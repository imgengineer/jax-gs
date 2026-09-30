import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.memory import SmemAllocator


@cute.kernel
def _projection_backward_kernel(
    xyz: cute.Tensor,
    log_scale: cute.Tensor,
    rotation: cute.Tensor,
    opacity: cute.Tensor,
    sh: cute.Tensor,
    view: cute.Tensor,
    intrinsic: cute.Tensor,
    center: cute.Tensor,
    color: cute.Tensor,
    cluster_ids: cute.Tensor,
    cluster_count: cute.Tensor,
    gmean: cute.Tensor,
    gdepth: cute.Tensor,
    gconic: cute.Tensor,
    gradius: cute.Tensor,
    gcolor: cute.Tensor,
    galpha: cute.Tensor,
    out_xyz: cute.Tensor,
    out_log_scale: cute.Tensor,
    out_rotation: cute.Tensor,
    out_opacity: cute.Tensor,
    out_sh: cute.Tensor,
    capacity: int,
    sh_dim: cutlass.Constexpr,
    degree: cutlass.Constexpr,
    near: float,
    cluster_size: cutlass.Constexpr,
    compacted: cutlass.Constexpr,
    compact_gradients: cutlass.Constexpr,
    rgb_only: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    # Gaussian-major pool storage needs a shared transpose for coalesced SH
    # gradient writes. Pad the coefficient stride to avoid bank conflicts
    # when neighboring lanes read different coefficients of the same point.
    sh_cache = SmemAllocator().allocate_tensor(
        cute.Float32, cute.make_layout((sh_dim * 3, 128), stride=(129, 1))
    )
    gid = bidx * bdx + tidx
    valid = gid < capacity
    if cutlass.const_expr(compacted):
        valid = valid and gid < cluster_count[0] * cluster_size
        if valid:
            gid = cluster_ids[gid // cluster_size] * cluster_size + gid % cluster_size
            valid = gid < capacity
    output_index = bidx * bdx + tidx if cutlass.const_expr(compact_gradients) else gid
    # Occluded or culled splats can have no upstream gradient. Their output
    # slots still need zeros, including the shared SH transpose below.
    has_gradient = False
    if valid:
        has_gradient = galpha[gid] != 0
        if cutlass.const_expr(not rgb_only):
            has_gradient = has_gradient | (gdepth[gid] != 0) | (gradius[gid] != 0)
        for component in cutlass.range_constexpr(2):
            has_gradient = has_gradient | (gmean[gid * 2 + component] != 0)
        for component in cutlass.range_constexpr(4):
            has_gradient = has_gradient | (gconic[gid * 4 + component] != 0)
        for component in cutlass.range_constexpr(3):
            has_gradient = has_gradient | (gcolor[gid * 3 + component] != 0)
        if not has_gradient:
            for component in cutlass.range_constexpr(3):
                out_xyz[output_index * 3 + component] = cute.Float32(0)
                out_log_scale[output_index * 3 + component] = cute.Float32(0)
            for component in cutlass.range_constexpr(4):
                out_rotation[output_index * 4 + component] = cute.Float32(0)
            out_opacity[output_index] = cute.Float32(0)
            for component in cutlass.range_constexpr(sh_dim * 3):
                sh_cache[component, tidx] = cute.Float32(0)
    if has_gradient:
        wx, wy, wz = xyz[gid * 3], xyz[gid * 3 + 1], xyz[gid * 3 + 2]
        x = view[0] * wx + view[1] * wy + view[2] * wz + view[3]
        y = view[4] * wx + view[5] * wy + view[6] * wz + view[7]
        z = view[8] * wx + view[9] * wy + view[10] * wz + view[11]
        safe_z = cute.max(z, cute.Float32(near))
        fx, fy = intrinsic[0], intrinsic[1]
        inv_z = 1.0 / safe_z
        inv_z2 = inv_z * inv_z
        gx = gmean[gid * 2] * fx * inv_z
        gy = gmean[gid * 2 + 1] * fy * inv_z
        gz = cute.Float32(0.0)
        if cutlass.const_expr(not rgb_only):
            gz = gdepth[gid]
        gsafe_z = -gmean[gid * 2] * fx * x * inv_z2
        gsafe_z -= gmean[gid * 2 + 1] * fy * y * inv_z2

        qw0, qx0, qy0, qz0 = (
            rotation[gid * 4],
            rotation[gid * 4 + 1],
            rotation[gid * 4 + 2],
            rotation[gid * 4 + 3],
        )
        qdenom = cute.sqrt(
            cute.max(qw0 * qw0 + qx0 * qx0 + qy0 * qy0 + qz0 * qz0, cute.Float32(1e-16))
        )
        qw, qx, qy, qz = qw0 / qdenom, qx0 / qdenom, qy0 / qdenom, qz0 / qdenom
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qw * qz)
        r02 = 2.0 * (qx * qz + qw * qy)
        r10 = 2.0 * (qx * qy + qw * qz)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qw * qx)
        r20 = 2.0 * (qx * qz - qw * qy)
        r21 = 2.0 * (qy * qz + qw * qx)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
        s0 = cute.exp(log_scale[gid * 3])
        s1 = cute.exp(log_scale[gid * 3 + 1])
        s2 = cute.exp(log_scale[gid * 3 + 2])
        m00 = (view[0] * r00 + view[1] * r10 + view[2] * r20) * s0
        m01 = (view[0] * r01 + view[1] * r11 + view[2] * r21) * s1
        m02 = (view[0] * r02 + view[1] * r12 + view[2] * r22) * s2
        m10 = (view[4] * r00 + view[5] * r10 + view[6] * r20) * s0
        m11 = (view[4] * r01 + view[5] * r11 + view[6] * r21) * s1
        m12 = (view[4] * r02 + view[5] * r12 + view[6] * r22) * s2
        m20 = (view[8] * r00 + view[9] * r10 + view[10] * r20) * s0
        m21 = (view[8] * r01 + view[9] * r11 + view[10] * r21) * s1
        m22 = (view[8] * r02 + view[9] * r12 + view[10] * r22) * s2
        j0 = fx * inv_z
        j1 = fy * inv_z
        j2 = fx * x * inv_z2
        j3 = fy * y * inv_z2
        a0, a1, a2 = j0 * m00 - j2 * m20, j0 * m01 - j2 * m21, j0 * m02 - j2 * m22
        b0, b1, b2 = j1 * m10 - j3 * m20, j1 * m11 - j3 * m21, j1 * m12 - j3 * m22
        a = a0 * a0 + a1 * a1 + a2 * a2 + 0.3
        b = a0 * b0 + a1 * b1 + a2 * b2
        d = b0 * b0 + b1 * b1 + b2 * b2 + 0.3
        raw_det = a * d - b * b
        det = cute.max(raw_det, cute.Float32(1e-8))
        gc00, gc01 = gconic[gid * 4], gconic[gid * 4 + 1]
        gc10, gc11 = gconic[gid * 4 + 2], gconic[gid * 4 + 3]
        gdet = cute.Float32(0.0)
        if raw_det > 1e-8:
            gdet = -(gc00 * d - (gc01 + gc10) * b + gc11 * a) / (det * det)
        ga = gc11 / det + gdet * d
        gb = -(gc01 + gc10) / det - 2.0 * b * gdet
        gd = gc00 / det + gdet * a

        if cutlass.const_expr(not rgb_only):
            disc = (a - d) * (a - d) + 4.0 * b * b
            disc_root = cute.sqrt(cute.max(disc, cute.Float32(1e-12)))
            eigen = 0.5 * (a + d + disc_root)
            geigen = gradius[gid] * 1.5 / cute.sqrt(eigen)
            gdisc = cute.Float32(0.0)
            if disc > 1e-12:
                gdisc = geigen * 0.25 / disc_root
            ga += 0.5 * geigen + 2.0 * (a - d) * gdisc
            gb += 8.0 * b * gdisc
            gd += 0.5 * geigen - 2.0 * (a - d) * gdisc

        ga0, ga1, ga2 = 2 * a0 * ga + b0 * gb, 2 * a1 * ga + b1 * gb, 2 * a2 * ga + b2 * gb
        gb0, gb1, gb2 = 2 * b0 * gd + a0 * gb, 2 * b1 * gd + a1 * gb, 2 * b2 * gd + a2 * gb
        gj0 = ga0 * m00 + ga1 * m01 + ga2 * m02
        gj1 = gb0 * m10 + gb1 * m11 + gb2 * m12
        gj2 = -(ga0 * m20 + ga1 * m21 + ga2 * m22)
        gj3 = -(gb0 * m20 + gb1 * m21 + gb2 * m22)
        gx += gj2 * fx * inv_z2
        gy += gj3 * fy * inv_z2
        gsafe_z -= (gj0 * fx + gj1 * fy) * inv_z2
        gsafe_z -= 2.0 * (gj2 * fx * x + gj3 * fy * y) * inv_z2 * inv_z
        if z > near:
            gz += gsafe_z

        gm00, gm01, gm02 = j0 * ga0, j0 * ga1, j0 * ga2
        gm10, gm11, gm12 = j1 * gb0, j1 * gb1, j1 * gb2
        gm20 = -j2 * ga0 - j3 * gb0
        gm21 = -j2 * ga1 - j3 * gb1
        gm22 = -j2 * ga2 - j3 * gb2
        out_log_scale[output_index * 3] = gm00 * m00 + gm10 * m10 + gm20 * m20
        out_log_scale[output_index * 3 + 1] = gm01 * m01 + gm11 * m11 + gm21 * m21
        out_log_scale[output_index * 3 + 2] = gm02 * m02 + gm12 * m12 + gm22 * m22
        gr00 = s0 * (view[0] * gm00 + view[4] * gm10 + view[8] * gm20)
        gr10 = s0 * (view[1] * gm00 + view[5] * gm10 + view[9] * gm20)
        gr20 = s0 * (view[2] * gm00 + view[6] * gm10 + view[10] * gm20)
        gr01 = s1 * (view[0] * gm01 + view[4] * gm11 + view[8] * gm21)
        gr11 = s1 * (view[1] * gm01 + view[5] * gm11 + view[9] * gm21)
        gr21 = s1 * (view[2] * gm01 + view[6] * gm11 + view[10] * gm21)
        gr02 = s2 * (view[0] * gm02 + view[4] * gm12 + view[8] * gm22)
        gr12 = s2 * (view[1] * gm02 + view[5] * gm12 + view[9] * gm22)
        gr22 = s2 * (view[2] * gm02 + view[6] * gm12 + view[10] * gm22)
        gqw = (
            -2 * qz * gr01
            + 2 * qy * gr02
            + 2 * qz * gr10
            - 2 * qx * gr12
            - 2 * qy * gr20
            + 2 * qx * gr21
        )
        gqx = (
            2 * qy * (gr01 + gr10)
            + 2 * qz * (gr02 + gr20)
            - 4 * qx * (gr11 + gr22)
            - 2 * qw * gr12
            + 2 * qw * gr21
        )
        gqy = (
            -4 * qy * (gr00 + gr22)
            + 2 * qx * (gr01 + gr10)
            + 2 * qw * gr02
            + 2 * qz * (gr12 + gr21)
            - 2 * qw * gr20
        )
        gqz = (
            -4 * qz * (gr00 + gr11)
            - 2 * qw * gr01
            + 2 * qx * (gr02 + gr20)
            + 2 * qw * gr10
            + 2 * qy * (gr12 + gr21)
        )
        qdot = gqw * qw + gqx * qx + gqy * qy + gqz * qz
        if qdenom > 1e-8:
            gqw = (gqw - qdot * qw) / qdenom
            gqx = (gqx - qdot * qx) / qdenom
            gqy = (gqy - qdot * qy) / qdenom
            gqz = (gqz - qdot * qz) / qdenom
        else:
            gqw /= qdenom
            gqx /= qdenom
            gqy /= qdenom
            gqz /= qdenom
        out_rotation[output_index * 4] = gqw
        out_rotation[output_index * 4 + 1] = gqx
        out_rotation[output_index * 4 + 2] = gqy
        out_rotation[output_index * 4 + 3] = gqz

        alpha = 1.0 / (1.0 + cute.exp(-opacity[gid]))
        out_opacity[output_index] = galpha[gid] * alpha * (1.0 - alpha)

        vx, vy, vz = wx - center[0], wy - center[1], wz - center[2]
        dir_denom = cute.sqrt(cute.max(vx * vx + vy * vy + vz * vz, cute.Float32(1e-16)))
        dx, dy, dz = vx / dir_denom, vy / dir_denom, vz / dir_denom
        xx, yy, zz = dx * dx, dy * dy, dz * dz
        gdx = cute.Float32(0.0)
        gdy = cute.Float32(0.0)
        gdz = cute.Float32(0.0)
        for channel in cutlass.range_constexpr(3):
            base = gid * sh_dim * 3 + channel
            for coefficient in cutlass.range_constexpr((degree + 1) * (degree + 1), sh_dim):
                sh_cache[coefficient * 3 + channel, tidx] = cute.Float32(0.0)
            gc = cute.Float32(0.0)
            if color[gid * 3 + channel] > 0.0:
                gc = gcolor[gid * 3 + channel]
            sh_cache[0 + channel, tidx] = gc * 0.28209479177387814
            if degree >= 1:
                sh_cache[3 + channel, tidx] = -gc * 0.4886025119029199 * dy
                sh_cache[6 + channel, tidx] = gc * 0.4886025119029199 * dz
                sh_cache[9 + channel, tidx] = -gc * 0.4886025119029199 * dx
                gdx -= gc * 0.4886025119029199 * sh[base + 9]
                gdy -= gc * 0.4886025119029199 * sh[base + 3]
                gdz += gc * 0.4886025119029199 * sh[base + 6]
            if degree >= 2:
                sh_cache[12 + channel, tidx] = gc * 1.0925484305920792 * dx * dy
                sh_cache[15 + channel, tidx] = -gc * 1.0925484305920792 * dy * dz
                sh_cache[18 + channel, tidx] = gc * 0.31539156525252005 * (2 * zz - xx - yy)
                sh_cache[21 + channel, tidx] = -gc * 1.0925484305920792 * dx * dz
                sh_cache[24 + channel, tidx] = gc * 0.5462742152960396 * (xx - yy)
                gdx += gc * (
                    1.0925484305920792 * dy * sh[base + 12]
                    - 0.6307831305050401 * dx * sh[base + 18]
                    - 1.0925484305920792 * dz * sh[base + 21]
                    + 1.0925484305920792 * dx * sh[base + 24]
                )
                gdy += gc * (
                    1.0925484305920792 * dx * sh[base + 12]
                    - 1.0925484305920792 * dz * sh[base + 15]
                    - 0.6307831305050401 * dy * sh[base + 18]
                    - 1.0925484305920792 * dy * sh[base + 24]
                )
                gdz += gc * (
                    -1.0925484305920792 * dy * sh[base + 15]
                    + 1.2615662610100802 * dz * sh[base + 18]
                    - 1.0925484305920792 * dx * sh[base + 21]
                )
            if degree >= 3:
                sh_cache[27 + channel, tidx] = -gc * 0.5900435899266435 * dy * (3 * xx - yy)
                sh_cache[30 + channel, tidx] = gc * 2.890611442640554 * dx * dy * dz
                sh_cache[33 + channel, tidx] = -gc * 0.4570457994644658 * dy * (4 * zz - xx - yy)
                sh_cache[36 + channel, tidx] = (
                    gc * 0.3731763325901154 * dz * (2 * zz - 3 * xx - 3 * yy)
                )
                sh_cache[39 + channel, tidx] = -gc * 0.4570457994644658 * dx * (4 * zz - xx - yy)
                sh_cache[42 + channel, tidx] = gc * 1.445305721320277 * dz * (xx - yy)
                sh_cache[45 + channel, tidx] = -gc * 0.5900435899266435 * dx * (xx - 3 * yy)
                gdx += gc * (
                    -3.540261539559861 * dx * dy * sh[base + 27]
                    + 2.890611442640554 * dy * dz * sh[base + 30]
                    + 0.9140915989289316 * dx * dy * sh[base + 33]
                    - 2.2390579955406924 * dx * dz * sh[base + 36]
                    - 0.4570457994644658 * (4 * zz - 3 * xx - yy) * sh[base + 39]
                    + 2.890611442640554 * dx * dz * sh[base + 42]
                    - 1.7701307697799305 * (xx - yy) * sh[base + 45]
                )
                gdy += gc * (
                    -1.7701307697799305 * (xx - yy) * sh[base + 27]
                    + 2.890611442640554 * dx * dz * sh[base + 30]
                    - 0.4570457994644658 * (4 * zz - xx - 3 * yy) * sh[base + 33]
                    - 2.2390579955406924 * dy * dz * sh[base + 36]
                    + 0.9140915989289316 * dx * dy * sh[base + 39]
                    - 2.890611442640554 * dy * dz * sh[base + 42]
                    + 3.540261539559861 * dx * dy * sh[base + 45]
                )
                gdz += gc * (
                    2.890611442640554 * dx * dy * sh[base + 30]
                    - 3.6563663957157264 * dy * dz * sh[base + 33]
                    + 0.3731763325901154 * (6 * zz - 3 * xx - 3 * yy) * sh[base + 36]
                    - 3.6563663957157264 * dx * dz * sh[base + 39]
                    + 1.445305721320277 * (xx - yy) * sh[base + 42]
                )

        dir_dot = gdx * dx + gdy * dy + gdz * dz
        gx_world, gy_world, gz_world = (gdx / dir_denom, gdy / dir_denom, gdz / dir_denom)
        if dir_denom > 1e-8:
            gx_world = (gdx - dir_dot * dx) / dir_denom
            gy_world = (gdy - dir_dot * dy) / dir_denom
            gz_world = (gdz - dir_dot * dz) / dir_denom
        out_xyz[output_index * 3] = view[0] * gx + view[4] * gy + view[8] * gz + gx_world
        out_xyz[output_index * 3 + 1] = view[1] * gx + view[5] * gy + view[9] * gz + gy_world
        out_xyz[output_index * 3 + 2] = view[2] * gx + view[6] * gy + view[10] * gz + gz_world

    active_block = True
    if cutlass.const_expr(compacted):
        active_block = bidx * bdx < cluster_count[0] * cluster_size
    # This condition is uniform across the CTA; partial blocks still have every
    # thread reach the barrier before reading other threads' SH gradients.
    if active_block:
        cute.arch.sync_threads()
        for offset in cutlass.range_constexpr(sh_dim * 3):
            flat = offset * 128 + tidx
            point = bidx * 128 + flat // (sh_dim * 3)
            coefficient = flat % (sh_dim * 3)
            write = point < capacity
            if cutlass.const_expr(compacted):
                write = write and point < cluster_count[0] * cluster_size
                if write:
                    point = cluster_ids[point // cluster_size] * cluster_size + point % cluster_size
                    write = point < capacity
            output_point = (
                bidx * 128 + flat // (sh_dim * 3)
                if cutlass.const_expr(compact_gradients)
                else point
            )
            if write:
                out_sh[output_point * sh_dim * 3 + coefficient] = sh_cache[
                    coefficient, flat // (sh_dim * 3)
                ]


@cute.jit
def launch_projection_backward(
    stream: cuda.CUstream,
    xyz: cute.Tensor,
    log_scale: cute.Tensor,
    rotation: cute.Tensor,
    opacity: cute.Tensor,
    sh: cute.Tensor,
    view: cute.Tensor,
    intrinsic: cute.Tensor,
    center: cute.Tensor,
    color: cute.Tensor,
    cluster_ids: cute.Tensor,
    cluster_count: cute.Tensor,
    gmean: cute.Tensor,
    gdepth: cute.Tensor,
    gconic: cute.Tensor,
    gradius: cute.Tensor,
    gcolor: cute.Tensor,
    galpha: cute.Tensor,
    out_xyz: cute.Tensor,
    out_log_scale: cute.Tensor,
    out_rotation: cute.Tensor,
    out_opacity: cute.Tensor,
    out_sh: cute.Tensor,
    *,
    capacity: int,
    sh_dim: cutlass.Constexpr,
    degree: cutlass.Constexpr,
    near: float,
    cluster_size: cutlass.Constexpr,
    compacted: cutlass.Constexpr,
    compact_gradients: cutlass.Constexpr,
    rgb_only: cutlass.Constexpr,
):
    block = 128
    if cutlass.const_expr(compacted and not compact_gradients):
        from .cluster_compact import _clear_parameter_grads

        _clear_parameter_grads(
            out_xyz, out_log_scale, out_rotation, out_opacity, out_sh, capacity, sh_dim
        ).launch(
            grid=[(capacity * max(sh_dim * 3, 4) + 255) // 256, 1, 1],
            block=[256, 1, 1],
            stream=stream,
        )
    _projection_backward_kernel(
        xyz,
        log_scale,
        rotation,
        opacity,
        sh,
        view,
        intrinsic,
        center,
        color,
        cluster_ids,
        cluster_count,
        gmean,
        gdepth,
        gconic,
        gradius,
        gcolor,
        galpha,
        out_xyz,
        out_log_scale,
        out_rotation,
        out_opacity,
        out_sh,
        capacity,
        sh_dim,
        degree,
        near,
        cluster_size,
        compacted,
        compact_gradients,
        rgb_only,
    ).launch(grid=[(capacity + block - 1) // block, 1, 1], block=[block, 1, 1], stream=stream)
