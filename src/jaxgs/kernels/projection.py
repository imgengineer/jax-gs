import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


@cute.kernel
def _projection_kernel(
    xyz: cute.Tensor,
    log_scale: cute.Tensor,
    rotation: cute.Tensor,
    opacity: cute.Tensor,
    sh: cute.Tensor,
    alive: cute.Tensor,
    view: cute.Tensor,
    intrinsic: cute.Tensor,
    center: cute.Tensor,
    cluster_ids: cute.Tensor,
    cluster_count: cute.Tensor,
    out_mean: cute.Tensor,
    out_depth: cute.Tensor,
    out_conic: cute.Tensor,
    out_radius: cute.Tensor,
    out_color: cute.Tensor,
    out_alpha: cute.Tensor,
    out_visible: cute.Tensor,
    capacity: int,
    sh_dim: cutlass.Constexpr,
    degree: cutlass.Constexpr,
    width: int,
    height: int,
    near: float,
    far: float,
    cluster_size: cutlass.Constexpr,
    compacted: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    gid = bidx * bdx + tidx
    valid = gid < capacity
    if cutlass.const_expr(compacted):
        valid = valid and gid < cluster_count[0] * cluster_size
        if valid:
            gid = cluster_ids[gid // cluster_size] * cluster_size + gid % cluster_size
            valid = gid < capacity
    if valid:
        wx, wy, wz = xyz[gid * 3], xyz[gid * 3 + 1], xyz[gid * 3 + 2]
        x = view[0] * wx + view[1] * wy + view[2] * wz + view[3]
        y = view[4] * wx + view[5] * wy + view[6] * wz + view[7]
        z = view[8] * wx + view[9] * wy + view[10] * wz + view[11]
        safe_z = cute.max(z, cute.Float32(near))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        u = fx * x / safe_z + cx
        v = fy * y / safe_z + cy
        out_mean[gid * 2] = u
        out_mean[gid * 2 + 1] = v
        out_depth[gid] = z

        qw, qx, qy, qz = (
            rotation[gid * 4],
            rotation[gid * 4 + 1],
            rotation[gid * 4 + 2],
            rotation[gid * 4 + 3],
        )
        qnorm = cute.rsqrt(cute.max(qw * qw + qx * qx + qy * qy + qz * qz, cute.Float32(1e-16)))
        qw, qx, qy, qz = qw * qnorm, qx * qnorm, qy * qnorm, qz * qnorm
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
        j0 = fx / safe_z
        j1 = fy / safe_z
        j2 = fx * x / (safe_z * safe_z)
        j3 = fy * y / (safe_z * safe_z)
        a0, a1, a2 = j0 * m00 - j2 * m20, j0 * m01 - j2 * m21, j0 * m02 - j2 * m22
        b0, b1, b2 = j1 * m10 - j3 * m20, j1 * m11 - j3 * m21, j1 * m12 - j3 * m22
        a = a0 * a0 + a1 * a1 + a2 * a2 + 0.3
        b = a0 * b0 + a1 * b1 + a2 * b2
        d = b0 * b0 + b1 * b1 + b2 * b2 + 0.3
        det = cute.max(a * d - b * b, cute.Float32(1e-8))
        out_conic[gid * 4] = d / det
        out_conic[gid * 4 + 1] = -b / det
        out_conic[gid * 4 + 2] = -b / det
        out_conic[gid * 4 + 3] = a / det
        radius = 3.0 * cute.sqrt(
            0.5 * (a + d + cute.sqrt(cute.max((a - d) * (a - d) + 4 * b * b, cute.Float32(1e-12))))
        )
        out_radius[gid] = radius
        alpha = 1.0 / (1.0 + cute.exp(-opacity[gid]))
        out_alpha[gid] = alpha
        is_visible = (
            alive[gid] != 0
            and z > near
            and z < far
            and alpha >= 1.0 / 255
            and u >= -0.15 * width
            and u <= 1.15 * width
            and v >= -0.15 * height
            and v <= 1.15 * height
        )
        out_visible[gid] = cutlass.Int8(1) if is_visible else cutlass.Int8(0)

        dx, dy, dz = wx - center[0], wy - center[1], wz - center[2]
        direction_norm = cute.rsqrt(cute.max(dx * dx + dy * dy + dz * dz, cute.Float32(1e-16)))
        dx, dy, dz = dx * direction_norm, dy * direction_norm, dz * direction_norm
        xx, yy, zz = dx * dx, dy * dy, dz * dz
        # Each point's coefficients are contiguous: load the active degree's
        # prefix with 128-bit loads instead of strided scalar loads.
        stride = sh_dim * 3
        coefficients = cute.make_rmem_tensor(
            min(stride, ((degree + 1) * (degree + 1) * 3 + 3) // 4 * 4), cute.Float32
        )
        cute.autovec_copy(
            cute.make_tensor(
                sh.iterator + cute.assume(gid * stride, divby=4 if stride % 4 == 0 else 1),
                coefficients.layout,
            ),
            coefficients,
        )
        for channel in cutlass.range_constexpr(3):
            value = 0.28209479177387814 * coefficients[channel]
            if degree >= 1:
                value += (
                    -0.4886025119029199 * dy * coefficients[channel + 3]
                    + 0.4886025119029199 * dz * coefficients[channel + 6]
                    - 0.4886025119029199 * dx * coefficients[channel + 9]
                )
            if degree >= 2:
                value += (
                    1.0925484305920792 * dx * dy * coefficients[channel + 12]
                    - 1.0925484305920792 * dy * dz * coefficients[channel + 15]
                    + 0.31539156525252005 * (2 * zz - xx - yy) * coefficients[channel + 18]
                    - 1.0925484305920792 * dx * dz * coefficients[channel + 21]
                    + 0.5462742152960396 * (xx - yy) * coefficients[channel + 24]
                )
            if degree >= 3:
                value += (
                    -0.5900435899266435 * dy * (3 * xx - yy) * coefficients[channel + 27]
                    + 2.890611442640554 * dx * dy * dz * coefficients[channel + 30]
                    - 0.4570457994644658 * dy * (4 * zz - xx - yy) * coefficients[channel + 33]
                    + 0.3731763325901154
                    * dz
                    * (2 * zz - 3 * xx - 3 * yy)
                    * coefficients[channel + 36]
                    - 0.4570457994644658 * dx * (4 * zz - xx - yy) * coefficients[channel + 39]
                    + 1.445305721320277 * dz * (xx - yy) * coefficients[channel + 42]
                    - 0.5900435899266435 * dx * (xx - 3 * yy) * coefficients[channel + 45]
                )
            out_color[gid * 3 + channel] = cute.max(value + 0.5, cute.Float32(0.0))


@cute.jit
def launch_projection(
    stream: cuda.CUstream,
    xyz: cute.Tensor,
    log_scale: cute.Tensor,
    rotation: cute.Tensor,
    opacity: cute.Tensor,
    sh: cute.Tensor,
    alive: cute.Tensor,
    view: cute.Tensor,
    intrinsic: cute.Tensor,
    center: cute.Tensor,
    cluster_ids: cute.Tensor,
    cluster_count: cute.Tensor,
    out_mean: cute.Tensor,
    out_depth: cute.Tensor,
    out_conic: cute.Tensor,
    out_radius: cute.Tensor,
    out_color: cute.Tensor,
    out_alpha: cute.Tensor,
    out_visible: cute.Tensor,
    *,
    capacity: int,
    sh_dim: cutlass.Constexpr,
    degree: cutlass.Constexpr,
    width: int,
    height: int,
    near: float,
    far: float,
    cluster_size: cutlass.Constexpr,
    compacted: cutlass.Constexpr,
    clear_invisible: cutlass.Constexpr = True,
):
    block = 128
    if cutlass.const_expr(compacted and clear_invisible):
        from .cluster_compact import _clear_projected

        _clear_projected(
            out_mean, out_depth, out_conic, out_radius, out_color, out_alpha, out_visible, capacity
        ).launch(grid=[(capacity * 4 + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream)
    elif cutlass.const_expr(compacted):
        from .cluster_compact import _clear_visible

        _clear_visible(out_visible, capacity).launch(
            grid=[(capacity + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
        )
    _projection_kernel(
        xyz,
        log_scale,
        rotation,
        opacity,
        sh,
        alive,
        view,
        intrinsic,
        center,
        cluster_ids,
        cluster_count,
        out_mean,
        out_depth,
        out_conic,
        out_radius,
        out_color,
        out_alpha,
        out_visible,
        capacity,
        sh_dim,
        degree,
        width,
        height,
        near,
        far,
        cluster_size,
        compacted,
    ).launch(grid=[(capacity + block - 1) // block, 1, 1], block=[block, 1, 1], stream=stream)
