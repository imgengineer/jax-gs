import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.memory


@cute.kernel
def _zero_grads(
    gmean: cute.Tensor,
    gconic: cute.Tensor,
    gdepth: cute.Tensor,
    gcolor: cute.Tensor,
    gopacity: cute.Tensor,
    capacity: int,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    i = bidx * bdx + tidx
    if i < capacity * 4:
        gconic[i] = 0.0
        if i < capacity * 3:
            gcolor[i] = 0.0
        if i < capacity * 2:
            gmean[i] = 0.0
        if i < capacity:
            gdepth[i] = 0.0
            gopacity[i] = 0.0


@cute.kernel
def _backward_tiled(
    mean: cute.Tensor,
    conic: cute.Tensor,
    depth: cute.Tensor,
    color: cute.Tensor,
    opacity: cute.Tensor,
    ids: cute.Tensor,
    valid: cute.Tensor,
    background: cute.Tensor,
    transmittance_cache: cute.Tensor,
    drgb: cute.Tensor,
    ddepth: cute.Tensor,
    dalpha: cute.Tensor,
    gmean: cute.Tensor,
    gconic: cute.Tensor,
    gdepth: cute.Tensor,
    gcolor: cute.Tensor,
    gopacity: cute.Tensor,
    width: int,
    height: int,
    tile_size: int,
    tiles_x: int,
    k_max: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    tile, _, _ = cute.arch.block_idx()
    block_size, _, _ = cute.arch.block_dim()
    smem = cutlass.memory.SmemAllocator()
    partial = smem.allocate_tensor(cutlass.Float32, cute.make_layout((k_max * 11,)))
    for entry in range(tidx, k_max * 11, block_size):
        partial[entry] = 0.0
    cute.arch.sync_threads()

    for local in range(tidx, tile_size * tile_size, block_size):
        px = (tile % tiles_x) * tile_size + local % tile_size
        py = (tile // tiles_x) * tile_size + local // tile_size
        if px < width and py < height:
            p = py * width + px
            gr = drgb[p * 3]
            gg = drgb[p * 3 + 1]
            gb = drgb[p * 3 + 2]
            gd = ddepth[p]
            gtrans = gr * background[0] + gg * background[1] + gb * background[2] - dalpha[p]
            for step in range(k_max):
                k = k_max - 1 - step
                if valid[tile * k_max + k] != 0:
                    gid = ids[tile * k_max + k]
                    dx = cute.Float32(px) + 0.5 - mean[gid * 2]
                    dy = cute.Float32(py) + 0.5 - mean[gid * 2 + 1]
                    c00, c01 = conic[gid * 4], conic[gid * 4 + 1]
                    c10, c11 = conic[gid * 4 + 2], conic[gid * 4 + 3]
                    exponent = -0.5 * (dx * (c00 * dx + c01 * dy) + dy * (c10 * dx + c11 * dy))
                    if exponent >= -4.5:
                        e = cute.exp(exponent)
                        raw_alpha = opacity[gid] * e
                        if raw_alpha >= 1.0 / 256:
                            a = cute.min(cute.Float32(255.0 / 256), raw_alpha)
                            trans = transmittance_cache[p * k_max + k]
                            weight = trans * a
                            gw = (
                                gr * color[gid * 3]
                                + gg * color[gid * 3 + 1]
                                + gb * color[gid * 3 + 2]
                                + gd * depth[gid]
                            )
                            ga = trans * (gw - gtrans)
                            gtrans = a * gw + (1.0 - a) * gtrans
                            offset = k * 11
                            cute.arch.atomic_add(
                                partial.iterator + offset + 6, weight * gd, scope="cta"
                            )
                            cute.arch.atomic_add(
                                partial.iterator + offset + 7, weight * gr, scope="cta"
                            )
                            cute.arch.atomic_add(
                                partial.iterator + offset + 8, weight * gg, scope="cta"
                            )
                            cute.arch.atomic_add(
                                partial.iterator + offset + 9, weight * gb, scope="cta"
                            )
                            if raw_alpha < 255.0 / 256:
                                ge = ga * raw_alpha
                                cute.arch.atomic_add(
                                    partial.iterator + offset + 10, ga * e, scope="cta"
                                )
                                cute.arch.atomic_add(
                                    partial.iterator + offset,
                                    ge * (c00 * dx + 0.5 * (c01 + c10) * dy),
                                    scope="cta",
                                )
                                cute.arch.atomic_add(
                                    partial.iterator + offset + 1,
                                    ge * (c11 * dy + 0.5 * (c01 + c10) * dx),
                                    scope="cta",
                                )
                                cute.arch.atomic_add(
                                    partial.iterator + offset + 2, -0.5 * ge * dx * dx, scope="cta"
                                )
                                cute.arch.atomic_add(
                                    partial.iterator + offset + 3, -0.5 * ge * dx * dy, scope="cta"
                                )
                                cute.arch.atomic_add(
                                    partial.iterator + offset + 4, -0.5 * ge * dy * dx, scope="cta"
                                )
                                cute.arch.atomic_add(
                                    partial.iterator + offset + 5, -0.5 * ge * dy * dy, scope="cta"
                                )

    cute.arch.sync_threads()
    for entry in range(tidx, k_max * 11, block_size):
        k = entry // 11
        field = entry % 11
        if valid[tile * k_max + k] != 0:
            value = partial[entry]
            if value != 0.0:
                gid = ids[tile * k_max + k]
                if field < 2:
                    cute.arch.atomic_add(gmean.iterator + gid * 2 + field, value)
                elif field < 6:
                    cute.arch.atomic_add(gconic.iterator + gid * 4 + field - 2, value)
                elif field == 6:
                    cute.arch.atomic_add(gdepth.iterator + gid, value)
                elif field < 10:
                    cute.arch.atomic_add(gcolor.iterator + gid * 3 + field - 7, value)
                else:
                    cute.arch.atomic_add(gopacity.iterator + gid, value)


@cute.jit
def launch_backward(
    stream: cuda.CUstream,
    mean: cute.Tensor,
    conic: cute.Tensor,
    depth: cute.Tensor,
    color: cute.Tensor,
    opacity: cute.Tensor,
    ids: cute.Tensor,
    valid: cute.Tensor,
    background: cute.Tensor,
    transmittance_cache: cute.Tensor,
    drgb: cute.Tensor,
    ddepth: cute.Tensor,
    dalpha: cute.Tensor,
    gmean: cute.Tensor,
    gconic: cute.Tensor,
    gdepth: cute.Tensor,
    gcolor: cute.Tensor,
    gopacity: cute.Tensor,
    *,
    width: int,
    height: int,
    tile_size: int,
    tiles_x: int,
    k_max: cutlass.Constexpr[int],
    capacity: int,
):
    block = 256
    _zero_grads(gmean, gconic, gdepth, gcolor, gopacity, capacity).launch(
        grid=[(capacity * 4 + block - 1) // block, 1, 1], block=[block, 1, 1], stream=stream
    )
    num_tiles = tiles_x * ((height + tile_size - 1) // tile_size)
    _backward_tiled(
        mean,
        conic,
        depth,
        color,
        opacity,
        ids,
        valid,
        background,
        transmittance_cache,
        drgb,
        ddepth,
        dalpha,
        gmean,
        gconic,
        gdepth,
        gcolor,
        gopacity,
        width,
        height,
        tile_size,
        tiles_x,
        k_max,
    ).launch(grid=[num_tiles, 1, 1], block=[block, 1, 1], stream=stream)
