"""Range-based rasterization with LiteGS's pixel-sized backward cache."""

import cuda.bindings.driver as cuda
import cutlass.cute as cute

from .rasterize_backward import _zero_grads


@cute.kernel
def _zero_fragments(stats: cute.Tensor, count: int):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 256 + tidx
    if i < count:
        stats[i] = cute.Float32(0.0)


@cute.jit
def _warp_sum(value):
    value += cute.arch.shuffle_sync_down(value, 16)
    value += cute.arch.shuffle_sync_down(value, 8)
    value += cute.arch.shuffle_sync_down(value, 4)
    value += cute.arch.shuffle_sync_down(value, 2)
    value += cute.arch.shuffle_sync_down(value, 1)
    return value


@cute.kernel
def _forward(
    mean: cute.Tensor,
    conic: cute.Tensor,
    depth: cute.Tensor,
    color: cute.Tensor,
    opacity: cute.Tensor,
    ids: cute.Tensor,
    tile_offsets: cute.Tensor,
    background: cute.Tensor,
    out_rgb: cute.Tensor,
    out_depth: cute.Tensor,
    out_alpha: cute.Tensor,
    out_final_t: cute.Tensor,
    out_last: cute.Tensor,
    width: int,
    height: int,
    tile_size: int,
    tiles_x: int,
):
    tidx, _, _ = cute.arch.thread_idx()
    tile, _, _ = cute.arch.block_idx()
    block, _, _ = cute.arch.block_dim()
    for local in range(tidx, tile_size * tile_size, block):
        px = (tile % tiles_x) * tile_size + local % tile_size
        py = (tile // tiles_x) * tile_size + local // tile_size
        if px < width and py < height:
            p = py * width + px
            red = cute.Float32(0.0)
            green = cute.Float32(0.0)
            blue = cute.Float32(0.0)
            weighted_depth = cute.Float32(0.0)
            transmittance = cute.Float32(1.0)
            index = tile_offsets[tile]
            end = tile_offsets[tile + 1]
            while index < end and transmittance >= 1.0 / 8192.0:
                gid = ids[index]
                dx = cute.Float32(px) + 0.5 - mean[gid * 2]
                dy = cute.Float32(py) + 0.5 - mean[gid * 2 + 1]
                exponent = -0.5 * (
                    dx * (conic[gid * 4] * dx + conic[gid * 4 + 1] * dy)
                    + dy * (conic[gid * 4 + 2] * dx + conic[gid * 4 + 3] * dy)
                )
                if exponent >= -4.5:
                    raw_alpha = opacity[gid] * cute.exp(exponent)
                    if raw_alpha >= 1.0 / 256.0:
                        a = cute.min(cute.Float32(255.0 / 256.0), raw_alpha)
                        weight = a * transmittance
                        red += weight * color[gid * 3]
                        green += weight * color[gid * 3 + 1]
                        blue += weight * color[gid * 3 + 2]
                        weighted_depth += weight * depth[gid]
                        transmittance *= 1.0 - a
                index += 1
            out_rgb[p * 3] = red + transmittance * background[0]
            out_rgb[p * 3 + 1] = green + transmittance * background[1]
            out_rgb[p * 3 + 2] = blue + transmittance * background[2]
            out_depth[p] = weighted_depth
            out_alpha[p] = 1.0 - transmittance
            out_final_t[p] = transmittance
            out_last[p] = index


@cute.kernel
def _backward_warp(
    mean: cute.Tensor,
    conic: cute.Tensor,
    depth: cute.Tensor,
    color: cute.Tensor,
    opacity: cute.Tensor,
    ids: cute.Tensor,
    tile_offsets: cute.Tensor,
    background: cute.Tensor,
    final_t: cute.Tensor,
    last: cute.Tensor,
    drgb: cute.Tensor,
    ddepth: cute.Tensor,
    dalpha: cute.Tensor,
    gmean: cute.Tensor,
    gconic: cute.Tensor,
    gdepth: cute.Tensor,
    gcolor: cute.Tensor,
    gopacity: cute.Tensor,
    fragment_stats: cute.Tensor,
    collect_stats: int,
    width: int,
    height: int,
    tile_size: int,
    tiles_x: int,
    blocks_per_tile: int,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    tile = bidx // blocks_per_tile
    local = (bidx % blocks_per_tile) * 256 + tidx
    px = (tile % tiles_x) * tile_size + local % tile_size
    py = (tile // tiles_x) * tile_size + local // tile_size
    in_image = local < tile_size * tile_size and px < width and py < height
    p = py * width + px
    start = tile_offsets[tile]
    last_index = start
    gr = cute.Float32(0.0)
    gg = cute.Float32(0.0)
    gb = cute.Float32(0.0)
    gd = cute.Float32(0.0)
    gtrans = cute.Float32(0.0)
    trans_after = cute.Float32(1.0)
    if in_image:
        last_index = last[p]
        gr, gg, gb = drgb[p * 3], drgb[p * 3 + 1], drgb[p * 3 + 2]
        gd = ddepth[p]
        gtrans = gr * background[0] + gg * background[1] + gb * background[2] - dalpha[p]
        trans_after = final_t[p]
    index = cute.arch.warp_redux_sync(last_index, "max") - 1
    lane = tidx % 32
    while index >= start:
        gid = ids[index]
        gmx = cute.Float32(0.0)
        gmy = cute.Float32(0.0)
        gc00 = cute.Float32(0.0)
        gc01 = cute.Float32(0.0)
        gc10 = cute.Float32(0.0)
        gc11 = cute.Float32(0.0)
        gz = cute.Float32(0.0)
        gcr = cute.Float32(0.0)
        gcg = cute.Float32(0.0)
        gcb = cute.Float32(0.0)
        go = cute.Float32(0.0)
        fragment_count = cute.Float32(0.0)
        fragment_weight = cute.Float32(0.0)
        if in_image and index < last_index:
            dx = cute.Float32(px) + 0.5 - mean[gid * 2]
            dy = cute.Float32(py) + 0.5 - mean[gid * 2 + 1]
            c00, c01 = conic[gid * 4], conic[gid * 4 + 1]
            c10, c11 = conic[gid * 4 + 2], conic[gid * 4 + 3]
            exponent = -0.5 * (dx * (c00 * dx + c01 * dy) + dy * (c10 * dx + c11 * dy))
            if exponent >= -4.5:
                e = cute.exp(exponent)
                raw_alpha = opacity[gid] * e
                if raw_alpha >= 1.0 / 256.0:
                    a = cute.min(cute.Float32(255.0 / 256.0), raw_alpha)
                    trans_before = trans_after / (1.0 - a)
                    weight = trans_before * a
                    fragment_count = 1.0
                    fragment_weight = weight
                    gw = (
                        gr * color[gid * 3]
                        + gg * color[gid * 3 + 1]
                        + gb * color[gid * 3 + 2]
                        + gd * depth[gid]
                    )
                    ga = trans_before * (gw - gtrans)
                    gtrans = a * gw + (1.0 - a) * gtrans
                    gz = weight * gd
                    gcr, gcg, gcb = weight * gr, weight * gg, weight * gb
                    if raw_alpha < 255.0 / 256.0:
                        ge = ga * raw_alpha
                        go = ga * e
                        gmx = ge * (c00 * dx + 0.5 * (c01 + c10) * dy)
                        gmy = ge * (c11 * dy + 0.5 * (c01 + c10) * dx)
                        gc00 = -0.5 * ge * dx * dx
                        gc01 = -0.5 * ge * dx * dy
                        gc10 = -0.5 * ge * dy * dx
                        gc11 = -0.5 * ge * dy * dy
                    trans_after = trans_before
        if collect_stats:
            count_sum = _warp_sum(fragment_count)
            weight_sum = _warp_sum(fragment_weight)
            square_sum = _warp_sum(go * go)
            if lane == 0:
                if count_sum != 0.0:
                    cute.arch.atomic_add(fragment_stats.iterator + gid * 3, count_sum)
                    cute.arch.atomic_add(fragment_stats.iterator + gid * 3 + 1, weight_sum)
                    cute.arch.atomic_add(fragment_stats.iterator + gid * 3 + 2, square_sum)
        gmx = _warp_sum(gmx)
        gmy = _warp_sum(gmy)
        gc00 = _warp_sum(gc00)
        gc01 = _warp_sum(gc01)
        gc10 = _warp_sum(gc10)
        gc11 = _warp_sum(gc11)
        gz = _warp_sum(gz)
        gcr = _warp_sum(gcr)
        gcg = _warp_sum(gcg)
        gcb = _warp_sum(gcb)
        go = _warp_sum(go)
        if lane == 0:
            if gmx != 0.0:
                cute.arch.atomic_add(gmean.iterator + gid * 2, gmx)
            if gmy != 0.0:
                cute.arch.atomic_add(gmean.iterator + gid * 2 + 1, gmy)
            if gc00 != 0.0:
                cute.arch.atomic_add(gconic.iterator + gid * 4, gc00)
            if gc01 != 0.0:
                cute.arch.atomic_add(gconic.iterator + gid * 4 + 1, gc01)
            if gc10 != 0.0:
                cute.arch.atomic_add(gconic.iterator + gid * 4 + 2, gc10)
            if gc11 != 0.0:
                cute.arch.atomic_add(gconic.iterator + gid * 4 + 3, gc11)
            if gz != 0.0:
                cute.arch.atomic_add(gdepth.iterator + gid, gz)
            if gcr != 0.0:
                cute.arch.atomic_add(gcolor.iterator + gid * 3, gcr)
            if gcg != 0.0:
                cute.arch.atomic_add(gcolor.iterator + gid * 3 + 1, gcg)
            if gcb != 0.0:
                cute.arch.atomic_add(gcolor.iterator + gid * 3 + 2, gcb)
            if go != 0.0:
                cute.arch.atomic_add(gopacity.iterator + gid, go)
        index -= 1


@cute.jit
def launch_forward_sorted(
    stream: cuda.CUstream,
    mean: cute.Tensor,
    conic: cute.Tensor,
    depth: cute.Tensor,
    color: cute.Tensor,
    opacity: cute.Tensor,
    ids: cute.Tensor,
    tile_offsets: cute.Tensor,
    background: cute.Tensor,
    out_rgb: cute.Tensor,
    out_depth: cute.Tensor,
    out_alpha: cute.Tensor,
    out_final_t: cute.Tensor,
    out_last: cute.Tensor,
    *,
    width: int,
    height: int,
    tile_size: int,
    tiles_x: int,
):
    tiles_y = (height + tile_size - 1) // tile_size
    _forward(
        mean,
        conic,
        depth,
        color,
        opacity,
        ids,
        tile_offsets,
        background,
        out_rgb,
        out_depth,
        out_alpha,
        out_final_t,
        out_last,
        width,
        height,
        tile_size,
        tiles_x,
    ).launch(grid=[tiles_x * tiles_y, 1, 1], block=[256, 1, 1], stream=stream)


@cute.jit
def launch_backward_sorted(
    stream: cuda.CUstream,
    mean: cute.Tensor,
    conic: cute.Tensor,
    depth: cute.Tensor,
    color: cute.Tensor,
    opacity: cute.Tensor,
    ids: cute.Tensor,
    tile_offsets: cute.Tensor,
    background: cute.Tensor,
    final_t: cute.Tensor,
    last: cute.Tensor,
    drgb: cute.Tensor,
    ddepth: cute.Tensor,
    dalpha: cute.Tensor,
    gmean: cute.Tensor,
    gconic: cute.Tensor,
    gdepth: cute.Tensor,
    gcolor: cute.Tensor,
    gopacity: cute.Tensor,
    fragment_stats: cute.Tensor,
    *,
    width: int,
    height: int,
    tile_size: int,
    tiles_x: int,
    capacity: int,
    collect_stats: int = 0,
):
    block = 256
    _zero_grads(gmean, gconic, gdepth, gcolor, gopacity, capacity).launch(
        grid=[(capacity * 4 + block - 1) // block, 1, 1], block=[block, 1, 1], stream=stream
    )
    if collect_stats:
        _zero_fragments(fragment_stats, capacity * 3).launch(
            grid=[(capacity * 3 + block - 1) // block, 1, 1], block=[block, 1, 1], stream=stream
        )
    tiles_y = (height + tile_size - 1) // tile_size
    blocks_per_tile = (tile_size * tile_size + block - 1) // block
    _backward_warp(
        mean,
        conic,
        depth,
        color,
        opacity,
        ids,
        tile_offsets,
        background,
        final_t,
        last,
        drgb,
        ddepth,
        dalpha,
        gmean,
        gconic,
        gdepth,
        gcolor,
        gopacity,
        fragment_stats,
        collect_stats,
        width,
        height,
        tile_size,
        tiles_x,
        blocks_per_tile,
    ).launch(grid=[tiles_x * tiles_y * blocks_per_tile, 1, 1], block=[block, 1, 1], stream=stream)
