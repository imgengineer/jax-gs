"""CuTe port of LiteGS binning.cu and speedy_splat.cuh's AccuTile slices.

Derived from speedy-splat (https://github.com/j-alex-hanson/speedy-splat)
which is based on gaussian-splatting. Original work © Inria and MPII.
Licensed under the Gaussian-Splatting License; see LICENSE.LiteGS.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


@cute.jit
def _intersection(px, py, c00, c01, c11, disc, level, is_y, coord):
    pu, pv, coeff = px, py, c11
    if is_y:
        pu, pv, coeff = py, px, c00
    delta = coord - pu
    root = cute.sqrt(cute.max(disc * delta * delta + level * coeff, 0.0))
    return ((-c01 * delta - root) / coeff + pv, (-c01 * delta + root) / coeff + pv)


@cute.jit
def _process_tiles(
    px,
    py,
    c00,
    c01,
    c11,
    alpha,
    tile_size,
    tile_height,
    tiles_x,
    tiles_y,
    gid,
    offset,
    max_pairs,
    keys,
    values,
    emit: cutlass.Constexpr,
):
    disc = c01 * c01 - c00 * c11
    level = 2.0 * cute.log(alpha * 255.0)
    x_term = cute.sqrt(-(c01 * c01 * level) / (disc * c00))
    y_term = cute.sqrt(-(c01 * c01 * level) / (disc * c11))
    if c01 >= 0:
        x_term, y_term = -x_term, -y_term
    argmin_x, argmin_y = py - y_term, px - x_term
    argmax_x, argmax_y = py + y_term, px + x_term
    xmin, _ = _intersection(px, py, c00, c01, c11, disc, level, True, argmin_x)
    ymin, _ = _intersection(px, py, c00, c01, c11, disc, level, False, argmin_y)
    _, xmax = _intersection(px, py, c00, c01, c11, disc, level, True, argmax_x)
    _, ymax = _intersection(px, py, c00, c01, c11, disc, level, False, argmax_y)
    x0 = cute.max(0, cute.min(tiles_x, cutlass.Int32(xmin / tile_size)))
    y0 = cute.max(0, cute.min(tiles_y, cutlass.Int32(ymin / tile_height)))
    x1 = cute.max(0, cute.min(tiles_x, cutlass.Int32((xmax + tile_size - 1) / tile_size)))
    y1 = cute.max(0, cute.min(tiles_y, cutlass.Int32((ymax + tile_height - 1) / tile_height)))
    is_y = y1 - y0 < x1 - x0
    block_u, block_v = cute.Float32(tile_size), cute.Float32(tile_height)
    u0, u1, v0, v1 = x0, x1, y0, y1
    bbox_u_min, bbox_u_max, bbox_v_min, bbox_v_max = xmin, xmax, ymin, ymax
    arg_min, arg_max = argmin_y, argmax_y
    if is_y:
        block_u, block_v = cute.Float32(tile_height), cute.Float32(tile_size)
        u0, u1, v0, v1 = y0, y1, x0, x1
        bbox_u_min, bbox_u_max, bbox_v_min, bbox_v_max = ymin, ymax, xmin, xmax
        arg_min, arg_max = argmin_x, argmax_x
    count = cutlass.Int32(0)
    if (y1 - y0) * (x1 - x0) > 0:
        upper_min, upper_max = bbox_v_max, bbox_v_min
        lower_min, lower_max = upper_min, upper_max
        line = cute.Float32(u0) * block_u
        if bbox_u_min <= line:
            lower_min, lower_max = _intersection(px, py, c00, c01, c11, disc, level, is_y, line)
        for u in range(u0, u1):
            next_line = line + block_u
            if next_line <= bbox_u_max:
                upper_min, upper_max = _intersection(
                    px, py, c00, c01, c11, disc, level, is_y, next_line
                )
            ellipse_min = cute.min(lower_min, upper_min)
            ellipse_max = cute.max(lower_max, upper_max)
            if line <= arg_min and arg_min < next_line:
                ellipse_min = bbox_v_min
            if line <= arg_max and arg_max < next_line:
                ellipse_max = bbox_v_max
            first = cute.max(v0, cute.min(v1, cutlass.Int32(ellipse_min / block_v)))
            end = cute.min(v1, cute.max(v0, cutlass.Int32(ellipse_max / block_v + 1)))
            if cutlass.const_expr(emit):
                for v in range(first, end):
                    key = v * tiles_x + u
                    if is_y:
                        key = u * tiles_x + v
                    slot = offset + count + v - first
                    if slot < max_pairs:
                        keys[slot] = keys.element_type(key)
                        values[slot] = gid
            count += end - first
            lower_min, lower_max = upper_min, upper_max
            line = next_line
    return count


@cute.kernel
def _pairs(
    mean: cute.Tensor,
    conic: cute.Tensor,
    alpha: cute.Tensor,
    visible: cute.Tensor,
    offsets: cute.Tensor,
    counts: cute.Tensor,
    keys: cute.Tensor,
    values: cute.Tensor,
    capacity: int,
    tile_size: cutlass.Constexpr,
    tile_height: cutlass.Constexpr,
    tiles_x: int,
    tiles_y: int,
    width: int,
    height: int,
    max_pairs: int,
    emit: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    gid = block * 256 + tid
    if gid < capacity:
        count = cutlass.Int32(0)
        if visible[gid] != 0:
            mx, my = mean[gid * 2], mean[gid * 2 + 1]
            c00, c01, c11 = conic[gid * 4], conic[gid * 4 + 1], conic[gid * 4 + 3]
            a = alpha[gid]
            if (
                a >= 1.0 / 255.0
                and c00 > 0.0
                and c11 > 0.0
                and c00 * c11 - c01 * c01 > 0.0
                and mx >= -0.15 * width
                and mx <= 1.15 * width
                and my >= -0.15 * height
                and my <= 1.15 * height
            ):
                offset = cutlass.Int32(0)
                if cutlass.const_expr(emit):
                    offset = offsets[gid]
                count = _process_tiles(
                    mx - 0.5,
                    my - 0.5,
                    c00,
                    c01,
                    c11,
                    a,
                    tile_size,
                    tile_height,
                    tiles_x,
                    tiles_y,
                    gid,
                    offset,
                    max_pairs,
                    keys,
                    values,
                    emit,
                )
        if cutlass.const_expr(not emit):
            counts[gid] = count


@cute.kernel
def _clear_pairs(tile_ids: cute.Tensor, gaussian_ids: cute.Tensor, max_pairs: int, num_tiles: int):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    i = block * 256 + tid
    if i < max_pairs:
        tile_ids[i] = tile_ids.element_type(num_tiles)
        gaussian_ids[i] = cutlass.Int32(0)


@cute.kernel
def _tile_ranges(tile_ids: cute.Tensor, offsets: cute.Tensor, max_pairs: int, num_tiles: int):
    """LiteGS tile_range_kernel, with contiguous offsets for empty tiles."""
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    i = block * 256 + tid
    # Widen unsigned keys before signed range arithmetic/comparisons. CuTe's
    # direct uint16 -> int32 promotion sign-extends values above 32767.
    if cutlass.Int32(cutlass.Uint32(tile_ids[0])) == num_tiles:
        # Empty fixed-capacity table: all entries are padding sentinels.
        if i <= num_tiles:
            offsets[i] = cutlass.Int32(0)
    elif i <= max_pairs:
        previous = cutlass.Int32(-1)
        current = cutlass.Int32(num_tiles)
        if i > 0:
            previous = cutlass.Int32(cutlass.Uint32(tile_ids[i - 1]))
        if i < max_pairs:
            current = cutlass.Int32(cutlass.Uint32(tile_ids[i]))
        # Each boundary owns a disjoint interval, including empty tiles.
        # A virtual final sentinel also closes tables with no spare capacity.
        for tile in range(previous + 1, current + 1):
            offsets[tile] = i


@cute.jit
def launch_tile_ranges(
    stream: cuda.CUstream,
    tile_ids: cute.Tensor,
    offsets: cute.Tensor,
    *,
    max_pairs: int,
    num_tiles: int,
):
    _tile_ranges(tile_ids, offsets, max_pairs, num_tiles).launch(
        grid=[(max(max_pairs, num_tiles) + 256) // 256, 1, 1], block=[256, 1, 1], stream=stream
    )


@cute.jit
def launch_count_pairs(
    stream: cuda.CUstream,
    mean: cute.Tensor,
    conic: cute.Tensor,
    alpha: cute.Tensor,
    visible: cute.Tensor,
    counts: cute.Tensor,
    *,
    capacity: int,
    tile_size: int,
    tile_height: int,
    tiles_x: int,
    tiles_y: int,
    width: int,
    height: int,
):
    _pairs(
        mean,
        conic,
        alpha,
        visible,
        counts,
        counts,
        counts,
        counts,
        capacity,
        tile_size,
        tile_height,
        tiles_x,
        tiles_y,
        width,
        height,
        0,
        False,
    ).launch(grid=[(capacity + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream)


@cute.jit
def launch_emit_pairs(
    stream: cuda.CUstream,
    mean: cute.Tensor,
    conic: cute.Tensor,
    alpha: cute.Tensor,
    visible: cute.Tensor,
    offsets: cute.Tensor,
    tile_ids: cute.Tensor,
    gaussian_ids: cute.Tensor,
    *,
    capacity: int,
    max_pairs: int,
    tile_size: int,
    tile_height: int,
    tiles_x: int,
    tiles_y: int,
    width: int,
    height: int,
):
    _clear_pairs(tile_ids, gaussian_ids, max_pairs, tiles_x * tiles_y).launch(
        grid=[(max_pairs + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
    )
    _pairs(
        mean,
        conic,
        alpha,
        visible,
        offsets,
        offsets,
        tile_ids,
        gaussian_ids,
        capacity,
        tile_size,
        tile_height,
        tiles_x,
        tiles_y,
        width,
        height,
        max_pairs,
        True,
    ).launch(grid=[(capacity + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream)
