"""CuTe port of LiteGS binning.cu and speedy_splat.cuh's AccuTile slices.

Derived from speedy-splat (https://github.com/j-alex-hanson/speedy-splat)
which is based on gaussian-splatting. Original work © Inria and MPII.
Licensed under the Gaussian-Splatting License; see LICENSE.LiteGS.

Counting and emission evaluate the same explicitly rounded ellipse slices, so
each Gaussian writes exactly the pairs it counted. Emission visits Gaussians in
depth order: neighboring lanes then write neighboring pair segments. Gaussians
with many pairs are emitted by their whole warp instead of one lane.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from . import half2 as h

# Gaussians with more pairs are emitted cooperatively by all 32 lanes.
_LANE_PAIRS = 16


@cute.jit
def _intersection(
    px,
    py,
    c00,
    c01,
    c11,
    disc,
    level,
    is_y,
    coord,
    side: cutlass.Constexpr = 0,
    sliced: cutlass.Constexpr = False,
):
    """Ellipse chord at coord with fixed roundings.

    These are the forms of the LiteGS-parity build: bounding-box points and a
    first row's lower line use fma(level, coeff, disc*delta*delta); all other
    slice lines (sliced) use fma(delta, disc*delta, level*coeff). side=-1/+1
    returns one end, fusing -c01*delta into the root's FMA; both ends share
    the rounded product.
    """
    pu, pv, coeff = px, py, c11
    if is_y:
        pu, pv, coeff = py, px, c00
    delta = coord - pu
    radicand = cute.Float32(0)
    if cutlass.const_expr(sliced):
        radicand = h.ffma(delta, h.fmul(disc, delta), h.fmul(level, coeff))
    else:
        radicand = h.ffma(level, coeff, h.fmul(h.fmul(disc, delta), delta))
    root = cute.sqrt(cute.max(radicand, 0.0))
    low, high = cute.Float32(0), cute.Float32(0)
    if cutlass.const_expr(side == 0):
        shared = h.fmul(c01, -delta)
        low, high = h.fsub(shared, root), h.fadd(shared, root)
    elif cutlass.const_expr(side < 0):
        low = h.ffma(-c01, delta, -root)
    else:
        high = h.ffma(-c01, delta, root)
    return low / coeff + pv, high / coeff + pv


@cute.jit
def _ellipse(px, py, c00, c01, c11, alpha, tile_size, tile_height, tiles_x, tiles_y):
    """AccuTile bounds. Rows u in [u0, u1) each intersect tiles [v0, v1) in v."""
    c01_squared = h.fmul(c01, c01)
    disc = h.fsub(c01_squared, h.fmul(c00, c11))
    level = 2.0 * cute.log(alpha * 255.0)
    numerator = h.fmul(c01_squared, -level)
    x_term = cute.sqrt(numerator / h.fmul(c00, disc))
    y_term = cute.sqrt(numerator / h.fmul(c11, disc))
    if c01 >= 0:
        x_term, y_term = -x_term, -y_term
    argmin_x, argmin_y = py - y_term, px - x_term
    argmax_x, argmax_y = py + y_term, px + x_term
    xmin, _ = _intersection(px, py, c00, c01, c11, disc, level, True, argmin_x, -1)
    ymin, _ = _intersection(px, py, c00, c01, c11, disc, level, False, argmin_y, -1)
    _, xmax = _intersection(px, py, c00, c01, c11, disc, level, True, argmax_x, 1)
    _, ymax = _intersection(px, py, c00, c01, c11, disc, level, False, argmax_y, 1)
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
    if (y1 - y0) * (x1 - x0) <= 0:
        u1 = u0
    return (
        disc,
        level,
        is_y,
        block_u,
        block_v,
        u0,
        u1,
        v0,
        v1,
        bbox_u_min,
        bbox_u_max,
        bbox_v_min,
        bbox_v_max,
        arg_min,
        arg_max,
    )


@cute.jit
def _span(ellipse, line, next_line, lower_min, lower_max, upper_min, upper_max):
    """Tiles [first, end) of the row between line and next_line."""
    _, _, _, _, block_v, _, _, v0, v1, _, _, bbox_v_min, bbox_v_max, arg_min, arg_max = ellipse
    ellipse_min = cute.min(lower_min, upper_min)
    ellipse_max = cute.max(lower_max, upper_max)
    if line <= arg_min and arg_min < next_line:
        ellipse_min = bbox_v_min
    if line <= arg_max and arg_max < next_line:
        ellipse_max = bbox_v_max
    first = cute.max(v0, cute.min(v1, cutlass.Int32(ellipse_min / block_v)))
    end = cute.min(v1, cute.max(v0, cutlass.Int32(ellipse_max / block_v + 1)))
    return first, end


@cute.jit
def _row(px, py, c00, c01, c11, ellipse, u):
    """Row u's tile span, computed independently but equal to _process_tiles's."""
    disc, level, is_y, block_u, _, u0, _, _, _, bbox_u_min, bbox_u_max, bbox_v_min, bbox_v_max = (
        ellipse[:13]
    )
    line = cute.Float32(u) * block_u
    next_line = line + block_u
    # A sequential scan reuses each previous upper line as the next lower line.
    lower_min, lower_max = bbox_v_max, bbox_v_min
    if u > u0:
        lower_min, lower_max = _intersection(
            px, py, c00, c01, c11, disc, level, is_y, line, 0, True
        )
    elif bbox_u_min <= line:
        lower_min, lower_max = _intersection(px, py, c00, c01, c11, disc, level, is_y, line)
    upper_min, upper_max = bbox_v_max, bbox_v_min
    if u > u0:
        upper_min, upper_max = lower_min, lower_max
    if next_line <= bbox_u_max:
        upper_min, upper_max = _intersection(
            px, py, c00, c01, c11, disc, level, is_y, next_line, 0, True
        )
    return _span(ellipse, line, next_line, lower_min, lower_max, upper_min, upper_max)


@cute.jit
def _key(is_y, u, v, tiles_x):
    key = v * tiles_x + u
    if is_y:
        key = u * tiles_x + v
    return key


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
    ellipse = _ellipse(px, py, c00, c01, c11, alpha, tile_size, tile_height, tiles_x, tiles_y)
    disc, level, is_y, block_u, _, u0, u1, _, _, bbox_u_min, bbox_u_max, bbox_v_min, bbox_v_max = (
        ellipse[:13]
    )
    count = cutlass.Int32(0)
    if u1 > u0:
        upper_min, upper_max = bbox_v_max, bbox_v_min
        lower_min, lower_max = upper_min, upper_max
        line = cute.Float32(u0) * block_u
        if bbox_u_min <= line:
            lower_min, lower_max = _intersection(px, py, c00, c01, c11, disc, level, is_y, line)
        for u in range(u0, u1):
            next_line = line + block_u
            if next_line <= bbox_u_max:
                upper_min, upper_max = _intersection(
                    px, py, c00, c01, c11, disc, level, is_y, next_line, 0, True
                )
            first, end = _span(ellipse, line, next_line, lower_min, lower_max, upper_min, upper_max)
            if cutlass.const_expr(emit):
                for v in range(first, end):
                    slot = offset + count + v - first
                    if slot < max_pairs:
                        keys[slot] = keys.element_type(_key(is_y, u, v, tiles_x))
                        values[slot] = gid
            count += end - first
            lower_min, lower_max = upper_min, upper_max
            line = next_line
    return count


@cute.jit
def _emit_by_warp(
    mean,
    conic,
    alpha,
    gid,
    offset,
    lane,
    keys,
    values,
    tile_size,
    tile_height,
    tiles_x,
    tiles_y,
    max_pairs,
):
    """All lanes emit one Gaussian: 32 rows at a time, then coalesced pair writes."""
    px, py = mean[gid * 2] - 0.5, mean[gid * 2 + 1] - 0.5
    c00, c01, c11 = conic[gid * 4], conic[gid * 4 + 1], conic[gid * 4 + 3]
    ellipse = _ellipse(px, py, c00, c01, c11, alpha[gid], tile_size, tile_height, tiles_x, tiles_y)
    is_y, u1 = ellipse[2], ellipse[6]
    base = ellipse[5]
    while base < u1:
        u = base + lane
        first, count = cutlass.Int32(0), cutlass.Int32(0)
        if u < u1:
            first, end = _row(px, py, c00, c01, c11, ellipse, u)
            count = end - first
        inclusive = count
        for level in cutlass.range_constexpr(5):
            neighbor = cute.arch.shuffle_sync_up(inclusive, 1 << level)
            if lane >= (1 << level):
                inclusive += neighbor
        before = inclusive - count
        total = cute.arch.shuffle_sync(inclusive, 31)
        chunk = cutlass.Int32(0)
        while chunk < total:
            # All lanes join the shuffles. A pair belongs to the last row
            # starting at or before it.
            pair = chunk + lane
            row = cutlass.Int32(0)
            for level in cutlass.range_constexpr(5):
                step = 16 >> level
                if cute.arch.shuffle_sync(before, row + step) <= pair:
                    row += step
            row_first = cute.arch.shuffle_sync(first, row)
            row_before = cute.arch.shuffle_sync(before, row)
            slot = offset + pair
            if pair < total and slot < max_pairs:
                keys[slot] = keys.element_type(
                    _key(is_y, base + row, row_first + pair - row_before, tiles_x)
                )
                values[slot] = gid
            chunk += 32
        offset += total
        base += 32


@cute.kernel
def _count(
    mean: cute.Tensor,
    conic: cute.Tensor,
    alpha: cute.Tensor,
    visible: cute.Tensor,
    counts: cute.Tensor,
    capacity: int,
    tile_size: cutlass.Constexpr,
    tile_height: cutlass.Constexpr,
    tiles_x: int,
    tiles_y: int,
    width: int,
    height: int,
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
                    0,
                    0,
                    counts,
                    counts,
                    False,
                )
        counts[gid] = count


@cute.kernel
def _emit(
    mean: cute.Tensor,
    conic: cute.Tensor,
    alpha: cute.Tensor,
    depth_order: cute.Tensor,
    ordered_counts: cute.Tensor,
    end_offsets: cute.Tensor,
    keys: cute.Tensor,
    values: cute.Tensor,
    capacity: int,
    tile_size: cutlass.Constexpr,
    tile_height: cutlass.Constexpr,
    tiles_x: int,
    tiles_y: int,
    max_pairs: int,
    lane_pairs: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tid % 32
    rank = block * 256 + tid
    gid, count, offset = cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)
    if rank < capacity:
        count = ordered_counts[rank]
        if count > 0:
            gid = depth_order[rank]
            offset = end_offsets[rank] - count
    # Counted Gaussians already passed the visibility and conic tests.
    if count > 0 and count <= lane_pairs:
        _process_tiles(
            mean[gid * 2] - 0.5,
            mean[gid * 2 + 1] - 0.5,
            conic[gid * 4],
            conic[gid * 4 + 1],
            conic[gid * 4 + 3],
            alpha[gid],
            tile_size,
            tile_height,
            tiles_x,
            tiles_y,
            gid,
            offset,
            max_pairs,
            keys,
            values,
            True,
        )
    pending = cutlass.Uint32(cute.arch.vote_ballot_sync(count > lane_pairs))
    while pending != 0:
        owner = cute.arch.popc((pending & (cutlass.Uint32(0) - pending)) - 1)
        pending &= pending - 1
        _emit_by_warp(
            mean,
            conic,
            alpha,
            cute.arch.shuffle_sync(gid, owner),
            cute.arch.shuffle_sync(offset, owner),
            lane,
            keys,
            values,
            tile_size,
            tile_height,
            tiles_x,
            tiles_y,
            max_pairs,
        )


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
    _count(
        mean,
        conic,
        alpha,
        visible,
        counts,
        capacity,
        tile_size,
        tile_height,
        tiles_x,
        tiles_y,
        width,
        height,
    ).launch(grid=[(capacity + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream)


@cute.jit
def launch_emit_pairs(
    stream: cuda.CUstream,
    mean: cute.Tensor,
    conic: cute.Tensor,
    alpha: cute.Tensor,
    depth_order: cute.Tensor,
    ordered_counts: cute.Tensor,
    end_offsets: cute.Tensor,
    tile_ids: cute.Tensor,
    gaussian_ids: cute.Tensor,
    *,
    capacity: int,
    max_pairs: int,
    tile_size: int,
    tile_height: int,
    tiles_x: int,
    tiles_y: int,
    lane_pairs: cutlass.Constexpr = _LANE_PAIRS,
):
    _clear_pairs(tile_ids, gaussian_ids, max_pairs, tiles_x * tiles_y).launch(
        grid=[(max_pairs + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
    )
    _emit(
        mean,
        conic,
        alpha,
        depth_order,
        ordered_counts,
        end_offsets,
        tile_ids,
        gaussian_ids,
        capacity,
        tile_size,
        tile_height,
        tiles_x,
        tiles_y,
        max_pairs,
        lane_pairs,
    ).launch(grid=[(capacity + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream)
