"""CuTe translation of LiteGS raster.cu's RGB half2 forward/backward.

Each single-warp block processes one tile. Each lane processes consecutive
vertical pixel pairs using forward differences, with transmittance scaled by 128.
Adapted from LiteGS (see LICENSE.LiteGS).
Unlike the source's uint16 counters, our last indices remain int32.

Tiles launch heaviest first: a one-block counting sort orders them by pair
count (forward) or by the pairs before their last contributor (backward).
A warp stages 32 splats at a time in shared memory; each lane loads the
next batch's parameters into registers while the current batch composites,
and the forward loop composites two splats per iteration. Training records
one contribution bit per pair; backward visits only set bits in reverse order.
These changes preserve each pixel's compositing order and arithmetic.

A splat's warp sums (backward gradients, forward statistics) share one
transposed shuffle butterfly (half2.warp_sums), after which each lane adds
one sum with a single atomic instruction for the warp. The float sums round
differently from LiteGS's shared-exponent integer reductions.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.memory import SmemAllocator

from . import half2 as h
from .rasterize_backward import _zero_grads
from .sorted_rasterize import _zero_fragments

_ORDER_THREADS = 1024
_ORDER_BUCKETS = 256
# Splats per forward loop iteration; divides the 32-splat staging batch.
_FORWARD_UNROLL = 2


@cute.jit
def _work_bucket(work, tile, from_offsets: cutlass.Constexpr):
    # Eight log2 buckets per octave, heaviest first.
    count = work[tile]
    if cutlass.const_expr(from_offsets):
        count = work[tile + 1] - count
    bits = h.float_bits(cute.Float32(cute.max(count, 0) + 1))
    return _ORDER_BUCKETS - 1 - cutlass.Int32((bits >> 20) - (127 << 3))


@cute.kernel
def _order_tiles(
    work: cute.Tensor, order: cute.Tensor, tiles: int, from_offsets: cutlass.Constexpr
):
    """Counting sort of tile IDs; order within a bucket is unspecified."""
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 32
    starts = SmemAllocator().allocate_tensor(cutlass.Int32, cute.make_layout(_ORDER_BUCKETS))
    if tid < _ORDER_BUCKETS:
        starts[tid] = cutlass.Int32(0)
    cute.arch.sync_threads()
    tile = tid
    while tile < tiles:
        cute.arch.atomic_add(
            starts.iterator + _work_bucket(work, tile, from_offsets), cutlass.Int32(1)
        )
        tile += _ORDER_THREADS
    cute.arch.sync_threads()
    if tid < 32:
        # Exclusive scan; each lane owns consecutive buckets.
        per_lane = _ORDER_BUCKETS // 32
        total = cutlass.Int32(0)
        for i in cutlass.range_constexpr(per_lane):
            total += starts[lane * per_lane + i]
        inclusive = total
        for level in cutlass.range_constexpr(5):
            neighbor = cute.arch.shuffle_sync_up(inclusive, 1 << level)
            if lane >= (1 << level):
                inclusive += neighbor
        running = inclusive - total
        for i in cutlass.range_constexpr(per_lane):
            count = starts[lane * per_lane + i]
            starts[lane * per_lane + i] = running
            running += count
    cute.arch.sync_threads()
    tile = tid
    while tile < tiles:
        position = cute.arch.atomic_add(
            starts.iterator + _work_bucket(work, tile, from_offsets), cutlass.Int32(1)
        )
        order[position] = tile
        tile += _ORDER_THREADS


@cute.jit
def _load_params(params, g, registers):
    # LiteGS's 32-byte PackedParams as two 128-bit loads.
    source = cute.make_tensor(params.iterator + cute.assume(g * 8, divby=4), cute.make_layout(8))
    cute.autovec_copy(source, registers)


@cute.jit
def _staging():
    allocator = SmemAllocator()
    staged = allocator.allocate_tensor(
        cutlass.Uint32, cute.make_layout((32, 8), stride=(8, 1)), byte_alignment=16
    )
    return staged, allocator.allocate_tensor(cutlass.Int32, cute.make_layout(32))


@cute.jit
def _stage(staged, staged_ids, lane, registers, g):
    cute.autovec_copy(registers, staged[lane, None])
    staged_ids[lane] = g


@cute.jit
def _unstage(staged, staged_ids, k, registers):
    cute.autovec_copy(staged[k, None], registers)
    return staged_ids[k]


@cute.jit
def _power(dx, dy, c00, c01, c11):
    # LiteGS's exponent and first vertical difference. Explicit FMAs keep the
    # forward and backward roundings identical under any instruction schedule.
    bxcy = h.ffma(c01, dx, h.fmul(c11, dy))
    axby = h.ffma(c00, dx, h.fmul(c01, dy))
    value = h.fmul(h.ffma(dx, axby, h.fmul(dy, bxcy)), -0.5)
    return value, h.ffma(c11, -0.5, bxcy)


@cute.jit
def _gradient_target(
    lane,
    norm,
    gmean,
    gconic,
    gcolor,
    gopacity,
    square_error,
    collect_stats: cutlass.Constexpr,
    symmetric_conic: cutlass.Constexpr,
):
    """Where this lane adds a backward sum: (address, stride, factor, source).

    After h.warp_sums, lane l holds geometry sum l >> 2 (x, y, c00, c01, c11,
    squared error); lanes 0-15 hold the RG sums and lanes 16-31 the BA sums.
    Source 0 adds the geometry sum, 1/2 a low/high color half, -1 nothing.
    The address points to Gaussian 0's entry; Gaussian g is stride * g later.
    """
    geometry, part = lane >> 2, lane & 3
    address = gmean.iterator.toint()
    stride, factor, source = cutlass.Int32(0), norm, cutlass.Int32(-1)
    if part == 0 and geometry < 2:
        address, stride, source = (gmean.iterator + geometry).toint(), 2, 0
    if part == 0 and geometry >= 2 and geometry < 5:
        address, stride, source = (gconic.iterator + geometry - 2).toint(), 4, 0
        if geometry == 4:
            address = (gconic.iterator + 3).toint()
        if cutlass.const_expr(symmetric_conic):
            if geometry == 3:
                factor = norm * 2
    if cutlass.const_expr(not symmetric_conic):
        if lane == 13:
            address, stride, source = (gconic.iterator + 2).toint(), 4, 0
    if cutlass.const_expr(collect_stats):
        if lane == 20:
            address, stride, factor, source = square_error.iterator.toint(), 1, 1.0, 0
    if lane == 1 or lane == 2:
        address, stride, source = (gcolor.iterator + lane - 1).toint(), 3, lane
    if lane == 17:
        address, stride, source = (gcolor.iterator + 2).toint(), 3, 1
    if lane == 18:
        address, stride, source = gopacity.iterator.toint(), 1, 2
    return address, stride, factor, source


@cute.kernel
def _pack(
    mean: cute.Tensor,
    conic: cute.Tensor,
    color: cute.Tensor,
    opacity: cute.Tensor,
    visible: cute.Tensor,
    params: cute.Tensor,
    capacity: int,
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    g = block * 256 + tid
    # Binning emits pairs only for visible Gaussians; skip the others' records.
    if g < capacity and visible[g] != 0:
        point = cute.make_rmem_tensor(8, cutlass.Uint32)
        point[0] = h.float_bits(mean[g * 2] - 0.5)
        point[1] = h.float_bits(mean[g * 2 + 1] - 0.5)
        point[2] = cutlass.Uint32(0)  # LiteGS's unused RGB-path depth field
        point[3] = h.pack(color[g * 3], color[g * 3 + 1])
        point[4] = h.float_bits(conic[g * 4])
        point[5] = h.float_bits(conic[g * 4 + 1])
        point[6] = h.float_bits(conic[g * 4 + 3])
        point[7] = h.pack(color[g * 3 + 2], opacity[g])
        destination = cute.make_tensor(
            params.iterator + cute.assume(g * 8, divby=4), cute.make_layout(8)
        )
        cute.autovec_copy(point, destination)


@cute.jit
def _composite(
    point,
    px,
    py,
    width,
    height,
    reg,
    lst,
    local,
    present,
    groups: cutlass.Constexpr,
    collect_stats: cutlass.Constexpr,
    record_contributions: cutlass.Constexpr,
):
    """Composite one splat; report active pixels, statistics and fragment validity."""
    dx = h.bits_float(point[0]) - cute.Float32(px)
    dy = h.bits_float(point[1]) - cute.Float32(py)
    c00 = h.bits_float(point[4])
    c01 = h.bits_float(point[5])
    c11 = h.bits_float(point[6])
    rg, ba = point[3], point[7]
    red, green = h.splat(rg), h.splat(rg, True)
    blue, opacity = h.splat(ba), h.splat(ba, True)
    value, diff = _power(dx, dy, c00, c01, c11)
    active = False
    contributing = False
    fragment_count = cutlass.Int32(0)
    weight_sum = cutlass.Uint32(0)
    for i in cutlass.range_constexpr(groups):
        v0 = value
        value += diff
        diff -= c11
        power = h.pack(v0, value)
        value += diff
        diff -= c11
        mask = h.gt_mask(reg[i, 3], h.pack(128 / 8192, 128 / 8192))
        # Native kernels render padded tiles; suppress outside-image fragments.
        if px >= width or py + i * 2 >= height:
            mask &= cutlass.Uint32(0xFFFF0000)
        if px >= width or py + i * 2 + 1 >= height:
            mask &= cutlass.Uint32(0x0000FFFF)
        if not present:
            mask = cutlass.Uint32(0)
        active = active or mask != 0
        if (mask & 0xFFFF) != 0:
            lst[i, 0] = local
        if (mask >> 16) != 0:
            lst[i, 1] = local
        alpha = h.mul(opacity, h.exp(power))
        valid = mask & h.ge_mask(alpha, h.pack(1 / 256, 1 / 256))
        if cutlass.const_expr(record_contributions):
            contributing = contributing or valid != 0
        alpha = h.minimum(alpha, h.pack(255 / 256, 255 / 256)) & valid
        weight = h.mul(reg[i, 3], alpha)
        if cutlass.const_expr(collect_stats):
            fragment_count += cutlass.Int32((valid & 1) + ((valid >> 16) & 1))
            weight_sum = h.add(weight_sum, weight)
        reg[i, 0] = h.fma(red, weight, reg[i, 0])
        reg[i, 1] = h.fma(green, weight, reg[i, 1])
        reg[i, 2] = h.fma(blue, weight, reg[i, 2])
        reg[i, 3] = h.mul(reg[i, 3], h.sub(h.pack(1, 1), alpha))
    return active, fragment_count, weight_sum, contributing


@cute.kernel
def _forward(
    params: cute.Tensor,
    ids: cute.Tensor,
    offsets: cute.Tensor,
    order: cute.Tensor,
    rgb: cute.Tensor,
    final_t: cute.Tensor,
    last: cute.Tensor,
    stats: cute.Tensor,
    backward_work: cute.Tensor,
    contribution_bits: cute.Tensor,
    width: int,
    height: int,
    tile_size: cutlass.Constexpr,
    tile_height: cutlass.Constexpr,
    tiles_x: int,
    tiles: int,
    collect_stats: cutlass.Constexpr,
    record_contributions: cutlass.Constexpr,
):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    groups = tile_size * tile_height // 64
    if block < tiles:
        tile = order[block]
        px = tile % tiles_x * tile_size + lane % tile_size
        py = tile // tiles_x * tile_height + lane // tile_size * groups * 2
        # r,g,b,t are packed half2; last indices use separate int32 registers.
        reg = cute.make_rmem_tensor((groups, 4), cutlass.Uint32)
        lst = cute.make_rmem_tensor((groups, 2), cutlass.Int32)
        reg.fill(0)
        lst.fill(0)
        for i in cutlass.range_constexpr(groups):
            reg[i, 3] = h.pack(128.0, 128.0)
        start, end = offsets[tile], offsets[tile + 1]
        index = start
        # One padding word per tile makes adjacent unaligned segments disjoint.
        bit_base = start // 32 + tile
        bits = cutlass.Uint32(0)
        active = True
        staged, staged_ids = _staging()
        points = cute.make_rmem_tensor((_FORWARD_UNROLL, 8), cutlass.Uint32)
        mine = cute.make_rmem_tensor(8, cutlass.Uint32)
        mine.fill(0)
        my_id = cutlass.Int32(0)
        if start + lane < end:
            my_id = ids[start + lane]
            _load_params(params, my_id, mine)
        while index < end and cute.arch.vote_any_sync(active):
            k = (index - start) % 32
            if k == 0:
                if cutlass.const_expr(record_contributions):
                    if index > start and lane == 0:
                        contribution_bits[bit_base + (index - start) // 32 - 1] = bits
                    bits = cutlass.Uint32(0)
                cute.arch.sync_warp()
                _stage(staged, staged_ids, lane, mine, my_id)
                upcoming = index + 32 + lane
                if upcoming < end:
                    my_id = ids[upcoming]
                    _load_params(params, my_id, mine)
                cute.arch.sync_warp()
            # Consecutive splats' exponentials overlap; only T is sequential.
            # A splat past the end is masked, and staged words stay finite.
            # Each splat's fragment count and weight sum, and its Gaussian ID.
            sums, splat_ids = [], []
            seen = cutlass.Int32(0)
            for u in cutlass.range_constexpr(_FORWARD_UNROLL):
                g = _unstage(staged, staged_ids, k + u, points[u, None])
                present = index + u < end
                active, fragments, weights, contributing = _composite(
                    points[u, None],
                    px,
                    py,
                    width,
                    height,
                    reg,
                    lst,
                    index - start + 1 + u,
                    present,
                    groups,
                    collect_stats,
                    record_contributions,
                )
                if cutlass.const_expr(record_contributions):
                    if cute.arch.vote_any_sync(contributing):
                        bits |= cutlass.Uint32(1) << (k + u)
                if cutlass.const_expr(collect_stats):
                    sums += [cute.Float32(fragments), h.sum_pair(weights) / 128]
                    splat_ids.append(g)
                    seen |= fragments
            if cutlass.const_expr(collect_stats):
                # Splats without valid fragments would only add zeros.
                if cute.arch.vote_any_sync(seen != 0):
                    total = h.warp_sums(sums, lane)
                    stat = lane >> (6 - len(sums).bit_length())
                    g = splat_ids[0]
                    for u in cutlass.range_constexpr(1, _FORWARD_UNROLL):
                        if stat >> 1 == u:
                            g = splat_ids[u]
                    if lane % (32 // len(sums)) == 0 and total != 0:
                        cute.arch.atomic_add(stats.iterator + g * 2 + (stat & 1), total)
            index += _FORWARD_UNROLL
        if cutlass.const_expr(record_contributions):
            if lane == 0 and index > start:
                contribution_bits[bit_base + (index - start - 1) // 32] = bits
        # Last indices bound the initialized bitmap prefix consumed by backward.
        tile_last = cutlass.Int32(0)
        for i in cutlass.range_constexpr(groups):
            tile_last = cute.max(tile_last, cute.max(lst[i, 0], lst[i, 1]))
        tile_last = cute.arch.warp_redux_sync(tile_last, "max")
        if lane == 0:
            backward_work[tile] = tile_last
        for i in cutlass.range_constexpr(groups):
            for k in cutlass.range_constexpr(2):
                y = py + i * 2 + k
                if px < width and y < height:
                    p = y * width + px
                    for c in cutlass.range_constexpr(3):
                        rgb[p * 3 + c] = cute.min(h.get(reg[i, c], k == 1) / 128, cute.Float32(1))
                    final_t[p] = h.get(reg[i, 3], k == 1) / 128
                    last[p] = start + lst[i, k]


@cute.kernel
def _backward(
    params: cute.Tensor,
    ids: cute.Tensor,
    offsets: cute.Tensor,
    order: cute.Tensor,
    final_t: cute.Tensor,
    last: cute.Tensor,
    contribution_bits: cute.Tensor,
    image_grad: cute.Tensor,
    grad_scale: cute.Tensor,
    gmean: cute.Tensor,
    gconic: cute.Tensor,
    gcolor: cute.Tensor,
    gopacity: cute.Tensor,
    square_error: cute.Tensor,
    width: int,
    height: int,
    tile_size: cutlass.Constexpr,
    tile_height: cutlass.Constexpr,
    tiles_x: int,
    tiles: int,
    collect_stats: cutlass.Constexpr,
    symmetric_conic: cutlass.Constexpr,
):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    groups = tile_size * tile_height // 64
    if block < tiles:
        tile = order[block]
        px = tile % tiles_x * tile_size + lane % tile_size
        py = tile // tiles_x * tile_height + lane // tile_size * groups * 2
        reg = cute.make_rmem_tensor((groups, 4), cutlass.Uint32)
        grad = cute.make_rmem_tensor((groups, 3), cutlass.Uint32)
        lst = cute.make_rmem_tensor((groups, 2), cutlass.Int32)
        reg.fill(0)
        grad.fill(0)
        start = offsets[tile]
        bit_base = start // 32 + tile
        bits = cutlass.Uint32(0)
        index = start
        scale = grad_scale[0]
        inv_scale = cute.Float32(1) / scale
        for i in cutlass.range_constexpr(groups):
            t0, t1 = cute.Float32(0), cute.Float32(0)
            lst[i, 0], lst[i, 1] = start, start
            if px < width and py + i * 2 < height:
                p = (py + i * 2) * width + px
                t0 = final_t[p] * 128
                lst[i, 0] = last[p]
            if px < width and py + i * 2 + 1 < height:
                p = (py + i * 2 + 1) * width + px
                t1 = final_t[p] * 128
                lst[i, 1] = last[p]
            reg[i, 3] = h.pack(t0, t1)
            index = cute.max(index, cute.max(lst[i, 0], lst[i, 1]))
            for c in cutlass.range_constexpr(3):
                v0, v1 = cute.Float32(0), cute.Float32(0)
                if px < width and py + i * 2 < height:
                    v0 = image_grad[((py + i * 2) * width + px) * 3 + c] * inv_scale
                if px < width and py + i * 2 + 1 < height:
                    v1 = image_grad[((py + i * 2 + 1) * width + px) * 3 + c] * inv_scale
                grad[i, c] = h.pack(v0, v1)
        index = cute.arch.warp_redux_sync(index, "max") - 1
        top = index
        address, stride, factor, source = _gradient_target(
            lane,
            scale / 128,
            gmean,
            gconic,
            gcolor,
            gopacity,
            square_error,
            collect_stats,
            symmetric_conic,
        )
        staged, staged_ids = _staging()
        point = cute.make_rmem_tensor(8, cutlass.Uint32)
        mine = cute.make_rmem_tensor(8, cutlass.Uint32)
        mine.fill(0)
        my_id = cutlass.Int32(0)
        chunk = (top - start) // 32
        if top >= start:
            bits = cutlass.Uint32(contribution_bits[bit_base + chunk])
            # The forward unroll may inspect pairs past every pixel's last index.
            bits &= cutlass.Uint32(0xFFFFFFFF) >> (31 - (top - start) % 32)
        if (bits & (cutlass.Uint32(1) << lane)) != 0:
            my_id = ids[start + chunk * 32 + lane]
            _load_params(params, my_id, mine)
        while index >= start:
            if bits != 0:
                cute.arch.sync_warp()
                _stage(staged, staged_ids, lane, mine, my_id)
                cute.arch.sync_warp()
            next_bits = cutlass.Uint32(0)
            if chunk > 0:
                next_bits = cutlass.Uint32(contribution_bits[bit_base + chunk - 1])
            if (next_bits & (cutlass.Uint32(1) << lane)) != 0:
                my_id = ids[start + (chunk - 1) * 32 + lane]
                _load_params(params, my_id, mine)
            while bits != 0:
                # Highest set bit preserves reverse compositing and half2 reductions.
                k = cutlass.Int32(cute.arch.bfind(bits))
                index = start + chunk * 32 + k
                g = _unstage(staged, staged_ids, k, point)
                dx = h.bits_float(point[0]) - cute.Float32(px)
                dy = h.bits_float(point[1]) - cute.Float32(py)
                c00 = h.bits_float(point[4])
                c01 = h.bits_float(point[5])
                c11 = h.bits_float(point[6])
                rg, ba = point[3], point[7]
                red, green = h.splat(rg), h.splat(rg, True)
                blue, opacity = h.splat(ba), h.splat(ba, True)
                value, diff = _power(dx, dy, c00, c01, c11)
                gr, gg, gb, ga, err = (
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                    cutlass.Uint32(0),
                )
                contributes = False
                basic, linear, quadratic = cute.Float32(0), cute.Float32(0), cute.Float32(0)
                for i in cutlass.range_constexpr(groups):
                    v0 = value
                    value += diff
                    diff -= c11
                    power = h.pack(v0, value)
                    value += diff
                    diff -= c11
                    gaussian = h.exp(power)
                    alpha = h.minimum(h.mul(opacity, gaussian), h.pack(255 / 256, 255 / 256))
                    mask = h.ge_mask(alpha, h.pack(1 / 256, 1 / 256))
                    if index >= lst[i, 0]:
                        mask &= cutlass.Uint32(0xFFFF0000)
                    if index >= lst[i, 1]:
                        mask &= cutlass.Uint32(0x0000FFFF)
                    # Masked fragments leave ordinary gradient sums unchanged.
                    # Statistics must skip empty groups: their squared-error
                    # term depends on the opacity gradient accumulated so far.
                    if not collect_stats or cute.arch.vote_any_sync(mask != 0):
                        contributes = True
                        alpha &= mask
                        gaussian &= mask
                        trans = h.minimum(
                            h.pack(128, 128),
                            h.mul(reg[i, 3], h.reciprocal(h.sub(h.pack(1, 1), alpha))),
                        )
                        reg[i, 3] = trans
                        weight = h.mul(alpha, trans)
                        gr = h.fma(weight, grad[i, 0], gr)
                        gg = h.fma(weight, grad[i, 1], gg)
                        gb = h.fma(weight, grad[i, 2], gb)
                        da = h.mul(h.mul(h.sub(red, reg[i, 0]), trans), grad[i, 0])
                        da = h.add(da, h.mul(h.mul(h.sub(green, reg[i, 1]), trans), grad[i, 1]))
                        da = h.add(da, h.mul(h.mul(h.sub(blue, reg[i, 2]), trans), grad[i, 2]))
                        reg[i, 0] = h.fma(alpha, h.sub(red, reg[i, 0]), reg[i, 0])
                        reg[i, 1] = h.fma(alpha, h.sub(green, reg[i, 1]), reg[i, 1])
                        reg[i, 2] = h.fma(alpha, h.sub(blue, reg[i, 2]), reg[i, 2])
                        ga = h.fma(da, gaussian, ga)
                        dp = h.mul(gaussian, h.mul(opacity, da))
                        if cutlass.const_expr(collect_stats):
                            # Source accumulates squared partial sums within each lane.
                            err = h.add(err, h.mul(h.mul(ga, h.pack(1 / 128, 1 / 128)), ga))
                        basic += h.sum_pair(dp)
                        if cutlass.const_expr(i == 0):
                            # Offsets (0, 1) give the same first and second moments.
                            upper = h.get(dp, True)
                            linear += upper
                            quadratic += upper
                        else:
                            offset = h.pack(i * 2, i * 2 + 1)
                            linear += h.sum_pair(h.mul(dp, offset))
                            quadratic += h.sum_pair(h.mul(h.mul(dp, offset), offset))
                # LiteGS skips the reductions/atomics for noncontributing splats.
                # Test fragment validity so arbitrary RGB cotangents also work when
                # their opacity gradient cancels but their color gradient does not.
                if contributes:
                    total = h.warp_sums(
                        (
                            -(c00 * dx + c01 * dy) * basic + c01 * linear,
                            -(c11 * dy + c01 * dx) * basic + c11 * linear,
                            -0.5 * dx * dx * basic,
                            (-dx * dy * basic + dx * linear) * 0.5,
                            -0.5 * dy * dy * basic + dy * linear - 0.5 * quadratic,
                            h.sum_pair(err) / 128,
                            cute.Float32(0),
                            cute.Float32(0),
                        ),
                        lane,
                    )
                    # Native half2 sums keep RG and BA paired across the warp.
                    colors = h.warp_sums(
                        (
                            h.pack_pair_sums(gr, gg),
                            h.pack_pair_sums(gb, ga),
                        ),
                        lane,
                        True,
                    )
                    if source == 1:
                        total = h.get(colors)
                    if source == 2:
                        total = h.get(colors, True)
                    if source >= 0:
                        target = cute.make_ptr(
                            cutlass.Float32,
                            address + cutlass.Int64(g * stride) * 4,
                            cute.AddressSpace.gmem,
                        )
                        cute.arch.atomic_add(target, total * factor)
                bits &= ~(cutlass.Uint32(1) << k)
            index = start + chunk * 32 - 1
            chunk -= 1
            bits = next_bits


@cute.kernel
def _zero_cluster_grads(
    cluster_ids: cute.Tensor,
    cluster_count: cute.Tensor,
    gmean: cute.Tensor,
    gconic: cute.Tensor,
    gcolor: cute.Tensor,
    gopacity: cute.Tensor,
    capacity: int,
    cluster_size: cutlass.Constexpr,
):
    """Zero the cotangents of one listed cluster per 128-thread block."""
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    if block < cluster_count[0]:
        first = cluster_ids[block] * cluster_size
        slot = tid
        while slot < cluster_size and first + slot < capacity:
            g = first + slot
            for k in cutlass.range_constexpr(2):
                gmean[g * 2 + k] = cute.Float32(0)
            for k in cutlass.range_constexpr(4):
                gconic[g * 4 + k] = cute.Float32(0)
            for k in cutlass.range_constexpr(3):
                gcolor[g * 3 + k] = cute.Float32(0)
            gopacity[g] = cute.Float32(0)
            slot += 128


@cute.jit
def launch_tile_order(
    stream: cuda.CUstream,
    work: cute.Tensor,
    order: cute.Tensor,
    *,
    tiles: int,
    from_offsets: cutlass.Constexpr,
):
    """Tile IDs by descending work: offsets[t + 1] - offsets[t], or work[t]."""
    _order_tiles(work, order, tiles, from_offsets).launch(
        grid=[1, 1, 1], block=[_ORDER_THREADS, 1, 1], stream=stream
    )


@cute.jit
def launch_forward(
    stream: cuda.CUstream,
    mean: cute.Tensor,
    conic: cute.Tensor,
    color: cute.Tensor,
    opacity: cute.Tensor,
    visible: cute.Tensor,
    ids: cute.Tensor,
    offsets: cute.Tensor,
    params: cute.Tensor,
    rgb: cute.Tensor,
    final_t: cute.Tensor,
    last: cute.Tensor,
    stats: cute.Tensor,
    backward_work: cute.Tensor,
    order: cute.Tensor,
    contribution_bits: cute.Tensor,
    *,
    width: int,
    height: int,
    tile_size: cutlass.Constexpr,
    tile_height: cutlass.Constexpr,
    capacity: int,
    collect_stats: cutlass.Constexpr,
    record_contributions: cutlass.Constexpr,
):
    tiles_x = (width + tile_size - 1) // tile_size
    tiles = tiles_x * ((height + tile_height - 1) // tile_height)
    launch_tile_order(stream, offsets, order, tiles=tiles, from_offsets=True)
    _pack(mean, conic, color, opacity, visible, params, capacity).launch(
        grid=[(capacity + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
    )
    if cutlass.const_expr(collect_stats):
        _zero_fragments(stats, capacity * 2).launch(
            grid=[(capacity * 2 + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
        )
    _forward(
        params,
        ids,
        offsets,
        order,
        rgb,
        final_t,
        last,
        stats,
        backward_work,
        contribution_bits,
        width,
        height,
        tile_size,
        tile_height,
        tiles_x,
        tiles,
        collect_stats,
        record_contributions,
    ).launch(grid=[tiles, 1, 1], block=[32, 1, 1], stream=stream)


@cute.jit
def launch_backward(
    stream: cuda.CUstream,
    params: cute.Tensor,
    ids: cute.Tensor,
    offsets: cute.Tensor,
    final_t: cute.Tensor,
    last: cute.Tensor,
    contribution_bits: cute.Tensor,
    backward_work: cute.Tensor,
    image_grad: cute.Tensor,
    grad_scale: cute.Tensor,
    cluster_ids: cute.Tensor,
    cluster_count: cute.Tensor,
    gmean: cute.Tensor,
    gconic: cute.Tensor,
    gdepth: cute.Tensor,
    gcolor: cute.Tensor,
    gopacity: cute.Tensor,
    square_error: cute.Tensor,
    order: cute.Tensor,
    *,
    width: int,
    height: int,
    tile_size: cutlass.Constexpr,
    tile_height: cutlass.Constexpr,
    capacity: int,
    collect_stats: cutlass.Constexpr,
    symmetric_conic: cutlass.Constexpr = False,
    cluster_size: cutlass.Constexpr = 0,
):
    """Zero the projected-field cotangents, then accumulate them per tile.

    With cluster_size, only the listed clusters' cotangents are zeroed: the
    table must reference no other Gaussian. Depth cotangents then stay
    unwritten, and opacity cotangents are zeroed everywhere for statistics.
    """
    tiles_x = (width + tile_size - 1) // tile_size
    tiles = tiles_x * ((height + tile_height - 1) // tile_height)
    launch_tile_order(stream, backward_work, order, tiles=tiles, from_offsets=False)
    if cutlass.const_expr(cluster_size == 0):
        _zero_grads(gmean, gconic, gdepth, gcolor, gopacity, capacity).launch(
            grid=[(capacity * 4 + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
        )
    else:
        _zero_cluster_grads(
            cluster_ids, cluster_count, gmean, gconic, gcolor, gopacity, capacity, cluster_size
        ).launch(
            grid=[(capacity + cluster_size - 1) // cluster_size, 1, 1],
            block=[128, 1, 1],
            stream=stream,
        )
        if cutlass.const_expr(collect_stats):
            _zero_fragments(gopacity, capacity).launch(
                grid=[(capacity + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
            )
    if cutlass.const_expr(collect_stats):
        _zero_fragments(square_error, capacity).launch(
            grid=[(capacity + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
        )
    _backward(
        params,
        ids,
        offsets,
        order,
        final_t,
        last,
        contribution_bits,
        image_grad,
        grad_scale,
        gmean,
        gconic,
        gcolor,
        gopacity,
        square_error,
        width,
        height,
        tile_size,
        tile_height,
        tiles_x,
        tiles,
        collect_stats,
        symmetric_conic,
    ).launch(grid=[tiles, 1, 1], block=[32, 1, 1], stream=stream)
