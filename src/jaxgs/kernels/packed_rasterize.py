"""CuTe translation of LiteGS raster.cu's RGB half2 forward/backward.

Each warp processes one tile (four warps per forward block, one per backward
block). Each lane processes consecutive vertical
pixel pairs using forward differences, with transmittance scaled by 128.
Adapted from LiteGS (see LICENSE.LiteGS).
Unlike the source's uint16 counters, our last indices remain int32.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from . import half2 as h
from .rasterize_backward import _zero_grads
from .sorted_rasterize import _zero_fragments


@cute.jit
def _load_params(params, g):
    # Load LiteGS's complete 32-byte PackedParams into registers together.
    source = cute.make_tensor(params.iterator + g * 8, cute.make_layout(8))
    registers = cute.make_rmem_tensor(8, cutlass.Uint32)
    cute.autovec_copy(source, registers)
    return registers


@cute.kernel
def _pack(
    mean: cute.Tensor,
    conic: cute.Tensor,
    color: cute.Tensor,
    opacity: cute.Tensor,
    params: cute.Tensor,
    capacity: int,
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    g = block * 256 + tid
    if g < capacity:
        point = cute.make_rmem_tensor(8, cutlass.Uint32)
        point[0] = h.float_bits(mean[g * 2] - 0.5)
        point[1] = h.float_bits(mean[g * 2 + 1] - 0.5)
        point[2] = cutlass.Uint32(0)  # LiteGS's unused RGB-path depth field
        point[3] = h.pack(color[g * 3], color[g * 3 + 1])
        point[4] = h.float_bits(conic[g * 4])
        point[5] = h.float_bits(conic[g * 4 + 1])
        point[6] = h.float_bits(conic[g * 4 + 3])
        point[7] = h.pack(color[g * 3 + 2], opacity[g])
        destination = cute.make_tensor(params.iterator + g * 8, cute.make_layout(8))
        cute.autovec_copy(point, destination)


@cute.kernel
def _forward(
    params: cute.Tensor,
    ids: cute.Tensor,
    offsets: cute.Tensor,
    rgb: cute.Tensor,
    final_t: cute.Tensor,
    last: cute.Tensor,
    stats: cute.Tensor,
    width: int,
    height: int,
    tile_size: cutlass.Constexpr,
    tile_height: cutlass.Constexpr,
    tiles_x: int,
    tiles: int,
    collect_stats: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tid % 32
    tile = block * 4 + tid // 32
    groups = tile_size * tile_height // 64
    if tile < tiles:
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
        active = True
        while index < end and cute.arch.vote_any_sync(active):
            g = ids[index]
            point = _load_params(params, g)
            dx = h.bits_float(point[0]) - cute.Float32(px)
            dy = h.bits_float(point[1]) - cute.Float32(py)
            c00 = h.bits_float(point[4])
            c01 = h.bits_float(point[5])
            c11 = h.bits_float(point[6])
            rg, ba = point[3], point[7]
            red, green = h.splat(rg), h.splat(rg, True)
            blue, opacity = h.splat(ba), h.splat(ba, True)
            bxcy = c11 * dy + c01 * dx
            value = -0.5 * (dx * (c00 * dx + c01 * dy) + dy * bxcy)
            diff = bxcy - 0.5 * c11
            active = False
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
                active = active or mask != 0
                if (mask & 0xFFFF) != 0:
                    lst[i, 0] = index - start + 1
                if (mask >> 16) != 0:
                    lst[i, 1] = index - start + 1
                alpha = h.mul(opacity, h.exp(power))
                valid = mask & h.ge_mask(alpha, h.pack(1 / 256, 1 / 256))
                alpha = h.minimum(alpha, h.pack(255 / 256, 255 / 256)) & valid
                weight = h.mul(reg[i, 3], alpha)
                if cutlass.const_expr(collect_stats):
                    fragment_count += cutlass.Int32((valid & 1) + ((valid >> 16) & 1))
                    weight_sum = h.add(weight_sum, weight)
                reg[i, 0] = h.fma(red, weight, reg[i, 0])
                reg[i, 1] = h.fma(green, weight, reg[i, 1])
                reg[i, 2] = h.fma(blue, weight, reg[i, 2])
                reg[i, 3] = h.mul(reg[i, 3], h.sub(h.pack(1, 1), alpha))
            if cutlass.const_expr(collect_stats):
                count = cute.arch.warp_redux_sync(fragment_count, "add")
                (total_weight,) = h.warp_sum_scaled((h.sum_pair(weight_sum) / 128,))
                if lane == 0:
                    cute.arch.atomic_add(stats.iterator + g * 2, cute.Float32(count))
                    cute.arch.atomic_add(stats.iterator + g * 2 + 1, total_weight)
            index += 1
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
    final_t: cute.Tensor,
    last: cute.Tensor,
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
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tid % 32
    tile = block
    groups = tile_size * tile_height // 64
    if tile < tiles:
        px = tile % tiles_x * tile_size + lane % tile_size
        py = tile // tiles_x * tile_height + lane // tile_size * groups * 2
        reg = cute.make_rmem_tensor((groups, 4), cutlass.Uint32)
        grad = cute.make_rmem_tensor((groups, 3), cutlass.Uint32)
        lst = cute.make_rmem_tensor((groups, 2), cutlass.Int32)
        reg.fill(0)
        grad.fill(0)
        start = offsets[tile]
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
        while index >= start:
            g = ids[index]
            point = _load_params(params, g)
            dx = h.bits_float(point[0]) - cute.Float32(px)
            dy = h.bits_float(point[1]) - cute.Float32(py)
            c00 = h.bits_float(point[4])
            c01 = h.bits_float(point[5])
            c11 = h.bits_float(point[6])
            rg, ba = point[3], point[7]
            red, green = h.splat(rg), h.splat(rg, True)
            blue, opacity = h.splat(ba), h.splat(ba, True)
            bxcy = c11 * dy + c01 * dx
            value = -0.5 * (dx * (c00 * dx + c01 * dy) + dy * bxcy)
            diff = bxcy - 0.5 * c11
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
                if cute.arch.vote_any_sync(mask != 0):
                    contributes = True
                    alpha &= mask
                    gaussian &= mask
                    trans = h.minimum(
                        h.pack(128, 128), h.mul(reg[i, 3], h.reciprocal(h.sub(h.pack(1, 1), alpha)))
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
                    offset = h.pack(i * 2, i * 2 + 1)
                    basic += h.sum_pair(dp)
                    linear += h.sum_pair(h.mul(dp, offset))
                    quadratic += h.sum_pair(h.mul(h.mul(dp, offset), offset))
            # LiteGS skips the reductions/atomics for noncontributing splats.
            # Test fragment validity so arbitrary RGB cotangents also work when
            # their opacity gradient cancels but their color gradient does not.
            if contributes:
                norm = scale / 128
                gx, gy = h.warp_sum_scaled(
                    (
                        -(c00 * dx + c01 * dy) * basic + c01 * linear,
                        -(c11 * dy + c01 * dx) * basic + c11 * linear,
                    )
                )
                gc00, gc01, gc11 = h.warp_sum_scaled(
                    (
                        -0.5 * dx * dx * basic,
                        (-dx * dy * basic + dx * linear) * 0.5,
                        -0.5 * dy * dy * basic + dy * linear - 0.5 * quadratic,
                    )
                )
                gx, gy = gx * norm, gy * norm
                gc00, gc01, gc11 = gc00 * norm, gc01 * norm, gc11 * norm
                # Native half2 reduction keeps RG and BA paired across the warp.
                rg_sum = h.warp_sum(h.pack(h.sum_pair(gr), h.sum_pair(gg)))
                ba_sum = h.warp_sum(h.pack(h.sum_pair(gb), h.sum_pair(ga)))
                if lane == 0:
                    cute.arch.atomic_add(gmean.iterator + g * 2, gx)
                    cute.arch.atomic_add(gmean.iterator + g * 2 + 1, gy)
                    cute.arch.atomic_add(gconic.iterator + g * 4, gc00)
                    if cutlass.const_expr(symmetric_conic):
                        cute.arch.atomic_add(gconic.iterator + g * 4 + 1, gc01 * 2)
                    else:
                        cute.arch.atomic_add(gconic.iterator + g * 4 + 1, gc01)
                        cute.arch.atomic_add(gconic.iterator + g * 4 + 2, gc01)
                    cute.arch.atomic_add(gconic.iterator + g * 4 + 3, gc11)
                    cute.arch.atomic_add(gcolor.iterator + g * 3, h.get(rg_sum) * norm)
                    cute.arch.atomic_add(gcolor.iterator + g * 3 + 1, h.get(rg_sum, True) * norm)
                    cute.arch.atomic_add(gcolor.iterator + g * 3 + 2, h.get(ba_sum) * norm)
                    cute.arch.atomic_add(gopacity.iterator + g, h.get(ba_sum, True) * norm)
                if cutlass.const_expr(collect_stats):
                    (error,) = h.warp_sum_scaled((h.sum_pair(err) / 128,))
                    if lane == 0:
                        cute.arch.atomic_add(square_error.iterator + g, error)
            index -= 1


@cute.jit
def launch_forward(
    stream: cuda.CUstream,
    mean: cute.Tensor,
    conic: cute.Tensor,
    color: cute.Tensor,
    opacity: cute.Tensor,
    ids: cute.Tensor,
    offsets: cute.Tensor,
    params: cute.Tensor,
    rgb: cute.Tensor,
    final_t: cute.Tensor,
    last: cute.Tensor,
    stats: cute.Tensor,
    *,
    width: int,
    height: int,
    tile_size: cutlass.Constexpr,
    tile_height: cutlass.Constexpr,
    capacity: int,
    collect_stats: cutlass.Constexpr,
):
    tiles_x = (width + tile_size - 1) // tile_size
    tiles = tiles_x * ((height + tile_height - 1) // tile_height)
    _pack(mean, conic, color, opacity, params, capacity).launch(
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
        rgb,
        final_t,
        last,
        stats,
        width,
        height,
        tile_size,
        tile_height,
        tiles_x,
        tiles,
        collect_stats,
    ).launch(grid=[(tiles + 3) // 4, 1, 1], block=[128, 1, 1], stream=stream)


@cute.jit
def launch_backward(
    stream: cuda.CUstream,
    params: cute.Tensor,
    ids: cute.Tensor,
    offsets: cute.Tensor,
    final_t: cute.Tensor,
    last: cute.Tensor,
    image_grad: cute.Tensor,
    grad_scale: cute.Tensor,
    gmean: cute.Tensor,
    gconic: cute.Tensor,
    gdepth: cute.Tensor,
    gcolor: cute.Tensor,
    gopacity: cute.Tensor,
    square_error: cute.Tensor,
    *,
    width: int,
    height: int,
    tile_size: cutlass.Constexpr,
    tile_height: cutlass.Constexpr,
    capacity: int,
    collect_stats: cutlass.Constexpr,
    symmetric_conic: cutlass.Constexpr = False,
):
    tiles_x = (width + tile_size - 1) // tile_size
    tiles = tiles_x * ((height + tile_height - 1) // tile_height)
    _zero_grads(gmean, gconic, gdepth, gcolor, gopacity, capacity).launch(
        grid=[(capacity * 4 + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
    )
    if cutlass.const_expr(collect_stats):
        _zero_fragments(square_error, capacity).launch(
            grid=[(capacity + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
        )
    _backward(
        params,
        ids,
        offsets,
        final_t,
        last,
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
