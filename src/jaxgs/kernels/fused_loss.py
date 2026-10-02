"""CuTe port of LiteGS fused_ssim/ssim.cu's L1 + SSIM loss and image gradient.

The source's MIT notice is in LICENSE.fused_ssim. Images use HWC layout.

The source runs two kernels: one writes per-pixel losses and SSIM partial
derivatives, the other blurs those partials into the image gradient. Here one
block computes a 32x16 output tile end to end. It recomputes the window
statistics over the 5-pixel halo its gradient needs, so neither the loss map
nor the partials reach global memory. Per channel, the block

1. copies the image and target tiles (36x52, zero padded) to shared memory,
   asynchronously while the previous channel finishes;
2. sums x, x^2, y, y^2 and xy horizontally (36x42);
3. sums them vertically into the window statistics of the 26x42 partials
   region, then evaluates SSIM, its partial derivatives and the loss;
4. blurs the partials horizontally (26x32), then vertically into the image
   gradient of the output tile.

Packed training also reduces the gradient magnitude while writing each tile,
so normalization does not need another traversal of the full image gradient.

Every blur adds its taps in the source's order. The partial derivatives use
approximate reciprocals of the SSIM denominators instead of the source's
divisions, which changes the gradient by ~1e-6 relative.
"""

import chex
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import jax
import jax.numpy as jnp
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.memory import SmemAllocator

from . import half2 as h

_GAUSS = (
    0.001028380123898387,
    0.0075987582094967365,
    0.036000773310661316,
    0.10936068743467331,
    0.21300552785396576,
    0.26601171493530273,
    0.21300552785396576,
    0.10936068743467331,
    0.036000773310661316,
    0.0075987582094967365,
    0.001028380123898387,
)
_RADIUS = 5
_TAPS = 2 * _RADIUS + 1
_TILE_W, _TILE_H = 32, 16
# Statistics and partials cover the tile and its halo; the image tile one more halo.
_STAT_W, _STAT_H = _TILE_W + 2 * _RADIUS, _TILE_H + 2 * _RADIUS
_IMAGE_W, _IMAGE_H = _STAT_W + 2 * _RADIUS, _STAT_H + 2 * _RADIUS
_THREADS = 256
# Consecutive outputs per thread in each pass; neighbors share their inputs.
_ROW_SUMS = 6
_COLUMN_SUMS = 5
_ROW_PARTIALS = 4
_COLUMN_PARTIALS = 2
# Padded row stride of the horizontally blurred partials (fewer bank conflicts).
_BLUR_STRIDE = _TILE_W + 1


@dsl_user_op
def _copy_async(destination, source, size, *, loc=None, ip=None):
    """cp.async of one float to shared memory; zero-filled when size is 0."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(destination.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
            cutlass.Int64(source.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
            cutlass.Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.ca.shared.global [$0], [$1], 4, $2;",
        "r,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _copy_tiles(image, target, tiles, tid, x0, y0, c, width, height, uint8_target):
    """Start copying channel c of the image and target tiles (zero padded)."""
    rows = _THREADS // _IMAGE_W
    if tid < rows * _IMAGE_W:
        col = tid % _IMAGE_W
        gx = x0 + col - 2 * _RADIUS
        r = tid // _IMAGE_W
        while r < _IMAGE_H:
            gy = y0 + r - 2 * _RADIUS
            p = cutlass.Int32(c)
            size = 0
            if gx >= 0 and gx < width and gy >= 0 and gy < height:
                p = (gy * width + gx) * 3 + c
                size = 4
            _copy_async(tiles.iterator + (r * _IMAGE_W + col), image.iterator + p, size)
            if cutlass.const_expr(uint8_target):
                value = cute.Float32(0)
                if size != 0:
                    value = cute.Float32(cutlass.Uint8(target[p])) * (1.0 / 255)
                tiles[1, r, col] = value
            else:
                _copy_async(
                    tiles.iterator + (_IMAGE_H * _IMAGE_W + r * _IMAGE_W + col),
                    target.iterator + p,
                    size,
                )
            r += rows
    cute.arch.cp_async_commit_group()


@cute.kernel
def _loss(
    image: cute.Tensor,
    target: cute.Tensor,
    block_loss: cute.Tensor,
    gradient: cute.Tensor,
    block_scale: cute.Tensor,
    width: cutlass.Constexpr,
    height: cutlass.Constexpr,
    uint8_target: cutlass.Constexpr,
    with_scale: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    x0, y0 = bx * _TILE_W, by * _TILE_H
    image_size = _IMAGE_H * _IMAGE_W
    stat_size = _STAT_H * _STAT_W
    smem = SmemAllocator()
    # The image tiles are dead once summed, the row sums once the partials exist.
    first = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(max(2 * image_size, 3 * stat_size)), byte_alignment=16
    )
    second = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout(max(5 * _IMAGE_H * _STAT_W, 3 * _STAT_H * _BLUR_STRIDE)),
        byte_alignment=16,
    )
    warp_loss = smem.allocate_tensor(cutlass.Float32, cute.make_layout(_THREADS // 32))
    if cutlass.const_expr(with_scale):
        warp_scale = smem.allocate_tensor(cutlass.Uint32, cute.make_layout(_THREADS // 32))
    tiles = cute.make_tensor(
        first.iterator, cute.make_layout((2, _IMAGE_H, _IMAGE_W), stride=(image_size, _IMAGE_W, 1))
    )
    partials = cute.make_tensor(
        first.iterator, cute.make_layout((3, _STAT_H, _STAT_W), stride=(stat_size, _STAT_W, 1))
    )
    row_sums = cute.make_tensor(
        second.iterator,
        cute.make_layout((5, _IMAGE_H, _STAT_W), stride=(_IMAGE_H * _STAT_W, _STAT_W, 1)),
    )
    blurred = cute.make_tensor(
        second.iterator,
        cute.make_layout(
            (3, _STAT_H, _BLUR_STRIDE), stride=(_STAT_H * _BLUR_STRIDE, _BLUR_STRIDE, 1)
        ),
    )
    # Each pass's first row and column for this thread.
    row_groups = _STAT_W // _ROW_SUMS
    sum_row, sum_col = tid // row_groups, tid % row_groups * _ROW_SUMS
    stat_col, stat_row = tid % _STAT_W, tid // _STAT_W * _COLUMN_SUMS
    blur_groups = _TILE_W // _ROW_PARTIALS
    blur_row, blur_col = tid // blur_groups, tid % blur_groups * _ROW_PARTIALS
    out_col, out_row = tid % _TILE_W, tid // _TILE_W * _COLUMN_PARTIALS
    stat_tasks = _STAT_W * ((_STAT_H + _COLUMN_SUMS - 1) // _COLUMN_SUMS)
    # Pixel values needed after the image tiles are overwritten.
    stat_pixels = cute.make_rmem_tensor((2, _COLUMN_SUMS), cutlass.Float32)
    out_pixels = cute.make_rmem_tensor((2, _COLUMN_PARTIALS), cutlass.Float32)
    loss = cute.Float32(0)
    scale_bits = cutlass.Uint32(0)
    _copy_tiles(image, target, tiles, tid, x0, y0, 0, width, height, uint8_target)
    for c in range(3):
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if tid < _IMAGE_H * row_groups:
            xs = cute.make_rmem_tensor(_ROW_SUMS + _TAPS - 1, cutlass.Float32)
            ys = cute.make_rmem_tensor(_ROW_SUMS + _TAPS - 1, cutlass.Float32)
            for i in cutlass.range_constexpr(_ROW_SUMS + _TAPS - 1):
                xs[i] = tiles[0, sum_row, sum_col + i]
                ys[i] = tiles[1, sum_row, sum_col + i]
            for o in cutlass.range_constexpr(_ROW_SUMS):
                sx, sx2, sy, sy2, sxy = (
                    cute.Float32(0),
                    cute.Float32(0),
                    cute.Float32(0),
                    cute.Float32(0),
                    cute.Float32(0),
                )
                for d in cutlass.range_constexpr(_TAPS):
                    w = _GAUSS[d]
                    a, b = xs[o + d], ys[o + d]
                    sx += a * w
                    sx2 += a * a * w
                    sy += b * w
                    sy2 += b * b * w
                    sxy += a * b * w
                row_sums[0, sum_row, sum_col + o] = sx
                row_sums[1, sum_row, sum_col + o] = sx2
                row_sums[2, sum_row, sum_col + o] = sy
                row_sums[3, sum_row, sum_col + o] = sy2
                row_sums[4, sum_row, sum_col + o] = sxy
        if tid < stat_tasks:
            for r in cutlass.range_constexpr(_COLUMN_SUMS):
                if stat_row + r < _STAT_H:
                    stat_pixels[0, r] = tiles[0, stat_row + r + _RADIUS, stat_col + _RADIUS]
                    stat_pixels[1, r] = tiles[1, stat_row + r + _RADIUS, stat_col + _RADIUS]
        for r in cutlass.range_constexpr(_COLUMN_PARTIALS):
            out_pixels[0, r] = tiles[0, out_row + r + 2 * _RADIUS, out_col + 2 * _RADIUS]
            out_pixels[1, r] = tiles[1, out_row + r + 2 * _RADIUS, out_col + 2 * _RADIUS]
        cute.arch.sync_threads()
        if tid < stat_tasks:
            # Rows stream through the window sums; each still adds its taps in order.
            sums = cute.make_rmem_tensor((5, _COLUMN_SUMS), cutlass.Float32)
            sums.fill(0)
            for i in cutlass.range_constexpr(_COLUMN_SUMS + _TAPS - 1):
                if stat_row + i < _IMAGE_H:
                    for q in cutlass.range_constexpr(5):
                        value = row_sums[q, stat_row + i, stat_col]
                        for r in cutlass.range_constexpr(_COLUMN_SUMS):
                            if cutlass.const_expr(0 <= i - r < _TAPS):
                                sums[q, r] += value * _GAUSS[i - r]
            for r in cutlass.range_constexpr(_COLUMN_SUMS):
                if stat_row + r < _STAT_H:
                    mx, xx, my, yy, xy = sums[0, r], sums[1, r], sums[2, r], sums[3, r], sums[4, r]
                    a = mx * mx + my * my + 0.0001
                    b = (xx - mx * mx) + (yy - my * my) + 0.0009
                    cc = 2.0 * mx * my + 0.0001
                    dd = 2.0 * (xy - mx * my) + 0.0009
                    gy, gx = y0 + stat_row + r - _RADIUS, x0 + stat_col - _RADIUS
                    # The partials blurred into the gradient include the
                    # source's -0.2 / N loss weight; outside the image they are 0.
                    p0, p1, p2 = cute.Float32(0), cute.Float32(0), cute.Float32(0)
                    if gx >= 0 and gx < width and gy >= 0 and gy < height:
                        weight = -0.2 / (width * height * 3)
                        inv_a, inv_b = cute.arch.rcp_approx(a), cute.arch.rcp_approx(b)
                        ssim = cc * dd * inv_a * inv_b
                        p0 = (
                            2.0 * my * (dd - cc) * inv_a * inv_b + 2.0 * mx * ssim * (inv_b - inv_a)
                        ) * weight
                        p1 = -ssim * inv_b * weight
                        p2 = 2.0 * cc * inv_a * inv_b * weight
                        if (
                            stat_row + r >= _RADIUS
                            and stat_row + r < _RADIUS + _TILE_H
                            and stat_col >= _RADIUS
                            and stat_col < _RADIUS + _TILE_W
                        ):
                            l1 = cute.abs(stat_pixels[0, r] - stat_pixels[1, r])
                            loss += 0.2 * (1.0 - ssim) + 0.8 * l1
                    partials[0, stat_row + r, stat_col] = p0
                    partials[1, stat_row + r, stat_col] = p1
                    partials[2, stat_row + r, stat_col] = p2
        cute.arch.sync_threads()
        if tid < _STAT_H * blur_groups:
            for k in cutlass.range_constexpr(3):
                values = cute.make_rmem_tensor(_ROW_PARTIALS + _TAPS - 1, cutlass.Float32)
                for i in cutlass.range_constexpr(_ROW_PARTIALS + _TAPS - 1):
                    values[i] = partials[k, blur_row, blur_col + i]
                for o in cutlass.range_constexpr(_ROW_PARTIALS):
                    v = cute.Float32(0)
                    for d in cutlass.range_constexpr(_TAPS):
                        v += values[o + d] * _GAUSS[d]
                    blurred[k, blur_row, blur_col + o] = v
        cute.arch.sync_threads()
        # The partials are consumed: start the next channel's tiles.
        if c < 2:
            _copy_tiles(image, target, tiles, tid, x0, y0, c + 1, width, height, uint8_target)
        for r in cutlass.range_constexpr(_COLUMN_PARTIALS):
            s0, s1, s2 = cute.Float32(0), cute.Float32(0), cute.Float32(0)
            for d in cutlass.range_constexpr(_TAPS):
                s0 += blurred[0, out_row + r + d, out_col] * _GAUSS[d]
                s1 += blurred[1, out_row + r + d, out_col] * _GAUSS[d]
                s2 += blurred[2, out_row + r + d, out_col] * _GAUSS[d]
            px, py = x0 + out_col, y0 + out_row + r
            if px < width and py < height:
                a, b = out_pixels[0, r], out_pixels[1, r]
                sign = cute.Float32(0)
                if a > b:
                    sign = 1.0
                elif a < b:
                    sign = -1.0
                value = s0 + 2.0 * a * s1 + b * s2 + 0.8 * sign / (width * height * 3)
                gradient[(py * width + px) * 3 + c] = value
                if cutlass.const_expr(with_scale):
                    # Positive float bits sort by magnitude; NaNs stay above infinity.
                    scale_bits = cute.max(scale_bits, h.float_bits(value) & 0x7FFFFFFF)
    for level in cutlass.range_constexpr(5):
        loss += cute.arch.shuffle_sync_bfly(loss, 16 >> level)
    if tid % 32 == 0:
        warp_loss[tid // 32] = loss
    if cutlass.const_expr(with_scale):
        scale_bits = cute.arch.warp_redux_sync(scale_bits, "max")
        if tid % 32 == 0:
            warp_scale[tid // 32] = scale_bits
    cute.arch.sync_threads()
    if tid == 0:
        total = cute.Float32(0)
        for w in cutlass.range_constexpr(_THREADS // 32):
            total += warp_loss[w]
        block = by * ((width + _TILE_W - 1) // _TILE_W) + bx
        block_loss[block] = total
        if cutlass.const_expr(with_scale):
            for w in cutlass.range_constexpr(_THREADS // 32):
                scale_bits = cute.max(scale_bits, warp_scale[w])
            block_scale[block] = h.bits_float(scale_bits)


@cute.jit
def _launch(
    stream: cuda.CUstream,
    image: cute.Tensor,
    target: cute.Tensor,
    block_loss: cute.Tensor,
    gradient: cute.Tensor,
    block_scale: cute.Tensor,
    *,
    width: cutlass.Constexpr,
    height: cutlass.Constexpr,
    uint8_target: cutlass.Constexpr,
    with_scale: cutlass.Constexpr,
):
    grid = [(width + _TILE_W - 1) // _TILE_W, (height + _TILE_H - 1) // _TILE_H, 1]
    _loss(
        image, target, block_loss, gradient, block_scale, width, height, uint8_target, with_scale
    ).launch(grid=grid, block=[_THREADS, 1, 1], stream=stream)


def fused_loss_and_grad(
    prediction: chex.Array, target: chex.Array
) -> tuple[chex.Array, chex.Array]:
    """Mean LiteGS L1+SSIM and its image gradient (weight 0.2).

    Prediction is float32; target is normalized float32 or uint8 RGB.
    Uint8 targets are normalized while loading tiles, without a full-image
    float32 intermediate.
    """
    loss, gradient, _ = _fused_loss(prediction, target, with_scale=False)
    return loss, gradient


def _fused_loss_and_grad_with_scale(
    prediction: chex.Array, target: chex.Array
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Also reduce the packed rasterizer's gradient scale while writing gradients."""
    loss, gradient, block_scale = _fused_loss(prediction, target, with_scale=True)
    scale = jnp.maximum(jnp.max(block_scale), 1e-12).reshape(1)
    return loss, gradient, scale


def _fused_loss(
    prediction: chex.Array, target: chex.Array, *, with_scale: bool
) -> tuple[chex.Array, chex.Array, chex.Array]:
    from cutlass.jax import cutlass_call

    height, width, channels = prediction.shape
    if channels != 3 or target.shape != prediction.shape:
        raise ValueError("L1+SSIM expects matching HWC RGB images")
    blocks = -(-width // _TILE_W) * -(-height // _TILE_H)
    # CuTe byte tensors expose signless i8; dispatch with the JAX dtype and
    # explicitly cast byte reads to Uint8 before normalizing them.
    call = cutlass_call(
        _launch,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((blocks,), jnp.float32),
            jax.ShapeDtypeStruct((prediction.size,), jnp.float32),
            jax.ShapeDtypeStruct((blocks if with_scale else 1,), jnp.float32),
        ),
        use_static_tensors=True,
        width=width,
        height=height,
        uint8_target=target.dtype == jnp.uint8,
        with_scale=with_scale,
    )
    block_loss, gradient, block_scale = call(prediction.reshape(-1), target.reshape(-1))
    return jnp.sum(block_loss) / prediction.size, gradient.reshape(prediction.shape), block_scale
