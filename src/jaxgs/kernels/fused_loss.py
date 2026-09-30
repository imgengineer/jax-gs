"""CuTe port of LiteGS fused_ssim/ssim.cu's separable L1 + SSIM kernels.

The source's MIT notice is in LICENSE.fused_ssim. Images use HWC layout;
loss/derivative caches keep LiteGS's CHW planes for coalesced access.
"""

import chex
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import jax
import jax.numpy as jnp
from cutlass.memory import SmemAllocator

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


@cute.kernel
def _forward(
    image: cute.Tensor,
    target: cute.Tensor,
    loss: cute.Tensor,
    partials: cute.Tensor,
    width: cutlass.Constexpr,
    height: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    lx, ly = tid % 16, tid // 16
    px, py = bx * 16 + lx, by * 16 + ly
    smem = SmemAllocator()
    tile = smem.allocate_tensor(cute.Float32, cute.make_layout((26, 26, 2), stride=(52, 2, 1)))
    scratch = smem.allocate_tensor(cute.Float32, cute.make_layout((26, 16, 5), stride=(80, 5, 1)))
    for c in range(3):
        for step in cutlass.range_constexpr(3):
            i = step * 256 + tid
            if i < 26 * 26:
                y, x = i // 26, i % 26
                gy, gx = by * 16 + y - 5, bx * 16 + x - 5
                a, b = cute.Float32(0), cute.Float32(0)
                if gx >= 0 and gx < width and gy >= 0 and gy < height:
                    p = (gy * width + gx) * 3 + c
                    a, b = image[p], target[p]
                tile[y, x, 0], tile[y, x, 1] = a, b
        cute.arch.sync_threads()
        l1 = cute.abs(tile[ly + 5, lx + 5, 0] - tile[ly + 5, lx + 5, 1])
        for step in cutlass.range_constexpr(2):
            y = ly + step * 16
            if y < 26:
                sx, sx2, sy, sy2, sxy = (
                    cute.Float32(0),
                    cute.Float32(0),
                    cute.Float32(0),
                    cute.Float32(0),
                    cute.Float32(0),
                )
                for d in cutlass.range_constexpr(11):
                    w = _GAUSS[d]
                    a, b = tile[y, lx + d, 0], tile[y, lx + d, 1]
                    sx += a * w
                    sx2 += a * a * w
                    sy += b * w
                    sy2 += b * b * w
                    sxy += a * b * w
                scratch[y, lx, 0], scratch[y, lx, 1] = sx, sx2
                scratch[y, lx, 2], scratch[y, lx, 3] = sy, sy2
                scratch[y, lx, 4] = sxy
        cute.arch.sync_threads()
        if px < width and py < height:
            mx, xx, my, yy, xy = (
                cute.Float32(0),
                cute.Float32(0),
                cute.Float32(0),
                cute.Float32(0),
                cute.Float32(0),
            )
            for d in cutlass.range_constexpr(11):
                w = _GAUSS[d]
                mx += scratch[ly + d, lx, 0] * w
                xx += scratch[ly + d, lx, 1] * w
                my += scratch[ly + d, lx, 2] * w
                yy += scratch[ly + d, lx, 3] * w
                xy += scratch[ly + d, lx, 4] * w
            a = mx * mx + my * my + 0.0001
            b = (xx - mx * mx) + (yy - my * my) + 0.0009
            cc = 2.0 * mx * my + 0.0001
            dd = 2.0 * (xy - mx * my) + 0.0009
            loss[c * width * height + py * width + px] = 0.2 * (1.0 - cc * dd / (a * b)) + 0.8 * l1
            partials[c * width * height + py * width + px] = (
                2.0 * my * dd / (a * b)
                - 2.0 * my * cc / (a * b)
                - 2.0 * mx * cc * dd / (a * a * b)
                + 2.0 * mx * cc * dd / (a * b * b)
            )
            partials[(3 + c) * width * height + py * width + px] = -cc * dd / (a * b * b)
            partials[(6 + c) * width * height + py * width + px] = 2.0 * cc / (a * b)
        cute.arch.sync_threads()


@cute.kernel
def _backward(
    image: cute.Tensor,
    target: cute.Tensor,
    partials: cute.Tensor,
    gradient: cute.Tensor,
    width: cutlass.Constexpr,
    height: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    lx, ly = tid % 16, tid // 16
    px, py = bx * 16 + lx, by * 16 + ly
    smem = SmemAllocator()
    tile = smem.allocate_tensor(cute.Float32, cute.make_layout((26, 26, 3), stride=(78, 3, 1)))
    scratch = smem.allocate_tensor(cute.Float32, cute.make_layout((26, 16, 3), stride=(48, 3, 1)))
    for c in range(3):
        for step in cutlass.range_constexpr(3):
            i = step * 256 + tid
            if i < 26 * 26:
                y, x = i // 26, i % 26
                gy, gx = by * 16 + y - 5, bx * 16 + x - 5
                for k in cutlass.range_constexpr(3):
                    v = cute.Float32(0)
                    if gx >= 0 and gx < width and gy >= 0 and gy < height:
                        v = partials[(k * 3 + c) * width * height + gy * width + gx] * (
                            -0.2 / (width * height * 3)
                        )
                    tile[y, x, k] = v
        cute.arch.sync_threads()
        for step in cutlass.range_constexpr(2):
            y = ly + step * 16
            if y < 26:
                for k in cutlass.range_constexpr(3):
                    v = cute.Float32(0)
                    for d in cutlass.range_constexpr(11):
                        v += tile[y, lx + d, k] * _GAUSS[d]
                    scratch[y, lx, k] = v
        cute.arch.sync_threads()
        if px < width and py < height:
            s0, s1, s2 = cute.Float32(0), cute.Float32(0), cute.Float32(0)
            for d in cutlass.range_constexpr(11):
                s0 += scratch[ly + d, lx, 0] * _GAUSS[d]
                s1 += scratch[ly + d, lx, 1] * _GAUSS[d]
                s2 += scratch[ly + d, lx, 2] * _GAUSS[d]
            p = (py * width + px) * 3 + c
            a, b = image[p], target[p]
            sign = cute.Float32(0)
            if a > b:
                sign = 1.0
            elif a < b:
                sign = -1.0
            gradient[p] = s0 + 2.0 * a * s1 + b * s2 + 0.8 * sign / (width * height * 3)
        cute.arch.sync_threads()


@cute.jit
def _launch(
    stream: cuda.CUstream,
    image: cute.Tensor,
    target: cute.Tensor,
    loss: cute.Tensor,
    partials: cute.Tensor,
    gradient: cute.Tensor,
    *,
    width: cutlass.Constexpr,
    height: cutlass.Constexpr,
):
    grid = [(width + 15) // 16, (height + 15) // 16, 1]
    _forward(image, target, loss, partials, width, height).launch(
        grid=grid, block=[256, 1, 1], stream=stream
    )
    _backward(image, target, partials, gradient, width, height).launch(
        grid=grid, block=[256, 1, 1], stream=stream
    )


def fused_loss_and_grad(
    prediction: chex.Array, target: chex.Array
) -> tuple[chex.Array, chex.Array]:
    """Mean LiteGS L1+SSIM and its image gradient (weight 0.2)."""
    from cutlass.jax import cutlass_call

    height, width, channels = prediction.shape
    if channels != 3 or target.shape != prediction.shape:
        raise ValueError("L1+SSIM expects matching HWC RGB images")
    call = cutlass_call(
        _launch,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((prediction.size,), jnp.float32),
            jax.ShapeDtypeStruct((prediction.size * 3,), jnp.float32),
            jax.ShapeDtypeStruct((prediction.size,), jnp.float32),
        ),
        use_static_tensors=True,
        width=width,
        height=height,
    )
    loss, _, gradient = call(prediction.reshape(-1), target.reshape(-1))
    return jnp.mean(loss), gradient.reshape(prediction.shape)
