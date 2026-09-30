import cuda.bindings.driver as cuda
import cutlass.cute as cute


@cute.kernel
def _forward_kernel(
    mean: cute.Tensor,
    conic: cute.Tensor,
    depth: cute.Tensor,
    color: cute.Tensor,
    opacity: cute.Tensor,
    ids: cute.Tensor,
    valid: cute.Tensor,
    background: cute.Tensor,
    out_rgb: cute.Tensor,
    out_depth: cute.Tensor,
    out_alpha: cute.Tensor,
    out_transmittance: cute.Tensor,
    width: int,
    height: int,
    tile_size: int,
    tiles_x: int,
    k_max: int,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    p = bidx * bdx + tidx
    if p < width * height:
        px = p % width
        py = p // width
        tile = (py // tile_size) * tiles_x + px // tile_size
        red = cute.Float32(0.0)
        green = cute.Float32(0.0)
        blue = cute.Float32(0.0)
        weighted_depth = cute.Float32(0.0)
        transmittance = cute.Float32(1.0)
        for k in range(k_max):
            out_transmittance[p * k_max + k] = transmittance
            offset = tile * k_max + k
            if valid[offset] != 0:
                gid = ids[offset]
                dx = cute.Float32(px) + 0.5 - mean[gid * 2]
                dy = cute.Float32(py) + 0.5 - mean[gid * 2 + 1]
                exponent = -0.5 * (
                    dx * (conic[gid * 4] * dx + conic[gid * 4 + 1] * dy)
                    + dy * (conic[gid * 4 + 2] * dx + conic[gid * 4 + 3] * dy)
                )
                if exponent <= 0:
                    raw_alpha = opacity[gid] * cute.exp(exponent)
                    if raw_alpha >= 1.0 / 256:
                        a = cute.min(cute.Float32(255.0 / 256), raw_alpha)
                        weight = a * transmittance
                        red += weight * color[gid * 3]
                        green += weight * color[gid * 3 + 1]
                        blue += weight * color[gid * 3 + 2]
                        weighted_depth += weight * depth[gid]
                        transmittance *= 1.0 - a
        out_rgb[p * 3] = red + transmittance * background[0]
        out_rgb[p * 3 + 1] = green + transmittance * background[1]
        out_rgb[p * 3 + 2] = blue + transmittance * background[2]
        out_depth[p] = weighted_depth
        out_alpha[p] = 1.0 - transmittance


@cute.jit
def launch_forward(
    stream: cuda.CUstream,
    mean: cute.Tensor,
    conic: cute.Tensor,
    depth: cute.Tensor,
    color: cute.Tensor,
    opacity: cute.Tensor,
    ids: cute.Tensor,
    valid: cute.Tensor,
    background: cute.Tensor,
    out_rgb: cute.Tensor,
    out_depth: cute.Tensor,
    out_alpha: cute.Tensor,
    out_transmittance: cute.Tensor,
    *,
    width: int,
    height: int,
    tile_size: int,
    tiles_x: int,
    k_max: int,
):
    block = 128
    _forward_kernel(
        mean,
        conic,
        depth,
        color,
        opacity,
        ids,
        valid,
        background,
        out_rgb,
        out_depth,
        out_alpha,
        out_transmittance,
        width,
        height,
        tile_size,
        tiles_x,
        k_max,
    ).launch(grid=[(width * height + block - 1) // block, 1, 1], block=[block, 1, 1], stream=stream)
