import chex
import jax
import jax.numpy as jnp

from ..config import CapacityConfig
from ..render.types import ProjectedGaussians, RenderResult, VisibilityTable
from ..scene.camera import Camera


def _forward_arrays(
    mean,
    conic,
    depth,
    color,
    opacity,
    ids,
    valid,
    background,
    width: int,
    height: int,
    tile_size: int,
    k_max: int,
):
    from cutlass.jax import cutlass_call

    from .rasterize_forward import launch_forward

    if jax.default_backend() != "gpu":
        raise RuntimeError("CuTe rasterization requires a JAX CUDA device")
    count = width * height
    shapes = (
        jax.ShapeDtypeStruct((count * 3,), jnp.float32),
        jax.ShapeDtypeStruct((count,), jnp.float32),
        jax.ShapeDtypeStruct((count,), jnp.float32),
        jax.ShapeDtypeStruct((count * k_max,), jnp.float32),
    )
    call = cutlass_call(
        launch_forward,
        compile_key=launch_forward,
        output_shape_dtype=shapes,
        use_static_tensors=True,
        width=width,
        height=height,
        tile_size=tile_size,
        tiles_x=(width + tile_size - 1) // tile_size,
        k_max=k_max,
    )
    rgb, depth, alpha, transmittance = call(
        mean.reshape(-1),
        conic.reshape(-1),
        depth,
        color.reshape(-1),
        opacity,
        ids.reshape(-1),
        valid.astype(jnp.int8).reshape(-1),
        background,
    )
    return (
        rgb.reshape(height, width, 3),
        depth.reshape(height, width),
        alpha.reshape(height, width),
    ), transmittance.reshape(count, k_max)


def rasterize_cute_with_cache(
    projected: ProjectedGaussians,
    table: VisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    background: chex.Array | None = None,
) -> tuple[RenderResult, chex.Array]:
    """CuTe forward renderer with transmittance cache for backward."""
    if config.raster_tile_height != config.tile_size:
        raise ValueError("The bounded CuTe rasterizer requires square tiles")
    background = jnp.zeros((3,), jnp.float32) if background is None else background
    result, cache = _forward_arrays(
        projected.mean,
        projected.conic,
        projected.depth,
        projected.color,
        projected.alpha,
        table.tile_gaussian_ids,
        table.tile_valid,
        background,
        camera.width,
        camera.height,
        config.tile_size,
        config.max_gaussians_per_tile,
    )
    return RenderResult(*result), cache


def rasterize_cute(
    projected: ProjectedGaussians,
    table: VisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    background: chex.Array | None = None,
) -> RenderResult:
    return rasterize_cute_with_cache(projected, table, camera, config, background)[0]


def rasterize_cute_vjp(
    projected: ProjectedGaussians,
    table: VisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    background: chex.Array | None = None,
) -> RenderResult:
    """CuTe rasterization with a custom VJP for projected Gaussian fields."""
    if config.raster_tile_height != config.tile_size:
        raise ValueError("The bounded CuTe rasterizer requires square tiles")
    from cutlass.jax import cutlass_call

    from .rasterize_backward import launch_backward

    width, height = camera.width, camera.height
    tile_size, k_max = config.tile_size, config.max_gaussians_per_tile
    background = jnp.zeros((3,), jnp.float32) if background is None else background

    @jax.custom_vjp
    def render(mean, conic, depth, color, opacity, ids, valid, bg):
        return _forward_arrays(
            mean, conic, depth, color, opacity, ids, valid, bg, width, height, tile_size, k_max
        )[0]

    def render_fwd(mean, conic, depth, color, opacity, ids, valid, bg):
        result, cache = _forward_arrays(
            mean, conic, depth, color, opacity, ids, valid, bg, width, height, tile_size, k_max
        )
        return result, (mean, conic, depth, color, opacity, ids, valid, bg, cache)

    def render_bwd(residual, cotangents):
        mean, conic, depth, color, opacity, ids, valid, bg, cache = residual
        drgb, ddepth, dalpha = cotangents
        capacity = depth.shape[0]
        shapes = (
            jax.ShapeDtypeStruct((mean.size,), jnp.float32),
            jax.ShapeDtypeStruct((conic.size,), jnp.float32),
            jax.ShapeDtypeStruct((depth.size,), jnp.float32),
            jax.ShapeDtypeStruct((color.size,), jnp.float32),
            jax.ShapeDtypeStruct((opacity.size,), jnp.float32),
        )
        call = cutlass_call(
            launch_backward,
            compile_key=launch_backward,
            output_shape_dtype=shapes,
            use_static_tensors=True,
            width=width,
            height=height,
            tile_size=tile_size,
            tiles_x=(width + tile_size - 1) // tile_size,
            k_max=k_max,
            capacity=capacity,
        )
        grads = call(
            mean.reshape(-1),
            conic.reshape(-1),
            depth,
            color.reshape(-1),
            opacity,
            ids.reshape(-1),
            valid.astype(jnp.int8).reshape(-1),
            bg,
            cache.reshape(-1),
            drgb.reshape(-1),
            ddepth.reshape(-1),
            dalpha.reshape(-1),
        )
        return (
            grads[0].reshape(mean.shape),
            grads[1].reshape(conic.shape),
            grads[2],
            grads[3].reshape(color.shape),
            grads[4],
            None,
            None,
            None,
        )

    render.defvjp(render_fwd, render_bwd)
    result = render(
        projected.mean,
        projected.conic,
        projected.depth,
        projected.color,
        projected.alpha,
        table.tile_gaussian_ids,
        table.tile_valid,
        background,
    )
    return RenderResult(*result)
