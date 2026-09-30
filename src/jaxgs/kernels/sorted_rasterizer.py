import chex
import jax
import jax.numpy as jnp

from ..config import CapacityConfig
from ..render.types import (
    FragmentStatistics,
    ProjectedGaussians,
    RenderResult,
    SortedVisibilityTable,
)
from ..scene.camera import Camera


def _forward_arrays(
    mean,
    conic,
    depth,
    color,
    opacity,
    ids,
    tile_offsets,
    background,
    camera: Camera,
    config: CapacityConfig,
):
    from cutlass.jax import cutlass_call

    from .sorted_rasterize import launch_forward_sorted

    if config.raster_tile_height != config.tile_size:
        raise ValueError(
            "The float32 diagnostic rasterizer requires square tiles; use packed_forward for rectangular tiles"
        )
    width, height = camera.width, camera.height
    pixels = width * height
    call = cutlass_call(
        launch_forward_sorted,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((pixels * 3,), jnp.float32),
            jax.ShapeDtypeStruct((pixels,), jnp.float32),
            jax.ShapeDtypeStruct((pixels,), jnp.float32),
            jax.ShapeDtypeStruct((pixels,), jnp.float32),
            jax.ShapeDtypeStruct((pixels,), jnp.int32),
        ),
        use_static_tensors=True,
        width=width,
        height=height,
        tile_size=config.tile_size,
        tiles_x=(width + config.tile_size - 1) // config.tile_size,
    )
    rgb, out_depth, alpha, final_t, last = call(
        mean.reshape(-1),
        conic.reshape(-1),
        depth,
        color.reshape(-1),
        opacity,
        ids,
        tile_offsets,
        background,
    )
    return (
        (
            rgb.reshape(height, width, 3),
            out_depth.reshape(height, width),
            alpha.reshape(height, width),
        ),
        (final_t, last),
    )


def rasterize_sorted_cute_vjp(
    projected: ProjectedGaussians,
    table: SortedVisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    background: chex.Array | None = None,
) -> RenderResult:
    """Tile-range rasterizer with a custom VJP and O(image pixels) cache."""
    from cutlass.jax import cutlass_call

    from .sorted_rasterize import launch_backward_sorted

    if jax.default_backend() != "gpu":
        raise RuntimeError("CuTe rasterization requires a JAX CUDA device")
    background = jnp.zeros((3,), jnp.float32) if background is None else background
    width, height = camera.width, camera.height

    @jax.custom_vjp
    def render(mean, conic, depth, color, opacity, ids, tile_offsets, bg):
        return _forward_arrays(
            mean, conic, depth, color, opacity, ids, tile_offsets, bg, camera, config
        )[0]

    def render_fwd(mean, conic, depth, color, opacity, ids, tile_offsets, bg):
        result, cache = _forward_arrays(
            mean, conic, depth, color, opacity, ids, tile_offsets, bg, camera, config
        )
        return result, (mean, conic, depth, color, opacity, ids, tile_offsets, bg, *cache)

    def render_bwd(residual, cotangents):
        (mean, conic, depth, color, opacity, ids, tile_offsets, bg, final_t, last) = residual
        drgb, ddepth, dalpha = cotangents
        capacity = depth.shape[0]
        call = cutlass_call(
            launch_backward_sorted,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((mean.size,), jnp.float32),
                jax.ShapeDtypeStruct((conic.size,), jnp.float32),
                jax.ShapeDtypeStruct((depth.size,), jnp.float32),
                jax.ShapeDtypeStruct((color.size,), jnp.float32),
                jax.ShapeDtypeStruct((opacity.size,), jnp.float32),
                jax.ShapeDtypeStruct((1,), jnp.float32),
            ),
            use_static_tensors=True,
            width=width,
            height=height,
            tile_size=config.tile_size,
            tiles_x=(width + config.tile_size - 1) // config.tile_size,
            capacity=capacity,
        )
        grads = call(
            mean.reshape(-1),
            conic.reshape(-1),
            depth,
            color.reshape(-1),
            opacity,
            ids,
            tile_offsets,
            bg,
            final_t,
            last,
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
    return RenderResult(
        *render(
            projected.mean,
            projected.conic,
            projected.depth,
            projected.color,
            projected.alpha,
            table.gaussian_ids,
            table.tile_offsets,
            background,
        )
    )


def rasterize_loss_and_grad(
    projected: ProjectedGaussians,
    table: SortedVisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    target: chex.Array,
    collect_stats: bool,
) -> tuple[chex.Array, ProjectedGaussians, FragmentStatistics]:
    """Explicit pullback exposes LiteGS fragment statistics outside autodiff."""
    from cutlass.jax import cutlass_call

    from .fused_loss import fused_loss_and_grad
    from .sorted_rasterize import launch_backward_sorted

    background = jnp.zeros((3,), jnp.float32)
    arrays, cache = _forward_arrays(
        projected.mean,
        projected.conic,
        projected.depth,
        projected.color,
        projected.alpha,
        table.gaussian_ids,
        table.tile_offsets,
        background,
        camera,
        config,
    )
    rgb, clip_pullback = jax.vjp(lambda x: jnp.clip(x, 0, 1), arrays[0])
    loss, drgb = fused_loss_and_grad(rgb, target)
    (drgb,) = clip_pullback(drgb)
    fields = (projected.mean, projected.conic, projected.depth, projected.color, projected.alpha)
    call = cutlass_call(
        launch_backward_sorted,
        output_shape_dtype=(
            *(jax.ShapeDtypeStruct((value.size,), jnp.float32) for value in fields),
            jax.ShapeDtypeStruct((config.max_gaussians * 3 if collect_stats else 1,), jnp.float32),
        ),
        use_static_tensors=True,
        width=camera.width,
        height=camera.height,
        tile_size=config.tile_size,
        tiles_x=(camera.width + config.tile_size - 1) // config.tile_size,
        capacity=config.max_gaussians,
        collect_stats=int(collect_stats),
    )
    grads = call(
        *(value.reshape(-1) for value in fields),
        table.gaussian_ids,
        table.tile_offsets,
        background,
        *cache,
        drgb.reshape(-1),
        jnp.zeros((camera.width * camera.height,), jnp.float32),
        jnp.zeros((camera.width * camera.height,), jnp.float32),
    )
    cotangents = projected.replace(
        mean=grads[0].reshape(projected.mean.shape),
        conic=grads[1].reshape(projected.conic.shape),
        depth=grads[2],
        color=grads[3].reshape(projected.color.shape),
        alpha=grads[4],
        radius=jnp.zeros_like(projected.radius),
        visible=jnp.zeros(projected.visible.shape, jax.dtypes.float0),
    )
    # count, compositing weight, sum(dL/dalpha), sum((dL/dalpha)^2)
    stats = (
        jnp.stack((grads[5][0::3], grads[5][1::3], grads[4], grads[5][2::3]), axis=1)
        if collect_stats
        else jnp.zeros((config.max_gaussians, 4), jnp.float32)
    )
    return loss, cotangents, jax.lax.stop_gradient(stats)
