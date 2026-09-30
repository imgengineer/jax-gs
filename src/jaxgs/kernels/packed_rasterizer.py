"""JAX entry points for the LiteGS RGB half2 training rasterizer."""

import chex
import jax
import jax.numpy as jnp

from ..config import CapacityConfig
from ..render.types import (
    FragmentStatistics,
    PackedRasterCache,
    ProjectedGaussians,
    SortedVisibilityTable,
)
from ..scene.camera import Camera


def packed_forward(
    projected: ProjectedGaussians,
    table: SortedVisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    collect_stats: bool = False,
) -> tuple[chex.Array, PackedRasterCache, chex.Array]:
    """Return RGB, (packed params, final T, last pair), and count/weight pairs."""
    from cutlass.jax import cutlass_call

    from .packed_rasterize import launch_forward

    if (config.raster_tile_height, config.tile_size) not in ((8, 8), (8, 16), (12, 16), (16, 16)):
        raise ValueError("Packed rasterizer supports tiles 8x8, 8x16, 12x16 and 16x16")
    pixels = camera.width * camera.height
    call = cutlass_call(
        launch_forward,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((config.max_gaussians * 8,), jnp.uint32),
            jax.ShapeDtypeStruct((pixels * 3,), jnp.float32),
            jax.ShapeDtypeStruct((pixels,), jnp.float32),
            jax.ShapeDtypeStruct((pixels,), jnp.int32),
            jax.ShapeDtypeStruct((config.max_gaussians * 2 if collect_stats else 1,), jnp.float32),
        ),
        use_static_tensors=True,
        width=camera.width,
        height=camera.height,
        tile_size=config.tile_size,
        tile_height=config.raster_tile_height,
        capacity=config.max_gaussians,
        collect_stats=int(collect_stats),
    )
    params, rgb, trans, last, stats = call(
        projected.mean.reshape(-1),
        projected.conic.reshape(-1),
        projected.color.reshape(-1),
        projected.alpha,
        table.gaussian_ids,
        table.tile_offsets,
    )
    return rgb.reshape(camera.height, camera.width, 3), (params, trans, last), stats


def packed_backward(
    projected: ProjectedGaussians,
    table: SortedVisibilityTable,
    cache: PackedRasterCache,
    image_grad: chex.Array,
    camera: Camera,
    config: CapacityConfig,
    collect_stats: bool = False,
    *,
    symmetric_conic: bool = False,
) -> tuple[ProjectedGaussians, chex.Array]:
    """Return field cotangents and per-Gaussian squared alpha gradients.

    symmetric_conic stores both off-diagonal contributions in [0, 1] for
    the analytic parameter pullback; [1, 0] remains zero. General VJPs use
    the default matrix-gradient convention.
    """
    from cutlass.jax import cutlass_call

    from .packed_rasterize import launch_backward

    scale = jnp.maximum(jnp.max(jnp.abs(image_grad)), 1e-12).reshape(1)
    fields = (projected.mean, projected.conic, projected.depth, projected.color, projected.alpha)
    call = cutlass_call(
        launch_backward,
        output_shape_dtype=(
            *(jax.ShapeDtypeStruct((value.size,), jnp.float32) for value in fields),
            jax.ShapeDtypeStruct((config.max_gaussians if collect_stats else 1,), jnp.float32),
        ),
        use_static_tensors=True,
        width=camera.width,
        height=camera.height,
        tile_size=config.tile_size,
        tile_height=config.raster_tile_height,
        capacity=config.max_gaussians,
        collect_stats=int(collect_stats),
        symmetric_conic=symmetric_conic,
    )
    params, trans, last = cache
    grads = call(
        params, table.gaussian_ids, table.tile_offsets, trans, last, image_grad.reshape(-1), scale
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
    return cotangents, grads[5] * scale[0] ** 2


def packed_loss_and_grad(
    projected: ProjectedGaussians,
    table: SortedVisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    target: chex.Array,
    collect_stats: bool,
    *,
    symmetric_conic: bool = False,
) -> tuple[chex.Array, ProjectedGaussians, FragmentStatistics]:
    """Loss, projected-field gradients and detached [C, 4] fragment statistics."""
    from .fused_loss import fused_loss_and_grad

    image, cache, fragments = packed_forward(projected, table, camera, config, collect_stats)
    loss, image_grad = fused_loss_and_grad(image, target)
    gradients, alpha_grad_sq_sum = packed_backward(
        projected,
        table,
        cache,
        image_grad,
        camera,
        config,
        collect_stats,
        symmetric_conic=symmetric_conic,
    )
    stats = (
        jnp.stack((fragments[0::2], fragments[1::2], gradients.alpha, alpha_grad_sq_sum), axis=1)
        if collect_stats
        else jnp.zeros((config.max_gaussians, 4), jnp.float32)
    )
    return loss, gradients, jax.lax.stop_gradient(stats)


def rasterize_packed_cute_vjp(
    projected: ProjectedGaussians,
    table: SortedVisibilityTable,
    camera: Camera,
    config: CapacityConfig,
) -> chex.Array:
    """Differentiable LiteGS RGB rendering; parameter gradients stay in CuTe."""

    @jax.custom_vjp
    def render(mean, conic, color, alpha):
        current = projected.replace(mean=mean, conic=conic, color=color, alpha=alpha)
        return packed_forward(current, table, camera, config)[0]

    def forward(mean, conic, color, alpha):
        current = projected.replace(mean=mean, conic=conic, color=color, alpha=alpha)
        image, cache, _ = packed_forward(current, table, camera, config)
        return image, cache

    def backward(cache, image_grad):
        gradients, _ = packed_backward(projected, table, cache, image_grad, camera, config)
        return gradients.mean, gradients.conic, gradients.color, gradients.alpha

    render.defvjp(forward, backward)
    return render(projected.mean, projected.conic, projected.color, projected.alpha)
