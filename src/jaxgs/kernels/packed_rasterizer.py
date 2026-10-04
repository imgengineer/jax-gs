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
from ..scene.types import VisibleClusters


def _tile_count(camera: Camera, config: CapacityConfig) -> int:
    tiles_x = (camera.width + config.tile_size - 1) // config.tile_size
    return tiles_x * ((camera.height + config.raster_tile_height - 1) // config.raster_tile_height)


def packed_forward(
    projected: ProjectedGaussians,
    table: SortedVisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    collect_stats: bool = False,
    packed_params: chex.Array | None = None,
) -> tuple[chex.Array, PackedRasterCache, chex.Array]:
    """Return RGB, the backward cache, and fragment statistics.

    The table may reference only visible Gaussians; others are not packed.
    """
    return _packed_forward(
        projected,
        table,
        camera,
        config,
        collect_stats,
        record_contributions=True,
        packed_params=packed_params,
    )


def _packed_forward(
    projected: ProjectedGaussians,
    table: SortedVisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    collect_stats: bool,
    *,
    record_contributions: bool,
    packed_params: chex.Array | None = None,
) -> tuple[chex.Array, PackedRasterCache, chex.Array]:
    from cutlass.jax import cutlass_call

    from .packed_rasterize import launch_forward

    if (config.raster_tile_height, config.tile_size) not in ((8, 8), (8, 16), (12, 16), (16, 16)):
        raise ValueError("Packed rasterizer supports tiles 8x8, 8x16, 12x16 and 16x16")
    pixels = camera.width * camera.height
    tiles = _tile_count(camera, config)
    # RGB-only calls never read or write backward state; retain scalar placeholders.
    cached_pixels = pixels if record_contributions else 1
    cached_tiles = tiles if record_contributions else 1
    bit_words = (config.visibility_capacity + 31) // 32 + tiles if record_contributions else 1
    use_packed_input = packed_params is not None
    call = cutlass_call(
        launch_forward,
        compile_key=launch_forward,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(
                (1 if use_packed_input else config.max_gaussians * 8,), jnp.uint32
            ),
            jax.ShapeDtypeStruct((pixels * 3,), jnp.float32),
            jax.ShapeDtypeStruct((cached_pixels,), jnp.float32),
            jax.ShapeDtypeStruct((cached_pixels,), jnp.int32),
            jax.ShapeDtypeStruct((config.max_gaussians * 2 if collect_stats else 1,), jnp.float32),
            jax.ShapeDtypeStruct((cached_tiles,), jnp.int32),
            jax.ShapeDtypeStruct((tiles,), jnp.int32),
            jax.ShapeDtypeStruct((bit_words,), jnp.uint32),
        ),
        use_static_tensors=True,
        width=camera.width,
        height=camera.height,
        tile_size=config.tile_size,
        tile_height=config.raster_tile_height,
        capacity=config.max_gaussians,
        collect_stats=int(collect_stats),
        record_contributions=record_contributions,
        use_packed_input=use_packed_input,
    )
    packed_input = packed_params if use_packed_input else jnp.zeros((1,), jnp.uint32)
    params_output, rgb, trans, last, stats, backward_work, _, contribution_bits = call(
        projected.mean.reshape(-1),
        projected.conic.reshape(-1),
        projected.color.reshape(-1),
        projected.alpha,
        projected.visible.astype(jnp.int8),
        table.gaussian_ids,
        table.tile_offsets,
        packed_input,
    )
    params = packed_params if use_packed_input else params_output
    return (
        rgb.reshape(camera.height, camera.width, 3),
        (params, trans, last, backward_work, contribution_bits),
        stats,
    )


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
    visible_clusters: VisibleClusters | None = None,
    image_grad_scale: chex.Array | None = None,
) -> tuple[ProjectedGaussians, chex.Array]:
    """Return field cotangents and per-Gaussian squared alpha gradients.

    symmetric_conic stores both off-diagonal contributions in [0, 1] for
    the analytic parameter pullback; [1, 0] remains zero. General VJPs use
    the default matrix-gradient convention. With visible_clusters, which must
    contain every Gaussian in the table, mean, conic and color cotangents are
    defined only in those clusters, depth cotangents nowhere, and alpha
    cotangents everywhere when collecting statistics.
    image_grad_scale may supply the same floored maximum absolute gradient,
    avoiding a full-image reduction when the loss kernel already computed it.
    """
    from cutlass.jax import cutlass_call

    from .packed_rasterize import launch_backward

    scale = (
        jnp.maximum(jnp.max(jnp.abs(image_grad)), 1e-12).reshape(1)
        if image_grad_scale is None
        else image_grad_scale
    )
    fields = (projected.mean, projected.conic, projected.depth, projected.color, projected.alpha)
    call = cutlass_call(
        launch_backward,
        compile_key=launch_backward,
        output_shape_dtype=(
            *(jax.ShapeDtypeStruct((value.size,), jnp.float32) for value in fields),
            jax.ShapeDtypeStruct((config.max_gaussians if collect_stats else 1,), jnp.float32),
            jax.ShapeDtypeStruct((_tile_count(camera, config),), jnp.int32),
        ),
        use_static_tensors=True,
        width=camera.width,
        height=camera.height,
        tile_size=config.tile_size,
        tile_height=config.raster_tile_height,
        capacity=config.max_gaussians,
        collect_stats=int(collect_stats),
        symmetric_conic=symmetric_conic,
        cluster_size=0 if visible_clusters is None else config.cluster_size,
    )
    params, trans, last, backward_work, contribution_bits = cache
    clusters = (
        (jnp.zeros(1, jnp.int32), jnp.zeros(1, jnp.int32))
        if visible_clusters is None
        else visible_clusters
    )
    grads = call(
        params,
        table.gaussian_ids,
        table.tile_offsets,
        trans,
        last,
        contribution_bits,
        backward_work,
        image_grad.reshape(-1),
        scale,
        *clusters,
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
    visible_clusters: VisibleClusters | None = None,
    packed_params: chex.Array | None = None,
) -> tuple[chex.Array, ProjectedGaussians, FragmentStatistics]:
    """Loss, projected-field gradients and detached [C, 4] fragment statistics.

    visible_clusters limits the defined gradients as in packed_backward.
    """
    from .fused_loss import _fused_loss_and_grad_with_scale

    image, cache, fragments = packed_forward(
        projected, table, camera, config, collect_stats, packed_params
    )
    loss, image_grad, image_grad_scale = _fused_loss_and_grad_with_scale(image, target)
    gradients, alpha_grad_sq_sum = packed_backward(
        projected,
        table,
        cache,
        image_grad,
        camera,
        config,
        collect_stats,
        symmetric_conic=symmetric_conic,
        visible_clusters=visible_clusters,
        image_grad_scale=image_grad_scale,
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
    def render(mean, conic, color, alpha, visible):
        current = projected.replace(
            mean=mean, conic=conic, color=color, alpha=alpha, visible=visible
        )
        return _packed_forward(current, table, camera, config, False, record_contributions=False)[0]

    def forward(mean, conic, color, alpha, visible):
        current = projected.replace(
            mean=mean, conic=conic, color=color, alpha=alpha, visible=visible
        )
        image, cache, _ = packed_forward(current, table, camera, config)
        return image, cache

    def backward(cache, image_grad):
        gradients, _ = packed_backward(projected, table, cache, image_grad, camera, config)
        return gradients.mean, gradients.conic, gradients.color, gradients.alpha, None

    render.defvjp(forward, backward)
    return render(
        projected.mean, projected.conic, projected.color, projected.alpha, projected.visible
    )
