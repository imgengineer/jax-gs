import chex
import jax.numpy as jnp

from ..config import CapacityConfig
from ..scene.camera import Camera
from .projection import support_radius
from .types import ProjectedGaussians, VisibilityTable


def build_visibility_table(
    projected: ProjectedGaussians,
    camera: Camera,
    config: CapacityConfig,
    cluster_tile_mask: chex.Array | None = None,
) -> VisibilityTable:
    """Reference binning; O(C*T) and intended only for small correctness cases."""
    tiles_x = (camera.width + config.tile_size - 1) // config.tile_size
    tile_height = config.raster_tile_height
    tiles_y = (camera.height + tile_height - 1) // tile_height
    tile_x = jnp.tile(jnp.arange(tiles_x), tiles_y) * config.tile_size
    tile_y = jnp.repeat(jnp.arange(tiles_y), tiles_x) * tile_height
    mean_x = projected.mean[:, 0][None, :]
    mean_y = projected.mean[:, 1][None, :]
    support_radii = support_radius(projected)[None, :]
    intersects = (
        (mean_x + support_radii >= tile_x[:, None])
        & (mean_x - support_radii < tile_x[:, None] + config.tile_size)
        & (mean_y + support_radii >= tile_y[:, None])
        & (mean_y - support_radii < tile_y[:, None] + tile_height)
        & projected.visible[None, :]
    )
    if cluster_tile_mask is not None:
        cluster_ids = jnp.arange(config.max_gaussians) // config.cluster_size
        intersects = intersects & cluster_tile_mask[cluster_ids].T
    tile_counts = jnp.sum(intersects, axis=1, dtype=jnp.int32)
    depths = jnp.where(intersects, projected.depth[None, :], jnp.inf)
    depth_order = jnp.argsort(depths, axis=1, stable=True)[:, : config.max_gaussians_per_tile]
    sorted_depths = jnp.take_along_axis(depths, depth_order, axis=1)
    padding = max(0, config.max_gaussians_per_tile - config.max_gaussians)
    depth_order = jnp.pad(depth_order, ((0, 0), (0, padding)))
    sorted_depths = jnp.pad(sorted_depths, ((0, 0), (0, padding)), constant_values=jnp.inf)
    valid_mask = jnp.isfinite(sorted_depths)
    return VisibilityTable(
        depth_order.astype(jnp.int32),
        sorted_depths,
        valid_mask,
        jnp.minimum(tile_counts, config.max_gaussians_per_tile),
        tile_counts > config.max_gaussians_per_tile,
    )
