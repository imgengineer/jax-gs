"""Reference cluster visibility against image tiles."""

import chex
import jax.numpy as jnp

from ..config import CapacityConfig
from ..scene.camera import Camera
from .projection import support_radius
from .types import ProjectedGaussians


def build_cluster_tile_mask(
    projected: ProjectedGaussians, camera: Camera, config: CapacityConfig
) -> chex.Array:
    """Return a conservative mask of clusters intersecting each image tile."""
    capacity = config.max_gaussians
    padding = config.num_clusters * config.cluster_size - capacity
    cluster_shape = (config.num_clusters, config.cluster_size)
    cluster_visible = jnp.pad(projected.visible, (0, padding)).reshape(cluster_shape)
    projected_x = jnp.pad(projected.mean[:, 0], (0, padding)).reshape(cluster_shape)
    projected_y = jnp.pad(projected.mean[:, 1], (0, padding)).reshape(cluster_shape)
    support_radii = jnp.pad(support_radius(projected), (0, padding)).reshape(cluster_shape)
    minimum_x = jnp.min(jnp.where(cluster_visible, projected_x - support_radii, jnp.inf), axis=1)
    maximum_x = jnp.max(jnp.where(cluster_visible, projected_x + support_radii, -jnp.inf), axis=1)
    minimum_y = jnp.min(jnp.where(cluster_visible, projected_y - support_radii, jnp.inf), axis=1)
    maximum_y = jnp.max(jnp.where(cluster_visible, projected_y + support_radii, -jnp.inf), axis=1)
    tiles_x = (camera.width + config.tile_size - 1) // config.tile_size
    tile_height = config.raster_tile_height
    tiles_y = (camera.height + tile_height - 1) // tile_height
    tile_x = jnp.tile(jnp.arange(tiles_x), tiles_y) * config.tile_size
    tile_y = jnp.repeat(jnp.arange(tiles_y), tiles_x) * tile_height
    return (
        (minimum_x[:, None] < tile_x[None] + config.tile_size)
        & (maximum_x[:, None] >= tile_x[None])
        & (minimum_y[:, None] < tile_y[None] + tile_height)
        & (maximum_y[:, None] >= tile_y[None])
    )
