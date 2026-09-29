import chex
import jax.numpy as jnp

from ..config import CapacityConfig
from ..scene.camera import Camera
from .projection import ProjectedGaussians


def cluster_culling(
    projected: ProjectedGaussians, camera: Camera, config: CapacityConfig
) -> chex.Array:
    """Conservative projected cluster AABB against each image tile."""
    c = config.max_gaussians
    pad = config.num_clusters * config.cluster_size - c
    shape = (config.num_clusters, config.cluster_size)
    visible = jnp.pad(projected.visible, (0, pad)).reshape(shape)
    x = jnp.pad(projected.mean[:, 0], (0, pad)).reshape(shape)
    y = jnp.pad(projected.mean[:, 1], (0, pad)).reshape(shape)
    radius = jnp.pad(projected.radius, (0, pad)).reshape(shape)
    minimum_x = jnp.min(jnp.where(visible, x - radius, jnp.inf), axis=1)
    maximum_x = jnp.max(jnp.where(visible, x + radius, -jnp.inf), axis=1)
    minimum_y = jnp.min(jnp.where(visible, y - radius, jnp.inf), axis=1)
    maximum_y = jnp.max(jnp.where(visible, y + radius, -jnp.inf), axis=1)
    tiles_x = (camera.width + config.tile_size - 1) // config.tile_size
    tiles_y = (camera.height + config.tile_size - 1) // config.tile_size
    tile_x = jnp.tile(jnp.arange(tiles_x), tiles_y) * config.tile_size
    tile_y = jnp.repeat(jnp.arange(tiles_y), tiles_x) * config.tile_size
    return (
        (minimum_x[:, None] < tile_x[None] + config.tile_size)
        & (maximum_x[:, None] >= tile_x[None])
        & (minimum_y[:, None] < tile_y[None] + config.tile_size)
        & (maximum_y[:, None] >= tile_y[None])
    )
