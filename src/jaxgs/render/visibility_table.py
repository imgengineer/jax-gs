import chex
import jax.numpy as jnp
from flax import struct

from ..config import CapacityConfig
from ..scene.camera import Camera
from .projection import ProjectedGaussians


@struct.dataclass
class VisibilityTable:
    tile_gaussian_ids: chex.Array
    tile_depths: chex.Array
    tile_valid: chex.Array
    tile_count: chex.Array
    overflow: chex.Array


@struct.dataclass
class SortedVisibilityTable:
    """LiteGS-style global tile/Gaussian pairs and per-tile ranges."""

    gaussian_ids: chex.Array
    tile_offsets: chex.Array
    point_counts: chex.Array
    pair_count: chex.Array
    overflow: chex.Array


def build_visibility_table(
    projected: ProjectedGaussians,
    camera: Camera,
    config: CapacityConfig,
    cluster_tile_mask: chex.Array | None = None,
) -> VisibilityTable:
    """Reference binning; O(C*T) and intended only for small correctness cases."""
    tiles_x = (camera.width + config.tile_size - 1) // config.tile_size
    tiles_y = (camera.height + config.tile_size - 1) // config.tile_size
    tile_x = jnp.tile(jnp.arange(tiles_x), tiles_y) * config.tile_size
    tile_y = jnp.repeat(jnp.arange(tiles_y), tiles_x) * config.tile_size
    px = projected.mean[:, 0][None, :]
    py = projected.mean[:, 1][None, :]
    radius = projected.radius[None, :]
    intersects = (
        (px + radius >= tile_x[:, None])
        & (px - radius < tile_x[:, None] + config.tile_size)
        & (py + radius >= tile_y[:, None])
        & (py - radius < tile_y[:, None] + config.tile_size)
        & projected.visible[None, :]
    )
    if cluster_tile_mask is not None:
        cluster_ids = jnp.arange(config.max_gaussians) // config.cluster_size
        intersects = intersects & cluster_tile_mask[cluster_ids].T
    counts = jnp.sum(intersects, axis=1, dtype=jnp.int32)
    depths = jnp.where(intersects, projected.depth[None, :], jnp.inf)
    order = jnp.argsort(depths, axis=1, stable=True)[:, : config.max_gaussians_per_tile]
    sorted_depths = jnp.take_along_axis(depths, order, axis=1)
    padding = max(0, config.max_gaussians_per_tile - config.max_gaussians)
    order = jnp.pad(order, ((0, 0), (0, padding)))
    sorted_depths = jnp.pad(sorted_depths, ((0, 0), (0, padding)), constant_values=jnp.inf)
    valid = jnp.isfinite(sorted_depths)
    return VisibilityTable(
        order.astype(jnp.int32),
        sorted_depths,
        valid,
        jnp.minimum(counts, config.max_gaussians_per_tile),
        counts > config.max_gaussians_per_tile,
    )
