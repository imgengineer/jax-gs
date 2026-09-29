import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..config import CapacityConfig
from ..render.projection import ProjectedGaussians
from ..render.visibility_table import VisibilityTable
from ..scene.camera import Camera


@struct.dataclass
class ClusterList:
    bounds: chex.Array  # [num_clusters, 4]
    ids: chex.Array  # [num_clusters]
    count: chex.Array  # [1]


def compact_clusters_cute(projected: ProjectedGaussians, config: CapacityConfig) -> ClusterList:
    from cutlass.jax import cutlass_call

    from .visibility import launch_cluster_compact

    if jax.default_backend() != "gpu":
        raise RuntimeError("CuTe binning requires a JAX CUDA device")
    c = config.max_gaussians
    clusters = config.num_clusters
    shapes = (
        jax.ShapeDtypeStruct((clusters * 4,), jnp.float32),
        jax.ShapeDtypeStruct((clusters,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    call = cutlass_call(
        launch_cluster_compact,
        output_shape_dtype=shapes,
        use_static_tensors=True,
        capacity=c,
        cluster_size=config.cluster_size,
        num_clusters=clusters,
    )
    bounds, ids, count = call(
        projected.mean.reshape(-1), projected.radius, projected.visible.astype(jnp.int8)
    )
    return ClusterList(bounds.reshape(clusters, 4), ids, count)


def build_visibility_table_cute(
    projected: ProjectedGaussians,
    camera: Camera,
    config: CapacityConfig,
    clusters: ClusterList | None = None,
) -> VisibilityTable:
    from cutlass.jax import cutlass_call

    from .visibility import launch_visibility_table

    clusters = compact_clusters_cute(projected, config) if clusters is None else clusters
    tiles_x = (camera.width + config.tile_size - 1) // config.tile_size
    tiles_y = (camera.height + config.tile_size - 1) // config.tile_size
    tiles = tiles_x * tiles_y
    k_max = config.max_gaussians_per_tile
    shapes = (
        jax.ShapeDtypeStruct((tiles * k_max,), jnp.int32),
        jax.ShapeDtypeStruct((tiles * k_max,), jnp.float32),
        jax.ShapeDtypeStruct((tiles * k_max,), jnp.int8),
        jax.ShapeDtypeStruct((tiles,), jnp.int32),
        jax.ShapeDtypeStruct((tiles,), jnp.int8),
    )
    call = cutlass_call(
        launch_visibility_table,
        output_shape_dtype=shapes,
        use_static_tensors=True,
        capacity=config.max_gaussians,
        cluster_size=config.cluster_size,
        k_max=k_max,
        tile_size=config.tile_size,
        tiles_x=tiles_x,
        num_tiles=tiles,
    )
    ids, depths, valid, count, overflow = call(
        projected.mean.reshape(-1),
        projected.depth,
        projected.radius,
        projected.visible.astype(jnp.int8),
        clusters.bounds.reshape(-1),
        clusters.ids,
        clusters.count,
    )
    valid = valid.reshape(tiles, k_max).astype(jnp.bool_)
    return VisibilityTable(
        ids.reshape(tiles, k_max),
        jnp.where(valid, depths.reshape(tiles, k_max), jnp.inf),
        valid,
        count,
        overflow.astype(jnp.bool_),
    )
