import jax
import jax.numpy as jnp

from ..config import CapacityConfig
from ..render.types import ProjectedGaussians, SortedVisibilityTable
from ..scene.camera import Camera


def build_sorted_visibility_table_cute(
    projected: ProjectedGaussians, camera: Camera, config: CapacityConfig
) -> SortedVisibilityTable:
    """LiteGS count/prefix/duplicate/sort/range pipeline with a static pair arena."""
    from cutlass.jax import cutlass_call

    from .sorted_visibility import launch_count_pairs, launch_emit_pairs, launch_tile_ranges

    if jax.default_backend() != "gpu":
        raise RuntimeError("CuTe binning requires a JAX CUDA device")
    capacity = config.max_gaussians
    max_pairs = config.visibility_capacity
    tiles_x = (camera.width + config.tile_size - 1) // config.tile_size
    tiles_y = (camera.height + config.raster_tile_height - 1) // config.raster_tile_height
    num_tiles = tiles_x * tiles_y
    # LiteGS limits radix sorting to the used tile bits. Narrow keys let XLA
    # avoid unused high bits too; num_tiles itself is the padding sentinel.
    tile_dtype = jnp.uint16 if num_tiles <= 65535 else jnp.uint32
    count = cutlass_call(
        launch_count_pairs,
        output_shape_dtype=jax.ShapeDtypeStruct((capacity,), jnp.int32),
        use_static_tensors=True,
        capacity=capacity,
        tile_size=config.tile_size,
        tile_height=config.raster_tile_height,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        width=camera.width,
        height=camera.height,
    )
    point_counts = count(
        projected.mean.reshape(-1),
        projected.conic.reshape(-1),
        projected.alpha,
        projected.visible.astype(jnp.int8),
    )
    # LiteGS sorts Gaussians by depth before duplication. A stable tile-only
    # sort then preserves depth order inside each tile (wrapper.py::Binning).
    # Emission walks the same depth order, writing each prefix-sum segment.
    depth_order = jnp.argsort(projected.depth, stable=True).astype(jnp.int32)
    ordered_counts = point_counts[depth_order]
    end_offsets = jnp.cumsum(ordered_counts, dtype=jnp.int32)
    pair_count = end_offsets[-1]
    emit = cutlass_call(
        launch_emit_pairs,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((max_pairs,), tile_dtype),
            jax.ShapeDtypeStruct((max_pairs,), jnp.int32),
        ),
        use_static_tensors=True,
        capacity=capacity,
        max_pairs=max_pairs,
        tile_size=config.tile_size,
        tile_height=config.raster_tile_height,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
    )
    tile_ids, gaussian_ids = emit(
        projected.mean.reshape(-1),
        projected.conic.reshape(-1),
        projected.alpha,
        depth_order,
        ordered_counts,
        end_offsets,
    )
    # One sort path keeps the training update asynchronous.
    tile_ids, gaussian_ids = jax.lax.sort(
        (tile_ids, gaussian_ids), dimension=0, num_keys=1, is_stable=True
    )
    ranges = cutlass_call(
        launch_tile_ranges,
        output_shape_dtype=jax.ShapeDtypeStruct((num_tiles + 1,), jnp.int32),
        use_static_tensors=True,
        max_pairs=max_pairs,
        num_tiles=num_tiles,
    )
    tile_offsets = ranges(tile_ids)
    return SortedVisibilityTable(
        gaussian_ids, tile_offsets, point_counts, pair_count, pair_count > max_pairs
    )
