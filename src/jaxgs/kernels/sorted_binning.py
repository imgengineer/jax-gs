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

    from .pair_sort import sort_pairs_by_tile
    from .sorted_visibility import (
        _SCAN_ITEMS,
        _SCAN_THREADS,
        launch_count_pairs,
        launch_emit_pairs,
        launch_pair_offsets,
        launch_tile_ranges,
    )

    if jax.default_backend() != "gpu":
        raise RuntimeError("CuTe binning requires a JAX CUDA device")
    capacity = config.max_gaussians
    max_pairs = config.visibility_capacity
    tiles_x = (camera.width + config.tile_size - 1) // config.tile_size
    tiles_y = (camera.height + config.raster_tile_height - 1) // config.raster_tile_height
    num_tiles = tiles_x * tiles_y
    # LiteGS limits radix sorting to the used tile bits; narrow keys halve
    # the key traffic. num_tiles is the padding key of filled arenas.
    tile_dtype = jnp.uint16 if num_tiles <= 65535 else jnp.uint32
    count = cutlass_call(
        launch_count_pairs,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct((capacity,), jnp.uint32),
        ),
        use_static_tensors=True,
        capacity=capacity,
        tile_size=config.tile_size,
        tile_height=config.raster_tile_height,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        width=camera.width,
        height=camera.height,
    )
    point_counts, depth_keys = count(
        projected.mean.reshape(-1),
        projected.conic.reshape(-1),
        projected.alpha,
        projected.depth,
        projected.visible.astype(jnp.int8),
    )
    # LiteGS sorts Gaussians by depth before duplication. A stable tile-only
    # sort then preserves depth order inside each tile (wrapper.py::Binning).
    # Emission walks the same depth order, writing each prefix-sum segment.
    # The uint32 keys order Gaussians with pairs exactly like their depths.
    _, depth_order = jax.lax.sort(
        (depth_keys, jnp.arange(capacity, dtype=jnp.int32)), num_keys=1, is_stable=True
    )
    scan_items = _SCAN_THREADS * _SCAN_ITEMS
    offsets = cutlass_call(
        launch_pair_offsets,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct(((capacity + scan_items - 1) // scan_items,), jnp.int32),
        ),
        use_static_tensors=True,
        capacity=capacity,
    )
    ordered_counts, end_offsets, _ = offsets(depth_order, point_counts)
    pair_count = end_offsets[-1]
    emit = cutlass_call(
        launch_emit_pairs,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((max_pairs,), tile_dtype),
            jax.ShapeDtypeStruct((max_pairs,), jnp.int32),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),  # queued large Gaussians
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ),
        use_static_tensors=True,
        capacity=capacity,
        max_pairs=max_pairs,
        tile_size=config.tile_size,
        tile_height=config.raster_tile_height,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
    )
    tile_ids, gaussian_ids, _, _ = emit(
        projected.mean.reshape(-1),
        projected.conic.reshape(-1),
        projected.alpha,
        depth_order,
        ordered_counts,
        end_offsets,
    )
    # A stable tile sort of only the emitted pairs; slots past them stay unread.
    tile_ids, gaussian_ids = sort_pairs_by_tile(tile_ids, gaussian_ids, pair_count, num_tiles)
    ranges = cutlass_call(
        launch_tile_ranges,
        output_shape_dtype=jax.ShapeDtypeStruct((num_tiles + 1,), jnp.int32),
        use_static_tensors=True,
        max_pairs=max_pairs,
        num_tiles=num_tiles,
    )
    tile_offsets = ranges(tile_ids, pair_count.reshape(1))
    return SortedVisibilityTable(
        gaussian_ids, tile_offsets, point_counts, pair_count, pair_count > max_pairs
    )
