"""Sparse tile layout and intersection primitives.

The upstream CUDA operators return runtime-length arrays.  JAX keeps those
arrays at static capacity and exposes the retained and required lengths on the
result objects.  Iterating either result yields only the upstream-compatible
array values.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Iterator, NamedTuple

import jax
import jax.numpy as jnp

from .low_level import (
    DEFAULT_ALPHA_THRESHOLD,
    DEFAULT_TRANSMITTANCE_THRESHOLD,
    MAX_ALPHA,
    _broadcast_means_with_absgrad_probe,
)


_WORD_BITS = 32
_MAX_INT32_INDEX = 2**30 - 2


def _static_int(name: str, value: int, *, minimum: int) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a static integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _normalize_count(value: jax.Array | int | None, capacity: int) -> jax.Array:
    if value is None:
        return jnp.asarray(capacity, dtype=jnp.int32)
    return jnp.clip(jnp.asarray(value, dtype=jnp.int32), 0, capacity)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PaddedSparseTileLayout:
    """Static-shape result of :func:`build_sparse_tile_layout`.

    The first ``valid_count`` rows of ``active_tiles``, ``tile_pixel_mask`` and
    ``tile_pixel_cumsum`` are meaningful.  ``required_count`` is the exact
    number of active tiles, so ``overflow`` cannot occur with the default
    capacity.
    """

    active_tiles: jax.Array
    active_tile_mask: jax.Array
    tile_pixel_mask: jax.Array
    tile_pixel_cumsum: jax.Array
    pixel_map: jax.Array
    valid_count: jax.Array
    required_count: jax.Array
    overflow: jax.Array

    def __iter__(self) -> Iterator[jax.Array]:
        yield self.active_tiles
        yield self.active_tile_mask
        yield self.tile_pixel_mask
        yield self.tile_pixel_cumsum
        yield self.pixel_map

    def tree_flatten(self):
        return (
            (
                self.active_tiles,
                self.active_tile_mask,
                self.tile_pixel_mask,
                self.tile_pixel_cumsum,
                self.pixel_map,
                self.valid_count,
                self.required_count,
                self.overflow,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PaddedSparseIntersections:
    """Static-shape result of :func:`isect_tiles_sparse`.

    ``tile_offsets`` covers the padded active-tile capacity and ends with a
    sentinel equal to ``valid_count``.  Only the first ``valid_count`` entries
    of ``flatten_ids`` are meaningful.
    """

    tile_offsets: jax.Array
    flatten_ids: jax.Array
    valid_count: jax.Array
    required_count: jax.Array
    active_tile_count: jax.Array
    overflow: jax.Array

    def __iter__(self) -> Iterator[jax.Array]:
        yield self.tile_offsets
        yield self.flatten_ids

    def tree_flatten(self):
        return (
            (
                self.tile_offsets,
                self.flatten_ids,
                self.valid_count,
                self.required_count,
                self.active_tile_count,
                self.overflow,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


class _DecodedSparsePixels(NamedTuple):
    active_ranks: jax.Array
    image_ids: jax.Array
    rows: jax.Array
    columns: jax.Array
    output_slots: jax.Array
    valid: jax.Array
    active_tile_count: jax.Array
    decoded_count: jax.Array
    layout_error: jax.Array


class _SparseGaussianData(NamedTuple):
    means2d: jax.Array
    conics: jax.Array
    colors: jax.Array | None
    opacities: jax.Array
    slot_count: int
    channels: int | None
    dense_gaussians_per_image: int | None
    image_count: int | None


class _SparseIntersectionData(NamedTuple):
    tile_offsets: jax.Array
    flatten_ids: jax.Array
    valid_count: jax.Array
    error: jax.Array


def _raise_sparse_rasterization_overflow() -> None:
    raise RuntimeError(
        "sparse rasterization received an incomplete or inconsistent padded "
        "layout; inspect return_info=True and increase the reported capacity"
    )


def build_sparse_tile_layout(
    pixels: jax.Array,
    image_ids: jax.Array,
    n_images: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    *,
    max_active_tiles: int | None = None,
) -> PaddedSparseTileLayout:
    """Build the tile bookkeeping for a packed set of requested pixels.

    ``pixels`` uses ``(row, column)`` order and must contain no duplicate
    ``(image, row, column)`` entries.  Coordinates and image ids must lie in
    the supplied tile grid.  These are the same preconditions as gsplat's CUDA
    operator.

    gsplat stores 64 pixels per ``uint64`` mask word.  This project keeps JAX
    x64 disabled, so each ``uint32`` word stores 32 pixels instead.  The mask
    remains raster ordered and is consumed through this module's public sparse
    functions rather than by assuming an upstream word width.
    """

    n_images = _static_int("n_images", n_images, minimum=0)
    tile_size = _static_int("tile_size", tile_size, minimum=1)
    tile_width = _static_int("tile_width", tile_width, minimum=1)
    tile_height = _static_int("tile_height", tile_height, minimum=1)
    dense_tile_count = n_images * tile_width * tile_height
    if dense_tile_count > _MAX_INT32_INDEX:
        raise ValueError("the image/tile grid is too large for int32 indexing")

    pixels = jax.lax.stop_gradient(jnp.asarray(pixels, dtype=jnp.int32))
    image_ids = jax.lax.stop_gradient(jnp.asarray(image_ids, dtype=jnp.int32))
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must have shape [P, 2]")
    pixel_count = pixels.shape[0]
    if image_ids.shape != (pixel_count,):
        raise ValueError("image_ids must have shape [P]")
    if n_images == 0 and pixel_count:
        raise ValueError("n_images must be positive when pixels are non-empty")

    default_capacity = min(pixel_count, dense_tile_count)
    if max_active_tiles is None:
        active_capacity = default_capacity
    else:
        active_capacity = _static_int(
            "max_active_tiles", max_active_tiles, minimum=0
        )
    if active_capacity > _MAX_INT32_INDEX:
        raise ValueError("max_active_tiles is too large for int32 indexing")

    rows = pixels[:, 0]
    columns = pixels[:, 1]
    tiles_per_image = tile_width * tile_height
    tile_rows = rows // tile_size
    tile_columns = columns // tile_size
    dense_tile_ids = (
        image_ids * tiles_per_image + tile_rows * tile_width + tile_columns
    )
    pixel_ids_in_tile = (
        (rows % tile_size) * tile_size + columns % tile_size
    )

    active_tile_mask_flat = jnp.zeros(
        (dense_tile_count,), dtype=jnp.bool_
    ).at[dense_tile_ids].set(True)
    required_count = jnp.count_nonzero(active_tile_mask_flat).astype(jnp.int32)
    valid_count = jnp.minimum(required_count, jnp.int32(active_capacity))
    overflow = required_count > active_capacity
    active_tiles = jnp.nonzero(
        active_tile_mask_flat, size=active_capacity, fill_value=-1
    )[0].astype(jnp.int32)
    active_slots = jnp.arange(active_capacity, dtype=jnp.int32) < valid_count
    active_tiles = jnp.where(active_slots, active_tiles, -1)

    original_ids = jnp.arange(pixel_count, dtype=jnp.int32)
    pixel_map = jnp.lexsort(
        (original_ids, pixel_ids_in_tile, dense_tile_ids)
    ).astype(jnp.int32)

    words_per_tile = (
        tile_size * tile_size + _WORD_BITS - 1
    ) // _WORD_BITS
    if active_capacity:
        safe_active_tiles = jnp.where(active_slots, active_tiles, 0)
        tile_to_active = jnp.full(
            (dense_tile_count,), -1, dtype=jnp.int32
        ).at[safe_active_tiles].max(
            jnp.where(
                active_slots,
                jnp.arange(active_capacity, dtype=jnp.int32),
                -1,
            )
        )
        active_ranks = tile_to_active[dense_tile_ids]
        retained = active_ranks >= 0
        safe_ranks = jnp.maximum(active_ranks, 0)
        word_ids = pixel_ids_in_tile // _WORD_BITS
        bit_ids = (pixel_ids_in_tile % _WORD_BITS).astype(jnp.uint32)
        bit_values = jnp.left_shift(jnp.uint32(1), bit_ids)
        tile_pixel_mask = jnp.zeros(
            (active_capacity, words_per_tile), dtype=jnp.uint32
        ).at[safe_ranks, word_ids].add(
            jnp.where(retained, bit_values, jnp.uint32(0))
        )
        tile_counts = jnp.zeros(
            (active_capacity,), dtype=jnp.int32
        ).at[safe_ranks].add(retained.astype(jnp.int32))
        tile_pixel_cumsum = jnp.cumsum(tile_counts, dtype=jnp.int32)
    else:
        tile_pixel_mask = jnp.zeros(
            (0, words_per_tile), dtype=jnp.uint32
        )
        tile_pixel_cumsum = jnp.zeros((1,), dtype=jnp.int32)

    return PaddedSparseTileLayout(
        active_tiles,
        active_tile_mask_flat.reshape(
            n_images, tile_height, tile_width
        ),
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        valid_count,
        required_count,
        overflow,
    )


def isect_tiles_sparse(
    means2d: jax.Array,
    radii: jax.Array,
    depths: jax.Array,
    tile_mask: jax.Array,
    active_tiles: jax.Array,
    n_images: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    image_ids: jax.Array | None = None,
    *,
    active_tile_count: jax.Array | int | None = None,
    max_intersections: int | None = None,
) -> PaddedSparseIntersections:
    """Intersect projected Gaussians with only the requested active tiles.

    Dense inputs have shape ``[I, N, ...]``.  Packed inputs have shape
    ``[nnz, ...]`` and require ``image_ids``.  Results are ordered by active
    tile, float32 depth, and flatten id.  The final key makes equal-depth output
    deterministic; upstream leaves tie ordering unspecified.

    By default, the static intersection capacity is the complete Cartesian
    bound ``number_of_gaussians * active_tile_capacity``.  Passing a smaller
    ``max_intersections`` saves memory but requires checking ``overflow``.
    """

    n_images = _static_int("n_images", n_images, minimum=1)
    tile_size = _static_int("tile_size", tile_size, minimum=1)
    tile_width = _static_int("tile_width", tile_width, minimum=1)
    tile_height = _static_int("tile_height", tile_height, minimum=1)
    tiles_per_image = tile_width * tile_height
    dense_tile_count = n_images * tiles_per_image
    if dense_tile_count > _MAX_INT32_INDEX:
        raise ValueError("the image/tile grid is too large for int32 indexing")

    means2d = jax.lax.stop_gradient(jnp.asarray(means2d))
    radii = jax.lax.stop_gradient(jnp.asarray(radii))
    depths = jax.lax.stop_gradient(jnp.asarray(depths))
    tile_mask = jax.lax.stop_gradient(jnp.asarray(tile_mask, dtype=jnp.bool_))
    active_tiles = jax.lax.stop_gradient(
        jnp.asarray(active_tiles, dtype=jnp.int32)
    )
    if tile_mask.shape != (n_images, tile_height, tile_width):
        raise ValueError(
            "tile_mask must have shape [n_images, tile_height, tile_width]"
        )
    if active_tiles.ndim != 1:
        raise ValueError("active_tiles must have shape [active_capacity]")
    active_capacity = active_tiles.shape[0]
    normalized_active_count = _normalize_count(
        active_tile_count, active_capacity
    )
    if active_tile_count is None:
        normalized_active_count = jnp.count_nonzero(
            (active_tiles >= 0) & (active_tiles < dense_tile_count)
        ).astype(jnp.int32)

    packed = means2d.ndim == 2
    if packed:
        if means2d.shape[-1] != 2:
            raise ValueError("packed means2d must have shape [nnz, 2]")
        gaussian_count = means2d.shape[0]
        if radii.shape != (gaussian_count, 2):
            raise ValueError("packed radii must have shape [nnz, 2]")
        if depths.shape != (gaussian_count,):
            raise ValueError("packed depths must have shape [nnz]")
        if image_ids is None:
            raise ValueError("packed inputs require image_ids")
        image_of = jax.lax.stop_gradient(
            jnp.asarray(image_ids, dtype=jnp.int32)
        )
        if image_of.shape != (gaussian_count,):
            raise ValueError("image_ids must have shape [nnz]")
        flat_means = means2d
        flat_radii = radii
        flat_depths = depths
    else:
        if means2d.ndim != 3 or means2d.shape[-1] != 2:
            raise ValueError("dense means2d must have shape [I, N, 2]")
        if means2d.shape[0] != n_images:
            raise ValueError("dense means2d image dimension must equal n_images")
        gaussians_per_image = means2d.shape[1]
        if radii.shape != (n_images, gaussians_per_image, 2):
            raise ValueError("dense radii shape does not match means2d")
        if depths.shape != (n_images, gaussians_per_image):
            raise ValueError("dense depths shape does not match means2d")
        gaussian_count = n_images * gaussians_per_image
        flat_means = means2d.reshape(gaussian_count, 2)
        flat_radii = radii.reshape(gaussian_count, 2)
        flat_depths = depths.reshape(gaussian_count)
        image_of = jnp.repeat(
            jnp.arange(n_images, dtype=jnp.int32), gaussians_per_image
        )

    complete_capacity = gaussian_count * active_capacity
    if complete_capacity > _MAX_INT32_INDEX:
        raise ValueError(
            "the complete sparse intersection bound is too large for int32 indexing; "
            "pass fewer active tiles or Gaussians"
        )
    if max_intersections is None:
        intersection_capacity = complete_capacity
    else:
        intersection_capacity = _static_int(
            "max_intersections", max_intersections, minimum=0
        )
    if intersection_capacity > _MAX_INT32_INDEX:
        raise ValueError("max_intersections is too large for int32 indexing")

    empty_ids = jnp.full(
        (intersection_capacity,), -1, dtype=jnp.int32
    )
    empty_offsets = jnp.zeros((active_capacity + 1,), dtype=jnp.int32)
    if gaussian_count == 0 or active_capacity == 0:
        return PaddedSparseIntersections(
            empty_offsets,
            empty_ids,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            normalized_active_count,
            jnp.asarray(False),
        )

    finite = (
        jnp.all(jnp.isfinite(flat_means), axis=-1)
        & jnp.all(jnp.isfinite(flat_radii), axis=-1)
        & jnp.isfinite(flat_depths)
    )
    safe_means = jnp.where(finite[:, None], flat_means, 0.0)
    safe_radii = jnp.where(finite[:, None], flat_radii, 0.0)
    tile_min = jnp.floor((safe_means - safe_radii) / tile_size).astype(
        jnp.int32
    )
    tile_max = jnp.ceil((safe_means + safe_radii) / tile_size).astype(
        jnp.int32
    )
    min_x = jnp.clip(tile_min[:, 0], 0, tile_width)
    min_y = jnp.clip(tile_min[:, 1], 0, tile_height)
    max_x = jnp.clip(tile_max[:, 0], 0, tile_width)
    max_y = jnp.clip(tile_max[:, 1], 0, tile_height)
    gaussian_valid = (
        finite
        & jnp.all(flat_radii > 0, axis=-1)
        & (image_of >= 0)
        & (image_of < n_images)
    )

    active_ranks = jnp.arange(active_capacity, dtype=jnp.int32)
    active_in_range = (
        (active_ranks < normalized_active_count)
        & (active_tiles >= 0)
        & (active_tiles < dense_tile_count)
    )
    safe_active_tiles = jnp.clip(active_tiles, 0, dense_tile_count - 1)
    active_enabled = active_in_range & tile_mask.reshape(-1)[safe_active_tiles]
    active_images = safe_active_tiles // tiles_per_image
    tile_ids_in_image = safe_active_tiles % tiles_per_image
    active_y = tile_ids_in_image // tile_width
    active_x = tile_ids_in_image % tile_width

    overlaps = (
        gaussian_valid[:, None]
        & active_enabled[None, :]
        & (image_of[:, None] == active_images[None, :])
        & (active_x[None, :] >= min_x[:, None])
        & (active_x[None, :] < max_x[:, None])
        & (active_y[None, :] >= min_y[:, None])
        & (active_y[None, :] < max_y[:, None])
    )
    required_count = jnp.count_nonzero(overlaps).astype(jnp.int32)
    valid_count = jnp.minimum(
        required_count, jnp.int32(intersection_capacity)
    )
    overflow = required_count > intersection_capacity
    if intersection_capacity == 0:
        return PaddedSparseIntersections(
            empty_offsets,
            empty_ids,
            valid_count,
            required_count,
            normalized_active_count,
            overflow,
        )

    overlap_positions = jnp.nonzero(
        overlaps.reshape(-1), size=intersection_capacity, fill_value=0
    )[0]
    selected_gaussians = overlap_positions // active_capacity
    selected_active_ranks = overlap_positions % active_capacity
    selected_valid = (
        jnp.arange(intersection_capacity, dtype=jnp.int32) < valid_count
    )
    selected_depths = flat_depths[selected_gaussians].astype(jnp.float32)
    order = jax.lax.stop_gradient(
        jnp.lexsort(
            (
                selected_gaussians,
                selected_depths,
                selected_active_ranks,
                (~selected_valid).astype(jnp.int32),
            )
        )
    )
    selected_gaussians = selected_gaussians[order]
    selected_active_ranks = selected_active_ranks[order]
    selected_valid = selected_valid[order]
    flatten_ids = jnp.where(
        selected_valid, selected_gaussians, -1
    ).astype(jnp.int32)

    safe_ranks = jnp.where(selected_valid, selected_active_ranks, 0)
    counts = jnp.zeros((active_capacity,), dtype=jnp.int32).at[
        safe_ranks
    ].add(selected_valid.astype(jnp.int32))
    tile_offsets = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(counts, dtype=jnp.int32),
        )
    )
    return PaddedSparseIntersections(
        tile_offsets,
        flatten_ids,
        valid_count,
        required_count,
        normalized_active_count,
        overflow,
    )


def _decode_sparse_pixels(
    active_tiles: jax.Array,
    tile_pixel_mask: jax.Array,
    tile_pixel_cumsum: jax.Array,
    pixel_map: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    *,
    active_tile_count: jax.Array | int | None,
    image_ids: jax.Array | None = None,
) -> _DecodedSparsePixels:
    """Decode the uint32 layout back into tile/raster-ordered pixels."""

    active_tiles = jax.lax.stop_gradient(
        jnp.asarray(active_tiles, dtype=jnp.int32)
    )
    raw_pixel_mask = jnp.asarray(tile_pixel_mask)
    if raw_pixel_mask.dtype != jnp.uint32:
        raise TypeError(
            "tile_pixel_mask must use jax-gs uint32 words; rebuild it with "
            "build_sparse_tile_layout"
        )
    tile_pixel_mask = jax.lax.stop_gradient(raw_pixel_mask)
    tile_pixel_cumsum = jax.lax.stop_gradient(
        jnp.asarray(tile_pixel_cumsum, dtype=jnp.int32)
    )
    pixel_map = jax.lax.stop_gradient(jnp.asarray(pixel_map, dtype=jnp.int32))
    if active_tiles.ndim != 1:
        raise ValueError("active_tiles must have shape [active_capacity]")
    active_capacity = active_tiles.shape[0]
    pixel_count = pixel_map.shape[0]
    if pixel_map.ndim != 1:
        raise ValueError("pixel_map must have shape [P]")
    words_per_tile = (tile_size * tile_size + _WORD_BITS - 1) // _WORD_BITS
    if tile_pixel_mask.shape != (active_capacity, words_per_tile):
        raise ValueError(
            "tile_pixel_mask must have shape [active_capacity, words_per_tile]"
        )
    expected_cumsum_shape = (active_capacity,) if active_capacity else (1,)
    if tile_pixel_cumsum.shape != expected_cumsum_shape:
        raise ValueError(
            "tile_pixel_cumsum must have shape [active_capacity], or [1] for "
            "an empty active-tile layout"
        )
    if image_ids is not None:
        image_ids = jax.lax.stop_gradient(
            jnp.asarray(image_ids, dtype=jnp.int32)
        )
        if image_ids.shape != (pixel_count,):
            raise ValueError("image_ids must have shape [P]")

    normalized_active_count = _normalize_count(
        active_tile_count, active_capacity
    )
    if active_tile_count is None:
        normalized_active_count = jnp.count_nonzero(active_tiles >= 0).astype(
            jnp.int32
        )
    active_slots = (
        jnp.arange(active_capacity, dtype=jnp.int32)
        < normalized_active_count
    )
    active_in_range = active_slots & (active_tiles >= 0)
    tile_pixel_count = tile_size * tile_size
    pixel_ids_in_tile = jnp.arange(tile_pixel_count, dtype=jnp.int32)
    word_ids = pixel_ids_in_tile // _WORD_BITS
    bit_ids = (pixel_ids_in_tile % _WORD_BITS).astype(jnp.uint32)
    requested = (
        (
            tile_pixel_mask[:, word_ids]
            >> bit_ids[None, :]
        )
        & jnp.uint32(1)
    ).astype(jnp.bool_)
    requested = requested & active_in_range[:, None]
    decoded_count = jnp.count_nonzero(requested).astype(jnp.int32)
    selected_positions = jnp.nonzero(
        requested.reshape(-1), size=pixel_count, fill_value=0
    )[0]
    active_ranks = selected_positions // tile_pixel_count
    selected_pixel_ids = selected_positions % tile_pixel_count
    selected_valid = (
        jnp.arange(pixel_count, dtype=jnp.int32)
        < jnp.minimum(decoded_count, jnp.int32(pixel_count))
    )

    safe_active_ranks = jnp.clip(
        active_ranks, 0, max(active_capacity - 1, 0)
    )
    if active_capacity:
        selected_tiles = active_tiles[safe_active_ranks]
    else:
        selected_tiles = jnp.zeros((pixel_count,), dtype=jnp.int32)
    tile_range = tile_width * tile_height
    selected_tile_valid = (
        selected_valid
        & (selected_tiles >= 0)
        & (safe_active_ranks < normalized_active_count)
    )
    selected_images = jnp.maximum(selected_tiles, 0) // tile_range
    selected_tiles_in_image = jnp.maximum(selected_tiles, 0) % tile_range
    selected_tile_rows = selected_tiles_in_image // tile_width
    selected_tile_columns = selected_tiles_in_image % tile_width
    rows = selected_tile_rows * tile_size + selected_pixel_ids // tile_size
    columns = (
        selected_tile_columns * tile_size + selected_pixel_ids % tile_size
    )
    coordinates_valid = (
        (rows >= 0)
        & (rows < image_height)
        & (columns >= 0)
        & (columns < image_width)
    )

    output_slots_in_range = (pixel_map >= 0) & (pixel_map < pixel_count)
    safe_output_slots = jnp.clip(pixel_map, 0, max(pixel_count - 1, 0))
    if pixel_count:
        permutation_counts = jnp.zeros(
            (pixel_count,), dtype=jnp.int32
        ).at[safe_output_slots].add(output_slots_in_range.astype(jnp.int32))
        permutation_valid = jnp.all(permutation_counts == 1)
    else:
        permutation_valid = jnp.asarray(True)

    bit_counts = jnp.sum(requested, axis=1, dtype=jnp.int32)
    if active_capacity:
        expected_cumsum = jnp.cumsum(bit_counts, dtype=jnp.int32)
        cumsum_valid = jnp.all(tile_pixel_cumsum == expected_cumsum)
        sorted_active_valid = jnp.all(
            jnp.where(
                active_slots[1:],
                active_tiles[1:] > active_tiles[:-1],
                True,
            )
        ) & jnp.all((~active_slots) | (active_tiles >= 0))
    else:
        cumsum_valid = tile_pixel_cumsum[0] == 0
        sorted_active_valid = jnp.asarray(True)

    image_ids_valid = jnp.asarray(True)
    if image_ids is not None and pixel_count:
        sorted_supplied_images = image_ids[safe_output_slots]
        image_ids_valid = jnp.all(
            (~selected_valid)
            | (~output_slots_in_range)
            | (sorted_supplied_images == selected_images)
        )

    valid = (
        selected_tile_valid
        & coordinates_valid
        & output_slots_in_range
    )
    layout_error = (
        (decoded_count != pixel_count)
        | (~permutation_valid)
        | (~cumsum_valid)
        | (~sorted_active_valid)
        | (~image_ids_valid)
        | jnp.any(selected_valid & ~valid)
    )
    return _DecodedSparsePixels(
        active_ranks.astype(jnp.int32),
        selected_images.astype(jnp.int32),
        rows.astype(jnp.int32),
        columns.astype(jnp.int32),
        pixel_map,
        valid,
        normalized_active_count,
        decoded_count,
        layout_error,
    )


def _flatten_sparse_gaussians(
    means2d: jax.Array,
    conics: jax.Array,
    colors: jax.Array | None,
    opacities: jax.Array,
    *,
    packed: bool,
) -> _SparseGaussianData:
    means2d = jnp.asarray(means2d)
    conics = jnp.asarray(conics)
    opacities = jnp.asarray(opacities)
    colors = None if colors is None else jnp.asarray(colors)
    packed = bool(packed or means2d.ndim == 2)
    if packed:
        if means2d.ndim != 2 or means2d.shape[-1] != 2:
            raise ValueError("packed means2d must have shape [nnz, 2]")
        slot_count = means2d.shape[0]
        if conics.shape != (slot_count, 3):
            raise ValueError("packed conics must have shape [nnz, 3]")
        if opacities.shape != (slot_count,):
            raise ValueError("packed opacities must have shape [nnz]")
        channels = None
        if colors is not None:
            if colors.ndim != 2 or colors.shape[0] != slot_count:
                raise ValueError("packed colors must have shape [nnz, channels]")
            channels = colors.shape[-1]
        return _SparseGaussianData(
            means2d,
            conics,
            colors,
            opacities,
            slot_count,
            channels,
            None,
            None,
        )

    if means2d.ndim < 3 or means2d.shape[-1] != 2:
        raise ValueError("dense means2d must have shape [..., N, 2]")
    image_shape = means2d.shape[:-2]
    gaussian_count = means2d.shape[-2]
    if conics.shape != image_shape + (gaussian_count, 3):
        raise ValueError("dense conics shape does not match means2d")
    if opacities.shape != image_shape + (gaussian_count,):
        raise ValueError("dense opacities shape does not match means2d")
    channels = None
    if colors is not None:
        if colors.shape[:-1] != image_shape + (gaussian_count,):
            raise ValueError("dense colors shape does not match means2d")
        channels = colors.shape[-1]
    image_count = math.prod(image_shape)
    slot_count = image_count * gaussian_count
    return _SparseGaussianData(
        means2d.reshape(slot_count, 2),
        conics.reshape(slot_count, 3),
        None if colors is None else colors.reshape(slot_count, channels),
        opacities.reshape(slot_count),
        slot_count,
        channels,
        gaussian_count,
        image_count,
    )


def _validate_sparse_intersections(
    tile_offsets: jax.Array,
    flatten_ids: jax.Array,
    active_capacity: int,
    *,
    valid_count: jax.Array | int | None,
) -> _SparseIntersectionData:
    tile_offsets = jax.lax.stop_gradient(
        jnp.asarray(tile_offsets, dtype=jnp.int32)
    )
    flatten_ids = jax.lax.stop_gradient(
        jnp.asarray(flatten_ids, dtype=jnp.int32)
    )
    if tile_offsets.shape != (active_capacity + 1,):
        raise ValueError("tile_offsets must have shape [active_capacity + 1]")
    if flatten_ids.ndim != 1:
        raise ValueError("flatten_ids must have shape [intersection_capacity]")
    intersection_capacity = flatten_ids.shape[0]
    normalized_count = _normalize_count(valid_count, intersection_capacity)
    if valid_count is None:
        normalized_count = jnp.count_nonzero(flatten_ids >= 0).astype(jnp.int32)
    offsets_in_range = (tile_offsets >= 0) & (
        tile_offsets <= normalized_count
    )
    offsets_ordered = jnp.all(tile_offsets[1:] >= tile_offsets[:-1])
    intersection_slots = jnp.arange(intersection_capacity, dtype=jnp.int32)
    prefix_valid = jnp.all(
        (intersection_slots >= normalized_count) | (flatten_ids >= 0)
    )
    error = (
        (tile_offsets[0] != 0)
        | (tile_offsets[-1] != normalized_count)
        | (~jnp.all(offsets_in_range))
        | (~offsets_ordered)
        | (~prefix_valid)
    )
    return _SparseIntersectionData(
        tile_offsets,
        flatten_ids,
        normalized_count,
        error,
    )


def _sparse_sample_weights(
    active_rank: jax.Array,
    image_id: jax.Array,
    row: jax.Array,
    column: jax.Array,
    pixel_valid: jax.Array,
    tile_enabled: jax.Array,
    gaussians: _SparseGaussianData,
    intersections: _SparseIntersectionData,
    *,
    absgrad_probe: jax.Array | None = None,
    alpha_threshold: float,
    transmittance_threshold: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return flatten ids, radiance weights and accepted-sample flags."""

    intersection_capacity = intersections.flatten_ids.shape[0]
    candidate_slots = jnp.arange(intersection_capacity, dtype=jnp.int32)
    start = intersections.tile_offsets[active_rank]
    end = intersections.tile_offsets[active_rank + 1]
    positions = start + candidate_slots
    safe_positions = jnp.clip(
        positions, 0, max(intersection_capacity - 1, 0)
    )
    flatten_ids = intersections.flatten_ids[safe_positions]
    safe_flatten_ids = jnp.clip(
        flatten_ids, 0, max(gaussians.slot_count - 1, 0)
    )
    candidate_valid = (
        (positions < end)
        & (positions < intersections.valid_count)
        & (flatten_ids >= 0)
        & (flatten_ids < gaussians.slot_count)
        & pixel_valid
        & tile_enabled
    )
    if gaussians.dense_gaussians_per_image is not None:
        candidate_valid = candidate_valid & (
            safe_flatten_ids // gaussians.dense_gaussians_per_image == image_id
        )

    selected_means = gaussians.means2d[safe_flatten_ids]
    if absgrad_probe is not None:
        selected_means = _broadcast_means_with_absgrad_probe(
            selected_means,
            absgrad_probe[safe_flatten_ids],
            jnp.zeros((1, 2), dtype=selected_means.dtype),
        )[:, 0, :]
    selected_conics = gaussians.conics[safe_flatten_ids]
    selected_opacities = gaussians.opacities[safe_flatten_ids]
    pixel_x = column.astype(selected_means.dtype) + 0.5
    pixel_y = row.astype(selected_means.dtype) + 0.5
    delta_x = pixel_x - selected_means[:, 0]
    delta_y = pixel_y - selected_means[:, 1]
    sigma = (
        0.5
        * (
            selected_conics[:, 0] * delta_x**2
            + selected_conics[:, 2] * delta_y**2
        )
        + selected_conics[:, 1] * delta_x * delta_y
    )
    alpha = jnp.minimum(selected_opacities * jnp.exp(-sigma), MAX_ALPHA)
    alpha = jnp.nan_to_num(
        alpha, nan=0.0, posinf=MAX_ALPHA, neginf=0.0
    )
    alpha_valid = (
        candidate_valid
        & jnp.isfinite(sigma)
        & (sigma >= 0.0)
        & (alpha >= alpha_threshold)
    )
    alpha = jnp.where(alpha_valid, alpha, 0.0)
    transmittance = jnp.concatenate(
        (
            jnp.ones((1,), dtype=alpha.dtype),
            jnp.cumprod(1.0 - alpha)[:-1],
        )
    )
    next_transmittance = transmittance * (1.0 - alpha)
    accepted = alpha_valid & (
        next_transmittance > transmittance_threshold
    )
    weights = jnp.where(accepted, alpha * transmittance, 0.0)
    return safe_flatten_ids, weights, accepted


def rasterize_to_pixels_sparse(
    means2d: jax.Array,
    conics: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    image_ids: jax.Array,
    active_tiles: jax.Array,
    tile_offsets: jax.Array,
    flatten_ids: jax.Array,
    tile_pixel_mask: jax.Array,
    tile_pixel_cumsum: jax.Array,
    pixel_map: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    backgrounds: jax.Array | None = None,
    masks: jax.Array | None = None,
    packed: bool = False,
    absgrad: bool = False,
    *,
    active_tile_count: jax.Array | int | None = None,
    valid_count: jax.Array | int | None = None,
    overflow: jax.Array | bool = False,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
    return_info: bool = False,
    _means2d_absgrad_probe: jax.Array | None = None,
):
    """Rasterize only the pixels encoded by a sparse tile layout.

    The returned arrays have shapes ``[P, channels]`` and ``[P, 1]`` in the
    original requested-pixel order.  All retained tile intersections are
    scanned, so this readable reference path has no per-tile candidate limit.

    JAX arrays cannot carry PyTorch's mutable ``means2d.absgrad`` side
    attribute. To request the equivalent statistic, pass ``absgrad=True``
    together with an independent, zero-valued
    ``_means2d_absgrad_probe`` matching ``means2d``. Differentiate the loss
    jointly with respect to ``means2d`` and the probe: the former gradient is
    the ordinary signed VJP and the latter sums the absolute per-pixel VJP.
    The probe does not change forward values.
    """

    means2d = jnp.asarray(means2d)
    if absgrad and _means2d_absgrad_probe is None:
        raise ValueError(
            "absgrad=True requires a zero-valued _means2d_absgrad_probe; "
            "differentiate the loss jointly with respect to means2d and the "
            "probe to obtain signed gradient and compositor AbsGrad"
        )
    if not absgrad and _means2d_absgrad_probe is not None:
        raise ValueError("_means2d_absgrad_probe requires absgrad=True")
    if _means2d_absgrad_probe is not None:
        _means2d_absgrad_probe = jnp.asarray(_means2d_absgrad_probe)
        if _means2d_absgrad_probe.shape != means2d.shape:
            raise ValueError(
                "_means2d_absgrad_probe must have the same shape as means2d"
            )
        if _means2d_absgrad_probe.dtype != means2d.dtype:
            raise TypeError(
                "_means2d_absgrad_probe must have the same dtype as means2d"
            )
    image_width = _static_int("image_width", image_width, minimum=1)
    image_height = _static_int("image_height", image_height, minimum=1)
    tile_size = _static_int("tile_size", tile_size, minimum=1)
    tile_width = _static_int("tile_width", tile_width, minimum=1)
    tile_height = _static_int("tile_height", tile_height, minimum=1)
    if tile_width * tile_size < image_width:
        raise ValueError("tile_width does not cover image_width")
    if tile_height * tile_size < image_height:
        raise ValueError("tile_height does not cover image_height")
    if not math.isfinite(alpha_threshold) or alpha_threshold <= 0.0:
        raise ValueError("alpha_threshold must be positive and finite")
    if (
        not math.isfinite(transmittance_threshold)
        or transmittance_threshold <= 0.0
    ):
        raise ValueError(
            "transmittance_threshold must be positive and finite"
        )

    gaussians = _flatten_sparse_gaussians(
        means2d, conics, colors, opacities, packed=packed
    )
    flat_absgrad_probe = (
        None
        if _means2d_absgrad_probe is None
        else _means2d_absgrad_probe.reshape(gaussians.slot_count, 2)
    )
    assert gaussians.colors is not None and gaussians.channels is not None
    image_ids = jnp.asarray(image_ids, dtype=jnp.int32)
    decoded = _decode_sparse_pixels(
        active_tiles,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        active_tile_count=active_tile_count,
        image_ids=image_ids,
    )
    intersections = _validate_sparse_intersections(
        tile_offsets,
        flatten_ids,
        jnp.asarray(active_tiles).shape[0],
        valid_count=valid_count,
    )
    pixel_count = decoded.output_slots.shape[0]
    channels = gaussians.channels
    dense_image_error = jnp.asarray(False)
    if gaussians.image_count is not None:
        dense_image_error = jnp.any(
            decoded.valid
            & (
                (decoded.image_ids < 0)
                | (decoded.image_ids >= gaussians.image_count)
            )
        )

    tile_enabled = jnp.ones((pixel_count,), dtype=jnp.bool_)
    mask_error = jnp.asarray(False)
    if masks is not None:
        masks = jnp.asarray(masks, dtype=jnp.bool_)
        if masks.ndim != 3 or masks.shape[1:] != (tile_height, tile_width):
            raise ValueError(
                "masks must have shape [n_images, tile_height, tile_width]"
            )
        if (
            gaussians.image_count is not None
            and masks.shape[0] != gaussians.image_count
        ):
            raise ValueError("masks image dimension does not match dense inputs")
        mask_images_valid = (decoded.image_ids >= 0) & (
            decoded.image_ids < masks.shape[0]
        )
        safe_images = jnp.clip(
            decoded.image_ids, 0, max(masks.shape[0] - 1, 0)
        )
        active_capacity = jnp.asarray(active_tiles).shape[0]
        if active_capacity:
            safe_tiles = jnp.maximum(
                jnp.asarray(active_tiles, dtype=jnp.int32)[
                    jnp.clip(
                        decoded.active_ranks,
                        0,
                        active_capacity - 1,
                    )
                ],
                0,
            )
        else:
            safe_tiles = jnp.zeros((pixel_count,), dtype=jnp.int32)
        tile_ids_in_image = safe_tiles % (tile_width * tile_height)
        tile_rows = tile_ids_in_image // tile_width
        tile_columns = tile_ids_in_image % tile_width
        tile_enabled = mask_images_valid & masks[
            safe_images, tile_rows, tile_columns
        ]
        mask_error = jnp.any(decoded.valid & ~mask_images_valid)

    if backgrounds is None:
        sorted_backgrounds = jnp.zeros(
            (pixel_count, channels), dtype=gaussians.colors.dtype
        )
        background_error = jnp.asarray(False)
    else:
        backgrounds = jnp.asarray(backgrounds)
        if backgrounds.ndim != 2 or backgrounds.shape[1] != channels:
            raise ValueError("backgrounds must have shape [n_images, channels]")
        if (
            gaussians.image_count is not None
            and backgrounds.shape[0] != gaussians.image_count
        ):
            raise ValueError(
                "backgrounds image dimension does not match dense inputs"
            )
        if pixel_count:
            sorted_image_ids = image_ids[
                jnp.clip(decoded.output_slots, 0, pixel_count - 1)
            ]
        else:
            sorted_image_ids = jnp.zeros((0,), dtype=jnp.int32)
        background_images_valid = (sorted_image_ids >= 0) & (
            sorted_image_ids < backgrounds.shape[0]
        )
        sorted_backgrounds = backgrounds[
            jnp.clip(
                sorted_image_ids, 0, max(backgrounds.shape[0] - 1, 0)
            )
        ]
        sorted_backgrounds = jnp.where(
            background_images_valid[:, None], sorted_backgrounds, 0.0
        )
        background_error = jnp.any(
            decoded.valid & ~background_images_valid
        )

    can_sample = (
        gaussians.slot_count > 0
        and intersections.flatten_ids.shape[0] > 0
        and pixel_count > 0
    )
    if can_sample:
        def render_pixel(pixel_index):
            selected_ids, weights, _ = _sparse_sample_weights(
                decoded.active_ranks[pixel_index],
                decoded.image_ids[pixel_index],
                decoded.rows[pixel_index],
                decoded.columns[pixel_index],
                decoded.valid[pixel_index],
                tile_enabled[pixel_index],
                gaussians,
                intersections,
                absgrad_probe=flat_absgrad_probe,
                alpha_threshold=alpha_threshold,
                transmittance_threshold=transmittance_threshold,
            )
            render = jnp.einsum(
                "k,kc->c",
                weights,
                gaussians.colors[selected_ids],
                precision=jax.lax.Precision.HIGHEST,
            )
            alpha = jnp.sum(weights)
            return render, alpha

        sorted_colors, sorted_alphas = jax.lax.map(
            render_pixel, jnp.arange(pixel_count, dtype=jnp.int32)
        )
    else:
        sorted_colors = jnp.zeros(
            (pixel_count, channels), dtype=gaussians.colors.dtype
        )
        sorted_alphas = jnp.zeros(
            (pixel_count,), dtype=gaussians.opacities.dtype
        )
    sorted_colors = sorted_colors + sorted_backgrounds * (
        1.0 - sorted_alphas[:, None]
    )

    safe_output_slots = jnp.clip(
        decoded.output_slots, 0, max(pixel_count - 1, 0)
    )
    render_colors = jnp.zeros_like(sorted_colors).at[safe_output_slots].set(
        jnp.where(decoded.valid[:, None], sorted_colors, 0.0)
    )
    render_alphas = jnp.zeros_like(sorted_alphas).at[safe_output_slots].set(
        jnp.where(decoded.valid, sorted_alphas, 0.0)
    )[:, None]

    input_overflow = jnp.asarray(overflow, dtype=jnp.bool_)
    layout_error = (
        decoded.layout_error
        | dense_image_error
        | mask_error
        | background_error
    )
    combined_overflow = (
        input_overflow | layout_error | intersections.error
    )
    info = {
        "active_tile_count": decoded.active_tile_count,
        "decoded_pixel_count": decoded.decoded_count,
        "intersection_valid_count": intersections.valid_count,
        "layout_overflow": input_overflow | layout_error,
        "intersection_overflow": input_overflow | intersections.error,
        "overflow": combined_overflow,
    }
    if not return_info:
        def report_overflow(_):
            jax.debug.callback(
                _raise_sparse_rasterization_overflow, ordered=True
            )
            return jnp.asarray(0, dtype=jnp.int32)

        jax.lax.cond(
            combined_overflow,
            report_overflow,
            lambda _: jnp.asarray(0, dtype=jnp.int32),
            operand=None,
        )
    if return_info:
        return render_colors, render_alphas, info
    return render_colors, render_alphas


__all__ = [
    "PaddedSparseIntersections",
    "PaddedSparseTileLayout",
    "build_sparse_tile_layout",
    "isect_tiles_sparse",
    "rasterize_to_pixels_sparse",
]
