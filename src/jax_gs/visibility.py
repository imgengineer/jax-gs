"""Per-pixel Gaussian visibility and contributor queries.

These functions share the sparse rasterizer's exact alpha/transmittance rules.
Upstream contributor lists use a runtime maximum length; JAX returns a padded
result with explicit per-pixel counts and overflow metadata instead.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .low_level import (
    DEFAULT_ALPHA_THRESHOLD,
    DEFAULT_TRANSMITTANCE_THRESHOLD,
)
from .sparse import (
    PaddedSparseTileLayout,
    _decode_sparse_pixels,
    _DecodedSparsePixels,
    _flatten_sparse_gaussians,
    _raise_sparse_rasterization_overflow,
    _sparse_sample_weights,
    _SparseGaussianData,
    _SparseIntersectionData,
    _static_int,
    _validate_sparse_intersections,
    build_sparse_tile_layout,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PaddedContributors:
    """Static contributor IDs and weights that unpack like gsplat's tuple.

    ``valid_counts[p]`` entries are meaningful for pixel ``p``.
    ``required_count`` is the maximum exact count over all pixels, and
    ``overflow`` reports a capacity or input-layout mismatch.
    """

    gaussian_ids: jax.Array
    weights: jax.Array
    valid_counts: jax.Array
    required_count: jax.Array
    overflow: jax.Array

    def __iter__(self) -> Iterator[jax.Array]:
        yield self.gaussian_ids
        yield self.weights

    def tree_flatten(self):
        return (
            (
                self.gaussian_ids,
                self.weights,
                self.valid_counts,
                self.required_count,
                self.overflow,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


class _SparseVisibility(NamedTuple):
    decoded: _DecodedSparsePixels
    gaussians: _SparseGaussianData
    intersections: _SparseIntersectionData
    flatten_ids: jax.Array
    local_ids: jax.Array
    weights: jax.Array
    accepted: jax.Array
    overflow: jax.Array


class _DenseSparseLayout(NamedTuple):
    layout: PaddedSparseTileLayout
    tile_offsets: jax.Array
    valid_count: jax.Array
    image_shape: tuple[int, ...]


def _validate_query_geometry(
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    alpha_threshold: float,
    transmittance_threshold: float,
) -> tuple[int, int, int, int, int]:
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
    if not math.isfinite(transmittance_threshold) or transmittance_threshold <= 0.0:
        raise ValueError("transmittance_threshold must be positive and finite")
    return image_width, image_height, tile_size, tile_width, tile_height


def _evaluate_sparse_visibility(
    means2d: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
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
    *,
    active_tile_count: jax.Array | int | None,
    valid_count: jax.Array | int | None,
    overflow: jax.Array | bool,
    alpha_threshold: float,
    transmittance_threshold: float,
) -> _SparseVisibility:
    (
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
    ) = _validate_query_geometry(
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        alpha_threshold,
        transmittance_threshold,
    )
    gaussians = _flatten_sparse_gaussians(
        means2d, conics, None, opacities, packed=means2d.ndim == 2
    )
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
    )
    intersections = _validate_sparse_intersections(
        tile_offsets,
        flatten_ids,
        jnp.asarray(active_tiles).shape[0],
        valid_count=valid_count,
    )
    pixel_count = decoded.output_slots.shape[0]
    intersection_capacity = intersections.flatten_ids.shape[0]
    dense_image_error = jnp.asarray(False)
    if gaussians.image_count is not None:
        dense_image_error = jnp.any(
            decoded.valid
            & ((decoded.image_ids < 0) | (decoded.image_ids >= gaussians.image_count))
        )

    if gaussians.slot_count > 0 and intersection_capacity > 0 and pixel_count > 0:

        def evaluate_pixel(pixel_index):
            return _sparse_sample_weights(
                decoded.active_ranks[pixel_index],
                decoded.image_ids[pixel_index],
                decoded.rows[pixel_index],
                decoded.columns[pixel_index],
                decoded.valid[pixel_index],
                jnp.asarray(True),
                gaussians,
                intersections,
                alpha_threshold=alpha_threshold,
                transmittance_threshold=transmittance_threshold,
            )

        selected_ids, weights, accepted = jax.lax.map(
            evaluate_pixel, jnp.arange(pixel_count, dtype=jnp.int32)
        )
    else:
        selected_ids = jnp.zeros((pixel_count, intersection_capacity), dtype=jnp.int32)
        weights = jnp.zeros(
            (pixel_count, intersection_capacity), dtype=gaussians.opacities.dtype
        )
        accepted = jnp.zeros((pixel_count, intersection_capacity), dtype=jnp.bool_)
    if gaussians.dense_gaussians_per_image is None:
        local_ids = selected_ids
    else:
        local_ids = selected_ids % gaussians.dense_gaussians_per_image
    combined_overflow = (
        jnp.asarray(overflow, dtype=jnp.bool_)
        | decoded.layout_error
        | intersections.error
        | dense_image_error
    )
    return _SparseVisibility(
        decoded,
        gaussians,
        intersections,
        selected_ids,
        local_ids.astype(jnp.int32),
        weights,
        accepted,
        combined_overflow,
    )


def _scatter_sorted_rows(
    values: jax.Array,
    decoded: _DecodedSparsePixels,
    fill_value: float,
) -> jax.Array:
    pixel_count = decoded.output_slots.shape[0]
    safe_slots = jnp.clip(decoded.output_slots, 0, max(pixel_count - 1, 0))
    valid_shape = (pixel_count,) + (1,) * (values.ndim - 1)
    valid = decoded.valid.reshape(valid_shape)
    initial = jnp.full(values.shape, fill_value, dtype=values.dtype)
    return initial.at[safe_slots].set(jnp.where(valid, values, fill_value))


def _report_query_overflow(overflow: jax.Array, return_info: bool) -> None:
    if return_info:
        return

    def report(_):
        jax.debug.callback(_raise_sparse_rasterization_overflow, ordered=True)
        return jnp.asarray(0, dtype=jnp.int32)

    jax.lax.cond(
        overflow,
        report,
        lambda _: jnp.asarray(0, dtype=jnp.int32),
        operand=None,
    )


def _visibility_info(visibility: _SparseVisibility) -> dict[str, jax.Array]:
    return {
        "active_tile_count": visibility.decoded.active_tile_count,
        "decoded_pixel_count": visibility.decoded.decoded_count,
        "intersection_valid_count": visibility.intersections.valid_count,
        "overflow": visibility.overflow,
    }


def rasterize_num_contributing_gaussians_sparse(
    means2d: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
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
    *,
    active_tile_count: jax.Array | int | None = None,
    valid_count: jax.Array | int | None = None,
    overflow: jax.Array | bool = False,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
    return_info: bool = False,
):
    """Count accepted contributors and accumulated alpha for ``P`` pixels."""

    visibility = _evaluate_sparse_visibility(
        means2d,
        conics,
        opacities,
        active_tiles,
        tile_offsets,
        flatten_ids,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        active_tile_count=active_tile_count,
        valid_count=valid_count,
        overflow=overflow,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    sorted_counts = jnp.sum(visibility.accepted, axis=-1, dtype=jnp.int32)
    sorted_alphas = jnp.sum(visibility.weights, axis=-1)
    counts = _scatter_sorted_rows(sorted_counts, visibility.decoded, 0)
    alphas = _scatter_sorted_rows(sorted_alphas, visibility.decoded, 0.0)
    counts = jax.lax.stop_gradient(counts)
    alphas = jax.lax.stop_gradient(alphas)
    _report_query_overflow(visibility.overflow, return_info)
    if return_info:
        return counts, alphas, _visibility_info(visibility)
    return counts, alphas


def rasterize_contributing_gaussian_ids_sparse(
    means2d: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    active_tiles: jax.Array,
    tile_offsets: jax.Array,
    flatten_ids: jax.Array,
    tile_pixel_mask: jax.Array,
    tile_pixel_cumsum: jax.Array,
    pixel_map: jax.Array,
    num_contributing_gaussians: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    *,
    active_tile_count: jax.Array | int | None = None,
    valid_count: jax.Array | int | None = None,
    overflow: jax.Array | bool = False,
    max_contributors: int | None = None,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
) -> PaddedContributors:
    """Return all accepted contributors in front-to-back order.

    The default capacity is a complete per-pixel upper bound.  A smaller
    ``max_contributors`` is allowed only with explicit ``overflow`` handling on
    the returned result.
    """

    visibility = _evaluate_sparse_visibility(
        means2d,
        conics,
        opacities,
        active_tiles,
        tile_offsets,
        flatten_ids,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        active_tile_count=active_tile_count,
        valid_count=valid_count,
        overflow=overflow,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    pixel_count, intersection_capacity = visibility.accepted.shape
    per_pixel_gaussian_bound = (
        visibility.gaussians.slot_count
        if visibility.gaussians.dense_gaussians_per_image is None
        else visibility.gaussians.dense_gaussians_per_image
    )
    complete_capacity = min(per_pixel_gaussian_bound, intersection_capacity)
    if max_contributors is None:
        contributor_capacity = complete_capacity
    else:
        contributor_capacity = _static_int(
            "max_contributors", max_contributors, minimum=0
        )
    supplied_counts = jnp.asarray(num_contributing_gaussians, dtype=jnp.int32)
    if supplied_counts.shape != (pixel_count,):
        raise ValueError("num_contributing_gaussians must have shape [P]")
    safe_slots = jnp.clip(visibility.decoded.output_slots, 0, max(pixel_count - 1, 0))
    sorted_supplied_counts = supplied_counts[safe_slots]
    sorted_counts = jnp.sum(visibility.accepted, axis=-1, dtype=jnp.int32)
    count_error = jnp.any(supplied_counts < 0) | jnp.any(
        visibility.decoded.valid & (sorted_supplied_counts != sorted_counts)
    )
    required_count = (
        jnp.max(sorted_counts) if pixel_count else jnp.asarray(0, jnp.int32)
    )
    valid_counts_sorted = jnp.minimum(sorted_counts, jnp.int32(contributor_capacity))

    if contributor_capacity and intersection_capacity:

        def select_pixel(pixel_index):
            positions = jnp.nonzero(
                visibility.accepted[pixel_index],
                size=contributor_capacity,
                fill_value=0,
            )[0]
            slot_valid = (
                jnp.arange(contributor_capacity, dtype=jnp.int32)
                < valid_counts_sorted[pixel_index]
            )
            ids = visibility.local_ids[pixel_index, positions]
            weights = visibility.weights[pixel_index, positions]
            return (
                jnp.where(slot_valid, ids, -1),
                jnp.where(slot_valid, weights, 0.0),
            )

        sorted_ids, sorted_weights = jax.lax.map(
            select_pixel, jnp.arange(pixel_count, dtype=jnp.int32)
        )
    else:
        sorted_ids = jnp.full((pixel_count, contributor_capacity), -1, dtype=jnp.int32)
        sorted_weights = jnp.zeros(
            (pixel_count, contributor_capacity),
            dtype=visibility.gaussians.opacities.dtype,
        )
    output_ids = _scatter_sorted_rows(sorted_ids, visibility.decoded, -1)
    output_weights = _scatter_sorted_rows(sorted_weights, visibility.decoded, 0.0)
    valid_counts = _scatter_sorted_rows(valid_counts_sorted, visibility.decoded, 0)
    result_overflow = (
        visibility.overflow | count_error | (required_count > contributor_capacity)
    )
    return PaddedContributors(
        jax.lax.stop_gradient(output_ids.astype(jnp.int32)),
        jax.lax.stop_gradient(output_weights),
        jax.lax.stop_gradient(valid_counts.astype(jnp.int32)),
        jax.lax.stop_gradient(required_count.astype(jnp.int32)),
        jax.lax.stop_gradient(result_overflow),
    )


def rasterize_top_contributing_gaussian_ids_sparse(
    means2d: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
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
    num_depth_samples: int,
    *,
    active_tile_count: jax.Array | int | None = None,
    valid_count: jax.Array | int | None = None,
    overflow: jax.Array | bool = False,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
    return_info: bool = False,
):
    """Return the strongest contributors, restored to front-to-back order."""

    num_depth_samples = _static_int("num_depth_samples", num_depth_samples, minimum=1)
    visibility = _evaluate_sparse_visibility(
        means2d,
        conics,
        opacities,
        active_tiles,
        tile_offsets,
        flatten_ids,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        active_tile_count=active_tile_count,
        valid_count=valid_count,
        overflow=overflow,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    pixel_count, intersection_capacity = visibility.weights.shape
    selection_capacity = max(intersection_capacity, num_depth_samples)
    pad_width = selection_capacity - intersection_capacity
    padded_weights = jnp.pad(visibility.weights, ((0, 0), (0, pad_width)))
    padded_ids = jnp.pad(visibility.local_ids, ((0, 0), (0, pad_width)))
    candidate_positions = jnp.arange(selection_capacity, dtype=jnp.int32)

    def select_pixel(pixel_index):
        weight_order = jnp.lexsort((candidate_positions, -padded_weights[pixel_index]))
        selected_positions = weight_order[:num_depth_samples]
        selected_weights = padded_weights[pixel_index, selected_positions]
        selected_ids = padded_ids[pixel_index, selected_positions]
        selected_valid = (selected_positions < intersection_capacity) & (
            selected_weights > 0.0
        )
        depth_order = jnp.lexsort(
            (
                selected_positions,
                (~selected_valid).astype(jnp.int32),
            )
        )
        selected_positions = selected_positions[depth_order]
        selected_weights = selected_weights[depth_order]
        selected_ids = selected_ids[depth_order]
        selected_valid = selected_valid[depth_order]
        return (
            jnp.where(selected_valid, selected_ids, -1),
            jnp.where(selected_valid, selected_weights, 0.0),
        )

    sorted_ids, sorted_weights = jax.lax.map(
        select_pixel, jnp.arange(pixel_count, dtype=jnp.int32)
    )
    output_ids = _scatter_sorted_rows(sorted_ids, visibility.decoded, -1).astype(
        jnp.int32
    )
    output_weights = _scatter_sorted_rows(sorted_weights, visibility.decoded, 0.0)
    output_ids = jax.lax.stop_gradient(output_ids)
    output_weights = jax.lax.stop_gradient(output_weights)
    _report_query_overflow(visibility.overflow, return_info)
    if return_info:
        counts = jnp.sum(visibility.accepted, axis=-1, dtype=jnp.int32)
        info = _visibility_info(visibility)
        info["valid_counts"] = _scatter_sorted_rows(
            jnp.minimum(counts, num_depth_samples), visibility.decoded, 0
        )
        return output_ids, output_weights, info
    return output_ids, output_weights


def _dense_layout_as_sparse(
    tile_offsets: jax.Array,
    flatten_ids: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    *,
    valid_count: jax.Array | int | None,
) -> _DenseSparseLayout:
    tile_offsets = jnp.asarray(tile_offsets, dtype=jnp.int32)
    flatten_ids = jnp.asarray(flatten_ids, dtype=jnp.int32)
    if tile_offsets.ndim < 2:
        raise ValueError("tile_offsets must have shape [..., TH, TW]")
    image_shape = tile_offsets.shape[:-2]
    tile_height, tile_width = tile_offsets.shape[-2:]
    image_count = math.prod(image_shape)
    if image_count < 1:
        raise ValueError("tile_offsets must contain at least one image")
    if tile_width * tile_size < image_width:
        raise ValueError("tile_offsets do not cover image_width")
    if tile_height * tile_size < image_height:
        raise ValueError("tile_offsets do not cover image_height")
    intersection_capacity = flatten_ids.shape[0]
    normalized_count = (
        jnp.count_nonzero(flatten_ids >= 0).astype(jnp.int32)
        if valid_count is None
        else jnp.clip(
            jnp.asarray(valid_count, dtype=jnp.int32),
            0,
            intersection_capacity,
        )
    )

    rows, columns = jnp.meshgrid(
        jnp.arange(image_height, dtype=jnp.int32),
        jnp.arange(image_width, dtype=jnp.int32),
        indexing="ij",
    )
    one_image_pixels = jnp.stack((rows.reshape(-1), columns.reshape(-1)), axis=-1)
    pixels = jnp.tile(one_image_pixels, (image_count, 1))
    image_ids = jnp.repeat(
        jnp.arange(image_count, dtype=jnp.int32), image_height * image_width
    )
    layout = build_sparse_tile_layout(
        pixels,
        image_ids,
        image_count,
        tile_size,
        tile_width,
        tile_height,
    )
    sparse_offsets = jnp.concatenate((tile_offsets.reshape(-1), normalized_count[None]))
    return _DenseSparseLayout(layout, sparse_offsets, normalized_count, image_shape)


def rasterize_num_contributing_gaussians(
    means2d: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    tile_offsets: jax.Array,
    flatten_ids: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    *,
    valid_count: jax.Array | int | None = None,
    overflow: jax.Array | bool = False,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
    return_info: bool = False,
):
    """Dense-image wrapper around the common contributor traversal."""

    dense = _dense_layout_as_sparse(
        tile_offsets,
        flatten_ids,
        image_width,
        image_height,
        tile_size,
        valid_count=valid_count,
    )
    result = rasterize_num_contributing_gaussians_sparse(
        means2d,
        conics,
        opacities,
        dense.layout.active_tiles,
        dense.tile_offsets,
        flatten_ids,
        dense.layout.tile_pixel_mask,
        dense.layout.tile_pixel_cumsum,
        dense.layout.pixel_map,
        image_width,
        image_height,
        tile_size,
        tile_offsets.shape[-1],
        tile_offsets.shape[-2],
        active_tile_count=dense.layout.valid_count,
        valid_count=dense.valid_count,
        overflow=overflow | dense.layout.overflow,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
        return_info=return_info,
    )
    output_shape = dense.image_shape + (image_height, image_width)
    if return_info:
        counts, alphas, info = result
        return counts.reshape(output_shape), alphas.reshape(output_shape), info
    counts, alphas = result
    return counts.reshape(output_shape), alphas.reshape(output_shape)


def rasterize_contributing_gaussian_ids(
    means2d: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    tile_offsets: jax.Array,
    flatten_ids: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    num_contributing_gaussians: jax.Array,
    *,
    valid_count: jax.Array | int | None = None,
    overflow: jax.Array | bool = False,
    max_contributors: int | None = None,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
) -> PaddedContributors:
    """Dense-image all-contributor query with a static sample capacity."""

    dense = _dense_layout_as_sparse(
        tile_offsets,
        flatten_ids,
        image_width,
        image_height,
        tile_size,
        valid_count=valid_count,
    )
    supplied_counts = jnp.asarray(num_contributing_gaussians, dtype=jnp.int32)
    expected_shape = dense.image_shape + (image_height, image_width)
    if supplied_counts.shape != expected_shape:
        raise ValueError("num_contributing_gaussians shape must match the dense image")
    result = rasterize_contributing_gaussian_ids_sparse(
        means2d,
        conics,
        opacities,
        dense.layout.active_tiles,
        dense.tile_offsets,
        flatten_ids,
        dense.layout.tile_pixel_mask,
        dense.layout.tile_pixel_cumsum,
        dense.layout.pixel_map,
        supplied_counts.reshape(-1),
        image_width,
        image_height,
        tile_size,
        tile_offsets.shape[-1],
        tile_offsets.shape[-2],
        active_tile_count=dense.layout.valid_count,
        valid_count=dense.valid_count,
        overflow=overflow | dense.layout.overflow,
        max_contributors=max_contributors,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    capacity = result.gaussian_ids.shape[-1]
    output_shape = expected_shape + (capacity,)
    return PaddedContributors(
        result.gaussian_ids.reshape(output_shape),
        result.weights.reshape(output_shape),
        result.valid_counts.reshape(expected_shape),
        result.required_count,
        result.overflow,
    )


def rasterize_top_contributing_gaussian_ids(
    means2d: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    tile_offsets: jax.Array,
    flatten_ids: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    num_depth_samples: int,
    *,
    valid_count: jax.Array | int | None = None,
    overflow: jax.Array | bool = False,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
    return_info: bool = False,
):
    """Dense-image top-contributor query."""

    dense = _dense_layout_as_sparse(
        tile_offsets,
        flatten_ids,
        image_width,
        image_height,
        tile_size,
        valid_count=valid_count,
    )
    result = rasterize_top_contributing_gaussian_ids_sparse(
        means2d,
        conics,
        opacities,
        dense.layout.active_tiles,
        dense.tile_offsets,
        flatten_ids,
        dense.layout.tile_pixel_mask,
        dense.layout.tile_pixel_cumsum,
        dense.layout.pixel_map,
        image_width,
        image_height,
        tile_size,
        tile_offsets.shape[-1],
        tile_offsets.shape[-2],
        num_depth_samples,
        active_tile_count=dense.layout.valid_count,
        valid_count=dense.valid_count,
        overflow=overflow | dense.layout.overflow,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
        return_info=return_info,
    )
    output_shape = dense.image_shape + (
        image_height,
        image_width,
        num_depth_samples,
    )
    if return_info:
        ids, weights, info = result
        info["valid_counts"] = info["valid_counts"].reshape(output_shape[:-1])
        return ids.reshape(output_shape), weights.reshape(output_shape), info
    ids, weights = result
    return ids.reshape(output_shape), weights.reshape(output_shape)


__all__ = [
    "PaddedContributors",
    "rasterize_contributing_gaussian_ids",
    "rasterize_contributing_gaussian_ids_sparse",
    "rasterize_num_contributing_gaussians",
    "rasterize_num_contributing_gaussians_sparse",
    "rasterize_top_contributing_gaussian_ids",
    "rasterize_top_contributing_gaussian_ids_sparse",
]
