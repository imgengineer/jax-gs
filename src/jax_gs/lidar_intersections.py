"""Static-shape angular tile intersections for legacy LiDAR rendering."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .lidar import (
    ANGLE_TO_PIXEL_SCALING_FACTOR,
    RowOffsetStructuredSpinningLidarModelParametersExt,
    relative_sensor_angles,
)
from .low_level import (
    PaddedIntersections,
    _as_static_int,
    _bits_for_count,
    _encode_high_word,
    _saturating_cumsum,
)


@dataclass(frozen=True)
class LidarSampleTileIdReturn:
    idx: jax.Array
    idxdense: jax.Array


jax.tree_util.register_dataclass(
    LidarSampleTileIdReturn,
    data_fields=("idx", "idxdense"),
    meta_fields=(),
)


def _round_kind(round_fn) -> str:
    name = getattr(round_fn, "__name__", "")
    if round_fn is jnp.floor or name == "floor":
        return "floor"
    if round_fn is jnp.ceil or name == "ceil":
        return "ceil"
    raise ValueError("round_fn must be jax.numpy.floor or jax.numpy.ceil")


def lidar_sample_tileid(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    relative_pixels,
    round_fn,
) -> LidarSampleTileIdReturn:
    """Map relative scaled angles to coarse and dense angular tile bounds."""

    kind = _round_kind(round_fn)
    pixels = jnp.asarray(relative_pixels)
    if pixels.shape[-1] != 2:
        raise ValueError("relative_pixels must end in [azimuth, elevation]")
    horizontal_span = ANGLE_TO_PIXEL_SCALING_FACTOR * lidar.fov_horiz_rad.span
    vertical_span = ANGLE_TO_PIXEL_SCALING_FACTOR * lidar.fov_vert_rad.span
    if horizontal_span <= 0.0 or vertical_span <= 0.0:
        raise ValueError("LiDAR FOV spans must be positive for tiling")

    normalized_azimuth = pixels[..., 0] / horizontal_span
    normalized_elevation = pixels[..., 1] / vertical_span
    dense_azimuth = round_fn(
        normalized_azimuth * lidar.tiling.cdf_resolution_azimuth
    ).astype(jnp.int32)
    dense_elevation = round_fn(
        normalized_elevation * lidar.tiling.cdf_resolution_elevation
    ).astype(jnp.int32)
    dense_azimuth = jnp.clip(dense_azimuth, 0, lidar.tiling.cdf_resolution_azimuth)
    dense_elevation = jnp.clip(
        dense_elevation, 0, lidar.tiling.cdf_resolution_elevation
    )

    cdf = lidar.tiling.cdf_elevation
    if kind == "ceil":
        previous = cdf[jnp.maximum(dense_elevation - 1, 0)] + 1
        elevation_tile = jnp.where(
            dense_elevation >= 1,
            jnp.minimum(previous, lidar.tiling.n_bins_elevation),
            cdf[dense_elevation],
        )
    else:
        elevation_tile = cdf[dense_elevation]
    azimuth_tile = round_fn(normalized_azimuth * lidar.tiling.n_bins_azimuth).astype(
        jnp.int32
    )
    azimuth_tile = jnp.clip(azimuth_tile, 0, lidar.tiling.n_bins_azimuth)
    return LidarSampleTileIdReturn(
        idx=jnp.stack((azimuth_tile, elevation_tile), axis=-1).astype(jnp.int32),
        idxdense=jnp.stack((dense_azimuth, dense_elevation), axis=-1).astype(jnp.int32),
    )


def has_any_rays_in_tile(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    begin,
    end,
) -> jax.Array:
    """Query the integral dense-ray mask over half-open bounds ``[begin,end)``."""

    begin = jnp.asarray(begin, dtype=jnp.int32)
    end = jnp.asarray(end, dtype=jnp.int32)
    if begin.shape != end.shape or begin.shape[-1] != 2:
        raise ValueError("begin and end must have matching [..., 2] shapes")
    max_azimuth = lidar.tiling.cdf_resolution_azimuth
    max_elevation = lidar.tiling.cdf_resolution_elevation
    begin_azimuth = jnp.clip(begin[..., 0], 0, max_azimuth)
    begin_elevation = jnp.clip(begin[..., 1], 0, max_elevation)
    end_azimuth = jnp.clip(end[..., 0], 0, max_azimuth)
    end_elevation = jnp.clip(end[..., 1], 0, max_elevation)
    cdf = lidar.tiling.cdf_dense_ray_mask
    ray_count = (
        cdf[end_elevation, end_azimuth]
        - cdf[begin_elevation, end_azimuth]
        - cdf[end_elevation, begin_azimuth]
        + cdf[begin_elevation, begin_azimuth]
    )
    full_horizontal_cover = (begin_azimuth <= 0) & (end_azimuth >= max_azimuth)
    return (ray_count > 0) | full_horizontal_cover


def _validate_dense_inputs(means, radii, depths):
    if means.ndim < 2 or means.shape[-1] != 2:
        raise ValueError("means2d must have shape [..., N, 2]")
    image_shape = means.shape[:-2]
    gaussian_count = means.shape[-2]
    if radii.shape != image_shape + (gaussian_count, 2):
        raise ValueError("radii shape does not match means2d")
    if depths.shape != image_shape + (gaussian_count,):
        raise ValueError("depths shape does not match means2d")
    return image_shape, gaussian_count


def isect_tiles_lidar(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    means2d,
    radii,
    depths,
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: int | None = None,
    image_ids=None,
    gaussian_ids=None,
    *,
    max_intersections: int | None = None,
    active_mask=None,
) -> PaddedIntersections:
    """Map angular Gaussian bounds to LiDAR tiles using a padded buffer.

    ``means2d`` and ``radii`` are expressed in radians multiplied by
    :data:`ANGLE_TO_PIXEL_SCALING_FACTOR`.  Only the first ``valid_count``
    encoded intersections are meaningful; ``overflow`` requests a larger
    static ``max_intersections``.
    """

    if not isinstance(lidar, RowOffsetStructuredSpinningLidarModelParametersExt):
        raise TypeError("lidar must contain angle maps and tiling state")
    if packed and segmented:
        raise ValueError("segmented sort is not supported for packed inputs")
    means = jax.lax.stop_gradient(jnp.asarray(means2d))
    radii = jax.lax.stop_gradient(jnp.asarray(radii))
    depths = jax.lax.stop_gradient(jnp.asarray(depths))
    if not jnp.issubdtype(depths.dtype, jnp.floating):
        raise TypeError("depths must have a floating-point dtype")

    if packed:
        if means.ndim != 2 or means.shape[-1] != 2:
            raise ValueError("packed means2d must have shape [nnz, 2]")
        flat_count = means.shape[0]
        if radii.shape != (flat_count, 2) or depths.shape != (flat_count,):
            raise ValueError("packed radii/depths shapes do not match means2d")
        if n_images is None or image_ids is None or gaussian_ids is None:
            raise ValueError(
                "packed mode requires n_images, image_ids, and gaussian_ids"
            )
        image_count = _as_static_int("n_images", n_images, minimum=1)
        image_of = jnp.asarray(image_ids, dtype=jnp.int32)
        gaussian_ids = jnp.asarray(gaussian_ids, dtype=jnp.int32)
        if image_of.shape != (flat_count,) or gaussian_ids.shape != (flat_count,):
            raise ValueError("packed image_ids and gaussian_ids must have shape [nnz]")
        flat_means = means
        flat_radii = radii
        flat_depths = depths
        output_shape = (flat_count,)
    else:
        image_shape, gaussian_count = _validate_dense_inputs(means, radii, depths)
        image_count = math.prod(image_shape) if image_shape else 1
        if n_images is not None and operator.index(n_images) != image_count:
            raise ValueError("n_images does not match dense leading dimensions")
        flat_count = image_count * gaussian_count
        flat_means = means.reshape((flat_count, 2))
        flat_radii = radii.reshape((flat_count, 2))
        flat_depths = depths.reshape((flat_count,))
        image_of = jnp.repeat(jnp.arange(image_count, dtype=jnp.int32), gaussian_count)
        output_shape = image_shape + (gaussian_count,)

    if active_mask is None:
        flat_active = jnp.ones((flat_count,), dtype=jnp.bool_)
    else:
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
        if active.shape != output_shape:
            raise ValueError("active_mask must match the Gaussian input shape")
        flat_active = active.reshape((flat_count,))

    tile_width = lidar.tiling.n_bins_azimuth
    tile_height = lidar.tiling.n_bins_elevation
    tile_count = tile_width * tile_height
    if max_intersections is None:
        max_intersections = flat_count * tile_count
    capacity = _as_static_int("max_intersections", max_intersections)
    if capacity > 2**30 - 2:
        raise ValueError("max_intersections is too large for int32 indexing")
    tile_bits = _bits_for_count(tile_count)
    if _bits_for_count(image_count) + tile_bits > 32:
        raise ValueError("image and LiDAR tile ids do not fit in 32 bits")

    if flat_count == 0:
        return PaddedIntersections(
            jnp.zeros(output_shape, dtype=jnp.int32),
            jnp.full((capacity, 2), -1, dtype=jnp.int32),
            jnp.full((capacity,), -1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
        )

    finite = (
        jnp.all(jnp.isfinite(flat_means), axis=-1)
        & jnp.all(jnp.isfinite(flat_radii), axis=-1)
        & jnp.isfinite(flat_depths)
    )
    nonzero_extent = jnp.all(flat_radii > 0, axis=-1)
    source_valid = (
        flat_active
        & finite
        & nonzero_extent
        & (image_of >= 0)
        & (image_of < image_count)
    )
    safe_means = jnp.where(finite[:, None], flat_means, 0.0)
    safe_radii = jnp.where(finite[:, None], flat_radii, 0.0)
    relative_mean = relative_sensor_angles(
        lidar, safe_means, ANGLE_TO_PIXEL_SCALING_FACTOR
    )
    begin_pixels = relative_mean - safe_radii
    end_pixels = relative_mean + safe_radii
    begin_azimuth = begin_pixels[:, 0]
    begin_elevation = begin_pixels[:, 1]
    end_azimuth = end_pixels[:, 0]
    end_elevation = end_pixels[:, 1]
    full_circle = 2.0 * math.pi * ANGLE_TO_PIXEL_SCALING_FACTOR
    horizontal_span = lidar.fov_horiz_rad.span * ANGLE_TO_PIXEL_SCALING_FACTOR
    vertical_span = lidar.fov_vert_rad.span * ANGLE_TO_PIXEL_SCALING_FACTOR

    full_cover = (begin_azimuth <= 0.0) & (end_azimuth >= horizontal_span)
    end_azimuth = jnp.minimum(end_azimuth, begin_azimuth + full_circle)
    overflow = end_azimuth > full_circle
    underflow = begin_azimuth < 0.0
    begin_a_azimuth = jnp.where(full_cover | underflow, 0.0, begin_azimuth)
    end_a_azimuth = jnp.where(
        full_cover,
        horizontal_span,
        jnp.where(overflow, full_circle, end_azimuth),
    )
    begin_b_azimuth = jnp.where(
        underflow & ~full_cover, begin_azimuth + full_circle, 0.0
    )
    end_b_azimuth = jnp.where(
        overflow & ~full_cover,
        end_azimuth - full_circle,
        jnp.where(underflow & ~full_cover, full_circle, 0.0),
    )

    def sample(azimuth, elevation, round_fn):
        pixels = jnp.stack(
            (
                jnp.clip(azimuth, 0.0, horizontal_span),
                jnp.clip(elevation, 0.0, vertical_span),
            ),
            axis=-1,
        )
        return lidar_sample_tileid(lidar, pixels, round_fn)

    begin_a = sample(begin_a_azimuth, begin_elevation, jnp.floor)
    end_a = sample(end_a_azimuth, end_elevation, jnp.ceil)
    begin_b = sample(begin_b_azimuth, begin_elevation, jnp.floor)
    end_b = sample(end_b_azimuth, end_elevation, jnp.ceil)
    has_rays_a = source_valid & has_any_rays_in_tile(
        lidar, begin_a.idxdense, end_a.idxdense
    )
    has_rays_b = source_valid & has_any_rays_in_tile(
        lidar, begin_b.idxdense, end_b.idxdense
    )
    has_rays = has_rays_a | has_rays_b

    elevation_begin = jnp.where(has_rays, begin_a.idx[:, 1], 0)
    elevation_end = jnp.where(has_rays, end_a.idx[:, 1], 0)
    begin_a_tile = jnp.where(has_rays_a, begin_a.idx[:, 0], 0)
    end_a_tile = jnp.where(has_rays_a, end_a.idx[:, 0], 0)
    begin_b_tile = jnp.where(has_rays_b, begin_b.idx[:, 0], 0)
    end_b_tile = jnp.where(has_rays_b, end_b.idx[:, 0], 0)
    periodic_azimuth = horizontal_span >= full_circle
    if periodic_azimuth:
        begin_a_tile = jnp.where(
            has_rays_b & underflow, begin_b_tile - tile_width, begin_a_tile
        )
        end_a_tile = jnp.where(
            has_rays_b & overflow, end_b_tile + tile_width, end_a_tile
        )
        begin_a_tile = jnp.maximum(begin_a_tile, end_a_tile - tile_width)
        end_a_tile = jnp.minimum(end_a_tile, begin_a_tile + tile_width)
        begin_b_tile = jnp.zeros_like(begin_b_tile)
        end_b_tile = jnp.zeros_like(end_b_tile)
    else:
        tile_overlap = (
            has_rays_a
            & has_rays_b
            & (begin_b_tile < end_a_tile)
            & (begin_a_tile < end_b_tile)
        )
        begin_a_tile = jnp.where(tile_overlap, 0, begin_a_tile)
        end_a_tile = jnp.where(tile_overlap, tile_width, end_a_tile)
        begin_b_tile = jnp.where(tile_overlap, 0, begin_b_tile)
        end_b_tile = jnp.where(tile_overlap, 0, end_b_tile)

    azimuth_count_a = (end_a_tile - begin_a_tile).astype(jnp.int32)
    azimuth_count_b = (end_b_tile - begin_b_tile).astype(jnp.int32)
    azimuth_count = azimuth_count_a + azimuth_count_b
    elevation_count = (elevation_end - elevation_begin).astype(jnp.int32)
    tiles_per_flat = jnp.maximum(elevation_count * azimuth_count, 0)
    tiles_per_gaussian = tiles_per_flat.reshape(output_shape)
    cumulative = _saturating_cumsum(tiles_per_flat, capacity + 1)
    total = cumulative[-1]
    valid_count = jnp.minimum(total, jnp.int32(capacity))
    result_overflow = total > capacity
    if capacity == 0:
        return PaddedIntersections(
            tiles_per_gaussian,
            jnp.full((0, 2), -1, dtype=jnp.int32),
            jnp.full((0,), -1, dtype=jnp.int32),
            valid_count,
            result_overflow,
        )

    ranks = jnp.arange(capacity, dtype=jnp.int32)
    source = jnp.searchsorted(cumulative, ranks, side="right")
    source = jnp.clip(source, 0, flat_count - 1)
    previous = jnp.where(source > 0, cumulative[jnp.maximum(source - 1, 0)], 0)
    local = ranks - previous
    selected_azimuth_count = jnp.maximum(azimuth_count[source], 1)
    azimuth_local = local % selected_azimuth_count
    elevation = elevation_begin[source] + local // selected_azimuth_count
    in_region_a = azimuth_local < azimuth_count_a[source]
    azimuth = jnp.where(
        in_region_a,
        begin_a_tile[source] + azimuth_local,
        begin_b_tile[source] + azimuth_local - azimuth_count_a[source],
    )
    if periodic_azimuth:
        azimuth = azimuth % tile_width
    tile_id = elevation * tile_width + azimuth
    selected_image = image_of[source]
    selected_depth = flat_depths[source].astype(jnp.float32)
    selected_valid = ranks < valid_count

    if sort:
        global_tile = selected_image * tile_count + tile_id
        order = jnp.lexsort(
            (
                source,
                selected_depth,
                global_tile,
                (~selected_valid).astype(jnp.int32),
            )
        )
        source = source[order]
        selected_image = selected_image[order]
        tile_id = tile_id[order]
        selected_depth = selected_depth[order]
        selected_valid = selected_valid[order]

    high_word = _encode_high_word(selected_image, tile_id, tile_bits)
    depth_word = jax.lax.bitcast_convert_type(selected_depth, jnp.int32)
    intersection_ids = jnp.stack((high_word, depth_word), axis=-1)
    intersection_ids = jnp.where(selected_valid[:, None], intersection_ids, -1)
    flatten_ids = jnp.where(selected_valid, source.astype(jnp.int32), -1)
    return PaddedIntersections(
        jax.lax.stop_gradient(tiles_per_gaussian),
        jax.lax.stop_gradient(intersection_ids),
        jax.lax.stop_gradient(flatten_ids),
        jax.lax.stop_gradient(valid_count),
        jax.lax.stop_gradient(result_overflow),
    )


__all__ = [
    "LidarSampleTileIdReturn",
    "has_any_rays_in_tile",
    "isect_tiles_lidar",
    "lidar_sample_tileid",
]
