# pyright: reportAttributeAccessIssue=none, reportOptionalMemberAccess=none, reportMissingImports=none, reportArgumentType=none, reportCallIssue=none, reportPossiblyUnboundVariable=none, reportGeneralTypeIssues=none, reportOptionalSubscript=none

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import replace
from functools import partial
from typing import Any, NamedTuple

import jax  # pyright: ignore[reportMissingImports]
import jax.numpy as jnp  # pyright: ignore[reportMissingImports]

from ._pallas import rasterize_to_pixels_pallas
from .cameras import fully_fused_projection
from .config import RasterizationConfig
from .external_distortion import (
    BivariateWindshieldModelParameters,
    validate_external_distortion,
)
from .intersections import intersect_tiles
from .lidar import RowOffsetStructuredSpinningLidarModelParametersExt
from .lidar_intersections import isect_tiles_lidar
from .low_level import (
    _bits_for_count,
    _broadcast_means_with_absgrad_probe,
    _encode_high_word,
    isect_offset_encode,
    rasterize_to_pixels,
)
from .rendering_types import (
    CameraModel,
    RasterizeMode,
    RendererConfig,
    RenderMode,
    render_mode_has_color,
    render_mode_has_depth_channel,
    render_mode_has_expected_depth,
    render_mode_has_hit_distance,
    resolve_renderer_config,
    resolve_tile_size,
)
from .spherical_harmonics import spherical_harmonics
from .three_dgut import (
    FThetaCameraDistortionParameters,
    RollingShutterType,
    UnscentedTransformParameters,
    _rasterize_eval3d_camera,
    _rasterize_eval3d_lidar,
    fully_fused_projection_with_ut,
)

ColorInput = jax.Array | tuple[jax.Array, jax.Array]


def _has_color(render_mode: RenderMode) -> bool:
    return render_mode_has_color(render_mode)


def _has_depth(render_mode: RenderMode) -> bool:
    return render_mode_has_depth_channel(render_mode)


def _camera_centers(viewmats: jax.Array) -> jax.Array:
    rotations = viewmats[..., :3, :3]
    translations = viewmats[..., :3, 3]
    return -jnp.einsum("...ji,...j->...i", rotations, translations)


def _normalize_color_input(
    colors: ColorInput | None,
    *,
    gaussian_count: int,
    camera_count: int,
    sh_degree: int | jax.Array | None,
) -> ColorInput | None:
    if colors is None:
        return None
    if isinstance(colors, tuple):
        if len(colors) != 2:
            raise ValueError("split SH colors must be a (sh0, sh_rest) pair")
        if sh_degree is None:
            raise ValueError("split SH colors require sh_degree")
        sh0, sh_rest = (jnp.asarray(value) for value in colors)
        if sh0.ndim != 3 or sh0.shape[0] != gaussian_count or sh0.shape[1] != 1:
            raise ValueError("sh0 must have shape [N, 1, channels]")
        if (
            sh_rest.ndim != 3
            or sh_rest.shape[0] != gaussian_count
            or sh_rest.shape[-1] != sh0.shape[-1]
        ):
            raise ValueError("sh_rest must have shape [N, K-1, channels]")
        return sh0, sh_rest

    colors = jnp.asarray(colors)
    if sh_degree is not None:
        if colors.ndim == 3 and colors.shape[0] == gaussian_count:
            return colors
        if (
            colors.ndim == 4
            and colors.shape[0] == camera_count
            and colors.shape[1] == gaussian_count
        ):
            return colors
        raise ValueError(
            "SH colors must have shape [N, K, channels] or [C, N, K, channels]"
        )
    if colors.ndim == 2 and colors.shape[0] == gaussian_count:
        return colors
    if (
        colors.ndim == 3
        and colors.shape[0] == camera_count
        and colors.shape[1] == gaussian_count
    ):
        return colors
    raise ValueError("colors must have shape [N, D] or [C, N, D]")


def _color_channels(colors: ColorInput | None) -> int:
    if colors is None:
        return 0
    if isinstance(colors, tuple):
        return int(colors[0].shape[-1])
    return int(colors.shape[-1])


def _join_sh_coefficients(colors: ColorInput) -> jax.Array:
    if isinstance(colors, tuple):
        return jnp.concatenate(colors, axis=-2)
    return colors


def _prepare_colors(
    means: jax.Array,
    colors: jax.Array | None,
    viewmats: jax.Array,
    sh_degree: int | None,
    camera_count: int,
) -> jax.Array | None:
    if colors is None:
        return None
    colors = jnp.asarray(colors)
    if sh_degree is not None:
        if colors.ndim not in {3, 4}:
            raise ValueError(
                "SH colors must have shape [N, K, channels] or [C, N, K, channels]"
            )
        if colors.ndim == 4 and colors.shape[0] != camera_count:
            raise ValueError("per-camera SH colors must match the camera count")
        centers = _camera_centers(viewmats)
        directions = means[None, :, :] - centers[:, None, :]
        if colors.ndim == 3:
            evaluate = lambda direction: spherical_harmonics(
                sh_degree, direction, colors
            )
            camera_inputs = directions
        else:
            evaluate = lambda values: spherical_harmonics(
                sh_degree, values[0], values[1]
            )
            camera_inputs = (directions, colors)
        if camera_count == 1:
            rendered = (
                evaluate(camera_inputs[0])[None, ...]
                if colors.ndim == 3
                else evaluate((directions[0], colors[0]))[None, ...]
            )
        else:
            rendered = jax.lax.map(evaluate, camera_inputs, batch_size=1)
        return jnp.maximum(rendered + 0.5, 0.0)
    if colors.ndim == 2:
        return jnp.broadcast_to(colors[None, ...], (camera_count,) + colors.shape)
    if colors.ndim == 3 and colors.shape[0] == camera_count:
        return colors
    raise ValueError("colors must have shape [N, D] or [C, N, D]")


def _render_camera_tiles(
    means2d: jax.Array,
    radii: jax.Array,
    depths: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    colors: jax.Array | None,
    valid: jax.Array,
    *,
    width: int,
    height: int,
    config: RasterizationConfig,
    background: jax.Array | None,
    render_mode: RenderMode,
    absgrad_probe: jax.Array | None = None,
    allow_accutile: bool = True,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    del allow_accutile
    tile_size = config.tile_size
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size
    tile_count = tile_width * tile_height
    candidate_limit = means2d.shape[0]
    channel_count = 0 if colors is None else colors.shape[-1]
    if _has_color(render_mode) and colors is None:
        raise ValueError(f"render_mode={render_mode!r} requires colors")

    local_y, local_x = jnp.meshgrid(
        jnp.arange(tile_size, dtype=means2d.dtype) + 0.5,
        jnp.arange(tile_size, dtype=means2d.dtype) + 0.5,
        indexing="ij",
    )
    local_x = local_x.reshape(-1)
    local_y = local_y.reshape(-1)
    pixel_count = tile_size * tile_size

    def render_tile(tile_id: jax.Array):
        tile_x = tile_id % tile_width
        tile_y = tile_id // tile_width
        x0 = tile_x * tile_size
        y0 = tile_y * tile_size
        x1 = x0 + tile_size
        y1 = y0 + tile_size

        overlaps = (
            valid
            & (means2d[..., 0] + radii[..., 0] > x0)
            & (means2d[..., 0] - radii[..., 0] < x1)
            & (means2d[..., 1] + radii[..., 1] > y0)
            & (means2d[..., 1] - radii[..., 1] < y1)
            & (opacities > 0.0)
        )
        candidate_count = jnp.count_nonzero(overlaps)
        scores = jnp.where(overlaps, -depths, -jnp.inf)
        selected_scores, selected_ids = jax.lax.top_k(scores, candidate_limit)
        selected_valid = jnp.isfinite(selected_scores)

        selected_means = means2d[selected_ids]
        selected_conics = conics[selected_ids]
        selected_opacities = opacities[selected_ids]
        selected_depths = depths[selected_ids]

        pixel_x = x0.astype(means2d.dtype) + local_x
        pixel_y = y0.astype(means2d.dtype) + local_y
        pixel_valid = (pixel_x < width) & (pixel_y < height)
        if absgrad_probe is None:
            dx = pixel_x[None, :] - selected_means[:, 0, None]
            dy = pixel_y[None, :] - selected_means[:, 1, None]
        else:
            means_per_pixel = _broadcast_means_with_absgrad_probe(
                selected_means,
                absgrad_probe[selected_ids],
                jnp.stack((pixel_x, pixel_y), axis=-1),
            )
            dx = pixel_x[None, :] - means_per_pixel[..., 0]
            dy = pixel_y[None, :] - means_per_pixel[..., 1]
        sigma = (
            0.5
            * (
                selected_conics[:, 0, None] * dx * dx
                + selected_conics[:, 2, None] * dy * dy
            )
            + selected_conics[:, 1, None] * dx * dy
        )
        alpha = selected_opacities[:, None] * jnp.exp(-jnp.maximum(sigma, 0.0))
        alpha = jnp.minimum(alpha, 0.999)
        alpha_valid = (
            selected_valid[:, None]
            & pixel_valid[None, :]
            & (sigma >= 0.0)
            & (alpha >= config.alpha_clip)
        )
        alpha = jnp.where(alpha_valid, alpha, 0.0)
        one_minus_alpha = 1.0 - alpha
        transmittance = jnp.concatenate(
            [
                jnp.ones((1, pixel_count), dtype=alpha.dtype),
                jnp.cumprod(one_minus_alpha, axis=0)[:-1],
            ],
            axis=0,
        )
        accepted = transmittance * (1.0 - alpha) > config.transmittance_eps
        weights = jnp.where(accepted, alpha * transmittance, 0.0)
        accumulated_alpha = jnp.sum(weights, axis=0)

        if colors is None:
            rendered_color = jnp.zeros((pixel_count, 0), dtype=means2d.dtype)
        else:
            rendered_color = jnp.einsum(
                "kp,kd->pd",
                weights,
                colors[selected_ids],
                precision=jax.lax.Precision.HIGHEST,
            )
        accumulated_depth = jnp.sum(weights * selected_depths[:, None], axis=0)
        expected_depth = accumulated_depth / jnp.maximum(
            accumulated_alpha, config.transmittance_eps
        )

        if render_mode == "RGB":
            rendered = rendered_color
        elif render_mode == "D":
            rendered = accumulated_depth[:, None]
        elif render_mode == "ED":
            rendered = expected_depth[:, None]
        elif render_mode == "RGB+D":
            rendered = jnp.concatenate(
                [rendered_color, accumulated_depth[:, None]], axis=-1
            )
        elif render_mode == "RGB+ED":
            rendered = jnp.concatenate(
                [rendered_color, expected_depth[:, None]], axis=-1
            )
        else:
            raise ValueError(f"unsupported render mode: {render_mode}")

        rendered = jnp.where(pixel_valid[:, None], rendered, 0.0)
        return (
            rendered.reshape(tile_size, tile_size, -1),
            accumulated_alpha.reshape(tile_size, tile_size, 1),
            candidate_count,
            jnp.asarray(False),
            candidate_count > config.max_gaussians_per_tile,
            selected_ids,
            selected_valid,
        )

    tile_ids = jnp.arange(tile_count, dtype=jnp.int32)
    (
        rendered_tiles,
        alpha_tiles,
        candidate_counts,
        overflows,
        candidate_limit_exceeded,
        selected_gaussian_ids,
        selected_valid,
    ) = jax.lax.map(
        # Reverse mode otherwise keeps every tile's [candidate, pixel]
        # compositing intermediates alive at once.
        jax.checkpoint(render_tile),
        tile_ids,
        batch_size=config.tile_batch_size,
    )

    intersection_capacity = tile_count * candidate_limit
    flat_selected_valid = selected_valid.reshape(-1)
    selected_positions = jnp.nonzero(
        flat_selected_valid,
        size=intersection_capacity,
        fill_value=0,
    )[0]
    intersection_count = jnp.sum(flat_selected_valid, dtype=jnp.int32)
    retained_counts = jnp.sum(selected_valid, axis=-1, dtype=jnp.int32)
    output_valid = (
        jnp.arange(intersection_capacity, dtype=jnp.int32) < intersection_count
    )
    intersection_gaussian_ids = selected_gaussian_ids.reshape(-1)[selected_positions]
    intersection_gaussian_ids = jnp.where(
        output_valid, intersection_gaussian_ids, -1
    ).astype(jnp.int32)
    source_tile_ids = jnp.repeat(tile_ids, candidate_limit)
    intersection_tile_ids = source_tile_ids[selected_positions]
    intersection_tile_ids = jnp.where(output_valid, intersection_tile_ids, -1).astype(
        jnp.int32
    )
    intersection_offsets = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(retained_counts[:-1], dtype=jnp.int32),
        )
    ).reshape(tile_height, tile_width)

    output_channels = rendered_tiles.shape[-1]
    rendered = (
        rendered_tiles.reshape(
            tile_height, tile_width, tile_size, tile_size, output_channels
        )
        .transpose(0, 2, 1, 3, 4)
        .reshape(tile_height * tile_size, tile_width * tile_size, output_channels)
    )[:height, :width]
    alphas = (
        alpha_tiles.reshape(tile_height, tile_width, tile_size, tile_size, 1)
        .transpose(0, 2, 1, 3, 4)
        .reshape(tile_height * tile_size, tile_width * tile_size, 1)
    )[:height, :width]

    if background is not None and _has_color(render_mode):
        color_channels = channel_count
        rendered = rendered.at[..., :color_channels].add(
            background[None, None, :color_channels] * (1.0 - alphas)
        )

    return (
        rendered,
        alphas,
        {
            "candidate_counts": candidate_counts.reshape(tile_height, tile_width),
            "tile_overflow": overflows.reshape(tile_height, tile_width),
            "candidate_limit_exceeded": candidate_limit_exceeded.reshape(
                tile_height, tile_width
            ),
            "intersection_count": intersection_count,
            "intersection_required_count": intersection_count,
            "intersection_overflow": jnp.asarray(False),
            "intersection_capacity": jnp.asarray(
                min(tile_count * means2d.shape[0], 2**31 - 1), dtype=jnp.int32
            ),
            "intersection_gaussian_ids": intersection_gaussian_ids,
            "intersection_tile_ids": intersection_tile_ids,
            "intersection_offsets": intersection_offsets,
        },
    )


def _automatic_intersection_capacity(
    gaussian_count: int,
    tile_count: int,
    config: RasterizationConfig,
) -> int:
    if config.max_intersections is not None:
        return config.max_intersections
    # Every Gaussian can conservatively intersect every tile. Until a dynamic
    # exact-size allocation is available, this is the only static upper bound
    # that cannot truncate a high-level render.
    return gaussian_count * tile_count


def _assemble_dense_intersection_metadata(
    depths: jax.Array,
    gaussian_ids: jax.Array,
    tile_ids: jax.Array,
    local_offsets: jax.Array,
    valid_counts: jax.Array,
    *,
    tile_width: int,
    tile_height: int,
) -> dict[str, jax.Array]:
    """Combine the per-camera intersections used by the dense renderer."""

    camera_count, camera_capacity = gaussian_ids.shape
    gaussian_count = depths.shape[-1]
    total_capacity = camera_count * camera_capacity
    positions = jnp.arange(camera_capacity, dtype=jnp.int32)
    slot_valid = (
        (positions[None, :] < valid_counts[:, None])
        & (gaussian_ids >= 0)
        & (tile_ids >= 0)
    )
    global_valid_count = jnp.sum(slot_valid, dtype=jnp.int32)
    count_bases = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(valid_counts[:-1], dtype=jnp.int32),
        )
    )
    isect_offsets = local_offsets + count_bases[:, None, None]

    if total_capacity == 0 or gaussian_count == 0:
        return {
            "tiles_per_gauss": jnp.zeros(
                (camera_count, gaussian_count), dtype=jnp.int32
            ),
            "isect_ids": jnp.full((total_capacity, 2), -1, dtype=jnp.int32),
            "flatten_ids": jnp.full((total_capacity,), -1, dtype=jnp.int32),
            "isect_offsets": isect_offsets.astype(jnp.int32),
            "isect_valid_count": global_valid_count,
        }

    flat_valid = slot_valid.reshape(-1)
    selected = jnp.nonzero(flat_valid, size=total_capacity, fill_value=0)[0]
    output_valid = jnp.arange(total_capacity, dtype=jnp.int32) < global_valid_count
    source_camera_ids = jnp.repeat(
        jnp.arange(camera_count, dtype=jnp.int32), camera_capacity
    )
    selected_camera_ids = source_camera_ids[selected]
    selected_gaussian_ids = gaussian_ids.reshape(-1)[selected]
    selected_tile_ids = tile_ids.reshape(-1)[selected]
    safe_gaussian_ids = jnp.clip(selected_gaussian_ids, 0, gaussian_count - 1)

    tile_bits = _bits_for_count(tile_width * tile_height)
    high_words = _encode_high_word(selected_camera_ids, selected_tile_ids, tile_bits)
    selected_depths = depths[selected_camera_ids, safe_gaussian_ids].astype(jnp.float32)
    depth_words = jax.lax.bitcast_convert_type(selected_depths, jnp.int32)
    isect_ids = jnp.stack((high_words, depth_words), axis=-1)
    isect_ids = jnp.where(output_valid[:, None], isect_ids, -1)

    flatten_ids = selected_camera_ids * gaussian_count + safe_gaussian_ids
    flatten_ids = jnp.where(output_valid, flatten_ids, -1).astype(jnp.int32)
    safe_flatten_ids = jnp.clip(flatten_ids, 0, camera_count * gaussian_count - 1)
    tiles_per_gauss = (
        jnp.zeros((camera_count * gaussian_count,), dtype=jnp.int32)
        .at[safe_flatten_ids]
        .add(output_valid.astype(jnp.int32))
    )

    return {
        "tiles_per_gauss": tiles_per_gauss.reshape(camera_count, gaussian_count),
        "isect_ids": isect_ids.astype(jnp.int32),
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets.astype(jnp.int32),
        "isect_valid_count": global_valid_count,
    }


def _pack_dense_metadata(
    radii: jax.Array,
    means2d: jax.Array,
    depths: jax.Array,
    conics: jax.Array,
    compensations: jax.Array | None,
    opacities: jax.Array,
    valid: jax.Array,
    intersections: dict[str, jax.Array],
) -> dict[str, Any]:
    """Create one global static packed view of dense projection metadata.

    Valid rows form a stable batch-camera-Gaussian prefix. Remaining static
    capacity is zero-filled and is excluded from both counters and CSR pointers.
    """

    batch_shape = valid.shape[:-2]
    batch_count = math.prod(batch_shape) if batch_shape else 1
    camera_count, gaussian_count = valid.shape[-2:]
    image_count = batch_count * camera_count
    capacity = image_count * gaussian_count
    flat_valid = valid.reshape(-1)
    selected = jnp.nonzero(flat_valid, size=capacity, fill_value=0)[0]
    valid_count = jnp.count_nonzero(flat_valid).astype(jnp.int32)
    output_valid = jnp.arange(capacity, dtype=jnp.int32) < valid_count

    def gather(values: jax.Array) -> jax.Array:
        trailing_shape = values.shape[len(batch_shape) + 2 :]
        flat_values = values.reshape((capacity,) + trailing_shape)
        gathered = flat_values[selected]
        mask = output_valid.reshape((capacity,) + (1,) * len(trailing_shape))
        return jnp.where(mask, gathered, jnp.zeros_like(gathered))

    if capacity == 0:
        batch_ids = jnp.empty((0,), dtype=jnp.int32)
        camera_ids = jnp.empty((0,), dtype=jnp.int32)
        gaussian_ids = jnp.empty((0,), dtype=jnp.int32)
    else:
        dense_slots = selected.astype(jnp.int32)
        batch_ids = dense_slots // (camera_count * gaussian_count)
        within_batch = dense_slots % (camera_count * gaussian_count)
        camera_ids = within_batch // gaussian_count
        gaussian_ids = within_batch % gaussian_count
        batch_ids = jnp.where(output_valid, batch_ids, -1)
        camera_ids = jnp.where(output_valid, camera_ids, -1).astype(jnp.int32)
        gaussian_ids = jnp.where(output_valid, gaussian_ids, -1).astype(jnp.int32)

        packed_slots = jnp.arange(capacity, dtype=jnp.int32)
        scatter_destinations = jnp.where(output_valid, dense_slots, jnp.int32(capacity))
        dense_to_packed = (
            jnp.full((capacity + 1,), -1, dtype=jnp.int32)
            .at[scatter_destinations]
            .set(jnp.where(output_valid, packed_slots, -1))[:capacity]
        )

    dense_flatten_ids = intersections["flatten_ids"].reshape(batch_count, -1)
    per_batch_isect_capacity = dense_flatten_ids.shape[1]
    global_isect_capacity = batch_count * per_batch_isect_capacity
    dense_isect_ids = intersections["isect_ids"].reshape(
        batch_count, per_batch_isect_capacity, 2
    )
    reported_isect_counts = intersections["isect_valid_count"].reshape(batch_count)
    isect_positions = jnp.arange(per_batch_isect_capacity, dtype=jnp.int32)
    dense_isect_valid = (
        (isect_positions[None, :] < reported_isect_counts[:, None])
        & (dense_flatten_ids >= 0)
        & (dense_isect_ids[..., 0] != -1)
    )
    per_batch_isect_counts = jnp.sum(dense_isect_valid, axis=1, dtype=jnp.int32)
    isect_valid_count = jnp.sum(per_batch_isect_counts, dtype=jnp.int32)
    flat_isect_valid = dense_isect_valid.reshape(-1)
    selected_isects = jnp.nonzero(
        flat_isect_valid, size=global_isect_capacity, fill_value=0
    )[0]
    output_isect_valid = (
        jnp.arange(global_isect_capacity, dtype=jnp.int32) < isect_valid_count
    )
    source_batch_ids = jnp.repeat(
        jnp.arange(batch_count, dtype=jnp.int32),
        per_batch_isect_capacity,
    )[selected_isects]
    selected_local_dense_ids = dense_flatten_ids.reshape(-1)[selected_isects]
    if gaussian_count == 0:
        safe_local_dense_ids = jnp.zeros_like(selected_local_dense_ids)
    else:
        safe_local_dense_ids = jnp.clip(
            selected_local_dense_ids,
            0,
            camera_count * gaussian_count - 1,
        )
    selected_global_dense_ids = (
        source_batch_ids * camera_count * gaussian_count + safe_local_dense_ids
    )
    if capacity == 0:
        packed_flatten_ids = jnp.full((global_isect_capacity,), -1, dtype=jnp.int32)
    else:
        mapped_ids = dense_to_packed[selected_global_dense_ids]
        packed_flatten_ids = jnp.where(
            output_isect_valid & (mapped_ids >= 0), mapped_ids, -1
        ).astype(jnp.int32)

    selected_isect_ids = dense_isect_ids.reshape(-1, 2)[selected_isects]
    tile_height, tile_width = intersections["isect_offsets"].shape[-2:]
    tile_bits = _bits_for_count(tile_width * tile_height)
    high_words = jax.lax.bitcast_convert_type(selected_isect_ids[:, 0], jnp.uint32)
    tile_ids = (high_words & jnp.uint32((1 << tile_bits) - 1)).astype(jnp.int32)
    local_camera_ids = (
        jnp.zeros_like(safe_local_dense_ids)
        if gaussian_count == 0
        else safe_local_dense_ids // gaussian_count
    )
    global_image_ids = source_batch_ids * camera_count + local_camera_ids
    packed_isect_ids = jnp.stack(
        (
            _encode_high_word(global_image_ids, tile_ids, tile_bits),
            selected_isect_ids[:, 1],
        ),
        axis=-1,
    )
    packed_isect_ids = jnp.where(
        output_isect_valid[:, None], packed_isect_ids, -1
    ).astype(jnp.int32)

    isect_bases = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(per_batch_isect_counts[:-1], dtype=jnp.int32),
        )
    )
    global_isect_offsets = (
        intersections["isect_offsets"].reshape(
            batch_count, camera_count, tile_height, tile_width
        )
        + isect_bases[:, None, None, None]
    )
    global_isect_offsets = global_isect_offsets.reshape(
        batch_shape + (camera_count, tile_height, tile_width)
    )

    projection_counts = jnp.sum(
        valid.reshape(image_count, gaussian_count), axis=1, dtype=jnp.int32
    )
    indptr = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(projection_counts, dtype=jnp.int32),
        )
    )

    packed: dict[str, Any] = {
        "batch_ids": batch_ids,
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "indptr": indptr,
        "radii": gather(radii),
        "means2d": gather(means2d),
        "depths": gather(depths),
        "conics": gather(conics),
        "compensations": (None if compensations is None else gather(compensations)),
        "opacities": gather(opacities),
        "valid": output_valid,
        "tiles_per_gauss": gather(intersections["tiles_per_gauss"]),
        "isect_ids": packed_isect_ids,
        "flatten_ids": packed_flatten_ids,
        "isect_offsets": global_isect_offsets,
        "isect_valid_count": isect_valid_count,
        "projection_valid_count": valid_count,
        "projection_capacity": jnp.asarray(capacity, dtype=jnp.int32),
    }
    return packed


def _compact_visible_ids(
    valid: jax.Array,
    *,
    capacity: int,
    backend: str = "auto",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return a stable bounded prefix of visible Gaussian ids.

    Every valid Gaussian contributes to at least one tile intersection.  The
    renderer cannot produce an exact result when the visible count itself is
    larger than the configured intersection capacity, so compacting to that
    same bound does not truncate any non-overflowing render.  The overflow bit
    remains explicit for callers that need to raise before using gradients.
    """

    del backend
    valid = jnp.asarray(valid, dtype=jnp.bool_)
    capacity = min(int(capacity), int(valid.shape[0]))
    positions = jnp.cumsum(valid.astype(jnp.int32)) - 1
    visible_count = jnp.sum(valid, dtype=jnp.int32)
    destinations = jnp.where(valid, positions, jnp.int32(capacity))
    gaussian_ids = (
        jnp.zeros((capacity,), dtype=jnp.int32)
        .at[destinations]
        .set(jnp.arange(valid.shape[0], dtype=jnp.int32), mode="drop")
    )
    gaussian_ids = jax.lax.stop_gradient(gaussian_ids)
    retained_count = jnp.minimum(visible_count, jnp.int32(capacity))
    retained_valid = jnp.arange(capacity, dtype=jnp.int32) < retained_count
    return (
        gaussian_ids,
        retained_valid,
        visible_count,
        visible_count > capacity,
    )


def _render_camera_intersections(
    means2d: jax.Array,
    radii: jax.Array,
    depths: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    colors: jax.Array | None,
    valid: jax.Array,
    *,
    width: int,
    height: int,
    config: RasterizationConfig,
    background: jax.Array | None,
    render_mode: RenderMode,
    absgrad_probe: jax.Array | None = None,
    allow_accutile: bool = True,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Rasterize one camera from a single sorted tile-intersection list."""

    if _has_color(render_mode) and colors is None:
        raise ValueError(f"render_mode={render_mode!r} requires colors")
    tile_size = config.tile_size
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size
    tile_count = tile_width * tile_height
    intersection_capacity = _automatic_intersection_capacity(
        means2d.shape[0], tile_count, config
    )
    use_fused = (
        means2d.shape[0] > 0
        and intersection_capacity > 0
        and config.projection_backend == "cute"
        and config.intersection_backend == "cute"
        and config.compositor_backend == "cute"
        and config.rasterize_mode == "classic"
        and config.intersection_mode != "aabb"
        and tile_size == 16
        and allow_accutile
        and render_mode == "RGB"
        and colors is not None
        and absgrad_probe is None
    )
    if use_fused:
        assert colors is not None
        feature_background = (
            jnp.zeros((colors.shape[-1],), dtype=colors.dtype)
            if background is None
            else background
        )
        from ._cute_intersections import (
            rasterize_accutile_cute_raw_fused as fused_rasterize,
        )

        with jax.named_scope("intersection_compositor_cute_raw_fused"):
            rendered, alphas, fused_info = fused_rasterize(
                radii,
                depths,
                means2d,
                conics,
                colors,
                opacities,
                valid,
                capacity=intersection_capacity,
                tile_size=tile_size,
                tile_width=tile_width,
                tile_height=tile_height,
                image_width=width,
                image_height=height,
                background=feature_background,
                max_gaussians_per_tile=config.max_gaussians_per_tile,
                max_candidates_per_tile=config.max_candidates_per_tile,
                alpha_threshold=config.alpha_clip,
                transmittance_threshold=config.transmittance_eps,
            )
        flat_offsets = fused_info["offsets"].reshape(-1)
        ends = jnp.concatenate(
            (flat_offsets[1:], fused_info["valid_count"][None]), axis=0
        )
        candidate_counts = jnp.maximum(ends - flat_offsets, 0).reshape(
            tile_height, tile_width
        )
        return (
            rendered,
            alphas,
            {
                "candidate_counts": candidate_counts,
                "tile_overflow": fused_info["tile_overflow"],
                "candidate_limit_exceeded": (
                    candidate_counts > config.max_gaussians_per_tile
                ),
                "intersection_count": fused_info["valid_count"],
                "intersection_required_count": fused_info["required_count"],
                "intersection_overflow": fused_info["overflow"],
                "intersection_capacity": jnp.asarray(
                    intersection_capacity, dtype=jnp.int32
                ),
                "intersection_gaussian_ids": fused_info["gaussian_ids"],
                "intersection_tile_ids": fused_info["tile_ids"],
                "intersection_offsets": fused_info["offsets"],
            },
        )
    with jax.named_scope("intersection_total"):
        intersections = intersect_tiles(
            means2d,
            radii,
            depths,
            valid,
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            max_intersections=intersection_capacity,
            backend=config.intersection_backend,
            sort_backend=config.sort_backend,
            conics=conics if allow_accutile else None,
            opacities=opacities if allow_accutile else None,
            alpha_threshold=config.alpha_clip,
            mode=config.intersection_mode if allow_accutile else "aabb",
        )
    flat_offsets = intersections.offsets.reshape(-1)
    ends = jnp.concatenate((flat_offsets[1:], intersections.valid_count[None]), axis=0)
    candidate_counts = jnp.maximum(ends - flat_offsets, 0).reshape(
        tile_height, tile_width
    )

    if render_mode == "RGB":
        features = colors
        feature_background = background
    elif render_mode in {"D", "ED"}:
        features = depths[:, None]
        feature_background = jnp.zeros((1,), dtype=depths.dtype)
    elif render_mode in {"RGB+D", "RGB+ED"}:
        assert colors is not None
        features = jnp.concatenate((colors, depths[:, None]), axis=-1)
        color_background = (
            jnp.zeros((colors.shape[-1],), dtype=colors.dtype)
            if background is None
            else background[: colors.shape[-1]]
        )
        feature_background = jnp.concatenate(
            (color_background, jnp.zeros((1,), dtype=colors.dtype)), axis=0
        )
    else:
        raise ValueError(f"unsupported render mode: {render_mode}")
    assert features is not None
    if feature_background is None:
        feature_background = jnp.zeros((features.shape[-1],), dtype=features.dtype)

    if config.compositor_backend in {"pallas", "cute"}:
        if absgrad_probe is not None:
            raise NotImplementedError(
                "the experimental optimized compositors do not support AbsGrad"
            )
        with jax.named_scope(f"compositing_{config.compositor_backend}"):
            if config.compositor_backend == "pallas":
                rendered, alphas, low_info = rasterize_to_pixels_pallas(
                    means2d,
                    conics,
                    features,
                    opacities,
                    width,
                    height,
                    tile_size,
                    intersections.offsets,
                    intersections.gaussian_ids,
                    backgrounds=feature_background,
                    valid_count=intersections.valid_count,
                    overflow=intersections.overflow,
                    max_gaussians_per_tile=config.max_gaussians_per_tile,
                    max_candidates_per_tile=config.max_candidates_per_tile,
                    alpha_threshold=config.alpha_clip,
                    transmittance_threshold=config.transmittance_eps,
                )
            elif config.compositor_backend == "cute":
                from ._cute_compositor import rasterize_to_pixels_cute

                rendered, alphas, low_info = rasterize_to_pixels_cute(
                    means2d,
                    conics,
                    features,
                    opacities,
                    width,
                    height,
                    tile_size,
                    intersections.offsets,
                    intersections.gaussian_ids,
                    backgrounds=feature_background,
                    valid_count=intersections.valid_count,
                    overflow=intersections.overflow,
                    max_gaussians_per_tile=config.max_gaussians_per_tile,
                    max_candidates_per_tile=config.max_candidates_per_tile,
                    alpha_threshold=config.alpha_clip,
                    transmittance_threshold=config.transmittance_eps,
                )
        tile_overflow = low_info["tile_overflow"]
    else:
        with jax.named_scope("compositing_jax"):
            rendered, alphas, low_info = rasterize_to_pixels(  # type: ignore
                means2d[None, ...],
                conics[None, ...],
                features[None, ...],
                opacities[None, ...],
                width,
                height,
                tile_size,
                intersections.offsets[None, ...],
                intersections.gaussian_ids,
                backgrounds=feature_background[None, ...],
                valid_count=intersections.valid_count,
                overflow=intersections.overflow,
                max_gaussians_per_tile=config.max_gaussians_per_tile,
                max_candidates_per_tile=config.max_candidates_per_tile,
                tile_batch_size=config.tile_batch_size,
                alpha_threshold=config.alpha_clip,
                transmittance_threshold=config.transmittance_eps,
                return_info=True,
                absgrad=absgrad_probe is not None,
                _means2d_absgrad_probe=(
                    None if absgrad_probe is None else absgrad_probe[None, ...]
                ),
            )
        rendered = rendered[0]
        alphas = alphas[0]
        tile_overflow = low_info["tile_overflow"][0]
    if render_mode == "ED":
        rendered = rendered / jnp.maximum(alphas, config.transmittance_eps)
    elif render_mode == "RGB+ED":
        rendered = rendered.at[..., -1].set(
            rendered[..., -1] / jnp.maximum(alphas[..., 0], config.transmittance_eps)
        )

    return (
        rendered,
        alphas,
        {
            "candidate_counts": candidate_counts,
            "tile_overflow": tile_overflow,
            "candidate_limit_exceeded": (
                candidate_counts > config.max_gaussians_per_tile
            ),
            "intersection_count": intersections.valid_count,
            "intersection_required_count": intersections.required_count,
            "intersection_overflow": intersections.overflow,
            "intersection_capacity": jnp.asarray(
                intersection_capacity, dtype=jnp.int32
            ),
            "intersection_gaussian_ids": intersections.gaussian_ids,
            "intersection_tile_ids": intersections.tile_ids,
            "intersection_offsets": intersections.offsets,
        },
    )


class _CameraRenderContext(NamedTuple):
    """What every camera in one rasterization call shares.

    The scene being rendered, the camera-model coefficients, and the
    choices and sizes the call was configured with. One camera's own
    arrays travel separately as the mapped-over inputs.
    """

    absgrad_probe_enabled: Any
    calc_compensations: Any
    camera_model: Any
    color_channels: Any
    colors: Any
    config: Any
    external_distortion_coeffs: Any
    extra_channels: Any
    extra_signals: Any
    extra_signals_sh_degree: Any
    ftheta_coeffs: Any
    height: Any
    lidar_coeffs: Any
    means: Any
    opacities: Any
    quats: Any
    radial_coeffs: Any
    rays: Any
    render_mode: Any
    return_normals: Any
    rolling_shutter: Any
    scales: Any
    sh_degree: Any
    tangential_coeffs: Any
    thin_prism_coeffs: Any
    tile_count: Any
    tile_height: Any
    tile_width: Any
    use_ut: Any
    viewmats_rs: Any
    visible_capacity: Any
    width: Any
    with_eval3d: Any


def _render_one_camera(camera_inputs, *, ctx: _CameraRenderContext):
    """Render one camera against everything the sweep of cameras shares.

    ``camera_inputs`` is the per-camera slice ``jax.lax.map`` hands over;
    ``ctx`` is fixed for the whole sweep and arrives bound by
    :func:`functools.partial`, which traces the same as the closure this used
    to be.
    """
    (
        camera_index,
        means2d_c,
        absgrad_probe_c,
        radii_c,
        depths_c,
        conics_c,
        compensation_c,
        colors_c,
        extra_c,
        valid_c,
        viewmat_c,
        K_c,
        bg,
    ) = camera_inputs
    if ctx.visible_capacity == ctx.means.shape[0]:
        visible_ids = jnp.arange(ctx.means.shape[0], dtype=jnp.int32)
        retained_valid = valid_c
        # Packing is bypassed, so an exact visibility reduction would add
        # a standalone launch solely for diagnostics. ``-1`` ctx.means not
        # counted; overflow is impossible because the full bucket is kept.
        visible_count = jnp.asarray(-1, dtype=jnp.int32)
        visible_overflow = jnp.asarray(False)
    else:
        with jax.named_scope("visible_pack"):
            (
                visible_ids,
                retained_valid,
                visible_count,
                visible_overflow,
            ) = _compact_visible_ids(
                valid_c,
                capacity=ctx.visible_capacity,
                backend=ctx.config.intersection_backend,
            )
    compacted = ctx.visible_capacity != ctx.means.shape[0]
    if compacted:
        with jax.named_scope("visible_gather"):
            means2d_c = means2d_c[visible_ids]
            absgrad_probe_c = absgrad_probe_c[visible_ids]
            radii_c = radii_c[visible_ids]
            depths_c = depths_c[visible_ids]
            conics_c = conics_c[visible_ids]
            opacities_c = ctx.opacities[visible_ids]
            if ctx.calc_compensations:
                opacities_c = opacities_c * compensation_c[visible_ids]
            selected_means = ctx.means[visible_ids]
            selected_quats = ctx.quats[visible_ids]
            selected_scales = ctx.scales[visible_ids]
    else:
        opacities_c = ctx.opacities
        if ctx.calc_compensations:
            opacities_c = opacities_c * compensation_c
        selected_means = ctx.means
        selected_quats = ctx.quats
        selected_scales = ctx.scales
    opacities_c = jnp.where(retained_valid, opacities_c, 0.0)

    colors_arg = None
    if _has_color(ctx.render_mode):
        assert ctx.colors is not None
        if ctx.sh_degree is not None:
            if isinstance(ctx.colors, tuple):
                all_coefficients = _join_sh_coefficients(ctx.colors)
            elif ctx.colors.ndim == 4:
                all_coefficients = ctx.colors[camera_index]
            else:
                all_coefficients = ctx.colors
            coefficients = (
                all_coefficients[visible_ids] if compacted else all_coefficients
            )
            camera_center = _camera_centers(viewmat_c[None, ...])[0]
            directions = selected_means - camera_center[None, :]
            with jax.named_scope("spherical_harmonics"):
                colors_arg = jnp.maximum(
                    spherical_harmonics(
                        ctx.sh_degree,
                        directions,
                        coefficients,
                        masks=retained_valid,
                    )
                    + 0.5,
                    0.0,
                )
        else:
            colors_arg = (
                jnp.where(retained_valid[:, None], colors_c[visible_ids], 0.0)
                if compacted
                else colors_c
            )

    extra_arg = None
    if ctx.extra_signals is not None:
        if ctx.extra_signals_sh_degree is not None:
            all_extra_coefficients = (
                ctx.extra_signals[camera_index]
                if ctx.extra_signals.ndim == 4
                else ctx.extra_signals
            )
            extra_coefficients = (
                all_extra_coefficients[visible_ids]
                if compacted
                else all_extra_coefficients
            )
            camera_center = _camera_centers(viewmat_c[None, ...])[0]
            directions = selected_means - camera_center[None, :]
            with jax.named_scope("extra_signals_spherical_harmonics"):
                extra_arg = (
                    spherical_harmonics(
                        ctx.extra_signals_sh_degree,
                        directions,
                        extra_coefficients,
                        masks=retained_valid,
                    )
                    + 0.5
                )
        else:
            extra_arg = (
                jnp.where(retained_valid[:, None], extra_c[visible_ids], 0.0)
                if compacted
                else extra_c
            )

    feature_parts = []
    feature_background_parts = []
    if render_mode_has_color(ctx.render_mode):
        assert colors_arg is not None
        feature_parts.append(colors_arg)
        feature_background_parts.append(bg[: ctx.color_channels])
    if extra_arg is not None:
        feature_parts.append(extra_arg)
        feature_background_parts.append(
            jnp.zeros((ctx.extra_channels,), dtype=extra_arg.dtype)
        )
    combined_features = (
        None if not feature_parts else jnp.concatenate(feature_parts, axis=-1)
    )
    combined_background = (
        jnp.zeros((0,), dtype=ctx.means.dtype)
        if not feature_background_parts
        else jnp.concatenate(feature_background_parts, axis=-1)
    )
    has_depth_channel = render_mode_has_depth_channel(ctx.render_mode)
    expected_depth = render_mode_has_expected_depth(ctx.render_mode)
    if combined_features is None:
        composite_mode: RenderMode = "ED" if expected_depth else "D"
    elif has_depth_channel:
        composite_mode = "RGB+ED" if expected_depth else "RGB+D"
    else:
        composite_mode = "RGB"

    if ctx.with_eval3d:
        intersection_capacity = _automatic_intersection_capacity(
            means2d_c.shape[0], ctx.tile_count, ctx.config
        )
        if ctx.lidar_coeffs is None:
            intersections = intersect_tiles(
                means2d_c,
                radii_c,
                depths_c,
                retained_valid & (opacities_c > 0.0),
                tile_size=ctx.config.tile_size,
                tile_width=ctx.tile_width,
                tile_height=ctx.tile_height,
                max_intersections=intersection_capacity,
                backend=ctx.config.intersection_backend,
                sort_backend=ctx.config.sort_backend,
                mode="aabb",
            )
            intersection_offsets = intersections.offsets
            intersection_gaussian_ids = intersections.gaussian_ids
            intersection_tile_ids = intersections.tile_ids
            intersection_valid_count = intersections.valid_count
            intersection_required_count = intersections.required_count
            intersection_overflow = intersections.overflow
        else:
            lidar_intersections = isect_tiles_lidar(
                ctx.lidar_coeffs,
                means2d_c[None, ...],
                radii_c[None, ...],
                depths_c[None, ...],
                max_intersections=intersection_capacity,
                active_mask=(retained_valid & (opacities_c > 0.0))[None, ...],
            )
            encoded_offsets = isect_offset_encode(
                lidar_intersections.isect_ids,
                1,
                ctx.tile_width,
                ctx.tile_height,
                valid_count=lidar_intersections.valid_count,
                overflow=lidar_intersections.overflow,
                return_info=True,
            )
            intersection_offsets = encoded_offsets.offsets[0]
            intersection_gaussian_ids = lidar_intersections.flatten_ids
            intersection_valid_count = lidar_intersections.valid_count
            intersection_required_count = jnp.sum(
                lidar_intersections.tiles_per_gaussian, dtype=jnp.int32
            )
            intersection_overflow = lidar_intersections.overflow
        flat_offsets = intersection_offsets.reshape(-1)
        if ctx.lidar_coeffs is not None:
            intersection_positions = jnp.arange(
                intersection_gaussian_ids.shape[0], dtype=jnp.int32
            )
            intersection_tile_ids = (
                jnp.searchsorted(
                    flat_offsets,
                    intersection_positions,
                    side="right",
                )
                - 1
            )
            intersection_tile_ids = jnp.where(
                intersection_positions < intersection_valid_count,
                intersection_tile_ids,
                -1,
            ).astype(jnp.int32)
        ends = jnp.concatenate(
            (flat_offsets[1:], intersection_valid_count[None]), axis=0
        )
        candidate_counts = jnp.maximum(ends - flat_offsets, 0).reshape(
            ctx.tile_height, ctx.tile_width
        )
        if composite_mode == "RGB":
            features = combined_features
            feature_background = combined_background
        elif composite_mode in {"D", "ED"}:
            features = depths_c[:, None]
            feature_background = jnp.zeros((1,), dtype=depths_c.dtype)
        else:
            assert composite_mode in {"RGB+D", "RGB+ED"}
            assert combined_features is not None
            features = jnp.concatenate((combined_features, depths_c[:, None]), axis=-1)
            feature_background = jnp.concatenate(
                (
                    combined_background,
                    jnp.zeros((1,), dtype=combined_background.dtype),
                ),
                axis=0,
            )
        assert features is not None
        camera_radial = (
            None
            if ctx.radial_coeffs is None
            else jnp.asarray(ctx.radial_coeffs)[camera_index]
        )
        camera_tangential = (
            None
            if ctx.tangential_coeffs is None
            else jnp.asarray(ctx.tangential_coeffs)[camera_index]
        )
        camera_thin_prism = (
            None
            if ctx.thin_prism_coeffs is None
            else jnp.asarray(ctx.thin_prism_coeffs)[camera_index]
        )
        camera_viewmat_rs = (
            None
            if ctx.viewmats_rs is None
            else jnp.asarray(ctx.viewmats_rs)[camera_index]
        )
        common_arguments = (
            selected_means,
            selected_quats,
            selected_scales,
            features,
            opacities_c,
            viewmat_c,
            K_c,
        )
        if ctx.lidar_coeffs is None:
            rendered, alpha, eval3d_info = _rasterize_eval3d_camera(
                *common_arguments,
                ctx.width,
                ctx.height,
                ctx.config.tile_size,
                intersection_offsets,
                intersection_gaussian_ids,
                intersection_valid_count,
                background=feature_background,
                mask=None,
                camera_model=ctx.camera_model,
                radial_coeffs=camera_radial,
                tangential_coeffs=camera_tangential,
                thin_prism_coeffs=camera_thin_prism,
                ftheta_coeffs=ctx.ftheta_coeffs,
                external_distortion_coeffs=ctx.external_distortion_coeffs,
                rolling_shutter=ctx.rolling_shutter,
                viewmat_rs=camera_viewmat_rs,
                rays=None if ctx.rays is None else ctx.rays[camera_index],
                use_hit_distance=render_mode_has_hit_distance(ctx.render_mode),
                return_normals=ctx.return_normals,
                flatten_index_offset=0,
                max_gaussians_per_tile=ctx.config.max_gaussians_per_tile,
                tile_batch_size=ctx.config.tile_batch_size,
            )
        else:
            rendered, alpha, eval3d_info = _rasterize_eval3d_lidar(
                *common_arguments,
                intersection_offsets,
                intersection_gaussian_ids,
                intersection_valid_count,
                ctx.lidar_coeffs,
                background=feature_background,
                mask=None,
                viewmat_rs=camera_viewmat_rs,
                rays=None if ctx.rays is None else ctx.rays[camera_index],
                use_hit_distance=render_mode_has_hit_distance(ctx.render_mode),
                return_normals=ctx.return_normals,
                flatten_index_offset=0,
                max_gaussians_per_tile=ctx.config.max_gaussians_per_tile,
                tile_batch_size=ctx.config.tile_batch_size,
            )
        if composite_mode == "ED":
            rendered = rendered / jnp.maximum(alpha, ctx.config.transmittance_eps)
        elif composite_mode == "RGB+ED":
            rendered = rendered.at[..., -1].set(
                rendered[..., -1]
                / jnp.maximum(alpha[..., 0], ctx.config.transmittance_eps)
            )
        tile_info = {
            "candidate_counts": candidate_counts,
            "tile_overflow": eval3d_info["tile_overflow"],
            "candidate_limit_exceeded": (
                candidate_counts > ctx.config.max_gaussians_per_tile
            ),
            "intersection_count": intersection_valid_count,
            "intersection_required_count": intersection_required_count,
            "intersection_overflow": intersection_overflow,
            "intersection_capacity": jnp.asarray(
                intersection_capacity, dtype=jnp.int32
            ),
            "intersection_gaussian_ids": intersection_gaussian_ids,
            "intersection_tile_ids": intersection_tile_ids,
            "intersection_offsets": intersection_offsets,
        }
        if ctx.return_normals:
            tile_info["normals"] = eval3d_info["normals"]
    else:
        render_impl = (
            _render_camera_tiles
            if ctx.config.backend == "reference"
            else _render_camera_intersections
        )
        rendered, alpha, tile_info = render_impl(
            means2d_c,
            radii_c,
            depths_c,
            conics_c,
            opacities_c,
            combined_features,
            retained_valid,
            width=ctx.width,
            height=ctx.height,
            config=ctx.config,
            background=combined_background,
            render_mode=composite_mode,
            absgrad_probe=(absgrad_probe_c if ctx.absgrad_probe_enabled else None),
            allow_accutile=not ctx.use_ut,
        )
    local_gaussian_ids = tile_info["intersection_gaussian_ids"]
    intersection_positions = jnp.arange(local_gaussian_ids.shape[0], dtype=jnp.int32)
    intersection_valid = (intersection_positions < tile_info["intersection_count"]) & (
        local_gaussian_ids >= 0
    )
    if visible_ids.shape[0] == 0:
        original_gaussian_ids = jnp.full_like(local_gaussian_ids, -1)
    else:
        safe_local_ids = jnp.clip(local_gaussian_ids, 0, visible_ids.shape[0] - 1)
        original_gaussian_ids = visible_ids[safe_local_ids]
        original_gaussian_ids = jnp.where(
            intersection_valid, original_gaussian_ids, -1
        ).astype(jnp.int32)
    tile_info["intersection_gaussian_ids"] = original_gaussian_ids
    if has_depth_channel:
        rendered_depth = rendered[..., -1:]
    if render_mode_has_color(ctx.render_mode):
        rendered_primary = rendered[..., : ctx.color_channels]
    if extra_arg is not None:
        rendered_extra = rendered[
            ..., ctx.color_channels : ctx.color_channels + ctx.extra_channels
        ]
    if render_mode_has_color(ctx.render_mode) and has_depth_channel:
        rendered = jnp.concatenate((rendered_primary, rendered_depth), axis=-1)
    elif render_mode_has_color(ctx.render_mode):
        rendered = rendered_primary
    else:
        rendered = rendered_depth
    if extra_arg is not None:
        tile_info["render_extra_signals"] = rendered_extra
    tile_info = {
        **tile_info,
        "intersection_overflow": (
            tile_info["intersection_overflow"] | visible_overflow
        ),
        "intersection_required_count": jnp.maximum(
            tile_info["intersection_required_count"],
            jnp.maximum(visible_count, 0),
        ),
        "visible_count": visible_count,
        "visible_capacity": jnp.asarray(ctx.visible_capacity, dtype=jnp.int32),
        "visible_overflow": visible_overflow,
    }
    return rendered, alpha, tile_info


def rasterization(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    opacities: jax.Array,
    colors: ColorInput | None,
    viewmats: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    near_plane: float | None = None,
    far_plane: float | None = None,
    radius_clip: float | None = None,
    eps2d: float | None = None,
    sh_degree: int | None = None,
    packed: bool = True,
    tile_size: int | None = None,
    backgrounds: jax.Array | None = None,
    render_mode: RenderMode = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: RasterizeMode | None = None,
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: CameraModel = "pinhole",
    segmented: bool = False,
    covars: jax.Array | None = None,
    with_ut: bool = False,
    with_eval3d: bool = False,
    return_normals: bool = False,
    global_z_order: bool = True,
    rays: jax.Array | None = None,
    radial_coeffs: jax.Array | None = None,
    tangential_coeffs: jax.Array | None = None,
    thin_prism_coeffs: jax.Array | None = None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None = None,
    lidar_coeffs: Any | None = None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None = None,
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: jax.Array | None = None,
    ut_params: UnscentedTransformParameters | None = None,
    extra_signals: jax.Array | None = None,
    extra_signals_sh_degree: int | None = None,
    renderer_config: RendererConfig | None = None,
    *,
    active_mask: jax.Array | None = None,
    config: RasterizationConfig = RasterizationConfig(),
    _means2d_offset: jax.Array | None = None,
    distributed_world_size: int = 1,
    distributed_axis_name: Hashable | None = None,
    _means2d_absgrad_probe: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
    """Differentiably rasterize fixed-capacity 3D Gaussians.

    The high-level call follows gsplat's stable 3DGS API. Pure-JAX rendering
    processes every depth-sorted tile candidate. ``max_gaussians_per_tile`` is
    a temporary chunk size for the intersection renderer; exceeding it is
    diagnostic only and does not set ``info['tile_overflow']``.

    With the default ``config.max_intersections=None``, the intersection buffer
    covers every Gaussian/tile pair. Set an explicit capacity only when the
    caller handles ``info['intersection_overflow']``. In JAX, multi-rank calls
    additionally pass a static ``distributed_world_size`` and a bound
    ``distributed_axis_name``; the single-rank distributed path needs neither.

    ``sparse_grad=True`` follows current-main's unbatched ``packed=True``
    contract and rejects distributed, UT (including the implicit f-theta UT
    path), and Eval3D modes. JAX parameter gradients nevertheless remain dense
    over fixed storage; the training adapter applies equivalent visible-row
    optimizer semantics. ``absgrad`` uses the explicit probe described below.

    Dense and fixed-capacity packed canonical metadata are exposed for all
    renderer paths, including reference, eval3d, and leading batch dimensions.

    ``_means2d_offset`` is a private zero-valued training probe. When provided,
    it must match the dense projected means shape and exposes their true
    reverse-mode gradient without changing the rendered forward pass.

    ``_means2d_absgrad_probe`` is an independent zero-valued probe for gsplat's
    compositor-side AbsGrad statistic. Differentiate the loss with respect to
    it while passing ``absgrad=True``; it never changes forward values.
    """

    if config.intersection_backend == "cute":
        if distributed:
            raise NotImplementedError(
                "the CuTe intersection backend does not support distributed rendering"
            )
        if with_eval3d or with_ut or camera_model != "pinhole":
            raise NotImplementedError(
                "the optimized AccuTile intersection backend currently supports "
                "single-camera-style pinhole 3DGS only"
            )
    if config.projection_backend == "cute":
        if distributed:
            raise NotImplementedError(
                "the optimized projection does not support distributed rendering"
            )
        if sparse_grad:
            raise NotImplementedError(
                "the optimized projection does not support sparse_grad"
            )
        if with_ut or with_eval3d or camera_model != "pinhole" or covars is not None:
            raise NotImplementedError(
                "the optimized projection supports pinhole quaternion/scale 3DGS only"
            )
    if _means2d_absgrad_probe is not None and not absgrad:
        raise ValueError("_means2d_absgrad_probe requires absgrad=True")
    if distributed and config.compositor_backend == "cute":
        raise NotImplementedError(
            "the CuTe compositor does not support distributed rendering"
        )
    if config.compositor_backend in {"pallas", "cute"} and absgrad:
        raise NotImplementedError("the optimized compositors do not support AbsGrad")
    if config.compositor_backend in {"pallas", "cute"} and with_eval3d:
        raise NotImplementedError("the optimized compositors do not support Eval3D")
    if _means2d_absgrad_probe is not None and with_eval3d:
        raise ValueError(
            "_means2d_absgrad_probe is not supported with with_eval3d=True"
        )
    if distributed and sparse_grad:
        raise NotImplementedError(
            "distributed rasterization does not support sparse_grad"
        )
    if distributed and absgrad:
        raise NotImplementedError("distributed rasterization does not support absgrad")
    if sparse_grad:
        if not packed:
            raise ValueError("sparse_grad=True requires packed=True")
        if jnp.ndim(means) != 2:
            raise ValueError("sparse_grad does not support batch dimensions")
        if with_ut:
            raise ValueError("sparse_grad does not support with_ut=True")
        if with_eval3d:
            raise ValueError("sparse_grad does not support with_eval3d=True")
        if camera_model == "ftheta":
            raise ValueError(
                "sparse_grad does not support camera_model='ftheta' because "
                "f-theta projection uses the UT path"
            )

    if distributed:
        from .distributed import rasterization as distributed_rasterization

        return distributed_rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
            world_size=distributed_world_size,
            axis_name=distributed_axis_name,
            active_mask=active_mask,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            eps2d=eps2d,
            sh_degree=sh_degree,
            packed=packed,
            tile_size=tile_size,
            backgrounds=backgrounds,
            render_mode=render_mode,
            sparse_grad=sparse_grad,
            absgrad=absgrad,
            rasterize_mode=rasterize_mode,
            channel_chunk=channel_chunk,
            camera_model=camera_model,
            segmented=segmented,
            covars=covars,
            with_ut=with_ut,
            with_eval3d=with_eval3d,
            return_normals=return_normals,
            global_z_order=global_z_order,
            rays=rays,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            ftheta_coeffs=ftheta_coeffs,
            lidar_coeffs=lidar_coeffs,
            external_distortion_coeffs=external_distortion_coeffs,
            rolling_shutter=rolling_shutter,
            viewmats_rs=viewmats_rs,
            ut_params=ut_params,
            extra_signals=extra_signals,
            extra_signals_sh_degree=extra_signals_sh_degree,
            renderer_config=renderer_config,
            config=config,
            _means2d_offset=_means2d_offset,
        )
    renderer_config = resolve_renderer_config(renderer_config, with_eval3d=with_eval3d)
    if camera_model == "lidar":
        if not isinstance(
            lidar_coeffs, RowOffsetStructuredSpinningLidarModelParametersExt
        ):
            raise ValueError("camera_model='lidar' requires lidar_coeffs")
        if not with_ut:
            raise ValueError("LiDAR rendering requires with_ut=True")
        if not with_eval3d:
            raise ValueError("LiDAR rendering requires with_eval3d=True")
        width = lidar_coeffs.n_columns
        height = lidar_coeffs.n_rows
    elif lidar_coeffs is not None:
        raise ValueError("lidar_coeffs requires camera_model='lidar'")
    if external_distortion_coeffs is not None:
        if not isinstance(
            external_distortion_coeffs, BivariateWindshieldModelParameters
        ):
            raise TypeError(
                "external_distortion_coeffs must be BivariateWindshieldModelParameters"
            )
        if camera_model == "lidar":
            raise ValueError("LiDAR cameras do not support external distortion")
        if not with_ut:
            raise ValueError("external distortion requires with_ut=True")
        validate_external_distortion(external_distortion_coeffs)
    if rays is not None and not with_eval3d:
        raise ValueError("Rays input is only supported with with_eval3d=True")
    if return_normals and not with_eval3d:
        raise ValueError("return_normals=True requires with_eval3d=True")
    if not (
        render_mode_has_color(render_mode) or render_mode_has_depth_channel(render_mode)
    ):
        raise ValueError(f"unsupported render mode: {render_mode}")
    if render_mode_has_hit_distance(render_mode) and not with_eval3d:
        raise ValueError("hit-distance render modes require with_eval3d=True")

    means = jnp.asarray(means)
    quats = jnp.asarray(quats)
    scales = jnp.asarray(scales)
    opacities = jnp.asarray(opacities)
    viewmats = jnp.asarray(viewmats)
    Ks = jnp.asarray(Ks)
    if means.ndim < 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [..., N, 3]")
    batch_shape = means.shape[:-2]
    if batch_shape:
        gaussian_count = means.shape[-2]
        batch_count = math.prod(batch_shape)
        if quats.shape != batch_shape + (gaussian_count, 4):
            raise ValueError("quats must have shape [..., N, 4]")
        if scales.shape != batch_shape + (gaussian_count, 3):
            raise ValueError("scales must have shape [..., N, 3]")
        if opacities.shape != batch_shape + (gaussian_count,):
            raise ValueError("opacities must have shape [..., N]")
        if viewmats.shape[:-3] != batch_shape or viewmats.shape[-2:] != (4, 4):
            raise ValueError("viewmats must have shape [..., C, 4, 4]")
        camera_count = viewmats.shape[-3]
        if Ks.shape != batch_shape + (camera_count, 3, 3):
            raise ValueError("Ks must have shape [..., C, 3, 3]")

        def flatten_batch(value: jax.Array) -> jax.Array:
            return value.reshape((batch_count,) + value.shape[len(batch_shape) :])

        flat_means = flatten_batch(means)
        flat_quats = flatten_batch(quats)
        flat_scales = flatten_batch(scales)
        flat_opacities = flatten_batch(opacities)
        flat_viewmats = flatten_batch(viewmats)
        flat_Ks = flatten_batch(Ks)
        if _means2d_offset is None:
            flat_means2d_offset = None
        else:
            means2d_offset = jnp.asarray(_means2d_offset)
            expected_offset_shape = batch_shape + (
                camera_count,
                gaussian_count,
                2,
            )
            if means2d_offset.shape != expected_offset_shape:
                raise ValueError(
                    f"_means2d_offset must have shape {expected_offset_shape}"
                )
            flat_means2d_offset = flatten_batch(means2d_offset)
        if _means2d_absgrad_probe is None:
            flat_means2d_absgrad_probe = None
        else:
            means2d_absgrad_probe = jnp.asarray(_means2d_absgrad_probe)
            expected_probe_shape = batch_shape + (
                camera_count,
                gaussian_count,
                2,
            )
            if means2d_absgrad_probe.shape != expected_probe_shape:
                raise ValueError(
                    f"_means2d_absgrad_probe must have shape {expected_probe_shape}"
                )
            if means2d_absgrad_probe.dtype != means.dtype:
                raise ValueError(
                    "_means2d_absgrad_probe must have the same dtype as means"
                )
            flat_means2d_absgrad_probe = flatten_batch(means2d_absgrad_probe)

        if active_mask is None:
            batched_active_mask = jnp.ones(
                batch_shape + (gaussian_count,), dtype=jnp.bool_
            )
        else:
            batched_active_mask = jnp.asarray(active_mask, dtype=jnp.bool_)
            if batched_active_mask.shape == (gaussian_count,):
                batched_active_mask = jnp.broadcast_to(
                    batched_active_mask, batch_shape + (gaussian_count,)
                )
            if batched_active_mask.shape != batch_shape + (gaussian_count,):
                raise ValueError("active_mask must have shape [N] or [..., N]")
        flat_active_mask = flatten_batch(batched_active_mask)

        if colors is None:
            flat_colors = None
        elif isinstance(colors, tuple):
            if len(colors) != 2:
                raise ValueError("split SH colors must be a (sh0, sh_rest) pair")
            color_values = tuple(jnp.asarray(value) for value in colors)
            if any(
                value.shape[: len(batch_shape)] != batch_shape
                or value.shape[len(batch_shape)] != gaussian_count
                for value in color_values
            ):
                raise ValueError("split SH colors must have shape [..., N, K, D]")
            flat_colors = tuple(flatten_batch(value) for value in color_values)
        else:
            color_values = jnp.asarray(colors)
            color_tail = color_values.shape[len(batch_shape) :]
            shared_per_gaussian = bool(color_tail and color_tail[0] == gaussian_count)
            per_camera = bool(
                len(color_tail) >= 2
                and color_tail[0] == camera_count
                and color_tail[1] == gaussian_count
            )
            if color_values.shape[: len(batch_shape)] != batch_shape or not (
                shared_per_gaussian or per_camera
            ):
                raise ValueError(
                    "colors must have shape [..., N, ...] or [..., C, N, ...]"
                )
            flat_colors = flatten_batch(color_values)

        if extra_signals is None:
            flat_extra_signals = None
        else:
            extra_values = jnp.asarray(extra_signals)
            if extra_signals_sh_degree is not None:
                shared_shape = (
                    extra_values.ndim == 3 and extra_values.shape[0] == gaussian_count
                )
                batched_shape = (
                    extra_values.shape[: len(batch_shape)] == batch_shape
                    and extra_values.ndim == len(batch_shape) + 3
                    and extra_values.shape[len(batch_shape)] == gaussian_count
                )
                if shared_shape:
                    extra_values = jnp.broadcast_to(
                        extra_values, batch_shape + extra_values.shape
                    )
                elif not batched_shape:
                    raise ValueError(
                        "SH extra_signals must have shape [N, K, E] or [..., N, K, E]"
                    )
            else:
                extra_tail = extra_values.shape[len(batch_shape) :]
                shared_per_gaussian = bool(
                    extra_tail and extra_tail[0] == gaussian_count
                )
                per_camera = bool(
                    len(extra_tail) >= 2
                    and extra_tail[0] == camera_count
                    and extra_tail[1] == gaussian_count
                )
                if extra_values.shape[: len(batch_shape)] != batch_shape or not (
                    shared_per_gaussian or per_camera
                ):
                    raise ValueError(
                        "extra_signals must have shape [..., N, E] or [..., C, N, E]"
                    )
            flat_extra_signals = flatten_batch(extra_values)

        if backgrounds is None:
            flat_backgrounds = None
        else:
            background_values = jnp.asarray(backgrounds)
            if background_values.ndim == 1:
                background_values = jnp.broadcast_to(
                    background_values,
                    batch_shape + (camera_count, background_values.shape[0]),
                )
            elif (
                background_values.ndim == 2
                and background_values.shape[0] == camera_count
            ):
                background_values = jnp.broadcast_to(
                    background_values, batch_shape + background_values.shape
                )
            elif background_values.shape[: len(batch_shape)] != batch_shape:
                raise ValueError(
                    "backgrounds must have shape [D], [C, D], or [..., C, D]"
                )
            if background_values.shape[len(batch_shape)] != camera_count:
                raise ValueError("backgrounds camera dimension does not match viewmats")
            flat_backgrounds = flatten_batch(background_values)

        def flatten_optional_gaussians(
            name: str, value: jax.Array | None, trailing_shape: tuple[int, ...]
        ) -> jax.Array | None:
            if value is None:
                return None
            array = jnp.asarray(value)
            expected = batch_shape + (gaussian_count,) + trailing_shape
            if array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            return flatten_batch(array)

        flat_covars = flatten_optional_gaussians("covars", covars, (3, 3))

        def flatten_optional_cameras(
            name: str, value: jax.Array | None
        ) -> jax.Array | None:
            if value is None:
                return None
            array = jnp.asarray(value)
            if (
                array.shape[: len(batch_shape)] != batch_shape
                or array.shape[len(batch_shape)] != camera_count
            ):
                raise ValueError(f"{name} must have shape [..., C, ...]")
            return flatten_batch(array)

        flat_radial = flatten_optional_cameras("radial_coeffs", radial_coeffs)
        flat_tangential = flatten_optional_cameras(
            "tangential_coeffs", tangential_coeffs
        )
        flat_thin_prism = flatten_optional_cameras(
            "thin_prism_coeffs", thin_prism_coeffs
        )
        flat_viewmats_rs = flatten_optional_cameras("viewmats_rs", viewmats_rs)
        if rays is None:
            flat_rays = None
        else:
            ray_values = jnp.asarray(rays)
            expected_rays = batch_shape + (
                camera_count,
                height,
                width,
                6,
            )
            if ray_values.shape != expected_rays:
                raise ValueError(f"rays must have shape {expected_rays}")
            flat_rays = flatten_batch(ray_values)

        def take_optional(value: jax.Array | None, index: jax.Array):
            return None if value is None else value[index]

        def render_batch(index: jax.Array):
            batch_colors = (
                None
                if flat_colors is None
                else (
                    tuple(value[index] for value in flat_colors)
                    if isinstance(flat_colors, tuple)
                    else flat_colors[index]
                )
            )
            return rasterization(
                flat_means[index],
                flat_quats[index],
                flat_scales[index],
                flat_opacities[index],
                batch_colors,
                flat_viewmats[index],
                flat_Ks[index],
                width,
                height,
                active_mask=flat_active_mask[index],
                near_plane=near_plane,
                far_plane=far_plane,
                radius_clip=radius_clip,
                eps2d=eps2d,
                sh_degree=sh_degree,
                # Render each batch densely, then assemble one global static
                # packed prefix at the outer recursion boundary.
                packed=False,
                tile_size=tile_size,
                backgrounds=take_optional(flat_backgrounds, index),
                render_mode=render_mode,
                sparse_grad=sparse_grad,
                absgrad=absgrad,
                rasterize_mode=rasterize_mode,
                channel_chunk=channel_chunk,
                distributed=distributed,
                camera_model=camera_model,
                segmented=segmented,
                covars=take_optional(flat_covars, index),
                with_ut=with_ut,
                with_eval3d=with_eval3d,
                return_normals=return_normals,
                global_z_order=global_z_order,
                rays=take_optional(flat_rays, index),
                radial_coeffs=take_optional(flat_radial, index),
                tangential_coeffs=take_optional(flat_tangential, index),
                thin_prism_coeffs=take_optional(flat_thin_prism, index),
                ftheta_coeffs=ftheta_coeffs,
                lidar_coeffs=lidar_coeffs,
                external_distortion_coeffs=external_distortion_coeffs,
                rolling_shutter=rolling_shutter,
                viewmats_rs=take_optional(flat_viewmats_rs, index),
                ut_params=ut_params,
                extra_signals=take_optional(flat_extra_signals, index),
                extra_signals_sh_degree=extra_signals_sh_degree,
                renderer_config=renderer_config,
                config=config,
                _means2d_offset=take_optional(flat_means2d_offset, index),
                _means2d_absgrad_probe=take_optional(flat_means2d_absgrad_probe, index),
            )

        batched_outputs = jax.lax.map(
            render_batch,
            jnp.arange(batch_count, dtype=jnp.int32),
        )

        def restore_batch(value: jax.Array) -> jax.Array:
            return value.reshape(batch_shape + value.shape[1:])

        restored_outputs = jax.tree.map(restore_batch, batched_outputs)
        if packed:
            rendered, alphas, info = restored_outputs
            packed_metadata_available = all(
                key in info
                for key in (
                    "flatten_ids",
                    "isect_ids",
                    "isect_offsets",
                    "isect_valid_count",
                )
            )
            if packed_metadata_available:
                info.update(
                    _pack_dense_metadata(
                        info["radii"],
                        info["means2d"],
                        info["depths"],
                        info["conics"],
                        info["compensations"],
                        info["opacities"],
                        info["valid"],
                        info,
                    )
                )
                info["packed_requested"] = jnp.asarray(True)
                info["packed_metadata_available"] = jnp.asarray(True)
                info["n_batches"] = jnp.asarray(batch_count, dtype=jnp.int32)
                info["n_cameras"] = jnp.asarray(camera_count, dtype=jnp.int32)
                for key in (
                    "width",
                    "height",
                    "tile_size",
                    "tile_width",
                    "tile_height",
                ):
                    info[key] = info[key].reshape(-1)[0]
            else:
                info["packed_requested"] = jnp.ones(batch_shape, dtype=jnp.bool_)
                info["packed_metadata_available"] = jnp.zeros(
                    batch_shape, dtype=jnp.bool_
                )
            return rendered, alphas, info
        return restored_outputs

    packed_metadata_requested = bool(packed)
    dense_metadata_requested = not packed_metadata_requested
    packed_requested = jnp.asarray(packed)
    sparse_grad_requested = jnp.asarray(sparse_grad)
    absgrad_requested = jnp.asarray(absgrad)
    absgrad_probe_enabled = _means2d_absgrad_probe is not None
    del packed, sparse_grad, absgrad, channel_chunk, segmented
    if viewmats.ndim == 2:
        viewmats = viewmats[None, ...]
    if Ks.ndim == 2:
        Ks = Ks[None, ...]
    if viewmats.shape[0] != Ks.shape[0]:
        raise ValueError("viewmats and Ks must contain the same number of cameras")
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [N, 3]")
    if active_mask is None:
        active_mask = jnp.ones((means.shape[0],), dtype=jnp.bool_)
    if active_mask.shape != (means.shape[0],):
        raise ValueError("active_mask must have shape [N]")
    colors = _normalize_color_input(
        colors,
        gaussian_count=means.shape[0],
        camera_count=viewmats.shape[0],
        sh_degree=sh_degree,
    )
    extra_signals = _normalize_color_input(
        extra_signals,
        gaussian_count=means.shape[0],
        camera_count=viewmats.shape[0],
        sh_degree=extra_signals_sh_degree,
    )
    if isinstance(extra_signals, tuple):
        raise TypeError("extra_signals must be a single array")
    if extra_signals is not None and extra_signals.shape[-1] == 0:
        raise ValueError("extra_signals must contain at least one channel")
    if _has_color(render_mode) and colors is None:
        raise ValueError(f"render_mode={render_mode!r} requires colors")
    if rays is not None:
        rays = jnp.asarray(rays)
        expected_rays = (viewmats.shape[0], height, width, 6)
        if rays.shape != expected_rays:
            raise ValueError(f"rays must have shape {expected_rays}")

    overrides: dict[str, Any] = {}
    if near_plane is not None:
        overrides["near_plane"] = near_plane
    if far_plane is not None:
        overrides["far_plane"] = far_plane
    if radius_clip is not None:
        overrides["radius_clip"] = radius_clip
    if eps2d is not None:
        overrides["eps2d"] = eps2d
    if tile_size is not None:
        overrides["tile_size"] = tile_size
    if rasterize_mode is not None:
        overrides["rasterize_mode"] = rasterize_mode
    if tile_size is None and config.tile_size == RasterizationConfig().tile_size:
        overrides["tile_size"] = resolve_tile_size(
            None,
            with_eval3d=with_eval3d,
            width=width,
            height=height,
        )
    if overrides:
        config = replace(config, **overrides)

    if config.rasterize_mode != "classic" and (with_ut or with_eval3d):
        raise ValueError(
            "3DGUT rendering only supports rasterize_mode='classic'. "
            f"Got rasterize_mode={config.rasterize_mode!r} with "
            f"with_ut={with_ut} and with_eval3d={with_eval3d}."
        )

    calc_compensations = config.rasterize_mode == "antialiased"
    has_nonlinear_camera = (
        radial_coeffs is not None
        or tangential_coeffs is not None
        or thin_prism_coeffs is not None
        or ftheta_coeffs is not None
        or external_distortion_coeffs is not None
        or rolling_shutter != RollingShutterType.GLOBAL
    )
    if has_nonlinear_camera and not with_ut:
        raise ValueError("distortion and rolling shutter require with_ut=True")
    use_ut = with_ut or with_eval3d or camera_model == "ftheta"
    if use_ut:
        if covars is not None:
            raise ValueError("3DGUT requires quats/scales rather than covars")
        with jax.named_scope("projection_ut"):
            radii, means2d, depths, conics, compensations, projection_valid = (
                fully_fused_projection_with_ut(
                    means,
                    quats,
                    scales,
                    opacities,
                    viewmats,
                    Ks,
                    width,
                    height,
                    eps2d=config.eps2d,
                    near_plane=config.near_plane,
                    far_plane=config.far_plane,
                    radius_clip=config.radius_clip,
                    calc_compensations=calc_compensations,
                    camera_model=camera_model,
                    ut_params=ut_params,
                    radial_coeffs=radial_coeffs,
                    tangential_coeffs=tangential_coeffs,
                    thin_prism_coeffs=thin_prism_coeffs,
                    ftheta_coeffs=ftheta_coeffs,
                    lidar_coeffs=lidar_coeffs,
                    external_distortion_coeffs=external_distortion_coeffs,
                    rolling_shutter=rolling_shutter,
                    viewmats_rs=viewmats_rs,
                    global_z_order=global_z_order,
                    active_mask=active_mask,
                    ut_chunk_size=config.ut_chunk_size,
                    alpha_threshold=config.alpha_clip,
                )
            )
    else:
        with jax.named_scope("projection"):
            if config.projection_backend == "cute":
                from ._cute_projection import (
                    fully_fused_projection_cute as project,
                )

                projection_outputs = project(
                    means,
                    viewmats,
                    Ks,
                    width,
                    height,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    eps2d=config.eps2d,
                    near_plane=config.near_plane,
                    far_plane=config.far_plane,
                    radius_clip=config.radius_clip,
                    calc_compensations=calc_compensations,
                    alpha_threshold=config.alpha_clip,
                    active_mask=active_mask,
                )
            else:
                projection_outputs = fully_fused_projection(
                    means,
                    viewmats,
                    Ks,
                    width,
                    height,
                    quats=quats,
                    scales=scales,
                    covars=covars,
                    eps2d=config.eps2d,
                    near_plane=config.near_plane,
                    far_plane=config.far_plane,
                    radius_clip=config.radius_clip,
                    calc_compensations=calc_compensations,
                    camera_model=camera_model,  # pyright: ignore[reportArgumentType]
                    opacities=opacities,
                    active_mask=active_mask,
                    alpha_threshold=config.alpha_clip,
                )
            radii, means2d, depths, conics, compensations, projection_valid = (
                projection_outputs
            )
    if _means2d_offset is not None:
        means2d_offset = jnp.asarray(_means2d_offset, dtype=means2d.dtype)
        if means2d_offset.shape != means2d.shape:
            raise ValueError(f"_means2d_offset must have shape {means2d.shape}")
        means2d = means2d + means2d_offset
    if _means2d_absgrad_probe is None:
        means2d_absgrad_probe = jnp.zeros_like(means2d)
    else:
        means2d_absgrad_probe = jnp.asarray(_means2d_absgrad_probe)
        if means2d_absgrad_probe.shape != means2d.shape:
            raise ValueError(f"_means2d_absgrad_probe must have shape {means2d.shape}")
        if means2d_absgrad_probe.dtype != means2d.dtype:
            raise ValueError(
                "_means2d_absgrad_probe must have the same dtype as means2d"
            )
    valid = projection_valid
    color_channels = _color_channels(colors) if _has_color(render_mode) else 0
    extra_channels = _color_channels(extra_signals)

    if backgrounds is None:
        backgrounds = jnp.zeros((viewmats.shape[0], color_channels), dtype=means.dtype)
    elif backgrounds.ndim == 1:
        backgrounds = jnp.broadcast_to(
            backgrounds[None, :], (viewmats.shape[0], backgrounds.shape[0])
        )
    if render_mode_has_color(render_mode) and backgrounds.shape != (
        viewmats.shape[0],
        color_channels,
    ):
        raise ValueError(
            "backgrounds must have shape [C, D] matching the primary colors"
        )

    if colors is None or sh_degree is not None:
        color_xs = jnp.zeros((viewmats.shape[0], 0, color_channels), dtype=means.dtype)
    else:
        assert not isinstance(colors, tuple)
        color_xs = _prepare_colors(means, colors, viewmats, None, viewmats.shape[0])
        assert color_xs is not None

    if extra_signals is None or extra_signals_sh_degree is not None:
        extra_xs = jnp.zeros((viewmats.shape[0], 0, extra_channels), dtype=means.dtype)
    else:
        extra_xs = _prepare_colors(
            means,
            extra_signals,
            viewmats,
            None,
            viewmats.shape[0],
        )
        assert extra_xs is not None

    compensation_xs = (
        compensations
        if compensations is not None
        else jnp.zeros((viewmats.shape[0], 0), dtype=means.dtype)
    )
    if lidar_coeffs is None:
        tile_width = (width + config.tile_size - 1) // config.tile_size
        tile_height = (height + config.tile_size - 1) // config.tile_size
    else:
        tile_width = lidar_coeffs.tiling.n_bins_azimuth
        tile_height = lidar_coeffs.tiling.n_bins_elevation
    tile_count = tile_width * tile_height
    if config.backend == "reference":
        visible_capacity = means.shape[0]
    else:
        visible_capacity = min(
            means.shape[0],
            _automatic_intersection_capacity(means.shape[0], tile_count, config),
        )

    ctx = _CameraRenderContext(
        absgrad_probe_enabled=absgrad_probe_enabled,
        calc_compensations=calc_compensations,
        camera_model=camera_model,
        color_channels=color_channels,
        colors=colors,
        config=config,
        external_distortion_coeffs=external_distortion_coeffs,
        extra_channels=extra_channels,
        extra_signals=extra_signals,
        extra_signals_sh_degree=extra_signals_sh_degree,
        ftheta_coeffs=ftheta_coeffs,
        height=height,
        lidar_coeffs=lidar_coeffs,
        means=means,
        opacities=opacities,
        quats=quats,
        radial_coeffs=radial_coeffs,
        rays=rays,
        render_mode=render_mode,
        return_normals=return_normals,
        rolling_shutter=rolling_shutter,
        scales=scales,
        sh_degree=sh_degree,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        tile_count=tile_count,
        tile_height=tile_height,
        tile_width=tile_width,
        use_ut=use_ut,
        viewmats_rs=viewmats_rs,
        visible_capacity=visible_capacity,
        width=width,
        with_eval3d=with_eval3d,
    )
    render_one_camera = partial(_render_one_camera, ctx=ctx)

    camera_inputs = (
        jnp.arange(viewmats.shape[0], dtype=jnp.int32),
        means2d,
        means2d_absgrad_probe,
        radii,
        depths,
        conics,
        compensation_xs,
        color_xs,
        extra_xs,
        valid,
        viewmats,
        Ks,
        backgrounds,
    )
    if viewmats.shape[0] == 1:
        camera_output = render_one_camera(tuple(value[0] for value in camera_inputs))
        renders, alphas, tile_info = jax.tree.map(
            lambda value: value[None, ...], camera_output
        )
    else:
        # Serial camera batches keep the candidate workspace bounded and avoid
        # the large nested vmap+scan compilation seen on some CUDA toolchains.
        renders, alphas, tile_info = jax.lax.map(render_one_camera, camera_inputs)
    dense_intersection_metadata = None
    if all(
        key in tile_info
        for key in (
            "intersection_gaussian_ids",
            "intersection_tile_ids",
            "intersection_offsets",
        )
    ):
        dense_intersection_metadata = _assemble_dense_intersection_metadata(
            depths,
            tile_info["intersection_gaussian_ids"],
            tile_info["intersection_tile_ids"],
            tile_info["intersection_offsets"],
            tile_info["intersection_count"],
            tile_width=tile_width,
            tile_height=tile_height,
        )
    packed_metadata_available = (
        packed_metadata_requested and dense_intersection_metadata is not None
    )
    info = {
        "means2d": means2d,
        "radii": radii,
        "depths": depths,
        "conics": conics,
        "compensations": compensations,
        "valid": valid,
        "candidate_counts": tile_info["candidate_counts"],
        "tile_overflow": tile_info["tile_overflow"],
        "candidate_limit_exceeded": tile_info["candidate_limit_exceeded"],
        "intersection_count": tile_info["intersection_count"],
        "intersection_required_count": tile_info["intersection_required_count"],
        "intersection_overflow": tile_info["intersection_overflow"],
        "intersection_capacity": tile_info["intersection_capacity"],
        "visible_count": tile_info["visible_count"],
        "visible_capacity": tile_info["visible_capacity"],
        "visible_overflow": tile_info["visible_overflow"],
        "active_count": jnp.count_nonzero(active_mask),
        "used_unscented_transform": jnp.asarray(use_ut),
        "eval3d_ewa_approximation": jnp.asarray(False),
        "eval3d_world_space": jnp.asarray(with_eval3d),
        "distributed_world_size": jnp.asarray(1, dtype=jnp.int32),
        "distributed_requested": jnp.asarray(distributed),
        "distributed_feature_exchange": jnp.asarray(False),
        "packed_requested": packed_requested,
        "packed_metadata_available": jnp.asarray(packed_metadata_available),
        "sparse_grad_requested": sparse_grad_requested,
        "sparse_grad_is_dense": jnp.asarray(True),
        "absgrad_requested": absgrad_requested,
        "absgrad_available": jnp.asarray(absgrad_probe_enabled),
        "absgrad_probe_enabled": jnp.asarray(absgrad_probe_enabled),
    }
    if dense_metadata_requested:
        info.update(
            {
                "batch_ids": None,
                "camera_ids": None,
                "gaussian_ids": None,
            }
        )
        if dense_intersection_metadata is not None:
            info.update(dense_intersection_metadata)
    if extra_signals is not None:
        info["render_extra_signals"] = tile_info["render_extra_signals"]
    if return_normals:
        info["normals"] = tile_info["normals"]
    projected_opacities = jnp.broadcast_to(
        opacities[None, :], (viewmats.shape[0], opacities.shape[0])
    )
    if compensations is not None:
        projected_opacities = projected_opacities * compensations
    projected_opacities = jnp.where(active_mask[None, :], projected_opacities, 0.0)
    info.update(
        {
            "opacities": projected_opacities,
            "tile_width": jnp.asarray(tile_width, dtype=jnp.int32),
            "tile_height": jnp.asarray(tile_height, dtype=jnp.int32),
            "width": jnp.asarray(width, dtype=jnp.int32),
            "height": jnp.asarray(height, dtype=jnp.int32),
            "tile_size": jnp.asarray(config.tile_size, dtype=jnp.int32),
            "n_batches": jnp.asarray(1, dtype=jnp.int32),
            "n_cameras": jnp.asarray(viewmats.shape[0], dtype=jnp.int32),
        }
    )
    if packed_metadata_available:
        assert dense_intersection_metadata is not None
        info.update(
            _pack_dense_metadata(
                radii,
                means2d,
                depths,
                conics,
                compensations,
                projected_opacities,
                valid,
                dense_intersection_metadata,
            )
        )
    return renders, alphas, info


def rasterization_inria_wrapper(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    opacities: jax.Array,
    colors: ColorInput,
    viewmats: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: int | None = None,
    backgrounds: jax.Array | None = None,
    **kwargs: Any,
) -> tuple[jax.Array, None, dict[str, Any]]:
    """Match gsplat's INRIA comparison wrapper using the pure-JAX backend.

    The external INRIA extension returns only its color image through this
    wrapper. Keeping the same return contract makes comparison scripts portable
    without introducing the separately licensed CUDA dependency.
    """

    render_colors, _, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
        near_plane=near_plane,
        far_plane=far_plane,
        eps2d=eps2d,
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        **kwargs,
    )
    return render_colors, None, {}
