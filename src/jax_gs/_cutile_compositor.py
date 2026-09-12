from __future__ import annotations

"""Single-camera float32 3DGS compositor through NVIDIA cuTile Python."""

import math
import operator
from functools import partial

import cuda.tile as ct
import cuda.tile.jax as ctj
import jax
import jax.numpy as jnp

from .low_level import (
    DEFAULT_ALPHA_THRESHOLD,
    DEFAULT_TRANSMITTANCE_THRESHOLD,
)

_SUPPORTED_CHANNELS = frozenset((1, 2, 3, 4, 8, 16, 32))
_TILE_SIZE = 16
_TILE_PIXELS = _TILE_SIZE * _TILE_SIZE
_MAX_ALPHA = 0.999
_MIN_ONE_MINUS_ALPHA = 1.0e-3
_DONE_CHECK_INTERVAL = 16


def _static_int(name: str, value: int, *, minimum: int = 0) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a static integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _static_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a static real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_cuda_tile_device() -> jax.Device:
    try:
        devices = jax.local_devices(backend="gpu")
    except RuntimeError as exc:
        raise RuntimeError(
            "compositor_backend='cuda_tile' requires an NVIDIA CUDA GPU"
        ) from exc
    if not devices or any("cuda" not in str(device).lower() for device in devices):
        raise RuntimeError(
            "compositor_backend='cuda_tile' requires NVIDIA CUDA devices"
        )
    capabilities = {getattr(device, "compute_capability", None) for device in devices}
    if len(capabilities) != 1 or None in capabilities:
        raise RuntimeError(
            "compositor_backend='cuda_tile' requires one shared compute capability"
        )
    return devices[0]


@ct.kernel
def _compositor_forward_kernel(
    means,
    conics,
    colors,
    opacities,
    offsets,
    ids,
    valid_count,
    foreground,
    alpha_out,
    accepted_final_transmittance,
    last_ids,
    tile_overflow,
    gaussian_count: ct.Constant[int],
    input_capacity: ct.Constant[int],
    image_width: ct.Constant[int],
    image_height: ct.Constant[int],
    tile_width: ct.Constant[int],
    per_tile_bound: ct.Constant[int],
    channels: ct.Constant[int],
    storage_channels: ct.Constant[int],
    alpha_threshold: ct.Constant[float],
    transmittance_threshold: ct.Constant[float],
):
    tile_id = ct.bid(0)
    tile_x = tile_id % tile_width
    tile_y = tile_id // tile_width
    lane = ct.arange(_TILE_PIXELS, dtype=ct.int32)
    pixel_x = tile_x * _TILE_SIZE + lane % _TILE_SIZE
    pixel_y = tile_y * _TILE_SIZE + lane // _TILE_SIZE
    pixel_valid = (pixel_x < image_width) & (pixel_y < image_height)
    px = ct.astype(pixel_x, ct.float32) + 0.5
    py = ct.astype(pixel_y, ct.float32) + 0.5

    bounded_count = ct.maximum(
        0,
        ct.minimum(ct.load(valid_count, (0,), shape=()), input_capacity),
    )
    start = ct.maximum(
        0,
        ct.minimum(ct.load(offsets, (tile_id,), shape=()), bounded_count),
    )
    end = bounded_count
    if tile_id + 1 < offsets.shape[0]:
        end = ct.load(offsets, (tile_id + 1,), shape=())
    end = ct.maximum(start, ct.minimum(end, bounded_count))
    rendered_end = ct.minimum(end, start + per_tile_bound)
    ct.store(
        tile_overflow,
        (tile_id,),
        ct.astype(end - start > per_tile_bound, ct.uint8),
    )

    channel = ct.arange(storage_channels, dtype=ct.int32)
    rendered = ct.zeros((_TILE_PIXELS, storage_channels), ct.float32)
    transmittance = ct.ones((_TILE_PIXELS,), ct.float32)
    accepted_transmittance = ct.ones((_TILE_PIXELS,), ct.float32)
    last_id = ct.full((_TILE_PIXELS,), -1, ct.int32)
    done = ~pixel_valid

    position = start
    while position < rendered_end:
        # Amortize the reduction while bounding dead work after every pixel is done.
        if (position - start) % _DONE_CHECK_INTERVAL == 0 and ct.max(
            ct.astype(~done, ct.int32)
        ) == 0:
            break
        raw_id = ct.load(ids, (position,), shape=())
        candidate_valid = raw_id >= 0
        gaussian_id = ct.maximum(0, ct.minimum(raw_id, gaussian_count - 1))
        # The wrapper validates nonempty attribute shapes; gaussian_id is clamped.
        mean_x = ct.load(means, (gaussian_id, 0), shape=(), check_bounds=False)
        mean_y = ct.load(means, (gaussian_id, 1), shape=(), check_bounds=False)
        conic_x = ct.load(conics, (gaussian_id, 0), shape=(), check_bounds=False)
        conic_xy = ct.load(conics, (gaussian_id, 1), shape=(), check_bounds=False)
        conic_y = ct.load(conics, (gaussian_id, 2), shape=(), check_bounds=False)
        opacity = ct.load(opacities, (gaussian_id,), shape=(), check_bounds=False)
        color = ct.gather(
            colors,
            (gaussian_id, channel),
            mask=channel < channels,
            padding_value=0.0,
            check_bounds=False,
        )

        dx = px - mean_x
        dy = py - mean_y
        sigma = 0.5 * (conic_x * dx * dx + conic_y * dy * dy)
        sigma = sigma + conic_xy * dx * dy
        finite_sigma = (~ct.isnan(sigma)) & (ct.abs(sigma) != float("inf"))
        weight_valid = candidate_valid & (~done) & finite_sigma & (sigma >= 0.0)
        visibility = ct.exp(-ct.where(weight_valid, sigma, 0.0))
        raw_alpha = opacity * visibility
        invalid_alpha = ct.isnan(raw_alpha) | (
            (ct.abs(raw_alpha) == float("inf")) & (raw_alpha < 0.0)
        )
        alpha = ct.where(
            invalid_alpha,
            0.0,
            ct.minimum(_MAX_ALPHA, raw_alpha),
        )
        contributes = weight_valid & (alpha >= alpha_threshold)
        next_transmittance = transmittance * (1.0 - alpha)
        accepted = contributes & (next_transmittance > transmittance_threshold)
        weight = ct.where(accepted, alpha * transmittance, 0.0)
        rendered = rendered + weight[:, None] * color[None, :]
        accepted_transmittance = ct.where(
            accepted, next_transmittance, accepted_transmittance
        )
        last_id = ct.where(accepted, position, last_id)
        transmittance = ct.where(contributes, next_transmittance, transmittance)
        done = done | (contributes & ~accepted)
        position += 1

    ct.scatter(
        foreground,
        (pixel_y[:, None], pixel_x[:, None], channel[None, :]),
        rendered,
        mask=pixel_valid[:, None] & (channel[None, :] < channels),
    )
    ct.scatter(
        alpha_out,
        (pixel_y, pixel_x),
        1.0 - accepted_transmittance,
        mask=pixel_valid,
    )
    ct.scatter(
        accepted_final_transmittance,
        (pixel_y, pixel_x),
        accepted_transmittance,
        mask=pixel_valid,
    )
    ct.scatter(last_ids, (pixel_y, pixel_x), last_id, mask=pixel_valid)


@ct.kernel
def _compositor_backward_kernel(
    means,
    conics,
    colors,
    opacities,
    offsets,
    ids,
    valid_count,
    accepted_final_transmittance,
    last_ids,
    render_cotangent,
    alpha_cotangent,
    geometry_gradient,
    color_gradient,
    gaussian_count: ct.Constant[int],
    input_capacity: ct.Constant[int],
    image_width: ct.Constant[int],
    image_height: ct.Constant[int],
    tile_width: ct.Constant[int],
    per_tile_bound: ct.Constant[int],
    channels: ct.Constant[int],
    storage_channels: ct.Constant[int],
    alpha_threshold: ct.Constant[float],
):
    tile_id = ct.bid(0)
    tile_x = tile_id % tile_width
    tile_y = tile_id // tile_width
    lane = ct.arange(_TILE_PIXELS, dtype=ct.int32)
    pixel_x = tile_x * _TILE_SIZE + lane % _TILE_SIZE
    pixel_y = tile_y * _TILE_SIZE + lane // _TILE_SIZE
    pixel_valid = (pixel_x < image_width) & (pixel_y < image_height)
    px = ct.astype(pixel_x, ct.float32) + 0.5
    py = ct.astype(pixel_y, ct.float32) + 0.5

    bounded_count = ct.maximum(
        0,
        ct.minimum(ct.load(valid_count, (0,), shape=()), input_capacity),
    )
    start = ct.maximum(
        0,
        ct.minimum(ct.load(offsets, (tile_id,), shape=()), bounded_count),
    )
    end = bounded_count
    if tile_id + 1 < offsets.shape[0]:
        end = ct.load(offsets, (tile_id + 1,), shape=())
    end = ct.maximum(start, ct.minimum(end, bounded_count))
    rendered_end = ct.minimum(end, start + per_tile_bound)

    channel = ct.arange(storage_channels, dtype=ct.int32)
    transmittance = ct.gather(
        accepted_final_transmittance,
        (pixel_y, pixel_x),
        mask=pixel_valid,
        padding_value=1.0,
    )
    pixel_last_id = ct.gather(
        last_ids,
        (pixel_y, pixel_x),
        mask=pixel_valid,
        padding_value=-1,
    )
    render_gradient = ct.gather(
        render_cotangent,
        (pixel_y[:, None], pixel_x[:, None], channel[None, :]),
        mask=pixel_valid[:, None] & (channel[None, :] < channels),
    )
    output_alpha_gradient = ct.gather(
        alpha_cotangent,
        (pixel_y, pixel_x),
        mask=pixel_valid,
    )
    trailing_cotangent = ct.zeros((_TILE_PIXELS,), ct.float32)

    # Forward records the last accepted pair per pixel, so later pairs cannot
    # contribute to any gradient in this tile.
    position = ct.minimum(rendered_end - 1, ct.max(pixel_last_id))
    while position >= start:
        raw_id = ct.load(ids, (position,), shape=())
        candidate_valid = raw_id >= 0
        gaussian_id = ct.maximum(0, ct.minimum(raw_id, gaussian_count - 1))
        # The wrapper validates nonempty attribute shapes; gaussian_id is clamped.
        mean_x = ct.load(means, (gaussian_id, 0), shape=(), check_bounds=False)
        mean_y = ct.load(means, (gaussian_id, 1), shape=(), check_bounds=False)
        conic_x = ct.load(conics, (gaussian_id, 0), shape=(), check_bounds=False)
        conic_xy = ct.load(conics, (gaussian_id, 1), shape=(), check_bounds=False)
        conic_y = ct.load(conics, (gaussian_id, 2), shape=(), check_bounds=False)
        opacity = ct.load(opacities, (gaussian_id,), shape=(), check_bounds=False)
        color = ct.gather(
            colors,
            (gaussian_id, channel),
            mask=channel < channels,
            padding_value=0.0,
            check_bounds=False,
        )

        dx = px - mean_x
        dy = py - mean_y
        sigma = 0.5 * (conic_x * dx * dx + conic_y * dy * dy)
        sigma = sigma + conic_xy * dx * dy
        finite_sigma = (~ct.isnan(sigma)) & (ct.abs(sigma) != float("inf"))
        active = (
            pixel_valid
            & candidate_valid
            & (position <= pixel_last_id)
            & finite_sigma
            & (sigma >= 0.0)
        )
        visibility = ct.exp(-ct.where(active, sigma, 0.0))
        raw_alpha = opacity * visibility
        invalid_alpha = ct.isnan(raw_alpha) | (
            (ct.abs(raw_alpha) == float("inf")) & (raw_alpha < 0.0)
        )
        alpha = ct.where(
            invalid_alpha,
            0.0,
            ct.minimum(_MAX_ALPHA, raw_alpha),
        )
        active = active & (alpha >= alpha_threshold)

        if ct.max(ct.astype(active, ct.int32)) > 0:
            one_minus_alpha = 1.0 - alpha
            transmittance_before = transmittance / ct.maximum(
                _MIN_ONE_MINUS_ALPHA, one_minus_alpha
            )
            weight = alpha * transmittance_before
            weight_cotangent = output_alpha_gradient + ct.sum(
                render_gradient * color[None, :], axis=1
            )
            alpha_chain_cotangent = (
                weight_cotangent * transmittance_before
                - trailing_cotangent / one_minus_alpha
            )
            clamp_cotangent = ct.where(
                raw_alpha < _MAX_ALPHA,
                1.0,
                ct.where(raw_alpha == _MAX_ALPHA, 0.5, 0.0),
            )
            raw_alpha_cotangent = alpha_chain_cotangent * clamp_cotangent
            sigma_cotangent = -raw_alpha_cotangent * raw_alpha

            active_float = ct.astype(active, ct.float32)
            means_x_local = (
                -sigma_cotangent * (conic_x * dx + conic_xy * dy) * active_float
            )
            means_y_local = (
                -sigma_cotangent * (conic_y * dy + conic_xy * dx) * active_float
            )
            conic_x_local = sigma_cotangent * 0.5 * dx * dx * active_float
            conic_xy_local = sigma_cotangent * dx * dy * active_float
            conic_y_local = sigma_cotangent * 0.5 * dy * dy * active_float
            opacity_local = raw_alpha_cotangent * visibility * active_float
            color_local = render_gradient * (weight * active_float)[:, None]

            # Reduce the six geometry/opacity fields together, padded to eight.
            # Negative destinations discard the padding; completion is the only
            # synchronization point, so relaxed atomic ordering is sufficient.
            field = ct.arange(8, dtype=ct.int32)
            means_local = ct.cat(
                (means_x_local[:, None], means_y_local[:, None]), axis=1
            )
            conic_pair = ct.cat(
                (conic_x_local[:, None], conic_xy_local[:, None]), axis=1
            )
            opacity_pair = ct.cat(
                (conic_y_local[:, None], opacity_local[:, None]), axis=1
            )
            geometry = ct.cat(
                (
                    ct.cat((means_local, conic_pair), axis=1),
                    ct.cat(
                        (opacity_pair, ct.zeros((_TILE_PIXELS, 2), ct.float32)), axis=1
                    ),
                ),
                axis=1,
            )
            ct.atomic_add(
                geometry_gradient,
                (gaussian_id, ct.where(field < 6, field, -1)),
                ct.sum(geometry, axis=0),
                memory_order=ct.MemoryOrder.RELAXED,
            )
            ct.atomic_add(
                color_gradient,
                (gaussian_id, channel),
                ct.where(channel < channels, ct.sum(color_local, axis=0), 0.0),
                memory_order=ct.MemoryOrder.RELAXED,
            )
            trailing_cotangent = ct.where(
                active,
                trailing_cotangent + weight_cotangent * weight,
                trailing_cotangent,
            )
            transmittance = ct.where(active, transmittance_before, transmittance)
        position -= 1


def _run_forward(
    means2d,
    conics,
    colors,
    opacities,
    offsets,
    flatten_ids,
    valid_count,
    *,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    tile_height, tile_width = offsets.shape
    tile_count = tile_height * tile_width
    gaussian_count = means2d.shape[0]
    input_capacity = flatten_ids.shape[0]
    channels = colors.shape[-1]
    storage_channels = 1 << (channels - 1).bit_length()
    outputs = (
        ctj.OutputPlaceholder((image_height, image_width, channels), jnp.float32),
        ctj.OutputPlaceholder((image_height, image_width), jnp.float32),
        ctj.OutputPlaceholder((image_height, image_width), jnp.float32),
        ctj.OutputPlaceholder((image_height, image_width), jnp.int32),
        ctj.OutputPlaceholder((tile_count,), jnp.uint8),
    )
    foreground, alpha, accepted, last, tile_overflow = ctj.cutile_call(
        (tile_count,),
        _compositor_forward_kernel,
        (
            means2d,
            conics,
            colors,
            opacities,
            offsets.reshape((-1,)),
            flatten_ids,
            valid_count.reshape((1,)),
            *outputs,
            gaussian_count,
            input_capacity,
            image_width,
            image_height,
            tile_width,
            per_tile_bound,
            channels,
            storage_channels,
            alpha_threshold,
            transmittance_threshold,
        ),
    )
    return foreground, alpha, accepted, last, tile_overflow.astype(jnp.bool_)


def _run_backward(
    means2d,
    conics,
    colors,
    opacities,
    offsets,
    flatten_ids,
    valid_count,
    accepted_final_transmittance,
    last_ids,
    rendered_cotangent,
    alpha_cotangent,
    *,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
):
    tile_height, tile_width = offsets.shape
    tile_count = tile_height * tile_width
    gaussian_count = means2d.shape[0]
    input_capacity = flatten_ids.shape[0]
    channels = colors.shape[-1]
    storage_channels = 1 << (channels - 1).bit_length()
    # Independent outputs avoid ordering geometry and color atomics together.
    geometry_gradient = jnp.zeros((gaussian_count, 6), jnp.float32)
    color_gradient = jnp.zeros((gaussian_count, channels), jnp.float32)
    geometry_gradient, color_gradient = ctj.cutile_call(
        (tile_count,),
        _compositor_backward_kernel,
        (
            means2d,
            conics,
            colors,
            opacities,
            offsets.reshape((-1,)),
            flatten_ids,
            valid_count.reshape((1,)),
            accepted_final_transmittance,
            last_ids,
            rendered_cotangent,
            alpha_cotangent,
            ctj.InputOutput(geometry_gradient),
            ctj.InputOutput(color_gradient),
            gaussian_count,
            input_capacity,
            image_width,
            image_height,
            tile_width,
            per_tile_bound,
            channels,
            storage_channels,
            alpha_threshold,
        ),
    )
    return (
        geometry_gradient[:, :2],
        geometry_gradient[:, 2:5],
        color_gradient,
        geometry_gradient[:, 5],
    )


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11))
def _composite_foreground(
    means2d,
    conics,
    colors,
    opacities,
    offsets,
    flatten_ids,
    valid_count,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    foreground, alpha, _, _, tile_overflow = _run_forward(
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    return foreground, alpha[..., None], tile_overflow


def _composite_foreground_fwd(
    means2d,
    conics,
    colors,
    opacities,
    offsets,
    flatten_ids,
    valid_count,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    foreground, alpha, accepted, last, tile_overflow = _run_forward(
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    residuals = (
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        accepted,
        last,
    )
    return (foreground, alpha[..., None], tile_overflow), residuals


def _composite_foreground_bwd(
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
    residuals,
    cotangents,
):
    del transmittance_threshold
    (
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        accepted,
        last,
    ) = residuals
    rendered_cotangent, alpha_cotangent, _ = cotangents
    gradients = _run_backward(
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        accepted,
        last,
        rendered_cotangent,
        alpha_cotangent[..., 0],
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
    )
    return (*gradients, None, None, None)


_composite_foreground.defvjp(_composite_foreground_fwd, _composite_foreground_bwd)


def rasterize_to_pixels_cutile(
    means2d: jax.Array,
    conics: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: jax.Array,
    flatten_ids: jax.Array,
    backgrounds: jax.Array | None = None,
    *,
    valid_count: jax.Array | int,
    overflow: jax.Array | bool = False,
    max_gaussians_per_tile: int = 512,
    max_candidates_per_tile: int | None = None,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Composite one camera with cuTile forward and backward kernels."""

    image_width = _static_int("image_width", image_width, minimum=1)
    image_height = _static_int("image_height", image_height, minimum=1)
    tile_size = _static_int("tile_size", tile_size, minimum=1)
    if tile_size != _TILE_SIZE:
        raise ValueError("the cuTile compositor currently requires tile_size=16")
    max_gaussians_per_tile = _static_int(
        "max_gaussians_per_tile", max_gaussians_per_tile, minimum=1
    )
    if max_candidates_per_tile is not None:
        max_candidates_per_tile = _static_int(
            "max_candidates_per_tile", max_candidates_per_tile, minimum=1
        )

    means2d = jnp.asarray(means2d)
    conics = jnp.asarray(conics)
    colors = jnp.asarray(colors)
    opacities = jnp.asarray(opacities)
    offsets = jnp.asarray(isect_offsets, jnp.int32)
    flatten_ids = jnp.asarray(flatten_ids, jnp.int32)
    valid_count_array = jnp.asarray(valid_count, jnp.int32)
    gaussian_count = means2d.shape[0]
    channels = colors.shape[-1]
    input_capacity = flatten_ids.shape[0]

    if gaussian_count == 0 or input_capacity == 0:
        raise ValueError(
            "the cuTile compositor requires non-empty fixed-capacity inputs"
        )
    if means2d.shape != (gaussian_count, 2):
        raise ValueError("means2d must have shape [N, 2]")
    if conics.shape != (gaussian_count, 3):
        raise ValueError("conics must have shape [N, 3]")
    if colors.shape != (gaussian_count, channels):
        raise ValueError("colors must have shape [N, channels]")
    if channels not in _SUPPORTED_CHANNELS:
        raise ValueError(
            "the cuTile compositor supports channel counts "
            f"{sorted(_SUPPORTED_CHANNELS)}, got {channels}"
        )
    if opacities.shape != (gaussian_count,):
        raise ValueError("opacities must have shape [N]")
    if offsets.ndim != 2:
        raise ValueError("isect_offsets must have shape [tile_height, tile_width]")
    if flatten_ids.ndim != 1:
        raise ValueError("flatten_ids must be one-dimensional")
    if valid_count_array.shape != ():
        raise ValueError("valid_count must be a scalar")
    if any(
        value.dtype != jnp.float32 for value in (means2d, conics, colors, opacities)
    ):
        raise TypeError("the cuTile compositor requires float32 inputs")
    if offsets.dtype != jnp.int32 or flatten_ids.dtype != jnp.int32:
        raise TypeError("the cuTile compositor requires int32 metadata")

    tile_height, tile_width = offsets.shape
    if tile_width * tile_size < image_width or tile_height * tile_size < image_height:
        raise ValueError("isect_offsets tile grid does not cover the image")
    if tile_height * tile_size >= image_height + tile_size:
        raise ValueError("isect_offsets has extra tile rows beyond the image")
    if tile_width * tile_size >= image_width + tile_size:
        raise ValueError("isect_offsets has extra tile columns beyond the image")
    if backgrounds is None:
        background_array = jnp.zeros((channels,), jnp.float32)
    else:
        background_array = jnp.asarray(backgrounds)
        if background_array.shape != (channels,):
            raise ValueError("backgrounds must have shape [channels]")
        if background_array.dtype != jnp.float32:
            raise TypeError("the cuTile compositor requires float32 inputs")

    _require_cuda_tile_device()
    per_tile_bound = min(gaussian_count, input_capacity)
    if max_candidates_per_tile is not None:
        per_tile_bound = min(per_tile_bound, max_candidates_per_tile)
    per_tile_bound = (
        math.ceil(per_tile_bound / max_gaussians_per_tile) * max_gaussians_per_tile
    )
    foreground, alphas, tile_overflow = _composite_foreground(
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count_array,
        image_width,
        image_height,
        per_tile_bound,
        _static_float("alpha_threshold", alpha_threshold),
        _static_float("transmittance_threshold", transmittance_threshold),
    )
    rendered = foreground + background_array[None, None, :] * (1.0 - alphas)
    tile_overflow = tile_overflow.reshape(offsets.shape)
    return (
        rendered,
        alphas,
        {
            "tile_overflow": tile_overflow,
            "overflow": jnp.asarray(overflow, jnp.bool_) | jnp.any(tile_overflow),
        },
    )


__all__ = ["rasterize_to_pixels_cutile"]
