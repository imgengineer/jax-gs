# pyright: reportMissingImports=false

"""Single-camera float32 3DGS compositor through CuTe DSL."""

from __future__ import annotations

from functools import partial
import math
import operator

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import cuda.bindings.driver as cuda

from .low_level import (
    DEFAULT_ALPHA_THRESHOLD,
    DEFAULT_TRANSMITTANCE_THRESHOLD,
)


_SUPPORTED_CHANNELS = frozenset((1, 2, 3, 4, 8, 16, 32))
_TILE_SIZE = 16
_TILE_PIXELS = _TILE_SIZE * _TILE_SIZE
_MAX_ALPHA = 0.999
_MIN_ONE_MINUS_ALPHA = 1.0e-3
_WIDE_BATCH_MAX_CHANNELS = 4
_WIDE_BATCH_LOADS_PER_THREAD = 2


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
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a static real scalar") from exc


def _cute_device() -> jax.Device:
    try:
        devices = jax.local_devices(backend="gpu")
    except RuntimeError as exc:
        raise RuntimeError(
            "compositor_backend='cute' requires an NVIDIA CUDA GPU"
        ) from exc
    if not devices or any("cuda" not in str(device).lower() for device in devices):
        raise RuntimeError(
            "compositor_backend='cute' requires NVIDIA CUDA devices"
        )
    capabilities = {
        getattr(device, "compute_capability", None) for device in devices
    }
    if len(capabilities) != 1 or None in capabilities:
        raise RuntimeError(
            "compositor_backend='cute' requires one shared compute capability"
        )
    return devices[0]


@cute.jit
def _clamp_int(value, minimum, maximum):
    return cutlass.min(cutlass.max(value, minimum), maximum)


@cute.jit
def _composite_tile_forward(
    means: cute.Tensor,
    conics: cute.Tensor,
    colors: cute.Tensor,
    opacities: cute.Tensor,
    ids: cute.Tensor,
    foreground: cute.Tensor,
    alpha_out: cute.Tensor,
    accepted_final_transmittance: cute.Tensor,
    last_ids: cute.Tensor,
    tile_overflow: cute.Tensor,
    tile_id,
    thread,
    start,
    end,
    gaussian_count: int,
    image_width: int,
    image_height: int,
    tile_width: int,
    per_tile_bound: int,
    channels: cutlass.Constexpr[int],
    batch_capacity: cutlass.Constexpr[int],
    alpha_threshold: float,
    transmittance_threshold: float,
):
    tile_x = tile_id % tile_width
    tile_y = tile_id // tile_width
    pixel_x = tile_x * _TILE_SIZE + (thread & (_TILE_SIZE - 1))
    pixel_y = tile_y * _TILE_SIZE + (thread >> 4)
    pixel_valid = pixel_x < image_width and pixel_y < image_height
    pixel_id = cutlass.min(
        pixel_y * image_width + pixel_x,
        image_width * image_height - 1,
    )

    candidate_count = end - start
    rendered_count = cutlass.min(candidate_count, per_tile_bound)
    if thread == 0:
        tile_overflow[tile_id] = cutlass.Uint8(
            1 if candidate_count > per_tile_bound else 0
        )

    smem = cutlass.utils.SmemAllocator()
    batch_ids = smem.allocate_tensor(cutlass.Int32, batch_capacity)
    batch_means = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((batch_capacity, 2))
    )
    batch_conics = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((batch_capacity, 3))
    )
    batch_opacities = smem.allocate_tensor(cutlass.Float32, batch_capacity)
    batch_colors = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((batch_capacity, channels))
    )

    rendered = cute.make_rmem_tensor(
        cute.make_layout(channels), cutlass.Float32
    )
    for channel in cutlass.range_constexpr(channels):
        rendered[channel] = 0.0
    transmittance = cutlass.Float32(1.0)
    accepted_transmittance = cutlass.Float32(1.0)
    last_id = cutlass.Int32(-1)
    done = not pixel_valid
    px = cutlass.Float32(pixel_x) + 0.5
    py = cutlass.Float32(pixel_y) + 0.5

    for batch_start in cutlass.range(
        cutlass.Int32(0), rendered_count, batch_capacity
    ):
        for slot in cutlass.range(
            cutlass.Int32(thread), batch_capacity, _TILE_PIXELS
        ):
            local = batch_start + slot
            if local < rendered_count:
                position = start + local
                raw_id = ids[position]
                gaussian_id = cutlass.Int32(-1)
                if raw_id >= 0:
                    gaussian_id = cutlass.min(raw_id, gaussian_count - 1)
                batch_ids[slot] = gaussian_id
                if gaussian_id >= 0:
                    batch_means[slot, 0] = means[gaussian_id, 0]
                    batch_means[slot, 1] = means[gaussian_id, 1]
                    batch_conics[slot, 0] = conics[gaussian_id, 0]
                    batch_conics[slot, 1] = conics[gaussian_id, 1]
                    batch_conics[slot, 2] = conics[gaussian_id, 2]
                    batch_opacities[slot] = opacities[gaussian_id]
                    for channel in cutlass.range_constexpr(channels):
                        batch_colors[slot, channel] = colors[
                            gaussian_id, channel
                        ]
        cute.arch.sync_threads()

        batch_size = cutlass.min(
            batch_capacity, rendered_count - batch_start
        )
        for t in cutlass.range(batch_size):
            if not done:
                gaussian_id = batch_ids[t]
                if gaussian_id >= 0:
                    dx = px - batch_means[t, 0]
                    dy = py - batch_means[t, 1]
                    sigma = 0.5 * (
                        batch_conics[t, 0] * dx * dx
                        + batch_conics[t, 2] * dy * dy
                    ) + batch_conics[t, 1] * dx * dy
                    weight_valid = cute.math.isfinite(sigma) and sigma >= 0.0
                    if weight_valid:
                        visibility = cute.math.exp(-sigma, fastmath=True)
                        raw_alpha = batch_opacities[t] * visibility
                        alpha = cutlass.min(_MAX_ALPHA, raw_alpha)
                        if cute.math.isnan(raw_alpha) or (
                            cute.math.isinf(raw_alpha) and raw_alpha < 0.0
                        ):
                            alpha = cutlass.Float32(0.0)
                        if alpha >= alpha_threshold:
                            next_transmittance = transmittance * (1.0 - alpha)
                            if next_transmittance > transmittance_threshold:
                                weight = alpha * transmittance
                                for channel in cutlass.range_constexpr(channels):
                                    rendered[channel] = (
                                        rendered[channel]
                                        + batch_colors[t, channel] * weight
                                    )
                                transmittance = next_transmittance
                                accepted_transmittance = next_transmittance
                                last_id = start + batch_start + t
                            else:
                                transmittance = next_transmittance
                                done = True
        cute.arch.sync_threads()

    if pixel_valid:
        for channel in cutlass.range_constexpr(channels):
            foreground[pixel_y, pixel_x, channel] = rendered[channel]
        alpha_out[pixel_y, pixel_x] = 1.0 - accepted_transmittance
        accepted_final_transmittance[pixel_y, pixel_x] = accepted_transmittance
        last_ids[pixel_y, pixel_x] = last_id


@cute.kernel
def _compositor_forward_kernel(
    means: cute.Tensor,
    conics: cute.Tensor,
    colors: cute.Tensor,
    opacities: cute.Tensor,
    offsets: cute.Tensor,
    ids: cute.Tensor,
    valid_count: cute.Tensor,
    foreground: cute.Tensor,
    alpha_out: cute.Tensor,
    accepted_final_transmittance: cute.Tensor,
    last_ids: cute.Tensor,
    tile_overflow: cute.Tensor,
    gaussian_count: int,
    input_capacity: int,
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_count: int,
    per_tile_bound: int,
    channels: cutlass.Constexpr[int],
    alpha_threshold: float,
    transmittance_threshold: float,
):
    tile_id, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    if tile_id < tile_count:
        bounded_valid_count = _clamp_int(
            valid_count[0], cutlass.Int32(0), input_capacity
        )
        start = _clamp_int(
            offsets[tile_id], cutlass.Int32(0), bounded_valid_count
        )
        end = bounded_valid_count
        if tile_id + 1 < tile_count:
            end = offsets[tile_id + 1]
        end = _clamp_int(end, start, bounded_valid_count)
        _composite_tile_forward(
            means,
            conics,
            colors,
            opacities,
            ids,
            foreground,
            alpha_out,
            accepted_final_transmittance,
            last_ids,
            tile_overflow,
            tile_id,
            thread,
            start,
            end,
            gaussian_count,
            image_width,
            image_height,
            tile_width,
            per_tile_bound,
            channels,
            _TILE_PIXELS
            * (
                _WIDE_BATCH_LOADS_PER_THREAD
                if channels <= _WIDE_BATCH_MAX_CHANNELS
                else 1
            ),
            alpha_threshold,
            transmittance_threshold,
        )


@cute.kernel
def _clear_float_kernel(
    output: cute.Tensor, size: cutlass.Constexpr[int]
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * _TILE_PIXELS + thread
    if index < size:
        flat = cute.make_tensor(output.iterator, cute.make_layout(size))
        flat[index] = 0.0


@cute.kernel
def _compositor_backward_kernel(
    means: cute.Tensor,
    conics: cute.Tensor,
    colors: cute.Tensor,
    opacities: cute.Tensor,
    offsets: cute.Tensor,
    ids: cute.Tensor,
    valid_count: cute.Tensor,
    accepted_final_transmittance: cute.Tensor,
    last_ids: cute.Tensor,
    render_cotangent: cute.Tensor,
    alpha_cotangent: cute.Tensor,
    packed_gradient: cute.Tensor,
    gaussian_count: int,
    input_capacity: int,
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_count: int,
    per_tile_bound: int,
    channels: cutlass.Constexpr[int],
    alpha_threshold: float,
    transmittance_threshold: float,
):
    del transmittance_threshold
    tile_id, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    if tile_id < tile_count:
        tile_x = tile_id % tile_width
        tile_y = tile_id // tile_width
        pixel_x = tile_x * _TILE_SIZE + (thread & (_TILE_SIZE - 1))
        pixel_y = tile_y * _TILE_SIZE + (thread >> 4)
        inside = pixel_x < image_width and pixel_y < image_height
        pixel_id = cutlass.min(
            pixel_y * image_width + pixel_x,
            image_width * image_height - 1,
        )
        px = cutlass.Float32(pixel_x) + 0.5
        py = cutlass.Float32(pixel_y) + 0.5

        bounded_valid_count = _clamp_int(
            valid_count[0], cutlass.Int32(0), input_capacity
        )
        start = _clamp_int(
            offsets[tile_id], cutlass.Int32(0), bounded_valid_count
        )
        end = bounded_valid_count
        if tile_id + 1 < tile_count:
            end = offsets[tile_id + 1]
        end = _clamp_int(end, start, bounded_valid_count)
        rendered_end = cutlass.min(end, start + per_tile_bound)

        loads_per_thread = (
            _WIDE_BATCH_LOADS_PER_THREAD
            if channels <= _WIDE_BATCH_MAX_CHANNELS
            else 1
        )
        batch_capacity = _TILE_PIXELS * loads_per_thread
        smem = cutlass.utils.SmemAllocator()
        batch_ids = smem.allocate_tensor(cutlass.Int32, batch_capacity)
        batch_means = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((batch_capacity, 2))
        )
        batch_conics = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((batch_capacity, 3))
        )
        batch_opacities = smem.allocate_tensor(cutlass.Float32, batch_capacity)
        batch_colors = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((batch_capacity, channels))
        )
        warp_last_ids = smem.allocate_tensor(cutlass.Int32, 8)
        block_last_id = smem.allocate_tensor(cutlass.Int32, 1)

        transmittance = cutlass.Float32(1.0)
        pixel_last_id = cutlass.Int32(-1)
        if inside:
            transmittance = accepted_final_transmittance[pixel_y, pixel_x]
            pixel_last_id = last_ids[pixel_y, pixel_x]
        warp_last_id = pixel_last_id
        for offset in (16, 8, 4, 2, 1):
            warp_last_id = cutlass.max(
                warp_last_id,
                cute.arch.shuffle_sync_bfly(warp_last_id, offset),
            )

        if thread < 8:
            warp_last_ids[thread] = -1
        cute.arch.sync_threads()
        if (thread & 31) == 0:
            warp_last_ids[thread >> 5] = warp_last_id
        cute.arch.sync_threads()
        if thread == 0:
            maximum = cutlass.Int32(-1)
            for warp in cutlass.range_constexpr(8):
                maximum = cutlass.max(maximum, warp_last_ids[warp])
            block_last_id[0] = maximum
        cute.arch.sync_threads()

        trailing_cotangent = cutlass.Float32(0.0)
        render_gradient = cute.make_rmem_tensor(
            cute.make_layout(channels), cutlass.Float32
        )
        for channel in cutlass.range_constexpr(channels):
            render_gradient[channel] = 0.0
        output_alpha_gradient = cutlass.Float32(0.0)
        if inside:
            for channel in cutlass.range_constexpr(channels):
                render_gradient[channel] = render_cotangent[
                    pixel_y, pixel_x, channel
                ]
            output_alpha_gradient = alpha_cotangent[pixel_y, pixel_x]

        batches = (
            rendered_end - start + batch_capacity - 1
        ) // batch_capacity
        for batch in cutlass.range(batches):
            batch_end = rendered_end - 1 - batch * batch_capacity
            batch_size = cutlass.min(
                batch_capacity, batch_end + 1 - start
            )
            if batch_end - (batch_size - 1) <= block_last_id[0]:
                for load in cutlass.range_constexpr(loads_per_thread):
                    slot = thread + load * _TILE_PIXELS
                    position = batch_end - slot
                    if position >= start:
                        raw_id = ids[position]
                        gaussian_id = cutlass.Int32(-1)
                        if raw_id >= 0:
                            gaussian_id = cutlass.min(
                                raw_id, gaussian_count - 1
                            )
                        batch_ids[slot] = gaussian_id
                        if gaussian_id >= 0:
                            batch_means[slot, 0] = means[gaussian_id, 0]
                            batch_means[slot, 1] = means[gaussian_id, 1]
                            batch_conics[slot, 0] = conics[gaussian_id, 0]
                            batch_conics[slot, 1] = conics[gaussian_id, 1]
                            batch_conics[slot, 2] = conics[gaussian_id, 2]
                            batch_opacities[slot] = opacities[gaussian_id]
                            for channel in cutlass.range_constexpr(channels):
                                batch_colors[slot, channel] = colors[
                                    gaussian_id, channel
                                ]
                cute.arch.sync_threads()

                t_start = cutlass.max(
                    cutlass.Int32(0), batch_end - warp_last_id
                )
                for t in cutlass.range(t_start, batch_size):
                    current_position = batch_end - t
                    gaussian_id = batch_ids[t]
                    valid = (
                        inside
                        and gaussian_id >= 0
                        and current_position <= pixel_last_id
                    )
                    visibility = cutlass.Float32(0.0)
                    alpha = cutlass.Float32(0.0)
                    dx = cutlass.Float32(0.0)
                    dy = cutlass.Float32(0.0)
                    conic_x = cutlass.Float32(0.0)
                    conic_xy = cutlass.Float32(0.0)
                    conic_y = cutlass.Float32(0.0)
                    if valid:
                        conic_x = batch_conics[t, 0]
                        conic_xy = batch_conics[t, 1]
                        conic_y = batch_conics[t, 2]
                        dx = px - batch_means[t, 0]
                        dy = py - batch_means[t, 1]
                        sigma = 0.5 * (
                            conic_x * dx * dx + conic_y * dy * dy
                        ) + conic_xy * dx * dy
                        valid = cute.math.isfinite(sigma) and sigma >= 0.0
                        if valid:
                            visibility = cute.math.exp(-sigma, fastmath=True)
                            raw_alpha = batch_opacities[t] * visibility
                            alpha = cutlass.min(_MAX_ALPHA, raw_alpha)
                            if cute.math.isnan(raw_alpha) or (
                                cute.math.isinf(raw_alpha) and raw_alpha < 0.0
                            ):
                                alpha = cutlass.Float32(0.0)
                            valid = alpha >= alpha_threshold

                    valid_mask = cute.arch.vote_ballot_sync(valid)
                    if valid_mask != 0:
                        color_local = cute.make_rmem_tensor(
                            cute.make_layout(channels), cutlass.Float32
                        )
                        for channel in cutlass.range_constexpr(channels):
                            color_local[channel] = 0.0
                        means_x_local = cutlass.Float32(0.0)
                        means_y_local = cutlass.Float32(0.0)
                        conic_x_local = cutlass.Float32(0.0)
                        conic_xy_local = cutlass.Float32(0.0)
                        conic_y_local = cutlass.Float32(0.0)
                        opacity_local = cutlass.Float32(0.0)
                        if valid:
                            one_minus_alpha = 1.0 - alpha
                            transmittance_before = cute.math.div(
                                transmittance,
                                cutlass.max(
                                    _MIN_ONE_MINUS_ALPHA, one_minus_alpha
                                ),
                            )
                            weight = alpha * transmittance_before
                            weight_cotangent = output_alpha_gradient
                            for channel in cutlass.range_constexpr(channels):
                                weight_cotangent = (
                                    weight_cotangent
                                    + render_gradient[channel]
                                    * batch_colors[t, channel]
                                )
                                color_local[channel] = (
                                    render_gradient[channel] * weight
                                )
                            alpha_chain_cotangent = (
                                weight_cotangent * transmittance_before
                                - cute.math.div(
                                    trailing_cotangent, one_minus_alpha
                                )
                            )
                            raw_alpha = batch_opacities[t] * visibility
                            clamp_cotangent = cutlass.Float32(0.0)
                            if raw_alpha < _MAX_ALPHA:
                                clamp_cotangent = 1.0
                            elif raw_alpha == _MAX_ALPHA:
                                clamp_cotangent = 0.5
                            raw_alpha_cotangent = (
                                alpha_chain_cotangent * clamp_cotangent
                            )
                            if clamp_cotangent != 0.0:
                                opacity_local = raw_alpha_cotangent * visibility
                                sigma_cotangent = -raw_alpha_cotangent * raw_alpha
                                means_x_local = -sigma_cotangent * (
                                    conic_x * dx + conic_xy * dy
                                )
                                means_y_local = -sigma_cotangent * (
                                    conic_y * dy + conic_xy * dx
                                )
                                conic_x_local = (
                                    sigma_cotangent * 0.5 * dx * dx
                                )
                                conic_xy_local = sigma_cotangent * dx * dy
                                conic_y_local = (
                                    sigma_cotangent * 0.5 * dy * dy
                                )
                            trailing_cotangent = (
                                trailing_cotangent + weight_cotangent * weight
                            )
                            transmittance = transmittance_before

                        for offset in (16, 8, 4, 2, 1):
                            for channel in cutlass.range_constexpr(channels):
                                color_local[channel] = (
                                    color_local[channel]
                                    + cute.arch.shuffle_sync_down(
                                        color_local[channel], offset
                                    )
                                )
                            means_x_local = (
                                means_x_local
                                + cute.arch.shuffle_sync_down(
                                    means_x_local, offset
                                )
                            )
                            means_y_local = (
                                means_y_local
                                + cute.arch.shuffle_sync_down(
                                    means_y_local, offset
                                )
                            )
                            conic_x_local = (
                                conic_x_local
                                + cute.arch.shuffle_sync_down(
                                    conic_x_local, offset
                                )
                            )
                            conic_xy_local = (
                                conic_xy_local
                                + cute.arch.shuffle_sync_down(
                                    conic_xy_local, offset
                                )
                            )
                            conic_y_local = (
                                conic_y_local
                                + cute.arch.shuffle_sync_down(
                                    conic_y_local, offset
                                )
                            )
                            opacity_local = (
                                opacity_local
                                + cute.arch.shuffle_sync_down(
                                    opacity_local, offset
                                )
                            )
                        if (thread & 31) == 0:
                            gradient_base = gaussian_id * (6 + channels)
                            cute.arch.atomic_add(
                                packed_gradient.iterator + gradient_base,
                                means_x_local,
                                sem="relaxed",
                                scope="gpu",
                            )
                            cute.arch.atomic_add(
                                packed_gradient.iterator + gradient_base + 1,
                                means_y_local,
                                sem="relaxed",
                                scope="gpu",
                            )
                            cute.arch.atomic_add(
                                packed_gradient.iterator + gradient_base + 2,
                                conic_x_local,
                                sem="relaxed",
                                scope="gpu",
                            )
                            cute.arch.atomic_add(
                                packed_gradient.iterator + gradient_base + 3,
                                conic_xy_local,
                                sem="relaxed",
                                scope="gpu",
                            )
                            cute.arch.atomic_add(
                                packed_gradient.iterator + gradient_base + 4,
                                conic_y_local,
                                sem="relaxed",
                                scope="gpu",
                            )
                            for channel in cutlass.range_constexpr(channels):
                                cute.arch.atomic_add(
                                    packed_gradient.iterator
                                    + gradient_base
                                    + 5
                                    + channel,
                                    color_local[channel],
                                    sem="relaxed",
                                    scope="gpu",
                                )
                            cute.arch.atomic_add(
                                packed_gradient.iterator
                                + gradient_base
                                + 5
                                + channels,
                                opacity_local,
                                sem="relaxed",
                                scope="gpu",
                            )
                cute.arch.sync_threads()


@cute.jit
def _launch_compositor_forward(
    stream: cuda.CUstream,
    means: cute.Tensor,
    conics: cute.Tensor,
    colors: cute.Tensor,
    opacities: cute.Tensor,
    offsets: cute.Tensor,
    ids: cute.Tensor,
    valid_count: cute.Tensor,
    foreground: cute.Tensor,
    alpha: cute.Tensor,
    accepted_final_transmittance: cute.Tensor,
    last_ids: cute.Tensor,
    tile_overflow: cute.Tensor,
    *,
    gaussian_count: int,
    input_capacity: int,
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_count: int,
    per_tile_bound: int,
    channels: cutlass.Constexpr[int],
    alpha_threshold: float,
    transmittance_threshold: float,
):
    _compositor_forward_kernel(
        means,
        conics,
        colors,
        opacities,
        offsets,
        ids,
        valid_count,
        foreground,
        alpha,
        accepted_final_transmittance,
        last_ids,
        tile_overflow,
        gaussian_count,
        input_capacity,
        image_width,
        image_height,
        tile_width,
        tile_count,
        per_tile_bound,
        channels,
        alpha_threshold,
        transmittance_threshold,
    ).launch(
        grid=[tile_count, 1, 1],
        block=[_TILE_PIXELS, 1, 1],
        stream=stream,
    )


@cute.jit
def _launch_compositor_backward(
    stream: cuda.CUstream,
    means: cute.Tensor,
    conics: cute.Tensor,
    colors: cute.Tensor,
    opacities: cute.Tensor,
    offsets: cute.Tensor,
    ids: cute.Tensor,
    valid_count: cute.Tensor,
    accepted_final_transmittance: cute.Tensor,
    last_ids: cute.Tensor,
    render_cotangent: cute.Tensor,
    alpha_cotangent: cute.Tensor,
    packed_gradient: cute.Tensor,
    *,
    gaussian_count: cutlass.Constexpr[int],
    input_capacity: int,
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_count: int,
    per_tile_bound: int,
    channels: cutlass.Constexpr[int],
    alpha_threshold: float,
    transmittance_threshold: float,
):
    gradient_size = gaussian_count * (6 + channels)
    blocks = (gradient_size + _TILE_PIXELS - 1) // _TILE_PIXELS
    _clear_float_kernel(packed_gradient, gradient_size).launch(
        grid=[blocks, 1, 1],
        block=[_TILE_PIXELS, 1, 1],
        stream=stream,
    )
    _compositor_backward_kernel(
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
        packed_gradient,
        gaussian_count,
        input_capacity,
        image_width,
        image_height,
        tile_width,
        tile_count,
        per_tile_bound,
        channels,
        alpha_threshold,
        transmittance_threshold,
    ).launch(
        grid=[tile_count, 1, 1],
        block=[_TILE_PIXELS, 1, 1],
        stream=stream,
    )


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
    call = cjax.cutlass_call(
        _launch_compositor_forward,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(
                (image_height, image_width, channels), jnp.float32
            ),
            jax.ShapeDtypeStruct((image_height, image_width), jnp.float32),
            jax.ShapeDtypeStruct((image_height, image_width), jnp.float32),
            jax.ShapeDtypeStruct((image_height, image_width), jnp.int32),
            jax.ShapeDtypeStruct((tile_count,), jnp.uint8),
        ),
        use_static_tensors=True,
        gaussian_count=gaussian_count,
        input_capacity=input_capacity,
        image_width=image_width,
        image_height=image_height,
        tile_width=tile_width,
        tile_count=tile_count,
        per_tile_bound=per_tile_bound,
        channels=channels,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    foreground, alpha, accepted, last_ids, tile_overflow = call(
        means2d,
        conics,
        colors,
        opacities,
        offsets.reshape(-1),
        flatten_ids,
        valid_count.reshape((1,)),
    )
    return foreground, alpha, accepted, last_ids, tile_overflow.astype(jnp.bool_)


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
    transmittance_threshold: float,
):
    tile_height, tile_width = offsets.shape
    tile_count = tile_height * tile_width
    gaussian_count = means2d.shape[0]
    input_capacity = flatten_ids.shape[0]
    channels = colors.shape[-1]
    gradient_width = 6 + channels
    call = cjax.cutlass_call(
        _launch_compositor_backward,
        output_shape_dtype=jax.ShapeDtypeStruct(
            (gaussian_count, gradient_width), jnp.float32
        ),
        use_static_tensors=True,
        gaussian_count=gaussian_count,
        input_capacity=input_capacity,
        image_width=image_width,
        image_height=image_height,
        tile_width=tile_width,
        tile_count=tile_count,
        per_tile_bound=per_tile_bound,
        channels=channels,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    packed = call(
        means2d,
        conics,
        colors,
        opacities,
        offsets.reshape(-1),
        flatten_ids,
        valid_count.reshape((1,)),
        accepted_final_transmittance,
        last_ids,
        rendered_cotangent,
        alpha_cotangent,
    )
    return (
        packed[:, :2],
        packed[:, 2:5],
        packed[:, 5 : 5 + channels],
        packed[:, 5 + channels],
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
    (
        foreground,
        alpha,
        accepted_final_transmittance,
        last_ids,
        tile_overflow,
    ) = _run_forward(
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
        accepted_final_transmittance,
        last_ids,
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
    (
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        accepted_final_transmittance,
        last_ids,
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
        accepted_final_transmittance,
        last_ids,
        rendered_cotangent,
        alpha_cotangent[..., 0],
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    return (*gradients, None, None, None)


getattr(_composite_foreground, "defvjp")(
    _composite_foreground_fwd, _composite_foreground_bwd
)


def rasterize_to_pixels_cute(
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
    """Composite one camera using CuTe DSL."""

    image_width = _static_int("image_width", image_width, minimum=1)
    image_height = _static_int("image_height", image_height, minimum=1)
    tile_size = _static_int("tile_size", tile_size, minimum=1)
    if tile_size != _TILE_SIZE:
        raise ValueError("the CuTe compositor currently requires tile_size=16")
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
            "the CuTe compositor requires non-empty fixed-capacity inputs"
        )
    if means2d.shape != (gaussian_count, 2):
        raise ValueError("means2d must have shape [N, 2]")
    if conics.shape != (gaussian_count, 3):
        raise ValueError("conics must have shape [N, 3]")
    if colors.shape != (gaussian_count, channels):
        raise ValueError("colors must have shape [N, channels]")
    if channels not in _SUPPORTED_CHANNELS:
        raise ValueError(
            "the CuTe compositor supports channel counts "
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
        value.dtype != jnp.float32
        for value in (means2d, conics, colors, opacities)
    ):
        raise TypeError("the CuTe compositor requires float32 inputs")
    if offsets.dtype != jnp.int32 or flatten_ids.dtype != jnp.int32:
        raise TypeError("the CuTe compositor requires int32 metadata")

    tile_height, tile_width = offsets.shape
    if (
        tile_width * tile_size < image_width
        or tile_height * tile_size < image_height
    ):
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
            raise TypeError("the CuTe compositor requires float32 inputs")

    _cute_device()
    per_tile_bound = min(gaussian_count, input_capacity)
    if max_candidates_per_tile is not None:
        per_tile_bound = min(per_tile_bound, max_candidates_per_tile)
    per_tile_bound = (
        math.ceil(per_tile_bound / max_gaussians_per_tile)
        * max_gaussians_per_tile
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
    return rendered, alphas, {
        "tile_overflow": tile_overflow,
        "overflow": jnp.asarray(overflow, jnp.bool_) | jnp.any(tile_overflow),
    }


__all__ = ["rasterize_to_pixels_cute"]
