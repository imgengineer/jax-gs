"""Working-set estimates, and the budget checks that refuse a run before it allocates."""

from __future__ import annotations

from collections.abc import Sequence
import sys

# Tests drive the trainer by patching seams on the package, for example
# monkeypatch.setattr(jax_gs.training, "rasterization", fake). Calls resolve
# through the package namespace at run time so those seams keep working now
# that the implementation lives in submodules.
_training = sys.modules[__package__]

from dataclasses import replace
from typing import NamedTuple
import math

import jax
import numpy as np

from ..config import RasterizationConfig, TrainConfig
from ..rasterization import _automatic_intersection_capacity
from ._scene import _training_render_size
from ._step import TrainingSafetyState, _PendingTrainStep
from .appearance import (
    APPEARANCE_FEATURE_DIM,
)


def _intersection_bucket_capacity(
    required: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Return the smallest configured power-of-two bucket covering required."""

    required = int(required)
    minimum = int(minimum)
    maximum = int(maximum)
    if required < 0:
        raise ValueError("required intersection capacity cannot be negative")
    if minimum <= 0 or minimum & (minimum - 1):
        raise ValueError("minimum intersection bucket must be a power of two")
    if maximum <= 0:
        raise ValueError("maximum intersection capacity must be positive")
    if required > maximum:
        raise RuntimeError(
            f"required intersections {required} exceed configured maximum {maximum}"
        )
    bucket = min(minimum, maximum)
    while bucket < required:
        bucket = min(bucket * 2, maximum)
    return bucket


def _training_intersection_limit(
    config: TrainConfig,
    physical_capacity: int,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
) -> int:
    height, width = _training_render_size(
        config, image_height=image_height, image_width=image_width
    )
    tile_width = math.ceil(width / config.rasterizer.tile_size)
    tile_height = math.ceil(height / config.rasterizer.tile_size)
    tile_count = tile_width * tile_height
    return _automatic_intersection_capacity(
        physical_capacity, tile_count, config.rasterizer
    )


def _training_config_with_intersection_capacity(
    config: TrainConfig, capacity: int
) -> TrainConfig:
    return replace(
        config,
        rasterizer=replace(config.rasterizer, max_intersections=int(capacity)),
    )


def _training_config_with_candidate_bound(
    config: TrainConfig, bound: int
) -> TrainConfig:
    return replace(
        config,
        rasterizer=replace(
            config.rasterizer, max_candidates_per_tile=int(bound)
        ),
    )


def _candidate_bound_for_occupancy(
    required: int, max_gaussians_per_tile: int
) -> int:
    """Return the bound covering ``required`` candidates in a tile.

    The compositor's loop length is ``ceil(bound / max_gaussians_per_tile)``,
    so only that quotient is observable and the bound is rounded to a whole
    number of chunks. Rounding that count to a power of two costs at most one
    surplus chunk but makes the sequence of bounds a doubling one, which is
    what keeps a scene that densifies from recompiling at every step: the
    occupancy has to double before the bound moves again.
    """

    required = max(int(required), 1)
    per_tile = int(max_gaussians_per_tile)
    chunks = math.ceil(required / per_tile)
    return (1 << (chunks - 1).bit_length()) * per_tile


def _mcmc_required_capacity(active_count: int, config: TrainConfig) -> int:
    """Return the scheduled MCMC population target from the active count."""

    active_count = int(active_count)
    target_count = min(
        config.strategy.cap_max, (active_count * 105) // 100
    )
    births = min(
        max(target_count - active_count, 0),
        config.strategy.max_new_per_refine,
    )
    return active_count + births


def _raise_training_overflow(tile_count: np.ndarray, intersection: np.ndarray) -> None:
    raise RuntimeError(
        "rasterization overflow would produce truncated gradients "
        f"(tiles={int(tile_count)}, intersection={bool(intersection)}); increase "
        "max_intersections or max_gaussians_per_tile"
    )


def _training_overflow_status(
    safety_state: TrainingSafetyState,
) -> tuple[int, bool]:
    max_overflow_tiles, intersection_overflow_seen = jax.device_get(
        (
            safety_state.max_overflow_tiles[...],
            safety_state.intersection_overflow_seen[...],
        )
    )
    return (
        int(np.max(np.asarray(max_overflow_tiles))),
        bool(np.any(np.asarray(intersection_overflow_seen))),
    )


class _PendingOverflow(NamedTuple):
    """The earliest uncommitted step that overflowed, and what it needs."""

    replay_start: int
    kind: str
    required: int


def _pending_overflow_suffix(
    pending: list[_PendingTrainStep],
) -> _PendingOverflow | None:
    values = jax.device_get(
        tuple(
            (
                record.metrics["overflow_tiles"],
                record.metrics["intersection_overflow"],
                record.metrics["intersection_required_count"],
                record.metrics["busiest_tile_candidates"],
            )
            for record in pending
        )
    )
    for index, (overflow_tiles, intersection, required, busiest) in enumerate(
        values
    ):
        # An overflowing intersection buffer is resolved first even when the
        # same step also truncated a tile: the candidate counts a tile bound
        # would be grown from are themselves drawn from that buffer, so they
        # under-report until it is large enough to hold the frame.
        if bool(np.any(np.asarray(intersection))):
            return _PendingOverflow(
                index, "intersection", int(np.max(np.asarray(required)))
            )
        if int(np.max(np.asarray(overflow_tiles))) > 0:
            return _PendingOverflow(
                index, "candidates", int(np.max(np.asarray(busiest)))
            )
    # The sticky flag says a step overflowed and no step admits to it. There
    # is nothing to grow from, so the caller falls back to refusing the run.
    return None


def estimate_rasterization_memory_bytes(
    capacity: int,
    width: int,
    height: int,
    rasterizer: RasterizationConfig,
    *,
    channels: int = 3,
) -> int:
    """Conservative forward-only rasterization workspace estimate."""

    tile_width = math.ceil(width / rasterizer.tile_size)
    tile_height = math.ceil(height / rasterizer.tile_size)
    tile_count = tile_width * tile_height
    projection = capacity * 64
    if rasterizer.compositor_backend == "pallas":
        pixel_count = math.ceil(rasterizer.tile_size**2 / 128) * 128
        # Render, alpha, and reverse-mode transmittance tile buffers. Count the
        # residual conservatively even when no gradient is requested.
        compositing = tile_count * pixel_count * (channels + 2) * 4
    else:
        compositing = (
            min(rasterizer.tile_batch_size, tile_count)
            * rasterizer.max_gaussians_per_tile
            * rasterizer.tile_size**2
            * (32 + channels * 4)
        )
    intersections = 0
    tile_scan = 0
    if rasterizer.backend != "reference":
        intersection_capacity = _automatic_intersection_capacity(
            capacity, tile_count, rasterizer
        )
        intersections = intersection_capacity * 96
    else:
        tile_scan = min(rasterizer.tile_batch_size, tile_count) * capacity * 16
    outputs = width * height * (channels + 1) * 4 * 3
    return int(
        projection
        + compositing
        + intersections
        + tile_scan
        + outputs
        + 256 * 2**20
    )


def _parameter_bytes(config: TrainConfig, physical_capacity: int) -> int:
    color_floats = (
        APPEARANCE_FEATURE_DIM + 3
        if config.app_opt
        else (config.model.sh_degree + 1) ** 2 * 3
    )
    floats_per_gaussian = 3 + 3 + 4 + 1 + color_floats
    return physical_capacity * floats_per_gaussian * 4


def _training_state_bytes(config: TrainConfig, physical_capacity: int) -> int:
    parameter_bytes = _parameter_bytes(config, physical_capacity)
    strategy_bytes = physical_capacity * (3 * 4 + 1)
    # Model parameters, two Adam moments, active mask, and strategy statistics.
    return parameter_bytes * 3 + strategy_bytes


def _appearance_mlp_workspace_bytes(
    config: TrainConfig,
    gaussian_capacity: int,
    camera_count: int,
    *,
    reverse_mode: bool,
) -> int:
    """Conservatively count materialized appearance tensors."""

    if not config.app_opt:
        return 0
    basis_count = (config.model.sh_degree + 1) ** 2
    mlp_input_width = (
        config.app_embed_dim + APPEARANCE_FEATURE_DIM + basis_count
    )
    # Directions and bases, concatenated input, two 64-wide hidden layers,
    # correction/corrected logits/direct RGB. Training reserves the same space
    # again for reverse-mode residuals and cotangents.
    floats_per_camera_gaussian = (
        3 + basis_count + mlp_input_width + 2 * 64 + 3 * 3
    )
    return (
        gaussian_capacity
        * camera_count
        * floats_per_camera_gaussian
        * 4
        * (2 if reverse_mode else 1)
    )


def estimate_training_memory_bytes(
    config: TrainConfig,
    *,
    physical_capacity: int | None = None,
    render_capacity: int | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
) -> int:
    """Estimate one device's peak for local state and rendered Gaussians."""

    if physical_capacity is None:
        physical_capacity = config.model.bucket_capacity()
    physical_capacity = int(physical_capacity)
    if not 0 < physical_capacity <= config.model.capacity:
        raise ValueError(
            "physical_capacity must be positive and not exceed the logical maximum"
        )
    if render_capacity is None:
        render_capacity = physical_capacity
    render_capacity = int(render_capacity)
    if render_capacity <= 0:
        raise ValueError("render_capacity must be positive")
    render_height, render_width = _training_render_size(
        config, image_height=image_height, image_width=image_width
    )

    parameter_bytes = _parameter_bytes(config, physical_capacity)
    # Parameters, gradients, two Adam moments, updates, and compiler temporaries.
    model_working_set = parameter_bytes * 6
    # Dense projection still visits the complete physical bucket before the
    # bounded visible set is packed. Include forward outputs, reverse-mode
    # residuals, and covariance temporaries explicitly for large buckets.
    projection_working_set = (
        render_capacity * 192 * config.data.batch_size
    )
    strategy_bytes = physical_capacity * (3 * 4 + 1)
    tile_pixels = config.rasterizer.tile_size**2
    tile_width = math.ceil(render_width / config.rasterizer.tile_size)
    tile_height = math.ceil(render_height / config.rasterizer.tile_size)
    render_tiles = tile_width * tile_height
    if config.rasterizer.compositor_backend == "pallas":
        pixel_count = math.ceil(tile_pixels / 128) * 128
        raster_workspace = render_tiles * pixel_count * 5 * 4
    else:
        raster_workspace = (
            config.rasterizer.tile_batch_size
            * config.rasterizer.max_gaussians_per_tile
            * tile_pixels
            * 64
        )
    intersection_workspace = 0
    intersection_capacity = 0
    if config.rasterizer.backend != "reference":
        intersection_capacity = config.rasterizer.max_intersections
        if intersection_capacity is None:
            intersection_capacity = max(
                1,
                min(
                    render_tiles
                    * (config.rasterizer.max_gaussians_per_tile + 1),
                    max(
                        config.rasterizer.max_gaussians_per_tile + 1,
                        physical_capacity * 8,
                    ),
                ),
            )
        # Padded ids plus conservative temporary storage for lexicographic sort.
        intersection_workspace = intersection_capacity * 96
    if config.rasterizer.compositor_backend == "pallas":
        # The backward kernel emits race-free per-intersection gradients for
        # mean (2), conic (3), RGB (3), and opacity (1) before owner scatter.
        raster_workspace += intersection_capacity * 9 * 4
    ut_workspace = 0
    if config.with_ut or config.with_eval3d:
        ut_workspace = (
            min(render_capacity, config.rasterizer.ut_chunk_size)
            * 7
            * 3
            * 4
            * 12
            * config.data.batch_size
        )
    # Distributed training passes global render_capacity while camera_count
    # remains the rank-local batch size.
    appearance_workspace = _appearance_mlp_workspace_bytes(
        config,
        render_capacity,
        config.data.batch_size,
        reverse_mode=True,
    )
    return int(
        model_working_set
        + projection_working_set
        + strategy_bytes
        + raster_workspace
        + intersection_workspace
        + ut_workspace
        + appearance_workspace
        + 512 * 2**20
    )


def estimate_bucket_transition_memory_bytes(
    config: TrainConfig,
    old_capacity: int,
    new_capacity: int,
    *,
    render_capacity: int | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
) -> int:
    """Estimate the old/new coexistence peak while growing a storage bucket."""

    old_capacity = int(old_capacity)
    new_capacity = int(new_capacity)
    if not 0 < old_capacity < new_capacity <= config.model.capacity:
        raise ValueError("invalid bucket transition")
    target_peak = estimate_training_memory_bytes(
        config,
        physical_capacity=new_capacity,
        render_capacity=render_capacity,
        image_height=image_height,
        image_width=image_width,
    )
    # Migration now pads the existing optimizer and strategy state directly;
    # it does not initialize a second all-zero Adam state. Runtime projection
    # and gradient workspaces are not live during the host-controlled resize,
    # so guard the larger of the steady target and old/new persistent overlap.
    migration_peak = (
        _training_state_bytes(config, old_capacity)
        + _training_state_bytes(config, new_capacity)
        + 256 * 2**20
    )
    return int(max(target_peak, migration_peak))


def _check_memory_budget(
    config: TrainConfig,
    *,
    physical_capacity: int,
    render_capacity: int | None = None,
    devices: Sequence[jax.Device] | None = None,
    label: str = "training",
    image_height: int | None = None,
    image_width: int | None = None,
) -> int:
    estimate = estimate_training_memory_bytes(
        config,
        physical_capacity=physical_capacity,
        render_capacity=render_capacity,
        image_height=image_height,
        image_width=image_width,
    )
    samples = _device_memory_samples(devices)
    print(
        f"estimated_{label}_peak_memory={estimate / 2**30:.2f}GiB "
        f"storage_capacity={physical_capacity} "
        f"devices={','.join(str(device) for device, _, _ in samples)}",
        flush=True,
    )
    unsafe = next(
        (
            (device, limit)
            for device, _, limit in samples
            if limit and estimate > int(limit * 0.70)
        ),
        None,
    )
    if unsafe is not None:
        device, limit = unsafe
        raise MemoryError(
            "estimated training working set exceeds 70% of JAX's device memory "
            f"limit ({estimate / 2**30:.2f} GiB estimated, "
            f"{limit / 2**30:.2f} GiB limit on selected device {device}). "
            "Reduce "
            "bucket_min_capacity, "
            "tile_batch_size, max_gaussians_per_tile, patch_size, or SH degree."
        )
    return estimate


def _check_bucket_transition_memory_budget(
    config: TrainConfig,
    old_capacity: int,
    new_capacity: int,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
) -> int:
    estimate = estimate_bucket_transition_memory_bytes(
        config,
        old_capacity,
        new_capacity,
        image_height=image_height,
        image_width=image_width,
    )
    bytes_in_use, limit = _training._device_memory_usage()
    # The optimized resize creates only the new persistent model, Adam moments,
    # mask, and strategy arrays. Include them on top of observed live buffers;
    # old buffers remain live until the replacement state is fully blocked.
    resize_allocation = (
        _training_state_bytes(config, new_capacity) + 256 * 2**20
    )
    projected = max(estimate, bytes_in_use + resize_allocation)
    print(
        f"estimated_capacity_transition_peak={projected / 2**30:.2f}GiB "
        f"capacity_growth={old_capacity}->{new_capacity}",
        flush=True,
    )
    if limit and projected > int(limit * 0.70):
        raise MemoryError(
            "bucket growth was stopped before allocation because old and new "
            "training states would exceed 70% of JAX's device-memory limit "
            f"({projected / 2**30:.2f} GiB projected, "
            f"{limit / 2**30:.2f} GiB limit). Lower the logical capacity, "
            "bucket minimum, SH degree, or rasterizer workspace limits."
        )
    return projected


def _check_distributed_bucket_transition_memory_budget(
    config: TrainConfig,
    world_size: int,
    old_capacity: int,
    new_capacity: int,
    *,
    devices: Sequence[jax.Device] | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
) -> int:
    """Preflight mapped per-device or host-stacked world resize memory."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    per_shard = estimate_bucket_transition_memory_bytes(
        config,
        old_capacity,
        new_capacity,
        render_capacity=world_size * new_capacity,
        image_height=image_height,
        image_width=image_width,
    )
    if devices is None:
        estimate = world_size * per_shard
        resize_allocation = (
            world_size * _training_state_bytes(config, new_capacity)
            + 256 * 2**20
        )
    else:
        estimate = per_shard
        resize_allocation = (
            _training_state_bytes(config, new_capacity) + 256 * 2**20
        )
    samples = _device_memory_samples(devices)
    projections = [
        (device, max(estimate, bytes_in_use + resize_allocation), limit)
        for device, bytes_in_use, limit in samples
    ]
    projected = max(value for _, value, _ in projections)
    print(
        f"estimated_distributed_transition_peak={projected / 2**30:.2f}GiB "
        f"world_size={world_size} "
        f"capacity_growth={old_capacity}->{new_capacity}",
        flush=True,
    )
    unsafe = next(
        (
            (device, value, limit)
            for device, value, limit in projections
            if limit and value > int(limit * 0.70)
        ),
        None,
    )
    if unsafe is not None:
        device, device_projected, limit = unsafe
        raise MemoryError(
            "distributed bucket growth was stopped before allocation because "
            "the world's old and new training states would exceed 70% of "
            f"JAX's device-memory limit ({device_projected / 2**30:.2f} GiB "
            f"projected, {limit / 2**30:.2f} GiB limit on selected device "
            f"{device}). Lower "
            "the logical capacity, bucket minimum, SH degree, or the world "
            "size per device."
        )
    return projected


def _device_memory_usage(
    device: jax.Device | None = None,
) -> tuple[int, int]:
    if device is None:
        device = jax.devices()[0]
    stats = device.memory_stats() or {}
    return int(stats.get("bytes_in_use", 0) or 0), int(
        stats.get("bytes_limit", 0) or 0
    )


def _device_memory_samples(
    devices: Sequence[jax.Device] | None,
) -> tuple[tuple[jax.Device, int, int], ...]:
    if devices is None:
        device = jax.devices()[0]
        bytes_in_use, limit = _training._device_memory_usage()
        return ((device, bytes_in_use, limit),)
    selected = tuple(devices)
    if not selected:
        raise ValueError("devices must not be empty")
    return tuple(
        (device, *_training._device_memory_usage(device))
        for device in selected
    )


def _check_evaluation_memory_budget(
    config: TrainConfig,
    *,
    physical_capacity: int,
    width: int,
    height: int,
    devices: Sequence[jax.Device] | None = None,
) -> int:
    """Check a full-resolution evaluation against live training allocations."""

    workspace = estimate_rasterization_memory_bytes(
        physical_capacity, width, height, config.rasterizer
    )
    workspace += _appearance_mlp_workspace_bytes(
        config, physical_capacity, 1, reverse_mode=False
    )
    samples = _device_memory_samples(devices)
    projections = [
        (device, bytes_in_use + workspace, limit)
        for device, bytes_in_use, limit in samples
    ]
    projected = max(value for _, value, _ in projections)
    print(
        f"estimated_evaluation_peak={projected / 2**30:.2f}GiB "
        f"storage_capacity={physical_capacity} resolution={width}x{height}",
        flush=True,
    )
    unsafe = next(
        (
            (device, value, limit)
            for device, value, limit in projections
            if limit and value > int(limit * 0.70)
        ),
        None,
    )
    if unsafe is not None:
        device, device_projected, limit = unsafe
        raise MemoryError(
            "full-resolution evaluation was stopped because live training "
            "state plus the render workspace would exceed 70% of JAX's "
            f"device-memory limit on selected device {device} "
            f"({device_projected / 2**30:.2f} GiB projected, "
            f"{limit / 2**30:.2f} GiB limit). Lower eval frequency, image "
            "resolution, max_intersections, max_gaussians_per_tile, or "
            "tile_batch_size."
        )
    return projected
