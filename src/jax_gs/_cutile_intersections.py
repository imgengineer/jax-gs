from __future__ import annotations

import importlib
import math
import operator
import os
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp

_COUNT_BLOCK_SIZE = 128
_EMIT_BLOCK_SIZE = 128
_TUNING_ENV = "JAX_GS_CUTILE_TUNING"
_TUNING_PROFILES = {
    "default": (128, 128, None, None),
    "small": (64, 64, None, None),
    "wide": (128, 256, None, None),
    "low_occupancy": (128, 128, None, 1),
}

__all__ = [
    "count_accutile_intersections_cutile",
    "emit_accutile_intersections_cutile",
]
_T = TypeVar("_T")


class _Constant(int, Generic[_T]):
    """Static-checking stand-in replaced by ``cuda.tile.Constant`` lazily."""


def _static_optional_int(
    name: str, value: str | None, *, minimum: int, maximum: int
) -> int | None:
    if value is None or value == "":
        return None
    try:
        result = operator.index(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _tuning_profile() -> tuple[int, int, int | None, int | None]:
    profile = os.environ.get(_TUNING_ENV, "default").strip().lower()
    try:
        count_block, emit_block, count_occupancy, emit_occupancy = (
            _TUNING_PROFILES[profile]
        )
    except KeyError as exc:
        raise ValueError(
            f"{_TUNING_ENV} must name one of "
            + ", ".join(sorted(_TUNING_PROFILES))
        ) from exc
    count_block = _static_optional_int(
        "JAX_GS_CUTILE_COUNT_BLOCK_SIZE",
        os.environ.get("JAX_GS_CUTILE_COUNT_BLOCK_SIZE"),
        minimum=32,
        maximum=1024,
    ) or count_block
    emit_block = _static_optional_int(
        "JAX_GS_CUTILE_EMIT_BLOCK_SIZE",
        os.environ.get("JAX_GS_CUTILE_EMIT_BLOCK_SIZE"),
        minimum=32,
        maximum=1024,
    ) or emit_block
    count_occupancy = _static_optional_int(
        "JAX_GS_CUTILE_COUNT_OCCUPANCY",
        os.environ.get("JAX_GS_CUTILE_COUNT_OCCUPANCY"),
        minimum=1,
        maximum=32,
    ) or count_occupancy
    emit_occupancy = _static_optional_int(
        "JAX_GS_CUTILE_EMIT_OCCUPANCY",
        os.environ.get("JAX_GS_CUTILE_EMIT_OCCUPANCY"),
        minimum=1,
        maximum=32,
    ) or emit_occupancy
    return count_block, emit_block, count_occupancy, emit_occupancy


def available_tuning_profiles() -> tuple[str, ...]:
    """Return semantic-equivalent cuTile variants for external autotuning."""

    return tuple(sorted(_TUNING_PROFILES))


def _cutile():
    try:
        ct = importlib.import_module("cuda.tile")
        ctj = importlib.import_module("cuda.tile.jax")
    except ImportError as exc:
        raise RuntimeError(
            "the cuTile AccuTile backend requires the optional cuda-tile "
            "package; install cuda-tile[tileiras]>=1.5.0 or use "
            "intersection_backend='jax'"
        ) from exc
    return ct, ctj


# Keep imports lazy for ordinary jax_gs import and CPU-only installs. Kernels need
# module-level decorated definitions, so build them once on first explicit use.
_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    ct, _ = _cutile()
    globals()["ct"] = ct
    globals()["_Constant"] = ct.Constant

    @ct.kernel
    def count_kernel(
        valid_ref, b_ref, disc_ref, t_ref, p_u_ref, p_v_ref,
        coefficient_ref, outer_min_ref, outer_max_ref, cross_min_ref,
        cross_max_ref, outer_bbox_min_ref, outer_bbox_max_ref,
        cross_bbox_min_ref, cross_bbox_max_ref, argmin_outer_ref,
        argmax_outer_ref, counts_ref, tile_size: _Constant[int],
        outer_steps: _Constant[int], block_size: _Constant[int],
    ):
        block_id = ct.bid(0)
        index = (block_id,)
        shape = (block_size,)
        valid = ct.load(
            valid_ref, index, shape, padding_mode=ct.PaddingMode.ZERO
        ) != 0
        b = ct.load(b_ref, index, shape, padding_mode=ct.PaddingMode.ZERO)
        disc = ct.load(disc_ref, index, shape, padding_mode=ct.PaddingMode.ZERO)
        t = ct.load(t_ref, index, shape, padding_mode=ct.PaddingMode.ZERO)
        p_u = ct.load(p_u_ref, index, shape, padding_mode=ct.PaddingMode.ZERO)
        p_v = ct.load(p_v_ref, index, shape, padding_mode=ct.PaddingMode.ZERO)
        coefficient = ct.load(
            coefficient_ref, index, shape, padding_mode=ct.PaddingMode.ZERO
        )
        outer_min = ct.load(
            outer_min_ref, index, shape, padding_mode=ct.PaddingMode.ZERO
        )
        outer_max = ct.load(
            outer_max_ref, index, shape, padding_mode=ct.PaddingMode.ZERO
        )
        cross_min = ct.load(
            cross_min_ref, index, shape, padding_mode=ct.PaddingMode.ZERO
        )
        cross_max = ct.load(
            cross_max_ref, index, shape, padding_mode=ct.PaddingMode.ZERO
        )
        outer_bbox_min = ct.load(
            outer_bbox_min_ref,
            index,
            shape,
            padding_mode=ct.PaddingMode.ZERO,
        )
        outer_bbox_max = ct.load(
            outer_bbox_max_ref,
            index,
            shape,
            padding_mode=ct.PaddingMode.ZERO,
        )
        cross_bbox_min = ct.load(
            cross_bbox_min_ref,
            index,
            shape,
            padding_mode=ct.PaddingMode.ZERO,
        )
        cross_bbox_max = ct.load(
            cross_bbox_max_ref,
            index,
            shape,
            padding_mode=ct.PaddingMode.ZERO,
        )
        argmin_outer = ct.load(
            argmin_outer_ref, index, shape, padding_mode=ct.PaddingMode.ZERO
        )
        argmax_outer = ct.load(
            argmax_outer_ref, index, shape, padding_mode=ct.PaddingMode.ZERO
        )

        disc = ct.where(valid, disc, -1.0)
        t = ct.where(valid, t, 1.0)
        coefficient = ct.where(valid, coefficient, 1.0)
        block = tile_size
        min_line = ct.astype(outer_min, ct.float32) * block
        h = min_line - p_u
        radicand = ct.maximum(disc * h * h + t * coefficient, 0.0)
        root = ct.where(
            radicand > 0.0, radicand * ct.rsqrt(radicand), 0.0
        )
        line_min = (-b * h - root) / coefficient + p_v
        line_max = (-b * h + root) / coefficient + p_v
        intersects = outer_bbox_min <= min_line
        previous_min = ct.where(intersects, line_min, cross_bbox_max)
        previous_max = ct.where(intersects, line_max, cross_bbox_min)
        counts = ct.zeros(shape, ct.int32)

        for outer_offset in range(outer_steps):
            outer = outer_min + outer_offset
            active = valid & (outer < outer_max)
            min_line = ct.astype(outer, ct.float32) * block
            max_line = min_line + block
            h = max_line - p_u
            radicand = ct.maximum(disc * h * h + t * coefficient, 0.0)
            root = ct.where(
                radicand > 0.0, radicand * ct.rsqrt(radicand), 0.0
            )
            line_min = (-b * h - root) / coefficient + p_v
            line_max = (-b * h + root) / coefficient + p_v
            intersects = max_line <= outer_bbox_max
            current_min = ct.where(intersects, line_min, previous_min)
            current_max = ct.where(intersects, line_max, previous_max)
            ellipse_min = ct.where(
                (min_line <= argmin_outer) & (argmin_outer < max_line),
                cross_bbox_min,
                ct.minimum(previous_min, current_min),
            )
            ellipse_max = ct.where(
                (min_line <= argmax_outer) & (argmax_outer < max_line),
                cross_bbox_max,
                ct.maximum(previous_max, current_max),
            )
            min_v = ct.maximum(
                cross_min,
                ct.minimum(
                    cross_max, ct.astype(ellipse_min / block, ct.int32)
                ),
            )
            max_v = ct.minimum(
                cross_max,
                ct.maximum(
                    cross_min,
                    ct.astype(ellipse_max / block + 1.0, ct.int32),
                ),
            )
            counts = counts + ct.where(active, max_v - min_v, 0)
            previous_min = ct.where(active, current_min, previous_min)
            previous_max = ct.where(active, current_max, previous_max)
        ct.store(counts_ref, index, counts)

    @ct.kernel
    def emit_kernel(
        valid_ref, is_y_ref, b_ref, disc_ref, t_ref, p_u_ref, p_v_ref,
        coefficient_ref, outer_min_ref, outer_max_ref, cross_min_ref,
        cross_max_ref, outer_bbox_min_ref, outer_bbox_max_ref,
        cross_bbox_min_ref, cross_bbox_max_ref, argmin_outer_ref,
        argmax_outer_ref, cumulative_ref, valid_count_ref,
        gaussian_ids_ref, tile_ids_ref,
        tile_size: _Constant[int], tile_width: _Constant[int],
        outer_steps: _Constant[int], capacity: _Constant[int],
        gaussian_count: _Constant[int], search_steps: _Constant[int],
        block_size: _Constant[int],
    ):
        block_id = ct.bid(0)
        rank = ct.arange(
            block_size, dtype=ct.int32, start=block_id * block_size
        )
        shape = (block_size,)
        valid_count = ct.maximum(
            0,
            ct.minimum(
                ct.load(valid_count_ref, (0,), shape=()), capacity
            ),
        )
        output_valid = rank < valid_count
        if block_id * block_size >= valid_count:
            padding = ct.full(shape, -1, ct.int32)
            ct.store(gaussian_ids_ref, (block_id,), padding)
            ct.store(tile_ids_ref, (block_id,), padding)
            return

        low = ct.zeros(shape, ct.int32)
        high = ct.full(shape, gaussian_count, ct.int32)
        for _ in range(search_steps):
            middle = (low + high) // 2
            safe_middle = ct.minimum(middle, gaussian_count - 1)
            end = ct.gather(cumulative_ref, safe_middle)
            move_right = end <= rank
            low = ct.where(move_right, middle + 1, low)
            high = ct.where(move_right, high, middle)
        owner = ct.minimum(low, gaussian_count - 1)
        starts = ct.gather(
            cumulative_ref,
            owner - 1,
            mask=owner > 0,
            padding_value=0,
        )
        local_rank = rank - starts

        valid = (ct.gather(valid_ref, owner) != 0) & output_valid
        is_y = ct.gather(is_y_ref, owner) != 0
        b = ct.gather(b_ref, owner)
        disc = ct.gather(disc_ref, owner)
        t = ct.gather(t_ref, owner)
        p_u = ct.gather(p_u_ref, owner)
        p_v = ct.gather(p_v_ref, owner)
        coefficient = ct.gather(coefficient_ref, owner)
        outer_min = ct.gather(outer_min_ref, owner)
        outer_max = ct.gather(outer_max_ref, owner)
        cross_min = ct.gather(cross_min_ref, owner)
        cross_max = ct.gather(cross_max_ref, owner)
        outer_bbox_min = ct.gather(outer_bbox_min_ref, owner)
        outer_bbox_max = ct.gather(outer_bbox_max_ref, owner)
        cross_bbox_min = ct.gather(cross_bbox_min_ref, owner)
        cross_bbox_max = ct.gather(cross_bbox_max_ref, owner)
        argmin_outer = ct.gather(argmin_outer_ref, owner)
        argmax_outer = ct.gather(argmax_outer_ref, owner)

        disc = ct.where(valid, disc, -1.0)
        t = ct.where(valid, t, 1.0)
        coefficient = ct.where(valid, coefficient, 1.0)
        block = tile_size
        min_line = ct.astype(outer_min, ct.float32) * block
        h = min_line - p_u
        radicand = ct.maximum(disc * h * h + t * coefficient, 0.0)
        root = ct.where(
            radicand > 0.0, radicand * ct.rsqrt(radicand), 0.0
        )
        line_min = (-b * h - root) / coefficient + p_v
        line_max = (-b * h + root) / coefficient + p_v
        intersects = outer_bbox_min <= min_line
        previous_min = ct.where(intersects, line_min, cross_bbox_max)
        previous_max = ct.where(intersects, line_max, cross_bbox_min)
        emitted = ct.zeros(shape, ct.int32)
        selected_cross = ct.zeros(shape, ct.int32)
        selected_outer = outer_min
        found = ct.zeros(shape, ct.bool_)

        for outer_offset in range(outer_steps):
            outer = outer_min + outer_offset
            active = valid & (outer < outer_max)
            min_line = ct.astype(outer, ct.float32) * block
            max_line = min_line + block
            h = max_line - p_u
            radicand = ct.maximum(disc * h * h + t * coefficient, 0.0)
            root = ct.where(
                radicand > 0.0, radicand * ct.rsqrt(radicand), 0.0
            )
            line_min = (-b * h - root) / coefficient + p_v
            line_max = (-b * h + root) / coefficient + p_v
            intersects = max_line <= outer_bbox_max
            current_min = ct.where(intersects, line_min, previous_min)
            current_max = ct.where(intersects, line_max, previous_max)
            ellipse_min = ct.where(
                (min_line <= argmin_outer) & (argmin_outer < max_line),
                cross_bbox_min,
                ct.minimum(previous_min, current_min),
            )
            ellipse_max = ct.where(
                (min_line <= argmax_outer) & (argmax_outer < max_line),
                cross_bbox_max,
                ct.maximum(previous_max, current_max),
            )
            min_v = ct.maximum(
                cross_min,
                ct.minimum(
                    cross_max, ct.astype(ellipse_min / block, ct.int32)
                ),
            )
            max_v = ct.minimum(
                cross_max,
                ct.maximum(
                    cross_min,
                    ct.astype(ellipse_max / block + 1.0, ct.int32),
                ),
            )
            span = ct.where(active, max_v - min_v, 0)
            covered = (
                output_valid
                & (~found)
                & (local_rank >= emitted)
                & (local_rank < emitted + span)
            )
            selected_cross = ct.where(
                covered, min_v + local_rank - emitted, selected_cross
            )
            selected_outer = ct.where(covered, outer, selected_outer)
            found = found | covered
            emitted = emitted + span
            previous_min = ct.where(active, current_min, previous_min)
            previous_max = ct.where(active, current_max, previous_max)

        tile_id = ct.where(
            is_y,
            selected_outer * tile_width + selected_cross,
            selected_cross * tile_width + selected_outer,
        )
        final_valid = output_valid & found
        ct.store(
            gaussian_ids_ref,
            (block_id,),
            ct.where(final_valid, owner, -1),
        )
        ct.store(
            tile_ids_ref,
            (block_id,),
            ct.where(final_valid, tile_id, -1),
        )

    _, _, count_occupancy, emit_occupancy = _tuning_profile()
    if count_occupancy is not None:
        count_kernel = count_kernel.replace_hints(occupancy=count_occupancy)
    if emit_occupancy is not None:
        emit_kernel = emit_kernel.replace_hints(occupancy=emit_occupancy)
    _KERNELS = count_kernel, emit_kernel
    return _KERNELS


def _require_cuda_tile_device() -> None:
    device = jax.devices()[0]
    if device.platform != "gpu" or "cuda" not in str(device).lower():
        raise RuntimeError(
            "the cuTile AccuTile backend requires an NVIDIA CUDA device; "
            "use intersection_backend='jax' on this device"
        )


def _state_arguments(state: Any) -> tuple[jax.Array, ...]:
    return (
        state.valid.astype(jnp.uint8),
        state.b,
        state.disc,
        state.t,
        state.p_u,
        state.p_v,
        state.coefficient,
        state.outer_min,
        state.outer_max,
        state.cross_min,
        state.cross_max,
        state.outer_bbox_min,
        state.outer_bbox_max,
        state.cross_bbox_min,
        state.cross_bbox_max,
        state.argmin_outer,
        state.argmax_outer,
    )


def count_accutile_intersections_cutile(
    state: Any,
    *,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> jax.Array:
    """Count AccuTile pairs with an opt-in NVIDIA cuTile kernel."""

    gaussian_count = state.valid.shape[0]
    if gaussian_count == 0:
        return jnp.zeros((0,), dtype=jnp.int32)
    _require_cuda_tile_device()
    ct, ctj = _cutile()
    count_kernel, _ = _kernels()
    count_block_size, _, _, _ = _tuning_profile()
    output = ctj.OutputPlaceholder((gaussian_count,), jnp.int32)
    return ctj.cutile_call(
        (ct.cdiv(gaussian_count, count_block_size),),
        count_kernel,
        (
            *_state_arguments(state),
            output,
            tile_size,
            max(tile_width, tile_height),
            count_block_size,
        ),
    )


def emit_accutile_intersections_cutile(
    state: Any,
    cumulative: jax.Array,
    valid_count: jax.Array,
    *,
    capacity: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> tuple[jax.Array, jax.Array]:
    """Emit the fixed Gaussian-major AccuTile prefix with NVIDIA cuTile."""

    gaussian_count = state.valid.shape[0]
    if capacity == 0:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        return empty, empty
    if gaussian_count == 0:
        padding = jnp.full((capacity,), -1, dtype=jnp.int32)
        return padding, padding
    _require_cuda_tile_device()
    ct, ctj = _cutile()
    _, emit_kernel = _kernels()
    _, emit_block_size, _, _ = _tuning_profile()
    gaussian_output = ctj.OutputPlaceholder((capacity,), jnp.int32)
    tile_output = ctj.OutputPlaceholder((capacity,), jnp.int32)
    return ctj.cutile_call(
        (ct.cdiv(capacity, emit_block_size),),
        emit_kernel,
        (
            state.valid.astype(jnp.uint8),
            state.is_y.astype(jnp.uint8),
            state.b,
            state.disc,
            state.t,
            state.p_u,
            state.p_v,
            state.coefficient,
            state.outer_min,
            state.outer_max,
            state.cross_min,
            state.cross_max,
            state.outer_bbox_min,
            state.outer_bbox_max,
            state.cross_bbox_min,
            state.cross_bbox_max,
            state.argmin_outer,
            state.argmax_outer,
            cumulative,
            jnp.asarray(valid_count, dtype=jnp.int32).reshape((1,)),
            gaussian_output,
            tile_output,
            tile_size,
            tile_width,
            max(tile_width, tile_height),
            capacity,
            gaussian_count,
            math.ceil(math.log2(gaussian_count + 1)),
            emit_block_size,
        ),
    )
