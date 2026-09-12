from __future__ import annotations

import math
import operator
import os
from typing import Any, NamedTuple

import cuda.tile as ct
import cuda.tile.jax as ctj
import jax
import jax.numpy as jnp

_COUNT_BLOCK_SIZE = 128
_EMIT_BLOCK_SIZE = 128
_PREFIX_BLOCK_SIZE = 256
_PREFIX_SCAN_CHUNK_SIZE = 256
_RADIX_BLOCK_SIZE = 256
_RADIX_LARGE_BLOCK_SIZE = 512
_RADIX_LARGE_CAPACITY = 1 << 22
_RADIX_SCAN_CHUNK_SIZE = 256
_RADIX_BITS = 5
_RADIX_SIZE = 1 << _RADIX_BITS
_STATE_FLOATS = 16
_STATE_BOUNDS = 4
_STATE_FLAGS = 2
_UINT64_MAX = (1 << 64) - 1
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
    "intersect_tiles_cutile",
]


class _Constant[T](int):
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
        count_block, emit_block, count_occupancy, emit_occupancy = _TUNING_PROFILES[
            profile
        ]
    except KeyError as exc:
        raise ValueError(
            f"{_TUNING_ENV} must name one of " + ", ".join(sorted(_TUNING_PROFILES))
        ) from exc
    count_block = (
        _static_optional_int(
            "JAX_GS_CUTILE_COUNT_BLOCK_SIZE",
            os.environ.get("JAX_GS_CUTILE_COUNT_BLOCK_SIZE"),
            minimum=32,
            maximum=1024,
        )
        or count_block
    )
    emit_block = (
        _static_optional_int(
            "JAX_GS_CUTILE_EMIT_BLOCK_SIZE",
            os.environ.get("JAX_GS_CUTILE_EMIT_BLOCK_SIZE"),
            minimum=32,
            maximum=1024,
        )
        or emit_block
    )
    count_occupancy = (
        _static_optional_int(
            "JAX_GS_CUTILE_COUNT_OCCUPANCY",
            os.environ.get("JAX_GS_CUTILE_COUNT_OCCUPANCY"),
            minimum=1,
            maximum=32,
        )
        or count_occupancy
    )
    emit_occupancy = (
        _static_optional_int(
            "JAX_GS_CUTILE_EMIT_OCCUPANCY",
            os.environ.get("JAX_GS_CUTILE_EMIT_OCCUPANCY"),
            minimum=1,
            maximum=32,
        )
        or emit_occupancy
    )
    return count_block, emit_block, count_occupancy, emit_occupancy


def available_tuning_profiles() -> tuple[str, ...]:
    """Return semantic-equivalent cuTile variants for external autotuning."""

    return tuple(sorted(_TUNING_PROFILES))


def _cutile():
    return ct, ctj


# The tuned count/emission variants are built once on first explicit use.
_KERNELS = None


class _NativeAccuTileState(NamedTuple):
    valid: jax.Array
    is_y: jax.Array
    b: jax.Array
    disc: jax.Array
    t: jax.Array
    p_u: jax.Array
    p_v: jax.Array
    coefficient: jax.Array
    outer_min: jax.Array
    outer_max: jax.Array
    cross_min: jax.Array
    cross_max: jax.Array
    outer_bbox_min: jax.Array
    outer_bbox_max: jax.Array
    cross_bbox_min: jax.Array
    cross_bbox_max: jax.Array
    argmin_outer: jax.Array
    argmax_outer: jax.Array


def _kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    ct, _ = _cutile()
    globals()["ct"] = ct
    globals()["_Constant"] = ct.Constant

    @ct.kernel
    def count_kernel(
        valid_ref,
        b_ref,
        disc_ref,
        t_ref,
        p_u_ref,
        p_v_ref,
        coefficient_ref,
        outer_min_ref,
        outer_max_ref,
        cross_min_ref,
        cross_max_ref,
        outer_bbox_min_ref,
        outer_bbox_max_ref,
        cross_bbox_min_ref,
        cross_bbox_max_ref,
        argmin_outer_ref,
        argmax_outer_ref,
        counts_ref,
        tile_size: _Constant[int],
        outer_steps: _Constant[int],
        block_size: _Constant[int],
    ):
        block_id = ct.bid(0)
        index = (block_id,)
        shape = (block_size,)
        valid = ct.load(valid_ref, index, shape, padding_mode=ct.PaddingMode.ZERO) != 0
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
        root = ct.where(radicand > 0.0, radicand * ct.rsqrt(radicand), 0.0)
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
            root = ct.where(radicand > 0.0, radicand * ct.rsqrt(radicand), 0.0)
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
                ct.minimum(cross_max, ct.astype(ellipse_min / block, ct.int32)),
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
        valid_ref,
        is_y_ref,
        b_ref,
        disc_ref,
        t_ref,
        p_u_ref,
        p_v_ref,
        coefficient_ref,
        outer_min_ref,
        outer_max_ref,
        cross_min_ref,
        cross_max_ref,
        outer_bbox_min_ref,
        outer_bbox_max_ref,
        cross_bbox_min_ref,
        cross_bbox_max_ref,
        argmin_outer_ref,
        argmax_outer_ref,
        cumulative_ref,
        valid_count_ref,
        depths_ref,
        first_output_ref,
        second_output_ref,
        encode_sort_keys: _Constant[bool],
        tile_size: _Constant[int],
        tile_width: _Constant[int],
        outer_steps: _Constant[int],
        capacity: _Constant[int],
        gaussian_count: _Constant[int],
        search_steps: _Constant[int],
        block_size: _Constant[int],
    ):
        block_id = ct.bid(0)
        rank = ct.arange(block_size, dtype=ct.int32, start=block_id * block_size)
        shape = (block_size,)
        valid_count = ct.maximum(
            0,
            ct.minimum(ct.load(valid_count_ref, (0,), shape=()), capacity),
        )
        output_valid = rank < valid_count
        if block_id * block_size >= valid_count:
            if encode_sort_keys:
                ct.store(
                    first_output_ref,
                    (block_id,),
                    ct.full(shape, _UINT64_MAX, ct.uint64),
                )
            else:
                ct.store(
                    first_output_ref,
                    (block_id,),
                    ct.full(shape, -1, ct.int32),
                )
            ct.store(
                second_output_ref,
                (block_id,),
                ct.full(shape, -1, ct.int32),
            )
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
        root = ct.where(radicand > 0.0, radicand * ct.rsqrt(radicand), 0.0)
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
            root = ct.where(radicand > 0.0, radicand * ct.rsqrt(radicand), 0.0)
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
                ct.minimum(cross_max, ct.astype(ellipse_min / block, ct.int32)),
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
        if encode_sort_keys:
            depth = ct.gather(depths_ref, owner)
            depth = ct.where(depth == 0.0, 0.0, depth)
            bits = ct.bitcast(depth, ct.uint32)
            ordered = ct.where(
                (bits & ct.astype(0x80000000, ct.uint32)) != 0,
                ~bits,
                bits ^ ct.astype(0x80000000, ct.uint32),
            )
            key = (
                ct.astype(tile_id, ct.uint64) << ct.astype(32, ct.uint64)
            ) | ct.astype(ordered, ct.uint64)
            ct.store(
                first_output_ref,
                (block_id,),
                ct.where(final_valid, key, ct.astype(_UINT64_MAX, ct.uint64)),
            )
            ct.store(
                second_output_ref,
                (block_id,),
                ct.where(final_valid, owner, -1),
            )
        else:
            ct.store(
                first_output_ref,
                (block_id,),
                ct.where(final_valid, owner, -1),
            )
            ct.store(
                second_output_ref,
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


@ct.kernel
def _prepare_count_kernel(
    means2d,
    radii,
    depths,
    conics,
    opacities,
    input_valid,
    state_floats,
    state_bounds,
    state_flags,
    counts_out,
    gaussian_count: ct.Constant[int],
    tile_size: ct.Constant[int],
    tile_width: ct.Constant[int],
    tile_height: ct.Constant[int],
    alpha_threshold: ct.Constant[float],
    outer_steps: ct.Constant[int],
    block_size: ct.Constant[int],
):
    rank = ct.arange(block_size, start=ct.bid(0) * block_size, dtype=ct.int32)
    in_bounds = rank < gaussian_count
    safe_rank = ct.minimum(rank, gaussian_count - 1)
    px = ct.gather(means2d, (safe_rank, 0), mask=in_bounds)
    py = ct.gather(means2d, (safe_rank, 1), mask=in_bounds)
    radius_x = ct.gather(radii, (safe_rank, 0), mask=in_bounds)
    radius_y = ct.gather(radii, (safe_rank, 1), mask=in_bounds)
    depth = ct.gather(depths, safe_rank, mask=in_bounds)
    a = ct.gather(conics, (safe_rank, 0), mask=in_bounds)
    b = ct.gather(conics, (safe_rank, 1), mask=in_bounds)
    c = ct.gather(conics, (safe_rank, 2), mask=in_bounds)
    opacity = ct.gather(opacities, safe_rank, mask=in_bounds)
    requested = ct.gather(input_valid, safe_rank, mask=in_bounds) != 0
    finite = (
        (~ct.isnan(px))
        & (~ct.isnan(py))
        & (~ct.isnan(depth))
        & (~ct.isnan(a))
        & (~ct.isnan(b))
        & (~ct.isnan(c))
        & (~ct.isnan(opacity))
        & (ct.abs(px) != float("inf"))
        & (ct.abs(py) != float("inf"))
        & (ct.abs(depth) != float("inf"))
        & (ct.abs(a) != float("inf"))
        & (ct.abs(b) != float("inf"))
        & (ct.abs(c) != float("inf"))
        & (ct.abs(opacity) != float("inf"))
    )
    base_valid = (
        in_bounds
        & requested
        & finite
        & (radius_x > 0)
        & (radius_y > 0)
        & (opacity > alpha_threshold)
        & (a > 0.0)
        & (c > 0.0)
    )

    safe_px = ct.where(base_valid, px, 0.0)
    safe_py = ct.where(base_valid, py, 0.0)
    safe_a = ct.where(base_valid, a, 1.0)
    safe_b = ct.where(base_valid, b, 0.0)
    safe_c = ct.where(base_valid, c, 1.0)
    safe_opacity = ct.where(base_valid, opacity, alpha_threshold)
    disc = safe_b * safe_b - safe_a * safe_c
    t = ct.minimum(
        3.33 * 3.33,
        2.0 * ct.log(safe_opacity / alpha_threshold),
    )
    ellipse_valid = (
        base_valid
        & (~ct.isnan(disc))
        & (~ct.isnan(t))
        & (ct.abs(disc) != float("inf"))
        & (ct.abs(t) != float("inf"))
        & (disc < 0.0)
        & (t > 0.0)
    )
    safe_disc = ct.where(ellipse_valid, disc, -1.0)
    safe_t = ct.where(ellipse_valid, t, 1.0)
    scale = -safe_t / safe_disc
    x_extent = ct.sqrt(ct.maximum(scale * safe_c, 0.0))
    y_extent = ct.sqrt(ct.maximum(scale * safe_a, 0.0))
    bbox_min_x = safe_px - x_extent
    bbox_min_y = safe_py - y_extent
    bbox_max_x = safe_px + x_extent
    bbox_max_y = safe_py + y_extent
    argmin_x = safe_py + safe_b * x_extent / safe_c
    argmin_y = safe_px + safe_b * y_extent / safe_a
    argmax_x = safe_py - safe_b * x_extent / safe_c
    argmax_y = safe_px - safe_b * y_extent / safe_a
    derived_valid = (
        ellipse_valid
        & (~ct.isnan(scale))
        & (~ct.isnan(x_extent))
        & (~ct.isnan(y_extent))
        & (~ct.isnan(bbox_min_x))
        & (~ct.isnan(bbox_min_y))
        & (~ct.isnan(bbox_max_x))
        & (~ct.isnan(bbox_max_y))
        & (~ct.isnan(argmin_x))
        & (~ct.isnan(argmin_y))
        & (~ct.isnan(argmax_x))
        & (~ct.isnan(argmax_y))
        & (ct.abs(scale) != float("inf"))
        & (ct.abs(x_extent) != float("inf"))
        & (ct.abs(y_extent) != float("inf"))
        & (ct.abs(bbox_min_x) != float("inf"))
        & (ct.abs(bbox_min_y) != float("inf"))
        & (ct.abs(bbox_max_x) != float("inf"))
        & (ct.abs(bbox_max_y) != float("inf"))
        & (ct.abs(argmin_x) != float("inf"))
        & (ct.abs(argmin_y) != float("inf"))
        & (ct.abs(argmax_x) != float("inf"))
        & (ct.abs(argmax_y) != float("inf"))
    )
    bbox_min_x = ct.where(derived_valid, bbox_min_x, 0.0)
    bbox_min_y = ct.where(derived_valid, bbox_min_y, 0.0)
    bbox_max_x = ct.where(derived_valid, bbox_max_x, 0.0)
    bbox_max_y = ct.where(derived_valid, bbox_max_y, 0.0)
    argmin_x = ct.where(derived_valid, argmin_x, 0.0)
    argmin_y = ct.where(derived_valid, argmin_y, 0.0)
    argmax_x = ct.where(derived_valid, argmax_x, 0.0)
    argmax_y = ct.where(derived_valid, argmax_y, 0.0)

    block = float(tile_size)
    rect_min_x = ct.maximum(
        0,
        ct.minimum(ct.astype(bbox_min_x / block, ct.int32), tile_width),
    )
    rect_min_y = ct.maximum(
        0,
        ct.minimum(ct.astype(bbox_min_y / block, ct.int32), tile_height),
    )
    rect_max_x = ct.maximum(
        0,
        ct.minimum(ct.astype(bbox_max_x / block + 1.0, ct.int32), tile_width),
    )
    rect_max_y = ct.maximum(
        0,
        ct.minimum(ct.astype(bbox_max_y / block + 1.0, ct.int32), tile_height),
    )
    span_x = rect_max_x - rect_min_x
    span_y = rect_max_y - rect_min_y
    state_valid = derived_valid & (span_x > 0) & (span_y > 0)
    is_y = span_y < span_x
    p_u = ct.where(is_y, safe_py, safe_px)
    p_v = ct.where(is_y, safe_px, safe_py)
    coefficient = ct.where(is_y, safe_a, safe_c)
    outer_min = ct.where(is_y, rect_min_y, rect_min_x)
    outer_max = ct.where(is_y, rect_max_y, rect_max_x)
    cross_min = ct.where(is_y, rect_min_x, rect_min_y)
    cross_max = ct.where(is_y, rect_max_x, rect_max_y)
    outer_bbox_min = ct.where(is_y, bbox_min_y, bbox_min_x)
    outer_bbox_max = ct.where(is_y, bbox_max_y, bbox_max_x)
    cross_bbox_min = ct.where(is_y, bbox_min_x, bbox_min_y)
    cross_bbox_max = ct.where(is_y, bbox_max_x, bbox_max_y)
    argmin_outer = ct.where(is_y, argmin_x, argmin_y)
    argmax_outer = ct.where(is_y, argmax_x, argmax_y)

    safe_disc = ct.where(state_valid, safe_disc, -1.0)
    safe_t = ct.where(state_valid, safe_t, 1.0)
    coefficient = ct.where(state_valid, coefficient, 1.0)
    min_line = ct.astype(outer_min, ct.float32) * block
    h = min_line - p_u
    radicand = ct.maximum(safe_disc * h * h + safe_t * coefficient, 0.0)
    root = ct.where(radicand > 0.0, radicand * ct.rsqrt(radicand), 0.0)
    line_min = (-safe_b * h - root) / coefficient + p_v
    line_max = (-safe_b * h + root) / coefficient + p_v
    intersects = outer_bbox_min <= min_line
    previous_min = ct.where(intersects, line_min, cross_bbox_max)
    previous_max = ct.where(intersects, line_max, cross_bbox_min)
    counts = ct.zeros((block_size,), ct.int32)
    for outer_offset in range(outer_steps):
        outer = outer_min + outer_offset
        active = state_valid & (outer < outer_max)
        min_line = ct.astype(outer, ct.float32) * block
        max_line = min_line + block
        h = max_line - p_u
        radicand = ct.maximum(safe_disc * h * h + safe_t * coefficient, 0.0)
        root = ct.where(radicand > 0.0, radicand * ct.rsqrt(radicand), 0.0)
        line_min = (-safe_b * h - root) / coefficient + p_v
        line_max = (-safe_b * h + root) / coefficient + p_v
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
            ct.minimum(cross_max, ct.astype(ellipse_min / block, ct.int32)),
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

    write = in_bounds
    ct.scatter(state_floats, (rank, 0), safe_b, mask=write)
    ct.scatter(state_floats, (rank, 1), safe_disc, mask=write)
    ct.scatter(state_floats, (rank, 2), safe_t, mask=write)
    ct.scatter(state_floats, (rank, 3), p_u, mask=write)
    ct.scatter(state_floats, (rank, 4), p_v, mask=write)
    ct.scatter(state_floats, (rank, 5), coefficient, mask=write)
    ct.scatter(state_floats, (rank, 6), outer_bbox_min, mask=write)
    ct.scatter(state_floats, (rank, 7), outer_bbox_max, mask=write)
    ct.scatter(state_floats, (rank, 8), cross_bbox_min, mask=write)
    ct.scatter(state_floats, (rank, 9), cross_bbox_max, mask=write)
    ct.scatter(state_floats, (rank, 10), argmin_outer, mask=write)
    ct.scatter(state_floats, (rank, 11), argmax_outer, mask=write)
    ct.scatter(state_bounds, (rank, 0), outer_min, mask=write)
    ct.scatter(state_bounds, (rank, 1), outer_max, mask=write)
    ct.scatter(state_bounds, (rank, 2), cross_min, mask=write)
    ct.scatter(state_bounds, (rank, 3), cross_max, mask=write)
    ct.scatter(
        state_flags,
        (rank, 0),
        ct.astype(state_valid, ct.uint8),
        mask=write,
    )
    ct.scatter(
        state_flags,
        (rank, 1),
        ct.astype(is_y, ct.uint8),
        mask=write,
    )
    ct.scatter(counts_out, rank, counts, mask=write)


@ct.kernel
def _prefix_local_kernel(
    counts,
    local_prefix,
    block_sums,
    block_size: ct.Constant[int],
):
    block_id = ct.bid(0)
    values = ct.load(
        counts,
        (block_id,),
        shape=(block_size,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    values = ct.astype(values, ct.int64)
    inclusive = ct.cumsum(values)
    ct.store(local_prefix, (block_id,), inclusive)
    ct.store(block_sums, (block_id,), ct.sum(values))


@ct.kernel
def _prefix_blocks_kernel(
    block_sums,
    block_bases,
    chunk_bases,
    valid_count_out,
    overflow_out,
    required_count_out,
    capacity: ct.Constant[int],
    block_count: ct.Constant[int],
):
    running = ct.astype(0, ct.int64)
    block = 0
    while block < block_count:
        value = ct.load(block_sums, (block,), shape=())
        ct.store(block_bases, (block,), running)
        running += value
        block += 1
    ct.store(chunk_bases, (0,), ct.astype(0, ct.int64))
    saturated = ct.minimum(running, ct.astype((1 << 30) - 1, ct.int64))
    required = ct.astype(saturated, ct.int32)
    ct.store(required_count_out, (0,), required)
    ct.store(valid_count_out, (0,), ct.minimum(required, capacity))
    ct.store(
        overflow_out,
        (0,),
        ct.astype(required > capacity, ct.uint8),
    )


@ct.kernel
def _prefix_scan_local_kernel(
    block_sums,
    block_bases,
    chunk_sums,
    block_count: ct.Constant[int],
    scan_chunk_size: ct.Constant[int],
):
    chunk_id = ct.bid(0)
    block = ct.arange(
        scan_chunk_size,
        start=chunk_id * scan_chunk_size,
        dtype=ct.int32,
    )
    values = ct.load(
        block_sums,
        (chunk_id,),
        shape=(scan_chunk_size,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    values = ct.where(block < block_count, values, 0)
    inclusive = ct.cumsum(values)
    ct.store(block_bases, (chunk_id,), inclusive - values)
    ct.store(chunk_sums, (chunk_id,), ct.sum(values))


@ct.kernel
def _prefix_scan_chunks_kernel(
    chunk_sums,
    chunk_bases,
    valid_count_out,
    overflow_out,
    required_count_out,
    capacity: ct.Constant[int],
    chunk_count: ct.Constant[int],
    scan_chunk_size: ct.Constant[int],
):
    chunk = ct.arange(scan_chunk_size, dtype=ct.int32)
    values = ct.load(
        chunk_sums,
        (0,),
        shape=(scan_chunk_size,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    values = ct.where(chunk < chunk_count, values, 0)
    inclusive = ct.cumsum(values)
    ct.store(chunk_bases, (0,), inclusive - values)
    running = ct.sum(values)
    saturated = ct.minimum(running, ct.astype((1 << 30) - 1, ct.int64))
    required = ct.astype(saturated, ct.int32)
    ct.store(required_count_out, (0,), required)
    ct.store(valid_count_out, (0,), ct.minimum(required, capacity))
    ct.store(
        overflow_out,
        (0,),
        ct.astype(required > capacity, ct.uint8),
    )


@ct.kernel
def _prefix_finalize_kernel(
    local_prefix,
    block_bases,
    chunk_bases,
    cumulative,
    gaussian_count: ct.Constant[int],
    capacity: ct.Constant[int],
    block_size: ct.Constant[int],
    scan_chunk_size: ct.Constant[int],
):
    block_id = ct.bid(0)
    rank = ct.arange(block_size, start=block_id * block_size, dtype=ct.int32)
    in_bounds = rank < gaussian_count
    local = ct.load(
        local_prefix,
        (block_id,),
        shape=(block_size,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    base = ct.load(block_bases, (block_id,), shape=()) + ct.load(
        chunk_bases,
        (block_id // scan_chunk_size,),
        shape=(),
    )
    capped = ct.minimum(base + local, ct.astype(capacity + 1, ct.int64))
    ct.scatter(
        cumulative,
        rank,
        ct.astype(capped, ct.int32),
        mask=in_bounds,
    )


@ct.kernel
def _prepare_sort_keys_kernel(
    gaussian_ids,
    tile_ids,
    depths,
    declared_valid_count,
    keys_out,
    values_out,
    capacity: ct.Constant[int],
    gaussian_count: ct.Constant[int],
    tile_count: ct.Constant[int],
    block_size: ct.Constant[int],
):
    rank = ct.arange(block_size, start=ct.bid(0) * block_size, dtype=ct.int32)
    in_bounds = rank < capacity
    gaussian_id = ct.gather(gaussian_ids, rank, mask=in_bounds, padding_value=-1)
    tile_id = ct.gather(tile_ids, rank, mask=in_bounds, padding_value=-1)
    count = ct.maximum(
        0,
        ct.minimum(ct.load(declared_valid_count, (0,), shape=()), capacity),
    )
    item_valid = (
        in_bounds
        & (rank < count)
        & (gaussian_id >= 0)
        & (gaussian_id < gaussian_count)
        & (tile_id >= 0)
        & (tile_id < tile_count)
    )
    safe_id = ct.maximum(0, ct.minimum(gaussian_id, gaussian_count - 1))
    depth = ct.gather(depths, safe_id, mask=in_bounds)
    item_valid = item_valid & (~ct.isnan(depth)) & (ct.abs(depth) != float("inf"))
    depth = ct.where(depth == 0.0, 0.0, depth)
    bits = ct.bitcast(depth, ct.uint32)
    ordered = ct.where(
        (bits & ct.astype(0x80000000, ct.uint32)) != 0,
        ~bits,
        bits ^ ct.astype(0x80000000, ct.uint32),
    )
    key = (ct.astype(tile_id, ct.uint64) << ct.astype(32, ct.uint64)) | ct.astype(
        ordered, ct.uint64
    )
    key = ct.where(item_valid, key, ct.astype(_UINT64_MAX, ct.uint64))
    ct.scatter(keys_out, rank, key, mask=in_bounds)
    ct.scatter(
        values_out,
        rank,
        ct.where(item_valid, gaussian_id, -1),
        mask=in_bounds,
    )


@ct.kernel
def _radix_histogram_kernel(
    keys,
    valid_count,
    histogram,
    local_ranks,
    capacity: ct.Constant[int],
    shift: ct.Constant[int],
    radix_size: ct.Constant[int],
    block_size: ct.Constant[int],
):
    block_id = ct.bid(0)
    count = ct.maximum(
        0,
        ct.minimum(ct.load(valid_count, (0,), shape=()), capacity),
    )
    if block_id * block_size >= count:
        return
    rank = ct.arange(block_size, start=block_id * block_size, dtype=ct.int32)
    in_bounds = rank < count
    key = ct.load(
        keys,
        (block_id,),
        shape=(block_size,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    key = ct.where(in_bounds, key, ct.astype(_UINT64_MAX, ct.uint64))
    digit = ct.astype(
        (key >> ct.astype(shift, ct.uint64)) & ct.astype(radix_size - 1, ct.uint64),
        ct.int32,
    )
    subgroups = block_size // 32
    digit = ct.reshape(digit, (subgroups, 32))
    valid = ct.reshape(in_bounds, (subgroups, 32))
    packed_buckets = (radix_size + 3) // 4
    bucket_group = ct.arange(packed_buckets, dtype=ct.int32)
    matches = (bucket_group[:, None, None] == digit[None, :, :] // 4) & valid[
        None, :, :
    ]
    slot = ct.astype(digit & 3, ct.uint32)
    # Four six-bit fields hold independent counts within 32-lane subgroups.
    packed = ct.astype(matches, ct.uint32) << (slot[None, :, :] * 6)
    local = ct.cumsum(packed, axis=2)
    local_count = ct.astype((local >> (slot[None, :, :] * 6)) & 63, ct.uint16)
    totals = ct.astype(ct.sum(packed, axis=2), ct.uint64)
    field = ct.arange(4, dtype=ct.uint64)
    # Expand subgroup totals before accumulating up to 512 lanes per bucket.
    wide = ct.sum(
        ((totals[:, :, None] >> (field[None, None, :] * 6)) & 63)
        << (field[None, None, :] * 16),
        axis=2,
    )
    bases = ct.cumsum(wide, axis=1) - wide
    group_bases = ct.astype(
        (bases[:, :, None] >> ct.astype(slot[None, :, :] * 16, ct.uint64)) & 0xFFFF,
        ct.uint16,
    )
    ranks = ct.reshape(
        ct.sum(ct.where(matches, group_bases + local_count - 1, 0), axis=0),
        (block_size,),
    )
    bucket_totals = ct.sum(wide, axis=1)
    counts = ct.astype(
        ct.reshape(
            (bucket_totals[:, None] >> (field[None, :] * 16)) & 0xFFFF,
            (packed_buckets * 4,),
        ),
        ct.int32,
    )
    ct.store(histogram, (block_id, 0), counts[None, :])
    # A rank is local to a block of at most 512 entries, so uint16 is exact.
    ct.store(local_ranks, (block_id,), ct.astype(ranks, ct.uint16))


@ct.kernel
def _radix_scan_local_kernel(
    histogram,
    valid_count,
    block_prefix,
    chunk_sums,
    capacity: ct.Constant[int],
    radix_block_size: ct.Constant[int],
    scan_chunk_size: ct.Constant[int],
    radix_size: ct.Constant[int],
):
    chunk_id = ct.bid(0)
    count = ct.maximum(
        0,
        ct.minimum(ct.load(valid_count, (0,), shape=()), capacity),
    )
    active_blocks = (count + radix_block_size - 1) // radix_block_size
    if chunk_id * scan_chunk_size >= active_blocks:
        return
    block = ct.arange(
        scan_chunk_size,
        start=chunk_id * scan_chunk_size,
        dtype=ct.int32,
    )
    active = block < active_blocks
    values = ct.load(
        histogram,
        (chunk_id, 0),
        shape=(scan_chunk_size, radix_size),
        padding_mode=ct.PaddingMode.ZERO,
    )
    values = ct.where(active[:, None], values, 0)
    inclusive = ct.cumsum(values, axis=0)
    ct.store(block_prefix, (chunk_id, 0), inclusive - values)
    ct.store(
        chunk_sums,
        (chunk_id, 0),
        ct.sum(values, axis=0)[None, :],
    )


@ct.kernel
def _radix_scan_chunks_kernel(
    chunk_sums,
    valid_count,
    chunk_prefix,
    capacity: ct.Constant[int],
    radix_block_size: ct.Constant[int],
    scan_chunk_size: ct.Constant[int],
    radix_size: ct.Constant[int],
    chunk_scan_size: ct.Constant[int],
):
    count = ct.maximum(
        0,
        ct.minimum(ct.load(valid_count, (0,), shape=()), capacity),
    )
    active_blocks = (count + radix_block_size - 1) // radix_block_size
    active_chunks = (active_blocks + scan_chunk_size - 1) // scan_chunk_size
    groups = ct.cdiv(chunk_sums.shape[0], chunk_scan_size)
    lane = ct.arange(chunk_scan_size, dtype=ct.int32)
    totals = ct.zeros((radix_size,), ct.int32)
    for group in range(groups):
        values = ct.load(
            chunk_sums,
            (group, 0),
            shape=(chunk_scan_size, radix_size),
            padding_mode=ct.PaddingMode.ZERO,
        )
        active = group * chunk_scan_size + lane < active_chunks
        values = ct.where(active[:, None], values, 0)
        totals += ct.sum(values, axis=0)

    # Include the bucket base in each chunk prefix, eliminating a separate
    # kernel and one metadata gather per scattered intersection.
    running = ct.cumsum(totals) - totals
    for group in range(groups):
        values = ct.load(
            chunk_sums,
            (group, 0),
            shape=(chunk_scan_size, radix_size),
            padding_mode=ct.PaddingMode.ZERO,
        )
        active = group * chunk_scan_size + lane < active_chunks
        values = ct.where(active[:, None], values, 0)
        inclusive = ct.cumsum(values, axis=0)
        ct.store(chunk_prefix, (group, 0), inclusive - values + running[None, :])
        running += ct.sum(values, axis=0)


@ct.kernel
def _radix_scatter_kernel(
    keys,
    values,
    valid_count,
    block_prefix,
    chunk_prefix,
    local_ranks,
    output_keys,
    output_values,
    capacity: ct.Constant[int],
    shift: ct.Constant[int],
    radix_size: ct.Constant[int],
    block_size: ct.Constant[int],
    scan_chunk_size: ct.Constant[int],
):
    block_id = ct.bid(0)
    count = ct.maximum(
        0,
        ct.minimum(ct.load(valid_count, (0,), shape=()), capacity),
    )
    if block_id * block_size >= count:
        return
    rank = ct.arange(block_size, start=block_id * block_size, dtype=ct.int32)
    in_bounds = rank < count
    key = ct.load(
        keys,
        (block_id,),
        shape=(block_size,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    key = ct.where(in_bounds, key, ct.astype(_UINT64_MAX, ct.uint64))
    value = ct.load(
        values,
        (block_id,),
        shape=(block_size,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    local = ct.astype(
        ct.load(
            local_ranks,
            (block_id,),
            shape=(block_size,),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.int32,
    )
    digit = ct.astype(
        (key >> ct.astype(shift, ct.uint64)) & ct.astype(radix_size - 1, ct.uint64),
        ct.int32,
    )
    prefix = ct.gather(block_prefix, (block_id, digit)) + ct.gather(
        chunk_prefix, (block_id // scan_chunk_size, digit)
    )
    target = prefix + local
    ct.scatter(output_keys, target, key, mask=in_bounds)
    ct.scatter(output_values, target, value, mask=in_bounds)


@ct.kernel
def _effective_count_kernel(
    keys,
    declared_valid_count,
    effective_count,
    capacity: ct.Constant[int],
):
    low = ct.astype(0, ct.int32)
    high = ct.maximum(
        0,
        ct.minimum(ct.load(declared_valid_count, (0,), shape=()), capacity),
    )
    while low < high:
        middle = low + (high - low) // 2
        key = ct.load(keys, (middle,), shape=())
        if key != ct.astype(_UINT64_MAX, ct.uint64):
            low = middle + 1
        else:
            high = middle
    ct.store(effective_count, (0,), low)


@ct.kernel
def _finalize_pairs_kernel(
    keys,
    values,
    effective_count,
    gaussian_ids_out,
    tile_ids_out,
    capacity: ct.Constant[int],
    block_size: ct.Constant[int],
):
    block_id = ct.bid(0)
    rank = ct.arange(block_size, start=block_id * block_size, dtype=ct.int32)
    in_bounds = rank < capacity
    key = ct.load(
        keys,
        (block_id,),
        shape=(block_size,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    key = ct.where(in_bounds, key, ct.astype(_UINT64_MAX, ct.uint64))
    value = ct.load(
        values,
        (block_id,),
        shape=(block_size,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    count = ct.load(effective_count, (0,), shape=())
    valid = in_bounds & (rank < count) & (key != ct.astype(_UINT64_MAX, ct.uint64))
    tile = ct.astype(key >> ct.astype(32, ct.uint64), ct.int32)
    ct.scatter(
        gaussian_ids_out,
        rank,
        ct.where(valid, value, -1),
        mask=in_bounds,
    )
    ct.scatter(
        tile_ids_out,
        rank,
        ct.where(valid, tile, -1),
        mask=in_bounds,
    )


@ct.kernel
def _offsets_kernel(
    keys,
    effective_count,
    offsets_out,
    capacity: ct.Constant[int],
    tile_count: ct.Constant[int],
    search_steps: ct.Constant[int],
    block_size: ct.Constant[int],
):
    tile = ct.arange(block_size, start=ct.bid(0) * block_size, dtype=ct.int32)
    in_bounds = tile < tile_count
    count = ct.load(effective_count, (0,), shape=())
    low = ct.zeros((block_size,), ct.int32)
    high = ct.full((block_size,), count, ct.int32)
    needle = ct.astype(tile, ct.uint64) << ct.astype(32, ct.uint64)
    for _ in range(search_steps):
        middle = low + (high - low) // 2
        safe_middle = ct.minimum(middle, capacity - 1)
        key = ct.gather(
            keys,
            safe_middle,
            mask=in_bounds & (middle < count),
            padding_value=_UINT64_MAX,
        )
        move_right = key < needle
        low = ct.where(move_right, middle + 1, low)
        high = ct.where(move_right, high, middle)
    ct.scatter(offsets_out, tile, low, mask=in_bounds)


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
    """Count AccuTile pairs with an NVIDIA cuTile kernel."""

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
            min(tile_width, tile_height),
            count_block_size,
        ),
    )


def _emit_accutile_cutile(
    state: Any,
    cumulative: jax.Array,
    valid_count: jax.Array,
    depths: jax.Array,
    *,
    capacity: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    encode_sort_keys: bool,
) -> tuple[jax.Array, jax.Array]:
    gaussian_count = state.valid.shape[0]
    first_dtype = jnp.uint64 if encode_sort_keys else jnp.int32
    if capacity == 0:
        return (
            jnp.zeros((0,), dtype=first_dtype),
            jnp.zeros((0,), dtype=jnp.int32),
        )
    if gaussian_count == 0:
        first_padding = _UINT64_MAX if encode_sort_keys else -1
        return (
            jnp.full((capacity,), first_padding, dtype=first_dtype),
            jnp.full((capacity,), -1, dtype=jnp.int32),
        )
    _require_cuda_tile_device()
    ct, ctj = _cutile()
    _, emit_kernel = _kernels()
    _, emit_block_size, _, _ = _tuning_profile()
    first_output = ctj.OutputPlaceholder((capacity,), first_dtype)
    second_output = ctj.OutputPlaceholder((capacity,), jnp.int32)
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
            depths,
            first_output,
            second_output,
            encode_sort_keys,
            tile_size,
            tile_width,
            min(tile_width, tile_height),
            capacity,
            gaussian_count,
            math.ceil(math.log2(gaussian_count + 1)),
            emit_block_size,
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

    return _emit_accutile_cutile(
        state,
        cumulative,
        valid_count,
        state.b,
        capacity=capacity,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        encode_sort_keys=False,
    )


def _native_state(
    state_floats: jax.Array,
    state_bounds: jax.Array,
    state_flags: jax.Array,
) -> _NativeAccuTileState:
    return _NativeAccuTileState(
        valid=state_flags[:, 0].astype(jnp.bool_),
        is_y=state_flags[:, 1].astype(jnp.bool_),
        b=state_floats[:, 0],
        disc=state_floats[:, 1],
        t=state_floats[:, 2],
        p_u=state_floats[:, 3],
        p_v=state_floats[:, 4],
        coefficient=state_floats[:, 5],
        outer_min=state_bounds[:, 0],
        outer_max=state_bounds[:, 1],
        cross_min=state_bounds[:, 2],
        cross_max=state_bounds[:, 3],
        outer_bbox_min=state_floats[:, 6],
        outer_bbox_max=state_floats[:, 7],
        cross_bbox_min=state_floats[:, 8],
        cross_bbox_max=state_floats[:, 9],
        argmin_outer=state_floats[:, 10],
        argmax_outer=state_floats[:, 11],
    )


def _prefix_counts_cutile(
    counts: jax.Array, *, capacity: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    gaussian_count = counts.shape[0]
    block_count = math.ceil(gaussian_count / _PREFIX_BLOCK_SIZE)
    local, block_sums = ctj.cutile_call(
        (block_count,),
        _prefix_local_kernel,
        (
            counts,
            ctj.OutputPlaceholder((gaussian_count,), jnp.int64),
            ctj.OutputPlaceholder((block_count,), jnp.int64),
            _PREFIX_BLOCK_SIZE,
        ),
    )
    if block_count <= _PREFIX_SCAN_CHUNK_SIZE:
        block_bases, chunk_bases, valid_count, overflow, required_count = (
            ctj.cutile_call(
                (1,),
                _prefix_blocks_kernel,
                (
                    block_sums,
                    ctj.OutputPlaceholder((block_count,), jnp.int64),
                    ctj.OutputPlaceholder((1,), jnp.int64),
                    ctj.OutputPlaceholder((1,), jnp.int32),
                    ctj.OutputPlaceholder((1,), jnp.uint8),
                    ctj.OutputPlaceholder((1,), jnp.int32),
                    capacity,
                    block_count,
                ),
            )
        )
    else:
        chunk_count = math.ceil(block_count / _PREFIX_SCAN_CHUNK_SIZE)
        block_bases, chunk_sums = ctj.cutile_call(
            (chunk_count,),
            _prefix_scan_local_kernel,
            (
                block_sums,
                ctj.OutputPlaceholder((block_count,), jnp.int64),
                ctj.OutputPlaceholder((chunk_count,), jnp.int64),
                block_count,
                _PREFIX_SCAN_CHUNK_SIZE,
            ),
        )
        chunk_bases, valid_count, overflow, required_count = ctj.cutile_call(
            (1,),
            _prefix_scan_chunks_kernel,
            (
                chunk_sums,
                ctj.OutputPlaceholder((chunk_count,), jnp.int64),
                ctj.OutputPlaceholder((1,), jnp.int32),
                ctj.OutputPlaceholder((1,), jnp.uint8),
                ctj.OutputPlaceholder((1,), jnp.int32),
                capacity,
                chunk_count,
                # The top level can exceed 256 chunks for more than 256**3 rows.
                1 << (chunk_count - 1).bit_length(),
            ),
        )
    cumulative = ctj.cutile_call(
        (block_count,),
        _prefix_finalize_kernel,
        (
            local,
            block_bases,
            chunk_bases,
            ctj.OutputPlaceholder((gaussian_count,), jnp.int32),
            gaussian_count,
            capacity,
            _PREFIX_BLOCK_SIZE,
            _PREFIX_SCAN_CHUNK_SIZE,
        ),
    )
    return (
        cumulative,
        valid_count[0],
        overflow[0].astype(jnp.bool_),
        required_count[0],
    )


def _radix_pass_cutile(
    keys: jax.Array,
    values: jax.Array,
    valid_count: jax.Array,
    *,
    shift: int,
    radix_size: int,
) -> tuple[jax.Array, jax.Array]:
    capacity = keys.shape[0]
    # Wider blocks amortize scan metadata on large jobs but regress small prefixes.
    block_size = (
        _RADIX_LARGE_BLOCK_SIZE
        if capacity >= _RADIX_LARGE_CAPACITY
        else _RADIX_BLOCK_SIZE
    )
    block_count = math.ceil(capacity / block_size)
    chunk_count = math.ceil(block_count / _RADIX_SCAN_CHUNK_SIZE)
    padded_block_count = chunk_count * _RADIX_SCAN_CHUNK_SIZE
    histogram, local_ranks = ctj.cutile_call(
        (block_count,),
        _radix_histogram_kernel,
        (
            keys,
            valid_count.reshape((1,)),
            ctj.OutputPlaceholder((padded_block_count, radix_size), jnp.int32),
            ctj.OutputPlaceholder((capacity,), jnp.uint16),
            capacity,
            shift,
            radix_size,
            block_size,
        ),
    )
    # Scan blocks in parallel chunks, then scan only the chunk totals.  A
    # bucket-by-bucket serial walk over every radix block dominates large jobs.
    block_prefix, chunk_sums = ctj.cutile_call(
        (chunk_count,),
        _radix_scan_local_kernel,
        (
            histogram,
            valid_count.reshape((1,)),
            ctj.OutputPlaceholder(histogram.shape, jnp.int32),
            ctj.OutputPlaceholder((chunk_count, radix_size), jnp.int32),
            capacity,
            block_size,
            _RADIX_SCAN_CHUNK_SIZE,
            radix_size,
        ),
    )
    chunk_prefix = ctj.cutile_call(
        (1,),
        _radix_scan_chunks_kernel,
        (
            chunk_sums,
            valid_count.reshape((1,)),
            ctj.OutputPlaceholder(chunk_sums.shape, jnp.int32),
            capacity,
            block_size,
            _RADIX_SCAN_CHUNK_SIZE,
            radix_size,
            min(_RADIX_SCAN_CHUNK_SIZE, 1 << (chunk_count - 1).bit_length()),
        ),
    )
    return ctj.cutile_call(
        (block_count,),
        _radix_scatter_kernel,
        (
            keys,
            values,
            valid_count.reshape((1,)),
            block_prefix,
            chunk_prefix,
            local_ranks,
            ctj.OutputPlaceholder((capacity,), jnp.uint64),
            ctj.OutputPlaceholder((capacity,), jnp.int32),
            capacity,
            shift,
            radix_size,
            block_size,
            _RADIX_SCAN_CHUNK_SIZE,
        ),
    )


def _sort_prepared_offsets_cutile(
    keys: jax.Array,
    values: jax.Array,
    valid_count: jax.Array,
    *,
    tile_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    capacity = keys.shape[0]
    block_count = math.ceil(capacity / _RADIX_BLOCK_SIZE)
    end_bit = 32 + tile_count.bit_length()
    for shift in range(0, end_bit, _RADIX_BITS):
        # Do not pay for 32 buckets when the final pass has fewer than five bits.
        pass_bits = min(_RADIX_BITS, end_bit - shift)
        keys, values = _radix_pass_cutile(
            keys,
            values,
            valid_count,
            shift=shift,
            radix_size=1 << pass_bits,
        )
    effective_count = ctj.cutile_call(
        (1,),
        _effective_count_kernel,
        (
            keys,
            valid_count.reshape((1,)),
            ctj.OutputPlaceholder((1,), jnp.int32),
            capacity,
        ),
    )
    sorted_gaussians, sorted_tiles = ctj.cutile_call(
        (block_count,),
        _finalize_pairs_kernel,
        (
            keys,
            values,
            effective_count,
            ctj.OutputPlaceholder((capacity,), jnp.int32),
            ctj.OutputPlaceholder((capacity,), jnp.int32),
            capacity,
            _RADIX_BLOCK_SIZE,
        ),
    )
    offset_blocks = math.ceil(tile_count / _RADIX_BLOCK_SIZE)
    offsets = ctj.cutile_call(
        (offset_blocks,),
        _offsets_kernel,
        (
            keys,
            effective_count,
            ctj.OutputPlaceholder((tile_count,), jnp.int32),
            capacity,
            tile_count,
            math.ceil(math.log2(capacity + 1)),
            _RADIX_BLOCK_SIZE,
        ),
    )
    return sorted_gaussians, sorted_tiles, offsets, effective_count[0]


def _sort_offsets_cutile(
    gaussian_ids: jax.Array,
    tile_ids: jax.Array,
    depths: jax.Array,
    valid_count: jax.Array,
    *,
    tile_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    capacity = gaussian_ids.shape[0]
    block_count = math.ceil(capacity / _RADIX_BLOCK_SIZE)
    keys, values = ctj.cutile_call(
        (block_count,),
        _prepare_sort_keys_kernel,
        (
            gaussian_ids,
            tile_ids,
            depths,
            valid_count.reshape((1,)),
            ctj.OutputPlaceholder((capacity,), jnp.uint64),
            ctj.OutputPlaceholder((capacity,), jnp.int32),
            capacity,
            depths.shape[0],
            tile_count,
            _RADIX_BLOCK_SIZE,
        ),
    )
    return _sort_prepared_offsets_cutile(
        keys,
        values,
        valid_count,
        tile_count=tile_count,
    )


def intersect_tiles_cutile(
    means2d: jax.Array,
    radii: jax.Array,
    depths: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    valid: jax.Array,
    *,
    capacity: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    alpha_threshold: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run AccuTile preparation, prefix, emission, and sorting in cuTile."""

    _require_cuda_tile_device()
    gaussian_count = means2d.shape[0]
    count_block, _, _, _ = _tuning_profile()
    block_count = math.ceil(gaussian_count / count_block)
    state_floats, state_bounds, state_flags, counts = ctj.cutile_call(
        (block_count,),
        _prepare_count_kernel,
        (
            means2d,
            radii,
            depths,
            conics,
            opacities,
            valid.astype(jnp.uint8),
            ctj.OutputPlaceholder((gaussian_count, _STATE_FLOATS), jnp.float32),
            ctj.OutputPlaceholder((gaussian_count, _STATE_BOUNDS), jnp.int32),
            ctj.OutputPlaceholder((gaussian_count, _STATE_FLAGS), jnp.uint8),
            ctj.OutputPlaceholder((gaussian_count,), jnp.int32),
            gaussian_count,
            tile_size,
            tile_width,
            tile_height,
            alpha_threshold,
            # AccuTile walks the shorter clipped span, bounded by either grid axis.
            min(tile_width, tile_height),
            count_block,
        ),
    )
    cumulative, valid_count, overflow, required_count = _prefix_counts_cutile(
        counts, capacity=capacity
    )
    if capacity == 0:
        empty = jnp.zeros((0,), jnp.int32)
        offsets = jnp.zeros((tile_height * tile_width,), jnp.int32)
        return empty, empty, offsets, valid_count, overflow, required_count
    state = _native_state(state_floats, state_bounds, state_flags)
    keys, values = _emit_accutile_cutile(
        state,
        cumulative,
        valid_count,
        depths,
        capacity=capacity,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        encode_sort_keys=True,
    )
    gaussian_ids, tile_ids, offsets, valid_count = _sort_prepared_offsets_cutile(
        keys,
        values,
        valid_count,
        tile_count=tile_width * tile_height,
    )
    return (
        gaussian_ids,
        tile_ids,
        offsets,
        valid_count,
        overflow,
        required_count,
    )
