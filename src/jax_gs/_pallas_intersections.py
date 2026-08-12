from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


_COUNT_AXIS_NAME = "jax_gs_pallas_accutile_count"
_COUNT_BLOCK_SIZE = 128
_EMIT_AXIS_NAME = "jax_gs_pallas_accutile_emit"
_EMIT_BLOCK_SIZE = 128


def _accutile_count_kernel(
    *,
    tile_size: int,
    outer_steps: int,
    named_grid: bool,
):
    def kernel(
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
    ):
        block_id = (
            jax.lax.axis_index(_COUNT_AXIS_NAME)
            if named_grid
            else pl.program_id(0)
        )
        start = block_id * _COUNT_BLOCK_SIZE
        layout = plgpu.Layout.WG_STRIDED((_COUNT_BLOCK_SIZE,), vec_size=1)

        def load(ref):
            value = ref.at[pl.ds(start, _COUNT_BLOCK_SIZE)][...]
            return plgpu.layout_cast(value, layout) if named_grid else value

        valid = load(valid_ref) != jnp.uint8(0)
        b = load(b_ref)
        disc = load(disc_ref)
        t = load(t_ref)
        p_u = load(p_u_ref)
        p_v = load(p_v_ref)
        coefficient = load(coefficient_ref)
        outer_min = load(outer_min_ref)
        outer_max = load(outer_max_ref)
        cross_min = load(cross_min_ref)
        cross_max = load(cross_max_ref)
        outer_bbox_min = load(outer_bbox_min_ref)
        outer_bbox_max = load(outer_bbox_max_ref)
        cross_bbox_min = load(cross_bbox_min_ref)
        cross_bbox_max = load(cross_bbox_max_ref)
        argmin_outer = load(argmin_outer_ref)
        argmax_outer = load(argmax_outer_ref)

        block = jnp.float32(tile_size)

        def ellipse_intersection(coordinate):
            h = coordinate - p_u
            radicand = jnp.maximum(
                disc * h * h + t * coefficient, 0.0
            )
            root = jnp.where(
                radicand > 0.0,
                radicand * jax.lax.rsqrt(radicand),
                0.0,
            )
            return (
                (-b * h - root) / coefficient + p_v,
                (-b * h + root) / coefficient + p_v,
            )

        min_line = outer_min.astype(jnp.float32) * block
        line_min, line_max = ellipse_intersection(min_line)
        intersects = outer_bbox_min <= min_line
        previous_min = jnp.where(intersects, line_min, cross_bbox_max)
        previous_max = jnp.where(intersects, line_max, cross_bbox_min)
        counts = jnp.zeros((_COUNT_BLOCK_SIZE,), dtype=jnp.int32)
        if named_grid:
            counts = plgpu.layout_cast(counts, layout)

        def count_outer(outer_offset, carry):
            previous_min, previous_max, counts = carry
            outer = outer_min + jnp.int32(outer_offset)
            active = valid & (outer < outer_max)
            min_line = outer.astype(jnp.float32) * block
            max_line = min_line + block
            line_min, line_max = ellipse_intersection(max_line)
            intersects = max_line <= outer_bbox_max
            current_min = jnp.where(intersects, line_min, previous_min)
            current_max = jnp.where(intersects, line_max, previous_max)
            ellipse_min = jnp.where(
                (min_line <= argmin_outer) & (argmin_outer < max_line),
                cross_bbox_min,
                jnp.minimum(previous_min, current_min),
            )
            ellipse_max = jnp.where(
                (min_line <= argmax_outer) & (argmax_outer < max_line),
                cross_bbox_max,
                jnp.maximum(previous_max, current_max),
            )
            min_v = jnp.maximum(
                cross_min,
                jnp.minimum(
                    cross_max, (ellipse_min / block).astype(jnp.int32)
                ),
            )
            max_v = jnp.minimum(
                cross_max,
                jnp.maximum(
                    cross_min,
                    (ellipse_max / block + 1.0).astype(jnp.int32),
                ),
            )
            counts = counts + jnp.where(active, max_v - min_v, 0)
            previous_min = jnp.where(active, current_min, previous_min)
            previous_max = jnp.where(active, current_max, previous_max)
            return previous_min, previous_max, counts

        _, _, counts = jax.lax.fori_loop(
            0,
            outer_steps,
            count_outer,
            (previous_min, previous_max, counts),
        )
        counts_ref.at[pl.ds(start, _COUNT_BLOCK_SIZE)][...] = counts

    return kernel


def count_accutile_intersections_pallas(
    state: Any,
    *,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    interpret: bool = False,
) -> jax.Array:
    """Count AccuTile pairs with one Mosaic program per 128 Gaussians.

    Geometry preparation, prefix sums, sorting, and offsets remain in JAX.
    The sequential outer-axis count scan is fused into one GPU kernel.
    """

    gaussian_count = state.valid.shape[0]
    if gaussian_count == 0:
        return jnp.zeros((0,), dtype=jnp.int32)
    if not interpret:
        device = jax.devices()[0]
        try:
            compute_capability = float(
                getattr(device, "compute_capability", 0.0)
            )
        except (TypeError, ValueError):
            compute_capability = 0.0
        if device.platform != "gpu" or compute_capability < 9.0:
            raise RuntimeError(
                "the Pallas AccuTile counter requires an NVIDIA "
                "Hopper-or-newer GPU; use intersection_backend='jax' on "
                "this device"
            )

    padded_count = (
        math.ceil(gaussian_count / _COUNT_BLOCK_SIZE) * _COUNT_BLOCK_SIZE
    )
    padding = padded_count - gaussian_count
    fields = (
        (state.valid.astype(jnp.uint8), 0),
        (state.b, 0.0),
        (state.disc, -1.0),
        (state.t, 1.0),
        (state.p_u, 0.0),
        (state.p_v, 0.0),
        (state.coefficient, 1.0),
        (state.outer_min, 0),
        (state.outer_max, 0),
        (state.cross_min, 0),
        (state.cross_max, 0),
        (state.outer_bbox_min, 0.0),
        (state.outer_bbox_max, 0.0),
        (state.cross_bbox_min, 0.0),
        (state.cross_bbox_max, 0.0),
        (state.argmin_outer, 0.0),
        (state.argmax_outer, 0.0),
    )
    inputs = tuple(
        value
        if padding == 0
        else jnp.pad(value, (0, padding), constant_values=fill)
        for value, fill in fields
    )
    kernel = _accutile_count_kernel(
        tile_size=tile_size,
        outer_steps=max(tile_width, tile_height),
        named_grid=not interpret,
    )
    out_type = jax.ShapeDtypeStruct((padded_count,), jnp.int32)
    block_count = padded_count // _COUNT_BLOCK_SIZE
    if interpret:
        gmem = pl.BlockSpec(memory_space=plgpu.GMEM)
        call = pl.pallas_call(
            kernel,
            out_shape=out_type,
            grid=(block_count,),
            in_specs=(gmem,) * len(inputs),
            out_specs=gmem,
            interpret=True,
            name="jax_gs_pallas_accutile_count",
        )
    else:
        call = plgpu.kernel(
            kernel,
            out_type=out_type,
            grid=(block_count,),
            grid_names=(_COUNT_AXIS_NAME,),
            compiler_params=plgpu.CompilerParams(
                lowering_semantics=plgpu.LoweringSemantics.Lane
            ),
        )
    return call(*inputs)[:gaussian_count]


def _accutile_emit_kernel(
    *,
    tile_size: int,
    tile_width: int,
    outer_steps: int,
    named_grid: bool,
):
    def kernel(
        owner_ref,
        local_ref,
        output_valid_ref,
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
        gaussian_ids_ref,
        tile_ids_ref,
    ):
        block_id = (
            jax.lax.axis_index(_EMIT_AXIS_NAME)
            if named_grid
            else pl.program_id(0)
        )
        start = block_id * _EMIT_BLOCK_SIZE
        layout = plgpu.Layout.WG_STRIDED((_EMIT_BLOCK_SIZE,), vec_size=1)

        def load(ref):
            value = ref.at[pl.ds(start, _EMIT_BLOCK_SIZE)][...]
            return plgpu.layout_cast(value, layout) if named_grid else value

        owner = load(owner_ref)
        local = load(local_ref)
        output_valid = load(output_valid_ref) != jnp.uint8(0)
        is_y = load(is_y_ref) != jnp.uint8(0)
        b = load(b_ref)
        disc = load(disc_ref)
        t = load(t_ref)
        p_u = load(p_u_ref)
        p_v = load(p_v_ref)
        coefficient = load(coefficient_ref)
        outer_min = load(outer_min_ref)
        outer_max = load(outer_max_ref)
        cross_min = load(cross_min_ref)
        cross_max = load(cross_max_ref)
        outer_bbox_min = load(outer_bbox_min_ref)
        outer_bbox_max = load(outer_bbox_max_ref)
        cross_bbox_min = load(cross_bbox_min_ref)
        cross_bbox_max = load(cross_bbox_max_ref)
        argmin_outer = load(argmin_outer_ref)
        argmax_outer = load(argmax_outer_ref)

        block = jnp.float32(tile_size)

        def ellipse_intersection(coordinate):
            h = coordinate - p_u
            radicand = jnp.maximum(
                disc * h * h + t * coefficient, 0.0
            )
            root = jnp.where(
                radicand > 0.0,
                radicand * jax.lax.rsqrt(radicand),
                0.0,
            )
            return (
                (-b * h - root) / coefficient + p_v,
                (-b * h + root) / coefficient + p_v,
            )

        min_line = outer_min.astype(jnp.float32) * block
        line_min, line_max = ellipse_intersection(min_line)
        intersects = outer_bbox_min <= min_line
        previous_min = jnp.where(intersects, line_min, cross_bbox_max)
        previous_max = jnp.where(intersects, line_max, cross_bbox_min)
        emitted = jnp.zeros((_EMIT_BLOCK_SIZE,), dtype=jnp.int32)
        selected_cross = jnp.zeros((_EMIT_BLOCK_SIZE,), dtype=jnp.int32)
        selected_outer = outer_min
        found = jnp.zeros((_EMIT_BLOCK_SIZE,), dtype=jnp.bool_)
        if named_grid:
            emitted = plgpu.layout_cast(emitted, layout)
            selected_cross = plgpu.layout_cast(selected_cross, layout)
            found = plgpu.layout_cast(found, layout)

        def emit_outer(outer_offset, carry):
            (
                previous_min,
                previous_max,
                emitted,
                selected_cross,
                selected_outer,
                found,
            ) = carry
            outer = outer_min + jnp.int32(outer_offset)
            active = outer < outer_max
            min_line = outer.astype(jnp.float32) * block
            max_line = min_line + block
            line_min, line_max = ellipse_intersection(max_line)
            intersects = max_line <= outer_bbox_max
            current_min = jnp.where(intersects, line_min, previous_min)
            current_max = jnp.where(intersects, line_max, previous_max)
            ellipse_min = jnp.where(
                (min_line <= argmin_outer) & (argmin_outer < max_line),
                cross_bbox_min,
                jnp.minimum(previous_min, current_min),
            )
            ellipse_max = jnp.where(
                (min_line <= argmax_outer) & (argmax_outer < max_line),
                cross_bbox_max,
                jnp.maximum(previous_max, current_max),
            )
            min_v = jnp.maximum(
                cross_min,
                jnp.minimum(
                    cross_max, (ellipse_min / block).astype(jnp.int32)
                ),
            )
            max_v = jnp.minimum(
                cross_max,
                jnp.maximum(
                    cross_min,
                    (ellipse_max / block + 1.0).astype(jnp.int32),
                ),
            )
            span = jnp.where(active, max_v - min_v, 0)
            covered = (
                output_valid
                & ~found
                & (local >= emitted)
                & (local < emitted + span)
            )
            selected_cross = jnp.where(
                covered, min_v + local - emitted, selected_cross
            )
            selected_outer = jnp.where(covered, outer, selected_outer)
            found = found | covered
            emitted = emitted + span
            previous_min = jnp.where(active, current_min, previous_min)
            previous_max = jnp.where(active, current_max, previous_max)
            return (
                previous_min,
                previous_max,
                emitted,
                selected_cross,
                selected_outer,
                found,
            )

        (
            _,
            _,
            _,
            selected_cross,
            selected_outer,
            found,
        ) = jax.lax.fori_loop(
            0,
            outer_steps,
            emit_outer,
            (
                previous_min,
                previous_max,
                emitted,
                selected_cross,
                selected_outer,
                found,
            ),
        )
        tile_ids = jnp.where(
            is_y,
            selected_outer * jnp.int32(tile_width) + selected_cross,
            selected_cross * jnp.int32(tile_width) + selected_outer,
        )
        valid = output_valid & found
        gaussian_ids = jnp.where(valid, owner, jnp.int32(-1))
        tile_ids = jnp.where(valid, tile_ids, jnp.int32(-1))
        gaussian_ids_ref.at[pl.ds(start, _EMIT_BLOCK_SIZE)][...] = gaussian_ids
        tile_ids_ref.at[pl.ds(start, _EMIT_BLOCK_SIZE)][...] = tile_ids

    return kernel


def emit_accutile_intersections_pallas(
    state: Any,
    cumulative: jax.Array,
    valid_count: jax.Array,
    *,
    capacity: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Emit Gaussian-major AccuTile pairs with a Mosaic output-slot scan.

    JAX marks each Gaussian's first retained slot and prefix-propagates those
    owners across the fixed output capacity. One Mosaic program then walks the
    owner ellipse for 128 slots and writes the corresponding Gaussian/tile
    pairs without a JAX kernel per outer-axis step. Sorting and offset
    construction remain in JAX.
    """

    if capacity == 0:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        return empty, empty
    if not interpret:
        device = jax.devices()[0]
        try:
            compute_capability = float(
                getattr(device, "compute_capability", 0.0)
            )
        except (TypeError, ValueError):
            compute_capability = 0.0
        if device.platform != "gpu" or compute_capability < 9.0:
            raise RuntimeError(
                "Pallas AccuTile emission requires an NVIDIA "
                "Hopper-or-newer GPU; use intersection_backend='jax' on "
                "this device"
            )

    gaussian_count = state.valid.shape[0]
    ranks = jnp.arange(capacity, dtype=jnp.int32)
    starts = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.int32), cumulative[:-1])
    )
    safe_starts = jnp.clip(starts, 0, capacity - 1)
    owner_markers = jnp.zeros((capacity,), dtype=jnp.int32).at[
        safe_starts
    ].max(
        jnp.where(
            starts < valid_count,
            jnp.arange(gaussian_count, dtype=jnp.int32) + 1,
            0,
        )
    )
    owner = jax.lax.associative_scan(jnp.maximum, owner_markers) - 1
    owner = jnp.clip(owner, 0, gaussian_count - 1)
    local = ranks - starts[owner]
    output_valid = ranks < valid_count

    padded_capacity = (
        math.ceil(capacity / _EMIT_BLOCK_SIZE) * _EMIT_BLOCK_SIZE
    )
    padding = padded_capacity - capacity
    fields = (
        (owner, 0),
        (local, 0),
        (output_valid.astype(jnp.uint8), 0),
        (state.is_y[owner].astype(jnp.uint8), 0),
        (state.b[owner], 0.0),
        (state.disc[owner], -1.0),
        (state.t[owner], 1.0),
        (state.p_u[owner], 0.0),
        (state.p_v[owner], 0.0),
        (state.coefficient[owner], 1.0),
        (state.outer_min[owner], 0),
        (state.outer_max[owner], 0),
        (state.cross_min[owner], 0),
        (state.cross_max[owner], 0),
        (state.outer_bbox_min[owner], 0.0),
        (state.outer_bbox_max[owner], 0.0),
        (state.cross_bbox_min[owner], 0.0),
        (state.cross_bbox_max[owner], 0.0),
        (state.argmin_outer[owner], 0.0),
        (state.argmax_outer[owner], 0.0),
    )
    inputs = tuple(
        value
        if padding == 0
        else jnp.pad(value, (0, padding), constant_values=fill)
        for value, fill in fields
    )
    kernel = _accutile_emit_kernel(
        tile_size=tile_size,
        tile_width=tile_width,
        outer_steps=max(tile_width, tile_height),
        named_grid=not interpret,
    )
    out_type = (
        jax.ShapeDtypeStruct((padded_capacity,), jnp.int32),
        jax.ShapeDtypeStruct((padded_capacity,), jnp.int32),
    )
    block_count = padded_capacity // _EMIT_BLOCK_SIZE
    if interpret:
        gmem = pl.BlockSpec(memory_space=plgpu.GMEM)
        call = pl.pallas_call(
            kernel,
            out_shape=out_type,
            grid=(block_count,),
            in_specs=(gmem,) * len(inputs),
            out_specs=(gmem, gmem),
            interpret=True,
            name="jax_gs_pallas_accutile_emit",
        )
    else:
        call = plgpu.kernel(
            kernel,
            out_type=out_type,
            grid=(block_count,),
            grid_names=(_EMIT_AXIS_NAME,),
            compiler_params=plgpu.CompilerParams(
                lowering_semantics=plgpu.LoweringSemantics.Lane
            ),
        )
    gaussian_ids, tile_ids = call(*inputs)
    return gaussian_ids[:capacity], tile_ids[:capacity]


__all__ = [
    "count_accutile_intersections_pallas",
    "emit_accutile_intersections_pallas",
]
