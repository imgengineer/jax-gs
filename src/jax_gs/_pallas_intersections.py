from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


_COUNT_AXIS_NAME = "jax_gs_pallas_accutile_count"
_COUNT_BLOCK_SIZE = 128


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

    The prepared geometry and all later emission/sort stages retain their JAX
    implementation. This deliberately small first intersection slice only
    fuses the sequential outer-axis count scan into one GPU kernel.
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


__all__ = ["count_accutile_intersections_pallas"]
