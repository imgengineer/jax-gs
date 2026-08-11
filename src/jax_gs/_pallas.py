from __future__ import annotations

from functools import partial
import math
import operator

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from .low_level import (
    DEFAULT_ALPHA_THRESHOLD,
    DEFAULT_TRANSMITTANCE_THRESHOLD,
    MAX_ALPHA,
)

_TILE_AXIS_NAME = "jax_gs_pallas_tile"


def _static_int(name: str, value: int, *, minimum: int = 0) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a static integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _compositor_kernel(
    *,
    gaussian_count: int,
    input_capacity: int,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_width: int,
    tile_count: int,
    pixel_count: int,
    tile_pixel_count: int,
    channels: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
    named_grid: bool,
):
    def kernel(
        means_ref,
        conics_ref,
        colors_ref,
        opacities_ref,
        offsets_ref,
        flatten_ids_ref,
        valid_count_ref,
        rendered_ref,
        alpha_ref,
        transmittance_ref,
        overflow_ref,
    ):
        tile_id = (
            jax.lax.axis_index(_TILE_AXIS_NAME)
            if named_grid
            else pl.program_id(0)
        )
        start = offsets_ref[tile_id]
        end = jnp.where(
            tile_id + 1 < tile_count,
            offsets_ref[jnp.minimum(tile_id + 1, tile_count - 1)],
            valid_count_ref[...],
        )
        candidate_count = jnp.maximum(end - start, 0)

        tile_y = tile_id // tile_width
        tile_x = tile_id % tile_width
        local_pixel = jnp.arange(pixel_count, dtype=jnp.int32)
        if named_grid:
            pixel_layout = plgpu.Layout.WG_STRIDED(
                (pixel_count,), vec_size=1
            )
            local_pixel = plgpu.layout_cast(
                local_pixel,
                pixel_layout,
            )
        pixel_x = tile_x * tile_size + local_pixel % tile_size
        pixel_y = tile_y * tile_size + local_pixel // tile_size
        pixel_valid = (
            (local_pixel < tile_pixel_count)
            & (pixel_x < image_width)
            & (pixel_y < image_height)
        )
        pixel_x = pixel_x.astype(jnp.float32) + 0.5
        pixel_y = pixel_y.astype(jnp.float32) + 0.5

        def pixel_fill(value):
            filled = jnp.full((pixel_count,), value, jnp.float32)
            if named_grid:
                filled = plgpu.layout_cast(filled, pixel_layout)
            return filled

        initial = (
            tuple(pixel_fill(0.0) for _ in range(channels)),
            pixel_fill(0.0),
            pixel_fill(1.0),
        )

        def composite_one(candidate_index, carry):
            rendered_channels, accumulated_alpha, transmittance = carry
            position = start + candidate_index
            safe_position = jnp.clip(position, 0, input_capacity - 1)
            gaussian_id = flatten_ids_ref[safe_position]
            safe_gaussian_id = jnp.clip(gaussian_id, 0, gaussian_count - 1)

            delta_x = pixel_x - means_ref[safe_gaussian_id, 0]
            delta_y = pixel_y - means_ref[safe_gaussian_id, 1]
            sigma = (
                0.5
                * (
                    conics_ref[safe_gaussian_id, 0] * delta_x**2
                    + conics_ref[safe_gaussian_id, 2] * delta_y**2
                )
                + conics_ref[safe_gaussian_id, 1] * delta_x * delta_y
            )
            opacity = opacities_ref[safe_gaussian_id]
            alpha = jnp.minimum(opacity * jnp.exp(-sigma), MAX_ALPHA)
            valid = (
                (candidate_index < candidate_count)
                & (position < valid_count_ref[...])
                & (position < input_capacity)
                & (gaussian_id >= 0)
                & pixel_valid
                & (jnp.abs(sigma) <= jnp.finfo(jnp.float32).max)
                & (sigma >= 0.0)
                & (alpha >= alpha_threshold)
            )
            alpha = jnp.where(valid, alpha, 0.0)
            accepted = (
                transmittance * (1.0 - alpha) > transmittance_threshold
            )
            weight = jnp.where(accepted, alpha * transmittance, 0.0)
            rendered_channels = tuple(
                rendered_channel
                + weight * colors_ref[safe_gaussian_id, channel]
                for channel, rendered_channel in enumerate(rendered_channels)
            )
            accumulated_alpha = accumulated_alpha + weight
            transmittance = transmittance * (1.0 - alpha)
            return rendered_channels, accumulated_alpha, transmittance

        iterations = jnp.minimum(candidate_count, per_tile_bound)
        rendered_channels, accumulated_alpha, transmittance = jax.lax.fori_loop(
            0, iterations, composite_one, initial
        )
        for channel, rendered_channel in enumerate(rendered_channels):
            rendered_ref[tile_id, channel, :] = jnp.where(
                pixel_valid, rendered_channel, 0.0
            )
        alpha_ref[tile_id, :] = jnp.where(
            pixel_valid, accumulated_alpha, 0.0
        )
        transmittance_ref[tile_id, :] = jnp.where(
            pixel_valid, transmittance, 1.0
        )
        overflow_ref[tile_id] = candidate_count > per_tile_bound

    return kernel


def _compositor_backward_kernel(
    *,
    gaussian_count: int,
    input_capacity: int,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_width: int,
    tile_count: int,
    pixel_count: int,
    tile_pixel_count: int,
    channels: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
    named_grid: bool,
):
    """Build a reverse compositor that emits one gradient per intersection.

    A tile owns a contiguous intersection segment, so these writes are
    race-free. The custom VJP below performs the separate many-to-one scatter
    from intersection slots back to Gaussian rows.
    """

    def kernel(
        means_ref,
        conics_ref,
        colors_ref,
        opacities_ref,
        offsets_ref,
        flatten_ids_ref,
        valid_count_ref,
        final_transmittance_ref,
        rendered_cotangent_ref,
        alpha_cotangent_ref,
        means_cotangent_ref,
        conics_cotangent_ref,
        colors_cotangent_ref,
        opacities_cotangent_ref,
    ):
        tile_id = (
            jax.lax.axis_index(_TILE_AXIS_NAME)
            if named_grid
            else pl.program_id(0)
        )
        start = offsets_ref[tile_id]
        end = jnp.where(
            tile_id + 1 < tile_count,
            offsets_ref[jnp.minimum(tile_id + 1, tile_count - 1)],
            valid_count_ref[...],
        )
        candidate_count = jnp.maximum(end - start, 0)
        # Walk the stored overflow tail too, writing explicit zero gradients
        # for candidates that the bounded forward pass did not render.
        iterations = jnp.minimum(
            candidate_count, jnp.maximum(input_capacity - start, 0)
        )

        tile_y = tile_id // tile_width
        tile_x = tile_id % tile_width
        local_pixel = jnp.arange(pixel_count, dtype=jnp.int32)
        if named_grid:
            pixel_layout = plgpu.Layout.WG_STRIDED(
                (pixel_count,), vec_size=1
            )
            local_pixel = plgpu.layout_cast(local_pixel, pixel_layout)
        pixel_x = tile_x * tile_size + local_pixel % tile_size
        pixel_y = tile_y * tile_size + local_pixel // tile_size
        pixel_valid = (
            (local_pixel < tile_pixel_count)
            & (pixel_x < image_width)
            & (pixel_y < image_height)
        )
        pixel_x = pixel_x.astype(jnp.float32) + 0.5
        pixel_y = pixel_y.astype(jnp.float32) + 0.5

        def pixel_cast(value):
            if named_grid:
                value = plgpu.layout_cast(value, pixel_layout)
            return value

        def pixel_fill(value):
            return pixel_cast(
                jnp.full((pixel_count,), value, dtype=jnp.float32)
            )

        rendered_cotangents = tuple(
            pixel_cast(rendered_cotangent_ref[tile_id, channel, :])
            for channel in range(channels)
        )
        alpha_cotangent = pixel_cast(alpha_cotangent_ref[tile_id, :])
        initial = (
            pixel_cast(final_transmittance_ref[tile_id, :]),
            pixel_fill(0.0),
        )

        def composite_one_reverse(reverse_step, carry):
            transmittance_after, trailing_cotangent = carry
            candidate_index = iterations - 1 - reverse_step
            position = start + candidate_index
            safe_position = jnp.clip(position, 0, input_capacity - 1)
            gaussian_id = flatten_ids_ref[safe_position]
            safe_gaussian_id = jnp.clip(gaussian_id, 0, gaussian_count - 1)

            delta_x = pixel_x - means_ref[safe_gaussian_id, 0]
            delta_y = pixel_y - means_ref[safe_gaussian_id, 1]
            conic_x = conics_ref[safe_gaussian_id, 0]
            conic_xy = conics_ref[safe_gaussian_id, 1]
            conic_y = conics_ref[safe_gaussian_id, 2]
            sigma = (
                0.5 * (conic_x * delta_x**2 + conic_y * delta_y**2)
                + conic_xy * delta_x * delta_y
            )
            exponential = jnp.exp(-sigma)
            raw_alpha = opacities_ref[safe_gaussian_id] * exponential
            clamped_alpha = jnp.minimum(raw_alpha, MAX_ALPHA)
            candidate_valid = (
                (candidate_index < per_tile_bound)
                & (position < valid_count_ref[...])
                & (position < input_capacity)
                & (gaussian_id >= 0)
            )
            alpha_valid = (
                candidate_valid
                & pixel_valid
                & (jnp.abs(sigma) <= jnp.finfo(jnp.float32).max)
                & (sigma >= 0.0)
                & (clamped_alpha >= alpha_threshold)
            )
            alpha = jnp.where(alpha_valid, clamped_alpha, 0.0)
            one_minus_alpha = 1.0 - alpha
            transmittance_before = transmittance_after / one_minus_alpha
            accepted = (
                transmittance_before * one_minus_alpha
                > transmittance_threshold
            )
            weight = jnp.where(
                accepted, alpha * transmittance_before, 0.0
            )

            weight_cotangent = alpha_cotangent
            for channel, rendered_cotangent in enumerate(
                rendered_cotangents
            ):
                weight_cotangent = (
                    weight_cotangent
                    + rendered_cotangent
                    * colors_ref[safe_gaussian_id, channel]
                )
            # ``trailing_cotangent`` is the suffix sum of later weight
            # cotangents. Recovering T_before from T_after avoids retaining a
            # candidate-by-pixel transmittance tensor in the forward pass.
            alpha_chain_cotangent = jnp.where(
                accepted,
                weight_cotangent * transmittance_before,
                0.0,
            ) - trailing_cotangent / one_minus_alpha

            clamp_cotangent = jnp.where(
                raw_alpha < MAX_ALPHA,
                1.0,
                jnp.where(raw_alpha == MAX_ALPHA, 0.5, 0.0),
            )
            raw_alpha_cotangent = jnp.where(
                alpha_valid,
                alpha_chain_cotangent * clamp_cotangent,
                0.0,
            )
            differentiable_alpha = alpha_valid & (clamp_cotangent != 0.0)
            opacity_pixel_cotangent = jnp.where(
                differentiable_alpha,
                raw_alpha_cotangent * exponential,
                0.0,
            )
            sigma_cotangent = jnp.where(
                differentiable_alpha,
                -raw_alpha_cotangent * raw_alpha,
                0.0,
            )

            means_cotangent_ref[position, 0] = jnp.sum(
                -sigma_cotangent
                * (conic_x * delta_x + conic_xy * delta_y)
            )
            means_cotangent_ref[position, 1] = jnp.sum(
                -sigma_cotangent
                * (conic_y * delta_y + conic_xy * delta_x)
            )
            conics_cotangent_ref[position, 0] = jnp.sum(
                sigma_cotangent * 0.5 * delta_x**2
            )
            conics_cotangent_ref[position, 1] = jnp.sum(
                sigma_cotangent * delta_x * delta_y
            )
            conics_cotangent_ref[position, 2] = jnp.sum(
                sigma_cotangent * 0.5 * delta_y**2
            )
            for channel, rendered_cotangent in enumerate(
                rendered_cotangents
            ):
                colors_cotangent_ref[position, channel] = jnp.sum(
                    rendered_cotangent * weight
                )
            opacities_cotangent_ref[position] = jnp.sum(
                opacity_pixel_cotangent
            )

            trailing_cotangent = (
                trailing_cotangent + weight_cotangent * weight
            )
            return transmittance_before, trailing_cotangent

        jax.lax.fori_loop(
            0,
            iterations,
            composite_one_reverse,
            initial,
        )

    return kernel


def _launch_compositor_kernel(
    kernel,
    out_type,
    inputs,
    *,
    tile_count: int,
    interpret: bool,
    name: str,
):
    if interpret:
        gmem = pl.BlockSpec(memory_space=plgpu.GMEM)
        call = pl.pallas_call(
            kernel,
            out_shape=out_type,
            grid=(tile_count,),
            in_specs=(gmem,) * len(inputs),
            out_specs=(gmem,) * len(out_type),
            interpret=True,
            name=name,
        )
    else:
        call = plgpu.kernel(
            kernel,
            out_type=out_type,
            grid=(tile_count,),
            grid_names=(_TILE_AXIS_NAME,),
            compiler_params=plgpu.CompilerParams(
                lowering_semantics=plgpu.LoweringSemantics.Lane
            ),
        )
    return call(*inputs)


def _run_compositor_forward(
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
    tile_size: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
    interpret: bool,
):
    gaussian_count = means2d.shape[0]
    input_capacity = flatten_ids.shape[0]
    tile_height, tile_width = offsets.shape
    tile_count = tile_height * tile_width
    tile_pixel_count = tile_size * tile_size
    pixel_count = math.ceil(tile_pixel_count / 128) * 128
    channels = colors.shape[-1]
    kernel = _compositor_kernel(
        gaussian_count=gaussian_count,
        input_capacity=input_capacity,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_count=tile_count,
        pixel_count=pixel_count,
        tile_pixel_count=tile_pixel_count,
        channels=channels,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
        named_grid=not interpret,
    )
    out_type = (
        jax.ShapeDtypeStruct(
            (tile_count, channels, pixel_count), jnp.float32
        ),
        jax.ShapeDtypeStruct((tile_count, pixel_count), jnp.float32),
        jax.ShapeDtypeStruct((tile_count, pixel_count), jnp.float32),
        jax.ShapeDtypeStruct((tile_count,), jnp.bool_),
    )
    return _launch_compositor_kernel(
        kernel,
        out_type,
        (
            means2d,
            conics,
            colors,
            opacities,
            offsets.reshape(-1),
            flatten_ids,
            valid_count,
        ),
        tile_count=tile_count,
        interpret=interpret,
        name="jax_gs_pallas_compositor_forward",
    )


def _run_compositor_backward(
    means2d,
    conics,
    colors,
    opacities,
    offsets,
    flatten_ids,
    valid_count,
    final_transmittance,
    rendered_cotangent,
    alpha_cotangent,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
    interpret: bool,
):
    gaussian_count = means2d.shape[0]
    input_capacity = flatten_ids.shape[0]
    tile_height, tile_width = offsets.shape
    tile_count = tile_height * tile_width
    tile_pixel_count = tile_size * tile_size
    pixel_count = final_transmittance.shape[-1]
    channels = colors.shape[-1]
    kernel = _compositor_backward_kernel(
        gaussian_count=gaussian_count,
        input_capacity=input_capacity,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_count=tile_count,
        pixel_count=pixel_count,
        tile_pixel_count=tile_pixel_count,
        channels=channels,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
        named_grid=not interpret,
    )
    out_type = (
        jax.ShapeDtypeStruct((input_capacity, 2), jnp.float32),
        jax.ShapeDtypeStruct((input_capacity, 3), jnp.float32),
        jax.ShapeDtypeStruct((input_capacity, channels), jnp.float32),
        jax.ShapeDtypeStruct((input_capacity,), jnp.float32),
    )
    return _launch_compositor_kernel(
        kernel,
        out_type,
        (
            means2d,
            conics,
            colors,
            opacities,
            offsets.reshape(-1),
            flatten_ids,
            valid_count,
            final_transmittance,
            rendered_cotangent,
            alpha_cotangent,
        ),
        tile_count=tile_count,
        interpret=interpret,
        name="jax_gs_pallas_compositor_backward",
    )


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13))
def _composite_tiles(
    means2d,
    conics,
    colors,
    opacities,
    offsets,
    flatten_ids,
    valid_count,
    image_width: int,
    image_height: int,
    tile_size: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
    interpret: bool,
):
    rendered, alpha, _, overflow = _run_compositor_forward(
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
        interpret=interpret,
    )
    return rendered, alpha, overflow


def _composite_tiles_fwd(
    means2d,
    conics,
    colors,
    opacities,
    offsets,
    flatten_ids,
    valid_count,
    image_width: int,
    image_height: int,
    tile_size: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
    interpret: bool,
):
    rendered, alpha, final_transmittance, overflow = (
        _run_compositor_forward(
            means2d,
            conics,
            colors,
            opacities,
            offsets,
            flatten_ids,
            valid_count,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            per_tile_bound=per_tile_bound,
            alpha_threshold=alpha_threshold,
            transmittance_threshold=transmittance_threshold,
            interpret=interpret,
        )
    )
    residuals = (
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        final_transmittance,
    )
    return (rendered, alpha, overflow), residuals


def _composite_tiles_bwd(
    image_width: int,
    image_height: int,
    tile_size: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
    interpret: bool,
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
        final_transmittance,
    ) = residuals
    rendered_cotangent, alpha_cotangent, _ = cotangents
    slot_cotangents = _run_compositor_backward(
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        final_transmittance,
        rendered_cotangent,
        alpha_cotangent,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
        interpret=interpret,
    )

    input_capacity = flatten_ids.shape[0]
    gaussian_count = means2d.shape[0]
    position_valid = (
        jnp.arange(input_capacity, dtype=jnp.int32)
        < jnp.clip(valid_count, 0, input_capacity)
    ) & (flatten_ids >= 0)
    safe_ids = jnp.clip(flatten_ids, 0, gaussian_count - 1)

    # The same Gaussian can touch many tiles. Accumulate the race-free slot
    # results outside Pallas so the kernel needs no global atomics.
    gaussian_cotangents = []
    for source, slot_cotangent in zip(
        (means2d, conics, colors, opacities), slot_cotangents
    ):
        mask = position_valid.reshape(
            (input_capacity,) + (1,) * (slot_cotangent.ndim - 1)
        )
        slot_cotangent = jnp.where(mask, slot_cotangent, 0.0)
        gaussian_cotangents.append(
            jnp.zeros_like(source).at[safe_ids].add(slot_cotangent)
        )

    return (*gaussian_cotangents, None, None, None)


_composite_tiles.defvjp(_composite_tiles_fwd, _composite_tiles_bwd)


def rasterize_to_pixels_pallas(
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
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Composite one camera with a dynamic-trip-count Pallas GPU kernel.

    This is an experimental counterpart to
    :func:`jax_gs.low_level.rasterize_to_pixels`. Projection, intersection
    generation, sorting, overflow reporting, and the public renderer stay in
    JAX; one Pallas program owns one image tile and walks only that tile's real
    candidate count. ``interpret=True`` runs the same kernel through Pallas's
    CPU interpreter for correctness tests. Its custom VJP reverses each tile,
    emits one gradient per stored intersection, and then scatter-adds repeated
    Gaussian rows with ordinary JAX.
    """

    image_width = _static_int("image_width", image_width, minimum=1)
    image_height = _static_int("image_height", image_height, minimum=1)
    tile_size = _static_int("tile_size", tile_size, minimum=1)
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
    valid_count = jnp.asarray(valid_count, jnp.int32)
    gaussian_count = means2d.shape[0]
    channels = colors.shape[-1]
    input_capacity = flatten_ids.shape[0]

    if means2d.shape != (gaussian_count, 2):
        raise ValueError("means2d must have shape [N, 2]")
    if conics.shape != (gaussian_count, 3):
        raise ValueError("conics must have shape [N, 3]")
    if colors.shape != (gaussian_count, channels) or channels < 1:
        raise ValueError("colors must have shape [N, channels]")
    if opacities.shape != (gaussian_count,):
        raise ValueError("opacities must have shape [N]")
    if offsets.ndim != 2:
        raise ValueError("isect_offsets must have shape [tile_height, tile_width]")
    if flatten_ids.ndim != 1:
        raise ValueError("flatten_ids must be one-dimensional")
    if valid_count.shape != ():
        raise ValueError("valid_count must be a scalar")
    if any(
        value.dtype != jnp.float32
        for value in (means2d, conics, colors, opacities)
    ):
        raise TypeError("the Pallas compositor currently requires float32 inputs")

    tile_height, tile_width = offsets.shape
    if (
        tile_width * tile_size < image_width
        or tile_height * tile_size < image_height
    ):
        raise ValueError("isect_offsets tile grid does not cover the image")
    if backgrounds is None:
        backgrounds = jnp.zeros((channels,), jnp.float32)
    else:
        backgrounds = jnp.asarray(backgrounds)
        if backgrounds.shape != (channels,):
            raise ValueError("backgrounds must have shape [channels]")
        if backgrounds.dtype != jnp.float32:
            raise TypeError(
                "the Pallas compositor currently requires float32 inputs"
            )

    if gaussian_count == 0 or input_capacity == 0:
        rendered = jnp.broadcast_to(
            backgrounds[None, None, :], (image_height, image_width, channels)
        )
        alphas = jnp.zeros((image_height, image_width, 1), jnp.float32)
        tile_overflow = jnp.zeros(offsets.shape, jnp.bool_)
        return rendered, alphas, {
            "tile_overflow": tile_overflow,
            "overflow": jnp.asarray(overflow, jnp.bool_),
        }

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
                "the Pallas compositor requires an NVIDIA Hopper-or-newer "
                "GPU; use compositor_backend='jax' on this device"
            )

    per_tile_bound = min(gaussian_count, input_capacity)
    if max_candidates_per_tile is not None:
        per_tile_bound = min(per_tile_bound, max_candidates_per_tile)
    per_tile_bound = (
        math.ceil(per_tile_bound / max_gaussians_per_tile)
        * max_gaussians_per_tile
    )
    tile_pixel_count = tile_size * tile_size
    rendered_tiles, alpha_tiles, tile_overflow = _composite_tiles(
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        image_width,
        image_height,
        tile_size,
        per_tile_bound,
        float(alpha_threshold),
        float(transmittance_threshold),
        interpret,
    )
    rendered_tiles = rendered_tiles.transpose(0, 2, 1)[:, :tile_pixel_count]
    alpha_tiles = alpha_tiles[:, :tile_pixel_count]

    rendered = (
        rendered_tiles.reshape(
            tile_height, tile_width, tile_size, tile_size, channels
        )
        .transpose(0, 2, 1, 3, 4)
        .reshape(tile_height * tile_size, tile_width * tile_size, channels)
    )[:image_height, :image_width]
    alphas = (
        alpha_tiles.reshape(tile_height, tile_width, tile_size, tile_size)
        .transpose(0, 2, 1, 3)
        .reshape(tile_height * tile_size, tile_width * tile_size, 1)
    )[:image_height, :image_width]
    rendered = rendered + backgrounds[None, None, :] * (1.0 - alphas)
    tile_overflow = tile_overflow.reshape(offsets.shape)
    return rendered, alphas, {
        "tile_overflow": tile_overflow,
        "overflow": jnp.asarray(overflow, jnp.bool_) | jnp.any(tile_overflow),
    }


__all__ = ["rasterize_to_pixels_pallas"]
