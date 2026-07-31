from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
import operator
from typing import Iterator, NamedTuple

import jax
import jax.numpy as jnp

from .intersections import _accutile_intersections_jax


MAX_ALPHA = 0.999
DEFAULT_ALPHA_THRESHOLD = 1.0 / 255.0
DEFAULT_TRANSMITTANCE_THRESHOLD = 1.0e-4


def _broadcast_means_per_pixel(
    means: jax.Array, pixel_template: jax.Array
) -> jax.Array:
    return jnp.broadcast_to(
        means[:, None, :],
        (means.shape[0], pixel_template.shape[0], means.shape[1]),
    )


@jax.custom_vjp
def _broadcast_means_with_absgrad_probe(
    means: jax.Array,
    absgrad_probe: jax.Array,
    pixel_template: jax.Array,
) -> jax.Array:
    """Broadcast means while retaining Gaussian-by-pixel VJP contributions.

    The ordinary input receives the signed sum over pixels. The zero-valued
    probe receives the sum of the componentwise absolute pixel contributions,
    matching gsplat's compositor-side AbsGrad reduction.
    """

    del absgrad_probe
    return _broadcast_means_per_pixel(means, pixel_template)


def _broadcast_means_with_absgrad_probe_fwd(
    means: jax.Array,
    absgrad_probe: jax.Array,
    pixel_template: jax.Array,
) -> tuple[jax.Array, None]:
    del absgrad_probe
    return _broadcast_means_per_pixel(means, pixel_template), None


def _broadcast_means_with_absgrad_probe_bwd(
    _residual: None, pixel_cotangent: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    signed_gradient = jnp.sum(pixel_cotangent, axis=1)
    absolute_gradient = jnp.sum(jnp.abs(pixel_cotangent), axis=1)
    pixel_template_gradient = jnp.zeros(
        pixel_cotangent.shape[1:], dtype=pixel_cotangent.dtype
    )
    return signed_gradient, absolute_gradient, pixel_template_gradient


_broadcast_means_with_absgrad_probe.defvjp(
    _broadcast_means_with_absgrad_probe_fwd,
    _broadcast_means_with_absgrad_probe_bwd,
)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _chunk_weights(
    alpha: jax.Array,
    incoming_transmittance: jax.Array,
    transmittance_threshold: float,
) -> tuple[jax.Array, jax.Array]:
    """Front-to-back weights for one chunk of depth-sorted candidates.

    ``alpha`` is ``[K, P]`` for K candidates over P pixels and
    ``incoming_transmittance`` is ``[P]``. Returns the compositing weights and
    the transmittance leaving the chunk.

    Reverse mode is written by hand. Differentiating the exclusive cumulative
    product generically is what makes every alpha-dependent gradient far more
    expensive than the colour path; the closed form below needs one suffix sum
    instead. Upstream's backward kernel recovers transmittance the same way, by
    dividing out ``1 - alpha``, which the ``MAX_ALPHA`` clamp keeps at or above
    1e-3. Invalid candidates arrive with ``alpha = 0``, so their divisor is one.
    """

    weights, outgoing, _ = _chunk_weights_with_residuals(
        alpha, incoming_transmittance, transmittance_threshold
    )
    return weights, outgoing


def _chunk_weights_with_residuals(
    alpha: jax.Array,
    incoming_transmittance: jax.Array,
    transmittance_threshold: float,
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
    one_minus_alpha = 1.0 - alpha
    exclusive = jnp.concatenate(
        (
            jnp.ones((1,) + alpha.shape[1:], dtype=alpha.dtype),
            jnp.cumprod(one_minus_alpha, axis=0)[:-1],
        ),
        axis=0,
    )
    transmittance = incoming_transmittance[None, :] * exclusive
    accepted = transmittance * one_minus_alpha > transmittance_threshold
    weights = jnp.where(accepted, alpha * transmittance, 0.0)
    chunk_product = jnp.prod(one_minus_alpha, axis=0)
    outgoing = incoming_transmittance * chunk_product
    residuals = (
        alpha,
        one_minus_alpha,
        exclusive,
        transmittance,
        accepted,
        weights,
        chunk_product,
        outgoing,
    )
    return weights, outgoing, residuals


def _chunk_weights_fwd(
    alpha: jax.Array,
    incoming_transmittance: jax.Array,
    transmittance_threshold: float,
):
    weights, outgoing, residuals = _chunk_weights_with_residuals(
        alpha, incoming_transmittance, transmittance_threshold
    )
    return (weights, outgoing), residuals


def _chunk_weights_bwd(_threshold: float, residuals, cotangents):
    (
        alpha,
        one_minus_alpha,
        exclusive,
        transmittance,
        accepted,
        weights,
        chunk_product,
        outgoing,
    ) = residuals
    weight_cotangent, outgoing_cotangent = cotangents

    # Every later candidate's weight carries this candidate's (1 - alpha)
    # factor, so its share is the strict suffix sum of the weight cotangents.
    scaled = weight_cotangent * weights
    inclusive_suffix = jnp.cumsum(scaled[::-1], axis=0)[::-1]
    strict_suffix = inclusive_suffix - scaled
    trailing = strict_suffix + outgoing_cotangent[None, :] * outgoing[None, :]

    direct = jnp.where(accepted, weight_cotangent * transmittance, 0.0)
    alpha_cotangent = direct - trailing / one_minus_alpha

    incoming_cotangent = jnp.sum(
        jnp.where(accepted, weight_cotangent * alpha * exclusive, 0.0), axis=0
    ) + outgoing_cotangent * chunk_product
    return alpha_cotangent, incoming_cotangent


_chunk_weights.defvjp(_chunk_weights_fwd, _chunk_weights_bwd)


def _raise_rasterization_overflow() -> None:
    raise RuntimeError(
        "low-level rasterization truncated its padded intersection buffer; "
        "increase the intersection capacity or request return_info=True and "
        "handle overflow"
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PaddedIntersections:
    """Static-shape result of :func:`isect_tiles`.

    ``isect_ids`` stores the two 32-bit words of gsplat's 64-bit key as
    ``[encoded_image_and_tile, float32_depth_bits]``. Invalid capacity slots
    are ``-1``. Keeping the words separate works when JAX's default x64 mode is
    disabled. Iteration yields gsplat's three public values; ``valid_count``
    and ``overflow`` remain available as JAX-specific attributes.
    """

    tiles_per_gaussian: jax.Array
    isect_ids: jax.Array
    flatten_ids: jax.Array
    valid_count: jax.Array
    overflow: jax.Array

    def __iter__(self) -> Iterator[jax.Array]:
        """Unpack like gsplat while retaining JAX capacity metadata."""

        yield self.tiles_per_gaussian
        yield self.isect_ids
        yield self.flatten_ids

    def tree_flatten(self):
        return (
            (
                self.tiles_per_gaussian,
                self.isect_ids,
                self.flatten_ids,
                self.valid_count,
                self.overflow,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


class PaddedOffsets(NamedTuple):
    """Offsets plus the metadata associated with their padded intersections."""

    offsets: jax.Array
    valid_count: jax.Array
    overflow: jax.Array


class PaddedRasterizationIndices(NamedTuple):
    """Static-shape pixel/Gaussian intersection indices."""

    gaussian_ids: jax.Array
    pixel_ids: jax.Array
    image_ids: jax.Array
    valid_count: jax.Array
    overflow: jax.Array


def _as_static_int(name: str, value: int, *, minimum: int = 0) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a static integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _bits_for_count(count: int) -> int:
    return max(1, (count - 1).bit_length())


def _saturating_cumsum(values: jax.Array, limit: int) -> jax.Array:
    """Cumulative sum capped at a small static limit without int32 overflow."""

    values = jnp.minimum(values.astype(jnp.int32), jnp.int32(limit))

    def combine(left: jax.Array, right: jax.Array) -> jax.Array:
        return jnp.minimum(left + right, jnp.int32(limit))

    return jax.lax.associative_scan(combine, values)


def _encode_high_word(
    image_ids: jax.Array, tile_ids: jax.Array, tile_bits: int
) -> jax.Array:
    high = (image_ids.astype(jnp.uint32) << jnp.uint32(tile_bits)) | tile_ids.astype(
        jnp.uint32
    )
    return jax.lax.bitcast_convert_type(high, jnp.int32)


def _decode_high_words(
    isect_ids: jax.Array, tile_bits: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    isect_ids = jnp.asarray(isect_ids)
    if isect_ids.ndim == 2 and isect_ids.shape[-1] == 2:
        high = isect_ids[:, 0]
        slot_valid = high != -1
    elif isect_ids.ndim == 1:
        if isect_ids.dtype in (jnp.int64, jnp.uint64):
            high_u32 = (isect_ids.astype(jnp.uint64) >> jnp.uint64(32)).astype(
                jnp.uint32
            )
            high = jax.lax.bitcast_convert_type(high_u32, jnp.int32)
        else:
            high = isect_ids.astype(jnp.int32)
        slot_valid = high != -1
    else:
        raise ValueError("isect_ids must have shape [K] or [K, 2]")

    high_u32 = jax.lax.bitcast_convert_type(high, jnp.uint32)
    image_ids = (high_u32 >> jnp.uint32(tile_bits)).astype(jnp.int32)
    tile_ids = (high_u32 & jnp.uint32((1 << tile_bits) - 1)).astype(jnp.int32)
    return image_ids, tile_ids, slot_valid


def _normalize_valid_count(
    valid_count: jax.Array | int | None, capacity: int
) -> jax.Array:
    if valid_count is None:
        return jnp.asarray(capacity, dtype=jnp.int32)
    return jnp.clip(jnp.asarray(valid_count, dtype=jnp.int32), 0, capacity)


def isect_tiles(
    means2d: jax.Array,
    radii: jax.Array,
    depths: jax.Array,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: int | None = None,
    image_ids: jax.Array | None = None,
    gaussian_ids: jax.Array | None = None,
    conics: jax.Array | None = None,
    opacities: jax.Array | None = None,
    *,
    max_intersections: int | None = None,
    active_mask: jax.Array | None = None,
) -> PaddedIntersections:
    """Map Gaussians to tiles using a fixed-capacity intersection buffer.

    The first ``valid_count`` entries are sorted by image, tile and depth.
    ``overflow`` is true when the exact number of tile intersections exceeds
    ``max_intersections``. The default capacity is one slot per input Gaussian;
    callers that require complete results should pass an explicit larger static
    capacity and handle ``overflow``.

    When both ``conics`` and ``opacities`` are provided, the mapping follows
    gsplat's opacity-aware AccuTile ellipse walk. Otherwise it uses the
    conservative radius AABB.
    """

    del segmented, gaussian_ids
    tile_size = _as_static_int("tile_size", tile_size, minimum=1)
    tile_width = _as_static_int("tile_width", tile_width, minimum=1)
    tile_height = _as_static_int("tile_height", tile_height, minimum=1)
    means2d = jnp.asarray(means2d)
    radii = jnp.asarray(radii)
    depths = jnp.asarray(depths)
    if (conics is None) != (opacities is None):
        raise ValueError("conics and opacities must be provided together")

    packed = bool(packed or means2d.ndim == 2)
    if packed:
        if means2d.ndim != 2 or means2d.shape[-1] != 2:
            raise ValueError("packed means2d must have shape [nnz, 2]")
        nnz = means2d.shape[0]
        if radii.shape != (nnz, 2) or depths.shape != (nnz,):
            raise ValueError("packed radii/depths shapes do not match means2d")
        if n_images is None or image_ids is None:
            raise ValueError("packed mode requires n_images and image_ids")
        image_count = _as_static_int("n_images", n_images, minimum=1)
        image_of = jnp.asarray(image_ids, dtype=jnp.int32)
        if image_of.shape != (nnz,):
            raise ValueError("image_ids must have shape [nnz]")
        flat_means = means2d
        flat_radii = radii
        flat_depths = depths
        if conics is None:
            flat_conics = None
            flat_opacities = None
        else:
            flat_conics = jnp.asarray(conics)
            flat_opacities = jnp.asarray(opacities)
            if flat_conics.shape != (nnz, 3):
                raise ValueError("packed conics must have shape [nnz, 3]")
            if flat_opacities.shape != (nnz,):
                raise ValueError("packed opacities must have shape [nnz]")
        output_shape = (nnz,)
        if active_mask is None:
            flat_active = jnp.ones((nnz,), dtype=jnp.bool_)
        else:
            flat_active = jnp.asarray(active_mask, dtype=jnp.bool_)
            if flat_active.shape != (nnz,):
                raise ValueError("packed active_mask must have shape [nnz]")
    else:
        if means2d.ndim < 3 or means2d.shape[-1] != 2:
            raise ValueError("means2d must have shape [..., N, 2]")
        image_shape = means2d.shape[:-2]
        gaussian_count = means2d.shape[-2]
        if radii.shape != image_shape + (gaussian_count, 2):
            raise ValueError("radii shape does not match means2d")
        if depths.shape != image_shape + (gaussian_count,):
            raise ValueError("depths shape does not match means2d")
        image_count = math.prod(image_shape)
        if image_count < 1:
            raise ValueError("means2d must contain at least one image")
        if n_images is not None and n_images != image_count:
            raise ValueError("n_images does not match the dense leading dimensions")
        flat_means = means2d.reshape(-1, 2)
        flat_radii = radii.reshape(-1, 2)
        flat_depths = depths.reshape(-1)
        if conics is None:
            flat_conics = None
            flat_opacities = None
        else:
            conics = jnp.asarray(conics)
            opacities = jnp.asarray(opacities)
            if conics.shape != image_shape + (gaussian_count, 3):
                raise ValueError("conics shape does not match means2d")
            if opacities.shape != image_shape + (gaussian_count,):
                raise ValueError("opacities shape does not match means2d")
            flat_conics = conics.reshape(-1, 3)
            flat_opacities = opacities.reshape(-1)
        image_of = jnp.repeat(
            jnp.arange(image_count, dtype=jnp.int32), gaussian_count
        )
        output_shape = image_shape + (gaussian_count,)
        if active_mask is None:
            flat_active = jnp.ones((flat_means.shape[0],), dtype=jnp.bool_)
        else:
            active_mask = jnp.asarray(active_mask, dtype=jnp.bool_)
            if active_mask.shape == (gaussian_count,):
                active_mask = jnp.broadcast_to(
                    active_mask, image_shape + (gaussian_count,)
                )
            if active_mask.shape != image_shape + (gaussian_count,):
                raise ValueError("active_mask must have shape [N] or [..., N]")
            flat_active = active_mask.reshape(-1)

    flat_count = flat_means.shape[0]
    if max_intersections is None:
        max_intersections = flat_count
    capacity = _as_static_int("max_intersections", max_intersections)
    if capacity > (2**30 - 2):
        raise ValueError("max_intersections is too large for int32 indexing")

    tile_bits = _bits_for_count(tile_width * tile_height)
    if _bits_for_count(image_count) + tile_bits > 32:
        raise ValueError("image and tile ids do not fit in the 32-bit high word")

    if flat_count == 0:
        return PaddedIntersections(
            jnp.zeros(output_shape, dtype=jnp.int32),
            jnp.full((capacity, 2), -1, dtype=jnp.int32),
            jnp.full((capacity,), -1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
        )

    ranks = jnp.arange(capacity, dtype=jnp.int32)
    valid_gaussian = (
        flat_active
        & jnp.isfinite(flat_depths)
        & (image_of >= 0)
        & (image_of < image_count)
    )
    if flat_conics is not None:
        assert flat_opacities is not None
        (
            tiles_per_flat,
            source_slots,
            selected_tiles,
            valid_count,
            overflow,
            _,
        ) = _accutile_intersections_jax(
            flat_means,
            flat_radii,
            flat_conics,
            flat_opacities,
            valid_gaussian,
            capacity=capacity,
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            alpha_threshold=DEFAULT_ALPHA_THRESHOLD,
        )
        selected_valid = (ranks < valid_count) & (source_slots >= 0)
    else:
        tile_mins = jnp.floor((flat_means - flat_radii) / tile_size).astype(
            jnp.int32
        )
        tile_maxs = jnp.ceil((flat_means + flat_radii) / tile_size).astype(
            jnp.int32
        )
        tile_mins = jnp.stack(
            (
                jnp.clip(tile_mins[:, 0], 0, tile_width),
                jnp.clip(tile_mins[:, 1], 0, tile_height),
            ),
            axis=-1,
        )
        tile_maxs = jnp.stack(
            (
                jnp.clip(tile_maxs[:, 0], 0, tile_width),
                jnp.clip(tile_maxs[:, 1], 0, tile_height),
            ),
            axis=-1,
        )
        spans = jnp.maximum(tile_maxs - tile_mins, 0)
        valid_gaussian = (
            valid_gaussian
            & jnp.all(flat_radii > 0, axis=-1)
            & jnp.all(jnp.isfinite(flat_means), axis=-1)
            & jnp.all(jnp.isfinite(flat_radii), axis=-1)
        )
        tiles_per_flat = jnp.where(
            valid_gaussian, spans[:, 0] * spans[:, 1], 0
        ).astype(jnp.int32)
        cumulative = _saturating_cumsum(tiles_per_flat, capacity + 1)
        total = cumulative[-1]
        valid_count = jnp.minimum(total, jnp.int32(capacity))
        overflow = total > capacity
        source_slots = jnp.searchsorted(cumulative, ranks, side="right")
        source_slots = jnp.clip(source_slots, 0, flat_count - 1)
        previous = jnp.where(
            source_slots > 0,
            cumulative[jnp.maximum(source_slots - 1, 0)],
            0,
        )
        local_ids = ranks - previous
        span_x = jnp.maximum(spans[source_slots, 0], 1)
        tile_x = tile_mins[source_slots, 0] + local_ids % span_x
        tile_y = tile_mins[source_slots, 1] + local_ids // span_x
        selected_tiles = tile_y * tile_width + tile_x
        selected_valid = ranks < valid_count

    tiles_per_gaussian = tiles_per_flat.reshape(output_shape)
    safe_source_slots = jnp.clip(source_slots, 0, flat_count - 1)
    selected_images = image_of[safe_source_slots]
    selected_depths = flat_depths[safe_source_slots].astype(jnp.float32)
    global_tiles = selected_images * (tile_width * tile_height) + selected_tiles

    if sort and capacity:
        order = jnp.lexsort(
            (
                source_slots,
                selected_depths,
                global_tiles,
                (~selected_valid).astype(jnp.int32),
            )
        )
        source_slots = source_slots[order]
        selected_images = selected_images[order]
        selected_tiles = selected_tiles[order]
        selected_depths = selected_depths[order]
        selected_valid = selected_valid[order]

    high_words = _encode_high_word(selected_images, selected_tiles, tile_bits)
    depth_words = jax.lax.bitcast_convert_type(selected_depths, jnp.int32)
    isect_ids = jnp.stack((high_words, depth_words), axis=-1)
    isect_ids = jnp.where(selected_valid[:, None], isect_ids, -1)
    flatten_ids = jnp.where(
        selected_valid, source_slots.astype(jnp.int32), -1
    )
    return PaddedIntersections(
        tiles_per_gaussian, isect_ids, flatten_ids, valid_count, overflow
    )


def isect_offset_encode(
    isect_ids: jax.Array,
    n_images: int,
    tile_width: int,
    tile_height: int,
    *,
    valid_count: jax.Array | int | None = None,
    overflow: jax.Array | bool = False,
    return_info: bool = False,
) -> jax.Array | PaddedOffsets:
    """Encode padded, sorted intersection ids as fixed-shape tile offsets."""

    n_images = _as_static_int("n_images", n_images, minimum=1)
    tile_width = _as_static_int("tile_width", tile_width, minimum=1)
    tile_height = _as_static_int("tile_height", tile_height, minimum=1)
    isect_ids = jnp.asarray(isect_ids)
    capacity = isect_ids.shape[0]
    valid_count = _normalize_valid_count(valid_count, capacity)
    tile_bits = _bits_for_count(tile_width * tile_height)
    image_ids, tile_ids, slot_valid = _decode_high_words(isect_ids, tile_bits)
    positions = jnp.arange(capacity, dtype=jnp.int32)
    valid = (
        (positions < valid_count)
        & slot_valid
        & (image_ids >= 0)
        & (image_ids < n_images)
        & (tile_ids >= 0)
        & (tile_ids < tile_width * tile_height)
    )
    dense_ids = image_ids * (tile_width * tile_height) + tile_ids
    safe_ids = jnp.clip(dense_ids, 0, n_images * tile_width * tile_height - 1)
    counts = jnp.zeros(
        (n_images * tile_width * tile_height,), dtype=jnp.int32
    ).at[safe_ids].add(valid.astype(jnp.int32))
    offsets = (jnp.cumsum(counts) - counts).reshape(
        n_images, tile_height, tile_width
    )
    if return_info:
        return PaddedOffsets(
            offsets, valid_count, jnp.asarray(overflow, dtype=jnp.bool_)
        )
    return offsets


def _unwrap_offsets(
    isect_offsets: jax.Array | PaddedOffsets,
    valid_count: jax.Array | int | None,
    overflow: jax.Array | bool,
    capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if isinstance(isect_offsets, PaddedOffsets):
        if valid_count is None:
            valid_count = isect_offsets.valid_count
        overflow = jnp.asarray(overflow, dtype=jnp.bool_) | isect_offsets.overflow
        isect_offsets = isect_offsets.offsets
    return (
        jnp.asarray(isect_offsets, dtype=jnp.int32),
        _normalize_valid_count(valid_count, capacity),
        jnp.asarray(overflow, dtype=jnp.bool_),
    )


def _validate_dense_raster_inputs(
    means2d: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    image_width: int,
    image_height: int,
    isect_offsets: jax.Array,
) -> tuple[tuple[int, ...], int, int, int, int]:
    image_width = _as_static_int("image_width", image_width, minimum=1)
    image_height = _as_static_int("image_height", image_height, minimum=1)
    image_shape = means2d.shape[:-2]
    gaussian_count = means2d.shape[-2]
    if means2d.shape != image_shape + (gaussian_count, 2):
        raise ValueError("means2d must have shape [..., N, 2]")
    if conics.shape != image_shape + (gaussian_count, 3):
        raise ValueError("conics shape does not match means2d")
    if opacities.shape != image_shape + (gaussian_count,):
        raise ValueError("opacities shape does not match means2d")
    if isect_offsets.shape[:-2] != image_shape:
        raise ValueError("isect_offsets leading dimensions must match means2d")
    return (
        image_shape,
        math.prod(image_shape),
        gaussian_count,
        image_width,
        image_height,
    )


def rasterize_to_indices_in_range(
    range_start: int,
    range_end: int,
    transmittances: jax.Array,
    means2d: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: jax.Array | PaddedOffsets,
    flatten_ids: jax.Array,
    *,
    max_intersections: int | None = None,
    valid_count: jax.Array | int | None = None,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
) -> PaddedRasterizationIndices:
    """Return a padded list of contributing Gaussian/pixel pairs.

    The scan is streaming: memory is proportional to the fixed output capacity,
    not to ``num_tile_intersections * tile_size**2``. ``range_start`` and
    ``range_end`` follow gsplat's block convention, where one block contains
    ``tile_size**2`` depth-sorted Gaussians per tile.
    """

    range_start = jnp.asarray(range_start)
    range_end = jnp.asarray(range_end)
    if range_start.shape != () or range_end.shape != ():
        raise ValueError("range_start and range_end must be scalar")
    if not jnp.issubdtype(range_start.dtype, jnp.integer) or not jnp.issubdtype(
        range_end.dtype, jnp.integer
    ):
        raise TypeError("range_start and range_end must be integers")
    range_start = jnp.maximum(range_start.astype(jnp.int32), 0)
    range_end = jnp.maximum(range_end.astype(jnp.int32), range_start)
    tile_size = _as_static_int("tile_size", tile_size, minimum=1)
    means2d = jnp.asarray(means2d)
    conics = jnp.asarray(conics)
    opacities = jnp.asarray(opacities)
    flatten_ids = jnp.asarray(flatten_ids, dtype=jnp.int32)
    input_capacity = flatten_ids.shape[0]
    if max_intersections is None:
        max_intersections = max(1, input_capacity * tile_size * tile_size)
    output_capacity = _as_static_int(
        "max_intersections", max_intersections, minimum=1
    )
    raw_offsets = (
        isect_offsets.offsets
        if isinstance(isect_offsets, PaddedOffsets)
        else jnp.asarray(isect_offsets)
    )
    image_shape, image_count, gaussian_count, image_width, image_height = (
        _validate_dense_raster_inputs(
            means2d,
            conics,
            opacities,
            image_width,
            image_height,
            raw_offsets,
        )
    )
    offsets, valid_count, _ = _unwrap_offsets(
        isect_offsets, valid_count, False, input_capacity
    )
    expected_transmittance_shape = image_shape + (image_height, image_width)
    transmittances = jnp.asarray(transmittances)
    if transmittances.shape != expected_transmittance_shape:
        raise ValueError(
            f"transmittances must have shape {expected_transmittance_shape}"
        )

    output = (
        jnp.full((output_capacity,), -1, dtype=jnp.int32),
        jnp.full((output_capacity,), -1, dtype=jnp.int32),
        jnp.full((output_capacity,), -1, dtype=jnp.int32),
    )
    if gaussian_count == 0 or input_capacity == 0:
        return PaddedRasterizationIndices(
            *output, jnp.asarray(0, jnp.int32), jnp.asarray(False)
        )

    tile_height, tile_width = offsets.shape[-2:]
    tiles_per_image = tile_height * tile_width
    offsets_flat = offsets.reshape(-1)
    positions = jnp.arange(input_capacity, dtype=jnp.int32)
    candidate_tiles = jnp.searchsorted(offsets_flat, positions, side="right") - 1
    candidate_tiles = jnp.clip(candidate_tiles, 0, offsets_flat.shape[0] - 1)
    block_size = tile_size * tile_size
    start_slot = jnp.minimum(range_start * block_size, input_capacity)
    end_slot = jnp.minimum(range_end * block_size, input_capacity)
    flat_means = means2d.reshape(image_count, gaussian_count, 2)
    flat_conics = conics.reshape(image_count, gaussian_count, 3)
    flat_opacities = opacities.reshape(image_count, gaussian_count)
    transmittance = transmittances.reshape(-1)
    count = jnp.asarray(0, dtype=jnp.int32)

    def candidate_body(index, carry):
        trans, gaussian_out, pixel_out, image_out, current_count = carry
        tile_global = candidate_tiles[index]
        image_id = tile_global // tiles_per_image
        tile_id = tile_global % tiles_per_image
        tile_y = tile_id // tile_width
        tile_x = tile_id % tile_width
        tile_start = offsets_flat[tile_global]
        local_index = index - tile_start
        flat_id = jnp.clip(flatten_ids[index], 0, image_count * gaussian_count - 1)
        gaussian_id = flat_id % gaussian_count
        flat_image_id = flat_id // gaussian_count
        candidate_valid = (
            (index < valid_count)
            & (flatten_ids[index] >= 0)
            & (flat_image_id == image_id)
            & (local_index >= start_slot)
            & (local_index < end_slot)
        )
        mean = flat_means[image_id, gaussian_id]
        conic = flat_conics[image_id, gaussian_id]
        opacity = flat_opacities[image_id, gaussian_id]

        def pixel_body(local_pixel, pixel_carry):
            trans, gaussian_out, pixel_out, image_out, current_count = pixel_carry
            local_y = local_pixel // tile_size
            local_x = local_pixel % tile_size
            pixel_x = tile_x * tile_size + local_x
            pixel_y = tile_y * tile_size + local_y
            pixel_valid = (pixel_x < image_width) & (pixel_y < image_height)
            safe_x = jnp.minimum(pixel_x, image_width - 1)
            safe_y = jnp.minimum(pixel_y, image_height - 1)
            pixel_id = safe_y * image_width + safe_x
            ray_id = image_id * image_height * image_width + pixel_id
            delta_x = safe_x.astype(means2d.dtype) + 0.5 - mean[0]
            delta_y = safe_y.astype(means2d.dtype) + 0.5 - mean[1]
            sigma = (
                0.5 * (conic[0] * delta_x**2 + conic[2] * delta_y**2)
                + conic[1] * delta_x * delta_y
            )
            alpha = jnp.minimum(opacity * jnp.exp(-sigma), MAX_ALPHA)
            alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=MAX_ALPHA, neginf=0.0)
            current_transmittance = trans[ray_id]
            contributes = (
                candidate_valid
                & pixel_valid
                & jnp.isfinite(sigma)
                & (sigma >= 0.0)
                & (alpha >= alpha_threshold)
                & (
                    current_transmittance * (1.0 - alpha)
                    > transmittance_threshold
                )
            )
            trans = trans.at[ray_id].set(
                jnp.where(
                    contributes,
                    current_transmittance * (1.0 - alpha),
                    current_transmittance,
                )
            )

            def write(outputs):
                gaussian_out, pixel_out, image_out = outputs
                gaussian_out = gaussian_out.at[current_count].set(gaussian_id)
                pixel_out = pixel_out.at[current_count].set(pixel_id)
                image_out = image_out.at[current_count].set(image_id)
                return gaussian_out, pixel_out, image_out

            gaussian_out, pixel_out, image_out = jax.lax.cond(
                contributes & (current_count < output_capacity),
                write,
                lambda outputs: outputs,
                (gaussian_out, pixel_out, image_out),
            )
            current_count = jnp.minimum(
                current_count + contributes.astype(jnp.int32), output_capacity + 1
            )
            return trans, gaussian_out, pixel_out, image_out, current_count

        return jax.lax.fori_loop(0, block_size, pixel_body, carry)

    transmittance, *output, count = jax.lax.fori_loop(
        0,
        input_capacity,
        candidate_body,
        (transmittance, *output, count),
    )
    del transmittance
    return PaddedRasterizationIndices(
        *output,
        jnp.minimum(count, output_capacity),
        count > output_capacity,
    )


def accumulate(
    means2d: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    colors: jax.Array,
    gaussian_ids: jax.Array,
    pixel_ids: jax.Array,
    image_ids: jax.Array,
    image_width: int,
    image_height: int,
    *,
    valid_count: jax.Array | int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Differentiably alpha-composite padded pixel/Gaussian intersections."""

    means2d = jnp.asarray(means2d)
    conics = jnp.asarray(conics)
    opacities = jnp.asarray(opacities)
    colors = jnp.asarray(colors)
    gaussian_ids = jnp.asarray(gaussian_ids, dtype=jnp.int32)
    pixel_ids = jnp.asarray(pixel_ids, dtype=jnp.int32)
    image_ids = jnp.asarray(image_ids, dtype=jnp.int32)
    if not (gaussian_ids.shape == pixel_ids.shape == image_ids.shape):
        raise ValueError("gaussian_ids, pixel_ids and image_ids must share shape [K]")
    if gaussian_ids.ndim != 1:
        raise ValueError("rasterization indices must be one-dimensional")
    image_shape = means2d.shape[:-2]
    gaussian_count = means2d.shape[-2]
    if colors.shape[:-1] != image_shape + (gaussian_count,):
        raise ValueError("colors shape does not match means2d")
    channels = colors.shape[-1]
    dummy_offsets = jnp.empty(image_shape + (1, 1), dtype=jnp.int32)
    image_shape, image_count, gaussian_count, image_width, image_height = (
        _validate_dense_raster_inputs(
            means2d,
            conics,
            opacities,
            image_width,
            image_height,
            dummy_offsets,
        )
    )
    capacity = gaussian_ids.shape[0]
    valid_count = _normalize_valid_count(valid_count, capacity)
    output_colors = jnp.zeros(
        (image_count * image_height * image_width, channels), dtype=colors.dtype
    )
    output_alphas = jnp.zeros(
        (image_count * image_height * image_width,), dtype=opacities.dtype
    )
    if capacity == 0 or gaussian_count == 0:
        return (
            output_colors.reshape(image_shape + (image_height, image_width, channels)),
            output_alphas.reshape(image_shape + (image_height, image_width, 1)),
        )

    positions = jnp.arange(capacity, dtype=jnp.int32)
    valid = (
        (positions < valid_count)
        & (gaussian_ids >= 0)
        & (gaussian_ids < gaussian_count)
        & (pixel_ids >= 0)
        & (pixel_ids < image_width * image_height)
        & (image_ids >= 0)
        & (image_ids < image_count)
    )
    safe_gaussian_ids = jnp.clip(gaussian_ids, 0, gaussian_count - 1)
    safe_pixel_ids = jnp.clip(pixel_ids, 0, image_width * image_height - 1)
    safe_image_ids = jnp.clip(image_ids, 0, image_count - 1)
    ray_ids = safe_image_ids * image_height * image_width + safe_pixel_ids
    ray_sort_ids = jnp.where(valid, ray_ids, image_count * image_height * image_width)
    order = jnp.lexsort((positions, ray_sort_ids))
    valid = valid[order]
    safe_gaussian_ids = safe_gaussian_ids[order]
    safe_pixel_ids = safe_pixel_ids[order]
    safe_image_ids = safe_image_ids[order]
    ray_ids = ray_ids[order]

    flat_means = means2d.reshape(image_count, gaussian_count, 2)
    flat_conics = conics.reshape(image_count, gaussian_count, 3)
    flat_opacities = opacities.reshape(image_count, gaussian_count)
    flat_colors = colors.reshape(image_count, gaussian_count, channels)
    pixel_x = safe_pixel_ids % image_width
    pixel_y = safe_pixel_ids // image_width
    means = flat_means[safe_image_ids, safe_gaussian_ids]
    conic = flat_conics[safe_image_ids, safe_gaussian_ids]
    delta_x = pixel_x.astype(means2d.dtype) + 0.5 - means[:, 0]
    delta_y = pixel_y.astype(means2d.dtype) + 0.5 - means[:, 1]
    sigma = (
        0.5 * (conic[:, 0] * delta_x**2 + conic[:, 2] * delta_y**2)
        + conic[:, 1] * delta_x * delta_y
    )
    alphas = jnp.minimum(
        flat_opacities[safe_image_ids, safe_gaussian_ids] * jnp.exp(-sigma),
        MAX_ALPHA,
    )
    alphas = jnp.where(valid, jnp.nan_to_num(alphas, nan=0.0), 0.0)
    ray_sort_ids = jnp.where(valid, ray_ids, image_count * image_height * image_width)
    segment_starts = jnp.concatenate(
        (
            jnp.ones((1,), dtype=jnp.bool_),
            ray_sort_ids[1:] != ray_sort_ids[:-1],
        )
    )
    one_minus_alpha = 1.0 - alphas

    def segmented_product(left, right):
        left_value, left_has_start = left
        right_value, right_has_start = right
        return (
            jnp.where(right_has_start, right_value, left_value * right_value),
            left_has_start | right_has_start,
        )

    inclusive, _ = jax.lax.associative_scan(
        segmented_product, (one_minus_alpha, segment_starts)
    )
    previous = jnp.concatenate((jnp.ones_like(inclusive[:1]), inclusive[:-1]))
    transmittance = jnp.where(segment_starts, 1.0, previous)
    weights = jnp.where(valid, alphas * transmittance, 0.0)
    output_colors = output_colors.at[ray_ids].add(
        weights[:, None] * flat_colors[safe_image_ids, safe_gaussian_ids]
    )
    output_alphas = output_alphas.at[ray_ids].add(weights)
    return (
        output_colors.reshape(image_shape + (image_height, image_width, channels)),
        output_alphas.reshape(image_shape + (image_height, image_width, 1)),
    )


def rasterize_to_pixels(
    means2d: jax.Array,
    conics: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: jax.Array | PaddedOffsets,
    flatten_ids: jax.Array,
    backgrounds: jax.Array | None = None,
    masks: jax.Array | None = None,
    packed: bool = False,
    absgrad: bool = False,
    *,
    valid_count: jax.Array | int | None = None,
    overflow: jax.Array | bool = False,
    max_gaussians_per_tile: int = 512,
    tile_batch_size: int = 1,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
    return_info: bool = False,
    _means2d_absgrad_probe: jax.Array | None = None,
):
    """Rasterize every retained tile intersection in fixed-size chunks.

    ``max_gaussians_per_tile`` controls the temporary chunk size; it does not
    truncate a tile. Only an overflowing input intersection buffer makes the
    result incomplete.

    JAX cannot attach a mutable ``.absgrad`` value during a later backward
    pass. To request the equivalent statistic, pass ``absgrad=True`` together
    with an independent, zero-valued ``_means2d_absgrad_probe`` matching
    ``means2d``. Differentiate the final loss jointly with respect to
    ``means2d`` and the probe: the former gradient remains the ordinary signed
    VJP, while the latter is the sum of absolute Gaussian-by-pixel compositor
    contributions. The probe does not affect the forward values.
    """

    if absgrad and _means2d_absgrad_probe is None:
        raise ValueError(
            "absgrad=True requires a zero-valued _means2d_absgrad_probe; "
            "differentiate the loss jointly with respect to means2d and the "
            "probe to obtain signed gradient and compositor AbsGrad"
        )
    if not absgrad and _means2d_absgrad_probe is not None:
        raise ValueError(
            "_means2d_absgrad_probe requires absgrad=True"
        )
    image_width = _as_static_int("image_width", image_width, minimum=1)
    image_height = _as_static_int("image_height", image_height, minimum=1)
    tile_size = _as_static_int("tile_size", tile_size, minimum=1)
    max_gaussians_per_tile = _as_static_int(
        "max_gaussians_per_tile", max_gaussians_per_tile, minimum=1
    )
    tile_batch_size = _as_static_int("tile_batch_size", tile_batch_size, minimum=1)
    means2d = jnp.asarray(means2d)
    conics = jnp.asarray(conics)
    colors = jnp.asarray(colors)
    opacities = jnp.asarray(opacities)
    flatten_ids = jnp.asarray(flatten_ids, dtype=jnp.int32)
    if _means2d_absgrad_probe is not None:
        _means2d_absgrad_probe = jnp.asarray(_means2d_absgrad_probe)
        if _means2d_absgrad_probe.shape != means2d.shape:
            raise ValueError(
                "_means2d_absgrad_probe must have the same shape as means2d"
            )
        if _means2d_absgrad_probe.dtype != means2d.dtype:
            raise TypeError(
                "_means2d_absgrad_probe must have the same dtype as means2d"
            )
    input_capacity = flatten_ids.shape[0]
    offsets, valid_count, input_overflow = _unwrap_offsets(
        isect_offsets, valid_count, overflow, input_capacity
    )
    image_shape = offsets.shape[:-2]
    image_count = math.prod(image_shape)
    tile_height, tile_width = offsets.shape[-2:]
    if image_count < 1:
        raise ValueError("isect_offsets must contain at least one image")
    if tile_width * tile_size < image_width or tile_height * tile_size < image_height:
        raise ValueError("isect_offsets tile grid does not cover the requested image")

    packed = bool(packed or means2d.ndim == 2)
    if packed:
        slot_count = means2d.shape[0]
        if means2d.shape != (slot_count, 2):
            raise ValueError("packed means2d must have shape [nnz, 2]")
        if conics.shape != (slot_count, 3):
            raise ValueError("packed conics must have shape [nnz, 3]")
        if opacities.shape != (slot_count,):
            raise ValueError("packed opacities must have shape [nnz]")
        if colors.shape[:-1] != (slot_count,):
            raise ValueError("packed colors must have shape [nnz, channels]")
        dense_gaussians_per_image = None
    else:
        if means2d.shape[:-2] != image_shape:
            raise ValueError("means2d leading dimensions must match isect_offsets")
        dense_gaussians_per_image = means2d.shape[-2]
        if conics.shape != image_shape + (dense_gaussians_per_image, 3):
            raise ValueError("conics shape does not match means2d")
        if opacities.shape != image_shape + (dense_gaussians_per_image,):
            raise ValueError("opacities shape does not match means2d")
        if colors.shape[:-1] != image_shape + (dense_gaussians_per_image,):
            raise ValueError("colors shape does not match means2d")
        slot_count = image_count * dense_gaussians_per_image
    channels = colors.shape[-1]
    flat_means = means2d.reshape(slot_count, 2)
    flat_conics = conics.reshape(slot_count, 3)
    flat_colors = colors.reshape(slot_count, channels)
    flat_opacities = opacities.reshape(slot_count)
    flat_absgrad_probe = (
        None
        if _means2d_absgrad_probe is None
        else _means2d_absgrad_probe.reshape(slot_count, 2)
    )

    if masks is None:
        masks = jnp.ones(offsets.shape, dtype=jnp.bool_)
    else:
        masks = jnp.asarray(masks, dtype=jnp.bool_)
        if masks.shape != offsets.shape:
            raise ValueError("masks must have the same shape as isect_offsets")

    if backgrounds is None:
        backgrounds = jnp.zeros(image_shape + (channels,), dtype=colors.dtype)
    else:
        backgrounds = jnp.asarray(backgrounds)
        if backgrounds.shape == (channels,):
            backgrounds = jnp.broadcast_to(backgrounds, image_shape + (channels,))
        if backgrounds.shape != image_shape + (channels,):
            raise ValueError("backgrounds must have shape [..., channels]")

    if slot_count == 0 or input_capacity == 0:
        render_colors = jnp.broadcast_to(
            backgrounds[..., None, None, :],
            image_shape + (image_height, image_width, channels),
        )
        render_alphas = jnp.zeros(
            image_shape + (image_height, image_width, 1), dtype=opacities.dtype
        )
        info = {
            "tile_overflow": jnp.zeros(offsets.shape, dtype=jnp.bool_),
            "overflow": input_overflow,
        }
        return (render_colors, render_alphas, info) if return_info else (
            render_colors,
            render_alphas,
        )

    offsets_flat = offsets.reshape(-1)
    mask_flat = masks.reshape(-1)
    tile_count = offsets_flat.shape[0]
    candidate_slots = jnp.arange(max_gaussians_per_tile, dtype=jnp.int32)
    # The chunk loop has to cover the worst case for a single tile. That is not
    # the whole intersection buffer: isect_tiles emits a tile at most once per
    # Gaussian, so one tile holds at most as many candidates as there are slots
    # it can draw from. Batched dense inputs narrow that further, because a
    # tile only accepts candidates from its own image.
    per_tile_bound = (
        slot_count if dense_gaussians_per_image is None else dense_gaussians_per_image
    )
    chunk_count = math.ceil(
        min(input_capacity, per_tile_bound) / max_gaussians_per_tile
    )
    # The busiest tile is a single scalar over the whole image, so it can also
    # gate the loop from outside the tile map: the predicate stays scalar
    # inside the tile batch's vmap and lets the loop skip chunks no tile can
    # reach at run time.
    tile_ends = jnp.concatenate(
        (
            offsets_flat[1:],
            jnp.asarray(valid_count, dtype=offsets_flat.dtype)[None],
        )
    )
    busiest_tile_count = jnp.max(jnp.maximum(tile_ends - offsets_flat, 0))
    local_y, local_x = jnp.meshgrid(
        jnp.arange(tile_size, dtype=means2d.dtype) + 0.5,
        jnp.arange(tile_size, dtype=means2d.dtype) + 0.5,
        indexing="ij",
    )
    local_x = local_x.reshape(-1)
    local_y = local_y.reshape(-1)
    pixel_count = tile_size * tile_size

    def render_tile(tile_global):
        start = offsets_flat[tile_global]
        end = jnp.where(
            tile_global + 1 < tile_count,
            offsets_flat[jnp.minimum(tile_global + 1, tile_count - 1)],
            valid_count,
        )
        count = jnp.maximum(end - start, 0)
        image_id = tile_global // (tile_height * tile_width)
        tile_id = tile_global % (tile_height * tile_width)
        tile_y = tile_id // tile_width
        tile_x = tile_id % tile_width
        pixel_x = tile_x.astype(means2d.dtype) * tile_size + local_x
        pixel_y = tile_y.astype(means2d.dtype) * tile_size + local_y
        pixel_valid = (pixel_x < image_width) & (pixel_y < image_height)

        initial = (
            jnp.zeros((pixel_count, channels), dtype=colors.dtype),
            jnp.zeros((pixel_count,), dtype=opacities.dtype),
            jnp.ones((pixel_count,), dtype=opacities.dtype),
        )

        def composite_chunk(chunk_index, carry):
            return jax.lax.cond(
                chunk_index * max_gaussians_per_tile < busiest_tile_count,
                # Recomputing the reached branch keeps the conditional's
                # reverse-mode residuals down to the carry. A conditional gives
                # both branches the same residual signature, so without this the
                # skipped branch materializes a zero-filled stand-in for every
                # [max_gaussians_per_tile, pixel] intermediate the reached
                # branch saves. That costs far more than the compositing it
                # stands in for, and the gate skips most chunks. The enclosing
                # loop already keeps XLA from folding the recomputation back
                # into the original, so the barrier that prevent_cse inserts
                # would only cost fusion opportunities.
                jax.checkpoint(
                    partial(composite_reached_chunk, chunk_index),
                    prevent_cse=False,
                ),
                lambda state: state,
                carry,
            )

        def composite_reached_chunk(chunk_index, carry):
            render, accumulated_alpha, incoming_transmittance = carry
            local_ids = chunk_index * max_gaussians_per_tile + candidate_slots
            positions = start + local_ids
            safe_positions = jnp.clip(positions, 0, input_capacity - 1)
            flat_ids = flatten_ids[safe_positions]
            safe_flat_ids = jnp.clip(flat_ids, 0, slot_count - 1)
            candidate_valid = (
                (local_ids < count)
                & (positions < valid_count)
                & (positions < input_capacity)
                & (flat_ids >= 0)
                & mask_flat[tile_global]
            )
            if dense_gaussians_per_image is not None:
                candidate_valid = candidate_valid & (
                    safe_flat_ids // dense_gaussians_per_image == image_id
                )

            means = flat_means[safe_flat_ids]
            selected_conics = flat_conics[safe_flat_ids]
            selected_colors = flat_colors[safe_flat_ids]
            selected_opacities = flat_opacities[safe_flat_ids]
            if flat_absgrad_probe is None:
                delta_x = pixel_x[None, :] - means[:, 0, None]
                delta_y = pixel_y[None, :] - means[:, 1, None]
            else:
                pixel_template = jnp.stack((pixel_x, pixel_y), axis=-1)
                means_per_pixel = _broadcast_means_with_absgrad_probe(
                    means,
                    flat_absgrad_probe[safe_flat_ids],
                    pixel_template,
                )
                delta_x = pixel_x[None, :] - means_per_pixel[..., 0]
                delta_y = pixel_y[None, :] - means_per_pixel[..., 1]
            sigma = (
                0.5
                * (
                    selected_conics[:, 0, None] * delta_x**2
                    + selected_conics[:, 2, None] * delta_y**2
                )
                + selected_conics[:, 1, None] * delta_x * delta_y
            )
            alpha = jnp.minimum(
                selected_opacities[:, None] * jnp.exp(-sigma), MAX_ALPHA
            )
            alpha = jnp.nan_to_num(
                alpha, nan=0.0, posinf=MAX_ALPHA, neginf=0.0
            )
            alpha_valid = (
                candidate_valid[:, None]
                & pixel_valid[None, :]
                & jnp.isfinite(sigma)
                & (sigma >= 0.0)
                & (alpha >= alpha_threshold)
            )
            alpha = jnp.where(alpha_valid, alpha, 0.0)
            weights, outgoing_transmittance = _chunk_weights(
                alpha, incoming_transmittance, transmittance_threshold
            )
            render = render + jnp.einsum(
                "kp,kc->pc",
                weights,
                selected_colors,
                precision=jax.lax.Precision.HIGHEST,
            )
            accumulated_alpha = accumulated_alpha + jnp.sum(weights, axis=0)
            return render, accumulated_alpha, outgoing_transmittance

        render, accumulated_alpha, _ = jax.lax.fori_loop(
            0,
            chunk_count,
            # Reverse mode would otherwise stack one [max_gaussians_per_tile,
            # pixel] intermediate set per chunk, so a tile's backward
            # workspace grew with the intersection capacity. Recomputing a
            # chunk keeps only the carry.
            jax.checkpoint(composite_chunk),
            initial,
        )
        render = jnp.where(pixel_valid[:, None], render, 0.0)
        accumulated_alpha = jnp.where(pixel_valid, accumulated_alpha, 0.0)
        return (
            render.reshape(tile_size, tile_size, channels),
            accumulated_alpha.reshape(tile_size, tile_size, 1),
            # The chunk loop is sized for the most candidates one tile can
            # hold. Offsets that claim more than that describe an intersection
            # buffer this call cannot render completely, so report it rather
            # than silently dropping the tail.
            count > chunk_count * max_gaussians_per_tile,
        )

    tile_ids = jnp.arange(tile_count, dtype=jnp.int32)
    rendered_tiles, alpha_tiles, tile_overflow = jax.lax.map(
        # Reverse mode otherwise keeps every chunk of every tile's
        # [max_gaussians_per_tile, pixel] compositing intermediates alive at
        # once, which dwarfs the forward workspace. Recomputing one tile batch
        # at a time is also what the upstream backward kernel does.
        jax.checkpoint(render_tile),
        tile_ids,
        batch_size=tile_batch_size,
    )
    render_colors = (
        rendered_tiles.reshape(
            image_count,
            tile_height,
            tile_width,
            tile_size,
            tile_size,
            channels,
        )
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(
            image_count,
            tile_height * tile_size,
            tile_width * tile_size,
            channels,
        )[:, :image_height, :image_width]
    )
    render_alphas = (
        alpha_tiles.reshape(
            image_count, tile_height, tile_width, tile_size, tile_size, 1
        )
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(
            image_count, tile_height * tile_size, tile_width * tile_size, 1
        )[:, :image_height, :image_width]
    )
    flat_backgrounds = backgrounds.reshape(image_count, channels)
    render_colors = render_colors + flat_backgrounds[:, None, None, :] * (
        1.0 - render_alphas
    )
    render_colors = render_colors.reshape(
        image_shape + (image_height, image_width, channels)
    )
    render_alphas = render_alphas.reshape(
        image_shape + (image_height, image_width, 1)
    )
    tile_overflow = tile_overflow.reshape(offsets.shape)
    info = {
        "tile_overflow": tile_overflow,
        "overflow": input_overflow | jnp.any(tile_overflow),
    }
    if not return_info:
        def report_overflow(_):
            jax.debug.callback(_raise_rasterization_overflow, ordered=True)
            return jnp.asarray(0, dtype=jnp.int32)

        jax.lax.cond(
            info["overflow"],
            report_overflow,
            lambda _: jnp.asarray(0, dtype=jnp.int32),
            operand=None,
        )
    return (render_colors, render_alphas, info) if return_info else (
        render_colors,
        render_alphas,
    )


__all__ = [
    "DEFAULT_ALPHA_THRESHOLD",
    "DEFAULT_TRANSMITTANCE_THRESHOLD",
    "MAX_ALPHA",
    "PaddedIntersections",
    "PaddedOffsets",
    "PaddedRasterizationIndices",
    "accumulate",
    "isect_offset_encode",
    "isect_tiles",
    "rasterize_to_indices_in_range",
    "rasterize_to_pixels",
]
