from __future__ import annotations

import operator
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._pallas_intersections import (
    count_accutile_intersections_pallas,
    emit_accutile_intersections_pallas,
)


_DIRECT_SORT_MIN_CAPACITY = 65_536
_GAUSSIAN_EXTEND = 3.33


class TileIntersections(NamedTuple):
    """Fixed-capacity, tile-major Gaussian intersections for one camera.

    The first ``valid_count`` entries of ``gaussian_ids`` and ``tile_ids`` are
    valid and sorted by ``(tile_id, depth, gaussian_id)``. Remaining entries are
    padded with ``-1``. ``offsets[y, x]`` is the start of tile ``(x, y)`` in
    those arrays; the end is the next flattened offset, or ``valid_count`` for
    the final tile.
    """

    gaussian_ids: jax.Array
    tile_ids: jax.Array
    offsets: jax.Array
    valid_count: jax.Array
    overflow: jax.Array
    required_count: jax.Array


class _AccuTileState(NamedTuple):
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


def _static_int(name: str, value: int, *, minimum: int) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a static integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _saturating_cumsum(counts: jax.Array, limit: int) -> jax.Array:
    counts = jnp.minimum(counts.astype(jnp.int32), jnp.int32(limit))
    return jax.lax.associative_scan(
        lambda left, right: jnp.minimum(left + right, jnp.int32(limit)),
        counts,
    )


def _map_intersections_jax(
    cumulative: jax.Array,
    min_x: jax.Array,
    min_y: jax.Array,
    span_x: jax.Array,
    valid_count: jax.Array,
    *,
    tile_width: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array]:
    ranks = jnp.arange(capacity, dtype=jnp.int32)
    gaussian_count = cumulative.shape[0]
    gaussian_ids = jnp.searchsorted(cumulative, ranks, side="right")
    gaussian_ids = jnp.clip(gaussian_ids, 0, gaussian_count - 1)
    previous = jnp.where(
        gaussian_ids > 0,
        cumulative[jnp.maximum(gaussian_ids - 1, 0)],
        0,
    )
    local_ids = ranks - previous
    safe_span_x = jnp.maximum(span_x[gaussian_ids], 1)
    tile_x = min_x[gaussian_ids] + local_ids % safe_span_x
    tile_y = min_y[gaussian_ids] + local_ids // safe_span_x
    output_valid = ranks < valid_count
    return (
        jnp.where(output_valid, gaussian_ids, -1).astype(jnp.int32),
        jnp.where(output_valid, tile_y * tile_width + tile_x, -1).astype(
            jnp.int32
        ),
    )


def _ellipse_intersection_jax(
    b: jax.Array,
    disc: jax.Array,
    t: jax.Array,
    p_u: jax.Array,
    p_v: jax.Array,
    coefficient: jax.Array,
    coordinate: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    h = coordinate - p_u
    root = jnp.sqrt(jnp.maximum(disc * h * h + t * coefficient, 0.0))
    return (
        (-b * h - root) / coefficient + p_v,
        (-b * h + root) / coefficient + p_v,
    )


def _prepare_accutile_state_jax(
    means2d: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    valid: jax.Array,
    *,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    alpha_threshold: float,
) -> _AccuTileState:
    """Prepare the per-Gaussian values used by the AccuTile column walk."""

    means2d = means2d.astype(jnp.float32)
    radii = radii.astype(jnp.float32)
    conics = conics.astype(jnp.float32)
    opacities = opacities.astype(jnp.float32)
    threshold = jnp.float32(alpha_threshold)
    block = jnp.float32(tile_size)

    px = means2d[:, 0]
    py = means2d[:, 1]
    radius_x = radii[:, 0]
    radius_y = radii[:, 1]
    a = conics[:, 0]
    b = conics[:, 1]
    c = conics[:, 2]
    finite = (
        jnp.all(jnp.isfinite(means2d), axis=-1)
        & jnp.all(jnp.isfinite(radii), axis=-1)
        & jnp.all(jnp.isfinite(conics), axis=-1)
        & jnp.isfinite(opacities)
    )
    base_valid = (
        valid
        & finite
        & (radius_x > 0.0)
        & (radius_y > 0.0)
        & (opacities > threshold)
        & (a > 0.0)
        & (c > 0.0)
    )

    safe_px = jnp.where(base_valid, px, 0.0)
    safe_py = jnp.where(base_valid, py, 0.0)
    safe_a = jnp.where(base_valid, a, 1.0)
    safe_b = jnp.where(base_valid, b, 0.0)
    safe_c = jnp.where(base_valid, c, 1.0)
    safe_opacity = jnp.where(base_valid, opacities, threshold)
    disc = safe_b * safe_b - safe_a * safe_c
    extent = jnp.float32(_GAUSSIAN_EXTEND)
    t = jnp.minimum(
        extent * extent,
        jnp.float32(2.0) * jnp.log(safe_opacity / threshold),
    )
    ellipse_valid = (
        base_valid
        & jnp.isfinite(disc)
        & jnp.isfinite(t)
        & (disc < 0.0)
        & (t > 0.0)
    )
    safe_disc = jnp.where(ellipse_valid, disc, -1.0)
    safe_t = jnp.where(ellipse_valid, t, 1.0)
    scale = -safe_t / safe_disc
    x_extent = jnp.sqrt(jnp.maximum(scale * safe_c, 0.0))
    y_extent = jnp.sqrt(jnp.maximum(scale * safe_a, 0.0))
    bbox_min_x = safe_px - x_extent
    bbox_min_y = safe_py - y_extent
    bbox_max_x = safe_px + x_extent
    bbox_max_y = safe_py + y_extent
    argmin_x = safe_py + safe_b * x_extent / safe_c
    argmin_y = safe_px + safe_b * y_extent / safe_a
    argmax_x = safe_py - safe_b * x_extent / safe_c
    argmax_y = safe_px - safe_b * y_extent / safe_a
    derived_finite = (
        ellipse_valid
        & jnp.isfinite(scale)
        & jnp.isfinite(x_extent)
        & jnp.isfinite(y_extent)
        & jnp.isfinite(bbox_min_x)
        & jnp.isfinite(bbox_min_y)
        & jnp.isfinite(bbox_max_x)
        & jnp.isfinite(bbox_max_y)
        & jnp.isfinite(argmin_x)
        & jnp.isfinite(argmin_y)
        & jnp.isfinite(argmax_x)
        & jnp.isfinite(argmax_y)
    )
    bbox_min_x = jnp.where(derived_finite, bbox_min_x, 0.0)
    bbox_min_y = jnp.where(derived_finite, bbox_min_y, 0.0)
    bbox_max_x = jnp.where(derived_finite, bbox_max_x, 0.0)
    bbox_max_y = jnp.where(derived_finite, bbox_max_y, 0.0)
    argmin_x = jnp.where(derived_finite, argmin_x, 0.0)
    argmin_y = jnp.where(derived_finite, argmin_y, 0.0)
    argmax_x = jnp.where(derived_finite, argmax_x, 0.0)
    argmax_y = jnp.where(derived_finite, argmax_y, 0.0)

    rect_min_x = jnp.clip(
        (bbox_min_x / block).astype(jnp.int32), 0, tile_width
    )
    rect_min_y = jnp.clip(
        (bbox_min_y / block).astype(jnp.int32), 0, tile_height
    )
    rect_max_x = jnp.clip(
        (bbox_max_x / block + 1.0).astype(jnp.int32), 0, tile_width
    )
    rect_max_y = jnp.clip(
        (bbox_max_y / block + 1.0).astype(jnp.int32), 0, tile_height
    )
    span_x = rect_max_x - rect_min_x
    span_y = rect_max_y - rect_min_y
    state_valid = derived_finite & (span_x > 0) & (span_y > 0)
    is_y = span_y < span_x

    return _AccuTileState(
        valid=state_valid,
        is_y=is_y,
        b=jnp.where(state_valid, safe_b, 0.0),
        disc=jnp.where(state_valid, safe_disc, -1.0),
        t=jnp.where(state_valid, safe_t, 1.0),
        p_u=jnp.where(is_y, safe_py, safe_px),
        p_v=jnp.where(is_y, safe_px, safe_py),
        coefficient=jnp.where(is_y, safe_a, safe_c),
        outer_min=jnp.where(is_y, rect_min_y, rect_min_x),
        outer_max=jnp.where(is_y, rect_max_y, rect_max_x),
        cross_min=jnp.where(is_y, rect_min_x, rect_min_y),
        cross_max=jnp.where(is_y, rect_max_x, rect_max_y),
        outer_bbox_min=jnp.where(is_y, bbox_min_y, bbox_min_x),
        outer_bbox_max=jnp.where(is_y, bbox_max_y, bbox_max_x),
        cross_bbox_min=jnp.where(is_y, bbox_min_x, bbox_min_y),
        cross_bbox_max=jnp.where(is_y, bbox_max_x, bbox_max_y),
        argmin_outer=jnp.where(is_y, argmin_x, argmin_y),
        argmax_outer=jnp.where(is_y, argmax_x, argmax_y),
    )


def _accutile_initial_span_jax(
    state: _AccuTileState, *, tile_size: int
) -> tuple[jax.Array, jax.Array]:
    min_line = state.outer_min.astype(jnp.float32) * jnp.float32(tile_size)
    line_min, line_max = _ellipse_intersection_jax(
        state.b,
        state.disc,
        state.t,
        state.p_u,
        state.p_v,
        state.coefficient,
        min_line,
    )
    intersects = state.outer_bbox_min <= min_line
    return (
        jnp.where(intersects, line_min, state.cross_bbox_max),
        jnp.where(intersects, line_max, state.cross_bbox_min),
    )


def _accutile_column_span_jax(
    state: _AccuTileState,
    outer_offset: jax.Array,
    previous_min: jax.Array,
    previous_max: jax.Array,
    *,
    tile_size: int,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    outer = state.outer_min + outer_offset
    active = state.valid & (outer < state.outer_max)
    min_line = outer.astype(jnp.float32) * jnp.float32(tile_size)
    max_line = min_line + jnp.float32(tile_size)
    line_min, line_max = _ellipse_intersection_jax(
        state.b,
        state.disc,
        state.t,
        state.p_u,
        state.p_v,
        state.coefficient,
        max_line,
    )
    intersects = max_line <= state.outer_bbox_max
    current_min = jnp.where(intersects, line_min, previous_min)
    current_max = jnp.where(intersects, line_max, previous_max)
    ellipse_min = jnp.where(
        (min_line <= state.argmin_outer)
        & (state.argmin_outer < max_line),
        state.cross_bbox_min,
        jnp.minimum(previous_min, current_min),
    )
    ellipse_max = jnp.where(
        (min_line <= state.argmax_outer)
        & (state.argmax_outer < max_line),
        state.cross_bbox_max,
        jnp.maximum(previous_max, current_max),
    )
    block = jnp.float32(tile_size)
    min_v = jnp.maximum(
        state.cross_min,
        jnp.minimum(
            state.cross_max, (ellipse_min / block).astype(jnp.int32)
        ),
    )
    max_v = jnp.minimum(
        state.cross_max,
        jnp.maximum(
            state.cross_min,
            (ellipse_max / block + 1.0).astype(jnp.int32),
        ),
    )
    return min_v, max_v, current_min, current_max, active, outer


def _count_accutile_intersections_jax(
    state: _AccuTileState,
    *,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> jax.Array:
    previous_min, previous_max = _accutile_initial_span_jax(
        state, tile_size=tile_size
    )
    counts = jnp.zeros(state.valid.shape, dtype=jnp.int32)

    def count_outer(
        outer_offset: int,
        carry: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        previous_min, previous_max, counts = carry
        min_v, max_v, current_min, current_max, active, _ = (
            _accutile_column_span_jax(
                state,
                jnp.int32(outer_offset),
                previous_min,
                previous_max,
                tile_size=tile_size,
            )
        )
        counts = counts + jnp.where(active, max_v - min_v, 0)
        previous_min = jnp.where(active, current_min, previous_min)
        previous_max = jnp.where(active, current_max, previous_max)
        return previous_min, previous_max, counts

    _, _, counts = jax.lax.fori_loop(
        0,
        max(tile_width, tile_height),
        count_outer,
        (previous_min, previous_max, counts),
    )
    return counts.astype(jnp.int32)


def _emit_accutile_intersections_jax(
    state: _AccuTileState,
    cumulative: jax.Array,
    valid_count: jax.Array,
    *,
    capacity: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> tuple[jax.Array, jax.Array]:
    """Fill the output buffer by asking each slot which run it belongs to.

    The ellipse walk is sequential along the outer axis, so the per-column
    spans still come from one pass over it. Emitting used to add a second,
    nested pass that scattered the entire input into the output buffer once per
    (outer, cross) pair. That costs the square of the tile grid however few
    intersections the scene produces, and grows with the square of the
    resolution. Instead every output slot resolves its own Gaussian from the
    prefix sums, the way the AABB path already does, and a pass over the outer
    axis only has to say which slots the runs it just measured cover.
    """

    if capacity == 0:
        return (
            jnp.full((capacity,), -1, dtype=jnp.int32),
            jnp.full((capacity,), -1, dtype=jnp.int32),
        )

    gaussian_count = state.valid.shape[0]
    ranks = jnp.arange(capacity, dtype=jnp.int32)
    owner = jnp.clip(
        jnp.searchsorted(cumulative, ranks, side="right"),
        0,
        gaussian_count - 1,
    )
    previous = jnp.where(
        owner > 0, cumulative[jnp.maximum(owner - 1, 0)], 0
    )
    local = ranks - previous
    output_valid = ranks < valid_count

    previous_min, previous_max = _accutile_initial_span_jax(
        state, tile_size=tile_size
    )
    emitted = jnp.zeros(state.valid.shape, dtype=jnp.int32)
    cross = jnp.zeros((capacity,), dtype=jnp.int32)
    outer_offset_of_slot = jnp.zeros((capacity,), dtype=jnp.int32)

    def emit_outer(
        outer_offset: int,
        carry: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        previous_min, previous_max, emitted, cross, outer_offset_of_slot = carry
        min_v, max_v, current_min, current_max, active, _ = (
            _accutile_column_span_jax(
                state,
                jnp.int32(outer_offset),
                previous_min,
                previous_max,
                tile_size=tile_size,
            )
        )
        span = jnp.where(active, max_v - min_v, 0)
        # The run this column contributes occupies the Gaussian's output slots
        # [emitted, emitted + span). A negative span covers nothing, which is
        # what the nested loop's cross_offset < span condition also did.
        run_start = emitted[owner]
        covered = (local >= run_start) & (local < run_start + span[owner])
        cross = jnp.where(covered, min_v[owner] + local - run_start, cross)
        outer_offset_of_slot = jnp.where(
            covered, jnp.int32(outer_offset), outer_offset_of_slot
        )
        emitted = emitted + span
        previous_min = jnp.where(active, current_min, previous_min)
        previous_max = jnp.where(active, current_max, previous_max)
        return previous_min, previous_max, emitted, cross, outer_offset_of_slot

    _, _, _, cross, outer_offset_of_slot = jax.lax.fori_loop(
        0,
        max(tile_width, tile_height),
        emit_outer,
        (previous_min, previous_max, emitted, cross, outer_offset_of_slot),
    )

    outer = state.outer_min[owner] + outer_offset_of_slot
    tile_ids = jnp.where(
        state.is_y[owner],
        outer * tile_width + cross,
        cross * tile_width + outer,
    )
    return (
        jnp.where(output_valid, owner, -1).astype(jnp.int32),
        jnp.where(output_valid, tile_ids, -1).astype(jnp.int32),
    )


def _accutile_intersections_jax(
    means2d: jax.Array,
    radii: jax.Array,
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
    """Return Gaussian-major pure-JAX AccuTile pairs before depth sorting."""

    state = _prepare_accutile_state_jax(
        means2d,
        radii,
        conics,
        opacities,
        valid,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        alpha_threshold=alpha_threshold,
    )
    counts = _count_accutile_intersections_jax(
        state,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    required_cumulative = _saturating_cumsum(counts, 2**30 - 1)
    required_count = required_cumulative[-1]
    cumulative = jnp.minimum(required_cumulative, jnp.int32(capacity + 1))
    valid_count = jnp.minimum(required_count, jnp.int32(capacity))
    overflow = required_count > capacity
    gaussian_ids, tile_ids = _emit_accutile_intersections_jax(
        state,
        cumulative,
        valid_count,
        capacity=capacity,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    return (
        counts,
        gaussian_ids,
        tile_ids,
        valid_count,
        overflow,
        required_count,
    )


def intersect_tiles(
    means2d: jax.Array,
    radii: jax.Array,
    depths: jax.Array,
    valid: jax.Array,
    *,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    max_intersections: int,
    backend: str = "auto",
    sort_backend: str = "auto",
    conics: jax.Array | None = None,
    opacities: jax.Array | None = None,
    alpha_threshold: float = 1.0 / 255.0,
    mode: str = "auto",
) -> TileIntersections:
    """Build a bounded tile-intersection list without a ``tiles x N`` matrix.

    Geometry inputs describe a single camera. Intersection construction is
    intentionally stop-gradient, like gsplat's CUDA intersection kernels; the
    returned integer indices remain safe to use for gathering differentiable
    Gaussian attributes in a later rasterizer.

    Memory is ``O(N + max_intersections + tile_count)``. When the exact number
    of intersections exceeds ``max_intersections``, the retained prefix is
    sorted normally and ``overflow`` is set. ``backend='pallas'`` replaces
    the AccuTile count and pair-emission scans; geometry preparation, prefix
    sums, sorting, and offsets deliberately remain in JAX.
    """

    tile_size = _static_int("tile_size", tile_size, minimum=1)
    tile_width = _static_int("tile_width", tile_width, minimum=1)
    tile_height = _static_int("tile_height", tile_height, minimum=1)
    capacity = _static_int("max_intersections", max_intersections, minimum=0)
    if backend not in {"auto", "jax", "pallas"}:
        raise ValueError("backend must be 'auto', 'jax', or 'pallas'")
    if sort_backend not in {"auto", "jax"}:
        raise ValueError("sort_backend must be 'auto' or 'jax'")
    if mode not in {"auto", "aabb", "accutile"}:
        raise ValueError("mode must be 'auto', 'aabb', or 'accutile'")
    if not math.isfinite(alpha_threshold) or alpha_threshold <= 0.0:
        raise ValueError("alpha_threshold must be positive and finite")
    if capacity > 2**30 - 2:
        raise ValueError("max_intersections is too large for int32 indexing")
    tile_count = tile_width * tile_height
    if tile_count > 2**30 - 2:
        raise ValueError("tile grid is too large for int32 indexing")

    means2d = jax.lax.stop_gradient(jnp.asarray(means2d))
    radii = jax.lax.stop_gradient(jnp.asarray(radii))
    depths = jax.lax.stop_gradient(jnp.asarray(depths))
    valid = jax.lax.stop_gradient(jnp.asarray(valid, dtype=jnp.bool_))
    if means2d.ndim != 2 or means2d.shape[-1] != 2:
        raise ValueError("means2d must have shape [N, 2]")
    gaussian_count = means2d.shape[0]
    if radii.shape != (gaussian_count, 2):
        raise ValueError("radii must have shape [N, 2]")
    if depths.shape != (gaussian_count,):
        raise ValueError("depths must have shape [N]")
    if valid.shape != (gaussian_count,):
        raise ValueError("valid must have shape [N]")
    if (conics is None) != (opacities is None):
        raise ValueError("conics and opacities must be provided together")
    if conics is not None:
        conics = jax.lax.stop_gradient(jnp.asarray(conics))
        opacities = jax.lax.stop_gradient(jnp.asarray(opacities))
        if conics.shape != (gaussian_count, 3):
            raise ValueError("conics must have shape [N, 3]")
        if opacities.shape != (gaussian_count,):
            raise ValueError("opacities must have shape [N]")

    padded_gaussians = jnp.full((capacity,), -1, dtype=jnp.int32)
    padded_tiles = jnp.full((capacity,), -1, dtype=jnp.int32)
    empty_offsets = jnp.zeros((tile_height, tile_width), dtype=jnp.int32)
    if gaussian_count == 0:
        return TileIntersections(
            padded_gaussians,
            padded_tiles,
            empty_offsets,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int32),
        )

    tight_inputs = conics is not None and opacities is not None
    use_accutile = mode != "aabb" and tight_inputs
    if mode == "accutile" and not tight_inputs:
        raise ValueError("mode='accutile' requires conics and opacities")

    with jax.named_scope("intersection_bounds_scan"):
        if use_accutile:
            assert conics is not None
            assert opacities is not None
            accutile_state = _prepare_accutile_state_jax(
                means2d,
                radii,
                conics,
                opacities,
                valid & jnp.isfinite(depths),
                tile_size=tile_size,
                tile_width=tile_width,
                tile_height=tile_height,
                alpha_threshold=alpha_threshold,
            )
            if backend == "pallas":
                tiles_per_gaussian = count_accutile_intersections_pallas(
                    accutile_state,
                    tile_size=tile_size,
                    tile_width=tile_width,
                    tile_height=tile_height,
                )
            else:
                tiles_per_gaussian = _count_accutile_intersections_jax(
                    accutile_state,
                    tile_size=tile_size,
                    tile_width=tile_width,
                    tile_height=tile_height,
                )
        else:
            finite = (
                jnp.all(jnp.isfinite(means2d), axis=-1)
                & jnp.all(jnp.isfinite(radii), axis=-1)
                & jnp.isfinite(depths)
            )
            safe_means = jnp.where(finite[:, None], means2d, 0.0)
            safe_radii = jnp.where(finite[:, None], radii, 0.0).astype(
                means2d.dtype
            )
            tile_min = jnp.floor((safe_means - safe_radii) / tile_size).astype(
                jnp.int32
            )
            tile_max = jnp.ceil((safe_means + safe_radii) / tile_size).astype(
                jnp.int32
            )
            min_x = jnp.clip(tile_min[:, 0], 0, tile_width)
            min_y = jnp.clip(tile_min[:, 1], 0, tile_height)
            max_x = jnp.clip(tile_max[:, 0], 0, tile_width)
            max_y = jnp.clip(tile_max[:, 1], 0, tile_height)
            span_x = jnp.maximum(max_x - min_x, 0)
            span_y = jnp.maximum(max_y - min_y, 0)
            gaussian_valid = valid & finite & jnp.all(radii > 0, axis=-1)
            tiles_per_gaussian = jnp.where(
                gaussian_valid, span_x * span_y, 0
            ).astype(jnp.int32)
        required_cumulative = _saturating_cumsum(
            tiles_per_gaussian, 2**30 - 1
        )
        required_count = required_cumulative[-1]
        cumulative = jnp.minimum(
            required_cumulative, jnp.int32(capacity + 1)
        )
        valid_count = jnp.minimum(required_count, jnp.int32(capacity))
        overflow = required_count > capacity
    if capacity == 0:
        return TileIntersections(
            padded_gaussians,
            padded_tiles,
            empty_offsets,
            valid_count,
            overflow,
            required_count,
        )

    if use_accutile:
        if backend == "pallas":
            with jax.named_scope("intersection_emit_accutile_pallas"):
                gaussian_ids, tile_ids = emit_accutile_intersections_pallas(
                    accutile_state,
                    cumulative,
                    valid_count,
                    capacity=capacity,
                    tile_size=tile_size,
                    tile_width=tile_width,
                    tile_height=tile_height,
                )
        else:
            with jax.named_scope("intersection_emit_accutile_jax"):
                gaussian_ids, tile_ids = _emit_accutile_intersections_jax(
                    accutile_state,
                    cumulative,
                    valid_count,
                    capacity=capacity,
                    tile_size=tile_size,
                    tile_width=tile_width,
                    tile_height=tile_height,
                )
    else:
        with jax.named_scope("intersection_map_jax"):
            gaussian_ids, tile_ids = _map_intersections_jax(
                cumulative,
                min_x,
                min_y,
                span_x,
                valid_count,
                tile_width=tile_width,
                capacity=capacity,
            )
    ranks = jnp.arange(capacity, dtype=jnp.int32)
    output_valid = ranks < valid_count

    if capacity >= _DIRECT_SORT_MIN_CAPACITY:
        tile_keys = jnp.where(output_valid, tile_ids, jnp.int32(tile_count))
        depth_keys = depths[jnp.maximum(gaussian_ids, 0)]
        with jax.named_scope("intersection_sort"):
            tile_keys, _, gaussian_ids = jax.lax.sort(
                (tile_keys, depth_keys, gaussian_ids),
                dimension=0,
                num_keys=3,
                is_stable=False,
            )
        output_valid = tile_keys < tile_count
        gaussian_ids = jnp.where(output_valid, gaussian_ids, -1).astype(jnp.int32)
        tile_ids = jnp.where(output_valid, tile_keys, -1).astype(jnp.int32)
        with jax.named_scope("intersection_offsets"):
            offsets = jnp.searchsorted(
                tile_keys,
                jnp.arange(tile_count, dtype=jnp.int32),
                side="left",
            ).astype(jnp.int32).reshape(tile_height, tile_width)
    else:
        with jax.named_scope("intersection_sort"):
            order = jax.lax.stop_gradient(
                jnp.lexsort(
                    (
                        gaussian_ids,
                        depths[gaussian_ids],
                        tile_ids,
                        (~output_valid).astype(jnp.int32),
                    )
                )
            )
            gaussian_ids = gaussian_ids[order]
            tile_ids = tile_ids[order]
            output_valid = output_valid[order]
            gaussian_ids = jnp.where(output_valid, gaussian_ids, -1).astype(jnp.int32)
            tile_ids = jnp.where(output_valid, tile_ids, -1).astype(jnp.int32)

        with jax.named_scope("intersection_offsets"):
            safe_tile_ids = jnp.clip(tile_ids, 0, tile_count - 1)
            tile_counts = jnp.zeros((tile_count,), dtype=jnp.int32).at[
                safe_tile_ids
            ].add(output_valid.astype(jnp.int32))
            offsets = (jnp.cumsum(tile_counts) - tile_counts).reshape(
                tile_height, tile_width
            )
    return TileIntersections(
        gaussian_ids,
        tile_ids,
        offsets,
        valid_count,
        overflow,
        required_count,
    )


__all__ = ["TileIntersections", "intersect_tiles"]
