from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs._pallas_intersections import count_accutile_intersections_pallas
from jax_gs.intersections import (
    _count_accutile_intersections_jax,
    _prepare_accutile_state_jax,
    intersect_tiles,
)


_GAUSSIAN_EXTEND = np.float32(3.33)


def _supports_native_pallas() -> bool:
    device = jax.devices()[0]
    try:
        compute_capability = float(
            getattr(device, "compute_capability", 0.0)
        )
    except (TypeError, ValueError):
        compute_capability = 0.0
    return device.platform == "gpu" and compute_capability >= 9.0


@pytest.mark.parametrize("interpret", [True, False], ids=["interpret", "native"])
def test_pallas_accutile_count_matches_jax_with_a_partial_final_block(interpret):
    if not interpret and not _supports_native_pallas():
        pytest.skip("native Pallas AccuTile counting requires a supported GPU")
    count = 129
    means = jnp.stack(
        (
            jnp.linspace(-2.0, 25.0, count, dtype=jnp.float32),
            jnp.linspace(19.0, -1.0, count, dtype=jnp.float32),
        ),
        axis=-1,
    )
    radii = jnp.full((count, 2), 4.0, jnp.float32)
    conics = jnp.tile(
        jnp.asarray([[0.18, 0.03, 0.24]], jnp.float32), (count, 1)
    )
    opacities = jnp.linspace(0.001, 0.9, count, dtype=jnp.float32)
    valid = jnp.arange(count) % 7 != 0
    state = _prepare_accutile_state_jax(
        means,
        radii,
        conics,
        opacities,
        valid,
        tile_size=4,
        tile_width=7,
        tile_height=5,
        alpha_threshold=1.0 / 255.0,
    )
    expected = jax.jit(
        lambda: _count_accutile_intersections_jax(
            state, tile_size=4, tile_width=7, tile_height=5
        )
    )()
    actual = jax.jit(
        lambda: count_accutile_intersections_pallas(
            state,
            tile_size=4,
            tile_width=7,
            tile_height=5,
            interpret=interpret,
        )
    )()
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


class _ReferenceResult(NamedTuple):
    gaussian_ids: np.ndarray
    tile_ids: np.ndarray
    offsets: np.ndarray
    valid_count: int
    overflow: bool
    all_pairs: tuple[tuple[int, int], ...]


def _ellipse_intersection(
    conic: np.ndarray,
    disc: np.float32,
    t: np.float32,
    mean: np.ndarray,
    is_y: bool,
    coord: np.float32,
) -> tuple[np.float32, np.float32]:
    a, b, c = conic
    p_u, p_v = (mean[1], mean[0]) if is_y else (mean[0], mean[1])
    coefficient = a if is_y else c
    h = np.float32(coord - p_u)
    root = np.sqrt(np.float32(disc * h * h + t * coefficient)).astype(
        np.float32
    )
    return (
        np.float32((-b * h - root) / coefficient + p_v),
        np.float32((-b * h + root) / coefficient + p_v),
    )


def _accutile_tiles(
    mean: np.ndarray,
    conic: np.ndarray,
    opacity: np.float32,
    *,
    alpha_threshold: float,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> list[int]:
    threshold = np.float32(alpha_threshold)
    if not np.isfinite(opacity) or opacity <= threshold:
        return []

    a, b, c = conic.astype(np.float32)
    disc = np.float32(b * b - a * c)
    if not np.all(np.isfinite(conic)) or a <= 0 or c <= 0 or disc >= 0:
        return []
    t = np.minimum(
        np.float32(_GAUSSIAN_EXTEND * _GAUSSIAN_EXTEND),
        np.float32(2.0) * np.log(np.float32(opacity / threshold)),
    ).astype(np.float32)
    if not np.isfinite(t) or t <= 0:
        return []

    scale = np.float32(-t / disc)
    x_extent = np.sqrt(np.float32(scale * c)).astype(np.float32)
    y_extent = np.sqrt(np.float32(scale * a)).astype(np.float32)
    bbox_min = np.array(
        [mean[0] - x_extent, mean[1] - y_extent], dtype=np.float32
    )
    bbox_max = np.array(
        [mean[0] + x_extent, mean[1] + y_extent], dtype=np.float32
    )
    bbox_argmin = np.array(
        [mean[1] + b * x_extent / c, mean[0] + b * y_extent / a],
        dtype=np.float32,
    )
    bbox_argmax = np.array(
        [mean[1] - b * x_extent / c, mean[0] - b * y_extent / a],
        dtype=np.float32,
    )

    block = np.float32(tile_size)
    rect_min = np.array(
        [
            np.clip(np.int32(bbox_min[0] / block), 0, tile_width),
            np.clip(np.int32(bbox_min[1] / block), 0, tile_height),
        ],
        dtype=np.int32,
    )
    rect_max = np.array(
        [
            np.clip(np.int32(bbox_max[0] / block + 1.0), 0, tile_width),
            np.clip(np.int32(bbox_max[1] / block + 1.0), 0, tile_height),
        ],
        dtype=np.int32,
    )
    spans = rect_max - rect_min
    if np.any(spans <= 0):
        return []

    is_y = bool(spans[1] < spans[0])
    if is_y:
        rect_min = rect_min[::-1]
        rect_max = rect_max[::-1]
        bbox_min = bbox_min[::-1]
        bbox_max = bbox_max[::-1]
        bbox_argmin = bbox_argmin[::-1]
        bbox_argmax = bbox_argmax[::-1]

    min_line = np.float32(rect_min[0]) * block
    intersect_max = (bbox_max[1], bbox_min[1])
    intersect_min = (
        _ellipse_intersection(conic, disc, t, mean, is_y, min_line)
        if bbox_min[0] <= min_line
        else intersect_max
    )
    tile_ids: list[int] = []
    for u in range(int(rect_min[0]), int(rect_max[0])):
        max_line = np.float32(min_line + block)
        if max_line <= bbox_max[0]:
            intersect_max = _ellipse_intersection(
                conic, disc, t, mean, is_y, max_line
            )

        if min_line <= bbox_argmin[1] < max_line:
            ellipse_min = bbox_min[1]
        else:
            ellipse_min = min(intersect_min[0], intersect_max[0])
        if min_line <= bbox_argmax[1] < max_line:
            ellipse_max = bbox_max[1]
        else:
            ellipse_max = max(intersect_min[1], intersect_max[1])

        min_v = max(
            int(rect_min[1]),
            min(int(rect_max[1]), int(np.int32(ellipse_min / block))),
        )
        max_v = min(
            int(rect_max[1]),
            max(
                int(rect_min[1]),
                int(np.int32(ellipse_max / block + 1.0)),
            ),
        )
        for v in range(min_v, max_v):
            tile_ids.append(
                u * tile_width + v if is_y else v * tile_width + u
            )
        intersect_min = intersect_max
        min_line = max_line
    return tile_ids


def _reference(
    means2d: np.ndarray,
    radii: np.ndarray,
    depths: np.ndarray,
    valid: np.ndarray,
    conics: np.ndarray,
    opacities: np.ndarray,
    *,
    alpha_threshold: float,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    capacity: int,
) -> _ReferenceResult:
    pairs: list[tuple[int, int]] = []
    for gaussian_id in range(means2d.shape[0]):
        finite = (
            np.all(np.isfinite(means2d[gaussian_id]))
            and np.all(np.isfinite(radii[gaussian_id]))
            and np.isfinite(depths[gaussian_id])
        )
        if (
            not valid[gaussian_id]
            or not finite
            or np.any(radii[gaussian_id] <= 0)
        ):
            continue
        pairs.extend(
            (gaussian_id, tile_id)
            for tile_id in _accutile_tiles(
                means2d[gaussian_id],
                conics[gaussian_id],
                opacities[gaussian_id],
                alpha_threshold=alpha_threshold,
                tile_size=tile_size,
                tile_width=tile_width,
                tile_height=tile_height,
            )
        )

    retained = pairs[:capacity]
    retained.sort(key=lambda pair: (pair[1], depths[pair[0]], pair[0]))
    gaussian_ids = np.full((capacity,), -1, dtype=np.int32)
    tile_ids = np.full((capacity,), -1, dtype=np.int32)
    for rank, (gaussian_id, tile_id) in enumerate(retained):
        gaussian_ids[rank] = gaussian_id
        tile_ids[rank] = tile_id

    tile_count = tile_width * tile_height
    counts = np.bincount(
        np.asarray([tile_id for _, tile_id in retained], dtype=np.int32),
        minlength=tile_count,
    ).astype(np.int32)
    offsets = (np.cumsum(counts, dtype=np.int32) - counts).reshape(
        tile_height, tile_width
    )
    return _ReferenceResult(
        gaussian_ids,
        tile_ids,
        offsets,
        len(retained),
        len(pairs) > capacity,
        tuple(pairs),
    )


def _covariance_case(
    std_major: float, std_minor: float, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.float32(np.cos(angle))
    sine = np.float32(np.sin(angle))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    covariance = rotation @ np.diag(
        np.square(np.array([std_major, std_minor], dtype=np.float32))
    ) @ rotation.T
    inverse = np.linalg.inv(covariance).astype(np.float32)
    return covariance, np.array(
        [inverse[0, 0], inverse[0, 1], inverse[1, 1]], dtype=np.float32
    )


def _opacity_radii(
    covariances: np.ndarray, opacities: np.ndarray, alpha_threshold: float
) -> np.ndarray:
    ratio = np.maximum(
        opacities.astype(np.float32) / np.float32(alpha_threshold),
        np.float32(1.0),
    )
    extent = np.minimum(
        _GAUSSIAN_EXTEND,
        np.sqrt(np.float32(2.0) * np.log(ratio)).astype(np.float32),
    )
    radii = np.ceil(
        extent[:, None]
        * np.sqrt(
            np.stack(
                (covariances[:, 0, 0], covariances[:, 1, 1]), axis=-1
            )
        )
    )
    return radii.astype(np.float32)


def _assert_matches_reference(
    means2d: np.ndarray,
    radii: np.ndarray,
    depths: np.ndarray,
    valid: np.ndarray,
    conics: np.ndarray,
    opacities: np.ndarray,
    *,
    alpha_threshold: float,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    capacity: int,
) -> _ReferenceResult:
    expected = _reference(
        means2d,
        radii,
        depths,
        valid,
        conics,
        opacities,
        alpha_threshold=alpha_threshold,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        capacity=capacity,
    )
    actual = intersect_tiles(
        jnp.asarray(means2d),
        jnp.asarray(radii),
        jnp.asarray(depths),
        jnp.asarray(valid),
        conics=jnp.asarray(conics),
        opacities=jnp.asarray(opacities),
        alpha_threshold=alpha_threshold,
        mode="accutile",
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        max_intersections=capacity,
        backend="jax",
        sort_backend="jax",
    )
    np.testing.assert_array_equal(
        np.asarray(actual.gaussian_ids), expected.gaussian_ids
    )
    np.testing.assert_array_equal(np.asarray(actual.tile_ids), expected.tile_ids)
    np.testing.assert_array_equal(np.asarray(actual.offsets), expected.offsets)
    assert int(actual.valid_count) == expected.valid_count
    assert bool(actual.overflow) == expected.overflow
    return expected


def test_accutile_matches_float32_reference_for_random_spd_conics_and_clipping():
    rng = np.random.default_rng(9237)
    count = 9
    tile_size = 4
    tile_width = 6
    tile_height = 5
    alpha_threshold = 1.0 / 255.0
    means2d = rng.uniform(
        [-3.0, -2.0],
        [tile_width * tile_size + 3.0, tile_height * tile_size + 2.0],
        size=(count, 2),
    ).astype(np.float32)
    covariances = []
    conics = []
    for major, minor, angle in zip(
        rng.uniform(1.2, 3.2, count),
        rng.uniform(0.4, 1.1, count),
        rng.uniform(-np.pi, np.pi, count),
        strict=True,
    ):
        covariance, conic = _covariance_case(major, minor, angle)
        covariances.append(covariance)
        conics.append(conic)
    covariances = np.asarray(covariances, dtype=np.float32)
    conics = np.asarray(conics, dtype=np.float32)
    opacities = rng.uniform(0.03, 0.9, count).astype(np.float32)
    radii = _opacity_radii(covariances, opacities, alpha_threshold)
    depths = np.linspace(0.5, 4.5, count, dtype=np.float32)[::-1].copy()
    valid = np.array([True, True, False, True, True, True, False, True, True])

    _assert_matches_reference(
        means2d,
        radii,
        depths,
        valid,
        conics,
        opacities,
        alpha_threshold=alpha_threshold,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        capacity=192,
    )


def test_accutile_rotated_ellipse_is_a_strict_subset_of_aabb_pairs():
    tile_size = 4
    tile_width = tile_height = 9
    alpha_threshold = 1.0 / 255.0
    covariance, conic = _covariance_case(6.0, 0.55, np.pi / 4.0)
    means2d = np.array([[18.25, 17.75]], dtype=np.float32)
    conics = conic[None]
    opacities = np.array([0.9], dtype=np.float32)
    radii = _opacity_radii(covariance[None], opacities, alpha_threshold)
    depths = np.array([1.0], dtype=np.float32)
    valid = np.array([True])
    kwargs = dict(
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        max_intersections=tile_width * tile_height,
        sort_backend="jax",
    )

    exact = _assert_matches_reference(
        means2d,
        radii,
        depths,
        valid,
        conics,
        opacities,
        alpha_threshold=alpha_threshold,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        capacity=tile_width * tile_height,
    )
    aabb = intersect_tiles(
        jnp.asarray(means2d),
        jnp.asarray(radii),
        jnp.asarray(depths),
        jnp.asarray(valid),
        backend="jax",
        **kwargs,
    )
    exact_pairs = set(exact.all_pairs)
    aabb_pairs = set(
        zip(
            np.asarray(aabb.gaussian_ids)[: int(aabb.valid_count)].tolist(),
            np.asarray(aabb.tile_ids)[: int(aabb.valid_count)].tolist(),
            strict=True,
        )
    )
    assert exact_pairs < aabb_pairs


def test_accutile_applies_opacity_threshold_before_the_ellipse_walk():
    alpha_threshold = 1.0 / 255.0
    _, conic = _covariance_case(2.0, 1.0, 0.2)
    means2d = np.repeat(np.array([[5.0, 5.0]], dtype=np.float32), 3, axis=0)
    conics = np.repeat(conic[None], 3, axis=0)
    opacities = np.asarray(
        [alpha_threshold * 0.9, alpha_threshold, alpha_threshold * 1.2],
        dtype=np.float32,
    )
    radii = np.repeat(np.array([[2.0, 2.0]], dtype=np.float32), 3, axis=0)
    depths = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    valid = np.ones((3,), dtype=np.bool_)

    expected = _assert_matches_reference(
        means2d,
        radii,
        depths,
        valid,
        conics,
        opacities,
        alpha_threshold=alpha_threshold,
        tile_size=4,
        tile_width=3,
        tile_height=3,
        capacity=8,
    )
    assert {gaussian_id for gaussian_id, _ in expected.all_pairs} == {2}


def test_accutile_overflow_keeps_gaussian_major_prefix_and_padding_is_minus_one():
    alpha_threshold = 1.0 / 255.0
    covariance, conic = _covariance_case(4.0, 1.2, -0.6)
    means2d = np.array(
        [[7.0, 7.0], [10.0, 8.0], [13.0, 10.0]], dtype=np.float32
    )
    covariances = np.repeat(covariance[None], 3, axis=0)
    conics = np.repeat(conic[None], 3, axis=0)
    opacities = np.array([0.8, 0.7, 0.6], dtype=np.float32)
    radii = _opacity_radii(covariances, opacities, alpha_threshold)
    depths = np.array([2.0, 1.0, 0.5], dtype=np.float32)
    valid = np.ones((3,), dtype=np.bool_)
    common = dict(
        alpha_threshold=alpha_threshold,
        tile_size=4,
        tile_width=5,
        tile_height=4,
    )
    full = _reference(
        means2d,
        radii,
        depths,
        valid,
        conics,
        opacities,
        capacity=60,
        **common,
    )
    assert len(full.all_pairs) > 6

    truncated = _assert_matches_reference(
        means2d,
        radii,
        depths,
        valid,
        conics,
        opacities,
        capacity=len(full.all_pairs) - 3,
        **common,
    )
    assert truncated.overflow

    padded = _assert_matches_reference(
        means2d,
        radii,
        depths,
        valid,
        conics,
        opacities,
        capacity=len(full.all_pairs) + 4,
        **common,
    )
    assert not padded.overflow
    np.testing.assert_array_equal(
        padded.gaussian_ids[padded.valid_count :], np.full((4,), -1, np.int32)
    )
    np.testing.assert_array_equal(
        padded.tile_ids[padded.valid_count :], np.full((4,), -1, np.int32)
    )


def test_accutile_matches_reference_on_a_wide_grid_with_multi_column_ellipses():
    # Emission resolves each output slot's Gaussian from the prefix sums, so
    # the cases that matter are the ones where a Gaussian spans many columns
    # and the runs of several Gaussians meet inside the buffer. A wide,
    # shallow grid also forces the walk onto its transposed axis.
    rng = np.random.default_rng(4471)
    count = 64
    tile_size = 4
    tile_width = 17
    tile_height = 3
    alpha_threshold = 1.0 / 255.0
    means2d = rng.uniform(
        [-6.0, -4.0],
        [tile_width * tile_size + 6.0, tile_height * tile_size + 4.0],
        size=(count, 2),
    ).astype(np.float32)
    covariances = []
    conics = []
    for major, minor, angle in zip(
        rng.uniform(1.5, 9.0, count),
        rng.uniform(0.3, 1.4, count),
        rng.uniform(-np.pi, np.pi, count),
        strict=True,
    ):
        covariance, conic = _covariance_case(major, minor, angle)
        covariances.append(covariance)
        conics.append(conic)
    covariances = np.asarray(covariances, dtype=np.float32)
    conics = np.asarray(conics, dtype=np.float32)
    opacities = rng.uniform(0.02, 0.99, count).astype(np.float32)
    radii = _opacity_radii(covariances, opacities, alpha_threshold)
    depths = np.linspace(0.4, 9.0, count, dtype=np.float32)[::-1].copy()
    valid = rng.random(count) > 0.15

    result = _assert_matches_reference(
        means2d,
        radii,
        depths,
        valid,
        conics,
        opacities,
        alpha_threshold=alpha_threshold,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        capacity=4096,
    )

    # A Gaussian reaches a tile at most once. The compositor's chunk loop is
    # sized on that, so it is worth asserting separately from the reference.
    emitted = int(result.valid_count)
    assert emitted > count, "the case should emit multi-tile runs"
    pairs = np.stack(
        (result.tile_ids[:emitted], result.gaussian_ids[:emitted]), axis=-1
    )
    assert len(np.unique(pairs, axis=0)) == emitted
