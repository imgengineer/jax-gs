import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_gs.intersections as intersections_module
from jax_gs.config import RasterizationConfig
from jax_gs.intersections import intersect_tiles


def test_intersections_are_tile_major_depth_sorted_and_padded():
    result = intersect_tiles(
        jnp.array(
            [[1.0, 1.0], [1.0, 1.0], [3.0, 1.0], [5.0, 1.0]],
            dtype=jnp.float32,
        ),
        jnp.full((4, 2), 0.75, dtype=jnp.float32),
        jnp.array([2.0, 1.0, 0.5, 0.25], dtype=jnp.float32),
        jnp.array([True, True, True, False]),
        tile_size=2,
        tile_width=3,
        tile_height=2,
        max_intersections=5,
    )

    np.testing.assert_array_equal(np.asarray(result.gaussian_ids), [1, 0, 2, -1, -1])
    np.testing.assert_array_equal(np.asarray(result.tile_ids), [0, 0, 1, -1, -1])
    np.testing.assert_array_equal(np.asarray(result.offsets), [[0, 2, 3], [3, 3, 3]])
    assert int(result.valid_count) == 3
    assert not bool(result.overflow)


def test_multi_tile_gaussian_and_overflow_are_bounded():
    full = intersect_tiles(
        jnp.array([[2.0, 1.0]], dtype=jnp.float32),
        jnp.array([[2.0, 0.75]], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([True]),
        tile_size=2,
        tile_width=2,
        tile_height=1,
        max_intersections=3,
    )
    np.testing.assert_array_equal(np.asarray(full.gaussian_ids), [0, 0, -1])
    np.testing.assert_array_equal(np.asarray(full.tile_ids), [0, 1, -1])
    np.testing.assert_array_equal(np.asarray(full.offsets), [[0, 1]])
    assert int(full.valid_count) == 2
    assert int(full.required_count) == 2
    assert not bool(full.overflow)

    truncated = intersect_tiles(
        jnp.array([[2.0, 1.0]], dtype=jnp.float32),
        jnp.array([[2.0, 0.75]], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([True]),
        tile_size=2,
        tile_width=2,
        tile_height=1,
        max_intersections=1,
    )
    assert truncated.gaussian_ids.shape == (1,)
    assert int(truncated.valid_count) == 1
    assert int(truncated.required_count) == 2
    assert bool(truncated.overflow)


def test_valid_values_reuse_one_jit_signature_and_indices_are_grad_safe():
    means2d = jnp.array([[1.0, 1.0], [3.0, 1.0]], dtype=jnp.float32)
    radii = jnp.full((2, 2), 0.75, dtype=jnp.float32)
    depths = jnp.array([2.0, 1.0], dtype=jnp.float32)

    @jax.jit
    def build(valid):
        return intersect_tiles(
            means2d,
            radii,
            depths,
            valid,
            tile_size=2,
            tile_width=2,
            tile_height=1,
            max_intersections=3,
        )

    both = build(jnp.array([True, True]))
    cache_size = build._cache_size()
    one = build(jnp.array([True, False]))
    assert build._cache_size() == cache_size == 1
    assert int(both.valid_count) == 2
    assert int(one.valid_count) == 1

    def selected_sum(features):
        intersections = build(jnp.array([True, True]))
        safe_ids = jnp.clip(intersections.gaussian_ids, 0, features.shape[0] - 1)
        mask = (
            jnp.arange(intersections.gaussian_ids.shape[0]) < intersections.valid_count
        )
        return jnp.sum(jnp.where(mask, features[safe_ids], 0.0))

    gradient = jax.grad(selected_sum)(jnp.array([3.0, 4.0], dtype=jnp.float32))
    np.testing.assert_array_equal(np.asarray(gradient), [1.0, 1.0])


def test_zero_count_prefix_and_equal_depth_ties_keep_gaussian_order():
    result = intersect_tiles(
        jnp.array(
            [[20.0, 20.0], [1.0, 1.0], [20.0, 20.0], [1.0, 1.0]],
            dtype=jnp.float32,
        ),
        jnp.full((4, 2), 0.75, dtype=jnp.float32),
        jnp.ones((4,), dtype=jnp.float32),
        jnp.array([False, True, False, True]),
        tile_size=2,
        tile_width=1,
        tile_height=1,
        max_intersections=5,
    )
    np.testing.assert_array_equal(np.asarray(result.gaussian_ids), [1, 3, -1, -1, -1])
    np.testing.assert_array_equal(np.asarray(result.tile_ids), [0, 0, -1, -1, -1])
    assert int(result.valid_count) == 2
    assert not bool(result.overflow)


def test_direct_sort_uses_gaussian_id_as_the_equal_depth_tiebreaker(monkeypatch):
    monkeypatch.setattr(intersections_module, "_DIRECT_SORT_MIN_CAPACITY", 1)
    result = intersect_tiles(
        jnp.array(
            [[20.0, 20.0], [1.0, 1.0], [20.0, 20.0], [1.0, 1.0]],
            dtype=jnp.float32,
        ),
        jnp.full((4, 2), 0.75, dtype=jnp.float32),
        jnp.ones((4,), dtype=jnp.float32),
        jnp.array([False, True, False, True]),
        tile_size=2,
        tile_width=1,
        tile_height=1,
        max_intersections=5,
        backend="jax",
    )
    np.testing.assert_array_equal(np.asarray(result.gaussian_ids), [1, 3, -1, -1, -1])


def test_empty_inputs_and_zero_capacity_are_safe():
    empty = intersect_tiles(
        jnp.empty((0, 2), dtype=jnp.float32),
        jnp.empty((0, 2), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.bool_),
        tile_size=2,
        tile_width=2,
        tile_height=2,
        max_intersections=3,
    )
    np.testing.assert_array_equal(np.asarray(empty.gaussian_ids), [-1, -1, -1])
    np.testing.assert_array_equal(np.asarray(empty.offsets), np.zeros((2, 2), np.int32))
    assert int(empty.valid_count) == 0
    assert not bool(empty.overflow)

    zero_capacity = intersect_tiles(
        jnp.array([[1.0, 1.0]], dtype=jnp.float32),
        jnp.ones((1, 2), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.array([True]),
        tile_size=2,
        tile_width=1,
        tile_height=1,
        max_intersections=0,
    )
    assert zero_capacity.gaussian_ids.shape == (0,)
    assert int(zero_capacity.valid_count) == 0
    assert int(zero_capacity.required_count) == 1
    assert bool(zero_capacity.overflow)


def test_removed_low_level_backend_aliases_are_rejected():
    inputs = (
        jnp.array(
            [[1.0, 1.0], [3.0, 1.0], [20.0, 20.0], [2.0, 3.0]],
            dtype=jnp.float32,
        ),
        jnp.array(
            [[0.75, 0.75], [2.0, 0.75], [1.0, 1.0], [1.5, 1.5]],
            dtype=jnp.float32,
        ),
        jnp.array([2.0, 1.0, 4.0, 0.5], dtype=jnp.float32),
        jnp.array([True, True, False, True]),
    )
    kwargs = {
        "tile_size": 2,
        "tile_width": 3,
        "tile_height": 2,
        "max_intersections": 11,
    }
    with pytest.raises(ValueError, match="backend must"):
        intersect_tiles(*inputs, **kwargs, backend="cutile")
    with pytest.raises(ValueError, match="sort_backend must"):
        intersect_tiles(*inputs, **kwargs, sort_backend="cutile")


def test_intersection_backend_is_validated():
    with pytest.raises(ValueError, match="intersection_backend"):
        RasterizationConfig(intersection_backend="invalid")


def test_sort_backend_is_validated():
    with pytest.raises(ValueError, match="sort_backend"):
        RasterizationConfig(sort_backend="invalid")


def test_intersection_mode_is_validated():
    with pytest.raises(ValueError, match="intersection_mode"):
        RasterizationConfig(intersection_mode="invalid")
