import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jax_gs.contrib.dynamic import HexPlaneField
from jax_gs.contrib.dynamic.regulation import (
    hexplane_regularization,
    plane_smoothness,
    time_l1,
    time_smoothness,
)


def test_plane_smoothness_is_zero_for_constant_and_linear_planes():
    constant = jnp.full((1, 4, 8, 8), 0.5)
    linear = jnp.broadcast_to(
        jnp.arange(8, dtype=jnp.float32)[None, None, :, None],
        (1, 4, 8, 8),
    )
    assert jnp.allclose(plane_smoothness([constant]), 0.0)
    assert jnp.allclose(plane_smoothness([linear]), 0.0)


def test_plane_smoothness_is_positive_for_curvature_and_sums_planes():
    coordinates = jnp.arange(8, dtype=jnp.float32)
    curved = jnp.broadcast_to(
        (coordinates**2)[None, None, :, None],
        (1, 2, 8, 4),
    )
    one = plane_smoothness([curved])
    two = plane_smoothness([curved, curved])
    assert float(one) > 0.0
    assert jnp.allclose(two, 2.0 * one)


def test_smoothness_validates_rank_and_skips_short_height():
    with pytest.raises(ValueError, match="4D"):
        plane_smoothness([jnp.zeros((8, 8))])
    short = jnp.zeros((1, 2, 2, 4), dtype=jnp.float16)
    result = plane_smoothness([short])
    assert result.dtype == jnp.float16
    assert jnp.array_equal(result, jnp.asarray(0.0, dtype=jnp.float16))


def test_time_smoothness_uses_same_math_and_jits():
    plane = jax.random.normal(jax.random.key(0), (1, 3, 6, 6))
    expected = plane_smoothness([plane])
    actual = jax.jit(time_smoothness)([plane])
    assert jnp.allclose(actual, expected)


def test_time_l1_values_sum_and_empty_sequence_contract():
    first = jnp.full((1, 1, 2, 2), 0.5)
    second = jnp.full((1, 1, 2, 2), 0.8)
    assert jnp.allclose(time_l1([jnp.ones_like(first)]), 0.0)
    assert jnp.allclose(time_l1([first, second]), 0.7)
    assert time_l1([]).shape == ()
    assert time_l1([]).dtype == jnp.float32


def test_time_l1_gradient_matches_mean_absolute_derivative():
    plane = jnp.full((1, 1, 4, 4), 0.5)
    gradient = jax.grad(lambda value: time_l1([value]))(plane)
    assert jnp.allclose(gradient, jnp.full_like(plane, -1.0 / 16.0))


def test_hexplane_regularization_uses_field_partition_and_weights():
    config = {
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 2,
        "resolution": [4, 5, 6, 3],
    }
    field = HexPlaneField(
        planes_config=config,
        multires=(1,),
        rngs=nnx.Rngs(1),
    )
    expected = (
        2.0 * plane_smoothness(field.spatial_planes())
        + 3.0 * time_smoothness(field.temporal_planes())
        + 4.0 * time_l1(field.temporal_planes())
    )
    actual = hexplane_regularization(
        field,
        lambda_plane_smooth=2.0,
        lambda_time_smooth=3.0,
        lambda_time_l1=4.0,
    )
    assert jnp.allclose(actual, expected)
