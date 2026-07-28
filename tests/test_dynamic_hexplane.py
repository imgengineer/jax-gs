from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from jax_gs.contrib.dynamic.hexplane import (
    HexPlaneField,
    _grid_sample_wrapper,
    _normalize_aabb,
)


_SMALL_CONFIG = {
    "grid_dimensions": 2,
    "input_coordinate_dim": 4,
    "output_coordinate_dim": 4,
    "resolution": [4, 5, 6, 3],
}


def test_hexplane_default_feature_dimension():
    field = HexPlaneField(rngs=nnx.Rngs(0))
    assert field.feat_dim == 64


def test_hexplane_custom_multires_sets_feature_dimension_and_plane_shapes():
    field = HexPlaneField(
        bounds=1.0,
        planes_config=_SMALL_CONFIG,
        multires=(1, 2, 4),
        rngs=nnx.Rngs(1),
    )
    assert field.feat_dim == 12
    assert len(field.grids) == 3
    assert len(field.grids[0]) == 6
    assert field.grids[0][0].shape == (1, 4, 5, 4)
    assert field.grids[1][0].shape == (1, 4, 10, 8)
    assert field.grids[1][2].shape == (1, 4, 3, 8)


def test_hexplane_temporal_planes_start_at_one_and_spatial_planes_are_uniform():
    field = HexPlaneField(
        planes_config=_SMALL_CONFIG,
        multires=(1,),
        rngs=nnx.Rngs(2),
    )
    assert len(field.spatial_planes()) == 3
    assert len(field.temporal_planes()) == 3
    assert all(jnp.all(plane >= 0.1) for plane in field.spatial_planes())
    assert all(jnp.all(plane < 0.5) for plane in field.spatial_planes())
    assert all(jnp.array_equal(plane, jnp.ones_like(plane)) for plane in field.temporal_planes())


def test_grid_sample_wrapper_matches_analytic_bilinear_values_and_border():
    grid = jnp.asarray([[[[0.0, 2.0], [4.0, 6.0]]]])
    coordinates = jnp.asarray(
        [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, -2.0]]
    )
    sampled = _grid_sample_wrapper(grid, coordinates)
    assert sampled.shape == (4, 1)
    assert jnp.allclose(sampled[:, 0], jnp.asarray([0.0, 3.0, 6.0, 2.0]))


def test_grid_sample_wrapper_supports_batched_trilinear_sampling():
    grid = jnp.arange(8, dtype=jnp.float32).reshape(1, 1, 2, 2, 2)
    coordinates = jnp.asarray([[[0.0, 0.0, 0.0]]])
    sampled = _grid_sample_wrapper(grid, coordinates)
    assert sampled.shape == (1, 1)
    assert jnp.allclose(sampled, 3.5)


def test_hexplane_forward_shape_finiteness_and_leading_dimension_flattening():
    field = HexPlaneField(
        planes_config=_SMALL_CONFIG,
        multires=(1, 2),
        rngs=nnx.Rngs(3),
    )
    points = jax.random.uniform(jax.random.key(4), (2, 3, 4), minval=-1.0)
    features = field(points)
    assert features.shape == (6, field.feat_dim)
    assert bool(jnp.all(jnp.isfinite(features)))


def test_hexplane_nnx_jit_matches_eager():
    field = HexPlaneField(
        planes_config=_SMALL_CONFIG,
        multires=(1,),
        rngs=nnx.Rngs(5),
    )
    points = jax.random.uniform(jax.random.key(6), (7, 4), minval=-1.0)
    eager = field(points)
    compiled = nnx.jit(lambda module, values: module(values))(field, points)
    assert jnp.allclose(compiled, eager)


def test_hexplane_gradients_reach_planes_and_coordinates():
    field = HexPlaneField(
        planes_config=_SMALL_CONFIG,
        multires=(1,),
        rngs=nnx.Rngs(7),
    )
    points = jnp.asarray([[0.13, -0.22, 0.31, 0.27]])
    parameter_gradients = nnx.grad(lambda module: jnp.sum(module(points)))(field)
    plane_gradients = [
        parameter_gradients.grids[0][index][...] for index in range(6)
    ]
    assert any(bool(jnp.any(gradient != 0.0)) for gradient in plane_gradients)

    coordinate_gradient = jax.grad(lambda values: jnp.sum(field(values)))(points)
    assert coordinate_gradient.shape == points.shape
    assert bool(jnp.any(coordinate_gradient != 0.0))


def test_hexplane_out_of_bounds_time_uses_border_values():
    field = HexPlaneField(
        planes_config=_SMALL_CONFIG,
        multires=(1,),
        rngs=nnx.Rngs(8),
    )
    low = field(jnp.asarray([[0.0, 0.0, 0.0, -2.0]]))
    low_border = field(jnp.asarray([[0.0, 0.0, 0.0, -1.0]]))
    high = field(jnp.asarray([[0.0, 0.0, 0.0, 2.0]]))
    high_border = field(jnp.asarray([[0.0, 0.0, 0.0, 1.0]]))
    assert jnp.allclose(low, low_border)
    assert jnp.allclose(high, high_border)


def test_hexplane_upstream_aabb_order_reverses_spatial_sign():
    aabb = jnp.asarray([[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]])
    normalized = _normalize_aabb(
        jnp.asarray([[2.0, 0.0, -2.0]]),
        aabb,
    )
    assert jnp.allclose(normalized, jnp.asarray([[-1.0, 0.0, 1.0]]))


def test_hexplane_rejects_invalid_input_and_grid_configuration():
    field = HexPlaneField(
        planes_config=_SMALL_CONFIG,
        multires=(1,),
        rngs=nnx.Rngs(9),
    )
    with pytest.raises(ValueError, match="last dim"):
        field(jnp.zeros((2, 3)))

    bad_config = dict(_SMALL_CONFIG, input_coordinate_dim=5)
    with pytest.raises(ValueError, match=r"len\(reso\)"):
        HexPlaneField(planes_config=bad_config, multires=(1,))
