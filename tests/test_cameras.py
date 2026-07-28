import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_gs.cameras as cameras_module
from jax_gs.cameras import (
    _pinhole_proj_factors,
    _pinhole_proj_world_factors,
    fisheye_proj,
    fully_fused_projection,
    ortho_proj,
    pinhole_proj,
    world_to_cam,
)
from jax_gs.math import quat_scale_to_covar_preci, quat_to_rotmat


def _intrinsics() -> jax.Array:
    return jnp.array(
        [[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )


def test_world_to_cam_known_transform_and_shapes() -> None:
    means = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]], dtype=jnp.float32)
    covars = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (2, 3, 3))
    viewmat = jnp.eye(4, dtype=jnp.float32).at[:3, 3].set(
        jnp.array([2.0, -1.0, 0.5])
    )[None]
    means_c, covars_c = jax.jit(world_to_cam)(means, covars, viewmat)
    assert means_c.shape == (1, 2, 3)
    assert covars_c.shape == (1, 2, 3, 3)
    np.testing.assert_allclose(means_c[0, 0], [3.0, 1.0, 3.5], atol=1e-6)
    np.testing.assert_allclose(covars_c[0, 0], np.eye(3), atol=1e-6)


def test_pinhole_and_ortho_known_values() -> None:
    means = jnp.array([[[[1.0, 2.0, 4.0]]]], dtype=jnp.float32)
    covars = jnp.eye(3, dtype=jnp.float32)[None, None, None]
    intrinsics = _intrinsics()
    means2d, covars2d = pinhole_proj(means, covars, intrinsics, 100, 100)
    np.testing.assert_allclose(means2d[0, 0, 0], [75.0, 100.0], atol=1e-5)
    jacobian = np.array([[25.0, 0.0, -6.25], [0.0, 25.0, -12.5]])
    np.testing.assert_allclose(covars2d[0, 0, 0], jacobian @ jacobian.T, rtol=1e-5)

    means2d_ortho, covars2d_ortho = ortho_proj(
        means, covars, intrinsics, 100, 100
    )
    np.testing.assert_allclose(means2d_ortho[0, 0, 0], [150.0, 250.0], atol=1e-5)
    np.testing.assert_allclose(covars2d_ortho[0, 0, 0], np.eye(2) * 1e4, atol=1e-3)


def test_pinhole_factor_projection_matches_covariance_projection() -> None:
    means = jnp.array([[0.2, -0.1, 3.0], [-0.3, 0.4, 4.0]], jnp.float32)
    quats = jnp.array(
        [[1.0, 0.2, -0.1, 0.3], [0.9, -0.2, 0.4, 0.1]], jnp.float32
    )
    scales = jnp.array([[0.1, 0.2, 0.3], [0.15, 0.08, 0.25]], jnp.float32)
    viewmat = jnp.array(
        [[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, -0.2], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]],
        jnp.float32,
    )[None]
    covars, _ = quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=False
    )
    means_c, covars_c = world_to_cam(means, covars, viewmat)
    expected_means, expected_covars = pinhole_proj(
        means_c, covars_c, _intrinsics(), 100, 100
    )

    world_factors = quat_to_rotmat(quats) * scales[..., None, :]
    camera_factors = jnp.einsum(
        "cij,njk->cnik", viewmat[:, :3, :3], world_factors
    )
    actual_means, actual_covars = _pinhole_proj_factors(
        means_c, camera_factors, _intrinsics(), 100, 100
    )
    np.testing.assert_allclose(actual_means, expected_means, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_covars, expected_covars, rtol=2e-5, atol=2e-5)

    world_means, world_covars = _pinhole_proj_world_factors(
        means_c,
        world_factors,
        viewmat[:, :3, :3],
        _intrinsics(),
        100,
        100,
    )
    np.testing.assert_allclose(world_means, expected_means, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(world_covars, expected_covars, rtol=2e-5, atol=2e-5)

    factor_loss = lambda value: sum(
        output.sum()
        for output in _pinhole_proj_world_factors(
            means_c,
            value,
            viewmat[:, :3, :3],
            _intrinsics(),
            100,
            100,
        )
    )
    camera_loss = lambda value: sum(
        output.sum()
        for output in _pinhole_proj_factors(
            means_c,
            jnp.einsum("cij,njk->cnik", viewmat[:, :3, :3], value),
            _intrinsics(),
            100,
            100,
        )
    )
    np.testing.assert_allclose(
        jax.grad(factor_loss)(world_factors),
        jax.grad(camera_loss)(world_factors),
        rtol=3e-5,
        atol=3e-5,
    )


def test_fully_fused_factor_route_matches_covariance_route(monkeypatch) -> None:
    monkeypatch.setattr(cameras_module, "_FACTOR_PROJECTION_MIN_GAUSSIANS", 1)
    means = jnp.array(
        [[0.2, -0.1, 3.0], [-0.3, 0.4, 4.0], [0.1, 0.2, 2.5]],
        jnp.float32,
    )
    quats = jnp.array(
        [[1.0, 0.2, -0.1, 0.3], [0.9, -0.2, 0.4, 0.1], [1.0, 0.0, 0.1, 0.0]],
        jnp.float32,
    )
    scales = jnp.array(
        [[0.1, 0.2, 0.3], [0.15, 0.08, 0.25], [0.12, 0.13, 0.09]],
        jnp.float32,
    )
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]

    def factor_projection(current_means, current_quats, current_scales):
        return fully_fused_projection(
            current_means,
            viewmats,
            _intrinsics(),
            100,
            100,
            quats=current_quats,
            scales=current_scales,
            calc_compensations=True,
        )

    def covariance_projection(current_means, current_quats, current_scales):
        covars, _ = quat_scale_to_covar_preci(
            current_quats,
            current_scales,
            compute_covar=True,
            compute_preci=False,
        )
        return fully_fused_projection(
            current_means,
            viewmats,
            _intrinsics(),
            100,
            100,
            covars=covars,
            calc_compensations=True,
        )

    actual = factor_projection(means, quats, scales)
    expected = covariance_projection(means, quats, scales)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_value, expected_value, rtol=3e-5, atol=3e-5)

    def loss(function, current_means, current_quats, current_scales):
        radii, means2d, depths, conics, compensations, valid = function(
            current_means, current_quats, current_scales
        )
        del radii
        return (
            jnp.sum(jnp.where(valid[..., None], means2d, 0.0))
            + jnp.sum(jnp.where(valid, depths, 0.0))
            + jnp.sum(jnp.where(valid[..., None], conics, 0.0))
            + jnp.sum(jnp.where(valid, compensations, 0.0))
        )

    actual_grads = jax.grad(
        lambda *values: loss(factor_projection, *values), argnums=(0, 1, 2)
    )(means, quats, scales)
    expected_grads = jax.grad(
        lambda *values: loss(covariance_projection, *values), argnums=(0, 1, 2)
    )(means, quats, scales)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, rtol=2e-4, atol=2e-4)


def test_fisheye_known_value_and_optical_axis_stability() -> None:
    intrinsics = _intrinsics()
    covars = jnp.eye(3, dtype=jnp.float32)[None, None, None]
    means = jnp.array([[[[1.0, 0.0, 1.0]]]], dtype=jnp.float32)
    means2d, _ = fisheye_proj(means, covars, intrinsics, 100, 100)
    np.testing.assert_allclose(
        means2d[0, 0, 0], [50.0 + 25.0 * np.pi, 50.0], rtol=1e-5
    )

    optical_axis = jnp.array([[[[0.0, 0.0, 2.0]]]], dtype=jnp.float32)
    objective = lambda value: sum(
        output.sum() for output in fisheye_proj(value, covars, intrinsics, 100, 100)
    )
    gradient = jax.jit(jax.grad(objective))(optical_axis)
    assert jnp.all(jnp.isfinite(gradient))


@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
def test_fully_fused_projection_valid_mask_and_grad(camera_model: str) -> None:
    means = jnp.array(
        [[0.0, 0.0, 2.0], [0.0, 0.0, -1.0], [0.2, 0.1, 3.0]],
        dtype=jnp.float32,
    )
    quats = jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32), (3, 1))
    scales = jnp.ones((3, 3), dtype=jnp.float32) * 0.1
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    active_mask = jnp.array([True, True, False])

    projection = lambda value: fully_fused_projection(
        value,
        viewmats,
        _intrinsics(),
        100,
        100,
        quats=quats,
        scales=scales,
        calc_compensations=True,
        camera_model=camera_model,
        active_mask=active_mask,
    )
    radii, means2d, depths, conics, compensations, valid = jax.jit(projection)(means)
    assert radii.shape == (1, 3, 2)
    assert means2d.shape == (1, 3, 2)
    assert depths.shape == (1, 3)
    assert conics.shape == (1, 3, 3)
    assert compensations.shape == (1, 3)
    np.testing.assert_array_equal(valid, [[True, False, False]])
    np.testing.assert_array_equal(radii[0, 1:], 0)

    objective = lambda value: jnp.sum(projection(value)[1]) + jnp.sum(
        projection(value)[3]
    )
    gradient = jax.jit(jax.grad(objective))(means)
    assert jnp.all(jnp.isfinite(gradient))


def test_fully_fused_accepts_triu_covariance() -> None:
    means = jnp.array([[0.0, 0.0, 2.0]], dtype=jnp.float32)
    covariance = jnp.array([[0.01, 0.0, 0.0, 0.01, 0.0, 0.01]], dtype=jnp.float32)
    output = fully_fused_projection(
        means,
        jnp.eye(4, dtype=jnp.float32)[None],
        _intrinsics(),
        100,
        100,
        covars=covariance,
    )
    assert output[0].shape == (1, 1, 2)
    assert bool(output[-1][0, 0])
