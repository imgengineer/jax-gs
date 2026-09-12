from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_gs.three_dgut import (
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    RollingShutterType,
    UnscentedTransformParameters,
    _world_rays_from_pixels,
    compute_ut_weights,
    fully_fused_projection_with_ut,
    project_camera_points,
    project_world_points,
    world_gaussian_sigma_points,
)


def _camera(width: int = 101, height: int = 101):
    viewmat = jnp.eye(4, dtype=jnp.float32)[None]
    K = jnp.asarray(
        [[[100.0, 0.0, width / 2.0], [0.0, 100.0, height / 2.0], [0.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )
    return viewmat, K


def test_ut_sigma_points_reconstruct_gaussian_moments() -> None:
    params = UnscentedTransformParameters(alpha=0.5, beta=2.0, kappa=0.0)
    means = jnp.asarray([[1.0, -2.0, 3.0]], dtype=jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    scales = jnp.asarray([[0.2, 0.3, 0.4]], dtype=jnp.float32)
    sigma_points = world_gaussian_sigma_points(means, quats, scales, params)
    weights_mean, weights_cov = compute_ut_weights(params)

    reconstructed_mean = jnp.sum(sigma_points * weights_mean[:, None], axis=-2)
    delta = sigma_points - reconstructed_mean[..., None, :]
    reconstructed_covariance = jnp.sum(
        weights_cov[:, None, None] * delta[..., :, :, None] * delta[..., :, None, :],
        axis=-3,
    )

    assert sigma_points.shape == (1, 7, 3)
    np.testing.assert_allclose(reconstructed_mean, means, atol=1.0e-6)
    np.testing.assert_allclose(
        reconstructed_covariance,
        np.diag(np.square(np.asarray(scales[0])))[None],
        atol=1.0e-6,
    )


def test_opencv_pinhole_and_fisheye_distortion() -> None:
    camera_points = jnp.asarray([[[0.2, 0.1, 1.0]]], dtype=jnp.float32)
    K = jnp.asarray(
        [[[100.0, 0.0, 20.0], [0.0, 80.0, 10.0], [0.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )
    radial = jnp.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    tangential = jnp.asarray([0.01, -0.02], dtype=jnp.float32)
    thin_prism = jnp.asarray([0.001, 0.0, -0.001, 0.0], dtype=jnp.float32)
    image_points, valid = project_camera_points(
        camera_points,
        K,
        200,
        200,
        radial_coeffs=radial,
        tangential_coeffs=tangential,
        thin_prism_coeffs=thin_prism,
    )
    r2 = 0.2**2 + 0.1**2
    radial_scale = 1.0 + 0.1 * r2
    expected_u = radial_scale * 0.2 + 2.0 * 0.01 * 0.2 * 0.1
    expected_u += -0.02 * (r2 + 2.0 * 0.2**2) + 0.001 * r2
    expected_v = radial_scale * 0.1 + 0.01 * (r2 + 2.0 * 0.1**2)
    expected_v += 2.0 * -0.02 * 0.2 * 0.1 - 0.001 * r2
    np.testing.assert_allclose(
        image_points[0, 0],
        [100.0 * expected_u + 20.0, 80.0 * expected_v + 10.0],
        atol=1.0e-5,
    )
    assert bool(valid[0, 0])

    fisheye_point = jnp.asarray([[[1.0, 0.0, 1.0]]], dtype=jnp.float32)
    fisheye_image, fisheye_valid = project_camera_points(
        fisheye_point,
        K,
        200,
        200,
        camera_model="fisheye",
        radial_coeffs=jnp.asarray([0.1, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )
    theta = math.pi / 4.0
    expected_radius = theta * (1.0 + 0.1 * theta * theta)
    np.testing.assert_allclose(
        fisheye_image[0, 0], [20.0 + 100.0 * expected_radius, 10.0], atol=2.0e-5
    )
    assert bool(fisheye_valid[0, 0])


def test_ftheta_projection_uses_polynomial_and_half_pixel_origin() -> None:
    parameters = FThetaCameraDistortionParameters(
        reference_poly=FThetaPolynomialType.ANGLE_TO_PIXELDIST,
        pixeldist_to_angle_poly=(0.0, 0.01, 0.0, 0.0, 0.0, 0.0),
        angle_to_pixeldist_poly=(0.0, 100.0, 0.0, 0.0, 0.0, 0.0),
        max_angle=math.pi / 2.0,
        linear_cde=(1.0, 0.0, 0.0),
    )
    camera_points = jnp.asarray([[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], dtype=jnp.float32)
    K = jnp.asarray(
        [[[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )
    image_points, valid = project_camera_points(
        camera_points,
        K,
        200,
        200,
        camera_model="ftheta",
        ftheta_coeffs=parameters,
    )

    np.testing.assert_allclose(image_points[0, 0], [10.5, 20.5], atol=1.0e-5)
    np.testing.assert_allclose(
        image_points[0, 1], [10.5 + 25.0 * math.pi, 20.5], atol=2.0e-5
    )
    np.testing.assert_array_equal(valid, [[True, True]])


def test_distorted_camera_projection_and_world_ray_unprojection_are_inverses():
    viewmat = jnp.eye(4, dtype=jnp.float32)
    K = jnp.asarray(
        [[100.0, 0.0, 20.0], [0.0, 80.0, 10.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    camera_direction = jnp.asarray([[0.2, 0.1, 1.0]], jnp.float32)
    expected_direction = camera_direction / jnp.linalg.norm(
        camera_direction, axis=-1, keepdims=True
    )
    radial = jnp.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], jnp.float32)
    tangential = jnp.asarray([0.01, -0.02], jnp.float32)
    thin_prism = jnp.asarray([0.001, 0.0, -0.001, 0.0], jnp.float32)
    pinhole_pixels, _ = project_camera_points(
        camera_direction[None],
        K[None],
        200,
        200,
        radial_coeffs=radial,
        tangential_coeffs=tangential,
        thin_prism_coeffs=thin_prism,
    )
    origins, directions, valid = _world_rays_from_pixels(
        pinhole_pixels[0],
        viewmat,
        K,
        200,
        200,
        camera_model="pinhole",
        radial_coeffs=radial,
        tangential_coeffs=tangential,
        thin_prism_coeffs=thin_prism,
        ftheta_coeffs=None,
        rolling_shutter=RollingShutterType.GLOBAL,
        viewmat_rs=None,
    )
    np.testing.assert_allclose(origins, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(directions, expected_direction, atol=2.0e-6)
    assert bool(valid[0])

    fisheye_radial = jnp.asarray([0.1, 0.0, 0.0, 0.0], jnp.float32)
    fisheye_pixels, _ = project_camera_points(
        camera_direction[None],
        K[None],
        200,
        200,
        camera_model="fisheye",
        radial_coeffs=fisheye_radial,
    )
    _, directions, valid = _world_rays_from_pixels(
        fisheye_pixels[0],
        viewmat,
        K,
        200,
        200,
        camera_model="fisheye",
        radial_coeffs=fisheye_radial,
        tangential_coeffs=None,
        thin_prism_coeffs=None,
        ftheta_coeffs=None,
        rolling_shutter=RollingShutterType.GLOBAL,
        viewmat_rs=None,
    )
    np.testing.assert_allclose(directions, expected_direction, atol=2.0e-6)
    assert bool(valid[0])

    ftheta = FThetaCameraDistortionParameters(
        reference_poly=FThetaPolynomialType.ANGLE_TO_PIXELDIST,
        pixeldist_to_angle_poly=(0.0, 0.01, 0.0, 0.0, 0.0, 0.0),
        angle_to_pixeldist_poly=(0.0, 100.0, 0.0, 0.0, 0.0, 0.0),
        max_angle=math.pi / 2.0,
        linear_cde=(1.0, 0.0, 0.0),
    )
    ftheta_pixels, _ = project_camera_points(
        camera_direction[None],
        K[None],
        200,
        200,
        camera_model="ftheta",
        ftheta_coeffs=ftheta,
    )
    _, directions, valid = _world_rays_from_pixels(
        ftheta_pixels[0],
        viewmat,
        K,
        200,
        200,
        camera_model="ftheta",
        radial_coeffs=None,
        tangential_coeffs=None,
        thin_prism_coeffs=None,
        ftheta_coeffs=ftheta,
        rolling_shutter=RollingShutterType.GLOBAL,
        viewmat_rs=None,
    )
    np.testing.assert_allclose(directions, expected_direction, atol=2.0e-6)
    assert bool(valid[0])


def test_rolling_shutter_world_ray_uses_pixel_readout_pose():
    start_view = jnp.eye(4, dtype=jnp.float32)
    end_view = start_view.at[0, 3].set(1.0)
    K = jnp.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        jnp.float32,
    )
    origins, directions, valid = _world_rays_from_pixels(
        jnp.asarray([[50.0, 50.0]], jnp.float32),
        start_view,
        K,
        101,
        101,
        camera_model="pinhole",
        radial_coeffs=None,
        tangential_coeffs=None,
        thin_prism_coeffs=None,
        ftheta_coeffs=None,
        rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
        viewmat_rs=end_view,
    )

    np.testing.assert_allclose(origins[0], [-0.5, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(directions[0], [0.0, 0.0, 1.0], atol=1.0e-6)
    assert bool(valid[0])


def test_rolling_shutter_iterates_pose_at_pixel_readout_time() -> None:
    start_view = jnp.eye(4, dtype=jnp.float32)[None]
    end_view = start_view.at[0, 0, 3].set(1.0)
    K = jnp.asarray(
        [[[100.0, 0.0, 50.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )
    world_point = jnp.asarray([[[0.0, 0.5, 2.0]]], dtype=jnp.float32)

    global_image, _ = project_world_points(world_point, start_view, K, 101, 101)
    rolling_image, rolling_valid = project_world_points(
        world_point,
        start_view,
        K,
        101,
        101,
        rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
        viewmats_rs=end_view,
        rolling_shutter_iterations=2,
    )

    np.testing.assert_allclose(global_image[0, 0], [50.0, 25.0], atol=1.0e-6)
    np.testing.assert_allclose(rolling_image[0, 0], [62.5, 25.0], atol=1.0e-5)
    assert bool(rolling_valid[0, 0])


def test_active_mask_is_jittable_with_static_chunked_shapes() -> None:
    means = jnp.asarray([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]], dtype=jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=jnp.float32)
    scales = jnp.asarray([[0.1, 0.1, 0.1]] * 2, dtype=jnp.float32)
    opacities = jnp.asarray([0.8, 0.8], dtype=jnp.float32)
    viewmats, Ks = _camera()

    projection = jax.jit(
        lambda mask: fully_fused_projection_with_ut(
            means,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            101,
            101,
            calc_compensations=True,
            active_mask=mask,
            ut_chunk_size=1,
        )
    )
    first = projection(jnp.asarray([True, False]))
    second = projection(jnp.asarray([False, True]))

    assert first[0].shape == second[0].shape == (1, 2, 2)
    assert first[1].shape == second[1].shape == (1, 2, 2)
    assert first[4] is not None and first[4].shape == (1, 2)
    np.testing.assert_array_equal(first[-1], [[True, False]])
    np.testing.assert_array_equal(second[-1], [[False, True]])
    np.testing.assert_array_equal(first[0][0, 1], 0)
    np.testing.assert_array_equal(second[0][0, 0], 0)
