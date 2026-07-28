from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.sensors import functional
from jax_gs.sensors.kernels.cameras import (
    BivariateWindshieldDistortion,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ReferencePolynomial,
    camera_rays_to_image_points,
    from_components,
    generate_image_points,
    image_points_to_camera_rays,
)


def _pinhole(*, distorted: bool = False) -> OpenCVPinholeProjection:
    return OpenCVPinholeProjection(
        focal_length=jnp.array([100.0, 120.0]),
        principal_point=jnp.array([50.0, 40.0]),
        radial_coeffs=jnp.array([0.01, -0.002, 0.0001, 0.0, 0.0, 0.0])
        if distorted
        else jnp.zeros(6),
        tangential_coeffs=jnp.array([0.001, -0.002])
        if distorted
        else jnp.zeros(2),
        thin_prism_coeffs=jnp.array([0.0001, 0.0, -0.0001, 0.0])
        if distorted
        else jnp.zeros(4),
        resolution=(100, 80),
    )


def _ftheta(reference: int) -> FThetaProjection:
    return FThetaProjection(
        principal_point=jnp.array([50.0, 40.0]),
        fw_poly=jnp.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
        bw_poly=jnp.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        A=jnp.array([1.0, 0.0, 0.0, 1.0]),
        resolution=(100, 80),
        reference_polynomial=reference,
        fw_poly_degree=1,
        bw_poly_degree=1,
        newton_iterations=10,
        max_angle=1.4,
        min_2d_norm=1.0e-6,
    )


def _fisheye() -> OpenCVFisheyeProjection:
    return OpenCVFisheyeProjection(
        principal_point=jnp.array([50.0, 40.0]),
        focal_length=jnp.array([100.0, 100.0]),
        forward_poly=jnp.array([0.01, -0.001, 0.0001, 0.0]),
        approx_backward_factor=jnp.array([1.0]),
        resolution=(100, 80),
        newton_iterations=10,
        max_angle=1.8,
        min_2d_norm=1.0e-6,
    )


def _identity_windshield(reference: int = 0) -> BivariateWindshieldDistortion:
    horizontal = jnp.array([0.0, 1.0, 0.0])
    vertical = jnp.array([0.0, 0.0, 1.0])
    return from_components(
        horizontal, vertical, horizontal, vertical, reference
    )


def test_pinhole_projection_and_backprojection_match_closed_form():
    projection = _pinhole()
    rays = jnp.array([[0.0, 0.0, 1.0], [0.1, 0.2, 1.0], [0.0, 0.0, -1.0]])
    points, valid = camera_rays_to_image_points(
        rays, projection, NoExternalDistortion()
    )
    np.testing.assert_allclose(points[:2], [[50.0, 40.0], [60.0, 64.0]])
    np.testing.assert_array_equal(valid, [True, True, False])
    np.testing.assert_array_equal(points[2], [0.0, 0.0])
    rays_back = image_points_to_camera_rays(
        points[:2], projection, NoExternalDistortion()
    )
    expected = rays[:2] / jnp.linalg.norm(rays[:2], axis=-1, keepdims=True)
    np.testing.assert_allclose(rays_back, expected, atol=1.0e-6)


def test_distorted_pinhole_matches_open_cv_formula_and_round_trips():
    projection = _pinhole(distorted=True)
    rays = jnp.array([[0.05, -0.03, 1.0], [-0.08, 0.04, 1.0]])
    points, valid = camera_rays_to_image_points(
        rays, projection, NoExternalDistortion()
    )
    x, y = rays[:, 0], rays[:, 1]
    radius_squared = x * x + y * y
    radius_fourth = radius_squared**2
    radius_sixth = radius_squared**3
    k = projection.radial_coeffs
    radial = (1 + k[0] * radius_squared + k[1] * radius_fourth + k[2] * radius_sixth) / (
        1 + k[3] * radius_squared + k[4] * radius_fourth + k[5] * radius_sixth
    )
    p = projection.tangential_coeffs
    s = projection.thin_prism_coeffs
    delta_x = 2 * p[0] * x * y + p[1] * (radius_squared + 2 * x * x) + s[0] * radius_squared + s[1] * radius_fourth
    delta_y = p[0] * (radius_squared + 2 * y * y) + 2 * p[1] * x * y + s[2] * radius_squared + s[3] * radius_fourth
    expected = jnp.stack((x * radial + delta_x, y * radial + delta_y), axis=-1)
    expected = expected * projection.focal_length + projection.principal_point
    np.testing.assert_allclose(points, expected, atol=1.0e-6)
    assert bool(jnp.all(valid))
    roundtrip, roundtrip_valid = camera_rays_to_image_points(
        image_points_to_camera_rays(points, projection, NoExternalDistortion()),
        projection,
        NoExternalDistortion(),
    )
    np.testing.assert_allclose(roundtrip, points, atol=1.0e-4)
    assert bool(jnp.all(roundtrip_valid))


@pytest.mark.parametrize("projection", [_pinhole(), _ftheta(0), _fisheye()])
@pytest.mark.parametrize("reference", [ReferencePolynomial.FORWARD, ReferencePolynomial.BACKWARD])
def test_identity_windshield_matches_no_external(projection, reference):
    rays = jnp.array([[0.0, 0.0, 1.0], [0.05, 0.02, 1.0], [-0.04, 0.01, 1.0]])
    plain = camera_rays_to_image_points(rays, projection, NoExternalDistortion())
    windshield = camera_rays_to_image_points(
        rays, projection, _identity_windshield(int(reference))
    )
    np.testing.assert_allclose(windshield[0], plain[0], atol=1.0e-4, rtol=1.0e-5)
    np.testing.assert_array_equal(windshield[1], plain[1])
    plain_rays = image_points_to_camera_rays(
        plain[0], projection, NoExternalDistortion()
    )
    windshield_rays = image_points_to_camera_rays(
        plain[0], projection, _identity_windshield(int(reference))
    )
    np.testing.assert_allclose(
        windshield_rays, plain_rays, atol=1.0e-4, rtol=1.0e-5
    )


@pytest.mark.parametrize("reference", [0, 1])
def test_ftheta_linear_polynomials_project_and_round_trip(reference):
    projection = _ftheta(reference)
    rays = jnp.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.04, 1.0]])
    points, valid = camera_rays_to_image_points(
        rays, projection, NoExternalDistortion()
    )
    expected = jnp.array(
        [
            [50.0, 40.0],
            [50.0 + 100.0 * jnp.arctan(0.05), 40.0],
            [50.0, 40.0 + 100.0 * jnp.arctan(0.04)],
        ]
    )
    np.testing.assert_allclose(points, expected, atol=2.0e-4)
    assert bool(jnp.all(valid))
    rays_back = image_points_to_camera_rays(
        points, projection, NoExternalDistortion()
    )
    normalized = rays / jnp.linalg.norm(rays, axis=-1, keepdims=True)
    np.testing.assert_allclose(rays_back, normalized, atol=2.0e-5)


def test_ftheta_marks_angle_frame_and_nan_failures_invalid():
    projection = _ftheta(0)
    rays = jnp.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 1.0], [jnp.nan, 0.0, 1.0]]
    )
    points, valid = camera_rays_to_image_points(
        rays, projection, NoExternalDistortion()
    )
    np.testing.assert_array_equal(valid, [False, False, False])
    np.testing.assert_array_equal(points[0], [0.0, 0.0])


def test_fisheye_round_trip_clamps_angle_and_zeroes_out_of_frame_points():
    projection = _fisheye()
    rays = jnp.array([[0.0, 0.0, 1.0], [0.1, 0.04, 1.0]])
    points, valid = camera_rays_to_image_points(
        rays, projection, NoExternalDistortion()
    )
    assert bool(jnp.all(valid))
    rays_back = image_points_to_camera_rays(
        points, projection, NoExternalDistortion()
    )
    expected = rays / jnp.linalg.norm(rays, axis=-1, keepdims=True)
    np.testing.assert_allclose(rays_back, expected, atol=2.0e-5)

    out_points, out_valid = camera_rays_to_image_points(
        jnp.array([[1.0, 0.0, 1.0]]), projection, NoExternalDistortion()
    )
    np.testing.assert_array_equal(out_points, [[0.0, 0.0]])
    np.testing.assert_array_equal(out_valid, [False])

    tiny_angle_projection = replace(projection, max_angle=0.01)
    over = jnp.array([[0.1, 0.0, 1.0]])
    at_limit = jnp.array([[jnp.sin(0.01), 0.0, jnp.cos(0.01)]])
    over_points, over_valid = camera_rays_to_image_points(
        over, tiny_angle_projection, NoExternalDistortion()
    )
    limit_points, limit_valid = camera_rays_to_image_points(
        at_limit, tiny_angle_projection, NoExternalDistortion()
    )
    np.testing.assert_allclose(over_points, limit_points, atol=1.0e-5)
    np.testing.assert_array_equal(over_valid, limit_valid)


@pytest.mark.parametrize("projection", [_pinhole(distorted=True), _ftheta(1), _fisheye()])
def test_camera_ops_are_jittable_and_differentiable(projection):
    rays = jnp.array([[0.02, 0.01, 1.0], [-0.03, 0.02, 1.0]])
    compiled = jax.jit(camera_rays_to_image_points)
    points, valid = compiled(rays, projection, NoExternalDistortion())
    assert points.shape == (2, 2)
    assert bool(jnp.all(valid))

    gradient = jax.grad(
        lambda value: jnp.sum(
            camera_rays_to_image_points(
                value, projection, NoExternalDistortion()
            )[0]
        )
    )(rays)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    inverse_gradient = jax.grad(
        lambda value: jnp.sum(
            image_points_to_camera_rays(
                value, projection, NoExternalDistortion()
            )
        )
    )(points)
    assert bool(jnp.all(jnp.isfinite(inverse_gradient)))


def test_windshield_coefficients_receive_gradients_in_the_active_slice():
    projection = _pinhole()
    distortion = _identity_windshield()
    rays = jnp.array([[0.02, 0.01, 1.0], [-0.03, 0.02, 1.0]])

    def loss(coefficients):
        current = replace(distortion, distortion_coeffs=coefficients)
        return jnp.sum(camera_rays_to_image_points(rays, projection, current)[0])

    gradient = jax.grad(loss)(distortion.distortion_coeffs)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.any(gradient[:21] != 0.0))
    np.testing.assert_array_equal(gradient[21:], jnp.zeros(21))


def test_generate_image_points_and_structured_functional_return():
    points = generate_image_points((2, 3))
    assert points.shape == (3, 2, 2)
    np.testing.assert_array_equal(points[0, 0], [0.5, 0.5])
    np.testing.assert_array_equal(points[2, 1], [1.5, 2.5])
    result = functional.camera_rays_to_image_points(
        jnp.array([[0.0, 0.0, 1.0]]), _pinhole(), NoExternalDistortion()
    )
    np.testing.assert_array_equal(result.image_points, [[50.0, 40.0]])
    np.testing.assert_array_equal(result.valid_flag, [True])
    compiled = jax.jit(functional.camera_rays_to_image_points)(
        jnp.array([[0.0, 0.0, 1.0]]), _pinhole(), NoExternalDistortion()
    )
    np.testing.assert_array_equal(compiled.image_points, result.image_points)


def test_none_external_distortion_is_rejected_explicitly():
    with pytest.raises(TypeError, match="NoExternalDistortion"):
        camera_rays_to_image_points(jnp.ones((1, 3)), _pinhole(), None)
