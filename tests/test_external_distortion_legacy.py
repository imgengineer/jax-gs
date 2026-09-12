import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_gs
from jax_gs.config import RasterizationConfig
from jax_gs.external_distortion import (
    coefficient_order,
    distort_camera_rays,
    eval_bivariate_polynomial,
)
from jax_gs.three_dgut import (
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    fully_fused_projection_with_ut,
    project_camera_points,
    rasterize_to_pixels_eval3d,
)


def _identity_parameters(order: int = 1):
    count = (order + 1) * (order + 2) // 2
    horizontal = jnp.zeros((count,), jnp.float32)
    vertical = jnp.zeros((count,), jnp.float32)
    if order >= 1:
        horizontal = horizontal.at[1].set(1.0)
        vertical = vertical.at[order + 1].set(1.0)
    return jax_gs.BivariateWindshieldModelParameters(
        horizontal,
        vertical,
        horizontal,
        vertical,
    )


def _shift_parameters(shift: float = 0.04):
    horizontal = jnp.asarray([shift, 1.0, 0.0], jnp.float32)
    vertical = jnp.asarray([0.0, 0.0, 1.0], jnp.float32)
    horizontal_inverse = jnp.asarray([-shift, 1.0, 0.0], jnp.float32)
    return jax_gs.BivariateWindshieldModelParameters(
        horizontal,
        vertical,
        horizontal_inverse,
        vertical,
    )


def _scene(*, leading_batch: bool = False):
    means = jnp.asarray([[0.0, 0.0, 2.0]], jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], jnp.float32)
    scales = jnp.asarray([[0.1, 0.1, 0.1]], jnp.float32)
    opacities = jnp.asarray([0.8], jnp.float32)
    colors = jnp.asarray([[1.0, 0.0, 0.0]], jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.asarray(
        [[[20.0, 0.0, 4.5], [0.0, 20.0, 4.5], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )
    if leading_batch:
        means = jnp.stack((means, means.at[0, 0].set(0.05)))
        quats = jnp.broadcast_to(quats, (2, 1, 4))
        scales = jnp.broadcast_to(scales, (2, 1, 3))
        opacities = jnp.broadcast_to(opacities, (2, 1))
        colors = jnp.broadcast_to(colors, (2, 1, 3))
        viewmats = jnp.broadcast_to(viewmats, (2, 1, 4, 4))
        Ks = jnp.broadcast_to(Ks, (2, 1, 3, 3))
    return means, quats, scales, opacities, colors, viewmats, Ks


def _config():
    return RasterizationConfig(
        backend="intersections",
        tile_size=4,
        max_gaussians_per_tile=2,
        max_intersections=16,
        tile_batch_size=1,
        ut_chunk_size=1,
    )


def test_root_external_distortion_api_and_mutable_compatibility_form():
    assert jax_gs.BivariateWindshieldModelParameters.MAX_ORDER == 5
    assert jax_gs.BivariateWindshieldModelParameters.MAX_COEFFS == 21
    assert jax_gs.ExternalDistortionReferencePolynomial.FORWARD == 1
    assert jax_gs.ExternalDistortionReferencePolynomial.BACKWARD == 2

    parameters = jax_gs.BivariateWindshieldModelParameters()
    identity = _identity_parameters()
    parameters.horizontal_poly = identity.horizontal_poly
    parameters.vertical_poly = identity.vertical_poly
    parameters.horizontal_poly_inverse = identity.horizontal_poly_inverse
    parameters.vertical_poly_inverse = identity.vertical_poly_inverse
    leaves = jax.tree.leaves(parameters)
    assert len(leaves) == 4

    rays = jnp.asarray([[0.1, 0.05, 1.0]], jnp.float32)
    compiled = jax.jit(distort_camera_rays)(rays, parameters)
    np.testing.assert_allclose(
        compiled,
        rays / jnp.linalg.norm(rays, axis=-1, keepdims=True),
        atol=1.0e-6,
    )


@pytest.mark.parametrize("order", range(6))
def test_bivariate_polynomial_orders_match_triangular_layout(order):
    count = (order + 1) * (order + 2) // 2
    assert coefficient_order(count) == order
    coefficients = jnp.arange(1, count + 1, dtype=jnp.float32)
    x = jnp.asarray([0.2, -0.4], jnp.float32)
    y = jnp.asarray([0.3, 0.1], jnp.float32)

    expected = np.zeros((2,), np.float32)
    index = 0
    for y_power in range(order + 1):
        for x_power in range(order - y_power + 1):
            expected += (
                np.asarray(coefficients[index])
                * np.asarray(x) ** x_power
                * np.asarray(y) ** y_power
            )
            index += 1
    np.testing.assert_allclose(
        eval_bivariate_polynomial(x, y, coefficients), expected, atol=1.0e-6
    )


def test_external_distortion_validates_complete_triangular_polynomials():
    with pytest.raises(ValueError, match="1, 3, 6"):
        jax_gs.BivariateWindshieldModelParameters(
            jnp.zeros((2,), jnp.float32),
            jnp.zeros((3,), jnp.float32),
            jnp.zeros((3,), jnp.float32),
            jnp.zeros((3,), jnp.float32),
        )
    with pytest.raises(ValueError, match="missing horizontal_poly"):
        distort_camera_rays(
            jnp.ones((1, 3), jnp.float32),
            jax_gs.BivariateWindshieldModelParameters(),
        )


@pytest.mark.parametrize("order", range(1, 6))
def test_identity_distortion_roundtrip_for_every_nonconstant_order(order):
    rays = jnp.asarray(
        [[0.1, 0.05, 1.0], [-0.05, 0.08, 1.0], [0.0, 0.0, 1.0]],
        jnp.float32,
    )
    parameters = _identity_parameters(order)
    distorted = distort_camera_rays(rays, parameters)
    recovered = distort_camera_rays(distorted, parameters, inverse=True)
    expected = rays / jnp.linalg.norm(rays, axis=-1, keepdims=True)
    np.testing.assert_allclose(distorted, expected, atol=1.0e-6)
    np.testing.assert_allclose(recovered, expected, atol=1.0e-6)


def test_external_distortion_math_is_jittable_and_differentiable():
    rays = jnp.asarray([[0.1, 0.05, 1.0], [-0.1, 0.02, 1.0]], jnp.float32)
    vertical = jnp.asarray([0.0, 0.0, 1.0], jnp.float32)

    def objective(horizontal):
        parameters = jax_gs.BivariateWindshieldModelParameters(
            horizontal,
            vertical,
            jnp.asarray([-horizontal[0], 1.0, 0.0]),
            vertical,
        )
        return distort_camera_rays(rays, parameters).sum()

    value, gradient = jax.jit(jax.value_and_grad(objective))(
        jnp.asarray([0.02, 1.0, 0.0], jnp.float32)
    )
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
    assert abs(float(gradient[0])) > 0.0


def test_root_camera_wrapper_projection_roundtrip_and_properties():
    camera = jax_gs.create_camera_model(
        "pinhole",
        width=640,
        height=480,
        principal_points=jnp.asarray([[320.0, 240.0]], jnp.float32),
        focal_lengths=jnp.asarray([[320.0, 320.0]], jnp.float32),
        external_distortion_coeffs=_shift_parameters(),
    )
    rays = jnp.asarray(
        [[[0.05, 0.03, 1.0], [0.0, 0.0, 1.0], [-0.05, 0.05, 1.0]]],
        jnp.float32,
    )
    image_points, projected_valid = jax.jit(
        lambda value: camera.camera_ray_to_image_point(value)
    )(rays)
    recovered, unprojected_valid = camera.image_point_to_camera_ray(image_points)
    expected = rays / jnp.linalg.norm(rays, axis=-1, keepdims=True)

    np.testing.assert_allclose(recovered, expected, atol=2.0e-6)
    np.testing.assert_array_equal(projected_valid, True)
    np.testing.assert_array_equal(unprojected_valid, True)
    np.testing.assert_allclose(camera.principal_points, [[320.0, 240.0]])
    np.testing.assert_allclose(camera.focal_lengths, [[320.0, 320.0]])
    assert camera.width == 640
    assert camera.height == 480


def test_root_camera_wrapper_shutter_world_methods_are_consistent():
    camera = jax_gs.create_camera_model(
        "pinhole",
        width=101,
        height=101,
        principal_points=jnp.asarray([[50.0, 50.0]], jnp.float32),
        focal_lengths=jnp.asarray([[100.0, 100.0]], jnp.float32),
        external_distortion_coeffs=_identity_parameters(),
        rs_type=jax_gs.RollingShutterType.ROLLING_TOP_TO_BOTTOM,
    )
    start = jnp.asarray([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], jnp.float32)
    end = start.at[0, 0].set(1.0)
    pixels = jnp.asarray([[[50.0, 50.0]]], jnp.float32)
    origins, directions, valid = camera.image_point_to_world_ray_shutter_pose(
        pixels, start, end
    )
    np.testing.assert_allclose(origins, [[[-0.5, 0.0, 0.0]]], atol=1.0e-6)
    np.testing.assert_allclose(directions, [[[0.0, 0.0, 1.0]]], atol=1.0e-6)
    np.testing.assert_array_equal(valid, True)
    np.testing.assert_allclose(camera.shutter_relative_frame_time(pixels), 0.5)

    world_points = jnp.asarray([[[0.0, 0.0, 2.0]]], jnp.float32)
    projected, projected_valid = camera.world_point_to_image_point_shutter_pose(
        world_points, start, end
    )
    np.testing.assert_allclose(projected, [[[75.0, 50.0]]], atol=1.0e-5)
    np.testing.assert_array_equal(projected_valid, True)


def test_orthographic_identity_external_distortion_is_a_noop():
    points = jnp.asarray([[[0.2, -0.3, 2.0], [-0.1, 0.4, 1.0]]], jnp.float32)
    K = jnp.asarray(
        [[[20.0, 0.0, 32.0], [0.0, 20.0, 32.0], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )
    plain = project_camera_points(points, K, 64, 64, camera_model="ortho")
    distorted = project_camera_points(
        points,
        K,
        64,
        64,
        camera_model="ortho",
        external_distortion_coeffs=_identity_parameters(),
    )
    np.testing.assert_allclose(distorted[0], plain[0], atol=1.0e-6)
    np.testing.assert_array_equal(distorted[1], plain[1])


def test_ut_projection_applies_external_distortion_with_jit_and_grad():
    means, quats, scales, opacities, _, viewmats, Ks = _scene()
    identity = fully_fused_projection_with_ut(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        8,
        8,
        external_distortion_coeffs=_identity_parameters(),
    )
    plain = fully_fused_projection_with_ut(
        means, quats, scales, opacities, viewmats, Ks, 8, 8
    )
    np.testing.assert_allclose(identity[1], plain[1], atol=2.0e-5)

    vertical = jnp.asarray([0.0, 0.0, 1.0], jnp.float32)

    def objective(horizontal):
        parameters = jax_gs.BivariateWindshieldModelParameters(
            horizontal,
            vertical,
            jnp.asarray([-horizontal[0], 1.0, 0.0]),
            vertical,
        )
        projected = fully_fused_projection_with_ut(
            means,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            8,
            8,
            external_distortion_coeffs=parameters,
        )
        return projected[1].sum()

    value, gradient = jax.jit(jax.value_and_grad(objective))(
        jnp.asarray([0.02, 1.0, 0.0], jnp.float32)
    )
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
    assert abs(float(gradient[0])) > 0.0


def test_ftheta_ut_accepts_order_five_root_windshield_parameters():
    means, quats, scales, opacities, _, viewmats, Ks = _scene()
    ftheta = FThetaCameraDistortionParameters(
        reference_poly=FThetaPolynomialType.ANGLE_TO_PIXELDIST,
        pixeldist_to_angle_poly=(0.0, 0.05, 0.0, 0.0, 0.0, 0.0),
        angle_to_pixeldist_poly=(0.0, 20.0, 0.0, 0.0, 0.0, 0.0),
        max_angle=math.pi / 2.0,
        linear_cde=(1.0, 0.0, 0.0),
    )
    plain = fully_fused_projection_with_ut(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        8,
        8,
        camera_model="ftheta",
        ftheta_coeffs=ftheta,
    )
    identity = jax.jit(
        lambda: fully_fused_projection_with_ut(
            means,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            8,
            8,
            camera_model="ftheta",
            ftheta_coeffs=ftheta,
            external_distortion_coeffs=_identity_parameters(5),
        )
    )()
    np.testing.assert_allclose(identity[1], plain[1], atol=5.0e-5)
    np.testing.assert_array_equal(identity[-1], plain[-1])


def test_low_level_eval3d_external_distortion_changes_scan_rays():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    offsets = jnp.zeros((1, 2, 2), jnp.int32)
    ids = jnp.asarray([0], jnp.int32)

    def render(parameters):
        return rasterize_to_pixels_eval3d(
            means,
            quats,
            scales,
            colors[None],
            opacities[None],
            viewmats,
            Ks,
            8,
            8,
            4,
            offsets,
            ids,
            external_distortion_coeffs=parameters,
            max_gaussians_per_tile=1,
        )

    plain = render(None)
    identity = jax.jit(lambda: render(_identity_parameters()))()
    shifted = render(_shift_parameters())
    np.testing.assert_allclose(identity[0], plain[0], atol=2.0e-6)
    np.testing.assert_allclose(identity[1], plain[1], atol=2.0e-6)
    assert not np.allclose(np.asarray(shifted[1]), np.asarray(plain[1]))


def test_high_level_external_distortion_contract_leading_batch_and_gradients():
    scene = _scene(leading_batch=True)
    means, quats, scales, opacities, colors, viewmats, Ks = scene
    with pytest.raises(ValueError, match="with_ut=True"):
        jax_gs.rasterization(
            *scene[:4],
            colors,
            viewmats,
            Ks,
            8,
            8,
            external_distortion_coeffs=_identity_parameters(),
            config=_config(),
        )
    with pytest.raises(ValueError, match="with_ut=True"):
        jax_gs.rasterization(
            *scene[:4],
            colors,
            viewmats,
            Ks,
            8,
            8,
            external_distortion_coeffs=(jax_gs.BivariateWindshieldModelParameters()),
            config=_config(),
        )

    vertical = jnp.asarray([0.0, 0.0, 1.0], jnp.float32)

    def objective(horizontal):
        parameters = jax_gs.BivariateWindshieldModelParameters(
            horizontal,
            vertical,
            jnp.asarray([-horizontal[0], 1.0, 0.0]),
            vertical,
        )
        rendered, alpha, _ = jax_gs.rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            8,
            8,
            with_ut=True,
            with_eval3d=True,
            external_distortion_coeffs=parameters,
            config=_config(),
        )
        return rendered.sum() + alpha.sum(), (rendered, alpha)

    (value, (rendered, alpha)), gradient = jax.jit(
        jax.value_and_grad(objective, has_aux=True)
    )(jnp.asarray([0.02, 1.0, 0.0], jnp.float32))
    assert rendered.shape == (2, 1, 8, 8, 3)
    assert alpha.shape == (2, 1, 8, 8, 1)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
    assert abs(float(gradient[0])) > 0.0


def test_external_distortion_rejects_lidar_pairing():
    parameters = jax_gs.RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=jnp.asarray([0.1, -0.1], jnp.float32),
        column_azimuths_rad=jnp.asarray([0.5, 0.0, -0.5], jnp.float32),
        row_azimuth_offsets_rad=jnp.zeros((2,), jnp.float32),
        spinning_frequency_hz=10.0,
        spinning_direction=jax_gs.SpinningDirection.CLOCKWISE,
    )
    lidar = jax_gs.RowOffsetStructuredSpinningLidarModelParametersExt(
        parameters,
        jax_gs.compute_lidar_angles_to_columns_map(parameters),
        jax_gs.compute_lidar_tiling(
            parameters,
            n_bins_elevation=1,
            max_pts_per_tile=4,
            resolution_elevation=10,
            densification_factor_azimuth=1,
        ),
    )
    with pytest.raises(ValueError, match="LiDAR"):
        project_camera_points(
            jnp.ones((1, 1, 3), jnp.float32),
            jnp.eye(3, dtype=jnp.float32)[None],
            1,
            1,
            camera_model="lidar",
            lidar_coeffs=lidar,
            external_distortion_coeffs=_identity_parameters(),
        )
