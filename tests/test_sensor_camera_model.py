import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_gs.sensors.functional.return_types import ImagePointsReturn
from jax_gs.sensors.kernels.cameras import (
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ReferencePolynomial,
    ShutterType,
    from_components,
)
from jax_gs.sensors.kernels.common import DynamicPose, Pose
from jax_gs.sensors.models import CameraModel
from jax_gs.sensors.models.common import compute_scaled_resolution


def _pinhole() -> OpenCVPinholeProjection:
    return OpenCVPinholeProjection(
        focal_length=jnp.array([100.0, 120.0]),
        principal_point=jnp.array([50.0, 40.0]),
        radial_coeffs=jnp.zeros(6),
        tangential_coeffs=jnp.zeros(2),
        thin_prism_coeffs=jnp.zeros(4),
        resolution=(100, 80),
    )


def _ftheta() -> FThetaProjection:
    return FThetaProjection(
        principal_point=jnp.array([50.0, 40.0]),
        fw_poly=jnp.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
        bw_poly=jnp.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        A=jnp.array([1.0, 0.0, 0.0, 1.0]),
        resolution=(100, 80),
        reference_polynomial=ReferencePolynomial.FORWARD,
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


def _static_pose() -> Pose:
    return Pose(jnp.zeros(3), jnp.array([1.0, 0.0, 0.0, 0.0]))


def _dynamic_pose() -> DynamicPose:
    return DynamicPose(
        start_pose=_static_pose(),
        end_pose=Pose(
            jnp.array([0.1, 0.0, 0.0]),
            jnp.array([1.0, 0.0, 0.0, 0.0]),
        ),
    )


def _model(
    projection: OpenCVPinholeProjection | FThetaProjection | OpenCVFisheyeProjection,
    *,
    shutter_type: ShutterType = ShutterType.GLOBAL,
) -> CameraModel:
    return CameraModel(
        projection,
        NoExternalDistortion(),
        (100, 80),
        shutter_type,
    )


@pytest.mark.parametrize(
    ("projection", "parameter_leaves"),
    [(_pinhole(), 5), (_ftheta(), 4), (_fisheye(), 4)],
)
def test_camera_model_registers_projection_arrays_as_nnx_params(
    projection, parameter_leaves
):
    model = _model(projection)
    assert isinstance(model, nnx.Module)
    assert type(model.projection) is type(projection)
    assert model.projection.resolution == projection.resolution
    assert model.resolution == (100, 80)
    assert model.shutter_type == ShutterType.GLOBAL
    assert len(jax.tree.leaves(nnx.state(model, nnx.Param))) == parameter_leaves


def test_camera_model_registers_windshield_coefficients_as_a_param():
    horizontal = jnp.array([0.0, 1.0, 0.0])
    vertical = jnp.array([0.0, 0.0, 1.0])
    distortion = from_components(
        horizontal,
        vertical,
        horizontal,
        vertical,
        ReferencePolynomial.FORWARD,
    )
    model = CameraModel(_pinhole(), distortion, (100, 80), ShutterType.GLOBAL)
    assert type(model.external_distortion) is type(distortion)
    np.testing.assert_array_equal(
        model.external_distortion.distortion_coeffs, distortion.distortion_coeffs
    )
    assert len(jax.tree.leaves(nnx.state(model, nnx.Param))) == 6


def test_camera_projection_jacobian_roundtrip_and_pixel_conversions():
    model = _model(_pinhole())
    rays = jnp.array([[0.1, 0.2, 1.0]])
    result = model.camera_rays_to_image_points(rays, return_jacobians=True)
    assert isinstance(result, ImagePointsReturn)
    np.testing.assert_allclose(result.image_points, [[60.0, 64.0]])
    np.testing.assert_allclose(
        result.jacobians,
        [[[100.0, 0.0, -10.0], [0.0, 120.0, -24.0]]],
        atol=1.0e-4,
    )
    recovered = model.image_points_to_camera_rays(result.image_points)
    np.testing.assert_allclose(
        recovered, rays / jnp.linalg.norm(rays, axis=-1, keepdims=True), atol=1.0e-6
    )
    pixels = jnp.array([[0, 1]], dtype=jnp.int32)
    np.testing.assert_array_equal(model.pixels_to_image_points(pixels), [[0.5, 1.5]])
    np.testing.assert_array_equal(
        model.image_points_to_pixels(jnp.array([[0.9, 1.9]])), pixels
    )


def test_world_projection_preserves_upstream_eager_filtering_and_metadata():
    model = _model(_pinhole())
    points = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    result = model.world_points_to_pixels_static_pose(
        points,
        _static_pose(),
        timestamp_us=11,
        return_T_sensor_world=True,
        return_valid_flag=True,
        return_valid_indices=True,
        return_timestamps=True,
    )
    np.testing.assert_array_equal(result.pixels, [[50, 40]])
    np.testing.assert_array_equal(result.valid_flag, [True, False])
    np.testing.assert_array_equal(result.valid_indices, [0])
    np.testing.assert_array_equal(result.timestamps_us, [11])
    assert result.T_sensor_world.shape == (1, 4, 4)

    all_result = model.world_points_to_pixels_static_pose(
        points,
        _static_pose(),
        return_valid_flag=True,
        return_all_projections=True,
    )
    assert all_result.pixels.shape == (2, 2)
    np.testing.assert_array_equal(all_result.valid_flag, [True, False])


def test_fixed_shape_model_projection_is_nnx_jittable():
    model = _model(_pinhole())
    pose = _static_pose()

    @nnx.jit
    def project(current_model, points):
        return current_model.world_points_to_image_points_static_pose(
            points,
            pose,
            return_valid_flag=True,
            return_all_projections=True,
        )

    result = project(model, jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
    assert result.image_points.shape == (2, 2)
    np.testing.assert_array_equal(result.valid_flag, [True, False])


def test_mean_and_shutter_pose_adapters_return_rays_and_timestamps():
    model = _model(_pinhole(), shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM)
    points = jnp.array([[50.5, 40.5]])
    mean = model.image_points_to_world_rays_mean_pose(
        points,
        _dynamic_pose(),
        start_timestamp_us=0,
        end_timestamp_us=10,
        return_T_sensor_world=True,
        return_timestamps=True,
    )
    np.testing.assert_allclose(mean.world_rays[:, :3], [[0.05, 0.0, 0.0]])
    np.testing.assert_array_equal(mean.timestamps_us, [5])
    assert mean.T_sensor_world.shape == (1, 4, 4)

    shutter = model.image_points_to_world_rays_shutter_pose(
        points,
        _dynamic_pose(),
        start_timestamp_us=0,
        end_timestamp_us=100,
        return_timestamps=True,
    )
    np.testing.assert_allclose(
        shutter.world_rays[:, :3], [[0.1 * 40.0 / 79.0, 0.0, 0.0]]
    )
    np.testing.assert_array_equal(shutter.timestamps_us, [int(100 * 40.0 / 79.0)])


@pytest.mark.parametrize("projection", [_pinhole(), _ftheta(), _fisheye()])
def test_transform_returns_new_model_and_preserves_nnx_gradient(projection):
    model = _model(projection)
    transformed = model.transform((0.5, 0.25), (1.0, 2.0))
    assert transformed is not model
    assert transformed.resolution == (50, 20)
    assert transformed.shutter_type == model.shutter_type
    assert type(transformed.external_distortion) is type(model.external_distortion)

    if isinstance(projection, FThetaProjection):
        gradients = nnx.grad(
            lambda value: value.transform(0.5).projection.fw_poly.sum()
        )(model)
        np.testing.assert_allclose(gradients._fw_poly[...], jnp.full((6,), 0.5))
    else:
        gradients = nnx.grad(
            lambda value: value.transform(0.5).projection.focal_length.sum()
        )(model)
        np.testing.assert_allclose(gradients._focal_length[...], [0.5, 0.5])


def test_scaled_resolution_matches_current_main_rules():
    assert compute_scaled_resolution((100, 80), 0.5) == (50, 40)
    assert compute_scaled_resolution((100, 80), (0.5, 0.25)) == (50, 20)
    assert compute_scaled_resolution((100, 80), 0.5, (12, 13)) == (12, 13)


def test_camera_model_rejects_missing_concrete_components():
    with pytest.raises(TypeError, match="projection"):
        CameraModel(None, NoExternalDistortion(), (100, 80), ShutterType.GLOBAL)
    with pytest.raises(TypeError, match="NoExternalDistortion"):
        CameraModel(_pinhole(), None, (100, 80), ShutterType.GLOBAL)


def test_pixel_conversions_are_instance_methods_like_upstream():
    # Upstream defines these as instance methods, so the explicit unbound form
    # CameraModel.method(model, values) is a valid call. As staticmethods that
    # form silently consumed the model as the coordinates.
    model = CameraModel(
        _pinhole(), NoExternalDistortion(), (100, 80), ShutterType.GLOBAL
    )
    pixels = jnp.asarray([[3, 4], [10, 20]], dtype=jnp.int32)
    bound = model.pixels_to_image_points(pixels)
    unbound = CameraModel.pixels_to_image_points(model, pixels)
    np.testing.assert_array_equal(np.asarray(bound), np.asarray(unbound))
    np.testing.assert_allclose(np.asarray(bound), np.asarray(pixels) + 0.5)

    image_points = jnp.asarray([[3.7, 4.2]], dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(model.image_points_to_pixels(image_points)),
        np.asarray(CameraModel.image_points_to_pixels(model, image_points)),
    )
