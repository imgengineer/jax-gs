import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.sensors import functional
from jax_gs.sensors.kernels.cameras import (
    NoExternalDistortion,
    OpenCVPinholeProjection,
    ShutterType,
    image_points_to_world_rays_shutter_pose,
    image_points_to_world_rays_static_pose,
    pixel_grid_to_world_rays_shutter_pose,
    project_world_points_mean_pose,
    project_world_points_shutter_pose,
    relative_frame_times,
)
from jax_gs.sensors.kernels.common import (
    DynamicPose,
    Pose,
    interpolate_dynamic_pose,
    quaternion_slerp_wxyz,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def _projection() -> OpenCVPinholeProjection:
    return OpenCVPinholeProjection(
        focal_length=jnp.array([100.0, 120.0]),
        principal_point=jnp.array([50.0, 40.0]),
        radial_coeffs=jnp.zeros(6),
        tangential_coeffs=jnp.zeros(2),
        thin_prism_coeffs=jnp.zeros(4),
        resolution=(100, 80),
    )


def _static_pose() -> Pose:
    return Pose(jnp.zeros(3), jnp.array([1.0, 0.0, 0.0, 0.0]))


def _dynamic_pose() -> DynamicPose:
    return DynamicPose(
        start_pose=_static_pose(),
        end_pose=Pose(jnp.array([0.1, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0, 0.0])),
    )


def test_quaternion_layout_helpers_accept_current_main_keyword_name():
    wxyz = jnp.asarray([[1.0, 2.0, 3.0, 4.0]])
    xyzw = wxyz_to_xyzw(quat=wxyz)
    np.testing.assert_array_equal(xyzw, [[2.0, 3.0, 4.0, 1.0]])
    np.testing.assert_array_equal(xyzw_to_wxyz(quat=xyzw), wxyz)


def test_pose_types_are_pytrees_and_equal_quaternion_slerp_has_finite_gradients():
    dynamic_pose = DynamicPose.from_static_pose(_static_pose())
    leaves = jax.tree_util.tree_leaves(dynamic_pose)
    assert len(leaves) == 4
    assert dynamic_pose.start_pose is not dynamic_pose.end_pose
    trajectory = dynamic_pose.to_trajectory()
    assert trajectory.control_count == 2
    np.testing.assert_array_equal(trajectory.control_times, [0.0, 1.0])

    start = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    end = start
    output = quaternion_slerp_wxyz(start, end, jnp.array([0.5]))
    np.testing.assert_allclose(output, start)
    gradient = jax.grad(
        lambda quaternion: jnp.sum(
            quaternion_slerp_wxyz(quaternion, end, jnp.array([0.5]))
        )
    )(start)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_pose_interpolation_uses_lerp_and_wxyz_slerp():
    half_angle = math.pi / 4.0
    start_rotation = jnp.array([1.0, 0.0, 0.0, 0.0])
    end_rotation = jnp.array([math.cos(half_angle), 0.0, math.sin(half_angle), 0.0])
    translation, rotation = interpolate_dynamic_pose(
        jnp.zeros(3),
        start_rotation,
        jnp.array([2.0, 4.0, 6.0]),
        end_rotation,
        jnp.array([0.5]),
    )
    np.testing.assert_allclose(translation, [[1.0, 2.0, 3.0]])
    expected = jnp.array([[math.cos(math.pi / 8.0), 0.0, math.sin(math.pi / 8.0), 0.0]])
    np.testing.assert_allclose(rotation, expected, atol=1.0e-6)


@pytest.mark.parametrize(
    "shutter,expected",
    [
        (ShutterType.ROLLING_TOP_TO_BOTTOM, [0.0, 40.0 / 79.0, 1.0]),
        (ShutterType.ROLLING_BOTTOM_TO_TOP, [1.0, 39.0 / 79.0, 0.0]),
        (ShutterType.ROLLING_LEFT_TO_RIGHT, [0.0, 50.0 / 99.0, 1.0]),
        (ShutterType.ROLLING_RIGHT_TO_LEFT, [1.0, 49.0 / 99.0, 0.0]),
        (ShutterType.GLOBAL, [0.0, 0.0, 0.0]),
    ],
)
def test_relative_frame_times_match_current_main_scan_boundaries(shutter, expected):
    points = jnp.array([[0.5, 0.5], [50.5, 40.5], [99.5, 79.5]])
    times = relative_frame_times(points, (100, 80), shutter)
    np.testing.assert_allclose(times, expected, atol=1.0e-7)


def test_mean_pose_projection_returns_midpoint_pose_timestamp_and_values():
    result = project_world_points_mean_pose(
        jnp.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
        _projection(),
        NoExternalDistortion(),
        _dynamic_pose(),
        (100, 80),
        start_timestamp_us=0,
        end_timestamp_us=10,
        return_valid_flags=True,
        return_timestamps=True,
        return_poses=True,
    )
    image_points, valid, timestamps, translations, rotations = result
    np.testing.assert_allclose(image_points, [[45.0, 40.0], [55.0, 40.0]])
    np.testing.assert_array_equal(valid, [True, True])
    np.testing.assert_array_equal(timestamps, [5, 5])
    np.testing.assert_allclose(translations, [[0.05, 0.0, 0.0]] * 2)
    np.testing.assert_allclose(rotations, [[1.0, 0.0, 0.0, 0.0]] * 2)


def test_static_world_ray_uses_sensor_to_world_pose():
    half_angle = math.pi / 4.0
    pose = Pose(
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([math.cos(half_angle), 0.0, math.sin(half_angle), 0.0]),
    )
    world_rays, timestamps, pose_t, pose_r = image_points_to_world_rays_static_pose(
        jnp.array([[50.0, 40.0]]),
        _projection(),
        NoExternalDistortion(),
        pose,
        timestamp_us=7,
        return_timestamps=True,
        return_poses=True,
    )
    np.testing.assert_allclose(world_rays[:, :3], [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(world_rays[:, 3:], [[1.0, 0.0, 0.0]], atol=1.0e-6)
    np.testing.assert_array_equal(timestamps, [7])
    np.testing.assert_allclose(pose_t, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(pose_r, pose.rotation[None])


def test_shutter_world_ray_uses_pixel_time_and_stop_gradient_timing():
    points = jnp.array([[50.5, 40.5]])
    world_rays, timestamps, pose_t, _ = image_points_to_world_rays_shutter_pose(
        points,
        _projection(),
        NoExternalDistortion(),
        (100, 80),
        ShutterType.ROLLING_TOP_TO_BOTTOM,
        _dynamic_pose(),
        start_timestamp_us=0,
        end_timestamp_us=100,
        return_timestamps=True,
        return_poses=True,
    )
    relative_time = 40.0 / 79.0
    np.testing.assert_allclose(world_rays[:, :3], [[0.1 * relative_time, 0.0, 0.0]])
    np.testing.assert_allclose(pose_t, world_rays[:, :3])
    np.testing.assert_array_equal(timestamps, [int(relative_time * 100)])

    origin_gradient = jax.jacrev(
        lambda image: image_points_to_world_rays_shutter_pose(
            image,
            _projection(),
            NoExternalDistortion(),
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            _dynamic_pose(),
        )[0][:, :3]
    )(points)
    np.testing.assert_array_equal(origin_gradient, jnp.zeros_like(origin_gradient))


def test_shutter_projection_converges_to_scanline_pose_and_is_jittable():
    world_points = jnp.array([[0.0, 0.0, 1.0]])

    def project(points):
        return project_world_points_shutter_pose(
            points,
            _projection(),
            NoExternalDistortion(),
            (100, 80),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            _dynamic_pose(),
            start_timestamp_us=0,
            end_timestamp_us=100,
            return_valid_flags=True,
            return_timestamps=True,
            return_poses=True,
        )

    image_points, valid, timestamps, pose_t, _ = jax.jit(project)(world_points)
    relative_time = 40.0 / 79.0
    np.testing.assert_allclose(
        image_points, [[50.0 - 10.0 * relative_time, 40.0]], atol=1.0e-5
    )
    np.testing.assert_array_equal(valid, [True])
    np.testing.assert_array_equal(timestamps, [int(relative_time * 100)])
    np.testing.assert_allclose(pose_t, [[0.1 * relative_time, 0.0, 0.0]])
    gradient = jax.grad(lambda points: jnp.sum(project(points)[0]))(world_points)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_ftheta_shutter_projection_can_converge_inward_from_out_of_frame_pose():
    from jax_gs.sensors.kernels.cameras import FThetaProjection

    projection = FThetaProjection(
        principal_point=jnp.array([50.0, 40.0]),
        fw_poly=jnp.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
        bw_poly=jnp.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        A=jnp.array([1.0, 0.0, 0.0, 1.0]),
        resolution=(100, 80),
        reference_polynomial=0,
        fw_poly_degree=1,
        bw_poly_degree=1,
        newton_iterations=10,
        max_angle=1.4,
        min_2d_norm=1.0e-6,
    )
    dynamic_pose = DynamicPose(
        start_pose=_static_pose(),
        end_pose=Pose(
            jnp.array([2.0, 0.0, 0.0]),
            jnp.array([1.0, 0.0, 0.0, 0.0]),
        ),
    )
    image_points, valid, timestamps, pose_t, _ = project_world_points_shutter_pose(
        jnp.array([[0.4, -0.4, 1.0]]),
        projection,
        NoExternalDistortion(),
        (100, 80),
        ShutterType.ROLLING_TOP_TO_BOTTOM,
        dynamic_pose,
        start_timestamp_us=0,
        end_timestamp_us=100,
        return_valid_flags=True,
        return_timestamps=True,
        return_poses=True,
    )
    assert bool(valid[0])
    assert bool(
        jnp.all((image_points[0] >= 0.0) & (image_points[0] < jnp.array([100.0, 80.0])))
    )
    assert int(timestamps[0]) < 50
    assert float(pose_t[0, 0]) < 1.0


def test_pixel_grid_and_functional_metadata_use_static_shapes():
    rays, timestamps, pose_t, pose_r = pixel_grid_to_world_rays_shutter_pose(
        _projection(),
        NoExternalDistortion(),
        (2, 3),
        ShutterType.GLOBAL,
        _dynamic_pose(),
        return_timestamps=True,
        return_poses=True,
    )
    assert rays.shape == (6, 6)
    assert timestamps.shape == (6,)
    assert pose_t.shape == (6, 3)
    assert pose_r.shape == (6, 4)

    result = functional.project_world_points_mean_pose(
        jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        _projection(),
        NoExternalDistortion(),
        (100, 80),
        DynamicPose.from_static_pose(_static_pose()),
        return_T_sensor_world=True,
        return_valid_flag=True,
        return_valid_indices=True,
        return_timestamps=True,
    )
    assert result.T_sensor_world.shape == (2, 4, 4)
    np.testing.assert_array_equal(result.valid_flag, [True, False])
    np.testing.assert_array_equal(result.valid_indices, [0, -1])
    assert result.timestamps_us.shape == (2,)


def test_timestamp_bounds_and_unknown_shutter_are_rejected():
    with pytest.raises(ValueError, match="provided together"):
        project_world_points_mean_pose(
            jnp.ones((1, 3)),
            _projection(),
            NoExternalDistortion(),
            _dynamic_pose(),
            (100, 80),
            start_timestamp_us=1,
        )
    with pytest.raises(ValueError, match="Unsupported ShutterType"):
        relative_frame_times(jnp.ones((1, 2)), (100, 80), 99)
