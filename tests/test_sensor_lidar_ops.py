import math
from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.sensors import functional
from jax_gs.sensors.functional.return_types import (
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)
from jax_gs.sensors.kernels.common import DynamicPose, Pose
from jax_gs.sensors.kernels.lidars import (
    REGISTERED_LIDAR_PROJECTION_NAMES,
    REGISTERED_LIDAR_PROJECTIONS,
    RowOffsetStructuredSpinningLidarProjection,
    SpinningDirection,
    elements_to_sensor_angles,
    generate_spinning_lidar_rays,
    inverse_project_spinning_lidar,
    script_class_name,
    sensor_angles_to_sensor_rays,
    sensor_rays_to_sensor_angles,
    validate_lidar_projection,
)


def _projection(
    *,
    has_offsets: bool = True,
    spinning_direction: SpinningDirection = SpinningDirection.CLOCKWISE,
) -> RowOffsetStructuredSpinningLidarProjection:
    return RowOffsetStructuredSpinningLidarProjection(
        row_elevations_rad=jnp.array([0.15, 0.0, -0.15]),
        column_azimuths_rad=jnp.array(
            [math.pi - 0.01, 1.5, 0.0, -1.5, -math.pi + 0.01]
        ),
        row_azimuth_offsets_rad=jnp.array([0.05, -0.02, 0.1])
        if has_offsets
        else jnp.zeros((0,), dtype=jnp.float32),
        fov_vert_start_rad=0.2,
        fov_vert_span_rad=0.4,
        fov_horiz_start_rad=math.pi,
        fov_horiz_span_rad=2.0 * math.pi,
        spinning_direction=spinning_direction,
        has_row_offsets=has_offsets,
    )


def _dynamic_pose() -> DynamicPose:
    identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    return DynamicPose(
        start_pose=Pose(jnp.zeros(3), identity),
        end_pose=Pose(jnp.array([0.1, 0.02, -0.01]), identity),
    )


def _element_grid(projection) -> jax.Array:
    rows, columns = jnp.meshgrid(
        jnp.arange(projection.row_elevations_rad.shape[0]),
        jnp.arange(projection.column_azimuths_rad.shape[0]),
        indexing="ij",
    )
    return jnp.stack((rows, columns), axis=-1).reshape((-1, 2)).astype(jnp.int32)


def _wrapped_difference(left, right):
    difference = left - right
    return jnp.arctan2(jnp.sin(difference), jnp.cos(difference))


def test_projection_type_is_frozen_registered_and_validated():
    projection = _projection()
    assert SpinningDirection.CLOCKWISE == 0
    assert SpinningDirection.COUNTERCLOCKWISE == 1
    assert REGISTERED_LIDAR_PROJECTIONS == (RowOffsetStructuredSpinningLidarProjection,)
    assert REGISTERED_LIDAR_PROJECTION_NAMES == (
        "RowOffsetStructuredSpinningLidarProjection",
    )
    assert script_class_name(projection) == REGISTERED_LIDAR_PROJECTION_NAMES[0]
    validate_lidar_projection(projection)
    with pytest.raises(FrozenInstanceError):
        projection.has_row_offsets = False

    bad_values = replace(
        projection,
        row_elevations_rad=jnp.array([0.1, jnp.nan, -0.1]),
    )
    with pytest.raises(ValueError, match=r"row_elevations_rad\[1\]"):
        validate_lidar_projection(bad_values)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"row_elevations_rad": jnp.zeros((1, 1))}, ValueError),
        ({"column_azimuths_rad": jnp.array([0, 1])}, TypeError),
        ({"row_elevations_rad": jnp.zeros((0,))}, ValueError),
        ({"row_azimuth_offsets_rad": jnp.zeros((2,))}, ValueError),
        ({"spinning_direction": 2}, ValueError),
        ({"fov_vert_span_rad": math.inf}, ValueError),
    ],
)
def test_projection_rejects_invalid_structure(changes, error):
    with pytest.raises(error):
        replace(_projection(), **changes)


def test_ray_angle_conversions_match_closed_form_and_round_trip():
    rays = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    )
    angles = sensor_rays_to_sensor_angles(rays)
    np.testing.assert_allclose(
        angles,
        [[0.0, 0.0], [0.0, math.pi / 2.0], [math.pi / 4.0, 0.0], [0.0, 0.0]],
        atol=1.0e-6,
    )
    unit_rays = sensor_angles_to_sensor_rays(angles[:3])
    expected = rays[:3] / jnp.linalg.norm(rays[:3], axis=-1, keepdims=True)
    np.testing.assert_allclose(unit_rays, expected, atol=1.0e-6)

    structured_angles = functional.sensor_rays_to_sensor_angles(rays)
    structured_rays = functional.sensor_angles_to_sensor_rays(angles)
    assert isinstance(structured_angles, SensorAnglesReturn)
    assert isinstance(structured_rays, SensorRayReturn)
    assert structured_angles.valid_flag is None
    assert structured_rays.valid_flag is None


def test_ray_angle_conversions_are_jittable_and_differentiable():
    rays = jnp.array([[1.0, 0.2, 0.1], [0.8, -0.3, -0.2]])
    compiled = jax.jit(sensor_rays_to_sensor_angles)(rays)
    recovered = jax.jit(sensor_angles_to_sensor_rays)(compiled)
    expected = rays / jnp.linalg.norm(rays, axis=-1, keepdims=True)
    np.testing.assert_allclose(recovered, expected, atol=1.0e-6)

    ray_gradient = jax.grad(lambda value: sensor_rays_to_sensor_angles(value).sum())(
        rays
    )
    angle_gradient = jax.grad(lambda value: sensor_angles_to_sensor_rays(value).sum())(
        compiled
    )
    assert bool(jnp.all(jnp.isfinite(ray_gradient)))
    assert bool(jnp.all(jnp.isfinite(angle_gradient)))


def test_elements_lookup_wraps_offsets_zeroes_invalid_and_accumulates_gradients():
    projection = _projection()
    elements = jnp.array([[0, 0], [1, 2], [2, 4], [-1, 0], [0, 5]])
    angles, valid = elements_to_sensor_angles(elements, projection)
    expected_first_azimuth = -math.pi + 0.04
    np.testing.assert_allclose(
        angles[:3],
        [
            [0.15, expected_first_azimuth],
            [0.0, -0.02],
            [-0.15, -math.pi + 0.11],
        ],
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(valid, [True, True, True, False, False])
    np.testing.assert_array_equal(angles[3:], jnp.zeros((2, 2)))

    shared_elements = jnp.array([[0, 0], [0, 1], [2, 1]], dtype=jnp.int32)

    def loss(row_elevations, column_azimuths, row_offsets):
        current = replace(
            projection,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=row_offsets,
        )
        return elements_to_sensor_angles(shared_elements, current)[0].sum()

    gradients = jax.grad(loss, argnums=(0, 1, 2))(
        projection.row_elevations_rad,
        projection.column_azimuths_rad,
        projection.row_azimuth_offsets_rad,
    )
    np.testing.assert_allclose(gradients[0], [2.0, 0.0, 1.0])
    np.testing.assert_allclose(gradients[1], [1.0, 2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(gradients[2], [2.0, 0.0, 1.0])


def test_generate_matches_explicit_grid_and_interpolates_pose_and_timestamps():
    projection = _projection()
    pose = _dynamic_pose()
    explicit = _element_grid(projection)
    generated = functional.generate_spinning_lidar_rays(
        projection,
        None,
        pose,
        start_timestamp_us=100,
        end_timestamp_us=500,
        return_T_sensor_world=True,
        return_timestamps=True,
    )
    selected = functional.generate_spinning_lidar_rays(projection, explicit, pose)
    assert isinstance(generated, WorldRaysReturn)
    np.testing.assert_allclose(generated.world_rays, selected.world_rays, atol=1.0e-6)
    assert generated.world_rays.shape == (
        projection.row_elevations_rad.size * projection.column_azimuths_rad.size,
        6,
    )
    assert generated.T_sensor_world.shape == (explicit.shape[0], 4, 4)
    expected_timestamps = jnp.tile(jnp.array([100, 200, 300, 400, 500]), 3)
    np.testing.assert_array_equal(generated.timestamps_us, expected_timestamps)

    alpha = explicit[:, 1].astype(jnp.float32) / 4.0
    expected_origins = alpha[:, None] * pose.end_pose.translation
    np.testing.assert_allclose(generated.world_rays[:, :3], expected_origins)
    np.testing.assert_allclose(generated.T_sensor_world[:, :3, 3], expected_origins)
    np.testing.assert_allclose(
        jnp.linalg.norm(generated.world_rays[:, 3:], axis=-1), 1.0, atol=1.0e-6
    )


def test_generate_invalid_elements_use_zero_ray_identity_pose_and_start_time():
    result = functional.generate_spinning_lidar_rays(
        _projection(),
        jnp.array([[1, 2], [-1, 0], [0, 5]], dtype=jnp.int32),
        _dynamic_pose(),
        start_timestamp_us=100,
        end_timestamp_us=500,
        return_T_sensor_world=True,
        return_timestamps=True,
    )
    np.testing.assert_array_equal(result.world_rays[1:], jnp.zeros((2, 6)))
    np.testing.assert_allclose(
        result.T_sensor_world[1:], jnp.broadcast_to(jnp.eye(4), (2, 4, 4))
    )
    np.testing.assert_array_equal(result.timestamps_us, [300, 100, 100])


def test_generate_is_jittable_and_routes_gradients_to_pose_and_tables():
    projection = _projection()
    pose = _dynamic_pose()
    elements = jnp.array([[0, 1], [1, 2], [2, 3]], dtype=jnp.int32)
    world_rays = jax.jit(
        lambda value: generate_spinning_lidar_rays(projection, value, pose)[0]
    )(elements)
    assert world_rays.shape == (3, 6)

    translation_gradient = jax.grad(
        lambda end_translation: generate_spinning_lidar_rays(
            projection,
            elements,
            DynamicPose(
                pose.start_pose,
                Pose(end_translation, pose.end_pose.rotation),
            ),
        )[0].sum()
    )(pose.end_pose.translation)
    projection_gradient = jax.grad(
        lambda value: generate_spinning_lidar_rays(value, elements, pose)[0].sum()
    )(projection)
    assert bool(jnp.all(jnp.isfinite(translation_gradient)))
    for leaf in jax.tree.leaves(projection_gradient):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_inverse_projection_round_trips_dynamic_generated_points_under_jit():
    projection = _projection()
    pose = _dynamic_pose()
    elements = _element_grid(projection)
    world_rays = generate_spinning_lidar_rays(projection, elements, pose)[0]
    world_points = world_rays[:, :3] + 10.0 * world_rays[:, 3:]
    expected_angles = elements_to_sensor_angles(elements, projection)[0]

    angles, valid = jax.jit(
        lambda points: inverse_project_spinning_lidar(
            projection, points, pose, max_iterations=10
        )[:2]
    )(world_points)
    np.testing.assert_array_equal(valid, jnp.ones(valid.shape, dtype=jnp.bool_))
    np.testing.assert_allclose(angles[:, 0], expected_angles[:, 0], atol=2.0e-6)
    np.testing.assert_allclose(
        _wrapped_difference(angles[:, 1], expected_angles[:, 1]),
        jnp.zeros(angles.shape[0]),
        atol=2.0e-6,
    )


def test_inverse_projection_marks_out_of_fov_and_has_finite_geometry_gradient():
    projection = _projection()
    pose = DynamicPose.from_static_pose(
        Pose(jnp.zeros(3), jnp.array([1.0, 0.0, 0.0, 0.0]))
    )
    points = jnp.array([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    angles, valid, *_ = inverse_project_spinning_lidar(projection, points, pose)
    np.testing.assert_array_equal(valid, [True, False])
    np.testing.assert_array_equal(angles[1], [0.0, 0.0])
    gradient = jax.grad(
        lambda value: inverse_project_spinning_lidar(projection, value, pose)[0].sum()
    )(jnp.array([[5.0, 0.2, 0.1]]))
    assert bool(jnp.all(jnp.isfinite(gradient)))


@pytest.mark.parametrize("max_iterations", [0, 33])
def test_inverse_projection_rejects_invalid_iteration_count(max_iterations):
    with pytest.raises(ValueError, match="max_iterations"):
        inverse_project_spinning_lidar(
            _projection(),
            jnp.array([[5.0, 0.0, 0.0]]),
            _dynamic_pose(),
            max_iterations=max_iterations,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stop_mean_relative_time_error": -1.0},
        {"stop_delta_mean_relative_time_error": math.inf},
        {"initial_relative_time": -0.1},
        {"initial_relative_time": 1.1},
    ],
)
def test_inverse_projection_rejects_invalid_solver_scalars(kwargs):
    with pytest.raises(ValueError):
        inverse_project_spinning_lidar(
            _projection(),
            jnp.array([[5.0, 0.0, 0.0]]),
            _dynamic_pose(),
            **kwargs,
        )


def test_timestamp_bounds_must_be_paired():
    with pytest.raises(ValueError, match="provided together"):
        generate_spinning_lidar_rays(
            _projection(),
            jnp.array([[0, 0]]),
            _dynamic_pose(),
            start_timestamp_us=1,
        )
    with pytest.raises(ValueError, match="provided together"):
        inverse_project_spinning_lidar(
            _projection(),
            jnp.array([[5.0, 0.0, 0.0]]),
            _dynamic_pose(),
            end_timestamp_us=2,
        )


def test_functional_inverse_gates_metadata_and_uses_static_padded_indices():
    projection = _projection()
    pose = DynamicPose.from_static_pose(
        Pose(jnp.zeros(3), jnp.array([1.0, 0.0, 0.0, 0.0]))
    )
    points = jnp.array([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    bare = functional.inverse_project_spinning_lidar(projection, points, pose)
    assert isinstance(bare, WorldPointsToSensorAnglesReturn)
    assert bare.valid_flag is None
    assert bare.valid_indices is None
    assert bare.T_sensor_world is None
    assert bare.timestamps_us is None

    full = functional.inverse_project_spinning_lidar(
        projection,
        points,
        pose,
        start_timestamp_us=10,
        end_timestamp_us=20,
        return_T_sensor_world=True,
        return_valid_flag=True,
        return_valid_indices=True,
        return_timestamps=True,
    )
    np.testing.assert_array_equal(full.valid_flag, [True, False])
    np.testing.assert_array_equal(full.valid_indices, [0, -1])
    assert full.T_sensor_world.shape == (2, 4, 4)
    assert full.timestamps_us.shape == (2,)
    expected_timestamp_dtype = jnp.int64 if jax.config.x64_enabled else jnp.int32
    assert full.timestamps_us.dtype == expected_timestamp_dtype
