import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_gs.sensors import functional
from jax_gs.sensors.kernels.common import DynamicPose, Pose
from jax_gs.sensors.models import (
    Frame,
    LidarFrame,
    LidarFrameSet,
    LidarModel,
    RowOffsetStructuredSpinningLidarProjection,
    SpinningDirection,
)


def _projection() -> RowOffsetStructuredSpinningLidarProjection:
    return RowOffsetStructuredSpinningLidarProjection(
        row_elevations_rad=jnp.array([0.15, 0.0, -0.15]),
        column_azimuths_rad=jnp.array(
            [math.pi - 0.01, 1.5, 0.0, -1.5, -math.pi + 0.01]
        ),
        row_azimuth_offsets_rad=jnp.array([0.05, -0.02, 0.1]),
        fov_vert_start_rad=0.2,
        fov_vert_span_rad=0.4,
        fov_horiz_start_rad=math.pi,
        fov_horiz_span_rad=2.0 * math.pi,
        spinning_direction=SpinningDirection.CLOCKWISE,
        has_row_offsets=True,
    )


def _static_pose() -> Pose:
    return Pose(jnp.zeros(3), jnp.array([1.0, 0.0, 0.0, 0.0]))


def _dynamic_pose() -> DynamicPose:
    return DynamicPose(
        start_pose=_static_pose(),
        end_pose=Pose(
            jnp.array([0.1, 0.02, -0.01]),
            jnp.array([1.0, 0.0, 0.0, 0.0]),
        ),
    )


def _model() -> LidarModel:
    return LidarModel(_projection())


def test_lidar_model_registers_angle_tables_as_nnx_params_and_reexports_types():
    model = _model()
    assert isinstance(model, nnx.Module)
    assert isinstance(model.projection, RowOffsetStructuredSpinningLidarProjection)
    assert SpinningDirection.CLOCKWISE.value == 0
    assert model.n_rows == 3
    assert model.n_columns == 5
    assert model.n_elements == 15
    assert model.fov_vert == (0.2, 0.4)
    assert model.fov_horiz == (math.pi, 2.0 * math.pi)
    assert model.spinning_direction is SpinningDirection.CLOCKWISE
    assert len(jax.tree.leaves(nnx.state(model, nnx.Param))) == 3
    assert LidarFrameSet == dict[str, LidarFrame]


def test_lidar_model_requires_a_concrete_projection_and_is_not_callable():
    with pytest.raises(TypeError, match="projection=None"):
        LidarModel(None)
    with pytest.raises(TypeError, match="unsupported"):
        LidarModel(object())
    with pytest.raises(NotImplementedError, match="projection methods"):
        _model().forward()
    with pytest.raises(NotImplementedError, match="not callable"):
        _model()()


def test_model_angle_ray_elements_and_distance_adapters():
    model = _model()
    angles = jnp.array([[0.1, 0.5], [-0.1, -1.0]])
    rays = model.sensor_angles_to_sensor_rays(angles, return_valid_flag=True)
    recovered = model.sensor_rays_to_sensor_angles(
        rays.sensor_rays, return_valid_flag=True
    )
    np.testing.assert_allclose(recovered.sensor_angles, angles, atol=1.0e-6)
    np.testing.assert_array_equal(rays.valid_flag, [True, True])
    np.testing.assert_array_equal(recovered.valid_flag, [True, True])

    zero = model.sensor_rays_to_sensor_angles(jnp.zeros((1, 3)), normalized=False)
    np.testing.assert_array_equal(zero.sensor_angles, [[0.0, 0.0]])

    elements = jnp.array([[0, 0], [2, 3]], dtype=jnp.int32)
    element_angles = model.elements_to_sensor_angles(elements, return_valid_flag=True)
    functional_angles = functional.elements_to_sensor_angles(
        elements, model.projection, return_valid_flag=True
    )
    np.testing.assert_allclose(
        element_angles.sensor_angles, functional_angles.sensor_angles
    )
    np.testing.assert_array_equal(element_angles.valid_flag, [True, True])
    element_rays = model.elements_to_sensor_rays(elements)
    distances = jnp.array([2.0, 3.0])
    np.testing.assert_allclose(
        model.elements_to_sensor_points(elements, distances),
        element_rays * distances[:, None],
    )


def test_model_fov_slack_and_relative_frame_times_follow_tables():
    projection = _projection()
    tight = LidarModel(projection, fov_eps_factor=0.0)
    loose = LidarModel(projection, fov_eps_factor=1.0e6)
    bottom = jnp.asarray(
        projection.fov_vert_start_rad - projection.fov_vert_span_rad,
        dtype=jnp.float32,
    )
    just_below = jnp.nextafter(bottom, jnp.asarray(-jnp.inf, dtype=bottom.dtype))
    angles = jnp.array([[just_below, 0.0]])
    np.testing.assert_array_equal(tight.valid_sensor_angles(angles), [False])
    np.testing.assert_array_equal(loose.valid_sensor_angles(angles), [True])
    np.testing.assert_array_equal(
        tight.valid_sensor_angles(jnp.array([[0.7, 0.0], [-7.0, 0.0]])),
        [False, False],
    )

    rows = jnp.array([0, 1, 2, 0, 1], dtype=jnp.int32)
    columns = jnp.arange(5, dtype=jnp.int32)
    table_angles = tight.elements_to_sensor_angles(
        jnp.stack((rows, columns), axis=-1)
    ).sensor_angles
    np.testing.assert_allclose(
        tight.sensor_angles_relative_frame_times(table_angles),
        jnp.linspace(0.0, 1.0, 5),
    )


def test_world_projection_filters_eager_and_matches_functional_metadata():
    model = _model()
    pose = _dynamic_pose()
    points = jnp.array([[5.0, 0.0, 0.0], [4.0, 1.0, -0.1], [0.0, 0.0, 5.0]])
    all_result = model.world_points_to_sensor_angles_shutter_pose(
        points,
        pose,
        start_timestamp_us=100,
        end_timestamp_us=500,
        return_T_sensor_world=True,
        return_valid_flag=True,
        return_valid_indices=True,
        return_timestamps=True,
        return_all_projections=True,
    )
    filtered = model.world_points_to_sensor_angles_shutter_pose(
        points,
        pose,
        start_timestamp_us=100,
        end_timestamp_us=500,
        return_T_sensor_world=True,
        return_valid_indices=True,
        return_timestamps=True,
    )
    functional_result = functional.inverse_project_spinning_lidar(
        model.projection,
        points,
        pose,
        start_timestamp_us=100,
        end_timestamp_us=500,
        return_T_sensor_world=True,
        return_valid_flag=True,
        return_timestamps=True,
    )
    np.testing.assert_allclose(
        all_result.sensor_angles, functional_result.sensor_angles
    )
    np.testing.assert_array_equal(all_result.valid_flag, functional_result.valid_flag)
    valid_count = int(jnp.sum(all_result.valid_flag))
    assert filtered.sensor_angles.shape == (valid_count, 2)
    assert filtered.T_sensor_world.shape == (valid_count, 4, 4)
    assert filtered.timestamps_us.shape == (valid_count,)
    np.testing.assert_array_equal(
        filtered.valid_indices, jnp.flatnonzero(all_result.valid_flag)
    )


def test_fixed_shape_lidar_model_projection_is_nnx_jittable():
    model = _model()
    pose = _dynamic_pose()

    @nnx.jit
    def project(current_model, points):
        return current_model.world_points_to_sensor_angles_shutter_pose(
            points,
            pose,
            return_valid_flag=True,
            return_all_projections=True,
        )

    result = project(model, jnp.array([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]))
    assert result.sensor_angles.shape == (2, 2)
    np.testing.assert_array_equal(result.valid_flag, [True, False])


def test_model_world_ray_adapter_and_nnx_table_gradients():
    model = _model()
    elements = jnp.array([[0, 0], [1, 2], [2, 4]], dtype=jnp.int32)
    result = model.elements_to_world_rays_shutter_pose(
        elements,
        _dynamic_pose(),
        start_timestamp_us=0,
        end_timestamp_us=100,
        return_T_sensor_world=True,
        return_timestamps=True,
    )
    assert result.world_rays.shape == (3, 6)
    assert result.T_sensor_world.shape == (3, 4, 4)
    np.testing.assert_array_equal(result.timestamps_us, [0, 50, 100])

    gradients = nnx.grad(
        lambda current: current.elements_to_sensor_angles(elements).sensor_angles.sum()
    )(model)
    assert bool(jnp.all(jnp.isfinite(gradients._row_elevations_rad[...])))
    assert bool(jnp.all(jnp.isfinite(gradients._column_azimuths_rad[...])))
    assert bool(jnp.all(jnp.isfinite(gradients._row_azimuth_offsets_rad[...])))


def test_dense_lidar_frame_uses_nnx_variables_and_reports_dimensions():
    frame = LidarFrame(
        "lidar",
        _model(),
        _static_pose(),
        0,
        100,
        jnp.zeros((3, 5, 2)),
        jnp.zeros((3, 5, 2)),
        timestamp_us=jnp.zeros((3, 5), dtype=jnp.int32),
        metadata={"weather": "clear"},
    )
    assert isinstance(frame, Frame)
    assert frame.is_dense and not frame.is_sparse
    assert frame.n_points == 15
    assert frame.max_returns == 2
    assert frame.metadata == {"weather": "clear"}
    assert isinstance(frame.distance_m, nnx.Variable)
    assert isinstance(frame.intensity, nnx.Variable)
    assert isinstance(frame.timestamp_us, nnx.Variable)
    assert not isinstance(frame.distance_m, nnx.Param)
    assert len(jax.tree.leaves(nnx.state(frame, nnx.Param))) == 5
    assert len(jax.tree.leaves(nnx.state(frame))) == 8


def test_sparse_lidar_frame_preserves_markers_and_optional_properties():
    frame = LidarFrame(
        "lidar",
        _model(),
        _dynamic_pose(),
        0,
        100,
        jnp.array([[1.0, jnp.nan], [2.0, 3.0]]),
        jnp.array([[0.5, 0.0], [0.4, 0.3]]),
        model_element=jnp.array([[0, 0], [1, 2]], dtype=jnp.int32),
        timestamp_us=jnp.array([0, 50], dtype=jnp.int32),
        optional_properties={"elongation": jnp.ones((2, 2))},
    )
    assert frame.is_sparse and not frame.is_dense
    assert frame.n_points == 2
    assert frame.max_returns == 2
    assert isinstance(frame.model_element, nnx.Variable)
    assert isinstance(frame.elongation, nnx.Variable)
    assert jnp.isnan(frame.distance_m[...][0, 1])
    assert float(frame.intensity[...][0, 1]) == 0.0
    np.testing.assert_array_equal(
        frame.optional_properties["elongation"], jnp.ones((2, 2))
    )
    assert isinstance(frame.pose, DynamicPose)
    assert len(jax.tree.leaves(nnx.state(frame, nnx.Param))) == 7


def test_lidar_frame_rejects_invalid_observation_shapes_and_collisions():
    common = (_model(), _static_pose(), 0, 100)
    with pytest.raises(ValueError, match="intensity"):
        LidarFrame("bad", *common, jnp.zeros((2, 3, 1)), jnp.zeros((2, 3, 2)))
    with pytest.raises(ValueError, match="dense distance_m"):
        LidarFrame("bad", *common, jnp.zeros((2, 3)), jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="dense timestamp_us"):
        LidarFrame(
            "bad",
            *common,
            jnp.zeros((2, 3, 1)),
            jnp.zeros((2, 3, 1)),
            timestamp_us=jnp.zeros((6,)),
        )
    with pytest.raises(ValueError, match="model_element"):
        LidarFrame(
            "bad",
            *common,
            jnp.zeros((2, 1)),
            jnp.zeros((2, 1)),
            model_element=jnp.zeros((2, 3), dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="sparse timestamp_us"):
        LidarFrame(
            "bad",
            *common,
            jnp.zeros((2, 1)),
            jnp.zeros((2, 1)),
            model_element=jnp.zeros((2, 2), dtype=jnp.int32),
            timestamp_us=jnp.zeros((2, 1), dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="distance_m"):
        LidarFrame(
            "bad",
            *common,
            jnp.zeros((2, 1)),
            jnp.zeros((2, 1)),
            model_element=jnp.zeros((2, 2), dtype=jnp.int32),
            optional_properties={"distance_m": jnp.zeros((2, 1))},
        )


def test_lidar_frame_without_optional_timestamps_is_not_callable():
    frame = LidarFrame(
        "lidar",
        _model(),
        _static_pose(),
        0,
        0,
        jnp.zeros((2, 1, 1)),
        jnp.zeros((2, 1, 1)),
    )
    assert frame.timestamp_us is None
    assert frame.optional_properties == {}
    with pytest.raises(NotImplementedError, match="data container"):
        frame.forward()
    with pytest.raises(NotImplementedError, match="data container"):
        frame()
