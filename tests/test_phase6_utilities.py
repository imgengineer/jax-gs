import jax
import jax.numpy as jnp
import pytest

import jax_gs
import jax_gs.trace as trace_module
from jax_gs.init_utils import knn_scale_init, multi_frame_depth_unprojection
from jax_gs.sensors import models as sensor_models
from jax_gs.sensors.kernels import projective_sensor_ops
from jax_gs.sensors.kernels.cameras import (
    REGISTERED_CAMERA_PROJECTIONS,
    REGISTERED_DISTORTIONS,
)
from jax_gs.sensors.kernels.common.tensor_ops import (
    raise_or_target_device,
    timestamp_bounds,
    to_dev,
    zero_like,
)
from jax_gs.sensors.kernels.lidars import dispatch
from jax_gs.sensors.kernels.lidars.types import (
    REGISTERED_LIDAR_PROJECTIONS,
    RowOffsetStructuredSpinningLidarProjection,
)
from jax_gs.training import TwoStageScheduler
from jax_gs.training.schedulers import TwoStageScheduleStep


def _lidar_projection() -> RowOffsetStructuredSpinningLidarProjection:
    return RowOffsetStructuredSpinningLidarProjection(
        row_elevations_rad=jnp.asarray([0.0], jnp.float32),
        column_azimuths_rad=jnp.asarray([0.0], jnp.float32),
        row_azimuth_offsets_rad=jnp.asarray([], jnp.float32),
        fov_vert_start_rad=0.5,
        fov_vert_span_rad=1.0,
        fov_horiz_start_rad=0.0,
        fov_horiz_span_rad=6.0,
        spinning_direction=0,
        has_row_offsets=False,
    )


def test_multiframe_unprojection_recovers_grid_and_uint8_colors():
    images = jnp.asarray(
        [[[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]]],
        dtype=jnp.uint8,
    )
    depths = jnp.ones((1, 2, 2), jnp.float32)
    masks = jnp.ones((1, 2, 2), jnp.bool_)
    poses = jnp.eye(4, dtype=jnp.float32)[None]
    intrinsics = jnp.asarray([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])

    xyz, rgb = multi_frame_depth_unprojection(images, depths, masks, poses, intrinsics)

    assert xyz.shape == (4, 3)
    assert jnp.allclose(
        xyz,
        jnp.asarray(
            [[-1.0, -1.0, 1.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        ),
    )
    assert jnp.allclose(rgb, images.reshape(-1, 3).astype(jnp.float32) / 255.0)


def test_multiframe_unprojection_masks_depth_and_reproducibly_subsamples():
    images = jnp.arange(4 * 3, dtype=jnp.float32).reshape(1, 2, 2, 3)
    depths = jnp.asarray([[[1.0, 0.0], [1.0, 1.0]]])
    masks = jnp.ones((1, 2, 2), jnp.bool_)
    poses = jnp.eye(4, dtype=jnp.float32)[None]
    intrinsics = jnp.eye(3, dtype=jnp.float32)[None]
    result_a = multi_frame_depth_unprojection(
        images, depths, masks, poses, intrinsics, 2, key=jax.random.key(4)
    )
    result_b = multi_frame_depth_unprojection(
        images, depths, masks, poses, intrinsics, 2, key=jax.random.key(4)
    )
    assert result_a[0].shape == (2, 3)
    assert jnp.array_equal(result_a[0], result_b[0])
    assert jnp.array_equal(result_a[1], result_b[1])


def test_multiframe_unprojection_validates_frames_and_returns_empty():
    args = (
        jnp.zeros((1, 1, 1, 3)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.eye(4)[None],
        jnp.eye(3)[None],
    )
    xyz, rgb = multi_frame_depth_unprojection(*args)
    assert xyz.shape == rgb.shape == (0, 3)
    with pytest.raises(ValueError, match="leading dim"):
        multi_frame_depth_unprojection(args[0], jnp.zeros((2, 1, 1)), *args[2:])


def test_knn_scale_init_matches_uniform_grid_across_chunks():
    xyz = (
        jnp.zeros((8, 3), jnp.float32)
        .at[:, 0]
        .set(jnp.arange(8, dtype=jnp.float32) * 0.5)
    )
    result = knn_scale_init(xyz, k=1, chunk_size=3)
    assert jnp.allclose(result, jnp.log(0.5), atol=1.0e-6)
    with pytest.raises(ValueError, match="at least"):
        knn_scale_init(xyz[:2], k=3)
    assert jnp.all(jnp.isfinite(knn_scale_init(jnp.zeros((5, 3)), k=2)))


def test_two_stage_scheduler_matches_current_main_boundaries():
    scheduler = TwoStageScheduler(2, 3, coarse_frame_index=1)
    coarse = scheduler.step(1, num_frames=4)
    fine = [scheduler.step(step, 4) for step in range(2, 7)]
    assert isinstance(coarse, TwoStageScheduleStep)
    assert coarse == TwoStageScheduleStep("coarse", 1, False)
    assert [step.frame_index for step in fine] == [0, 1, 2, 3, 0]
    assert all(step.stage == "fine" and step.shuffle for step in fine)
    with pytest.raises(ValueError, match="global_step"):
        scheduler.step(-1, 4)
    with pytest.raises(ValueError, match="num_frames"):
        scheduler.step(0, 0)


def test_tensor_ops_preserve_device_and_enforce_conversion_policy():
    value = jnp.asarray([1.0, 2.0], jnp.float32)
    device = raise_or_target_device(value, allow_device_transfer=False)
    assert to_dev(value, device, jnp.float32, False).device == device
    with pytest.raises(RuntimeError, match="allow_device_transfer"):
        to_dev(value, device, jnp.int32, False)
    converted = to_dev(value, device, jnp.int32, True)
    assert converted.dtype == jnp.int32
    zeros = zero_like((2, 3), value)
    assert zeros.dtype == value.dtype and zeros.device == device
    assert timestamp_bounds(None, None) == (0, 0)
    assert timestamp_bounds(4, 9) == (4, 9)
    with pytest.raises(ValueError, match="provided together"):
        timestamp_bounds(4, None)


def test_lidar_dispatch_covers_registered_type_and_routes_calls():
    expected_operations = {
        "sensor_rays_to_sensor_angles",
        "sensor_angles_to_sensor_rays",
        "elements_to_sensor_angles",
        "generate_spinning_lidar_rays",
        "inverse_project_spinning_lidar",
    }
    assert set(dispatch._DISPATCH_TABLES) == expected_operations
    for table in dispatch._DISPATCH_TABLES.values():
        assert set(table) == set(REGISTERED_LIDAR_PROJECTIONS)
    projection = _lidar_projection()
    angles = dispatch.sensor_rays_to_sensor_angles(
        jnp.asarray([[1.0, 0.0, 0.0]]), projection
    )
    assert jnp.allclose(angles, 0.0)
    with pytest.raises(TypeError, match="Unsupported LiDAR projection"):
        dispatch._lookup(
            dispatch._DISPATCH_TABLES["sensor_rays_to_sensor_angles"], object()
        )


def test_trace_helpers_preserve_range_order_and_decorator_result(monkeypatch):
    calls = []

    class FakeAnnotation:
        def __init__(self, name, kwargs):
            self.name = name
            self.kwargs = kwargs

        def __enter__(self):
            calls.append(("push", self.name, self.kwargs))

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append(("pop", self.name, {}))
            return False

    monkeypatch.setattr(
        trace_module,
        "_annotation",
        lambda name, kwargs: FakeAnnotation(name, kwargs),
    )
    trace_module.trace_push("outer", category="test")
    with trace_module.trace_range("inner"):
        pass

    @trace_module.trace_function("leaf")
    def decorated():
        return 7

    assert decorated() == 7
    trace_module.trace_pop()
    assert calls == [
        ("push", "outer", {"category": "test"}),
        ("push", "inner", {}),
        ("pop", "inner", {}),
        ("push", "leaf", {}),
        ("pop", "leaf", {}),
        ("pop", "outer", {}),
    ]
    with pytest.raises(RuntimeError, match="without a matching"):
        trace_module.trace_pop()


def test_current_main_root_and_sensor_model_reexports_are_complete():
    assert "__version__" in jax_gs.__all__
    assert "load_checkpoint_appearance_image_names" in jax_gs.__all__
    assert callable(jax_gs.load_checkpoint_appearance_image_names)
    expected_model_exports = {
        "BivariateWindshieldDistortion",
        "CameraProjection",
        "DynamicPose",
        "ImagePointsReturn",
        "OpenCVPinholeProjection",
        "Pose",
        "SensorAnglesReturn",
        "Trajectory",
        "WorldRaysReturn",
        "from_components",
    }
    assert expected_model_exports <= set(sensor_models.__all__)
    assert all(hasattr(sensor_models, name) for name in expected_model_exports)


def test_projective_dispatch_tables_cover_registered_cartesian_product():
    expected = {
        (projection, distortion)
        for projection in REGISTERED_CAMERA_PROJECTIONS
        for distortion in REGISTERED_DISTORTIONS
    }
    assert len(projective_sensor_ops._DISPATCH_TABLES) == 7
    for table in projective_sensor_ops._DISPATCH_TABLES.values():
        assert set(table) == expected
        assert all(callable(backend) for backend in table.values())
    with pytest.raises(TypeError, match="NoExternalDistortion"):
        projective_sensor_ops.camera_rays_to_image_points(
            jnp.asarray([[0.0, 0.0, 1.0]]), object(), None
        )
