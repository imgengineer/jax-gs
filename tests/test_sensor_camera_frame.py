import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_gs.sensors.kernels.cameras import (
    NoExternalDistortion,
    OpenCVPinholeProjection,
    ShutterType,
)
from jax_gs.sensors.kernels.common import DynamicPose, Pose
from jax_gs.sensors.models import CameraModel, Frame, ImageFrame


def _camera() -> CameraModel:
    projection = OpenCVPinholeProjection(
        focal_length=jnp.array([100.0, 120.0]),
        principal_point=jnp.array([50.0, 40.0]),
        radial_coeffs=jnp.zeros(6),
        tangential_coeffs=jnp.zeros(2),
        thin_prism_coeffs=jnp.zeros(4),
        resolution=(100, 80),
    )
    return CameraModel(
        projection,
        NoExternalDistortion(),
        (100, 80),
        ShutterType.GLOBAL,
    )


def _pose(translation=(0.0, 0.0, 0.0)) -> Pose:
    return Pose(
        jnp.asarray(translation, dtype=jnp.float32),
        jnp.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_image_frame_stores_image_as_buffer_and_pose_as_parameters():
    frame = ImageFrame(
        "cam",
        _camera(),
        _pose(),
        10,
        20,
        jnp.zeros((80, 100, 3)),
        {"exposure": 1.0},
    )
    assert isinstance(frame, Frame)
    assert isinstance(frame.image, nnx.Variable)
    assert not isinstance(frame.image, nnx.Param)
    assert (frame.height, frame.width, frame.channels) == (80, 100, 3)
    assert frame.frame_id == "cam"
    assert frame.metadata == {"exposure": 1.0}
    assert frame.is_rolling_shutter
    assert frame.frame_duration_us == 10
    np.testing.assert_array_equal(frame.pose.translation, jnp.zeros(3))

    parameter_state = nnx.state(frame, nnx.Param)
    assert len(jax.tree.leaves(parameter_state)) == 7
    all_state = nnx.state(frame)
    assert len(jax.tree.leaves(all_state)) == 8


def test_dynamic_frame_pose_rebuilds_from_mutable_nnx_state():
    dynamic_pose = DynamicPose(_pose(), _pose((0.1, 0.0, 0.0)))
    frame = ImageFrame(
        "rolling",
        _camera(),
        dynamic_pose,
        0,
        100,
        jnp.zeros((2, 3, 1)),
    )
    assert isinstance(frame.pose, DynamicPose)
    np.testing.assert_allclose(frame.pose.end_pose.translation, [0.1, 0.0, 0.0])
    frame.end_pose_translation[...] = jnp.array([0.2, 0.0, 0.0])
    np.testing.assert_allclose(frame.pose.end_pose.translation, [0.2, 0.0, 0.0])
    assert len(jax.tree.leaves(nnx.state(frame, nnx.Param))) == 9


def test_frame_validates_pose_and_image_shape_and_is_not_callable():
    with pytest.raises(TypeError, match="Pose"):
        Frame("bad", object(), 0, 0)
    with pytest.raises(ValueError, match="image must be"):
        ImageFrame("bad", _camera(), _pose(), 0, 0, jnp.zeros((2, 3)))
    frame = ImageFrame("cam", _camera(), _pose(), 0, 0, jnp.zeros((2, 3, 1)))
    assert not frame.is_rolling_shutter
    with pytest.raises(NotImplementedError, match="data container"):
        frame.forward()
    with pytest.raises(NotImplementedError, match="data container"):
        frame()

    base_frame = Frame("base", _pose(), 0, 0)
    with pytest.raises(NotImplementedError, match="state container"):
        base_frame.forward()
