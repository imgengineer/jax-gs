"""Image observation state backed by Flax NNX variables."""

from __future__ import annotations

from typing import Any

from flax import nnx
import jax.numpy as jnp

from ...kernels.common.pose import DynamicPose, Pose
from ..common.frame import Frame, FrameId
from .camera_model import CameraModel


class ImageFrame(Frame):
    """Pair a trainable camera and pose with a non-parameter image buffer."""

    def __init__(
        self,
        frame_id: FrameId,
        camera_model: CameraModel,
        pose: Pose | DynamicPose,
        timestamp_start_us: int,
        timestamp_end_us: int,
        image,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        image_array = jnp.asarray(image)
        if image_array.ndim != 3:
            raise ValueError(f"image must be (H, W, C), got {image_array.shape}")
        super().__init__(
            frame_id,
            pose,
            timestamp_start_us,
            timestamp_end_us,
            metadata,
        )
        self.camera_model = camera_model
        self.image = nnx.Variable(image_array)

    @property
    def height(self) -> int:
        return int(self.image[...].shape[0])

    @property
    def width(self) -> int:
        return int(self.image[...].shape[1])

    @property
    def channels(self) -> int:
        return int(self.image[...].shape[2])

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError("ImageFrame is a data container, not a callable model")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


ImageFrameGroup = dict[FrameId, ImageFrame]


__all__ = ["ImageFrame", "ImageFrameGroup"]
