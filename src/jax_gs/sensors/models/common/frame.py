"""Flax NNX state for a trainable sensor frame."""

from __future__ import annotations

from typing import Any, TypeAlias

import jax.numpy as jnp
from flax import nnx

from ...kernels.common.pose import DynamicPose, Pose

FrameId: TypeAlias = str  # noqa: UP040 - preserve runtime equality with str


class Frame(nnx.Module):
    """Common frame metadata and learnable static or dynamic pose state."""

    def __init__(
        self,
        frame_id: FrameId,
        pose: Pose | DynamicPose,
        timestamp_start_us: int,
        timestamp_end_us: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.frame_id = str(frame_id)
        self.timestamp_start_us = int(timestamp_start_us)
        self.timestamp_end_us = int(timestamp_end_us)
        self.metadata = {} if metadata is None else dict(metadata)

        if isinstance(pose, Pose):
            self._pose_kind = "static"
            self.pose_translation = nnx.Param(jnp.asarray(pose.translation))
            self.pose_rotation = nnx.Param(jnp.asarray(pose.rotation))
        elif isinstance(pose, DynamicPose):
            self._pose_kind = "dynamic"
            self.start_pose_translation = nnx.Param(
                jnp.asarray(pose.start_pose.translation)
            )
            self.start_pose_rotation = nnx.Param(jnp.asarray(pose.start_pose.rotation))
            self.end_pose_translation = nnx.Param(
                jnp.asarray(pose.end_pose.translation)
            )
            self.end_pose_rotation = nnx.Param(jnp.asarray(pose.end_pose.rotation))
        else:
            raise TypeError("pose must be Pose or DynamicPose")

    @property
    def pose(self) -> Pose | DynamicPose:
        """Rebuild the pure-JAX pose value from the current NNX parameters."""

        if self._pose_kind == "static":
            return Pose(
                translation=self.pose_translation[...],
                rotation=self.pose_rotation[...],
            )
        return DynamicPose(
            start_pose=Pose(
                translation=self.start_pose_translation[...],
                rotation=self.start_pose_rotation[...],
            ),
            end_pose=Pose(
                translation=self.end_pose_translation[...],
                rotation=self.end_pose_rotation[...],
            ),
        )

    @property
    def is_rolling_shutter(self) -> bool:
        return self.timestamp_start_us != self.timestamp_end_us

    @property
    def frame_duration_us(self) -> int:
        return self.timestamp_end_us - self.timestamp_start_us

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError("Frame is a state container, not a callable model")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


__all__ = ["Frame", "FrameId"]
