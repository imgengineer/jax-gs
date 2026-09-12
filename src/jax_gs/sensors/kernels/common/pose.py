"""JAX pytree pose and trajectory types used by sensor operations."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def _vector(value, size: int, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating-point dtype")
    return array


@dataclass(frozen=True)
class Pose:
    """Static sensor-to-world pose with a ``wxyz`` quaternion."""

    translation: jax.Array
    rotation: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "translation", _vector(self.translation, 3, "translation")
        )
        object.__setattr__(self, "rotation", _vector(self.rotation, 4, "rotation"))


@dataclass(frozen=True)
class Trajectory:
    """Piecewise pose trajectory at normalized control times."""

    control_poses: tuple[Pose, ...]
    control_count: int
    control_times: jax.Array

    def __post_init__(self) -> None:
        count = int(self.control_count)
        if count != len(self.control_poses):
            raise ValueError("control_count must equal len(control_poses)")
        times = jnp.asarray(self.control_times)
        if times.shape != (count,):
            raise ValueError(f"control_times must have shape ({count},)")
        if not jnp.issubdtype(times.dtype, jnp.floating):
            raise TypeError("control_times must have a floating-point dtype")
        object.__setattr__(self, "control_count", count)
        object.__setattr__(self, "control_times", times)


@dataclass(frozen=True)
class DynamicPose:
    """Two-pose trajectory over normalized frame time ``[0, 1]``."""

    start_pose: Pose
    end_pose: Pose

    @staticmethod
    def from_static_pose(pose: Pose) -> DynamicPose:
        end_pose = Pose(
            translation=jnp.array(pose.translation),
            rotation=jnp.array(pose.rotation),
        )
        return DynamicPose(start_pose=pose, end_pose=end_pose)

    def to_trajectory(self) -> Trajectory:
        return Trajectory(
            control_poses=(self.start_pose, self.end_pose),
            control_count=2,
            control_times=jnp.asarray(
                (0.0, 1.0), dtype=self.start_pose.translation.dtype
            ),
        )


jax.tree_util.register_dataclass(
    Pose, data_fields=("translation", "rotation"), meta_fields=()
)
jax.tree_util.register_dataclass(
    Trajectory,
    data_fields=("control_poses", "control_times"),
    meta_fields=("control_count",),
)
jax.tree_util.register_dataclass(
    DynamicPose, data_fields=("start_pose", "end_pose"), meta_fields=()
)


__all__ = ["DynamicPose", "Pose", "Trajectory"]
