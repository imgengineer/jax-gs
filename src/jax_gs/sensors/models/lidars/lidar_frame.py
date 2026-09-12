"""Flax NNX observation state for dense or sparse LiDAR frames."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from ...kernels.common.pose import DynamicPose, Pose
from ..common.frame import Frame, FrameId
from .lidar_model import LidarModel


def _validate_lidar_observations(
    distance_m,
    intensity,
    model_element,
    timestamp_us,
) -> None:
    if intensity.shape != distance_m.shape:
        raise ValueError(
            "intensity must have the same shape as distance_m, got "
            f"{intensity.shape} and {distance_m.shape}"
        )
    if model_element is None:
        if distance_m.ndim != 3:
            raise ValueError(
                f"dense distance_m must be (H, W, R), got {distance_m.shape}"
            )
        if timestamp_us is not None and timestamp_us.shape != distance_m.shape[:2]:
            raise ValueError(
                f"dense timestamp_us must be (H, W), got {timestamp_us.shape}"
            )
        return
    if distance_m.ndim != 2:
        raise ValueError(f"sparse distance_m must be (N, R), got {distance_m.shape}")
    if model_element.ndim != 2 or model_element.shape[1] != 2:
        raise ValueError(f"model_element must be (N, 2), got {model_element.shape}")
    if model_element.shape[0] != distance_m.shape[0]:
        raise ValueError("model_element first dimension must match sparse distance_m")
    if timestamp_us is not None and timestamp_us.shape != (distance_m.shape[0],):
        raise ValueError(f"sparse timestamp_us must be (N,), got {timestamp_us.shape}")


class LidarFrame(Frame):
    """Pair a LiDAR model and trainable pose with observation buffers."""

    def __init__(
        self,
        frame_id: FrameId,
        lidar_model: LidarModel,
        pose: Pose | DynamicPose,
        timestamp_start_us: int,
        timestamp_end_us: int,
        distance_m,
        intensity,
        model_element=None,
        timestamp_us=None,
        optional_properties: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        distance = jnp.asarray(distance_m)
        intensity_array = jnp.asarray(intensity)
        element_array = None if model_element is None else jnp.asarray(model_element)
        timestamp_array = None if timestamp_us is None else jnp.asarray(timestamp_us)
        _validate_lidar_observations(
            distance, intensity_array, element_array, timestamp_array
        )
        super().__init__(
            frame_id,
            pose,
            timestamp_start_us,
            timestamp_end_us,
            metadata,
        )
        self.lidar_model = lidar_model
        self.distance_m = nnx.Variable(distance)
        self.intensity = nnx.Variable(intensity_array)
        self.model_element = (
            None if element_array is None else nnx.Variable(element_array)
        )
        self.timestamp_us = (
            None if timestamp_array is None else nnx.Variable(timestamp_array)
        )
        self._optional_property_names = ()
        if optional_properties:
            names = []
            for name, value in optional_properties.items():
                if hasattr(self, name):
                    raise ValueError(
                        f"optional property {name!r} collides with an existing "
                        "LidarFrame attribute"
                    )
                setattr(self, name, nnx.Variable(jnp.asarray(value)))
                names.append(name)
            self._optional_property_names = tuple(names)

    @property
    def is_sparse(self) -> bool:
        return self.model_element is not None

    @property
    def is_dense(self) -> bool:
        return self.model_element is None

    @property
    def n_points(self) -> int:
        shape = self.distance_m[...].shape
        return int(shape[0]) if self.is_sparse else int(shape[0] * shape[1])

    @property
    def max_returns(self) -> int:
        return int(self.distance_m[...].shape[-1])

    @property
    def optional_properties(self) -> dict[str, jax.Array]:
        return {
            name: getattr(self, name)[...] for name in self._optional_property_names
        }

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "LidarFrame is a data container, not a callable model"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


LidarFrameSet = dict[FrameId, LidarFrame]


__all__ = ["LidarFrame", "LidarFrameSet"]
