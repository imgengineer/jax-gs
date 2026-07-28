"""Structured return values for stateless sensor operations."""

from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class ImagePointsReturn:
    image_points: jax.Array
    valid_flag: jax.Array
    jacobians: jax.Array | None = None


@dataclass(frozen=True)
class PixelsReturn:
    pixels: jax.Array
    valid_flag: jax.Array


@dataclass(frozen=True)
class WorldPointsToImagePointsReturn:
    image_points: jax.Array
    T_sensor_world: jax.Array | None = None
    valid_flag: jax.Array | None = None
    valid_indices: jax.Array | None = None
    timestamps_us: jax.Array | None = None


@dataclass(frozen=True)
class WorldPointsToPixelsReturn:
    pixels: jax.Array
    T_sensor_world: jax.Array | None = None
    valid_flag: jax.Array | None = None
    valid_indices: jax.Array | None = None
    timestamps_us: jax.Array | None = None


@dataclass(frozen=True)
class WorldRaysReturn:
    world_rays: jax.Array
    T_sensor_world: jax.Array | None = None
    timestamps_us: jax.Array | None = None


@dataclass(frozen=True)
class SensorAnglesReturn:
    sensor_angles: jax.Array
    valid_flag: jax.Array | None = None


@dataclass(frozen=True)
class SensorRayReturn:
    sensor_rays: jax.Array
    valid_flag: jax.Array | None = None


@dataclass(frozen=True)
class WorldPointsToSensorAnglesReturn:
    sensor_angles: jax.Array
    T_sensor_world: jax.Array | None = None
    valid_flag: jax.Array | None = None
    valid_indices: jax.Array | None = None
    timestamps_us: jax.Array | None = None


for _return_type in (
    ImagePointsReturn,
    PixelsReturn,
    WorldPointsToImagePointsReturn,
    WorldPointsToPixelsReturn,
    WorldRaysReturn,
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToSensorAnglesReturn,
):
    jax.tree_util.register_dataclass(
        _return_type,
        data_fields=tuple(_return_type.__dataclass_fields__),
        meta_fields=(),
    )


__all__ = [
    "ImagePointsReturn",
    "PixelsReturn",
    "SensorAnglesReturn",
    "SensorRayReturn",
    "WorldPointsToImagePointsReturn",
    "WorldPointsToPixelsReturn",
    "WorldPointsToSensorAnglesReturn",
    "WorldRaysReturn",
]
