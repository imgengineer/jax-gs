"""Single-key dispatch for the pure-JAX spinning-LiDAR operations."""

from __future__ import annotations

from collections.abc import Callable

from . import ops as lidar_ops
from .types import (
    REGISTERED_LIDAR_PROJECTION_NAMES,
    REGISTERED_LIDAR_PROJECTIONS,
    RowOffsetStructuredSpinningLidarProjection,
    script_class_name,
)


DispatchKey = type[RowOffsetStructuredSpinningLidarProjection]

_SENSOR_RAYS_TO_SENSOR_ANGLES_BACKENDS: dict[DispatchKey, Callable] = {
    RowOffsetStructuredSpinningLidarProjection: lidar_ops.sensor_rays_to_sensor_angles,
}
_SENSOR_ANGLES_TO_SENSOR_RAYS_BACKENDS: dict[DispatchKey, Callable] = {
    RowOffsetStructuredSpinningLidarProjection: lidar_ops.sensor_angles_to_sensor_rays,
}
_ELEMENTS_TO_SENSOR_ANGLES_BACKENDS: dict[DispatchKey, Callable] = {
    RowOffsetStructuredSpinningLidarProjection: lidar_ops.elements_to_sensor_angles,
}
_GENERATE_SPINNING_LIDAR_RAYS_BACKENDS: dict[DispatchKey, Callable] = {
    RowOffsetStructuredSpinningLidarProjection: lidar_ops.generate_spinning_lidar_rays,
}
_INVERSE_PROJECT_SPINNING_LIDAR_BACKENDS: dict[DispatchKey, Callable] = {
    RowOffsetStructuredSpinningLidarProjection: lidar_ops.inverse_project_spinning_lidar,
}

_DISPATCH_TABLES = {
    "sensor_rays_to_sensor_angles": _SENSOR_RAYS_TO_SENSOR_ANGLES_BACKENDS,
    "sensor_angles_to_sensor_rays": _SENSOR_ANGLES_TO_SENSOR_RAYS_BACKENDS,
    "elements_to_sensor_angles": _ELEMENTS_TO_SENSOR_ANGLES_BACKENDS,
    "generate_spinning_lidar_rays": _GENERATE_SPINNING_LIDAR_RAYS_BACKENDS,
    "inverse_project_spinning_lidar": _INVERSE_PROJECT_SPINNING_LIDAR_BACKENDS,
}
_REGISTERED_CLASS_NAMES = dict(
    zip(REGISTERED_LIDAR_PROJECTIONS, REGISTERED_LIDAR_PROJECTION_NAMES, strict=True)
)


def _lookup(table: dict[DispatchKey, Callable], projection: object) -> Callable:
    projection_name = script_class_name(projection)
    for projection_class, backend in table.items():
        if projection_name == _REGISTERED_CLASS_NAMES[projection_class]:
            return backend
    raise TypeError(f"Unsupported LiDAR projection: {projection_name}")


def sensor_rays_to_sensor_angles(sensor_rays, projection, **kwargs):
    return _lookup(_SENSOR_RAYS_TO_SENSOR_ANGLES_BACKENDS, projection)(
        sensor_rays, **kwargs
    )


def sensor_angles_to_sensor_rays(sensor_angles, projection, **kwargs):
    return _lookup(_SENSOR_ANGLES_TO_SENSOR_RAYS_BACKENDS, projection)(
        sensor_angles, **kwargs
    )


def elements_to_sensor_angles(elements, projection, **kwargs):
    return _lookup(_ELEMENTS_TO_SENSOR_ANGLES_BACKENDS, projection)(
        elements, projection, **kwargs
    )


def generate_spinning_lidar_rays(projection, *args, **kwargs):
    return _lookup(_GENERATE_SPINNING_LIDAR_RAYS_BACKENDS, projection)(
        projection, *args, **kwargs
    )


def inverse_project_spinning_lidar(projection, *args, **kwargs):
    return _lookup(_INVERSE_PROJECT_SPINNING_LIDAR_BACKENDS, projection)(
        projection, *args, **kwargs
    )


__all__ = [
    "REGISTERED_LIDAR_PROJECTIONS",
    "_DISPATCH_TABLES",
    "elements_to_sensor_angles",
    "generate_spinning_lidar_rays",
    "inverse_project_spinning_lidar",
    "sensor_angles_to_sensor_rays",
    "sensor_rays_to_sensor_angles",
]
