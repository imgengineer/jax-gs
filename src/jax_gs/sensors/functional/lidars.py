"""Structured wrappers for pure-JAX spinning-LiDAR operations."""

from __future__ import annotations

import jax

from ..kernels.common.pose import DynamicPose
from ..kernels.common.utils import poses_to_matrix, valid_flags_to_indices
from ..kernels.lidars import ops as _kernel_ops
from ..kernels.lidars.types import RowOffsetStructuredSpinningLidarProjection
from .return_types import (
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)


def sensor_rays_to_sensor_angles(
    sensor_rays: jax.Array,
    *,
    allow_device_transfer: bool = False,
) -> SensorAnglesReturn:
    return SensorAnglesReturn(
        sensor_angles=_kernel_ops.sensor_rays_to_sensor_angles(
            sensor_rays, allow_device_transfer=allow_device_transfer
        )
    )


def sensor_angles_to_sensor_rays(
    sensor_angles: jax.Array,
    *,
    allow_device_transfer: bool = False,
) -> SensorRayReturn:
    return SensorRayReturn(
        sensor_rays=_kernel_ops.sensor_angles_to_sensor_rays(
            sensor_angles, allow_device_transfer=allow_device_transfer
        )
    )


def elements_to_sensor_angles(
    elements: jax.Array,
    projection: RowOffsetStructuredSpinningLidarProjection,
    *,
    return_valid_flag: bool = False,
    allow_device_transfer: bool = False,
) -> SensorAnglesReturn:
    sensor_angles, valid_flag = _kernel_ops.elements_to_sensor_angles(
        elements,
        projection,
        allow_device_transfer=allow_device_transfer,
    )
    return SensorAnglesReturn(
        sensor_angles=sensor_angles,
        valid_flag=valid_flag if return_valid_flag else None,
    )


def generate_spinning_lidar_rays(
    projection: RowOffsetStructuredSpinningLidarProjection,
    elements: jax.Array | None,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_T_sensor_world: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldRaysReturn:
    world_rays, timestamps, pose_t, pose_r = _kernel_ops.generate_spinning_lidar_rays(
        projection,
        elements,
        dynamic_pose,
        start_timestamp_us=start_timestamp_us,
        end_timestamp_us=end_timestamp_us,
        return_timestamps=return_timestamps,
        return_poses=return_T_sensor_world,
        allow_device_transfer=allow_device_transfer,
    )
    return WorldRaysReturn(
        world_rays=world_rays,
        T_sensor_world=poses_to_matrix(pose_t, pose_r)
        if return_T_sensor_world
        else None,
        timestamps_us=timestamps if return_timestamps else None,
    )


def inverse_project_spinning_lidar(
    projection: RowOffsetStructuredSpinningLidarProjection,
    world_points: jax.Array,
    dynamic_pose: DynamicPose,
    *,
    max_iterations: int = 10,
    stop_mean_relative_time_error: float = 1.0e-4,
    stop_delta_mean_relative_time_error: float = 1.0e-6,
    initial_relative_time: float = 0.5,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_T_sensor_world: bool = False,
    return_valid_flag: bool = False,
    return_valid_indices: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldPointsToSensorAnglesReturn:
    sensor_angles, valid_flag, timestamps, pose_t, pose_r = (
        _kernel_ops.inverse_project_spinning_lidar(
            projection,
            world_points,
            dynamic_pose,
            max_iterations=max_iterations,
            stop_mean_relative_time_error=stop_mean_relative_time_error,
            stop_delta_mean_relative_time_error=stop_delta_mean_relative_time_error,
            initial_relative_time=initial_relative_time,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_timestamps=return_timestamps,
            return_poses=return_T_sensor_world,
            allow_device_transfer=allow_device_transfer,
        )
    )
    return WorldPointsToSensorAnglesReturn(
        sensor_angles=sensor_angles,
        T_sensor_world=poses_to_matrix(pose_t, pose_r)
        if return_T_sensor_world
        else None,
        valid_flag=valid_flag if return_valid_flag else None,
        valid_indices=valid_flags_to_indices(valid_flag)
        if return_valid_indices
        else None,
        timestamps_us=timestamps if return_timestamps else None,
    )


__all__ = [
    "elements_to_sensor_angles",
    "generate_spinning_lidar_rays",
    "inverse_project_spinning_lidar",
    "sensor_angles_to_sensor_rays",
    "sensor_rays_to_sensor_angles",
]
