"""Stateless structured wrappers for pure-JAX camera operations."""

from __future__ import annotations

import jax

from ..kernels.cameras import ops as _kernel_ops
from ..kernels.cameras.types import (
    CameraProjection,
    ExternalDistortion,
    ShutterType,
)
from ..kernels.common.pose import DynamicPose, Pose
from ..kernels.common.utils import poses_to_matrix, valid_flags_to_indices
from .return_types import (
    ImagePointsReturn,
    WorldPointsToImagePointsReturn,
    WorldRaysReturn,
)


def camera_rays_to_image_points(
    camera_rays: jax.Array,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    *,
    allow_device_transfer: bool = False,
) -> ImagePointsReturn:
    """Project camera-space rays and package points with their valid mask."""

    image_points, valid_flag = _kernel_ops.camera_rays_to_image_points(
        camera_rays,
        projection,
        external_distortion,
        allow_device_transfer=allow_device_transfer,
    )
    return ImagePointsReturn(image_points=image_points, valid_flag=valid_flag)


def image_points_to_camera_rays(
    image_points: jax.Array,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    *,
    allow_device_transfer: bool = False,
) -> jax.Array:
    """Back-project image coordinates to unit directions in camera space."""

    return _kernel_ops.image_points_to_camera_rays(
        image_points,
        projection,
        external_distortion,
        allow_device_transfer=allow_device_transfer,
    )


def generate_image_points(
    resolution: tuple[int, int],
    device: jax.Device | str | None = None,
    *,
    allow_device_transfer: bool = False,
) -> jax.Array:
    """Generate ``(x + .5, y + .5)`` coordinates for a complete image."""

    return _kernel_ops.generate_image_points(
        resolution,
        device=device,
        allow_device_transfer=allow_device_transfer,
    )


def project_world_points_mean_pose(
    world_points: jax.Array,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_T_sensor_world: bool = False,
    return_valid_flag: bool = False,
    return_valid_indices: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldPointsToImagePointsReturn:
    """Project world points at the mean pose and package optional metadata."""

    image_points, valid_flag, timestamps, pose_t, pose_r = (
        _kernel_ops.project_world_points_mean_pose(
            world_points,
            projection,
            external_distortion,
            dynamic_pose,
            resolution,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_valid_flags=return_valid_flag or return_valid_indices,
            return_timestamps=return_timestamps,
            return_poses=return_T_sensor_world,
            allow_device_transfer=allow_device_transfer,
        )
    )
    return WorldPointsToImagePointsReturn(
        image_points=image_points,
        T_sensor_world=poses_to_matrix(pose_t, pose_r)
        if return_T_sensor_world
        else None,
        valid_flag=valid_flag if return_valid_flag else None,
        valid_indices=valid_flags_to_indices(valid_flag)
        if return_valid_indices
        else None,
        timestamps_us=timestamps if return_timestamps else None,
    )


def project_world_points_shutter_pose(
    world_points: jax.Array,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    max_iterations: int = 10,
    stop_mean_error_px: float = 0.001,
    stop_delta_mean_error_px: float = 0.00001,
    initial_relative_time: float = 0.5,
    return_T_sensor_world: bool = False,
    return_valid_flag: bool = False,
    return_valid_indices: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldPointsToImagePointsReturn:
    """Project world points with per-point rolling-shutter compensation."""

    image_points, valid_flag, timestamps, pose_t, pose_r = (
        _kernel_ops.project_world_points_shutter_pose(
            world_points,
            projection,
            external_distortion,
            resolution,
            shutter_type,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            max_iterations=max_iterations,
            stop_mean_error_px=stop_mean_error_px,
            stop_delta_mean_error_px=stop_delta_mean_error_px,
            initial_relative_time=initial_relative_time,
            return_valid_flags=return_valid_flag or return_valid_indices,
            return_timestamps=return_timestamps,
            return_poses=return_T_sensor_world,
            allow_device_transfer=allow_device_transfer,
        )
    )
    return WorldPointsToImagePointsReturn(
        image_points=image_points,
        T_sensor_world=poses_to_matrix(pose_t, pose_r)
        if return_T_sensor_world
        else None,
        valid_flag=valid_flag if return_valid_flag else None,
        valid_indices=valid_flags_to_indices(valid_flag)
        if return_valid_indices
        else None,
        timestamps_us=timestamps if return_timestamps else None,
    )


def image_points_to_world_rays_static_pose(
    image_points: jax.Array,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    pose: Pose,
    *,
    timestamp_us: int | None = None,
    return_T_sensor_world: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldRaysReturn:
    """Back-project image points through a fixed sensor-to-world pose."""

    world_rays, timestamps, pose_t, pose_r = (
        _kernel_ops.image_points_to_world_rays_static_pose(
            image_points,
            projection,
            external_distortion,
            pose,
            timestamp_us=timestamp_us,
            return_timestamps=return_timestamps,
            return_poses=return_T_sensor_world,
            allow_device_transfer=allow_device_transfer,
        )
    )
    return WorldRaysReturn(
        world_rays=world_rays,
        T_sensor_world=poses_to_matrix(pose_t, pose_r)
        if return_T_sensor_world
        else None,
        timestamps_us=timestamps if return_timestamps else None,
    )


def image_points_to_world_rays_shutter_pose(
    image_points: jax.Array,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_T_sensor_world: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldRaysReturn:
    """Back-project image points at their rolling-shutter poses."""

    world_rays, timestamps, pose_t, pose_r = (
        _kernel_ops.image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            external_distortion,
            resolution,
            shutter_type,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_timestamps=return_timestamps,
            return_poses=return_T_sensor_world,
            allow_device_transfer=allow_device_transfer,
        )
    )
    return WorldRaysReturn(
        world_rays=world_rays,
        T_sensor_world=poses_to_matrix(pose_t, pose_r)
        if return_T_sensor_world
        else None,
        timestamps_us=timestamps if return_timestamps else None,
    )


def pixel_grid_to_world_rays_shutter_pose(
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_T_sensor_world: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldRaysReturn:
    """Generate world rays for every pixel in row-major order."""

    world_rays, timestamps, pose_t, pose_r = (
        _kernel_ops.pixel_grid_to_world_rays_shutter_pose(
            projection,
            external_distortion,
            resolution,
            shutter_type,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_timestamps=return_timestamps,
            return_poses=return_T_sensor_world,
            allow_device_transfer=allow_device_transfer,
        )
    )
    return WorldRaysReturn(
        world_rays=world_rays,
        T_sensor_world=poses_to_matrix(pose_t, pose_r)
        if return_T_sensor_world
        else None,
        timestamps_us=timestamps if return_timestamps else None,
    )


__all__ = [
    "camera_rays_to_image_points",
    "generate_image_points",
    "image_points_to_camera_rays",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
    "pixel_grid_to_world_rays_shutter_pose",
    "project_world_points_mean_pose",
    "project_world_points_shutter_pose",
]
