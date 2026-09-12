"""Readable pure-JAX spinning-LiDAR projection operations."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ....math import quat_to_rotmat
from ..common.pose import DynamicPose
from ..common.pose_interp import (
    interpolate_dynamic_pose,
    unpack_dynamic_pose_components,
)
from .types import (
    RowOffsetStructuredSpinningLidarProjection,
    SpinningDirection,
)

_TWO_PI = 2.0 * math.pi
_MAX_INVERSE_ITERATIONS = 32


def _matrix(value, columns: int, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {array.shape}")
    return array


def _check_projection(projection: object) -> None:
    if not isinstance(projection, RowOffsetStructuredSpinningLidarProjection):
        raise TypeError(f"Unsupported LiDAR projection: {type(projection).__name__}")


def _normalize_angle(angle: jax.Array) -> jax.Array:
    pi = jnp.asarray(math.pi, dtype=angle.dtype)
    two_pi = jnp.asarray(_TWO_PI, dtype=angle.dtype)
    wrapped = jnp.fmod(angle + pi, two_pi)
    wrapped = jnp.where(wrapped < 0.0, wrapped + two_pi, wrapped)
    return wrapped - pi


def _timestamp_bounds(
    start_timestamp_us: int | None, end_timestamp_us: int | None
) -> tuple[int, int]:
    if start_timestamp_us is None and end_timestamp_us is None:
        return 0, 0
    if start_timestamp_us is None or end_timestamp_us is None:
        raise ValueError(
            "start_timestamp_us and end_timestamp_us must be provided together"
        )
    return int(start_timestamp_us), int(end_timestamp_us)


def _timestamps(
    relative_time: jax.Array, start_timestamp_us: int, end_timestamp_us: int
) -> jax.Array:
    timestamp_dtype = jnp.int64 if jax.config.x64_enabled else jnp.int32
    start = jnp.asarray(start_timestamp_us, dtype=relative_time.dtype)
    duration = jnp.asarray(
        end_timestamp_us - start_timestamp_us, dtype=relative_time.dtype
    )
    return (start + relative_time * duration).astype(timestamp_dtype)


def _safe_normalize(vector: jax.Array) -> jax.Array:
    floor = 1.0e-40 if vector.dtype == jnp.float64 else 1.0e-20
    inverse_norm = jax.lax.rsqrt(
        jnp.maximum(
            jnp.sum(vector * vector, axis=-1, keepdims=True),
            jnp.asarray(floor, dtype=vector.dtype),
        )
    )
    return vector * inverse_norm


def sensor_rays_to_sensor_angles(
    sensor_rays: jax.Array,
    *,
    allow_device_transfer: bool = False,
) -> jax.Array:
    """Convert +X-forward sensor rays to ``[elevation, azimuth]``."""

    del allow_device_transfer
    rays = _matrix(sensor_rays, 3, "sensor_rays")
    horizontal_norm = jnp.hypot(rays[:, 0], rays[:, 1])
    return jnp.stack(
        (
            jnp.arctan2(rays[:, 2], horizontal_norm),
            jnp.arctan2(rays[:, 1], rays[:, 0]),
        ),
        axis=-1,
    )


def sensor_angles_to_sensor_rays(
    sensor_angles: jax.Array,
    *,
    allow_device_transfer: bool = False,
) -> jax.Array:
    """Convert ``[elevation, azimuth]`` to unit +X-forward rays."""

    del allow_device_transfer
    angles = _matrix(sensor_angles, 2, "sensor_angles")
    elevation, azimuth = angles[:, 0], angles[:, 1]
    cos_elevation = jnp.cos(elevation)
    return jnp.stack(
        (
            cos_elevation * jnp.cos(azimuth),
            cos_elevation * jnp.sin(azimuth),
            jnp.sin(elevation),
        ),
        axis=-1,
    )


def elements_to_sensor_angles(
    elements: jax.Array,
    projection: RowOffsetStructuredSpinningLidarProjection,
    *,
    allow_device_transfer: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Look up element angles, returning zeros for out-of-bounds elements."""

    del allow_device_transfer
    _check_projection(projection)
    elements = _matrix(elements, 2, "elements").astype(jnp.int32)
    rows, columns = elements[:, 0], elements[:, 1]
    n_rows = projection.row_elevations_rad.shape[0]
    n_columns = projection.column_azimuths_rad.shape[0]
    valid = (rows >= 0) & (rows < n_rows) & (columns >= 0) & (columns < n_columns)
    safe_rows = jnp.clip(rows, 0, n_rows - 1)
    safe_columns = jnp.clip(columns, 0, n_columns - 1)
    row_elevations = jnp.asarray(projection.row_elevations_rad, dtype=jnp.float32)
    column_azimuths = jnp.asarray(projection.column_azimuths_rad, dtype=jnp.float32)
    elevation = row_elevations[safe_rows]
    azimuth = column_azimuths[safe_columns]
    if projection.has_row_offsets:
        offsets = jnp.asarray(projection.row_azimuth_offsets_rad, dtype=jnp.float32)
        azimuth = _normalize_angle(azimuth + offsets[safe_rows])
    angles = jnp.stack((elevation, azimuth), axis=-1)
    return jnp.where(valid[:, None], angles, 0.0), valid


def _full_element_grid(projection) -> jax.Array:
    n_rows = projection.row_elevations_rad.shape[0]
    n_columns = projection.column_azimuths_rad.shape[0]
    flat = jnp.arange(n_rows * n_columns, dtype=jnp.int32)
    return jnp.stack((flat // n_columns, flat % n_columns), axis=-1)


def generate_spinning_lidar_rays(
    projection: RowOffsetStructuredSpinningLidarProjection,
    elements: jax.Array | None,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array | None]:
    """Generate rolling-shutter world rays in row-major element order."""

    del allow_device_transfer
    _check_projection(projection)
    start, end = _timestamp_bounds(start_timestamp_us, end_timestamp_us)
    if elements is None:
        elements = _full_element_grid(projection)
    else:
        elements = _matrix(elements, 2, "elements").astype(jnp.int32)
    angles, valid = elements_to_sensor_angles(elements, projection)
    columns = elements[:, 1]
    n_columns = projection.column_azimuths_rad.shape[0]
    relative_time = (
        columns.astype(jnp.float32) / (n_columns - 1)
        if n_columns > 1
        else jnp.zeros(columns.shape, dtype=jnp.float32)
    )
    alpha = jnp.clip(relative_time, 0.0, 1.0)
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, dtype=jnp.float32
    )
    pose_t, pose_r = interpolate_dynamic_pose(start_t, start_r, end_t, end_r, alpha)
    sensor_rays = sensor_angles_to_sensor_rays(angles)
    directions = jnp.einsum("nij,nj->ni", quat_to_rotmat(pose_r), sensor_rays)
    world_rays = jnp.concatenate((pose_t, directions), axis=-1)
    world_rays = jnp.where(valid[:, None], world_rays, 0.0)
    valid_pose_t = jnp.where(valid[:, None], pose_t, 0.0)
    identity = jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=pose_r.dtype)
    valid_pose_r = jnp.where(valid[:, None], pose_r, identity)
    timestamps = jnp.where(valid, _timestamps(relative_time, start, end), start)
    return (
        world_rays,
        jax.lax.stop_gradient(timestamps) if return_timestamps else None,
        jax.lax.stop_gradient(valid_pose_t) if return_poses else None,
        jax.lax.stop_gradient(valid_pose_r) if return_poses else None,
    )


def _sensor_angles_to_relative_time(
    sensor_angles: jax.Array,
    projection: RowOffsetStructuredSpinningLidarProjection,
) -> jax.Array:
    dtype = sensor_angles.dtype
    row_elevations = jnp.asarray(projection.row_elevations_rad, dtype=dtype)
    column_azimuths = jnp.asarray(projection.column_azimuths_rad, dtype=dtype)
    row_indices = jnp.argmin(
        jnp.abs(sensor_angles[:, :1] - row_elevations[None]), axis=-1
    )
    azimuth = sensor_angles[:, 1]
    if projection.has_row_offsets:
        offsets = jnp.asarray(projection.row_azimuth_offsets_rad, dtype=dtype)
        azimuth = azimuth - offsets[row_indices]
    differences = azimuth[:, None] - column_azimuths[None]
    two_pi = jnp.asarray(_TWO_PI, dtype=dtype)
    wrapped = differences - two_pi * jnp.rint(differences / two_pi)
    column_indices = jnp.argmin(jnp.abs(wrapped), axis=-1)
    if column_azimuths.shape[0] <= 1:
        return jnp.zeros(sensor_angles.shape[:1], dtype=dtype)
    return column_indices.astype(dtype) / (column_azimuths.shape[0] - 1)


def _angles_in_fov(
    sensor_angles: jax.Array,
    projection: RowOffsetStructuredSpinningLidarProjection,
) -> jax.Array:
    elevation, azimuth = sensor_angles[:, 0], sensor_angles[:, 1]
    vertical = (elevation <= projection.fov_vert_start_rad) & (
        elevation >= projection.fov_vert_start_rad - projection.fov_vert_span_rad
    )
    if projection.spinning_direction == SpinningDirection.COUNTERCLOCKWISE:
        relative_azimuth = azimuth - projection.fov_horiz_start_rad
    else:
        relative_azimuth = projection.fov_horiz_start_rad - azimuth
    relative_azimuth = jnp.mod(relative_azimuth, _TWO_PI)
    return vertical & (relative_azimuth <= projection.fov_horiz_span_rad)


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
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
    """Inverse-project world points with a frozen-time fixed-point solve."""

    del allow_device_transfer
    _check_projection(projection)
    points = _matrix(world_points, 3, "world_points")
    max_iterations = int(max_iterations)
    if not 1 <= max_iterations <= _MAX_INVERSE_ITERATIONS:
        raise ValueError("max_iterations must be in [1, 32]")
    for value, name in (
        (stop_mean_relative_time_error, "stop_mean_relative_time_error"),
        (stop_delta_mean_relative_time_error, "stop_delta_mean_relative_time_error"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (
        not math.isfinite(initial_relative_time)
        or not 0.0 <= initial_relative_time <= 1.0
    ):
        raise ValueError("initial_relative_time must be finite and in [0, 1]")
    start, end = _timestamp_bounds(start_timestamp_us, end_timestamp_us)
    dtype = points.dtype
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, dtype=dtype
    )
    count = points.shape[0]
    relative_time = jnp.full((count,), initial_relative_time, dtype=dtype)
    previous_time_difference = jnp.ones((count,), dtype=dtype)
    result_angles = jnp.zeros((count, 2), dtype=dtype)
    result_valid = jnp.zeros((count,), dtype=jnp.bool_)
    pose_t = jnp.zeros((count, 3), dtype=dtype)
    pose_r = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=dtype), (count, 4)
    )
    active = jnp.ones((count,), dtype=jnp.bool_)

    for iteration in range(max_iterations):
        alpha = jax.lax.stop_gradient(jnp.clip(relative_time, 0.0, 1.0))
        candidate_t, candidate_r = interpolate_dynamic_pose(
            start_t, start_r, end_t, end_r, alpha
        )
        rotation = quat_to_rotmat(candidate_r)
        sensor_points = jnp.einsum(
            "nij,nj->ni",
            jnp.swapaxes(rotation, -1, -2),
            points - candidate_t,
        )
        sensor_rays = _safe_normalize(sensor_points)
        candidate_angles = sensor_rays_to_sensor_angles(sensor_rays)
        candidate_valid = _angles_in_fov(candidate_angles, projection)

        pose_t = jnp.where(active[:, None], candidate_t, pose_t)
        pose_r = jnp.where(active[:, None], candidate_r, pose_r)
        successful = active & candidate_valid
        result_angles = jnp.where(successful[:, None], candidate_angles, result_angles)
        result_valid = jnp.where(active, candidate_valid, result_valid)

        new_relative_time = jax.lax.stop_gradient(
            _sensor_angles_to_relative_time(candidate_angles, projection)
        )
        time_difference = jnp.abs(new_relative_time - relative_time)
        mean_converged = time_difference < stop_mean_relative_time_error
        delta_converged = (iteration > 0) & (
            jnp.abs(time_difference - previous_time_difference)
            < stop_delta_mean_relative_time_error
        )
        continuing = successful & ~mean_converged & ~delta_converged
        previous_time_difference = jnp.where(
            continuing, time_difference, previous_time_difference
        )
        relative_time = jnp.where(continuing, new_relative_time, relative_time)
        active = continuing

    differentiable_angles = jnp.where(
        result_valid[:, None], result_angles, jax.lax.stop_gradient(result_angles)
    )
    timestamps = _timestamps(relative_time, start, end)
    return (
        differentiable_angles,
        result_valid,
        jax.lax.stop_gradient(timestamps) if return_timestamps else None,
        jax.lax.stop_gradient(pose_t) if return_poses else None,
        jax.lax.stop_gradient(pose_r) if return_poses else None,
    )


__all__ = [
    "elements_to_sensor_angles",
    "generate_spinning_lidar_rays",
    "inverse_project_spinning_lidar",
    "sensor_angles_to_sensor_rays",
    "sensor_rays_to_sensor_angles",
]
