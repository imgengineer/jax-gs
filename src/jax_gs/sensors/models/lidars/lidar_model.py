"""Flax NNX stateful model for a structured spinning LiDAR."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from ... import functional as F
from ...functional.return_types import (
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)
from ...kernels.common.pose import DynamicPose
from ...kernels.lidars.types import (
    RowOffsetStructuredSpinningLidarProjection,
    SpinningDirection,
)
from ..common.utils import (
    compact_valid_indices,
    filter_by_validity,
)


_TWO_PI = 2.0 * jnp.pi


class LidarModel(nnx.Module):
    """Trainable angle tables plus pure-JAX spinning-LiDAR adapters."""

    def __init__(
        self,
        projection: RowOffsetStructuredSpinningLidarProjection,
        fov_eps_factor: float = 4.0,
    ) -> None:
        if projection is None:
            raise TypeError(
                "projection=None is not supported; pass a concrete "
                "RowOffsetStructuredSpinningLidarProjection"
            )
        if not isinstance(projection, RowOffsetStructuredSpinningLidarProjection):
            raise TypeError(
                f"unsupported LiDAR projection: {type(projection).__name__}"
            )
        self._row_elevations_rad = nnx.Param(
            jnp.asarray(projection.row_elevations_rad)
        )
        self._column_azimuths_rad = nnx.Param(
            jnp.asarray(projection.column_azimuths_rad)
        )
        self._row_azimuth_offsets_rad = nnx.Param(
            jnp.asarray(projection.row_azimuth_offsets_rad)
        )
        self._fov_vert_start_rad = float(projection.fov_vert_start_rad)
        self._fov_vert_span_rad = float(projection.fov_vert_span_rad)
        self._fov_horiz_start_rad = float(projection.fov_horiz_start_rad)
        self._fov_horiz_span_rad = float(projection.fov_horiz_span_rad)
        self._spinning_direction = int(projection.spinning_direction)
        self._has_row_offsets = bool(projection.has_row_offsets)
        self._fov_eps_rad = float(fov_eps_factor) * float(jnp.finfo(jnp.float32).eps)

    @property
    def projection(self) -> RowOffsetStructuredSpinningLidarProjection:
        return RowOffsetStructuredSpinningLidarProjection(
            row_elevations_rad=self._row_elevations_rad[...],
            column_azimuths_rad=self._column_azimuths_rad[...],
            row_azimuth_offsets_rad=self._row_azimuth_offsets_rad[...],
            fov_vert_start_rad=self._fov_vert_start_rad,
            fov_vert_span_rad=self._fov_vert_span_rad,
            fov_horiz_start_rad=self._fov_horiz_start_rad,
            fov_horiz_span_rad=self._fov_horiz_span_rad,
            spinning_direction=self._spinning_direction,
            has_row_offsets=self._has_row_offsets,
        )

    def sensor_rays_to_sensor_angles(
        self,
        sensor_rays: jax.Array,
        *,
        normalized: bool = True,
        return_valid_flag: bool = False,
    ) -> SensorAnglesReturn:
        rays = jnp.asarray(sensor_rays)
        if not normalized:
            squared_norm = jnp.sum(rays * rays, axis=-1, keepdims=True)
            rays = rays * jax.lax.rsqrt(
                jnp.maximum(squared_norm, jnp.asarray(1.0e-20, rays.dtype))
            )
        result = F.sensor_rays_to_sensor_angles(
            rays, allow_device_transfer=True
        )
        valid = (
            self._valid_sensor_angles(result.sensor_angles)
            if return_valid_flag
            else None
        )
        return SensorAnglesReturn(result.sensor_angles, valid)

    def sensor_angles_to_sensor_rays(
        self,
        sensor_angles: jax.Array,
        *,
        return_valid_flag: bool = False,
    ) -> SensorRayReturn:
        result = F.sensor_angles_to_sensor_rays(
            sensor_angles, allow_device_transfer=True
        )
        valid = (
            self._valid_sensor_angles(sensor_angles)
            if return_valid_flag
            else None
        )
        return SensorRayReturn(result.sensor_rays, valid)

    def elements_to_sensor_angles(
        self,
        elements: jax.Array,
        *,
        return_valid_flag: bool = False,
    ) -> SensorAnglesReturn:
        return F.elements_to_sensor_angles(
            elements,
            self.projection,
            return_valid_flag=return_valid_flag,
            allow_device_transfer=True,
        )

    def elements_to_sensor_rays(self, elements: jax.Array) -> jax.Array:
        angles = self.elements_to_sensor_angles(elements).sensor_angles
        return self.sensor_angles_to_sensor_rays(angles).sensor_rays

    def elements_to_sensor_points(
        self, elements: jax.Array, element_distances: jax.Array
    ) -> jax.Array:
        return self.elements_to_sensor_rays(elements) * jnp.asarray(
            element_distances
        )[:, None]

    def elements_to_world_rays_shutter_pose(
        self,
        elements: jax.Array | None,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        return F.generate_spinning_lidar_rays(
            self.projection,
            elements,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )

    def world_points_to_sensor_angles_shutter_pose(
        self,
        world_points: jax.Array,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        max_iterations: int = 10,
        stop_mean_relative_time_error: float = 1.0e-4,
        stop_delta_mean_relative_time_error: float = 1.0e-6,
        initial_relative_time: float = 0.5,
        return_T_sensor_world: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToSensorAnglesReturn:
        result = F.inverse_project_spinning_lidar(
            self.projection,
            world_points,
            dynamic_pose,
            max_iterations=max_iterations,
            stop_mean_relative_time_error=stop_mean_relative_time_error,
            stop_delta_mean_relative_time_error=stop_delta_mean_relative_time_error,
            initial_relative_time=initial_relative_time,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=True,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )
        valid = result.valid_flag
        if valid is None:
            raise RuntimeError("internal inverse projection omitted validity")
        return WorldPointsToSensorAnglesReturn(
            sensor_angles=filter_by_validity(
                result.sensor_angles, valid, return_all_projections
            ),
            T_sensor_world=filter_by_validity(
                result.T_sensor_world, valid, return_all_projections
            )
            if return_T_sensor_world
            else None,
            valid_flag=valid if return_valid_flag else None,
            valid_indices=compact_valid_indices(valid)
            if return_valid_indices
            else None,
            timestamps_us=filter_by_validity(
                result.timestamps_us, valid, return_all_projections
            )
            if return_timestamps
            else None,
        )

    def sensor_angles_relative_frame_times(
        self, sensor_angles: jax.Array
    ) -> jax.Array:
        projection = self.projection
        angles = jnp.asarray(sensor_angles)
        row_indices = jnp.argmin(
            jnp.abs(
                angles[:, :1] - projection.row_elevations_rad[None]
            ),
            axis=-1,
        )
        azimuth = angles[:, 1]
        if projection.has_row_offsets:
            azimuth = azimuth - projection.row_azimuth_offsets_rad[row_indices]
        differences = azimuth[:, None] - projection.column_azimuths_rad[None]
        column_indices = jnp.argmin(
            jnp.abs(self._normalize_angle(differences)), axis=-1
        )
        if self.n_columns <= 1:
            return jnp.zeros(angles.shape[:1], dtype=angles.dtype)
        return column_indices.astype(angles.dtype) / (self.n_columns - 1)

    @staticmethod
    def _normalize_angle(angle: jax.Array) -> jax.Array:
        wrapped = jnp.fmod(angle + jnp.pi, _TWO_PI)
        wrapped = jnp.where(wrapped < 0.0, wrapped + _TWO_PI, wrapped)
        return wrapped - jnp.pi

    def _relative_sensor_angles(self, sensor_angles: jax.Array) -> jax.Array:
        projection = self.projection
        angles = jnp.asarray(sensor_angles)
        relative_elevation = projection.fov_vert_start_rad - angles[:, 0]
        if projection.spinning_direction == SpinningDirection.COUNTERCLOCKWISE:
            relative_azimuth = angles[:, 1] - projection.fov_horiz_start_rad
        else:
            relative_azimuth = projection.fov_horiz_start_rad - angles[:, 1]
        relative_azimuth = jnp.mod(relative_azimuth, _TWO_PI)
        return jnp.stack((relative_elevation, relative_azimuth), axis=-1)

    def _valid_sensor_angles(self, sensor_angles: jax.Array) -> jax.Array:
        relative = self._relative_sensor_angles(sensor_angles)
        slack = 2.0 * self._fov_eps_rad
        return (
            (relative[:, 0] >= -slack)
            & (relative[:, 0] <= self._fov_vert_span_rad + slack)
            & (relative[:, 1] <= self._fov_horiz_span_rad + slack)
        )

    def valid_sensor_angles(self, sensor_angles: jax.Array) -> jax.Array:
        return self._valid_sensor_angles(sensor_angles)

    @property
    def n_rows(self) -> int:
        return int(self._row_elevations_rad[...].shape[0])

    @property
    def n_columns(self) -> int:
        return int(self._column_azimuths_rad[...].shape[0])

    @property
    def n_elements(self) -> int:
        return self.n_rows * self.n_columns

    @property
    def fov_vert(self) -> tuple[float, float]:
        return self._fov_vert_start_rad, self._fov_vert_span_rad

    @property
    def fov_horiz(self) -> tuple[float, float]:
        return self._fov_horiz_start_rad, self._fov_horiz_span_rad

    @property
    def spinning_direction(self) -> SpinningDirection:
        return SpinningDirection(self._spinning_direction)

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "LidarModel.forward() is not implemented; use its projection methods"
        )

    def __call__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "LidarModel is not callable; use its projection methods"
        )


__all__ = ["LidarModel"]
