"""Flax NNX camera state layered over the pure-JAX functional API."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from ... import functional as F
from ...functional.return_types import (
    ImagePointsReturn,
    PixelsReturn,
    WorldPointsToImagePointsReturn,
    WorldPointsToPixelsReturn,
    WorldRaysReturn,
)
from ...kernels.cameras import ops as _camera_ops
from ...kernels.cameras.types import (
    BivariateWindshieldDistortion,
    CameraProjection,
    ExternalDistortion,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ShutterType,
)
from ...kernels.common.pose import DynamicPose, Pose
from ..common.utils import (
    compact_valid_indices,
    compute_scaled_resolution,
    filter_by_validity,
)


class CameraModel(nnx.Module):
    """Trainable camera parameters with readable projection/ray adapters.

    Kernel and functional calls keep fixed, input-aligned shapes. For upstream
    compatibility, world-point methods dynamically remove invalid rows by
    default; that post-processing is intentionally outside the JIT contract.
    Set ``return_all_projections=True`` and consume ``valid_flag`` in compiled
    code. Dynamic ``valid_indices`` are likewise a host-side convenience.
    """

    def __init__(
        self,
        projection: CameraProjection,
        external_distortion: ExternalDistortion,
        resolution: tuple[int, int],
        shutter_type: ShutterType,
    ) -> None:
        if projection is None:
            raise TypeError("projection must be a concrete camera projection")
        if external_distortion is None:
            raise TypeError(
                "external_distortion must be concrete; use NoExternalDistortion()"
            )

        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.shutter_type = ShutterType(shutter_type)
        self._store_projection(projection)
        self._store_external_distortion(external_distortion)

    def _store_projection(self, projection: CameraProjection) -> None:
        self._projection_resolution = tuple(projection.resolution)
        if isinstance(projection, OpenCVPinholeProjection):
            self._projection_kind = "opencv_pinhole"
            self._focal_length = nnx.Param(jnp.asarray(projection.focal_length))
            self._principal_point = nnx.Param(jnp.asarray(projection.principal_point))
            self._radial_coeffs = nnx.Param(jnp.asarray(projection.radial_coeffs))
            self._tangential_coeffs = nnx.Param(
                jnp.asarray(projection.tangential_coeffs)
            )
            self._thin_prism_coeffs = nnx.Param(
                jnp.asarray(projection.thin_prism_coeffs)
            )
            return
        if isinstance(projection, FThetaProjection):
            self._projection_kind = "ftheta"
            self._principal_point = nnx.Param(jnp.asarray(projection.principal_point))
            self._fw_poly = nnx.Param(jnp.asarray(projection.fw_poly))
            self._bw_poly = nnx.Param(jnp.asarray(projection.bw_poly))
            self._A = nnx.Param(jnp.asarray(projection.A))
            self._reference_polynomial = int(projection.reference_polynomial)
            self._fw_poly_degree = int(projection.fw_poly_degree)
            self._bw_poly_degree = int(projection.bw_poly_degree)
            self._newton_iterations = int(projection.newton_iterations)
            self._max_angle = float(projection.max_angle)
            self._min_2d_norm = float(projection.min_2d_norm)
            return
        if isinstance(projection, OpenCVFisheyeProjection):
            self._projection_kind = "opencv_fisheye"
            self._principal_point = nnx.Param(jnp.asarray(projection.principal_point))
            self._focal_length = nnx.Param(jnp.asarray(projection.focal_length))
            self._forward_poly = nnx.Param(jnp.asarray(projection.forward_poly))
            self._approx_backward_factor = nnx.Param(
                jnp.asarray(projection.approx_backward_factor)
            )
            self._newton_iterations = int(projection.newton_iterations)
            self._max_angle = float(projection.max_angle)
            self._min_2d_norm = float(projection.min_2d_norm)
            return
        raise TypeError(f"unsupported camera projection: {type(projection).__name__}")

    def _store_external_distortion(
        self, external_distortion: ExternalDistortion
    ) -> None:
        if isinstance(external_distortion, NoExternalDistortion):
            self._external_distortion_kind = "none"
            return
        if isinstance(external_distortion, BivariateWindshieldDistortion):
            self._external_distortion_kind = "bivariate_windshield"
            self._distortion_coeffs = nnx.Param(
                jnp.asarray(external_distortion.distortion_coeffs)
            )
            self._distortion_reference_polynomial = int(
                external_distortion.reference_polynomial
            )
            self._h_poly_degree = int(external_distortion.h_poly_degree)
            self._v_poly_degree = int(external_distortion.v_poly_degree)
            return
        raise TypeError(
            f"unsupported external distortion: {type(external_distortion).__name__}"
        )

    @property
    def projection(self) -> CameraProjection:
        """Build a pure-JAX projection value from the current NNX parameters."""

        if self._projection_kind == "opencv_pinhole":
            return OpenCVPinholeProjection(
                focal_length=self._focal_length[...],
                principal_point=self._principal_point[...],
                radial_coeffs=self._radial_coeffs[...],
                tangential_coeffs=self._tangential_coeffs[...],
                thin_prism_coeffs=self._thin_prism_coeffs[...],
                resolution=self._projection_resolution,
            )
        if self._projection_kind == "ftheta":
            return FThetaProjection(
                principal_point=self._principal_point[...],
                fw_poly=self._fw_poly[...],
                bw_poly=self._bw_poly[...],
                A=self._A[...],
                resolution=self._projection_resolution,
                reference_polynomial=self._reference_polynomial,
                fw_poly_degree=self._fw_poly_degree,
                bw_poly_degree=self._bw_poly_degree,
                newton_iterations=self._newton_iterations,
                max_angle=self._max_angle,
                min_2d_norm=self._min_2d_norm,
            )
        return OpenCVFisheyeProjection(
            principal_point=self._principal_point[...],
            focal_length=self._focal_length[...],
            forward_poly=self._forward_poly[...],
            approx_backward_factor=self._approx_backward_factor[...],
            resolution=self._projection_resolution,
            newton_iterations=self._newton_iterations,
            max_angle=self._max_angle,
            min_2d_norm=self._min_2d_norm,
        )

    @property
    def external_distortion(self) -> ExternalDistortion:
        """Build a pure-JAX external-distortion value from NNX state."""

        if self._external_distortion_kind == "none":
            return NoExternalDistortion()
        return BivariateWindshieldDistortion(
            distortion_coeffs=self._distortion_coeffs[...],
            reference_polynomial=self._distortion_reference_polynomial,
            h_poly_degree=self._h_poly_degree,
            v_poly_degree=self._v_poly_degree,
        )

    @staticmethod
    def _build_image_points_return(
        result: WorldPointsToImagePointsReturn,
        *,
        return_T_sensor_world: bool,
        return_valid_flag: bool,
        return_valid_indices: bool,
        return_timestamps: bool,
        return_all_projections: bool,
    ) -> WorldPointsToImagePointsReturn:
        valid = result.valid_flag
        if valid is None:
            raise RuntimeError("internal projection result omitted its validity mask")
        return WorldPointsToImagePointsReturn(
            image_points=filter_by_validity(
                result.image_points, valid, return_all_projections
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

    def world_points_to_image_points_static_pose(
        self,
        world_points: jax.Array,
        pose: Pose,
        *,
        timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToImagePointsReturn:
        result = F.project_world_points_mean_pose(
            world_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            DynamicPose.from_static_pose(pose),
            start_timestamp_us=timestamp_us,
            end_timestamp_us=timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=True,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )
        return self._build_image_points_return(
            result,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=return_valid_flag,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            return_all_projections=return_all_projections,
        )

    def world_points_to_image_points_mean_pose(
        self,
        world_points: jax.Array,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToImagePointsReturn:
        result = F.project_world_points_mean_pose(
            world_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=True,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )
        return self._build_image_points_return(
            result,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=return_valid_flag,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            return_all_projections=return_all_projections,
        )

    def world_points_to_image_points_shutter_pose(
        self,
        world_points: jax.Array,
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
        return_all_projections: bool = False,
    ) -> WorldPointsToImagePointsReturn:
        result = F.project_world_points_shutter_pose(
            world_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            self.shutter_type,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            max_iterations=max_iterations,
            stop_mean_error_px=stop_mean_error_px,
            stop_delta_mean_error_px=stop_delta_mean_error_px,
            initial_relative_time=initial_relative_time,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=True,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )
        return self._build_image_points_return(
            result,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=return_valid_flag,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            return_all_projections=return_all_projections,
        )

    def _points_to_pixels_return(
        self, result: WorldPointsToImagePointsReturn
    ) -> WorldPointsToPixelsReturn:
        return WorldPointsToPixelsReturn(
            pixels=self.image_points_to_pixels(result.image_points),
            T_sensor_world=result.T_sensor_world,
            valid_flag=result.valid_flag,
            valid_indices=result.valid_indices,
            timestamps_us=result.timestamps_us,
        )

    def world_points_to_pixels_static_pose(
        self,
        world_points: jax.Array,
        pose: Pose,
        *,
        timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToPixelsReturn:
        return self._points_to_pixels_return(
            self.world_points_to_image_points_static_pose(
                world_points,
                pose,
                timestamp_us=timestamp_us,
                return_T_sensor_world=return_T_sensor_world,
                return_valid_flag=return_valid_flag,
                return_valid_indices=return_valid_indices,
                return_timestamps=return_timestamps,
                return_all_projections=return_all_projections,
            )
        )

    def world_points_to_pixels_mean_pose(
        self,
        world_points: jax.Array,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToPixelsReturn:
        return self._points_to_pixels_return(
            self.world_points_to_image_points_mean_pose(
                world_points,
                dynamic_pose,
                start_timestamp_us=start_timestamp_us,
                end_timestamp_us=end_timestamp_us,
                return_T_sensor_world=return_T_sensor_world,
                return_valid_flag=return_valid_flag,
                return_valid_indices=return_valid_indices,
                return_timestamps=return_timestamps,
                return_all_projections=return_all_projections,
            )
        )

    def world_points_to_pixels_shutter_pose(
        self,
        world_points: jax.Array,
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
        return_all_projections: bool = False,
    ) -> WorldPointsToPixelsReturn:
        return self._points_to_pixels_return(
            self.world_points_to_image_points_shutter_pose(
                world_points,
                dynamic_pose,
                start_timestamp_us=start_timestamp_us,
                end_timestamp_us=end_timestamp_us,
                max_iterations=max_iterations,
                stop_mean_error_px=stop_mean_error_px,
                stop_delta_mean_error_px=stop_delta_mean_error_px,
                initial_relative_time=initial_relative_time,
                return_T_sensor_world=return_T_sensor_world,
                return_valid_flag=return_valid_flag,
                return_valid_indices=return_valid_indices,
                return_timestamps=return_timestamps,
                return_all_projections=return_all_projections,
            )
        )

    def image_points_to_world_rays_static_pose(
        self,
        image_points: jax.Array,
        pose: Pose,
        *,
        timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        return F.image_points_to_world_rays_static_pose(
            image_points,
            self.projection,
            self.external_distortion,
            pose,
            timestamp_us=timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )

    def image_points_to_world_rays_mean_pose(
        self,
        image_points: jax.Array,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        points = jnp.asarray(image_points)
        pose = _camera_ops.mean_pose_to_static_pose(
            dynamic_pose, dtype=points.dtype
        )
        timestamp = None
        if start_timestamp_us is not None and end_timestamp_us is not None:
            timestamp = (start_timestamp_us + end_timestamp_us) // 2
        return self.image_points_to_world_rays_static_pose(
            points,
            pose,
            timestamp_us=timestamp,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
        )

    def image_points_to_world_rays_shutter_pose(
        self,
        image_points: jax.Array,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        return F.image_points_to_world_rays_shutter_pose(
            image_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            self.shutter_type,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )

    def pixels_to_world_rays_static_pose(
        self,
        pixels: jax.Array,
        pose: Pose,
        *,
        timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        return self.image_points_to_world_rays_static_pose(
            self.pixels_to_image_points(pixels),
            pose,
            timestamp_us=timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
        )

    def pixels_to_world_rays_mean_pose(
        self,
        pixels: jax.Array,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        return self.image_points_to_world_rays_mean_pose(
            self.pixels_to_image_points(pixels),
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
        )

    def pixels_to_world_rays_shutter_pose(
        self,
        pixels: jax.Array,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        return self.image_points_to_world_rays_shutter_pose(
            self.pixels_to_image_points(pixels),
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
        )

    def camera_rays_to_image_points(
        self,
        camera_rays: jax.Array,
        *,
        return_jacobians: bool = False,
    ) -> ImagePointsReturn:
        projection = self.projection
        distortion = self.external_distortion
        result = F.camera_rays_to_image_points(
            camera_rays,
            projection,
            distortion,
            allow_device_transfer=True,
        )
        if not return_jacobians:
            return result

        def project_one(ray: jax.Array) -> jax.Array:
            return F.camera_rays_to_image_points(
                ray[None], projection, distortion
            ).image_points[0]

        jacobians = jax.vmap(jax.jacrev(project_one))(jnp.asarray(camera_rays))
        return ImagePointsReturn(
            image_points=jax.lax.stop_gradient(result.image_points),
            valid_flag=result.valid_flag,
            jacobians=jacobians,
        )

    def camera_rays_to_pixels(self, camera_rays: jax.Array) -> PixelsReturn:
        result = self.camera_rays_to_image_points(camera_rays)
        return PixelsReturn(
            pixels=self.image_points_to_pixels(result.image_points),
            valid_flag=result.valid_flag,
        )

    def image_points_to_camera_rays(self, image_points: jax.Array) -> jax.Array:
        return F.image_points_to_camera_rays(
            image_points,
            self.projection,
            self.external_distortion,
            allow_device_transfer=True,
        )

    def pixels_to_camera_rays(self, pixels: jax.Array) -> jax.Array:
        return self.image_points_to_camera_rays(self.pixels_to_image_points(pixels))

    def pixels_to_image_points(self, pixels: jax.Array) -> jax.Array:
        """Convert pixel indices to continuous image coordinates (pixel centers)."""

        return jnp.asarray(pixels, dtype=jnp.float32) + 0.5

    def image_points_to_pixels(self, image_points: jax.Array) -> jax.Array:
        """Convert continuous image coordinates to integer pixel indices."""

        return jnp.floor(image_points).astype(jnp.int32)

    def image_points_relative_frame_times(
        self, image_points: jax.Array
    ) -> jax.Array:
        return _camera_ops.relative_frame_times(
            image_points, self.resolution, self.shutter_type
        )

    def transform(
        self,
        image_domain_scale: float | tuple[float, float],
        image_domain_offset: tuple[float, float] = (0.0, 0.0),
        new_resolution: tuple[int, int] | None = None,
    ) -> "CameraModel":
        if isinstance(image_domain_scale, tuple):
            scale_x, scale_y = image_domain_scale
        else:
            scale_x = scale_y = image_domain_scale
        model_resolution = compute_scaled_resolution(
            self.resolution, image_domain_scale, new_resolution
        )
        projection = self.projection.transform(
            (scale_x, scale_y), image_domain_offset, model_resolution
        )
        return CameraModel(
            projection=projection,
            external_distortion=self.external_distortion,
            resolution=model_resolution,
            shutter_type=self.shutter_type,
        )


__all__ = ["CameraModel"]
