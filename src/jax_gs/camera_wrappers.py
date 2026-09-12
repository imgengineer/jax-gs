"""Readable pure-JAX equivalents of current-main's root camera wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .external_distortion import BivariateWindshieldModelParameters
from .lidar import (
    LegacyLidarModel,
    RowOffsetStructuredSpinningLidarModelParametersExt,
)
from .math import quat_to_rotmat
from .rendering_types import CameraModel
from .three_dgut import (
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    RollingShutterType,
    _normalize_rolling_shutter,
    _world_rays_from_pixels,
    project_camera_points,
    project_world_points,
    shutter_relative_frame_time,
    unproject_image_points,
)


def _array_ending_in(value, size: int, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim < 1 or array.shape[-1] != size:
        raise ValueError(f"{name} must end in {size} values")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating-point dtype")
    return array


def _viewmat_from_pose(pose: jax.Array) -> jax.Array:
    pose = _array_ending_in(pose, 7, "pose")
    rotation = quat_to_rotmat(pose[..., 3:])
    viewmat = jnp.broadcast_to(jnp.eye(4, dtype=pose.dtype), pose.shape[:-1] + (4, 4))
    viewmat = viewmat.at[..., :3, :3].set(rotation)
    return viewmat.at[..., :3, 3].set(pose[..., :3])


@dataclass(frozen=True)
class RootCameraModel:
    """Type-erased root camera model with gsplat-compatible methods."""

    width: int
    height: int
    camera_model: CameraModel
    Ks: jax.Array
    radial_coeffs: jax.Array | None
    tangential_coeffs: jax.Array | None
    thin_prism_coeffs: jax.Array | None
    ftheta_coeffs: FThetaCameraDistortionParameters | None
    external_distortion_coeffs: BivariateWindshieldModelParameters | None
    rs_type: RollingShutterType
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt | None

    @property
    def principal_points(self) -> jax.Array:
        principal = self.Ks[..., :2, 2]
        return principal + 0.5 if self.camera_model == "ftheta" else principal

    @property
    def focal_lengths(self) -> jax.Array:
        if self.camera_model == "ftheta":
            assert self.ftheta_coeffs is not None
            if (
                self.ftheta_coeffs.reference_poly
                == FThetaPolynomialType.PIXELDIST_TO_ANGLE
            ):
                focal = 1.0 / self.ftheta_coeffs.pixeldist_to_angle_poly[1]
            else:
                focal = self.ftheta_coeffs.angle_to_pixeldist_poly[1]
            return jnp.full_like(self.Ks[..., :2, 2], focal)
        return jnp.stack((self.Ks[..., 0, 0], self.Ks[..., 1, 1]), axis=-1)

    def camera_ray_to_image_point(
        self, camera_ray: jax.Array, margin_factor: float = 0.0
    ) -> tuple[jax.Array, jax.Array]:
        return project_camera_points(
            camera_ray,
            self.Ks,
            self.width,
            self.height,
            camera_model=self.camera_model,
            margin_factor=margin_factor,
            radial_coeffs=self.radial_coeffs,
            tangential_coeffs=self.tangential_coeffs,
            thin_prism_coeffs=self.thin_prism_coeffs,
            ftheta_coeffs=self.ftheta_coeffs,
            lidar_coeffs=self.lidar_coeffs,
            external_distortion_coeffs=self.external_distortion_coeffs,
        )

    def image_point_to_camera_ray(
        self, image_points: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        _, directions, valid = unproject_image_points(
            image_points,
            self.Ks,
            camera_model=self.camera_model,
            radial_coeffs=self.radial_coeffs,
            tangential_coeffs=self.tangential_coeffs,
            thin_prism_coeffs=self.thin_prism_coeffs,
            ftheta_coeffs=self.ftheta_coeffs,
            lidar_coeffs=self.lidar_coeffs,
            external_distortion_coeffs=self.external_distortion_coeffs,
        )
        return directions, valid

    def shutter_relative_frame_time(self, image_points: jax.Array) -> jax.Array:
        if self.lidar_coeffs is not None:
            return LegacyLidarModel(self.lidar_coeffs).shutter_relative_frame_time(
                image_points
            )
        return shutter_relative_frame_time(
            image_points, self.width, self.height, self.rs_type
        )

    def image_point_to_world_ray_shutter_pose(
        self,
        image_points: jax.Array,
        pose_start: jax.Array,
        pose_end: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _world_rays_from_pixels(
            image_points,
            _viewmat_from_pose(pose_start),
            self.Ks,
            self.width,
            self.height,
            camera_model=self.camera_model,
            radial_coeffs=self.radial_coeffs,
            tangential_coeffs=self.tangential_coeffs,
            thin_prism_coeffs=self.thin_prism_coeffs,
            ftheta_coeffs=self.ftheta_coeffs,
            rolling_shutter=self.rs_type,
            viewmat_rs=_viewmat_from_pose(pose_end),
            lidar_coeffs=self.lidar_coeffs,
            external_distortion_coeffs=self.external_distortion_coeffs,
        )

    def world_point_to_image_point_shutter_pose(
        self,
        world_points: jax.Array,
        pose_start: jax.Array,
        pose_end: jax.Array,
        margin_factor: float = 0.0,
    ) -> tuple[jax.Array, jax.Array]:
        return project_world_points(
            world_points,
            _viewmat_from_pose(pose_start),
            self.Ks,
            self.width,
            self.height,
            camera_model=self.camera_model,
            radial_coeffs=self.radial_coeffs,
            tangential_coeffs=self.tangential_coeffs,
            thin_prism_coeffs=self.thin_prism_coeffs,
            ftheta_coeffs=self.ftheta_coeffs,
            lidar_coeffs=self.lidar_coeffs,
            external_distortion_coeffs=self.external_distortion_coeffs,
            rolling_shutter=self.rs_type,
            viewmats_rs=_viewmat_from_pose(pose_end),
            margin_factor=margin_factor,
        )


def create_camera_model(
    camera_model: CameraModel,
    width: int | None = None,
    height: int | None = None,
    principal_points: jax.Array | None = None,
    focal_lengths: jax.Array | None = None,
    radial_coeffs: jax.Array | None = None,
    tangential_coeffs: jax.Array | None = None,
    thin_prism_coeffs: jax.Array | None = None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None = None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None = None,
    rs_type: RollingShutterType = RollingShutterType.GLOBAL,
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt | None = None,
) -> RootCameraModel:
    """Create a root camera wrapper without a compiled CUDA custom class."""

    shutter = _normalize_rolling_shutter(rs_type)
    if camera_model == "lidar":
        if not isinstance(
            lidar_coeffs, RowOffsetStructuredSpinningLidarModelParametersExt
        ):
            raise ValueError("camera_model='lidar' requires lidar_coeffs")
        if external_distortion_coeffs is not None:
            raise ValueError("LiDAR cameras do not support external distortion")
        return RootCameraModel(
            width=lidar_coeffs.n_columns,
            height=lidar_coeffs.n_rows,
            camera_model=camera_model,
            Ks=jnp.eye(3, dtype=lidar_coeffs.row_elevations_rad.dtype),
            radial_coeffs=None,
            tangential_coeffs=None,
            thin_prism_coeffs=None,
            ftheta_coeffs=None,
            external_distortion_coeffs=None,
            rs_type=shutter,
            lidar_coeffs=lidar_coeffs,
        )
    if width is None or height is None:
        raise ValueError("width and height are required for non-LiDAR cameras")
    if int(width) < 1 or int(height) < 1:
        raise ValueError("width and height must be positive")
    if principal_points is None:
        raise ValueError("principal_points is required for non-LiDAR cameras")
    principal = _array_ending_in(principal_points, 2, "principal_points")
    batch_shape = principal.shape[:-1]

    if camera_model == "ftheta":
        if ftheta_coeffs is None:
            raise ValueError("ftheta requires ftheta_coeffs")
        if focal_lengths is not None:
            raise ValueError("ftheta does not support focal_lengths")
        focal = jnp.ones(batch_shape + (2,), dtype=principal.dtype)
    else:
        if ftheta_coeffs is not None:
            raise ValueError(f"{camera_model} does not support ftheta_coeffs")
        if focal_lengths is None:
            raise ValueError(f"focal_lengths is required for {camera_model}")
        focal = _array_ending_in(focal_lengths, 2, "focal_lengths")
        focal = jnp.broadcast_to(focal, batch_shape + (2,))

    Ks = jnp.broadcast_to(jnp.eye(3, dtype=principal.dtype), batch_shape + (3, 3))
    Ks = Ks.at[..., 0, 0].set(focal[..., 0])
    Ks = Ks.at[..., 1, 1].set(focal[..., 1])
    Ks = Ks.at[..., :2, 2].set(principal)

    # Reuse the projection boundary for model-specific coefficient validation.
    probe = jnp.zeros(batch_shape + (1, 3), dtype=principal.dtype).at[..., 2].set(1.0)
    project_camera_points(
        probe,
        Ks,
        int(width),
        int(height),
        camera_model=camera_model,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        ftheta_coeffs=ftheta_coeffs,
        external_distortion_coeffs=external_distortion_coeffs,
    )
    return RootCameraModel(
        width=int(width),
        height=int(height),
        camera_model=camera_model,
        Ks=Ks,
        radial_coeffs=None if radial_coeffs is None else jnp.asarray(radial_coeffs),
        tangential_coeffs=(
            None if tangential_coeffs is None else jnp.asarray(tangential_coeffs)
        ),
        thin_prism_coeffs=(
            None if thin_prism_coeffs is None else jnp.asarray(thin_prism_coeffs)
        ),
        ftheta_coeffs=ftheta_coeffs,
        external_distortion_coeffs=external_distortion_coeffs,
        rs_type=shutter,
        lidar_coeffs=None,
    )


__all__ = ["RootCameraModel", "create_camera_model"]
