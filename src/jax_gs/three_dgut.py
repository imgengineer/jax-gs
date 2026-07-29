"""Static-shape 3DGUT projection for distorted and rolling-shutter cameras.

The CUDA 3DGUT path in gsplat projects seven unscented-transform (UT) sigma
points for every Gaussian.  This module keeps that representation dense and
fixed-size so an ``active_mask`` can change without retracing a jitted step.
For large fixed-capacity buffers the projection is evaluated in static chunks,
which bounds the peak size of the sigma-point intermediates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any

import jax
import jax.numpy as jnp

from .external_distortion import (
    BivariateWindshieldModelParameters,
    distort_camera_rays,
    validate_external_distortion,
)
from .math import quat_to_rotmat, safe_normalize
from .lidar import (
    generate_lidar_image_points,
    LegacyLidarModel,
    RowOffsetStructuredSpinningLidarModelParametersExt,
)
from .rendering_types import CameraModel, RendererConfig, resolve_renderer_config


class RollingShutterType(IntEnum):
    """Readout directions used by gsplat's rolling-shutter cameras."""

    ROLLING_TOP_TO_BOTTOM = 0
    ROLLING_LEFT_TO_RIGHT = 1
    ROLLING_BOTTOM_TO_TOP = 2
    ROLLING_RIGHT_TO_LEFT = 3
    GLOBAL = 4


class FThetaPolynomialType(IntEnum):
    """Which FTheta calibration polynomial is the fitted reference."""

    PIXELDIST_TO_ANGLE = 0
    ANGLE_TO_PIXELDIST = 1


@dataclass(frozen=True)
class UnscentedTransformParameters:
    """Parameters for the seven-point scaled unscented transform."""

    alpha: float = 0.1
    beta: float = 2.0
    kappa: float = 0.0
    in_image_margin_factor: float = 0.1
    require_all_sigma_points_valid: bool = True

    def __post_init__(self) -> None:
        scaled_dimension = self.alpha * self.alpha * (3.0 + self.kappa)
        if scaled_dimension <= 0.0:
            raise ValueError(
                "alpha**2 * (3 + kappa) must be positive for the UT"
            )
        if self.in_image_margin_factor < 0.0:
            raise ValueError("in_image_margin_factor must be non-negative")


@dataclass(frozen=True)
class FThetaCameraDistortionParameters:
    """Shared gsplat FTheta calibration parameters.

    Polynomial coefficients are in ascending order and both polynomials have
    degree at most five.  ``linear_cde`` represents ``[[c, d], [e, 1]]``.
    """

    reference_poly: FThetaPolynomialType
    pixeldist_to_angle_poly: tuple[float, float, float, float, float, float]
    angle_to_pixeldist_poly: tuple[float, float, float, float, float, float]
    max_angle: float
    linear_cde: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference_poly", FThetaPolynomialType(self.reference_poly)
        )
        object.__setattr__(
            self,
            "pixeldist_to_angle_poly",
            tuple(float(value) for value in self.pixeldist_to_angle_poly),
        )
        object.__setattr__(
            self,
            "angle_to_pixeldist_poly",
            tuple(float(value) for value in self.angle_to_pixeldist_poly),
        )
        object.__setattr__(
            self, "linear_cde", tuple(float(value) for value in self.linear_cde)
        )
        if len(self.pixeldist_to_angle_poly) != 6:
            raise ValueError("pixeldist_to_angle_poly must contain 6 coefficients")
        if len(self.angle_to_pixeldist_poly) != 6:
            raise ValueError("angle_to_pixeldist_poly must contain 6 coefficients")
        if len(self.linear_cde) != 3:
            raise ValueError("linear_cde must contain 3 coefficients")
        if self.max_angle <= 0.0:
            raise ValueError("max_angle must be positive")


def _safe_denominator(value: jax.Array, eps: float = 1.0e-8) -> jax.Array:
    eps_value = jnp.asarray(eps, dtype=value.dtype)
    sign = jnp.where(value < 0.0, -jnp.ones_like(value), jnp.ones_like(value))
    return jnp.where(jnp.abs(value) < eps_value, sign * eps_value, value)


def _normalize_rolling_shutter(
    rolling_shutter: RollingShutterType | int | str,
) -> RollingShutterType:
    if isinstance(rolling_shutter, str):
        key = rolling_shutter.upper()
        if not key.startswith("ROLLING_") and key != "GLOBAL":
            key = f"ROLLING_{key}"
        try:
            return RollingShutterType[key]
        except KeyError as error:
            raise ValueError(
                f"Unsupported rolling shutter type: {rolling_shutter}"
            ) from error
    try:
        return RollingShutterType(rolling_shutter)
    except ValueError as error:
        raise ValueError(
            f"Unsupported rolling shutter type: {rolling_shutter}"
        ) from error


def compute_ut_weights(
    ut_params: UnscentedTransformParameters | None = None,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
    """Return the seven mean and covariance weights for a 3D Gaussian."""

    if ut_params is None:
        ut_params = UnscentedTransformParameters()
    dimension = 3.0
    scaled_dimension = ut_params.alpha**2 * (dimension + ut_params.kappa)
    lambda_value = scaled_dimension - dimension
    center_mean = lambda_value / scaled_dimension
    center_cov = center_mean + 1.0 - ut_params.alpha**2 + ut_params.beta
    other = 1.0 / (2.0 * scaled_dimension)
    weights_mean = jnp.asarray([center_mean] + [other] * 6, dtype=dtype)
    weights_cov = jnp.asarray([center_cov] + [other] * 6, dtype=dtype)
    return weights_mean, weights_cov


def world_gaussian_sigma_points(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    ut_params: UnscentedTransformParameters | None = None,
) -> jax.Array:
    """Generate ``[..., N, 7, 3]`` world-space UT sigma points."""

    if ut_params is None:
        ut_params = UnscentedTransformParameters()
    means = jnp.asarray(means)
    quats = jnp.asarray(quats)
    scales = jnp.asarray(scales)
    if means.shape[-1] != 3:
        raise ValueError("means must have shape [..., N, 3]")
    if quats.shape != means.shape[:-1] + (4,):
        raise ValueError("quats must have shape [..., N, 4]")
    if scales.shape != means.shape:
        raise ValueError("scales must have shape [..., N, 3]")

    rotation = quat_to_rotmat(quats)
    scaled_dimension = ut_params.alpha**2 * (3.0 + ut_params.kappa)
    factor = rotation * scales[..., None, :]
    deltas = jnp.sqrt(jnp.asarray(scaled_dimension, means.dtype)) * jnp.swapaxes(
        factor, -1, -2
    )
    center = means[..., None, :]
    return jnp.concatenate((center, center + deltas, center - deltas), axis=-2)


def _matrix_to_quaternion(matrix: jax.Array) -> jax.Array:
    """Convert proper rotation matrices to normalized ``wxyz`` quaternions."""

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]
    q_abs = jnp.sqrt(
        jnp.maximum(
            jnp.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                axis=-1,
            ),
            0.0,
        )
    )
    candidates = jnp.stack(
        (
            jnp.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), -1),
            jnp.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), -1),
            jnp.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), -1),
            jnp.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), -1),
        ),
        axis=-2,
    )
    denominator = 2.0 * jnp.maximum(q_abs, jnp.asarray(0.1, matrix.dtype))
    candidates = candidates / denominator[..., :, None]
    best = jnp.argmax(q_abs, axis=-1)
    quaternion = jnp.take_along_axis(
        candidates, best[..., None, None], axis=-2
    )[..., 0, :]
    return safe_normalize(quaternion)


def _quaternion_slerp(
    start: jax.Array, end: jax.Array, time: jax.Array
) -> jax.Array:
    start = safe_normalize(start)
    end = safe_normalize(end)
    dot = jnp.sum(start * end, axis=-1)
    end = jnp.where((dot < 0.0)[..., None], -end, end)
    dot = jnp.abs(dot)
    dot_clipped = jnp.clip(dot, -1.0 + 1.0e-7, 1.0 - 1.0e-7)
    angle = jnp.arccos(dot_clipped)
    sin_angle = jnp.sin(angle)
    time = jnp.asarray(time, dtype=start.dtype)
    weight_start = jnp.sin((1.0 - time) * angle) / sin_angle
    weight_end = jnp.sin(time * angle) / sin_angle
    spherical = weight_start[..., None] * start + weight_end[..., None] * end
    linear = (1.0 - time)[..., None] * start + time[..., None] * end
    return safe_normalize(jnp.where((dot > 0.9995)[..., None], linear, spherical))


def _transform_world_points(
    world_points: jax.Array, translation: jax.Array, quaternion: jax.Array
) -> jax.Array:
    quaternion = safe_normalize(quaternion)
    vector = quaternion[..., 1:]
    scalar = quaternion[..., :1]
    first_cross = jnp.cross(vector, world_points, axis=-1)
    second_cross = jnp.cross(vector, first_cross, axis=-1)
    return world_points + 2.0 * (scalar * first_cross + second_cross) + translation


def shutter_relative_frame_time(
    pixel_coords: jax.Array,
    width: int,
    height: int,
    rolling_shutter: RollingShutterType | int | str,
) -> jax.Array:
    """Map pixel positions to frame time using gsplat's readout convention."""

    shutter = _normalize_rolling_shutter(rolling_shutter)
    pixel_coords = jnp.asarray(pixel_coords)
    x = pixel_coords[..., 0]
    y = pixel_coords[..., 1]
    if shutter == RollingShutterType.GLOBAL:
        return jnp.zeros_like(x)
    if shutter == RollingShutterType.ROLLING_TOP_TO_BOTTOM:
        return jnp.floor(y) / float(height - 1) if height > 1 else jnp.full_like(y, 0.5)
    if shutter == RollingShutterType.ROLLING_LEFT_TO_RIGHT:
        return jnp.floor(x) / float(width - 1) if width > 1 else jnp.full_like(x, 0.5)
    if shutter == RollingShutterType.ROLLING_BOTTOM_TO_TOP:
        return (
            (height - jnp.ceil(y)) / float(height - 1)
            if height > 1
            else jnp.full_like(y, 0.5)
        )
    return (
        (width - jnp.ceil(x)) / float(width - 1)
        if width > 1
        else jnp.full_like(x, 0.5)
    )


def _check_image_bounds(
    image_points: jax.Array, width: int, height: int, margin_factor: float
) -> jax.Array:
    margin_x = width * margin_factor
    margin_y = height * margin_factor
    return (
        (image_points[..., 0] >= -margin_x)
        & (image_points[..., 0] < width + margin_x)
        & (image_points[..., 1] >= -margin_y)
        & (image_points[..., 1] < height + margin_y)
    )


def _broadcast_coefficients(
    coefficients: jax.Array | None,
    prefix_shape: tuple[int, ...],
    valid_sizes: tuple[int, ...],
    output_size: int,
    dtype: jnp.dtype,
    name: str,
) -> jax.Array:
    if coefficients is None:
        return jnp.zeros(prefix_shape + (output_size,), dtype=dtype)
    coefficients = jnp.asarray(coefficients, dtype=dtype)
    if coefficients.ndim == 0 or coefficients.shape[-1] not in valid_sizes:
        expected = " or ".join(str(size) for size in valid_sizes)
        raise ValueError(f"{name} must end in {expected} coefficients")
    coefficients = jnp.broadcast_to(
        coefficients, prefix_shape + (coefficients.shape[-1],)
    )
    if coefficients.shape[-1] < output_size:
        coefficients = jnp.pad(
            coefficients,
            ((0, 0),) * (coefficients.ndim - 1)
            + ((0, output_size - coefficients.shape[-1]),),
        )
    return coefficients


def _prepare_camera_parameters(
    camera_model: CameraModel,
    Ks: jax.Array,
    radial_coeffs: jax.Array | None,
    tangential_coeffs: jax.Array | None,
    thin_prism_coeffs: jax.Array | None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None,
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt | None = None,
) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None]:
    prefix_shape = Ks.shape[:-2]
    dtype = Ks.dtype
    if camera_model == "pinhole":
        if lidar_coeffs is not None:
            raise ValueError("lidar_coeffs requires camera_model='lidar'")
        if ftheta_coeffs is not None:
            raise ValueError("pinhole cameras do not accept ftheta_coeffs")
        return (
            _broadcast_coefficients(
                radial_coeffs, prefix_shape, (4, 6), 6, dtype, "radial_coeffs"
            ),
            _broadcast_coefficients(
                tangential_coeffs,
                prefix_shape,
                (2,),
                2,
                dtype,
                "tangential_coeffs",
            ),
            _broadcast_coefficients(
                thin_prism_coeffs,
                prefix_shape,
                (4,),
                4,
                dtype,
                "thin_prism_coeffs",
            ),
        )
    if camera_model == "fisheye":
        if lidar_coeffs is not None:
            raise ValueError("lidar_coeffs requires camera_model='lidar'")
        if tangential_coeffs is not None or thin_prism_coeffs is not None:
            raise ValueError(
                "fisheye cameras only support radial_coeffs distortion"
            )
        if ftheta_coeffs is not None:
            raise ValueError("fisheye cameras do not accept ftheta_coeffs")
        return (
            _broadcast_coefficients(
                radial_coeffs, prefix_shape, (4,), 4, dtype, "radial_coeffs"
            ),
            None,
            None,
        )
    if (
        radial_coeffs is not None
        or tangential_coeffs is not None
        or thin_prism_coeffs is not None
    ):
        raise ValueError(f"{camera_model} cameras do not support OpenCV distortion")
    if camera_model == "ftheta":
        if lidar_coeffs is not None:
            raise ValueError("lidar_coeffs requires camera_model='lidar'")
        if ftheta_coeffs is None:
            raise ValueError("ftheta cameras require ftheta_coeffs")
    elif camera_model == "ortho":
        if lidar_coeffs is not None:
            raise ValueError("lidar_coeffs requires camera_model='lidar'")
        if ftheta_coeffs is not None:
            raise ValueError("ortho cameras do not accept ftheta_coeffs")
    elif camera_model == "lidar":
        if not isinstance(
            lidar_coeffs, RowOffsetStructuredSpinningLidarModelParametersExt
        ):
            raise ValueError("camera_model='lidar' requires lidar_coeffs")
        if ftheta_coeffs is not None:
            raise ValueError("lidar cameras do not accept ftheta_coeffs")
    else:
        raise ValueError(f"Unsupported camera model: {camera_model}")
    return None, None, None


def _opencv_pinhole_project(
    camera_points: jax.Array,
    Ks: jax.Array,
    radial_coeffs: jax.Array,
    tangential_coeffs: jax.Array,
    thin_prism_coeffs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    x, y, z = jnp.moveaxis(camera_points, -1, 0)
    z_safe = _safe_denominator(z)
    u = x / z_safe
    v = y / z_safe
    r2 = u * u + v * v
    k1, k2, k3, k4, k5, k6 = (
        radial_coeffs[..., index, None] for index in range(6)
    )
    p1 = tangential_coeffs[..., 0, None]
    p2 = tangential_coeffs[..., 1, None]
    s1, s2, s3, s4 = (
        thin_prism_coeffs[..., index, None] for index in range(4)
    )
    numerator = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    denominator = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
    radial = numerator / _safe_denominator(denominator)
    delta_x = 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u)
    delta_x = delta_x + r2 * (s1 + r2 * s2)
    delta_y = p1 * (r2 + 2.0 * v * v) + 2.0 * p2 * u * v
    delta_y = delta_y + r2 * (s3 + r2 * s4)
    distorted = jnp.stack((radial * u + delta_x, radial * v + delta_y), -1)
    focal = jnp.stack((Ks[..., 0, 0], Ks[..., 1, 1]), -1)[..., None, :]
    principal = Ks[..., :2, 2][..., None, :]
    image_points = distorted * focal + principal
    valid = (z > 0.0) & (jnp.abs(denominator) > 1.0e-8) & (radial > 0.8)
    image_points = jnp.where((z > 0.0)[..., None], image_points, 0.0)
    return image_points, valid


def _opencv_fisheye_project(
    camera_points: jax.Array,
    Ks: jax.Array,
    radial_coeffs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    x, y, z = jnp.moveaxis(camera_points, -1, 0)
    eps = jnp.asarray(1.0e-7, camera_points.dtype)
    r2 = x * x + y * y
    radius = jnp.sqrt(r2 + eps * eps)
    theta = jnp.arctan2(radius, z)
    theta2 = theta * theta
    k1, k2, k3, k4 = (
        radial_coeffs[..., index, None] for index in range(4)
    )
    theta_distorted = theta * (
        1.0
        + theta2
        * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))
    )
    scale = theta_distorted / radius
    normalized = jnp.stack((scale * x, scale * y), -1)
    focal = jnp.stack((Ks[..., 0, 0], Ks[..., 1, 1]), -1)[..., None, :]
    principal = Ks[..., :2, 2][..., None, :]
    image_points = normalized * focal + principal
    valid = (z > 0.0) & (theta_distorted >= 0.0)
    return image_points, valid


def _polyval_ascending(coefficients: tuple[float, ...], value: jax.Array) -> jax.Array:
    result = jnp.full_like(value, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        result = result * value + jnp.asarray(coefficient, value.dtype)
    return result


def _invert_polynomial(
    reference: tuple[float, ...],
    approximate_inverse: tuple[float, ...],
    target: jax.Array,
    iterations: int = 3,
) -> jax.Array:
    derivative = tuple(
        index * reference[index] for index in range(1, len(reference))
    )
    estimate = _polyval_ascending(approximate_inverse, target)
    for _ in range(iterations):
        residual = _polyval_ascending(reference, estimate) - target
        slope = _polyval_ascending(derivative, estimate)
        estimate = estimate - residual / _safe_denominator(slope, 1.0e-6)
    return estimate


def _ftheta_project(
    camera_points: jax.Array,
    Ks: jax.Array,
    parameters: FThetaCameraDistortionParameters,
) -> tuple[jax.Array, jax.Array]:
    x, y, z = jnp.moveaxis(camera_points, -1, 0)
    eps = jnp.asarray(1.0e-7, camera_points.dtype)
    radius = jnp.sqrt(x * x + y * y + eps * eps)
    theta_full = jnp.arctan2(radius, z)
    max_angle = jnp.asarray(parameters.max_angle, camera_points.dtype)
    theta = jnp.minimum(theta_full, max_angle)
    if parameters.reference_poly == FThetaPolynomialType.PIXELDIST_TO_ANGLE:
        pixel_distance = _invert_polynomial(
            parameters.pixeldist_to_angle_poly,
            parameters.angle_to_pixeldist_poly,
            theta,
        )
    else:
        pixel_distance = _polyval_ascending(
            parameters.angle_to_pixeldist_poly, theta
        )
    base_x = pixel_distance * x / radius
    base_y = pixel_distance * y / radius
    c, d, e = (
        jnp.asarray(value, camera_points.dtype) for value in parameters.linear_cde
    )
    principal = Ks[..., :2, 2][..., None, :] + 0.5
    image_points = jnp.stack(
        (c * base_x + d * base_y, e * base_x + base_y), axis=-1
    )
    image_points = image_points + principal
    valid = (z > 0.0) & (theta_full < max_angle)
    image_points = jnp.where((z > 0.0)[..., None], image_points, 0.0)
    return image_points, valid


def _project_camera_points_prepared(
    camera_points: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    camera_model: CameraModel,
    margin_factor: float,
    radial_coeffs: jax.Array | None,
    tangential_coeffs: jax.Array | None,
    thin_prism_coeffs: jax.Array | None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None,
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt | None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None,
) -> tuple[jax.Array, jax.Array]:
    projection_points = camera_points
    external_valid = jnp.ones(camera_points.shape[:-1], dtype=jnp.bool_)
    if external_distortion_coeffs is not None:
        if camera_model == "lidar":
            raise ValueError("LiDAR cameras do not support external distortion")
        if camera_model == "ortho":
            proxy_rays = jnp.concatenate(
                (
                    camera_points[..., :2],
                    jnp.ones(camera_points.shape[:-1] + (1,), camera_points.dtype),
                ),
                axis=-1,
            )
            distorted = distort_camera_rays(
                proxy_rays, external_distortion_coeffs
            )
            external_valid = (
                (distorted[..., 2] > 0.0)
                & jnp.all(jnp.isfinite(distorted), axis=-1)
            )
            distorted_xy = distorted[..., :2] / _safe_denominator(
                distorted[..., 2]
            )[..., None]
            projection_points = jnp.concatenate(
                (distorted_xy, camera_points[..., 2:3]), axis=-1
            )
        else:
            projection_points = distort_camera_rays(
                camera_points, external_distortion_coeffs
            )
            external_valid = jnp.all(jnp.isfinite(projection_points), axis=-1)
    if camera_model == "pinhole":
        assert radial_coeffs is not None
        assert tangential_coeffs is not None
        assert thin_prism_coeffs is not None
        image_points, valid = _opencv_pinhole_project(
            projection_points,
            Ks,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
    elif camera_model == "ortho":
        focal = jnp.stack((Ks[..., 0, 0], Ks[..., 1, 1]), -1)[..., None, :]
        principal = Ks[..., :2, 2][..., None, :]
        image_points = projection_points[..., :2] * focal + principal
        valid = projection_points[..., 2] > 0.0
        image_points = jnp.where(valid[..., None], image_points, 0.0)
    elif camera_model == "fisheye":
        assert radial_coeffs is not None
        image_points, valid = _opencv_fisheye_project(
            projection_points, Ks, radial_coeffs
        )
    elif camera_model == "ftheta":
        assert camera_model == "ftheta" and ftheta_coeffs is not None
        image_points, valid = _ftheta_project(
            projection_points, Ks, ftheta_coeffs
        )
    else:
        assert camera_model == "lidar" and lidar_coeffs is not None
        image_points, valid = LegacyLidarModel(
            lidar_coeffs
        ).camera_ray_to_image_point(projection_points, margin_factor)
    if camera_model != "lidar":
        valid = valid & _check_image_bounds(
            image_points, width, height, margin_factor
        )
    valid = valid & external_valid & jnp.all(jnp.isfinite(image_points), axis=-1)
    return image_points, valid


def project_camera_points(
    camera_points: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    *,
    camera_model: CameraModel = "pinhole",
    margin_factor: float = 0.1,
    radial_coeffs: jax.Array | None = None,
    tangential_coeffs: jax.Array | None = None,
    thin_prism_coeffs: jax.Array | None = None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None = None,
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt | None = None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Project camera-space points through a supported distorted camera."""

    camera_points = jnp.asarray(camera_points)
    Ks = jnp.asarray(Ks)
    if camera_points.shape[-1] != 3:
        raise ValueError("camera_points must end in 3 coordinates")
    if Ks.shape[-2:] != (3, 3):
        raise ValueError("Ks must have shape [..., C, 3, 3]")
    radial, tangential, thin_prism = _prepare_camera_parameters(
        camera_model,
        Ks,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        lidar_coeffs,
    )
    if external_distortion_coeffs is not None:
        validate_external_distortion(external_distortion_coeffs)
        if camera_model == "lidar":
            raise ValueError("LiDAR cameras do not support external distortion")
    return _project_camera_points_prepared(
        camera_points,
        Ks,
        width,
        height,
        camera_model,
        margin_factor,
        radial,
        tangential,
        thin_prism,
        ftheta_coeffs,
        lidar_coeffs,
        external_distortion_coeffs,
    )


def _viewmats_to_pose(viewmats: jax.Array) -> tuple[jax.Array, jax.Array]:
    return viewmats[..., :3, 3], _matrix_to_quaternion(viewmats[..., :3, :3])


def _project_world_points_from_poses(
    world_points: jax.Array,
    start_translation: jax.Array,
    start_quaternion: jax.Array,
    end_translation: jax.Array,
    end_quaternion: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    camera_model: CameraModel,
    margin_factor: float,
    radial_coeffs: jax.Array | None,
    tangential_coeffs: jax.Array | None,
    thin_prism_coeffs: jax.Array | None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None,
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt | None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None,
    rolling_shutter: RollingShutterType,
    rolling_shutter_iterations: int,
) -> tuple[jax.Array, jax.Array]:
    start_points = _transform_world_points(
        world_points,
        start_translation[..., None, :],
        start_quaternion[..., None, :],
    )
    start_image, start_valid = _project_camera_points_prepared(
        start_points,
        Ks,
        width,
        height,
        camera_model,
        margin_factor,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        lidar_coeffs,
        external_distortion_coeffs,
    )
    if rolling_shutter == RollingShutterType.GLOBAL and lidar_coeffs is None:
        return start_image, start_valid

    end_points = _transform_world_points(
        world_points,
        end_translation[..., None, :],
        end_quaternion[..., None, :],
    )
    end_image, end_valid = _project_camera_points_prepared(
        end_points,
        Ks,
        width,
        height,
        camera_model,
        margin_factor,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        lidar_coeffs,
        external_distortion_coeffs,
    )
    initially_valid = start_valid | end_valid
    initial_image = jnp.where(start_valid[..., None], start_image, end_image)
    image = initial_image
    iteration_valid = initially_valid
    for _ in range(rolling_shutter_iterations):
        if lidar_coeffs is None:
            relative_time = shutter_relative_frame_time(
                image, width, height, rolling_shutter
            )
        else:
            relative_time = LegacyLidarModel(
                lidar_coeffs
            ).shutter_relative_frame_time(image)
        translation = (
            (1.0 - relative_time)[..., None] * start_translation[..., None, :]
            + relative_time[..., None] * end_translation[..., None, :]
        )
        quaternion = _quaternion_slerp(
            start_quaternion[..., None, :],
            end_quaternion[..., None, :],
            relative_time,
        )
        camera_points = _transform_world_points(
            world_points, translation, quaternion
        )
        image, iteration_valid = _project_camera_points_prepared(
            camera_points,
            Ks,
            width,
            height,
            camera_model,
            margin_factor,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            ftheta_coeffs,
            lidar_coeffs,
            external_distortion_coeffs,
        )
    final_image = jnp.where(initially_valid[..., None], image, initial_image)
    return final_image, initially_valid & iteration_valid


def project_world_points(
    world_points: jax.Array,
    viewmats: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    *,
    camera_model: CameraModel = "pinhole",
    radial_coeffs: jax.Array | None = None,
    tangential_coeffs: jax.Array | None = None,
    thin_prism_coeffs: jax.Array | None = None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None = None,
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt | None = None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None = None,
    rolling_shutter: RollingShutterType | int | str = RollingShutterType.GLOBAL,
    viewmats_rs: jax.Array | None = None,
    margin_factor: float = 0.1,
    rolling_shutter_iterations: int = 10,
) -> tuple[jax.Array, jax.Array]:
    """Project ``[..., C, M, 3]`` world points, including rolling shutter."""

    world_points = jnp.asarray(world_points)
    viewmats = jnp.asarray(viewmats)
    Ks = jnp.asarray(Ks)
    shutter = _normalize_rolling_shutter(rolling_shutter)
    if viewmats.shape[-2:] != (4, 4):
        raise ValueError("viewmats must have shape [..., C, 4, 4]")
    if Ks.shape[:-2] != viewmats.shape[:-2] or Ks.shape[-2:] != (3, 3):
        raise ValueError("Ks and viewmats must have matching camera dimensions")
    if world_points.shape[:-2] != viewmats.shape[:-2]:
        raise ValueError("world_points must have shape [..., C, M, 3]")
    if rolling_shutter_iterations < 1:
        raise ValueError("rolling_shutter_iterations must be at least 1")
    if camera_model == "lidar":
        if lidar_coeffs is not None:
            width = lidar_coeffs.n_columns
            height = lidar_coeffs.n_rows
        end_viewmats = viewmats if viewmats_rs is None else jnp.asarray(viewmats_rs)
        if end_viewmats.shape != viewmats.shape:
            raise ValueError("viewmats_rs must have the same shape as viewmats")
    elif shutter == RollingShutterType.GLOBAL:
        end_viewmats = viewmats
    else:
        if viewmats_rs is None:
            raise ValueError("viewmats_rs is required for rolling shutter")
        end_viewmats = jnp.asarray(viewmats_rs)
        if end_viewmats.shape != viewmats.shape:
            raise ValueError("viewmats_rs must have the same shape as viewmats")
    radial, tangential, thin_prism = _prepare_camera_parameters(
        camera_model,
        Ks,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        lidar_coeffs,
    )
    if external_distortion_coeffs is not None:
        validate_external_distortion(external_distortion_coeffs)
        if camera_model == "lidar":
            raise ValueError("LiDAR cameras do not support external distortion")
    start_translation, start_quaternion = _viewmats_to_pose(viewmats)
    end_translation, end_quaternion = _viewmats_to_pose(end_viewmats)
    return _project_world_points_from_poses(
        world_points,
        start_translation,
        start_quaternion,
        end_translation,
        end_quaternion,
        Ks,
        width,
        height,
        camera_model,
        margin_factor,
        radial,
        tangential,
        thin_prism,
        ftheta_coeffs,
        lidar_coeffs,
        external_distortion_coeffs,
        shutter,
        rolling_shutter_iterations,
    )


def _camera_centers_at_mid_shutter(
    means: jax.Array,
    start_translation: jax.Array,
    start_quaternion: jax.Array,
    end_translation: jax.Array,
    end_quaternion: jax.Array,
    rolling_shutter: RollingShutterType,
) -> jax.Array:
    camera_count = start_translation.shape[-2]
    means_per_camera = jnp.broadcast_to(
        means[..., None, :, :],
        means.shape[:-2] + (camera_count, means.shape[-2], 3),
    )
    if rolling_shutter == RollingShutterType.GLOBAL:
        translation = start_translation[..., None, :]
        quaternion = start_quaternion[..., None, :]
    else:
        translation = 0.5 * (start_translation + end_translation)
        quaternion = _quaternion_slerp(
            start_quaternion, end_quaternion, jnp.full(start_quaternion.shape[:-1], 0.5)
        )
        translation = translation[..., None, :]
        quaternion = quaternion[..., None, :]
    return _transform_world_points(means_per_camera, translation, quaternion)


def _project_gaussian_chunk(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    active_mask: jax.Array,
    opacities: jax.Array | None,
    *,
    start_translation: jax.Array,
    start_quaternion: jax.Array,
    end_translation: jax.Array,
    end_quaternion: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    camera_model: CameraModel,
    ut_params: UnscentedTransformParameters,
    radial_coeffs: jax.Array | None,
    tangential_coeffs: jax.Array | None,
    thin_prism_coeffs: jax.Array | None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None,
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt | None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None,
    rolling_shutter: RollingShutterType,
    rolling_shutter_iterations: int,
    global_z_order: bool,
    alpha_threshold: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    dummy_means = jnp.zeros_like(means)
    dummy_quats = jnp.zeros_like(quats).at[..., 0].set(1.0)
    dummy_scales = jnp.ones_like(scales)
    means_safe = jnp.where(active_mask[..., None], means, dummy_means)
    quats_safe = jnp.where(active_mask[..., None], quats, dummy_quats)
    scales_safe = jnp.where(active_mask[..., None], scales, dummy_scales)

    sigma_points = world_gaussian_sigma_points(
        means_safe, quats_safe, scales_safe, ut_params
    )
    camera_count = start_translation.shape[-2]
    gaussian_count = means.shape[-2]
    batch_shape = means.shape[:-2]
    sigma_points = jnp.broadcast_to(
        sigma_points[..., None, :, :, :],
        batch_shape + (camera_count, gaussian_count, 7, 3),
    )
    sigma_points_flat = sigma_points.reshape(
        batch_shape + (camera_count, gaussian_count * 7, 3)
    )
    points_2d, valid_points = _project_world_points_from_poses(
        sigma_points_flat,
        start_translation,
        start_quaternion,
        end_translation,
        end_quaternion,
        Ks,
        width,
        height,
        camera_model,
        ut_params.in_image_margin_factor,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        lidar_coeffs,
        external_distortion_coeffs,
        rolling_shutter,
        rolling_shutter_iterations,
    )
    points_2d = points_2d.reshape(
        batch_shape + (camera_count, gaussian_count, 7, 2)
    )
    valid_points = valid_points.reshape(
        batch_shape + (camera_count, gaussian_count, 7)
    )
    weights_mean, weights_cov = compute_ut_weights(
        ut_params, dtype=means.dtype
    )
    weight_shape = (1,) * (points_2d.ndim - 2) + (7, 1)
    if ut_params.require_all_sigma_points_valid:
        cumulative_valid = jnp.cumprod(
            valid_points.astype(jnp.int32), axis=-1
        ).astype(jnp.bool_)
        valid_gaussian = cumulative_valid[..., -1]
        mean_weights = weights_mean.reshape(weight_shape) * cumulative_valid[..., None]
        covariance_weights = (
            weights_cov.reshape(weight_shape) * cumulative_valid[..., None]
        )
    else:
        valid_gaussian = jnp.any(valid_points, axis=-1)
        mean_weights = weights_mean.reshape(weight_shape)
        covariance_weights = weights_cov.reshape(weight_shape)
    means2d = jnp.sum(mean_weights * points_2d, axis=-2)
    delta = points_2d - means2d[..., None, :]
    outer = delta[..., :, :, None] * delta[..., :, None, :]
    covars2d = jnp.sum(covariance_weights[..., None] * outer, axis=-3)
    covars2d = 0.5 * (covars2d + jnp.swapaxes(covars2d, -1, -2))

    center_shutter = (
        RollingShutterType.ROLLING_LEFT_TO_RIGHT
        if lidar_coeffs is not None
        else rolling_shutter
    )
    means_camera = _camera_centers_at_mid_shutter(
        means_safe,
        start_translation,
        start_quaternion,
        end_translation,
        end_quaternion,
        center_shutter,
    )
    center_z = means_camera[..., 2]
    valid_gaussian = valid_gaussian & (center_z >= near_plane) & (center_z <= far_plane)
    eps = jnp.asarray(jnp.finfo(means.dtype).eps, means.dtype)
    valid_parameters = (jnp.sum(quats_safe * quats_safe, axis=-1) > eps) & jnp.all(
        scales_safe > eps, axis=-1
    )
    valid_gaussian = valid_gaussian & valid_parameters[..., None, :]
    valid_gaussian = valid_gaussian & active_mask[..., None, :]

    covariance_xx = covars2d[..., 0, 0]
    covariance_xy = covars2d[..., 0, 1]
    covariance_yy = covars2d[..., 1, 1]
    det_original = covariance_xx * covariance_yy - covariance_xy * covariance_xy
    covariance_xx = covariance_xx + jnp.asarray(eps2d, means.dtype)
    covariance_yy = covariance_yy + jnp.asarray(eps2d, means.dtype)
    determinant = covariance_xx * covariance_yy - covariance_xy * covariance_xy
    valid_gaussian = valid_gaussian & (determinant > 0.0)
    valid_gaussian = valid_gaussian & (covariance_xx > 0.0) & (covariance_yy > 0.0)
    safe_determinant = jnp.maximum(determinant, jnp.asarray(1.0e-10, means.dtype))
    conics = jnp.stack(
        (
            covariance_yy / safe_determinant,
            -covariance_xy / safe_determinant,
            covariance_xx / safe_determinant,
        ),
        axis=-1,
    )
    compensation = jnp.sqrt(
        jnp.maximum(
            det_original / safe_determinant,
            jnp.asarray(0.005**2, means.dtype),
        )
    )

    extend = jnp.full_like(determinant, 3.33)
    if opacities is not None:
        opacity = opacities[..., None, :] * compensation
        alpha_threshold_array = jnp.asarray(alpha_threshold, opacity.dtype)
        valid_gaussian = valid_gaussian & (opacity >= alpha_threshold_array)
        extend = jnp.minimum(
            extend,
            jnp.sqrt(
                2.0
                * jnp.log(jnp.maximum(opacity / alpha_threshold_array, 1.0))
            ),
        )
    half_trace = 0.5 * (covariance_xx + covariance_yy)
    largest_eigenvalue = half_trace + jnp.sqrt(
        jnp.maximum(half_trace * half_trace - determinant, 0.01)
    )
    eigen_radius = extend * jnp.sqrt(jnp.maximum(largest_eigenvalue, 0.0))
    radii = jnp.ceil(
        jnp.minimum(
            extend[..., None]
            * jnp.sqrt(
                jnp.maximum(jnp.stack((covariance_xx, covariance_yy), -1), 0.0)
            ),
            eigen_radius[..., None],
        )
    )
    valid_gaussian = valid_gaussian & (jnp.max(radii, axis=-1) > radius_clip)
    if lidar_coeffs is None:
        image_size = jnp.asarray((width, height), dtype=means.dtype)
        valid_gaussian = valid_gaussian & jnp.all(
            (means2d + radii > 0.0) & (means2d - radii < image_size), axis=-1
        )
    finite = (
        jnp.all(jnp.isfinite(means2d), axis=-1)
        & jnp.all(jnp.isfinite(conics), axis=-1)
        & jnp.isfinite(center_z)
        & jnp.isfinite(compensation)
    )
    valid_gaussian = valid_gaussian & finite
    if global_z_order:
        depths = center_z
    else:
        depths = jnp.linalg.norm(means_camera, axis=-1)
    radii = jnp.where(valid_gaussian[..., None], radii, 0.0).astype(jnp.int32)
    means2d = jnp.where(valid_gaussian[..., None], means2d, 0.0)
    depths = jnp.where(valid_gaussian, depths, 0.0)
    conics = jnp.where(valid_gaussian[..., None], conics, 0.0)
    compensation = jnp.where(valid_gaussian, compensation, 0.0)
    return radii, means2d, depths, conics, compensation, valid_gaussian


def _pad_gaussian_axis(array: jax.Array, padding: int, axis: int) -> jax.Array:
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, padding)
    return jnp.pad(array, tuple(pad_width))


def _merge_projected_chunks(
    value: jax.Array,
    batch_shape: tuple[int, ...],
    camera_count: int,
    padded_count: int,
    gaussian_count: int,
) -> jax.Array:
    batch_rank = len(batch_shape)
    permutation = (
        list(range(1, batch_rank + 1))
        + [batch_rank + 1, 0, batch_rank + 2]
        + list(range(batch_rank + 3, value.ndim))
    )
    value = jnp.transpose(value, permutation)
    feature_shape = value.shape[batch_rank + 3 :]
    value = value.reshape(batch_shape + (camera_count, padded_count) + feature_shape)
    index = (
        (slice(None),) * (batch_rank + 1)
        + (slice(0, gaussian_count),)
        + (slice(None),) * len(feature_shape)
    )
    return value[index]


def fully_fused_projection_with_ut(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    opacities: jax.Array | None,
    viewmats: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1.0e10,
    radius_clip: float = 0.0,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
    ut_params: UnscentedTransformParameters | None = None,
    radial_coeffs: jax.Array | None = None,
    tangential_coeffs: jax.Array | None = None,
    thin_prism_coeffs: jax.Array | None = None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None = None,
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt | None = None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None = None,
    rolling_shutter: RollingShutterType | int | str = RollingShutterType.GLOBAL,
    viewmats_rs: jax.Array | None = None,
    global_z_order: bool = True,
    alpha_threshold: float = 1.0 / 255.0,
    *,
    active_mask: jax.Array | None = None,
    rolling_shutter_iterations: int = 10,
    ut_chunk_size: int = 16384,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array,
]:
    """Project a fixed-capacity Gaussian buffer with 3DGUT.

    The dense return is ``(radii, means2d, depths, conics, compensations,
    valid)`` with shapes ``[..., C, N, ...]``.  ``active_mask`` has shape
    ``[..., N]`` and changes values without changing any compiled shape.
    ``ut_chunk_size`` is static and limits peak UT temporary memory for buffers
    such as the default one-million-Gaussian capacity used by the model.
    """

    if external_distortion_coeffs is not None:
        validate_external_distortion(external_distortion_coeffs)
        if camera_model == "lidar":
            raise ValueError("LiDAR cameras do not support external distortion")
    if ut_params is None:
        ut_params = UnscentedTransformParameters()
    shutter = _normalize_rolling_shutter(rolling_shutter)
    means = jnp.asarray(means)
    quats = jnp.asarray(quats)
    scales = jnp.asarray(scales)
    viewmats = jnp.asarray(viewmats)
    Ks = jnp.asarray(Ks)
    if camera_model == "lidar" and isinstance(
        lidar_coeffs, RowOffsetStructuredSpinningLidarModelParametersExt
    ):
        width = lidar_coeffs.n_columns
        height = lidar_coeffs.n_rows
    if means.ndim < 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [..., N, 3]")
    batch_shape = means.shape[:-2]
    gaussian_count = means.shape[-2]
    if gaussian_count == 0:
        raise ValueError("at least one Gaussian storage slot is required")
    if quats.shape != batch_shape + (gaussian_count, 4):
        raise ValueError("quats must have shape [..., N, 4]")
    if scales.shape != means.shape:
        raise ValueError("scales must have shape [..., N, 3]")
    if not batch_shape and viewmats.ndim == 2:
        viewmats = viewmats[None, ...]
    if not batch_shape and Ks.ndim == 2:
        Ks = Ks[None, ...]
    if viewmats.shape[:-3] != batch_shape or viewmats.shape[-2:] != (4, 4):
        raise ValueError("viewmats must have shape [..., C, 4, 4]")
    if Ks.shape[:-3] != batch_shape or Ks.shape[-2:] != (3, 3):
        raise ValueError("Ks must have shape [..., C, 3, 3]")
    if viewmats.shape[-3] != Ks.shape[-3]:
        raise ValueError("viewmats and Ks must contain the same cameras")
    if ut_chunk_size < 1:
        raise ValueError("ut_chunk_size must be positive")
    if rolling_shutter_iterations < 1:
        raise ValueError("rolling_shutter_iterations must be at least 1")

    camera_count = viewmats.shape[-3]
    if active_mask is None:
        active_mask = jnp.ones(batch_shape + (gaussian_count,), dtype=jnp.bool_)
    else:
        active_mask = jnp.broadcast_to(
            jnp.asarray(active_mask, dtype=jnp.bool_),
            batch_shape + (gaussian_count,),
        )
    if opacities is not None:
        opacities = jnp.broadcast_to(
            jnp.asarray(opacities, dtype=means.dtype),
            batch_shape + (gaussian_count,),
        )
    if camera_model == "lidar":
        end_viewmats = viewmats if viewmats_rs is None else jnp.asarray(viewmats_rs)
        if end_viewmats.shape != viewmats.shape:
            raise ValueError("viewmats_rs must have the same shape as viewmats")
    elif shutter == RollingShutterType.GLOBAL:
        end_viewmats = viewmats
    else:
        if viewmats_rs is None:
            raise ValueError("viewmats_rs is required for rolling shutter")
        end_viewmats = jnp.asarray(viewmats_rs)
        if end_viewmats.shape != viewmats.shape:
            raise ValueError("viewmats_rs must have the same shape as viewmats")
    radial, tangential, thin_prism = _prepare_camera_parameters(
        camera_model,
        Ks,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        lidar_coeffs,
    )
    start_translation, start_quaternion = _viewmats_to_pose(viewmats)
    end_translation, end_quaternion = _viewmats_to_pose(end_viewmats)

    def project_chunk(
        means_chunk: jax.Array,
        quats_chunk: jax.Array,
        scales_chunk: jax.Array,
        mask_chunk: jax.Array,
        opacity_chunk: jax.Array | None,
    ):
        return _project_gaussian_chunk(
            means_chunk,
            quats_chunk,
            scales_chunk,
            mask_chunk,
            opacity_chunk,
            start_translation=start_translation,
            start_quaternion=start_quaternion,
            end_translation=end_translation,
            end_quaternion=end_quaternion,
            Ks=Ks,
            width=width,
            height=height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            camera_model=camera_model,
            ut_params=ut_params,
            radial_coeffs=radial,
            tangential_coeffs=tangential,
            thin_prism_coeffs=thin_prism,
            ftheta_coeffs=ftheta_coeffs,
            lidar_coeffs=lidar_coeffs,
            external_distortion_coeffs=external_distortion_coeffs,
            rolling_shutter=shutter,
            rolling_shutter_iterations=rolling_shutter_iterations,
            global_z_order=global_z_order,
            alpha_threshold=alpha_threshold,
        )

    chunk_size = min(ut_chunk_size, gaussian_count)
    if gaussian_count <= chunk_size:
        radii, means2d, depths, conics, compensation, valid = project_chunk(
            means, quats, scales, active_mask, opacities
        )
    else:
        chunk_count = (gaussian_count + chunk_size - 1) // chunk_size
        padded_count = chunk_count * chunk_size
        padding = padded_count - gaussian_count
        means_padded = _pad_gaussian_axis(means, padding, -2)
        quats_padded = _pad_gaussian_axis(quats, padding, -2)
        scales_padded = _pad_gaussian_axis(scales, padding, -2)
        mask_padded = _pad_gaussian_axis(active_mask, padding, -1)
        arrays = [
            jnp.moveaxis(
                array.reshape(
                    batch_shape
                    + (chunk_count, chunk_size)
                    + array.shape[-1:]
                ),
                len(batch_shape),
                0,
            )
            for array in (means_padded, quats_padded, scales_padded)
        ]
        mask_chunks = jnp.moveaxis(
            mask_padded.reshape(batch_shape + (chunk_count, chunk_size)),
            len(batch_shape),
            0,
        )
        if opacities is None:
            projected = jax.lax.map(
                lambda values: project_chunk(*values, None),
                (*arrays, mask_chunks),
            )
        else:
            opacity_padded = _pad_gaussian_axis(opacities, padding, -1)
            opacity_chunks = jnp.moveaxis(
                opacity_padded.reshape(batch_shape + (chunk_count, chunk_size)),
                len(batch_shape),
                0,
            )
            projected = jax.lax.map(
                lambda values: project_chunk(*values),
                (*arrays, mask_chunks, opacity_chunks),
            )
        radii, means2d, depths, conics, compensation, valid = (
            _merge_projected_chunks(
                value,
                batch_shape,
                camera_count,
                padded_count,
                gaussian_count,
            )
            for value in projected
        )
    return (
        radii,
        means2d,
        depths,
        conics,
        compensation if calc_compensations else None,
        valid,
    )


fully_fused_projection_3dgut = fully_fused_projection_with_ut


def _opencv_pinhole_unproject(
    pixel_coords: jax.Array,
    K: jax.Array,
    radial_coeffs: jax.Array,
    tangential_coeffs: jax.Array,
    thin_prism_coeffs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Invert OpenCV pinhole distortion with gsplat's five Newton steps."""

    principal = K[..., :2, 2][..., None, :]
    focal = jnp.stack((K[..., 0, 0], K[..., 1, 1]), axis=-1)[..., None, :]
    target = (pixel_coords - principal) / focal
    x = target[..., 0]
    y = target[..., 1]
    xd = x
    yd = y
    k1, k2, k3, k4, k5, k6 = (
        radial_coeffs[..., index][..., None] for index in range(6)
    )
    p1, p2 = (
        tangential_coeffs[..., index][..., None] for index in range(2)
    )
    s1, s2, s3, s4 = (
        thin_prism_coeffs[..., index][..., None] for index in range(4)
    )
    running = jnp.ones_like(x, dtype=jnp.bool_)
    converged = jnp.zeros_like(x, dtype=jnp.bool_)
    for _ in range(5):
        radius = x * x + y * y
        radius2 = radius * radius
        numerator = 1.0 + radius * (k1 + radius * (k2 + radius * k3))
        denominator = 1.0 + radius * (k4 + radius * (k5 + radius * k6))
        radial = numerator / _safe_denominator(denominator)
        fx = (
            radial * x
            + 2.0 * p1 * x * y
            + p2 * (radius + 2.0 * x * x)
            + s1 * radius
            + s2 * radius2
            - xd
        )
        fy = (
            radial * y
            + 2.0 * p2 * x * y
            + p1 * (radius + 2.0 * y * y)
            + s3 * radius
            + s4 * radius2
            - yd
        )
        numerator_r = k1 + radius * (2.0 * k2 + radius * (3.0 * k3))
        denominator_r = k4 + radius * (2.0 * k5 + radius * (3.0 * k6))
        radial_r = (
            numerator_r * denominator - numerator * denominator_r
        ) / _safe_denominator(denominator * denominator)
        radial_x = 2.0 * x * radial_r
        radial_y = 2.0 * y * radial_r
        fx_x = radial + radial_x * x + 2.0 * p1 * y + 6.0 * p2 * x
        fx_x = fx_x + 2.0 * x * (s1 + 2.0 * s2 * radius)
        fx_y = radial_y * x + 2.0 * p1 * x + 2.0 * p2 * y
        fx_y = fx_y + 2.0 * y * (s1 + 2.0 * s2 * radius)
        fy_x = radial_x * y + 2.0 * p2 * y + 2.0 * p1 * x
        fy_x = fy_x + 2.0 * x * (s3 + 2.0 * s4 * radius)
        fy_y = radial + radial_y * y + 2.0 * p2 * x + 6.0 * p1 * y
        fy_y = fy_y + 2.0 * y * (s3 + 2.0 * s4 * radius)
        determinant = fx_y * fy_x - fx_x * fy_y
        step_valid = (
            running
            & (radial > 0.0)
            & (jnp.abs(determinant) >= 1.0e-6)
            & jnp.isfinite(determinant)
        )
        dx = (fx * fy_y - fy * fx_y) / _safe_denominator(determinant, 1.0e-6)
        dy = (fy * fx_x - fx * fy_x) / _safe_denominator(determinant, 1.0e-6)
        x = jnp.where(step_valid, x + dx, x)
        y = jnp.where(step_valid, y + dy, y)
        just_converged = step_valid & (jnp.abs(dx) < 1.0e-6) & (
            jnp.abs(dy) < 1.0e-6
        )
        converged = converged | just_converged
        running = step_valid & ~just_converged
    return jnp.stack((x, y), axis=-1), converged


def _opencv_fisheye_unproject(
    pixel_coords: jax.Array,
    K: jax.Array,
    radial_coeffs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    principal = K[..., :2, 2][..., None, :]
    focal = jnp.stack((K[..., 0, 0], K[..., 1, 1]), axis=-1)[..., None, :]
    normalized = (pixel_coords - principal) / focal
    distance = jnp.linalg.norm(normalized, axis=-1)
    k1, k2, k3, k4 = (
        radial_coeffs[..., index][..., None] for index in range(4)
    )
    theta = distance
    running = jnp.ones_like(distance, dtype=jnp.bool_)
    converged = jnp.zeros_like(distance, dtype=jnp.bool_)
    for _ in range(20):
        theta2 = theta * theta
        value = theta * (
            1.0
            + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))
        )
        derivative = 1.0 + theta2 * (
            3.0 * k1
            + theta2 * (5.0 * k2 + theta2 * (7.0 * k3 + theta2 * 9.0 * k4))
        )
        step_valid = running & (jnp.abs(derivative) >= 1.0e-8)
        step = (value - distance) / _safe_denominator(derivative)
        theta = jnp.where(step_valid, theta - step, theta)
        just_converged = step_valid & (jnp.abs(step) < 1.0e-6)
        converged = converged | just_converged
        running = step_valid & ~just_converged
    scale = jnp.sin(theta) / _safe_denominator(distance, 1.0e-6)
    direction = jnp.concatenate(
        (scale[..., None] * normalized, jnp.cos(theta)[..., None]), axis=-1
    )
    centered = distance < 1.0e-6
    direction = jnp.where(
        centered[..., None],
        jnp.asarray((0.0, 0.0, 1.0), direction.dtype),
        direction,
    )
    valid = (centered | converged) & (theta >= 0.0) & (theta < jnp.pi)
    return direction, valid


def _ftheta_unproject(
    pixel_coords: jax.Array,
    K: jax.Array,
    parameters: FThetaCameraDistortionParameters,
) -> tuple[jax.Array, jax.Array]:
    c, d, e = (
        jnp.asarray(value, pixel_coords.dtype) for value in parameters.linear_cde
    )
    centered = pixel_coords - (K[..., :2, 2][..., None, :] + 0.5)
    determinant = c - e * d
    normalized = jnp.stack(
        (
            centered[..., 0] - d * centered[..., 1],
            -e * centered[..., 0] + c * centered[..., 1],
        ),
        axis=-1,
    ) / _safe_denominator(determinant)
    distance = jnp.linalg.norm(normalized, axis=-1)
    if parameters.reference_poly == FThetaPolynomialType.PIXELDIST_TO_ANGLE:
        theta = _polyval_ascending(parameters.pixeldist_to_angle_poly, distance)
    else:
        theta = _invert_polynomial(
            parameters.angle_to_pixeldist_poly,
            parameters.pixeldist_to_angle_poly,
            distance,
        )
    scale = jnp.sin(theta) / _safe_denominator(distance, 1.0e-6)
    direction = jnp.concatenate(
        (scale[..., None] * normalized, jnp.cos(theta)[..., None]), axis=-1
    )
    at_center = distance < 1.0e-6
    direction = jnp.where(
        at_center[..., None],
        jnp.asarray((0.0, 0.0, 1.0), direction.dtype),
        direction,
    )
    valid = (
        jnp.isfinite(theta)
        & (theta >= 0.0)
        & (theta < jnp.asarray(parameters.max_angle, theta.dtype))
        & (jnp.abs(determinant) >= 1.0e-8)
    )
    return direction, valid


def unproject_image_points(
    image_points: jax.Array,
    Ks: jax.Array,
    *,
    camera_model: CameraModel = "pinhole",
    radial_coeffs: jax.Array | None = None,
    tangential_coeffs: jax.Array | None = None,
    thin_prism_coeffs: jax.Array | None = None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None = None,
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt | None = None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return camera-space ray origins, directions, and validity.

    ``image_points`` has shape ``[..., M, 2]`` and ``Ks`` has the matching
    camera prefix ``[..., 3, 3]``. Orthographic origins lie on the camera's
    z=0 image plane; perspective-model origins are zero.
    """

    image_points = jnp.asarray(image_points)
    Ks = jnp.asarray(Ks)
    if image_points.ndim < 2 or image_points.shape[-1] != 2:
        raise ValueError("image_points must have shape [..., M, 2]")
    if Ks.shape[-2:] != (3, 3):
        raise ValueError("Ks must have shape [..., 3, 3]")
    has_opencv_distortion = (
        radial_coeffs is not None
        or tangential_coeffs is not None
        or thin_prism_coeffs is not None
    )
    radial, tangential, thin_prism = _prepare_camera_parameters(
        camera_model,
        Ks,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        ftheta_coeffs,
        lidar_coeffs,
    )
    if external_distortion_coeffs is not None:
        validate_external_distortion(external_distortion_coeffs)
        if camera_model == "lidar":
            raise ValueError("LiDAR cameras do not support external distortion")

    camera_origins = jnp.zeros(
        image_points.shape[:-1] + (3,), image_points.dtype
    )
    if camera_model == "pinhole":
        if has_opencv_distortion:
            assert (
                radial is not None
                and tangential is not None
                and thin_prism is not None
            )
            normalized, valid = _opencv_pinhole_unproject(
                image_points, Ks, radial, tangential, thin_prism
            )
        else:
            principal = Ks[..., :2, 2][..., None, :]
            focal = jnp.stack(
                (Ks[..., 0, 0], Ks[..., 1, 1]), axis=-1
            )[..., None, :]
            normalized = (image_points - principal) / focal
            valid = jnp.ones(image_points.shape[:-1], dtype=jnp.bool_)
        camera_directions = jnp.concatenate(
            (
                normalized,
                jnp.ones(normalized.shape[:-1] + (1,), normalized.dtype),
            ),
            axis=-1,
        )
    elif camera_model == "ortho":
        principal = Ks[..., :2, 2][..., None, :]
        focal = jnp.stack(
            (Ks[..., 0, 0], Ks[..., 1, 1]), axis=-1
        )[..., None, :]
        normalized = (image_points - principal) / focal
        valid = jnp.ones(image_points.shape[:-1], dtype=jnp.bool_)
        if external_distortion_coeffs is not None:
            proxy = jnp.concatenate(
                (
                    normalized,
                    jnp.ones(normalized.shape[:-1] + (1,), normalized.dtype),
                ),
                axis=-1,
            )
            undistorted = distort_camera_rays(
                proxy, external_distortion_coeffs, inverse=True
            )
            distortion_valid = (
                (undistorted[..., 2] > 0.0)
                & jnp.all(jnp.isfinite(undistorted), axis=-1)
            )
            normalized = undistorted[..., :2] / _safe_denominator(
                undistorted[..., 2]
            )[..., None]
            valid = valid & distortion_valid
        camera_origins = camera_origins.at[..., :2].set(normalized)
        camera_directions = jnp.broadcast_to(
            jnp.asarray((0.0, 0.0, 1.0), image_points.dtype),
            camera_origins.shape,
        )
    elif camera_model == "fisheye":
        assert radial is not None
        camera_directions, valid = _opencv_fisheye_unproject(
            image_points, Ks, radial
        )
    elif camera_model == "ftheta":
        if ftheta_coeffs is None:
            raise ValueError("ftheta cameras require ftheta_coeffs")
        camera_directions, valid = _ftheta_unproject(
            image_points, Ks, ftheta_coeffs
        )
    else:
        assert camera_model == "lidar" and lidar_coeffs is not None
        lidar_rays = LegacyLidarModel(lidar_coeffs).image_point_to_camera_ray(
            image_points
        )
        camera_directions = lidar_rays.sensor_rays
        valid = lidar_rays.valid_flag

    if external_distortion_coeffs is not None and camera_model != "ortho":
        camera_directions = distort_camera_rays(
            camera_directions, external_distortion_coeffs, inverse=True
        )
    camera_directions = safe_normalize(camera_directions)
    valid = (
        valid
        & jnp.all(jnp.isfinite(camera_origins), axis=-1)
        & jnp.all(jnp.isfinite(camera_directions), axis=-1)
    )
    camera_origins = jnp.where(valid[..., None], camera_origins, 0.0)
    camera_directions = jnp.where(valid[..., None], camera_directions, 0.0)
    return camera_origins, camera_directions, valid


def _world_rays_from_pixels(
    pixel_coords: jax.Array,
    viewmat: jax.Array,
    K: jax.Array,
    width: int,
    height: int,
    *,
    camera_model: CameraModel,
    radial_coeffs: jax.Array | None,
    tangential_coeffs: jax.Array | None,
    thin_prism_coeffs: jax.Array | None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None,
    rolling_shutter: RollingShutterType,
    viewmat_rs: jax.Array | None,
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt | None = None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    camera_origins, camera_directions, valid = unproject_image_points(
        pixel_coords,
        K,
        camera_model=camera_model,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        ftheta_coeffs=ftheta_coeffs,
        lidar_coeffs=lidar_coeffs,
        external_distortion_coeffs=external_distortion_coeffs,
    )

    shutter = _normalize_rolling_shutter(rolling_shutter)
    if lidar_coeffs is not None:
        end_viewmat = viewmat if viewmat_rs is None else viewmat_rs
    else:
        end_viewmat = viewmat if shutter == RollingShutterType.GLOBAL else viewmat_rs
    if end_viewmat is None:
        raise ValueError("viewmats_rs is required for rolling shutter")
    start_translation, start_quaternion = _viewmats_to_pose(viewmat)
    end_translation, end_quaternion = _viewmats_to_pose(end_viewmat)
    if lidar_coeffs is None:
        relative_time = shutter_relative_frame_time(
            pixel_coords, width, height, shutter
        )
    else:
        relative_time = LegacyLidarModel(
            lidar_coeffs
        ).shutter_relative_frame_time(pixel_coords)
    translation = (
        (1.0 - relative_time)[..., None] * start_translation[..., None, :]
        + relative_time[..., None] * end_translation[..., None, :]
    )
    quaternion = _quaternion_slerp(
        start_quaternion[..., None, :],
        end_quaternion[..., None, :],
        relative_time,
    )
    rotation_cw = quat_to_rotmat(quaternion)
    ray_origins = jnp.einsum(
        "...ji,...j->...i", rotation_cw, camera_origins - translation
    )
    ray_directions = jnp.einsum(
        "...ji,...j->...i", rotation_cw, camera_directions
    )
    valid = (
        valid
        & jnp.all(jnp.isfinite(ray_origins), axis=-1)
        & jnp.all(jnp.isfinite(ray_directions), axis=-1)
    )
    return ray_origins, safe_normalize(ray_directions), valid


def _rasterize_eval3d_camera(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    viewmat: jax.Array,
    K: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: jax.Array,
    gaussian_ids: jax.Array,
    valid_count: jax.Array,
    *,
    background: jax.Array,
    mask: jax.Array | None,
    camera_model: CameraModel,
    radial_coeffs: jax.Array | None,
    tangential_coeffs: jax.Array | None,
    thin_prism_coeffs: jax.Array | None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None,
    rolling_shutter: RollingShutterType,
    viewmat_rs: jax.Array | None,
    rays: jax.Array | None,
    use_hit_distance: bool,
    return_normals: bool,
    flatten_index_offset: jax.Array | int,
    max_gaussians_per_tile: int,
    tile_batch_size: int,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Rasterize one camera using exact world-space Gaussian responses."""

    gaussian_count = means.shape[0]
    if gaussian_count < 1:
        raise ValueError("eval3d requires at least one Gaussian slot")
    if max_gaussians_per_tile < 1:
        raise ValueError("max_gaussians_per_tile must be positive")
    tile_height, tile_width = isect_offsets.shape
    tile_count = tile_height * tile_width
    candidate_capacity = min(gaussian_count, gaussian_ids.shape[0])
    candidate_slots = jnp.arange(candidate_capacity, dtype=jnp.int32)
    flat_offsets = isect_offsets.reshape(-1)
    local_y, local_x = jnp.meshgrid(
        jnp.arange(tile_size, dtype=means.dtype) + 0.5,
        jnp.arange(tile_size, dtype=means.dtype) + 0.5,
        indexing="ij",
    )
    local_x = local_x.reshape(-1)
    local_y = local_y.reshape(-1)
    pixel_count = tile_size * tile_size

    def render_tile(tile_id: jax.Array):
        tile_x = tile_id % tile_width
        tile_y = tile_id // tile_width
        start = flat_offsets[tile_id]
        end = jnp.where(
            tile_id + 1 < tile_count,
            flat_offsets[jnp.minimum(tile_id + 1, tile_count - 1)],
            valid_count,
        )
        candidate_count = jnp.maximum(end - start, 0)
        positions = start + candidate_slots
        safe_positions = jnp.clip(positions, 0, gaussian_ids.shape[0] - 1)
        selected_ids = gaussian_ids[safe_positions]
        selected_valid = (
            (candidate_slots < candidate_count)
            & (positions < valid_count)
            & (selected_ids >= 0)
            & (selected_ids < gaussian_count)
        )
        selected_ids = jnp.clip(selected_ids, 0, gaussian_count - 1)

        pixel_x = tile_x.astype(means.dtype) * tile_size + local_x
        pixel_y = tile_y.astype(means.dtype) * tile_size + local_y
        pixel_coords = jnp.stack((pixel_x, pixel_y), axis=-1)
        pixel_valid = (pixel_x < image_width) & (pixel_y < image_height)
        if rays is None:
            ray_origins, ray_directions, ray_valid = _world_rays_from_pixels(
                pixel_coords,
                viewmat,
                K,
                image_width,
                image_height,
                camera_model=camera_model,
                radial_coeffs=radial_coeffs,
                tangential_coeffs=tangential_coeffs,
                thin_prism_coeffs=thin_prism_coeffs,
                ftheta_coeffs=ftheta_coeffs,
                external_distortion_coeffs=external_distortion_coeffs,
                rolling_shutter=rolling_shutter,
                viewmat_rs=viewmat_rs,
            )
        else:
            pixel_columns = jnp.floor(pixel_x).astype(jnp.int32)
            pixel_rows = jnp.floor(pixel_y).astype(jnp.int32)
            pixel_indices = jnp.clip(
                pixel_rows * image_width + pixel_columns,
                0,
                image_width * image_height - 1,
            )
            selected_rays = rays.reshape(-1, 6)[pixel_indices]
            ray_origins = selected_rays[:, :3]
            raw_directions = selected_rays[:, 3:]
            direction_norm = jnp.linalg.norm(raw_directions, axis=-1)
            ray_valid = (
                jnp.all(jnp.isfinite(selected_rays), axis=-1)
                & (direction_norm > 1.0e-8)
            )
            ray_directions = safe_normalize(raw_directions)

        selected_means = means[selected_ids]
        selected_scales = scales[selected_ids]
        rotations = quat_to_rotmat(quats[selected_ids])
        origin_delta = ray_origins[None, :, :] - selected_means[:, None, :]
        local_origins = jnp.einsum("kij,kpi->kpj", rotations, origin_delta)
        local_directions = jnp.einsum(
            "kij,pi->kpj", rotations, ray_directions
        )
        safe_scales = _safe_denominator(selected_scales)
        local_origins = local_origins / safe_scales[:, None, :]
        local_directions = safe_normalize(
            local_directions / safe_scales[:, None, :]
        )
        hit_t = jnp.sum(local_directions * -local_origins, axis=-1)
        distance_vector = jnp.cross(local_directions, local_origins, axis=-1)
        distance_squared = jnp.sum(distance_vector * distance_vector, axis=-1)
        distance_squared = jnp.where(hit_t < 0.0, jnp.inf, distance_squared)
        alpha = jnp.minimum(
            opacities[selected_ids, None] * jnp.exp(-0.5 * distance_squared),
            0.99,
        )
        parameter_valid = (
            jnp.all(jnp.isfinite(selected_means), axis=-1)
            & jnp.all(jnp.isfinite(selected_scales), axis=-1)
            & jnp.all(jnp.abs(selected_scales) > 1.0e-8, axis=-1)
        )
        alpha_valid = (
            selected_valid[:, None]
            & parameter_valid[:, None]
            & pixel_valid[None, :]
            & ray_valid[None, :]
            & jnp.isfinite(alpha)
            & (alpha >= 1.0 / 255.0)
        )
        if mask is not None:
            alpha_valid = alpha_valid & mask[tile_y, tile_x]
        alpha = jnp.where(alpha_valid, alpha, 0.0)
        transmittance = jnp.concatenate(
            (
                jnp.ones((1, pixel_count), dtype=alpha.dtype),
                jnp.cumprod(1.0 - alpha, axis=0)[:-1],
            ),
            axis=0,
        )
        accepted = transmittance * (1.0 - alpha) > 1.0e-4
        weights = jnp.where(accepted, alpha * transmittance, 0.0)
        accumulated_alpha = jnp.sum(weights, axis=0)
        accumulated_samples = accepted & (alpha > 0.0)
        sample_counts = jnp.sum(
            accumulated_samples, axis=0, dtype=jnp.int32
        )
        flatten_positions = (
            jnp.asarray(flatten_index_offset, dtype=jnp.int32) + positions
        )
        last_ids = jnp.max(
            jnp.where(
                accumulated_samples,
                flatten_positions[:, None],
                jnp.int32(-1),
            ),
            axis=0,
        )
        hit_distance = jnp.linalg.norm(
            selected_scales[:, None, :]
            * local_directions
            * hit_t[..., None],
            axis=-1,
        )
        selected_colors = jnp.broadcast_to(
            colors[selected_ids, None, :],
            (candidate_capacity, pixel_count, colors.shape[-1]),
        )
        if use_hit_distance:
            selected_colors = selected_colors.at[..., -1].set(hit_distance)
        rendered = jnp.einsum(
            "kp,kpd->pd",
            weights,
            selected_colors,
            precision=jax.lax.Precision.HIGHEST,
        )
        rendered = rendered + (1.0 - accumulated_alpha[:, None]) * background
        rendered = jnp.where(pixel_valid[:, None], rendered, 0.0)

        gaussian_normals = rotations[:, :, 2]
        normal_dot_ray = jnp.einsum(
            "ki,pi->kp", gaussian_normals, ray_directions
        )
        oriented_normals = jnp.where(
            normal_dot_ray[..., None] > 0.0,
            -gaussian_normals[:, None, :],
            gaussian_normals[:, None, :],
        )
        oriented_normals = safe_normalize(oriented_normals)
        rendered_normals = jnp.einsum(
            "kp,kpi->pi",
            weights,
            oriented_normals,
            precision=jax.lax.Precision.HIGHEST,
        )
        rendered_normals = jnp.where(
            pixel_valid[:, None] & return_normals,
            rendered_normals,
            0.0,
        )
        return (
            rendered.reshape(tile_size, tile_size, colors.shape[-1]),
            accumulated_alpha.reshape(tile_size, tile_size, 1),
            rendered_normals.reshape(tile_size, tile_size, 3),
            last_ids.reshape(tile_size, tile_size),
            sample_counts.reshape(tile_size, tile_size),
            candidate_count,
            candidate_count > candidate_capacity,
            candidate_count > max_gaussians_per_tile,
        )

    (
        rendered_tiles,
        alpha_tiles,
        normal_tiles,
        last_id_tiles,
        sample_count_tiles,
        candidate_counts,
        tile_overflow,
        candidate_limit_exceeded,
    ) = jax.lax.map(
        # Reverse mode otherwise keeps every tile's [candidate, pixel]
        # compositing intermediates alive at once.
        jax.checkpoint(render_tile),
        jnp.arange(tile_count, dtype=jnp.int32),
        batch_size=tile_batch_size,
    )
    rendered = (
        rendered_tiles.reshape(
            tile_height, tile_width, tile_size, tile_size, colors.shape[-1]
        )
        .transpose(0, 2, 1, 3, 4)
        .reshape(
            tile_height * tile_size,
            tile_width * tile_size,
            colors.shape[-1],
        )
    )[:image_height, :image_width]
    alphas = (
        alpha_tiles.reshape(tile_height, tile_width, tile_size, tile_size, 1)
        .transpose(0, 2, 1, 3, 4)
        .reshape(tile_height * tile_size, tile_width * tile_size, 1)
    )[:image_height, :image_width]
    normals = (
        normal_tiles.reshape(tile_height, tile_width, tile_size, tile_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(tile_height * tile_size, tile_width * tile_size, 3)
    )[:image_height, :image_width]
    last_ids = (
        last_id_tiles.reshape(tile_height, tile_width, tile_size, tile_size)
        .transpose(0, 2, 1, 3)
        .reshape(tile_height * tile_size, tile_width * tile_size)
    )[:image_height, :image_width]
    sample_counts = (
        sample_count_tiles.reshape(tile_height, tile_width, tile_size, tile_size)
        .transpose(0, 2, 1, 3)
        .reshape(tile_height * tile_size, tile_width * tile_size)
    )[:image_height, :image_width]
    return rendered, alphas, {
        "candidate_counts": candidate_counts.reshape(tile_height, tile_width),
        "tile_overflow": tile_overflow.reshape(tile_height, tile_width),
        "candidate_limit_exceeded": candidate_limit_exceeded.reshape(
            tile_height, tile_width
        ),
        "normals": normals,
        "last_ids": last_ids,
        "sample_counts": sample_counts,
    }


def _rasterize_eval3d_lidar(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    viewmat: jax.Array,
    K: jax.Array,
    isect_offsets: jax.Array,
    gaussian_ids: jax.Array,
    valid_count: jax.Array,
    lidar_coeffs: RowOffsetStructuredSpinningLidarModelParametersExt,
    *,
    background: jax.Array,
    mask: jax.Array | None,
    viewmat_rs: jax.Array | None,
    rays: jax.Array | None,
    use_hit_distance: bool,
    return_normals: bool,
    flatten_index_offset: jax.Array | int,
    max_gaussians_per_tile: int,
    tile_batch_size: int,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Rasterize one structured LiDAR using its angular tile-to-ray map."""

    gaussian_count = means.shape[0]
    if gaussian_count < 1:
        raise ValueError("eval3d requires at least one Gaussian slot")
    if max_gaussians_per_tile < 1:
        raise ValueError("max_gaussians_per_tile must be positive")
    tile_height = lidar_coeffs.tiling.n_bins_elevation
    tile_width = lidar_coeffs.tiling.n_bins_azimuth
    if isect_offsets.shape != (tile_height, tile_width):
        raise ValueError("LiDAR isect_offsets must match its angular tile grid")
    element_limit = lidar_coeffs.tiling.max_elements_per_tile
    if element_limit < 1:
        raise ValueError("LiDAR tiling must contain at least one ray element")

    image_height = lidar_coeffs.n_rows
    image_width = lidar_coeffs.n_columns
    image_size = image_height * image_width
    tile_count = tile_height * tile_width
    candidate_capacity = min(gaussian_count, gaussian_ids.shape[0])
    candidate_slots = jnp.arange(candidate_capacity, dtype=jnp.int32)
    element_slots = jnp.arange(element_limit, dtype=jnp.int32)
    flat_offsets = isect_offsets.reshape(-1)

    if rays is None:
        image_points = generate_lidar_image_points(lidar_coeffs).reshape(-1, 2)
        all_ray_origins, all_ray_directions, all_ray_valid = (
            _world_rays_from_pixels(
                image_points,
                viewmat,
                K,
                image_width,
                image_height,
                camera_model="lidar",
                radial_coeffs=None,
                tangential_coeffs=None,
                thin_prism_coeffs=None,
                ftheta_coeffs=None,
                rolling_shutter=RollingShutterType.GLOBAL,
                viewmat_rs=viewmat_rs,
                lidar_coeffs=lidar_coeffs,
            )
        )
    else:
        supplied_rays = rays.reshape(image_size, 6)
        all_ray_origins = supplied_rays[:, :3]
        raw_directions = supplied_rays[:, 3:]
        direction_norm = jnp.linalg.norm(raw_directions, axis=-1)
        all_ray_valid = (
            jnp.all(jnp.isfinite(supplied_rays), axis=-1)
            & (direction_norm > 1.0e-8)
        )
        all_ray_directions = safe_normalize(raw_directions)

    def render_tile(tile_id: jax.Array):
        start = flat_offsets[tile_id]
        end = jnp.where(
            tile_id + 1 < tile_count,
            flat_offsets[jnp.minimum(tile_id + 1, tile_count - 1)],
            valid_count,
        )
        candidate_count = jnp.maximum(end - start, 0)
        positions = start + candidate_slots
        safe_positions = jnp.clip(positions, 0, gaussian_ids.shape[0] - 1)
        selected_ids = gaussian_ids[safe_positions]
        selected_valid = (
            (candidate_slots < candidate_count)
            & (positions < valid_count)
            & (selected_ids >= 0)
            & (selected_ids < gaussian_count)
        )
        selected_ids = jnp.clip(selected_ids, 0, gaussian_count - 1)

        element_start, element_count = lidar_coeffs.tiling.tiles_pack_info[tile_id]
        element_positions = element_start + element_slots
        safe_element_positions = jnp.clip(
            element_positions,
            0,
            lidar_coeffs.tiling.tiles_to_elements_map.shape[0] - 1,
        )
        elements = lidar_coeffs.tiling.tiles_to_elements_map[
            safe_element_positions
        ]
        columns = elements[:, 0]
        rows = elements[:, 1]
        element_valid = (
            (element_slots < element_count)
            & (columns >= 0)
            & (columns < image_width)
            & (rows >= 0)
            & (rows < image_height)
        )
        pixel_ids = jnp.clip(rows * image_width + columns, 0, image_size - 1)
        ray_origins = all_ray_origins[pixel_ids]
        ray_directions = all_ray_directions[pixel_ids]
        ray_valid = all_ray_valid[pixel_ids]

        selected_means = means[selected_ids]
        selected_scales = scales[selected_ids]
        rotations = quat_to_rotmat(quats[selected_ids])
        origin_delta = ray_origins[None, :, :] - selected_means[:, None, :]
        local_origins = jnp.einsum("kij,kpi->kpj", rotations, origin_delta)
        local_directions = jnp.einsum(
            "kij,pi->kpj", rotations, ray_directions
        )
        safe_scales = _safe_denominator(selected_scales)
        local_origins = local_origins / safe_scales[:, None, :]
        local_directions = safe_normalize(
            local_directions / safe_scales[:, None, :]
        )
        hit_t = jnp.sum(local_directions * -local_origins, axis=-1)
        distance_vector = jnp.cross(local_directions, local_origins, axis=-1)
        distance_squared = jnp.sum(distance_vector * distance_vector, axis=-1)
        distance_squared = jnp.where(hit_t < 0.0, jnp.inf, distance_squared)
        alpha = jnp.minimum(
            opacities[selected_ids, None] * jnp.exp(-0.5 * distance_squared),
            0.99,
        )
        parameter_valid = (
            jnp.all(jnp.isfinite(selected_means), axis=-1)
            & jnp.all(jnp.isfinite(selected_scales), axis=-1)
            & jnp.all(jnp.abs(selected_scales) > 1.0e-8, axis=-1)
        )
        alpha_valid = (
            selected_valid[:, None]
            & parameter_valid[:, None]
            & element_valid[None, :]
            & ray_valid[None, :]
            & jnp.isfinite(alpha)
            & (alpha >= 1.0 / 255.0)
        )
        if mask is not None:
            tile_y = tile_id // tile_width
            tile_x = tile_id % tile_width
            alpha_valid = alpha_valid & mask[tile_y, tile_x]
        alpha = jnp.where(alpha_valid, alpha, 0.0)
        transmittance = jnp.concatenate(
            (
                jnp.ones((1, element_limit), dtype=alpha.dtype),
                jnp.cumprod(1.0 - alpha, axis=0)[:-1],
            ),
            axis=0,
        )
        accepted = transmittance * (1.0 - alpha) > 1.0e-4
        weights = jnp.where(accepted, alpha * transmittance, 0.0)
        accumulated_alpha = jnp.sum(weights, axis=0)
        accumulated_samples = accepted & (alpha > 0.0)
        sample_counts = jnp.sum(
            accumulated_samples, axis=0, dtype=jnp.int32
        )
        flatten_positions = (
            jnp.asarray(flatten_index_offset, dtype=jnp.int32) + positions
        )
        last_ids = jnp.max(
            jnp.where(
                accumulated_samples,
                flatten_positions[:, None],
                jnp.int32(-1),
            ),
            axis=0,
        )
        hit_distance = jnp.linalg.norm(
            selected_scales[:, None, :]
            * local_directions
            * hit_t[..., None],
            axis=-1,
        )
        selected_colors = jnp.broadcast_to(
            colors[selected_ids, None, :],
            (candidate_capacity, element_limit, colors.shape[-1]),
        )
        if use_hit_distance:
            selected_colors = selected_colors.at[..., -1].set(hit_distance)
        rendered = jnp.einsum(
            "kp,kpd->pd",
            weights,
            selected_colors,
            precision=jax.lax.Precision.HIGHEST,
        )
        rendered = rendered + (1.0 - accumulated_alpha[:, None]) * background
        rendered = jnp.where(element_valid[:, None], rendered, 0.0)
        accumulated_alpha = jnp.where(element_valid, accumulated_alpha, 0.0)
        last_ids = jnp.where(element_valid, last_ids, -1)
        sample_counts = jnp.where(element_valid, sample_counts, 0)

        gaussian_normals = rotations[:, :, 2]
        normal_dot_ray = jnp.einsum(
            "ki,pi->kp", gaussian_normals, ray_directions
        )
        oriented_normals = jnp.where(
            normal_dot_ray[..., None] > 0.0,
            -gaussian_normals[:, None, :],
            gaussian_normals[:, None, :],
        )
        oriented_normals = safe_normalize(oriented_normals)
        rendered_normals = jnp.einsum(
            "kp,kpi->pi",
            weights,
            oriented_normals,
            precision=jax.lax.Precision.HIGHEST,
        )
        rendered_normals = jnp.where(
            element_valid[:, None] & return_normals,
            rendered_normals,
            0.0,
        )
        return (
            pixel_ids,
            element_valid,
            rendered,
            accumulated_alpha[:, None],
            rendered_normals,
            last_ids,
            sample_counts,
            candidate_count,
            candidate_count > candidate_capacity,
            candidate_count > max_gaussians_per_tile,
        )

    (
        pixel_ids,
        element_valid,
        rendered_elements,
        alpha_elements,
        normal_elements,
        last_id_elements,
        sample_count_elements,
        candidate_counts,
        tile_overflow,
        candidate_limit_exceeded,
    ) = jax.lax.map(
        # Reverse mode otherwise keeps every tile's [candidate, pixel]
        # compositing intermediates alive at once.
        jax.checkpoint(render_tile),
        jnp.arange(tile_count, dtype=jnp.int32),
        batch_size=tile_batch_size,
    )
    flat_pixel_ids = pixel_ids.reshape(-1)
    flat_element_valid = element_valid.reshape(-1)
    rendered = jnp.zeros(
        (image_size, colors.shape[-1]), dtype=colors.dtype
    ).at[flat_pixel_ids].add(
        jnp.where(
            flat_element_valid[:, None],
            rendered_elements.reshape(-1, colors.shape[-1]),
            0.0,
        )
    )
    alphas = jnp.zeros((image_size, 1), dtype=opacities.dtype).at[
        flat_pixel_ids
    ].add(
        jnp.where(
            flat_element_valid[:, None], alpha_elements.reshape(-1, 1), 0.0
        )
    )
    normals = jnp.zeros((image_size, 3), dtype=means.dtype).at[
        flat_pixel_ids
    ].add(
        jnp.where(
            flat_element_valid[:, None], normal_elements.reshape(-1, 3), 0.0
        )
    )
    last_ids = jnp.full((image_size,), -1, dtype=jnp.int32).at[
        flat_pixel_ids
    ].max(
        jnp.where(flat_element_valid, last_id_elements.reshape(-1), -1)
    )
    sample_counts = jnp.zeros((image_size,), dtype=jnp.int32).at[
        flat_pixel_ids
    ].add(
        jnp.where(flat_element_valid, sample_count_elements.reshape(-1), 0)
    )
    output_shape = (image_height, image_width)
    return (
        rendered.reshape(output_shape + (colors.shape[-1],)),
        alphas.reshape(output_shape + (1,)),
        {
            "candidate_counts": candidate_counts.reshape(
                tile_height, tile_width
            ),
            "tile_overflow": tile_overflow.reshape(tile_height, tile_width),
            "candidate_limit_exceeded": candidate_limit_exceeded.reshape(
                tile_height, tile_width
            ),
            "normals": normals.reshape(output_shape + (3,)),
            "last_ids": last_ids.reshape(output_shape),
            "sample_counts": sample_counts.reshape(output_shape),
        },
    )


def rasterize_to_pixels_eval3d(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    viewmats: jax.Array,
    Ks: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: jax.Array,
    flatten_ids: jax.Array,
    backgrounds: jax.Array | None = None,
    masks: jax.Array | None = None,
    camera_model: CameraModel = "pinhole",
    ut_params: UnscentedTransformParameters | None = None,
    rays: jax.Array | None = None,
    radial_coeffs: jax.Array | None = None,
    tangential_coeffs: jax.Array | None = None,
    thin_prism_coeffs: jax.Array | None = None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None = None,
    lidar_coeffs: Any | None = None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None = None,
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: jax.Array | None = None,
    use_hit_distance: bool = False,
    return_normals: bool = False,
    renderer_config: RendererConfig | None = None,
    *,
    active_mask: jax.Array | None = None,
    max_gaussians_per_tile: int = 512,
    tile_batch_size: int = 1,
    return_info: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Rasterize exact world-space Gaussian responses from padded intersections."""

    del ut_params
    resolve_renderer_config(renderer_config, with_eval3d=True)
    if camera_model == "lidar":
        if not isinstance(
            lidar_coeffs, RowOffsetStructuredSpinningLidarModelParametersExt
        ):
            raise ValueError("camera_model='lidar' requires lidar_coeffs")
        image_width = lidar_coeffs.n_columns
        image_height = lidar_coeffs.n_rows
    elif lidar_coeffs is not None:
        raise ValueError("lidar_coeffs requires camera_model='lidar'")
    if external_distortion_coeffs is not None:
        validate_external_distortion(external_distortion_coeffs)
        if camera_model == "lidar":
            raise ValueError("LiDAR cameras do not support external distortion")
    means = jnp.asarray(means)
    quats = jnp.asarray(quats)
    scales = jnp.asarray(scales)
    colors = jnp.asarray(colors)
    opacities = jnp.asarray(opacities)
    viewmats = jnp.asarray(viewmats)
    Ks = jnp.asarray(Ks)
    if means.ndim < 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [..., N, 3]")
    batch_shape = means.shape[:-2]
    gaussian_count = means.shape[-2]
    if quats.shape != batch_shape + (gaussian_count, 4):
        raise ValueError("quats must have shape [..., N, 4]")
    if scales.shape != batch_shape + (gaussian_count, 3):
        raise ValueError("scales must have shape [..., N, 3]")
    if viewmats.shape[:-3] != batch_shape or viewmats.shape[-2:] != (4, 4):
        raise ValueError("viewmats must have shape [..., C, 4, 4]")
    camera_count = viewmats.shape[-3]
    if Ks.shape != batch_shape + (camera_count, 3, 3):
        raise ValueError("Ks must have shape [..., C, 3, 3]")
    if colors.shape == batch_shape + (gaussian_count, colors.shape[-1]):
        colors = jnp.broadcast_to(
            colors[..., None, :, :],
            batch_shape + (camera_count, gaussian_count, colors.shape[-1]),
        )
    if colors.shape[:-1] != batch_shape + (camera_count, gaussian_count):
        raise ValueError("colors must have shape [..., C, N, channels]")
    if opacities.shape == batch_shape + (gaussian_count,):
        opacities = jnp.broadcast_to(
            opacities[..., None, :],
            batch_shape + (camera_count, gaussian_count),
        )
    if opacities.shape != batch_shape + (camera_count, gaussian_count):
        raise ValueError("opacities must have shape [..., C, N]")
    if rays is not None:
        rays = jnp.asarray(rays)
        expected_rays = batch_shape + (
            camera_count,
            image_height,
            image_width,
            6,
        )
        if rays.shape != expected_rays:
            raise ValueError(f"rays must have shape {expected_rays}")

    if lidar_coeffs is None:
        tile_height = (image_height + tile_size - 1) // tile_size
        tile_width = (image_width + tile_size - 1) // tile_size
    else:
        tile_height = lidar_coeffs.tiling.n_bins_elevation
        tile_width = lidar_coeffs.tiling.n_bins_azimuth
    offsets = jnp.asarray(isect_offsets, dtype=jnp.int32)
    expected_offsets = batch_shape + (camera_count, tile_height, tile_width)
    if offsets.shape != expected_offsets:
        raise ValueError(f"isect_offsets must have shape {expected_offsets}")
    supplied_ids = jnp.asarray(flatten_ids, dtype=jnp.int32)
    if supplied_ids.ndim != 1:
        raise ValueError("flatten_ids must have shape [K]")
    if supplied_ids.shape[0] == 0:
        supplied_ids = jnp.full((1,), -1, dtype=jnp.int32)
    supplied_valid_count = jnp.count_nonzero(supplied_ids >= 0)
    supplied_slots = jnp.arange(supplied_ids.shape[0], dtype=jnp.int32)
    batch_count = math.prod(batch_shape) if batch_shape else 1
    image_count = batch_count * camera_count
    flat_means = means.reshape(batch_count, gaussian_count, 3)
    flat_quats = quats.reshape(batch_count, gaussian_count, 4)
    flat_scales = scales.reshape(batch_count, gaussian_count, 3)
    flat_colors = colors.reshape(image_count, gaussian_count, colors.shape[-1])
    flat_opacities = opacities.reshape(image_count, gaussian_count)
    flat_viewmats = viewmats.reshape(image_count, 4, 4)
    flat_Ks = Ks.reshape(image_count, 3, 3)
    flat_offsets = offsets.reshape(image_count, tile_height * tile_width)
    flat_rays = (
        None
        if rays is None
        else rays.reshape(image_count, image_height, image_width, 6)
    )

    if active_mask is not None:
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
        if active.shape == (gaussian_count,):
            active = jnp.broadcast_to(active, batch_shape + (gaussian_count,))
        if active.shape != batch_shape + (gaussian_count,):
            raise ValueError("active_mask must have shape [N] or [..., N]")
        flat_active = active.reshape(batch_count, gaussian_count)
    else:
        flat_active = jnp.ones((batch_count, gaussian_count), dtype=jnp.bool_)

    if backgrounds is None:
        flat_backgrounds = jnp.zeros(
            (image_count, colors.shape[-1]), colors.dtype
        )
    else:
        background_values = jnp.broadcast_to(
            jnp.asarray(backgrounds),
            batch_shape + (camera_count, colors.shape[-1]),
        )
        flat_backgrounds = background_values.reshape(
            image_count, colors.shape[-1]
        )
    flat_masks = None
    if masks is not None:
        mask_values = jnp.asarray(masks, dtype=jnp.bool_)
        if mask_values.shape != expected_offsets:
            raise ValueError("masks must have the same shape as isect_offsets")
        flat_masks = mask_values.reshape(image_count, tile_height, tile_width)

    def flatten_camera_optional(value: jax.Array | None) -> jax.Array | None:
        if value is None:
            return None
        array = jnp.asarray(value)
        if array.shape[: len(batch_shape) + 1] != batch_shape + (camera_count,):
            raise ValueError("camera parameters must have shape [..., C, ...]")
        return array.reshape((image_count,) + array.shape[len(batch_shape) + 1 :])

    flat_radial = flatten_camera_optional(radial_coeffs)
    flat_tangential = flatten_camera_optional(tangential_coeffs)
    flat_thin_prism = flatten_camera_optional(thin_prism_coeffs)
    flat_viewmats_rs = flatten_camera_optional(viewmats_rs)

    def take_optional(value: jax.Array | None, index: jax.Array):
        return None if value is None else value[index]

    def render_image(index: jax.Array):
        image_start = flat_offsets[index, 0]
        next_start = flat_offsets[jnp.minimum(index + 1, image_count - 1), 0]
        image_end = jnp.where(
            index + 1 < image_count, next_start, supplied_valid_count
        )
        image_intersection_count = jnp.maximum(image_end - image_start, 0)
        source_positions = image_start + supplied_slots
        safe_positions = jnp.clip(
            source_positions, 0, supplied_ids.shape[0] - 1
        )
        global_ids = supplied_ids[safe_positions]
        image_base = index * gaussian_count
        local_valid = (
            (supplied_slots < image_intersection_count)
            & (source_positions < supplied_valid_count)
            & (global_ids >= image_base)
            & (global_ids < image_base + gaussian_count)
        )
        local_ids = jnp.where(local_valid, global_ids - image_base, -1)
        local_offsets = (flat_offsets[index] - image_start).reshape(
            tile_height, tile_width
        )
        batch_index = index // camera_count
        image_opacities = jnp.where(
            flat_active[batch_index], flat_opacities[index], 0.0
        )
        common_arguments = (
            flat_means[batch_index],
            flat_quats[batch_index],
            flat_scales[batch_index],
            flat_colors[index],
            image_opacities,
            flat_viewmats[index],
            flat_Ks[index],
        )
        if lidar_coeffs is not None:
            return _rasterize_eval3d_lidar(
                *common_arguments,
                local_offsets,
                local_ids,
                image_intersection_count,
                lidar_coeffs,
                background=flat_backgrounds[index],
                mask=take_optional(flat_masks, index),
                viewmat_rs=take_optional(flat_viewmats_rs, index),
                rays=take_optional(flat_rays, index),
                use_hit_distance=use_hit_distance,
                return_normals=return_normals,
                flatten_index_offset=image_start,
                max_gaussians_per_tile=max_gaussians_per_tile,
                tile_batch_size=tile_batch_size,
            )
        return _rasterize_eval3d_camera(
            *common_arguments,
            image_width,
            image_height,
            tile_size,
            local_offsets,
            local_ids,
            image_intersection_count,
            background=flat_backgrounds[index],
            mask=take_optional(flat_masks, index),
            camera_model=camera_model,
            radial_coeffs=take_optional(flat_radial, index),
            tangential_coeffs=take_optional(flat_tangential, index),
            thin_prism_coeffs=take_optional(flat_thin_prism, index),
            ftheta_coeffs=ftheta_coeffs,
            external_distortion_coeffs=external_distortion_coeffs,
            rolling_shutter=_normalize_rolling_shutter(rolling_shutter),
            viewmat_rs=take_optional(flat_viewmats_rs, index),
            rays=take_optional(flat_rays, index),
            use_hit_distance=use_hit_distance,
            return_normals=return_normals,
            flatten_index_offset=image_start,
            max_gaussians_per_tile=max_gaussians_per_tile,
            tile_batch_size=tile_batch_size,
        )

    renders, alphas, info = jax.lax.map(
        render_image,
        jnp.arange(image_count, dtype=jnp.int32),
        batch_size=1,
    )
    output_prefix = batch_shape + (camera_count,)
    renders = renders.reshape(
        output_prefix + (image_height, image_width, colors.shape[-1])
    )
    alphas = alphas.reshape(output_prefix + (image_height, image_width, 1))
    info = jax.tree.map(
        lambda value: value.reshape(output_prefix + value.shape[1:]), info
    )
    return (renders, alphas, info) if return_info else (renders, alphas)


def rasterize_to_pixels_eval3d_extra(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    viewmats: jax.Array,
    Ks: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: jax.Array,
    flatten_ids: jax.Array,
    backgrounds: jax.Array | None = None,
    masks: jax.Array | None = None,
    camera_model: CameraModel = "pinhole",
    ut_params: UnscentedTransformParameters | None = None,
    rays: jax.Array | None = None,
    radial_coeffs: jax.Array | None = None,
    tangential_coeffs: jax.Array | None = None,
    thin_prism_coeffs: jax.Array | None = None,
    ftheta_coeffs: FThetaCameraDistortionParameters | None = None,
    lidar_coeffs: Any | None = None,
    external_distortion_coeffs: BivariateWindshieldModelParameters | None = None,
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: jax.Array | None = None,
    return_sample_counts: bool = False,
    use_hit_distance: bool = False,
    return_normals: bool = False,
    renderer_config: RendererConfig | None = None,
    return_last_ids: bool = True,
    unsafe_masked_tile_outputs: bool = False,
    *,
    active_mask: jax.Array | None = None,
    max_gaussians_per_tile: int = 512,
    tile_batch_size: int = 1,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
    """Eval3d rasterization with current-main's optional debug outputs.

    Masked tiles always receive safe values. This is also a valid result when
    ``unsafe_masked_tile_outputs=True``, whose upstream contract merely permits
    (but does not require) leaving those values undefined.
    """

    del unsafe_masked_tile_outputs
    renders, alphas, info = rasterize_to_pixels_eval3d(
        means,
        quats,
        scales,
        colors,
        opacities,
        viewmats,
        Ks,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        masks=masks,
        camera_model=camera_model,
        ut_params=ut_params,
        rays=rays,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        ftheta_coeffs=ftheta_coeffs,
        lidar_coeffs=lidar_coeffs,
        external_distortion_coeffs=external_distortion_coeffs,
        rolling_shutter=rolling_shutter,
        viewmats_rs=viewmats_rs,
        use_hit_distance=use_hit_distance,
        return_normals=return_normals,
        renderer_config=renderer_config,
        active_mask=active_mask,
        max_gaussians_per_tile=max_gaussians_per_tile,
        tile_batch_size=tile_batch_size,
        return_info=True,
    )
    return (
        renders,
        alphas,
        info["last_ids"] if return_last_ids else None,
        info["sample_counts"] if return_sample_counts else None,
        info["normals"] if return_normals else None,
    )


__all__ = [
    "FThetaCameraDistortionParameters",
    "FThetaPolynomialType",
    "RollingShutterType",
    "UnscentedTransformParameters",
    "compute_ut_weights",
    "fully_fused_projection_3dgut",
    "fully_fused_projection_with_ut",
    "project_camera_points",
    "project_world_points",
    "rasterize_to_pixels_eval3d",
    "rasterize_to_pixels_eval3d_extra",
    "shutter_relative_frame_time",
    "unproject_image_points",
    "world_gaussian_sigma_points",
]
