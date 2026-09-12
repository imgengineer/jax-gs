"""Readable pure-JAX implementations of current-main camera kernel ops."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from ....math import quat_to_rotmat
from ..common.pose import DynamicPose, Pose
from ..common.pose_interp import (
    interpolate_dynamic_pose,
    unpack_dynamic_pose_components,
)
from .types import (
    BivariateWindshieldDistortion,
    CameraProjection,
    ExternalDistortion,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ShutterType,
)

_NORMALIZE_NORM_SQUARED_FLOOR = 1.0e-20
_DENOMINATOR_EPSILON = 1.0e-12
_MIN_RADIAL_DISTORTION = 0.8
_MAX_RADIAL_DISTORTION = 1.2
_FTHETA_NEWTON_EPSILON = 1.0e-10
_FTHETA_IFT_DERIVATIVE_EPSILON = 1.0e-10
_FISHEYE_RAY_XY_NORM_EPSILON = 1.0e-6
_FISHEYE_IFT_DERIVATIVE_EPSILON = 1.0e-10


def _matrix_input(value, columns: int, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {array.shape}")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating-point dtype")
    return array


def _safe_nonzero(value: jax.Array, epsilon: float = _DENOMINATOR_EPSILON) -> jax.Array:
    eps = jnp.asarray(epsilon, dtype=value.dtype)
    signed_eps = jnp.where(value < 0.0, -eps, eps)
    return jnp.where(jnp.abs(value) >= eps, value, signed_eps)


def _normalize(vector: jax.Array) -> jax.Array:
    floor = jnp.asarray(_NORMALIZE_NORM_SQUARED_FLOOR, dtype=vector.dtype)
    norm = jnp.sqrt(
        jnp.maximum(jnp.sum(vector * vector, axis=-1, keepdims=True), floor)
    )
    return vector / norm


def _eval_bivariate_polynomial(
    x: jax.Array, y: jax.Array, coefficients: jax.Array, degree: int
) -> jax.Array:
    """Evaluate the triangular ``[1,x,...,y,xy,...]`` coefficient layout."""

    result = jnp.zeros_like(x)
    index = 0
    for y_power in range(degree + 1):
        for x_power in range(degree - y_power + 1):
            result = result + coefficients[index] * jnp.power(x, x_power) * jnp.power(
                y, y_power
            )
            index += 1
    return result


def _apply_bivariate_distortion(
    camera_rays: jax.Array,
    distortion: BivariateWindshieldDistortion,
    *,
    inverse: bool,
) -> jax.Array:
    reference_is_forward = distortion.reference_polynomial == 0
    use_inverse_slice = inverse == reference_is_forward
    base = 21 if use_inverse_slice else 0
    coefficients = jnp.asarray(distortion.distortion_coeffs, dtype=camera_rays.dtype)
    horizontal = coefficients[base : base + 6]
    vertical = coefficients[base + 6 : base + 21]

    normalized = _normalize(camera_rays)
    phi = jnp.arcsin(normalized[:, 0])
    theta = jnp.arcsin(normalized[:, 1])
    adjusted_phi = _eval_bivariate_polynomial(
        phi, theta, horizontal, distortion.h_poly_degree
    )
    adjusted_theta = _eval_bivariate_polynomial(
        phi, theta, vertical, distortion.v_poly_degree
    )
    x = jnp.sin(adjusted_phi)
    y = jnp.sin(adjusted_theta)
    z_squared = jnp.clip(1.0 - x * x - y * y, 0.0, 1.0)
    z_sign = jnp.where(normalized[:, 2] >= 0.0, 1.0, -1.0)
    return jnp.stack((x, y, jnp.sqrt(z_squared) * z_sign), axis=-1)


def _apply_external_distortion(
    camera_rays: jax.Array,
    external_distortion: ExternalDistortion,
    *,
    inverse: bool,
) -> jax.Array:
    if isinstance(external_distortion, NoExternalDistortion):
        return camera_rays
    if isinstance(external_distortion, BivariateWindshieldDistortion):
        return _apply_bivariate_distortion(
            camera_rays, external_distortion, inverse=inverse
        )
    raise TypeError(
        f"Unsupported external distortion class: {type(external_distortion).__name__}"
    )


def _pinhole_distortion(
    xy: jax.Array, projection: OpenCVPinholeProjection
) -> tuple[jax.Array, jax.Array, jax.Array]:
    dtype = xy.dtype
    radial_coeffs = jnp.asarray(projection.radial_coeffs, dtype=dtype)
    tangential = jnp.asarray(projection.tangential_coeffs, dtype=dtype)
    thin_prism = jnp.asarray(projection.thin_prism_coeffs, dtype=dtype)
    x, y = xy[:, 0], xy[:, 1]
    radius_squared = x * x + y * y
    radius_fourth = radius_squared * radius_squared
    radius_sixth = radius_fourth * radius_squared
    numerator = (
        1.0
        + radial_coeffs[0] * radius_squared
        + radial_coeffs[1] * radius_fourth
        + radial_coeffs[2] * radius_sixth
    )
    denominator = _safe_nonzero(
        1.0
        + radial_coeffs[3] * radius_squared
        + radial_coeffs[4] * radius_fourth
        + radial_coeffs[5] * radius_sixth
    )
    radial = numerator / denominator
    xy_product = x * y
    delta_x = (
        2.0 * tangential[0] * xy_product
        + tangential[1] * (radius_squared + 2.0 * x * x)
        + thin_prism[0] * radius_squared
        + thin_prism[1] * radius_fourth
    )
    delta_y = (
        tangential[0] * (radius_squared + 2.0 * y * y)
        + 2.0 * tangential[1] * xy_product
        + thin_prism[2] * radius_squared
        + thin_prism[3] * radius_fourth
    )
    return radial, jnp.stack((delta_x, delta_y), axis=-1), radius_squared


def _pinhole_project(
    camera_rays: jax.Array, projection: OpenCVPinholeProjection
) -> tuple[jax.Array, jax.Array]:
    dtype = camera_rays.dtype
    focal_length = jnp.asarray(projection.focal_length, dtype=dtype)
    principal_point = jnp.asarray(projection.principal_point, dtype=dtype)
    front_facing = camera_rays[:, 2] > 0.0
    inverse_z = jnp.reciprocal(_safe_nonzero(camera_rays[:, 2]))
    xy = camera_rays[:, :2] * inverse_z[:, None]
    radial, delta, radius_squared = _pinhole_distortion(xy, projection)
    normal_point = (xy * radial[:, None] + delta) * focal_length + principal_point

    width, height = projection.resolution
    resolution_length = jnp.sqrt(
        jnp.asarray(width * width + height * height, dtype=dtype)
    )
    inverse_radius = jnp.where(
        radius_squared > 0.0,
        jax.lax.rsqrt(jnp.maximum(radius_squared, jnp.finfo(dtype).tiny)),
        0.0,
    )
    boundary_point = (
        xy * (resolution_length * inverse_radius)[:, None] + principal_point
    )
    radial_in_range = (radial >= _MIN_RADIAL_DISTORTION) & (
        radial <= _MAX_RADIAL_DISTORTION
    )
    image_points = jnp.where(radial_in_range[:, None], normal_point, boundary_point)
    image_points = jnp.where(front_facing[:, None], image_points, 0.0)
    valid = (
        front_facing
        & (image_points[:, 0] >= 0.0)
        & (image_points[:, 0] < width)
        & (image_points[:, 1] >= 0.0)
        & (image_points[:, 1] < height)
    )
    return image_points, valid


def _pinhole_backproject(
    image_points: jax.Array, projection: OpenCVPinholeProjection
) -> jax.Array:
    dtype = image_points.dtype
    focal_length = jnp.asarray(projection.focal_length, dtype=dtype)
    principal_point = jnp.asarray(projection.principal_point, dtype=dtype)
    distorted_xy = (image_points - principal_point) / _safe_nonzero(focal_length)
    xy = distorted_xy
    for _ in range(10):
        radial, delta, _ = _pinhole_distortion(xy, projection)
        xy = (distorted_xy - delta) / _safe_nonzero(radial)[:, None]
    return _normalize(
        jnp.concatenate((xy, jnp.ones((xy.shape[0], 1), dtype=dtype)), axis=-1)
    )


def _poly_eval(value: jax.Array, coefficients: jax.Array, degree: int) -> jax.Array:
    result = jnp.zeros_like(value)
    for index in range(degree, -1, -1):
        result = result * value + coefficients[index]
    return result


def _poly_derivative(
    value: jax.Array, coefficients: jax.Array, degree: int
) -> jax.Array:
    result = jnp.zeros_like(value)
    for index in range(degree, 0, -1):
        result = result * value + index * coefficients[index]
    return result


def _solve_polynomial_newton(
    target: jax.Array,
    initial_guess: jax.Array,
    coefficients: jax.Array,
    degree: int,
    iterations: int,
) -> jax.Array:
    solution = initial_guess
    active = jnp.ones(target.shape, dtype=jnp.bool_)
    epsilon = jnp.asarray(_FTHETA_NEWTON_EPSILON, dtype=target.dtype)
    for _ in range(iterations):
        derivative = _poly_derivative(solution, coefficients, degree)
        can_step = active & (jnp.abs(derivative) >= epsilon)
        safe_derivative = jnp.where(can_step, derivative, 1.0)
        delta = jnp.where(
            can_step,
            (_poly_eval(solution, coefficients, degree) - target) / safe_derivative,
            0.0,
        )
        solution = jnp.where(can_step, solution - delta, solution)
        active = can_step & (jnp.abs(delta) >= epsilon)
    return jnp.maximum(solution, 0.0)


def _ftheta_project_unbounded(
    camera_rays: jax.Array, projection: FThetaProjection
) -> tuple[jax.Array, jax.Array]:
    dtype = camera_rays.dtype
    principal_point = jnp.asarray(projection.principal_point, dtype=dtype)
    forward_poly = jnp.asarray(projection.fw_poly, dtype=dtype)
    backward_poly = jnp.asarray(projection.bw_poly, dtype=dtype)
    normalized = _normalize(camera_rays)
    front_facing = camera_rays[:, 2] > 0.0
    theta = jnp.arccos(jnp.clip(normalized[:, 2], -1.0, 1.0))
    within_angle = theta <= projection.max_angle

    if projection.reference_polynomial == 0:
        radius = _poly_eval(theta, forward_poly, projection.fw_poly_degree)
    else:
        initial = _poly_eval(theta, forward_poly, projection.fw_poly_degree)
        radius_star = jax.lax.stop_gradient(
            _solve_polynomial_newton(
                theta,
                initial,
                backward_poly,
                projection.bw_poly_degree,
                projection.newton_iterations,
            )
        )
        value = _poly_eval(radius_star, backward_poly, projection.bw_poly_degree)
        derivative = _poly_derivative(
            radius_star, backward_poly, projection.bw_poly_degree
        )
        safe_derivative = jnp.where(
            jnp.abs(derivative) > _FTHETA_IFT_DERIVATIVE_EPSILON,
            derivative,
            1.0,
        )
        radius = radius_star - (value - theta) / safe_derivative

    xy_norm = jnp.linalg.norm(normalized[:, :2], axis=-1)
    safe_xy_norm = jnp.maximum(
        xy_norm, jnp.asarray(projection.min_2d_norm, dtype=dtype)
    )
    offset = normalized[:, :2] * (radius / safe_xy_norm)[:, None]
    affine = jnp.asarray(projection.A, dtype=dtype).reshape(2, 2)
    projected = offset @ affine.T + principal_point
    on_axis = xy_norm <= projection.min_2d_norm
    image_points = jnp.where(on_axis[:, None], principal_point, projected)
    projection_succeeded = front_facing & within_angle
    image_points = jnp.where(projection_succeeded[:, None], image_points, 0.0)
    return image_points, projection_succeeded


def _ftheta_points_in_frame(
    image_points: jax.Array, projection: FThetaProjection
) -> jax.Array:
    width, height = projection.resolution
    return (
        jnp.all(jnp.isfinite(image_points), axis=-1)
        & (image_points[:, 0] >= 0.0)
        & (image_points[:, 0] < width)
        & (image_points[:, 1] >= 0.0)
        & (image_points[:, 1] < height)
    )


def _ftheta_project(
    camera_rays: jax.Array, projection: FThetaProjection
) -> tuple[jax.Array, jax.Array]:
    image_points, projection_succeeded = _ftheta_project_unbounded(
        camera_rays, projection
    )
    valid = projection_succeeded & _ftheta_points_in_frame(image_points, projection)
    return image_points, valid


def _ftheta_backproject(
    image_points: jax.Array, projection: FThetaProjection
) -> jax.Array:
    dtype = image_points.dtype
    principal_point = jnp.asarray(projection.principal_point, dtype=dtype)
    forward_poly = jnp.asarray(projection.fw_poly, dtype=dtype)
    backward_poly = jnp.asarray(projection.bw_poly, dtype=dtype)
    inverse_affine = jnp.asarray(projection.Ainv, dtype=dtype).reshape(2, 2)
    transformed = (image_points - principal_point) @ inverse_affine.T
    radius = jnp.linalg.norm(transformed, axis=-1)
    on_axis = radius <= projection.min_2d_norm

    if projection.reference_polynomial == 1:
        theta = _poly_eval(radius, backward_poly, projection.bw_poly_degree)
    else:
        initial = _poly_eval(radius, backward_poly, projection.bw_poly_degree)
        theta_star = jax.lax.stop_gradient(
            _solve_polynomial_newton(
                radius,
                initial,
                forward_poly,
                projection.fw_poly_degree,
                projection.newton_iterations,
            )
        )
        value = _poly_eval(theta_star, forward_poly, projection.fw_poly_degree)
        derivative = _poly_derivative(
            theta_star, forward_poly, projection.fw_poly_degree
        )
        safe_derivative = jnp.where(
            jnp.abs(derivative) > _FTHETA_IFT_DERIVATIVE_EPSILON,
            derivative,
            1.0,
        )
        theta = theta_star - (value - radius) / safe_derivative

    safe_radius = jnp.maximum(radius, jnp.asarray(projection.min_2d_norm, dtype=dtype))
    xy = transformed * (jnp.sin(theta) / safe_radius)[:, None]
    raw_ray = jnp.concatenate((xy, jnp.cos(theta)[:, None]), axis=-1)
    ray = _normalize(raw_ray)
    optical_axis = jnp.asarray((0.0, 0.0, 1.0), dtype=dtype)
    return jnp.where(on_axis[:, None], optical_axis, ray)


def _fisheye_forward_polynomial(theta: jax.Array, coefficients: jax.Array) -> jax.Array:
    theta_squared = theta * theta
    factor = 1.0 + theta_squared * (
        coefficients[0]
        + theta_squared
        * (
            coefficients[1]
            + theta_squared * (coefficients[2] + theta_squared * coefficients[3])
        )
    )
    return theta * factor


def _fisheye_forward_derivative(theta: jax.Array, coefficients: jax.Array) -> jax.Array:
    theta_squared = theta * theta
    return 1.0 + theta_squared * (
        3.0 * coefficients[0]
        + theta_squared
        * (
            5.0 * coefficients[1]
            + theta_squared
            * (7.0 * coefficients[2] + theta_squared * 9.0 * coefficients[3])
        )
    )


def _solve_fisheye_newton(
    delta: jax.Array, projection: OpenCVFisheyeProjection, dtype
) -> jax.Array:
    coefficients = jnp.asarray(projection.forward_poly, dtype=dtype)
    initial_factor = jax.lax.stop_gradient(
        jnp.asarray(projection.approx_backward_factor[0], dtype=dtype)
    )
    solution = initial_factor * delta
    for _ in range(projection.newton_iterations):
        derivative = _fisheye_forward_derivative(solution, coefficients)
        residual = _fisheye_forward_polynomial(solution, coefficients) - delta
        solution = solution - residual / derivative
    return solution


def _fisheye_project(
    camera_rays: jax.Array, projection: OpenCVFisheyeProjection
) -> tuple[jax.Array, jax.Array]:
    dtype = camera_rays.dtype
    principal_point = jnp.asarray(projection.principal_point, dtype=dtype)
    focal_length = jnp.asarray(projection.focal_length, dtype=dtype)
    coefficients = jnp.asarray(projection.forward_poly, dtype=dtype)
    front_facing = camera_rays[:, 2] > 0.0
    raw_xy_norm = jnp.linalg.norm(camera_rays[:, :2], axis=-1)
    xy_norm = jnp.maximum(
        raw_xy_norm, jnp.asarray(_FISHEYE_RAY_XY_NORM_EPSILON, dtype=dtype)
    )
    theta = jnp.minimum(
        jnp.arctan2(xy_norm, camera_rays[:, 2]),
        jnp.asarray(projection.max_angle, dtype=dtype),
    )
    delta = _fisheye_forward_polynomial(theta, coefficients)
    scaled = camera_rays[:, :2] * (delta / xy_norm)[:, None] * focal_length
    projected = scaled + principal_point
    width, height = projection.resolution
    out_of_bounds = (
        (projected[:, 0] < 0.0)
        | (projected[:, 0] >= width)
        | (projected[:, 1] < 0.0)
        | (projected[:, 1] >= height)
    )
    valid = front_facing & ~out_of_bounds
    image_points = jnp.where(valid[:, None], projected, 0.0)
    return image_points, valid


def _fisheye_backproject(
    image_points: jax.Array, projection: OpenCVFisheyeProjection
) -> jax.Array:
    dtype = image_points.dtype
    principal_point = jnp.asarray(projection.principal_point, dtype=dtype)
    focal_length = jnp.asarray(projection.focal_length, dtype=dtype)
    coefficients = jnp.asarray(projection.forward_poly, dtype=dtype)
    normalized = (image_points - principal_point) / focal_length
    delta = jnp.linalg.norm(normalized, axis=-1)
    on_axis = delta <= projection.min_2d_norm
    theta_star = jax.lax.stop_gradient(_solve_fisheye_newton(delta, projection, dtype))
    derivative = _fisheye_forward_derivative(theta_star, coefficients)
    good_derivative = jnp.abs(derivative) > _FISHEYE_IFT_DERIVATIVE_EPSILON
    # The forward returns Newton's final iterate exactly. This zero-valued
    # surrogate gives JAX the same implicit-function gradient as upstream.
    surrogate = jnp.where(
        good_derivative,
        (delta - _fisheye_forward_polynomial(theta_star, coefficients))
        / jax.lax.stop_gradient(derivative),
        0.0,
    )
    theta = theta_star + surrogate - jax.lax.stop_gradient(surrogate)
    safe_delta = jnp.maximum(delta, jnp.asarray(projection.min_2d_norm, dtype=dtype))
    xy = normalized * (jnp.sin(theta) / safe_delta)[:, None]
    raw_ray = jnp.concatenate((xy, jnp.cos(theta)[:, None]), axis=-1)
    ray = _normalize(raw_ray)
    optical_axis = jnp.asarray((0.0, 0.0, 1.0), dtype=dtype)
    return jnp.where(on_axis[:, None], optical_axis, ray)


def _check_pair(
    projection: CameraProjection, external_distortion: ExternalDistortion
) -> None:
    if external_distortion is None:
        raise TypeError(
            "external_distortion=None is not supported; pass "
            "NoExternalDistortion() explicitly"
        )
    if not isinstance(
        projection,
        (OpenCVPinholeProjection, FThetaProjection, OpenCVFisheyeProjection),
    ) or not isinstance(
        external_distortion,
        (NoExternalDistortion, BivariateWindshieldDistortion),
    ):
        raise TypeError(
            "Unsupported camera projection/distortion pair: "
            f"({type(projection).__name__}, {type(external_distortion).__name__})"
        )


def camera_rays_to_image_points(
    camera_rays: jax.Array,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    *,
    allow_device_transfer: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Project ``(N, 3)`` camera rays and return points plus validity flags."""

    del (
        allow_device_transfer
    )  # JAX placement is explicit and has no implicit CUDA copy.
    _check_pair(projection, external_distortion)
    rays = _matrix_input(camera_rays, 3, "camera_rays")
    rays = _apply_external_distortion(rays, external_distortion, inverse=False)
    if isinstance(projection, OpenCVPinholeProjection):
        return _pinhole_project(rays, projection)
    if isinstance(projection, FThetaProjection):
        return _ftheta_project(rays, projection)
    if isinstance(projection, OpenCVFisheyeProjection):
        return _fisheye_project(rays, projection)
    raise AssertionError("unreachable projection dispatch")


def image_points_to_camera_rays(
    image_points: jax.Array,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    *,
    allow_device_transfer: bool = False,
) -> jax.Array:
    """Back-project ``(N, 2)`` image coordinates to unit camera rays."""

    del allow_device_transfer
    _check_pair(projection, external_distortion)
    points = _matrix_input(image_points, 2, "image_points")
    if isinstance(projection, OpenCVPinholeProjection):
        rays = _pinhole_backproject(points, projection)
    elif isinstance(projection, FThetaProjection):
        rays = _ftheta_backproject(points, projection)
    elif isinstance(projection, OpenCVFisheyeProjection):
        rays = _fisheye_backproject(points, projection)
    else:
        raise AssertionError("unreachable projection dispatch")  # noqa: TRY004
    rays = _apply_external_distortion(rays, external_distortion, inverse=True)
    return _normalize(rays)


def generate_image_points(
    resolution: tuple[int, int],
    *,
    device: jax.Device | str | None = None,
    allow_device_transfer: bool = False,
) -> jax.Array:
    """Return an ``(H, W, 2)`` row-major grid of pixel-center coordinates."""

    del allow_device_transfer
    width, height = int(resolution[0]), int(resolution[1])
    if width < 0 or height < 0:
        raise ValueError("resolution dimensions must be non-negative")
    x = jnp.arange(width, dtype=jnp.float32) + 0.5
    y = jnp.arange(height, dtype=jnp.float32) + 0.5
    xx, yy = jnp.meshgrid(x, y, indexing="xy")
    points = jnp.stack((xx, yy), axis=-1)
    if device is None:
        return points
    if isinstance(device, str):
        backend = "gpu" if device.startswith("cuda") else device.split(":", 1)[0]
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        devices = jax.devices(backend)
        if index >= len(devices):
            raise ValueError(
                f"device index {index} is unavailable for backend {backend}"
            )
        device = devices[index]
    return jax.device_put(points, device)


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


def _timestamps_from_relative_time(
    relative_time: jax.Array, start_timestamp_us: int, end_timestamp_us: int
) -> jax.Array:
    # JAX x64 is deliberately optional in this repository. Small timestamps
    # retain upstream values in int32 mode; enabling x64 restores int64 range.
    timestamp_dtype = jnp.int64 if jax.config.x64_enabled else jnp.int32
    start = jnp.asarray(start_timestamp_us, dtype=relative_time.dtype)
    duration = jnp.asarray(
        end_timestamp_us - start_timestamp_us, dtype=relative_time.dtype
    )
    return (start + relative_time * duration).astype(timestamp_dtype)


def _projection_at_resolution(
    projection: CameraProjection, resolution: tuple[int, int]
) -> CameraProjection:
    resolution = (int(resolution[0]), int(resolution[1]))
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError("resolution width and height must be positive")
    return replace(projection, resolution=resolution)


def _world_to_camera_points(
    world_points: jax.Array,
    translations: jax.Array,
    rotations: jax.Array,
) -> jax.Array:
    rotation_matrices = quat_to_rotmat(rotations)
    relative = world_points - translations
    return jnp.einsum("nij,nj->ni", jnp.swapaxes(rotation_matrices, -1, -2), relative)


def _camera_to_world_directions(
    camera_rays: jax.Array, rotations: jax.Array
) -> jax.Array:
    return jnp.einsum("nij,nj->ni", quat_to_rotmat(rotations), camera_rays)


def relative_frame_times(
    image_points: jax.Array,
    resolution: tuple[int, int],
    shutter_type: ShutterType | int,
) -> jax.Array:
    """Map image coordinates to normalized rolling-shutter readout times."""

    points = _matrix_input(image_points, 2, "image_points")
    width, height = int(resolution[0]), int(resolution[1])
    safe_width = max(width, 2) - 1
    safe_height = max(height, 2) - 1
    try:
        shutter = ShutterType(shutter_type)
    except ValueError as error:
        raise ValueError(
            f"Unsupported ShutterType: {shutter_type!r} (allowed: 1..5)"
        ) from error
    if shutter == ShutterType.ROLLING_TOP_TO_BOTTOM:
        return jnp.clip(jnp.floor(points[:, 1]), 0, safe_height) / safe_height
    if shutter == ShutterType.ROLLING_BOTTOM_TO_TOP:
        return jnp.clip(height - jnp.ceil(points[:, 1]), 0, safe_height) / safe_height
    if shutter == ShutterType.ROLLING_LEFT_TO_RIGHT:
        return jnp.clip(jnp.floor(points[:, 0]), 0, safe_width) / safe_width
    if shutter == ShutterType.ROLLING_RIGHT_TO_LEFT:
        return jnp.clip(width - jnp.ceil(points[:, 0]), 0, safe_width) / safe_width
    return jnp.zeros((points.shape[0],), dtype=points.dtype)


def mean_pose_to_static_pose(
    dynamic_pose: DynamicPose,
    device: jax.Device | None = None,
    dtype=jnp.float32,
) -> Pose:
    """Return the LERP/SLERP midpoint of a two-pose trajectory."""

    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, device=device, dtype=dtype
    )
    pose_t, pose_r = interpolate_dynamic_pose(
        start_t, start_r, end_t, end_r, jnp.asarray((0.5,), dtype=dtype)
    )
    return Pose(translation=pose_t[0], rotation=pose_r[0])


def project_world_points_mean_pose(
    world_points: jax.Array,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    dynamic_pose: DynamicPose,
    resolution: tuple[int, int],
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_valid_flags: bool = False,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[
    jax.Array,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
    """Project world points with the sensor pose at frame midpoint."""

    del allow_device_transfer
    points = _matrix_input(world_points, 3, "world_points")
    projection = _projection_at_resolution(projection, resolution)
    start, end = _timestamp_bounds(start_timestamp_us, end_timestamp_us)
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, dtype=points.dtype
    )
    relative_time = jnp.asarray((0.5,), dtype=points.dtype)
    pose_t, pose_r = interpolate_dynamic_pose(
        start_t, start_r, end_t, end_r, relative_time
    )
    pose_t = jnp.broadcast_to(pose_t, (points.shape[0], 3))
    pose_r = jnp.broadcast_to(pose_r, (points.shape[0], 4))
    camera_points = _world_to_camera_points(points, pose_t, pose_r)
    image_points, valid_flags = camera_rays_to_image_points(
        camera_points, projection, external_distortion
    )
    timestamps = _timestamps_from_relative_time(
        jnp.full((points.shape[0],), 0.5, dtype=points.dtype), start, end
    )
    return (
        image_points,
        valid_flags if return_valid_flags else None,
        timestamps if return_timestamps else None,
        pose_t if return_poses else None,
        pose_r if return_poses else None,
    )


def project_world_points_shutter_pose(
    world_points: jax.Array,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType | int,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    max_iterations: int = 10,
    stop_mean_error_px: float = 0.001,
    stop_delta_mean_error_px: float = 0.00001,
    initial_relative_time: float = 0.5,
    return_valid_flags: bool = False,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[
    jax.Array,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
    """Project world points with a fixed-iteration rolling-shutter solve."""

    del allow_device_transfer
    points = _matrix_input(world_points, 3, "world_points")
    projection = _projection_at_resolution(projection, resolution)
    shutter = ShutterType(shutter_type)
    max_iterations = int(max_iterations)
    if not 1 <= max_iterations <= 32:
        raise ValueError("max_iterations must be in [1, 32]")
    start, end = _timestamp_bounds(start_timestamp_us, end_timestamp_us)
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, dtype=points.dtype
    )

    count = points.shape[0]
    relative_time = jnp.full((count,), initial_relative_time, dtype=points.dtype)
    previous_image_points = jnp.zeros((count, 2), dtype=points.dtype)
    image_points = jnp.zeros((count, 2), dtype=points.dtype)
    valid_flags = jnp.zeros((count,), dtype=jnp.bool_)
    pose_t = jnp.broadcast_to(start_t, (count, 3))
    pose_r = jnp.broadcast_to(start_r, (count, 4))
    active = jnp.ones((count,), dtype=jnp.bool_)

    for iteration in range(max_iterations):
        candidate_t, candidate_r = interpolate_dynamic_pose(
            start_t, start_r, end_t, end_r, relative_time
        )
        camera_points = _world_to_camera_points(points, candidate_t, candidate_r)
        if isinstance(projection, FThetaProjection):
            distorted_rays = _apply_external_distortion(
                camera_points, external_distortion, inverse=False
            )
            candidate_points, candidate_valid = _ftheta_project_unbounded(
                distorted_rays, projection
            )
        else:
            candidate_points, candidate_valid = camera_rays_to_image_points(
                camera_points, projection, external_distortion
            )
        behind_camera = camera_points[:, 2] <= 0.0
        candidate_points = jnp.where(
            behind_camera[:, None], image_points, candidate_points
        )
        candidate_valid = candidate_valid & ~behind_camera

        image_points = jnp.where(active[:, None], candidate_points, image_points)
        valid_flags = jnp.where(active, candidate_valid, valid_flags)
        pose_t = jnp.where(active[:, None], candidate_t, pose_t)
        pose_r = jnp.where(active[:, None], candidate_r, pose_r)

        next_relative_time = jax.lax.stop_gradient(
            relative_frame_times(candidate_points, resolution, shutter)
        )
        pixel_error = jnp.linalg.norm(candidate_points - previous_image_points, axis=-1)
        delta_converged = (iteration > 0) & (pixel_error < stop_delta_mean_error_px)
        approximate_error = jnp.abs(next_relative_time - relative_time) * max(
            resolution
        )
        mean_converged = approximate_error < stop_mean_error_px
        continue_iteration = (
            active
            & candidate_valid
            & ~delta_converged
            & ~mean_converged
            & (shutter != ShutterType.GLOBAL)
        )
        previous_image_points = jnp.where(
            continue_iteration[:, None], candidate_points, previous_image_points
        )
        relative_time = jnp.where(continue_iteration, next_relative_time, relative_time)
        active = continue_iteration

    if isinstance(projection, FThetaProjection):
        valid_flags = valid_flags & _ftheta_points_in_frame(image_points, projection)
    timestamps = _timestamps_from_relative_time(relative_time, start, end)
    return (
        image_points,
        valid_flags if return_valid_flags else None,
        timestamps if return_timestamps else None,
        pose_t if return_poses else None,
        pose_r if return_poses else None,
    )


def image_points_to_world_rays_static_pose(
    image_points: jax.Array,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    pose: Pose,
    *,
    timestamp_us: int | None = None,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array | None]:
    """Back-project image coordinates through one sensor-to-world pose."""

    del allow_device_transfer
    points = _matrix_input(image_points, 2, "image_points")
    camera_rays = image_points_to_camera_rays(points, projection, external_distortion)
    count = points.shape[0]
    pose_t = jnp.broadcast_to(
        jnp.asarray(pose.translation, dtype=points.dtype), (count, 3)
    )
    pose_r = jnp.broadcast_to(
        jnp.asarray(pose.rotation, dtype=points.dtype), (count, 4)
    )
    directions = _camera_to_world_directions(camera_rays, pose_r)
    world_rays = jnp.concatenate((pose_t, directions), axis=-1)
    timestamp = 0 if timestamp_us is None else int(timestamp_us)
    timestamps = _timestamps_from_relative_time(
        jnp.zeros((count,), dtype=points.dtype), timestamp, timestamp
    )
    return (
        world_rays,
        timestamps if return_timestamps else None,
        pose_t if return_poses else None,
        pose_r if return_poses else None,
    )


def image_points_to_world_rays_shutter_pose(
    image_points: jax.Array,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType | int,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array | None]:
    """Back-project rays at the pose implied by each pixel's scan time."""

    del allow_device_transfer
    points = _matrix_input(image_points, 2, "image_points")
    projection = _projection_at_resolution(projection, resolution)
    start, end = _timestamp_bounds(start_timestamp_us, end_timestamp_us)
    relative_time = jax.lax.stop_gradient(
        relative_frame_times(points, resolution, shutter_type)
    )
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, dtype=points.dtype
    )
    pose_t, pose_r = interpolate_dynamic_pose(
        start_t, start_r, end_t, end_r, relative_time
    )
    camera_rays = image_points_to_camera_rays(points, projection, external_distortion)
    directions = _camera_to_world_directions(camera_rays, pose_r)
    world_rays = jnp.concatenate((pose_t, directions), axis=-1)
    timestamps = _timestamps_from_relative_time(relative_time, start, end)
    return (
        world_rays,
        timestamps if return_timestamps else None,
        pose_t if return_poses else None,
        pose_r if return_poses else None,
    )


def pixel_grid_to_world_rays_shutter_pose(
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType | int,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array | None]:
    """Generate rolling-shutter world rays for a complete pixel grid."""

    points = generate_image_points(resolution).reshape((-1, 2))
    return image_points_to_world_rays_shutter_pose(
        points,
        projection,
        external_distortion,
        resolution,
        shutter_type,
        dynamic_pose,
        start_timestamp_us=start_timestamp_us,
        end_timestamp_us=end_timestamp_us,
        return_timestamps=return_timestamps,
        return_poses=return_poses,
        allow_device_transfer=allow_device_transfer,
    )


__all__ = [  # noqa: RUF022 - preserve the public compatibility order
    "camera_rays_to_image_points",
    "generate_image_points",
    "image_points_to_camera_rays",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
    "interpolate_dynamic_pose",
    "mean_pose_to_static_pose",
    "pixel_grid_to_world_rays_shutter_pose",
    "project_world_points_mean_pose",
    "project_world_points_shutter_pose",
    "relative_frame_times",
    "unpack_dynamic_pose_components",
]
