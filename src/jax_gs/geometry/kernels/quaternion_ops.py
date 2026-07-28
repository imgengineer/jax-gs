"""Pure-JAX quaternion operators in gsplat's ``xyzw`` convention."""

from __future__ import annotations

from numbers import Real

import jax
import jax.numpy as jnp


SLERP_SMALL_ANGLE_DOT_THRESHOLD = 0.9995

_FLOAT_DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64))


def _expect_array(name: str, value: object) -> jax.Array:
    if not isinstance(value, jax.Array):
        raise TypeError(f"{name} must be a JAX array, got {type(value).__name__}")
    return value


def _require_float(name: str, value: jax.Array) -> None:
    if value.dtype not in _FLOAT_DTYPES:
        raise TypeError(
            f"{name} must have dtype float32 or float64, got {value.dtype}"
        )


def _require_last_dim(name: str, value: jax.Array, size: int) -> None:
    if value.ndim < 1 or value.shape[-1] != size:
        raise ValueError(
            f"{name} must have last dimension {size}, got shape {value.shape}"
        )


def _validate_quaternion(name: str, value: object) -> jax.Array:
    array = _expect_array(name, value)
    _require_float(name, array)
    _require_last_dim(name, array, 4)
    return array


def _validate_vector(name: str, value: object) -> jax.Array:
    array = _expect_array(name, value)
    _require_float(name, array)
    _require_last_dim(name, array, 3)
    return array


def _require_same_dtype(
    first_name: str,
    first: jax.Array,
    second_name: str,
    second: jax.Array,
) -> None:
    if first.dtype != second.dtype:
        raise TypeError(
            f"{second_name} must have the same dtype as {first_name} "
            f"({first.dtype}); got {second.dtype}"
        )


def _identity_like(quaternion: jax.Array) -> jax.Array:
    identity = jnp.zeros_like(quaternion)
    return identity.at[..., 3].set(1)


def _normalization_epsilon(dtype: jnp.dtype) -> float:
    return 1.0e-12 if dtype == jnp.dtype(jnp.float64) else 1.0e-7


def _quat_multiply_raw(q1: jax.Array, q2: jax.Array) -> jax.Array:
    vector1, scalar1 = q1[..., :3], q1[..., 3:4]
    vector2, scalar2 = q2[..., :3], q2[..., 3:4]
    vector = (
        scalar1 * vector2
        + scalar2 * vector1
        + jnp.cross(vector1, vector2)
    )
    scalar = scalar1 * scalar2 - jnp.sum(
        vector1 * vector2, axis=-1, keepdims=True
    )
    return jnp.concatenate((vector, scalar), axis=-1)


def _quat_rotate_vector_raw(quaternion: jax.Array, vector: jax.Array) -> jax.Array:
    q_vector = quaternion[..., :3]
    q_scalar = quaternion[..., 3:4]
    first_cross = jnp.cross(q_vector, vector)
    second_cross = jnp.cross(q_vector, first_cross)
    return vector + 2 * (q_scalar * first_cross + second_cross)


def _blend_parameter(q_flat: jax.Array, t: float | jax.Array) -> jax.Array:
    count = q_flat.shape[0]
    if isinstance(t, Real):
        return jnp.full((count,), float(t), dtype=q_flat.dtype)
    t_array = _expect_array("t", t)
    if not jnp.issubdtype(t_array.dtype, jnp.floating):
        raise TypeError(
            "t must be a floating-point array when array-valued, "
            f"got {t_array.dtype}"
        )
    t_flat = jnp.asarray(t_array, dtype=q_flat.dtype).reshape(-1)
    if t_flat.size == 1:
        return jnp.broadcast_to(t_flat, (count,))
    if t_flat.size != count:
        raise ValueError(
            f"t must have batch size 1 or {count} when array-valued; "
            f"got {t_flat.size}"
        )
    return t_flat


def _slerp_rows(q1: jax.Array, q2: jax.Array, t: jax.Array) -> jax.Array:
    """SLERP rows with upstream hemisphere and NLERP fallback semantics."""

    t = t[..., None]
    dot = jnp.sum(q1 * q2, axis=-1, keepdims=True)
    q2_short = jnp.where(dot < 0, -q2, q2)
    cosine = jnp.clip(
        jnp.sum(q1 * q2_short, axis=-1, keepdims=True), -1, 1
    )
    use_lerp = cosine > SLERP_SMALL_ANGLE_DOT_THRESHOLD

    linear_raw = (1 - t) * q1 + t * q2_short
    linear_norm_sq = jnp.sum(linear_raw * linear_raw, axis=-1, keepdims=True)
    linear_safe_norm_sq = jnp.where(linear_norm_sq > 0, linear_norm_sq, 1)
    linear = linear_raw * jax.lax.rsqrt(linear_safe_norm_sq)

    # The unused acos branch must stay finite at cosine == 1 for JAX's VJP.
    spherical_cosine = jnp.where(use_lerp, 0, cosine)
    theta = jnp.arccos(spherical_cosine)
    sin_theta = jnp.sin(theta)
    weight1 = jnp.sin((1 - t) * theta) / sin_theta
    weight2 = jnp.sin(t * theta) / sin_theta
    spherical = weight1 * q1 + weight2 * q2_short
    return jnp.where(use_lerp, linear, spherical)


def quat_normalize_safe(quat: jax.Array) -> jax.Array:
    """Normalize quaternion(s), mapping near-zero rows to ``[0, 0, 0, 1]``."""

    quat = _validate_quaternion("quat", quat)
    norm_sq = jnp.sum(quat * quat, axis=-1, keepdims=True)
    small = jnp.abs(norm_sq) < _normalization_epsilon(quat.dtype)
    safe_norm_sq = jnp.where(small, 1, norm_sq)
    normalized = quat * jax.lax.rsqrt(safe_norm_sq)
    return jnp.where(small, _identity_like(quat), normalized)


def quat_conjugate(q: jax.Array) -> jax.Array:
    """Return quaternion conjugate(s) in ``xyzw`` storage order."""

    q = _validate_quaternion("q", q)
    return jnp.concatenate((-q[..., :3], q[..., 3:4]), axis=-1)


def quat_inverse(q: jax.Array) -> jax.Array:
    """Return the inverse of unit quaternion(s)."""

    return quat_conjugate(q)


def quat_multiply(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Hamilton product ``q1 * q2`` without implicit batch broadcasting."""

    q1 = _validate_quaternion("q1", q1)
    q2 = _validate_quaternion("q2", q2)
    _require_same_dtype("q1", q1, "q2", q2)
    if q1.shape != q2.shape:
        raise ValueError(
            f"q1 and q2 must have the same shape; got {q1.shape} vs {q2.shape}"
        )
    return _quat_multiply_raw(q1, q2)


def quat_rotate_vector(q: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate vector(s) with unit quaternion(s), without batch broadcasting."""

    q = _validate_quaternion("q", q)
    v = _validate_vector("v", v)
    _require_same_dtype("q", q, "v", v)
    if q.shape[:-1] != v.shape[:-1]:
        raise ValueError(
            "q and v must share the same batch dimensions; got leading shape "
            f"{q.shape[:-1]} vs {v.shape[:-1]}"
        )
    return _quat_rotate_vector_raw(q, v)


def quat_to_matrix(quat: jax.Array) -> jax.Array:
    """Convert quaternion(s) to row-major 3-by-3 rotation matrices."""

    quat = quat_normalize_safe(quat)
    x, y, z, w = jnp.moveaxis(quat, -1, 0)
    return jnp.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def quat_slerp(
    q1: jax.Array, q2: jax.Array, t: float | jax.Array
) -> jax.Array:
    """Spherically interpolate matching quaternion batches."""

    q1 = _validate_quaternion("q1", q1)
    q2 = _validate_quaternion("q2", q2)
    _require_same_dtype("q1", q1, "q2", q2)
    if q1.shape != q2.shape:
        raise ValueError(
            f"q1 and q2 must have the same shape; got {q1.shape} vs {q2.shape}"
        )
    original_shape = q1.shape
    q1_flat = q1.reshape((-1, 4))
    q2_flat = q2.reshape((-1, 4))
    t_flat = _blend_parameter(q1_flat, t)
    return _slerp_rows(q1_flat, q2_flat, t_flat).reshape(original_shape)


def quat_lerp(q1: jax.Array, q2: jax.Array, t: float) -> jax.Array:
    """Normalized linear interpolation with shortest-hemisphere correction."""

    if not isinstance(t, Real):
        raise TypeError(
            "quat_lerp expects a scalar Python float or int for t; "
            "use quat_slerp for array interpolation parameters"
        )
    q1 = _validate_quaternion("q1", q1)
    q2 = _validate_quaternion("q2", q2)
    _require_same_dtype("q1", q1, "q2", q2)
    if q1.shape != q2.shape:
        raise ValueError(
            f"q1 and q2 must have the same shape; got {q1.shape} vs {q2.shape}"
        )
    dot = jnp.sum(q1 * q2, axis=-1, keepdims=True)
    q2_short = jnp.where(dot < 0, -q2, q2)
    raw = (1 - float(t)) * q1 + float(t) * q2_short
    return raw * jax.lax.rsqrt(jnp.sum(raw * raw, axis=-1, keepdims=True))


def quat_from_axis_angle(axis: jax.Array, angle: jax.Array) -> jax.Array:
    """Convert axis/angle rows to ``xyzw`` quaternions.

    As upstream, the axis is not normalized by this operation.
    """

    axis = _validate_vector("axis", axis)
    angle = _expect_array("angle", angle)
    _require_float("angle", angle)
    _require_same_dtype("axis", axis, "angle", angle)
    axis_flat = axis.reshape((-1, 3))
    row_count = axis_flat.shape[0]
    if angle.ndim == 0:
        if row_count != 1:
            raise ValueError(
                "axis-angle batch mismatch: scalar angle requires axis with "
                f"exactly one quaternion row, got {row_count} rows from axis"
            )
        angle_flat = angle.reshape((1,))
    else:
        angle_flat = angle.reshape((-1,))
        if angle_flat.shape[0] != row_count:
            raise ValueError(
                "axis and angle batch mismatch: "
                f"{row_count} axis rows vs {angle_flat.shape[0]} angle rows"
            )
    half_angle = 0.5 * angle_flat
    quaternion = jnp.concatenate(
        (
            axis_flat * jnp.sin(half_angle)[:, None],
            jnp.cos(half_angle)[:, None],
        ),
        axis=-1,
    )
    return quaternion.reshape(axis.shape[:-1] + (4,))


def quat_angular_distance(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Return geodesic rotation distance in radians."""

    q1 = _validate_quaternion("q1", q1)
    q2 = _validate_quaternion("q2", q2)
    _require_same_dtype("q1", q1, "q2", q2)
    if q1.shape != q2.shape:
        raise ValueError(
            f"q1 and q2 must have the same shape; got {q1.shape} vs {q2.shape}"
        )
    q1 = q1 / jnp.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / jnp.linalg.norm(q2, axis=-1, keepdims=True)
    cosine = jnp.clip(jnp.abs(jnp.sum(q1 * q2, axis=-1)), 0, 1)
    return 2 * jnp.arccos(cosine)


def quat_identity(
    shape: tuple[int, ...] = (),
    dtype: jnp.dtype = jnp.float32,
    *,
    device: jax.Device | None = None,
) -> jax.Array:
    """Create identity quaternion(s).

    Unlike PyTorch upstream, JAX has a well-defined default device, so the
    keyword is optional. Passing a JAX device places the result explicitly.
    """

    dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError(f"dtype must be a floating-point dtype, got {dtype}")
    quaternion = jnp.zeros(tuple(shape) + (4,), dtype=dtype)
    quaternion = quaternion.at[..., 3].set(1)
    return jax.device_put(quaternion, device) if device is not None else quaternion


def _so3_log(quaternion: jax.Array) -> jax.Array:
    norm_sq = jnp.sum(quaternion * quaternion, axis=-1, keepdims=True)
    needs_normalization = jnp.abs(norm_sq - 1) > 1.0e-12
    safe_norm_sq = jnp.where(norm_sq > 0, norm_sq, 1)
    normalized = quaternion * jax.lax.rsqrt(safe_norm_sq)
    quaternion = jnp.where(needs_normalization, normalized, quaternion)

    vector = jnp.where(quaternion[..., 3:4] < 0, -quaternion[..., :3], quaternion[..., :3])
    scalar_abs = jnp.abs(quaternion[..., 3:4])
    vector_norm_sq = jnp.sum(vector * vector, axis=-1, keepdims=True)
    small = vector_norm_sq < 1.0e-12
    series_scale = 2 - vector_norm_sq / 3 + 2 * vector_norm_sq**2 / 45

    safe_vector_norm_sq = jnp.where(small, 1, vector_norm_sq)
    vector_norm = jnp.sqrt(safe_vector_norm_sq)
    angle = 2 * jnp.arctan2(vector_norm, scalar_abs)
    regular = vector * (angle / vector_norm)
    return jnp.where(small, vector * series_scale, regular)


def _so3_exp(rotation_vector: jax.Array) -> jax.Array:
    angle_sq = jnp.sum(rotation_vector * rotation_vector, axis=-1, keepdims=True)
    small = angle_sq < 1.0e-6

    quarter_angle_sq = 0.25 * angle_sq
    series_scale = 0.5 - quarter_angle_sq / 12 + angle_sq**2 / 2880
    series_scalar = 1 - quarter_angle_sq * (
        1 - angle_sq / 24 + angle_sq**2 / 720
    )

    safe_angle_sq = jnp.where(small, 1, angle_sq)
    angle = jnp.sqrt(safe_angle_sq)
    half_angle = 0.5 * angle
    regular_scale = jnp.sin(half_angle) / angle
    regular_scalar = jnp.cos(half_angle)

    vector = jnp.where(small, rotation_vector * series_scale, rotation_vector * regular_scale)
    scalar = jnp.where(small, series_scalar, regular_scalar)
    return jnp.concatenate((vector, scalar), axis=-1)


def quat_manifold_interp(
    q1: jax.Array, q2: jax.Array, t: float | jax.Array
) -> jax.Array:
    """Evaluate ``q1 * exp(t * log(conjugate(q1) * q2))`` row-wise."""

    q1 = _validate_quaternion("q1", q1)
    q2 = _validate_quaternion("q2", q2)
    _require_same_dtype("q1", q1, "q2", q2)
    if q1.shape != q2.shape:
        raise ValueError(
            f"q1 and q2 must have the same shape; got {q1.shape} vs {q2.shape}"
        )
    original_shape = q1.shape
    q1_flat = q1.reshape((-1, 4))
    q2_flat = q2.reshape((-1, 4))
    t_flat = _blend_parameter(q1_flat, t)
    relative = _quat_multiply_raw(
        jnp.concatenate((-q1_flat[:, :3], q1_flat[:, 3:4]), axis=-1),
        q2_flat,
    )
    increment = _so3_exp(t_flat[:, None] * _so3_log(relative))
    return _quat_multiply_raw(q1_flat, increment).reshape(original_shape)


__all__ = [
    "SLERP_SMALL_ANGLE_DOT_THRESHOLD",
    "quat_angular_distance",
    "quat_conjugate",
    "quat_from_axis_angle",
    "quat_identity",
    "quat_inverse",
    "quat_lerp",
    "quat_manifold_interp",
    "quat_multiply",
    "quat_normalize_safe",
    "quat_rotate_vector",
    "quat_slerp",
    "quat_to_matrix",
]
