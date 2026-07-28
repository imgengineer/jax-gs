"""Differentiable two-pose interpolation in ``wxyz`` convention."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .pose import DynamicPose


SLERP_SMALL_ANGLE_DOT_THRESHOLD = 0.9995


def _normalize_quaternion(quaternion: jax.Array) -> jax.Array:
    squared_norm = jnp.sum(quaternion * quaternion, axis=-1, keepdims=True)
    norm = jnp.sqrt(jnp.maximum(squared_norm, jnp.asarray(1.0e-20, quaternion.dtype)))
    return quaternion / norm


def quaternion_slerp_wxyz(
    start: jax.Array, end: jax.Array, relative_time: jax.Array
) -> jax.Array:
    """SLERP with hemisphere correction and an NLERP small-angle branch."""

    start = _normalize_quaternion(jnp.asarray(start))
    end = _normalize_quaternion(jnp.asarray(end, dtype=start.dtype))
    relative_time = jnp.asarray(relative_time, dtype=start.dtype)
    while relative_time.ndim < start.ndim:
        relative_time = relative_time[..., None]
    dot = jnp.sum(start * end, axis=-1, keepdims=True)
    end = jnp.where(dot < 0.0, -end, end)
    dot = jnp.sum(start * end, axis=-1, keepdims=True)
    close = dot > SLERP_SMALL_ANGLE_DOT_THRESHOLD
    safe_dot = jnp.minimum(dot, jnp.asarray(1.0 - 1.0e-6, start.dtype))
    linear = _normalize_quaternion(
        start * (1.0 - relative_time) + end * relative_time
    )
    theta = jnp.arccos(safe_dot)
    sin_theta = jnp.sin(theta)
    weight_start = jnp.sin((1.0 - relative_time) * theta) / sin_theta
    weight_end = jnp.sin(relative_time * theta) / sin_theta
    spherical = weight_start * start + weight_end * end
    return jnp.where(close, linear, spherical)


def interpolate_dynamic_pose(
    start_translation: jax.Array,
    start_rotation: jax.Array,
    end_translation: jax.Array,
    end_rotation: jax.Array,
    relative_time: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """LERP translations and SLERP rotations at ``relative_time`` values."""

    relative_time = jnp.asarray(relative_time, dtype=jnp.asarray(start_translation).dtype)
    translation = (
        jnp.asarray(start_translation) * (1.0 - relative_time[..., None])
        + jnp.asarray(end_translation) * relative_time[..., None]
    )
    count = relative_time.size
    start = jnp.broadcast_to(jnp.asarray(start_rotation), (count, 4))
    end = jnp.broadcast_to(jnp.asarray(end_rotation), (count, 4))
    rotation = quaternion_slerp_wxyz(start, end, relative_time.reshape((count,)))
    return translation.reshape((count, 3)), rotation


def unpack_dynamic_pose_components(
    dynamic_pose: DynamicPose,
    device: jax.Device | None = None,
    dtype=None,
    allow_device_transfer: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return start/end translation and rotation arrays.

    ``allow_device_transfer`` is retained for signature compatibility; JAX
    device movement is performed only when ``device`` is explicitly supplied.
    """

    del allow_device_transfer
    arrays = (
        dynamic_pose.start_pose.translation,
        dynamic_pose.start_pose.rotation,
        dynamic_pose.end_pose.translation,
        dynamic_pose.end_pose.rotation,
    )
    if dtype is not None:
        arrays = tuple(jnp.asarray(value, dtype=dtype) for value in arrays)
    if device is not None:
        arrays = tuple(jax.device_put(value, device) for value in arrays)
    return arrays


__all__ = [
    "SLERP_SMALL_ANGLE_DOT_THRESHOLD",
    "interpolate_dynamic_pose",
    "quaternion_slerp_wxyz",
    "unpack_dynamic_pose_components",
]
