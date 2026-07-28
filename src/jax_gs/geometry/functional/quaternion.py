"""Quaternion operations in gsplat's ``xyzw`` convention."""

from ..kernels.quaternion_ops import (
    quat_angular_distance,
    quat_conjugate,
    quat_from_axis_angle,
    quat_identity,
    quat_inverse,
    quat_lerp,
    quat_manifold_interp,
    quat_multiply,
    quat_normalize_safe,
    quat_rotate_vector,
    quat_slerp,
    quat_to_matrix,
)

__all__ = [
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
