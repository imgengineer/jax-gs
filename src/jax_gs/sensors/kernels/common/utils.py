"""Quaternion, pose-matrix, and fixed-shape validity helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ....math import quat_to_rotmat


def wxyz_to_xyzw(quat: jax.Array) -> jax.Array:
    quat = jnp.asarray(quat)
    return jnp.concatenate((quat[..., 1:], quat[..., :1]), axis=-1)


def xyzw_to_wxyz(quat: jax.Array) -> jax.Array:
    quat = jnp.asarray(quat)
    return jnp.concatenate((quat[..., 3:], quat[..., :3]), axis=-1)


def poses_to_matrix(
    translations: jax.Array | None, rotations: jax.Array | None
) -> jax.Array | None:
    if translations is None or rotations is None:
        return None
    translations = jnp.asarray(translations)
    rotations = jnp.asarray(rotations, dtype=translations.dtype)
    matrices = jnp.broadcast_to(
        jnp.eye(4, dtype=translations.dtype), translations.shape[:-1] + (4, 4)
    )
    matrices = matrices.at[..., :3, :3].set(quat_to_rotmat(rotations))
    return matrices.at[..., :3, 3].set(translations)


def valid_flags_to_indices(valid_flags: jax.Array | None) -> jax.Array | None:
    """Return a static-length index buffer padded with ``-1``."""

    if valid_flags is None:
        return None
    valid_flags = jnp.asarray(valid_flags, dtype=jnp.bool_)
    return jnp.nonzero(valid_flags, size=valid_flags.size, fill_value=-1)[0].astype(
        jnp.int32
    )


__all__ = [
    "poses_to_matrix",
    "valid_flags_to_indices",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
