"""Small, differentiable math primitives used by JAX-GS.

Quaternion inputs follow gsplat's ``wxyz`` convention.  The functions in this
module deliberately operate on arbitrary leading batch dimensions so they can
be composed with :func:`jax.jit` and :func:`jax.vmap` without reshaping data.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import Array


def safe_normalize(
    x: Array,
    axis: int | tuple[int, ...] = -1,
    eps: float = 1e-8,
) -> Array:
    """L2-normalize ``x`` while keeping zero inputs and gradients finite."""

    x = jnp.asarray(x)
    eps_array = jnp.asarray(eps, dtype=x.dtype)
    squared_norm = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    inv_norm = jax_safe_rsqrt(jnp.maximum(squared_norm, eps_array * eps_array))
    return x * inv_norm


def jax_safe_rsqrt(x: Array) -> Array:
    """Reciprocal square root helper kept separate for easy lowering by JAX."""

    return jnp.reciprocal(jnp.sqrt(x))


def quat_to_rotmat(quats: Array, eps: float = 1e-8) -> Array:
    """Convert unnormalized ``wxyz`` quaternions to rotation matrices.

    A zero quaternion maps to the identity matrix.  This matches the algebraic
    fallback of gsplat while avoiding NaNs in both the forward and backward
    passes.
    """

    quats = safe_normalize(jnp.asarray(quats), axis=-1, eps=eps)
    w, x, y, z = jnp.moveaxis(quats, -1, 0)
    values = jnp.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    )
    return values.reshape(quats.shape[:-1] + (3, 3))


normalized_quat_to_rotmat = quat_to_rotmat


def quat_scale_to_matrix(quats: Array, scales: Array) -> Array:
    """Return the Gaussian factor ``R @ diag(scales)``."""

    rotation = quat_to_rotmat(quats)
    return rotation * jnp.asarray(scales)[..., None, :]


def full_to_triu(matrix: Array) -> Array:
    """Flatten symmetric 3x3 matrices as ``[00, 01, 02, 11, 12, 22]``."""

    matrix = jnp.asarray(matrix)
    return jnp.stack(
        (
            matrix[..., 0, 0],
            0.5 * (matrix[..., 0, 1] + matrix[..., 1, 0]),
            0.5 * (matrix[..., 0, 2] + matrix[..., 2, 0]),
            matrix[..., 1, 1],
            0.5 * (matrix[..., 1, 2] + matrix[..., 2, 1]),
            matrix[..., 2, 2],
        ),
        axis=-1,
    )


def triu_to_full(triu: Array) -> Array:
    """Expand ``[00, 01, 02, 11, 12, 22]`` to a symmetric 3x3 matrix."""

    triu = jnp.asarray(triu)
    xx, xy, xz, yy, yz, zz = jnp.moveaxis(triu, -1, 0)
    values = jnp.stack((xx, xy, xz, xy, yy, yz, xz, yz, zz), axis=-1)
    return values.reshape(triu.shape[:-1] + (3, 3))


def _safe_reciprocal_scale(scales: Array, eps: float) -> Array:
    eps_array = jnp.asarray(eps, dtype=scales.dtype)
    safe_sign = jnp.where(scales < 0.0, -jnp.ones_like(scales), jnp.ones_like(scales))
    safe_scales = jnp.where(jnp.abs(scales) < eps_array, safe_sign * eps_array, scales)
    return jnp.reciprocal(safe_scales)


def quat_scale_to_covar_preci(
    quats: Array,
    scales: Array,
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
    eps: float = 1e-8,
) -> tuple[Optional[Array], Optional[Array]]:
    """Convert quaternion/scale pairs to covariance and precision matrices.

    ``compute_covar``, ``compute_preci`` and ``triu`` are intended as static
    configuration arguments when this function is directly jitted.
    """

    quats = jnp.asarray(quats)
    scales = jnp.asarray(scales)
    rotation = quat_to_rotmat(quats, eps=eps)

    covariance = None
    if compute_covar:
        factor = rotation * scales[..., None, :]
        covariance = jnp.einsum("...ij,...kj->...ik", factor, factor)
        if triu:
            covariance = full_to_triu(covariance)

    precision = None
    if compute_preci:
        inv_scales = _safe_reciprocal_scale(scales, eps)
        factor_inv = rotation * inv_scales[..., None, :]
        precision = jnp.einsum("...ij,...kj->...ik", factor_inv, factor_inv)
        if triu:
            precision = full_to_triu(precision)

    return covariance, precision


quat_scale_to_covariance_precision = quat_scale_to_covar_preci
