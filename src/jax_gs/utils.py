"""JAX equivalents of gsplat v1.5.3's public utility functions."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .losses import depth_to_normal, depth_to_points
from .math import quat_to_rotmat


def normalized_quat_to_rotmat(quat: jax.Array) -> jax.Array:
    """Convert a normalized ``wxyz`` quaternion to a rotation matrix."""

    return quat_to_rotmat(quat)


def log_transform(x: jax.Array) -> jax.Array:
    """Apply gsplat's signed logarithmic coordinate transform."""

    x = jnp.asarray(x)
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def inverse_log_transform(y: jax.Array) -> jax.Array:
    """Invert :func:`log_transform`."""

    y = jnp.asarray(y)
    return jnp.sign(y) * jnp.expm1(jnp.abs(y))


def get_projection_matrix(
    znear: float | jax.Array,
    zfar: float | jax.Array,
    fovX: float | jax.Array,
    fovY: float | jax.Array,
    device: Any = "cuda",
) -> jax.Array:
    """Create the OpenGL-style projection matrix used by gsplat's wrappers."""

    del device
    znear, zfar, fovX, fovY = (
        jnp.asarray(value) for value in (znear, zfar, fovX, fovY)
    )
    dtype = jnp.result_type(znear, zfar, fovX, fovY, jnp.float32)
    znear, zfar, fovX, fovY = (
        value.astype(dtype) for value in (znear, zfar, fovX, fovY)
    )
    top = jnp.tan(fovY / 2) * znear
    right = jnp.tan(fovX / 2) * znear
    matrix = jnp.zeros((4, 4), dtype=dtype)
    matrix = matrix.at[0, 0].set(znear / right)
    matrix = matrix.at[1, 1].set(znear / top)
    matrix = matrix.at[3, 2].set(1.0)
    matrix = matrix.at[2, 2].set(zfar / (zfar - znear))
    matrix = matrix.at[2, 3].set(-(zfar * znear) / (zfar - znear))
    return matrix


def save_ply(
    splats: Mapping[str, Any],
    dir: str | Path,
    colors: jax.Array | None = None,
) -> Path:
    """Write the deprecated conventional 3DGS PLY representation."""

    warnings.warn(
        "save_ply() is deprecated; use export_splats() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    host = {name: np.asarray(jax.device_get(value)) for name, value in splats.items()}
    means = host["means"]
    count = means.shape[0]
    scales = host["scales"]
    quats = host["quats"]
    opacities = host["opacities"].reshape(count)
    sh0 = host["sh0"].transpose(0, 2, 1).reshape(count, -1)
    shN = host["shN"].transpose(0, 2, 1).reshape(count, -1)
    valid = np.ones((count,), dtype=bool)
    for values in (means, scales, quats, opacities, sh0, shN):
        finite = np.isfinite(values)
        valid &= finite if values.ndim == 1 else finite.all(axis=1)
    means, scales, quats, opacities, sh0, shN = (
        values[valid] for values in (means, scales, quats, opacities, sh0, shN)
    )

    if colors is None:
        color_data = np.concatenate((sh0, shN), axis=1)
        color_names = [f"f_dc_{index}" for index in range(sh0.shape[1])]
        color_names += [f"f_rest_{index}" for index in range(shN.shape[1])]
    else:
        color_data = np.asarray(jax.device_get(colors))[valid]
        color_data = (color_data - 0.5) / 0.2820947917738781
        color_names = [f"f_dc_{index}" for index in range(color_data.shape[1])]
    names = ["x", "y", "z", "nx", "ny", "nz", *color_names, "opacity"]
    names += [f"scale_{index}" for index in range(scales.shape[1])]
    names += [f"rot_{index}" for index in range(quats.shape[1])]
    records = np.concatenate(
        (
            means,
            np.zeros_like(means),
            color_data,
            opacities[:, None],
            scales,
            quats,
        ),
        axis=1,
    ).astype("<f4", copy=False)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {records.shape[0]}",
        *[f"property float {name}" for name in names],
        "end_header",
        "",
    ]
    path = Path(dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        handle.write(records.tobytes())
    return path


__all__ = [
    "depth_to_normal",
    "depth_to_points",
    "get_projection_matrix",
    "inverse_log_transform",
    "log_transform",
    "normalized_quat_to_rotmat",
    "save_ply",
]
