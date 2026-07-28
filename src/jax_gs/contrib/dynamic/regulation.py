"""Pure-JAX regularizers for HexPlane feature planes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


def _second_difference_squared(planes: Sequence[jax.Array]) -> jax.Array:
    """Sum mean squared second differences along each plane's H axis."""

    total: jax.Array | None = None
    first_plane: jax.Array | None = None
    for plane in planes:
        first_plane = plane if first_plane is None else first_plane
        if plane.ndim != 4:
            raise ValueError(
                "Expected 4D plane tensors (B, C, H, W); "
                f"got {plane.ndim}D shape {tuple(plane.shape)}."
            )
        if plane.shape[-2] < 3:
            continue
        first = plane[..., 1:, :] - plane[..., :-1, :]
        second = first[..., 1:, :] - first[..., :-1, :]
        contribution = jnp.mean(jnp.square(second))
        total = contribution if total is None else total + contribution

    if total is not None:
        return total
    dtype = jnp.float32 if first_plane is None else first_plane.dtype
    return jnp.zeros((), dtype=dtype)


def plane_smoothness(planes: Sequence[jax.Array]) -> jax.Array:
    """Spatial second-difference regularization, summed across planes."""

    return _second_difference_squared(planes)


def time_smoothness(planes: Sequence[jax.Array]) -> jax.Array:
    """Temporal second-difference regularization, summed across planes."""

    return _second_difference_squared(planes)


def time_l1(planes: Sequence[jax.Array]) -> jax.Array:
    """Sum each temporal plane's mean absolute deviation from one."""

    total: jax.Array | None = None
    first_plane: jax.Array | None = None
    for plane in planes:
        first_plane = plane if first_plane is None else first_plane
        contribution = jnp.mean(jnp.abs(1.0 - plane))
        total = contribution if total is None else total + contribution
    if total is not None:
        return total
    dtype = jnp.float32 if first_plane is None else first_plane.dtype
    return jnp.zeros((), dtype=dtype)


def hexplane_regularization(
    field: "HexPlaneField",
    lambda_plane_smooth: float = 1.0,
    lambda_time_smooth: float = 1.0,
    lambda_time_l1: float = 1.0,
) -> jax.Array:
    """Apply all three regularizers using the field's plane partition."""

    spatial = field.spatial_planes()
    temporal = field.temporal_planes()
    return (
        lambda_plane_smooth * plane_smoothness(spatial)
        + lambda_time_smooth * time_smoothness(temporal)
        + lambda_time_l1 * time_l1(temporal)
    )


if TYPE_CHECKING:
    from .hexplane import HexPlaneField


__all__ = [
    "plane_smoothness",
    "time_smoothness",
    "time_l1",
    "hexplane_regularization",
]
