"""Small host-side helpers shared by stateful sensor models."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ...kernels.common.utils import (
    poses_to_matrix,
    valid_flags_to_indices,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def compute_scaled_resolution(
    original_resolution: tuple[int, int],
    scale: float | tuple[float, float],
    new_resolution: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Compute a scaled ``(width, height)`` resolution."""

    if new_resolution is not None:
        return (int(new_resolution[0]), int(new_resolution[1]))
    if isinstance(scale, tuple):
        scale_x, scale_y = scale
    else:
        scale_x = scale_y = scale
    return (
        int(round(original_resolution[0] * scale_x)),
        int(round(original_resolution[1] * scale_y)),
    )


def filter_by_validity(
    data: jax.Array | None,
    valid_flags: jax.Array | None,
    return_all: bool,
) -> jax.Array | None:
    """Apply upstream-compatible dynamic filtering outside ``jax.jit``.

    Stateful model methods use this compatibility helper by default. Compiled
    callers should request all projections and consume the fixed-shape mask.
    """

    if data is None or valid_flags is None or return_all:
        return data
    return data[jnp.asarray(valid_flags, dtype=jnp.bool_)]


def compact_valid_indices(valid_flags: jax.Array) -> jax.Array:
    """Return dynamic-length valid indices for the model compatibility API."""

    return jnp.flatnonzero(jnp.asarray(valid_flags, dtype=jnp.bool_)).astype(jnp.int32)


__all__ = [
    "compact_valid_indices",
    "compute_scaled_resolution",
    "filter_by_validity",
    "poses_to_matrix",
    "valid_flags_to_indices",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
