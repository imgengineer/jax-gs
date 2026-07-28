"""JAX device and timestamp helpers shared by sensor kernels."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def raise_or_target_device(
    tensor: jax.Array, allow_device_transfer: bool
) -> jax.Device:
    """Return a JAX array's device, transferring non-JAX input when allowed."""

    if isinstance(tensor, jax.Array):
        return tensor.device
    if not allow_device_transfer:
        raise RuntimeError(
            "Non-JAX inputs require allow_device_transfer=True before sensor ops"
        )
    return jax.device_put(tensor).device


def to_dev(
    tensor: jax.Array,
    device: jax.Device,
    dtype: Any,
    allow_device_transfer: bool,
) -> jax.Array:
    """Place ``tensor`` on ``device`` with ``dtype`` under an explicit policy."""

    array = jnp.asarray(tensor)
    target_dtype = jnp.dtype(dtype)
    needs_transfer = array.device != device or array.dtype != target_dtype
    if needs_transfer and not allow_device_transfer:
        raise RuntimeError(
            f"Array on {array.device} (dtype={array.dtype}) requires transfer "
            f"to {device} (dtype={target_dtype}); set "
            "allow_device_transfer=True"
        )
    if array.dtype != target_dtype:
        array = array.astype(target_dtype)
    return jax.device_put(array, device)


def zero_like(shape: tuple[int, ...], reference: jax.Array) -> jax.Array:
    """Allocate zeros with the same dtype and device as ``reference``."""

    return jax.device_put(jnp.zeros(shape, dtype=reference.dtype), reference.device)


def timestamp_bounds(
    start_timestamp_us: int | None, end_timestamp_us: int | None
) -> tuple[int, int]:
    """Normalize optional timestamp bounds, requiring both or neither."""

    if start_timestamp_us is None and end_timestamp_us is None:
        return 0, 0
    if start_timestamp_us is None or end_timestamp_us is None:
        raise ValueError(
            "start_timestamp_us and end_timestamp_us must be provided together"
        )
    return int(start_timestamp_us), int(end_timestamp_us)


__all__ = ["raise_or_target_device", "timestamp_bounds", "to_dev", "zero_like"]
