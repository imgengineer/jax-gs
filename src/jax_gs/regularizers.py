"""Pure-JAX occlusion and targeted total-variation regularizers."""

from __future__ import annotations

from collections.abc import Iterable
import os

import jax
import jax.numpy as jnp


ENFORCE_CONTRACTS = (
    os.environ.get("GSPLAT_ENFORCE_CONTRACTS") == "1"
    or os.environ.get("PYTHONOPTIMIZE") == "0"
)


def compute_tv_loss_targeted(
    image: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Anisotropic NCHW total variation, optionally restricted by a mask."""

    if image.ndim != 4:
        raise ValueError(
            "compute_tv_loss_targeted: expected 4D image (B, C, H, W), "
            f"got {image.ndim}D shape {tuple(image.shape)}."
        )
    vertical = jnp.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    horizontal = jnp.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    if mask is None:
        return (jnp.sum(vertical) + jnp.sum(horizontal)) / image.size

    if ENFORCE_CONTRACTS and not bool(jnp.all((mask == 0) | (mask == 1))):
        raise ValueError(
            "compute_tv_loss_targeted: mask must be binary (values in {0, 1})."
        )
    mask_vertical = mask[:, :, 1:, :]
    mask_horizontal = mask[:, :, :, 1:]
    channels = image.shape[1]
    vertical_count = jnp.sum(mask_vertical) * channels + 1.0e-8
    horizontal_count = jnp.sum(mask_horizontal) * channels + 1.0e-8
    return (
        jnp.sum(vertical * mask_vertical) / vertical_count
        + jnp.sum(horizontal * mask_horizontal) / horizontal_count
    )


def dilate_mask(mask: jax.Array, kernel_size: int = 3) -> jax.Array:
    """Dilate a 2D, CHW, or NCHW mask with a square max-pool window."""

    if (
        not isinstance(kernel_size, int)
        or kernel_size < 1
        or kernel_size % 2 == 0
    ):
        raise ValueError(
            "dilate_mask: kernel_size must be a positive odd integer, "
            f"got {kernel_size!r}."
        )
    original_ndim = mask.ndim
    if original_ndim == 2:
        values = mask[None, None, ...].astype(jnp.float32)
    elif original_ndim == 3:
        values = mask[None, ...].astype(jnp.float32)
    elif original_ndim == 4:
        values = mask.astype(jnp.float32)
    else:
        raise ValueError(
            f"dilate_mask: expected 2D / 3D / 4D mask, got {original_ndim}D."
        )
    padding = kernel_size // 2
    dilated = jax.lax.reduce_window(
        values,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(1, 1, kernel_size, kernel_size),
        window_strides=(1, 1, 1, 1),
        padding=((0, 0), (0, 0), (padding, padding), (padding, padding)),
    )
    if original_ndim == 2:
        return dilated[0, 0]
    if original_ndim == 3:
        return dilated[0]
    return dilated


def create_invisible_mask(masks: Iterable[jax.Array | str]) -> jax.Array:
    """Return the clipped union of tensor masks or inverted PNG masks."""

    mask_values = list(masks)
    if not mask_values:
        raise ValueError("create_invisible_mask: at least one mask is required.")
    arrays = []
    for index, mask in enumerate(mask_values):
        if isinstance(mask, str):
            from PIL import Image
            import numpy as np

            array = np.asarray(Image.open(mask), dtype=np.float32) / 255.0
            if array.ndim == 3:
                array = array[:, :, 0]
            value = jnp.asarray(1.0 - array, dtype=jnp.float32)
        elif isinstance(mask, jax.Array):
            value = mask.astype(jnp.float32)
        else:
            raise TypeError(
                "create_invisible_mask: expected JAX Array or str path at "
                f"index {index}, got {type(mask).__name__}."
            )
        arrays.append(value)
    expected_shape = arrays[0].shape
    for index, value in enumerate(arrays[1:], 1):
        if value.shape != expected_shape:
            raise ValueError(
                f"create_invisible_mask: mask {index} shape {tuple(value.shape)} "
                f"!= mask 0 shape {tuple(expected_shape)}."
            )
    return jnp.clip(jnp.max(jnp.stack(arrays), axis=0), 0.0, 1.0)


__all__ = ["compute_tv_loss_targeted", "dilate_mask", "create_invisible_mask"]
