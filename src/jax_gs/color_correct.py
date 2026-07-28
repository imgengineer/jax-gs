"""Pure-JAX color correction utilities for evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def color_correct_quadratic(
    img: jax.Array,
    ref: jax.Array,
    num_iters: int = 5,
    eps: float = 0.5 / 255,
) -> jax.Array:
    """Iteratively fit a cross-channel quadratic warp from ``img`` to ``ref``."""

    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f"img's {img.shape[-1]} and ref's {ref.shape[-1]} channels must match"
        )
    channel_count = img.shape[-1]
    image = img.reshape((-1, channel_count))
    reference = ref.reshape((-1, channel_count))

    def is_unclipped(value: jax.Array) -> jax.Array:
        return (value >= eps) & (value <= 1.0 - eps)

    original_mask = is_unclipped(image)
    for _ in range(num_iters):
        terms = [
            image[:, channel : channel + 1] * image[:, channel:]
            for channel in range(channel_count)
        ]
        terms.extend((image, jnp.ones_like(image[:, :1])))
        design = jnp.concatenate(terms, axis=-1)
        coefficients = []
        for channel in range(channel_count):
            target = reference[:, channel]
            mask = (
                original_mask[:, channel]
                & is_unclipped(image[:, channel])
                & is_unclipped(target)
            )
            masked_design = jnp.where(mask[:, None], design, 0.0)
            masked_target = jnp.where(mask, target, 0.0)
            coefficients.append(
                jnp.linalg.lstsq(masked_design, masked_target, rcond=-1.0)[0]
            )
        warp = jnp.stack(coefficients, axis=-1)
        image = jnp.clip(design @ warp, 0.0, 1.0)
    return image.reshape(img.shape)


def color_correct_affine(img: jax.Array, ref: jax.Array) -> jax.Array:
    """Fit and invert an independent affine color mapping per channel."""

    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f"img's {img.shape[-1]} and ref's {ref.shape[-1]} channels must match"
        )
    channel_count = img.shape[-1]
    image = img.reshape((-1, channel_count))
    reference = ref.reshape((-1, channel_count))
    reference_mean = jnp.mean(reference, axis=0)
    image_mean = jnp.mean(image, axis=0)
    cross_mean = jnp.mean(reference * image, axis=0)
    reference_square_mean = jnp.mean(reference * reference, axis=0)
    variance = jnp.maximum(
        reference_square_mean - reference_mean * reference_mean,
        1.0e-8,
    )
    slope = (cross_mean - reference_mean * image_mean) / variance
    offset = image_mean - slope * reference_mean
    slope = jnp.where(jnp.abs(slope) < 1.0e-8, 1.0, slope)
    corrected = jnp.clip((image - offset) / slope, 0.0, 1.0)
    return corrected.reshape(img.shape)


__all__ = ["color_correct_affine", "color_correct_quadratic"]
