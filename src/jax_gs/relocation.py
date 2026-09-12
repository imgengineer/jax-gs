"""Pure-JAX Gaussian relocation primitive from gsplat's MCMC strategy."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_relocation(
    opacities: jax.Array,
    scales: jax.Array,
    ratios: jax.Array,
    binoms: jax.Array,
    min_opacity: float = 0.005,
) -> tuple[jax.Array, jax.Array]:
    """Compute opacity and scale after duplicating sampled Gaussians.

    This is Equation 9 of 3D Gaussian Splatting as Markov Chain Monte Carlo,
    matching gsplat v1.5.3's CUDA primitive. ``ratios`` are clipped to the
    binomial table's supported range without mutating the caller's array.
    """

    opacities = jnp.asarray(opacities)
    scales = jnp.asarray(scales)
    ratios = jnp.asarray(ratios)
    binoms = jnp.asarray(binoms)
    count = opacities.shape[0]
    if opacities.shape != (count,):
        raise ValueError("opacities must have shape (N,)")
    if scales.shape != (count, 3):
        raise ValueError("scales must have shape (N, 3)")
    if ratios.shape != (count,):
        raise ValueError("ratios must have shape (N,)")
    if binoms.ndim != 2 or binoms.shape[0] != binoms.shape[1]:
        raise ValueError("binoms must be a square matrix")

    maximum_ratio = binoms.shape[0]
    ratios = jnp.clip(ratios, 1, maximum_ratio).astype(jnp.int32)
    new_opacities = 1.0 - jnp.power(
        1.0 - opacities, 1.0 / ratios.astype(opacities.dtype)
    )
    new_opacities = jnp.clip(
        new_opacities,
        jnp.asarray(min_opacity, dtype=opacities.dtype),
        1.0 - jnp.finfo(opacities.dtype).eps,
    )
    orders = jnp.arange(maximum_ratio, dtype=jnp.int32)
    signs = jnp.where(orders % 2 == 0, 1.0, -1.0).astype(opacities.dtype)
    powers = signs / jnp.sqrt((orders + 1).astype(opacities.dtype))
    powers = powers[None, :] * jnp.power(new_opacities[:, None], orders[None, :] + 1)

    def add_order(index: int, denominator: jax.Array) -> jax.Array:
        valid_terms = orders <= index
        contribution = jnp.sum(
            jnp.where(valid_terms, binoms[index, :] * powers, 0.0), axis=-1
        )
        return denominator + jnp.where(index < ratios, contribution, 0.0)

    denominator = jax.lax.fori_loop(
        0,
        maximum_ratio,
        add_order,
        jnp.zeros_like(opacities),
    )
    scale_factor = opacities / denominator
    return new_opacities, scales * scale_factor[:, None]
