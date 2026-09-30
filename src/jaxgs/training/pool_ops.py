"""Slot operations shared by production and reference density control."""

import chex
import jax
import jax.numpy as jnp

from ..scene.point import GaussianArrays
from .optimizer import AdamState, reset_adam_slots


def prune_step(
    pool: GaussianArrays, state: AdamState, remove: chex.Array
) -> tuple[GaussianArrays, AdamState]:
    """Release masked live slots and reset their optimizer state without resizing."""
    remove = jax.lax.stop_gradient(remove) & pool.alive
    alive = pool.alive & ~remove
    next_pool = pool.replace(
        alive=alive, free_mask=~alive, n_active=pool.n_active - jnp.sum(remove, dtype=jnp.int32)
    )
    return next_pool, reset_adam_slots(state, remove)
