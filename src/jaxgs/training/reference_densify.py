import chex
import jax
import jax.numpy as jnp

from ..render.projection import quaternion_to_matrix
from ..scene.point import GaussianPool
from .optimizer import AdamState, reset_adam_slots


def densify_step(
    pool: GaussianPool,
    state: AdamState,
    gradient_stats: chex.Array,
    key: chex.Array,
    *,
    max_new: int,
    threshold: float,
    split_scale: float = 0.05,
    allocator: str = "jax",
) -> tuple[GaussianPool, AdamState, chex.Array]:
    """Clone small Gaussians and split large ones into free fixed-capacity slots."""
    capacity = pool.xyz.shape[0]
    if not 0 < max_new <= capacity:
        raise ValueError("max_new must be within pool capacity")
    scores = jax.lax.stop_gradient(gradient_stats)
    candidates = jnp.argsort(jnp.where(pool.alive, -scores, jnp.inf), stable=True)[:max_new]
    if allocator == "cute":
        from ..kernels.allocator import allocate_free_slots_cute

        free_slots, slots_valid = allocate_free_slots_cute(pool.free_mask, max_new)
    elif allocator == "jax":
        free_slots = jnp.argsort(~pool.free_mask, stable=True)[:max_new]
        slots_valid = jnp.arange(max_new) < jnp.sum(pool.free_mask)
    else:
        raise ValueError(f"unknown allocator: {allocator}")
    valid = (
        pool.alive[candidates]
        & (scores[candidates] >= threshold)
        & pool.free_mask[free_slots]
        & slots_valid
    )
    jitter = jax.random.normal(key, (max_new, 3))
    split = jnp.max(jnp.exp(pool.log_scale[candidates]), axis=1) > split_scale

    def write_one(i, current):
        parent = candidates[i]
        slot = free_slots[i]
        scale = jnp.exp(current.log_scale[parent])
        parent_xyz = current.xyz[parent]
        offset = jnp.where(
            split[i],
            quaternion_to_matrix(current.rotation[parent]) @ (jitter[i] * scale / 1.6),
            jnp.zeros((3,), scale.dtype),
        )
        child_scale = jnp.where(
            split[i], current.log_scale[parent] - jnp.log(1.6), current.log_scale[parent]
        )
        return current.replace(
            xyz=current.xyz.at[parent].set(parent_xyz - offset).at[slot].set(parent_xyz + offset),
            log_scale=current.log_scale.at[parent].set(child_scale).at[slot].set(child_scale),
            rotation=current.rotation.at[slot].set(current.rotation[parent]),
            opacity=current.opacity.at[slot].set(current.opacity[parent]),
            sh=current.sh.at[slot].set(current.sh[parent]),
            alive=current.alive.at[slot].set(True),
            free_mask=current.free_mask.at[slot].set(False),
            n_active=current.n_active + 1,
        )

    next_pool = jax.lax.fori_loop(
        0, max_new, lambda i, p: jax.lax.cond(valid[i], write_one, lambda _, q: q, i, p), pool
    )
    new_slots = (
        jnp.zeros((capacity,), dtype=jnp.int32).at[free_slots].add(valid.astype(jnp.int32)) > 0
    )
    return next_pool, reset_adam_slots(state, new_slots), jnp.sum(valid, dtype=jnp.int32)


def reset_opacity(
    pool: GaussianPool, state: AdamState, max_alpha: float = 0.01
) -> tuple[GaussianPool, AdamState]:
    alpha = jax.nn.sigmoid(pool.opacity)
    logit = jnp.log(max_alpha / (1 - max_alpha))
    reset = pool.alive & (alpha[:, 0] > max_alpha)
    pool = pool.replace(opacity=jnp.where(reset[:, None], logit, pool.opacity))
    return pool, state.replace(
        m=state.m.replace(opacity=jnp.where(reset[:, None], 0, state.m.opacity)),
        v=state.v.replace(opacity=jnp.where(reset[:, None], 0, state.v.opacity)),
    )
