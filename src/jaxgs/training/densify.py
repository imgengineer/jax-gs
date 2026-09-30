"""Fixed-capacity form of LiteGS DensityControllerTamingGS.

See LiteGS/litegs/training/densify.py: opacity-weighted fragment variance,
weighted sampling without replacement, append-only clone/split, weight pruning.
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp

from ..render.projection import quaternion_to_matrix
from ..render.types import FragmentStatistics
from ..scene.point import GaussianPool
from .optimizer import AdamState, create_adam_state, reset_adam_slots
from .pool_ops import prune_step


def compute_densification_scores(
    pool: GaussianPool, fragment_stats: FragmentStatistics
) -> tuple[chex.Array, chex.Array]:
    """Return opacity-weighted gradient scores and an unused-slot prune mask."""
    fragment_count, compositing_weight, alpha_grad_sum, alpha_grad_sq_sum = jax.lax.stop_gradient(
        fragment_stats
    ).T
    gradient_variance = jnp.maximum(
        alpha_grad_sq_sum / (fragment_count + 1) - (alpha_grad_sum / (fragment_count + 1)) ** 2, 0
    )
    scores = jnp.nan_to_num(
        gradient_variance * fragment_count * jax.nn.sigmoid(pool.opacity[:, 0]) ** 2,
        nan=0,
        posinf=0,
        neginf=0,
    )
    return jnp.where(pool.alive, scores, 0), pool.alive & (compositing_weight == 0)


@partial(jax.jit, static_argnames=("cluster_size", "allocator", "percent_dense"))
def densify_step(
    pool: GaussianPool,
    state: AdamState,
    stats: FragmentStatistics,
    key: chex.Array,
    target_count: int | chex.Array,
    scene_radius: float | chex.Array,
    cluster_size: int = 128,
    allocator: str = "cute",
    percent_dense: float = 0.01,
) -> tuple[GaussianPool, AdamState, chex.Array, chex.Array]:
    """Keep parents unchanged; append split/clone children into reusable slots."""
    capacity = pool.xyz.shape[0]
    scores, prune_mask = compute_densification_scores(pool, stats)
    prune_candidate_count = jnp.sum(prune_mask, dtype=jnp.int32)
    candidate_count = jnp.minimum(
        jnp.maximum(target_count - pool.n_active, 1) + prune_candidate_count, pool.n_active
    )
    prune_count = prune_candidate_count // cluster_size * cluster_size
    prune_mask = prune_mask & (jnp.cumsum(prune_mask) <= prune_count)
    updated_pool, state = prune_step(pool, state, prune_mask)
    max_births = jnp.minimum(
        candidate_count // cluster_size * cluster_size,
        (capacity - updated_pool.n_active) // cluster_size * cluster_size,
    )
    sample_key, jitter_key = jax.random.split(key)
    # Exponential-race/Gumbel sampling is the same weighted-without-replacement
    # distribution used by torch.multinomial; zero-weight live points come last.
    priorities = jnp.where(
        pool.alive,
        jnp.log(jnp.maximum(scores, 1e-30)) + jax.random.gumbel(sample_key, scores.shape),
        -jnp.inf,
    )
    candidates = jnp.argsort(-priorities, stable=True)
    split_mask = jnp.max(jnp.exp(pool.log_scale[candidates]), axis=1) > percent_dense * scene_radius
    candidate_mask = jnp.arange(capacity) < candidate_count
    candidate_order = jnp.argsort(
        jnp.where(candidate_mask, jnp.where(split_mask, 0, 1), 2), stable=True
    )
    candidates, split_mask = candidates[candidate_order], split_mask[candidate_order]
    if allocator == "cute":
        from ..kernels.allocator import allocate_free_slots_cute

        free_slots, slot_available = allocate_free_slots_cute(updated_pool.free_mask, capacity)
    else:
        free_slots = jnp.argsort(~updated_pool.free_mask, stable=True)
        slot_available = jnp.arange(capacity) < jnp.sum(updated_pool.free_mask)
    child_valid = (jnp.arange(capacity) < max_births) & slot_available & pool.alive[candidates]
    child_slots = jnp.where(child_valid, free_slots, capacity)
    local_jitter = jax.random.normal(jitter_key, (capacity, 3)) * jnp.exp(
        pool.log_scale[candidates]
    )
    rotation_matrices = quaternion_to_matrix(pool.rotation[candidates])
    world_offsets = jnp.einsum(
        "nij,nj->ni", rotation_matrices, local_jitter, precision=jax.lax.Precision.HIGHEST
    )
    child_parameters = {
        "xyz": pool.xyz[candidates] + jnp.where(split_mask[:, None], world_offsets, 0),
        "log_scale": pool.log_scale[candidates] - jnp.where(split_mask[:, None], jnp.log(1.6), 0),
        "rotation": pool.rotation[candidates],
        "opacity": pool.opacity[candidates],
        "sh": pool.sh[candidates],
    }
    updated_pool = updated_pool.replace(
        **{
            name: getattr(updated_pool, name).at[child_slots].set(value, mode="drop")
            for name, value in child_parameters.items()
        }
    )
    newborn_mask = jnp.zeros_like(pool.alive).at[child_slots].set(True, mode="drop")
    alive_mask = updated_pool.alive | newborn_mask
    updated_pool = updated_pool.replace(
        alive=alive_mask, free_mask=~alive_mask, n_active=jnp.sum(alive_mask, dtype=jnp.int32)
    )
    return updated_pool, reset_adam_slots(state, newborn_mask), jnp.sum(newborn_mask), prune_count


@jax.jit
def decay_opacity(pool: GaussianPool, state: AdamState) -> tuple[GaussianPool, AdamState]:
    """LiteGS decay halves alpha, clamps at 1/128, and clears all Adam moments."""
    alpha = jnp.maximum(jax.nn.sigmoid(pool.opacity) * 0.5, 1 / 128)
    pool = pool.replace(
        opacity=jnp.where(pool.alive[:, None], jnp.log(alpha / (1 - alpha)), pool.opacity)
    )
    return pool, create_adam_state(pool)
