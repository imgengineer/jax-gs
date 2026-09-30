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
from ..scene.point import GaussianArrays
from .optimizer import AdamState, create_adam_state, reset_adam_slots
from .pool_ops import prune_step


def compute_densification_scores(
    pool: GaussianArrays, fragment_stats: FragmentStatistics
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
    pool: GaussianArrays,
    state: AdamState,
    stats: FragmentStatistics,
    key: chex.Array,
    target_count: int | chex.Array,
    scene_radius: float | chex.Array,
    cluster_size: int = 128,
    allocator: str = "cute",
    percent_dense: float = 0.01,
) -> tuple[GaussianArrays, AdamState, chex.Array, chex.Array]:
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
    if allocator == "cute":
        from ..kernels.partitioner import stable_split_partition_cute

        candidate_order = stable_split_partition_cute(split_mask, candidate_count)
    else:
        candidate_indices = jnp.arange(capacity, dtype=jnp.int32)
        candidate_mask = candidate_indices < candidate_count
        split_prefix = jnp.cumsum(candidate_mask & split_mask, dtype=jnp.int32)
        # Stable partition: selected splits, selected clones, then the unused tail.
        destinations = jnp.where(
            candidate_mask,
            jnp.where(
                split_mask, split_prefix - 1, split_prefix[-1] + candidate_indices - split_prefix
            ),
            candidate_indices,
        )
        candidate_order = (
            jnp.zeros_like(candidate_indices)
            .at[destinations]
            .set(candidate_indices, unique_indices=True)
        )
    candidates, split_mask = candidates[candidate_order], split_mask[candidate_order]
    if allocator == "cute":
        from ..kernels.allocator import allocate_free_slots_cute

        free_slots, slot_available = allocate_free_slots_cute(updated_pool.free_mask, capacity)
    else:
        free_slots = jnp.argsort(~updated_pool.free_mask, stable=True)
        slot_available = jnp.arange(capacity) < jnp.sum(updated_pool.free_mask)

    def generate_children(width):
        parents, split = candidates[:width], split_mask[:width]
        child_valid = (
            (jnp.arange(width) < max_births) & slot_available[:width] & pool.alive[parents]
        )
        child_slots = jnp.where(child_valid, free_slots[:width], capacity)
        local_jitter = jax.random.normal(jitter_key, (width, 3)) * jnp.exp(pool.log_scale[parents])
        rotation_matrices = quaternion_to_matrix(pool.rotation[parents])
        world_offsets = jnp.einsum(
            "nij,nj->ni", rotation_matrices, local_jitter, precision=jax.lax.Precision.HIGHEST
        )
        child_parameters = {
            "xyz": pool.xyz[parents] + jnp.where(split[:, None], world_offsets, 0),
            "log_scale": pool.log_scale[parents] - jnp.where(split[:, None], jnp.log(1.6), 0),
            "rotation": pool.rotation[parents],
            "opacity": pool.opacity[parents],
            "sh": pool.sh[parents],
        }
        # Read parents from the original pool before writing released slots.
        child_pool = updated_pool.replace(
            **{
                name: getattr(updated_pool, name).at[child_slots].set(value, mode="drop")
                for name, value in child_parameters.items()
            }
        )
        newborn_mask = jnp.zeros_like(pool.alive).at[child_slots].set(True, mode="drop")
        return child_pool, newborn_mask

    if (
        str(jax.random.key_impl(jitter_key)) == "threefry2x32"
        and jax.config.jax_threefry_partitionable
    ):
        # Partitionable Threefry preserves the random prefix across shapes.
        # All buckets compile in this JIT; device budgets do not change shapes.
        widths = [0]
        width = min(cluster_size, capacity)
        while width < capacity:
            widths.append(width)
            width = min(width * 4, capacity)
        widths.append(capacity)
        bucket = jnp.searchsorted(jnp.asarray(widths), max_births, side="left")
        updated_pool, newborn_mask = jax.lax.switch(
            bucket, tuple(partial(generate_children, width) for width in widths)
        )
    else:
        updated_pool, newborn_mask = generate_children(capacity)
    alive_mask = updated_pool.alive | newborn_mask
    updated_pool = updated_pool.replace(
        alive=alive_mask, free_mask=~alive_mask, n_active=jnp.sum(alive_mask, dtype=jnp.int32)
    )
    return updated_pool, reset_adam_slots(state, newborn_mask), jnp.sum(newborn_mask), prune_count


@jax.jit
def decay_opacity(pool: GaussianArrays, state: AdamState) -> tuple[GaussianArrays, AdamState]:
    """LiteGS decay halves alpha, clamps at 1/128, and clears all Adam moments."""
    alpha = jnp.maximum(jax.nn.sigmoid(pool.opacity) * 0.5, 1 / 128)
    pool = pool.replace(
        opacity=jnp.where(pool.alive[:, None], jnp.log(alpha / (1 - alpha)), pool.opacity)
    )
    return pool, create_adam_state(pool)
