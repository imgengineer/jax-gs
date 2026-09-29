"""Fixed-capacity form of LiteGS DensityControllerTamingGS.

See LiteGS/litegs/training/densify.py: opacity-weighted fragment variance,
weighted sampling without replacement, append-only clone/split, weight pruning.
"""

from functools import partial

import jax
import jax.numpy as jnp

from ..render.projection import quaternion_to_matrix
from .optimizer import clear_slots, create_adam_state
from .reference_densify import prune_step


def fragment_scores(pool, stats):
    count, weight, error, error_square = jax.lax.stop_gradient(stats).T
    variance = jnp.maximum(error_square / (count + 1) - (error / (count + 1)) ** 2, 0)
    score = jnp.nan_to_num(
        variance * count * jax.nn.sigmoid(pool.opacity[:, 0]) ** 2, nan=0, posinf=0, neginf=0
    )
    return jnp.where(pool.alive, score, 0), pool.alive & (weight == 0)


@partial(jax.jit, static_argnames=("cluster_size", "allocator", "percent_dense"))
def densify_step(
    pool,
    state,
    stats,
    key,
    target_count,
    scene_radius,
    cluster_size=128,
    allocator="cute",
    percent_dense=0.01,
):
    """Keep parents unchanged; append split/clone children into reusable slots."""
    capacity = pool.xyz.shape[0]
    score, remove = fragment_scores(pool, stats)
    raw_prune_count = jnp.sum(remove, dtype=jnp.int32)
    raw_budget = jnp.minimum(
        jnp.maximum(target_count - pool.n_active, 1) + raw_prune_count, pool.n_active
    )
    prune_count = raw_prune_count // cluster_size * cluster_size
    remove = remove & (jnp.cumsum(remove) <= prune_count)
    current, state = prune_step(pool, state, remove)
    budget = jnp.minimum(
        raw_budget // cluster_size * cluster_size,
        (capacity - current.n_active) // cluster_size * cluster_size,
    )
    sample_key, jitter_key = jax.random.split(key)
    # Exponential-race/Gumbel sampling is the same weighted-without-replacement
    # distribution used by torch.multinomial; zero-weight live points come last.
    priorities = jnp.where(
        pool.alive,
        jnp.log(jnp.maximum(score, 1e-30)) + jax.random.gumbel(sample_key, score.shape),
        -jnp.inf,
    )
    candidates = jnp.argsort(-priorities, stable=True)
    split = jnp.max(jnp.exp(pool.log_scale[candidates]), axis=1) > percent_dense * scene_radius
    selected = jnp.arange(capacity) < raw_budget
    order = jnp.argsort(jnp.where(selected, jnp.where(split, 0, 1), 2), stable=True)
    candidates, split = candidates[order], split[order]
    if allocator == "cute":
        from ..kernels.allocator import allocate_free_slots_cute

        slots, available = allocate_free_slots_cute(current.free_mask, capacity)
    else:
        slots = jnp.argsort(~current.free_mask, stable=True)
        available = jnp.arange(capacity) < jnp.sum(current.free_mask)
    valid = (jnp.arange(capacity) < budget) & available & pool.alive[candidates]
    destination = jnp.where(valid, slots, capacity)
    jitter = jax.random.normal(jitter_key, (capacity, 3)) * jnp.exp(pool.log_scale[candidates])
    rotation = quaternion_to_matrix(pool.rotation[candidates])
    shift = jnp.einsum("nij,nj->ni", rotation, jitter, precision=jax.lax.Precision.HIGHEST)
    values = {
        "xyz": pool.xyz[candidates] + jnp.where(split[:, None], shift, 0),
        "log_scale": pool.log_scale[candidates] - jnp.where(split[:, None], jnp.log(1.6), 0),
        "rotation": pool.rotation[candidates],
        "opacity": pool.opacity[candidates],
        "sh": pool.sh[candidates],
    }
    current = current.replace(
        **{
            name: getattr(current, name).at[destination].set(value, mode="drop")
            for name, value in values.items()
        }
    )
    newborn = jnp.zeros_like(pool.alive).at[destination].set(True, mode="drop")
    alive = current.alive | newborn
    current = current.replace(
        alive=alive, free_mask=~alive, n_active=jnp.sum(alive, dtype=jnp.int32)
    )
    return current, clear_slots(state, newborn), jnp.sum(newborn), prune_count


@jax.jit
def decay_opacity(pool, state):
    """LiteGS decay halves alpha, clamps at 1/128, and clears all Adam moments."""
    alpha = jnp.maximum(jax.nn.sigmoid(pool.opacity) * 0.5, 1 / 128)
    pool = pool.replace(
        opacity=jnp.where(pool.alive[:, None], jnp.log(alpha / (1 - alpha)), pool.opacity)
    )
    return pool, create_adam_state(pool)
