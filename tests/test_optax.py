import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jaxgs import CapacityConfig, create_pool, seed_pool
from jaxgs.training.optimizer import (
    adam_transform,
    clear_slots,
    create_adam_state,
    optax_adam_update,
    sparse_adam_update,
)


@pytest.mark.parametrize("compact", [False, True])
def test_optax_matches_sparse_adam_with_visibility_changes_and_slot_reuse(compact):
    capacity, cluster_size = 9, 2  # Include a partial final cluster.
    pool = seed_pool(
        create_pool(CapacityConfig(capacity, cluster_size, 8, 4, 3)),
        jnp.ones((capacity, 3)),
        jnp.full((capacity, 3), 0.4),
    )
    pool = pool.replace(
        alive=pool.alive.at[3].set(False),
        free_mask=pool.free_mask.at[3].set(True),
        n_active=jnp.array(8),
    )
    state = create_adam_state(pool)
    state = state.replace(
        m=jax.tree.map(lambda x: x + 0.2, state.m), v=jax.tree.map(lambda x: x + 0.1, state.v)
    )
    expected_pool, expected_state = pool, state
    names = ("xyz", "log_scale", "rotation", "opacity", "sh")
    rng = np.random.default_rng(3)
    update = jax.jit(
        lambda p, s, g, visible, step: optax_adam_update(
            p, s, g, visible, step, 2.0, cluster_size=cluster_size, compact_gradients=compact
        )
    )
    patterns = (
        [False, True, False, True, True],
        [False] * 5,
        [True] * 5,
        [True, False, True, False, False],
    )
    assert isinstance(adam_transform(), optax.GradientTransformationExtraArgs)
    for iteration, pattern in enumerate(patterns):
        visible = jnp.repeat(jnp.array(pattern), cluster_size)[:capacity]
        gradients = tuple(
            jnp.asarray(rng.normal(size=getattr(pool, name).shape), jnp.float32) for name in names
        )
        # Visible zero gradients must decay existing momentum.
        gradients = tuple(g.at[2].set(0) for g in gradients)
        if iteration == 2:
            slots = jnp.arange(capacity) == 3
            state, expected_state = clear_slots(state, slots), clear_slots(expected_state, slots)
            pool = pool.replace(
                alive=jnp.ones(capacity, jnp.bool_),
                free_mask=jnp.zeros(capacity, jnp.bool_),
                n_active=jnp.array(9),
            )
            expected_pool = expected_pool.replace(
                alive=pool.alive, free_mask=pool.free_mask, n_active=pool.n_active
            )
        if compact:
            ids = np.flatnonzero(np.asarray(visible))
            supplied = tuple(
                jnp.full_like(g, jnp.nan).at[: len(ids)].set(g[ids]) for g in gradients
            )
        else:
            supplied = gradients
        previous_pool, previous_state = pool, state
        step = jnp.array(iteration * 5000, jnp.int32)
        pool, state = update(pool, state, supplied, visible, step)
        expected_pool, expected_state = sparse_adam_update(
            expected_pool, expected_state, gradients, visible, step, 2.0
        )
        for actual, expected in zip(
            jax.tree.leaves((pool, state)),
            jax.tree.leaves((expected_pool, expected_state)),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
        inactive = np.asarray(~(visible & pool.alive))
        for actual, previous in zip(
            jax.tree.leaves((pool, state.m, state.v, state.step)),
            jax.tree.leaves(
                (previous_pool, previous_state.m, previous_state.v, previous_state.step)
            ),
            strict=True,
        ):
            if actual.ndim:
                np.testing.assert_array_equal(
                    np.asarray(actual)[inactive], np.asarray(previous)[inactive]
                )
        assert update._cache_size() == 1
