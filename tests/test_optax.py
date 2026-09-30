import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jaxgs import CapacityConfig, create_gaussians, seed_gaussians
from jaxgs.training.optimizer import (
    create_adam_state,
    create_adam_transform,
    optax_adam_update,
    reset_adam_slots,
    sparse_adam_update,
)


@pytest.mark.parametrize("compact", [False, True])
def test_optax_matches_sparse_adam_with_visibility_changes_and_slot_reuse(compact):
    capacity, cluster_size = 9, 2  # Include a partial final cluster.
    pool = seed_gaussians(
        create_gaussians(CapacityConfig(capacity, cluster_size, 8, 4, 3)),
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
    assert isinstance(create_adam_transform(), optax.GradientTransformationExtraArgs)
    for iteration, pattern in enumerate(patterns):
        visible = jnp.repeat(jnp.array(pattern), cluster_size)[:capacity]
        gradients = tuple(
            jnp.asarray(rng.normal(size=getattr(pool, name).shape), jnp.float32) for name in names
        )
        # Visible zero gradients must decay existing momentum.
        gradients = tuple(g.at[2].set(0) for g in gradients)
        if iteration == 2:
            slots = jnp.arange(capacity) == 3
            state, expected_state = (
                reset_adam_slots(state, slots),
                reset_adam_slots(expected_state, slots),
            )
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


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("compact", [False, True])
def test_active_sh_gradients_preserve_existing_higher_order_momentum(degree, compact):
    capacity, cluster_size = 9, 2
    pool = seed_gaussians(
        create_gaussians(CapacityConfig(capacity, cluster_size, 8, 4, 3)),
        jnp.ones((capacity, 3)),
        jnp.full((capacity, 3), 0.4),
    )
    state = create_adam_state(pool)
    state = state.replace(
        m=jax.tree.map(lambda x: x + 0.2, state.m),
        v=jax.tree.map(lambda x: x + 0.1, state.v),
    )
    visible = jnp.repeat(jnp.array([True, False, True, False, True]), cluster_size)[:capacity]
    gradients = tuple(
        jnp.ones_like(getattr(pool, name))
        for name in ("xyz", "log_scale", "rotation", "opacity", "sh")
    )
    gradients = (*gradients[:-1], gradients[-1].at[:, (degree + 1) ** 2 :].set(0))
    supplied = gradients
    if compact:
        ids = np.flatnonzero(visible)
        supplied = tuple(jnp.full_like(g, jnp.nan).at[: len(ids)].set(g[ids]) for g in gradients)
    expected = sparse_adam_update(pool, state, gradients, visible, jnp.array(7), 2.0)
    update = jax.jit(
        lambda p, s, g: optax_adam_update(
            p,
            s,
            g,
            visible,
            jnp.array(7),
            2.0,
            cluster_size=cluster_size,
            compact_gradients=compact,
            active_degree=degree,
        )
    )
    actual = update(pool, state, supplied)
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(result, reference, rtol=2e-6, atol=2e-7)
    if degree < 3:
        higher_order = np.asarray(actual[1].m.sh)[:, (degree + 1) ** 2 :]
        np.testing.assert_allclose(higher_order[np.asarray(visible)], 0.18, atol=2e-7)
        np.testing.assert_array_equal(
            higher_order[~np.asarray(visible)],
            np.asarray(state.m.sh)[~np.asarray(visible), (degree + 1) ** 2 :],
        )


@pytest.mark.parametrize("degree", [-1, 4])
def test_optax_rejects_invalid_active_sh_degree(degree):
    pool = create_gaussians(CapacityConfig(1, 1, 1, 4, 3))
    gradients = tuple(
        getattr(pool, name) for name in ("xyz", "log_scale", "rotation", "opacity", "sh")
    )
    with pytest.raises(ValueError, match="active_degree"):
        optax_adam_update(
            pool, create_adam_state(pool), gradients, pool.alive, 0, 1.0, active_degree=degree
        )
