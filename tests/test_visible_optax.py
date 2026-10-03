"""Optax transformations applied only to the rows of visible clusters."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jaxgs import CapacityConfig, create_gaussians, seed_gaussians
from jaxgs.scene.types import PARAMETER_NAMES, ParameterArrays
from jaxgs.training.optimizer import create_adam_state, optax_update

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="The visible-cluster Optax kernel requires a JAX CUDA device",
)


def _pool(capacity, cluster_size, rng):
    pool = seed_gaussians(
        create_gaussians(CapacityConfig(capacity, cluster_size, 8, 4, 3)),
        jnp.asarray(rng.normal(size=(capacity, 3)), jnp.float32),
        jnp.asarray(rng.uniform(size=(capacity, 3)), jnp.float32),
    )
    pool = pool.replace(sh=jnp.asarray(rng.normal(size=pool.sh.shape), jnp.float32))
    dead = jnp.arange(capacity) % 7 == 3  # free slots inside visible clusters
    return pool.replace(alive=~dead, free_mask=dead, n_active=jnp.sum(~dead))


def _visible(capacity, cluster_size, pattern):
    from jaxgs.kernels.cluster_compact import compact_visible_clusters

    clusters = -(-capacity // cluster_size)
    visible = jnp.repeat(jnp.asarray((pattern * clusters)[:clusters]), cluster_size)[:capacity]
    return visible, compact_visible_clusters(visible, cluster_size)


def _compact(rng, pool, visible, cluster_size, sh_dim):
    """Compact gradients in visible-cluster order; unused rows are NaN."""
    capacity = pool.xyz.shape[0]
    slots = np.flatnonzero(np.asarray(visible))
    rank = np.cumsum(np.asarray(visible)[::cluster_size]) - 1
    rows = rank[slots // cluster_size] * cluster_size + slots % cluster_size
    shapes = [getattr(pool, name).shape[1:] for name in PARAMETER_NAMES]
    shapes[-1] = (sh_dim, 3)
    dense, compact = [], []
    for shape in shapes:
        values = rng.normal(size=(capacity, *shape)).astype(np.float32)
        values[slots[::5]] = 0  # visible zero gradients still decay momentum
        packed = np.full((capacity, *shape), np.nan, np.float32)
        packed[rows] = values[slots]
        dense.append(jnp.asarray(values))
        compact.append(jnp.asarray(packed))
    return dense, compact


@pytest.mark.parametrize("cluster_size,capacity", [(2, 9), (65, 257), (128, 300)])
@pytest.mark.parametrize("degree", [0, 3])
def test_visible_update_matches_dense_optax(cluster_size, capacity, degree):
    rng = np.random.default_rng(cluster_size + degree)
    pool = _pool(capacity, cluster_size, rng)
    state = create_adam_state(pool)
    state = state.replace(
        m=jax.tree.map(lambda x: x + 0.2, state.m),
        v=jax.tree.map(lambda x: x + 0.1, state.v),
        step=jnp.arange(capacity, dtype=jnp.int32),
    )
    for pattern in ([True, False, True, True], [False], [True]):
        visible, clusters = _visible(capacity, cluster_size, pattern)
        dense, compact = _compact(rng, pool, visible, cluster_size, (degree + 1) ** 2)
        kwargs = dict(cluster_size=cluster_size, compact_gradients=True, active_degree=degree)
        expected = jax.jit(lambda p, s, g: optax_update(p, s, g, visible, 37, 2.0, **kwargs))(
            pool, state, tuple(compact)
        )
        actual = jax.jit(
            lambda p, s, g: optax_update(
                p, s, g, visible, 37, 2.0, compacted_clusters=clusters, **kwargs
            )
        )(pool, state, tuple(compact))
        for result, reference in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_array_equal(result, reference)
        untouched = ~np.asarray(visible & pool.alive)
        for result, before in zip(
            jax.tree.leaves((actual[0], actual[1])), jax.tree.leaves((pool, state)), strict=True
        ):
            if result.ndim:
                np.testing.assert_array_equal(
                    np.asarray(result)[untouched], np.asarray(before)[untouched]
                )


def test_visible_update_runs_other_row_local_transforms():
    from jaxgs.kernels.visible_optax import update_visible_clusters

    capacity, cluster_size = 300, 128
    rng = np.random.default_rng(5)
    pool = _pool(capacity, cluster_size, rng)
    params = ParameterArrays(*(getattr(pool, name) for name in PARAMETER_NAMES))
    tx = optax.with_extra_args_support(
        optax.chain(optax.clip(0.5), optax.trace(decay=0.9), optax.scale(-0.05))
    )
    state = tx.init(params)
    visible, clusters = _visible(capacity, cluster_size, [True, False, True])
    dense, compact = _compact(rng, pool, visible, cluster_size, pool.sh.shape[1])
    updated, updated_state = jax.jit(
        lambda p, s, g: update_visible_clusters(
            tx, p, s, g, pool.alive, clusters, cluster_size=cluster_size
        )
    )(params, state, ParameterArrays(*compact))
    updates, dense_state = tx.update(ParameterArrays(*dense), state, params)
    dense_params = optax.apply_updates(params, updates)
    active = np.asarray(visible & pool.alive)
    for result, new, old in zip(
        jax.tree.leaves((updated, updated_state)),
        jax.tree.leaves((dense_params, dense_state)),
        jax.tree.leaves((params, state)),
        strict=True,
    ):
        expected = np.where(active.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)


def test_visible_update_requires_per_slot_state():
    from jaxgs.kernels.visible_optax import update_visible_clusters

    rng = np.random.default_rng(1)
    pool = _pool(16, 8, rng)
    params = ParameterArrays(*(getattr(pool, name) for name in PARAMETER_NAMES))
    tx = optax.with_extra_args_support(optax.scale_by_adam())  # scalar step count
    visible, clusters = _visible(16, 8, [True])
    with pytest.raises(ValueError, match="one row per pool slot"):
        update_visible_clusters(
            tx, params, tx.init(params), params, pool.alive, clusters, cluster_size=8
        )
    truncated = params.replace(xyz=jnp.zeros((16, 4)))
    with pytest.raises(ValueError, match="compact gradients"):
        update_visible_clusters(
            optax.with_extra_args_support(optax.trace(0.9)),
            params,
            optax.trace(0.9).init(params),
            truncated,
            pool.alive,
            clusters,
            cluster_size=8,
        )


def test_dense_update_without_the_triton_backend(monkeypatch):
    import sys

    from jax.experimental import pallas

    from jaxgs.kernels import visible_optax
    from jaxgs.training import optimizer

    try:
        visible_optax.available.cache_clear()
        monkeypatch.setattr(visible_optax.jax, "default_backend", lambda: "cpu")
        assert not visible_optax.available()
        visible_optax.available.cache_clear()
        monkeypatch.undo()
        monkeypatch.delattr(pallas, "triton", raising=False)
        monkeypatch.setitem(sys.modules, "jax.experimental.pallas.triton", None)
        assert not visible_optax.available()
    finally:
        monkeypatch.undo()
        visible_optax.available.cache_clear()
    assert visible_optax.available()
    # Without the kernel, compacted clusters fall back to the dense update.
    rng = np.random.default_rng(2)
    pool = _pool(300, 128, rng)
    state = create_adam_state(pool)
    visible, clusters = _visible(300, 128, [True, False, True])
    _, compact = _compact(rng, pool, visible, 128, 16)
    kwargs = dict(cluster_size=128, compact_gradients=True, active_degree=3)
    expected = optax_update(pool, state, tuple(compact), visible, 5, 1.0, **kwargs)
    monkeypatch.setattr(optimizer, "_visible_kernel_available", lambda: False)
    actual = optax_update(
        pool, state, tuple(compact), visible, 5, 1.0, compacted_clusters=clusters, **kwargs
    )
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("degree", [None, 0, 1, 2, 3])
@pytest.mark.parametrize("fallback", [False, True])
def test_deferred_sh_matches_autodiff_with_invisible_and_free_slots(monkeypatch, degree, fallback):
    from jaxgs.reference.sh import eval_sh
    from jaxgs.training import optimizer

    rng = np.random.default_rng(73)
    capacity, cluster_size = 257, 65
    pool = _pool(capacity, cluster_size, rng)
    center = jnp.array([0.1, -0.2, 0.3])
    pool = pool.replace(xyz=pool.xyz.at[0].set(center))
    state = create_adam_state(pool)
    state = state.replace(
        m=jax.tree.map(lambda x: x + 0.2, state.m),
        v=jax.tree.map(lambda x: x + 0.1, state.v),
    )
    delta = pool.xyz - center
    direction = delta / jnp.maximum(jnp.linalg.norm(delta, axis=1, keepdims=True), 1e-8)
    if fallback:
        monkeypatch.setattr(optimizer, "_visible_kernel_available", lambda: False)
    for pattern in ([True, False, True, True], [False]):
        visible, clusters = _visible(capacity, cluster_size, pattern)
        dense, compact = _compact(rng, pool, visible, cluster_size, 1)
        # Clipping is already included in the supplied color cotangent. Keep
        # reference RGB positive so autodiff only reconstructs the SH basis.
        _, pullback = jax.vjp(
            lambda sh: eval_sh(sh, direction, 3 if degree is None else degree),
            jnp.zeros_like(pool.sh),
        )
        (sh_gradient,) = pullback(dense[-1][:, 0])
        expected = jax.jit(
            lambda p, s, g: optax_update(p, s, g, visible, 37, 2.0, active_degree=degree)
        )(pool, state, (*dense[:-1], sh_gradient))
        actual = jax.jit(
            lambda p, s, g: optax_update(
                p,
                s,
                g,
                visible,
                37,
                2.0,
                cluster_size=cluster_size,
                compact_gradients=True,
                active_degree=degree,
                compacted_clusters=clusters,
                sh_pullback_center=center,
            )
        )(pool, state, tuple(compact))
        for result, reference in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(result, reference, rtol=2e-6, atol=2e-7)
