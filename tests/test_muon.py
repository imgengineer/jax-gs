"""Muon for each Gaussian's SH color map, with LiteGS Adam for the other fields."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optax.contrib._muon import MuonDimensionNumbers, orthogonalize_via_newton_schulz

from jaxgs import CapacityConfig, create_gaussians, seed_gaussians
from jaxgs.scene.types import PARAMETER_NAMES, ParameterArrays
from jaxgs.training.optimizer import (
    MUON_PROGRAM_SHAPE,
    MUON_UPDATE_RMS,
    _orthogonalize,
    _orthogonalize_rank_one,
    _parameter_learning_rates,
    create_adam_state,
    create_adam_transform,
    create_muon_transform,
    optax_update,
    reset_adam_slots,
)

_COEFFICIENTS = jnp.asarray((3.4445, -4.7750, 2.0315))


def _optax_orthogonalize(x):
    """optax.contrib.muon's orthogonalization of each [R, C] matrix in [B, R, C]."""
    with jax.default_matmul_precision("highest"):
        return orthogonalize_via_newton_schulz(
            x, _COEFFICIENTS, 5, "frobenius", 1e-8, MuonDimensionNumbers(1, 2)
        )


def _relative(actual, expected):
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


@pytest.mark.parametrize("rows", [15, 8, 3, 1])
def test_newton_schulz_matches_optax_contrib_muon(rows):
    rng = np.random.default_rng(rows)
    # Matrices of very different scales, including nearly zero ones.
    scales = 10 ** rng.uniform(-6, 2, (64, 1, 1))
    x = jnp.asarray(rng.normal(size=(64, rows, 3)) * scales, jnp.float32)
    expected = _optax_orthogonalize(x)
    assert _relative(_orthogonalize(x, 3), expected) < 1e-5
    # Zero rows and a zero fourth column (executor padding) change nothing.
    padded = jnp.pad(x, ((0, 0), (1, 16 - rows - 1), (0, 1)))
    np.testing.assert_allclose(
        _orthogonalize(padded, 3)[:, 1 : 1 + rows, :3], _orthogonalize(x, 3), rtol=1e-5, atol=1e-6
    )
    if rows == 1:
        assert _relative(_orthogonalize_rank_one(x), expected) < 1e-5
        np.testing.assert_allclose(
            _orthogonalize_rank_one(padded)[:, 1:2, :3],
            _orthogonalize_rank_one(x),
            rtol=1e-5,
            atol=1e-6,
        )


def _pool(capacity, rng):
    pool = seed_gaussians(
        create_gaussians(CapacityConfig(capacity, 2, 8, 4, 3)),
        jnp.asarray(rng.normal(size=(capacity, 3)), jnp.float32),
        jnp.asarray(rng.uniform(size=(capacity, 3)), jnp.float32),
    )
    return pool.replace(sh=jnp.asarray(rng.normal(size=pool.sh.shape), jnp.float32))


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_muon_update_matches_reference(degree):
    capacity = 64
    rng = np.random.default_rng(degree)
    pool = _pool(capacity, rng)
    params = ParameterArrays(*(getattr(pool, name) for name in PARAMETER_NAMES))
    gradients = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape), jnp.float32), params)
    state = create_adam_state(pool)
    state = state.replace(
        m=jax.tree.map(lambda x: x + jnp.asarray(rng.normal(size=x.shape), x.dtype), state.m),
        v=jax.tree.map(lambda x: x + 0.1, state.v),
        step=jnp.asarray(rng.integers(0, 4, capacity), jnp.int32),
    )
    active = jnp.asarray(rng.random(capacity) < 0.7)
    rates = _parameter_learning_rates(16, 100, 2.0, None, _default_optimization())
    sh_dim = (degree + 1) ** 2
    updates, new_state = create_muon_transform(sh_dim).update(
        gradients, state, params, active=active, rates=rates
    )
    adam_updates, adam_state = create_adam_transform().update(
        gradients, state, params, active=active, rates=rates
    )
    # Position, scale, rotation and opacity keep LiteGS Adam.
    for name in PARAMETER_NAMES[:-1]:
        np.testing.assert_array_equal(getattr(updates, name), getattr(adam_updates, name))
        np.testing.assert_array_equal(getattr(new_state.m, name), getattr(adam_state.m, name))
        np.testing.assert_array_equal(getattr(new_state.v, name), getattr(adam_state.v, name))
    np.testing.assert_array_equal(new_state.step, state.step + active)
    # SH: Optax's Nesterov momentum, bias-corrected per slot, orthogonalized per
    # Gaussian as a DC row and a [sh_dim - 1, 3] matrix of higher coefficients.
    count = (np.asarray(state.step, np.float64) + 1)[:, None, None]
    g, m = np.asarray(gradients.sh, np.float64), np.asarray(state.m.sh, np.float64)
    momentum = 0.95 * m + 0.05 * g
    nesterov = 0.95 * momentum / (1 - 0.95 ** (count + 1)) + 0.05 * g / (1 - 0.95**count)
    nesterov = jnp.asarray(nesterov, jnp.float32)
    expected = jnp.zeros_like(nesterov)
    expected = expected.at[:, :1].set(
        MUON_UPDATE_RMS * 3**0.5 * _optax_orthogonalize(nesterov[:, :1])
    )
    if sh_dim > 1:
        rest = nesterov[:, 1:sh_dim]
        expected = expected.at[:, 1:sh_dim].set(
            MUON_UPDATE_RMS * max(sh_dim - 1, 3) ** 0.5 * _optax_orthogonalize(rest)
        )
    expected = jnp.where(active[:, None, None], -rates.sh * expected, 0)
    np.testing.assert_allclose(updates.sh, expected, rtol=2e-5, atol=1e-9)
    expected_momentum = np.where(np.asarray(active)[:, None, None], momentum, m)
    np.testing.assert_allclose(new_state.m.sh, expected_momentum, rtol=1e-6, atol=1e-7)
    # Muon keeps no SH second moments; frozen slots keep their state.
    assert new_state.v.sh is state.v.sh
    inactive = ~np.asarray(active)
    np.testing.assert_array_equal(np.asarray(updates.sh)[inactive], 0)


def _default_optimization():
    from jaxgs.config import load_config

    return load_config().optimization


def test_muon_update_rms_matches_its_setting():
    """Each orthogonalized full-rank matrix has an update RMS of about update_rms."""
    rng = np.random.default_rng(7)
    pool = _pool(256, rng)
    params = ParameterArrays(*(getattr(pool, name) for name in PARAMETER_NAMES))
    gradients = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape), jnp.float32), params)
    rates = ParameterArrays(1.0, 1.0, 1.0, 1.0, 1.0)
    updates, _ = create_muon_transform(16, 0.4).update(
        gradients, create_adam_state(pool), params, active=pool.alive, rates=rates
    )
    rest_rms = jnp.sqrt(jnp.mean(updates.sh[:, 1:] ** 2, axis=(1, 2)))
    # Newton-Schulz leaves singular values in about [0.68, 1.19] (as in Optax).
    assert 0.4 * 0.68 < float(jnp.median(rest_rms)) < 0.4 * 1.19


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="The visible-cluster Optax kernel requires a JAX CUDA device",
)
@pytest.mark.parametrize("cluster_size,capacity,degree", [(128, 300, 3), (65, 257, 1), (2, 9, 0)])
def test_visible_muon_matches_dense_update(cluster_size, capacity, degree):
    from test_visible_optax import _compact, _visible
    from test_visible_optax import _pool as _cluster_pool

    from jaxgs.kernels.visible_optax import ProgramShape

    rng = np.random.default_rng(cluster_size)
    pool = _cluster_pool(capacity, cluster_size, rng)
    state = create_adam_state(pool)
    state = state.replace(
        m=jax.tree.map(lambda x: x + 0.2, state.m),
        step=jnp.arange(capacity, dtype=jnp.int32) % 5,
    )
    transform = create_muon_transform((degree + 1) ** 2)
    for pattern in ([True, False, True, True], [False], [True]):
        visible, clusters = _visible(capacity, cluster_size, pattern)
        _, compact = _compact(rng, pool, visible, cluster_size, (degree + 1) ** 2)
        kwargs = dict(
            cluster_size=cluster_size,
            compact_gradients=True,
            active_degree=degree,
            transform=transform,
        )
        expected = jax.jit(lambda p, s, g: optax_update(p, s, g, visible, 37, 2.0, **kwargs))(
            pool, state, tuple(compact)
        )
        for shape in (MUON_PROGRAM_SHAPE, ProgramShape()):
            actual = jax.jit(
                lambda p, s, g, shape=shape: optax_update(
                    p,
                    s,
                    g,
                    visible,
                    37,
                    2.0,
                    compacted_clusters=clusters,
                    program_shape=shape,
                    **kwargs,
                )
            )(pool, state, tuple(compact))
            for result, reference in zip(
                jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
            ):
                np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-9)


def test_muon_state_supports_slot_resets():
    rng = np.random.default_rng(3)
    pool = _pool(32, rng)
    params = ParameterArrays(*(getattr(pool, name) for name in PARAMETER_NAMES))
    transform = create_muon_transform(16)
    state = transform.init(params)
    gradients = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape), jnp.float32), params)
    rates = ParameterArrays(1e-3, 1e-3, 1e-3, 1e-3, 1e-3)
    _, state = transform.update(gradients, state, params, active=pool.alive, rates=rates)
    assert float(jnp.max(jnp.abs(state.m.sh))) > 0 and not np.any(state.v.sh)
    reset = jnp.arange(32) % 3 == 0
    state = reset_adam_slots(state, reset)
    np.testing.assert_array_equal(np.asarray(state.m.sh)[np.asarray(reset)], 0)
    np.testing.assert_array_equal(np.asarray(state.step)[np.asarray(reset)], 0)
