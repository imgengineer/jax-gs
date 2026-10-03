"""Deferred SH gradients agree with an independent automatic derivative."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs.reference.sh import eval_sh
from jaxgs.scene.types import ParameterArrays
from jaxgs.training.optimizer import create_adam_transform
from jaxgs.training.sh_pullback import with_sh_pullback


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("channels", [3, 4])
def test_sh_reconstruction_against_autodiff(degree, channels):
    rng = np.random.default_rng(84 + degree)
    count = 17
    center = jnp.asarray([0.1, -0.3, 0.2] + ([0.0] if channels == 4 else []))
    xyz = jnp.asarray(rng.normal(size=(count, channels)), jnp.float32)
    xyz = xyz.at[0].set(center)
    params = ParameterArrays(
        xyz,
        jnp.zeros((count, 3)),
        jnp.zeros((count, 4)),
        jnp.zeros((count, 1)),
        jnp.zeros((count, 16, channels)),
    )
    colors = jnp.asarray(rng.normal(size=(count, 1, channels)), jnp.float32).at[::5].set(0)
    delta = xyz[:, :3] - center[:3]
    direction = delta / jnp.maximum(jnp.linalg.norm(delta, axis=1, keepdims=True), 1e-8)
    _, pullback = jax.vjp(lambda sh: eval_sh(sh, direction, degree), params.sh)
    (expected_sh,) = pullback(colors[:, 0])
    gradients = jax.tree.map(jnp.zeros_like, params).replace(sh=colors)
    tx = create_adam_transform()
    state = tx.init(params)
    state = state.replace(
        m=jax.tree.map(lambda x: x + 0.03, state.m), v=jax.tree.map(lambda x: x + 0.01, state.v)
    )
    active = jnp.arange(count) % 3 != 0
    rates = jax.tree.map(lambda _: 0.004, params)
    expected = tx.update(
        gradients.replace(sh=expected_sh), state, params, active=active, rates=rates
    )
    actual = jax.jit(
        lambda g, s, p: with_sh_pullback(tx, degree).update(
            g, s, p, sh_center=center[None], active=active, rates=rates
        )
    )(gradients, state, params)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
