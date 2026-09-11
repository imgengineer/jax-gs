import cuda.tile as ct
import cuda.tile.jax as ctj
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@ct.kernel
def _add_kernel(a, b, output):
    block = ct.bid(0)
    lhs = ct.load(a, (block,), shape=(128,), padding_mode=ct.PaddingMode.ZERO)
    rhs = ct.load(b, (block,), shape=(128,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(output, (block,), lhs + rhs)


def _add_impl(a, b):
    return ctj.cutile_call(
        (ct.cdiv(a.shape[0], 128),),
        _add_kernel,
        (a, b, ctj.OutputPlaceholder(a.shape, a.dtype)),
    )


@jax.custom_vjp
def _add(a, b):
    return _add_impl(a, b)


def _add_fwd(a, b):
    return _add_impl(a, b), None


def _add_bwd(_, cotangent):
    return cotangent, cotangent


_add.defvjp(_add_fwd, _add_bwd)


def test_cutile_call_runs_under_jit_and_custom_vjp():
    device = jax.devices()[0]
    if device.platform != "gpu" or "cuda" not in str(device).lower():
        pytest.skip("cuTile bridge requires an NVIDIA CUDA GPU")
    a = jnp.arange(257, dtype=jnp.float32)
    b = jnp.linspace(-1.0, 1.0, 257, dtype=jnp.float32)
    actual, gradients = jax.jit(
        jax.value_and_grad(lambda x, y: jnp.sum(_add(x, y)), argnums=(0, 1))
    )(a, b)
    np.testing.assert_allclose(actual, jnp.sum(a + b))
    np.testing.assert_array_equal(gradients[0], jnp.ones_like(a))
    np.testing.assert_array_equal(gradients[1], jnp.ones_like(b))
