import jax.numpy as jnp
import numpy as np

from jaxgs.reference.loss import photometric_loss


def test_identical_image_has_zero_photometric_loss():
    for value in (0.0, 0.5, 1.0):
        image = jnp.full((8, 8, 3), value)
        np.testing.assert_allclose(photometric_loss(image, image), 0.0, atol=1e-7)


def test_cute_fused_loss_value_and_image_gradient():
    import importlib.util

    import jax
    import pytest

    if jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None:
        pytest.skip("CuTe fused loss requires JAX CUDA")
    from jaxgs.kernels.fused_loss import fused_loss_and_grad

    rng = np.random.default_rng(41)
    for shape in ((3, 5, 3), (17, 31, 3), (64, 49, 3)):
        prediction = jnp.asarray(rng.uniform(0.02, 0.98, shape), jnp.float32)
        target = jnp.asarray(rng.uniform(0.02, 0.98, shape), jnp.float32)
        with jax.default_matmul_precision("highest"):
            expected_loss, expected_grad = jax.value_and_grad(photometric_loss)(prediction, target)
        loss, gradient = jax.jit(fused_loss_and_grad)(prediction, target)
        np.testing.assert_allclose(loss, expected_loss, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(gradient, expected_grad, rtol=2e-4, atol=2e-7)
    # LiteGS chooses the zero L1 subgradient for equal pixels.
    image = jnp.full((17, 19, 3), 0.5)
    loss, gradient = fused_loss_and_grad(image, image)
    np.testing.assert_allclose(loss, 0, atol=2e-6)
    np.testing.assert_allclose(gradient, 0, atol=2e-7)
