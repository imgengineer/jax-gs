import jax
import jax.numpy as jnp
import numpy as np

from jax_gs.math import (
    full_to_triu,
    quat_scale_to_covar_preci,
    quat_to_rotmat,
    safe_normalize,
    triu_to_full,
)


def test_safe_normalize_known_values_and_zero() -> None:
    values = jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    normalized = jax.jit(safe_normalize)(values)
    np.testing.assert_allclose(normalized[0], [0.6, 0.8, 0.0], rtol=1e-6)
    np.testing.assert_array_equal(normalized[1], [0.0, 0.0, 0.0])
    gradient = jax.grad(lambda x: jnp.sum(safe_normalize(x)))(values)
    assert jnp.all(jnp.isfinite(gradient))


def test_quaternion_rotation_known_values_and_vmap() -> None:
    quats = jnp.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32
    )
    rotations = jax.jit(quat_to_rotmat)(quats)
    np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(rotations[1], np.diag([-1.0, -1.0, 1.0]), atol=1e-6)
    vmapped = jax.vmap(quat_to_rotmat)(quats)
    np.testing.assert_allclose(vmapped, rotations, atol=1e-6)


def test_quaternion_scale_covariance_precision_full_and_triu() -> None:
    quat = jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    scale = jnp.array([[2.0, 3.0, 4.0]], dtype=jnp.float32)
    covariance, precision = quat_scale_to_covar_preci(quat, scale)
    np.testing.assert_allclose(covariance[0], np.diag([4.0, 9.0, 16.0]), atol=1e-6)
    np.testing.assert_allclose(
        precision[0], np.diag([0.25, 1.0 / 9.0, 1.0 / 16.0]), atol=1e-6
    )

    covariance_triu, precision_triu = quat_scale_to_covar_preci(
        quat, scale, triu=True
    )
    assert covariance_triu.shape == (1, 6)
    assert precision_triu.shape == (1, 6)
    np.testing.assert_allclose(triu_to_full(covariance_triu), covariance, atol=1e-6)
    np.testing.assert_allclose(full_to_triu(precision), precision_triu, atol=1e-6)


def test_covariance_has_finite_gradients() -> None:
    quat = jnp.array([0.8, 0.1, -0.2, 0.3], dtype=jnp.float32)
    scale = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)

    def objective(q: jax.Array, s: jax.Array) -> jax.Array:
        covariance, precision = quat_scale_to_covar_preci(q, s)
        return jnp.sum(covariance) + 1e-3 * jnp.sum(precision)

    gradients = jax.jit(jax.grad(objective, argnums=(0, 1)))(quat, scale)
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients)
