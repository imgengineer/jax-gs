import jax
import jax.numpy as jnp
import numpy as np

from jax_gs.spherical_harmonics import eval_sh_bases, spherical_harmonics


def test_degree_zero_known_value() -> None:
    directions = jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32)
    coefficients = jnp.ones((1, 1, 3), dtype=jnp.float32)
    result = spherical_harmonics(0, directions, coefficients)
    np.testing.assert_allclose(result, np.full((1, 3), 0.2820948), rtol=1e-6)


def test_degree_one_z_basis_and_shapes_through_degree_four() -> None:
    directions = jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32)
    coefficients = jnp.zeros((1, 25, 3), dtype=jnp.float32).at[:, 2, :].set(1.0)
    result = spherical_harmonics(1, directions, coefficients)
    np.testing.assert_allclose(result, np.full((1, 3), 0.48860252), rtol=1e-6)
    for degree in range(5):
        assert eval_sh_bases(degree, directions).shape == (1, (degree + 1) ** 2)


def test_dynamic_degree_is_jittable_and_mask_is_respected() -> None:
    directions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    coefficients = jnp.ones((2, 25, 3), dtype=jnp.float32)
    compiled = jax.jit(spherical_harmonics)
    result0 = compiled(jnp.array(0), directions, coefficients)
    result4 = compiled(jnp.array(4), directions, coefficients)
    assert result0.shape == result4.shape == (2, 3)
    assert not np.allclose(result0, result4)
    masked = spherical_harmonics(4, directions, coefficients, jnp.array([True, False]))
    np.testing.assert_array_equal(masked[1], 0.0)


def test_spherical_harmonics_has_finite_gradients() -> None:
    directions = jnp.array([[0.2, -0.3, 0.9]], dtype=jnp.float32)
    coefficients = jnp.arange(75, dtype=jnp.float32).reshape(1, 25, 3) / 100.0
    objective = lambda value: jnp.sum(spherical_harmonics(4, value, coefficients))
    gradient = jax.jit(jax.grad(objective))(directions)
    assert jnp.all(jnp.isfinite(gradient))
