import jax
import jax.numpy as jnp
import pytest

import jax_gs
from jax_gs.color_correct import color_correct_affine, color_correct_quadratic


def test_color_correct_root_exports():
    assert jax_gs.color_correct_affine is color_correct_affine
    assert jax_gs.color_correct_quadratic is color_correct_quadratic


def test_affine_color_correction_inverts_per_channel_mapping_and_jits():
    reference = jnp.linspace(0.1, 0.8, 8 * 9 * 3).reshape(8, 9, 3)
    slope = jnp.asarray([0.7, 0.8, 0.9])
    offset = jnp.asarray([0.05, 0.04, 0.03])
    image = reference * slope + offset
    corrected = jax.jit(color_correct_affine)(image, reference)
    assert jnp.allclose(corrected, reference, atol=2.0e-5)


def test_quadratic_color_correction_identity_is_finite_and_improves_error():
    key = jax.random.key(0)
    reference = jax.random.uniform(key, (16, 12, 3), minval=0.05, maxval=0.9)
    image = jnp.clip(reference + 0.08 * reference * reference, 0.0, 1.0)
    corrected = color_correct_quadratic(image, reference, num_iters=2)
    before = jnp.mean(jnp.square(image - reference))
    after = jnp.mean(jnp.square(corrected - reference))
    assert corrected.shape == image.shape
    assert bool(jnp.all(jnp.isfinite(corrected)))
    assert after < before


@pytest.mark.parametrize("function", [color_correct_affine, color_correct_quadratic])
def test_color_correction_rejects_channel_mismatch(function):
    with pytest.raises(ValueError, match="channels must match"):
        function(jnp.zeros((2, 2, 3)), jnp.zeros((2, 2, 1)))
