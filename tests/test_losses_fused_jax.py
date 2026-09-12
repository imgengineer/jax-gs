import jax
import jax.numpy as jnp
from flax import nnx

from jax_gs.losses import (
    gaussian_density_reg,
    gaussian_scale_reg,
    gaussian_z_scale_reg,
    out_of_bound_loss,
)
from jax_gs.losses_fused import FusedGaussianLosses


def _inputs():
    return (
        jnp.asarray([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]),
        jnp.asarray([0.2, 0.8]),
        jnp.asarray([0.1, 0.9]),
        jnp.asarray([[0.0, 2.0, 0.0], [3.0, 0.0, -2.0]]),
        jnp.full((2, 3), 2.0),
        jnp.asarray([1.0, 0.0]),
    )


def test_fused_gaussian_losses_match_individual_functions_and_jit():
    values = _inputs()
    module = FusedGaussianLosses(z_scale_threshold=0.5)
    actual = nnx.jit(lambda model, args: model(*args))(module, values)
    expected = (
        gaussian_scale_reg(values[0], values[5]),
        gaussian_density_reg(values[1], values[5]),
        gaussian_z_scale_reg(values[2], 0.5),
        out_of_bound_loss(values[3], values[4]),
    )
    for result, reference in zip(actual, expected):
        assert jnp.array_equal(result, reference)


def test_fused_gaussian_losses_have_direct_jax_gradients():
    module = FusedGaussianLosses(z_scale_threshold=0.5)
    scales, densities, z_scales, positions, cuboid_dims, visibility = _inputs()

    def objective(a, b, c, d):
        outputs = module(a, b, c, d, cuboid_dims, visibility)
        return sum(jnp.sum(output) for output in outputs)

    gradients = jax.grad(objective, argnums=(0, 1, 2, 3))(
        scales,
        densities,
        z_scales,
        positions,
    )
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients)


def test_fused_gaussian_losses_empty_shapes():
    outputs = FusedGaussianLosses(0.5)(
        jnp.empty((0, 3)),
        jnp.empty((0,)),
        jnp.empty((0,)),
        jnp.empty((0, 3)),
        jnp.empty((0, 3)),
    )
    assert [value.shape for value in outputs] == [(0, 3), (0,), (0,), (0, 3)]
