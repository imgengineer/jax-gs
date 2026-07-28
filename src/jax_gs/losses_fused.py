"""Readable pure-JAX facade for the current-main fused Gaussian losses."""

from __future__ import annotations

from flax import nnx
import jax

from .losses import (
    gaussian_density_reg,
    gaussian_scale_reg,
    gaussian_z_scale_reg,
    out_of_bound_loss,
)


class FusedGaussianLosses(nnx.Module):
    """Compute the four Gaussian regularizers through one NNX call boundary."""

    def __init__(self, z_scale_threshold: float = 0.0) -> None:
        self.z_scale_threshold = z_scale_threshold

    def __call__(
        self,
        scales: jax.Array,
        densities: jax.Array,
        z_scales: jax.Array,
        positions: jax.Array,
        cuboid_dims: jax.Array,
        visibility: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return (
            gaussian_scale_reg(scales, visibility=visibility),
            gaussian_density_reg(densities, visibility=visibility),
            gaussian_z_scale_reg(z_scales, self.z_scale_threshold),
            out_of_bound_loss(positions, cuboid_dims),
        )

    forward = __call__


__all__ = ["FusedGaussianLosses"]
