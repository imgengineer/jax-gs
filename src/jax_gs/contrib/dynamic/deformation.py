"""NNX deformation network and per-Gaussian dynamic-mask bookkeeping."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


def _linear(
    in_features: int,
    out_features: int,
    *,
    rngs: nnx.Rngs,
    zero_init: bool = False,
) -> nnx.Linear:
    """Construct a Linear layer with the same initialization family as Torch."""

    if zero_init:
        kernel_init = jax.nn.initializers.zeros
        bias_init = jax.nn.initializers.zeros
    else:
        bound = 1.0 / math.sqrt(in_features)
        kernel_init = jax.nn.initializers.uniform(bound)
        bias_init = jax.nn.initializers.uniform(bound)
    return nnx.Linear(
        in_features,
        out_features,
        kernel_init=kernel_init,
        bias_init=bias_init,
        rngs=rngs,
    )


class DeformNetwork(nnx.Module):
    """MLP that adds learned deltas to means, quaternions, and opacities.

    The three output heads start at zero, so a newly constructed network is an
    exact identity map. Time is already encoded by the HexPlane features and
    remains a reserved argument, matching the upstream interface.

    ``rngs`` is optional for source compatibility. Passing an explicit
    ``nnx.Rngs`` is recommended when initialization must be controlled.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}.")

        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = [_linear(feature_dim, hidden_dim, rngs=rngs)]
        layers.extend(
            _linear(hidden_dim, hidden_dim, rngs=rngs) for _ in range(num_layers - 1)
        )
        self.trunk = nnx.List(layers)
        self.pos_head = _linear(hidden_dim, 3, rngs=rngs, zero_init=True)
        self.quat_head = _linear(hidden_dim, 4, rngs=rngs, zero_init=True)
        self.opacity_head = _linear(hidden_dim, 1, rngs=rngs, zero_init=True)

    def __call__(
        self,
        means: jax.Array,
        quats: jax.Array,
        opacities: jax.Array,
        t: jax.Array,
        plane_features: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Apply per-Gaussian deltas while preserving the input shapes."""

        del t
        count = means.shape[0]
        if (
            quats.shape[0] != count
            or opacities.shape[0] != count
            or plane_features.shape[0] != count
        ):
            raise ValueError(
                "DeformNetwork: batch dim mismatch — "
                f"means {means.shape[0]}, quats {quats.shape[0]}, "
                f"opacities {opacities.shape[0]}, "
                f"plane_features {plane_features.shape[0]}."
            )
        if plane_features.shape[-1] != self.feature_dim:
            raise ValueError(
                "DeformNetwork: plane_features last dim "
                f"{plane_features.shape[-1]} != feature_dim {self.feature_dim}."
            )
        if not (means.dtype == quats.dtype == opacities.dtype == plane_features.dtype):
            raise ValueError(
                "DeformNetwork: dtype mismatch — "
                f"means {means.dtype}, quats {quats.dtype}, "
                f"opacities {opacities.dtype}, "
                f"plane_features {plane_features.dtype}."
            )

        hidden = plane_features
        for layer in self.trunk:
            hidden = jax.nn.relu(layer(hidden))
        return (
            means + self.pos_head(hidden),
            quats + self.quat_head(hidden),
            opacities + self.opacity_head(hidden),
        )

    forward = __call__


class DeformationTable:
    """Mutable JAX boolean table that follows per-Gaussian topology changes."""

    def __init__(
        self,
        num_gaussians: int,
        device: jax.Device | None = None,
    ) -> None:
        if num_gaussians < 0:
            raise ValueError(
                f"DeformationTable: num_gaussians must be >= 0, got {num_gaussians}."
            )
        mask = jnp.zeros((num_gaussians,), dtype=jnp.bool_)
        self.mask = jax.device_put(mask, device) if device is not None else mask

    def __len__(self) -> int:
        return int(self.mask.shape[0])

    def set_indices(self, indices: jax.Array, value: bool = True) -> None:
        """Set the dynamic flag for the selected Gaussian rows."""

        self.mask = self.mask.at[indices].set(value)

    def prune(self, keep_mask: jax.Array) -> None:
        """Keep only rows selected by a concrete boolean mask."""

        if keep_mask.shape != self.mask.shape:
            raise ValueError(
                "DeformationTable.prune: keep_mask shape "
                f"{tuple(keep_mask.shape)} != table shape {tuple(self.mask.shape)}."
            )
        self.mask = self.mask[keep_mask]

    def duplicate(self, indices: jax.Array) -> None:
        """Append one child per index, inheriting each parent's flag."""

        self.mask = jnp.concatenate((self.mask, self.mask[indices]), axis=0)

    def split(self, indices: jax.Array, factor: int = 2) -> None:
        """Replace selected parents with ``factor`` inheriting children."""

        if factor < 1:
            raise ValueError(
                f"DeformationTable.split: factor must be >= 1, got {factor}."
            )
        keep = jnp.ones(self.mask.shape, dtype=jnp.bool_)
        keep = keep.at[indices].set(False)
        children = jnp.repeat(self.mask[indices], factor)
        self.mask = jnp.concatenate((self.mask[keep], children), axis=0)


__all__ = ["DeformNetwork", "DeformationTable"]
