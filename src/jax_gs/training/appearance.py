"""Pure-JAX appearance optimization module matching gsplat's trainer."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from ..config import TrainConfig
from ..math import safe_normalize
from ..model import rgb_to_sh
from ..spherical_harmonics import MAX_SH_DEGREE, _all_sh_bases

APPEARANCE_FEATURE_DIM = 32


def _linear(
    in_features: int,
    out_features: int,
    *,
    rngs: nnx.Rngs,
    zero_init: bool = False,
) -> nnx.Linear:
    if zero_init:
        kernel_init = jax.nn.initializers.zeros
        bias_init = jax.nn.initializers.zeros
    else:
        bound = 1.0 / math.sqrt(in_features)

        def symmetric_uniform(key, shape, dtype=jnp.float32):
            return jax.random.uniform(
                key,
                shape,
                dtype,
                minval=-bound,
                maxval=bound,
            )

        kernel_init = symmetric_uniform
        bias_init = symmetric_uniform
    return nnx.Linear(
        in_features,
        out_features,
        kernel_init=kernel_init,
        bias_init=bias_init,
        rngs=rngs,
    )


class AppearanceOptModule(nnx.Module):
    """Predict per-camera, per-Gaussian color-logit corrections.

    This is the Flax NNX equivalent of ``examples/utils.py`` in gsplat.  The
    output layer starts at zero, matching the initialization applied by
    ``simple_trainer.py`` immediately after constructing the Torch module.
    Consequently a fresh module leaves the trainer's base color logits
    unchanged.
    """

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be positive, got {n}")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if embed_dim < 0:
            raise ValueError(f"embed_dim must be non-negative, got {embed_dim}")
        if not 0 <= sh_degree <= MAX_SH_DEGREE:
            raise ValueError(
                f"sh_degree must be in [0, {MAX_SH_DEGREE}], got {sh_degree}"
            )
        if mlp_width < 1:
            raise ValueError(f"mlp_width must be positive, got {mlp_width}")
        if mlp_depth < 1:
            raise ValueError(f"mlp_depth must be positive, got {mlp_depth}")

        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.sh_degree = sh_degree
        self.embeds = nnx.Embed(
            n,
            embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=1.0),
            rngs=rngs,
        )

        input_dim = embed_dim + feature_dim + (sh_degree + 1) ** 2
        layers = [_linear(input_dim, mlp_width, rngs=rngs), jax.nn.relu]
        for _ in range(mlp_depth - 1):
            layers.extend((_linear(mlp_width, mlp_width, rngs=rngs), jax.nn.relu))
        layers.append(_linear(mlp_width, 3, rngs=rngs, zero_init=True))
        self.color_head = nnx.List(layers)

    def __call__(
        self,
        features: jax.Array,
        embed_ids: jax.Array | None,
        dirs: jax.Array,
        sh_degree: int | jax.Array,
    ) -> jax.Array:
        """Return color-logit corrections with shape ``[C, N, 3]``.

        ``features`` has shape ``[N, feature_dim]``, ``embed_ids`` is either
        ``[C]`` or ``None``, and ``dirs`` has shape ``[C, N, 3]``.
        """

        features = jnp.asarray(features)
        dirs = jnp.asarray(dirs)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [N, feature_dim], got {features.shape}"
            )
        if dirs.ndim != 3 or dirs.shape[-1] != 3:
            raise ValueError(f"dirs must have shape [C, N, 3], got {dirs.shape}")
        if isinstance(sh_degree, int) and not 0 <= sh_degree <= self.sh_degree:
            raise ValueError(
                f"sh_degree must be in [0, {self.sh_degree}], got {sh_degree}"
            )

        camera_count, gaussian_count = dirs.shape[:2]
        if features.shape[0] != gaussian_count:
            raise ValueError(
                "features and dirs Gaussian counts must match, got "
                f"{features.shape[0]} and {gaussian_count}"
            )

        if embed_ids is None:
            camera_embeddings = jnp.zeros(
                (camera_count, self.embed_dim), dtype=features.dtype
            )
        else:
            embed_ids = jnp.asarray(embed_ids)
            if embed_ids.shape != (camera_count,):
                raise ValueError(
                    f"embed_ids must have shape [{camera_count}], got {embed_ids.shape}"
                )
            camera_embeddings = self.embeds(embed_ids)

        camera_embeddings = jnp.broadcast_to(
            camera_embeddings[:, None, :],
            (camera_count, gaussian_count, self.embed_dim),
        )
        gaussian_features = jnp.broadcast_to(
            features[None, :, :],
            (camera_count, gaussian_count, self.feature_dim),
        )

        basis_count = (self.sh_degree + 1) ** 2
        normalized_dirs = safe_normalize(dirs, axis=-1, eps=1.0e-12)
        sh_bases = _all_sh_bases(normalized_dirs)[..., :basis_count]
        requested_basis_count = (jnp.asarray(sh_degree) + 1) ** 2
        sh_bases = sh_bases * (jnp.arange(basis_count) < requested_basis_count)

        if self.embed_dim > 0:
            hidden = jnp.concatenate(
                (camera_embeddings, gaussian_features, sh_bases), axis=-1
            )
        else:
            hidden = jnp.concatenate((gaussian_features, sh_bases), axis=-1)
        for layer in self.color_head:
            hidden = layer(hidden)
        return hidden

    forward = __call__


def create_appearance_optimizer(
    module: AppearanceOptModule, config: TrainConfig
) -> nnx.Optimizer:
    """Create current-main embedding/head Adam parameter groups."""

    base_learning_rate = config.app_opt_lr * math.sqrt(config.data.batch_size)
    transforms = {
        "embeds": optax.chain(
            optax.add_decayed_weights(config.app_opt_reg),
            optax.adam(base_learning_rate * 10.0, eps=1.0e-8),
        ),
        "head": optax.adam(base_learning_rate, eps=1.0e-8),
    }
    parameters = nnx.as_pure(nnx.state(module, nnx.Param))

    def label(path, _value):
        key = getattr(path[0], "key", None) if path else None
        if key == "embeds":
            return "embeds"
        if key == "color_head":
            return "head"
        raise KeyError(f"unexpected appearance parameter path {path!r}")

    labels = jax.tree_util.tree_map_with_path(label, parameters)
    optimizer = nnx.Optimizer(
        module,
        optax.multi_transform(transforms, labels),
        wrt=nnx.Param,
    )
    optimizer._jax_gs_appearance_contract = (
        "appearance_multi_adam_v1",
        config.data.batch_size,
        float(config.app_opt_lr),
        float(config.app_opt_reg),
    )
    return optimizer


def bake_appearance_sh(
    module: AppearanceOptModule,
    features: jax.Array,
    color_logits: jax.Array,
    *,
    sh_degree: int | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Bake the upstream canonical zero-camera appearance into degree-zero SH."""

    features = jnp.asarray(features)
    color_logits = jnp.asarray(color_logits)
    if color_logits.shape != (features.shape[0], 3):
        raise ValueError(
            f"color_logits must have shape [N, 3], got {color_logits.shape}"
        )
    directions = jnp.zeros((1, features.shape[0], 3), dtype=features.dtype)
    corrections = module(features, None, directions, sh_degree)[0]
    rgb = jax.nn.sigmoid(color_logits + corrections)
    sh0 = rgb_to_sh(rgb)[:, None, :]
    sh_rest = jnp.zeros((features.shape[0], 0, 3), dtype=rgb.dtype)
    return sh0, sh_rest


__all__ = [
    "APPEARANCE_FEATURE_DIM",
    "AppearanceOptModule",
    "bake_appearance_sh",
    "create_appearance_optimizer",
]
