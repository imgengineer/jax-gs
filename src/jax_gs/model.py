from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from .config import MAX_MODEL_CAPACITY, ModelConfig
from .init_utils import knn_scale_init


SH_C0 = 0.28209479177387814


class MeansParam(nnx.Param):
    pass


class ScalesParam(nnx.Param):
    pass


class QuatsParam(nnx.Param):
    pass


class OpacitiesParam(nnx.Param):
    pass


class Sh0Param(nnx.Param):
    pass


class ShRestParam(nnx.Param):
    pass


class FeaturesParam(nnx.Param):
    pass


class ColorsParam(nnx.Param):
    pass


def inverse_sigmoid(value: jax.Array | float) -> jax.Array:
    value = jnp.asarray(value)
    value = jnp.clip(value, 1.0e-6, 1.0 - 1.0e-6)
    return jnp.log(value) - jnp.log1p(-value)


def rgb_to_sh(rgb: jax.Array) -> jax.Array:
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh: jax.Array) -> jax.Array:
    return sh * SH_C0 + 0.5


class GaussianModel(nnx.Module):
    """Bucketed Flax NNX storage for 3D Gaussian parameters.

    Shapes remain static within one physical bucket. Densification and pruning
    mutate ``active_mask`` and slot contents; when a bucket fills, the model,
    Adam state, and strategy state grow together to the next bucket.
    """

    def __init__(
        self,
        means: jax.Array,
        log_scales: jax.Array,
        quats: jax.Array,
        opacity_logits: jax.Array,
        sh0: jax.Array | None,
        sh_rest: jax.Array | None,
        active_mask: jax.Array,
        *,
        features: jax.Array | None = None,
        colors: jax.Array | None = None,
        max_capacity: int | None = None,
    ) -> None:
        capacity = int(means.shape[0])
        if means.shape != (capacity, 3):
            raise ValueError(f"means must have shape [capacity, 3], got {means.shape}")
        if log_scales.shape != (capacity, 3):
            raise ValueError("log_scales must match means")
        if quats.shape != (capacity, 4):
            raise ValueError("quats must have shape [capacity, 4]")
        if opacity_logits.shape != (capacity,):
            raise ValueError("opacity_logits must have shape [capacity]")
        has_sh = sh0 is not None or sh_rest is not None
        has_appearance = features is not None or colors is not None
        if (sh0 is None) != (sh_rest is None):
            raise ValueError("sh0 and sh_rest must be provided together")
        if (features is None) != (colors is None):
            raise ValueError("features and colors must be provided together")
        if has_sh == has_appearance:
            raise ValueError(
                "exactly one of SH or appearance colors must be provided"
            )
        if has_sh:
            assert sh0 is not None and sh_rest is not None
            if sh0.shape != (capacity, 1, 3):
                raise ValueError("sh0 must have shape [capacity, 1, 3]")
            if (
                sh_rest.ndim != 3
                or sh_rest.shape[0] != capacity
                or sh_rest.shape[-1] != 3
            ):
                raise ValueError("sh_rest must have shape [capacity, K-1, 3]")
        else:
            assert features is not None and colors is not None
            if (
                features.ndim != 2
                or features.shape[0] != capacity
                or features.shape[1] < 1
            ):
                raise ValueError(
                    "features must have shape [capacity, feature_dim]"
                )
            if colors.shape != (capacity, 3):
                raise ValueError("colors must have shape [capacity, 3]")
        if active_mask.shape != (capacity,):
            raise ValueError("active_mask must have shape [capacity]")
        if max_capacity is None:
            max_capacity = capacity
        max_capacity = int(max_capacity)
        if max_capacity > MAX_MODEL_CAPACITY:
            raise ValueError(
                f"max_capacity cannot exceed {MAX_MODEL_CAPACITY:,}"
            )
        if max_capacity < capacity:
            raise ValueError("max_capacity cannot be smaller than physical capacity")

        self.max_capacity = max_capacity
        self.means = MeansParam(jnp.asarray(means, dtype=jnp.float32))
        self.log_scales = ScalesParam(jnp.asarray(log_scales, dtype=jnp.float32))
        self.quats = QuatsParam(jnp.asarray(quats, dtype=jnp.float32))
        self.opacity_logits = OpacitiesParam(
            jnp.asarray(opacity_logits, dtype=jnp.float32)
        )
        if has_sh:
            assert sh0 is not None and sh_rest is not None
            self.sh0 = Sh0Param(jnp.asarray(sh0, dtype=jnp.float32))
            self.sh_rest = ShRestParam(
                jnp.asarray(sh_rest, dtype=jnp.float32)
            )
        else:
            assert features is not None and colors is not None
            self.features = FeaturesParam(
                jnp.asarray(features, dtype=jnp.float32)
            )
            self.colors = ColorsParam(jnp.asarray(colors, dtype=jnp.float32))
        self.active_mask = nnx.Variable(jnp.asarray(active_mask, dtype=jnp.bool_))

    @property
    def capacity(self) -> int:
        """Current physical array length, not the logical maximum."""

        return int(self.means[...].shape[0])

    @property
    def sh_degree(self) -> int:
        if self.has_appearance:
            raise ValueError("appearance models do not store SH coefficients")
        basis_count = 1 + int(self.sh_rest[...].shape[1])
        return int(round(basis_count**0.5)) - 1

    @property
    def sh_coeffs(self) -> jax.Array:
        if self.has_appearance:
            raise ValueError("appearance models do not store SH coefficients")
        return jnp.concatenate([self.sh0[...], self.sh_rest[...]], axis=1)

    @property
    def has_appearance(self) -> bool:
        return hasattr(self, "features")

    @property
    def scales(self) -> jax.Array:
        return jnp.exp(self.log_scales[...])

    @property
    def normalized_quats(self) -> jax.Array:
        norms = jnp.linalg.norm(self.quats[...], axis=-1, keepdims=True)
        identity = jnp.zeros_like(self.quats[...]).at[:, 0].set(1.0)
        return jnp.where(norms > 1.0e-12, self.quats[...] / norms, identity)

    @property
    def opacities(self) -> jax.Array:
        return jax.nn.sigmoid(self.opacity_logits[...]) * self.active_mask[...]

    @property
    def active_count(self) -> jax.Array:
        return jnp.count_nonzero(self.active_mask[...])

    def activated(
        self,
        *,
        sh_degree: int | None = None,
        split_sh: bool = False,
    ) -> dict[str, Any]:
        if self.has_appearance:
            if sh_degree is not None or split_sh:
                raise ValueError(
                    "SH activation options are unavailable for appearance models"
                )
            return {
                "means": self.means[...],
                "quats": self.quats[...],
                "scales": self.scales,
                "opacities": self.opacities,
                "features": self.features[...],
                "colors": self.colors[...],
                "active_mask": self.active_mask[...],
            }

        sh0 = self.sh0[...]
        sh_rest = self.sh_rest[...]
        stored_basis_count = 1 + sh_rest.shape[1]
        if sh_degree is not None:
            basis_count = (sh_degree + 1) ** 2
            if basis_count > stored_basis_count:
                raise ValueError(
                    f"requested SH degree {sh_degree}, model only stores degree "
                    f"{self.sh_degree}"
                )
            sh_rest = sh_rest[:, : basis_count - 1]
        coeffs = (
            (sh0, sh_rest)
            if split_sh
            else jnp.concatenate([sh0, sh_rest], axis=1)
        )
        return {
            "means": self.means[...],
            # Projection normalizes defensively, and training normalizes the
            # stored value after every optimizer update. Avoid a redundant
            # full-capacity norm/div pass here for million-slot buckets.
            "quats": self.quats[...],
            "scales": self.scales,
            "opacities": self.opacities,
            "sh_coeffs": coeffs,
            "active_mask": self.active_mask[...],
        }

    def normalize_quaternions(self) -> None:
        self.quats[...] = self.normalized_quats

    def state_dict(self) -> dict[str, jax.Array]:
        state = {
            "means": self.means[...],
            "log_scales": self.log_scales[...],
            "quats": self.quats[...],
            "opacity_logits": self.opacity_logits[...],
            "active_mask": self.active_mask[...],
        }
        if self.has_appearance:
            state["features"] = self.features[...]
            state["colors"] = self.colors[...]
        else:
            state["sh0"] = self.sh0[...]
            state["sh_rest"] = self.sh_rest[...]
        return state

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        max_capacity: int | None = None,
    ) -> "GaussianModel":
        arrays = {name: jnp.asarray(value) for name, value in state.items()}
        return cls(
            arrays["means"],
            arrays["log_scales"],
            arrays["quats"],
            arrays["opacity_logits"],
            arrays.get("sh0"),
            arrays.get("sh_rest"),
            arrays["active_mask"],
            features=arrays.get("features"),
            colors=arrays.get("colors"),
            max_capacity=max_capacity,
        )

    @classmethod
    def empty(
        cls,
        config: ModelConfig = ModelConfig(),
        *,
        device: jax.Device | None = None,
        physical_capacity: int | None = None,
        appearance_feature_dim: int | None = None,
    ) -> "GaussianModel":
        capacity = (
            config.bucket_capacity()
            if physical_capacity is None
            else int(physical_capacity)
        )
        if capacity <= 0 or capacity > config.capacity:
            raise ValueError("physical_capacity must be in [1, config.capacity]")
        means = jnp.zeros((capacity, 3), dtype=jnp.float32)
        log_scales = jnp.full(
            (capacity, 3), jnp.log(config.initial_scale), dtype=jnp.float32
        )
        quats = jnp.zeros((capacity, 4), dtype=jnp.float32).at[:, 0].set(1.0)
        opacity_logits = jnp.full(
            (capacity,), inverse_sigmoid(config.initial_opacity), dtype=jnp.float32
        )
        active_mask = jnp.zeros((capacity,), dtype=jnp.bool_)
        if appearance_feature_dim is not None:
            if appearance_feature_dim < 1:
                raise ValueError("appearance_feature_dim must be positive")
            sh0 = None
            sh_rest = None
            features = jnp.zeros(
                (capacity, appearance_feature_dim), dtype=jnp.float32
            )
            colors = jnp.zeros((capacity, 3), dtype=jnp.float32)
        else:
            basis_count = (config.sh_degree + 1) ** 2
            sh0 = jnp.zeros((capacity, 1, 3), dtype=jnp.float32)
            sh_rest = jnp.zeros(
                (capacity, basis_count - 1, 3), dtype=jnp.float32
            )
            features = None
            colors = None
        if device is not None:
            means, log_scales, quats, opacity_logits, active_mask = (
                jax.device_put(
                    (means, log_scales, quats, opacity_logits, active_mask),
                    device,
                )
            )
            if sh0 is not None:
                sh0, sh_rest = jax.device_put((sh0, sh_rest), device)
            else:
                features, colors = jax.device_put((features, colors), device)
        return cls(
            means,
            log_scales,
            quats,
            opacity_logits,
            sh0,
            sh_rest,
            active_mask,
            features=features,
            colors=colors,
            max_capacity=config.capacity,
        )

    @classmethod
    def from_point_cloud(
        cls,
        points: np.ndarray | jax.Array,
        colors: np.ndarray | jax.Array,
        config: ModelConfig = ModelConfig(),
        *,
        device: jax.Device | None = None,
        physical_capacity: int | None = None,
        num_workers: int = 4,
        appearance_feature_dim: int | None = None,
        feature_key: jax.Array | None = None,
    ) -> "GaussianModel":
        points_np = np.asarray(points, dtype=np.float32)
        colors_np = np.asarray(colors, dtype=np.float32)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            raise ValueError("points must have shape [N, 3]")
        if colors_np.shape != points_np.shape:
            raise ValueError("colors must have shape [N, 3]")
        count = points_np.shape[0]
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if count > config.capacity:
            raise ValueError(
                f"point cloud contains {count} points, capacity is {config.capacity}"
            )
        if colors_np.size and colors_np.max() > 1.0:
            colors_np = colors_np / 255.0
        colors_np = np.clip(colors_np, 0.0, 1.0)

        initial_log_scales = np.full(
            (count, 3), np.log(config.initial_scale), np.float32
        )
        if count > 1:
            neighbor_count = min(3, count - 1)
            neighbor_log_scales = np.asarray(
                knn_scale_init(jnp.asarray(points_np), k=neighbor_count)
            ) + np.float32(np.log(config.initial_scale))
            initial_log_scales = np.repeat(
                neighbor_log_scales[:, None], 3, axis=1
            )

        capacity = (
            config.bucket_capacity(count)
            if physical_capacity is None
            else int(physical_capacity)
        )
        if capacity < count or capacity > config.capacity:
            raise ValueError(
                "physical_capacity must cover the point cloud and not exceed "
                "the logical maximum"
            )
        means = np.zeros((capacity, 3), np.float32)
        means[:count] = points_np
        log_scales = np.full(
            (capacity, 3), np.log(config.initial_scale), dtype=np.float32
        )
        log_scales[:count] = initial_log_scales
        quats = np.zeros((capacity, 4), np.float32)
        quats[:, 0] = 1.0
        opacity_logits = np.full(
            (capacity,), float(inverse_sigmoid(config.initial_opacity)), np.float32
        )
        active_mask = np.zeros((capacity,), np.bool_)
        active_mask[:count] = True

        if appearance_feature_dim is not None:
            if appearance_feature_dim < 1:
                raise ValueError("appearance_feature_dim must be positive")
            if feature_key is None:
                raise ValueError(
                    "feature_key is required for appearance initialization"
                )
            sh0 = None
            sh_rest = None
            features = np.zeros(
                (capacity, appearance_feature_dim), dtype=np.float32
            )
            features[:count] = np.asarray(
                jax.random.uniform(
                    feature_key,
                    (count, appearance_feature_dim),
                    dtype=jnp.float32,
                )
            )
            color_logits = np.zeros((capacity, 3), dtype=np.float32)
            color_logits[:count] = np.asarray(
                inverse_sigmoid(jnp.asarray(colors_np))
            )
        else:
            basis_count = (config.sh_degree + 1) ** 2
            sh0 = np.zeros((capacity, 1, 3), np.float32)
            sh0[:count, 0] = np.asarray(rgb_to_sh(jnp.asarray(colors_np)))
            sh_rest = np.zeros(
                (capacity, basis_count - 1, 3), np.float32
            )
            features = None
            color_logits = None

        def place(value: Any) -> jax.Array | None:
            if value is None:
                return None
            array = jnp.asarray(value)
            return jax.device_put(array, device) if device is not None else array

        return cls(
            place(means),
            place(log_scales),
            place(quats),
            place(opacity_logits),
            place(sh0),
            place(sh_rest),
            place(active_mask),
            features=place(features),
            colors=place(color_logits),
            max_capacity=config.capacity,
        )
