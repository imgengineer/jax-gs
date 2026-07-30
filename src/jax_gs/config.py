from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Literal, Mapping
import warnings


MAX_MODEL_CAPACITY = 10_000_000


def _available_system_workers() -> int:
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except (AttributeError, OSError):
        return max(int(os.cpu_count() or 1), 1)


@dataclass(frozen=True)
class ModelConfig:
    """Logical Gaussian limit and physical storage-bucket configuration.

    ``capacity`` is the logical maximum. Parameter, Adam, and strategy arrays
    start at ``bucket_min_capacity`` (or the first bucket covering the input
    point cloud) and double only when refinement needs more physical slots.
    Within each bucket, ``active_mask`` changes without retriggering JIT.
    """

    capacity: int = 1_000_000
    sh_degree: int = 3
    initial_opacity: float = 0.1
    initial_scale: float = 1.0
    bucket_min_capacity: int = 65_536

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.capacity > MAX_MODEL_CAPACITY:
            raise ValueError(
                f"capacity cannot exceed {MAX_MODEL_CAPACITY:,}"
            )
        if not 0 <= self.sh_degree <= 4:
            raise ValueError("sh_degree must be between 0 and 4")
        if not 0.0 < self.initial_opacity < 1.0:
            raise ValueError("initial_opacity must be in (0, 1)")
        if self.initial_scale <= 0.0:
            raise ValueError("initial_scale must be positive")
        if self.bucket_min_capacity <= 0:
            raise ValueError("bucket_min_capacity must be positive")

    def bucket_capacity(self, required: int = 0) -> int:
        """Return the smallest configured physical bucket covering ``required``."""

        required = int(required)
        if required < 0:
            raise ValueError("required capacity cannot be negative")
        if required > self.capacity:
            raise ValueError(
                f"required capacity {required} exceeds logical maximum {self.capacity}"
            )
        bucket = min(self.bucket_min_capacity, self.capacity)
        while bucket < required:
            bucket = min(bucket * 2, self.capacity)
        return bucket


@dataclass(frozen=True)
class RasterizationConfig:
    """Compile-time renderer limits and numerical settings."""

    tile_size: int = 16
    max_gaussians_per_tile: int = 512
    tile_batch_size: int = 64
    near_plane: float = 0.01
    far_plane: float = 1.0e10
    eps2d: float = 0.3
    radius_clip: float = 0.0
    alpha_clip: float = 1.0 / 255.0
    transmittance_eps: float = 1.0e-4
    rasterize_mode: str = "classic"
    ut_chunk_size: int = 16_384
    backend: str = "auto"
    intersection_backend: str = "auto"
    intersection_mode: str = "auto"
    sort_backend: str = "auto"
    max_intersections: int | None = None

    def __post_init__(self) -> None:
        removed_backends = []
        for field_name in ("backend", "intersection_backend", "sort_backend"):
            if getattr(self, field_name) == "cutile":
                object.__setattr__(self, field_name, "jax")
                removed_backends.append(field_name)
        if removed_backends:
            warnings.warn(
                "cuTile backends were removed; using pure JAX for "
                + ", ".join(removed_backends),
                DeprecationWarning,
                stacklevel=2,
            )
        if self.backend not in {
            "auto",
            "jax",
            "intersections",
            "reference",
        }:
            raise ValueError(
                "backend must be 'auto', 'jax', 'intersections', or 'reference'"
            )
        if self.intersection_backend not in {"auto", "jax"}:
            raise ValueError(
                "intersection_backend must be 'auto' or 'jax'"
            )
        if self.intersection_mode not in {"auto", "aabb", "accutile"}:
            raise ValueError(
                "intersection_mode must be 'auto', 'aabb', or 'accutile'"
            )
        if self.sort_backend not in {"auto", "jax"}:
            raise ValueError("sort_backend must be 'auto' or 'jax'")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.max_gaussians_per_tile <= 0:
            raise ValueError("max_gaussians_per_tile must be positive")
        if self.max_intersections is not None and self.max_intersections <= 0:
            raise ValueError("max_intersections must be positive when provided")
        if self.tile_batch_size <= 0:
            raise ValueError("tile_batch_size must be positive")
        if self.ut_chunk_size <= 0:
            raise ValueError("ut_chunk_size must be positive")
        if self.near_plane <= 0.0 or self.far_plane <= self.near_plane:
            raise ValueError("invalid near/far plane")
        if self.rasterize_mode not in {"classic", "antialiased"}:
            raise ValueError("rasterize_mode must be 'classic' or 'antialiased'")


@dataclass(frozen=True)
class OptimizerConfig:
    means_lr: float = 1.6e-4
    scales_lr: float = 5.0e-3
    quats_lr: float = 1.0e-3
    opacities_lr: float = 5.0e-2
    sh0_lr: float = 2.5e-3
    sh_rest_lr: float = 2.5e-3 / 20.0
    means_lr_final_scale: float = 0.01
    max_steps: int = 30_000
    eps: float = 1.0e-15


@dataclass(frozen=True)
class StrategyConfig:
    kind: str = "default"
    refine_start: int = 500
    refine_stop: int = 15_000
    refine_every: int = 100
    reset_every: int = 3_000
    max_new_per_refine: int = 8_192
    grow_grad2d: float = 2.0e-4
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_opacity: float = 0.005
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: str = "means2d"
    reset_opacity: float = 0.01
    cap_max: int = 1_000_000
    noise_lr: float = 5.0e5
    noise_injection_stop_iter: int = -1
    noise_opacity_t: float = 0.005
    noise_opacity_k: float = 100.0

    def __post_init__(self) -> None:
        if self.kind not in {"default", "mcmc"}:
            raise ValueError("strategy kind must be 'default' or 'mcmc'")
        if self.refine_every <= 0 or self.reset_every <= 0:
            raise ValueError("refinement intervals must be positive")
        if self.max_new_per_refine <= 0:
            raise ValueError("max_new_per_refine must be positive")
        if self.key_for_gradient not in {"means2d", "gradient_2dgs"}:
            raise ValueError(
                "key_for_gradient must be 'means2d' or 'gradient_2dgs'"
            )
        if self.cap_max <= 0:
            raise ValueError("cap_max must be positive")
        if self.noise_lr < 0.0:
            raise ValueError("noise_lr must be non-negative")
        if self.noise_opacity_k < 0.0:
            raise ValueError("noise_opacity_k must be non-negative")


@dataclass(frozen=True)
class DataConfig:
    root: str = "/home/lzc/datasets/stump"
    image_dir: str = "images_8"
    test_every: int = 8
    patch_size: int | None = None
    batch_size: int = 1
    shuffle_seed: int = 42
    num_workers: int = 4

    def __post_init__(self) -> None:
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        available_workers = _available_system_workers()
        if self.num_workers > available_workers:
            warnings.warn(
                f"num_workers={self.num_workers} exceeds the "
                f"{available_workers} workers available to this process; "
                "oversubscription may reduce performance and increase memory use",
                RuntimeWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rasterizer: RasterizationConfig = field(default_factory=RasterizationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    global_scale: float = 1.0
    normalize_world_space: bool = True
    model_type: Literal["3dgs", "2dgs"] = "3dgs"
    packed: bool = False
    sparse_grad: bool = False
    visible_adam: bool = False
    steps: int = 30_000
    sh_degree_interval: int = 1_000
    ssim_lambda: float = 0.2
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 1.0e-3
    app_opt_reg: float = 1.0e-6
    pose_opt: bool = False
    pose_opt_lr: float = 1.0e-5
    pose_opt_reg: float = 1.0e-6
    pose_noise: float = 0.0
    normal_loss: bool = False
    normal_lambda: float = 5.0e-2
    normal_start_iter: int = 7_000
    dist_loss: bool = False
    dist_lambda: float = 1.0e-2
    dist_start_iter: int = 3_000
    random_background: bool = False
    camera_model: str = "pinhole"
    with_ut: bool = False
    with_eval3d: bool = False
    seed: int = 42
    checkpoint_every: int = 5_000
    eval_every: int = 1_000
    intersection_bucket_min_capacity: int = 65_536
    output_dir: str = "outputs/default"

    def __post_init__(self) -> None:
        if not math.isfinite(self.global_scale) or self.global_scale <= 0.0:
            raise ValueError("global_scale must be finite and positive")
        if self.model_type not in {"3dgs", "2dgs"}:
            raise ValueError("model_type must be '3dgs' or '2dgs'")
        if self.sparse_grad and not self.packed:
            raise ValueError("sparse_grad=True requires packed=True")
        if self.sparse_grad and self.visible_adam:
            raise ValueError("sparse_grad and visible_adam are mutually exclusive")
        if self.model_type == "2dgs" and self.visible_adam:
            raise ValueError("2DGS training does not support visible_adam")
        if self.sparse_grad and self.with_ut:
            raise ValueError("sparse_grad does not support with_ut=True")
        if self.sparse_grad and self.with_eval3d:
            raise ValueError("sparse_grad does not support with_eval3d=True")
        if self.sparse_grad and self.camera_model == "ftheta":
            raise ValueError(
                "sparse_grad does not support camera_model='ftheta' because "
                "f-theta projection uses the UT path"
            )
        if self.camera_model not in {"pinhole", "ortho", "fisheye", "ftheta"}:
            raise ValueError("unsupported camera_model")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if not 0.0 <= self.ssim_lambda <= 1.0:
            raise ValueError("ssim_lambda must be in [0, 1]")
        if self.opacity_reg < 0.0 or self.scale_reg < 0.0:
            raise ValueError("3DGS regularization weights must be non-negative")
        if self.app_embed_dim < 0:
            raise ValueError("app_embed_dim must be non-negative")
        if self.app_opt_lr < 0.0:
            raise ValueError("app_opt_lr must be non-negative")
        if self.app_opt_reg < 0.0:
            raise ValueError("app_opt_reg must be non-negative")
        if self.pose_opt_lr < 0.0:
            raise ValueError("pose_opt_lr must be non-negative")
        if self.pose_opt_reg < 0.0:
            raise ValueError("pose_opt_reg must be non-negative")
        if self.pose_noise < 0.0:
            raise ValueError("pose_noise must be non-negative")
        if self.model_type == "2dgs" and (
            self.opacity_reg != 0.0 or self.scale_reg != 0.0
        ):
            raise ValueError(
                "opacity_reg and scale_reg are available only for 3DGS"
            )
        if self.normal_lambda < 0.0 or self.dist_lambda < 0.0:
            raise ValueError("2DGS regularization weights must be non-negative")
        if self.normal_start_iter < 0 or self.dist_start_iter < 0:
            raise ValueError(
                "2DGS regularization start iterations must be non-negative"
            )
        if self.intersection_bucket_min_capacity <= 0 or (
            self.intersection_bucket_min_capacity
            & (self.intersection_bucket_min_capacity - 1)
        ):
            raise ValueError(
                "intersection_bucket_min_capacity must be a positive power of two"
            )

    @property
    def densification_gradient_key(self) -> Literal["means2d", "gradient_2dgs"]:
        """Screen-space gradient consumed by the unified trainer."""

        return "gradient_2dgs" if self.model_type == "2dgs" else "means2d"

    @classmethod
    def for_model_type(
        cls,
        model_type: Literal["3dgs", "2dgs"],
        *,
        strategy_kind: Literal["default", "mcmc"] = "default",
    ) -> "TrainConfig":
        """Create the current-main example profile for one Gaussian model."""

        if strategy_kind not in {"default", "mcmc"}:
            raise ValueError("strategy_kind must be 'default' or 'mcmc'")
        if model_type == "3dgs" and strategy_kind == "mcmc":
            return cls(
                model=ModelConfig(initial_opacity=0.5, initial_scale=0.1),
                model_type="3dgs",
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=StrategyConfig(kind="mcmc", verbose=True),
            )
        if model_type == "3dgs":
            return cls(model_type="3dgs")
        if model_type == "2dgs":
            if strategy_kind != "default":
                raise ValueError(
                    "2DGS training supports only the default strategy"
                )
            return cls(
                model_type="2dgs",
                rasterizer=RasterizationConfig(
                    near_plane=0.2,
                    far_plane=200.0,
                ),
                strategy=StrategyConfig(
                    prune_opacity=0.05,
                    key_for_gradient="gradient_2dgs",
                ),
            )
        raise ValueError("model_type must be '3dgs' or '2dgs'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainConfig":
        rasterizer_values = dict(values.get("rasterizer", {}))
        legacy_backends = {
            "backend": {"cuda_ffi": "jax"},
            "intersection_backend": {"pallas": "jax"},
            "sort_backend": {"cuda_ffi": "jax"},
        }
        migrated = []
        for key, replacements in legacy_backends.items():
            old_value = rasterizer_values.get(key)
            if isinstance(old_value, str) and old_value in replacements:
                rasterizer_values[key] = replacements[old_value]
                migrated.append(f"{key}={old_value!r}")
        if migrated:
            warnings.warn(
                "migrated legacy rasterizer settings to pure JAX: "
                + ", ".join(migrated),
                UserWarning,
                stacklevel=2,
            )
        return cls(
            model=ModelConfig(**values.get("model", {})),
            rasterizer=RasterizationConfig(**rasterizer_values),
            optimizer=OptimizerConfig(**values.get("optimizer", {})),
            strategy=StrategyConfig(**values.get("strategy", {})),
            data=DataConfig(**values.get("data", {})),
            **{
                key: value
                for key, value in values.items()
                if key not in {"model", "rasterizer", "optimizer", "strategy", "data"}
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
