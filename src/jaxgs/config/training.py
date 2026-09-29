"""Typed training settings; default.toml is the single source of defaults."""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .capacity import CapacityConfig


@dataclass(frozen=True)
class ModelConfig:
    sh_degree: int
    images: str
    resolution: int
    white_background: bool
    data_device: str
    eval: bool


@dataclass(frozen=True)
class PipelineConfig:
    cluster_size: int
    tile_size: tuple[int, int]
    sparse_grad: bool
    device_preload: bool
    enable_transmitance: bool
    enable_depth: bool
    input_color_type: str


@dataclass(frozen=True)
class OptimizationConfig:
    iterations: int
    position_lr_init: float
    position_lr_final: float
    position_lr_max_steps: int
    feature_lr: float
    opacity_lr: float
    scaling_lr: float
    rotation_lr: float
    lambda_dssim: float
    reg_weight: float
    learnable_viewproj: bool


@dataclass(frozen=True)
class DensifyConfig:
    densification_interval: int
    densify_from: int
    densify_until: int
    opacity_reset_interval: int
    opacity_reset_mode: str
    prune_mode: str
    target_primitives: int
    densify_grad_threshold: float
    opacity_threshold: float
    screen_size_threshold: int
    percent_dense: float

    def end_epoch(self, epochs: int) -> int:
        if self.densify_until < 0:
            return int(epochs * 0.8 / self.opacity_reset_interval) * self.opacity_reset_interval + 1
        return self.densify_until


@dataclass(frozen=True)
class RuntimeConfig:
    max_gaussians: int
    max_visibility_pairs: int
    optimizer: str
    seed: int


@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig
    pipeline: PipelineConfig
    optimization: OptimizationConfig
    densify: DensifyConfig
    runtime: RuntimeConfig

    @property
    def capacity(self) -> CapacityConfig:
        height, width = self.pipeline.tile_size
        return CapacityConfig(
            max_gaussians=self.runtime.max_gaussians,
            cluster_size=self.pipeline.cluster_size,
            tile_size=width,
            tile_height=height,
            sh_degree=self.model.sh_degree,
            max_visibility_pairs=self.runtime.max_visibility_pairs,
        )

    def validate(self) -> None:
        op, dp = self.optimization, self.densify
        if (
            min(
                op.iterations,
                op.position_lr_max_steps,
                dp.densification_interval,
                dp.opacity_reset_interval,
                dp.target_primitives,
            )
            <= 0
        ):
            raise ValueError("iterations, schedule lengths and target_primitives must be positive")
        if dp.densify_from < 0 or dp.densify_until < -1 or dp.percent_dense <= 0:
            raise ValueError("invalid densification window or percent_dense")
        if self.model.resolution != -1 and self.model.resolution <= 0:
            raise ValueError("resolution must be -1 or positive")
        rates = (
            op.position_lr_init,
            op.position_lr_final,
            op.feature_lr,
            op.opacity_lr,
            op.scaling_lr,
            op.rotation_lr,
        )
        if any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError("learning rates must be finite and nonnegative")
        if (op.position_lr_init == 0) != (op.position_lr_final == 0):
            raise ValueError("position learning rates must both be positive or both zero")
        if dp.target_primitives > self.capacity.max_gaussians:
            raise ValueError("target_primitives exceeds runtime.max_gaussians")
        if self.pipeline.tile_size not in ((8, 8), (8, 16), (12, 16), (16, 16)):
            raise ValueError("production tile_size must be 8x8, 8x16, 12x16 or 16x16")
        if self.runtime.optimizer not in ("optax", "cute"):
            raise ValueError("optimizer must be optax or cute")
        # These source switches are recorded for parity, but their nondefault
        # paths are not implemented by the production RGB training backend.
        supported = {
            "white_background": (self.model.white_background, False),
            "data_device": (self.model.data_device, "cuda"),
            "sparse_grad": (self.pipeline.sparse_grad, True),
            "device_preload": (self.pipeline.device_preload, True),
            "enable_transmitance": (self.pipeline.enable_transmitance, False),
            "enable_depth": (self.pipeline.enable_depth, False),
            "input_color_type": (self.pipeline.input_color_type, "sh"),
            "lambda_dssim": (op.lambda_dssim, 0.2),
            "reg_weight": (op.reg_weight, 0.0),
            "learnable_viewproj": (op.learnable_viewproj, False),
            "opacity_reset_mode": (dp.opacity_reset_mode, "decay"),
            "prune_mode": (dp.prune_mode, "weight"),
        }
        for name, (actual, expected) in supported.items():
            if actual != expected:
                raise ValueError(f"production training currently requires {name}={expected!r}")


def load_config(path: str | Path | None = None) -> TrainingConfig:
    """Overlay a TOML file on the packaged defaults; reject unknown settings."""
    values = tomllib.loads(Path(__file__).with_name("default.toml").read_text())
    if path is not None:
        for section, overrides in tomllib.loads(Path(path).read_text()).items():
            if section not in values:
                raise ValueError(f"unknown configuration section: {section}")
            unknown = overrides.keys() - values[section].keys()
            if unknown:
                raise ValueError(f"unknown {section} settings: {sorted(unknown)}")
            values[section].update(overrides)
    values["pipeline"]["tile_size"] = tuple(values["pipeline"]["tile_size"])
    config = TrainingConfig(
        ModelConfig(**values["model"]),
        PipelineConfig(**values["pipeline"]),
        OptimizationConfig(**values["optimization"]),
        DensifyConfig(**values["densify"]),
        RuntimeConfig(**values["runtime"]),
    )
    config.validate()
    return config
