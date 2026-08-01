"""Growing, compacting and checkpointing the model, optimizer and strategy together."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from flax import nnx
import jax

from ..config import TrainConfig
from ..model import GaussianModel
from ..strategy import (
    StrategyState,
)
from ._scene import SceneTransform
from .appearance import (
    AppearanceOptModule,
)
from .pose import CameraOptModule

# Tests drive the trainer by patching seams on the package, for example
# monkeypatch.setattr(jax_gs.training, "rasterization", fake). Calls resolve
# through the package namespace at run time so those seams keep working now
# that the implementation lives in submodules.
_training = sys.modules[__package__]


def _block_nnx_state(*nodes: Any) -> None:
    for node in nodes:
        state = nnx.as_pure(nnx.state(node))
        for leaf in jax.tree.leaves(state):
            if isinstance(leaf, jax.Array):
                leaf.block_until_ready()


def _initial_storage_capacity(config: TrainConfig, point_count: int) -> int:
    if point_count > config.model.capacity:
        raise ValueError(
            f"point cloud contains {point_count} points, but the logical "
            f"maximum is {config.model.capacity}"
        )
    return config.model.bucket_capacity(point_count)


def _grow_training_state(
    config: TrainConfig,
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    strategy_state: StrategyState,
    new_capacity: int,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
) -> tuple[GaussianModel, nnx.Optimizer, StrategyState]:
    _training._check_bucket_transition_memory_budget(
        config,
        model.capacity,
        new_capacity,
        image_height=image_height,
        image_width=image_width,
    )
    resized = _training.resize_training_state(
        model,
        optimizer,
        strategy_state,
        new_capacity,
        config.model,
        config.optimizer,
    )
    # Device allocation is asynchronous. Do not drop the valid old state until
    # every new buffer is known to have been allocated and copied successfully.
    _block_nnx_state(*resized)
    return resized


def _save_compacted_training_checkpoint(
    directory: Path,
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    strategy_state: StrategyState,
    *,
    step: int,
    config: TrainConfig,
    intersection_capacity: int,
    scene_transform: SceneTransform,
    scene_scale: float,
    pose_adjust: CameraOptModule | None = None,
    pose_optimizer: nnx.Optimizer | None = None,
    pose_image_names: tuple[str, ...] | None = None,
    appearance_module: AppearanceOptModule | None = None,
    appearance_optimizer: nnx.Optimizer | None = None,
    appearance_image_names: tuple[str, ...] | None = None,
) -> Path:
    compact_count = _training.compact_training_state(model, optimizer, strategy_state)
    compact_count.block_until_ready()
    return _training.save_checkpoint(
        directory,
        model,
        optimizer=optimizer,
        strategy_state=strategy_state,
        step=step,
        config=config,
        intersection_capacity=intersection_capacity,
        scene_transform=scene_transform.matrix,
        scene_scale=scene_scale,
        pose_module=pose_adjust,
        pose_optimizer=pose_optimizer,
        pose_image_names=pose_image_names,
        appearance_module=appearance_module,
        appearance_optimizer=appearance_optimizer,
        appearance_image_names=appearance_image_names,
    )
