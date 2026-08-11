"""Growing, compacting and checkpointing the model, optimizer and strategy together."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from pathlib import Path
import sys
from typing import Any, NamedTuple

from flax import nnx
import jax
import numpy as np

from ..capacity import (
    _distributed_local_capacity,
    _distributed_world_size,
    resize_distributed_training_state,
    resize_training_state,
)
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


class DistributedCapacityDecision(NamedTuple):
    """What the host decided after reading one distributed step's metrics.

    ``replay_required`` distinguishes the two overflow kinds. A refine that
    would have truncated a gradient freezes the whole step on device, so after
    growing, the host must run the same step again with the same inputs and
    keys. A commit overflow only dropped growth the plan could not fit; the
    step itself committed, so growing before the next refine is enough.
    """

    grew: bool
    replay_required: bool
    old_capacity: int
    new_capacity: int


def make_distributed_resize_step(
    config: TrainConfig,
    *,
    axis_name: Hashable = "rank",
    devices: Sequence[jax.Device] | None = None,
) -> Callable[
    [GaussianModel, nnx.Optimizer, StrategyState, int],
    tuple[GaussianModel, nnx.Optimizer, StrategyState],
]:
    """Map one shard resize over the devices that own the training world.

    A model returned by ``nnx.pmap`` carries a rank-partitioned leading axis.
    Unstacking, resizing, and restacking it on the host loses that placement,
    so the next mapped train step rejects the replicated arrays. Keeping the
    ordinary single-shard resize inside the same mapped axis preserves the
    placement while changing the static per-shard capacity.
    """

    @nnx.pmap(
        in_axes=(0, 0, 0, None),
        out_axes=(0, 0, 0),
        static_broadcasted_argnums=(3,),
        axis_name=axis_name,
        devices=devices,
    )
    def resize_step(
        model: GaussianModel,
        optimizer: nnx.Optimizer,
        strategy_state: StrategyState,
        new_capacity: int,
    ) -> tuple[GaussianModel, nnx.Optimizer, StrategyState]:
        return resize_training_state(
            model,
            optimizer,
            strategy_state,
            new_capacity,
            config.model,
            config.optimizer,
        )

    return resize_step


def synchronize_distributed_capacity(
    config: TrainConfig,
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    strategy_state: StrategyState,
    metrics: dict[str, jax.Array],
    *,
    image_height: int | None = None,
    image_width: int | None = None,
    devices: Sequence[jax.Device] | None = None,
    resize_step: Callable[
        [GaussianModel, nnx.Optimizer, StrategyState, int],
        tuple[GaussianModel, nnx.Optimizer, StrategyState],
    ]
    | None = None,
) -> tuple[
    GaussianModel, nnx.Optimizer, StrategyState, DistributedCapacityDecision
]:
    """Grow every shard when a distributed step reports a capacity overflow.

    This is the host half of distributed refinement: the step synchronizes
    preflight and post-update ``refine_*_required_capacity`` across ranks with
    ``pmax`` and reports the overflow kind, and this reads those metrics and
    applies the same bucket rule the single-process trainer uses, growing all
    shards together so the world keeps one static capacity. A live ``pmap``
    caller passes ``resize_step=make_distributed_resize_step(...)`` to retain
    rank placement and ``devices`` so every selected device is preflighted;
    the fallback is for unplaced host-stacked/vmap state.

    An exact restore rebuilds the checkpoint's recorded ``local_capacity``;
    restoring directly into a differently shaped target instead requires the
    explicit distributed checkpoint reshard path.

    Raises ``RuntimeError`` when the requirement cannot be met at
    ``max_capacity``, because replaying the frozen step at an unchanged
    capacity would fail the same way forever.
    """

    world_size = _distributed_world_size(model, optimizer, strategy_state)
    local_capacity = _distributed_local_capacity(model, world_size)
    skip_overflow = bool(
        np.any(np.asarray(metrics.get("refine_capacity_overflow", False)))
    )
    commit_overflow = bool(
        np.any(np.asarray(metrics.get("refine_commit_overflow", False)))
    )
    if not skip_overflow and not commit_overflow:
        return model, optimizer, strategy_state, DistributedCapacityDecision(
            False, False, local_capacity, local_capacity
        )

    required = max(
        int(np.max(np.asarray(metrics["refine_required_capacity"]))),
        int(
            np.max(
                np.asarray(
                    metrics.get("refine_commit_required_capacity", 0)
                )
            )
        ),
    )
    if skip_overflow and required > model.max_capacity:
        raise RuntimeError(
            "distributed refinement needs capacity "
            f"{required} but max_capacity is {model.max_capacity}; the frozen "
            "step would replay forever, so stop refinement or raise the limit"
        )
    bounded_required = min(required, model.max_capacity)
    target_capacity = max(
        local_capacity, config.model.bucket_capacity(bounded_required)
    )
    if target_capacity <= local_capacity:
        if skip_overflow:
            raise RuntimeError(
                "a distributed step reported a refine capacity overflow but "
                f"the bucket rule keeps capacity {local_capacity}; replaying "
                "would fail the same way forever"
            )
        return model, optimizer, strategy_state, DistributedCapacityDecision(
            False, False, local_capacity, local_capacity
        )

    _training._check_distributed_bucket_transition_memory_budget(
        config,
        world_size,
        local_capacity,
        target_capacity,
        devices=devices,
        image_height=image_height,
        image_width=image_width,
    )
    if resize_step is None:
        model, optimizer, strategy_state = resize_distributed_training_state(
            model,
            optimizer,
            strategy_state,
            target_capacity,
            config.model,
            config.optimizer,
        )
    else:
        model, optimizer, strategy_state = resize_step(
            model, optimizer, strategy_state, target_capacity
        )
    _training._block_nnx_state(model, optimizer, strategy_state)
    return model, optimizer, strategy_state, DistributedCapacityDecision(
        True, skip_overflow, local_capacity, target_capacity
    )
