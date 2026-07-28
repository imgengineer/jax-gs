from __future__ import annotations

import math
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp

from .config import ModelConfig, OptimizerConfig
from .model import GaussianModel
from .optimizers import reorder_optimizer_slots
from .strategy import StrategyState


def _validate_scene_capacity_operation(
    scene: Any,
    capacity: int,
    operation: str,
) -> None:
    if scene is None:
        return
    validate_capacity = getattr(scene, "validate_slot_capacity", None)
    operation_fn = getattr(scene, operation, None)
    if not callable(validate_capacity) or not callable(operation_fn):
        raise TypeError(
            "scene must implement the fixed-slot capacity interface"
        )
    validate_capacity(capacity)


def _resize_state_tree(
    old_state: Any,
    *,
    old_capacity: int,
    new_capacity: int,
) -> Any:
    def resize(old_value: Any) -> Any:
        if not isinstance(old_value, jax.Array):
            return old_value
        if (
            old_value.ndim > 0
            and old_value.shape[0] == old_capacity
        ):
            padding = ((0, new_capacity - old_capacity),) + (
                (0, 0),
            ) * (old_value.ndim - 1)
            return jnp.pad(old_value, padding)
        return old_value

    return jax.tree.map(resize, old_state)


def resize_training_state(
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    strategy_state: StrategyState,
    new_capacity: int,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    scene: Any = None,
) -> tuple[GaussianModel, nnx.Optimizer, StrategyState]:
    """Grow all capacity-leading training arrays while preserving their state.

    When provided, ``scene`` is padded independently with the same physical
    capacity change. Its values are not copied from ``model``.
    """

    old_capacity = model.capacity
    new_capacity = int(new_capacity)
    if new_capacity <= old_capacity:
        raise ValueError("new_capacity must be larger than the current capacity")
    if new_capacity > model.max_capacity:
        raise ValueError("new_capacity exceeds the model's logical maximum")
    _validate_scene_capacity_operation(
        scene, old_capacity, "resize_slot_capacity"
    )
    del optimizer_config
    padding = new_capacity - old_capacity

    means = jnp.pad(model.means[...], ((0, padding), (0, 0)))
    log_scales = jnp.pad(
        model.log_scales[...],
        ((0, padding), (0, 0)),
        constant_values=math.log(model_config.initial_scale),
    )
    quats = jnp.pad(model.quats[...], ((0, padding), (0, 0)))
    quats = quats.at[old_capacity:, 0].set(1.0)
    initial_opacity_logit = math.log(model_config.initial_opacity) - math.log1p(
        -model_config.initial_opacity
    )
    opacity_logits = jnp.pad(
        model.opacity_logits[...],
        ((0, padding),),
        constant_values=initial_opacity_logit,
    )
    if model.has_appearance:
        sh0 = None
        sh_rest = None
        features = jnp.pad(
            model.features[...], ((0, padding), (0, 0))
        )
        colors = jnp.pad(model.colors[...], ((0, padding), (0, 0)))
    else:
        sh0 = jnp.pad(model.sh0[...], ((0, padding), (0, 0), (0, 0)))
        sh_rest = jnp.pad(
            model.sh_rest[...], ((0, padding), (0, 0), (0, 0))
        )
        features = None
        colors = None
    active_mask = jnp.pad(
        model.active_mask[...], ((0, padding),), constant_values=False
    )
    new_model = GaussianModel(
        means,
        log_scales,
        quats,
        opacity_logits,
        sh0,
        sh_rest,
        active_mask,
        features=features,
        colors=colors,
        max_capacity=model.max_capacity,
    )

    resized_optimizer_state = _resize_state_tree(
        nnx.as_pure(nnx.state(optimizer)),
        old_capacity=old_capacity,
        new_capacity=new_capacity,
    )
    nnx.update(optimizer, resized_optimizer_state)

    resized_strategy_state = _resize_state_tree(
        nnx.as_pure(nnx.state(strategy_state)),
        old_capacity=old_capacity,
        new_capacity=new_capacity,
    )
    nnx.update(strategy_state, resized_strategy_state)
    if scene is not None:
        scene.resize_slot_capacity(new_capacity)
    return new_model, optimizer, strategy_state


@nnx.jit(donate_argnames=("model", "optimizer", "strategy_state"))
def _compact_training_state(
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    strategy_state: StrategyState,
) -> tuple[jax.Array, jax.Array]:
    """Move every active slot and its optimizer/statistics state to a prefix."""

    active = model.active_mask[...]
    capacity = model.capacity
    active_count = jnp.count_nonzero(active)
    active_positions = jnp.cumsum(active.astype(jnp.int32)) - 1
    inactive_positions = (
        active_count + jnp.cumsum((~active).astype(jnp.int32)) - 1
    )
    destinations = jnp.where(active, active_positions, inactive_positions)
    order = jnp.zeros((capacity,), dtype=jnp.int32).at[destinations].set(
        jnp.arange(capacity, dtype=jnp.int32)
    )

    model.means[...] = model.means[...][order]
    model.log_scales[...] = model.log_scales[...][order]
    model.quats[...] = model.quats[...][order]
    model.opacity_logits[...] = model.opacity_logits[...][order]
    if model.has_appearance:
        model.features[...] = model.features[...][order]
        model.colors[...] = model.colors[...][order]
    else:
        model.sh0[...] = model.sh0[...][order]
        model.sh_rest[...] = model.sh_rest[...][order]
    model.active_mask[...] = jnp.arange(capacity) < active_count

    strategy_state.grad_accum[...] = strategy_state.grad_accum[...][order]
    strategy_state.visible_count[...] = strategy_state.visible_count[...][order]
    strategy_state.max_radii[...] = strategy_state.max_radii[...][order]
    if hasattr(strategy_state, "dynamic_mask"):
        strategy_state.dynamic_mask[...] = strategy_state.dynamic_mask[...][order]
    reorder_optimizer_slots(optimizer, order)
    return active_count, order


def compact_training_state(
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    strategy_state: StrategyState,
    scene: Any = None,
) -> jax.Array:
    """Move every active training and optional scene row to one prefix."""

    _validate_scene_capacity_operation(
        scene, model.capacity, "apply_slot_permutation"
    )
    active_count, order = _compact_training_state(
        model, optimizer, strategy_state
    )
    if scene is not None:
        scene.apply_slot_permutation(order)
    return active_count


def initial_physical_capacity(config: ModelConfig, point_count: int) -> int:
    return config.bucket_capacity(point_count)


__all__ = [
    "compact_training_state",
    "initial_physical_capacity",
    "resize_training_state",
]
