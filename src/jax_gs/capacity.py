from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp

from .config import ModelConfig, OptimizerConfig
from .model import GaussianModel
from .optimizers import (
    create_optimizer,
    create_row_selective_optimizer,
    create_visible_adam_optimizer,
    reorder_optimizer_slots,
)
from .strategy import StrategyState


def _distributed_world_size(*nodes: Any) -> int:
    """Return the shared leading world axis of a stacked shard set."""

    leading = set()
    for node in nodes:
        for leaf in jax.tree.leaves(nnx.as_pure(nnx.state(node))):
            if not isinstance(leaf, jax.Array) or leaf.ndim == 0:
                raise ValueError(
                    "distributed shards must be stacked over a leading world "
                    "axis; found an unsharded leaf"
                )
            leading.add(int(leaf.shape[0]))
    if len(leading) != 1:
        raise ValueError(
            "distributed shards must share one leading world axis, got "
            f"{sorted(leading)}"
        )
    return leading.pop()


def _distributed_local_capacity(model: GaussianModel, world_size: int) -> int:
    means = model.means[...]
    if means.ndim != 3 or means.shape[0] != world_size or means.shape[2] != 3:
        raise ValueError(
            "distributed model means must have shape [world, capacity, 3], "
            f"got {means.shape}"
        )
    return int(means.shape[1])


def _unstack_graph(graph: Any, index: int) -> Any:
    graphdef, state = nnx.split(graph)
    return nnx.merge(
        graphdef, jax.tree.map(lambda value: value[index], state)
    )


def _stack_graphs(graphs: Sequence[Any]) -> Any:
    graphdef, first_state = nnx.split(graphs[0])
    states = [first_state] + [nnx.split(graph)[1] for graph in graphs[1:]]
    return nnx.merge(
        graphdef, jax.tree.map(lambda *values: jnp.stack(values), *states)
    )


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


def resize_distributed_training_state(
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    strategy_state: StrategyState,
    new_capacity: int,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
) -> tuple[GaussianModel, nnx.Optimizer, StrategyState]:
    """Grow every shard of a stacked training world to one common capacity.

    The three nodes must be the stacked ``[world, ...]`` objects a bound
    ``nnx.pmap`` maps over. Every shard is resized with the ordinary
    single-process rules and the world is restacked, so all ranks keep one
    identical physical capacity and the mapped step stays compilable. A
    distributed capacity overflow is reported for the whole world, so hosts
    must grow all shards together and replay the skipped step rather than
    resizing one rank. Shards are materialized individually, so the transition
    transiently needs roughly one extra copy of the world.
    """

    world_size = _distributed_world_size(model, optimizer, strategy_state)
    # Reject a model that is not a [world, capacity, 3] shard set before any
    # shard is resized.
    _distributed_local_capacity(model, world_size)
    shards = [
        resize_training_state(
            _unstack_graph(model, rank),
            _unstack_graph(optimizer, rank),
            _unstack_graph(strategy_state, rank),
            new_capacity,
            model_config,
            optimizer_config,
        )
        for rank in range(world_size)
    ]
    return (
        _stack_graphs([shard[0] for shard in shards]),
        _stack_graphs([shard[1] for shard in shards]),
        _stack_graphs([shard[2] for shard in shards]),
    )


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


def _active_rows(state: Any, capacity: int, count: int) -> Any:
    """Take the active prefix of every capacity-leading array in a state tree."""

    def take(value: Any) -> Any:
        if isinstance(value, jax.Array) and value.ndim > 0 and value.shape[0] == capacity:
            return value[:count]
        return value

    return jax.tree.map(take, state)


def _pad_rows(state: Any, count: int, capacity: int) -> Any:
    """Pad every ``count``-leading array back out to the shard capacity."""

    def pad(value: Any) -> Any:
        if isinstance(value, jax.Array) and value.ndim > 0 and value.shape[0] == count:
            padding = ((0, capacity - count),) + ((0, 0),) * (value.ndim - 1)
            return jnp.pad(value, padding)
        return value

    return jax.tree.map(pad, state)


def reshard_distributed_training_state(
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    strategy_state: StrategyState,
    safety_state: Any,
    world_size: int,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    *,
    local_capacity: int | None = None,
) -> tuple[GaussianModel, nnx.Optimizer, StrategyState, Any]:
    """Redistribute a stacked training world across a different rank count.

    Every other distributed primitive keeps the world size fixed, because the
    mapped step is compiled for it; this is the one operation that changes it,
    so it is host work between steps and never inside the map.

    The active Gaussians are collected in rank order and dealt back out in
    contiguous blocks, so reading the new world in rank order returns the same
    global sequence that went in. That makes resharding composable: widening
    and narrowing again restores the original assignment exactly, which
    matters because this exists to move a checkpoint between world sizes and a
    world may be resharded more than once. A strided deal, which is how a
    fresh run splits its initial point cloud, would not compose that way,
    since the sequence it produces is not the one it consumed. Remainders go
    to the lowest ranks, so shard sizes differ by at most one.

    Each Gaussian's optimizer moments and densification statistics travel with
    its row, since they describe the Gaussian rather than the rank that
    happened to hold it. Shards may hold different active counts before and
    after; ``local_capacity`` defaults to the smallest configured bucket
    covering the busiest new shard. Source optimizer steps must agree. A fresh
    optimizer graph is built for the target world so current-main's
    world-scaled learning rates, betas, and epsilon change with it; the step
    and per-row moments are then loaded into that graph.

    Sticky overflow state is reduced rather than dropped: a world that has seen
    an intersection overflow still has, whichever rank saw it, so the flag is
    OR-reduced and the tile high-water maximized, then replicated.
    """

    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    old_world = _distributed_world_size(
        model, optimizer, strategy_state, safety_state
    )
    old_capacity = _distributed_local_capacity(model, old_world)
    optimizer_steps = jax.device_get(optimizer.step[...])
    if optimizer_steps.shape != (old_world,):
        raise ValueError(
            "distributed optimizer must hold one step counter per shard"
        )
    if not bool(jnp.all(optimizer_steps == optimizer_steps[0])):
        raise ValueError(
            "distributed shards disagree on the optimizer step: "
            f"{optimizer_steps.tolist()}"
        )

    source_optimizer_config = getattr(
        optimizer, "_jax_gs_optimizer_config", None
    )
    if source_optimizer_config != optimizer_config:
        raise ValueError(
            "optimizer_config must match the source optimizer contract"
        )
    optimizer_kind = getattr(optimizer, "_jax_gs_optimizer_kind", None)
    optimizer_factories = {
        "adam": create_optimizer,
        "row_selective_adam": create_row_selective_optimizer,
        "visible_adam": create_visible_adam_optimizer,
    }
    if optimizer_kind not in optimizer_factories:
        raise ValueError(
            "source optimizer does not record a supported optimizer kind"
        )
    optimizer_batch_size = getattr(
        optimizer, "_jax_gs_batch_size", None
    )
    optimizer_scene_scale = getattr(
        optimizer, "_jax_gs_scene_scale", None
    )

    # Compact first so each shard's active rows are a prefix we can slice.
    shards = []
    for rank in range(old_world):
        shard = (
            _unstack_graph(model, rank),
            _unstack_graph(optimizer, rank),
            _unstack_graph(strategy_state, rank),
        )
        count = int(compact_training_state(*shard))
        shards.append((shard, count))

    total = sum(count for _, count in shards)
    block, remainder = divmod(total, world_size)
    assignments = []
    start = 0
    for rank in range(world_size):
        size = block + (1 if rank < remainder else 0)
        assignments.append(list(range(start, start + size)))
        start += size
    new_counts = [len(rows) for rows in assignments]
    busiest = max(new_counts) if new_counts else 0
    if local_capacity is None:
        local_capacity = model_config.bucket_capacity(busiest)
    local_capacity = int(local_capacity)
    if local_capacity < busiest:
        raise ValueError(
            f"local_capacity {local_capacity} cannot hold the busiest new "
            f"shard's {busiest} Gaussians"
        )

    # Global row -> (old rank, row within that shard), in rank order.
    origins: list[tuple[int, int]] = []
    for rank, (_, count) in enumerate(shards):
        origins.extend((rank, row) for row in range(count))

    def gather(node_index: int, rows: list[tuple[int, int]]) -> Any:
        states = [
            _active_rows(
                nnx.as_pure(nnx.state(shard[node_index])), old_capacity, count
            )
            for shard, count in shards
        ]
        if not rows:
            # A rank with no Gaussians still needs the right tree shape.
            return _active_rows(
                nnx.as_pure(nnx.state(shards[0][0][node_index])),
                old_capacity,
                0,
            )
        selected = [states[rank] for rank, _ in rows]
        indices = [row for _, row in rows]

        def pick(*values: Any) -> Any:
            first = values[0]
            if not isinstance(first, jax.Array) or first.ndim == 0:
                return first
            return jnp.stack(
                [value[index] for value, index in zip(values, indices, strict=True)]
            )

        return jax.tree.map(pick, *selected)

    new_shards = []
    for rank in range(world_size):
        rows = [origins[index] for index in assignments[rank]]
        count = new_counts[rank]
        model_state = _pad_rows(gather(0, rows), count, local_capacity)
        optimizer_state = _pad_rows(gather(1, rows), count, local_capacity)
        strategy = _pad_rows(gather(2, rows), count, local_capacity)

        new_model = _rebuilt_shard_model(
            model_state,
            count,
            local_capacity,
            model.has_appearance,
            model.max_capacity,
            model_config,
        )
        # Fresh containers: the old shards are still the source rows for the
        # ranks built after this one.
        new_optimizer = optimizer_factories[optimizer_kind](
            new_model,
            optimizer_config,
            batch_size=optimizer_batch_size,
            world_size=world_size,
            scene_scale=optimizer_scene_scale,
        )
        nnx.update(new_optimizer, optimizer_state)
        new_strategy = _unstack_graph(strategy_state, 0)
        nnx.update(new_strategy, strategy)
        new_shards.append((new_model, new_optimizer, new_strategy))

    safety = _reduced_safety_state(safety_state, old_world, world_size)
    return (
        _stack_graphs([shard[0] for shard in new_shards]),
        _stack_graphs([shard[1] for shard in new_shards]),
        _stack_graphs([shard[2] for shard in new_shards]),
        safety,
    )


def _rebuilt_shard_model(
    state: Any,
    count: int,
    capacity: int,
    has_appearance: bool,
    max_capacity: int,
    model_config: ModelConfig,
) -> GaussianModel:
    """Materialize one resharded model shard at ``capacity`` from gathered rows.

    Built from the state rather than by updating a shard in place, because the
    shards are still the source rows for the ranks not yet built. Slots past
    the active prefix get the same fresh-slot values a capacity resize gives
    them, so an inactive slot means the same thing however it came to exist.
    """

    log_scales = state["log_scales"].at[count:].set(
        math.log(model_config.initial_scale)
    )
    quats = state["quats"].at[count:].set(0.0).at[count:, 0].set(1.0)
    initial_opacity_logit = math.log(model_config.initial_opacity) - math.log1p(
        -model_config.initial_opacity
    )
    opacity_logits = state["opacity_logits"].at[count:].set(
        initial_opacity_logit
    )
    return GaussianModel(
        state["means"],
        log_scales,
        quats,
        opacity_logits,
        None if has_appearance else state["sh0"],
        None if has_appearance else state["sh_rest"],
        jnp.arange(capacity) < count,
        features=state["features"] if has_appearance else None,
        colors=state["colors"] if has_appearance else None,
        max_capacity=max_capacity,
    )


def _reduced_safety_state(safety_state: Any, old_world: int, world_size: int) -> Any:
    """Carry sticky overflow across a reshard by reducing then replicating."""

    shards = [_unstack_graph(safety_state, rank) for rank in range(old_world)]
    tiles = jnp.max(
        jnp.stack([shard.max_overflow_tiles[...] for shard in shards])
    )
    seen = jnp.any(
        jnp.stack([shard.intersection_overflow_seen[...] for shard in shards])
    )
    replicas = []
    for _ in range(world_size):
        replica = _unstack_graph(safety_state, 0)
        replica.max_overflow_tiles[...] = tiles
        replica.intersection_overflow_seen[...] = seen
        replicas.append(replica)
    return _stack_graphs(replicas)
