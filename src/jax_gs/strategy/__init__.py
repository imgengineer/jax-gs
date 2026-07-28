from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from collections.abc import Mapping
from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp

from ..config import StrategyConfig
from ..math import quat_to_rotmat
from ..model import GaussianModel, inverse_sigmoid
from ..optimizers import reset_optimizer_slots
from ..relocation import compute_relocation


_MCMC_BINOMIALS = jnp.asarray(
    [
        [
            math.comb(row, column) if column <= row else 0
            for column in range(51)
        ]
        for row in range(51)
    ],
    dtype=jnp.float32,
)


class Strategy:
    """Base interface for fixed-capacity JAX densification strategies."""

    def check_sanity(
        self,
        params: GaussianModel,
        optimizers: nnx.Optimizer,
    ) -> None:
        if not isinstance(params, GaussianModel):
            raise TypeError("params must be a GaussianModel in the pure-JAX port")
        if not isinstance(optimizers, nnx.Optimizer):
            raise TypeError("optimizers must be one Flax NNX Optimizer")

    def step_pre_backward(
        self,
        params: GaussianModel,
        optimizers: nnx.Optimizer,
        state: "StrategyState",
        step: int,
        info: Mapping[str, Any],
    ) -> None:
        del state, step, info
        self.check_sanity(params, optimizers)

    def step_post_backward(self, *args, **kwargs):
        """Strategy-specific callback implemented by concrete strategies."""

        del args, kwargs
        return None


class StrategyState(nnx.Module):
    def __init__(self, capacity: int = 0, *, scene_scale: float = 1.0) -> None:
        capacity = self._validate_capacity(capacity)
        self.grad_accum = nnx.Variable(jnp.zeros((capacity,), jnp.float32))
        self.visible_count = nnx.Variable(jnp.zeros((capacity,), jnp.float32))
        self.max_radii = nnx.Variable(jnp.zeros((capacity,), jnp.float32))
        self.scene_scale = nnx.Variable(
            jnp.asarray(scene_scale, dtype=jnp.float32)
        )
        self.last_new_count = nnx.Variable(jnp.array(0, jnp.int32))
        self.last_pruned_count = nnx.Variable(jnp.array(0, jnp.int32))
        self.capacity_overflow = nnx.Variable(jnp.array(False))

    @staticmethod
    def _validate_capacity(capacity: int) -> int:
        try:
            capacity = operator.index(capacity)
        except TypeError as exc:
            raise TypeError("capacity must be an integer") from exc
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        return capacity

    def ensure_capacity(self, capacity: int) -> None:
        """Materialize a current-main lazy state for one physical bucket."""

        capacity = self._validate_capacity(capacity)
        current_capacity = self.grad_accum.shape[0]
        if current_capacity == capacity:
            return
        if current_capacity != 0:
            raise ValueError(
                "strategy state capacity does not match the model; use the "
                "training-state resize helper for an initialized state"
            )
        self.grad_accum = nnx.Variable(jnp.zeros((capacity,), jnp.float32))
        self.visible_count = nnx.Variable(jnp.zeros((capacity,), jnp.float32))
        self.max_radii = nnx.Variable(jnp.zeros((capacity,), jnp.float32))

    def reset_statistics(self) -> None:
        self.grad_accum[...] = 0.0
        self.visible_count[...] = 0.0
        self.max_radii[...] = 0.0


class DensificationStats(NamedTuple):
    """Per-Gaussian statistics accumulated by the default strategy."""

    grad_sum: jax.Array
    count: jax.Array
    max_radii: jax.Array


def build_densification_stats(
    screen_grad: jax.Array,
    radii: jax.Array,
    valid: jax.Array,
    active: jax.Array,
    width: int | jax.Array,
    height: int | jax.Array,
) -> DensificationStats:
    """Build dense multi-camera statistics using gsplat's normalization.

    ``screen_grad`` is the signed reverse-mode gradient for projected means.
    This helper deliberately does not approximate AbsGrad with
    ``abs(screen_grad)``: gsplat accumulates absolute per-pixel contributions
    before they can cancel, so true AbsGrad must come from the compositor.
    """

    screen_grad = jnp.asarray(screen_grad)
    radii = jnp.asarray(radii)
    valid = jnp.asarray(valid, dtype=jnp.bool_)
    active = jnp.asarray(active, dtype=jnp.bool_)
    if screen_grad.ndim != 3 or screen_grad.shape[-1] != 2:
        raise ValueError("screen_grad must have shape [C, N, 2]")
    camera_count, gaussian_count, _ = screen_grad.shape
    if camera_count == 0:
        raise ValueError("screen_grad must contain at least one camera")
    if radii.shape != screen_grad.shape:
        raise ValueError("radii must have shape [C, N, 2]")
    if valid.shape != (camera_count, gaussian_count):
        raise ValueError("valid must have shape [C, N]")
    if active.shape != (gaussian_count,):
        raise ValueError("active must have shape [N]")

    width = jnp.asarray(width, dtype=screen_grad.dtype)
    height = jnp.asarray(height, dtype=screen_grad.dtype)
    camera_scale = jnp.asarray(camera_count, dtype=screen_grad.dtype)
    gradient_scale = jnp.stack((width, height)) * (0.5 * camera_scale)
    visible = valid & active[None, :] & jnp.all(radii > 0.0, axis=-1)
    normalized_grad = jnp.where(
        visible[..., None], screen_grad * gradient_scale, 0.0
    )
    grad_sum = jnp.sum(jnp.linalg.norm(normalized_grad, axis=-1), axis=0)
    count = jnp.sum(visible.astype(jnp.float32), axis=0)
    normalized_radii = jnp.max(radii, axis=-1) / jnp.maximum(width, height)
    max_radii = jnp.max(jnp.where(visible, normalized_radii, 0.0), axis=0)
    return DensificationStats(grad_sum, count, max_radii)


def _build_packed_densification_stats(
    screen_grad: jax.Array,
    radii: jax.Array,
    valid: jax.Array,
    gaussian_ids: jax.Array,
    active: jax.Array,
    width: int | jax.Array,
    height: int | jax.Array,
    n_cameras: int | jax.Array,
    valid_count: int | jax.Array | None,
) -> DensificationStats:
    """Scatter fixed-capacity packed projection statistics to model rows."""

    screen_grad = jnp.asarray(screen_grad)
    radii = jnp.asarray(radii)
    valid = jnp.asarray(valid, dtype=jnp.bool_)
    gaussian_ids = jnp.asarray(gaussian_ids, dtype=jnp.int32)
    active = jnp.asarray(active, dtype=jnp.bool_)
    if screen_grad.ndim != 2 or screen_grad.shape[-1] != 2:
        raise ValueError("packed screen gradient must have shape [P, 2]")
    packed_capacity = screen_grad.shape[0]
    if radii.shape != (packed_capacity, 2):
        raise ValueError("packed radii must have shape [P, 2]")
    if valid.shape != (packed_capacity,):
        raise ValueError("packed valid must have shape [P]")
    if gaussian_ids.shape != (packed_capacity,):
        raise ValueError("packed gaussian_ids must have shape [P]")
    if active.ndim != 1:
        raise ValueError("active must have shape [N]")
    gaussian_count = active.shape[0]
    if gaussian_count == 0:
        raise ValueError("active must contain at least one Gaussian slot")
    if valid_count is None:
        valid_count = packed_capacity

    positions = jnp.arange(packed_capacity, dtype=jnp.int32)
    safe_ids = jnp.clip(gaussian_ids, 0, gaussian_count - 1)
    visible = (
        (positions < jnp.asarray(valid_count, dtype=jnp.int32))
        & valid
        & (gaussian_ids >= 0)
        & (gaussian_ids < gaussian_count)
        & active[safe_ids]
        & jnp.all(radii > 0.0, axis=-1)
    )
    width_value = jnp.asarray(width, dtype=screen_grad.dtype)
    height_value = jnp.asarray(height, dtype=screen_grad.dtype)
    camera_scale = jnp.asarray(n_cameras, dtype=screen_grad.dtype)
    gradient_scale = jnp.stack((width_value, height_value)) * (
        0.5 * camera_scale
    )
    gradient_norm = jnp.linalg.norm(screen_grad * gradient_scale, axis=-1)
    grad_sum = jnp.zeros((gaussian_count,), jnp.float32).at[safe_ids].add(
        jnp.where(visible, gradient_norm, 0.0).astype(jnp.float32)
    )
    count = jnp.zeros((gaussian_count,), jnp.float32).at[safe_ids].add(
        visible.astype(jnp.float32)
    )
    normalized_radii = jnp.max(radii, axis=-1) / jnp.maximum(
        width_value, height_value
    )
    max_radii = jnp.zeros((gaussian_count,), jnp.float32).at[safe_ids].max(
        jnp.where(visible, normalized_radii, 0.0).astype(jnp.float32)
    )
    return DensificationStats(grad_sum, count, max_radii)


@nnx.jit(donate_argnames=("state",))
def accumulate_densification_stats(
    state: StrategyState, stats: DensificationStats
) -> None:
    """Merge one dense rendering batch into a strategy state."""

    expected_shape = state.grad_accum.shape
    if stats.grad_sum.shape != expected_shape:
        raise ValueError("stats.grad_sum must match the strategy capacity")
    if stats.count.shape != expected_shape:
        raise ValueError("stats.count must match the strategy capacity")
    if stats.max_radii.shape != expected_shape:
        raise ValueError("stats.max_radii must match the strategy capacity")
    state.grad_accum[...] += stats.grad_sum
    state.visible_count[...] += stats.count
    state.max_radii[...] = jnp.maximum(
        state.max_radii[...], stats.max_radii
    )


@nnx.jit(donate_argnames=("state",))
def update_strategy_state(
    state: StrategyState,
    model: GaussianModel,
    means_gradient: jax.Array,
    visible: jax.Array,
    radii: jax.Array | None = None,
) -> None:
    """Accumulate fixed-shape densification statistics after a train step."""

    gradient_norm = jnp.linalg.norm(means_gradient, axis=-1)
    active_visible = visible.astype(jnp.bool_) & model.active_mask[...]
    state.grad_accum[...] += jnp.where(active_visible, gradient_norm, 0.0)
    state.visible_count[...] += active_visible.astype(jnp.float32)
    if radii is not None:
        state.max_radii[...] = jnp.maximum(
            state.max_radii[...], jnp.where(active_visible, radii, 0.0)
        )


def _scatter_mask(indices: jax.Array, valid: jax.Array, capacity: int) -> jax.Array:
    return jnp.zeros((capacity,), dtype=jnp.bool_).at[indices].set(valid)


def _validate_fixed_slot_scene(scene: Any, capacity: int) -> None:
    if scene is None:
        return
    validate_capacity = getattr(scene, "validate_slot_capacity", None)
    apply_transaction = getattr(scene, "apply_slot_transaction", None)
    if not callable(validate_capacity) or not callable(apply_transaction):
        raise TypeError("scene must implement the fixed-slot transaction interface")
    validate_capacity(capacity)


def _commit_and_strip_slot_transactions(
    scene: Any,
    result: Mapping[str, jax.Array],
) -> dict[str, jax.Array]:
    """Commit private fixed-shape lineage copies outside compiled refinement."""

    public_result = dict(result)
    for transaction_name in ("copy", "relocate", "birth"):
        source_key = f"_slot_{transaction_name}_sources"
        target_key = f"_slot_{transaction_name}_targets"
        valid_key = f"_slot_{transaction_name}_valid"
        sources = public_result.pop(source_key, None)
        targets = public_result.pop(target_key, None)
        valid = public_result.pop(valid_key, None)
        present = (sources is not None, targets is not None, valid is not None)
        if any(present) and not all(present):
            raise RuntimeError(
                f"incomplete private {transaction_name!r} slot transaction"
            )
        if sources is not None and scene is not None:
            scene.apply_slot_transaction(
                source_slots=sources,
                target_slots=targets,
                valid=valid,
            )
    return public_result


def _sample_weighted_ids(
    key: jax.Array,
    weights: jax.Array,
    sample_count: int,
) -> tuple[jax.Array, jax.Array]:
    """Sample with replacement without materializing ``[sample_count, N]``."""

    weights = jnp.asarray(weights)
    weights = jnp.where(jnp.isfinite(weights) & (weights > 0.0), weights, 0.0)
    cumulative = jnp.cumsum(weights)
    total = cumulative[-1]
    has_donor = jnp.isfinite(total) & (total > 0.0)
    samples = jax.random.uniform(
        key,
        (sample_count,),
        dtype=weights.dtype,
        maxval=jnp.where(has_donor, total, jnp.asarray(1.0, weights.dtype)),
    )
    sampled_ids = jnp.searchsorted(cumulative, samples, side="right")
    sampled_ids = jnp.clip(sampled_ids, 0, weights.shape[0] - 1)
    return jax.lax.stop_gradient(sampled_ids.astype(jnp.int32)), has_donor


class _DefaultGrowthEvents(NamedTuple):
    """Fixed-shape duplicate/split events selected for one refinement."""

    active: jax.Array
    parent_ids: jax.Array
    parent_ranks: jax.Array
    selected: jax.Array
    is_split: jax.Array
    planned_new_count: jax.Array
    free_count: jax.Array
    free_ids: jax.Array
    valid_new: jax.Array
    capacity_overflow: jax.Array


def _default_growth_events(
    model: GaussianModel,
    state: StrategyState,
    scene_scale: jax.Array,
    config: StrategyConfig,
    step: int | jax.Array | None,
) -> _DefaultGrowthEvents:
    """Select a stable, bounded sequence of current-main growth events.

    Parents retain the existing score order. Each ranked parent expands to a
    duplicate event followed by a split event, matching upstream's operation
    order when both predicates hold. ``max_new_per_refine`` bounds the total
    number of events rather than the number of distinct parents.
    """

    capacity = model.capacity
    allocation_count = min(config.max_new_per_refine, capacity)
    active = model.active_mask[...]
    average_gradient = state.grad_accum[...] / jnp.maximum(
        state.visible_count[...], 1.0
    )
    high_gradient = average_gradient > config.grow_grad2d
    max_scale = jnp.max(jnp.exp(model.log_scales[...]), axis=-1)
    small = max_scale <= config.grow_scale3d * scene_scale
    duplicate = active & high_gradient & small
    split = active & high_gradient & ~small
    if config.refine_scale2d_stop_iter > 0 and step is not None:
        split |= active & (
            (jnp.asarray(step) < config.refine_scale2d_stop_iter)
            & (state.max_radii[...] > config.grow_scale2d)
        )

    growth_score = jnp.maximum(average_gradient, state.max_radii[...])
    candidate = duplicate | split
    ranked_scores, ranked_parents = jax.lax.top_k(
        jnp.where(candidate, growth_score, -jnp.inf),
        allocation_count,
    )
    ranked_valid = jnp.isfinite(ranked_scores)
    raw_valid = jnp.stack(
        (
            ranked_valid & duplicate[ranked_parents],
            ranked_valid & split[ranked_parents],
        ),
        axis=-1,
    ).reshape(-1)
    raw_parent_ids = jnp.repeat(ranked_parents, 2)
    raw_parent_ranks = jnp.repeat(
        jnp.arange(allocation_count, dtype=jnp.int32), 2
    )
    raw_is_split = jnp.tile(
        jnp.asarray([False, True], dtype=jnp.bool_), allocation_count
    )
    event_positions = jnp.nonzero(
        raw_valid, size=allocation_count, fill_value=0
    )[0]
    parent_ids = raw_parent_ids[event_positions]
    parent_ranks = raw_parent_ranks[event_positions]
    is_split = raw_is_split[event_positions]
    planned_new_count = jnp.minimum(
        jnp.count_nonzero(raw_valid), allocation_count
    )
    selected = (
        jnp.arange(allocation_count, dtype=jnp.int32) < planned_new_count
    )

    free_scores, free_ids = jax.lax.top_k(
        (~active).astype(jnp.int32), allocation_count
    )
    free_count = jnp.count_nonzero(~active)
    capacity_overflow = planned_new_count > free_count
    valid_new = selected & (free_scores > 0) & ~capacity_overflow
    return _DefaultGrowthEvents(
        active,
        parent_ids,
        parent_ranks,
        selected,
        is_split,
        planned_new_count,
        free_count,
        free_ids,
        valid_new,
        capacity_overflow,
    )


def _scatter_any(
    indices: jax.Array, valid: jax.Array, capacity: int
) -> jax.Array:
    return (
        jnp.zeros((capacity,), dtype=jnp.int32)
        .at[indices]
        .add(valid.astype(jnp.int32))
        > 0
    )


def _default_split_opacity_logits(
    opacity_logits: jax.Array,
    split: jax.Array,
    revised_opacity: bool,
) -> jax.Array:
    if not revised_opacity:
        return opacity_logits
    opacity = jax.nn.sigmoid(opacity_logits)
    revised = 1.0 - jnp.sqrt(1.0 - opacity)
    return jnp.where(split, inverse_sigmoid(revised), opacity_logits)


def _default_prune_mask(
    active: jax.Array,
    opacity_logits: jax.Array,
    log_scales: jax.Array,
    max_radii: jax.Array,
    scene_scale: jax.Array,
    config: StrategyConfig,
    step: int | jax.Array | None,
) -> jax.Array:
    low_opacity = jax.nn.sigmoid(opacity_logits) < config.prune_opacity
    too_large = (
        jnp.max(jnp.exp(log_scales), axis=-1)
        > config.prune_scale3d * scene_scale
    )
    if step is not None:
        if config.refine_scale2d_stop_iter > 0:
            too_large |= (
                jnp.asarray(step) < config.refine_scale2d_stop_iter
            ) & (max_radii > config.prune_scale2d)
        too_large &= jnp.asarray(step) > config.reset_every
    return active & (low_opacity | too_large)


@nnx.jit(static_argnames=("config",))
def _default_refine_plan(
    model: GaussianModel,
    state: StrategyState,
    scene_scale: jax.Array,
    step: int | jax.Array | None,
    *,
    config: StrategyConfig,
) -> dict[str, jax.Array]:
    events = _default_growth_events(
        model, state, scene_scale, config, step
    )
    selected_split = events.selected & events.is_split
    split_parents = _scatter_any(
        events.parent_ids, selected_split, model.capacity
    )
    scale_reduction = jnp.asarray(jnp.log(1.6), model.log_scales[...].dtype)
    grown_log_scales = model.log_scales[...] - jnp.where(
        split_parents[:, None], scale_reduction, 0.0
    )
    grown_opacity_logits = _default_split_opacity_logits(
        model.opacity_logits[...], split_parents, config.revised_opacity
    )
    prune_parents = _default_prune_mask(
        events.active,
        grown_opacity_logits,
        grown_log_scales,
        state.max_radii[...],
        scene_scale,
        config,
        step,
    )

    child_log_scales = model.log_scales[...][events.parent_ids] - jnp.where(
        selected_split[:, None], scale_reduction, 0.0
    )
    child_opacity_logits = _default_split_opacity_logits(
        model.opacity_logits[...][events.parent_ids],
        selected_split,
        config.revised_opacity,
    )
    prune_children = _default_prune_mask(
        events.selected,
        child_opacity_logits,
        child_log_scales,
        state.max_radii[...][events.parent_ids],
        scene_scale,
        config,
        step,
    )
    pruned_count = jnp.count_nonzero(prune_parents) + jnp.count_nonzero(
        prune_children
    )
    active_after_prune_count = jnp.count_nonzero(
        events.active & ~prune_parents
    )
    active_after_prune_count += jnp.count_nonzero(
        events.selected & ~prune_children
    )
    return {
        "planned_new_count": events.planned_new_count,
        "pruned_count": pruned_count,
        "active_after_prune_count": active_after_prune_count,
        "free_count": events.free_count,
        "required_capacity": (
            jnp.count_nonzero(events.active) + events.planned_new_count
        ),
        "capacity_overflow": events.capacity_overflow,
    }


@nnx.jit(
    static_argnames=("config",),
    donate_argnames=("model", "state", "optimizer"),
)
def _default_refine(
    model: GaussianModel,
    state: StrategyState,
    optimizer: nnx.Optimizer,
    key: jax.Array,
    scene_scale: jax.Array,
    step: int | jax.Array | None,
    *,
    config: StrategyConfig,
) -> dict[str, jax.Array]:
    capacity = model.capacity
    allocation_count = min(config.max_new_per_refine, capacity)
    events = _default_growth_events(
        model, state, scene_scale, config, step
    )
    valid_new = events.valid_new
    valid_split = valid_new & events.is_split

    old_means = model.means[...]
    old_log_scales = model.log_scales[...]
    old_quats = model.quats[...]
    old_opacity_logits = model.opacity_logits[...]
    if model.has_appearance:
        old_features = model.features[...]
        old_colors = model.colors[...]
    else:
        old_sh0 = model.sh0[...]
        old_sh_rest = model.sh_rest[...]
    old_grad_accum = state.grad_accum[...]
    old_visible_count = state.visible_count[...]
    old_max_radii = state.max_radii[...]

    parent_scale = jnp.exp(old_log_scales[events.parent_ids])
    rotations = quat_to_rotmat(old_quats[events.parent_ids])
    ranked_noise = jax.random.normal(
        key, (2, allocation_count, 3), dtype=parent_scale.dtype
    )
    noise = ranked_noise[:, events.parent_ranks]
    offsets = jnp.einsum("nij,nj,bnj->bni", rotations, parent_scale, noise)
    parent_means = old_means[events.parent_ids]
    parent_log_scales = old_log_scales[events.parent_ids]
    parent_opacity_logits = old_opacity_logits[events.parent_ids]
    scale_reduction = jnp.asarray(jnp.log(1.6), parent_log_scales.dtype)
    child_means = parent_means + jnp.where(
        valid_split[:, None], offsets[1], 0.0
    )
    child_log_scales = parent_log_scales - jnp.where(
        valid_split[:, None], scale_reduction, 0.0
    )
    child_opacity_logits = _default_split_opacity_logits(
        parent_opacity_logits, valid_split, config.revised_opacity
    )

    split_parents = _scatter_any(
        events.parent_ids, valid_split, capacity
    )
    parent_offsets = jnp.zeros_like(old_means).at[events.parent_ids].add(
        jnp.where(valid_split[:, None], offsets[0], 0.0)
    )
    model.means[...] = old_means + parent_offsets
    model.log_scales[...] = old_log_scales - jnp.where(
        split_parents[:, None], scale_reduction, 0.0
    )
    model.opacity_logits[...] = _default_split_opacity_logits(
        old_opacity_logits, split_parents, config.revised_opacity
    )

    def assign(target: jax.Array, source: jax.Array) -> jax.Array:
        old = target[events.free_ids]
        mask = valid_new.reshape((allocation_count,) + (1,) * (target.ndim - 1))
        return target.at[events.free_ids].set(jnp.where(mask, source, old))

    model.means[...] = assign(model.means[...], child_means)
    model.log_scales[...] = assign(model.log_scales[...], child_log_scales)
    model.quats[...] = assign(model.quats[...], old_quats[events.parent_ids])
    model.opacity_logits[...] = assign(
        model.opacity_logits[...], child_opacity_logits
    )
    if model.has_appearance:
        model.features[...] = assign(
            model.features[...], old_features[events.parent_ids]
        )
        model.colors[...] = assign(
            model.colors[...], old_colors[events.parent_ids]
        )
    else:
        model.sh0[...] = assign(model.sh0[...], old_sh0[events.parent_ids])
        model.sh_rest[...] = assign(
            model.sh_rest[...], old_sh_rest[events.parent_ids]
        )
    state.grad_accum[...] = assign(
        state.grad_accum[...], old_grad_accum[events.parent_ids]
    )
    state.visible_count[...] = assign(
        state.visible_count[...], old_visible_count[events.parent_ids]
    )
    state.max_radii[...] = assign(
        state.max_radii[...], old_max_radii[events.parent_ids]
    )

    new_slots = _scatter_mask(events.free_ids, valid_new, capacity)
    grown_active = events.active | new_slots
    prune_candidates = _default_prune_mask(
        grown_active,
        model.opacity_logits[...],
        model.log_scales[...],
        state.max_radii[...],
        scene_scale,
        config,
        step,
    )
    prune = prune_candidates & ~events.capacity_overflow
    model.active_mask[...] = grown_active & ~prune
    reset_optimizer_slots(optimizer, new_slots | split_parents | prune)
    state.last_new_count[...] = jnp.count_nonzero(valid_new)
    state.last_pruned_count[...] = jnp.count_nonzero(prune)
    state.capacity_overflow[...] = events.capacity_overflow
    state.grad_accum[...] = jnp.where(
        events.capacity_overflow, old_grad_accum, 0.0
    )
    state.visible_count[...] = jnp.where(
        events.capacity_overflow, old_visible_count, 0.0
    )
    state.max_radii[...] = jnp.where(
        events.capacity_overflow, old_max_radii, 0.0
    )
    return {
        "new_count": state.last_new_count[...],
        "pruned_count": state.last_pruned_count[...],
        "capacity_overflow": state.capacity_overflow[...],
        "active_count": model.active_count,
        "_slot_copy_sources": events.parent_ids,
        "_slot_copy_targets": events.free_ids,
        "_slot_copy_valid": valid_new,
    }


def _mcmc_refine_decisions(
    model: GaussianModel,
    config: StrategyConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    active = model.active_mask[...]
    opacity = jax.nn.sigmoid(model.opacity_logits[...])
    finite_opacity = jnp.isfinite(opacity)
    dead = active & (~finite_opacity | (opacity <= config.prune_opacity))
    donor_mask = active & ~dead & finite_opacity
    has_donor = jnp.any(donor_mask)
    active_count = jnp.count_nonzero(active)
    target_count = (active_count * 105) // 100
    target_births = jnp.maximum(target_count - active_count, 0)
    remaining_to_cap = jnp.maximum(config.cap_max - active_count, 0)
    planned_new_count = jnp.where(
        has_donor,
        jnp.minimum(
            jnp.minimum(target_births, config.max_new_per_refine),
            remaining_to_cap,
        ),
        0,
    )
    free_count = jnp.count_nonzero(~active)
    return dead, active_count, planned_new_count, free_count, donor_mask


@nnx.jit(static_argnames=("config",))
def _mcmc_refine_plan(
    model: GaussianModel,
    *,
    config: StrategyConfig,
) -> dict[str, jax.Array]:
    dead, active_count, planned_new_count, free_count, donor_mask = (
        _mcmc_refine_decisions(model, config)
    )
    return {
        "planned_new_count": planned_new_count,
        "planned_relocate_count": jnp.where(
            jnp.any(donor_mask),
            jnp.minimum(jnp.count_nonzero(dead), config.max_new_per_refine),
            0,
        ),
        "active_after_prune_count": active_count,
        "free_count": free_count,
        "required_capacity": active_count + planned_new_count,
        "capacity_overflow": planned_new_count > free_count,
    }


@nnx.jit(
    static_argnames=("config",),
    donate_argnames=("model", "state", "optimizer"),
)
def _mcmc_refine(
    model: GaussianModel,
    state: StrategyState,
    optimizer: nnx.Optimizer,
    key: jax.Array,
    scene_scale: jax.Array,
    *,
    config: StrategyConfig,
) -> dict[str, jax.Array]:
    """Fixed-slot MCMC relocation and birth step.

    Low-opacity slots are relocated from opacity-weighted live donors. Free
    slots receive a bounded five-percent population increase. The topology is
    static even when no slots are available.
    """

    del scene_scale
    capacity = model.capacity
    allocation_count = min(config.max_new_per_refine, capacity)
    active = model.active_mask[...]
    opacity = jax.nn.sigmoid(model.opacity_logits[...])
    dead, _, planned_new_count, free_count, donor_mask = _mcmc_refine_decisions(
        model, config
    )
    capacity_overflow = planned_new_count > free_count
    old_grad_accum = state.grad_accum[...]
    old_visible_count = state.visible_count[...]
    old_max_radii = state.max_radii[...]
    donor_weights = jnp.where(donor_mask, opacity + 1.0e-8, 0.0)
    key_relocate, key_birth = jax.random.split(key)

    def sampled_ratios(ids: jax.Array, valid: jax.Array) -> jax.Array:
        counts = jnp.zeros((capacity,), dtype=jnp.int32)
        counts = counts.at[ids].add(valid.astype(jnp.int32))
        return counts[ids] + 1

    def sampled_mask(ids: jax.Array, valid: jax.Array) -> jax.Array:
        counts = jnp.zeros((capacity,), dtype=jnp.int32)
        return counts.at[ids].add(valid.astype(jnp.int32)) > 0

    def update_sampled(
        target: jax.Array,
        ids: jax.Array,
        values: jax.Array,
        valid: jax.Array,
    ) -> jax.Array:
        changed = sampled_mask(ids, valid)
        scattered = target.at[ids].set(values)
        mask = changed.reshape((capacity,) + (1,) * (target.ndim - 1))
        return jnp.where(mask, scattered, target)

    def copy_to_slots(
        target: jax.Array,
        target_ids: jax.Array,
        source_ids: jax.Array,
        valid: jax.Array,
    ) -> jax.Array:
        old = target[target_ids]
        source = target[source_ids]
        mask = valid.reshape((valid.shape[0],) + (1,) * (target.ndim - 1))
        return target.at[target_ids].set(jnp.where(mask, source, old))

    relocate_count = min(config.max_new_per_refine, capacity)
    dead_scores, dead_ids = jax.lax.top_k(dead.astype(jnp.int32), relocate_count)
    relocate_donor_ids, has_relocate_donor = _sample_weighted_ids(
        key_relocate,
        donor_weights,
        relocate_count,
    )
    valid_relocate = (
        (dead_scores > 0) & has_relocate_donor & ~capacity_overflow
    )
    relocated_opacity, relocated_scales = compute_relocation(
        opacity[relocate_donor_ids],
        jnp.exp(model.log_scales[...][relocate_donor_ids]),
        sampled_ratios(relocate_donor_ids, valid_relocate),
        _MCMC_BINOMIALS,
        min_opacity=config.prune_opacity,
    )
    relocated_opacity = jnp.clip(
        relocated_opacity,
        config.prune_opacity,
        1.0 - jnp.finfo(relocated_opacity.dtype).eps,
    )
    model.opacity_logits[...] = update_sampled(
        model.opacity_logits[...],
        relocate_donor_ids,
        inverse_sigmoid(relocated_opacity),
        valid_relocate,
    )
    model.log_scales[...] = update_sampled(
        model.log_scales[...],
        relocate_donor_ids,
        jnp.log(relocated_scales),
        valid_relocate,
    )

    model.means[...] = copy_to_slots(
        model.means[...], dead_ids, relocate_donor_ids, valid_relocate
    )
    model.log_scales[...] = copy_to_slots(
        model.log_scales[...], dead_ids, relocate_donor_ids, valid_relocate
    )
    model.quats[...] = copy_to_slots(
        model.quats[...], dead_ids, relocate_donor_ids, valid_relocate
    )
    model.opacity_logits[...] = copy_to_slots(
        model.opacity_logits[...], dead_ids, relocate_donor_ids, valid_relocate
    )
    if model.has_appearance:
        model.features[...] = copy_to_slots(
            model.features[...], dead_ids, relocate_donor_ids, valid_relocate
        )
        model.colors[...] = copy_to_slots(
            model.colors[...], dead_ids, relocate_donor_ids, valid_relocate
        )
    else:
        model.sh0[...] = copy_to_slots(
            model.sh0[...], dead_ids, relocate_donor_ids, valid_relocate
        )
        model.sh_rest[...] = copy_to_slots(
            model.sh_rest[...], dead_ids, relocate_donor_ids, valid_relocate
        )

    free_scores, free_ids = jax.lax.top_k(
        (~active).astype(jnp.int32), allocation_count
    )
    birth_opacity = jax.nn.sigmoid(model.opacity_logits[...])
    birth_weights = jnp.where(active, birth_opacity + 1.0e-8, 0.0)
    birth_donor_ids, has_birth_donor = _sample_weighted_ids(
        key_birth,
        birth_weights,
        allocation_count,
    )
    valid_birth = (free_scores > 0) & (
        jnp.arange(allocation_count) < planned_new_count
    ) & has_birth_donor & ~capacity_overflow
    donor_birth_opacity, donor_birth_scales = compute_relocation(
        birth_opacity[birth_donor_ids],
        jnp.exp(model.log_scales[...][birth_donor_ids]),
        sampled_ratios(birth_donor_ids, valid_birth),
        _MCMC_BINOMIALS,
        min_opacity=config.prune_opacity,
    )
    donor_birth_opacity = jnp.clip(
        donor_birth_opacity,
        config.prune_opacity,
        1.0 - jnp.finfo(donor_birth_opacity.dtype).eps,
    )
    model.opacity_logits[...] = update_sampled(
        model.opacity_logits[...],
        birth_donor_ids,
        inverse_sigmoid(donor_birth_opacity),
        valid_birth,
    )
    model.log_scales[...] = update_sampled(
        model.log_scales[...],
        birth_donor_ids,
        jnp.log(donor_birth_scales),
        valid_birth,
    )

    model.means[...] = copy_to_slots(
        model.means[...], free_ids, birth_donor_ids, valid_birth
    )
    model.log_scales[...] = copy_to_slots(
        model.log_scales[...], free_ids, birth_donor_ids, valid_birth
    )
    model.quats[...] = copy_to_slots(
        model.quats[...], free_ids, birth_donor_ids, valid_birth
    )
    model.opacity_logits[...] = copy_to_slots(
        model.opacity_logits[...], free_ids, birth_donor_ids, valid_birth
    )
    if model.has_appearance:
        model.features[...] = copy_to_slots(
            model.features[...], free_ids, birth_donor_ids, valid_birth
        )
        model.colors[...] = copy_to_slots(
            model.colors[...], free_ids, birth_donor_ids, valid_birth
        )
    else:
        model.sh0[...] = copy_to_slots(
            model.sh0[...], free_ids, birth_donor_ids, valid_birth
        )
        model.sh_rest[...] = copy_to_slots(
            model.sh_rest[...], free_ids, birth_donor_ids, valid_birth
        )

    new_slots = _scatter_mask(free_ids, valid_birth, capacity)
    model.active_mask[...] = active | new_slots
    changed_slots = (
        sampled_mask(relocate_donor_ids, valid_relocate)
        | sampled_mask(dead_ids, valid_relocate)
        | sampled_mask(birth_donor_ids, valid_birth)
        | sampled_mask(free_ids, valid_birth)
    )
    reset_optimizer_slots(optimizer, changed_slots)
    state.last_new_count[...] = jnp.count_nonzero(valid_birth)
    state.last_pruned_count[...] = 0
    state.capacity_overflow[...] = capacity_overflow
    state.grad_accum[...] = jnp.where(capacity_overflow, old_grad_accum, 0.0)
    state.visible_count[...] = jnp.where(
        capacity_overflow, old_visible_count, 0.0
    )
    state.max_radii[...] = jnp.where(capacity_overflow, old_max_radii, 0.0)
    return {
        "new_count": state.last_new_count[...],
        "pruned_count": state.last_pruned_count[...],
        "relocated_count": jnp.count_nonzero(valid_relocate),
        "capacity_overflow": state.capacity_overflow[...],
        "active_count": model.active_count,
        "_slot_relocate_sources": relocate_donor_ids,
        "_slot_relocate_targets": dead_ids,
        "_slot_relocate_valid": valid_relocate,
        "_slot_birth_sources": birth_donor_ids,
        "_slot_birth_targets": free_ids,
        "_slot_birth_valid": valid_birth,
    }


@nnx.jit(
    static_argnames=("maximum_opacity",),
    donate_argnames=("model", "optimizer"),
)
def reset_opacities(
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    *,
    maximum_opacity: float,
) -> None:
    maximum_logit = inverse_sigmoid(maximum_opacity)
    changed = model.active_mask[...] & (model.opacity_logits[...] > maximum_logit)
    model.opacity_logits[...] = jnp.where(
        changed, maximum_logit, model.opacity_logits[...]
    )
    reset_optimizer_slots(
        optimizer,
        changed,
        parameter_names=("opacity_logits",),
    )


@dataclass(frozen=True, init=False)
class DefaultStrategy(Strategy):
    """Original 3DGS refinement adapted to fixed-capacity NNX state."""

    config: StrategyConfig | None = None
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3_000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: str = "means2d"

    def __init__(
        self,
        prune_opa: float | StrategyConfig = 0.005,
        grow_grad2d: float = 0.0002,
        grow_scale3d: float = 0.01,
        grow_scale2d: float = 0.05,
        prune_scale3d: float = 0.1,
        prune_scale2d: float = 0.15,
        refine_scale2d_stop_iter: int = 0,
        refine_start_iter: int = 500,
        refine_stop_iter: int = 15_000,
        reset_every: int = 3_000,
        refine_every: int = 100,
        pause_refine_after_reset: int = 0,
        absgrad: bool = False,
        revised_opacity: bool = False,
        verbose: bool = False,
        key_for_gradient: str = "means2d",
        *,
        config: StrategyConfig | None = None,
    ) -> None:
        if isinstance(prune_opa, StrategyConfig):
            if config is not None:
                raise TypeError("config was provided both positionally and by keyword")
            config = prune_opa
            prune_opa = 0.005
        elif config is not None and not isinstance(config, StrategyConfig):
            raise TypeError("config must be a StrategyConfig")

        for name, value in (
            ("config", config),
            ("prune_opa", prune_opa),
            ("grow_grad2d", grow_grad2d),
            ("grow_scale3d", grow_scale3d),
            ("grow_scale2d", grow_scale2d),
            ("prune_scale3d", prune_scale3d),
            ("prune_scale2d", prune_scale2d),
            ("refine_scale2d_stop_iter", refine_scale2d_stop_iter),
            ("refine_start_iter", refine_start_iter),
            ("refine_stop_iter", refine_stop_iter),
            ("reset_every", reset_every),
            ("refine_every", refine_every),
            ("pause_refine_after_reset", pause_refine_after_reset),
            ("absgrad", absgrad),
            ("revised_opacity", revised_opacity),
            ("verbose", verbose),
            ("key_for_gradient", key_for_gradient),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        config = self.config
        if config is None:
            config = StrategyConfig(
                kind="default",
                refine_start=self.refine_start_iter,
                refine_stop=self.refine_stop_iter,
                refine_every=self.refine_every,
                reset_every=self.reset_every,
                grow_grad2d=self.grow_grad2d,
                grow_scale3d=self.grow_scale3d,
                grow_scale2d=self.grow_scale2d,
                prune_opacity=self.prune_opa,
                prune_scale3d=self.prune_scale3d,
                prune_scale2d=self.prune_scale2d,
                refine_scale2d_stop_iter=self.refine_scale2d_stop_iter,
                pause_refine_after_reset=self.pause_refine_after_reset,
                absgrad=self.absgrad,
                revised_opacity=self.revised_opacity,
                verbose=self.verbose,
                key_for_gradient=self.key_for_gradient,
                reset_opacity=self.prune_opa * 2.0,
            )
        elif config.kind != "default":
            raise ValueError("DefaultStrategy requires a default StrategyConfig")
        object.__setattr__(self, "config", config)
        for name, value in (
            ("prune_opa", config.prune_opacity),
            ("grow_grad2d", config.grow_grad2d),
            ("grow_scale3d", config.grow_scale3d),
            ("grow_scale2d", config.grow_scale2d),
            ("prune_scale3d", config.prune_scale3d),
            ("prune_scale2d", config.prune_scale2d),
            ("refine_scale2d_stop_iter", config.refine_scale2d_stop_iter),
            ("refine_start_iter", config.refine_start),
            ("refine_stop_iter", config.refine_stop),
            ("reset_every", config.reset_every),
            ("refine_every", config.refine_every),
            ("pause_refine_after_reset", config.pause_refine_after_reset),
            ("absgrad", config.absgrad),
            ("revised_opacity", config.revised_opacity),
            ("verbose", config.verbose),
            ("key_for_gradient", config.key_for_gradient),
        ):
            object.__setattr__(self, name, value)

    def initialize_state(
        self,
        scene_scale: float = 1.0,
        *,
        capacity: int | None = None,
    ) -> StrategyState:
        """Initialize current-main state, optionally binding a JAX capacity.

        Upstream initializes statistics lazily because their length is unknown
        until the first callback. This port uses a zero-length state for the
        same no-argument lifecycle and materializes it from ``params.capacity``
        at the first callback. Existing jax-gs calls passed the physical
        capacity as the first integer argument; that form remains supported.
        """

        if capacity is None:
            try:
                legacy_capacity = operator.index(scene_scale)
            except TypeError:
                pass
            else:
                capacity = legacy_capacity
                scene_scale = 1.0
        return StrategyState(
            0 if capacity is None else capacity,
            scene_scale=scene_scale,
        )

    def check_sanity(
        self, params: GaussianModel, optimizers: nnx.Optimizer
    ) -> None:
        super().check_sanity(params, optimizers)

    def step_pre_backward(
        self,
        params: GaussianModel,
        optimizers: nnx.Optimizer,
        state: StrategyState,
        step: int,
        info: Mapping[str, Any],
    ) -> None:
        del step
        self.check_sanity(params, optimizers)
        state.ensure_capacity(params.capacity)
        if self.key_for_gradient not in info:
            raise AssertionError(
                f"{self.key_for_gradient!r} is required in rasterization info"
            )

    def should_refine(self, step: int) -> bool:
        return (
            self.config.refine_start < step < self.config.refine_stop
            and step % self.config.refine_every == 0
            and step % self.config.reset_every
            >= self.config.pause_refine_after_reset
        )

    def should_reset(self, step: int) -> bool:
        return step > 0 and step % self.config.reset_every == 0

    def plan_refine(
        self,
        model: GaussianModel,
        state: StrategyState,
        scene_scale: float | jax.Array,
        *,
        step: int | jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        state.ensure_capacity(model.capacity)
        return _default_refine_plan(
            model,
            state,
            jnp.asarray(scene_scale, jnp.float32),
            step,
            config=self.config,
        )

    def required_capacity(
        self,
        model: GaussianModel,
        state: StrategyState,
        scene_scale: float | jax.Array,
        *,
        step: int | jax.Array | None = None,
    ) -> jax.Array:
        return self.plan_refine(
            model, state, scene_scale, step=step
        )["required_capacity"]

    def refine(
        self,
        model: GaussianModel,
        state: StrategyState,
        optimizer: nnx.Optimizer,
        key: jax.Array,
        scene_scale: float | jax.Array,
        *,
        step: int | jax.Array | None = None,
        scene: Any = None,
    ) -> dict[str, jax.Array]:
        _validate_fixed_slot_scene(scene, model.capacity)
        state.ensure_capacity(model.capacity)
        result = _default_refine(
            model,
            state,
            optimizer,
            key,
            jnp.asarray(scene_scale, jnp.float32),
            step,
            config=self.config,
        )
        return _commit_and_strip_slot_transactions(scene, result)

    def step_post_backward(
        self,
        params: GaussianModel,
        optimizers: nnx.Optimizer,
        state: StrategyState,
        step: int,
        info: Mapping[str, Any],
        packed: bool = False,
        scene: Any = None,
        *,
        key: jax.Array | None = None,
        scene_scale: float | jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Accumulate explicit screen gradients, then refine/reset on schedule.

        JAX arrays do not retain an intermediate ``.grad`` attribute. Callers
        provide ``<key_for_gradient>_gradient`` in dense ``[C, N, 2]`` or
        padded packed ``[P, 2]`` form. With ``absgrad=True``, callers instead
        provide the compositor-side statistic as
        ``<key_for_gradient>_absgrad``; this explicit array replaces gsplat's
        mutable PyTorch ``.absgrad`` side attribute. The former
        capacity-shaped gradient form remains accepted for compatibility with
        earlier jax-gs checkpoints and callback code in signed mode.
        """

        self.check_sanity(params, optimizers)
        _validate_fixed_slot_scene(scene, params.capacity)
        state.ensure_capacity(params.capacity)
        if step >= self.refine_stop_iter:
            return {}
        if self.absgrad:
            absgrad_key = f"{self.key_for_gradient}_absgrad"
            gradient = info.get(absgrad_key)
            if gradient is None:
                raise ValueError(
                    f"info must provide {absgrad_key!r} when absgrad=True"
                )
        else:
            gradient = info.get("means_gradient")
            if gradient is None:
                gradient = info.get(f"{self.key_for_gradient}_gradient")
            if gradient is None:
                raise ValueError(
                    "info must provide 'means_gradient' or "
                    f"'{self.key_for_gradient}_gradient' in the pure-JAX port"
                )
        gradient = jnp.asarray(gradient)
        radii = info.get("radii")
        valid = info.get("valid")
        is_screen_gradient = gradient.shape[-1:] == (2,)
        if packed:
            if not is_screen_gradient:
                raise ValueError("packed strategy gradient must have shape [P, 2]")
            if radii is None:
                raise ValueError("packed screen statistics require 'radii'")
            radii_array = jnp.asarray(radii)
            if valid is None:
                valid = jnp.all(radii_array > 0.0, axis=-1)
            gaussian_ids = info.get("gaussian_ids")
            if gaussian_ids is None:
                raise ValueError("packed screen statistics require 'gaussian_ids'")
            for required_key in ("width", "height", "n_cameras"):
                if required_key not in info:
                    raise ValueError(
                        "packed screen statistics require "
                        f"{required_key!r}"
                    )
            stats = _build_packed_densification_stats(
                gradient,
                radii_array,
                valid,
                gaussian_ids,
                params.active_mask[...],
                info["width"],
                info["height"],
                info["n_cameras"],
                info.get("projection_valid_count"),
            )
            accumulate_densification_stats(state, stats)
        elif gradient.ndim == 3 and is_screen_gradient:
            if gradient.shape[1] != params.capacity:
                raise ValueError("dense screen gradient must have shape [C, N, 2]")
            if radii is None:
                raise ValueError("dense screen statistics require 'radii'")
            radii_array = jnp.asarray(radii)
            if valid is None:
                valid = jnp.all(radii_array > 0.0, axis=-1)
            for required_key in ("width", "height"):
                if required_key not in info:
                    raise ValueError(
                        "dense screen statistics require "
                        f"{required_key!r}"
                    )
            stats = build_densification_stats(
                gradient,
                radii_array,
                valid,
                params.active_mask[...],
                info["width"],
                info["height"],
            )
            accumulate_densification_stats(state, stats)
        else:
            if gradient.ndim != 2 or gradient.shape[0] != params.capacity:
                raise ValueError(
                    "strategy gradient must have shape [C, N, 2], [P, 2], "
                    "or the legacy [N, channels]"
                )
            visible = info.get("visible")
            if visible is None:
                if radii is None:
                    visible = params.active_mask[...]
                else:
                    radii_array = jnp.asarray(radii)
                    if radii_array.shape[0] != params.capacity:
                        raise ValueError(
                            "radii must have capacity as its first axis"
                        )
                    visible = jnp.max(
                        radii_array.reshape(params.capacity, -1), axis=-1
                    ) > 0.0
            visible = jnp.asarray(visible, dtype=jnp.bool_)
            if visible.shape != (params.capacity,):
                raise ValueError("visible must have shape (capacity,)")
            max_radii = None
            if radii is not None:
                radii_array = jnp.asarray(radii)
                if radii_array.shape[0] != params.capacity:
                    raise ValueError(
                        "radii must have capacity as its first axis"
                    )
                max_radii = jnp.max(
                    radii_array.reshape(params.capacity, -1), axis=-1
                )
            update_strategy_state(state, params, gradient, visible, max_radii)
        if scene_scale is None:
            scene_scale = jnp.copy(state.scene_scale[...])

        result: dict[str, jax.Array] = {}
        refine_overflow = False
        if key is None:
            key = jax.random.fold_in(jax.random.key(0), step)
        if self.should_refine(step):
            refine_result = self.refine(
                params,
                state,
                optimizers,
                key,
                scene_scale,
                step=step,
                scene=scene,
            )
            result.update(refine_result)
            refine_overflow = bool(
                jax.device_get(refine_result["capacity_overflow"])
            )
        if self.should_reset(step):
            if refine_overflow:
                result["opacity_reset"] = jnp.asarray(False)
            else:
                reset_opacities(
                    params,
                    optimizers,
                    maximum_opacity=self.config.reset_opacity,
                )
                result["opacity_reset"] = jnp.asarray(True)
        return result


@dataclass(frozen=True, init=False)
class MCMCStrategy(DefaultStrategy):
    config: StrategyConfig | None = None
    refine_stop_iter: int = 25_000
    cap_max: int = 1_000_000
    noise_lr: float = 5.0e5
    noise_injection_stop_iter: int = -1
    min_opacity: float = 0.005
    noise_opacity_t: float = 0.005
    noise_opacity_k: float = 100.0

    def __init__(
        self,
        cap_max: int | StrategyConfig = 1_000_000,
        noise_lr: float = 5.0e5,
        refine_start_iter: int = 500,
        refine_stop_iter: int = 25_000,
        noise_injection_stop_iter: int = -1,
        refine_every: int = 100,
        min_opacity: float = 0.005,
        verbose: bool = False,
        noise_opacity_t: float = 0.005,
        noise_opacity_k: float = 100.0,
        *,
        config: StrategyConfig | None = None,
    ) -> None:
        if isinstance(cap_max, StrategyConfig):
            if config is not None:
                raise TypeError("config was provided both positionally and by keyword")
            config = cap_max
            cap_max = 1_000_000
        elif config is not None and not isinstance(config, StrategyConfig):
            raise TypeError("config must be a StrategyConfig")

        for name, value in (
            ("config", config),
            ("prune_opa", min_opacity),
            ("grow_grad2d", 0.0002),
            ("grow_scale3d", 0.01),
            ("grow_scale2d", 0.05),
            ("prune_scale3d", 0.1),
            ("prune_scale2d", 0.15),
            ("refine_scale2d_stop_iter", 0),
            ("refine_start_iter", refine_start_iter),
            ("refine_stop_iter", refine_stop_iter),
            ("reset_every", 3_000),
            ("refine_every", refine_every),
            ("pause_refine_after_reset", 0),
            ("absgrad", False),
            ("revised_opacity", False),
            ("verbose", verbose),
            ("key_for_gradient", "means2d"),
            ("cap_max", cap_max),
            ("noise_lr", noise_lr),
            ("noise_injection_stop_iter", noise_injection_stop_iter),
            ("min_opacity", min_opacity),
            ("noise_opacity_t", noise_opacity_t),
            ("noise_opacity_k", noise_opacity_k),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        config = self.config
        if config is None:
            config = StrategyConfig(
                kind="mcmc",
                refine_start=self.refine_start_iter,
                refine_stop=self.refine_stop_iter,
                refine_every=self.refine_every,
                prune_opacity=self.min_opacity,
                cap_max=self.cap_max,
                noise_lr=self.noise_lr,
                noise_injection_stop_iter=self.noise_injection_stop_iter,
                noise_opacity_t=self.noise_opacity_t,
                noise_opacity_k=self.noise_opacity_k,
                verbose=self.verbose,
            )
        elif config.kind != "mcmc":
            raise ValueError("MCMCStrategy requires an mcmc StrategyConfig")
        object.__setattr__(self, "config", config)
        for name, value in (
            ("refine_start_iter", config.refine_start),
            ("refine_stop_iter", config.refine_stop),
            ("refine_every", config.refine_every),
            ("min_opacity", config.prune_opacity),
            ("prune_opa", config.prune_opacity),
            ("cap_max", config.cap_max),
            ("noise_lr", config.noise_lr),
            ("noise_injection_stop_iter", config.noise_injection_stop_iter),
            ("noise_opacity_t", config.noise_opacity_t),
            ("noise_opacity_k", config.noise_opacity_k),
            ("verbose", config.verbose),
        ):
            object.__setattr__(self, name, value)

    def initialize_state(self, capacity: int | None = None) -> StrategyState:
        """Initialize MCMC state, binding capacity lazily when omitted."""

        return StrategyState(0 if capacity is None else capacity)

    def should_reset(self, step: int) -> bool:
        del step
        return False

    def plan_refine(
        self,
        model: GaussianModel,
        state: StrategyState,
        scene_scale: float | jax.Array,
        *,
        step: int | jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        del scene_scale, step
        state.ensure_capacity(model.capacity)
        return _mcmc_refine_plan(model, config=self.config)

    def required_capacity(
        self,
        model: GaussianModel,
        state: StrategyState,
        scene_scale: float | jax.Array,
        *,
        step: int | jax.Array | None = None,
    ) -> jax.Array:
        return self.plan_refine(
            model, state, scene_scale, step=step
        )["required_capacity"]

    def refine(
        self,
        model: GaussianModel,
        state: StrategyState,
        optimizer: nnx.Optimizer,
        key: jax.Array,
        scene_scale: float | jax.Array,
        *,
        step: int | jax.Array | None = None,
        scene: Any = None,
    ) -> dict[str, jax.Array]:
        del step
        _validate_fixed_slot_scene(scene, model.capacity)
        state.ensure_capacity(model.capacity)
        result = _mcmc_refine(
            model,
            state,
            optimizer,
            key,
            jnp.asarray(scene_scale, jnp.float32),
            config=self.config,
        )
        return _commit_and_strip_slot_transactions(scene, result)

    def step_post_backward(
        self,
        params: GaussianModel,
        optimizers: nnx.Optimizer,
        state: StrategyState,
        step: int,
        info: Mapping[str, Any],
        lr: float,
        scene: Any = None,
        *,
        key: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        del info
        self.check_sanity(params, optimizers)
        _validate_fixed_slot_scene(scene, params.capacity)
        state.ensure_capacity(params.capacity)
        if key is None:
            key = jax.random.fold_in(jax.random.key(0), step)
        refine_key, noise_key = jax.random.split(key)
        result: dict[str, jax.Array] = {}
        refine_overflow = False
        if self.should_refine(step):
            refine_result = self.refine(
                params,
                state,
                optimizers,
                refine_key,
                1.0,
                step=step,
                scene=scene,
            )
            result.update(refine_result)
            refine_overflow = bool(
                jax.device_get(refine_result["capacity_overflow"])
            )
        noise_stop = self.noise_injection_stop_iter
        if noise_stop < 0 or step < noise_stop:
            if refine_overflow:
                result["noise_injected"] = jnp.asarray(False)
            else:
                from .ops import inject_noise_to_position

                inject_noise_to_position(
                    params,
                    optimizers,
                    state,
                    noise_scale=lr * self.noise_lr,
                    t=self.noise_opacity_t,
                    k=self.noise_opacity_k,
                    key=noise_key,
                )
                result["noise_injected"] = jnp.asarray(True)
        return result


__all__ = [
    "DensificationStats",
    "DefaultStrategy",
    "MCMCStrategy",
    "Strategy",
    "StrategyState",
    "accumulate_densification_stats",
    "build_densification_stats",
    "reset_opacities",
    "update_strategy_state",
]
