from __future__ import annotations

from collections.abc import Mapping
import math
import operator
from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from ..config import OptimizerConfig
from ..model import GaussianModel


_PARAMETER_LABELS = {
    "means": "means",
    "log_scales": "scales",
    "quats": "quats",
    "opacity_logits": "opacities",
    "sh0": "sh0",
    "sh_rest": "sh_rest",
    "features": "features",
    "colors": "colors",
}


def _label_parameter_tree(parameters: Any) -> Any:
    def label(path: tuple[Any, ...], _: Any) -> str:
        if not path:
            raise ValueError("unexpected empty parameter path")
        key = getattr(path[0], "key", None)
        if key not in _PARAMETER_LABELS:
            raise KeyError(f"no optimizer label for parameter {key!r}")
        return _PARAMETER_LABELS[key]

    return jax.tree_util.tree_map_with_path(label, parameters)


def _row_selective(
    inner: optax.GradientTransformation,
) -> optax.GradientTransformationExtraArgs:
    """Keep optimizer state and parameter updates unchanged for hidden rows."""

    inner = optax.with_extra_args_support(inner)

    def update(
        updates: Any,
        state: optax.OptState,
        params: Any = None,
        *,
        visible_mask: jax.Array,
        **extra_args: Any,
    ) -> tuple[Any, optax.OptState]:
        visible_mask = jnp.asarray(visible_mask)
        if visible_mask.ndim != 1:
            raise ValueError("visible_mask must have shape [N]")
        if visible_mask.dtype != jnp.bool_:
            raise TypeError("visible_mask must be boolean")

        update_arrays = [
            value
            for value in jax.tree.leaves(updates)
            if hasattr(value, "shape") and hasattr(value, "dtype")
        ]
        if any(
            value.ndim == 0 or value.shape[0] != visible_mask.shape[0]
            for value in update_arrays
        ):
            raise ValueError(
                "visible_mask length does not match parameter rows"
            )

        transformed, candidate_state = inner.update(
            updates, state, params, **extra_args
        )
        capacity = visible_mask.shape[0]

        def keep_visible(candidate: Any, previous: Any) -> Any:
            if not (
                hasattr(candidate, "shape")
                and candidate.ndim > 0
                and candidate.shape[0] == capacity
            ):
                return candidate
            mask = visible_mask.reshape(
                (capacity,) + (1,) * (candidate.ndim - 1)
            )
            return jnp.where(mask, candidate, previous)

        def mask_update(value: Any) -> Any:
            if not hasattr(value, "shape"):
                return value
            mask = visible_mask.reshape(
                (capacity,) + (1,) * (value.ndim - 1)
            )
            return jnp.where(mask, value, jnp.zeros_like(value))

        selected_state = jax.tree.map(keep_visible, candidate_state, state)
        selected_updates = jax.tree.map(mask_update, transformed)
        return selected_updates, selected_state

    return optax.GradientTransformationExtraArgs(inner.init, update)


class _UncorrectedAdamState(NamedTuple):
    count: jax.Array
    mu: Any
    nu: Any


def _scale_by_uncorrected_adam(
    *,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1.0e-8,
) -> optax.GradientTransformation:
    """Match current-main SelectiveAdam's CUDA update without bias correction."""

    def init(params: Any) -> _UncorrectedAdamState:
        return _UncorrectedAdamState(
            count=jnp.zeros([], dtype=jnp.int32),
            mu=jax.tree.map(jnp.zeros_like, params),
            nu=jax.tree.map(jnp.zeros_like, params),
        )

    def update(
        updates: Any,
        state: _UncorrectedAdamState,
        params: Any = None,
    ) -> tuple[Any, _UncorrectedAdamState]:
        del params
        mu = jax.tree.map(
            lambda gradient, previous: (
                b1 * previous + (1.0 - b1) * gradient
            ),
            updates,
            state.mu,
        )
        nu = jax.tree.map(
            lambda gradient, previous: (
                b2 * previous + (1.0 - b2) * jnp.square(gradient)
            ),
            updates,
            state.nu,
        )
        scaled = jax.tree.map(
            lambda first, second: first / (jnp.sqrt(second) + eps),
            mu,
            nu,
        )
        return scaled, _UncorrectedAdamState(
            count=optax.safe_increment(state.count),
            mu=mu,
            nu=nu,
        )

    return optax.GradientTransformation(init, update)


def _create_optimizer(
    model: GaussianModel,
    config: OptimizerConfig,
    *,
    row_selective: bool,
    bias_correction: bool = True,
    batch_size: int = 1,
    world_size: int = 1,
    scene_scale: float = 1.0,
) -> nnx.Optimizer:
    try:
        batch_size = operator.index(batch_size)
        world_size = operator.index(world_size)
    except TypeError as exc:
        raise TypeError("batch_size and world_size must be integers") from exc
    if batch_size <= 0 or world_size <= 0:
        raise ValueError("batch_size and world_size must be positive")
    try:
        scene_scale = float(scene_scale)
    except (TypeError, ValueError) as exc:
        raise TypeError("scene_scale must be a real scalar") from exc
    if not math.isfinite(scene_scale) or scene_scale < 0.0:
        raise ValueError("scene_scale must be finite and non-negative")
    effective_batch_size = batch_size * world_size
    if effective_batch_size > 10:
        raise ValueError(
            "current-main Adam requires effective batch size <= 10 because "
            "its scaled beta1 becomes negative above that limit"
        )
    learning_rate_scale = math.sqrt(effective_batch_size)
    b1 = 1.0 - effective_batch_size * (1.0 - 0.9)
    b2 = 1.0 - effective_batch_size * (1.0 - 0.999)
    eps = config.eps / learning_rate_scale
    means_schedule = optax.exponential_decay(
        init_value=config.means_lr * scene_scale * learning_rate_scale,
        transition_steps=max(config.max_steps, 1),
        decay_rate=config.means_lr_final_scale,
        staircase=False,
    )

    def adam(learning_rate: Any) -> optax.GradientTransformation:
        if bias_correction:
            return optax.adam(
                learning_rate, b1=b1, b2=b2, eps=eps
            )
        return optax.chain(
            _scale_by_uncorrected_adam(b1=b1, b2=b2, eps=eps),
            optax.scale_by_learning_rate(learning_rate),
        )

    transforms: Mapping[str, optax.GradientTransformation] = {
        "means": adam(means_schedule),
        "scales": adam(config.scales_lr * learning_rate_scale),
        "quats": adam(config.quats_lr * learning_rate_scale),
        "opacities": adam(config.opacities_lr * learning_rate_scale),
    }
    transforms = {
        **transforms,
        **(
            {
                "features": adam(config.sh0_lr * learning_rate_scale),
                "colors": adam(config.sh0_lr * learning_rate_scale),
            }
            if model.has_appearance
            else {
                "sh0": adam(config.sh0_lr * learning_rate_scale),
                "sh_rest": adam(
                    config.sh_rest_lr * learning_rate_scale
                ),
            }
        ),
    }
    if row_selective:
        transforms = {
            name: _row_selective(transform)
            for name, transform in transforms.items()
        }
    parameter_state = nnx.as_pure(nnx.state(model, nnx.Param))
    labels = _label_parameter_tree(parameter_state)
    optimizer = nnx.Optimizer(
        model,
        optax.multi_transform(transforms, labels),
        wrt=nnx.Param,
    )
    optimizer._jax_gs_batch_size = batch_size
    optimizer._jax_gs_world_size = world_size
    optimizer._jax_gs_scene_scale = scene_scale
    optimizer._jax_gs_optimizer_config = config
    return optimizer


def create_optimizer(
    model: GaussianModel,
    config: OptimizerConfig = OptimizerConfig(),
    *,
    batch_size: int = 1,
    world_size: int = 1,
    scene_scale: float = 1.0,
) -> nnx.Optimizer:
    """Create current-main batch- and scene-scaled Adam for Gaussian rows."""

    return _create_optimizer(
        model,
        config,
        row_selective=False,
        batch_size=batch_size,
        world_size=world_size,
        scene_scale=scene_scale,
    )


def create_row_selective_optimizer(
    model: GaussianModel,
    config: OptimizerConfig = OptimizerConfig(),
    *,
    batch_size: int = 1,
    world_size: int = 1,
    scene_scale: float = 1.0,
) -> nnx.Optimizer:
    """Create Adam whose moments and parameters update only on visible rows.

    Parameters and optimizer moments retain their ordinary dense NNX layout.
    Call ``optimizer.update(..., visible_mask=mask)`` with a boolean ``[N]``
    mask. Global Adam and schedule counters still advance on every update.
    """

    return _create_optimizer(
        model,
        config,
        row_selective=True,
        batch_size=batch_size,
        world_size=world_size,
        scene_scale=scene_scale,
    )


def create_visible_adam_optimizer(
    model: GaussianModel,
    config: OptimizerConfig = OptimizerConfig(),
    *,
    batch_size: int = 1,
    world_size: int = 1,
    scene_scale: float = 1.0,
) -> nnx.Optimizer:
    """Create current-main SelectiveAdam with dense, row-masked state.

    Unlike :func:`create_row_selective_optimizer`, this uses the uncorrected
    first and second moments from gsplat's fused SelectiveAdam kernel. Hidden
    rows retain their parameters and moments bit-for-bit while global optimizer
    and learning-rate schedule counters advance.
    """

    return _create_optimizer(
        model,
        config,
        row_selective=True,
        bias_correction=False,
        batch_size=batch_size,
        world_size=world_size,
        scene_scale=scene_scale,
    )


def reset_optimizer_slots(
    optimizer: nnx.Optimizer,
    reset_mask: jax.Array,
    *,
    parameter_names: tuple[str, ...] | None = None,
) -> None:
    """Clear Adam moments for slots that were pruned or newly allocated."""

    capacity = reset_mask.shape[0]
    state = nnx.state(optimizer.opt_state)
    selected = None if parameter_names is None else frozenset(parameter_names)

    def reset(path: tuple[Any, ...], value: Any) -> Any:
        if not isinstance(value, jax.Array):
            return value
        if value.ndim == 0 or value.shape[0] != capacity:
            return value
        if selected is not None and not any(
            getattr(entry, "key", None) in selected for entry in path
        ):
            return value
        mask = reset_mask.reshape((capacity,) + (1,) * (value.ndim - 1))
        return jnp.where(mask, jnp.zeros_like(value), value)

    nnx.update(
        optimizer.opt_state,
        jax.tree_util.tree_map_with_path(reset, state),
    )


def reset_optimizer_indices(
    optimizer: nnx.Optimizer,
    slot_ids: jax.Array,
    valid: jax.Array,
    *,
    capacity: int,
    parameter_names: tuple[str, ...] | None = None,
) -> None:
    """Clear Adam moments at a fixed-size set of unique slot indices."""

    state = nnx.state(optimizer.opt_state)
    selected = None if parameter_names is None else frozenset(parameter_names)

    def reset(path: tuple[Any, ...], value: Any) -> Any:
        if not isinstance(value, jax.Array):
            return value
        if value.ndim == 0 or value.shape[0] != capacity:
            return value
        if selected is not None and not any(
            getattr(entry, "key", None) in selected for entry in path
        ):
            return value
        old = value[slot_ids]
        mask = valid.reshape((slot_ids.shape[0],) + (1,) * (value.ndim - 1))
        cleared = jnp.where(mask, jnp.zeros_like(old), old)
        return value.at[slot_ids].set(cleared)

    nnx.update(
        optimizer.opt_state,
        jax.tree_util.tree_map_with_path(reset, state),
    )


def reorder_optimizer_slots(
    optimizer: nnx.Optimizer,
    order: jax.Array,
) -> None:
    """Apply a slot permutation to every capacity-leading optimizer leaf."""

    capacity = order.shape[0]
    state = nnx.state(optimizer.opt_state)

    def reorder(value: Any) -> Any:
        if not isinstance(value, jax.Array):
            return value
        if value.ndim == 0 or value.shape[0] != capacity:
            return value
        return value[order]

    nnx.update(optimizer.opt_state, jax.tree.map(reorder, state))


def mask_inactive_gradients(grads: Any, active_mask: jax.Array) -> Any:
    """Zero any leading-capacity gradient leaves for inactive slots."""

    capacity = active_mask.shape[0]

    def mask(value: Any) -> Any:
        if not isinstance(value, jax.Array):
            return value
        if value.ndim == 0 or value.shape[0] != capacity:
            return value
        broadcast_mask = active_mask.reshape(
            (capacity,) + (1,) * (value.ndim - 1)
        )
        return jnp.where(broadcast_mask, value, 0.0)

    return jax.tree.map(mask, grads)


class SelectiveAdam:
    """Compatibility wrapper for current-main uncorrected SelectiveAdam."""

    def __init__(
        self,
        model: GaussianModel,
        config: OptimizerConfig = OptimizerConfig(),
    ) -> None:
        self.optimizer = create_visible_adam_optimizer(model, config)

    @property
    def step(self) -> jax.Array:
        return self.optimizer.step[...]

    def update(
        self,
        model: GaussianModel,
        grads: Any,
        visible_mask: jax.Array | None = None,
    ) -> Any:
        active_mask = model.active_mask[...]
        if visible_mask is None:
            visible_mask = active_mask
        else:
            visible_mask = jnp.asarray(visible_mask)
            if visible_mask.ndim != 1:
                raise ValueError("visible_mask must have shape [N]")
            if visible_mask.dtype != jnp.bool_:
                raise TypeError("visible_mask must be boolean")
            if visible_mask.shape != active_mask.shape:
                raise ValueError(
                    "visible_mask length does not match parameter rows"
                )
            visible_mask = visible_mask & active_mask
        updates = self.optimizer.update(
            model, grads, visible_mask=visible_mask
        )
        normalized_quats = model.normalized_quats
        model.quats[...] = jnp.where(
            visible_mask[:, None], normalized_quats, model.quats[...]
        )
        return updates

    def reset_slots(self, reset_mask: jax.Array) -> None:
        reset_optimizer_slots(self.optimizer, reset_mask)
