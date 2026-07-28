"""Fixed-capacity dynamic-scene densification strategy."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from ...model import GaussianModel
from ...strategy import (
    DefaultStrategy,
    StrategyState,
    _default_growth_events,
)
from .deformation import DeformationTable


class DynamicStrategyState(StrategyState):
    """Default strategy statistics plus one fixed-capacity dynamic mask."""

    def __init__(self, capacity: int, *, init_dynamic: bool = True) -> None:
        super().__init__(capacity)
        self.dynamic_mask = nnx.Variable(
            jnp.full((capacity,), bool(init_dynamic), dtype=jnp.bool_)
        )


def _new_slot_inheritance(
    model: GaussianModel,
    state: StrategyState,
    scene_scale: jax.Array,
    strategy: DefaultStrategy,
    step: int | jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Recover the parent/free-slot mapping used by ``DefaultStrategy``."""

    events = _default_growth_events(
        model,
        state,
        scene_scale,
        strategy.config,
        step,
    )
    return events.parent_ids, events.free_ids, events.valid_new


class DynamicStrategy(DefaultStrategy):
    """Default fixed-slot refinement with parent-inherited dynamic flags.

    Upstream resizes a runtime-length mask. This port keeps a mask with the
    model's physical capacity: inactive values are irrelevant, newly allocated
    slots inherit their selected parent's flag, and pruned slots are cleared.
    """

    def initialize_state(
        self,
        capacity: int,
        *,
        init_dynamic: bool = True,
    ) -> DynamicStrategyState:
        return DynamicStrategyState(capacity, init_dynamic=init_dynamic)

    def refine(
        self,
        model: GaussianModel,
        state: StrategyState,
        optimizer: nnx.Optimizer,
        key: jax.Array,
        scene_scale: float | jax.Array,
        *,
        step: int | jax.Array | None = None,
        scene=None,
    ) -> dict[str, jax.Array]:
        if not isinstance(state, DynamicStrategyState):
            raise RuntimeError(
                "DynamicStrategy.refine called without a dynamic state; "
                "create it with strategy.initialize_state(capacity)."
            )

        scale = jnp.asarray(scene_scale, dtype=jnp.float32)
        parent_ids, free_ids, valid_new = _new_slot_inheritance(
            model,
            state,
            scale,
            self,
            step,
        )
        previous_mask = jnp.array(state.dynamic_mask[...], copy=True)
        previous_active = jnp.array(model.active_mask[...], copy=True)
        parent_flags = previous_mask[parent_ids]

        statistics = super().refine(
            model,
            state,
            optimizer,
            key,
            scale,
            step=step,
            scene=scene,
        )

        current_active = model.active_mask[...]
        pruned = previous_active & ~current_active
        updated_mask = previous_mask & ~pruned
        surviving_new = valid_new & current_active[free_ids]
        inherited = surviving_new & parent_flags
        target_flags = jnp.where(
            valid_new, inherited, updated_mask[free_ids]
        )
        refined_mask = updated_mask.at[free_ids].set(target_flags)
        state.dynamic_mask[...] = jnp.where(
            statistics["capacity_overflow"],
            previous_mask,
            refined_mask,
        )
        return statistics


__all__ = ["DynamicStrategy"]
