"""Fixed-capacity dynamic-scene densification strategy."""

from __future__ import annotations

import operator
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from ...model import GaussianModel
from ...strategy import (
    DefaultStrategy,
    StrategyState,
    _default_growth_events,
)
from .deformation import DeformationTable as DeformationTable  # noqa: PLC0414


class DynamicStrategyState(StrategyState):
    """Default strategy statistics plus one fixed-capacity dynamic mask."""

    def __init__(
        self,
        capacity: int,
        *,
        init_dynamic: bool = True,
        scene_scale: float = 1.0,
    ) -> None:
        super().__init__(capacity, scene_scale=scene_scale)
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
        scene_scale: float = 1.0,
        num_gaussians: int = 0,
        device: Any | None = None,
        init_dynamic: bool = True,
        *,
        capacity: int | None = None,
    ) -> DynamicStrategyState:
        """Initialize dynamic state on the upstream calling surface.

        Upstream sizes the mask by ``num_gaussians`` and places it on
        ``device``; this port sizes fixed-capacity state by the trailing JAX
        ``capacity`` extension and leaves placement to the caller, the same
        adaptation :meth:`DefaultStrategy.initialize_state` documents. A first
        positional integer keeps meaning the capacity, as it does there, so
        existing jax-gs calls stay valid.
        """

        del device
        if capacity is None:
            try:
                legacy_capacity = operator.index(scene_scale)
            except TypeError:
                pass
            else:
                capacity = legacy_capacity
                scene_scale = 1.0
        if capacity is None:
            capacity = num_gaussians
        return DynamicStrategyState(
            capacity,
            init_dynamic=init_dynamic,
            scene_scale=scene_scale,
        )

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
            raise RuntimeError(  # noqa: TRY004 - preserve the public state-contract error
                "DynamicStrategy.refine called without a dynamic state; "
                "create it with strategy.initialize_state(capacity=...)."
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
        target_flags = jnp.where(valid_new, inherited, updated_mask[free_ids])
        refined_mask = updated_mask.at[free_ids].set(target_flags)
        state.dynamic_mask[...] = jnp.where(
            statistics["capacity_overflow"],
            previous_mask,
            refined_mask,
        )
        return statistics


__all__ = ["DynamicStrategy"]
