"""Flax NNX storage for row-aligned trainable Gaussian scene components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp

from .base import Scene


def _array(value: Any) -> jax.Array:
    if isinstance(value, nnx.Variable):
        return value[...]
    return jnp.asarray(value)


def _component_dict(component: Mapping[str, Any] | nnx.Dict) -> nnx.Dict:
    if isinstance(component, nnx.Dict):
        return component
    if not isinstance(component, Mapping):
        raise TypeError(
            "component must be a mapping or flax.nnx.Dict; "
            f"got {type(component).__name__}"
        )
    return nnx.Dict(
        {
            key: value
            if isinstance(value, nnx.Variable)
            else nnx.Param(jnp.asarray(value))
            for key, value in component.items()
        }
    )


def _like_variable(template: Any, value: jax.Array) -> Any:
    if isinstance(template, nnx.Variable):
        return type(template)(value)
    return value


class GaussianScene(nnx.Module, Scene):
    """Trainable Gaussian tensors plus row-aligned component sidecars.

    ``nnx.Dict`` is the JAX analogue of upstream's ``ParameterDict``. Plain
    mappings are accepted and their array values are wrapped in ``nnx.Param``.
    Upstream-compatible ``on_*`` hooks retain dynamic-array semantics, while
    fixed-capacity strategies use :meth:`apply_slot_transaction`.
    """

    component_index: jax.Array = nnx.data()

    def __init__(self, id: str) -> None:
        Scene.__init__(self, id)
        self.splats = nnx.Dict()
        self.signal = nnx.Dict()
        self.component_names: list[str] = []
        self.component_index = jnp.zeros((0,), dtype=jnp.int32)

    def put(
        self, name: str, component: Mapping[str, Any] | nnx.Dict
    ) -> None:
        """Append a component during initialization and keep sidecars aligned."""

        if not name:
            raise ValueError("component name must not be empty")
        if name in self.component_names:
            raise ValueError(f"Component {name!r} already exists in scene")
        component = _component_dict(component)
        if len(component) == 0 or "means" not in component:
            raise ValueError("component splats must not be empty")

        component_count = int(_array(component["means"]).shape[0])
        if len(self.splats) == 0:
            # Preserve nnx.Dict identity, matching the first-ParameterDict rule.
            self.splats = component
            self.component_names = [name]
            self.component_index = jnp.zeros(
                (component_count,), dtype=jnp.int32
            )
        else:
            appended = {}
            for key, current in self.splats.items():
                combined = jnp.concatenate(
                    (_array(current), _array(component[key])), axis=0
                )
                appended[key] = _like_variable(current, combined)
            self.splats = nnx.Dict(appended)
            self.component_names.append(name)
            new_component_id = len(self.component_names) - 1
            self.component_index = jnp.concatenate(
                (
                    self.component_index,
                    jnp.full(
                        (component_count,),
                        new_component_id,
                        dtype=self.component_index.dtype,
                    ),
                )
            )
            for key, value in self.signal.items():
                pad = jnp.zeros(
                    (component_count,) + value.shape[1:], dtype=value.dtype
                )
                self.signal[key] = jnp.concatenate((value, pad), axis=0)
        self.validate()

    @classmethod
    def from_splats(
        cls,
        splats: Mapping[str, Any] | nnx.Dict,
        id: str,
        signal: Mapping[str, Any] | None = None,
    ) -> "GaussianScene":
        splats = _component_dict(splats)
        if len(splats) == 0 or "means" not in splats:
            raise ValueError(
                "from_splats requires a non-empty nnx.Dict containing 'means'"
            )
        scene = cls(id)
        if signal is not None:
            scene.signal = nnx.Dict(
                {key: jnp.asarray(value) for key, value in signal.items()}
            )
        scene.put(id, splats)
        return scene

    def validate(self) -> None:
        required_keys = ("means", "scales", "quats", "opacities")
        missing = [key for key in required_keys if key not in self.splats]
        if len(self.splats) > 0 and missing:
            raise ValueError(f"missing required splat keys: {missing}")

        count = self.num_gaussians()
        if not all(_array(value).shape[0] == count for value in self.splats.values()):
            raise ValueError(
                "every splat array must have leading dim == "
                f"num_gaussians: {count}"
            )
        if not all(value.shape[0] == count for value in self.signal.values()):
            raise ValueError(
                "every signal array must have leading dim == "
                f"num_gaussians: {count}"
            )
        if self.component_index.shape != (count,):
            raise ValueError(
                f"component_index shape {self.component_index.shape} != ({count},)"
            )
        if len(self.splats) > 0:
            if not self.component_names:
                raise ValueError("component_names must not be empty")
            if count > 0:
                minimum = int(jax.device_get(jnp.min(self.component_index)))
                maximum = int(jax.device_get(jnp.max(self.component_index)))
                if minimum < 0:
                    raise ValueError("component_index must be non-negative")
                if maximum >= len(self.component_names):
                    raise ValueError(
                        "component_index refers to an unknown component"
                    )

    def num_gaussians(self) -> int:
        return int(_array(self.splats["means"]).shape[0]) if "means" in self.splats else 0

    def _component_id(self, component: str | int) -> int:
        if isinstance(component, int):
            if component < 0 or component >= len(self.component_names):
                raise KeyError(f"Unknown component index: {component}")
            return component
        try:
            return self.component_names.index(component)
        except ValueError as exc:
            raise KeyError(f"Unknown component name: {component}") from exc

    def get(self, component: str | int) -> dict[str, object]:
        component_id = self._component_id(component)
        mask = self.component_index == component_id
        return {
            "name": self.component_names[component_id],
            "index": component_id,
            "mask": mask,
            "splats": {
                key: _array(value)[mask] for key, value in self.splats.items()
            },
            "signal": {key: value[mask] for key, value in self.signal.items()},
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "splats": {
                key: jnp.array(_array(value)) for key, value in self.splats.items()
            },
            "splats_requires_grad": {
                key: isinstance(value, nnx.Param)
                for key, value in self.splats.items()
            },
            "signal": {
                key: jnp.array(value) for key, value in self.signal.items()
            },
            "component_names": list(self.component_names),
            "component_index": jnp.array(self.component_index),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "GaussianScene":
        if "id" not in state:
            raise KeyError("state_dict missing required 'id' entry")
        scene = cls(state["id"])
        trainable: Mapping[str, bool] = state.get("splats_requires_grad", {})
        scene.splats = nnx.Dict(
            {
                key: nnx.Param(jnp.array(value))
                if trainable.get(key, True)
                else jnp.array(value)
                for key, value in state["splats"].items()
            }
        )
        scene.signal = nnx.Dict(
            {
                key: jnp.array(value)
                for key, value in state.get("signal", {}).items()
            }
        )
        scene.component_names = list(state.get("component_names", []))
        has_splats = "means" in scene.splats
        component_index = state.get("component_index")
        if component_index is None:
            count = scene.num_gaussians() if has_splats else 0
            component_index = jnp.zeros((count,), dtype=jnp.int32)
        scene.component_index = jnp.asarray(component_index, dtype=jnp.int32)
        if has_splats and not scene.component_names:
            scene.component_names = [scene.id]
        scene.validate()
        return scene

    def _append_signal_rows(self, indices: jax.Array) -> None:
        for key, value in self.signal.items():
            self.signal[key] = jnp.concatenate((value, value[indices]), axis=0)

    def validate_slot_capacity(self, capacity: int) -> None:
        """Validate that fixed-slot sidecars match one model bucket.

        Fixed-slot topology keeps inactive rows physically present. Their
        values are intentionally unspecified; the model's ``active_mask`` is
        the source of truth for whether a row's lineage is live.
        """

        capacity = int(capacity)
        if self.num_gaussians() != capacity:
            raise ValueError(
                "scene slot capacity must match model capacity: "
                f"{self.num_gaussians()} != {capacity}"
            )
        if self.component_index.shape != (capacity,):
            raise ValueError(
                "scene component_index must have shape "
                f"({capacity},), got {self.component_index.shape}"
            )
        for name, value in self.signal.items():
            if value.shape[0] != capacity:
                raise ValueError(
                    f"scene signal {name!r} must have leading dimension "
                    f"{capacity}, got {value.shape[0]}"
                )
        self.validate()

    def resize_slot_capacity(self, new_capacity: int) -> None:
        """Pad every scene row store for a larger fixed-capacity bucket.

        New rows are inactive according to the paired model's ``active_mask``.
        Splat and signal values are therefore neutral zero padding, while
        ``component_index`` uses the existing component id ``0`` only to keep
        the scene's membership invariant valid.
        """

        old_capacity = self.num_gaussians()
        self.validate_slot_capacity(old_capacity)
        new_capacity = int(new_capacity)
        if new_capacity <= old_capacity:
            raise ValueError(
                "new scene slot capacity must exceed the current capacity"
            )
        padding = new_capacity - old_capacity
        for key, value in self.splats.items():
            rows = _array(value)
            pad_width = ((0, padding),) + ((0, 0),) * (rows.ndim - 1)
            self.splats[key] = _like_variable(value, jnp.pad(rows, pad_width))
        self.component_index = jnp.pad(
            self.component_index,
            ((0, padding),),
            constant_values=0,
        )
        for key, value in self.signal.items():
            pad_width = ((0, padding),) + ((0, 0),) * (value.ndim - 1)
            self.signal[key] = jnp.pad(value, pad_width)
        self.validate()

    def apply_slot_permutation(self, order: jax.Array) -> None:
        """Apply one stable fixed-slot order to every scene row store."""

        capacity = self.num_gaussians()
        self.validate_slot_capacity(capacity)
        order = jnp.asarray(order, dtype=jnp.int32)
        if order.shape != (capacity,):
            raise ValueError(
                f"scene slot order must have shape ({capacity},), got {order.shape}"
            )
        expected = jnp.arange(capacity, dtype=jnp.int32)
        if not bool(jax.device_get(jnp.array_equal(jnp.sort(order), expected))):
            raise ValueError("scene slot order must be a permutation")

        splat_snapshots = {
            key: jnp.array(_array(value)) for key, value in self.splats.items()
        }
        for key, value in self.splats.items():
            reordered = splat_snapshots[key][order]
            if isinstance(value, nnx.Variable):
                value[...] = reordered
            else:
                self.splats[key] = reordered
        component_snapshot = jnp.array(self.component_index)
        signal_snapshots = {
            key: jnp.array(value) for key, value in self.signal.items()
        }
        self.component_index = component_snapshot[order]
        for key in self.signal:
            self.signal[key] = signal_snapshots[key][order]
        self.validate()

    def apply_slot_transaction(
        self,
        *,
        source_slots: jax.Array,
        target_slots: jax.Array,
        valid: jax.Array | None = None,
    ) -> None:
        """Copy fixed-slot lineage without resizing scene sidecars.

        Every valid pair performs ``target <- source`` for
        ``component_index`` and every signal. All sources are read from the
        sidecars as they existed before the transaction, so overlapping copy
        chains cannot observe earlier writes. Invalid entries are padding and
        leave their targets untouched. Splat parameters are not mirrored here;
        a separately stored ``scene.splats`` must be kept in sync by its owner.
        """

        capacity = self.num_gaussians()
        self.validate_slot_capacity(capacity)
        source_slots = jnp.asarray(source_slots, dtype=jnp.int32)
        target_slots = jnp.asarray(target_slots, dtype=jnp.int32)
        if source_slots.ndim != 1 or target_slots.ndim != 1:
            raise ValueError("source_slots and target_slots must be one-dimensional")
        if source_slots.shape != target_slots.shape:
            raise ValueError("source_slots and target_slots must have matching shapes")
        if valid is None:
            valid = jnp.ones(source_slots.shape, dtype=jnp.bool_)
        else:
            valid = jnp.asarray(valid, dtype=jnp.bool_)
        if valid.shape != source_slots.shape:
            raise ValueError("valid must match source_slots and target_slots")

        invalid_live_pair = valid & (
            (source_slots < 0)
            | (source_slots >= capacity)
            | (target_slots < 0)
            | (target_slots >= capacity)
        )
        if bool(jax.device_get(jnp.any(invalid_live_pair))):
            raise ValueError(
                "valid fixed-slot transaction indices must be inside scene capacity"
            )
        if capacity == 0 or source_slots.shape[0] == 0:
            return

        safe_sources = jnp.clip(source_slots, 0, capacity - 1)
        safe_targets = jnp.clip(target_slots, 0, capacity - 1)
        write_ids = jnp.arange(source_slots.shape[0], dtype=jnp.int32) + 1
        last_write = jnp.zeros((capacity,), dtype=jnp.int32).at[
            safe_targets
        ].max(jnp.where(valid, write_ids, 0))
        selected_pair = jnp.maximum(last_write - 1, 0)

        def copy_from_snapshot(value: jax.Array) -> jax.Array:
            snapshot = jnp.array(value)
            mask = (last_write > 0).reshape(
                last_write.shape + (1,) * (value.ndim - 1)
            )
            replacements = snapshot[safe_sources[selected_pair]]
            return jnp.where(mask, replacements, snapshot)

        self.component_index = copy_from_snapshot(self.component_index)
        for key, value in self.signal.items():
            self.signal[key] = copy_from_snapshot(value)

    def on_duplicate(self, sel: jax.Array) -> None:
        self.component_index = jnp.concatenate(
            (self.component_index, self.component_index[sel]), axis=0
        )
        self._append_signal_rows(sel)

    def on_split(self, sel: jax.Array, rest: jax.Array) -> None:
        self.component_index = jnp.concatenate(
            (
                self.component_index[rest],
                self.component_index[sel],
                self.component_index[sel],
            ),
            axis=0,
        )
        for key, value in self.signal.items():
            self.signal[key] = jnp.concatenate(
                (value[rest], value[sel], value[sel]), axis=0
            )

    def on_remove(self, remove_mask: jax.Array) -> None:
        keep = ~remove_mask
        self.component_index = self.component_index[keep]
        for key, value in self.signal.items():
            self.signal[key] = value[keep]

    def on_relocate(
        self, dead_indices: jax.Array, sampled_indices: jax.Array
    ) -> None:
        self.component_index = self.component_index.at[dead_indices].set(
            self.component_index[sampled_indices]
        )
        for key, value in self.signal.items():
            self.signal[key] = value.at[dead_indices].set(value[sampled_indices])

    def on_sample_add(self, sampled_indices: jax.Array) -> None:
        self.component_index = jnp.concatenate(
            (self.component_index, self.component_index[sampled_indices]), axis=0
        )
        self._append_signal_rows(sampled_indices)

    def on_permute(self, order: jax.Array) -> None:
        self.component_index = self.component_index[order]
        for key, value in self.signal.items():
            self.signal[key] = value[order]


__all__ = ["GaussianScene"]
