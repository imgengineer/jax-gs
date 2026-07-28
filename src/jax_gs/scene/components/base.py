"""Minimal scene interface shared by trainable and inference containers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax


class Scene(ABC):
    """Named scene with optional Gaussian-topology sidecar hooks."""

    def __init__(self, id: str) -> None:
        self.id = id

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("Scene id must be a non-empty string")
        self._id = value

    @abstractmethod
    def put(self, name: str, component: object) -> None:
        """Add a named component to the scene."""

    @abstractmethod
    def get(self, component: str) -> object:
        """Return a component from the scene."""

    def on_duplicate(self, sel: jax.Array) -> None:
        del sel

    def on_split(self, sel: jax.Array, rest: jax.Array) -> None:
        del sel, rest

    def on_remove(self, remove_mask: jax.Array) -> None:
        del remove_mask

    def on_relocate(
        self, dead_indices: jax.Array, sampled_indices: jax.Array
    ) -> None:
        del dead_indices, sampled_indices

    def on_sample_add(self, sampled_indices: jax.Array) -> None:
        del sampled_indices

    def on_permute(self, order: jax.Array) -> None:
        del order


__all__ = ["Scene"]
