"""Minimal scene-to-render-function dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...scene import GaussianScene


class Stage:
    """Pair Gaussian scenes with render functions and dispatch by scene id."""

    def __init__(self) -> None:
        self._scenes: dict[str, tuple[GaussianScene, Callable]] = {}

    def add_scene(self, scene: GaussianScene, render_fn: Callable) -> None:
        """Register ``scene`` and its renderer under ``scene.id``."""

        if scene.id in self._scenes:
            raise ValueError(
                f"Scene {scene.id!r} already registered on this Stage"
            )
        self._scenes[scene.id] = (scene, render_fn)

    def scene_ids(self) -> list[str]:
        """Return registered ids in insertion order."""

        return list(self._scenes.keys())

    def get_scene(self, scene_id: str) -> GaussianScene:
        """Return a registered scene or raise a descriptive ``KeyError``."""

        if scene_id not in self._scenes:
            raise KeyError(
                f"Scene {scene_id!r} not registered; "
                f"available: {list(self._scenes.keys())}"
            )
        return self._scenes[scene_id][0]

    def render(self, scene_id: str, **kwargs: Any) -> Any:
        """Call the registered renderer as ``fn(splats=scene.splats, **kwargs)``."""

        if scene_id not in self._scenes:
            raise KeyError(
                f"Scene {scene_id!r} not registered; "
                f"available: {list(self._scenes.keys())}"
            )
        scene, render_fn = self._scenes[scene_id]
        return render_fn(splats=scene.splats, **kwargs)


__all__ = ["Stage"]
