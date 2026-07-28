"""Scene-type dispatcher for experimental rendering."""

from __future__ import annotations

from typing import Any

from ....scene import GaussianInferenceScene
from ..types import RenderReturn
from .gaussian_inference import rasterize_gaussian_inference_scene


def render_scene(
    scene: Any,
    *,
    out: RenderReturn | None = None,
    **request: Any,
) -> RenderReturn:
    """Render a ``GaussianInferenceScene`` and tag the selected path."""

    if not isinstance(scene, GaussianInferenceScene):
        raise TypeError(
            "render_scene requires a GaussianInferenceScene; "
            f"got {type(scene).__name__}"
        )
    result = rasterize_gaussian_inference_scene(scene, out=out, **request)
    result.metadata["render_path"] = "inference"
    return result


__all__ = ["render_scene"]
