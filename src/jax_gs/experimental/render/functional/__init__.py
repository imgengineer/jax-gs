"""Stateless experimental render entry points."""

from ..types import RenderReturn
from .gaussian_inference import rasterize_gaussian_inference_scene
from .render_scene import render_scene

__all__ = ["RenderReturn", "rasterize_gaussian_inference_scene", "render_scene"]
