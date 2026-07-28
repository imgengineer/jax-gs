"""Public rendering types shared by the high-level rasterizers.

Keeping policy objects and render-mode queries separate from numerical kernels
makes the current-main API usable without importing backend-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CameraModel = Literal["pinhole", "ortho", "fisheye", "ftheta", "lidar"]
RenderMode = Literal[
    "RGB",
    "d",
    "Ed",
    "D",
    "ED",
    "RGB-d",
    "RGB-Ed",
    "RGB+D",
    "RGB+ED",
]
RasterizeMode = Literal["classic", "antialiased"]


class RendererConfig:
    """Base class for public rasterizer selection policies."""

    def __new__(cls, *args, **kwargs):
        if cls is RendererConfig:
            raise TypeError(
                "RendererConfig is a base class; instantiate "
                "RendererConfig_MixedBatch or RendererConfig_ParallelBatch."
            )
        return super().__new__(cls)


@dataclass
class RendererConfig_MixedBatch(RendererConfig):
    """Use serial tile batches in forward and batch-parallel backward."""


@dataclass
class RendererConfig_ParallelBatch(RendererConfig):
    """Use batch-parallel eval3d forward and backward semantics."""


def _validate_renderer_config(renderer_config: RendererConfig) -> None:
    if renderer_config is None:
        raise TypeError("renderer_config must be a RendererConfig instance, got None.")
    if not isinstance(renderer_config, RendererConfig):
        raise TypeError(
            "renderer_config must be a RendererConfig instance, "
            f"got {type(renderer_config).__name__}."
        )
    if isinstance(
        renderer_config,
        (RendererConfig_MixedBatch, RendererConfig_ParallelBatch),
    ):
        return
    raise NotImplementedError(
        f"Unsupported renderer_config type: {type(renderer_config).__name__}."
    )


def resolve_renderer_config(
    renderer_config: RendererConfig | None,
    *,
    with_eval3d: bool,
) -> RendererConfig:
    """Resolve the default policy and validate its supported numerical path."""

    if renderer_config is None:
        renderer_config = RendererConfig_MixedBatch()
    _validate_renderer_config(renderer_config)
    if not with_eval3d and isinstance(
        renderer_config, RendererConfig_ParallelBatch
    ):
        raise ValueError(
            "RendererConfig_ParallelBatch requires with_eval3d=True; the "
            "non-eval3d path only supports RendererConfig_MixedBatch."
        )
    return renderer_config


def render_mode_has_color(mode: RenderMode) -> bool:
    return mode in {"RGB", "RGB-d", "RGB-Ed", "RGB+D", "RGB+ED"}


def render_mode_has_hit_distance(mode: RenderMode) -> bool:
    return mode in {"d", "Ed", "RGB-d", "RGB-Ed"}


def render_mode_has_depth(mode: RenderMode) -> bool:
    return mode in {"D", "ED", "RGB+D", "RGB+ED"}


def render_mode_has_expected_depth(mode: RenderMode) -> bool:
    return mode in {"Ed", "ED", "RGB-Ed", "RGB+ED"}


def render_mode_has_depth_channel(mode: RenderMode) -> bool:
    return render_mode_has_depth(mode) or render_mode_has_hit_distance(mode)


def render_mode_has_only_depth_channel(mode: RenderMode) -> bool:
    return render_mode_has_depth_channel(mode) and not render_mode_has_color(mode)


def render_mode_has_only_color(mode: RenderMode) -> bool:
    return not render_mode_has_depth_channel(mode) and render_mode_has_color(mode)


def resolve_tile_size(
    tile_size: int | None,
    *,
    with_eval3d: bool,
    width: int,
    height: int,
) -> int:
    """Choose current-main's path-specific tile-size default."""

    if tile_size is not None:
        return tile_size
    if with_eval3d:
        return 16 if min(width, height) >= 1080 else 8
    return 16


__all__ = [
    "CameraModel",
    "RasterizeMode",
    "RendererConfig",
    "RendererConfig_MixedBatch",
    "RendererConfig_ParallelBatch",
    "RenderMode",
    "render_mode_has_color",
    "render_mode_has_depth",
    "render_mode_has_depth_channel",
    "render_mode_has_expected_depth",
    "render_mode_has_hit_distance",
    "render_mode_has_only_color",
    "render_mode_has_only_depth_channel",
]
