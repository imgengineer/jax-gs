"""Render interfaces with an explicit CuTe or reference backend."""

import chex

from ..config import CapacityConfig
from ..scene.camera import Camera
from .types import ProjectedGaussians, RenderResult, VisibilityTable


def rasterize_forward(
    projected: ProjectedGaussians,
    table: VisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    *,
    backend: str = "cute",
    background: chex.Array | None = None,
) -> RenderResult:
    """Forward rendering with a bounded diagnostic tile table."""
    if backend == "reference":
        from ..reference.rasterizer_jax import rasterize_jax

        return rasterize_jax(projected, table, camera, config, background)
    if backend == "cute":
        from ..kernels.rasterizer import rasterize_cute

        return rasterize_cute(projected, table, camera, config, background)
    raise ValueError(f"unknown backend: {backend}")


def rasterize(
    projected: ProjectedGaussians,
    table: VisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    *,
    backend: str = "cute",
    background: chex.Array | None = None,
) -> RenderResult:
    """Differentiable bounded-table rendering with CuTe's explicit custom VJP."""
    if backend == "reference":
        from ..reference.rasterizer_jax import rasterize_jax

        return rasterize_jax(projected, table, camera, config, background)
    if backend == "cute":
        from ..kernels.rasterizer import rasterize_cute_vjp

        return rasterize_cute_vjp(projected, table, camera, config, background)
    raise ValueError(f"unknown backend: {backend}")
