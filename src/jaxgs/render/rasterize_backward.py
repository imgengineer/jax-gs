import chex

from ..config import CapacityConfig
from ..reference.rasterizer_jax import RenderResult, rasterize_jax
from ..scene.camera import Camera
from .projection import ProjectedGaussians
from .visibility_table import VisibilityTable


def rasterize(
    projected: ProjectedGaussians,
    table: VisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    *,
    backend: str = "cute",
    background: chex.Array | None = None,
) -> RenderResult:
    """Differentiable rasterization; CuTe uses its explicit custom VJP."""
    if backend == "reference":
        return rasterize_jax(projected, table, camera, config, background)
    if backend == "cute":
        from ..kernels.rasterizer import rasterize_cute_vjp

        return rasterize_cute_vjp(projected, table, camera, config, background)
    raise ValueError(f"unknown backend: {backend}")
