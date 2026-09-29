import chex

from ..config import CapacityConfig
from ..reference.rasterizer_jax import RenderResult, rasterize_jax
from ..scene.camera import Camera
from .projection import ProjectedGaussians
from .visibility_table import VisibilityTable


def rasterize_forward(
    projected: ProjectedGaussians,
    table: VisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    *,
    backend: str = "cute",
    background: chex.Array | None = None,
) -> RenderResult:
    if backend == "reference":
        return rasterize_jax(projected, table, camera, config, background)
    if backend == "cute":
        from ..kernels.rasterizer import rasterize_cute

        return rasterize_cute(projected, table, camera, config, background)
    raise ValueError(f"unknown backend: {backend}")
