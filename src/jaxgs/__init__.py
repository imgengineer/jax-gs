"""Fixed-capacity Gaussian splatting in JAX."""

from .config import CapacityConfig
from .scene.camera import Camera
from .scene.point import (
    GaussianModel,
    GaussianPool,
    create_pool,
    estimate_initial_scales,
    seed_pool,
)

__all__ = [
    "Camera",
    "CapacityConfig",
    "GaussianModel",
    "GaussianPool",
    "create_pool",
    "estimate_initial_scales",
    "seed_pool",
]
