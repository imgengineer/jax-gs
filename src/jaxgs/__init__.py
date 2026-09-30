"""Fixed-capacity Gaussian splatting in JAX."""

from .config import CapacityConfig
from .scene.camera import Camera
from .scene.point import (
    GaussianArrays,
    GaussianModel,
    create_gaussians,
    estimate_initial_scales,
    seed_gaussians,
)

__all__ = [
    "Camera",
    "CapacityConfig",
    "GaussianModel",
    "GaussianArrays",
    "create_gaussians",
    "estimate_initial_scales",
    "seed_gaussians",
]
