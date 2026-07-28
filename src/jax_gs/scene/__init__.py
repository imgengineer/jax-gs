"""Scene containers and inference packing."""

from .components.base import Scene
from .components.gaussian_inference_scene import GaussianInferenceScene
from .components.gaussian_scene import GaussianScene
from .sh_compression import SHCompressionMode

__all__ = [
    "functional",
    "Scene",
    "GaussianScene",
    "GaussianInferenceScene",
    "SHCompressionMode",
]


def __getattr__(name: str):
    if name == "functional":
        from importlib import import_module

        module = import_module(f"{__name__}.functional")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
