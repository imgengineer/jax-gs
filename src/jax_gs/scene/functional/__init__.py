"""Public functional scene operations, loaded lazily."""

__all__ = ["pack_gaussian_inference_scene"]


def __getattr__(name: str):
    if name == "pack_gaussian_inference_scene":
        from .gaussian_inference import pack_gaussian_inference_scene

        globals()[name] = pack_gaussian_inference_scene
        return pack_gaussian_inference_scene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
