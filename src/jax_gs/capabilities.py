"""Feature availability queries matching gsplat's public build API.

Upstream uses these functions for compile-time CUDA feature flags. Pure JAX
has no equivalent build matrix, so each query reports whether this checkout
currently implements the corresponding public subsystem.
"""


def has_2dgs() -> bool:
    return True


def has_3dgs() -> bool:
    return True


def has_3dgut() -> bool:
    return True


def has_adam() -> bool:
    return True


def has_camera_wrappers() -> bool:
    return True


def has_losses() -> bool:
    return True


def has_reloc() -> bool:
    return True


__all__ = [
    "has_2dgs",
    "has_3dgs",
    "has_3dgut",
    "has_adam",
    "has_camera_wrappers",
    "has_losses",
    "has_reloc",
]
