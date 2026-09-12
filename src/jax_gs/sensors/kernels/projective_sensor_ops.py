"""Cross-family dispatch for pure-JAX projective camera operations."""

from __future__ import annotations

import itertools
from collections.abc import Callable

from .cameras import ops as camera_ops
from .cameras.types import (
    REGISTERED_CAMERA_PROJECTIONS,
    REGISTERED_DISTORTIONS,
    script_class_name,
)
from .cameras.types import (
    BivariateWindshieldDistortion as BivariateWindshieldDistortion,  # noqa: PLC0414
)
from .cameras.types import (
    FThetaProjection as FThetaProjection,  # noqa: PLC0414
)
from .cameras.types import (
    NoExternalDistortion as NoExternalDistortion,  # noqa: PLC0414
)
from .cameras.types import (
    OpenCVFisheyeProjection as OpenCVFisheyeProjection,  # noqa: PLC0414
)
from .cameras.types import (
    OpenCVPinholeProjection as OpenCVPinholeProjection,  # noqa: PLC0414
)

DispatchKey = tuple[type, type]
_REGISTERED_PAIRS = tuple(
    itertools.product(REGISTERED_CAMERA_PROJECTIONS, REGISTERED_DISTORTIONS)
)


def _backend_table(backend: Callable) -> dict[DispatchKey, Callable]:
    return {pair: backend for pair in _REGISTERED_PAIRS}


_CAMERA_RAYS_TO_IMAGE_POINTS_BACKENDS = _backend_table(
    camera_ops.camera_rays_to_image_points
)
_IMAGE_POINTS_TO_CAMERA_RAYS_BACKENDS = _backend_table(
    camera_ops.image_points_to_camera_rays
)
_PROJECT_WORLD_POINTS_MEAN_POSE_BACKENDS = _backend_table(
    camera_ops.project_world_points_mean_pose
)
_PROJECT_WORLD_POINTS_SHUTTER_POSE_BACKENDS = _backend_table(
    camera_ops.project_world_points_shutter_pose
)
_IMAGE_POINTS_TO_WORLD_RAYS_STATIC_POSE_BACKENDS = _backend_table(
    camera_ops.image_points_to_world_rays_static_pose
)
_IMAGE_POINTS_TO_WORLD_RAYS_SHUTTER_POSE_BACKENDS = _backend_table(
    camera_ops.image_points_to_world_rays_shutter_pose
)
_PIXEL_GRID_TO_WORLD_RAYS_SHUTTER_POSE_BACKENDS = _backend_table(
    camera_ops.pixel_grid_to_world_rays_shutter_pose
)

_DISPATCH_TABLES = {
    "camera_rays_to_image_points": _CAMERA_RAYS_TO_IMAGE_POINTS_BACKENDS,
    "image_points_to_camera_rays": _IMAGE_POINTS_TO_CAMERA_RAYS_BACKENDS,
    "project_world_points_mean_pose": _PROJECT_WORLD_POINTS_MEAN_POSE_BACKENDS,
    "project_world_points_shutter_pose": _PROJECT_WORLD_POINTS_SHUTTER_POSE_BACKENDS,
    "image_points_to_world_rays_static_pose": (
        _IMAGE_POINTS_TO_WORLD_RAYS_STATIC_POSE_BACKENDS
    ),
    "image_points_to_world_rays_shutter_pose": (
        _IMAGE_POINTS_TO_WORLD_RAYS_SHUTTER_POSE_BACKENDS
    ),
    "pixel_grid_to_world_rays_shutter_pose": (
        _PIXEL_GRID_TO_WORLD_RAYS_SHUTTER_POSE_BACKENDS
    ),
}


def _lookup(
    table: dict[DispatchKey, Callable],
    projection: object,
    external_distortion: object,
) -> Callable:
    if external_distortion is None:
        raise TypeError(
            "external_distortion=None is not supported; pass "
            "NoExternalDistortion() explicitly"
        )
    names = (
        script_class_name(projection),
        script_class_name(external_distortion),
    )
    for (projection_type, distortion_type), backend in table.items():
        if names == (projection_type.__name__, distortion_type.__name__):
            return backend
    raise TypeError(
        f"Unsupported camera projection/distortion pair: ({names[0]}, {names[1]})"
    )


def camera_rays_to_image_points(camera_rays, projection, external_distortion, **kwargs):
    return _lookup(
        _CAMERA_RAYS_TO_IMAGE_POINTS_BACKENDS,
        projection,
        external_distortion,
    )(camera_rays, projection, external_distortion, **kwargs)


def image_points_to_camera_rays(
    image_points, projection, external_distortion, **kwargs
):
    return _lookup(
        _IMAGE_POINTS_TO_CAMERA_RAYS_BACKENDS,
        projection,
        external_distortion,
    )(image_points, projection, external_distortion, **kwargs)


def project_world_points_mean_pose(
    world_points, projection, external_distortion, *args, **kwargs
):
    return _lookup(
        _PROJECT_WORLD_POINTS_MEAN_POSE_BACKENDS,
        projection,
        external_distortion,
    )(world_points, projection, external_distortion, *args, **kwargs)


def project_world_points_shutter_pose(
    world_points, projection, external_distortion, *args, **kwargs
):
    return _lookup(
        _PROJECT_WORLD_POINTS_SHUTTER_POSE_BACKENDS,
        projection,
        external_distortion,
    )(world_points, projection, external_distortion, *args, **kwargs)


def image_points_to_world_rays_static_pose(
    image_points, projection, external_distortion, *args, **kwargs
):
    return _lookup(
        _IMAGE_POINTS_TO_WORLD_RAYS_STATIC_POSE_BACKENDS,
        projection,
        external_distortion,
    )(image_points, projection, external_distortion, *args, **kwargs)


def image_points_to_world_rays_shutter_pose(
    image_points, projection, external_distortion, *args, **kwargs
):
    return _lookup(
        _IMAGE_POINTS_TO_WORLD_RAYS_SHUTTER_POSE_BACKENDS,
        projection,
        external_distortion,
    )(image_points, projection, external_distortion, *args, **kwargs)


def pixel_grid_to_world_rays_shutter_pose(
    projection, external_distortion, *args, **kwargs
):
    return _lookup(
        _PIXEL_GRID_TO_WORLD_RAYS_SHUTTER_POSE_BACKENDS,
        projection,
        external_distortion,
    )(projection, external_distortion, *args, **kwargs)


__all__ = [  # noqa: RUF022 - preserve the public compatibility order
    "_DISPATCH_TABLES",
    "camera_rays_to_image_points",
    "image_points_to_camera_rays",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
    "pixel_grid_to_world_rays_shutter_pose",
    "project_world_points_mean_pose",
    "project_world_points_shutter_pose",
]
