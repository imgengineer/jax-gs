"""Opt-in value validation for camera projection parameter objects."""

from __future__ import annotations

import numpy as np

from .types import (
    FThetaProjection,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
)


def _finite(array, name: str) -> np.ndarray:
    values = np.asarray(array)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def validate_camera_projection(
    projection: OpenCVPinholeProjection | FThetaProjection | OpenCVFisheyeProjection,
) -> None:
    """Raise ``ValueError`` for value-level invariants not checked by shape."""

    if isinstance(projection, OpenCVPinholeProjection):
        for name in (
            "focal_length",
            "principal_point",
            "radial_coeffs",
            "tangential_coeffs",
            "thin_prism_coeffs",
        ):
            _finite(getattr(projection, name), name)
        return
    if isinstance(projection, FThetaProjection):
        _finite(projection.principal_point, "principal_point")
        fw_poly = _finite(projection.fw_poly, "fw_poly")
        bw_poly = _finite(projection.bw_poly, "bw_poly")
        if fw_poly[0] != 0.0:
            raise ValueError("fw_poly[0] must be 0 (radial polynomial passes through origin)")
        if bw_poly[0] != 0.0:
            raise ValueError("bw_poly[0] must be 0 (radial polynomial passes through origin)")
        a = _finite(projection.A, "A")
        determinant = a[0] * a[3] - a[1] * a[2]
        if abs(float(determinant)) < 1.0e-12:
            raise ValueError("A must be non-singular (det != 0)")
        return
    if isinstance(projection, OpenCVFisheyeProjection):
        _finite(projection.principal_point, "principal_point")
        focal_length = _finite(projection.focal_length, "focal_length")
        if np.any(focal_length <= 0.0):
            raise ValueError("focal_length values must be > 0")
        _finite(projection.forward_poly, "forward_poly")
        _finite(projection.approx_backward_factor, "approx_backward_factor")
        return
    raise TypeError(f"Unknown camera projection class: {type(projection).__name__}")


__all__ = ["validate_camera_projection"]
