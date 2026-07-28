"""Pure-JAX structured spinning-LiDAR parameter types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import jax
import jax.numpy as jnp


class SpinningDirection(IntEnum):
    CLOCKWISE = 0
    COUNTERCLOCKWISE = 1


def _vector(value, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating-point dtype")
    return array


@dataclass(frozen=True)
class RowOffsetStructuredSpinningLidarProjection:
    """Per-row elevation and per-column azimuth lookup tables."""

    row_elevations_rad: jax.Array
    column_azimuths_rad: jax.Array
    row_azimuth_offsets_rad: jax.Array
    fov_vert_start_rad: float
    fov_vert_span_rad: float
    fov_horiz_start_rad: float
    fov_horiz_span_rad: float
    spinning_direction: int
    has_row_offsets: bool

    def __post_init__(self) -> None:
        row_elevations = _vector(self.row_elevations_rad, "row_elevations_rad")
        column_azimuths = _vector(
            self.column_azimuths_rad, "column_azimuths_rad"
        )
        row_offsets = _vector(
            self.row_azimuth_offsets_rad, "row_azimuth_offsets_rad"
        )
        if row_elevations.size == 0:
            raise ValueError("row_elevations_rad must be non-empty")
        if column_azimuths.size == 0:
            raise ValueError("column_azimuths_rad must be non-empty")
        has_offsets = bool(self.has_row_offsets)
        expected_offsets = row_elevations.shape if has_offsets else (0,)
        if row_offsets.shape != expected_offsets:
            raise ValueError(
                "row_azimuth_offsets_rad must match the row table when "
                "has_row_offsets is true and be empty otherwise"
            )
        scalar_names = (
            "fov_vert_start_rad",
            "fov_vert_span_rad",
            "fov_horiz_start_rad",
            "fov_horiz_span_rad",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        direction = int(self.spinning_direction)
        if direction not in (0, 1):
            raise ValueError(
                "spinning_direction must be 0 (CLOCKWISE) or 1 (COUNTERCLOCKWISE)"
            )
        object.__setattr__(self, "row_elevations_rad", row_elevations)
        object.__setattr__(self, "column_azimuths_rad", column_azimuths)
        object.__setattr__(self, "row_azimuth_offsets_rad", row_offsets)
        object.__setattr__(self, "spinning_direction", direction)
        object.__setattr__(self, "has_row_offsets", has_offsets)


REGISTERED_LIDAR_PROJECTIONS = (RowOffsetStructuredSpinningLidarProjection,)
REGISTERED_LIDAR_PROJECTION_NAMES = (
    "RowOffsetStructuredSpinningLidarProjection",
)


def script_class_name(obj: object) -> str:
    return type(obj).__name__


jax.tree_util.register_dataclass(
    RowOffsetStructuredSpinningLidarProjection,
    data_fields=(
        "row_elevations_rad",
        "column_azimuths_rad",
        "row_azimuth_offsets_rad",
    ),
    meta_fields=(
        "fov_vert_start_rad",
        "fov_vert_span_rad",
        "fov_horiz_start_rad",
        "fov_horiz_span_rad",
        "spinning_direction",
        "has_row_offsets",
    ),
)


__all__ = [
    "REGISTERED_LIDAR_PROJECTIONS",
    "REGISTERED_LIDAR_PROJECTION_NAMES",
    "RowOffsetStructuredSpinningLidarProjection",
    "SpinningDirection",
    "script_class_name",
]
