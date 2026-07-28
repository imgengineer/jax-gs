"""Opt-in value validation for spinning-LiDAR tables."""

from __future__ import annotations

import numpy as np

from .types import RowOffsetStructuredSpinningLidarProjection


def _check_finite(value, name: str) -> None:
    array = np.asarray(value)
    invalid = np.flatnonzero(~np.isfinite(array))
    if invalid.size:
        index = int(invalid[0])
        raise ValueError(f"{name}[{index}] must be finite; got {array[index]!r}")


def validate_lidar_projection(
    projection: RowOffsetStructuredSpinningLidarProjection,
) -> None:
    if not isinstance(projection, RowOffsetStructuredSpinningLidarProjection):
        raise TypeError(
            f"Unknown LiDAR projection class: {type(projection).__name__}"
        )
    _check_finite(projection.row_elevations_rad, "row_elevations_rad")
    _check_finite(projection.column_azimuths_rad, "column_azimuths_rad")
    if projection.has_row_offsets:
        _check_finite(projection.row_azimuth_offsets_rad, "row_azimuth_offsets_rad")


__all__ = ["validate_lidar_projection"]
