"""Legacy root-level structured-LiDAR parameters and preprocessing.

Current gsplat ``main`` still exposes this API beside ``gsplat.sensors``.  The
legacy API stores angles as ``[azimuth, elevation]``; the newer sensors API
stores ``[elevation, azimuth]``.  Keeping the implementations separate makes
that otherwise subtle compatibility boundary explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import operator

import jax
import jax.numpy as jnp
import numpy as np


ANGLE_TO_PIXEL_SCALING_FACTOR = 1024.0


class SpinningDirection(Enum):
    CLOCKWISE = 0
    COUNTER_CLOCKWISE = 1


def relative_clock_rotation(
    angle_ref, angle, direction: SpinningDirection
) -> jax.Array:
    angle_ref = jnp.asarray(angle_ref)
    angle = jnp.asarray(angle)
    if direction == SpinningDirection.CLOCKWISE:
        return angle_ref - angle
    return angle - angle_ref


def normalize_angle(angle, *, start: float, scale: float = 1.0) -> jax.Array:
    angle = jnp.asarray(angle)
    period = jnp.asarray(2.0 * math.pi * scale, dtype=angle.dtype)
    return jnp.mod(angle - start, period) + start


def relative_angle(
    angle_ref,
    angle,
    direction: SpinningDirection,
    scale: float = 1.0,
) -> jax.Array:
    return normalize_angle(
        relative_clock_rotation(angle_ref, angle, direction),
        start=0.0,
        scale=scale,
    )


def angle_range_wrap_around(start_angle, end_angle, scale: float = 1.0) -> jax.Array:
    return jnp.abs(jnp.asarray(end_angle) - jnp.asarray(start_angle)) >= (
        2.0 * math.pi * scale
    )


def normalize_azimuth(angle, scale: float = 1.0) -> jax.Array:
    return normalize_angle(angle, start=0.0, scale=scale)


def normalize_elevation(angle, scale: float = 1.0) -> jax.Array:
    return jnp.clip(
        normalize_angle(angle, start=-scale * math.pi, scale=scale),
        -scale * math.pi / 2.0,
        scale * math.pi / 2.0,
    )


@dataclass(frozen=True, kw_only=True)
class FOV:
    """Angular field of view, starting at ``start`` along ``direction``."""

    start: float
    span: float
    direction: SpinningDirection

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.span):
            raise ValueError("FOV start and span must be finite")
        if self.span < 0.0:
            raise ValueError("FOV span must be nonnegative")
        if not isinstance(self.direction, SpinningDirection):
            raise TypeError("FOV direction must be a SpinningDirection")

    @property
    def end(self) -> float:
        if self.direction == SpinningDirection.COUNTER_CLOCKWISE:
            return self.start + self.span
        return self.start - self.span


@dataclass(frozen=True, kw_only=True)
class LidarModelParameters:
    fov_vert_rad: FOV
    fov_horiz_rad: FOV
    fov_eps_rad: float


@dataclass(frozen=True, kw_only=True)
class SpinningLidarModelParameters(LidarModelParameters):
    spinning_frequency_hz: float
    spinning_direction: SpinningDirection


@dataclass(frozen=True, kw_only=True)
class StructuredLidarModelParameters(LidarModelParameters):
    n_rows: int
    n_columns: int


@dataclass(frozen=True, kw_only=True)
class StructuredSpinningLidarModelParameters(
    StructuredLidarModelParameters, SpinningLidarModelParameters
):
    pass


def _float32_vector(value, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype != jnp.float32:
        raise TypeError(f"{name} must have dtype float32")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def _array_hash(array: jax.Array) -> int:
    host = np.ascontiguousarray(np.asarray(array))
    header = f"{host.shape}{host.dtype}".encode()
    digest = hashlib.sha256(header + host.tobytes()).digest()
    return int.from_bytes(digest[:8], "big")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, kw_only=True, eq=False)
class RowOffsetStructuredSpinningLidarModelParameters(
    StructuredSpinningLidarModelParameters
):
    """Legacy trainable angle tables for a structured spinning LiDAR."""

    row_elevations_rad: jax.Array
    column_azimuths_rad: jax.Array
    row_azimuth_offsets_rad: jax.Array

    def __init__(
        self,
        *,
        row_elevations_rad,
        column_azimuths_rad,
        row_azimuth_offsets_rad,
        spinning_frequency_hz: float,
        spinning_direction: SpinningDirection,
        fov_eps_factor: int = 4,
    ) -> None:
        rows = _float32_vector(row_elevations_rad, "row_elevations_rad")
        columns = _float32_vector(column_azimuths_rad, "column_azimuths_rad")
        offsets = _float32_vector(
            row_azimuth_offsets_rad, "row_azimuth_offsets_rad"
        )
        if offsets.shape != rows.shape:
            raise ValueError("row_azimuth_offsets_rad must match the row table")
        if not isinstance(spinning_direction, SpinningDirection):
            raise TypeError("spinning_direction must be a SpinningDirection")
        try:
            fov_eps_factor = operator.index(fov_eps_factor)
        except TypeError as exc:
            raise TypeError("fov_eps_factor must be an integer") from exc
        if fov_eps_factor <= 0:
            raise ValueError("fov_eps_factor must be positive")
        if not math.isfinite(spinning_frequency_hz):
            raise ValueError("spinning_frequency_hz must be finite")

        object.__setattr__(self, "row_elevations_rad", rows)
        object.__setattr__(self, "column_azimuths_rad", columns)
        object.__setattr__(self, "row_azimuth_offsets_rad", offsets)
        object.__setattr__(self, "n_rows", rows.shape[0])
        object.__setattr__(self, "n_columns", columns.shape[0])
        object.__setattr__(self, "spinning_frequency_hz", float(spinning_frequency_hz))
        object.__setattr__(self, "spinning_direction", spinning_direction)
        object.__setattr__(
            self,
            "fov_eps_rad",
            float(fov_eps_factor * np.finfo(np.float32).eps),
        )
        object.__setattr__(self, "fov_vert_rad", self._compute_fov_vert_rad())
        object.__setattr__(
            self,
            "fov_horiz_rad",
            self._compute_fov_horiz_rad(spinning_direction),
        )
        self._validate_ordering()

    def _validate_ordering(self) -> None:
        rows = np.asarray(self.row_elevations_rad)
        columns = np.asarray(self.column_azimuths_rad)
        row_relative = np.mod(rows[0] - rows, 2.0 * math.pi)
        if not np.all(np.diff(row_relative) > 0.0):
            raise ValueError("row elevations must be sorted in descending order")
        if np.any(np.abs(rows - rows[0]) >= 2.0 * math.pi):
            raise ValueError("row elevations must not wrap around the first row")
        if self.spinning_direction == SpinningDirection.CLOCKWISE:
            column_relative = np.mod(columns[0] - columns, 2.0 * math.pi)
        else:
            column_relative = np.mod(columns - columns[0], 2.0 * math.pi)
        if not np.all(np.diff(column_relative) > 0.0):
            raise ValueError(
                "column azimuths must be sorted in the spinning direction"
            )
        if np.any(np.abs(columns - columns[0]) >= 2.0 * math.pi):
            raise ValueError("column azimuths must not wrap around the first column")

    def _compute_fov_vert_rad(self) -> FOV:
        rows = np.asarray(self.row_elevations_rad)
        return FOV(
            start=float(rows[0]),
            span=float(np.mod(rows[0] - rows[-1], 2.0 * math.pi)),
            direction=SpinningDirection.CLOCKWISE,
        )

    def _compute_fov_horiz_rad(
        self, spinning_direction: SpinningDirection
    ) -> FOV:
        columns = np.asarray(self.column_azimuths_rad)
        offsets = np.asarray(self.row_azimuth_offsets_rad)
        extremes = columns[[0, -1]][None, :] + offsets[:, None]
        if spinning_direction == SpinningDirection.COUNTER_CLOCKWISE:
            start = float(np.min(extremes[:, 0]))
            relative_ends = np.mod(extremes[:, -1] - start, 2.0 * math.pi)
        else:
            start = float(np.max(extremes[:, 0]))
            relative_ends = np.mod(start - extremes[:, -1], 2.0 * math.pi)
        if np.any(np.abs(extremes[:, -1] - start) >= 2.0 * math.pi):
            span = 2.0 * math.pi
        else:
            span = float(np.max(relative_ends))
        return FOV(start=start, span=span, direction=spinning_direction)

    @property
    def device(self):
        return self.row_elevations_rad.device

    @property
    def dtype(self):
        return self.row_elevations_rad.dtype

    def __hash__(self) -> int:
        return hash(
            (
                self.fov_vert_rad,
                self.fov_horiz_rad,
                self.fov_eps_rad,
                self.n_rows,
                self.n_columns,
                self.spinning_frequency_hz,
                self.spinning_direction,
                _array_hash(self.row_elevations_rad),
                _array_hash(self.column_azimuths_rad),
                _array_hash(self.row_azimuth_offsets_rad),
            )
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, RowOffsetStructuredSpinningLidarModelParameters):
            return NotImplemented
        scalar_equal = (
            self.fov_vert_rad == other.fov_vert_rad
            and self.fov_horiz_rad == other.fov_horiz_rad
            and self.fov_eps_rad == other.fov_eps_rad
            and self.spinning_frequency_hz == other.spinning_frequency_hz
            and self.spinning_direction == other.spinning_direction
        )
        return scalar_equal and all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in (
                (self.row_elevations_rad, other.row_elevations_rad),
                (self.column_azimuths_rad, other.column_azimuths_rad),
                (self.row_azimuth_offsets_rad, other.row_azimuth_offsets_rad),
            )
        )

    def create_elements(self) -> jax.Array:
        """Return row-major legacy elements as ``[azimuth, elevation]``."""

        flat = jnp.arange(self.n_rows * self.n_columns, dtype=jnp.int32)
        return jnp.stack((flat % self.n_columns, flat // self.n_columns), axis=-1)

    def elements_to_sensor_angles(self, elements) -> jax.Array:
        """Map legacy ``[azimuth, elevation]`` indices to angles in that order."""

        elements = jnp.asarray(elements, dtype=jnp.int32)
        if elements.shape[-1] != 2:
            raise ValueError("elements must end in two indices")
        azimuth_index = elements[..., 0]
        elevation_index = elements[..., 1]
        azimuth = normalize_azimuth(
            self.column_azimuths_rad[azimuth_index]
            + self.row_azimuth_offsets_rad[elevation_index]
        )
        elevation = normalize_elevation(
            self.row_elevations_rad[elevation_index]
        )
        return jnp.stack((azimuth, elevation), axis=-1)

    def tree_flatten(self):
        children = (
            self.row_elevations_rad,
            self.column_azimuths_rad,
            self.row_azimuth_offsets_rad,
        )
        auxiliary = (
            self.fov_vert_rad,
            self.fov_horiz_rad,
            self.fov_eps_rad,
            self.n_rows,
            self.n_columns,
            self.spinning_frequency_hz,
            self.spinning_direction,
        )
        return children, auxiliary

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        obj = object.__new__(cls)
        names = (
            "fov_vert_rad",
            "fov_horiz_rad",
            "fov_eps_rad",
            "n_rows",
            "n_columns",
            "spinning_frequency_hz",
            "spinning_direction",
        )
        for name, value in zip(names, auxiliary, strict=True):
            object.__setattr__(obj, name, value)
        for name, value in zip(
            (
                "row_elevations_rad",
                "column_azimuths_rad",
                "row_azimuth_offsets_rad",
            ),
            children,
            strict=True,
        ):
            object.__setattr__(obj, name, value)
        return obj


@jax.tree_util.register_pytree_node_class
@dataclass
class LidarTiling:
    """Precomputed mapping between angular tiles and structured rays."""

    n_bins_azimuth: int
    n_bins_elevation: int
    cdf_elevation: jax.Array
    cdf_dense_ray_mask: jax.Array
    tiles_pack_info: jax.Array
    tiles_to_elements_map: jax.Array
    _max_elements_per_tile: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.n_bins_azimuth = operator.index(self.n_bins_azimuth)
        self.n_bins_elevation = operator.index(self.n_bins_elevation)
        if self.n_bins_azimuth < 1 or self.n_bins_elevation < 1:
            raise ValueError("LiDAR tile counts must be positive")
        self.cdf_elevation = jnp.asarray(self.cdf_elevation, dtype=jnp.int32)
        self.cdf_dense_ray_mask = jnp.asarray(
            self.cdf_dense_ray_mask, dtype=jnp.int32
        )
        self.tiles_pack_info = jnp.asarray(self.tiles_pack_info, dtype=jnp.int32)
        self.tiles_to_elements_map = jnp.asarray(
            self.tiles_to_elements_map, dtype=jnp.int32
        )
        if self.cdf_elevation.ndim != 1:
            raise ValueError("cdf_elevation must be one-dimensional")
        if int(np.asarray(self.cdf_elevation[-1])) != self.n_bins_elevation:
            raise ValueError("cdf_elevation[-1] must equal n_bins_elevation")
        if self.cdf_dense_ray_mask.ndim != 2:
            raise ValueError("cdf_dense_ray_mask must be two-dimensional")
        if self.cdf_dense_ray_mask.shape[0] != self.cdf_elevation.shape[0]:
            raise ValueError("dense ray mask and elevation CDF shapes disagree")
        tile_count = self.n_bins_azimuth * self.n_bins_elevation
        if self.tiles_pack_info.shape != (tile_count, 2):
            raise ValueError("tiles_pack_info must have shape [tile_count, 2]")
        if self.tiles_to_elements_map.ndim != 2 or self.tiles_to_elements_map.shape[1] != 2:
            raise ValueError("tiles_to_elements_map must have shape [N, 2]")
        self._max_elements_per_tile = int(
            np.max(np.asarray(self.tiles_pack_info[:, 1]), initial=0)
        )

    @property
    def cdf_resolution_elevation(self) -> int:
        return self.cdf_dense_ray_mask.shape[0] - 1

    @property
    def cdf_resolution_azimuth(self) -> int:
        return self.cdf_dense_ray_mask.shape[1] - 1

    @property
    def max_elements_per_tile(self) -> int:
        return self._max_elements_per_tile

    def tree_flatten(self):
        return (
            (
                self.cdf_elevation,
                self.cdf_dense_ray_mask,
                self.tiles_pack_info,
                self.tiles_to_elements_map,
            ),
            (
                self.n_bins_azimuth,
                self.n_bins_elevation,
                self._max_elements_per_tile,
            ),
        )

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        obj = object.__new__(cls)
        (
            obj.n_bins_azimuth,
            obj.n_bins_elevation,
            obj._max_elements_per_tile,
        ) = auxiliary
        (
            obj.cdf_elevation,
            obj.cdf_dense_ray_mask,
            obj.tiles_pack_info,
            obj.tiles_to_elements_map,
        ) = children
        return obj


@jax.tree_util.register_pytree_node_class
class RowOffsetStructuredSpinningLidarModelParametersExt(
    RowOffsetStructuredSpinningLidarModelParameters
):
    """Legacy parameters plus angle lookup and tile acceleration tables."""

    def __init__(
        self,
        params: RowOffsetStructuredSpinningLidarModelParameters,
        angles_to_columns_map,
        tiling: LidarTiling,
    ) -> None:
        if not isinstance(params, RowOffsetStructuredSpinningLidarModelParameters):
            raise TypeError("params must be structured spinning-LiDAR parameters")
        if not isinstance(tiling, LidarTiling):
            raise TypeError("tiling must be a LidarTiling")
        for name in (
            "fov_vert_rad",
            "fov_horiz_rad",
            "fov_eps_rad",
            "n_rows",
            "n_columns",
            "spinning_frequency_hz",
            "spinning_direction",
            "row_elevations_rad",
            "column_azimuths_rad",
            "row_azimuth_offsets_rad",
        ):
            object.__setattr__(self, name, getattr(params, name))
        angle_map = jnp.asarray(angles_to_columns_map)
        if angle_map.ndim != 2 or not jnp.issubdtype(angle_map.dtype, jnp.integer):
            raise TypeError("angles_to_columns_map must be a 2-D integer array")
        if angle_map.shape[0] % self.n_rows or angle_map.shape[1] % self.n_columns:
            raise ValueError("angles_to_columns_map resolution must scale both tables")
        if angle_map.shape[0] // self.n_rows != angle_map.shape[1] // self.n_columns:
            raise ValueError("angles_to_columns_map must use one resolution factor")
        object.__setattr__(self, "angles_to_columns_map", angle_map)
        object.__setattr__(self, "tiling", tiling)

    def tree_flatten(self):
        base_children, auxiliary = super().tree_flatten()
        return base_children + (self.angles_to_columns_map, self.tiling), auxiliary

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        base = RowOffsetStructuredSpinningLidarModelParameters.tree_unflatten(
            auxiliary, children[:3]
        )
        obj = object.__new__(cls)
        for name in (
            "fov_vert_rad",
            "fov_horiz_rad",
            "fov_eps_rad",
            "n_rows",
            "n_columns",
            "spinning_frequency_hz",
            "spinning_direction",
            "row_elevations_rad",
            "column_azimuths_rad",
            "row_azimuth_offsets_rad",
        ):
            object.__setattr__(obj, name, getattr(base, name))
        object.__setattr__(obj, "angles_to_columns_map", children[3])
        object.__setattr__(obj, "tiling", children[4])
        return obj


def relative_sensor_angles(
    lidar: SpinningLidarModelParameters, coord, scale: float = 1.0
) -> jax.Array:
    """Return legacy relative angles in ``[azimuth, elevation]`` order."""

    coord = jnp.asarray(coord)
    azimuth = relative_angle(
        lidar.fov_horiz_rad.start * scale,
        coord[..., 0],
        lidar.spinning_direction,
        scale,
    )
    elevation = relative_clock_rotation(
        lidar.fov_vert_rad.start * scale,
        coord[..., 1],
        SpinningDirection.CLOCKWISE,
    )
    return jnp.stack((azimuth, elevation), axis=-1)


def valid_sensor_angles(
    lidar: SpinningLidarModelParameters, coord, scale: float = 1.0
) -> jax.Array:
    coord = jnp.asarray(coord)
    vertical_start = lidar.fov_vert_rad.start + lidar.fov_eps_rad
    horizontal_start = (
        lidar.fov_horiz_rad.start + lidar.fov_eps_rad
        if lidar.spinning_direction == SpinningDirection.CLOCKWISE
        else lidar.fov_horiz_rad.start - lidar.fov_eps_rad
    )
    relative_elevation = relative_clock_rotation(
        vertical_start * scale,
        coord[..., 1],
        SpinningDirection.CLOCKWISE,
    )
    relative_azimuth = relative_angle(
        horizontal_start * scale,
        coord[..., 0],
        lidar.spinning_direction,
        scale,
    )
    return (
        relative_elevation
        <= (lidar.fov_vert_rad.span + 2.0 * lidar.fov_eps_rad) * scale
    ) & (
        relative_azimuth
        <= (lidar.fov_horiz_rad.span + 2.0 * lidar.fov_eps_rad) * scale
    )


@dataclass(frozen=True)
class SensorRayReturn:
    sensor_rays: jax.Array
    valid_flag: jax.Array


jax.tree_util.register_dataclass(
    SensorRayReturn,
    data_fields=("sensor_rays", "valid_flag"),
    meta_fields=(),
)


def sensor_angles_to_rays(
    lidar: SpinningLidarModelParameters, sensor_angles
) -> SensorRayReturn:
    """Convert legacy ``[azimuth, elevation]`` angles to unit rays."""

    angles = jnp.asarray(sensor_angles, dtype=lidar.dtype)
    if angles.shape[-1] != 2:
        raise ValueError("sensor_angles must end in two coordinates")
    azimuth, elevation = angles[..., 0], angles[..., 1]
    cosine = jnp.cos(elevation)
    rays = jnp.stack(
        (
            jnp.cos(azimuth) * cosine,
            jnp.sin(azimuth) * cosine,
            jnp.sin(elevation),
        ),
        axis=-1,
    )
    return SensorRayReturn(rays, valid_sensor_angles(lidar, angles))


def compute_angles_to_columns_map(
    lidar: RowOffsetStructuredSpinningLidarModelParameters,
    resolution_factor: int = 4,
    dtype=jnp.int32,
) -> jax.Array:
    """Precompute nearest structured columns on a regular angular grid."""

    resolution_factor = operator.index(resolution_factor)
    if resolution_factor < 1:
        raise ValueError("resolution_factor must be positive")
    dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(dtype, jnp.integer):
        raise TypeError("angles-to-columns dtype must be integral")
    if jnp.iinfo(dtype).max < lidar.n_columns - 1:
        raise ValueError("dtype cannot store the maximum column index")

    elevations, azimuths = jnp.meshgrid(
        jnp.linspace(
            lidar.fov_vert_rad.start,
            lidar.fov_vert_rad.end,
            resolution_factor * lidar.n_rows,
            dtype=lidar.dtype,
        ),
        jnp.linspace(
            lidar.fov_horiz_rad.start,
            lidar.fov_horiz_rad.end,
            resolution_factor * lidar.n_columns,
            dtype=lidar.dtype,
        ),
        indexing="ij",
    )
    grid_angles = jnp.stack((azimuths, elevations), axis=-1)
    grid_rays = sensor_angles_to_rays(lidar, grid_angles).sensor_rays
    elements = lidar.create_elements()
    element_rays = sensor_angles_to_rays(
        lidar, lidar.elements_to_sensor_angles(elements)
    ).sensor_rays

    # This table is deterministic preprocessing, not a differentiable runtime
    # op.  A host KD-tree avoids materializing an enormous all-pairs matrix for
    # production sensors while the returned state remains a JAX array.
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(element_rays))
    _, nearest = tree.query(np.asarray(grid_rays).reshape((-1, 3)))
    columns = (nearest % lidar.n_columns).astype(np.dtype(dtype))
    return jnp.asarray(columns.reshape(grid_angles.shape[:-1]), dtype=dtype)


def angles_to_dense_ray_mask_cdf(
    parameters: RowOffsetStructuredSpinningLidarModelParameters,
    angles,
    *,
    resolution_elevation: int,
    resolution_azimuth: int,
) -> jax.Array:
    resolution_elevation = operator.index(resolution_elevation)
    resolution_azimuth = operator.index(resolution_azimuth)
    if resolution_elevation < 1 or resolution_azimuth < 1:
        raise ValueError("dense mask resolutions must be positive")
    relative = relative_sensor_angles(parameters, angles)
    normalized_azimuth = relative[..., 0] / parameters.fov_horiz_rad.span
    normalized_elevation = relative[..., 1] / parameters.fov_vert_rad.span
    azimuth_indices = (
        (normalized_azimuth * resolution_azimuth).astype(jnp.int32)
        % resolution_azimuth
    )
    elevation_indices = (
        (normalized_elevation * resolution_elevation).astype(jnp.int32)
        % resolution_elevation
    )
    flat_indices = azimuth_indices + elevation_indices * resolution_azimuth
    mask = jnp.zeros(
        (resolution_elevation * resolution_azimuth,), dtype=jnp.int32
    ).at[flat_indices.reshape(-1)].set(1)
    padded = jnp.pad(mask.reshape((resolution_elevation, resolution_azimuth)), ((1, 0), (1, 0)))
    return jnp.cumsum(jnp.cumsum(padded, axis=0), axis=1, dtype=jnp.int32)


def angles_to_tile_indices(
    parameters: RowOffsetStructuredSpinningLidarModelParameters,
    angles,
    *,
    n_bins_azimuth: int,
    n_bins_elevation: int,
    cdf_elevation,
) -> jax.Array:
    del n_bins_elevation
    cdf = jnp.asarray(cdf_elevation)
    resolution = cdf.shape[0] - 1
    relative = relative_sensor_angles(parameters, angles)
    normalized_azimuth = (
        relative[..., 0] / parameters.fov_horiz_rad.span * n_bins_azimuth
    )
    normalized_elevation = (
        relative[..., 1] / parameters.fov_vert_rad.span * resolution
    )
    azimuth_index = normalized_azimuth.astype(jnp.int32) % n_bins_azimuth
    dense_elevation = jnp.clip(
        normalized_elevation, 0, resolution - 1
    ).astype(jnp.int32)
    elevation_index = cdf[dense_elevation].astype(jnp.int32)
    return azimuth_index + elevation_index * n_bins_azimuth


def compute_tiles_to_elements_map(
    parameters: RowOffsetStructuredSpinningLidarModelParameters,
    *,
    n_bins_azimuth: int,
    densification_factor_azimuth: int,
    cdf_elevation,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_bins_azimuth = operator.index(n_bins_azimuth)
    densification_factor_azimuth = operator.index(
        densification_factor_azimuth
    )
    cdf = jnp.asarray(cdf_elevation)
    n_bins_elevation = int(np.asarray(cdf[-1]))
    elements = parameters.create_elements()
    angles = parameters.elements_to_sensor_angles(elements)
    tile_indices = angles_to_tile_indices(
        parameters,
        angles,
        n_bins_azimuth=n_bins_azimuth,
        n_bins_elevation=n_bins_elevation,
        cdf_elevation=cdf,
    )
    tile_count = n_bins_azimuth * n_bins_elevation
    counts = jnp.bincount(tile_indices, length=tile_count).astype(jnp.int32)
    starts = jnp.cumsum(counts, dtype=jnp.int32) - counts
    pack_info = jnp.stack((starts, counts), axis=-1)
    sorted_elements = elements[jnp.argsort(tile_indices, stable=True)]
    dense_mask = angles_to_dense_ray_mask_cdf(
        parameters,
        angles,
        resolution_elevation=cdf.shape[0] - 1,
        resolution_azimuth=n_bins_azimuth * densification_factor_azimuth,
    )
    return sorted_elements, pack_info, dense_mask


def compute_histogram_equalization(
    parameters: RowOffsetStructuredSpinningLidarModelParameters,
    *,
    n_bins_elevation: int,
    max_pts_per_tile: int,
    resolution_elevation: int,
) -> tuple[int, jax.Array]:
    n_bins_elevation = operator.index(n_bins_elevation)
    max_pts_per_tile = operator.index(max_pts_per_tile)
    resolution_elevation = operator.index(resolution_elevation)
    if min(n_bins_elevation, max_pts_per_tile, resolution_elevation) < 1:
        raise ValueError("histogram parameters must be positive")
    angles = parameters.elements_to_sensor_angles(
        parameters.create_elements()
    ).reshape((parameters.n_rows, parameters.n_columns, 2))
    relative = np.asarray(relative_sensor_angles(parameters, angles))
    azimuth = relative[..., 0]
    elevation = relative[..., 1]
    epsilon = 2.0 * np.finfo(np.float32).eps
    elevation_range = (-epsilon, parameters.fov_vert_rad.span + epsilon)
    azimuth_range = (-epsilon, parameters.fov_horiz_rad.span + epsilon)
    if not np.all(
        (elevation >= elevation_range[0]) & (elevation <= elevation_range[1])
    ):
        raise ValueError("element elevations fall outside the computed FOV")
    if not np.all(
        (azimuth >= azimuth_range[0]) & (azimuth <= azimuth_range[1])
    ):
        raise ValueError("element azimuths fall outside the computed FOV")

    histogram, _ = np.histogram(
        elevation, bins=resolution_elevation, range=elevation_range
    )
    cumulative = np.zeros((resolution_elevation + 1,), dtype=np.float64)
    cumulative[1:] = np.cumsum(histogram)
    cumulative = cumulative / cumulative[-1] * n_bins_elevation
    edges = [0]
    next_bin = 1
    for index, value in enumerate(cumulative):
        if value >= next_bin:
            edges.append(index)
            next_bin += 1
    if len(edges) != n_bins_elevation + 1:
        raise ValueError("elevation histogram cannot form the requested bins")
    edges[-1] = resolution_elevation
    elevation_edges = (
        np.asarray(edges, dtype=np.float64)
        / resolution_elevation
        * parameters.fov_vert_rad.span
    )
    elevation_histogram, _ = np.histogram(elevation, bins=elevation_edges)
    n_bins_azimuth = max(
        1,
        int(math.ceil(float(np.mean(elevation_histogram)) / max_pts_per_tile)),
    )
    while True:
        histogram_2d, _, _ = np.histogram2d(
            azimuth.reshape(-1),
            elevation.reshape(-1),
            bins=(n_bins_azimuth, elevation_edges),
            range=(azimuth_range, elevation_range),
        )
        if histogram_2d.max(initial=0) <= max_pts_per_tile:
            break
        n_bins_azimuth += 1
    return n_bins_azimuth, jnp.asarray(cumulative, dtype=jnp.float32)


def compute_tiling(
    lidar_params: RowOffsetStructuredSpinningLidarModelParameters,
    n_bins_elevation: int = 16,
    max_pts_per_tile: int = 16 * 16,
    resolution_elevation: int = 1600,
    densification_factor_azimuth: int = 8,
) -> LidarTiling:
    n_bins_azimuth, cdf = compute_histogram_equalization(
        lidar_params,
        n_bins_elevation=n_bins_elevation,
        max_pts_per_tile=max_pts_per_tile,
        resolution_elevation=resolution_elevation,
    )
    elements, pack_info, dense_mask = compute_tiles_to_elements_map(
        lidar_params,
        n_bins_azimuth=n_bins_azimuth,
        densification_factor_azimuth=densification_factor_azimuth,
        cdf_elevation=cdf,
    )
    return LidarTiling(
        n_bins_azimuth=n_bins_azimuth,
        n_bins_elevation=n_bins_elevation,
        cdf_elevation=cdf.astype(jnp.int32),
        cdf_dense_ray_mask=dense_mask,
        tiles_pack_info=pack_info,
        tiles_to_elements_map=elements,
    )


class LegacyLidarModel:
    """Readable root-level projection adapter used by 3DGUT rendering."""

    def __init__(
        self, params: RowOffsetStructuredSpinningLidarModelParametersExt
    ) -> None:
        if not isinstance(
            params, RowOffsetStructuredSpinningLidarModelParametersExt
        ):
            raise TypeError("legacy LiDAR model requires extended parameters")
        self.params = params
        self.width = params.n_columns
        self.height = params.n_rows
        # These FOV values were derived and validated when the parameter object
        # was built.  Reusing the static metadata keeps model construction safe
        # while ``params`` contains JAX tracers under ``jax.jit``.
        self.fov_vert_rad = params.fov_vert_rad
        self.fov_horiz_rad = params.fov_horiz_rad

    def __getattr__(self, name):
        return getattr(self.params, name)

    def relative_sensor_angles(self, angles, scale: float = 1.0) -> jax.Array:
        return relative_sensor_angles(self, angles, scale)

    def valid_sensor_angles(self, angles, scale: float = 1.0) -> jax.Array:
        return valid_sensor_angles(self, angles, scale)

    def camera_ray_to_image_point(
        self, camera_ray, margin_factor: float = 0.0
    ) -> tuple[jax.Array, jax.Array]:
        ray = jnp.asarray(camera_ray)
        squared_norm = jnp.sum(ray * ray, axis=-1, keepdims=True)
        ray = ray * jax.lax.rsqrt(jnp.maximum(squared_norm, 1.0e-20))
        azimuth = jnp.arctan2(ray[..., 1], ray[..., 0])
        elevation = jnp.arcsin(jnp.clip(ray[..., 2], -1.0, 1.0))
        angles = jnp.stack((azimuth, elevation), axis=-1)
        relative = self.relative_sensor_angles(angles)
        margin = jnp.asarray(
            (margin_factor * self.fov_horiz_rad.span, margin_factor * self.fov_vert_rad.span),
            dtype=ray.dtype,
        )
        spans = jnp.asarray(
            (self.fov_horiz_rad.span, self.fov_vert_rad.span), dtype=ray.dtype
        )
        valid = jnp.all((relative >= -margin) & (relative <= spans + margin), axis=-1)
        image_point = angles * ANGLE_TO_PIXEL_SCALING_FACTOR
        valid = valid & (squared_norm[..., 0] >= 1.0e-12)
        valid = valid & jnp.all(jnp.isfinite(ray), axis=-1)
        return image_point, valid & jnp.all(jnp.isfinite(image_point), axis=-1)

    def element_to_image_point(self, row, column) -> jax.Array:
        elevation = self.row_elevations_rad[jnp.asarray(row, dtype=jnp.int32)]
        azimuth = (
            self.column_azimuths_rad[jnp.asarray(column, dtype=jnp.int32)]
            + self.row_azimuth_offsets_rad[jnp.asarray(row, dtype=jnp.int32)]
        )
        azimuth = jnp.where(azimuth > math.pi, azimuth - 2.0 * math.pi, azimuth)
        azimuth = jnp.where(azimuth <= -math.pi, azimuth + 2.0 * math.pi, azimuth)
        return jnp.stack((azimuth, elevation), axis=-1) * ANGLE_TO_PIXEL_SCALING_FACTOR

    def image_point_to_camera_ray(self, image_point) -> SensorRayReturn:
        angles = jnp.asarray(image_point) / ANGLE_TO_PIXEL_SCALING_FACTOR
        return sensor_angles_to_rays(self, angles)

    def shutter_relative_frame_time(self, image_point) -> jax.Array:
        angles = jnp.asarray(image_point) / ANGLE_TO_PIXEL_SCALING_FACTOR
        if self.n_columns <= 1:
            return jnp.zeros(angles.shape[:-1], dtype=angles.dtype)
        vertical_resolution = self.fov_vert_rad.span / (
            self.angles_to_columns_map.shape[0] - 1
        )
        horizontal_resolution = self.fov_horiz_rad.span / (
            self.angles_to_columns_map.shape[1] - 1
        )
        relative = self.relative_sensor_angles(angles)
        vertical = jnp.clip(
            relative[..., 1] / vertical_resolution + 0.5,
            0,
            self.angles_to_columns_map.shape[0] - 1,
        ).astype(jnp.int32)
        horizontal = jnp.clip(
            relative[..., 0] / horizontal_resolution + 0.5,
            0,
            self.angles_to_columns_map.shape[1] - 1,
        ).astype(jnp.int32)
        column = self.angles_to_columns_map[vertical, horizontal]
        return column.astype(angles.dtype) / (self.n_columns - 1)


def generate_lidar_image_points(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
) -> jax.Array:
    model = LegacyLidarModel(lidar)
    rows, columns = jnp.meshgrid(
        jnp.arange(lidar.n_rows, dtype=jnp.int32),
        jnp.arange(lidar.n_columns, dtype=jnp.int32),
        indexing="ij",
    )
    return model.element_to_image_point(rows, columns)


__all__ = [
    "ANGLE_TO_PIXEL_SCALING_FACTOR",
    "FOV",
    "LegacyLidarModel",
    "LidarModelParameters",
    "LidarTiling",
    "RowOffsetStructuredSpinningLidarModelParameters",
    "RowOffsetStructuredSpinningLidarModelParametersExt",
    "SensorRayReturn",
    "SpinningDirection",
    "SpinningLidarModelParameters",
    "StructuredLidarModelParameters",
    "StructuredSpinningLidarModelParameters",
    "angle_range_wrap_around",
    "angles_to_dense_ray_mask_cdf",
    "angles_to_tile_indices",
    "compute_angles_to_columns_map",
    "compute_histogram_equalization",
    "compute_tiles_to_elements_map",
    "compute_tiling",
    "generate_lidar_image_points",
    "normalize_angle",
    "normalize_azimuth",
    "normalize_elevation",
    "relative_angle",
    "relative_clock_rotation",
    "relative_sensor_angles",
    "sensor_angles_to_rays",
    "valid_sensor_angles",
]
