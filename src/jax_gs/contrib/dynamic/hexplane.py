"""Multi-resolution HexPlane spatio-temporal feature field in Flax NNX."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

_DEFAULT_PLANE_CONFIG = {
    "grid_dimensions": 2,
    "input_coordinate_dim": 4,
    "output_coordinate_dim": 32,
    "resolution": [64, 64, 64, 25],
}
_DEFAULT_MULTIRES = (1, 2)


def _normalize_aabb(points: jax.Array, aabb: jax.Array) -> jax.Array:
    """Linearly map spatial points to ``[-1, 1]`` using upstream ordering."""

    return (points - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def _normalized_index(
    coordinate: jax.Array,
    size: int,
    *,
    align_corners: bool,
) -> jax.Array:
    if align_corners:
        index = (coordinate + 1.0) * ((size - 1) / 2.0)
    else:
        index = ((coordinate + 1.0) * size - 1.0) / 2.0
    return jnp.clip(index, 0.0, float(size - 1))


def _sample_2d(
    grid: jax.Array,
    coordinates: jax.Array,
    *,
    align_corners: bool,
) -> jax.Array:
    """Border-padded bilinear sampling for one ``(C, H, W)`` grid."""

    height, width = grid.shape[-2:]
    x = _normalized_index(coordinates[:, 0], width, align_corners=align_corners)
    y = _normalized_index(coordinates[:, 1], height, align_corners=align_corners)
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, width - 1)
    y1 = jnp.minimum(y0 + 1, height - 1)
    wx = (x - x0).astype(grid.dtype)[:, None]
    wy = (y - y0).astype(grid.dtype)[:, None]

    def gather(row: jax.Array, column: jax.Array) -> jax.Array:
        return jnp.moveaxis(grid[:, row, column], 0, -1)

    top = gather(y0, x0) * (1.0 - wx) + gather(y0, x1) * wx
    bottom = gather(y1, x0) * (1.0 - wx) + gather(y1, x1) * wx
    return top * (1.0 - wy) + bottom * wy


def _sample_3d(
    grid: jax.Array,
    coordinates: jax.Array,
    *,
    align_corners: bool,
) -> jax.Array:
    """Border-padded trilinear sampling for one ``(C, D, H, W)`` grid."""

    depth, height, width = grid.shape[-3:]
    x = _normalized_index(coordinates[:, 0], width, align_corners=align_corners)
    y = _normalized_index(coordinates[:, 1], height, align_corners=align_corners)
    z = _normalized_index(coordinates[:, 2], depth, align_corners=align_corners)
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    z0 = jnp.floor(z).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, width - 1)
    y1 = jnp.minimum(y0 + 1, height - 1)
    z1 = jnp.minimum(z0 + 1, depth - 1)
    wx = (x - x0).astype(grid.dtype)[:, None]
    wy = (y - y0).astype(grid.dtype)[:, None]
    wz = (z - z0).astype(grid.dtype)[:, None]

    def gather(
        layer: jax.Array,
        row: jax.Array,
        column: jax.Array,
    ) -> jax.Array:
        return jnp.moveaxis(grid[:, layer, row, column], 0, -1)

    def sample_layer(layer: jax.Array) -> jax.Array:
        top = gather(layer, y0, x0) * (1.0 - wx) + gather(layer, y0, x1) * wx
        bottom = gather(layer, y1, x0) * (1.0 - wx) + gather(layer, y1, x1) * wx
        return top * (1.0 - wy) + bottom * wy

    return sample_layer(z0) * (1.0 - wz) + sample_layer(z1) * wz


def _grid_sample_wrapper(
    grid: jax.Array,
    coordinates: jax.Array,
    align_corners: bool = True,
) -> jax.Array:
    """JAX equivalent of the upstream 2D/3D ``grid_sample`` wrapper."""

    grid_dim = coordinates.shape[-1]
    if grid.ndim == grid_dim + 1:
        grid = grid[None, ...]
    if coordinates.ndim == 2:
        coordinates = coordinates[None, ...]
    if grid_dim not in (2, 3):
        raise NotImplementedError(
            f"_grid_sample_wrapper supports 2D / 3D coords only; got {grid_dim}D."
        )
    if grid.shape[0] != coordinates.shape[0]:
        raise ValueError(
            "_grid_sample_wrapper requires matching grid and coordinate "
            f"batch sizes, got {grid.shape[0]} and {coordinates.shape[0]}."
        )

    sample = _sample_2d if grid_dim == 2 else _sample_3d
    sampled = jax.vmap(
        lambda one_grid, one_coordinates: sample(
            one_grid,
            one_coordinates,
            align_corners=align_corners,
        )
    )(grid, coordinates)
    return jnp.squeeze(sampled, axis=0) if sampled.shape[0] == 1 else sampled


def _init_grid_params(
    grid_nd: int,
    in_dim: int,
    out_dim: int,
    resolution: Sequence[int],
    *,
    rngs: nnx.Rngs,
    minimum: float = 0.1,
    maximum: float = 0.5,
) -> nnx.List:
    if in_dim != len(resolution):
        raise ValueError(
            f"_init_grid_param: in_dim={in_dim} != len(reso)={len(resolution)}."
        )
    if grid_nd > in_dim:
        raise ValueError(f"_init_grid_param: grid_nd={grid_nd} > in_dim={in_dim}.")

    has_time_planes = in_dim == 4
    planes = []
    for coordinate_pair in itertools.combinations(range(in_dim), grid_nd):
        shape = (1, out_dim) + tuple(
            resolution[index] for index in reversed(coordinate_pair)
        )
        if has_time_planes and 3 in coordinate_pair:
            values = jnp.ones(shape, dtype=jnp.float32)
        else:
            values = jax.random.uniform(
                rngs.params(),
                shape,
                dtype=jnp.float32,
                minval=minimum,
                maxval=maximum,
            )
        planes.append(nnx.Param(values))
    return nnx.List(planes)


def _interpolate_ms_features(
    points: jax.Array,
    grids: nnx.List,
    grid_dimensions: int,
    concat_features: bool,
) -> jax.Array:
    coordinate_groups = tuple(
        itertools.combinations(range(points.shape[-1]), grid_dimensions)
    )
    scale_features = []
    summed: jax.Array | None = None
    for scale_grids in grids:
        product: jax.Array | None = None
        for plane, coordinate_group in zip(scale_grids, coordinate_groups):
            sampled = _grid_sample_wrapper(
                plane[...], points[..., coordinate_group]
            ).reshape(points.shape[0], -1)
            product = sampled if product is None else product * sampled
        if product is None:
            continue
        if concat_features:
            scale_features.append(product)
        else:
            summed = product if summed is None else summed + product

    if concat_features:
        return jnp.concatenate(scale_features, axis=-1)
    if summed is not None:
        return summed
    return jnp.zeros((points.shape[0], 0), dtype=points.dtype)


class HexPlaneField(nnx.Module):
    """Multi-resolution six-plane decomposition of a 4D feature field."""

    _SPATIAL_PLANE_IDXS = (0, 1, 3)
    _TEMPORAL_PLANE_IDXS = (2, 4, 5)

    def __init__(
        self,
        bounds: float = 1.6,
        planes_config: dict | None = None,
        multires: Sequence[int] | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        config = (
            dict(planes_config)
            if planes_config is not None
            else dict(_DEFAULT_PLANE_CONFIG)
        )
        multires_values = list(_DEFAULT_MULTIRES if multires is None else multires)
        rngs = nnx.Rngs(0) if rngs is None else rngs

        self.bounds = float(bounds)
        self.aabb = nnx.Variable(
            jnp.asarray(
                [
                    [bounds, bounds, bounds],
                    [-bounds, -bounds, -bounds],
                ],
                dtype=jnp.float32,
            )
        )
        self.grid_config = config
        self.multires = multires_values
        self.concat_features = True

        all_grids = []
        self.feat_dim = 0
        for multiplier in multires_values:
            base_resolution = list(config["resolution"])
            resolution = [
                value * multiplier for value in base_resolution[:3]
            ] + base_resolution[3:]
            scale_grids = _init_grid_params(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                resolution=resolution,
                rngs=rngs,
            )
            if self.concat_features:
                self.feat_dim += int(scale_grids[-1].shape[1])
            else:
                self.feat_dim = int(scale_grids[-1].shape[1])
            all_grids.append(scale_grids)
        self.grids = nnx.List(all_grids)

    def __call__(self, xyzt: jax.Array) -> jax.Array:
        """Sample features at points whose last coordinate dimension is four."""

        if xyzt.shape[-1] != 4:
            raise ValueError(
                "HexPlaneField.forward: xyzt last dim must be 4, "
                f"got shape {tuple(xyzt.shape)}."
            )
        xyz = xyzt[..., :3]
        time = xyzt[..., 3:]
        normalized_xyz = _normalize_aabb(xyz, self.aabb[...])
        points = jnp.concatenate((normalized_xyz, time), axis=-1).reshape(-1, 4)
        return _interpolate_ms_features(
            points,
            self.grids,
            self.grid_config["grid_dimensions"],
            self.concat_features,
        )

    forward = __call__

    def spatial_planes(self) -> list[jax.Array]:
        """Return spatial ``xy``, ``xz``, and ``yz`` planes at every scale."""

        return [
            scale_grids[index][...]
            for scale_grids in self.grids
            for index in self._SPATIAL_PLANE_IDXS
        ]

    def temporal_planes(self) -> list[jax.Array]:
        """Return temporal ``xt``, ``yt``, and ``zt`` planes at every scale."""

        return [
            scale_grids[index][...]
            for scale_grids in self.grids
            for index in self._TEMPORAL_PLANE_IDXS
        ]


__all__ = ["HexPlaneField"]
