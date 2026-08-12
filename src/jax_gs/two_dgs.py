"""Pure-JAX, fixed-capacity rasterization for 2D Gaussian splatting.

The public functions follow gsplat 1.5.3's 2DGS API.  Unlike the CUDA
implementation, this module deliberately keeps every intermediate shape static:
inactive storage slots are masked and every tile owns a fixed-size candidate
buffer.  This makes pruning and densification compatible with a jitted training
step without recompilation when the active set changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterator, Literal

import jax
import jax.numpy as jnp

from .config import RasterizationConfig
from .intersections import TileIntersections, intersect_tiles
from .low_level import (
    PaddedOffsets,
    PaddedRasterizationIndices,
    _bits_for_count,
    _encode_high_word,
)
from .math import quat_scale_to_matrix, safe_normalize
from .rasterization import (
    _assemble_dense_intersection_metadata,
    _automatic_intersection_capacity,
    _prepare_colors,
)


RenderMode2DGS = Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"]
DepthMode2DGS = Literal["expected", "median"]


def _broadcast_ray_transforms(
    ray_transforms: jax.Array, pixel_template: jax.Array
) -> jax.Array:
    return jnp.broadcast_to(
        ray_transforms[:, None, :, :],
        (
            ray_transforms.shape[0],
            pixel_template.shape[0],
            ray_transforms.shape[1],
            ray_transforms.shape[2],
        ),
    )


@jax.custom_jvp
def _broadcast_ray_transforms_with_densify_probe(
    ray_transforms: jax.Array,
    densify_probe: jax.Array,
    pixel_template: jax.Array,
) -> jax.Array:
    """Broadcast transforms while exposing the signed densification VJP."""

    del densify_probe
    return _broadcast_ray_transforms(ray_transforms, pixel_template)


@_broadcast_ray_transforms_with_densify_probe.defjvp
def _broadcast_ray_transforms_with_densify_probe_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    ray_transforms, densify_probe, pixel_template = primals
    ray_tangent, densify_tangent, _ = tangents
    del densify_probe
    output = _broadcast_ray_transforms(ray_transforms, pixel_template)
    output_tangent = _broadcast_ray_transforms(ray_tangent, pixel_template)
    w_m_z = ray_transforms[:, 2, 2]
    output_tangent = output_tangent.at[..., 0, 2].add(
        densify_tangent[:, 0, None] * w_m_z[:, None]
    )
    output_tangent = output_tangent.at[..., 1, 2].add(
        densify_tangent[:, 1, None] * w_m_z[:, None]
    )
    return output, output_tangent


@jax.custom_vjp
def _broadcast_ray_transforms_with_densify_and_absgrad_probes(
    ray_transforms: jax.Array,
    densify_probe: jax.Array,
    densify_absgrad_probe: jax.Array,
    pixel_template: jax.Array,
) -> jax.Array:
    """Expose current-main's 2DGS densification VJP in pure JAX.

    The renderer's 3D surfel branch differentiates through the projective ray
    transform. Upstream records a separate two-vector per Gaussian from the
    z-components of the first two transform rows, scaled by ``w_M.z``. Both
    probes are forward no-ops; one receives the signed pixel sum and the other
    receives the componentwise absolute pixel sum.
    """

    del densify_probe, densify_absgrad_probe
    return _broadcast_ray_transforms(ray_transforms, pixel_template)


def _broadcast_ray_transforms_with_densify_and_absgrad_probes_fwd(
    ray_transforms: jax.Array,
    densify_probe: jax.Array,
    densify_absgrad_probe: jax.Array,
    pixel_template: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    del densify_probe, densify_absgrad_probe
    return (
        _broadcast_ray_transforms(ray_transforms, pixel_template),
        (ray_transforms[:, 2, 2], pixel_template),
    )


def _broadcast_ray_transforms_with_densify_and_absgrad_probes_bwd(
    residual: tuple[jax.Array, jax.Array], pixel_cotangent: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    w_m_z, pixel_template = residual
    transform_gradient = jnp.sum(pixel_cotangent, axis=1)
    pixel_densify = jnp.stack(
        (
            pixel_cotangent[..., 0, 2] * w_m_z[:, None],
            pixel_cotangent[..., 1, 2] * w_m_z[:, None],
        ),
        axis=-1,
    )
    densify_gradient = jnp.sum(pixel_densify, axis=1)
    densify_absgrad = jnp.sum(jnp.abs(pixel_densify), axis=1)
    pixel_template_gradient = jnp.zeros_like(pixel_template)
    return (
        transform_gradient,
        densify_gradient,
        densify_absgrad,
        pixel_template_gradient,
    )


_broadcast_ray_transforms_with_densify_and_absgrad_probes.defvjp(
    _broadcast_ray_transforms_with_densify_and_absgrad_probes_fwd,
    _broadcast_ray_transforms_with_densify_and_absgrad_probes_bwd,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PaddedProjection2DGS:
    """Static packed 2DGS projection with gsplat-compatible iteration.

    The first ``valid_count`` entries contain visible projections.  Remaining
    entries are padding, identified by ``-1`` ids and zero-valued projection
    data.  Iteration yields gsplat's nine public packed
    values; ``valid_count`` and ``overflow`` are JAX-specific metadata.
    """

    batch_ids: jax.Array
    camera_ids: jax.Array
    gaussian_ids: jax.Array
    indptr: jax.Array
    radii: jax.Array
    means2d: jax.Array
    depths: jax.Array
    ray_transforms: jax.Array
    normals: jax.Array
    valid_count: jax.Array
    overflow: jax.Array

    def __iter__(self) -> Iterator[jax.Array]:
        yield self.batch_ids
        yield self.camera_ids
        yield self.gaussian_ids
        yield self.indptr
        yield self.radii
        yield self.means2d
        yield self.depths
        yield self.ray_transforms
        yield self.normals

    def tree_flatten(self):
        return (
            (
                self.batch_ids,
                self.camera_ids,
                self.gaussian_ids,
                self.indptr,
                self.radii,
                self.means2d,
                self.depths,
                self.ray_transforms,
                self.normals,
                self.valid_count,
                self.overflow,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


def _pack_projection_2dgs(
    radii: jax.Array,
    means2d: jax.Array,
    depths: jax.Array,
    ray_transforms: jax.Array,
    normals: jax.Array,
) -> PaddedProjection2DGS:
    """Compact dense ``[..., C, N, ...]`` projection into a static buffer."""

    batch_count = math.prod(radii.shape[:-3])
    camera_count = radii.shape[-3]
    gaussian_count = radii.shape[-2]
    capacity = batch_count * camera_count * gaussian_count
    flat_valid = jnp.all(radii.reshape(capacity, 2) > 0, axis=-1)
    selected = jnp.nonzero(flat_valid, size=capacity, fill_value=0)[0]
    valid_count = jnp.count_nonzero(flat_valid).astype(jnp.int32)
    output_valid = jnp.arange(capacity, dtype=jnp.int32) < valid_count
    group_counts = jnp.sum(
        flat_valid.reshape(batch_count * camera_count, gaussian_count),
        axis=-1,
        dtype=jnp.int32,
    )
    indptr = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(group_counts, dtype=jnp.int32),
        )
    )

    gaussian_ids = selected % gaussian_count
    camera_ids = (selected // gaussian_count) % camera_count
    batch_ids = selected // (camera_count * gaussian_count)
    return PaddedProjection2DGS(
        batch_ids=jnp.where(output_valid, batch_ids, -1).astype(jnp.int32),
        camera_ids=jnp.where(output_valid, camera_ids, -1).astype(jnp.int32),
        gaussian_ids=jnp.where(output_valid, gaussian_ids, -1).astype(jnp.int32),
        indptr=indptr,
        radii=jnp.where(
            output_valid[:, None], radii.reshape(capacity, 2)[selected], 0
        ),
        means2d=jnp.where(
            output_valid[:, None], means2d.reshape(capacity, 2)[selected], 0.0
        ),
        depths=jnp.where(
            output_valid, depths.reshape(capacity)[selected], 0.0
        ),
        ray_transforms=jnp.where(
            output_valid[:, None, None],
            ray_transforms.reshape(capacity, 3, 3)[selected],
            0.0,
        ),
        normals=jnp.where(
            output_valid[:, None], normals.reshape(capacity, 3)[selected], 0.0
        ),
        valid_count=valid_count,
        overflow=jnp.asarray(False),
    )


def _pack_dense_metadata_2dgs(
    projection: PaddedProjection2DGS,
    opacities: jax.Array,
    intersections: dict[str, jax.Array],
) -> dict[str, Any]:
    """Create one global packed view across leading batch dimensions."""

    batch_shape = opacities.shape[:-2]
    batch_count = math.prod(batch_shape) if batch_shape else 1
    camera_count, gaussian_count = opacities.shape[-2:]
    capacity = batch_count * camera_count * gaussian_count
    packed_slots = jnp.arange(capacity, dtype=jnp.int32)
    output_valid = packed_slots < projection.valid_count

    if capacity == 0:
        dense_to_packed = jnp.empty((0,), dtype=jnp.int32)
        global_dense_slots = jnp.empty((0,), dtype=jnp.int32)
        safe_camera_ids = projection.camera_ids
        safe_gaussian_ids = projection.gaussian_ids
    else:
        safe_batch_ids = jnp.clip(
            projection.batch_ids, 0, batch_count - 1
        )
        safe_camera_ids = jnp.clip(projection.camera_ids, 0, camera_count - 1)
        safe_gaussian_ids = jnp.clip(
            projection.gaussian_ids, 0, gaussian_count - 1
        )
        global_dense_slots = (
            (safe_batch_ids * camera_count + safe_camera_ids) * gaussian_count
            + safe_gaussian_ids
        )
        scatter_destinations = jnp.where(
            output_valid, global_dense_slots, jnp.int32(capacity)
        )
        dense_to_packed = jnp.full(
            (capacity + 1,), -1, dtype=jnp.int32
        ).at[scatter_destinations].set(
            jnp.where(output_valid, packed_slots, -1)
        )[:capacity]

    def gather(values: jax.Array) -> jax.Array:
        trailing_shape = values.shape[len(batch_shape) + 2 :]
        if capacity == 0:
            return jnp.zeros((0,) + trailing_shape, dtype=values.dtype)
        flat_values = values.reshape((capacity,) + trailing_shape)
        gathered = flat_values[global_dense_slots]
        mask = output_valid.reshape(
            (capacity,) + (1,) * len(trailing_shape)
        )
        return jnp.where(mask, gathered, jnp.zeros_like(gathered))

    dense_flatten_ids = intersections["flatten_ids"].reshape(batch_count, -1)
    per_batch_isect_capacity = dense_flatten_ids.shape[1]
    global_isect_capacity = batch_count * per_batch_isect_capacity
    dense_isect_ids = intersections["isect_ids"].reshape(
        batch_count, per_batch_isect_capacity, 2
    )
    reported_isect_counts = intersections["isect_valid_count"].reshape(
        batch_count
    )
    isect_positions = jnp.arange(
        per_batch_isect_capacity, dtype=jnp.int32
    )
    dense_isect_valid = (
        (isect_positions[None, :] < reported_isect_counts[:, None])
        & (dense_flatten_ids >= 0)
        & (dense_isect_ids[..., 0] != -1)
    )
    per_batch_isect_counts = jnp.sum(
        dense_isect_valid, axis=1, dtype=jnp.int32
    )
    isect_valid_count = jnp.sum(per_batch_isect_counts, dtype=jnp.int32)
    selected_isects = jnp.nonzero(
        dense_isect_valid.reshape(-1),
        size=global_isect_capacity,
        fill_value=0,
    )[0]
    output_isect_valid = (
        jnp.arange(global_isect_capacity, dtype=jnp.int32)
        < isect_valid_count
    )
    source_batch_ids = jnp.repeat(
        jnp.arange(batch_count, dtype=jnp.int32),
        per_batch_isect_capacity,
    )[selected_isects]
    selected_local_dense_ids = dense_flatten_ids.reshape(-1)[selected_isects]
    if gaussian_count == 0:
        safe_local_dense_ids = jnp.zeros_like(selected_local_dense_ids)
    else:
        safe_local_dense_ids = jnp.clip(
            selected_local_dense_ids,
            0,
            camera_count * gaussian_count - 1,
        )
    selected_global_dense_ids = (
        source_batch_ids * camera_count * gaussian_count
        + safe_local_dense_ids
    )
    if capacity == 0:
        packed_flatten_ids = jnp.full(
            (global_isect_capacity,), -1, dtype=jnp.int32
        )
    else:
        mapped_ids = dense_to_packed[selected_global_dense_ids]
        packed_flatten_ids = jnp.where(
            output_isect_valid & (mapped_ids >= 0), mapped_ids, -1
        ).astype(jnp.int32)

    selected_isect_ids = dense_isect_ids.reshape(-1, 2)[selected_isects]
    tile_height, tile_width = intersections["isect_offsets"].shape[-2:]
    tile_bits = _bits_for_count(tile_width * tile_height)
    high_words = jax.lax.bitcast_convert_type(
        selected_isect_ids[:, 0], jnp.uint32
    )
    tile_ids = (
        high_words & jnp.uint32((1 << tile_bits) - 1)
    ).astype(jnp.int32)
    local_camera_ids = (
        jnp.zeros_like(safe_local_dense_ids)
        if gaussian_count == 0
        else safe_local_dense_ids // gaussian_count
    )
    global_image_ids = source_batch_ids * camera_count + local_camera_ids
    packed_isect_ids = jnp.stack(
        (
            _encode_high_word(global_image_ids, tile_ids, tile_bits),
            selected_isect_ids[:, 1],
        ),
        axis=-1,
    )
    packed_isect_ids = jnp.where(
        output_isect_valid[:, None], packed_isect_ids, -1
    ).astype(jnp.int32)

    isect_bases = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(per_batch_isect_counts[:-1], dtype=jnp.int32),
        )
    )
    global_isect_offsets = intersections["isect_offsets"].reshape(
        batch_count, camera_count, tile_height, tile_width
    ) + isect_bases[:, None, None, None]
    global_isect_offsets = global_isect_offsets.reshape(
        batch_shape + (camera_count, tile_height, tile_width)
    )

    return {
        "batch_ids": projection.batch_ids,
        "camera_ids": projection.camera_ids,
        "gaussian_ids": projection.gaussian_ids,
        "indptr": projection.indptr,
        "radii": projection.radii,
        "means2d": projection.means2d,
        "depths": projection.depths,
        "ray_transforms": projection.ray_transforms,
        "opacities": gather(opacities),
        "normals": projection.normals,
        "gradient_2dgs": jnp.zeros_like(projection.means2d),
        "valid": output_valid,
        "tiles_per_gauss": gather(intersections["tiles_per_gauss"]),
        "isect_ids": packed_isect_ids,
        "flatten_ids": packed_flatten_ids,
        "isect_offsets": global_isect_offsets,
        "isect_valid_count": isect_valid_count,
        "projection_valid_count": projection.valid_count,
        "projection_capacity": jnp.asarray(capacity, dtype=jnp.int32),
    }


def _has_color(render_mode: RenderMode2DGS) -> bool:
    return render_mode in {"RGB", "RGB+D", "RGB+ED"}


def _safe_denominator(value: jax.Array, eps: float = 1.0e-8) -> jax.Array:
    eps_array = jnp.asarray(eps, dtype=value.dtype)
    sign = jnp.where(value < 0.0, -jnp.ones_like(value), jnp.ones_like(value))
    return jnp.where(jnp.abs(value) < eps_array, sign * eps_array, value)


def fully_fused_projection_2dgs(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    viewmats: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1.0e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    *,
    active_mask: jax.Array | None = None,
) -> (
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    | PaddedProjection2DGS
):
    """Project fixed-capacity 2D Gaussian surfels into one or more cameras.

    Returns ``(radii, means2d, depths, ray_transforms, normals)`` with dense
    shapes ``[..., C, N, ...]``.  ``ray_transforms`` is ``K @ [R*s_x,
    R*s_y, mean_camera]`` and is consumed directly by the ray--surfel
    intersection used during rasterization.  Normals are in camera space and
    face the camera.

    With ``packed=True``, the result iterates as gsplat's nine-value packed
    tuple, including CSR ``indptr`` for batch-camera groups, but keeps the dense
    projection capacity so its shape remains static. Callers use
    ``valid_count`` to identify the meaningful prefix. ``sparse_grad=True``
    requires this packed, unbatched form, while JAX keeps the underlying
    gradients in dense fixed-capacity storage.
    """

    del eps2d
    means = jnp.asarray(means)
    if sparse_grad:
        if not packed:
            raise ValueError("sparse_grad=True requires packed=True")
        if means.ndim != 2:
            raise ValueError("sparse_grad does not support batch dimensions")
    quats = jnp.asarray(quats)
    scales = jnp.asarray(scales)
    viewmats = jnp.asarray(viewmats)
    Ks = jnp.asarray(Ks)

    if means.ndim < 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [..., N, 3]")
    if quats.shape != means.shape[:-1] + (4,):
        raise ValueError("quats must have shape [..., N, 4]")
    if scales.shape != means.shape:
        raise ValueError("scales must have shape [..., N, 3]")
    if means.ndim == 2 and viewmats.ndim == 2:
        viewmats = viewmats[None, ...]
    if means.ndim == 2 and Ks.ndim == 2:
        Ks = Ks[None, ...]

    batch_shape = means.shape[:-2]
    if viewmats.ndim < 3 or viewmats.shape[-2:] != (4, 4):
        raise ValueError("viewmats must have shape [..., C, 4, 4]")
    if Ks.ndim < 3 or Ks.shape[-2:] != (3, 3):
        raise ValueError("Ks must have shape [..., C, 3, 3]")
    if viewmats.shape[:-3] != batch_shape or Ks.shape[:-3] != batch_shape:
        raise ValueError("means, viewmats, and Ks must share batch dimensions")
    if viewmats.shape[-3] != Ks.shape[-3]:
        raise ValueError("viewmats and Ks must contain the same number of cameras")

    rotation_cw = viewmats[..., :3, :3]
    translation_cw = viewmats[..., :3, 3]
    means_camera = jnp.einsum("...cij,...nj->...cni", rotation_cw, means)
    means_camera = means_camera + translation_cw[..., None, :]

    # 2DGS uses only the two tangent scales.  A unit third column provides a
    # scale-independent surface normal, matching the fused CUDA projection.
    surface_scales = jnp.concatenate(
        (scales[..., :2], jnp.ones_like(scales[..., 2:3])), axis=-1
    )
    tangent_frame_world = quat_scale_to_matrix(quats, surface_scales)
    tangent_frame_camera = jnp.einsum(
        "...cij,...njk->...cnik", rotation_cw, tangent_frame_world
    )

    normals = tangent_frame_camera[..., :, 2]
    facing = jnp.sum(-normals * means_camera, axis=-1, keepdims=True) > 0.0
    normals = normals * jnp.where(facing, 1.0, -1.0)

    camera_from_surfel = jnp.concatenate(
        (tangent_frame_camera[..., :, :2], means_camera[..., :, None]), axis=-1
    )
    ray_transforms = jnp.einsum(
        "...cij,...cnjk->...cnik", Ks, camera_from_surfel
    )

    row_u = ray_transforms[..., 0, :]
    row_v = ray_transforms[..., 1, :]
    row_w = ray_transforms[..., 2, :]
    signature = jnp.asarray((1.0, 1.0, -1.0), dtype=means.dtype)
    distance = jnp.sum(signature * row_w * row_w, axis=-1)
    conditioned = jnp.abs(distance) > jnp.asarray(1.0e-8, means.dtype)
    factors = signature / jnp.where(conditioned, distance, 1.0)[..., None]
    means2d = jnp.stack(
        (
            jnp.sum(factors * row_u * row_w, axis=-1),
            jnp.sum(factors * row_v * row_w, axis=-1),
        ),
        axis=-1,
    )
    extent_sq = jnp.stack(
        (
            means2d[..., 0] ** 2 - jnp.sum(factors * row_u * row_u, axis=-1),
            means2d[..., 1] ** 2 - jnp.sum(factors * row_v * row_v, axis=-1),
        ),
        axis=-1,
    )
    radius = jnp.ceil(3.33 * jnp.sqrt(jnp.maximum(extent_sq, 1.0e-4)))
    depths = means_camera[..., 2]

    valid = conditioned & (depths > near_plane) & (depths < far_plane)
    valid = valid & (
        (radius[..., 0] > radius_clip) | (radius[..., 1] > radius_clip)
    )
    valid = valid & (
        (means2d[..., 0] + radius[..., 0] > 0.0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0.0)
        & (means2d[..., 1] - radius[..., 1] < height)
    )
    finite = (
        jnp.all(jnp.isfinite(means2d), axis=-1)
        & jnp.isfinite(depths)
        & jnp.all(jnp.isfinite(ray_transforms), axis=(-2, -1))
        & jnp.all(jnp.isfinite(normals), axis=-1)
    )
    valid = valid & finite
    if active_mask is not None:
        active_mask = jnp.broadcast_to(
            jnp.asarray(active_mask, dtype=jnp.bool_), means.shape[:-1]
        )
        valid = valid & active_mask[..., None, :]

    radii = jnp.where(valid[..., None], radius, 0.0).astype(jnp.int32)
    if packed:
        return _pack_projection_2dgs(
            radii, means2d, depths, ray_transforms, normals
        )
    return radii, means2d, depths, ray_transforms, normals


def _render_camera_tiles_2dgs(
    means2d: jax.Array,
    radii: jax.Array,
    depths: jax.Array,
    ray_transforms: jax.Array,
    normals: jax.Array,
    opacities: jax.Array,
    colors: jax.Array | None,
    valid: jax.Array,
    *,
    width: int,
    height: int,
    config: RasterizationConfig,
    background: jax.Array,
    render_mode: RenderMode2DGS,
    distloss: bool,
    densify_probe: jax.Array,
    densify_absgrad_probe: jax.Array | None,
    precomputed_intersections: TileIntersections | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    dict[str, jax.Array],
]:
    tile_size = config.tile_size
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size
    tile_count = tile_width * tile_height
    candidate_limit = min(config.max_gaussians_per_tile, means2d.shape[0])
    candidate_capacity = means2d.shape[0]
    color_channels = 0 if colors is None else colors.shape[-1]
    if _has_color(render_mode) and colors is None:
        raise ValueError(f"render_mode={render_mode!r} requires colors")

    local_y, local_x = jnp.meshgrid(
        jnp.arange(tile_size, dtype=means2d.dtype) + 0.5,
        jnp.arange(tile_size, dtype=means2d.dtype) + 0.5,
        indexing="ij",
    )
    local_x = local_x.reshape(-1)
    local_y = local_y.reshape(-1)
    pixel_count = tile_size * tile_size
    intersections = precomputed_intersections
    built_intersections = (
        intersections is None and config.backend != "reference"
    )
    intersection_capacity = (
        0 if intersections is None else intersections.gaussian_ids.shape[0]
    )
    if built_intersections:
        intersection_capacity = _automatic_intersection_capacity(
            means2d.shape[0], tile_count, config
        )
        intersections = intersect_tiles(
            means2d,
            radii,
            depths,
            valid & (opacities > 0.0),
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            max_intersections=intersection_capacity,
            backend=config.intersection_backend,
            sort_backend=config.sort_backend,
            mode="aabb",
        )
    if intersections is not None:
        intersection_offsets = intersections.offsets.reshape(-1)
        candidate_slots = jnp.arange(candidate_capacity, dtype=jnp.int32)

    def render_tile(tile_id: jax.Array):
        tile_x = tile_id % tile_width
        tile_y = tile_id // tile_width
        x0 = tile_x * tile_size
        y0 = tile_y * tile_size
        x1 = x0 + tile_size
        y1 = y0 + tile_size

        if intersections is None:
            overlaps = (
                valid
                & ((radii[..., 0] > 0) | (radii[..., 1] > 0))
                & (means2d[..., 0] + radii[..., 0] > x0)
                & (means2d[..., 0] - radii[..., 0] < x1)
                & (means2d[..., 1] + radii[..., 1] > y0)
                & (means2d[..., 1] - radii[..., 1] < y1)
                & (opacities > 0.0)
            )
            candidate_count = jnp.count_nonzero(overlaps)
            scores = jnp.where(overlaps, -depths, -jnp.inf)
            selected_scores, selected_ids = jax.lax.top_k(
                scores, candidate_capacity
            )
            selected_valid = jnp.isfinite(selected_scores)
        else:
            start = intersection_offsets[tile_id]
            end = jnp.where(
                tile_id + 1 < tile_count,
                intersection_offsets[jnp.minimum(tile_id + 1, tile_count - 1)],
                intersections.valid_count,
            )
            candidate_count = jnp.maximum(end - start, 0)
            positions = start + candidate_slots
            safe_positions = jnp.clip(
                positions, 0, intersections.gaussian_ids.shape[0] - 1
            )
            selected_ids = intersections.gaussian_ids[safe_positions]
            selected_valid = (
                (candidate_slots < candidate_count)
                & (positions < intersections.valid_count)
                & (selected_ids >= 0)
            )
            selected_ids = jnp.clip(selected_ids, 0, means2d.shape[0] - 1)
        selected_valid = (
            selected_valid
            & valid[selected_ids]
            & (opacities[selected_ids] > 0.0)
        )

        selected_means = means2d[selected_ids]
        selected_transforms = ray_transforms[selected_ids]
        selected_normals = normals[selected_ids]
        selected_opacities = opacities[selected_ids]
        selected_depths = depths[selected_ids]

        pixel_x = x0.astype(means2d.dtype) + local_x
        pixel_y = y0.astype(means2d.dtype) + local_y
        pixel_valid = (pixel_x < width) & (pixel_y < height)

        pixel_template = jnp.stack((pixel_x, pixel_y), axis=-1)
        if densify_absgrad_probe is None:
            transforms_per_pixel = (
                _broadcast_ray_transforms_with_densify_probe(
                    selected_transforms,
                    densify_probe[selected_ids],
                    pixel_template,
                )
            )
        else:
            transforms_per_pixel = (
                _broadcast_ray_transforms_with_densify_and_absgrad_probes(
                    selected_transforms,
                    densify_probe[selected_ids],
                    densify_absgrad_probe[selected_ids],
                    pixel_template,
                )
            )
        row_u = transforms_per_pixel[..., 0, :]
        row_v = transforms_per_pixel[..., 1, :]
        row_w = transforms_per_pixel[..., 2, :]
        h_u = pixel_x[None, :, None] * row_w - row_u
        h_v = pixel_y[None, :, None] * row_w - row_v
        ray_cross = jnp.cross(h_u, h_v, axis=-1)
        denominator = ray_cross[..., 2]
        intersection_valid = jnp.abs(denominator) > 1.0e-8
        denominator = _safe_denominator(denominator)
        surfel_u = ray_cross[..., 0] / denominator
        surfel_v = ray_cross[..., 1] / denominator
        weight_3d = surfel_u * surfel_u + surfel_v * surfel_v

        dx = selected_means[:, 0, None] - pixel_x[None, :]
        dy = selected_means[:, 1, None] - pixel_y[None, :]
        weight_2d = 2.0 * (dx * dx + dy * dy)
        sigma = 0.5 * jnp.minimum(weight_3d, weight_2d)
        alpha = jnp.minimum(
            selected_opacities[:, None] * jnp.exp(-jnp.maximum(sigma, 0.0)),
            0.999,
        )
        alpha_valid = (
            selected_valid[:, None]
            & pixel_valid[None, :]
            & intersection_valid
            & (sigma >= 0.0)
            & (alpha >= config.alpha_clip)
        )
        alpha = jnp.where(alpha_valid, alpha, 0.0)

        transmittance = jnp.concatenate(
            (
                jnp.ones((1, pixel_count), dtype=alpha.dtype),
                jnp.cumprod(1.0 - alpha, axis=0)[:-1],
            ),
            axis=0,
        )
        # gsplat excludes the sample that would cross the termination threshold.
        accepted = transmittance * (1.0 - alpha) > config.transmittance_eps
        weights = jnp.where(accepted, alpha * transmittance, 0.0)
        accumulated_alpha = jnp.sum(weights, axis=0)

        if colors is None:
            rendered_color = jnp.zeros((pixel_count, 0), dtype=means2d.dtype)
        else:
            rendered_color = jnp.einsum(
                "kp,kd->pd",
                weights,
                colors[selected_ids],
                precision=jax.lax.Precision.HIGHEST,
            )
            rendered_color = rendered_color + background[None, :color_channels] * (
                1.0 - accumulated_alpha[:, None]
            )

        accumulated_depth = jnp.sum(weights * selected_depths[:, None], axis=0)
        expected_depth = accumulated_depth / jnp.maximum(
            accumulated_alpha, config.transmittance_eps
        )
        expected_depth = jnp.where(accumulated_alpha > 0.0, expected_depth, 0.0)

        candidate_indices = jnp.arange(candidate_capacity, dtype=jnp.int32)[
            :, None
        ]
        median_eligible = (weights > 0.0) & (transmittance > 0.5)
        median_index = jnp.max(
            jnp.where(median_eligible, candidate_indices, -1), axis=0
        )
        median_depth = jnp.take_along_axis(
            selected_depths[:, None], jnp.maximum(median_index, 0)[None, :], axis=0
        )[0]
        median_depth = jnp.where(median_index >= 0, median_depth, 0.0)

        previous_visibility = jnp.concatenate(
            (
                jnp.zeros((1, pixel_count), dtype=weights.dtype),
                jnp.cumsum(weights, axis=0)[:-1],
            ),
            axis=0,
        )
        weighted_depth = weights * selected_depths[:, None]
        previous_weighted_depth = jnp.concatenate(
            (
                jnp.zeros((1, pixel_count), dtype=weights.dtype),
                jnp.cumsum(weighted_depth, axis=0)[:-1],
            ),
            axis=0,
        )
        distortion = 2.0 * jnp.sum(
            weights
            * (
                selected_depths[:, None] * previous_visibility
                - previous_weighted_depth
            ),
            axis=0,
        )
        if not distloss:
            distortion = jnp.zeros_like(distortion)

        rendered_normals = jnp.einsum(
            "kp,kd->pd",
            weights,
            selected_normals,
            precision=jax.lax.Precision.HIGHEST,
        )
        if render_mode == "RGB":
            rendered = rendered_color
        elif render_mode == "D":
            rendered = accumulated_depth[:, None]
        elif render_mode == "ED":
            rendered = expected_depth[:, None]
        elif render_mode == "RGB+D":
            rendered = jnp.concatenate(
                (rendered_color, accumulated_depth[:, None]), axis=-1
            )
        elif render_mode == "RGB+ED":
            rendered = jnp.concatenate(
                (rendered_color, expected_depth[:, None]), axis=-1
            )
        else:
            raise ValueError(f"unsupported render mode: {render_mode}")

        rendered = jnp.where(pixel_valid[:, None], rendered, 0.0)
        accumulated_alpha = jnp.where(pixel_valid, accumulated_alpha, 0.0)
        rendered_normals = jnp.where(
            pixel_valid[:, None], rendered_normals, 0.0
        )
        distortion = jnp.where(pixel_valid, distortion, 0.0)
        median_depth = jnp.where(pixel_valid, median_depth, 0.0)
        expected_depth = jnp.where(pixel_valid, expected_depth, 0.0)
        return (
            rendered.reshape(tile_size, tile_size, -1),
            accumulated_alpha.reshape(tile_size, tile_size, 1),
            rendered_normals.reshape(tile_size, tile_size, 3),
            distortion.reshape(tile_size, tile_size, 1),
            median_depth.reshape(tile_size, tile_size, 1),
            expected_depth.reshape(tile_size, tile_size, 1),
            candidate_count,
            candidate_count > candidate_capacity,
            candidate_count > candidate_limit,
            selected_ids[:candidate_limit],
            selected_valid[:candidate_limit],
        )

    tile_ids = jnp.arange(tile_count, dtype=jnp.int32)
    mapped = jax.lax.map(
        # Reverse mode otherwise keeps every tile's [candidate, pixel]
        # compositing intermediates alive at once.
        jax.checkpoint(render_tile),
        tile_ids,
        batch_size=min(config.tile_batch_size, tile_count),
    )
    (
        rendered_tiles,
        alpha_tiles,
        normal_tiles,
        distortion_tiles,
        median_tiles,
        expected_tiles,
        candidate_counts,
        overflows,
        candidate_limit_exceeded,
        candidate_ids,
        candidate_valid,
    ) = mapped

    def untile(values: jax.Array) -> jax.Array:
        channels = values.shape[-1]
        return (
            values.reshape(
                tile_height, tile_width, tile_size, tile_size, channels
            )
            .transpose(0, 2, 1, 3, 4)
            .reshape(tile_height * tile_size, tile_width * tile_size, channels)
        )[:height, :width]

    return (
        untile(rendered_tiles),
        untile(alpha_tiles),
        untile(normal_tiles),
        untile(distortion_tiles),
        untile(median_tiles),
        untile(expected_tiles),
        {
            "candidate_counts": candidate_counts.reshape(tile_height, tile_width),
            "tile_overflow": overflows.reshape(tile_height, tile_width),
            "candidate_limit_exceeded": candidate_limit_exceeded.reshape(
                tile_height, tile_width
            ),
            "candidate_ids": candidate_ids.reshape(
                tile_height, tile_width, candidate_limit
            ),
            "candidate_valid": candidate_valid.reshape(
                tile_height, tile_width, candidate_limit
            ),
            "intersection_count": (
                intersections.valid_count
                if intersections is not None
                else jnp.sum(candidate_counts, dtype=jnp.int32)
            ),
            "intersection_required_count": (
                intersections.required_count
                if intersections is not None
                else jnp.sum(candidate_counts, dtype=jnp.int32)
            ),
            "intersection_overflow": (
                intersections.overflow
                if intersections is not None
                else jnp.asarray(False)
            ),
            "intersection_capacity": jnp.asarray(
                intersection_capacity
                if intersections is not None
                else min(tile_count * means2d.shape[0], 2**31 - 1),
                dtype=jnp.int32,
            ),
            **(
                {
                    "intersection_gaussian_ids": intersections.gaussian_ids,
                    "intersection_tile_ids": intersections.tile_ids,
                    "intersection_offsets": intersections.offsets,
                }
                if intersections is not None
                else {}
            ),
        },
    )


def _depth_to_surface_normals(
    depths: jax.Array,
    viewmats: jax.Array,
    Ks: jax.Array,
) -> jax.Array:
    """Convert per-camera z-depth maps to world-space surface normals."""

    camera_count, height, width, _ = depths.shape
    if height < 3 or width < 3:
        return jnp.zeros((camera_count, height, width, 3), dtype=depths.dtype)

    pixel_y, pixel_x = jnp.meshgrid(
        jnp.arange(height, dtype=depths.dtype) + 0.5,
        jnp.arange(width, dtype=depths.dtype) + 0.5,
        indexing="ij",
    )

    def normals_for_camera(depth: jax.Array, viewmat: jax.Array, K: jax.Array):
        camera_to_world = jnp.linalg.inv(viewmat)
        direction_camera = jnp.stack(
            (
                (pixel_x - K[0, 2]) / _safe_denominator(K[0, 0]),
                (pixel_y - K[1, 2]) / _safe_denominator(K[1, 1]),
                jnp.ones_like(pixel_x),
            ),
            axis=-1,
        )
        direction_world = jnp.einsum(
            "ij,hwj->hwi", camera_to_world[:3, :3], direction_camera
        )
        points = camera_to_world[:3, 3] + depth * direction_world
        delta_y = points[2:, 1:-1] - points[:-2, 1:-1]
        delta_x = points[1:-1, 2:] - points[1:-1, :-2]
        normal = safe_normalize(jnp.cross(delta_y, delta_x, axis=-1), axis=-1)
        return jnp.pad(normal, ((1, 1), (1, 1), (0, 0)))

    if camera_count == 1:
        return normals_for_camera(depths[0], viewmats[0], Ks[0])[None, ...]
    return jax.lax.map(
        lambda values: normals_for_camera(*values),
        (depths, viewmats, Ks),
        batch_size=1,
    )


def rasterization_2dgs(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    opacities: jax.Array,
    colors: jax.Array | None,
    viewmats: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    near_plane: float | None = None,
    far_plane: float | None = None,
    radius_clip: float | None = None,
    eps2d: float | None = None,
    sh_degree: int | None = None,
    packed: bool = False,
    tile_size: int | None = None,
    backgrounds: jax.Array | None = None,
    render_mode: RenderMode2DGS = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    depth_mode: DepthMode2DGS = "expected",
    *,
    active_mask: jax.Array | None = None,
    max_gaussians_per_tile: int | None = None,
    tile_batch_size: int | None = None,
    config: RasterizationConfig = RasterizationConfig(),
    _gradient_2dgs_offset: jax.Array | None = None,
    _gradient_2dgs_absgrad_probe: jax.Array | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    dict[str, Any],
]:
    """Differentiably rasterize fixed-capacity 2D Gaussian surfels.

    The return order matches gsplat 1.5.3: rendered channels, alpha, rendered
    normals, normals reconstructed from depth, distortion, median depth, and an
    information dictionary.  ``info['render_expected_depth']`` exposes expected
    depth for every render mode.

    ``config.max_gaussians_per_tile`` is a diagnostic threshold and compositing
    chunk size.  Every retained tile intersection is still composited; tiles
    above the threshold set ``info['candidate_limit_exceeded']``.
    """

    if config.compositor_backend in {"pallas", "cuda_ffi"}:
        raise NotImplementedError(
            "the experimental Pallas and CUDA FFI compositors only support 3DGS"
        )
    if _gradient_2dgs_absgrad_probe is not None and not absgrad:
        raise ValueError(
            "_gradient_2dgs_absgrad_probe requires absgrad=True"
        )
    if sparse_grad:
        if not packed:
            raise ValueError("sparse_grad=True requires packed=True")
        if jnp.ndim(means) != 2:
            raise ValueError("sparse_grad does not support batch dimensions")

    means = jnp.asarray(means)
    quats = jnp.asarray(quats)
    scales = jnp.asarray(scales)
    opacities = jnp.asarray(opacities)
    viewmats = jnp.asarray(viewmats)
    Ks = jnp.asarray(Ks)
    if means.ndim < 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [..., N, 3]")
    batch_shape = means.shape[:-2]
    if batch_shape:
        gaussian_count = means.shape[-2]
        batch_count = math.prod(batch_shape)
        if quats.shape != batch_shape + (gaussian_count, 4):
            raise ValueError("quats must have shape [..., N, 4]")
        if scales.shape != batch_shape + (gaussian_count, 3):
            raise ValueError("scales must have shape [..., N, 3]")
        if opacities.shape != batch_shape + (gaussian_count,):
            raise ValueError("opacities must have shape [..., N]")
        if viewmats.shape[:-3] != batch_shape or viewmats.shape[-2:] != (4, 4):
            raise ValueError("viewmats must have shape [..., C, 4, 4]")
        camera_count = viewmats.shape[-3]
        if Ks.shape != batch_shape + (camera_count, 3, 3):
            raise ValueError("Ks must have shape [..., C, 3, 3]")

        def flatten_batch(value: jax.Array) -> jax.Array:
            return value.reshape((batch_count,) + value.shape[len(batch_shape) :])

        flat_means = flatten_batch(means)
        flat_quats = flatten_batch(quats)
        flat_scales = flatten_batch(scales)
        flat_opacities = flatten_batch(opacities)
        flat_viewmats = flatten_batch(viewmats)
        flat_Ks = flatten_batch(Ks)
        if _gradient_2dgs_offset is None:
            flat_gradient_2dgs_offset = None
        else:
            gradient_2dgs_offset = jnp.asarray(_gradient_2dgs_offset)
            expected_offset_shape = batch_shape + (
                camera_count,
                gaussian_count,
                2,
            )
            if gradient_2dgs_offset.shape != expected_offset_shape:
                raise ValueError(
                    "_gradient_2dgs_offset must have shape "
                    f"{expected_offset_shape}"
                )
            flat_gradient_2dgs_offset = flatten_batch(gradient_2dgs_offset)
        if _gradient_2dgs_absgrad_probe is None:
            flat_gradient_2dgs_absgrad_probe = None
        else:
            gradient_2dgs_absgrad_probe = jnp.asarray(
                _gradient_2dgs_absgrad_probe
            )
            expected_probe_shape = batch_shape + (
                camera_count,
                gaussian_count,
                2,
            )
            if gradient_2dgs_absgrad_probe.shape != expected_probe_shape:
                raise ValueError(
                    "_gradient_2dgs_absgrad_probe must have shape "
                    f"{expected_probe_shape}"
                )
            if gradient_2dgs_absgrad_probe.dtype != means.dtype:
                raise ValueError(
                    "_gradient_2dgs_absgrad_probe must have the same dtype "
                    "as means"
                )
            flat_gradient_2dgs_absgrad_probe = flatten_batch(
                gradient_2dgs_absgrad_probe
            )

        if active_mask is None:
            batched_active_mask = jnp.ones(
                batch_shape + (gaussian_count,), dtype=jnp.bool_
            )
        else:
            batched_active_mask = jnp.asarray(active_mask, dtype=jnp.bool_)
            if batched_active_mask.shape == (gaussian_count,):
                batched_active_mask = jnp.broadcast_to(
                    batched_active_mask, batch_shape + (gaussian_count,)
                )
            if batched_active_mask.shape != batch_shape + (gaussian_count,):
                raise ValueError("active_mask must have shape [N] or [..., N]")
        flat_active_mask = flatten_batch(batched_active_mask)

        if colors is None:
            flat_colors = None
        else:
            color_values = jnp.asarray(colors)
            color_tail = color_values.shape[len(batch_shape) :]
            shared_per_gaussian = bool(
                color_tail and color_tail[0] == gaussian_count
            )
            per_camera = bool(
                len(color_tail) >= 2
                and color_tail[0] == camera_count
                and color_tail[1] == gaussian_count
            )
            if (
                color_values.shape[: len(batch_shape)] != batch_shape
                or not (shared_per_gaussian or per_camera)
            ):
                raise ValueError(
                    "colors must have shape [..., N, ...] or [..., C, N, ...]"
                )
            flat_colors = flatten_batch(color_values)

        if backgrounds is None:
            flat_backgrounds = None
        else:
            background_values = jnp.asarray(backgrounds)
            if background_values.ndim == 1:
                background_values = jnp.broadcast_to(
                    background_values,
                    batch_shape + (camera_count, background_values.shape[0]),
                )
            elif (
                background_values.ndim == 2
                and background_values.shape[0] == camera_count
            ):
                background_values = jnp.broadcast_to(
                    background_values, batch_shape + background_values.shape
                )
            elif background_values.shape[: len(batch_shape)] != batch_shape:
                raise ValueError(
                    "backgrounds must have shape [D], [C, D], or [..., C, D]"
                )
            if background_values.shape[len(batch_shape)] != camera_count:
                raise ValueError(
                    "backgrounds camera dimension does not match viewmats"
                )
            flat_backgrounds = flatten_batch(background_values)

        def render_batch(index: jax.Array):
            return rasterization_2dgs(
                flat_means[index],
                flat_quats[index],
                flat_scales[index],
                flat_opacities[index],
                None if flat_colors is None else flat_colors[index],
                flat_viewmats[index],
                flat_Ks[index],
                width,
                height,
                near_plane=near_plane,
                far_plane=far_plane,
                radius_clip=radius_clip,
                eps2d=eps2d,
                sh_degree=sh_degree,
                # Render each scene densely, then assemble one global static
                # packed prefix at the outer recursion boundary.
                packed=False,
                tile_size=tile_size,
                backgrounds=(
                    None if flat_backgrounds is None else flat_backgrounds[index]
                ),
                render_mode=render_mode,
                sparse_grad=sparse_grad,
                absgrad=absgrad,
                distloss=distloss,
                depth_mode=depth_mode,
                active_mask=flat_active_mask[index],
                max_gaussians_per_tile=max_gaussians_per_tile,
                tile_batch_size=tile_batch_size,
                config=config,
                _gradient_2dgs_offset=(
                    None
                    if flat_gradient_2dgs_offset is None
                    else flat_gradient_2dgs_offset[index]
                ),
                _gradient_2dgs_absgrad_probe=(
                    None
                    if flat_gradient_2dgs_absgrad_probe is None
                    else flat_gradient_2dgs_absgrad_probe[index]
                ),
            )

        batched_outputs = jax.lax.map(
            render_batch,
            jnp.arange(batch_count, dtype=jnp.int32),
        )

        def restore_batch(value: jax.Array) -> jax.Array:
            return value.reshape(batch_shape + value.shape[1:])

        restored_outputs = jax.tree.map(restore_batch, batched_outputs)
        if packed:
            *images, info = restored_outputs
            packed_metadata_available = all(
                key in info
                for key in (
                    "flatten_ids",
                    "isect_ids",
                    "isect_offsets",
                    "isect_valid_count",
                )
            )
            if packed_metadata_available:
                projection = _pack_projection_2dgs(
                    info["radii"],
                    info["means2d"],
                    info["depths"],
                    info["ray_transforms"],
                    info["normals"],
                )
                info.update(
                    _pack_dense_metadata_2dgs(
                        projection,
                        info["opacities"],
                        info,
                    )
                )
                info["packed_requested"] = jnp.asarray(True)
                info["packed_metadata_available"] = jnp.asarray(True)
                info["n_batches"] = jnp.asarray(
                    batch_count, dtype=jnp.int32
                )
                info["n_cameras"] = jnp.asarray(
                    camera_count, dtype=jnp.int32
                )
                for key in (
                    "width",
                    "height",
                    "tile_size",
                    "tile_width",
                    "tile_height",
                ):
                    info[key] = info[key].reshape(-1)[0]
            else:
                info["packed_requested"] = jnp.ones(
                    batch_shape, dtype=jnp.bool_
                )
                info["packed_metadata_available"] = jnp.zeros(
                    batch_shape, dtype=jnp.bool_
                )
            return (*images, info)
        return restored_outputs

    packed_metadata_requested = bool(packed)
    dense_metadata_requested = not packed_metadata_requested
    sparse_grad_requested = jnp.asarray(sparse_grad)
    absgrad_requested = jnp.asarray(absgrad)
    absgrad_probe_enabled = _gradient_2dgs_absgrad_probe is not None
    del packed, sparse_grad, absgrad
    if render_mode not in {"RGB", "D", "ED", "RGB+D", "RGB+ED"}:
        raise ValueError(f"unsupported render mode: {render_mode}")
    if depth_mode not in {"expected", "median"}:
        raise ValueError("depth_mode must be 'expected' or 'median'")

    means = jnp.asarray(means)
    quats = jnp.asarray(quats)
    scales = jnp.asarray(scales)
    opacities = jnp.asarray(opacities)
    viewmats = jnp.asarray(viewmats)
    Ks = jnp.asarray(Ks)
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [N, 3]")
    gaussian_count = means.shape[0]
    if gaussian_count == 0:
        raise ValueError("at least one fixed-capacity Gaussian slot is required")
    if quats.shape != (gaussian_count, 4):
        raise ValueError("quats must have shape [N, 4]")
    if scales.shape != (gaussian_count, 3):
        raise ValueError("scales must have shape [N, 3]")
    if opacities.shape != (gaussian_count,):
        raise ValueError("opacities must have shape [N]")
    if viewmats.ndim == 2:
        viewmats = viewmats[None, ...]
    if Ks.ndim == 2:
        Ks = Ks[None, ...]
    if viewmats.ndim != 3 or viewmats.shape[-2:] != (4, 4):
        raise ValueError("viewmats must have shape [C, 4, 4]")
    if Ks.ndim != 3 or Ks.shape[-2:] != (3, 3):
        raise ValueError("Ks must have shape [C, 3, 3]")
    if viewmats.shape[0] != Ks.shape[0]:
        raise ValueError("viewmats and Ks must contain the same number of cameras")
    if _gradient_2dgs_offset is not None:
        gradient_2dgs_offset = jnp.asarray(
            _gradient_2dgs_offset, dtype=means.dtype
        )
        expected_offset_shape = (viewmats.shape[0], gaussian_count, 2)
        if gradient_2dgs_offset.shape != expected_offset_shape:
            raise ValueError(
                "_gradient_2dgs_offset must have shape "
                f"{expected_offset_shape}"
            )
    else:
        gradient_2dgs_offset = jnp.zeros(
            (viewmats.shape[0], gaussian_count, 2), dtype=means.dtype
        )
    expected_probe_shape = (viewmats.shape[0], gaussian_count, 2)
    if _gradient_2dgs_absgrad_probe is None:
        gradient_2dgs_absgrad_probe = jnp.zeros(
            expected_probe_shape, dtype=means.dtype
        )
    else:
        gradient_2dgs_absgrad_probe = jnp.asarray(
            _gradient_2dgs_absgrad_probe
        )
        if gradient_2dgs_absgrad_probe.shape != expected_probe_shape:
            raise ValueError(
                "_gradient_2dgs_absgrad_probe must have shape "
                f"{expected_probe_shape}"
            )
        if gradient_2dgs_absgrad_probe.dtype != means.dtype:
            raise ValueError(
                "_gradient_2dgs_absgrad_probe must have the same dtype as means"
            )
    if active_mask is None:
        active_mask = jnp.ones((gaussian_count,), dtype=jnp.bool_)
    else:
        active_mask = jnp.asarray(active_mask, dtype=jnp.bool_)
    if active_mask.shape != (gaussian_count,):
        raise ValueError("active_mask must have shape [N]")

    overrides: dict[str, Any] = {}
    if near_plane is not None:
        overrides["near_plane"] = near_plane
    if far_plane is not None:
        overrides["far_plane"] = far_plane
    if radius_clip is not None:
        overrides["radius_clip"] = radius_clip
    if eps2d is not None:
        overrides["eps2d"] = eps2d
    if tile_size is not None:
        overrides["tile_size"] = tile_size
    if max_gaussians_per_tile is not None:
        overrides["max_gaussians_per_tile"] = max_gaussians_per_tile
    if tile_batch_size is not None:
        overrides["tile_batch_size"] = tile_batch_size
    if overrides:
        config = replace(config, **overrides)

    radii, means2d, depths, ray_transforms, normals_camera = (
        fully_fused_projection_2dgs(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d=config.eps2d,
            near_plane=config.near_plane,
            far_plane=config.far_plane,
            radius_clip=config.radius_clip,
            active_mask=active_mask,
        )
    )
    valid = jnp.all(radii > 0, axis=-1) & active_mask[None, :]
    projected_opacities = jnp.broadcast_to(
        opacities[None, :], (viewmats.shape[0], gaussian_count)
    )
    projected_opacities = jnp.where(valid, projected_opacities, 0.0)

    prepared_colors = (
        _prepare_colors(means, colors, viewmats, sh_degree, viewmats.shape[0])
        if _has_color(render_mode)
        else None
    )
    if _has_color(render_mode) and prepared_colors is None:
        raise ValueError(f"render_mode={render_mode!r} requires colors")
    if prepared_colors is None:
        color_values = jnp.zeros(
            (viewmats.shape[0], gaussian_count, 0), dtype=means.dtype
        )
        color_channels = 0
    else:
        color_values = prepared_colors
        color_channels = prepared_colors.shape[-1]

    if backgrounds is None:
        background_values = jnp.zeros(
            (viewmats.shape[0], color_channels), dtype=means.dtype
        )
    else:
        background_values = jnp.asarray(backgrounds)
        if background_values.ndim == 1:
            background_values = jnp.broadcast_to(
                background_values[None, :],
                (viewmats.shape[0], background_values.shape[0]),
            )
        if background_values.shape != (viewmats.shape[0], color_channels):
            raise ValueError("backgrounds must have shape [D] or [C, D]")

    def render_camera(camera_inputs):
        (
            means2d_camera,
            densify_probe_camera,
            absgrad_probe_camera,
            radii_camera,
            depths_camera,
            transforms_camera,
            normals_camera_value,
            opacity_camera,
            colors_camera,
            valid_camera,
            background_camera,
        ) = camera_inputs
        colors_arg = colors_camera if _has_color(render_mode) else None
        return _render_camera_tiles_2dgs(
            means2d_camera,
            radii_camera,
            depths_camera,
            transforms_camera,
            normals_camera_value,
            opacity_camera,
            colors_arg,
            valid_camera,
            width=width,
            height=height,
            config=config,
            background=background_camera,
            render_mode=render_mode,
            distloss=distloss,
            densify_probe=densify_probe_camera,
            densify_absgrad_probe=(
                absgrad_probe_camera if absgrad_probe_enabled else None
            ),
        )

    camera_inputs = (
        means2d,
        gradient_2dgs_offset,
        gradient_2dgs_absgrad_probe,
        radii,
        depths,
        ray_transforms,
        normals_camera,
        projected_opacities,
        color_values,
        valid,
        background_values,
    )
    if viewmats.shape[0] == 1:
        camera_outputs = render_camera(tuple(value[0] for value in camera_inputs))
        camera_outputs = jax.tree.map(lambda value: value[None, ...], camera_outputs)
    else:
        camera_outputs = jax.lax.map(
            render_camera,
            camera_inputs,
        )
    (
        renders,
        alphas,
        rendered_normals_camera,
        render_distort,
        render_median,
        render_expected,
        tile_info,
    ) = camera_outputs

    tile_width = (width + config.tile_size - 1) // config.tile_size
    tile_height = (height + config.tile_size - 1) // config.tile_size
    dense_intersection_metadata = None
    if config.backend != "reference":
        dense_intersection_metadata = _assemble_dense_intersection_metadata(
            depths,
            tile_info["intersection_gaussian_ids"],
            tile_info["intersection_tile_ids"],
            tile_info["intersection_offsets"],
            tile_info["intersection_count"],
            tile_width=tile_width,
            tile_height=tile_height,
        )
    packed_metadata_available = (
        packed_metadata_requested and dense_intersection_metadata is not None
    )

    camera_to_world = jnp.linalg.inv(viewmats)
    rendered_normals = jnp.einsum(
        "cij,chwj->chwi", camera_to_world[:, :3, :3], rendered_normals_camera
    )
    surface_depth = render_median if depth_mode == "median" else render_expected
    surface_normals = _depth_to_surface_normals(surface_depth, viewmats, Ks)
    surface_normals = jnp.where(alphas > 0.0, surface_normals, 0.0)

    min_tile_x = jnp.floor(
        (means2d[..., 0] - radii[..., 0]) / config.tile_size
    ).astype(jnp.int32)
    min_tile_y = jnp.floor(
        (means2d[..., 1] - radii[..., 1]) / config.tile_size
    ).astype(jnp.int32)
    max_tile_x = jnp.ceil(
        (means2d[..., 0] + radii[..., 0]) / config.tile_size
    ).astype(jnp.int32) - 1
    max_tile_y = jnp.ceil(
        (means2d[..., 1] + radii[..., 1]) / config.tile_size
    ).astype(jnp.int32) - 1
    min_tile_x = jnp.clip(min_tile_x, 0, tile_width - 1)
    max_tile_x = jnp.clip(max_tile_x, 0, tile_width - 1)
    min_tile_y = jnp.clip(min_tile_y, 0, tile_height - 1)
    max_tile_y = jnp.clip(max_tile_y, 0, tile_height - 1)
    tiles_per_gauss = (max_tile_x - min_tile_x + 1) * (
        max_tile_y - min_tile_y + 1
    )
    tiles_per_gauss = jnp.where(valid, tiles_per_gauss, 0)

    info: dict[str, Any] = {
        "camera_ids": None,
        "gaussian_ids": None,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "ray_transforms": ray_transforms,
        "opacities": projected_opacities,
        "normals": normals_camera,
        "render_normals_camera": rendered_normals_camera,
        "render_expected_depth": render_expected,
        "render_median_depth": render_median,
        "render_distort": render_distort,
        "valid": valid,
        "tiles_per_gauss": tiles_per_gauss,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "width": width,
        "height": height,
        "tile_size": config.tile_size,
        "n_batches": jnp.asarray(1, dtype=jnp.int32),
        "n_cameras": jnp.asarray(viewmats.shape[0], dtype=jnp.int32),
        "candidate_counts": tile_info["candidate_counts"],
        "tile_overflow": tile_info["tile_overflow"],
        "candidate_limit_exceeded": tile_info["candidate_limit_exceeded"],
        "intersection_count": tile_info["intersection_count"],
        "intersection_required_count": tile_info[
            "intersection_required_count"
        ],
        "intersection_overflow": tile_info["intersection_overflow"],
        "intersection_capacity": tile_info["intersection_capacity"],
        "candidate_ids": tile_info["candidate_ids"],
        "candidate_valid": tile_info["candidate_valid"],
        "active_count": jnp.count_nonzero(active_mask),
        "packed_requested": jnp.asarray(packed_metadata_requested),
        "packed_metadata_available": jnp.asarray(
            packed_metadata_available
        ),
        "sparse_grad_requested": sparse_grad_requested,
        "sparse_grad_is_dense": jnp.asarray(True),
        "absgrad_requested": absgrad_requested,
        "absgrad_available": jnp.asarray(absgrad_probe_enabled),
        "absgrad_probe_enabled": jnp.asarray(absgrad_probe_enabled),
        # Upstream exposes an independent forward-zero densification tensor.
        # JAX callers obtain its VJP through ``_gradient_2dgs_offset``.
        "gradient_2dgs": jnp.zeros_like(means2d),
    }
    if dense_metadata_requested and dense_intersection_metadata is not None:
        info.update(dense_intersection_metadata)
    if packed_metadata_available:
        assert dense_intersection_metadata is not None
        projection = _pack_projection_2dgs(
            radii,
            means2d,
            depths,
            ray_transforms,
            normals_camera,
        )
        info.update(
            _pack_dense_metadata_2dgs(
                projection,
                projected_opacities,
                dense_intersection_metadata,
            )
        )
    return (
        renders,
        alphas,
        rendered_normals,
        surface_normals,
        render_distort,
        render_median,
        info,
    )


def rasterization_2dgs_inria_wrapper(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    opacities: jax.Array,
    colors: jax.Array,
    viewmats: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: int | None = None,
    backgrounds: jax.Array | None = None,
    depth_ratio: float = 0.0,
    **kwargs: Any,
) -> tuple[tuple[jax.Array, jax.Array], dict[str, Any]]:
    """Expose the diff-surfel-rasterization/INRIA calling convention.

    The color result always has one appended depth channel.  ``depth_ratio=0``
    selects expected depth, ``1`` selects median depth, and intermediate values
    linearly blend the two, matching the reference wrapper.
    """

    kwargs.pop("render_mode", None)
    kwargs.pop("depth_mode", None)
    kwargs.setdefault("distloss", True)
    (
        render_colors,
        render_alphas,
        render_normals,
        _surface_normals,
        render_distort,
        render_median,
        info,
    ) = rasterization_2dgs(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
        near_plane=near_plane,
        far_plane=far_plane,
        eps2d=eps2d,
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        render_mode="RGB+ED",
        depth_mode="expected",
        **kwargs,
    )
    expected_depth = info["render_expected_depth"]
    blended_depth = (1.0 - depth_ratio) * expected_depth + depth_ratio * render_median
    render_colors = jnp.concatenate((render_colors[..., :-1], blended_depth), axis=-1)

    viewmats_array = jnp.asarray(viewmats)
    Ks_array = jnp.asarray(Ks)
    if viewmats_array.ndim == 2:
        viewmats_array = viewmats_array[None, ...]
    if Ks_array.ndim == 2:
        Ks_array = Ks_array[None, ...]
    surface_normals = _depth_to_surface_normals(
        blended_depth, viewmats_array, Ks_array
    )
    surface_normals = surface_normals * jax.lax.stop_gradient(render_alphas)

    meta = dict(info)
    meta.update(
        {
            "normals_rend": render_normals,
            "normals_surf": surface_normals,
            "render_distloss": render_distort,
        }
    )
    return (render_colors, render_alphas), meta


def accumulate_2dgs(
    means2d: jax.Array,
    ray_transforms: jax.Array,
    opacities: jax.Array,
    colors: jax.Array,
    normals: jax.Array,
    gaussian_ids: jax.Array,
    pixel_ids: jax.Array,
    image_ids: jax.Array,
    image_width: int,
    image_height: int,
    *,
    valid_count: jax.Array | int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pure-JAX alpha compositing for padded 2DGS pixel intersections."""

    image_shape = means2d.shape[:-2]
    image_count = math.prod(image_shape)
    gaussian_count = means2d.shape[-2]
    channels = colors.shape[-1]
    capacity = gaussian_ids.shape[0]
    if capacity == 0:
        return (
            jnp.zeros(
                image_shape + (image_height, image_width, channels), colors.dtype
            ),
            jnp.zeros(image_shape + (image_height, image_width, 1), opacities.dtype),
            jnp.zeros(image_shape + (image_height, image_width, 3), normals.dtype),
        )
    if valid_count is None:
        valid_count = capacity
    positions = jnp.arange(capacity, dtype=jnp.int32)
    gaussian_ids = jnp.asarray(gaussian_ids, dtype=jnp.int32)
    pixel_ids = jnp.asarray(pixel_ids, dtype=jnp.int32)
    image_ids = jnp.asarray(image_ids, dtype=jnp.int32)
    valid = (
        (positions < jnp.asarray(valid_count, jnp.int32))
        & (gaussian_ids >= 0)
        & (gaussian_ids < gaussian_count)
        & (pixel_ids >= 0)
        & (pixel_ids < image_width * image_height)
        & (image_ids >= 0)
        & (image_ids < image_count)
    )
    safe_gaussian = jnp.clip(gaussian_ids, 0, gaussian_count - 1)
    safe_pixel = jnp.clip(pixel_ids, 0, image_width * image_height - 1)
    safe_image = jnp.clip(image_ids, 0, image_count - 1)
    ray_ids = safe_image * image_width * image_height + safe_pixel
    order = jnp.argsort(
        jnp.where(valid, ray_ids, image_count * image_width * image_height),
        stable=True,
    )
    valid = valid[order]
    safe_gaussian = safe_gaussian[order]
    safe_pixel = safe_pixel[order]
    safe_image = safe_image[order]
    ray_ids = ray_ids[order]

    flat_means = means2d.reshape(image_count, gaussian_count, 2)
    flat_transforms = ray_transforms.reshape(image_count, gaussian_count, 3, 3)
    flat_opacities = opacities.reshape(image_count, gaussian_count)
    flat_colors = colors.reshape(image_count, gaussian_count, channels)
    flat_normals = normals.reshape(image_count, gaussian_count, 3)
    pixel_x = safe_pixel % image_width
    pixel_y = safe_pixel // image_width
    pixel_xf = pixel_x.astype(means2d.dtype) + 0.5
    pixel_yf = pixel_y.astype(means2d.dtype) + 0.5
    selected_means = flat_means[safe_image, safe_gaussian]
    transform = flat_transforms[safe_image, safe_gaussian]
    h_u = pixel_xf[:, None] * transform[:, 2, :] - transform[:, 0, :]
    h_v = pixel_yf[:, None] * transform[:, 2, :] - transform[:, 1, :]
    cross = jnp.cross(h_u, h_v, axis=-1)
    denominator = _safe_denominator(cross[:, 2])
    local_u = cross[:, 0] / denominator
    local_v = cross[:, 1] / denominator
    sigma_3d = local_u * local_u + local_v * local_v
    delta_x = pixel_xf - selected_means[:, 0]
    delta_y = pixel_yf - selected_means[:, 1]
    sigma_2d = 2.0 * (delta_x * delta_x + delta_y * delta_y)
    sigma = 0.5 * jnp.minimum(sigma_3d, sigma_2d)
    alpha = jnp.minimum(
        flat_opacities[safe_image, safe_gaussian] * jnp.exp(-sigma), 0.999
    )
    alpha = jnp.where(valid & jnp.isfinite(alpha), alpha, 0.0)
    sorted_ray_ids = jnp.where(
        valid, ray_ids, image_count * image_width * image_height
    )
    segment_starts = jnp.concatenate(
        (
            jnp.ones((1,), dtype=jnp.bool_),
            sorted_ray_ids[1:] != sorted_ray_ids[:-1],
        )
    )

    def segmented_product(left, right):
        left_value, left_start = left
        right_value, right_start = right
        return (
            jnp.where(right_start, right_value, left_value * right_value),
            left_start | right_start,
        )

    inclusive, _ = jax.lax.associative_scan(
        segmented_product, (1.0 - alpha, segment_starts)
    )
    previous = jnp.concatenate((jnp.ones_like(inclusive[:1]), inclusive[:-1]))
    transmittance = jnp.where(segment_starts, 1.0, previous)
    weights = jnp.where(valid, alpha * transmittance, 0.0)
    total_pixels = image_count * image_width * image_height
    rendered = jnp.zeros((total_pixels, channels), colors.dtype).at[ray_ids].add(
        weights[:, None] * flat_colors[safe_image, safe_gaussian]
    )
    accumulated = jnp.zeros((total_pixels,), opacities.dtype).at[ray_ids].add(
        weights
    )
    rendered_normals = jnp.zeros((total_pixels, 3), normals.dtype).at[ray_ids].add(
        weights[:, None] * flat_normals[safe_image, safe_gaussian]
    )
    return (
        rendered.reshape(image_shape + (image_height, image_width, channels)),
        accumulated.reshape(image_shape + (image_height, image_width, 1)),
        rendered_normals.reshape(image_shape + (image_height, image_width, 3)),
    )


def rasterize_to_indices_in_range_2dgs(
    range_start: int,
    range_end: int,
    transmittances: jax.Array,
    means2d: jax.Array,
    ray_transforms: jax.Array,
    opacities: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: jax.Array | PaddedOffsets,
    flatten_ids: jax.Array,
    *,
    max_intersections: int | None = None,
    valid_count: jax.Array | int | None = None,
    max_pair_evaluations: int = 8_000_000,
) -> PaddedRasterizationIndices:
    """Return an exact, statically padded list of 2DGS pixel hits."""

    flatten_ids = jnp.asarray(flatten_ids, dtype=jnp.int32)
    input_capacity = flatten_ids.shape[0]
    if max_intersections is None:
        max_intersections = max(1, input_capacity * tile_size * tile_size)
    if max_intersections <= 0:
        raise ValueError("max_intersections must be positive")
    if input_capacity == 0:
        invalid = jnp.full((max_intersections,), -1, dtype=jnp.int32)
        return PaddedRasterizationIndices(
            invalid,
            invalid,
            invalid,
            jnp.asarray(0, jnp.int32),
            jnp.asarray(False),
        )
    block_size = tile_size * tile_size
    pair_count = input_capacity * block_size
    if pair_count > max_pair_evaluations:
        raise MemoryError(
            f"2DGS reference indices would evaluate {pair_count} pairs; "
            "lower the padded intersection capacity or tile size"
        )
    offsets = (
        isect_offsets.offsets
        if isinstance(isect_offsets, PaddedOffsets)
        else jnp.asarray(isect_offsets)
    )
    if valid_count is None:
        valid_count = (
            isect_offsets.valid_count
            if isinstance(isect_offsets, PaddedOffsets)
            else input_capacity
        )
    image_shape = means2d.shape[:-2]
    image_count = math.prod(image_shape)
    gaussian_count = means2d.shape[-2]
    transmittances = jnp.asarray(transmittances, dtype=opacities.dtype)
    expected_transmittance_size = image_count * image_height * image_width
    if transmittances.size != expected_transmittance_size:
        raise ValueError(
            "transmittances must contain one value per image pixel"
        )
    flat_initial_transmittance = transmittances.reshape(-1)
    tile_height, tile_width = offsets.shape[-2:]
    tile_count = image_count * tile_height * tile_width
    positions = jnp.arange(input_capacity, dtype=jnp.int32)
    offsets_flat = offsets.reshape(-1)
    tile_global = jnp.searchsorted(offsets_flat, positions, side="right") - 1
    tile_global = jnp.clip(tile_global, 0, tile_count - 1)
    tile_start = offsets_flat[tile_global]
    local_index = positions - tile_start
    in_range = (local_index >= range_start * block_size) & (
        local_index < range_end * block_size
    )
    image_id = tile_global // (tile_height * tile_width)
    tile_id = tile_global % (tile_height * tile_width)
    tile_y = tile_id // tile_width
    tile_x = tile_id % tile_width
    flat_id = jnp.clip(flatten_ids, 0, image_count * gaussian_count - 1)
    gaussian_id = flat_id % gaussian_count
    flat_image_id = flat_id // gaussian_count
    local_pixels = jnp.arange(block_size, dtype=jnp.int32)
    local_y = local_pixels // tile_size
    local_x = local_pixels % tile_size
    pixel_x = tile_x[:, None] * tile_size + local_x[None, :]
    pixel_y = tile_y[:, None] * tile_size + local_y[None, :]
    pixel_valid = (pixel_x < image_width) & (pixel_y < image_height)
    base_valid = (
        (positions < jnp.asarray(valid_count, jnp.int32))
        & (flatten_ids >= 0)
        & (flat_image_id == image_id)
        & in_range
    )
    flat_means = means2d.reshape(image_count, gaussian_count, 2)
    flat_transforms = ray_transforms.reshape(image_count, gaussian_count, 3, 3)
    flat_opacity = opacities.reshape(image_count, gaussian_count)
    selected_means = flat_means[image_id, gaussian_id]
    transforms = flat_transforms[image_id, gaussian_id]
    pixel_xf = pixel_x.astype(means2d.dtype) + 0.5
    pixel_yf = pixel_y.astype(means2d.dtype) + 0.5
    h_u = (
        pixel_xf[..., None] * transforms[:, None, 2, :]
        - transforms[:, None, 0, :]
    )
    h_v = (
        pixel_yf[..., None] * transforms[:, None, 2, :]
        - transforms[:, None, 1, :]
    )
    cross = jnp.cross(h_u, h_v, axis=-1)
    denominator = _safe_denominator(cross[..., 2])
    sigma_3d = (cross[..., 0] / denominator) ** 2 + (
        cross[..., 1] / denominator
    ) ** 2
    dx = pixel_xf - selected_means[:, None, 0]
    dy = pixel_yf - selected_means[:, None, 1]
    sigma = 0.5 * jnp.minimum(sigma_3d, 2.0 * (dx * dx + dy * dy))
    alpha = jnp.minimum(
        flat_opacity[image_id, gaussian_id, None] * jnp.exp(-sigma), 0.999
    )
    pair_valid = (
        base_valid[:, None]
        & pixel_valid
        & jnp.isfinite(sigma)
        & jnp.isfinite(alpha)
        & (sigma >= 0.0)
        & (alpha >= 1.0 / 255.0)
    ).reshape(-1)
    pair_alpha = jnp.where(pair_valid, alpha.reshape(-1), 0.0)
    pair_ray_ids = (
        image_id[:, None] * (image_width * image_height)
        + pixel_y * image_width
        + pixel_x
    ).reshape(-1)
    total_rays = image_count * image_width * image_height
    pair_order = jnp.argsort(
        jnp.where(pair_valid, pair_ray_ids, total_rays), stable=True
    )
    sorted_valid = pair_valid[pair_order]
    sorted_alpha = pair_alpha[pair_order]
    sorted_rays = jnp.clip(pair_ray_ids[pair_order], 0, total_rays - 1)
    segment_starts = jnp.concatenate(
        (
            jnp.ones((1,), dtype=jnp.bool_),
            sorted_rays[1:] != sorted_rays[:-1],
        )
    )

    def segmented_product(left, right):
        left_value, left_start = left
        right_value, right_start = right
        return (
            jnp.where(right_start, right_value, left_value * right_value),
            left_start | right_start,
        )

    inclusive, _ = jax.lax.associative_scan(
        segmented_product, (1.0 - sorted_alpha, segment_starts)
    )
    exclusive = jnp.concatenate((jnp.ones_like(inclusive[:1]), inclusive[:-1]))
    exclusive = jnp.where(segment_starts, 1.0, exclusive)
    current_transmittance = flat_initial_transmittance[sorted_rays] * exclusive
    sorted_accepted = (
        sorted_valid
        & (
            current_transmittance * (1.0 - sorted_alpha)
            > 1.0e-4
        )
    )
    valid_pairs = jnp.zeros_like(sorted_accepted).at[pair_order].set(sorted_accepted)
    selected_pairs = jnp.nonzero(valid_pairs, size=max_intersections, fill_value=0)[0]
    exact_count = jnp.count_nonzero(valid_pairs)
    output_valid = jnp.arange(max_intersections) < exact_count
    intersection_id = selected_pairs // block_size
    local_pixel_id = selected_pairs % block_size
    output_gaussian = gaussian_id[intersection_id]
    output_image = image_id[intersection_id]
    output_pixel_x = tile_x[intersection_id] * tile_size + local_pixel_id % tile_size
    output_pixel_y = tile_y[intersection_id] * tile_size + local_pixel_id // tile_size
    output_pixel = output_pixel_y * image_width + output_pixel_x
    return PaddedRasterizationIndices(
        jnp.where(output_valid, output_gaussian, -1).astype(jnp.int32),
        jnp.where(output_valid, output_pixel, -1).astype(jnp.int32),
        jnp.where(output_valid, output_image, -1).astype(jnp.int32),
        jnp.minimum(exact_count, max_intersections).astype(jnp.int32),
        exact_count > max_intersections,
    )


def rasterize_to_pixels_2dgs(
    means2d: jax.Array,
    ray_transforms: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    normals: jax.Array,
    densify: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: jax.Array,
    flatten_ids: jax.Array,
    backgrounds: jax.Array | None = None,
    masks: jax.Array | None = None,
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    *,
    max_gaussians_per_tile: int = 512,
    tile_batch_size: int = 1,
    return_info: bool = False,
    _densify_absgrad_probe: jax.Array | None = None,
):
    """Low-level 2DGS compatibility renderer with bounded tile workspace.

    The supplied padded ``isect_offsets`` and ``flatten_ids`` determine the
    exact tile candidate order. Invalid padded ids are ignored.
    """

    densify_absgrad_probe_enabled = _densify_absgrad_probe is not None
    if densify_absgrad_probe_enabled and not absgrad:
        raise ValueError("_densify_absgrad_probe requires absgrad=True")
    means2d = jnp.asarray(means2d)
    densify_values = jnp.asarray(densify)
    if densify_values.shape != means2d.shape:
        raise ValueError("densify must have the same shape as means2d")
    if densify_values.dtype != means2d.dtype:
        raise ValueError("densify must have the same dtype as means2d")
    if _densify_absgrad_probe is None:
        densify_absgrad_values = jnp.zeros_like(densify_values)
    else:
        densify_absgrad_values = jnp.asarray(_densify_absgrad_probe)
        if densify_absgrad_values.shape != means2d.shape:
            raise ValueError(
                "_densify_absgrad_probe must have the same shape as means2d"
            )
        if densify_absgrad_values.dtype != means2d.dtype:
            raise ValueError(
                "_densify_absgrad_probe must have the same dtype as means2d"
            )
    tile_height = (image_height + tile_size - 1) // tile_size
    tile_width = (image_width + tile_size - 1) // tile_size
    offsets = jnp.asarray(isect_offsets, dtype=jnp.int32)
    if offsets.ndim < 2 or offsets.shape[-2:] != (tile_height, tile_width):
        raise ValueError("isect_offsets must cover the image tile grid")

    if packed:
        if means2d.ndim != 2 or means2d.shape[-1] != 2:
            raise ValueError("packed means2d must have shape [P, 2]")
        packed_count = means2d.shape[0]
        if ray_transforms.shape != (packed_count, 3, 3):
            raise ValueError("packed ray_transforms must have shape [P, 3, 3]")
        if colors.ndim != 2 or colors.shape[0] != packed_count:
            raise ValueError("packed colors must have shape [P, channels]")
        if opacities.shape != (packed_count,):
            raise ValueError("packed opacities must have shape [P]")
        if normals.shape != (packed_count, 3):
            raise ValueError("packed normals must have shape [P, 3]")
        image_shape = offsets.shape[:-2]
        gaussian_count = packed_count
        transforms = ray_transforms
        projected_means = means2d
        color_values = colors
        opacity_values = opacities
        normal_values = normals
        densify_probes = densify_values
        densify_absgrad_probes = densify_absgrad_values
    else:
        if means2d.ndim < 3 or means2d.shape[-1] != 2:
            raise ValueError("means2d must have shape [..., N, 2]")
        image_shape = means2d.shape[:-2]
        gaussian_count = means2d.shape[-2]
        if offsets.shape != image_shape + (tile_height, tile_width):
            raise ValueError(
                "isect_offsets must match the image batch and cover the tile grid"
            )
        transforms = ray_transforms.reshape(
            math.prod(image_shape), gaussian_count, 3, 3
        )
        projected_means = means2d.reshape(
            math.prod(image_shape), gaussian_count, 2
        )
        color_values = colors.reshape(
            math.prod(image_shape), gaussian_count, colors.shape[-1]
        )
        opacity_values = opacities.reshape(
            math.prod(image_shape), gaussian_count
        )
        normal_values = normals.reshape(
            math.prod(image_shape), gaussian_count, 3
        )
        densify_probes = densify_values.reshape(
            math.prod(image_shape), gaussian_count, 2
        )
        densify_absgrad_probes = densify_absgrad_values.reshape(
            math.prod(image_shape), gaussian_count, 2
        )

    image_count = math.prod(image_shape)
    supplied_ids = jnp.asarray(flatten_ids, dtype=jnp.int32)
    if supplied_ids.ndim != 1:
        raise ValueError("flatten_ids must have shape [K]")
    if supplied_ids.shape[0] == 0:
        supplied_ids = jnp.full((1,), -1, dtype=jnp.int32)
        supplied_capacity = 0
    else:
        supplied_capacity = supplied_ids.shape[0]
    supplied_valid_count = jnp.count_nonzero(supplied_ids >= 0)
    flat_offsets = offsets.reshape(image_count, tile_height * tile_width)
    intersection_slots = jnp.arange(supplied_ids.shape[0], dtype=jnp.int32)
    row_u = transforms[..., 0, :]
    row_v = transforms[..., 1, :]
    row_w = transforms[..., 2, :]
    signature = jnp.asarray((1.0, 1.0, -1.0), means2d.dtype)
    distance = jnp.sum(signature * row_w * row_w, axis=-1)
    factors = signature / jnp.where(
        jnp.abs(distance) > 1.0e-8, distance, 1.0
    )[..., None]
    extent_sq = jnp.stack(
        (
            projected_means[..., 0] ** 2
            - jnp.sum(factors * row_u * row_u, axis=-1),
            projected_means[..., 1] ** 2
            - jnp.sum(factors * row_v * row_v, axis=-1),
        ),
        axis=-1,
    )
    radii = jnp.ceil(
        3.33 * jnp.sqrt(jnp.maximum(extent_sq, 1.0e-4))
    ).astype(jnp.int32)
    depths = transforms[..., 2, 2]
    valid = (
        (jnp.abs(distance) > 1.0e-8)
        & (depths > 0.0)
        & (opacity_values > 0.0)
        & jnp.all(jnp.isfinite(projected_means), axis=-1)
        & jnp.all(jnp.isfinite(radii), axis=-1)
    )
    if backgrounds is None:
        background_values = jnp.zeros(
            (image_count, colors.shape[-1]), colors.dtype
        )
    else:
        background_values = jnp.broadcast_to(
            jnp.asarray(backgrounds), image_shape + (colors.shape[-1],)
        ).reshape(image_count, colors.shape[-1])
    config = RasterizationConfig(
        tile_size=tile_size,
        max_gaussians_per_tile=max_gaussians_per_tile,
        tile_batch_size=tile_batch_size,
    )

    def call_one(index):
        image_start = flat_offsets[index, 0]
        next_image_start = flat_offsets[
            jnp.minimum(index + 1, image_count - 1), 0
        ]
        image_end = jnp.where(
            index + 1 < image_count, next_image_start, supplied_valid_count
        )
        image_count_intersections = jnp.maximum(image_end - image_start, 0)
        source_positions = image_start + intersection_slots
        safe_positions = jnp.clip(
            source_positions, 0, supplied_ids.shape[0] - 1
        )
        global_ids = supplied_ids[safe_positions]
        local_valid = (
            (intersection_slots < image_count_intersections)
            & (source_positions < supplied_valid_count)
            & (global_ids >= 0)
        )
        if packed:
            local_valid = local_valid & (global_ids < gaussian_count)
            local_ids = jnp.where(local_valid, global_ids, -1)
            means_for_image = projected_means
            transforms_for_image = transforms
            normals_for_image = normal_values
            opacities_for_image = opacity_values
            colors_for_image = color_values
            densify_for_image = densify_probes
            densify_absgrad_for_image = densify_absgrad_probes
        else:
            gaussian_base = index * gaussian_count
            local_valid = local_valid & (
                (global_ids >= gaussian_base)
                & (global_ids < gaussian_base + gaussian_count)
            )
            local_ids = jnp.where(local_valid, global_ids - gaussian_base, -1)
            means_for_image = projected_means[index]
            transforms_for_image = transforms[index]
            normals_for_image = normal_values[index]
            opacities_for_image = opacity_values[index]
            colors_for_image = color_values[index]
            densify_for_image = densify_probes[index]
            densify_absgrad_for_image = densify_absgrad_probes[index]
        local_offsets = (flat_offsets[index] - image_start).reshape(
            tile_height, tile_width
        )
        supplied_intersections = TileIntersections(
            gaussian_ids=local_ids,
            tile_ids=jnp.full_like(local_ids, -1),
            offsets=local_offsets,
            valid_count=jnp.minimum(
                image_count_intersections, jnp.asarray(supplied_capacity, jnp.int32)
            ),
            overflow=jnp.asarray(False),
            required_count=image_count_intersections,
        )
        return _render_camera_tiles_2dgs(
            means_for_image,
            radii if packed else radii[index],
            depths if packed else depths[index],
            transforms_for_image,
            normals_for_image,
            opacities_for_image,
            colors_for_image,
            valid if packed else valid[index],
            width=image_width,
            height=image_height,
            config=config,
            background=background_values[index],
            render_mode="RGB",
            distloss=distloss,
            densify_probe=densify_for_image,
            densify_absgrad_probe=(
                densify_absgrad_for_image
                if densify_absgrad_probe_enabled
                else None
            ),
            precomputed_intersections=supplied_intersections,
        )

    if image_count == 1:
        single_output = call_one(jnp.asarray(0, dtype=jnp.int32))
        outputs = jax.tree.map(lambda value: value[None, ...], single_output)
    else:
        outputs = jax.lax.map(
            call_one, jnp.arange(image_count, dtype=jnp.int32), batch_size=1
        )
    rendered, alpha, rendered_normals, distortion, median, _, info = outputs
    if masks is not None:
        tile_mask = jnp.asarray(masks, dtype=jnp.bool_).reshape(
            image_count, tile_height, tile_width
        )
        pixel_mask = jnp.repeat(
            jnp.repeat(tile_mask, tile_size, axis=-2), tile_size, axis=-1
        )[:, :image_height, :image_width, None]
        background_pixels = background_values[:, None, None, :]
        rendered = jnp.where(pixel_mask, rendered, background_pixels)
        alpha = jnp.where(pixel_mask, alpha, 0.0)
        rendered_normals = jnp.where(pixel_mask, rendered_normals, 0.0)
        distortion = jnp.where(pixel_mask, distortion, 0.0)
        median = jnp.where(pixel_mask, median, 0.0)
    output_shape = image_shape + (image_height, image_width)
    result = (
        rendered.reshape(output_shape + (colors.shape[-1],)),
        alpha.reshape(output_shape + (1,)),
        rendered_normals.reshape(output_shape + (3,)),
        distortion.reshape(output_shape + (1,)),
        median.reshape(output_shape + (1,)),
    )
    return result + (info,) if return_info else result


__all__ = [
    "PaddedProjection2DGS",
    "accumulate_2dgs",
    "fully_fused_projection_2dgs",
    "rasterization_2dgs",
    "rasterization_2dgs_inria_wrapper",
    "rasterize_to_indices_in_range_2dgs",
    "rasterize_to_pixels_2dgs",
]
