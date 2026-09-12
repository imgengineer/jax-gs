from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jax_gs.low_level import isect_offset_encode, isect_tiles
from jax_gs.two_dgs import (
    PaddedProjection2DGS,
    accumulate_2dgs,
    fully_fused_projection_2dgs,
    rasterize_to_indices_in_range_2dgs,
    rasterize_to_pixels_2dgs,
)

pytestmark = pytest.mark.resource_heavy


def test_packed_projection_is_static_jittable_and_preserves_ids():
    means = jnp.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]], dtype=jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=jnp.float32)
    scales = jnp.asarray([[0.2, 0.2, 0.01]] * 2, dtype=jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.asarray(
        [[[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )

    project = jax.jit(
        lambda value: fully_fused_projection_2dgs(
            value, quats, scales, viewmats, Ks, 8, 8, packed=True
        )
    )
    packed = project(means)

    assert isinstance(packed, PaddedProjection2DGS)
    assert len(tuple(packed)) == 9
    assert packed.indptr.dtype == jnp.int32
    assert packed.indptr.shape == (2,)
    assert jnp.array_equal(packed.indptr, jnp.asarray([0, 1], jnp.int32))
    assert packed.radii.shape == (2, 2)
    assert int(packed.valid_count) == 1
    assert not bool(packed.overflow)
    assert int(packed.batch_ids[0]) == 0
    assert int(packed.camera_ids[0]) == 0
    assert int(packed.gaussian_ids[0]) == 0
    assert int(packed.gaussian_ids[1]) == -1
    assert project._cache_size() == 1


def test_packed_projection_indptr_tracks_batch_camera_groups_and_pytree():
    means = jnp.asarray(
        [
            [[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, 2.0], [0.1, 0.0, 3.0]],
        ],
        dtype=jnp.float32,
    )
    quats = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0], jnp.float32), (2, 2, 4))
    scales = jnp.full((2, 2, 3), 0.2, dtype=jnp.float32)
    viewmats = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (2, 3, 4, 4))
    viewmats = viewmats.at[:, 1, 2, 3].set(-4.0)
    Ks = jnp.broadcast_to(
        jnp.asarray(
            [[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        ),
        (2, 3, 3, 3),
    )
    active_mask = jnp.asarray([[True, False], [False, True]])

    project = jax.jit(
        lambda mask: fully_fused_projection_2dgs(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            8,
            8,
            packed=True,
            active_mask=mask,
        )
    )
    packed = project(active_mask)

    public_values = tuple(packed)
    assert len(public_values) == 9
    for actual, expected in zip(
        public_values,
        (
            packed.batch_ids,
            packed.camera_ids,
            packed.gaussian_ids,
            packed.indptr,
            packed.radii,
            packed.means2d,
            packed.depths,
            packed.ray_transforms,
            packed.normals,
        ),
        strict=True,
    ):
        assert jnp.array_equal(actual, expected)

    expected_counts = jnp.asarray([1, 0, 1, 1, 0, 1], jnp.int32)
    expected_indptr = jnp.asarray([0, 1, 1, 2, 3, 3, 4], jnp.int32)
    assert packed.indptr.dtype == jnp.int32
    assert packed.indptr.shape == (7,)
    assert jnp.array_equal(jnp.diff(packed.indptr), expected_counts)
    assert jnp.array_equal(packed.indptr, expected_indptr)
    valid_count = int(packed.valid_count)
    assert int(packed.indptr[-1]) == valid_count == 4

    capacity = 2 * 3 * 2
    assert packed.radii.shape == (capacity, 2)
    assert jnp.all(packed.batch_ids[valid_count:] == -1)
    assert jnp.all(packed.camera_ids[valid_count:] == -1)
    assert jnp.all(packed.gaussian_ids[valid_count:] == -1)
    assert jnp.all(packed.radii[valid_count:] == 0)

    leaves, tree = jax.tree.flatten(packed)
    restored = jax.tree.unflatten(tree, leaves)
    assert isinstance(restored, PaddedProjection2DGS)
    assert jnp.array_equal(restored.indptr, expected_indptr)
    assert int(restored.valid_count) == 4

    empty = project(jnp.zeros_like(active_mask))
    assert jnp.array_equal(empty.indptr, jnp.zeros((7,), jnp.int32))
    assert int(empty.indptr[-1]) == int(empty.valid_count) == 0
    assert project._cache_size() == 1


def test_packed_low_level_pixels_match_dense_layout():
    means = jnp.asarray([[0.0, 0.0, 2.0]], dtype=jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    scales = jnp.asarray([[0.2, 0.2, 0.01]], dtype=jnp.float32)
    viewmats = jnp.stack((jnp.eye(4, dtype=jnp.float32), jnp.eye(4, dtype=jnp.float32)))
    viewmats = viewmats.at[1, 0, 3].set(0.05)
    Ks = jnp.broadcast_to(
        jnp.asarray(
            [[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        ),
        (2, 3, 3),
    )
    dense = fully_fused_projection_2dgs(means, quats, scales, viewmats, Ks, 8, 8)
    packed = fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, 8, 8, packed=True
    )
    assert isinstance(packed, PaddedProjection2DGS)
    radii, means2d, depths, transforms, normals = dense
    colors = jnp.asarray([[[1.0, 0.25, 0.5]], [[0.1, 0.75, 0.3]]], dtype=jnp.float32)
    opacities = jnp.asarray([[0.8], [0.6]], dtype=jnp.float32)
    tile_size = 4
    tile_width = tile_height = 2

    dense_intersections = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        max_intersections=16,
    )
    dense_offsets = isect_offset_encode(
        dense_intersections.isect_ids,
        2,
        tile_width,
        tile_height,
        valid_count=dense_intersections.valid_count,
    )
    dense_outputs = rasterize_to_pixels_2dgs(
        means2d,
        transforms,
        colors,
        opacities,
        normals,
        means2d,
        8,
        8,
        tile_size,
        dense_offsets,
        dense_intersections.flatten_ids,
        max_gaussians_per_tile=2,
    )

    packed_valid = jnp.arange(packed.radii.shape[0]) < packed.valid_count
    packed_intersections = isect_tiles(
        packed.means2d,
        packed.radii,
        packed.depths,
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_images=2,
        image_ids=packed.camera_ids,
        active_mask=packed_valid,
        max_intersections=16,
    )
    packed_offsets = isect_offset_encode(
        packed_intersections.isect_ids,
        2,
        tile_width,
        tile_height,
        valid_count=packed_intersections.valid_count,
    )
    safe_cameras = jnp.clip(packed.camera_ids, 0, 1)
    safe_gaussians = jnp.clip(packed.gaussian_ids, 0, 0)
    packed_colors = jnp.where(
        packed_valid[:, None], colors[safe_cameras, safe_gaussians], 0.0
    )
    packed_opacities = jnp.where(
        packed_valid, opacities[safe_cameras, safe_gaussians], 0.0
    )
    packed_outputs = rasterize_to_pixels_2dgs(
        packed.means2d,
        packed.ray_transforms,
        packed_colors,
        packed_opacities,
        packed.normals,
        packed.means2d,
        8,
        8,
        tile_size,
        packed_offsets,
        packed_intersections.flatten_ids,
        packed=True,
        max_gaussians_per_tile=2,
    )

    for packed_value, dense_value in zip(packed_outputs, dense_outputs):
        assert jnp.allclose(packed_value, dense_value, rtol=1.0e-5, atol=1.0e-6)


def test_low_level_2dgs_padded_indices_accumulate_and_pixels():
    means = jnp.asarray([[0.0, 0.0, 2.0]], dtype=jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    scales = jnp.asarray([[0.2, 0.2, 0.01]], dtype=jnp.float32)
    opacities = jnp.asarray([0.8], dtype=jnp.float32)
    colors = jnp.asarray([[1.0, 0.25, 0.5]], dtype=jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.asarray(
        [[[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )
    radii, means2d, depths, transforms, normals = fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, 8, 8
    )
    intersections = isect_tiles(means2d, radii, depths, 4, 2, 2, max_intersections=4)
    offsets = isect_offset_encode(
        intersections.isect_ids,
        1,
        2,
        2,
        valid_count=intersections.valid_count,
        return_info=True,
    )
    pixel_outputs = rasterize_to_pixels_2dgs(
        means2d,
        transforms,
        colors[None],
        opacities[None],
        normals,
        means2d,
        8,
        8,
        4,
        offsets.offsets,
        intersections.flatten_ids,
        max_gaussians_per_tile=2,
    )
    assert pixel_outputs[0].shape == (1, 8, 8, 3)
    assert float(pixel_outputs[1][0, 4, 4, 0]) > 0.0
    empty_pixels = rasterize_to_pixels_2dgs(
        means2d,
        transforms,
        colors[None],
        opacities[None],
        normals,
        means2d,
        8,
        8,
        4,
        offsets.offsets,
        jnp.full_like(intersections.flatten_ids, -1),
        backgrounds=jnp.asarray([[0.1, 0.2, 0.3]], jnp.float32),
        max_gaussians_per_tile=2,
    )
    assert jnp.allclose(empty_pixels[0][0, 4, 4], jnp.asarray([0.1, 0.2, 0.3]))
    assert float(empty_pixels[1][0, 4, 4, 0]) == 0.0

    indices = rasterize_to_indices_in_range_2dgs(
        0,
        10,
        jnp.ones((1, 8, 8), jnp.float32),
        means2d,
        transforms,
        opacities[None],
        8,
        8,
        4,
        offsets,
        intersections.flatten_ids,
        max_intersections=128,
        valid_count=intersections.valid_count,
    )
    rendered, alpha, rendered_normals = accumulate_2dgs(
        means2d,
        transforms,
        opacities[None],
        colors[None],
        normals,
        indices.gaussian_ids,
        indices.pixel_ids,
        indices.image_ids,
        8,
        8,
        valid_count=indices.valid_count,
    )
    assert rendered.shape == (1, 8, 8, 3)
    assert alpha.shape == (1, 8, 8, 1)
    assert rendered_normals.shape == (1, 8, 8, 3)
    assert float(alpha[0, 4, 4, 0]) > 0.0

    default_capacity = rasterize_to_indices_in_range_2dgs(
        0,
        10,
        jnp.ones((1, 8, 8), jnp.float32),
        means2d,
        transforms,
        opacities[None],
        8,
        8,
        4,
        offsets,
        intersections.flatten_ids,
        valid_count=intersections.valid_count,
    )
    assert default_capacity.gaussian_ids.shape == (
        intersections.flatten_ids.shape[0] * 16,
    )
    assert int(default_capacity.valid_count) == int(indices.valid_count)

    blocked = rasterize_to_indices_in_range_2dgs(
        0,
        10,
        jnp.full((1, 8, 8), 1.0e-5, jnp.float32),
        means2d,
        transforms,
        opacities[None],
        8,
        8,
        4,
        offsets,
        intersections.flatten_ids,
        max_intersections=128,
        valid_count=intersections.valid_count,
    )
    assert int(blocked.valid_count) == 0


def test_low_level_2dgs_densify_probe_matches_ray_transform_vjp():
    means = jnp.asarray([[0.0, 0.0, 2.0]], dtype=jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    scales = jnp.asarray([[0.5, 0.5, 0.01]], dtype=jnp.float32)
    opacities = jnp.asarray([[0.8]], dtype=jnp.float32)
    colors = jnp.asarray([[[1.0, 0.25, 0.5]]], dtype=jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.asarray(
        [[[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )
    radii, means2d, depths, transforms, normals = fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, 8, 8
    )
    intersections = isect_tiles(means2d, radii, depths, 4, 2, 2, max_intersections=4)
    offsets = isect_offset_encode(
        intersections.isect_ids,
        1,
        2,
        2,
        valid_count=intersections.valid_count,
    )
    color_cotangent = jnp.linspace(0.1, 1.0, 8 * 8 * 3).reshape(1, 8, 8, 3)
    alpha_cotangent = jnp.linspace(0.2, 0.9, 8 * 8).reshape(1, 8, 8, 1)

    def loss(current_transforms, densify):
        rendered, alpha, *_ = rasterize_to_pixels_2dgs(
            means2d,
            current_transforms,
            colors,
            opacities,
            normals,
            densify,
            8,
            8,
            4,
            offsets,
            intersections.flatten_ids,
            max_gaussians_per_tile=1,
        )
        return jnp.sum(rendered * color_cotangent) + jnp.sum(alpha * alpha_cotangent)

    transform_gradient, densify_gradient = jax.jit(jax.grad(loss, argnums=(0, 1)))(
        transforms, jnp.zeros_like(means2d)
    )
    expected_densify = jnp.stack(
        (
            transform_gradient[..., 0, 2] * transforms[..., 2, 2],
            transform_gradient[..., 1, 2] * transforms[..., 2, 2],
        ),
        axis=-1,
    )

    assert jnp.any(jnp.abs(expected_densify) > 1.0e-6)
    assert jnp.allclose(densify_gradient, expected_densify, rtol=1.0e-5, atol=1.0e-6)
    assert loss(transforms, jnp.zeros_like(means2d)) == loss(
        transforms, jnp.ones_like(means2d)
    )
