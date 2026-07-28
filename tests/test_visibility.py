import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.low_level import (
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)
from jax_gs.sparse import build_sparse_tile_layout, isect_tiles_sparse
from jax_gs.visibility import (
    PaddedContributors,
    rasterize_contributing_gaussian_ids,
    rasterize_contributing_gaussian_ids_sparse,
    rasterize_num_contributing_gaussians,
    rasterize_num_contributing_gaussians_sparse,
    rasterize_top_contributing_gaussian_ids,
    rasterize_top_contributing_gaussian_ids_sparse,
)


def _one_pixel_visibility_scene(opacities):
    opacities = jnp.asarray(opacities, jnp.float32)[None]
    gaussian_count = opacities.shape[1]
    means2d = jnp.full((1, gaussian_count, 2), 0.5, jnp.float32)
    conics = jnp.zeros((1, gaussian_count, 3), jnp.float32)
    radii = jnp.ones((1, gaussian_count, 2), jnp.int32)
    depths = jnp.arange(1, gaussian_count + 1, dtype=jnp.float32)[None]
    pixels = jnp.asarray([[0, 0]], jnp.int32)
    pixel_image_ids = jnp.asarray([0], jnp.int32)
    layout = build_sparse_tile_layout(
        pixels, pixel_image_ids, 1, 4, 1, 1
    )
    intersections = isect_tiles_sparse(
        means2d,
        radii,
        depths,
        layout.active_tile_mask,
        layout.active_tiles,
        1,
        4,
        1,
        1,
        active_tile_count=layout.valid_count,
    )
    return means2d, conics, opacities, layout, intersections


def _random_scene():
    rng = np.random.default_rng(71)
    image_count, gaussian_count = 2, 7
    means2d = jnp.asarray(
        rng.uniform([0.0, 0.0], [9.0, 7.0], (image_count, gaussian_count, 2)),
        jnp.float32,
    )
    inverse_scale = rng.uniform(0.08, 0.24, (image_count, gaussian_count))
    conics = jnp.asarray(
        np.stack(
            (inverse_scale, np.zeros_like(inverse_scale), inverse_scale),
            axis=-1,
        ),
        jnp.float32,
    )
    opacities = jnp.asarray(
        rng.uniform(0.1, 0.85, (image_count, gaussian_count)), jnp.float32
    )
    radii = jnp.asarray(
        rng.integers(2, 6, (image_count, gaussian_count, 2)), jnp.int32
    )
    depths = jnp.asarray(
        np.stack(
            (
                np.arange(gaussian_count) + 0.25,
                np.arange(gaussian_count) + 0.75,
            )
        ),
        jnp.float32,
    )
    pixels = jnp.asarray(
        [[0, 0], [2, 5], [6, 8], [3, 1], [1, 7], [5, 4], [4, 0]],
        jnp.int32,
    )
    image_ids = jnp.asarray([0, 1, 0, 1, 1, 0, 0], jnp.int32)
    return means2d, conics, opacities, radii, depths, pixels, image_ids


def _random_layout(scene, *, packed):
    means2d, _conics, _opacities, radii, depths, pixels, image_ids = scene
    layout = build_sparse_tile_layout(pixels, image_ids, 2, 4, 3, 2)
    if packed:
        gaussian_image_ids = jnp.repeat(
            jnp.arange(2, dtype=jnp.int32), means2d.shape[1]
        )
        intersections = isect_tiles_sparse(
            means2d.reshape(-1, 2),
            radii.reshape(-1, 2),
            depths.reshape(-1),
            layout.active_tile_mask,
            layout.active_tiles,
            2,
            4,
            3,
            2,
            image_ids=gaussian_image_ids,
            active_tile_count=layout.valid_count,
        )
    else:
        intersections = isect_tiles_sparse(
            means2d,
            radii,
            depths,
            layout.active_tile_mask,
            layout.active_tiles,
            2,
            4,
            3,
            2,
            active_tile_count=layout.valid_count,
        )
    return layout, intersections


def _dense_intersections(scene, *, packed):
    means2d, _conics, _opacities, radii, depths, _pixels, _image_ids = scene
    capacity = 2 * means2d.shape[1] * 3 * 2
    if packed:
        gaussian_image_ids = jnp.repeat(
            jnp.arange(2, dtype=jnp.int32), means2d.shape[1]
        )
        intersections = isect_tiles(
            means2d.reshape(-1, 2),
            radii.reshape(-1, 2),
            depths.reshape(-1),
            4,
            3,
            2,
            packed=True,
            n_images=2,
            image_ids=gaussian_image_ids,
            max_intersections=capacity,
        )
    else:
        intersections = isect_tiles(
            means2d,
            radii,
            depths,
            4,
            3,
            2,
            n_images=2,
            max_intersections=capacity,
        )
    offsets = isect_offset_encode(
        intersections.isect_ids,
        2,
        3,
        2,
        valid_count=intersections.valid_count,
    )
    return offsets, intersections


def test_sparse_visibility_exact_weights_top_selection_and_padding():
    means2d, conics, opacities, layout, intersections = (
        _one_pixel_visibility_scene([0.1, 0.8, 0.5, 0.001])
    )
    common = (
        means2d,
        conics,
        opacities,
        layout.active_tiles,
        intersections.tile_offsets,
        intersections.flatten_ids,
        layout.tile_pixel_mask,
        layout.tile_pixel_cumsum,
        layout.pixel_map,
    )
    counts, alphas = rasterize_num_contributing_gaussians_sparse(
        *common,
        1,
        1,
        4,
        1,
        1,
        active_tile_count=layout.valid_count,
        valid_count=intersections.valid_count,
    )
    np.testing.assert_array_equal(counts, [3])
    np.testing.assert_allclose(alphas, [0.91], rtol=1e-6, atol=1e-6)

    contributors = jax.jit(
        lambda: rasterize_contributing_gaussian_ids_sparse(
            *common,
            counts,
            1,
            1,
            4,
            1,
            1,
            active_tile_count=layout.valid_count,
            valid_count=intersections.valid_count,
        )
    )()
    assert isinstance(contributors, PaddedContributors)
    ids, weights = contributors
    np.testing.assert_array_equal(ids, [[0, 1, 2, -1]])
    np.testing.assert_allclose(
        weights, [[0.1, 0.72, 0.09, 0.0]], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(contributors.valid_counts, [3])
    assert int(contributors.required_count) == 3
    assert not bool(contributors.overflow)

    top_ids, top_weights = rasterize_top_contributing_gaussian_ids_sparse(
        *common,
        1,
        1,
        4,
        1,
        1,
        2,
        active_tile_count=layout.valid_count,
        valid_count=intersections.valid_count,
    )
    np.testing.assert_array_equal(top_ids, [[0, 1]])
    np.testing.assert_allclose(top_weights, [[0.1, 0.72]], rtol=1e-6, atol=1e-6)

    truncated = rasterize_contributing_gaussian_ids_sparse(
        *common,
        counts,
        1,
        1,
        4,
        1,
        1,
        active_tile_count=layout.valid_count,
        valid_count=intersections.valid_count,
        max_contributors=1,
    )
    np.testing.assert_array_equal(truncated.gaussian_ids, [[0]])
    assert int(truncated.required_count) == 3
    assert bool(truncated.overflow)


def test_sparse_visibility_excludes_threshold_crossing_sample():
    means2d, conics, opacities, layout, intersections = (
        _one_pixel_visibility_scene([0.98] * 8)
    )
    counts, alphas = rasterize_num_contributing_gaussians_sparse(
        means2d,
        conics,
        opacities,
        layout.active_tiles,
        intersections.tile_offsets,
        intersections.flatten_ids,
        layout.tile_pixel_mask,
        layout.tile_pixel_cumsum,
        layout.pixel_map,
        1,
        1,
        4,
        1,
        1,
        active_tile_count=layout.valid_count,
        valid_count=intersections.valid_count,
    )
    np.testing.assert_array_equal(counts, [2])
    np.testing.assert_allclose(alphas, [0.9996], rtol=1e-6, atol=1e-6)


@jax.jit
def _zero_gradient_visibility(opacities, common):
    counts, alphas = rasterize_num_contributing_gaussians_sparse(
        common[0],
        common[1],
        opacities,
        *common[2:],
        1,
        1,
        4,
        1,
        1,
    )
    return counts, alphas


def test_visibility_queries_are_stop_gradient_like_upstream_no_grad():
    means2d, conics, opacities, layout, intersections = (
        _one_pixel_visibility_scene([0.2, 0.3])
    )
    common = (
        means2d,
        conics,
        layout.active_tiles,
        intersections.tile_offsets,
        intersections.flatten_ids,
        layout.tile_pixel_mask,
        layout.tile_pixel_cumsum,
        layout.pixel_map,
    )
    gradient = jax.grad(
        lambda value: _zero_gradient_visibility(value, common)[1].sum()
    )(opacities)
    np.testing.assert_array_equal(gradient, np.zeros_like(opacities))


@pytest.mark.resource_heavy
def test_dense_and_sparse_visibility_queries_match_for_dense_and_packed():
    scene = _random_scene()
    means2d, conics, opacities, _radii, _depths, pixels, image_ids = scene

    for packed in (False, True):
        layout, sparse_intersections = _random_layout(scene, packed=packed)
        dense_offsets, dense_intersections = _dense_intersections(
            scene, packed=packed
        )
        query_means = means2d.reshape(-1, 2) if packed else means2d
        query_conics = conics.reshape(-1, 3) if packed else conics
        query_opacities = opacities.reshape(-1) if packed else opacities

        dense_counts, dense_alphas = rasterize_num_contributing_gaussians(
            query_means,
            query_conics,
            query_opacities,
            dense_offsets,
            dense_intersections.flatten_ids,
            9,
            7,
            4,
            valid_count=dense_intersections.valid_count,
        )
        sparse_counts, sparse_alphas = (
            rasterize_num_contributing_gaussians_sparse(
                query_means,
                query_conics,
                query_opacities,
                layout.active_tiles,
                sparse_intersections.tile_offsets,
                sparse_intersections.flatten_ids,
                layout.tile_pixel_mask,
                layout.tile_pixel_cumsum,
                layout.pixel_map,
                9,
                7,
                4,
                3,
                2,
                active_tile_count=layout.valid_count,
                valid_count=sparse_intersections.valid_count,
            )
        )
        gathered_counts = dense_counts[
            image_ids, pixels[:, 0], pixels[:, 1]
        ]
        gathered_alphas = dense_alphas[
            image_ids, pixels[:, 0], pixels[:, 1]
        ]
        np.testing.assert_array_equal(sparse_counts, gathered_counts)
        np.testing.assert_allclose(
            sparse_alphas, gathered_alphas, rtol=1e-6, atol=1e-6
        )

        dense_all = rasterize_contributing_gaussian_ids(
            query_means,
            query_conics,
            query_opacities,
            dense_offsets,
            dense_intersections.flatten_ids,
            9,
            7,
            4,
            dense_counts,
            valid_count=dense_intersections.valid_count,
        )
        sparse_all = rasterize_contributing_gaussian_ids_sparse(
            query_means,
            query_conics,
            query_opacities,
            layout.active_tiles,
            sparse_intersections.tile_offsets,
            sparse_intersections.flatten_ids,
            layout.tile_pixel_mask,
            layout.tile_pixel_cumsum,
            layout.pixel_map,
            sparse_counts,
            9,
            7,
            4,
            3,
            2,
            active_tile_count=layout.valid_count,
            valid_count=sparse_intersections.valid_count,
        )
        dense_ids = dense_all.gaussian_ids[
            image_ids, pixels[:, 0], pixels[:, 1]
        ]
        dense_weights = dense_all.weights[
            image_ids, pixels[:, 0], pixels[:, 1]
        ]
        np.testing.assert_array_equal(sparse_all.gaussian_ids, dense_ids)
        np.testing.assert_allclose(
            sparse_all.weights, dense_weights, rtol=1e-6, atol=1e-6
        )
        assert not bool(sparse_all.overflow)

        dense_top_ids, dense_top_weights = (
            rasterize_top_contributing_gaussian_ids(
                query_means,
                query_conics,
                query_opacities,
                dense_offsets,
                dense_intersections.flatten_ids,
                9,
                7,
                4,
                3,
                valid_count=dense_intersections.valid_count,
            )
        )
        sparse_top_ids, sparse_top_weights = (
            rasterize_top_contributing_gaussian_ids_sparse(
                query_means,
                query_conics,
                query_opacities,
                layout.active_tiles,
                sparse_intersections.tile_offsets,
                sparse_intersections.flatten_ids,
                layout.tile_pixel_mask,
                layout.tile_pixel_cumsum,
                layout.pixel_map,
                9,
                7,
                4,
                3,
                2,
                3,
                active_tile_count=layout.valid_count,
                valid_count=sparse_intersections.valid_count,
            )
        )
        np.testing.assert_array_equal(
            sparse_top_ids,
            dense_top_ids[image_ids, pixels[:, 0], pixels[:, 1]],
        )
        np.testing.assert_allclose(
            sparse_top_weights,
            dense_top_weights[image_ids, pixels[:, 0], pixels[:, 1]],
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.resource_heavy
def test_dense_visibility_alpha_matches_dense_rasterizer():
    scene = _random_scene()
    means2d, conics, opacities, _radii, _depths, _pixels, _image_ids = scene
    offsets, intersections = _dense_intersections(scene, packed=False)
    counts, query_alpha = rasterize_num_contributing_gaussians(
        means2d,
        conics,
        opacities,
        offsets,
        intersections.flatten_ids,
        9,
        7,
        4,
        valid_count=intersections.valid_count,
    )
    _, raster_alpha, _ = rasterize_to_pixels(
        means2d,
        conics,
        jnp.zeros((*opacities.shape, 1), jnp.float32),
        opacities,
        9,
        7,
        4,
        offsets,
        intersections.flatten_ids,
        valid_count=intersections.valid_count,
        max_gaussians_per_tile=means2d.shape[1],
        return_info=True,
    )
    assert counts.dtype == jnp.int32
    np.testing.assert_allclose(query_alpha, raster_alpha[..., 0], rtol=1e-6, atol=1e-6)
