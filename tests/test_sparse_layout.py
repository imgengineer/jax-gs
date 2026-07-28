import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.sparse import build_sparse_tile_layout, isect_tiles_sparse


def _reference_layout(pixels, image_ids, n_images, tile_size, tile_width, tile_height):
    pixels = np.asarray(pixels, np.int32)
    image_ids = np.asarray(image_ids, np.int32)
    rows, columns = pixels[:, 0], pixels[:, 1]
    tiles_per_image = tile_width * tile_height
    tile_ids = (
        image_ids * tiles_per_image
        + (rows // tile_size) * tile_width
        + columns // tile_size
    )
    in_tile = (rows % tile_size) * tile_size + columns % tile_size
    pixel_map = np.lexsort((np.arange(len(pixels)), in_tile, tile_ids)).astype(
        np.int32
    )
    active_tiles, counts = np.unique(tile_ids, return_counts=True)
    active_tiles = active_tiles.astype(np.int32)
    active_mask = np.zeros(n_images * tiles_per_image, bool)
    active_mask[active_tiles] = True
    words = (tile_size * tile_size + 31) // 32
    bitmask = np.zeros((len(active_tiles), words), np.uint32)
    active_rank = {int(tile_id): rank for rank, tile_id in enumerate(active_tiles)}
    for tile_id, pixel_id in zip(tile_ids, in_tile, strict=True):
        rank = active_rank[int(tile_id)]
        bitmask[rank, pixel_id // 32] |= np.uint32(1 << (pixel_id % 32))
    return (
        active_tiles,
        active_mask.reshape(n_images, tile_height, tile_width),
        bitmask,
        np.cumsum(counts, dtype=np.int32),
        pixel_map,
    )


def _reference_sparse_intersections(
    means2d,
    radii,
    depths,
    tile_mask,
    active_tiles,
    n_images,
    tile_size,
    tile_width,
    tile_height,
    image_ids=None,
):
    means2d = np.asarray(means2d)
    radii = np.asarray(radii)
    depths = np.asarray(depths)
    if means2d.ndim == 3:
        gaussians_per_image = means2d.shape[1]
        flat_means = means2d.reshape(-1, 2)
        flat_radii = radii.reshape(-1, 2)
        flat_depths = depths.reshape(-1)
        image_of = np.repeat(np.arange(n_images), gaussians_per_image)
    else:
        flat_means = means2d
        flat_radii = radii
        flat_depths = depths
        image_of = np.asarray(image_ids)

    records = []
    tiles_per_image = tile_width * tile_height
    active_rank = {int(tile_id): rank for rank, tile_id in enumerate(active_tiles)}
    for gaussian_id, (mean, radius, depth, image_id) in enumerate(
        zip(flat_means, flat_radii, flat_depths, image_of, strict=True)
    ):
        if (
            not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(radius))
            or not np.isfinite(depth)
            or not np.all(radius > 0)
            or image_id < 0
            or image_id >= n_images
        ):
            continue
        lower = np.floor((mean - radius) / tile_size).astype(np.int32)
        upper = np.ceil((mean + radius) / tile_size).astype(np.int32)
        min_x, min_y = np.clip(lower, 0, [tile_width, tile_height])
        max_x, max_y = np.clip(upper, 0, [tile_width, tile_height])
        for tile_y in range(min_y, max_y):
            for tile_x in range(min_x, max_x):
                dense_id = int(
                    image_id * tiles_per_image + tile_y * tile_width + tile_x
                )
                if dense_id not in active_rank or not tile_mask[image_id, tile_y, tile_x]:
                    continue
                records.append(
                    (active_rank[dense_id], np.float32(depth), gaussian_id)
                )
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    flatten_ids = np.asarray([record[2] for record in records], np.int32)
    counts = np.bincount(
        [record[0] for record in records], minlength=len(active_tiles)
    ).astype(np.int32)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int32)))
    return offsets, flatten_ids


def test_sparse_tile_layout_matches_raster_order_contract():
    pixels = jnp.asarray(
        [[7, 7], [0, 1], [4, 0], [3, 3], [0, 0]], dtype=jnp.int32
    )
    image_ids = jnp.asarray([0, 0, 0, 0, 1], dtype=jnp.int32)

    layout = jax.jit(
        lambda p, i: build_sparse_tile_layout(p, i, 2, 4, 2, 2)
    )(pixels, image_ids)
    active_tiles, active_mask, bitmask, cumsum, pixel_map = layout

    np.testing.assert_array_equal(active_tiles[: layout.valid_count], [0, 2, 3, 4])
    np.testing.assert_array_equal(active_tiles[layout.valid_count :], [-1])
    np.testing.assert_array_equal(
        active_mask,
        np.asarray(
            [
                [[True, False], [True, True]],
                [[True, False], [False, False]],
            ]
        ),
    )
    assert bitmask.dtype == jnp.uint32
    np.testing.assert_array_equal(
        bitmask[: layout.valid_count, 0],
        np.asarray([(1 << 1) | (1 << 15), 1, 1 << 15, 1], np.uint32),
    )
    np.testing.assert_array_equal(cumsum[: layout.valid_count], [2, 3, 4, 5])
    np.testing.assert_array_equal(pixel_map, [1, 3, 2, 0, 4])
    assert int(layout.valid_count) == 4
    assert int(layout.required_count) == 4
    assert not bool(layout.overflow)


def test_sparse_tile_layout_uses_all_32_bits_in_each_word():
    pixels = jnp.asarray([[0, 0], [3, 7], [4, 0], [7, 7]], jnp.int32)
    image_ids = jnp.zeros((4,), jnp.int32)
    layout = build_sparse_tile_layout(pixels, image_ids, 1, 8, 1, 1)

    assert layout.tile_pixel_mask.shape == (1, 2)
    np.testing.assert_array_equal(
        layout.tile_pixel_mask[0],
        np.asarray([0x80000001, 0x80000001], np.uint32),
    )
    np.testing.assert_array_equal(layout.tile_pixel_cumsum, [4])


@pytest.mark.parametrize("tile_size", [4, 8])
def test_sparse_tile_layout_matches_random_reference(tile_size):
    rng = np.random.default_rng(100 + tile_size)
    n_images, width, height = 3, 29, 21
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size
    pixels, image_ids = [], []
    for image_id, count in enumerate([41, 0, 53]):
        flat = rng.choice(width * height, size=count, replace=False)
        pixels.append(np.stack((flat // width, flat % width), axis=-1))
        image_ids.append(np.full((count,), image_id, np.int32))
    pixels = np.concatenate(pixels).astype(np.int32)
    image_ids = np.concatenate(image_ids)

    result = build_sparse_tile_layout(
        pixels,
        image_ids,
        n_images,
        tile_size,
        tile_width,
        tile_height,
    )
    expected = _reference_layout(
        pixels,
        image_ids,
        n_images,
        tile_size,
        tile_width,
        tile_height,
    )
    valid_count = int(result.valid_count)
    np.testing.assert_array_equal(result.active_tiles[:valid_count], expected[0])
    np.testing.assert_array_equal(result.active_tile_mask, expected[1])
    np.testing.assert_array_equal(result.tile_pixel_mask[:valid_count], expected[2])
    np.testing.assert_array_equal(
        result.tile_pixel_cumsum[:valid_count], expected[3]
    )
    np.testing.assert_array_equal(result.pixel_map, expected[4])


def test_sparse_tile_layout_empty_case_and_explicit_overflow():
    empty = build_sparse_tile_layout(
        jnp.empty((0, 2), jnp.int32),
        jnp.empty((0,), jnp.int32),
        2,
        4,
        3,
        2,
    )
    assert empty.active_tiles.shape == (0,)
    assert empty.active_tile_mask.shape == (2, 2, 3)
    assert empty.tile_pixel_mask.shape == (0, 1)
    np.testing.assert_array_equal(empty.tile_pixel_cumsum, [0])
    assert empty.pixel_map.shape == (0,)
    assert int(empty.valid_count) == 0
    assert not bool(empty.overflow)

    pixels = jnp.asarray([[0, 0], [0, 4], [4, 0], [4, 4]], jnp.int32)
    truncated = build_sparse_tile_layout(
        pixels,
        jnp.zeros((4,), jnp.int32),
        1,
        4,
        2,
        2,
        max_active_tiles=2,
    )
    np.testing.assert_array_equal(truncated.active_tiles, [0, 1])
    assert int(truncated.valid_count) == 2
    assert int(truncated.required_count) == 4
    assert bool(truncated.overflow)


def test_sparse_intersections_dense_order_offsets_and_jit():
    means2d = jnp.asarray([[[2.0, 2.0], [6.0, 2.0], [2.0, 2.0]]])
    radii = jnp.asarray([[[3.0, 2.0], [1.0, 1.0], [0.0, 2.0]]])
    depths = jnp.asarray([[2.0, 1.0, 0.5]])
    tile_mask = jnp.ones((1, 1, 2), dtype=jnp.bool_)
    active_tiles = jnp.asarray([0, 1], dtype=jnp.int32)

    result = jax.jit(
        lambda m, r, d: isect_tiles_sparse(
            m, r, d, tile_mask, active_tiles, 1, 4, 2, 1
        )
    )(means2d, radii, depths)
    offsets, flatten_ids = result

    np.testing.assert_array_equal(offsets, [0, 1, 3])
    np.testing.assert_array_equal(flatten_ids[:3], [0, 1, 0])
    assert int(result.valid_count) == 3
    assert int(result.required_count) == 3
    assert int(result.active_tile_count) == 2
    assert not bool(result.overflow)


def test_sparse_intersections_packed_and_masked_tiles():
    means2d = jnp.asarray([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    radii = jnp.ones((3, 2), dtype=jnp.float32)
    depths = jnp.asarray([2.0, 3.0, 1.0])
    image_ids = jnp.asarray([1, 0, 1], dtype=jnp.int32)
    tile_mask = jnp.asarray([[[True, False]], [[True, False]]])
    active_tiles = jnp.asarray([0, 2], dtype=jnp.int32)

    result = isect_tiles_sparse(
        means2d,
        radii,
        depths,
        tile_mask,
        active_tiles,
        2,
        4,
        2,
        1,
        image_ids=image_ids,
    )

    np.testing.assert_array_equal(result.tile_offsets, [0, 1, 3])
    np.testing.assert_array_equal(result.flatten_ids[:3], [1, 2, 0])

    masked = isect_tiles_sparse(
        means2d,
        radii,
        depths,
        tile_mask.at[1, 0, 0].set(False),
        active_tiles,
        2,
        4,
        2,
        1,
        image_ids=image_ids,
    )
    np.testing.assert_array_equal(masked.tile_offsets, [0, 1, 1])
    np.testing.assert_array_equal(masked.flatten_ids[:1], [1])


def test_sparse_intersections_match_random_reference():
    rng = np.random.default_rng(19)
    n_images, gaussian_count = 3, 27
    tile_size, tile_width, tile_height = 4, 6, 5
    means2d = rng.uniform([-5.0, -4.0], [29.0, 24.0], (n_images, gaussian_count, 2))
    radii = rng.integers(1, 8, (n_images, gaussian_count, 2)).astype(np.float32)
    radii[0, :3] = 0.0
    depths = rng.uniform(0.1, 10.0, (n_images, gaussian_count)).astype(np.float32)
    tile_mask = rng.random((n_images, tile_height, tile_width)) < 0.55
    active_tiles = np.flatnonzero(tile_mask.reshape(-1)).astype(np.int32)

    result = isect_tiles_sparse(
        means2d,
        radii,
        depths,
        tile_mask,
        active_tiles,
        n_images,
        tile_size,
        tile_width,
        tile_height,
    )
    expected_offsets, expected_ids = _reference_sparse_intersections(
        means2d,
        radii,
        depths,
        tile_mask,
        active_tiles,
        n_images,
        tile_size,
        tile_width,
        tile_height,
    )
    np.testing.assert_array_equal(result.tile_offsets, expected_offsets)
    np.testing.assert_array_equal(
        result.flatten_ids[: result.valid_count], expected_ids
    )
    assert int(result.required_count) == len(expected_ids)
    assert not bool(result.overflow)


def test_sparse_intersections_zero_active_tiles_and_overflow_metadata():
    means2d = jnp.asarray([[[2.0, 2.0], [6.0, 2.0]]])
    radii = jnp.asarray([[[3.0, 2.0], [1.0, 1.0]]])
    depths = jnp.asarray([[2.0, 1.0]])
    tile_mask = jnp.ones((1, 1, 2), dtype=jnp.bool_)

    empty = isect_tiles_sparse(
        means2d,
        radii,
        depths,
        tile_mask,
        jnp.empty((0,), jnp.int32),
        1,
        4,
        2,
        1,
    )
    np.testing.assert_array_equal(empty.tile_offsets, [0])
    assert empty.flatten_ids.shape == (0,)
    assert int(empty.required_count) == 0

    truncated = isect_tiles_sparse(
        means2d,
        radii,
        depths,
        tile_mask,
        jnp.asarray([0, 1], jnp.int32),
        1,
        4,
        2,
        1,
        max_intersections=1,
    )
    assert truncated.flatten_ids.shape == (1,)
    assert int(truncated.valid_count) == 1
    assert int(truncated.required_count) == 3
    assert bool(truncated.overflow)
    assert int(truncated.tile_offsets[-1]) == 1


def test_sparse_intersections_accept_padded_active_tile_layout():
    pixels = jnp.asarray([[0, 0], [0, 4]], jnp.int32)
    layout = build_sparse_tile_layout(
        pixels,
        jnp.zeros((2,), jnp.int32),
        1,
        4,
        2,
        1,
        max_active_tiles=4,
    )
    means2d = jnp.asarray([[[2.0, 2.0]]])
    radii = jnp.asarray([[[5.0, 2.0]]])
    depths = jnp.asarray([[1.0]])

    result = isect_tiles_sparse(
        means2d,
        radii,
        depths,
        layout.active_tile_mask,
        layout.active_tiles,
        1,
        4,
        2,
        1,
        active_tile_count=layout.valid_count,
    )
    np.testing.assert_array_equal(result.tile_offsets, [0, 1, 2, 2, 2])
    np.testing.assert_array_equal(result.flatten_ids[:2], [0, 0])
    assert int(result.active_tile_count) == 2
