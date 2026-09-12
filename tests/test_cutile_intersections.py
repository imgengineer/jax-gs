import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_gs._cutile_intersections as cutile_intersections
from jax_gs._cutile_intersections import (
    _prefix_counts_cutile,
    _radix_pass_cutile,
    _sort_offsets_cutile,
)
from jax_gs.intersections import intersect_tiles

pytestmark = pytest.mark.gpu


def _require_cutile() -> None:
    device = jax.devices()[0]
    if device.platform != "gpu" or "cuda" not in str(device).lower():
        pytest.skip("cuTile intersections require an NVIDIA CUDA GPU")


@pytest.mark.parametrize(
    "capacity,empty,tile_width,tile_height",
    [
        (17, False, 7, 5),
        (257, False, 7, 5),
        (4096, False, 7, 5),
        (257, True, 7, 5),
        (4096, False, 2, 13),
        (4096, False, 13, 2),
    ],
)
def test_cutile_complete_topology_matches_jax(capacity, empty, tile_width, tile_height):
    _require_cutile()
    count = 257
    key = jax.random.key(capacity)
    means = jax.random.uniform(
        key, (count, 2), minval=-16.0, maxval=112.0, dtype=jnp.float32
    )
    radii = jax.random.randint(key, (count, 2), 1, 20, dtype=jnp.int32)
    depths = jax.random.uniform(
        jax.random.fold_in(key, 1),
        (count,),
        minval=0.01,
        maxval=20.0,
        dtype=jnp.float32,
    )
    a = jax.random.uniform(
        jax.random.fold_in(key, 2), (count,), minval=0.02, maxval=0.2
    )
    c = jax.random.uniform(
        jax.random.fold_in(key, 3), (count,), minval=0.02, maxval=0.2
    )
    b = jnp.sqrt(a * c) * jax.random.uniform(
        jax.random.fold_in(key, 4), (count,), minval=-0.8, maxval=0.8
    )
    conics = jnp.stack((a, b, c), axis=-1).astype(jnp.float32)
    opacities = jax.random.uniform(
        jax.random.fold_in(key, 5), (count,), dtype=jnp.float32
    )
    valid = jax.random.bernoulli(jax.random.fold_in(key, 6), 0.85, (count,))
    if empty:
        valid = jnp.zeros((count,), jnp.bool_)

    def run(backend):
        return intersect_tiles(
            means,
            radii,
            depths,
            valid,
            tile_size=16,
            tile_width=tile_width,
            tile_height=tile_height,
            max_intersections=capacity,
            backend=backend,
            sort_backend=backend,
            conics=conics,
            opacities=opacities,
            mode="accutile",
        )

    actual = jax.jit(lambda: run("cuda_tile"))()
    expected = jax.jit(lambda: run("jax"))()
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


@pytest.mark.parametrize(
    "block_size,scan_chunk_size", [(256, 256), (512, 256), (512, 4)]
)
def test_cutile_radix_sort_is_stable_across_scan_chunks(
    monkeypatch, block_size, scan_chunk_size
):
    _require_cutile()
    monkeypatch.setattr(cutile_intersections, "_RADIX_LARGE_CAPACITY", 0)
    monkeypatch.setattr(cutile_intersections, "_RADIX_LARGE_BLOCK_SIZE", block_size)
    monkeypatch.setattr(cutile_intersections, "_RADIX_SCAN_CHUNK_SIZE", scan_chunk_size)
    capacity = 70_013
    valid_count = capacity - 13
    tile_count = 97
    generator = np.random.default_rng(17)
    tile_ids = generator.integers(0, tile_count, size=capacity, dtype=np.int32)
    depths = generator.normal(size=capacity).astype(np.float32)
    depths[::19] = np.float32(0.0)
    depths[1::23] = np.float32(-0.0)
    gaussian_ids = np.arange(capacity, dtype=np.int32)
    order = np.lexsort(
        (
            np.arange(valid_count, dtype=np.int32),
            depths[:valid_count],
            tile_ids[:valid_count],
        )
    )
    expected_gaussians = np.full((capacity,), -1, np.int32)
    expected_tiles = np.full((capacity,), -1, np.int32)
    expected_gaussians[:valid_count] = gaussian_ids[:valid_count][order]
    expected_tiles[:valid_count] = tile_ids[:valid_count][order]
    expected_offsets = np.searchsorted(
        expected_tiles[:valid_count], np.arange(tile_count), side="left"
    ).astype(np.int32)

    actual = jax.jit(
        lambda gaussian_ids_, tile_ids_, depths_: _sort_offsets_cutile(
            gaussian_ids_,
            tile_ids_,
            depths_,
            jnp.asarray(valid_count, jnp.int32),
            tile_count=tile_count,
        )
    )(jnp.asarray(gaussian_ids), jnp.asarray(tile_ids), jnp.asarray(depths))
    expected = (
        expected_gaussians,
        expected_tiles,
        expected_offsets,
        np.asarray(valid_count, np.int32),
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


@pytest.mark.parametrize("capacity", [50_000, 200_000])
def test_cutile_hierarchical_prefix_matches_numpy(capacity):
    _require_cutile()
    generator = np.random.default_rng(capacity)
    counts = generator.integers(0, 4, size=70_013, dtype=np.int32)
    required_count = int(counts.astype(np.int64).sum())
    expected_cumulative = np.minimum(
        np.cumsum(counts, dtype=np.int64), capacity + 1
    ).astype(np.int32)

    actual = jax.jit(lambda values: _prefix_counts_cutile(values, capacity=capacity))(
        jnp.asarray(counts)
    )

    np.testing.assert_array_equal(actual[0], expected_cumulative)
    assert int(actual[1]) == min(required_count, capacity)
    assert bool(actual[2]) is (required_count > capacity)
    assert int(actual[3]) == required_count


@pytest.mark.parametrize("count", [32, 33, 70])
@pytest.mark.parametrize("capacity", [0, 17, 1000])
def test_cutile_prefix_scans_every_top_level_chunk(monkeypatch, count, capacity):
    _require_cutile()
    # Scale down the hierarchy to exercise the 256**3 Gaussian boundary cheaply.
    monkeypatch.setattr(cutile_intersections, "_PREFIX_BLOCK_SIZE", 2)
    monkeypatch.setattr(cutile_intersections, "_PREFIX_SCAN_CHUNK_SIZE", 4)
    counts = np.arange(count, dtype=np.int32) % 4
    counts[-1] = 7
    expected_cumulative = np.cumsum(counts, dtype=np.int64)
    required = int(expected_cumulative[-1])

    actual = jax.jit(lambda x: _prefix_counts_cutile(x, capacity=capacity))(
        jnp.asarray(counts)
    )

    np.testing.assert_array_equal(
        actual[0], np.minimum(expected_cumulative, capacity + 1)
    )
    assert int(actual[1]) == min(required, capacity)
    assert bool(actual[2]) is (required > capacity)
    assert int(actual[3]) == required


def test_cutile_radix_sort_handles_an_empty_prefix():
    _require_cutile()
    capacity = 257
    actual = jax.jit(
        lambda: _sort_offsets_cutile(
            jnp.full((capacity,), -1, jnp.int32),
            jnp.full((capacity,), -1, jnp.int32),
            jnp.zeros((capacity,), jnp.float32),
            jnp.asarray(0, jnp.int32),
            tile_count=7,
        )
    )()
    expected = (
        np.full((capacity,), -1, np.int32),
        np.full((capacity,), -1, np.int32),
        np.zeros((7,), np.int32),
        np.asarray(0, np.int32),
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


@pytest.mark.parametrize("block_size", [256, 512])
def test_cutile_radix_sort_preserves_full_blocks_of_equal_keys(monkeypatch, block_size):
    _require_cutile()
    monkeypatch.setattr(cutile_intersections, "_RADIX_LARGE_CAPACITY", 0)
    monkeypatch.setattr(cutile_intersections, "_RADIX_LARGE_BLOCK_SIZE", block_size)
    capacity = 4097
    valid_count = 1025
    ids = np.arange(capacity, dtype=np.int32)[::-1].copy()
    actual = jax.jit(
        lambda ids_: _sort_offsets_cutile(
            ids_,
            jnp.full((capacity,), 2, jnp.int32),
            jnp.ones((capacity,), jnp.float32),
            jnp.asarray(valid_count, jnp.int32),
            tile_count=4,
        )
    )(jnp.asarray(ids))
    expected_ids = np.full(capacity, -1, np.int32)
    expected_ids[:valid_count] = ids[:valid_count]
    expected_tiles = np.full(capacity, -1, np.int32)
    expected_tiles[:valid_count] = 2
    np.testing.assert_array_equal(actual[0], expected_ids)
    np.testing.assert_array_equal(actual[1], expected_tiles)
    np.testing.assert_array_equal(actual[2], [0, 0, 0, valid_count])
    assert int(actual[3]) == valid_count


def test_cutile_radix_sort_drops_malformed_prefix_entries():
    _require_cutile()
    actual = jax.jit(
        lambda: _sort_offsets_cutile(
            jnp.asarray([0, 9, 1, 2, -1], jnp.int32),
            jnp.asarray([0, 0, 1, 2, -1], jnp.int32),
            jnp.asarray([1.0, 2.0, jnp.nan], jnp.float32),
            jnp.asarray(4, jnp.int32),
            tile_count=3,
        )
    )()
    expected = (
        np.asarray([0, 1, -1, -1, -1], np.int32),
        np.asarray([0, 1, -1, -1, -1], np.int32),
        np.asarray([0, 1, 2], np.int32),
        np.asarray(2, np.int32),
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


@pytest.mark.parametrize("block_size", [256, 512])
@pytest.mark.parametrize("radix_size", [2, 4, 32, 64])
def test_cutile_packed_radix_pass_matches_stable_sort(
    monkeypatch, block_size, radix_size
):
    _require_cutile()
    monkeypatch.setattr(cutile_intersections, "_RADIX_LARGE_CAPACITY", 0)
    monkeypatch.setattr(cutile_intersections, "_RADIX_LARGE_BLOCK_SIZE", block_size)
    capacity = 1537
    shift = 11
    generator = np.random.default_rng(29)
    keys = generator.integers(0, 1 << 63, size=capacity, dtype=np.uint64)
    # Exercise the largest count in a single packed field, including slot 3.
    keys[:512] = (radix_size - 1) << shift
    values = np.arange(capacity, dtype=np.int32)

    with jax.enable_x64(True):
        run = jax.jit(
            lambda keys_, values_, count_: _radix_pass_cutile(
                keys_, values_, count_, shift=shift, radix_size=radix_size
            )
        )
        device_keys = jnp.asarray(keys)
        device_values = jnp.asarray(values)
        for count in (0, 1, 31, 32, 33, 511, 512, 513, 1031):
            actual = run(device_keys, device_values, jnp.asarray(count, jnp.int32))
            digits = (keys[:count] >> shift) & (radix_size - 1)
            order = np.argsort(digits, kind="stable")
            # The radix pass only defines the active prefix of its output.
            np.testing.assert_array_equal(actual[0][:count], keys[:count][order])
            np.testing.assert_array_equal(actual[1][:count], values[:count][order])
