import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs._cutile_intersections import _sort_offsets_cutile
from jax_gs.intersections import intersect_tiles

pytestmark = pytest.mark.gpu


def _require_cutile() -> None:
    device = jax.devices()[0]
    if device.platform != "gpu" or "cuda" not in str(device).lower():
        pytest.skip("cuTile intersections require an NVIDIA CUDA GPU")


@pytest.mark.parametrize("capacity", [17, 257, 4096])
def test_cutile_complete_topology_matches_jax(capacity):
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

    def run(backend):
        return intersect_tiles(
            means,
            radii,
            depths,
            valid,
            tile_size=16,
            tile_width=7,
            tile_height=5,
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


def test_cutile_radix_sort_is_stable_across_scan_chunks():
    _require_cutile()
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
