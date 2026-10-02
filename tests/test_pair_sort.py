"""Stable tile sort of the emitted pairs, limited to the pair count."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe kernels require JAX CUDA and NVIDIA CUTLASS DSL",
)


@pytest.mark.parametrize("num_tiles", [1, 200, 8034, 65535, 300000])
@pytest.mark.parametrize("count", [0, 1, 2047, 2049, 70001])
def test_pairs_sort_stably_by_tile(num_tiles, count):
    from jaxgs.kernels.pair_sort import sort_pairs_by_tile

    rng = np.random.default_rng(num_tiles + count)
    max_pairs = 70001
    dtype = np.uint16 if num_tiles <= 65535 else np.uint32
    # Few distinct tiles in places, so equal keys span warps and blocks.
    keys = rng.integers(0, num_tiles, max_pairs).astype(dtype)
    keys[: max_pairs // 3] = rng.integers(0, min(num_tiles, 3), max_pairs // 3)
    values = rng.permutation(max_pairs).astype(np.int32)
    garbage = rng.integers(0, 1 << 16, max_pairs).astype(dtype)
    arena = np.where(np.arange(max_pairs) < count, keys, garbage)
    sorted_keys, sorted_values = jax.jit(sort_pairs_by_tile, static_argnums=3)(
        jnp.asarray(arena), jnp.asarray(values), jnp.int32(count), num_tiles
    )
    order = np.argsort(keys[:count], kind="stable")
    np.testing.assert_array_equal(np.asarray(sorted_keys)[:count], keys[:count][order])
    np.testing.assert_array_equal(np.asarray(sorted_values)[:count], values[:count][order])


def test_pair_count_is_clamped_to_the_arena():
    from jaxgs.kernels.pair_sort import sort_pairs_by_tile

    rng = np.random.default_rng(7)
    keys = rng.integers(0, 500, 5000).astype(np.uint16)
    values = np.arange(5000, dtype=np.int32)
    sorted_keys, sorted_values = sort_pairs_by_tile(
        jnp.asarray(keys), jnp.asarray(values), jnp.int32(9000), 500
    )
    order = np.argsort(keys, kind="stable")
    np.testing.assert_array_equal(sorted_keys, keys[order])
    np.testing.assert_array_equal(sorted_values, values[order])


@pytest.mark.parametrize(
    "num_tiles,expected",
    [(1, (1, 1)), (256, (1, 8)), (257, (2, 5)), (8034, (2, 7)), (65535, (2, 8)), (300000, (3, 7))],
)
def test_digit_plan(num_tiles, expected):
    from jaxgs.kernels.pair_sort import digit_plan

    assert digit_plan(num_tiles) == expected
