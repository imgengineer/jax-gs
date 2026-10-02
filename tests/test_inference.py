"""RGB-only rendering keeps the training image without its backward storage."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig
from jaxgs.render.types import ProjectedGaussians, SortedVisibilityTable

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="Packed inference requires JAX CUDA and CuTe",
)


@pytest.mark.parametrize("tile_height,tile_width", [(8, 8), (8, 16), (12, 16), (16, 16)])
def test_rgb_only_matches_training_without_pixel_or_tile_caches(tile_height, tile_width):
    from jaxgs.kernels.packed_rasterizer import _packed_forward, packed_forward

    rng = np.random.default_rng(59)
    count = 97  # Multiple staged batches and a partial last batch.
    width, height = tile_width + 3, tile_height + 1
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 8, 4, width, height)
    config = CapacityConfig(count, 1, count, tile_width, 0, count * 4, tile_height=tile_height)
    matrix = rng.normal(size=(count, 2, 2)).astype(np.float32) * 0.3
    conic = matrix @ matrix.transpose(0, 2, 1) + np.eye(2, dtype=np.float32) * 0.01
    opacity = rng.uniform(0.05, 0.95, count).astype(np.float32)
    opacity[::5] = 0.001
    projected = ProjectedGaussians(
        jnp.asarray(rng.uniform((0, 0), (width, height), (count, 2)), jnp.float32),
        jnp.arange(count, dtype=jnp.float32),
        jnp.asarray(conic),
        jnp.ones(count),
        jnp.asarray(rng.random((count, 3)), jnp.float32),
        jnp.asarray(opacity),
        jnp.ones(count, jnp.bool_),
    )
    table = SortedVisibilityTable(
        jnp.tile(jnp.arange(count, dtype=jnp.int32), 4),
        jnp.arange(5, dtype=jnp.int32) * count,
        jnp.full(count, 4, jnp.int32),
        jnp.asarray(count * 4, jnp.int32),
        jnp.asarray(False),
    )
    image, cache, stats = _packed_forward(
        projected, table, camera, config, False, record_contributions=False
    )
    expected, training_cache, _ = packed_forward(projected, table, camera, config)
    np.testing.assert_array_equal(image, expected)
    assert np.any(np.asarray(image) > 0)
    assert all(value.shape == (1,) for value in (*cache[1:], stats))
    assert training_cache[1].shape == training_cache[2].shape == (width * height,)
    assert training_cache[3].shape == (4,)
