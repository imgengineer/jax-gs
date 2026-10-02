"""Contribution masks preserve RGB cotangents across holes and opaque tails."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig
from jaxgs.render.types import ProjectedGaussians, SortedVisibilityTable

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="Packed rasterizer requires JAX CUDA and CuTe",
)


@pytest.mark.parametrize("kind", ["holes", "empty", "opaque"])
@pytest.mark.parametrize("collect_stats", [False, True])
@pytest.mark.parametrize("tile_height,tile_width", [(8, 8), (8, 16), (12, 16), (16, 16)])
def test_contribution_bits_and_backward_ignore_unwritten_words(
    kind, collect_stats, tile_height, tile_width
):
    from jaxgs.kernels.packed_rasterizer import (
        packed_backward,
        packed_forward,
        rasterize_packed_cute_vjp,
    )

    # Empty tiles, unaligned offsets, 31/32/33/65 pairs, and image-edge tiles.
    counts = np.array([0, 31, 32, 33, 65, 1], np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    count = int(offsets[-1])
    config = CapacityConfig(count, 1, count, tile_width, 0, 256, tile_height=tile_height)
    width, height = tile_width + 3, tile_height * 2 + 1
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 9, 8, width, height)
    tile = np.repeat(np.arange(counts.size), counts)
    mean = np.stack((tile % 2 * tile_width, tile // 2 * tile_height), axis=1) + 0.5
    alpha = np.where(np.arange(count) % 2 == 0, 0.25, 0.001).astype(np.float32)
    if kind == "empty":
        alpha.fill(0)
    if kind == "opaque":
        alpha.fill(0.99)
    projected = ProjectedGaussians(
        jnp.asarray(mean, jnp.float32),
        jnp.arange(count, dtype=jnp.float32),
        jnp.broadcast_to(jnp.eye(2) * (0 if kind == "opaque" else 20), (count, 2, 2)),
        jnp.ones(count),
        jnp.full((count, 3), 0.5),
        jnp.asarray(alpha),
        jnp.ones(count, jnp.bool_),
    )
    table = SortedVisibilityTable(
        jnp.asarray(np.pad(np.arange(count, dtype=np.int32), (0, 256 - count), constant_values=-1)),
        jnp.asarray(offsets),
        jnp.ones(count, jnp.int32),
        jnp.asarray(count, jnp.int32),
        jnp.asarray(False),
    )
    image, cache, stats = packed_forward(projected, table, camera, config, collect_stats)
    np.testing.assert_array_equal(
        rasterize_packed_cute_vjp(projected, table, camera, config), image
    )
    expected = np.zeros(count, bool)
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        trans = np.float16(128)
        for i in range(start, end):
            opacity = np.float16(alpha[i])
            if trans > 128 / 8192 and opacity >= 1 / 256:
                expected[i] = True
                trans = np.float16(trans * np.float16(1 - opacity))

    # Poison padding and words beyond early termination; backward must not read them.
    poisoned = np.full(cache[4].shape, np.uint32(0xFFFFFFFF))
    actual_bits = np.asarray(cache[4])
    work = np.asarray(cache[3])
    for t, start in enumerate(offsets[:-1]):
        base = start // 32 + t
        for chunk in range((int(work[t]) + 31) // 32):
            bits = sum(
                int(expected[i]) << int(i - start - chunk * 32)
                for i in range(start + chunk * 32, min(start + (chunk + 1) * 32, offsets[t + 1]))
            )
            assert actual_bits[base + chunk] == bits
            word_work = min(32, int(work[t]) - chunk * 32)
            poisoned[base + chunk] = bits | ((0xFFFFFFFF << word_work) & 0xFFFFFFFF)
    if collect_stats:
        pixel_counts = np.ones(count, np.float32)
        if kind == "opaque":
            widths = np.where(tile % 2 == 0, tile_width, 3)
            heights = np.minimum(tile_height, height - tile // 2 * tile_height)
            pixel_counts = widths * heights
        np.testing.assert_array_equal(np.asarray(stats)[0::2], expected * pixel_counts)

    # Equal R/G colors cancel opacity/geometry derivatives, but not color derivatives.
    incoming = jnp.broadcast_to(jnp.array([1.0, -1.0, 0.0]), image.shape)
    grads, squares = packed_backward(
        projected, table, cache, incoming, camera, config, collect_stats
    )
    poisoned_cache = (*cache[:4], jnp.asarray(poisoned))
    other, other_squares = packed_backward(
        projected, table, poisoned_cache, incoming, camera, config, collect_stats
    )
    for field in ("mean", "conic", "color", "alpha"):
        np.testing.assert_array_equal(getattr(other, field), getattr(grads, field))
    if collect_stats:
        np.testing.assert_array_equal(other_squares, squares)
    np.testing.assert_array_equal(grads.alpha, 0)
    np.testing.assert_array_equal(grads.mean, 0)
    np.testing.assert_array_equal(grads.conic, 0)
    np.testing.assert_array_equal(np.asarray(grads.color)[:, 0] > 0, expected)
    np.testing.assert_array_equal(grads.color[:, 1], -grads.color[:, 0])
    np.testing.assert_array_equal(grads.color[:, 2], 0)


@pytest.mark.parametrize("tile_height,tile_width", [(8, 8), (8, 16), (12, 16), (16, 16)])
def test_empty_pixel_groups_preserve_accumulated_gradients(tile_height, tile_width):
    from jaxgs.kernels.packed_rasterizer import packed_backward, packed_forward

    count = 65
    groups = tile_height * tile_width // 64
    width, height = tile_width + 3, tile_height + 1
    config = CapacityConfig(count, 1, count, tile_width, 0, 96, tile_height=tile_height)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 8, 4, width, height)
    rng = np.random.default_rng(14)
    # Each narrow splat hits exactly one group, leaving the other groups
    # empty before or after a nonzero partial opacity/color gradient.
    mean = np.stack((np.arange(count) % 3 * 2 + 0.7, np.arange(count) % groups * 2 + 0.4), 1)
    projected = ProjectedGaussians(
        jnp.asarray(mean, jnp.float32),
        jnp.arange(count, dtype=jnp.float32),
        jnp.broadcast_to(jnp.eye(2) * 20, (count, 2, 2)),
        jnp.ones(count),
        jnp.asarray(rng.random((count, 3)), jnp.float32),
        jnp.asarray(np.where(np.arange(count) % 5 == 0, 0.001, 0.3), jnp.float32),
        jnp.ones(count, jnp.bool_),
    )
    table = SortedVisibilityTable(
        jnp.pad(jnp.arange(count, dtype=jnp.int32), (0, 96 - count), constant_values=-1),
        jnp.array([0, count, count, count, count], jnp.int32),
        jnp.ones(count, jnp.int32),
        jnp.array(count, jnp.int32),
        jnp.array(False),
    )
    image, cache, _ = packed_forward(projected, table, camera, config, True)
    incoming = jnp.asarray(rng.normal(size=image.shape), jnp.float32)
    ordinary, _ = packed_backward(projected, table, cache, incoming, camera, config, False)
    with_stats, _ = packed_backward(projected, table, cache, incoming, camera, config, True)
    for field in ("mean", "conic", "color", "alpha"):
        expected = np.asarray(getattr(with_stats, field))
        assert np.any(expected != 0), field
        np.testing.assert_allclose(getattr(ordinary, field), expected, rtol=1e-5, atol=1e-8)
