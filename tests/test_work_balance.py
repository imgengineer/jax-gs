"""Heaviest-first tile scheduling, staged splat batches and cooperative pair emission."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig
from jaxgs.render.types import ProjectedGaussians

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe kernels require JAX CUDA and the cute extra",
)


def test_half_pair_sums_preserve_rounding_and_nonfinite_values():
    import cutlass.cute as cute
    from cutlass.jax import cutlass_call

    from jaxgs.kernels import half2 as h

    @cute.kernel
    def sums(a, b, result, reference, size: int):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        i = block * 256 + tid
        if i < size:
            result[i] = h.pack_pair_sums(a[i], b[i])
            reference[i] = h.pack(h.sum_pair(a[i]), h.sum_pair(b[i]))

    @cute.jit
    def launch(stream, a, b, result, reference, *, size: int):
        sums(a, b, result, reference, size).launch(
            grid=[(size + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
        )

    # Every half encoding appears in both positions, followed by random pairs.
    half = np.arange(65536, dtype=np.uint32)
    rng = np.random.default_rng(44)
    a = np.concatenate(
        [half | (((half * 7919) & 65535) << 16), rng.integers(0, 1 << 32, 65537, dtype=np.uint32)]
    )
    b = np.concatenate(
        [
            half[::-1] | (((half * 8191) & 65535) << 16),
            rng.integers(0, 1 << 32, 65537, dtype=np.uint32),
        ]
    )
    call = cutlass_call(
        launch,
        output_shape_dtype=(jax.ShapeDtypeStruct(a.shape, jnp.uint32),) * 2,
        use_static_tensors=True,
        size=a.size,
    )
    actual, expected = [np.asarray(x).view(np.uint16) for x in call(jnp.asarray(a), jnp.asarray(b))]
    actual_nan, expected_nan = (actual & 0x7FFF) > 0x7C00, (expected & 0x7FFF) > 0x7C00
    np.testing.assert_array_equal(actual_nan, expected_nan)
    np.testing.assert_array_equal(actual[~expected_nan], expected[~expected_nan])


def _splats(rng, count, width, height, radius=(0.4, 40.0), alpha=(0.005, 0.99), huge=0):
    """Small, large, thin and nearly degenerate screen-space ellipses."""
    mean = rng.uniform((-0.1 * width, -0.1 * height), (1.1 * width, 1.1 * height), (count, 2))
    scale = np.exp(rng.uniform(np.log(radius[0]), np.log(radius[1]), (count, 2)))
    scale[:huge] = rng.uniform(0.3, 0.6, (huge, 2)) * max(width, height)
    thin = (rng.random(count) < 0.25) & (np.arange(count) >= huge)
    scale[thin, 1] = scale[thin, 0] * rng.uniform(0.005, 0.05, thin.sum())
    angle = rng.uniform(0, np.pi, count)
    rotation = np.stack(
        [
            np.stack([np.cos(angle), -np.sin(angle)], -1),
            np.stack([np.sin(angle), np.cos(angle)], -1),
        ],
        -2,
    )
    covariance = rotation @ (scale[:, :, None] ** 2 * np.eye(2)) @ rotation.transpose(0, 2, 1)
    conic = np.linalg.inv(covariance)
    conic = (conic + conic.transpose(0, 2, 1)) / 2
    depth = rng.integers(1, count // 3, count).astype(np.float32)  # includes ties
    return ProjectedGaussians(
        jnp.asarray(mean, jnp.float32),
        jnp.asarray(depth),
        jnp.asarray(conic, jnp.float32),
        jnp.asarray(3 * scale.max(1), jnp.float32),
        jnp.asarray(rng.uniform(0.05, 0.95, (count, 3)), jnp.float32),
        jnp.asarray(rng.uniform(*alpha, count), jnp.float32),
        jnp.asarray(rng.random(count) < 0.9),
    )


@pytest.mark.parametrize("tile_height,tile_width", [(8, 16), (16, 16)])
def test_every_emission_path_fills_exactly_the_counted_segments(tile_height, tile_width):
    from cutlass.jax import cutlass_call

    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
    from jaxgs.kernels.sorted_visibility import launch_emit_pairs

    rng = np.random.default_rng(3)
    width, height, capacity = 640, 360, 2048
    projected = _splats(rng, capacity, width, height, radius=(0.4, 60.0), huge=8)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 300, 300, 320, 180, width, height)
    config = CapacityConfig(capacity, 128, 1, tile_width, 0, 1 << 19, tile_height=tile_height)
    table = build_sorted_visibility_table_cute(projected, camera, config)
    counts = np.asarray(table.point_counts)
    tiles_x, tiles_y = -(-width // tile_width), -(-height // tile_height)
    tiles = tiles_x * tiles_y
    # Exercise per-lane and whole-warp emission, including more than 32 rows.
    assert np.any((counts > 0) & (counts <= 16)) and np.any((counts > 16) & (counts < 64))
    assert counts.max() > 32 * max(tiles_x, tiles_y) or tiles_x < 33 or tiles_y < 33
    order = jnp.argsort(projected.depth, stable=True).astype(jnp.int32)
    ordered = table.point_counts[order]
    ends = jnp.cumsum(ordered, dtype=jnp.int32)
    max_pairs = config.visibility_capacity

    def emit(lane_pairs):
        call = cutlass_call(
            launch_emit_pairs,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((max_pairs,), jnp.uint16),
                jax.ShapeDtypeStruct((max_pairs,), jnp.int32),
                jax.ShapeDtypeStruct((capacity,), jnp.int32),
                jax.ShapeDtypeStruct((1,), jnp.int32),
            ),
            use_static_tensors=True,
            capacity=capacity,
            max_pairs=max_pairs,
            tile_size=tile_width,
            tile_height=tile_height,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            lane_pairs=lane_pairs,
            fill_padding=True,
        )
        mean, conic = projected.mean.reshape(-1), projected.conic.reshape(-1)
        outputs = call(mean, conic, projected.alpha, order, ordered, ends)
        return tuple(map(np.asarray, outputs))

    total = int(table.pair_count)
    assert total == int(ends[-1]) == counts.sum()
    keys, ids, _, queued = emit(1 << 30)
    assert queued[0] == 0
    for lane_pairs in (0, 16):
        shared_keys, shared_ids, _, queued = emit(lane_pairs)
        np.testing.assert_array_equal(shared_keys, keys)
        np.testing.assert_array_equal(shared_ids, ids)
        assert queued[0] == np.sum(counts > lane_pairs)
    # The counting and emitting kernels agree: every counted pair is written
    # and nothing past the pairs.
    assert (keys[:total] < tiles).all() and (keys[total:] == tiles).all()
    assert (ids[total:] == -1).all()
    np.testing.assert_array_equal(ids[:total], np.repeat(np.asarray(order), np.asarray(ordered)))
    pairs = ids[:total].astype(np.int64) * tiles + keys[:total]
    assert np.unique(pairs).size == total
    offsets = np.asarray(table.tile_offsets)
    assert offsets[-1] == total
    sorted_ids = np.asarray(table.gaussian_ids)[:total]
    tile_of_pair = np.repeat(np.arange(tiles), np.diff(offsets))
    same_tile = tile_of_pair[1:] == tile_of_pair[:-1]
    depth = np.asarray(projected.depth)[sorted_ids]
    rank = np.argsort(np.asarray(order))[sorted_ids]
    assert (np.diff(depth)[same_tile] >= 0).all()
    assert (np.diff(rank)[same_tile] > 0).all()


@pytest.mark.parametrize("capacity", [1, 5, 1023, 1024, 1025, 70001])
def test_pair_offsets_scan_depth_ordered_counts(capacity):
    from cutlass.jax import cutlass_call

    from jaxgs.kernels.sorted_visibility import _SCAN_ITEMS, _SCAN_THREADS, launch_pair_offsets

    rng = np.random.default_rng(capacity)
    counts = rng.integers(0, 300, capacity).astype(np.int32)
    counts[rng.random(capacity) < 0.5] = 0
    order = rng.permutation(capacity).astype(np.int32)
    call = cutlass_call(
        launch_pair_offsets,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct((-(-capacity // (_SCAN_THREADS * _SCAN_ITEMS)),), jnp.int32),
        ),
        use_static_tensors=True,
        capacity=capacity,
    )
    ordered, ends, _ = call(jnp.asarray(order), jnp.asarray(counts))
    np.testing.assert_array_equal(ordered, counts[order])
    np.testing.assert_array_equal(ends, np.cumsum(counts[order]))


@pytest.mark.parametrize("tiles", [1, 7, 1025, 9000])
@pytest.mark.parametrize("from_offsets", [False, True])
def test_tile_order_is_a_heaviest_first_permutation(tiles, from_offsets):
    from cutlass.jax import cutlass_call

    from jaxgs.kernels.packed_rasterize import launch_tile_order

    rng = np.random.default_rng(tiles)
    work = rng.integers(0, 5000, tiles).astype(np.int32)
    work[rng.random(tiles) < 0.2] = 0
    source = np.concatenate([[0], np.cumsum(work)]) if from_offsets else work
    order = cutlass_call(
        launch_tile_order,
        output_shape_dtype=jax.ShapeDtypeStruct((tiles,), jnp.int32),
        use_static_tensors=True,
        tiles=tiles,
        from_offsets=from_offsets,
    )(jnp.asarray(source, jnp.int32))
    order = np.asarray(order)
    np.testing.assert_array_equal(np.sort(order), np.arange(tiles))
    # Eight buckets per octave of work + 1, as in the kernel.
    bucket = (np.float32(work + 1).view(np.uint32) >> 20).astype(np.int64)
    assert (np.diff(bucket[order]) <= 0).all()


@pytest.mark.parametrize("collect_stats", [False, True])
def test_packed_staging_across_batches_matches_reference(collect_stats):
    from jaxgs.kernels.packed_rasterizer import (
        packed_backward,
        packed_forward,
        rasterize_packed_cute_vjp,
    )
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
    from jaxgs.reference.rasterizer_jax import rasterize_jax
    from jaxgs.render.visibility_table import build_visibility_table

    rng = np.random.default_rng(11)
    width, height, count = 40, 20, 180
    projected = _splats(rng, count, width, height, radius=(2.0, 9.0), alpha=(0.04, 0.3))
    # Opaque splats at median depth saturate the left tiles partway through.
    opaque = jnp.stack(jnp.meshgrid(jnp.array([4.0, 12.0]), jnp.arange(4) * 6 + 1.0), -1)
    projected = projected.replace(
        mean=(projected.mean * 0.8 + jnp.array([0.1 * width, 0.1 * height]))
        .at[:8]
        .set(opaque.reshape(8, 2)),
        depth=projected.depth.at[:8].set(jnp.median(projected.depth)),
        conic=projected.conic.at[:8].set(jnp.eye(2) / 400),
        alpha=projected.alpha.at[:8].set(0.99),
        visible=jnp.ones(count, bool),
    )
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 20, 10, width, height)
    config = CapacityConfig(count, 1, count, 16, 0, count * 12, tile_height=8)
    table = build_sorted_visibility_table_cute(projected, camera, config)
    image, cache, _ = packed_forward(projected, table, camera, config, collect_stats)
    reference_config = CapacityConfig(count, 1, count, 64, 0)
    reference_table = build_visibility_table(projected, camera, reference_config)
    expected = rasterize_jax(projected, reference_table, camera, reference_config).rgb
    # Half2 compositing of up to 180 splats per pixel (a few half ULPs at scale 128).
    np.testing.assert_allclose(image, expected, atol=5e-3)
    assert np.linalg.norm(image - expected) / np.linalg.norm(expected) < 2e-3

    # The forward reports the pairs before each tile's last contributor.
    offsets = np.asarray(table.tile_offsets)
    last = np.asarray(cache[2]).reshape(height, width)
    tiles_x = -(-width // 16)
    tile = (np.arange(height)[:, None] // 8) * tiles_x + np.arange(width)[None, :] // 16
    expected_work = np.zeros(offsets.size - 1, np.int64)
    np.maximum.at(expected_work, tile.reshape(-1), (last - offsets[tile]).reshape(-1))
    np.testing.assert_array_equal(cache[3], expected_work)
    assert expected_work.max() > 64  # three or more staged batches
    # Some tiles stop early, inside a staged batch.
    assert ((expected_work < np.diff(offsets)) & (expected_work % 32 != 0)).any()

    weights = jnp.asarray(rng.normal(0, 0.01, image.shape), jnp.float32)
    grads, _ = packed_backward(projected, table, cache, weights, camera, config, collect_stats)

    def reference_loss(mean, conic, color, alpha):
        current = projected.replace(mean=mean, conic=conic, color=color, alpha=alpha)
        rgb = rasterize_jax(current, reference_table, camera, reference_config).rgb
        return jnp.sum(rgb * weights)

    def packed_loss(mean, conic, color, alpha):
        current = projected.replace(mean=mean, conic=conic, color=color, alpha=alpha)
        return jnp.sum(rasterize_packed_cute_vjp(current, table, camera, config) * weights)

    fields = (projected.mean, projected.conic, projected.color, projected.alpha)
    with jax.default_matmul_precision("highest"):
        reference = jax.grad(reference_loss, argnums=(0, 1, 2, 3))(*fields)
    packed = jax.grad(packed_loss, argnums=(0, 1, 2, 3))(*fields)
    for name, actual, via_vjp, target in zip(
        ("mean", "conic", "color", "alpha"),
        (grads.mean, grads.conic, grads.color, grads.alpha),
        packed,
        reference,
        strict=True,
    ):
        np.testing.assert_allclose(via_vjp, actual, rtol=1e-5, atol=1e-8, err_msg=name)
        scale = np.max(np.abs(target))
        error = np.linalg.norm(np.asarray(actual) - target) / np.linalg.norm(target)
        assert error < 0.03, (name, error, scale)
