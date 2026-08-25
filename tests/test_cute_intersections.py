import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax_gs._cute_intersections import (
    _emit_accutile_intersections_cute,
    _prepare_accutile_counts_cute,
    intersection_prefix_cute,
    intersection_sort_offsets_cute,
    rasterize_accutile_cute_raw_fused,
)
from jax_gs.intersections import _accutile_intersections_jax, intersect_tiles
from jax_gs.low_level import rasterize_to_pixels


def _supports_cute():
    device = jax.devices()[0]
    return device.platform == "gpu" and "cuda" in str(device).lower()


@pytest.mark.parametrize("count", [6, 257, 70_001])
def test_cute_intersection_prefix_matches_saturated_jax_scan(count):
    if not _supports_cute():
        pytest.skip("CuTe intersections require an NVIDIA CUDA GPU")
    values = (jnp.arange(count, dtype=jnp.int32) % 11) - 2
    values = values.at[min(count - 1, 4)].set(2**30 - 2)
    capacity = 193

    def reference(counts):
        limit = jnp.int32(2**30 - 1)
        cumulative = jax.lax.associative_scan(
            lambda left, right: jnp.where(
                left >= limit - right, limit, left + right
            ),
            jnp.maximum(counts, 0),
        )
        required = cumulative[-1]
        return (
            jnp.minimum(cumulative, jnp.int32(capacity + 1)),
            jnp.minimum(required, jnp.int32(capacity)),
            required > capacity,
            required,
        )

    expected = jax.jit(reference)(values)
    actual = jax.jit(
        lambda counts: intersection_prefix_cute(counts, capacity=capacity)
    )(values)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


def test_cute_accutile_preparation_count_and_emission_match_jax():
    if not _supports_cute():
        pytest.skip("CuTe intersections require an NVIDIA CUDA GPU")
    tile_width = tile_height = 2
    capacity = 17
    means = jnp.asarray(
        [[8, 8], [24, 8], [8, 24], [24, 24], [16, 16]], jnp.float32
    )
    radii = jnp.full((5, 2), 12, jnp.int32)
    depths = jnp.asarray([1.0, 0.0, -0.0, 2.0, jnp.nan], jnp.float32)
    conics = jnp.tile(jnp.asarray([[0.04, 0.0, 0.04]], jnp.float32), (5, 1))
    opacities = jnp.asarray([0.8, 0.7, 0.6, 0.5, 0.9], jnp.float32)
    valid = jnp.ones((5,), jnp.bool_)

    def run_reference(means_, radii_, depths_, conics_, opacities_, valid_):
        return _accutile_intersections_jax(
            means_,
            radii_,
            conics_,
            opacities_,
            valid_ & jnp.isfinite(depths_),
            capacity=capacity,
            tile_size=16,
            tile_width=tile_width,
            tile_height=tile_height,
            alpha_threshold=1 / 255,
        )

    expected = jax.jit(run_reference)(
        means, radii, depths, conics, opacities, valid
    )

    def run_cute(means_, radii_, depths_, conics_, opacities_, valid_):
        state_floats, state_bounds, state_is_y, counts = (
            _prepare_accutile_counts_cute(
                means_,
                radii_,
                depths_,
                conics_,
                opacities_,
                valid_,
                tile_width=tile_width,
                tile_height=tile_height,
                alpha_threshold=1 / 255,
            )
        )
        cumulative, count, overflow, required = intersection_prefix_cute(
            counts, capacity=capacity
        )
        gaussian_ids, tile_ids = _emit_accutile_intersections_cute(
            state_floats,
            state_bounds,
            state_is_y,
            cumulative,
            count,
            capacity=capacity,
            tile_width=tile_width,
        )
        return counts, gaussian_ids, tile_ids, count, overflow, required

    actual = jax.jit(run_cute)(
        means, radii, depths, conics, opacities, valid
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


@pytest.mark.parametrize("candidate_bound", [32, 2])
def test_cute_raw_fused_matches_staged_jax_forward_and_gradients(
    candidate_bound,
):
    if not _supports_cute():
        pytest.skip("CuTe intersections require an NVIDIA CUDA GPU")
    tile_width = tile_height = 2
    capacity = 17
    means = jnp.asarray(
        [[8, 8], [24, 8], [8, 24], [24, 24], [16, 16]], jnp.float32
    )
    radii = jnp.full((5, 2), 12, jnp.int32)
    depths = jnp.asarray([1.0, 0.0, -0.0, 2.0, jnp.nan], jnp.float32)
    conics = jnp.tile(jnp.asarray([[0.04, 0.0, 0.04]], jnp.float32), (5, 1))
    colors = jnp.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]],
        jnp.float32,
    )
    opacities = jnp.asarray([0.8, 0.7, 0.6, 0.5, 0.9], jnp.float32)
    valid = jnp.ones((5,), jnp.bool_)
    background = jnp.asarray([0.1, 0.2, 0.3], jnp.float32)

    def staged(means_, conics_, colors_, opacities_, background_):
        topology = intersect_tiles(
            jax.lax.stop_gradient(means_),
            radii,
            depths,
            valid,
            tile_size=16,
            tile_width=tile_width,
            tile_height=tile_height,
            max_intersections=capacity,
            backend="jax",
            sort_backend="jax",
            conics=jax.lax.stop_gradient(conics_),
            opacities=jax.lax.stop_gradient(opacities_),
            alpha_threshold=1 / 255,
            mode="accutile",
        )
        rendered, alpha, _ = rasterize_to_pixels(
            means_[None, ...],
            conics_[None, ...],
            colors_[None, ...],
            opacities_[None, ...],
            32,
            32,
            16,
            topology.offsets[None, ...],
            topology.gaussian_ids,
            backgrounds=background_[None, ...],
            valid_count=topology.valid_count,
            max_gaussians_per_tile=candidate_bound,
            max_candidates_per_tile=candidate_bound,
            return_info=True,
        )
        return rendered[0], alpha[0], topology

    def fused(means_, conics_, colors_, opacities_, background_):
        return rasterize_accutile_cute_raw_fused(
            radii,
            depths,
            means_,
            conics_,
            colors_,
            opacities_,
            valid,
            capacity=capacity,
            tile_size=16,
            tile_width=tile_width,
            tile_height=tile_height,
            image_width=32,
            image_height=32,
            background=background_,
            max_gaussians_per_tile=candidate_bound,
            max_candidates_per_tile=candidate_bound,
            alpha_threshold=1 / 255,
            transmittance_threshold=1e-4,
        )

    expected = jax.jit(staged)(means, conics, colors, opacities, background)
    actual = jax.jit(fused)(means, conics, colors, opacities, background)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=3e-5, atol=3e-6)
    topology = expected[2]
    flat_offsets = topology.offsets.reshape(-1)
    ends = jnp.concatenate((flat_offsets[1:], topology.valid_count[None]))
    np.testing.assert_array_equal(
        actual[2]["tile_overflow"],
        ((ends - flat_offsets) > candidate_bound).reshape(
            tile_height, tile_width
        ),
    )
    for actual_value, expected_value in (
        (actual[2]["gaussian_ids"], topology.gaussian_ids),
        (actual[2]["tile_ids"], topology.tile_ids),
        (actual[2]["offsets"], topology.offsets),
        (actual[2]["valid_count"], topology.valid_count),
        (actual[2]["overflow"], topology.overflow),
        (actual[2]["required_count"], topology.required_count),
    ):
        np.testing.assert_array_equal(actual_value, expected_value)

    def loss(function, *arguments):
        rendered, alpha = function(*arguments)[:2]
        return jnp.mean(rendered) + jnp.mean(alpha)

    differentiable = (means, conics, colors, opacities, background)
    expected_gradient = jax.jit(
        jax.value_and_grad(
            lambda *args: loss(staged, *args), argnums=(0, 1, 2, 3, 4)
        )
    )(*differentiable)
    actual_gradient = jax.jit(
        jax.value_and_grad(
            lambda *args: loss(fused, *args), argnums=(0, 1, 2, 3, 4)
        )
    )(*differentiable)
    np.testing.assert_allclose(
        actual_gradient[0], expected_gradient[0], rtol=3e-5, atol=3e-6
    )
    for actual_value, expected_value in zip(
        actual_gradient[1], expected_gradient[1], strict=True
    ):
        np.testing.assert_allclose(
            actual_value, expected_value, rtol=2e-4, atol=2e-5
        )


def test_cute_intersection_sort_offsets_preserves_ties_and_padding():
    if not _supports_cute():
        pytest.skip("CuTe intersections require an NVIDIA CUDA GPU")
    sort = jax.jit(
        lambda gaussian_ids, tile_ids, depths, valid_count: (
            intersection_sort_offsets_cute(
                gaussian_ids,
                tile_ids,
                depths,
                valid_count,
                tile_count=4,
                segment_capacity=2,
            )
        )
    )
    actual = sort(
        jnp.asarray([0, 1, 2, 3, 4, -1, -1], jnp.int32),
        jnp.asarray([2, 1, 1, 1, 0, -1, -1], jnp.int32),
        jnp.asarray([0.0, -0.0, 0.0, -2.0, 1.0], jnp.float32),
        jnp.asarray(5, jnp.int32),
    )
    expected = (
        np.asarray([4, 3, 1, 2, 0, -1, -1], np.int32),
        np.asarray([0, 1, 1, 1, 2, -1, -1], np.int32),
        np.asarray([0, 1, 4, 5], np.int32),
        np.asarray(5, np.int32),
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


@pytest.mark.parametrize("tile_count", [17, 97])
def test_cute_intersection_sort_offsets_is_stable_across_blocks(tile_count):
    if not _supports_cute():
        pytest.skip("CuTe intersections require an NVIDIA CUDA GPU")
    capacity = 4097
    valid_count = capacity - 13
    generator = np.random.default_rng(17)
    tile_ids = generator.integers(
        0, tile_count, size=capacity, dtype=np.int32
    )
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

    sort = jax.jit(
        lambda gaussian_ids, tile_ids, depths, count: (
            intersection_sort_offsets_cute(
                gaussian_ids,
                tile_ids,
                depths,
                count,
                tile_count=tile_count,
                segment_capacity=512,
            )
        )
    )
    actual = sort(
        jnp.asarray(gaussian_ids),
        jnp.asarray(tile_ids),
        jnp.asarray(depths),
        jnp.asarray(valid_count, jnp.int32),
    )
    expected = (
        expected_gaussians,
        expected_tiles,
        expected_offsets,
        np.asarray(valid_count, np.int32),
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


def test_cute_intersection_sort_offsets_clips_malformed_valid_prefix():
    if not _supports_cute():
        pytest.skip("CuTe intersections require an NVIDIA CUDA GPU")
    actual = jax.jit(
        lambda: intersection_sort_offsets_cute(
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
