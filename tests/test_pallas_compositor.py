import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax_gs._pallas import rasterize_to_pixels_pallas
from jax_gs.low_level import rasterize_to_pixels


def _inputs():
    means = jnp.asarray(
        [[1.0, 1.0], [2.0, 2.0], [5.0, 1.0], [6.0, 2.0], [3.0, 3.0]],
        jnp.float32,
    )
    conics = jnp.tile(jnp.asarray([[1.0, 0.0, 1.0]], jnp.float32), (5, 1))
    colors = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        jnp.float32,
    )
    opacities = jnp.asarray([0.5, 0.4, 0.3, 0.2, 0.1], jnp.float32)
    offsets = jnp.asarray([[0, 3]], jnp.int32)
    flatten_ids = jnp.asarray(
        [0, 1, 4, 2, 3, -1, -1, -1, -1, -1], jnp.int32
    )
    background = jnp.asarray([0.05, 0.1, 0.15], jnp.float32)
    return means, conics, colors, opacities, offsets, flatten_ids, background


def _reference(max_candidates_per_tile):
    means, conics, colors, opacities, offsets, flatten_ids, background = _inputs()
    return rasterize_to_pixels(
        means[None, ...],
        conics[None, ...],
        colors[None, ...],
        opacities[None, ...],
        8,
        4,
        4,
        offsets[None, ...],
        flatten_ids,
        backgrounds=background[None, ...],
        valid_count=jnp.asarray(5, jnp.int32),
        max_gaussians_per_tile=2,
        max_candidates_per_tile=max_candidates_per_tile,
        return_info=True,
    )


def _pallas(max_candidates_per_tile):
    means, conics, colors, opacities, offsets, flatten_ids, background = _inputs()
    return rasterize_to_pixels_pallas(
        means,
        conics,
        colors,
        opacities,
        8,
        4,
        4,
        offsets,
        flatten_ids,
        backgrounds=background,
        valid_count=jnp.asarray(5, jnp.int32),
        max_gaussians_per_tile=2,
        max_candidates_per_tile=max_candidates_per_tile,
        interpret=True,
    )


def test_pallas_compositor_matches_pure_jax_forward_in_interpret_mode():
    expected = _reference(5)
    actual = jax.jit(lambda: _pallas(5))()

    np.testing.assert_allclose(actual[0], expected[0][0], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(actual[1], expected[1][0], rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(
        np.asarray(actual[2]["tile_overflow"]),
        np.asarray(expected[2]["tile_overflow"][0]),
    )
    assert not bool(actual[2]["overflow"])


def test_pallas_compositor_preserves_candidate_bound_overflow_semantics():
    # The public bound sizes a fixed K-wide chunk loop. A promise of one with
    # K=2 therefore still renders two candidates before reporting overflow.
    expected = _reference(1)
    actual = _pallas(1)

    np.testing.assert_allclose(actual[0], expected[0][0], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(actual[1], expected[1][0], rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(
        np.asarray(actual[2]["tile_overflow"]),
        np.asarray(expected[2]["tile_overflow"][0]),
    )
    assert bool(actual[2]["overflow"])


def test_pallas_compositor_rejects_native_cpu_lowering():
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU rejection is specific to CPU test runs")
    means, conics, colors, opacities, offsets, flatten_ids, background = _inputs()
    with pytest.raises(RuntimeError, match="Hopper-or-newer"):
        rasterize_to_pixels_pallas(
            means,
            conics,
            colors,
            opacities,
            8,
            4,
            4,
            offsets,
            flatten_ids,
            backgrounds=background,
            valid_count=jnp.asarray(5, jnp.int32),
        )
