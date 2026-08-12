from typing import Any, cast

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


def _supports_native_pallas():
    device = jax.devices()[0]
    try:
        compute_capability = float(
            getattr(device, "compute_capability", 0.0)
        )
    except (TypeError, ValueError):
        compute_capability = 0.0
    return device.platform == "gpu" and compute_capability >= 9.0


def _reference(max_candidates_per_tile):
    means, conics, colors, opacities, offsets, flatten_ids, background = _inputs()
    return cast(
        tuple[Any, Any, dict[str, Any]],
        rasterize_to_pixels(
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
        ),
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


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("max_candidates_per_tile", [1, 5])
@pytest.mark.parametrize("interpret", [True, False], ids=["interpret", "native"])
def test_pallas_compositor_backward_matches_pure_jax(
    channels, max_candidates_per_tile, interpret,
):
    if not interpret and not _supports_native_pallas():
        pytest.skip("native Pallas compositing requires a supported GPU")
    inputs = _inputs()
    if channels == 1:
        inputs = (
            inputs[0],
            inputs[1],
            inputs[2][:, :1],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6][:1],
        )
    elif channels == 4:
        inputs = (
            inputs[0],
            inputs[1],
            jnp.concatenate(
                (
                    inputs[2],
                    jnp.linspace(0.1, 0.9, 5, dtype=jnp.float32)[:, None],
                ),
                axis=-1,
            ),
            inputs[3],
            inputs[4],
            inputs[5],
            jnp.concatenate(
                (inputs[6], jnp.asarray([0.2], dtype=jnp.float32))
            ),
        )
    # Gaussian 1 appears in both tiles, exercising the post-kernel scatter-add.
    flatten_ids = inputs[5].at[3].set(1)
    render_cotangent = jnp.linspace(
        -0.7, 0.9, 8 * 4 * channels, dtype=jnp.float32
    ).reshape(4, 8, channels)
    alpha_cotangent = jnp.linspace(
        0.6, -0.4, 8 * 4, dtype=jnp.float32
    ).reshape(4, 8, 1)

    def loss(compositor, means, conics, colors, opacities, background):
        if compositor == "jax":
            result = rasterize_to_pixels(
                means[None, ...],
                conics[None, ...],
                colors[None, ...],
                opacities[None, ...],
                8,
                4,
                4,
                inputs[4][None, ...],
                flatten_ids,
                backgrounds=background[None, ...],
                valid_count=jnp.asarray(5, jnp.int32),
                max_gaussians_per_tile=2,
                max_candidates_per_tile=max_candidates_per_tile,
                return_info=True,
            )
            rendered, alphas = result[0][0], result[1][0]
        else:
            rendered, alphas, _ = rasterize_to_pixels_pallas(
                means,
                conics,
                colors,
                opacities,
                8,
                4,
                4,
                inputs[4],
                flatten_ids,
                backgrounds=background,
                valid_count=jnp.asarray(5, jnp.int32),
                max_gaussians_per_tile=2,
                max_candidates_per_tile=max_candidates_per_tile,
                interpret=interpret,
            )
        return jnp.sum(rendered * render_cotangent) + jnp.sum(
            alphas * alpha_cotangent
        )

    differentiable_inputs = (
        inputs[0].at[0].set(jnp.asarray([0.5, 0.5], jnp.float32)),
        inputs[1],
        inputs[2],
        inputs[3].at[0].set(jnp.asarray(0.9999, jnp.float32)),
        inputs[6],
    )
    expected = jax.jit(
        jax.value_and_grad(
            lambda *args: loss("jax", *args), argnums=(0, 1, 2, 3, 4)
        )
    )(*differentiable_inputs)
    actual = jax.jit(
        jax.value_and_grad(
            lambda *args: loss("pallas", *args), argnums=(0, 1, 2, 3, 4)
        )
    )(*differentiable_inputs)

    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-6, atol=2e-7)
    for actual_gradient, expected_gradient in zip(actual[1], expected[1]):
        np.testing.assert_allclose(
            actual_gradient, expected_gradient, rtol=2e-5, atol=2e-6
        )


def test_pallas_compositor_large_empty_tail_gradients_stay_zero():
    if not _supports_native_pallas():
        pytest.skip("native Pallas compositing requires a supported GPU")
    gaussian_count = 257
    means = jnp.zeros((gaussian_count, 2), jnp.float32).at[0].set(
        jnp.asarray([0.5, 0.5], jnp.float32)
    )
    conics = jnp.zeros((gaussian_count, 3), jnp.float32)
    colors = jnp.zeros((gaussian_count, 1), jnp.float32).at[0, 0].set(1.0)
    opacities = jnp.zeros((gaussian_count,), jnp.float32).at[0].set(0.5)
    offsets = jnp.zeros((1, 1), jnp.int32)
    flatten_ids = jnp.full((512,), -1, jnp.int32).at[0].set(0)

    def loss(current_means):
        rendered, alpha, _ = rasterize_to_pixels_pallas(
            current_means,
            conics,
            colors,
            opacities,
            16,
            16,
            16,
            offsets,
            flatten_ids,
            valid_count=jnp.asarray(1, jnp.int32),
            max_gaussians_per_tile=512,
            max_candidates_per_tile=512,
        )
        return jnp.sum(rendered) + jnp.sum(alpha)

    gradient = jax.jit(jax.grad(loss))(means)
    np.testing.assert_array_equal(np.asarray(gradient[1:]), 0.0)


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
