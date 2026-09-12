from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs._cutile_compositor import rasterize_to_pixels_cutile
from jax_gs.low_level import rasterize_to_pixels


def _supports_cutile():
    device = jax.devices()[0]
    return device.platform == "gpu" and "cuda" in str(device).lower()


def _inputs(channels=3):
    means = jnp.asarray(
        [[1.0, 1.0], [2.0, 2.0], [5.0, 1.0], [6.0, 2.0], [3.0, 3.0]],
        jnp.float32,
    )
    conics = jnp.tile(jnp.asarray([[1.0, 0.0, 1.0]], jnp.float32), (5, 1))
    base_colors = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 1.0, 0.0, 0.4],
            [0.0, 0.0, 1.0, 0.6],
            [1.0, 1.0, 0.0, 0.8],
            [1.0, 0.0, 1.0, 1.0],
        ],
        jnp.float32,
    )
    repeats = (channels + base_colors.shape[1] - 1) // base_colors.shape[1]
    colors = jnp.tile(base_colors, (1, repeats))[:, :channels]
    opacities = jnp.asarray([0.5, 0.4, 0.3, 0.2, 0.1], jnp.float32)
    offsets = jnp.asarray([[0, 3]], jnp.int32)
    flatten_ids = jnp.asarray([0, 1, 4, 2, 3, -1, -1, -1, -1, -1], jnp.int32)
    background = jnp.linspace(0.05, 0.2, channels, dtype=jnp.float32)
    return means, conics, colors, opacities, offsets, flatten_ids, background


def _reference(inputs, max_candidates_per_tile=5, *, width=32, height=16):
    means, conics, colors, opacities, offsets, flatten_ids, background = inputs
    return cast(
        tuple[Any, Any, dict[str, Any]],
        rasterize_to_pixels(
            means[None, ...],
            conics[None, ...],
            colors[None, ...],
            opacities[None, ...],
            width,
            height,
            16,
            offsets[None, ...],
            flatten_ids,
            backgrounds=background[None, ...],
            valid_count=jnp.asarray(5, jnp.int32),
            max_gaussians_per_tile=2,
            max_candidates_per_tile=max_candidates_per_tile,
            return_info=True,
        ),
    )


def _cutile(inputs, max_candidates_per_tile=5, *, width=32, height=16):
    means, conics, colors, opacities, offsets, flatten_ids, background = inputs
    return rasterize_to_pixels_cutile(
        means,
        conics,
        colors,
        opacities,
        width,
        height,
        16,
        offsets,
        flatten_ids,
        backgrounds=background,
        valid_count=jnp.asarray(5, jnp.int32),
        max_gaussians_per_tile=2,
        max_candidates_per_tile=max_candidates_per_tile,
    )


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 8, 16, 32])
def test_cutile_matches_jax_forward_alpha_and_overflow(channels):
    if not _supports_cutile():
        pytest.skip("cuTile compositing requires an NVIDIA CUDA GPU")
    expected = _reference(_inputs(channels))
    actual = jax.jit(lambda: _cutile(_inputs(channels)))()
    np.testing.assert_allclose(actual[0], expected[0][0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1][0], rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(
        actual[2]["tile_overflow"], expected[2]["tile_overflow"][0]
    )
    assert not bool(actual[2]["overflow"])


def test_cutile_matches_jax_edge_cases():
    if not _supports_cutile():
        pytest.skip("cuTile compositing requires an NVIDIA CUDA GPU")
    inputs = _inputs()
    cases = (
        (*inputs[:5], inputs[5].at[1].set(-1), inputs[6]),
        (*inputs[:3], inputs[3].at[0].set(jnp.nan), *inputs[4:]),
    )
    for case in cases:
        expected = _reference(case)
        actual = _cutile(case)
        np.testing.assert_allclose(actual[0], expected[0][0], rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(actual[1], expected[1][0], rtol=3e-5, atol=3e-6)

    expected = _reference(inputs, width=17, height=9)
    actual = _cutile(inputs, width=17, height=9)
    np.testing.assert_allclose(actual[0], expected[0][0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1][0], rtol=3e-5, atol=3e-6)


def test_cutile_preserves_rounded_candidate_bound_overflow():
    if not _supports_cutile():
        pytest.skip("cuTile compositing requires an NVIDIA CUDA GPU")
    expected = _reference(_inputs(), 1)
    actual = _cutile(_inputs(), 1)
    np.testing.assert_allclose(actual[0], expected[0][0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1][0], rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(
        actual[2]["tile_overflow"], expected[2]["tile_overflow"][0]
    )
    assert bool(actual[2]["overflow"])


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 8, 16, 32])
@pytest.mark.parametrize(("width", "height"), [(32, 16), (17, 9)])
def test_cutile_backward_matches_jax_with_repeated_ids(channels, width, height):
    if not _supports_cutile():
        pytest.skip("cuTile compositing requires an NVIDIA CUDA GPU")
    inputs = _inputs(channels)
    # The repeated Gaussian contributes in both tiles, including a partial tile.
    conics = inputs[1].at[1].set(jnp.asarray([0.02, 0.0, 0.02], jnp.float32))
    inputs = (inputs[0], conics, *inputs[2:])
    inputs = (*inputs[:5], inputs[5].at[3].set(1), inputs[6])
    render_cotangent = jnp.linspace(
        -0.7, 0.9, width * height * channels, dtype=jnp.float32
    ).reshape(height, width, channels)
    alpha_cotangent = jnp.linspace(
        0.6, -0.4, width * height, dtype=jnp.float32
    ).reshape(height, width, 1)

    def loss(backend, means, conics, colors, opacities, background):
        current = (means, conics, colors, opacities, inputs[4], inputs[5], background)
        if backend == "jax":
            rendered, alphas = _reference(current, width=width, height=height)[:2]
            rendered, alphas = rendered[0], alphas[0]
        else:
            rendered, alphas = _cutile(current, width=width, height=height)[:2]
        return jnp.sum(rendered * render_cotangent) + jnp.sum(alphas * alpha_cotangent)

    differentiable_inputs = (inputs[0], inputs[1], inputs[2], inputs[3], inputs[6])
    expected = jax.jit(
        jax.value_and_grad(lambda *args: loss("jax", *args), argnums=(0, 1, 2, 3, 4))
    )(*differentiable_inputs)
    actual = jax.jit(
        jax.value_and_grad(
            lambda *args: loss("cuda_tile", *args), argnums=(0, 1, 2, 3, 4)
        )
    )(*differentiable_inputs)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-5, atol=3e-6)
    for actual_gradient, expected_gradient in zip(actual[1], expected[1]):
        np.testing.assert_allclose(
            actual_gradient, expected_gradient, rtol=2e-4, atol=2e-5
        )


@pytest.mark.parametrize(
    ("channels", "count"),
    ((1, 513), (8, 257)),
    ids=("wide-batch", "wide-channel-fallback"),
)
def test_cutile_compositor_crosses_candidate_batch_boundary(channels, count):
    if not _supports_cutile():
        pytest.skip("cuTile compositing requires an NVIDIA CUDA GPU")
    means = jnp.full((count, 2), 0.5, jnp.float32)
    conics = jnp.zeros((count, 3), jnp.float32)
    colors = jnp.linspace(0.1, 0.9, count * channels, dtype=jnp.float32).reshape(
        (count, channels)
    )
    opacities = jnp.full((count,), 0.005, jnp.float32)
    offsets = jnp.zeros((1, 1), jnp.int32)
    ids = jnp.arange(count, dtype=jnp.int32)
    background = jnp.linspace(0.1, 0.2, channels, dtype=jnp.float32)

    def loss(backend, colors_, opacities_):
        if backend == "jax":
            rendered, alpha = rasterize_to_pixels(
                means[None, ...],
                conics[None, ...],
                colors_[None, ...],
                opacities_[None, ...],
                16,
                16,
                16,
                offsets[None, ...],
                ids,
                backgrounds=background[None, ...],
                valid_count=jnp.asarray(count, jnp.int32),
                max_gaussians_per_tile=128,
                max_candidates_per_tile=count,
            )
            rendered, alpha = rendered[0], alpha[0]
        else:
            rendered, alpha, _ = rasterize_to_pixels_cutile(
                means,
                conics,
                colors_,
                opacities_,
                16,
                16,
                16,
                offsets,
                ids,
                backgrounds=background,
                valid_count=jnp.asarray(count, jnp.int32),
                max_gaussians_per_tile=128,
                max_candidates_per_tile=count,
            )
        return jnp.mean(rendered) + 0.01 * jnp.mean(alpha)

    expected = jax.jit(
        jax.value_and_grad(
            lambda colors_, opacities_: loss("jax", colors_, opacities_),
            argnums=(0, 1),
        )
    )(colors, opacities)
    actual = jax.jit(
        jax.value_and_grad(
            lambda colors_, opacities_: loss("cuda_tile", colors_, opacities_),
            argnums=(0, 1),
        )
    )(colors, opacities)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-5, atol=3e-6)
    for actual_gradient, expected_gradient in zip(actual[1], expected[1]):
        np.testing.assert_allclose(
            actual_gradient, expected_gradient, rtol=2e-4, atol=2e-5
        )


def test_cutile_clamp_and_threshold_boundaries_match_jax():
    if not _supports_cutile():
        pytest.skip("cuTile compositing requires an NVIDIA CUDA GPU")
    means = jnp.asarray([[0.5, 0.5], [0.5, 0.5]], jnp.float32)
    conics = jnp.zeros((2, 3), jnp.float32)
    colors = jnp.asarray([[1.0], [2.0]], jnp.float32)
    opacities = jnp.asarray([0.999, 1.0 / 255.0], jnp.float32)
    offsets = jnp.asarray([[0, 2]], jnp.int32)
    ids = jnp.asarray([0, 1, -1, -1], jnp.int32)
    background = jnp.zeros((1,), jnp.float32)

    def loss(backend, opacity):
        current = (means, conics, colors, opacity, offsets, ids, background)
        rendered, alpha = (
            _reference(current, 2)[:2] if backend == "jax" else _cutile(current, 2)[:2]
        )
        if backend == "jax":
            rendered, alpha = rendered[0], alpha[0]
        return rendered[0, 0].sum() + alpha[0, 0].sum()

    expected = jax.value_and_grad(lambda value: loss("jax", value))(opacities)
    actual = jax.value_and_grad(lambda value: loss("cuda_tile", value))(opacities)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-4, atol=2e-5)


def test_cutile_rejects_unsupported_tile_and_channels():
    inputs = _inputs(3)
    with pytest.raises(ValueError, match="tile_size=16"):
        rasterize_to_pixels_cutile(
            *inputs[:4],
            32,
            16,
            8,
            inputs[4],
            inputs[5],
            backgrounds=inputs[6],
            valid_count=jnp.asarray(5, jnp.int32),
        )
    with pytest.raises(ValueError, match="channel counts"):
        rasterize_to_pixels_cutile(
            inputs[0],
            inputs[1],
            jnp.zeros((5, 5), jnp.float32),
            inputs[3],
            32,
            16,
            16,
            inputs[4],
            inputs[5],
            backgrounds=jnp.zeros((5,), jnp.float32),
            valid_count=jnp.asarray(5, jnp.int32),
        )
