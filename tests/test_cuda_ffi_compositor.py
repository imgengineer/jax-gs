import os
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax_gs._cuda_ffi import rasterize_to_pixels_cuda_ffi
from jax_gs.low_level import rasterize_to_pixels


def _supports_cuda_ffi():
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
    colors = base_colors[:, :channels]
    opacities = jnp.asarray([0.5, 0.4, 0.3, 0.2, 0.1], jnp.float32)
    offsets = jnp.asarray([[0, 3]], jnp.int32)
    flatten_ids = jnp.asarray(
        [0, 1, 4, 2, 3, -1, -1, -1, -1, -1], jnp.int32
    )
    background = jnp.linspace(0.05, 0.2, channels, dtype=jnp.float32)
    return means, conics, colors, opacities, offsets, flatten_ids, background


def _reference(inputs, max_candidates_per_tile=5):
    means, conics, colors, opacities, offsets, flatten_ids, background = inputs
    return cast(
        tuple[Any, Any, dict[str, Any]],
        rasterize_to_pixels(
            means[None, ...],
            conics[None, ...],
            colors[None, ...],
            opacities[None, ...],
            32,
            16,
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


def _cuda_ffi(inputs, max_candidates_per_tile=5):
    means, conics, colors, opacities, offsets, flatten_ids, background = inputs
    return rasterize_to_pixels_cuda_ffi(
        means,
        conics,
        colors,
        opacities,
        32,
        16,
        16,
        offsets,
        flatten_ids,
        backgrounds=background,
        valid_count=jnp.asarray(5, jnp.int32),
        max_gaussians_per_tile=2,
        max_candidates_per_tile=max_candidates_per_tile,
    )


def test_cuda_ffi_module_import_is_lazy_without_a_library(tmp_path):
    environment = {
        **os.environ,
        "JAX_GS_CUDA_FFI_LIBRARY": str(tmp_path / "missing.so"),
    }
    result = subprocess.run(
        [sys.executable, "-c", "import jax_gs._cuda_ffi"],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_cuda_ffi_matches_jax_forward_alpha_and_overflow(channels):
    if not _supports_cuda_ffi():
        pytest.skip("CUDA FFI compositing requires an NVIDIA CUDA GPU")
    expected = _reference(_inputs(channels), 5)
    actual = jax.jit(lambda: _cuda_ffi(_inputs(channels), 5))()
    np.testing.assert_allclose(actual[0], expected[0][0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1][0], rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(
        np.asarray(actual[2]["tile_overflow"]),
        np.asarray(expected[2]["tile_overflow"][0]),
    )
    assert not bool(actual[2]["overflow"])


def test_cuda_ffi_ignores_negative_ids_inside_valid_prefix():
    if not _supports_cuda_ffi():
        pytest.skip("CUDA FFI compositing requires an NVIDIA CUDA GPU")
    inputs = _inputs()
    inputs = (*inputs[:5], inputs[5].at[1].set(-1), inputs[6])
    expected = _reference(inputs)
    actual = _cuda_ffi(inputs)
    np.testing.assert_allclose(actual[0], expected[0][0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1][0], rtol=3e-5, atol=3e-6)


def test_cuda_ffi_nan_opacity_matches_jax():
    if not _supports_cuda_ffi():
        pytest.skip("CUDA FFI compositing requires an NVIDIA CUDA GPU")
    inputs = _inputs()
    inputs = (*inputs[:3], inputs[3].at[0].set(jnp.nan), *inputs[4:])
    expected = _reference(inputs)
    actual = _cuda_ffi(inputs)
    np.testing.assert_allclose(actual[0], expected[0][0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1][0], rtol=3e-5, atol=3e-6)


def test_cuda_ffi_partial_tiles_match_jax():
    if not _supports_cuda_ffi():
        pytest.skip("CUDA FFI compositing requires an NVIDIA CUDA GPU")
    means, conics, colors, opacities, offsets, ids, background = _inputs()
    expected = rasterize_to_pixels(
        means[None, ...],
        conics[None, ...],
        colors[None, ...],
        opacities[None, ...],
        17,
        9,
        16,
        offsets[None, ...],
        ids,
        backgrounds=background[None, ...],
        valid_count=jnp.asarray(5, jnp.int32),
        max_gaussians_per_tile=2,
        max_candidates_per_tile=5,
        return_info=True,
    )
    actual = rasterize_to_pixels_cuda_ffi(
        means,
        conics,
        colors,
        opacities,
        17,
        9,
        16,
        offsets,
        ids,
        backgrounds=background,
        valid_count=jnp.asarray(5, jnp.int32),
        max_gaussians_per_tile=2,
        max_candidates_per_tile=5,
    )
    np.testing.assert_allclose(actual[0], expected[0][0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1][0], rtol=3e-5, atol=3e-6)


def test_cuda_ffi_preserves_rounded_candidate_bound_overflow():
    if not _supports_cuda_ffi():
        pytest.skip("CUDA FFI compositing requires an NVIDIA CUDA GPU")
    expected = _reference(_inputs(), 1)
    actual = _cuda_ffi(_inputs(), 1)
    np.testing.assert_allclose(actual[0], expected[0][0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1][0], rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(
        np.asarray(actual[2]["tile_overflow"]),
        np.asarray(expected[2]["tile_overflow"][0]),
    )
    assert bool(actual[2]["overflow"])


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_cuda_ffi_backward_matches_jax_with_repeated_ids(channels):
    if not _supports_cuda_ffi():
        pytest.skip("CUDA FFI compositing requires an NVIDIA CUDA GPU")
    inputs = _inputs(channels)
    inputs = (*inputs[:5], inputs[5].at[3].set(1), inputs[6])
    render_cotangent = jnp.linspace(
        -0.7, 0.9, 32 * 16 * channels, dtype=jnp.float32
    ).reshape(16, 32, channels)
    alpha_cotangent = jnp.linspace(
        0.6, -0.4, 32 * 16, dtype=jnp.float32
    ).reshape(16, 32, 1)

    def loss(backend, means, conics, colors, opacities, background):
        current = (means, conics, colors, opacities, inputs[4], inputs[5], background)
        if backend == "jax":
            rendered, alphas = _reference(current)[:2]
            rendered, alphas = rendered[0], alphas[0]
        else:
            rendered, alphas = _cuda_ffi(current)[:2]
        return jnp.sum(rendered * render_cotangent) + jnp.sum(
            alphas * alpha_cotangent
        )

    differentiable_inputs = (inputs[0], inputs[1], inputs[2], inputs[3], inputs[6])
    expected = jax.jit(
        jax.value_and_grad(
            lambda *args: loss("jax", *args), argnums=(0, 1, 2, 3, 4)
        )
    )(*differentiable_inputs)
    actual = jax.jit(
        jax.value_and_grad(
            lambda *args: loss("cuda_ffi", *args), argnums=(0, 1, 2, 3, 4)
        )
    )(*differentiable_inputs)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-5, atol=3e-6)
    for actual_gradient, expected_gradient in zip(actual[1], expected[1]):
        np.testing.assert_allclose(
            actual_gradient, expected_gradient, rtol=2e-4, atol=2e-5
        )


def test_cuda_ffi_rejects_missing_prebuilt_library(monkeypatch, tmp_path):
    if not _supports_cuda_ffi():
        pytest.skip("CUDA FFI compositing requires an NVIDIA CUDA GPU")
    import jax_gs._cuda_ffi as cuda_ffi_module

    monkeypatch.setattr(cuda_ffi_module, "_REGISTERED", False)
    monkeypatch.setattr(cuda_ffi_module, "_LIBRARY", None)
    monkeypatch.setenv(
        "JAX_GS_CUDA_FFI_LIBRARY", str(tmp_path / "missing.so")
    )
    with pytest.raises(RuntimeError, match="does not name a file"):
        _cuda_ffi(_inputs())


def test_cuda_ffi_clips_valid_count_to_input_capacity():
    if not _supports_cuda_ffi():
        pytest.skip("CUDA FFI compositing requires an NVIDIA CUDA GPU")
    inputs = _inputs()
    means, conics, colors, opacities, offsets, ids, background = inputs
    expected = rasterize_to_pixels_cuda_ffi(
        means,
        conics,
        colors,
        opacities,
        32,
        16,
        16,
        offsets,
        ids,
        backgrounds=background,
        valid_count=jnp.asarray(ids.shape[0], jnp.int32),
        max_gaussians_per_tile=2,
        max_candidates_per_tile=5,
    )
    actual = rasterize_to_pixels_cuda_ffi(
        means,
        conics,
        colors,
        opacities,
        32,
        16,
        16,
        offsets,
        ids,
        backgrounds=background,
        valid_count=jnp.asarray(ids.shape[0] + 100, jnp.int32),
        max_gaussians_per_tile=2,
        max_candidates_per_tile=5,
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=3e-5, atol=3e-6)


def test_cuda_ffi_rejects_unsupported_shape_and_channels():
    inputs = _inputs(3)
    with pytest.raises(ValueError, match="tile_size=16"):
        rasterize_to_pixels_cuda_ffi(
            *inputs[:4], 32, 16, 8, inputs[4], inputs[5],
            backgrounds=inputs[6], valid_count=jnp.asarray(5, jnp.int32),
        )
    colors = jnp.zeros((5, 5), jnp.float32)
    with pytest.raises(ValueError, match="channel counts"):
        rasterize_to_pixels_cuda_ffi(
            inputs[0], inputs[1], colors, inputs[3], 32, 16, 16,
            inputs[4], inputs[5], backgrounds=jnp.zeros((5,), jnp.float32),
            valid_count=jnp.asarray(5, jnp.int32),
        )


def test_cuda_ffi_clamp_and_threshold_boundaries_match_jax():
    if not _supports_cuda_ffi():
        pytest.skip("CUDA FFI compositing requires an NVIDIA CUDA GPU")
    means = jnp.asarray([[0.5, 0.5], [0.5, 0.5]], jnp.float32)
    conics = jnp.zeros((2, 3), jnp.float32)
    colors = jnp.asarray([[1.0], [2.0]], jnp.float32)
    opacities = jnp.asarray([0.999, 1.0 / 255.0], jnp.float32)
    offsets = jnp.asarray([[0, 2]], jnp.int32)
    ids = jnp.asarray([0, 1, -1, -1], jnp.int32)
    background = jnp.zeros((1,), jnp.float32)
    inputs = (means, conics, colors, opacities, offsets, ids, background)

    def loss(backend, opacity):
        current = (means, conics, colors, opacity, offsets, ids, background)
        rendered, alpha = (
            _reference(current, 2)[:2]
            if backend == "jax"
            else _cuda_ffi(current, 2)[:2]
        )
        if backend == "jax":
            rendered, alpha = rendered[0], alpha[0]
        return rendered[0, 0].sum() + alpha[0, 0].sum()

    expected = jax.value_and_grad(lambda value: loss("jax", value))(opacities)
    actual = jax.value_and_grad(lambda value: loss("cuda_ffi", value))(opacities)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-4, atol=2e-5)
