from __future__ import annotations

from dataclasses import replace
import importlib
import os
from pathlib import Path
import subprocess
import sys

import jax  # pyright: ignore[reportMissingImports]
import jax.numpy as jnp  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]


def _supports_cuda() -> bool:
    device = jax.devices()[0]
    return device.platform == "gpu" and "cuda" in str(device).lower()


def test_cuda_intersections_ffi_module_import_is_lazy_without_a_library(
    tmp_path,
):
    script = """
import os
os.environ['JAX_GS_CUDA_INTERSECTIONS_FFI_LIBRARY'] = r'%s'
from jax_gs import _cuda_intersections_ffi
assert _cuda_intersections_ffi._LIBRARY is None
assert not _cuda_intersections_ffi._REGISTERED
""" % (tmp_path / "missing.so")
    subprocess.run([sys.executable, "-c", script], check=True)


def test_cuda_intersection_prefix_matches_saturated_jax_scan():
    if not _supports_cuda():
        pytest.skip("CUDA intersection FFI requires an NVIDIA CUDA GPU")
    from jax_gs._cuda_intersections_ffi import intersection_prefix_cuda_ffi

    counts = jnp.asarray(
        [0, 3, 2**30 - 2, 9, 0, 4], dtype=jnp.int32
    )
    capacity = 193

    def reference(values):
        limit = jnp.int32(2**30 - 1)
        cumulative = jax.lax.associative_scan(
            lambda left, right: jnp.where(
                left >= limit - right, limit, left + right
            ),
            jnp.maximum(values, 0),
        )
        required = cumulative[-1]
        return (
            jnp.minimum(cumulative, jnp.int32(capacity + 1)),
            jnp.minimum(required, jnp.int32(capacity)),
            required > capacity,
            required,
        )

    expected = jax.jit(reference)(counts)
    actual = jax.jit(
        lambda values: intersection_prefix_cuda_ffi(
            values, capacity=capacity
        )
    )(counts)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            np.asarray(actual_value), np.asarray(expected_value)
        )


def test_cuda_intersection_sort_offsets_preserves_ties_and_padding():
    if not _supports_cuda():
        pytest.skip("CUDA intersection FFI requires an NVIDIA CUDA GPU")
    from jax_gs._cuda_intersections_ffi import (
        intersection_sort_offsets_cuda_ffi,
    )

    gaussian_ids = jnp.asarray([0, 1, 2, 3, 4, -1, -1], jnp.int32)
    tile_ids = jnp.asarray([2, 1, 1, 1, 0, -1, -1], jnp.int32)
    depths = jnp.asarray([0.0, -0.0, 0.0, -2.0, 1.0], jnp.float32)
    valid_count = jnp.asarray(5, jnp.int32)
    actual = jax.jit(
        lambda: intersection_sort_offsets_cuda_ffi(
            gaussian_ids,
            tile_ids,
            depths,
            valid_count,
            tile_count=4,
        )
    )()

    expected_gaussians = np.asarray([4, 3, 1, 2, 0, -1, -1], np.int32)
    expected_tiles = np.asarray([0, 1, 1, 1, 2, -1, -1], np.int32)
    expected_offsets = np.asarray([0, 1, 4, 5], np.int32)
    expected_valid_count = np.asarray(5, np.int32)
    for actual_value, expected_value in zip(
        actual,
        (
            expected_gaussians,
            expected_tiles,
            expected_offsets,
            expected_valid_count,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual_value), expected_value)


def test_cuda_intersection_sort_offsets_clips_malformed_valid_prefix():
    if not _supports_cuda():
        pytest.skip("CUDA intersection FFI requires an NVIDIA CUDA GPU")
    from jax_gs._cuda_intersections_ffi import (
        intersection_sort_offsets_cuda_ffi,
    )

    actual = jax.jit(
        lambda: intersection_sort_offsets_cuda_ffi(
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
        np.testing.assert_array_equal(np.asarray(actual_value), expected_value)


def test_fused_intersection_compositor_matches_staged_forward_and_gradients():
    if not _supports_cuda():
        pytest.skip("fused CUDA intersection/compositor requires CUDA")
    from jax_gs._cuda_ffi import (
        _cuda_device as compositor_device,
        _ensure_registered as ensure_compositor,
        _run_forward as compositor_forward,
        rasterize_to_pixels_cuda_ffi,
    )
    from jax_gs._cuda_intersections_ffi import (
        _run_intersection_compositor_forward,
        intersection_prefix_cuda_ffi,
        intersection_sort_offsets_cuda_ffi,
        rasterize_accutile_cuda_ffi_fused,
    )
    from jax_gs.intersections import (
        _count_accutile_intersections_jax,
        _emit_accutile_intersections_jax,
        _prepare_accutile_state_jax,
    )

    ensure_compositor(compositor_device())
    tile_size = 16
    tile_width = tile_height = 2
    capacity = 17
    means2d = jnp.asarray(
        [[8, 8], [24, 8], [8, 24], [24, 24], [16, 16]], jnp.float32
    )
    radii = jnp.full((5, 2), 12, jnp.float32)
    depths = jnp.asarray([1.0, 0.0, -0.0, 2.0, jnp.nan], jnp.float32)
    conics = jnp.tile(jnp.asarray([[0.04, 0.0, 0.04]], jnp.float32), (5, 1))
    colors = jnp.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]],
        jnp.float32,
    )
    opacities = jnp.asarray([0.8, 0.7, 0.6, 0.5, 0.9], jnp.float32)
    valid = jnp.asarray([True, True, True, True, True])

    def prepared(means, conic_values, opacity_values):
        return _prepare_accutile_state_jax(
            jax.lax.stop_gradient(means),
            radii,
            jax.lax.stop_gradient(conic_values),
            jax.lax.stop_gradient(opacity_values),
            valid & jnp.isfinite(depths),
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            alpha_threshold=1 / 255,
        )

    def staged_topology(means, conic_values, opacity_values):
        state = prepared(means, conic_values, opacity_values)
        counts = _count_accutile_intersections_jax(
            state,
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        cumulative, count, overflow, required = intersection_prefix_cuda_ffi(
            counts, capacity=capacity
        )
        gaussian_ids, tile_ids = _emit_accutile_intersections_jax(
            state,
            cumulative,
            count,
            capacity=capacity,
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        gaussian_ids, tile_ids, offsets, count = (
            intersection_sort_offsets_cuda_ffi(
                gaussian_ids,
                tile_ids,
                depths,
                count,
                tile_count=tile_width * tile_height,
            )
        )
        return gaussian_ids, tile_ids, offsets, count, overflow, required

    def staged_raw(means, conic_values, color_values, opacity_values):
        topology = staged_topology(means, conic_values, opacity_values)
        forward = compositor_forward(
            means,
            conic_values,
            color_values,
            opacity_values,
            topology[2].reshape((tile_height, tile_width)),
            topology[0],
            topology[3],
            image_width=32,
            image_height=32,
            per_tile_bound=32,
            alpha_threshold=1 / 255,
            transmittance_threshold=1e-4,
        )
        return *topology, *forward

    def fused_raw(means, conic_values, color_values, opacity_values):
        return _run_intersection_compositor_forward(
            prepared(means, conic_values, opacity_values),
            depths,
            means,
            conic_values,
            color_values,
            opacity_values,
            capacity=capacity,
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            image_width=32,
            image_height=32,
            per_tile_bound=32,
            alpha_threshold=1 / 255,
            transmittance_threshold=1e-4,
        )

    staged = jax.jit(staged_raw)(means2d, conics, colors, opacities)
    fused = jax.jit(fused_raw)(means2d, conics, colors, opacities)
    for actual, expected in zip(fused, staged, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def staged_loss(means, conic_values, color_values, opacity_values):
        topology = staged_topology(
            jax.lax.stop_gradient(means),
            jax.lax.stop_gradient(conic_values),
            jax.lax.stop_gradient(opacity_values),
        )
        rendered, alpha, _ = rasterize_to_pixels_cuda_ffi(
            means,
            conic_values,
            color_values,
            opacity_values,
            32,
            32,
            tile_size,
            topology[2].reshape((tile_height, tile_width)),
            topology[0],
            valid_count=topology[3],
            backgrounds=jnp.zeros((3,), jnp.float32),
            max_gaussians_per_tile=32,
            max_candidates_per_tile=32,
        )
        return jnp.mean(rendered) + jnp.mean(alpha)

    def fused_loss(means, conic_values, color_values, opacity_values):
        rendered, alpha, _ = rasterize_accutile_cuda_ffi_fused(
            prepared(means, conic_values, opacity_values),
            depths,
            means,
            conic_values,
            color_values,
            opacity_values,
            capacity=capacity,
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            image_width=32,
            image_height=32,
            background=jnp.zeros((3,), jnp.float32),
            max_gaussians_per_tile=32,
            max_candidates_per_tile=32,
            alpha_threshold=1 / 255,
            transmittance_threshold=1e-4,
        )
        return jnp.mean(rendered) + jnp.mean(alpha)

    staged_value, staged_gradients = jax.jit(
        jax.value_and_grad(staged_loss, argnums=(0, 1, 2, 3))
    )(means2d, conics, colors, opacities)
    fused_value, fused_gradients = jax.jit(
        jax.value_and_grad(fused_loss, argnums=(0, 1, 2, 3))
    )(means2d, conics, colors, opacities)
    np.testing.assert_array_equal(np.asarray(fused_value), np.asarray(staged_value))
    for actual, expected in zip(fused_gradients, staged_gradients, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-7
        )


def _small_strict_raster_case():
    means = jnp.asarray(
        [
            [-0.3, -0.2, 3.0],
            [0.2, -0.1, 3.2],
            [-0.1, 0.2, 2.8],
            [0.3, 0.25, 3.5],
        ],
        jnp.float32,
    )
    quats = jnp.tile(
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]], jnp.float32), (4, 1)
    )
    scales = jnp.full((4, 3), 0.2, jnp.float32)
    opacities = jnp.asarray([0.8, 0.7, 0.6, 0.5], jnp.float32)
    colors = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        jnp.float32,
    )
    viewmats = jnp.eye(4, dtype=jnp.float32)[None, ...]
    intrinsics = jnp.asarray(
        [[[30.0, 0.0, 9.0], [0.0, 30.0, 8.5], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )
    return (
        means,
        quats,
        scales,
        opacities,
        colors,
        jnp.ones((4,), jnp.bool_),
        viewmats,
        intrinsics,
    )


def _strict_fused_config():
    from jax_gs.config import RasterizationConfig

    return RasterizationConfig(
        backend="intersections",
        projection_backend="cuda_ffi_strict",
        intersection_backend="cuda_tile_cub",
        compositor_backend="cuda_ffi",
        intersection_mode="accutile",
        rasterize_mode="classic",
        tile_size=16,
        radius_clip=0.0,
        max_intersections=64,
        max_gaussians_per_tile=32,
        max_candidates_per_tile=32,
    )


def test_rasterization_strict_triple_routes_fused_with_full_parity(monkeypatch):
    if not _supports_cuda():
        pytest.skip("fused CUDA rasterization requires CUDA")
    fused_module = importlib.import_module("jax_gs._cuda_intersections_ffi")
    raster_module = importlib.import_module("jax_gs.rasterization")
    original_fused = fused_module.rasterize_accutile_cuda_ffi_fused
    calls = []

    def spy(*args, **kwargs):
        calls.append(True)
        return original_fused(*args, **kwargs)

    monkeypatch.setattr(
        fused_module, "rasterize_accutile_cuda_ffi_fused", spy
    )
    inputs = _small_strict_raster_case()
    fused_config = _strict_fused_config()
    staged_config = replace(fused_config, intersection_backend="jax")
    background = jnp.asarray([[0.1, 0.2, 0.3]], jnp.float32)

    def make_forward(name, render_config):
        def forward(*values):
            (
                means,
                quats,
                scales,
                opacities,
                colors,
                mask,
                viewmats,
                intrinsics,
            ) = values
            rendered, alpha, info = raster_module.rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats,
                intrinsics,
                18,
                17,
                backgrounds=background,
                active_mask=mask,
                config=render_config,
            )
            return (
                rendered,
                alpha,
                info["candidate_counts"],
                info["intersection_count"],
                info["intersection_required_count"],
                info["intersection_overflow"],
                info["tile_overflow"],
                info["flatten_ids"],
                info["isect_ids"],
                info["isect_offsets"],
                info["isect_valid_count"],
            )

        forward.__name__ = name
        return forward

    staged_fn = jax.jit(make_forward("staged_forward", staged_config))
    staged = staged_fn(*inputs)
    jax.block_until_ready(staged)
    assert not calls

    fused_fn = jax.jit(make_forward("fused_forward", fused_config))
    fused = fused_fn(*inputs)
    jax.block_until_ready(fused)
    assert calls
    for actual, expected in zip(fused, staged, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def make_loss(name, render_config):
        def loss(
            means,
            quats,
            scales,
            opacities,
            colors,
            mask,
            viewmats,
            intrinsics,
        ):
            rendered, alpha, _ = raster_module.rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats,
                intrinsics,
                18,
                17,
                backgrounds=background,
                active_mask=mask,
                config=render_config,
            )
            return jnp.mean(rendered) + jnp.mean(alpha)

        loss.__name__ = name
        return jax.jit(
            jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))
        )

    staged_value, staged_gradients = make_loss(
        "staged_loss", staged_config
    )(*inputs)
    fused_value, fused_gradients = make_loss("fused_loss", fused_config)(
        *inputs
    )
    np.testing.assert_array_equal(np.asarray(fused_value), np.asarray(staged_value))
    for actual, expected in zip(fused_gradients, staged_gradients, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-7
        )


def test_rasterization_adjacent_backend_avoids_fused_path(monkeypatch):
    if not _supports_cuda():
        pytest.skip("CUDA rasterization requires CUDA")
    fused_module = importlib.import_module("jax_gs._cuda_intersections_ffi")
    raster_module = importlib.import_module("jax_gs.rasterization")

    def fail(*args, **kwargs):
        raise AssertionError("adjacent backend unexpectedly selected fusion")

    monkeypatch.setattr(
        fused_module, "rasterize_accutile_cuda_ffi_fused", fail
    )
    config = replace(_strict_fused_config(), intersection_backend="jax")
    means, quats, scales, opacities, colors, mask, viewmats, intrinsics = (
        _small_strict_raster_case()
    )
    rendered, alpha, _ = jax.jit(
        lambda: raster_module.rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            intrinsics,
            18,
            17,
            backgrounds=jnp.asarray([[0.1, 0.2, 0.3]], jnp.float32),
            active_mask=mask,
            config=config,
        )
    )()
    jax.block_until_ready((rendered, alpha))


def test_cuda_tile_cub_rejects_non_float32_depths_before_native_launch():
    script = """
import jax.numpy as jnp
from jax_gs.intersections import intersect_tiles
try:
    intersect_tiles(
        jnp.zeros((1, 2), jnp.float64),
        jnp.ones((1, 2), jnp.float64),
        jnp.ones((1,), jnp.float64),
        jnp.ones((1,), jnp.bool_),
        tile_size=4,
        tile_width=1,
        tile_height=1,
        max_intersections=1,
        backend='cuda_tile_cub',
        conics=jnp.ones((1, 3), jnp.float64),
        opacities=jnp.ones((1,), jnp.float64),
        mode='accutile',
    )
except ValueError as exc:
    assert 'float32 depths' in str(exc)
else:
    raise AssertionError('float64 depths were accepted')
"""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "1"
    subprocess.run(
        [sys.executable, "-c", script], env=environment, check=True
    )


def test_cuda_intersections_ffi_rejects_missing_prebuilt_library(
    tmp_path, monkeypatch
):
    if not _supports_cuda():
        pytest.skip("CUDA intersection FFI requires an NVIDIA CUDA GPU")
    module = importlib.import_module("jax_gs._cuda_intersections_ffi")
    monkeypatch.setattr(module, "_REGISTERED", False)
    monkeypatch.setattr(module, "_LIBRARY", None)
    missing = Path(tmp_path) / "missing.so"
    monkeypatch.setenv(
        "JAX_GS_CUDA_INTERSECTIONS_FFI_LIBRARY", str(missing)
    )
    with pytest.raises(RuntimeError, match="does not name a file"):
        module.intersection_prefix_cuda_ffi(
            jnp.asarray([1], jnp.int32), capacity=1
        )
