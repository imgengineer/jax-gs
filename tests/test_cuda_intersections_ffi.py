from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


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
