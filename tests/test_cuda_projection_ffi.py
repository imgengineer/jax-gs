# pyright: reportMissingImports=false

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax_gs._cuda_projection_ffi import fully_fused_projection_cuda_ffi
from jax_gs.cameras import fully_fused_projection
from jax_gs.config import RasterizationConfig
from jax_gs.rasterization import rasterization


def _supports_cuda_ffi() -> bool:
    device = jax.devices()[0]
    return device.platform == "gpu" and "cuda" in str(device).lower()


def _inputs(gaussian_count=257, camera_count=2, seed=7):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 5)
    means = jax.random.normal(
        keys[0], (gaussian_count, 3), dtype=jnp.float32
    ).at[:, 2].add(4.0)
    quats = jax.random.normal(
        keys[1], (gaussian_count, 4), dtype=jnp.float32
    )
    scales = jnp.exp(
        jax.random.normal(
            keys[2], (gaussian_count, 3), dtype=jnp.float32
        )
        - 3.0
    )
    opacities = jax.nn.sigmoid(
        jax.random.normal(keys[3], (gaussian_count,), dtype=jnp.float32)
    )
    colors = jax.nn.sigmoid(
        jax.random.normal(
            keys[4], (gaussian_count, 3), dtype=jnp.float32
        )
    )
    active_mask = jnp.arange(gaussian_count) % 7 != 0
    viewmats = jnp.broadcast_to(
        jnp.eye(4, dtype=jnp.float32), (camera_count, 4, 4)
    )
    if camera_count > 1:
        viewmats = viewmats.at[1, 0, 3].set(0.2)
    intrinsics = jnp.broadcast_to(
        jnp.asarray(
            [[500.0, 0.0, 320.0], [0.0, 510.0, 180.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        ),
        (camera_count, 3, 3),
    )
    return (
        means,
        quats,
        scales,
        opacities,
        colors,
        active_mask,
        viewmats,
        intrinsics,
    )


def _project(function, inputs, *, compensations=False):
    means, quats, scales, opacities, _, active_mask, viewmats, intrinsics = inputs
    return function(
        means,
        viewmats,
        intrinsics,
        640,
        360,
        quats=quats,
        scales=scales,
        opacities=opacities,
        active_mask=active_mask,
        radius_clip=3.0,
        calc_compensations=compensations,
    )


def test_cuda_projection_is_publicly_exported():
    import jax_gs

    assert (
        jax_gs.fully_fused_projection_cuda_ffi
        is fully_fused_projection_cuda_ffi
    )


def test_cuda_projection_module_import_is_lazy_without_a_library(tmp_path):
    environment = {
        **os.environ,
        "JAX_GS_CUDA_PROJECTION_FFI_LIBRARY": str(tmp_path / "missing.so"),
    }
    result = subprocess.run(
        [sys.executable, "-c", "import jax_gs._cuda_projection_ffi"],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cuda_projection_rejects_non_float32_before_launch():
    inputs = list(_inputs(4, 1))
    inputs[0] = inputs[0].astype(jnp.float16)
    with pytest.raises(ValueError, match="float32"):
        _project(fully_fused_projection_cuda_ffi, tuple(inputs))


@pytest.mark.parametrize("compensations", [False, True])
def test_cuda_projection_matches_dense_jax_contract(compensations):
    if not _supports_cuda_ffi():
        pytest.skip("strict CUDA projection requires an NVIDIA CUDA GPU")
    inputs = _inputs()
    arguments = (*inputs[:4], inputs[5], *inputs[6:])

    def project(
        function,
        means,
        quats,
        scales,
        opacities,
        active_mask,
        viewmats,
        intrinsics,
    ):
        values = (
            means,
            quats,
            scales,
            opacities,
            inputs[4],
            active_mask,
            viewmats,
            intrinsics,
        )
        return _project(function, values, compensations=compensations)

    expected = jax.jit(
        lambda *values: project(fully_fused_projection, *values)
    )(*arguments)
    actual = jax.jit(
        lambda *values: project(fully_fused_projection_cuda_ffi, *values)
    )(*arguments)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_array_equal(actual[3], expected[3])
    if compensations:
        np.testing.assert_array_equal(actual[4], expected[4])
    else:
        assert actual[4] is expected[4] is None
    np.testing.assert_array_equal(actual[5], expected[5])


def test_cuda_projection_preserves_adversarial_topology_boundaries():
    if not _supports_cuda_ffi():
        pytest.skip("strict CUDA projection requires an NVIDIA CUDA GPU")
    inputs = list(_inputs(16, 2))
    means = inputs[0]
    means = means.at[0, 2].set(0.0)
    means = means.at[1, 2].set(jnp.float32(0.01))
    means = means.at[2, 2].set(
        jnp.nextafter(jnp.float32(0.01), jnp.float32(0.0))
    )
    means = means.at[3, 2].set(
        jnp.nextafter(jnp.float32(0.01), jnp.float32(1.0))
    )
    means = means.at[4].set(jnp.asarray([jnp.nan, 0.0, 1.0]))
    means = means.at[5].set(jnp.asarray([jnp.inf, 0.0, 1.0]))
    means = means.at[6].set(jnp.asarray([20.0, -20.0, 1.0]))
    inputs[0] = means
    inputs[1] = inputs[1].at[7].set(jnp.zeros(4, jnp.float32))
    inputs[1] = inputs[1].at[8].set(jnp.full(4, 1.0e-20, jnp.float32))
    inputs[3] = inputs[3].at[9].set(jnp.float32(1.0 / 255.0))
    inputs[3] = inputs[3].at[10].set(
        jnp.nextafter(jnp.float32(1.0 / 255.0), jnp.float32(0.0))
    )
    inputs[5] = inputs[5].at[11].set(False)
    inputs = tuple(inputs)
    arguments = (*inputs[:4], inputs[5], *inputs[6:])

    def project(function, means_, quats_, scales_, opacities_, mask_, views_, Ks_):
        return function(
            means_,
            views_,
            Ks_,
            640,
            360,
            quats=quats_,
            scales=scales_,
            opacities=opacities_,
            active_mask=mask_,
            radius_clip=3.0,
        )

    expected = jax.jit(
        lambda *values: project(fully_fused_projection, *values)
    )(*arguments)
    actual = jax.jit(
        lambda *values: project(fully_fused_projection_cuda_ffi, *values)
    )(*arguments)
    for actual_output, expected_output in zip(actual, expected):
        if actual_output is None:
            assert expected_output is None
        else:
            np.testing.assert_array_equal(actual_output, expected_output)


def test_cuda_projection_vjp_matches_authoritative_jax_recompute():
    if not _supports_cuda_ffi():
        pytest.skip("strict CUDA projection requires an NVIDIA CUDA GPU")
    inputs = _inputs(97, 2)
    means, quats, scales, opacities, _, active_mask, viewmats, intrinsics = inputs
    weights = _inputs(97, 2, seed=19)
    means_weight = jnp.broadcast_to(weights[0][None, :, :2], (2, 97, 2))
    depth_weight = jnp.broadcast_to(weights[0][None, :, 2], (2, 97))
    conic_weight = jnp.broadcast_to(weights[2][None, :, :], (2, 97, 3))

    def loss(function, means_, quats_, scales_, opacities_, viewmats_, intrinsics_):
        _, means2d, depths, conics, _, _ = function(
            means_,
            viewmats_,
            intrinsics_,
            640,
            360,
            quats=quats_,
            scales=scales_,
            opacities=opacities_,
            active_mask=active_mask,
            radius_clip=3.0,
        )
        return (
            jnp.vdot(means2d, means_weight)
            + jnp.vdot(depths, depth_weight)
            + jnp.vdot(conics, conic_weight)
        )

    differentiate = lambda function: jax.jit(
        jax.grad(
            lambda *values: loss(function, *values),
            argnums=(0, 1, 2, 3, 4, 5),
        )
    )(means, quats, scales, opacities, viewmats, intrinsics)
    expected = differentiate(fully_fused_projection)
    actual = differentiate(fully_fused_projection_cuda_ffi)
    for actual_gradient, expected_gradient in zip(actual, expected):
        np.testing.assert_array_equal(actual_gradient, expected_gradient)


def test_cuda_projection_renderer_preserves_intersection_topology():
    if not _supports_cuda_ffi():
        pytest.skip("strict CUDA projection requires an NVIDIA CUDA GPU")
    inputs = list(_inputs(64, 1))
    inputs[-1] = jnp.asarray(
        [[[50.0, 0.0, 32.0], [0.0, 51.0, 24.0], [0.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )
    inputs = tuple(inputs)
    means, quats, scales, opacities, colors, active_mask, viewmats, intrinsics = inputs

    def render(
        projection_backend,
        means_,
        quats_,
        scales_,
        opacities_,
        colors_,
        active_mask_,
        viewmats_,
        intrinsics_,
    ):
        return rasterization(
            means_,
            quats_,
            scales_,
            opacities_,
            colors_,
            viewmats_,
            intrinsics_,
            64,
            48,
            active_mask=active_mask_,
            config=RasterizationConfig(
                backend="intersections",
                projection_backend=projection_backend,
                intersection_backend="jax",
                intersection_mode="accutile",
                compositor_backend="jax",
                max_intersections=4096,
            ),
        )

    expected = jax.jit(lambda *values: render("jax", *values))(*inputs)
    actual = jax.jit(
        lambda *values: render("cuda_ffi_strict", *values)
    )(*inputs)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    for name in (
        "radii",
        "valid",
        "flatten_ids",
        "isect_ids",
        "isect_offsets",
        "isect_valid_count",
        "intersection_count",
        "intersection_required_count",
    ):
        np.testing.assert_array_equal(actual[2][name], expected[2][name])
    assert int(actual[2]["intersection_count"][0]) > 0

    def loss(projection_backend, means_, quats_, scales_, opacities_, colors_):
        rendered, alpha, _ = render(
            projection_backend,
            means_,
            quats_,
            scales_,
            opacities_,
            colors_,
            active_mask,
            viewmats,
            intrinsics,
        )
        return jnp.mean(rendered) + 0.01 * jnp.mean(alpha)

    differentiate = lambda projection_backend: jax.jit(
        jax.grad(
            lambda *values: loss(projection_backend, *values),
            argnums=(0, 1, 2, 3, 4),
        )
    )(means, quats, scales, opacities, colors)
    expected_gradients = differentiate("jax")
    actual_gradients = differentiate("cuda_ffi_strict")
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients
    ):
        np.testing.assert_allclose(
            actual_gradient,
            expected_gradient,
            rtol=1e-4,
            atol=1e-7,
        )


@pytest.mark.resource_heavy
def test_cuda_projection_factor_route_preserves_topology():
    if not _supports_cuda_ffi():
        pytest.skip("strict CUDA projection requires an NVIDIA CUDA GPU")
    gaussian_count = 262_144
    active_count = 64
    base = _inputs(active_count, 1)
    means = jnp.zeros((gaussian_count, 3), jnp.float32).at[:active_count].set(base[0])
    quats = jnp.zeros((gaussian_count, 4), jnp.float32).at[:active_count].set(base[1])
    scales = jnp.zeros((gaussian_count, 3), jnp.float32).at[:active_count].set(base[2])
    opacities = jnp.zeros((gaussian_count,), jnp.float32).at[:active_count].set(base[3])
    colors = jnp.zeros((gaussian_count, 3), jnp.float32)
    active_mask = jnp.arange(gaussian_count) < active_count
    inputs = (means, quats, scales, opacities, colors, active_mask, base[6], base[7])
    expected = jax.jit(lambda: _project(fully_fused_projection, inputs))()
    actual = jax.jit(
        lambda: _project(fully_fused_projection_cuda_ffi, inputs)
    )()
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_array_equal(actual[5], expected[5])
