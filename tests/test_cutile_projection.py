import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs._cutile_projection import fully_fused_projection_cutile
from jax_gs.cameras import fully_fused_projection
from jax_gs.config import RasterizationConfig
from jax_gs.rasterization import rasterization

pytestmark = pytest.mark.gpu


def _supports_cutile() -> bool:
    return jax.devices()[0].platform == "gpu"


def _inputs(gaussian_count=257, camera_count=2, seed=7):
    keys = jax.random.split(jax.random.key(seed), 5)
    means = (
        jax.random.normal(keys[0], (gaussian_count, 3), dtype=jnp.float32)
        .at[:, 2]
        .add(4.0)
    )
    quats = jax.random.normal(keys[1], (gaussian_count, 4), dtype=jnp.float32)
    scales = jnp.exp(
        jax.random.normal(keys[2], (gaussian_count, 3), dtype=jnp.float32) - 3.0
    )
    opacities = jax.nn.sigmoid(
        jax.random.normal(keys[3], (gaussian_count,), dtype=jnp.float32)
    )
    active_mask = jnp.arange(gaussian_count) % 7 != 0
    viewmats = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (camera_count, 4, 4))
    if camera_count > 1:
        viewmats = viewmats.at[1, 0, 3].set(0.2)
    intrinsics = jnp.broadcast_to(
        jnp.asarray(
            [[500.0, 0.0, 320.0], [0.0, 510.0, 180.0], [0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        ),
        (camera_count, 3, 3),
    )
    return means, quats, scales, opacities, active_mask, viewmats, intrinsics


def _project(function, inputs):
    means, quats, scales, opacities, active_mask, viewmats, intrinsics = inputs
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
    )


def _assert_projection_matches(actual, expected):
    for index in (0, 1, 2, 5):
        np.testing.assert_array_equal(actual[index], expected[index])
    assert actual[4] is expected[4] is None
    np.testing.assert_allclose(
        actual[3], expected[3], rtol=2e-4, atol=1e-6, equal_nan=True
    )


def test_cutile_projection_matches_jax_and_preserves_topology():
    if not _supports_cutile():
        pytest.skip("cuTile projection requires a GPU")
    inputs = _inputs()
    expected = jax.jit(lambda *values: _project(fully_fused_projection, values))(
        *inputs
    )
    actual = jax.jit(lambda *values: _project(fully_fused_projection_cutile, values))(
        *inputs
    )
    _assert_projection_matches(actual, expected)


def test_cutile_projection_matches_antialiased_compensation():
    if not _supports_cutile():
        pytest.skip("cuTile projection requires a GPU")
    inputs = _inputs()

    def project(function, values):
        means, quats, scales, opacities, active_mask, viewmats, intrinsics = values
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
            calc_compensations=True,
        )

    expected = jax.jit(lambda *values: project(fully_fused_projection, values))(*inputs)
    actual = jax.jit(lambda *values: project(fully_fused_projection_cutile, values))(
        *inputs
    )
    for index in (0, 1, 2, 5):
        np.testing.assert_array_equal(actual[index], expected[index])
    np.testing.assert_allclose(
        actual[3], expected[3], rtol=2e-4, atol=1e-6, equal_nan=True
    )
    np.testing.assert_allclose(
        actual[4], expected[4], rtol=2e-4, atol=1e-6, equal_nan=True
    )


def test_cutile_projection_preserves_adversarial_topology_boundaries():
    if not _supports_cutile():
        pytest.skip("cuTile projection requires a GPU")
    inputs = list(_inputs(16, 2))
    means = inputs[0]
    means = means.at[0, 2].set(0.0)
    means = means.at[1, 2].set(jnp.float32(0.01))
    means = means.at[2, 2].set(jnp.nextafter(jnp.float32(0.01), jnp.float32(0.0)))
    means = means.at[3, 2].set(jnp.nextafter(jnp.float32(0.01), jnp.float32(1.0)))
    means = means.at[4].set(jnp.asarray([jnp.nan, 0.0, 1.0]))
    means = means.at[5].set(jnp.asarray([jnp.inf, 0.0, 1.0]))
    means = means.at[6].set(jnp.asarray([20.0, -20.0, 1.0]))
    inputs[0] = means
    inputs[1] = inputs[1].at[7].set(jnp.zeros(4, jnp.float32))
    inputs[1] = inputs[1].at[8].set(jnp.full(4, 1.0e-20, jnp.float32))
    inputs[3] = inputs[3].at[9].set(jnp.float32(1.0 / 255.0))
    inputs[3] = (
        inputs[3].at[10].set(jnp.nextafter(jnp.float32(1.0 / 255.0), jnp.float32(0.0)))
    )
    inputs[4] = inputs[4].at[11].set(False)
    expected = jax.jit(lambda *values: _project(fully_fused_projection, values))(
        *inputs
    )
    actual = jax.jit(lambda *values: _project(fully_fused_projection_cutile, values))(
        *inputs
    )
    _assert_projection_matches(actual, expected)


def test_cutile_projection_vjp_matches_authoritative_jax_recompute():
    if not _supports_cutile():
        pytest.skip("cuTile projection requires a GPU")
    inputs = _inputs(97, 2)
    means, quats, scales, opacities, active_mask, viewmats, intrinsics = inputs
    weights = _inputs(97, 2, seed=19)
    means_weight = jnp.broadcast_to(weights[0][None, :, :2], (2, 97, 2))
    depth_weight = jnp.broadcast_to(weights[0][None, :, 2], (2, 97))
    conic_weight = jnp.broadcast_to(weights[2][None, :, :], (2, 97, 3))
    compensation_weight = jnp.broadcast_to(weights[3][None, :], (2, 97))

    def loss(function, means_, quats_, scales_, opacities_, viewmats_, intrinsics_):
        _, means2d, depths, conics, compensations, _ = function(
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
            calc_compensations=True,
        )
        assert compensations is not None
        return (
            jnp.vdot(means2d, means_weight)
            + jnp.vdot(depths, depth_weight)
            + jnp.vdot(conics, conic_weight)
            + jnp.vdot(compensations, compensation_weight)
        )

    differentiate = lambda function: jax.jit(
        jax.grad(
            lambda *values: loss(function, *values),
            argnums=(0, 1, 2, 3, 4, 5),
        )
    )(means, quats, scales, opacities, viewmats, intrinsics)
    expected = differentiate(fully_fused_projection)
    actual = differentiate(fully_fused_projection_cutile)
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_gradient, expected_gradient)


def test_cutile_backends_route_through_high_level_rasterizer():
    if not _supports_cutile():
        pytest.skip("cuTile rasterization requires a GPU")
    inputs = list(_inputs(64, 1))
    inputs[-1] = jnp.asarray(
        [[[50.0, 0.0, 32.0], [0.0, 51.0, 24.0], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )
    colors = jax.random.uniform(jax.random.key(31), (64, 3), dtype=jnp.float32)

    def render(
        projection_backend,
        intersection_backend,
        compositor_backend,
        values,
    ):
        (
            means,
            quats,
            scales,
            opacities,
            active_mask,
            viewmats,
            intrinsics,
            colors_,
        ) = values
        return rasterization(
            means,
            quats,
            scales,
            opacities,
            colors_,
            viewmats,
            intrinsics,
            64,
            48,
            active_mask=active_mask,
            config=RasterizationConfig(
                backend="intersections",
                projection_backend=projection_backend,
                intersection_backend=intersection_backend,
                compositor_backend=compositor_backend,
                intersection_mode="accutile",
                max_intersections=4096,
                max_gaussians_per_tile=64,
                max_candidates_per_tile=64,
            ),
        )

    values = (*inputs, colors)
    expected = jax.jit(lambda *arrays: render("jax", "jax", "jax", arrays))(*values)
    actual = jax.jit(
        lambda *arrays: render("cuda_tile", "cuda_tile", "cuda_tile", arrays)
    )(*values)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-4, atol=3e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-4, atol=3e-6)
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
