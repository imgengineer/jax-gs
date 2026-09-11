from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.resource_heavy

import jax_gs.two_dgs as two_dgs_module
from jax_gs.config import RasterizationConfig
from jax_gs.two_dgs import (
    fully_fused_projection_2dgs,
    rasterization_2dgs,
    rasterization_2dgs_inria_wrapper,
)


def _camera(width: int = 5, height: int = 5, focal: float = 10.0):
    viewmat = jnp.eye(4, dtype=jnp.float32)[None, ...]
    K = jnp.asarray(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )[None, ...]
    return viewmat, K


def _single_surfel():
    means = jnp.asarray([[0.0, 0.0, 2.0]], dtype=jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    scales = jnp.asarray([[0.2, 0.2, 0.01]], dtype=jnp.float32)
    opacities = jnp.asarray([0.8], dtype=jnp.float32)
    colors = jnp.asarray([[1.0, 0.25, 0.5]], dtype=jnp.float32)
    return means, quats, scales, opacities, colors


def _config(max_gaussians_per_tile: int = 4):
    return RasterizationConfig(
        tile_size=4,
        max_gaussians_per_tile=max_gaussians_per_tile,
        tile_batch_size=1,
    )


def test_2dgs_rejects_cutile_compositor():
    means, quats, scales, opacities, colors = _single_surfel()
    viewmats, Ks = _camera()
    with pytest.raises(NotImplementedError, match="cuTile.*3DGS"):
        rasterization_2dgs(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            5,
            5,
            config=RasterizationConfig(
                compositor_backend="cuda_tile"
            ),
        )


def test_fully_fused_projection_2dgs_matches_face_on_geometry():
    means, quats, scales, _, _ = _single_surfel()
    viewmats, Ks = _camera()

    radii, means2d, depths, transforms, normals = fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, 5, 5
    )

    np.testing.assert_array_equal(np.asarray(radii), [[[4, 4]]])
    np.testing.assert_allclose(np.asarray(means2d), [[[2.5, 2.5]]], atol=1e-6)
    np.testing.assert_allclose(np.asarray(depths), [[2.0]], atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(transforms),
        [[[[2.0, 0.0, 5.0], [0.0, 2.0, 5.0], [0.0, 0.0, 2.0]]]],
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(normals), [[[0.0, 0.0, -1.0]]])

    inactive_radii, *_ = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        5,
        5,
        active_mask=jnp.asarray([False]),
    )
    np.testing.assert_array_equal(np.asarray(inactive_radii), 0)


def test_leading_batch_dims_match_individual_2dgs_calls_and_are_jittable():
    means, quats, scales, opacities, colors = _single_surfel()
    viewmats, Ks = _camera(width=8, height=8)
    batched_means = jnp.stack((means, means.at[0, 0].set(0.1))).reshape(
        1, 2, 1, 3
    )
    batched_quats = jnp.broadcast_to(quats, (1, 2) + quats.shape)
    batched_scales = jnp.broadcast_to(scales, (1, 2) + scales.shape)
    batched_opacities = jnp.broadcast_to(opacities, (1, 2) + opacities.shape)
    batched_colors = jnp.broadcast_to(colors, (1, 2) + colors.shape)
    batched_viewmats = jnp.broadcast_to(viewmats, (1, 2) + viewmats.shape)
    batched_Ks = jnp.broadcast_to(Ks, (1, 2) + Ks.shape)
    config = RasterizationConfig(
        backend="intersections",
        tile_size=4,
        max_gaussians_per_tile=2,
        max_intersections=16,
        tile_batch_size=1,
    )

    @jax.jit
    def render(current_means):
        return rasterization_2dgs(
            current_means,
            batched_quats,
            batched_scales,
            batched_opacities,
            batched_colors,
            batched_viewmats,
            batched_Ks,
            8,
            8,
            backgrounds=jnp.asarray([[0.1, 0.2, 0.3]], jnp.float32),
            config=config,
        )

    outputs = render(batched_means)
    expected = [
        rasterization_2dgs(
            batched_means[0, index],
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            8,
            8,
            backgrounds=jnp.asarray([[0.1, 0.2, 0.3]], jnp.float32),
            config=config,
        )
        for index in range(2)
    ]

    assert outputs[0].shape == (1, 2, 1, 8, 8, 3)
    assert outputs[-1]["valid"].shape == (1, 2, 1, 1)
    assert outputs[-1]["active_count"].shape == (1, 2)
    for output_index in range(6):
        expected_value = jnp.stack(
            [value[output_index] for value in expected]
        )
        assert jnp.allclose(outputs[output_index][0], expected_value)

    gradient = jax.jit(jax.grad(lambda value: render(value)[0].sum()))(
        batched_means
    )
    assert gradient.shape == batched_means.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_screen_gradient_probe_zero_offset_is_identity_and_has_gradient():
    means, quats, scales, opacities, colors = _single_surfel()
    scales = scales.at[0, :2].set(0.5)
    viewmats, Ks = _camera(width=8, height=8)
    config = RasterizationConfig(
        backend="intersections",
        tile_size=4,
        max_gaussians_per_tile=1,
        max_intersections=8,
        tile_batch_size=1,
    )
    zero_offset = jnp.zeros((1, 1, 2), dtype=jnp.float32)

    def render(current_means, offset=None):
        return rasterization_2dgs(
            current_means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            8,
            8,
            render_mode="RGB+ED",
            distloss=True,
            config=config,
            _gradient_2dgs_offset=offset,
        )

    base_outputs = render(means)
    zero_outputs = render(means, zero_offset)
    for zero_image, base_image in zip(
        zero_outputs[:6], base_outputs[:6], strict=True
    ):
        np.testing.assert_array_equal(zero_image, base_image)

    base_info = base_outputs[-1]
    zero_info = zero_outputs[-1]
    assert set(zero_info) == set(base_info)
    for key, base_value in base_info.items():
        zero_value = zero_info[key]
        if base_value is None:
            assert zero_value is None
        else:
            np.testing.assert_array_equal(zero_value, base_value)

    def pixel_loss(
        current_means,
        current_quats,
        current_scales,
        current_opacities,
        current_colors,
        offset,
    ):
        outputs = rasterization_2dgs(
            current_means,
            current_quats,
            current_scales,
            current_opacities,
            current_colors,
            viewmats,
            Ks,
            8,
            8,
            render_mode="RGB+ED",
            distloss=True,
            config=config,
            _gradient_2dgs_offset=offset,
        )
        return outputs[0][0, 4, 4, 0]

    model_gradient = jax.grad(pixel_loss, argnums=(0, 1, 2, 3, 4))
    base_gradients = model_gradient(
        means, quats, scales, opacities, colors, None
    )
    zero_gradients = model_gradient(
        means, quats, scales, opacities, colors, zero_offset
    )
    for zero_gradient, base_gradient in zip(
        zero_gradients, base_gradients, strict=True
    ):
        np.testing.assert_array_equal(zero_gradient, base_gradient)

    offset_gradient = jax.grad(
        lambda offset: pixel_loss(
            means, quats, scales, opacities, colors, offset
        )
    )(zero_offset)
    assert bool(jnp.all(jnp.isfinite(offset_gradient)))
    assert bool(jnp.any(offset_gradient != 0.0))

    probe_offset = jnp.asarray([[[0.1, -0.05]]], dtype=jnp.float32)
    probe_outputs = render(means, probe_offset)
    for probe_image, base_image in zip(
        probe_outputs[:6], base_outputs[:6], strict=True
    ):
        np.testing.assert_array_equal(probe_image, base_image)
    np.testing.assert_array_equal(
        probe_outputs[-1]["gradient_2dgs"],
        base_info["gradient_2dgs"],
    )
    np.testing.assert_array_equal(
        base_info["gradient_2dgs"], jnp.zeros_like(base_info["means2d"])
    )

    with pytest.raises(ValueError, match="_gradient_2dgs_offset"):
        render(means, jnp.zeros((1, 2), dtype=jnp.float32))


def test_screen_gradient_probe_leading_batch_slices_offsets():
    means, quats, scales, opacities, colors = _single_surfel()
    viewmats, Ks = _camera(width=8, height=8)
    batch_size = 2
    batched_means = jnp.broadcast_to(means, (batch_size,) + means.shape)
    batched_quats = jnp.broadcast_to(quats, (batch_size,) + quats.shape)
    batched_scales = jnp.broadcast_to(scales, (batch_size,) + scales.shape)
    batched_opacities = jnp.broadcast_to(
        opacities, (batch_size,) + opacities.shape
    )
    batched_colors = jnp.broadcast_to(colors, (batch_size,) + colors.shape)
    batched_viewmats = jnp.broadcast_to(
        viewmats, (batch_size,) + viewmats.shape
    )
    batched_Ks = jnp.broadcast_to(Ks, (batch_size,) + Ks.shape)
    offsets = jnp.asarray(
        [[[[0.1, 0.0]]], [[[-0.05, 0.025]]]], dtype=jnp.float32
    )
    config = RasterizationConfig(
        backend="intersections",
        tile_size=4,
        max_gaussians_per_tile=1,
        max_intersections=8,
        tile_batch_size=1,
    )

    batched_outputs = rasterization_2dgs(
        batched_means,
        batched_quats,
        batched_scales,
        batched_opacities,
        batched_colors,
        batched_viewmats,
        batched_Ks,
        8,
        8,
        config=config,
        _gradient_2dgs_offset=offsets,
    )
    separate_outputs = [
        rasterization_2dgs(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            8,
            8,
            config=config,
            _gradient_2dgs_offset=offsets[index],
        )
        for index in range(batch_size)
    ]

    for output_index in range(6):
        expected = jnp.stack(
            [output[output_index] for output in separate_outputs]
        )
        np.testing.assert_allclose(
            batched_outputs[output_index], expected, rtol=1.0e-6, atol=1.0e-7
        )
    np.testing.assert_allclose(
        batched_outputs[-1]["gradient_2dgs"],
        jnp.stack([output[-1]["gradient_2dgs"] for output in separate_outputs]),
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="_gradient_2dgs_offset"):
        rasterization_2dgs(
            batched_means,
            batched_quats,
            batched_scales,
            batched_opacities,
            batched_colors,
            batched_viewmats,
            batched_Ks,
            8,
            8,
            config=config,
            _gradient_2dgs_offset=jnp.zeros((2, 1, 2), dtype=jnp.float32),
        )


@pytest.mark.parametrize("backend", ["reference", "intersections"])
def test_2dgs_absgrad_probe_sums_before_symmetric_pixel_cancellation(backend):
    means, quats, scales, opacities, colors = _single_surfel()
    scales = scales.at[0, :2].set(0.5)
    viewmats, Ks = _camera(width=2, height=1, focal=4.0)
    zero_offset = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    zero_probe = jnp.zeros_like(zero_offset)
    config = RasterizationConfig(
        backend=backend,
        intersection_backend="jax",
        sort_backend="jax",
        tile_size=2,
        max_gaussians_per_tile=1,
        max_intersections=1,
        tile_batch_size=1,
    )
    packed_results = []

    for packed in (False, True):
        def render(current_means, offset, probe):
            return rasterization_2dgs(
                current_means,
                quats,
                scales,
                opacities,
                colors,
                viewmats,
                Ks,
                2,
                1,
                packed=packed,
                absgrad=True,
                config=config,
                _gradient_2dgs_offset=offset,
                **(
                    {}
                    if probe is None
                    else {"_gradient_2dgs_absgrad_probe": probe}
                ),
            )

        def objective(current_means, offset, probe):
            outputs = render(current_means, offset, probe)
            return jnp.sum(outputs[0]) + 0.25 * jnp.sum(outputs[1]), outputs[:2]

        baseline = jax.jit(
            jax.value_and_grad(
                lambda current_means, offset: objective(
                    current_means, offset, None
                ),
                argnums=(0, 1),
                has_aux=True,
            )
        )
        probed = jax.jit(
            jax.value_and_grad(objective, argnums=(0, 1, 2), has_aux=True)
        )
        (baseline_value, baseline_images), baseline_gradients = baseline(
            means, zero_offset
        )
        (probed_value, probed_images), probed_gradients = probed(
            means, zero_offset, zero_probe
        )

        np.testing.assert_array_equal(probed_value, baseline_value)
        for probed_image, baseline_image in zip(
            probed_images, baseline_images, strict=True
        ):
            np.testing.assert_array_equal(probed_image, baseline_image)
        np.testing.assert_allclose(
            probed_gradients[0], baseline_gradients[0], rtol=1.0e-6, atol=1.0e-7
        )
        np.testing.assert_allclose(
            probed_gradients[1], baseline_gradients[1], rtol=1.0e-6, atol=1.0e-7
        )
        assert abs(float(probed_gradients[1][0, 0, 0])) < 1.0e-6
        assert float(probed_gradients[2][0, 0, 0]) > 1.0e-4

        info = render(means, zero_offset, zero_probe)[-1]
        assert bool(info["absgrad_requested"])
        assert bool(info["absgrad_available"])
        assert bool(info["absgrad_probe_enabled"])
        packed_results.append(
            (*probed_images, probed_gradients[0], probed_gradients[1:])
        )

    for packed_value, dense_value in zip(
        packed_results[1], packed_results[0], strict=True
    ):
        if isinstance(packed_value, tuple):
            for packed_gradient, dense_gradient in zip(
                packed_value, dense_value, strict=True
            ):
                np.testing.assert_allclose(
                    packed_gradient, dense_gradient, rtol=1.0e-6, atol=1.0e-7
                )
        else:
            np.testing.assert_allclose(
                packed_value, dense_value, rtol=1.0e-6, atol=1.0e-7
            )


def test_2dgs_absgrad_probe_leading_batch_slices_and_jits():
    means, quats, scales, opacities, colors = _single_surfel()
    scales = scales.at[0, :2].set(0.5)
    viewmats, Ks = _camera(width=2, height=1, focal=4.0)
    batch_size = 2
    batched_means = jnp.broadcast_to(means, (batch_size,) + means.shape)
    batched_quats = jnp.broadcast_to(quats, (batch_size,) + quats.shape)
    batched_scales = jnp.broadcast_to(scales, (batch_size,) + scales.shape)
    batched_opacities = jnp.broadcast_to(
        opacities, (batch_size,) + opacities.shape
    )
    batched_colors = jnp.broadcast_to(colors, (batch_size,) + colors.shape)
    batched_viewmats = jnp.broadcast_to(
        viewmats, (batch_size,) + viewmats.shape
    )
    batched_Ks = jnp.broadcast_to(Ks, (batch_size,) + Ks.shape)
    zero_probes = jnp.zeros((batch_size, 1, 1, 2), dtype=jnp.float32)
    batch_weights = jnp.asarray([1.0, 2.0], dtype=jnp.float32).reshape(
        2, 1, 1, 1, 1
    )
    config = RasterizationConfig(
        backend="intersections",
        intersection_backend="jax",
        sort_backend="jax",
        tile_size=2,
        max_gaussians_per_tile=1,
        max_intersections=1,
        tile_batch_size=1,
    )

    def batched_loss(current_means, probes):
        rendered = rasterization_2dgs(
            current_means,
            batched_quats,
            batched_scales,
            batched_opacities,
            batched_colors,
            batched_viewmats,
            batched_Ks,
            2,
            1,
            packed=False,
            absgrad=True,
            config=config,
            _gradient_2dgs_absgrad_probe=probes,
        )[0]
        return jnp.sum(rendered * batch_weights)

    batched_gradients = jax.jit(
        jax.grad(batched_loss, argnums=(0, 1))
    )(batched_means, zero_probes)

    def single_loss(current_means, probe, weight):
        rendered = rasterization_2dgs(
            current_means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            2,
            1,
            packed=False,
            absgrad=True,
            config=config,
            _gradient_2dgs_absgrad_probe=probe,
        )[0]
        return weight * jnp.sum(rendered)

    single_gradients = jax.jit(
        jax.grad(single_loss, argnums=(0, 1))
    )
    expected = [
        single_gradients(means, zero_probes[index], index + 1.0)
        for index in range(batch_size)
    ]
    for gradient_index in range(2):
        np.testing.assert_allclose(
            batched_gradients[gradient_index],
            jnp.stack([value[gradient_index] for value in expected]),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    assert jnp.all(batched_gradients[1][:, 0, 0, 0] > 0.0)


def test_2dgs_absgrad_probe_requires_absgrad():
    means, quats, scales, opacities, colors = _single_surfel()
    viewmats, Ks = _camera(width=2, height=1, focal=4.0)
    with pytest.raises(ValueError, match="requires absgrad=True"):
        rasterization_2dgs(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            2,
            1,
            _gradient_2dgs_absgrad_probe=jnp.zeros(
                (1, 1, 2), dtype=jnp.float32
            ),
        )


def test_per_camera_sh_coefficients_match_separate_2dgs_calls():
    means, quats, scales, opacities, _ = _single_surfel()
    viewmats, Ks = _camera(width=8, height=8)
    viewmats = jnp.concatenate((viewmats, viewmats), axis=0)
    viewmats = viewmats.at[1, 0, 3].set(0.05)
    Ks = jnp.concatenate((Ks, Ks), axis=0)
    sh = jnp.asarray(
        [[[[0.1, 0.1, 0.1]]], [[[0.3, 0.2, 0.1]]]], jnp.float32
    )
    config = RasterizationConfig(
        backend="intersections",
        tile_size=4,
        max_gaussians_per_tile=2,
        max_intersections=16,
        tile_batch_size=1,
    )

    outputs = rasterization_2dgs(
        means,
        quats,
        scales,
        opacities,
        sh,
        viewmats,
        Ks,
        8,
        8,
        sh_degree=0,
        config=config,
    )
    separate = [
        rasterization_2dgs(
            means,
            quats,
            scales,
            opacities,
            sh[index],
            viewmats[index : index + 1],
            Ks[index : index + 1],
            8,
            8,
            sh_degree=0,
            config=config,
        )
        for index in range(2)
    ]

    for output_index in range(6):
        expected = jnp.concatenate(
            [value[output_index] for value in separate], axis=0
        )
        assert jnp.allclose(outputs[output_index], expected)


@pytest.mark.parametrize(
    ("render_mode", "channels", "expected_depth"),
    (("RGB", 3, None), ("RGB+D", 4, 1.6), ("RGB+ED", 4, 2.0)),
)
def test_rasterization_2dgs_render_modes_and_auxiliary_maps(
    render_mode: str, channels: int, expected_depth: float | None
):
    means, quats, scales, opacities, colors = _single_surfel()
    viewmats, Ks = _camera()

    renders, alphas, normals, surface_normals, distort, median, info = (
        rasterization_2dgs(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            5,
            5,
            render_mode=render_mode,
            distloss=True,
            config=_config(),
        )
    )

    assert renders.shape == (1, 5, 5, channels)
    assert alphas.shape == (1, 5, 5, 1)
    assert normals.shape == surface_normals.shape == (1, 5, 5, 3)
    assert distort.shape == median.shape == (1, 5, 5, 1)
    np.testing.assert_allclose(np.asarray(alphas[0, 2, 2, 0]), 0.8, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(normals[0, 2, 2]), [0.0, 0.0, -0.8], atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(surface_normals[0, 2, 2]), [0.0, 0.0, -1.0], atol=1e-5
    )
    np.testing.assert_allclose(np.asarray(distort), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(median[0, 2, 2, 0]), 2.0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(info["render_expected_depth"][0, 2, 2, 0]), 2.0, atol=1e-6
    )
    if expected_depth is not None:
        np.testing.assert_allclose(
            np.asarray(renders[0, 2, 2, -1]), expected_depth, atol=1e-6
        )


def test_tile_chunks_composite_all_candidates_without_overflow():
    means = jnp.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=jnp.float32)
    scales = jnp.asarray([[0.25, 0.25, 0.01]] * 2, dtype=jnp.float32)
    opacities = jnp.asarray([0.6, 0.6], dtype=jnp.float32)
    colors = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32
    )
    viewmats, Ks = _camera(width=8, height=8)
    Ks = Ks.at[:, 0, 2].set(4.5).at[:, 1, 2].set(4.5)

    def render(current_colors, candidates_per_chunk):
        return rasterization_2dgs(
            means,
            quats,
            scales,
            opacities,
            current_colors,
            viewmats,
            Ks,
            8,
            8,
            render_mode="RGB+ED",
            distloss=True,
            config=RasterizationConfig(
                backend="intersections",
                intersection_backend="jax",
                sort_backend="jax",
                tile_size=8,
                max_gaussians_per_tile=candidates_per_chunk,
                max_intersections=4,
                tile_batch_size=1,
            ),
        )

    chunked = render(colors, 1)
    unchunked = render(colors, 2)
    for actual, expected in zip(chunked[:6], unchunked[:6], strict=True):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-6
        )
    np.testing.assert_array_equal(
        np.asarray(chunked[-1]["candidate_counts"]), [[[2]]]
    )
    assert not bool(jnp.any(chunked[-1]["tile_overflow"]))
    assert bool(jnp.all(chunked[-1]["candidate_limit_exceeded"]))

    chunked_gradient = jax.grad(lambda value: render(value, 1)[0].sum())(colors)
    unchunked_gradient = jax.grad(lambda value: render(value, 2)[0].sum())(colors)
    np.testing.assert_allclose(
        np.asarray(chunked_gradient),
        np.asarray(unchunked_gradient),
        rtol=2e-4,
        atol=2e-5,
    )
    assert bool(jnp.any(chunked_gradient[1] != 0.0))


def test_active_mask_is_jittable_and_candidate_limit_is_diagnostic():
    means = jnp.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=jnp.float32)
    scales = jnp.asarray([[0.25, 0.25, 0.01]] * 2, dtype=jnp.float32)
    opacities = jnp.asarray([0.6, 0.6], dtype=jnp.float32)
    colors = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    viewmats, Ks = _camera(width=8, height=8)
    Ks = Ks.at[:, 0, 2].set(4.5).at[:, 1, 2].set(4.5)
    overflow_config = RasterizationConfig(
        tile_size=8, max_gaussians_per_tile=1, tile_batch_size=1
    )

    def tile_stats(mask):
        *_, info = rasterization_2dgs(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            8,
            8,
            active_mask=mask,
            config=overflow_config,
        )
        return (
            info["candidate_counts"],
            info["tile_overflow"],
            info["candidate_limit_exceeded"],
        )

    jitted_stats = jax.jit(tile_stats)
    counts, overflow, limit_exceeded = jitted_stats(jnp.asarray([True, True]))
    np.testing.assert_array_equal(np.asarray(counts), [[[2]]])
    np.testing.assert_array_equal(np.asarray(overflow), [[[False]]])
    np.testing.assert_array_equal(np.asarray(limit_exceeded), [[[True]]])

    counts, overflow, limit_exceeded = jitted_stats(jnp.asarray([True, False]))
    np.testing.assert_array_equal(np.asarray(counts), [[[1]]])
    np.testing.assert_array_equal(np.asarray(overflow), [[[False]]])
    np.testing.assert_array_equal(np.asarray(limit_exceeded), [[[False]]])

    full_config = RasterizationConfig(
        tile_size=8, max_gaussians_per_tile=2, tile_batch_size=1
    )
    outputs = rasterization_2dgs(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        8,
        8,
        render_mode="RGB+ED",
        distloss=True,
        config=full_config,
    )
    expected_depth = outputs[-1]["render_expected_depth"][0, 4, 4, 0]
    np.testing.assert_allclose(np.asarray(expected_depth), 16.0 / 7.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(outputs[5][0, 4, 4, 0]), 2.0, atol=1e-6)
    assert float(outputs[4][0, 4, 4, 0]) > 0.0

    def render_sum(means_value):
        return rasterization_2dgs(
            means_value,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            8,
            8,
            config=full_config,
        )[0].sum()

    _, directional_derivative = jax.jvp(
        render_sum, (means,), (jnp.ones_like(means),)
    )
    assert bool(jnp.isfinite(directional_derivative))


def test_intersection_backend_matches_reference_2dgs():
    means = jnp.asarray([[0.0, 0.0, 2.0], [0.1, -0.05, 3.0]], dtype=jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=jnp.float32)
    scales = jnp.asarray([[0.2, 0.15, 0.01], [0.18, 0.22, 0.01]], dtype=jnp.float32)
    opacities = jnp.asarray([0.6, 0.45], dtype=jnp.float32)
    colors = jnp.asarray([[1.0, 0.2, 0.0], [0.0, 0.3, 1.0]], dtype=jnp.float32)
    viewmats, Ks = _camera(width=8, height=8)

    def render(backend):
        return rasterization_2dgs(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            8,
            8,
            backgrounds=jnp.asarray([[0.1, 0.2, 0.3]], dtype=jnp.float32),
            render_mode="RGB+ED",
            distloss=True,
            config=RasterizationConfig(
                backend=backend,
                tile_size=4,
                max_gaussians_per_tile=2,
                max_intersections=32,
                tile_batch_size=2,
            ),
        )

    fast = render("intersections")
    reference = render("reference")
    for actual, expected in zip(fast[:6], reference[:6]):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-6
        )
    assert not bool(fast[-1]["intersection_overflow"][0])


def test_dense_metadata_reuses_rendered_2dgs_tile_intersections(monkeypatch):
    means = jnp.asarray(
        [[-0.1, 0.0, 2.0], [0.15, 0.05, 3.0]], dtype=jnp.float32
    )
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=jnp.float32)
    scales = jnp.asarray([[0.2, 0.2, 0.01]] * 2, dtype=jnp.float32)
    opacities = jnp.asarray([0.7, 0.5], dtype=jnp.float32)
    colors = jnp.asarray(
        [[1.0, 0.2, 0.0], [0.0, 0.3, 1.0]], dtype=jnp.float32
    )
    viewmats, Ks = _camera(width=8, height=8)
    viewmats = jnp.concatenate(
        (viewmats, viewmats.at[0, 0, 3].set(0.1)), axis=0
    )
    Ks = jnp.broadcast_to(Ks, (2, 3, 3))
    config = RasterizationConfig(
        backend="intersections",
        intersection_backend="jax",
        sort_backend="jax",
        tile_size=4,
        max_gaussians_per_tile=2,
        max_intersections=32,
        tile_batch_size=1,
    )

    original_intersect_tiles = two_dgs_module.intersect_tiles
    intersection_builds = 0

    def count_intersection_builds(*args, **kwargs):
        nonlocal intersection_builds
        intersection_builds += 1
        if intersection_builds > 1:
            raise AssertionError("dense metadata must not rebuild intersections")
        return original_intersect_tiles(*args, **kwargs)

    monkeypatch.setattr(
        two_dgs_module, "intersect_tiles", count_intersection_builds
    )
    outputs = rasterization_2dgs(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        8,
        8,
        packed=False,
        distloss=True,
        config=config,
    )
    renders, alphas, normals, surface_normals, distort, median, info = outputs

    assert intersection_builds == 1
    assert renders.shape == (2, 8, 8, 3)
    assert alphas.shape == (2, 8, 8, 1)
    assert normals.shape == surface_normals.shape == (2, 8, 8, 3)
    assert distort.shape == median.shape == (2, 8, 8, 1)
    assert info["camera_ids"] is None
    assert info["gaussian_ids"] is None

    tile_count = 4
    tile_bits = max(1, (tile_count - 1).bit_length())
    expected_offsets = []
    expected_flatten_ids = []
    expected_isect_ids = []
    expected_tiles_per_gauss = np.zeros((2, 2), dtype=np.int32)
    count_base = 0
    candidate_counts = np.asarray(info["candidate_counts"]).reshape(2, tile_count)
    candidate_ids = np.asarray(info["candidate_ids"]).reshape(
        2, tile_count, 2
    )
    candidate_valid = np.asarray(info["candidate_valid"]).reshape(
        2, tile_count, 2
    )
    projected_depths = np.asarray(info["depths"], dtype=np.float32)

    for camera_id in range(2):
        local_counts = candidate_counts[camera_id]
        local_offsets = np.concatenate(
            (np.asarray([0], np.int32), np.cumsum(local_counts[:-1], dtype=np.int32))
        )
        expected_offsets.append(local_offsets.reshape(2, 2) + count_base)
        for tile_id in range(tile_count):
            valid_ids = candidate_ids[camera_id, tile_id][
                candidate_valid[camera_id, tile_id]
            ]
            assert valid_ids.shape[0] == local_counts[tile_id]
            for gaussian_id in valid_ids:
                expected_flatten_ids.append(camera_id * 2 + gaussian_id)
                high_word = (camera_id << tile_bits) | tile_id
                depth_word = projected_depths[camera_id, gaussian_id].view(np.int32)
                expected_isect_ids.append((high_word, depth_word))
                expected_tiles_per_gauss[camera_id, gaussian_id] += 1
        count_base += int(local_counts.sum())

    valid_count = int(info["isect_valid_count"])
    assert valid_count == count_base == int(jnp.sum(info["intersection_count"]))
    assert info["isect_ids"].shape == (64, 2)
    assert info["flatten_ids"].shape == (64,)
    assert info["isect_offsets"].shape == (2, 2, 2)
    np.testing.assert_array_equal(
        np.asarray(info["isect_ids"][:valid_count]),
        np.asarray(expected_isect_ids, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(info["flatten_ids"][:valid_count]),
        np.asarray(expected_flatten_ids, dtype=np.int32),
    )
    np.testing.assert_array_equal(np.asarray(info["isect_ids"])[valid_count:], -1)
    np.testing.assert_array_equal(
        np.asarray(info["flatten_ids"])[valid_count:], -1
    )
    np.testing.assert_array_equal(
        np.asarray(info["isect_offsets"]), np.asarray(expected_offsets)
    )
    np.testing.assert_array_equal(
        np.asarray(info["tiles_per_gauss"]), expected_tiles_per_gauss
    )


def test_packed_2dgs_metadata_is_a_stable_prefix_and_preserves_rendering():
    means = jnp.asarray(
        [
            [-0.1, 0.0, 2.0],
            [0.15, 0.05, 3.0],
            [0.0, 0.0, 2.5],
        ],
        dtype=jnp.float32,
    )
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=jnp.float32)
    scales = jnp.asarray([[0.2, 0.2, 0.01]] * 3, dtype=jnp.float32)
    opacities = jnp.asarray([0.7, 0.5, 0.9], dtype=jnp.float32)
    colors = jnp.asarray(
        [[1.0, 0.2, 0.0], [0.0, 0.3, 1.0], [0.5, 0.5, 0.5]],
        dtype=jnp.float32,
    )
    active_mask = jnp.asarray([True, True, False])
    viewmats, Ks = _camera(width=8, height=8)
    viewmats = jnp.concatenate(
        (viewmats, viewmats.at[0, 0, 3].set(0.1)), axis=0
    )
    Ks = jnp.broadcast_to(Ks, (2, 3, 3))
    config = RasterizationConfig(
        backend="intersections",
        intersection_backend="jax",
        sort_backend="jax",
        tile_size=4,
        max_gaussians_per_tile=3,
        max_intersections=32,
        tile_batch_size=1,
    )

    def render(current_means, *, packed):
        return rasterization_2dgs(
            current_means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            8,
            8,
            packed=packed,
            active_mask=active_mask,
            render_mode="RGB+ED",
            distloss=True,
            config=config,
        )

    dense_outputs = render(means, packed=False)
    packed_outputs = render(means, packed=True)
    dense_info = dense_outputs[-1]
    packed_info = packed_outputs[-1]

    for packed_image, dense_image in zip(
        packed_outputs[:6], dense_outputs[:6], strict=True
    ):
        np.testing.assert_array_equal(packed_image, dense_image)

    camera_count = viewmats.shape[0]
    gaussian_count = means.shape[0]
    capacity = camera_count * gaussian_count
    dense_valid = np.asarray(dense_info["valid"]).reshape(-1)
    selected = np.flatnonzero(dense_valid)
    valid_count = selected.shape[0]

    assert int(packed_info["projection_valid_count"]) == valid_count
    assert int(packed_info["projection_capacity"]) == capacity
    assert bool(packed_info["packed_metadata_available"])
    np.testing.assert_array_equal(
        np.asarray(packed_info["batch_ids"][:valid_count]), 0
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["camera_ids"][:valid_count]),
        selected // gaussian_count,
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["gaussian_ids"][:valid_count]),
        selected % gaussian_count,
    )
    for key in ("batch_ids", "camera_ids", "gaussian_ids"):
        assert packed_info[key].shape == (capacity,)
        assert packed_info[key].dtype == jnp.int32
        np.testing.assert_array_equal(
            np.asarray(packed_info[key][valid_count:]), -1
        )

    projection_fields = {
        "radii": (capacity, 2),
        "means2d": (capacity, 2),
        "depths": (capacity,),
        "ray_transforms": (capacity, 3, 3),
        "opacities": (capacity,),
        "normals": (capacity, 3),
        "gradient_2dgs": (capacity, 2),
        "tiles_per_gauss": (capacity,),
    }
    for key, shape in projection_fields.items():
        assert packed_info[key].shape == shape
        dense_values = np.asarray(dense_info[key]).reshape(
            (capacity,) + shape[1:]
        )
        np.testing.assert_array_equal(
            np.asarray(packed_info[key][:valid_count]), dense_values[selected]
        )
        np.testing.assert_array_equal(
            np.asarray(packed_info[key][valid_count:]), 0
        )

    intersection_count = int(dense_info["isect_valid_count"])
    dense_to_packed = np.full((capacity,), -1, dtype=np.int32)
    dense_to_packed[selected] = np.arange(valid_count, dtype=np.int32)
    expected_flatten_ids = dense_to_packed[
        np.asarray(dense_info["flatten_ids"][:intersection_count])
    ]
    assert int(packed_info["isect_valid_count"]) == intersection_count
    np.testing.assert_array_equal(
        np.asarray(packed_info["flatten_ids"][:intersection_count]),
        expected_flatten_ids,
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["flatten_ids"])[intersection_count:], -1
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["isect_ids"]),
        np.asarray(dense_info["isect_ids"]),
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["isect_offsets"]),
        np.asarray(dense_info["isect_offsets"]),
    )

    def image_loss(current_means, *, packed):
        images = render(current_means, packed=packed)[:6]
        return sum(jnp.sum(image) for image in images)

    dense_gradient = jax.grad(image_loss)(means, packed=False)
    packed_gradient = jax.grad(image_loss)(means, packed=True)
    np.testing.assert_allclose(
        packed_gradient, dense_gradient, rtol=1.0e-6, atol=1.0e-6
    )


def test_packed_2dgs_leading_batch_is_one_global_stable_prefix():
    batch_count, camera_count, gaussian_count = 2, 2, 3
    means = jnp.asarray(
        [
            [[-0.1, 0.0, 2.0], [0.15, 0.05, 3.0], [0.0, 0.0, 2.5]],
            [[0.05, 0.0, 2.2], [-0.15, 0.05, 2.8], [0.2, -0.1, 3.2]],
        ],
        dtype=jnp.float32,
    )
    quats = jnp.zeros(
        (batch_count, gaussian_count, 4), dtype=jnp.float32
    ).at[..., 0].set(1.0)
    scales = jnp.broadcast_to(
        jnp.asarray([0.2, 0.2, 0.01], dtype=jnp.float32),
        (batch_count, gaussian_count, 3),
    )
    opacities = jnp.asarray(
        [[0.7, 0.5, 0.9], [0.6, 0.8, 0.55]], dtype=jnp.float32
    )
    colors = jnp.asarray(
        [
            [[1.0, 0.2, 0.0], [0.0, 0.3, 1.0], [0.5, 0.5, 0.5]],
            [[0.2, 0.9, 0.1], [0.8, 0.1, 0.3], [0.1, 0.2, 1.0]],
        ],
        dtype=jnp.float32,
    )
    second_camera = jnp.eye(4, dtype=jnp.float32).at[0, 3].set(0.1)
    cameras = jnp.stack((jnp.eye(4, dtype=jnp.float32), second_camera))
    viewmats = jnp.broadcast_to(
        cameras, (batch_count, camera_count, 4, 4)
    )
    K = jnp.asarray(
        [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    Ks = jnp.broadcast_to(K, (batch_count, camera_count, 3, 3))
    active_mask = jnp.asarray(
        [[True, False, True], [False, True, True]], dtype=jnp.bool_
    )
    config = RasterizationConfig(
        backend="intersections",
        intersection_backend="jax",
        sort_backend="jax",
        tile_size=4,
        max_gaussians_per_tile=3,
        max_intersections=32,
        tile_batch_size=1,
    )

    def render(current_means, current_opacities, *, packed):
        return rasterization_2dgs(
            current_means,
            quats,
            scales,
            current_opacities,
            colors,
            viewmats,
            Ks,
            8,
            8,
            packed=packed,
            active_mask=active_mask,
            render_mode="RGB+ED",
            distloss=True,
            config=config,
        )

    dense_outputs = render(means, opacities, packed=False)
    packed_outputs = render(means, opacities, packed=True)
    for packed_image, dense_image in zip(
        packed_outputs[:6], dense_outputs[:6], strict=True
    ):
        np.testing.assert_array_equal(packed_image, dense_image)

    dense_info = dense_outputs[-1]
    packed_info = packed_outputs[-1]
    for key in (
        "render_normals_camera",
        "render_expected_depth",
        "render_median_depth",
        "render_distort",
    ):
        np.testing.assert_array_equal(packed_info[key], dense_info[key])

    capacity = batch_count * camera_count * gaussian_count
    dense_valid = np.asarray(dense_info["valid"]).reshape(
        batch_count, camera_count, gaussian_count
    )
    selected = np.flatnonzero(dense_valid.reshape(-1))
    valid_count = selected.size
    expected_batch_ids = selected // (camera_count * gaussian_count)
    within_batch = selected % (camera_count * gaussian_count)
    expected_camera_ids = within_batch // gaussian_count
    expected_gaussian_ids = within_batch % gaussian_count

    assert bool(packed_info["packed_requested"])
    assert bool(packed_info["packed_metadata_available"])
    assert int(packed_info["n_batches"]) == batch_count
    assert int(packed_info["n_cameras"]) == camera_count
    assert int(packed_info["projection_capacity"]) == capacity
    assert int(packed_info["projection_valid_count"]) == valid_count
    np.testing.assert_array_equal(
        np.asarray(packed_info["valid"]), np.arange(capacity) < valid_count
    )
    for key, expected in (
        ("batch_ids", expected_batch_ids),
        ("camera_ids", expected_camera_ids),
        ("gaussian_ids", expected_gaussian_ids),
    ):
        values = np.asarray(packed_info[key])
        assert values.shape == (capacity,)
        np.testing.assert_array_equal(values[:valid_count], expected)
        np.testing.assert_array_equal(values[valid_count:], -1)

    expected_indptr = np.concatenate(
        (
            np.zeros((1,), dtype=np.int32),
            np.cumsum(
                dense_valid.sum(axis=-1).reshape(-1), dtype=np.int32
            ),
        )
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["indptr"]), expected_indptr
    )

    projection_shapes = {
        "radii": (capacity, 2),
        "means2d": (capacity, 2),
        "depths": (capacity,),
        "ray_transforms": (capacity, 3, 3),
        "normals": (capacity, 3),
        "opacities": (capacity,),
        "gradient_2dgs": (capacity, 2),
        "tiles_per_gauss": (capacity,),
    }
    for key, shape in projection_shapes.items():
        packed_values = np.asarray(packed_info[key])
        dense_values = np.asarray(dense_info[key]).reshape(
            (capacity,) + shape[1:]
        )
        assert packed_values.shape == shape
        np.testing.assert_array_equal(
            packed_values[:valid_count], dense_values[selected]
        )
        np.testing.assert_array_equal(packed_values[valid_count:], 0)

    per_batch_isect_capacity = dense_info["flatten_ids"].shape[-1]
    global_isect_capacity = batch_count * per_batch_isect_capacity
    dense_isect_counts = np.asarray(
        dense_info["isect_valid_count"]
    ).reshape(batch_count)
    dense_flatten_ids = np.asarray(dense_info["flatten_ids"]).reshape(
        batch_count, per_batch_isect_capacity
    )
    dense_isect_ids = np.asarray(dense_info["isect_ids"]).reshape(
        batch_count, per_batch_isect_capacity, 2
    )
    dense_to_packed = np.full((capacity,), -1, dtype=np.int32)
    dense_to_packed[selected] = np.arange(valid_count, dtype=np.int32)
    tile_count = int(dense_info["tile_width"].reshape(-1)[0]) * int(
        dense_info["tile_height"].reshape(-1)[0]
    )
    tile_bits = max(1, (tile_count - 1).bit_length())
    tile_mask = np.uint32((1 << tile_bits) - 1)
    expected_flatten_ids = []
    expected_isect_ids = []
    for batch_id, count in enumerate(dense_isect_counts):
        local_flatten_ids = dense_flatten_ids[batch_id, :count]
        expected_flatten_ids.extend(
            dense_to_packed[
                batch_id * camera_count * gaussian_count + local_flatten_ids
            ]
        )
        local_words = dense_isect_ids[batch_id, :count]
        tile_ids = local_words[:, 0].view(np.uint32) & tile_mask
        local_camera_ids = local_flatten_ids // gaussian_count
        global_image_ids = batch_id * camera_count + local_camera_ids
        global_high_words = (
            global_image_ids.astype(np.uint32) << np.uint32(tile_bits)
        ) | tile_ids
        expected_isect_ids.extend(
            np.stack(
                (global_high_words.view(np.int32), local_words[:, 1]), axis=-1
            )
        )

    global_isect_count = len(expected_flatten_ids)
    assert packed_info["flatten_ids"].shape == (global_isect_capacity,)
    assert packed_info["isect_ids"].shape == (global_isect_capacity, 2)
    assert int(packed_info["isect_valid_count"]) == global_isect_count
    np.testing.assert_array_equal(
        np.asarray(packed_info["flatten_ids"][:global_isect_count]),
        expected_flatten_ids,
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["isect_ids"][:global_isect_count]),
        expected_isect_ids,
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["flatten_ids"])[global_isect_count:], -1
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["isect_ids"])[global_isect_count:], -1
    )

    dense_offsets = np.asarray(dense_info["isect_offsets"])
    isect_bases = np.concatenate(
        (
            np.zeros((1,), dtype=np.int32),
            np.cumsum(dense_isect_counts[:-1], dtype=np.int32),
        )
    )
    expected_offsets = dense_offsets + isect_bases.reshape(
        (batch_count,) + (1,) * (dense_offsets.ndim - 1)
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["isect_offsets"]), expected_offsets
    )

    def objective(current_means, current_opacities, *, packed):
        outputs = render(current_means, current_opacities, packed=packed)
        return jnp.sum(outputs[0]) + 0.1 * jnp.sum(outputs[1])

    dense_gradients = jax.grad(objective, argnums=(0, 1))(
        means, opacities, packed=False
    )
    packed_gradients = jax.grad(objective, argnums=(0, 1))(
        means, opacities, packed=True
    )
    for packed_gradient, dense_gradient in zip(
        packed_gradients, dense_gradients, strict=True
    ):
        np.testing.assert_allclose(
            packed_gradient, dense_gradient, rtol=1.0e-6, atol=1.0e-6
        )


def test_rasterization_2dgs_inria_wrapper_blends_depth_and_exposes_maps():
    means, quats, scales, opacities, colors = _single_surfel()
    viewmats, Ks = _camera()

    (renders, alphas), meta = rasterization_2dgs_inria_wrapper(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        5,
        5,
        depth_ratio=0.5,
        config=_config(),
    )

    assert renders.shape == (1, 5, 5, 4)
    assert alphas.shape == (1, 5, 5, 1)
    np.testing.assert_allclose(np.asarray(renders[0, 2, 2, -1]), 2.0, atol=1e-6)
    assert meta["normals_rend"].shape == (1, 5, 5, 3)
    assert meta["normals_surf"].shape == (1, 5, 5, 3)
    assert meta["render_distloss"].shape == (1, 5, 5, 1)
