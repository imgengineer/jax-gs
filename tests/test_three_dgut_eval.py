import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.config import RasterizationConfig
from jax_gs.rasterization import rasterization
from jax_gs.rendering import RendererConfig_ParallelBatch
from jax_gs.three_dgut import (
    rasterize_to_pixels_eval3d,
    rasterize_to_pixels_eval3d_extra,
)

pytestmark = pytest.mark.resource_heavy


def test_eval3d_compatibility_entry_is_finite():
    means = jnp.asarray([[0.0, 0.0, 2.0]], jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], jnp.float32)
    scales = jnp.asarray([[0.1, 0.1, 0.1]], jnp.float32)
    colors = jnp.asarray([[[1.0, 0.0, 0.0]]], jnp.float32)
    opacities = jnp.asarray([[0.8]], jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.asarray(
        [[[20.0, 0.0, 4.0], [0.0, 20.0, 4.0], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )
    render, alpha = rasterize_to_pixels_eval3d(
        means,
        quats,
        scales,
        colors,
        opacities,
        viewmats,
        Ks,
        8,
        8,
        4,
        jnp.zeros((1, 2, 2), jnp.int32),
        jnp.zeros((1,), jnp.int32),
        max_gaussians_per_tile=4,
    )
    assert render.shape == (1, 8, 8, 3)
    assert alpha.shape == (1, 8, 8, 1)
    assert jnp.all(jnp.isfinite(render))
    assert float(alpha.max()) > 0.0


def test_eval3d_matches_world_ray_distance_equation_and_gradients():
    means = jnp.asarray([[0.0, 0.0, 2.0]], jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], jnp.float32)
    scales = jnp.asarray([[0.1, 0.1, 0.1]], jnp.float32)
    colors = jnp.asarray([[[1.0, 0.0, 0.0]]], jnp.float32)
    opacities = jnp.asarray([[0.8]], jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.asarray(
        [[[20.0, 0.0, 4.5], [0.0, 20.0, 4.5], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )
    offsets = jnp.zeros((1, 2, 2), jnp.int32)
    flatten_ids = jnp.asarray([0], jnp.int32)

    @jax.jit
    def render(current_means):
        return rasterize_to_pixels_eval3d(
            current_means,
            quats,
            scales,
            colors,
            opacities,
            viewmats,
            Ks,
            8,
            8,
            4,
            offsets,
            flatten_ids,
            max_gaussians_per_tile=1,
        )

    rendered, alpha = render(means)
    np.testing.assert_allclose(np.asarray(alpha[0, 4, 4, 0]), 0.8, atol=1.0e-6)
    normalized_x = 1.0 / 20.0
    distance_squared = (
        2.0 * normalized_x / (0.1 * np.sqrt(1.0 + normalized_x**2))
    ) ** 2
    expected_off_axis = 0.8 * np.exp(-0.5 * distance_squared)
    np.testing.assert_allclose(
        np.asarray(alpha[0, 4, 5, 0]), expected_off_axis, rtol=1.0e-5
    )
    np.testing.assert_allclose(np.asarray(rendered[..., 0]), np.asarray(alpha[..., 0]))

    gradient = jax.grad(lambda value: render(value)[0].sum())(means)
    assert gradient.shape == means.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_eval3d_chunk_size_does_not_truncate_tile_candidates():
    means = jnp.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]], jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, jnp.float32)
    scales = jnp.full((2, 3), 0.1, jnp.float32)
    colors = jnp.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], jnp.float32)
    opacities = jnp.asarray([[0.5, 0.5]], jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.asarray(
        [[[20.0, 0.0, 4.5], [0.0, 20.0, 4.5], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )

    def render(chunk_size):
        return rasterize_to_pixels_eval3d(
            means,
            quats,
            scales,
            colors,
            opacities,
            viewmats,
            Ks,
            8,
            8,
            8,
            jnp.zeros((1, 1, 1), jnp.int32),
            jnp.asarray([0, 1], jnp.int32),
            max_gaussians_per_tile=chunk_size,
            return_info=True,
        )

    chunked_render, chunked_alpha, chunked_info = render(1)
    full_render, full_alpha, _ = render(2)

    assert not jnp.any(chunked_info["tile_overflow"])
    assert jnp.any(chunked_info["candidate_limit_exceeded"])
    assert jnp.allclose(chunked_render, full_render)
    assert jnp.allclose(chunked_alpha, full_alpha)


def test_eval3d_low_level_supports_leading_batch_dimensions():
    means = jnp.asarray([[[0.0, 0.0, 2.0]], [[0.0, 0.0, 3.0]]], jnp.float32)
    quats = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0], jnp.float32), (2, 1, 4))
    scales = jnp.full((2, 1, 3), 0.1, jnp.float32)
    colors = jnp.asarray([[[[1.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0]]]], jnp.float32)
    opacities = jnp.asarray([[[0.8]], [[0.6]]], jnp.float32)
    viewmats = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (2, 1, 4, 4))
    Ks = jnp.broadcast_to(
        jnp.asarray(
            [[20.0, 0.0, 4.5], [0.0, 20.0, 4.5], [0.0, 0.0, 1.0]],
            jnp.float32,
        ),
        (2, 1, 3, 3),
    )
    offsets = jnp.stack(
        (jnp.zeros((1, 2, 2), jnp.int32), jnp.ones((1, 2, 2), jnp.int32))
    )
    rendered, alpha = jax.jit(
        lambda: rasterize_to_pixels_eval3d(
            means,
            quats,
            scales,
            colors,
            opacities,
            viewmats,
            Ks,
            8,
            8,
            4,
            offsets,
            jnp.asarray([0, 1], jnp.int32),
            max_gaussians_per_tile=1,
        )
    )()

    assert rendered.shape == (2, 1, 8, 8, 3)
    assert alpha.shape == (2, 1, 8, 8, 1)
    np.testing.assert_allclose(np.asarray(alpha[:, 0, 4, 4, 0]), [0.8, 0.6])


def test_high_level_eval3d_uses_exact_world_space_response():
    means = jnp.asarray([[0.0, 0.0, 2.0]], jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], jnp.float32)
    scales = jnp.asarray([[0.1, 0.1, 0.1]], jnp.float32)
    colors = jnp.asarray([[1.0, 0.0, 0.0]], jnp.float32)
    opacities = jnp.asarray([0.8], jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.asarray(
        [[[20.0, 0.0, 4.5], [0.0, 20.0, 4.5], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )
    rendered, alpha, info = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        8,
        8,
        with_eval3d=True,
        config=RasterizationConfig(
            backend="intersections",
            tile_size=4,
            max_gaussians_per_tile=1,
            max_intersections=16,
            tile_batch_size=1,
            ut_chunk_size=1,
        ),
    )

    normalized_x = 1.0 / 20.0
    distance_squared = (
        2.0 * normalized_x / (0.1 * np.sqrt(1.0 + normalized_x**2))
    ) ** 2
    expected_off_axis = 0.8 * np.exp(-0.5 * distance_squared)
    np.testing.assert_allclose(np.asarray(alpha[0, 4, 4, 0]), 0.8, atol=1.0e-6)
    np.testing.assert_allclose(
        np.asarray(alpha[0, 4, 5, 0]), expected_off_axis, rtol=1.0e-5
    )
    np.testing.assert_allclose(np.asarray(rendered[..., 0]), np.asarray(alpha[..., 0]))
    assert bool(info["eval3d_world_space"])
    assert not bool(info["eval3d_ewa_approximation"])


def _current_main_eval_scene():
    means = jnp.asarray([[0.0, 0.0, 2.0]], jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], jnp.float32)
    scales = jnp.asarray([[0.1, 0.1, 0.1]], jnp.float32)
    opacities = jnp.asarray([0.8], jnp.float32)
    colors = jnp.asarray([[1.0, 0.0, 0.0]], jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.asarray(
        [[[20.0, 0.0, 4.5], [0.0, 20.0, 4.5], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )
    config = RasterizationConfig(
        backend="intersections",
        tile_size=4,
        max_gaussians_per_tile=1,
        max_intersections=16,
        tile_batch_size=1,
        ut_chunk_size=1,
    )
    return means, quats, scales, opacities, colors, viewmats, Ks, config


def test_eval3d_custom_rays_hit_distance_and_normals():
    means, quats, scales, opacities, colors, viewmats, Ks, config = (
        _current_main_eval_scene()
    )
    rays = jnp.zeros((1, 8, 8, 6), jnp.float32)
    rays = rays.at[..., 5].set(1.0)

    accumulated, alpha, accumulated_info = rasterization(
        means,
        quats,
        scales,
        opacities,
        None,
        viewmats,
        Ks,
        8,
        8,
        render_mode="d",
        with_eval3d=True,
        return_normals=True,
        rays=rays,
        config=config,
    )
    expected, _, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        None,
        viewmats,
        Ks,
        8,
        8,
        render_mode="Ed",
        with_eval3d=True,
        rays=rays,
        config=config,
    )
    combined, _, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        8,
        8,
        render_mode="RGB-d",
        with_eval3d=True,
        rays=rays,
        config=config,
    )
    parallel, parallel_alpha, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        None,
        viewmats,
        Ks,
        8,
        8,
        render_mode="d",
        with_eval3d=True,
        rays=rays,
        renderer_config=RendererConfig_ParallelBatch(),
        config=config,
    )

    np.testing.assert_allclose(np.asarray(alpha[0, 4, 4, 0]), 0.8, atol=1e-6)
    np.testing.assert_allclose(np.asarray(accumulated[0, 4, 4, 0]), 1.6, atol=1e-5)
    np.testing.assert_allclose(np.asarray(expected[0, 4, 4, 0]), 2.0, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(combined[0, 4, 4]), [0.8, 0.0, 0.0, 1.6], atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(accumulated_info["normals"][0, 4, 4]),
        [0.0, 0.0, -0.8],
        atol=1e-5,
    )
    np.testing.assert_allclose(np.asarray(parallel), np.asarray(accumulated))
    np.testing.assert_allclose(np.asarray(parallel_alpha), np.asarray(alpha))


def test_extra_signals_are_returned_separately_and_differentiable():
    means, quats, scales, opacities, colors, viewmats, Ks, config = (
        _current_main_eval_scene()
    )
    extra = jnp.asarray([[2.0, -1.0]], jnp.float32)

    @jax.jit
    def objective(current_extra):
        rendered, alpha, info = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            8,
            8,
            extra_signals=current_extra,
            config=config,
        )
        return rendered.sum() + info["render_extra_signals"].sum(), (
            rendered,
            alpha,
            info["render_extra_signals"],
        )

    (loss, (rendered, alpha, rendered_extra)), gradient = jax.value_and_grad(
        objective, has_aux=True
    )(extra)
    assert rendered.shape == (1, 8, 8, 3)
    assert rendered_extra.shape == (1, 8, 8, 2)
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_allclose(np.asarray(rendered[0, 4, 4, 0]), 0.8, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(rendered_extra[0, 4, 4]), [1.6, -0.8], atol=1e-6
    )
    np.testing.assert_allclose(np.asarray(alpha[0, 4, 4, 0]), 0.8, atol=1e-6)


def test_extra_signal_sh_uses_unclamped_bias():
    means, quats, scales, opacities, colors, viewmats, Ks, config = (
        _current_main_eval_scene()
    )
    extra_sh = jnp.zeros((1, 1, 1), jnp.float32)
    _, _, info = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        8,
        8,
        extra_signals=extra_sh,
        extra_signals_sh_degree=0,
        config=config,
    )
    np.testing.assert_allclose(
        np.asarray(info["render_extra_signals"][0, 4, 4, 0]),
        0.4,
        atol=1e-6,
    )


def test_eval3d_extra_returns_last_ids_sample_counts_and_normals():
    means, quats, scales, opacities, colors, viewmats, Ks, _ = (
        _current_main_eval_scene()
    )
    offsets = jnp.zeros((1, 2, 2), jnp.int32)
    rays = jnp.zeros((1, 8, 8, 6), jnp.float32).at[..., 5].set(1.0)
    rendered, alpha, last_ids, sample_counts, normals = (
        rasterize_to_pixels_eval3d_extra(
            means,
            quats,
            scales,
            colors[None],
            opacities[None],
            viewmats,
            Ks,
            8,
            8,
            4,
            offsets,
            jnp.asarray([0], jnp.int32),
            rays=rays,
            return_sample_counts=True,
            return_normals=True,
            max_gaussians_per_tile=1,
        )
    )

    assert rendered.shape == (1, 8, 8, 3)
    np.testing.assert_allclose(np.asarray(alpha[0, 4, 4, 0]), 0.8, atol=1e-6)
    assert int(last_ids[0, 4, 4]) == 0
    assert int(sample_counts[0, 4, 4]) == 1
    np.testing.assert_allclose(
        np.asarray(normals[0, 4, 4]), [0.0, 0.0, -0.8], atol=1e-6
    )
