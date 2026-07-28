import numpy as np

import jax
import jax.numpy as jnp

from jax_gs.compositing import composite_sorted_tile


def _dense_reference(
    gaussian_ids,
    candidate_count,
    means2d,
    conics,
    opacities,
    features,
    pixel_coords,
    pixel_valid,
    *,
    alpha_threshold=1.0 / 255.0,
    transmittance_threshold=1.0e-4,
):
    capacity = gaussian_ids.shape[0]
    positions = jnp.arange(capacity, dtype=jnp.int32)
    valid_id = (
        (positions < candidate_count)
        & (gaussian_ids >= 0)
        & (gaussian_ids < means2d.shape[0])
    )
    safe_ids = jnp.clip(gaussian_ids, 0, means2d.shape[0] - 1)
    means = means2d[safe_ids]
    selected_conics = conics[safe_ids]
    dx = pixel_coords[None, :, 0] - means[:, None, 0]
    dy = pixel_coords[None, :, 1] - means[:, None, 1]
    sigma = (
        0.5
        * (
            selected_conics[:, None, 0] * dx**2
            + selected_conics[:, None, 2] * dy**2
        )
        + selected_conics[:, None, 1] * dx * dy
    )
    alpha = jnp.minimum(
        opacities[safe_ids, None] * jnp.exp(-jnp.maximum(sigma, 0.0)), 0.999
    )
    alpha = jnp.where(
        valid_id[:, None]
        & pixel_valid[None, :]
        & jnp.isfinite(sigma)
        & (sigma >= 0.0)
        & (alpha >= alpha_threshold),
        alpha,
        0.0,
    )
    transmittance = jnp.concatenate(
        (jnp.ones_like(alpha[:1]), jnp.cumprod(1.0 - alpha, axis=0)[:-1]),
        axis=0,
    )
    weights = jnp.where(
        transmittance * (1.0 - alpha) > transmittance_threshold,
        alpha * transmittance,
        0.0,
    )
    rendered = jnp.einsum("kp,kd->pd", weights, features[safe_ids])
    return rendered, jnp.sum(weights, axis=0)[:, None]


def _scene():
    gaussian_ids = jnp.array([2, 0, 1, -1, -1], dtype=jnp.int32)
    means2d = jnp.array(
        [[0.5, 0.5], [1.5, 0.5], [1.0, 1.25]], dtype=jnp.float32
    )
    conics = jnp.array(
        [[1.0, 0.0, 1.0], [0.8, 0.1, 1.2], [1.1, -0.05, 0.9]],
        dtype=jnp.float32,
    )
    opacities = jnp.array([0.35, 0.55, 0.45], dtype=jnp.float32)
    features = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]], dtype=jnp.float32
    )
    pixel_coords = jnp.array(
        [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]],
        dtype=jnp.float32,
    )
    pixel_valid = jnp.array([True, True, True, False])
    return (
        gaussian_ids,
        means2d,
        conics,
        opacities,
        features,
        pixel_coords,
        pixel_valid,
    )


def test_chunked_values_and_gradients_match_dense_formula():
    ids, means, conics, opacities, features, pixels, pixel_valid = _scene()

    def chunked_loss(means, conics, opacities, features):
        rendered, alpha = composite_sorted_tile(
            ids,
            3,
            means,
            conics,
            opacities,
            features,
            pixels,
            pixel_valid,
            chunk_size=2,
        )
        return rendered, alpha, rendered.sum() + 0.37 * alpha.sum()

    def dense_loss(means, conics, opacities, features):
        rendered, alpha = _dense_reference(
            ids, 3, means, conics, opacities, features, pixels, pixel_valid
        )
        return rendered, alpha, rendered.sum() + 0.37 * alpha.sum()

    rendered, alpha, _ = chunked_loss(means, conics, opacities, features)
    expected_rendered, expected_alpha, _ = dense_loss(
        means, conics, opacities, features
    )
    np.testing.assert_allclose(rendered, expected_rendered, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(alpha, expected_alpha, rtol=1e-6, atol=1e-6)

    chunked_grads = jax.grad(lambda *args: chunked_loss(*args)[2], argnums=(0, 1, 2, 3))(
        means, conics, opacities, features
    )
    dense_grads = jax.grad(lambda *args: dense_loss(*args)[2], argnums=(0, 1, 2, 3))(
        means, conics, opacities, features
    )
    for actual, expected in zip(chunked_grads, dense_grads):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_candidate_count_is_jittable_and_zero_candidates_are_safe():
    ids, means, conics, opacities, features, pixels, pixel_valid = _scene()

    @jax.jit
    def render(candidate_count):
        return composite_sorted_tile(
            ids,
            candidate_count,
            means,
            conics,
            opacities,
            features,
            pixels,
            pixel_valid,
            chunk_size=2,
        )

    empty_render, empty_alpha = render(jnp.asarray(0, dtype=jnp.int32))
    cache_size = render._cache_size()
    partial_render, partial_alpha = render(jnp.asarray(2, dtype=jnp.int32))
    assert render._cache_size() == cache_size == 1
    np.testing.assert_array_equal(empty_render, np.zeros_like(empty_render))
    np.testing.assert_array_equal(empty_alpha, np.zeros_like(empty_alpha))
    assert np.any(np.asarray(partial_render) != 0.0)
    assert np.any(np.asarray(partial_alpha) != 0.0)


def test_opaque_early_termination_skips_later_candidates():
    ids = jnp.array([0, 1, 2], dtype=jnp.int32)
    means = jnp.zeros((3, 2), dtype=jnp.float32)
    conics = jnp.zeros((3, 3), dtype=jnp.float32)
    opacities = jnp.array([0.95, 0.9, 0.9], dtype=jnp.float32)
    features = jnp.array([[1.0], [100.0], [1000.0]], dtype=jnp.float32)
    pixels = jnp.zeros((1, 2), dtype=jnp.float32)
    valid = jnp.array([True])

    def loss(feature_values):
        rendered, alpha = composite_sorted_tile(
            ids,
            3,
            means,
            conics,
            opacities,
            feature_values,
            pixels,
            valid,
            chunk_size=1,
            transmittance_threshold=0.01,
        )
        return rendered[0, 0] + alpha[0, 0]

    rendered, alpha = composite_sorted_tile(
        ids,
        3,
        means,
        conics,
        opacities,
        features,
        pixels,
        valid,
        chunk_size=1,
        transmittance_threshold=0.01,
    )
    np.testing.assert_allclose(rendered, [[0.95]], rtol=1e-6)
    np.testing.assert_allclose(alpha, [[0.95]], rtol=1e-6)
    np.testing.assert_allclose(jax.grad(loss)(features), [[0.95], [0.0], [0.0]])


def test_empty_storage_and_all_invalid_pixels_are_safe():
    rendered, alpha = composite_sorted_tile(
        jnp.empty((0,), dtype=jnp.int32),
        0,
        jnp.empty((0, 2), dtype=jnp.float32),
        jnp.empty((0, 3), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0, 3), dtype=jnp.float32),
        jnp.array([[0.5, 0.5], [1.5, 0.5]], dtype=jnp.float32),
        jnp.array([True, True]),
        chunk_size=2,
    )
    assert rendered.shape == (2, 3)
    assert alpha.shape == (2, 1)
    np.testing.assert_array_equal(rendered, np.zeros((2, 3), np.float32))
    np.testing.assert_array_equal(alpha, np.zeros((2, 1), np.float32))

    ids, means, conics, opacities, features, pixels, _ = _scene()
    rendered, alpha = composite_sorted_tile(
        ids,
        3,
        means,
        conics,
        opacities,
        features,
        pixels,
        jnp.zeros((pixels.shape[0],), dtype=jnp.bool_),
        chunk_size=2,
    )
    np.testing.assert_array_equal(rendered, np.zeros_like(rendered))
    np.testing.assert_array_equal(alpha, np.zeros_like(alpha))
