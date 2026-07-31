import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax_gs.low_level import (
    accumulate,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_indices_in_range,
    rasterize_to_pixels,
)


def test_intersections_are_depth_sorted_padded_and_jittable():
    means2d = jnp.array([[[1.0, 1.0], [1.0, 1.0]]], dtype=jnp.float32)
    radii = jnp.ones((1, 2, 2), dtype=jnp.float32)
    depths = jnp.array([[2.0, 1.0]], dtype=jnp.float32)

    @jax.jit
    def intersect(active_mask):
        return isect_tiles(
            means2d,
            radii,
            depths,
            2,
            1,
            1,
            max_intersections=3,
            active_mask=active_mask,
        )

    result = intersect(jnp.array([True, True]))
    np.testing.assert_array_equal(np.asarray(result.tiles_per_gaussian), [[1, 1]])
    np.testing.assert_array_equal(np.asarray(result.flatten_ids), [1, 0, -1])
    assert int(result.valid_count) == 2
    assert not bool(result.overflow)
    cache_size = intersect._cache_size()

    masked = intersect(jnp.array([True, False]))
    assert intersect._cache_size() == cache_size == 1
    np.testing.assert_array_equal(np.asarray(masked.flatten_ids), [0, -1, -1])

    overflow = isect_tiles(
        means2d, radii, depths, 2, 1, 1, max_intersections=1
    )
    assert int(overflow.valid_count) == 1
    assert bool(overflow.overflow)


def test_isect_tiles_unpacks_like_upstream_three_value_contract():
    result = isect_tiles(
        jnp.asarray([[[1.0, 1.0]]], jnp.float32),
        jnp.asarray([[[1.0, 1.0]]], jnp.float32),
        jnp.asarray([[1.0]], jnp.float32),
        tile_size=2,
        tile_width=1,
        tile_height=1,
        max_intersections=1,
    )

    tiles_per_gauss, isect_ids, flatten_ids = result
    assert tiles_per_gauss is result.tiles_per_gaussian
    assert isect_ids is result.isect_ids
    assert flatten_ids is result.flatten_ids
    assert int(result.valid_count) == 1
    assert not bool(result.overflow)


def test_offsets_decode_multiple_images_and_non_power_of_two_tile_count():
    means2d = jnp.array([[[0.5, 0.5]], [[4.5, 0.5]]], dtype=jnp.float32)
    radii = jnp.full((2, 1, 2), 0.4, dtype=jnp.float32)
    depths = jnp.ones((2, 1), dtype=jnp.float32)
    intersections = isect_tiles(
        means2d, radii, depths, 2, 3, 1, max_intersections=2
    )
    offsets = isect_offset_encode(
        intersections.isect_ids,
        2,
        3,
        1,
        valid_count=intersections.valid_count,
        overflow=intersections.overflow,
        return_info=True,
    )
    np.testing.assert_array_equal(
        np.asarray(offsets.offsets), [[[0, 1, 1]], [[1, 1, 1]]]
    )
    assert int(offsets.valid_count) == 2
    assert not bool(offsets.overflow)


def test_indices_accumulate_and_bounded_pixel_rasterization_agree():
    means2d = jnp.array([[[1.0, 1.0]]], dtype=jnp.float32)
    radii = jnp.ones((1, 1, 2), dtype=jnp.float32)
    depths = jnp.ones((1, 1), dtype=jnp.float32)
    conics = jnp.zeros((1, 1, 3), dtype=jnp.float32)
    opacities = jnp.array([[0.5]], dtype=jnp.float32)
    colors = jnp.array([[[1.0, 0.0, 0.0]]], dtype=jnp.float32)
    intersections = isect_tiles(
        means2d, radii, depths, 2, 1, 1, max_intersections=1
    )
    offsets = isect_offset_encode(
        intersections.isect_ids,
        1,
        1,
        1,
        valid_count=intersections.valid_count,
        overflow=intersections.overflow,
        return_info=True,
    )

    indices = jax.jit(
        lambda: rasterize_to_indices_in_range(
            0,
            1,
            jnp.ones((1, 2, 2), dtype=jnp.float32),
            means2d,
            conics,
            opacities,
            2,
            2,
            2,
            offsets,
            intersections.flatten_ids,
            max_intersections=5,
        )
    )()
    assert int(indices.valid_count) == 4
    assert not bool(indices.overflow)
    np.testing.assert_array_equal(np.asarray(indices.gaussian_ids), [0, 0, 0, 0, -1])
    np.testing.assert_array_equal(np.asarray(indices.pixel_ids), [0, 1, 2, 3, -1])

    default_capacity = rasterize_to_indices_in_range(
        0,
        1,
        jnp.ones((1, 2, 2), dtype=jnp.float32),
        means2d,
        conics,
        opacities,
        2,
        2,
        2,
        offsets,
        intersections.flatten_ids,
    )
    assert default_capacity.gaussian_ids.shape == (4,)
    assert int(default_capacity.valid_count) == 4

    @jax.jit
    def dynamic_range(start, end):
        return rasterize_to_indices_in_range(
            start,
            end,
            jnp.ones((1, 2, 2), dtype=jnp.float32),
            means2d,
            conics,
            opacities,
            2,
            2,
            2,
            offsets,
            intersections.flatten_ids,
            max_intersections=5,
        )

    assert int(dynamic_range(jnp.int32(0), jnp.int32(1)).valid_count) == 4
    cache_size = dynamic_range._cache_size()
    assert int(dynamic_range(jnp.int32(1), jnp.int32(2)).valid_count) == 0
    assert dynamic_range._cache_size() == cache_size == 1

    accumulated_colors, accumulated_alphas = accumulate(
        means2d,
        conics,
        opacities,
        colors,
        indices.gaussian_ids,
        indices.pixel_ids,
        indices.image_ids,
        2,
        2,
        valid_count=indices.valid_count,
    )
    expected_colors = np.zeros((1, 2, 2, 3), dtype=np.float32)
    expected_colors[..., 0] = 0.5
    np.testing.assert_allclose(np.asarray(accumulated_colors), expected_colors)
    np.testing.assert_allclose(np.asarray(accumulated_alphas), 0.5)

    rendered, alphas, info = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        2,
        2,
        2,
        offsets,
        intersections.flatten_ids,
        backgrounds=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
        max_gaussians_per_tile=1,
        return_info=True,
    )
    expected_with_background = expected_colors.copy()
    expected_with_background[..., 2] = 0.5
    np.testing.assert_allclose(np.asarray(rendered), expected_with_background)
    np.testing.assert_allclose(np.asarray(alphas), 0.5)
    assert not bool(info["overflow"])

    color_grad = jax.grad(
        lambda value: rasterize_to_pixels(
            means2d,
            conics,
            value,
            opacities,
            2,
            2,
            2,
            offsets,
            intersections.flatten_ids,
            max_gaussians_per_tile=1,
        )[0].sum()
    )(colors)
    assert jnp.all(jnp.isfinite(color_grad))


def test_pixel_index_capacity_reports_overflow_without_large_allocation():
    means2d = jnp.array([[[1.0, 1.0]]], dtype=jnp.float32)
    conics = jnp.zeros((1, 1, 3), dtype=jnp.float32)
    opacities = jnp.array([[0.5]], dtype=jnp.float32)
    intersections = isect_tiles(
        means2d,
        jnp.ones((1, 1, 2), dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
        2,
        1,
        1,
        max_intersections=1,
    )
    offsets = isect_offset_encode(
        intersections.isect_ids, 1, 1, 1, valid_count=intersections.valid_count
    )
    result = rasterize_to_indices_in_range(
        0,
        1,
        jnp.ones((1, 2, 2), dtype=jnp.float32),
        means2d,
        conics,
        opacities,
        2,
        2,
        2,
        offsets,
        intersections.flatten_ids,
        valid_count=intersections.valid_count,
        max_intersections=3,
    )
    assert result.gaussian_ids.shape == (3,)
    assert int(result.valid_count) == 3
    assert bool(result.overflow)


def test_rasterize_to_pixels_composites_all_candidates_across_tile_chunks():
    means2d = jnp.array(
        [[[1.0, 1.0], [1.0, 1.0]]], dtype=jnp.float32
    )
    conics = jnp.zeros((1, 2, 3), dtype=jnp.float32)
    opacities = jnp.full((1, 2), 0.5, dtype=jnp.float32)
    colors = jnp.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=jnp.float32
    )
    offsets = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    flatten_ids = jnp.array([0, 1], dtype=jnp.int32)

    def render(current_colors):
        return rasterize_to_pixels(
            means2d,
            conics,
            current_colors,
            opacities,
            2,
            2,
            2,
            offsets,
            flatten_ids,
            backgrounds=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
            valid_count=jnp.asarray(2, dtype=jnp.int32),
            max_gaussians_per_tile=1,
            return_info=True,
        )

    rendered, alphas, info = jax.jit(render)(colors)
    expected = np.broadcast_to(
        np.array([0.5, 0.25, 0.25], dtype=np.float32), (1, 2, 2, 3)
    )
    np.testing.assert_allclose(np.asarray(rendered), expected, atol=1e-6)
    np.testing.assert_allclose(np.asarray(alphas), 0.75, atol=1e-6)
    assert not bool(jnp.any(info["tile_overflow"]))
    assert not bool(info["overflow"])

    gradient = jax.grad(lambda value: render(value)[0].sum())(colors)
    np.testing.assert_allclose(
        np.asarray(gradient),
        np.array([[[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]], np.float32),
        atol=1e-6,
    )


def test_rasterize_absgrad_sums_absolute_per_pixel_contributions_before_cancellation():
    means2d = jnp.asarray([[[1.0, 0.5]]], jnp.float32)
    conics = jnp.asarray([[[1.0, 0.0, 1.0]]], jnp.float32)
    colors = jnp.asarray([[[1.0]]], jnp.float32)
    opacities = jnp.asarray([[0.5]], jnp.float32)
    offsets = jnp.zeros((1, 1, 1), jnp.int32)
    flatten_ids = jnp.asarray([0], jnp.int32)
    color_cotangents = jnp.ones((1, 1, 2, 1), jnp.float32)
    alpha_cotangents = jnp.zeros((1, 1, 2, 1), jnp.float32)

    def render(current_means, probe=None):
        return rasterize_to_pixels(
            current_means,
            conics,
            colors,
            opacities,
            2,
            1,
            2,
            offsets,
            flatten_ids,
            valid_count=1,
            max_gaussians_per_tile=1,
            return_info=True,
            absgrad=probe is not None,
            _means2d_absgrad_probe=probe,
        )[:2]

    def objective(current_means, probe, current_color_cotangents):
        rendered, alpha = render(current_means, probe)
        return (
            jnp.sum(rendered * current_color_cotangents)
            + jnp.sum(alpha * alpha_cotangents),
            (rendered, alpha),
        )

    baseline_outputs, pullback = jax.vjp(render, means2d)
    signed_gradient = pullback((color_cotangents, alpha_cotangents))[0]
    np.testing.assert_allclose(signed_gradient[..., 0], 0.0, atol=1.0e-7)

    probe = jnp.zeros_like(means2d)
    compiled = jax.jit(
        jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)
    )
    (_, probed_outputs), (probed_signed_gradient, compiled_absgrad) = compiled(
        means2d, probe, color_cotangents
    )
    for probed_output, baseline_output in zip(
        probed_outputs, baseline_outputs, strict=True
    ):
        np.testing.assert_allclose(probed_output, baseline_output, atol=0.0)
    np.testing.assert_allclose(
        probed_signed_gradient, signed_gradient, rtol=1.0e-6, atol=1.0e-7
    )
    expected_x = 0.5 * np.exp(-0.125)
    np.testing.assert_allclose(
        compiled_absgrad,
        np.asarray([[[expected_x, 0.0]]], np.float32),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert float(compiled_absgrad[0, 0, 0]) > float(
        jnp.abs(signed_gradient[0, 0, 0])
    )

    vmapped = jax.vmap(lambda cot: compiled(means2d, probe, cot)[1][1])(
        jnp.stack((color_cotangents, -color_cotangents))
    )
    np.testing.assert_allclose(vmapped[0], vmapped[1], rtol=1.0e-6)
    second_order = jax.grad(
        lambda value: jax.grad(
            lambda current_probe: objective(
                value, current_probe, color_cotangents
            )[0]
        )(probe).sum()
    )(means2d)
    assert jnp.all(jnp.isfinite(second_order))


def test_absgrad_probe_preserves_parameter_gradients_and_matches_pixel_vjps():
    means2d = jnp.asarray(
        [[[0.75, 0.8], [1.3, 1.2]]], jnp.float32
    )
    conics = jnp.asarray(
        [[[1.0, 0.1, 0.8], [0.7, -0.05, 1.1]]], jnp.float32
    )
    colors = jnp.asarray(
        [[[0.9, 0.2], [0.1, 0.8]]], jnp.float32
    )
    opacities = jnp.asarray([[0.45, 0.35]], jnp.float32)
    background = jnp.asarray([[0.2, -0.1]], jnp.float32)
    offsets = jnp.zeros((1, 1, 1), jnp.int32)
    flatten_ids = jnp.asarray([0, 1], jnp.int32)
    color_cotangents = jnp.asarray(
        [[[[0.2, -0.3], [0.7, 0.1]], [[-0.4, 0.5], [0.3, -0.2]]]],
        jnp.float32,
    )
    alpha_cotangents = jnp.asarray(
        [[[[0.15], [-0.25]], [[0.4], [-0.1]]]], jnp.float32
    )

    def render(
        current_means,
        current_conics,
        current_colors,
        current_opacities,
        current_background,
        probe=None,
    ):
        return rasterize_to_pixels(
            current_means,
            current_conics,
            current_colors,
            current_opacities,
            2,
            2,
            2,
            offsets,
            flatten_ids,
            backgrounds=current_background,
            valid_count=2,
            max_gaussians_per_tile=1,
            return_info=True,
            absgrad=probe is not None,
            _means2d_absgrad_probe=probe,
        )[:2]

    def objective(*parameters, probe=None):
        rendered, alpha = render(*parameters, probe=probe)
        loss = jnp.sum(rendered * color_cotangents) + jnp.sum(
            alpha * alpha_cotangents
        )
        return loss, (rendered, alpha)

    parameters = (means2d, conics, colors, opacities, background)
    (_, baseline_outputs), baseline_gradients = jax.value_and_grad(
        objective, argnums=(0, 1, 2, 3, 4), has_aux=True
    )(*parameters)
    probe = jnp.zeros_like(means2d)

    def probed_objective(*values):
        return objective(*values[:-1], probe=values[-1])

    (_, probed_outputs), probed_gradients = jax.value_and_grad(
        probed_objective, argnums=(0, 1, 2, 3, 4, 5), has_aux=True
    )(*parameters, probe)
    for probed_output, baseline_output in zip(
        probed_outputs, baseline_outputs, strict=True
    ):
        np.testing.assert_allclose(probed_output, baseline_output, atol=0.0)
    for probed_gradient, baseline_gradient in zip(
        probed_gradients[:-1], baseline_gradients, strict=True
    ):
        np.testing.assert_allclose(
            probed_gradient, baseline_gradient, rtol=2.0e-6, atol=2.0e-7
        )

    _, pullback = jax.vjp(
        lambda value: render(value, *parameters[1:]), means2d
    )
    expected_absgrad = jnp.zeros_like(means2d)
    for y in range(2):
        for x in range(2):
            isolated_color = jnp.zeros_like(color_cotangents).at[0, y, x].set(
                color_cotangents[0, y, x]
            )
            isolated_alpha = jnp.zeros_like(alpha_cotangents).at[0, y, x].set(
                alpha_cotangents[0, y, x]
            )
            pixel_gradient = pullback((isolated_color, isolated_alpha))[0]
            expected_absgrad = expected_absgrad + jnp.abs(pixel_gradient)
    np.testing.assert_allclose(
        probed_gradients[-1], expected_absgrad, rtol=2.0e-6, atol=2.0e-7
    )


@pytest.mark.parametrize("packed", [False, True])
def test_absgrad_probe_scatter_accumulates_repeated_gaussian_across_tiles(
    packed,
):
    dense_means = jnp.asarray([[[2.0, 0.5]]], jnp.float32)
    dense_conics = jnp.asarray([[[1.0, 0.0, 1.0]]], jnp.float32)
    dense_colors = jnp.asarray([[[1.0]]], jnp.float32)
    dense_opacities = jnp.asarray([[0.5]], jnp.float32)
    means2d = dense_means[0] if packed else dense_means
    conics = dense_conics[0] if packed else dense_conics
    colors = dense_colors[0] if packed else dense_colors
    opacities = dense_opacities[0] if packed else dense_opacities
    offsets = jnp.asarray([[[0, 1]]], jnp.int32)
    flatten_ids = jnp.asarray([0, 0], jnp.int32)
    color_cotangents = jnp.ones((1, 1, 4, 1), jnp.float32)
    alpha_cotangents = jnp.zeros((1, 1, 4, 1), jnp.float32)

    def objective(current_means, probe):
        rendered, alpha = rasterize_to_pixels(
            current_means,
            conics,
            colors,
            opacities,
            4,
            1,
            2,
            offsets,
            flatten_ids,
            packed=packed,
            absgrad=True,
            valid_count=2,
            max_gaussians_per_tile=1,
            _means2d_absgrad_probe=probe,
        )
        return jnp.sum(rendered * color_cotangents) + jnp.sum(
            alpha * alpha_cotangents
        )

    signed_gradient, absolute_gradient = jax.grad(
        objective, argnums=(0, 1)
    )(means2d, jnp.zeros_like(means2d))
    np.testing.assert_allclose(signed_gradient[..., 0], 0.0, atol=1.0e-7)
    distances = np.asarray([-1.5, -0.5, 0.5, 1.5], np.float32)
    expected_x = np.sum(0.5 * np.exp(-0.5 * distances**2) * np.abs(distances))
    expected = np.zeros_like(np.asarray(means2d))
    expected[..., 0] = expected_x
    np.testing.assert_allclose(
        absolute_gradient, expected, rtol=2.0e-6, atol=2.0e-7
    )


def test_absgrad_probe_contract_is_explicit_and_shape_dtype_checked():
    arguments = (
        jnp.asarray([[[1.0, 0.5]]], jnp.float32),
        jnp.asarray([[[1.0, 0.0, 1.0]]], jnp.float32),
        jnp.asarray([[[1.0]]], jnp.float32),
        jnp.asarray([[0.5]], jnp.float32),
        2,
        1,
        2,
        jnp.zeros((1, 1, 1), jnp.int32),
        jnp.asarray([0], jnp.int32),
    )
    with pytest.raises(ValueError, match="zero-valued.*probe"):
        rasterize_to_pixels(*arguments, absgrad=True)
    with pytest.raises(ValueError, match="requires absgrad=True"):
        rasterize_to_pixels(
            *arguments,
            _means2d_absgrad_probe=jnp.zeros((1, 1, 2), jnp.float32),
        )
    with pytest.raises(ValueError, match="same shape"):
        rasterize_to_pixels(
            *arguments,
            absgrad=True,
            _means2d_absgrad_probe=jnp.zeros((1, 2), jnp.float32),
        )
    with pytest.raises(TypeError, match="same dtype"):
        rasterize_to_pixels(
            *arguments,
            absgrad=True,
            _means2d_absgrad_probe=jnp.zeros((1, 1, 2), jnp.float16),
        )


def test_a_tile_claiming_more_candidates_than_slots_reports_overflow():
    # The chunk loop is sized for the most candidates one tile can hold, which
    # is the slot count rather than the whole intersection buffer: isect_tiles
    # emits a tile at most once per Gaussian. Offsets that claim more than that
    # cannot come from isect_tiles, so the render is incomplete and has to say
    # so instead of dropping the tail.
    slot_count = 2
    means2d = jnp.full((1, slot_count, 2), 0.5, jnp.float32)
    conics = jnp.tile(
        jnp.asarray([[[1.0, 0.0, 1.0]]], jnp.float32), (1, slot_count, 1)
    )
    colors = jnp.ones((1, slot_count, 3), jnp.float32)
    opacities = jnp.full((1, slot_count), 0.5, jnp.float32)
    offsets = jnp.zeros((1, 1, 1), jnp.int32)
    # One tile whose every slot repeats well past the slot count.
    repeated_ids = jnp.tile(jnp.arange(slot_count, dtype=jnp.int32), 8)

    _, _, info = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        1,
        1,
        1,
        offsets,
        repeated_ids,
        max_gaussians_per_tile=1,
        return_info=True,
    )
    assert bool(jnp.any(info["tile_overflow"]))
    assert bool(info["overflow"])

    # The same buffer with a truthful offset table renders without complaint.
    _, _, honest = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        1,
        1,
        1,
        offsets,
        repeated_ids[:slot_count],
        max_gaussians_per_tile=1,
        return_info=True,
    )
    assert not bool(jnp.any(honest["tile_overflow"]))
    assert not bool(honest["overflow"])
