import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.low_level import (
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)
from jax_gs.sparse import (
    build_sparse_tile_layout,
    isect_tiles_sparse,
    rasterize_to_pixels_sparse,
)


def _scene():
    means2d = jnp.asarray(
        [
            [[1.5, 1.5], [4.5, 2.5], [7.0, 5.0], [2.0, 5.5], [8.0, 1.0]],
            [[2.5, 2.0], [5.5, 1.5], [7.5, 5.5], [1.5, 5.0], [4.0, 4.0]],
        ],
        jnp.float32,
    )
    conics = jnp.asarray(
        [
            [
                [0.18, 0.00, 0.14],
                [0.12, 0.01, 0.15],
                [0.20, -0.02, 0.16],
                [0.11, 0.00, 0.13],
                [0.16, 0.01, 0.19],
            ],
            [
                [0.15, -0.01, 0.18],
                [0.13, 0.00, 0.12],
                [0.17, 0.02, 0.14],
                [0.14, -0.01, 0.16],
                [0.10, 0.01, 0.11],
            ],
        ],
        jnp.float32,
    )
    colors = jnp.asarray(
        [
            [
                [1.0, 0.1, 0.2],
                [0.2, 0.9, 0.1],
                [0.1, 0.2, 1.0],
                [0.8, 0.7, 0.1],
                [0.4, 0.2, 0.7],
            ],
            [
                [0.2, 0.8, 0.9],
                [0.9, 0.3, 0.1],
                [0.2, 0.4, 0.8],
                [0.7, 0.2, 0.6],
                [0.3, 0.9, 0.4],
            ],
        ],
        jnp.float32,
    )
    opacities = jnp.asarray(
        [[0.55, 0.75, 0.45, 0.62, 0.38], [0.68, 0.51, 0.73, 0.42, 0.59]],
        jnp.float32,
    )
    radii = jnp.asarray(
        [
            [[4, 4], [5, 4], [4, 4], [4, 4], [3, 3]],
            [[4, 4], [4, 4], [4, 4], [3, 4], [5, 5]],
        ],
        jnp.int32,
    )
    depths = jnp.asarray(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.5, 4.5, 5.5]],
        jnp.float32,
    )
    return means2d, conics, colors, opacities, radii, depths


def _pixels():
    return (
        jnp.asarray(
            [[0, 0], [3, 4], [6, 8], [2, 7], [5, 1], [1, 3], [6, 0], [4, 6]],
            jnp.int32,
        ),
        jnp.asarray([0, 1, 0, 1, 0, 1, 1, 0], jnp.int32),
    )


def _sparse_layout(scene, pixels, pixel_image_ids, *, packed=False):
    means2d, _conics, _colors, _opacities, radii, depths = scene
    image_count, tile_size, tile_width, tile_height = 2, 4, 3, 2
    layout = build_sparse_tile_layout(
        pixels,
        pixel_image_ids,
        image_count,
        tile_size,
        tile_width,
        tile_height,
    )
    if packed:
        gaussian_image_ids = jnp.repeat(
            jnp.arange(image_count, dtype=jnp.int32), means2d.shape[1]
        )
        intersections = isect_tiles_sparse(
            means2d.reshape(-1, 2),
            radii.reshape(-1, 2),
            depths.reshape(-1),
            layout.active_tile_mask,
            layout.active_tiles,
            image_count,
            tile_size,
            tile_width,
            tile_height,
            image_ids=gaussian_image_ids,
            active_tile_count=layout.valid_count,
        )
    else:
        intersections = isect_tiles_sparse(
            means2d,
            radii,
            depths,
            layout.active_tile_mask,
            layout.active_tiles,
            image_count,
            tile_size,
            tile_width,
            tile_height,
            active_tile_count=layout.valid_count,
        )
    return layout, intersections


def _dense_render(scene, *, backgrounds=None, masks=None, packed=False):
    means2d, conics, colors, opacities, radii, depths = scene
    image_count, tile_size, tile_width, tile_height = 2, 4, 3, 2
    capacity = image_count * means2d.shape[1] * tile_width * tile_height
    if packed:
        gaussian_image_ids = jnp.repeat(
            jnp.arange(image_count, dtype=jnp.int32), means2d.shape[1]
        )
        means2d = means2d.reshape(-1, 2)
        conics = conics.reshape(-1, 3)
        colors = colors.reshape(-1, colors.shape[-1])
        opacities = opacities.reshape(-1)
        radii = radii.reshape(-1, 2)
        depths = depths.reshape(-1)
        intersections = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
            packed=True,
            n_images=image_count,
            image_ids=gaussian_image_ids,
            max_intersections=capacity,
        )
    else:
        intersections = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
            n_images=image_count,
            max_intersections=capacity,
        )
    offsets = isect_offset_encode(
        intersections.isect_ids,
        image_count,
        tile_width,
        tile_height,
        valid_count=intersections.valid_count,
    )
    return rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        9,
        7,
        tile_size,
        offsets,
        intersections.flatten_ids,
        backgrounds=backgrounds,
        masks=masks,
        packed=packed,
        valid_count=intersections.valid_count,
        max_gaussians_per_tile=means2d.shape[-2] if not packed else means2d.shape[0],
        return_info=True,
    )[:2]


def _sparse_render(
    scene,
    pixels,
    pixel_image_ids,
    *,
    backgrounds=None,
    masks=None,
    packed=False,
):
    means2d, conics, colors, opacities, _radii, _depths = scene
    layout, intersections = _sparse_layout(
        scene, pixels, pixel_image_ids, packed=packed
    )
    if packed:
        means2d = means2d.reshape(-1, 2)
        conics = conics.reshape(-1, 3)
        colors = colors.reshape(-1, colors.shape[-1])
        opacities = opacities.reshape(-1)
    return rasterize_to_pixels_sparse(
        means2d,
        conics,
        colors,
        opacities,
        pixel_image_ids,
        layout.active_tiles,
        intersections.tile_offsets,
        intersections.flatten_ids,
        layout.tile_pixel_mask,
        layout.tile_pixel_cumsum,
        layout.pixel_map,
        9,
        7,
        4,
        3,
        2,
        backgrounds=backgrounds,
        masks=masks,
        packed=packed,
        active_tile_count=layout.valid_count,
        valid_count=intersections.valid_count,
        overflow=layout.overflow | intersections.overflow,
        return_info=True,
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.resource_heavy
def test_sparse_pixels_match_dense_gather_and_jit(packed):
    scene = _scene()
    pixels, image_ids = _pixels()
    backgrounds = jnp.asarray([[0.05, 0.1, 0.15], [0.2, 0.1, 0.05]])

    sparse_colors, sparse_alphas, info = jax.jit(
        lambda: _sparse_render(
            scene,
            pixels,
            image_ids,
            backgrounds=backgrounds,
            packed=packed,
        )
    )()
    dense_colors, dense_alphas = _dense_render(
        scene, backgrounds=backgrounds, packed=packed
    )
    expected_colors = dense_colors[image_ids, pixels[:, 0], pixels[:, 1]]
    expected_alphas = dense_alphas[image_ids, pixels[:, 0], pixels[:, 1]]

    np.testing.assert_allclose(sparse_colors, expected_colors, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(sparse_alphas, expected_alphas, rtol=2e-6, atol=2e-6)
    assert int(info["decoded_pixel_count"]) == len(pixels)
    assert not bool(info["overflow"])


def test_sparse_pixels_match_dense_tile_masks():
    scene = _scene()
    pixels, image_ids = _pixels()
    masks = jnp.asarray(
        [
            [[True, False, True], [False, True, True]],
            [[False, True, True], [True, False, True]],
        ]
    )
    backgrounds = jnp.asarray([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])

    sparse_colors, sparse_alphas, info = _sparse_render(
        scene,
        pixels,
        image_ids,
        backgrounds=backgrounds,
        masks=masks,
    )
    dense_colors, dense_alphas = _dense_render(
        scene, backgrounds=backgrounds, masks=masks
    )

    np.testing.assert_allclose(
        sparse_colors,
        dense_colors[image_ids, pixels[:, 0], pixels[:, 1]],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        sparse_alphas,
        dense_alphas[image_ids, pixels[:, 0], pixels[:, 1]],
        rtol=2e-6,
        atol=2e-6,
    )
    assert not bool(info["overflow"])


@pytest.mark.resource_heavy
def test_sparse_pixels_gradients_match_dense_gather():
    scene = _scene()
    pixels, image_ids = _pixels()
    means2d, conics, colors, opacities, radii, depths = scene
    backgrounds = jnp.asarray([[0.05, 0.1, 0.15], [0.2, 0.1, 0.05]])
    layout, sparse_intersections = _sparse_layout(scene, pixels, image_ids)
    dense_capacity = 2 * means2d.shape[1] * 3 * 2
    dense_intersections = isect_tiles(
        means2d,
        radii,
        depths,
        4,
        3,
        2,
        n_images=2,
        max_intersections=dense_capacity,
    )
    dense_offsets = isect_offset_encode(
        dense_intersections.isect_ids,
        2,
        3,
        2,
        valid_count=dense_intersections.valid_count,
    )
    color_cotangent = jnp.linspace(0.1, 1.0, len(pixels) * 3).reshape(-1, 3)
    alpha_cotangent = jnp.linspace(0.2, 0.9, len(pixels)).reshape(-1, 1)

    def sparse_loss(m, c, rgb, opacity, background):
        rendered, alpha, _ = rasterize_to_pixels_sparse(
            m,
            c,
            rgb,
            opacity,
            image_ids,
            layout.active_tiles,
            sparse_intersections.tile_offsets,
            sparse_intersections.flatten_ids,
            layout.tile_pixel_mask,
            layout.tile_pixel_cumsum,
            layout.pixel_map,
            9,
            7,
            4,
            3,
            2,
            backgrounds=background,
            active_tile_count=layout.valid_count,
            valid_count=sparse_intersections.valid_count,
            return_info=True,
        )
        return jnp.sum(rendered * color_cotangent) + jnp.sum(alpha * alpha_cotangent)

    def dense_loss(m, c, rgb, opacity, background):
        rendered, alpha, _ = rasterize_to_pixels(
            m,
            c,
            rgb,
            opacity,
            9,
            7,
            4,
            dense_offsets,
            dense_intersections.flatten_ids,
            backgrounds=background,
            valid_count=dense_intersections.valid_count,
            max_gaussians_per_tile=means2d.shape[1],
            return_info=True,
        )
        gathered_colors = rendered[image_ids, pixels[:, 0], pixels[:, 1]]
        gathered_alphas = alpha[image_ids, pixels[:, 0], pixels[:, 1]]
        return jnp.sum(gathered_colors * color_cotangent) + jnp.sum(
            gathered_alphas * alpha_cotangent
        )

    sparse_grad = jax.jit(jax.grad(sparse_loss, argnums=(0, 1, 2, 3, 4)))(
        means2d, conics, colors, opacities, backgrounds
    )
    dense_grad = jax.jit(jax.grad(dense_loss, argnums=(0, 1, 2, 3, 4)))(
        means2d, conics, colors, opacities, backgrounds
    )
    for actual, expected in zip(sparse_grad, dense_grad, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_sparse_pixels_empty_gaussians_return_background():
    pixels = jnp.asarray([[0, 0], [3, 4]], jnp.int32)
    image_ids = jnp.asarray([0, 1], jnp.int32)
    layout = build_sparse_tile_layout(pixels, image_ids, 2, 4, 3, 2)
    intersections = isect_tiles_sparse(
        jnp.empty((2, 0, 2)),
        jnp.empty((2, 0, 2)),
        jnp.empty((2, 0)),
        layout.active_tile_mask,
        layout.active_tiles,
        2,
        4,
        3,
        2,
        active_tile_count=layout.valid_count,
    )
    backgrounds = jnp.asarray([[0.1, 0.2], [0.3, 0.4]])
    rendered, alpha, info = rasterize_to_pixels_sparse(
        jnp.empty((2, 0, 2)),
        jnp.empty((2, 0, 3)),
        jnp.empty((2, 0, 2)),
        jnp.empty((2, 0)),
        image_ids,
        layout.active_tiles,
        intersections.tile_offsets,
        intersections.flatten_ids,
        layout.tile_pixel_mask,
        layout.tile_pixel_cumsum,
        layout.pixel_map,
        9,
        7,
        4,
        3,
        2,
        backgrounds=backgrounds,
        active_tile_count=layout.valid_count,
        valid_count=intersections.valid_count,
        return_info=True,
    )
    np.testing.assert_array_equal(rendered, backgrounds)
    np.testing.assert_array_equal(alpha, np.zeros((2, 1)))
    assert not bool(info["overflow"])


@pytest.mark.parametrize("packed", [False, True])
def test_sparse_pixels_absgrad_probe_matches_per_pixel_vjps(packed):
    scene = _scene()
    pixels, image_ids = _pixels()
    pixels = pixels[:3]
    image_ids = image_ids[:3]
    means2d, conics, colors, opacities, _radii, _depths = scene
    layout, intersections = _sparse_layout(scene, pixels, image_ids, packed=packed)
    if packed:
        means2d = means2d.reshape(-1, 2)
        conics = conics.reshape(-1, 3)
        colors = colors.reshape(-1, colors.shape[-1])
        opacities = opacities.reshape(-1)
    color_cotangent = jnp.asarray(
        [[0.7, -0.2, 0.4], [-0.3, 0.8, 0.1], [0.2, -0.5, 0.9]],
        jnp.float32,
    )
    alpha_cotangent = jnp.asarray([[0.6], [-0.4], [0.3]], jnp.float32)

    def render(m, *, absgrad=False, probe=None):
        return rasterize_to_pixels_sparse(
            m,
            conics,
            colors,
            opacities,
            image_ids,
            layout.active_tiles,
            intersections.tile_offsets,
            intersections.flatten_ids,
            layout.tile_pixel_mask,
            layout.tile_pixel_cumsum,
            layout.pixel_map,
            9,
            7,
            4,
            3,
            2,
            packed=packed,
            absgrad=absgrad,
            active_tile_count=layout.valid_count,
            valid_count=intersections.valid_count,
            _means2d_absgrad_probe=probe,
        )

    def joint_loss(m, probe):
        rendered, alpha = render(m, absgrad=True, probe=probe)
        return jnp.sum(rendered * color_cotangent) + jnp.sum(alpha * alpha_cotangent)

    zero_probe = jnp.zeros_like(means2d)
    signed_gradient, absolute_gradient = jax.jit(jax.grad(joint_loss, argnums=(0, 1)))(
        means2d, zero_probe
    )

    def ordinary_loss(m):
        rendered, alpha = render(m)
        return jnp.sum(rendered * color_cotangent) + jnp.sum(alpha * alpha_cotangent)

    expected_signed = jax.grad(ordinary_loss)(means2d)
    expected_absolute = jnp.zeros_like(means2d)
    for pixel_index in range(pixels.shape[0]):

        def pixel_loss(m, pixel_index=pixel_index):
            rendered, alpha = render(m)
            return jnp.sum(
                rendered[pixel_index] * color_cotangent[pixel_index]
            ) + jnp.sum(alpha[pixel_index] * alpha_cotangent[pixel_index])

        expected_absolute = expected_absolute + jnp.abs(jax.grad(pixel_loss)(means2d))

    np.testing.assert_allclose(signed_gradient, expected_signed, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        absolute_gradient, expected_absolute, rtol=2e-5, atol=2e-5
    )
    ordinary_render = render(means2d)
    probed_render = render(means2d, absgrad=True, probe=zero_probe)
    for actual, expected in zip(probed_render, ordinary_render, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_sparse_pixels_absgrad_probe_contract_errors():
    scene = _scene()
    pixels, image_ids = _pixels()
    means2d, conics, colors, opacities, _radii, _depths = scene
    layout, intersections = _sparse_layout(scene, pixels, image_ids)

    def call(*, absgrad, probe):
        return rasterize_to_pixels_sparse(
            means2d,
            conics,
            colors,
            opacities,
            image_ids,
            layout.active_tiles,
            intersections.tile_offsets,
            intersections.flatten_ids,
            layout.tile_pixel_mask,
            layout.tile_pixel_cumsum,
            layout.pixel_map,
            9,
            7,
            4,
            3,
            2,
            absgrad=absgrad,
            active_tile_count=layout.valid_count,
            valid_count=intersections.valid_count,
            _means2d_absgrad_probe=probe,
        )

    with pytest.raises(ValueError, match="requires.*probe"):
        call(absgrad=True, probe=None)
    with pytest.raises(ValueError, match="requires absgrad=True"):
        call(absgrad=False, probe=jnp.zeros_like(means2d))
    with pytest.raises(ValueError, match="same shape"):
        call(absgrad=True, probe=jnp.zeros((1, 2), means2d.dtype))
    with pytest.raises(TypeError, match="same dtype"):
        call(absgrad=True, probe=jnp.zeros(means2d.shape, jnp.float16))


def test_sparse_backward_does_not_store_every_pixel():
    # Per-pixel sampling is rematerialized, so reverse mode keeps one pixel's
    # candidate weights alive instead of all of them. Without it this scene
    # measured a backward workspace 17 times the forward one.
    scene = _scene()
    pixels, pixel_image_ids = _pixels()

    def render(means2d, conics, colors, opacities):
        return _sparse_render(
            (means2d, conics, colors, opacities, scene[4], scene[5]),
            pixels,
            pixel_image_ids,
        )[0]

    def loss(means2d, conics, colors, opacities):
        rendered, alpha = _sparse_render(
            (means2d, conics, colors, opacities, scene[4], scene[5]),
            pixels,
            pixel_image_ids,
        )[:2]
        return jnp.mean(rendered**2) + jnp.mean(alpha**2)

    args = scene[:4]
    forward = jax.jit(render).lower(*args).compile()
    backward = (
        jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3))).lower(*args).compile()
    )
    try:
        forward_temp = forward.memory_analysis().temp_size_in_bytes
        backward_temp = backward.memory_analysis().temp_size_in_bytes
    except (AttributeError, NotImplementedError) as exc:  # pragma: no cover
        pytest.skip(f"memory analysis is unavailable: {exc}")

    assert backward_temp < 8 * max(forward_temp, 1)
