import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.config import RasterizationConfig
from jax_gs.intersections import intersect_tiles
from jax_gs.rasterization import rasterization, rasterization_inria_wrapper


def _scene():
    means = jnp.array(
        [[0.0, 0.0, 3.0], [0.2, 0.0, 4.0], [-0.2, 0.1, 2.5], [0.0, 0.0, -1.0]],
        jnp.float32,
    )
    quats = jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]], jnp.float32), (4, 1))
    scales = jnp.full((4, 3), 0.1, jnp.float32)
    opacities = jnp.full((4,), 0.5, jnp.float32)
    colors = jnp.eye(4, 3, dtype=jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.array([[[50.0, 0.0, 16.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]]])
    return means, quats, scales, opacities, colors, viewmats, Ks


def _symmetric_absgrad_scene():
    means = jnp.asarray([[0.0, 0.0, 2.0]], dtype=jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    scales = jnp.asarray([[0.3, 0.3, 0.3]], dtype=jnp.float32)
    opacities = jnp.asarray([0.8], dtype=jnp.float32)
    colors = jnp.asarray([[1.0]], dtype=jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None, ...]
    Ks = jnp.asarray(
        [[[4.0, 0.0, 1.0], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )
    return means, quats, scales, opacities, colors, viewmats, Ks


def _assert_global_packed_metadata(dense_info, packed_info):
    dense_valid = np.asarray(dense_info["valid"])
    camera_count, gaussian_count = dense_valid.shape[-2:]
    batch_count = int(np.prod(dense_valid.shape[:-2], dtype=np.int64)) or 1
    dense_valid = dense_valid.reshape(
        batch_count, camera_count, gaussian_count
    )
    capacity = batch_count * camera_count * gaussian_count
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

    dense_flatten_ids = np.asarray(dense_info["flatten_ids"]).reshape(
        batch_count, -1
    )
    dense_isect_ids = np.asarray(dense_info["isect_ids"]).reshape(
        batch_count, dense_flatten_ids.shape[1], 2
    )
    dense_isect_counts = np.asarray(
        dense_info["isect_valid_count"]
    ).reshape(batch_count)
    dense_to_packed = np.full((capacity,), -1, dtype=np.int32)
    dense_to_packed[selected] = np.arange(valid_count, dtype=np.int32)
    tile_height, tile_width = np.asarray(dense_info["isect_offsets"]).shape[-2:]
    tile_count = tile_height * tile_width
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
        local_high_words = local_words[:, 0].view(np.uint32)
        tile_ids = local_high_words & tile_mask
        camera_ids = local_flatten_ids // gaussian_count
        global_image_ids = batch_id * camera_count + camera_ids
        global_high_words = (
            global_image_ids.astype(np.uint32) << np.uint32(tile_bits)
        ) | tile_ids
        expected_isect_ids.extend(
            np.stack((global_high_words.view(np.int32), local_words[:, 1]), axis=-1)
        )

    isect_valid_count = len(expected_flatten_ids)
    isect_capacity = batch_count * dense_flatten_ids.shape[1]
    assert int(packed_info["isect_valid_count"]) == isect_valid_count
    assert packed_info["flatten_ids"].shape == (isect_capacity,)
    assert packed_info["isect_ids"].shape == (isect_capacity, 2)
    np.testing.assert_array_equal(
        np.asarray(packed_info["flatten_ids"][:isect_valid_count]),
        expected_flatten_ids,
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["isect_ids"][:isect_valid_count]),
        expected_isect_ids,
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["flatten_ids"])[isect_valid_count:], -1
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["isect_ids"])[isect_valid_count:], -1
    )

    dense_offsets = np.asarray(dense_info["isect_offsets"]).reshape(
        batch_count, camera_count, tile_height, tile_width
    )
    count_bases = np.concatenate(
        (
            np.zeros((1,), dtype=np.int32),
            np.cumsum(dense_isect_counts[:-1], dtype=np.int32),
        )
    )
    expected_offsets = dense_offsets + count_bases[:, None, None, None]
    np.testing.assert_array_equal(
        np.asarray(packed_info["isect_offsets"]).reshape(
            batch_count, camera_count, tile_height, tile_width
        ),
        expected_offsets,
    )


def test_rasterization_shapes_depth_modes_and_gradients():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    config = RasterizationConfig(
        tile_size=8, max_gaussians_per_tile=4, tile_batch_size=2
    )

    def objective(current_means):
        render, alpha, info = rasterization(
            current_means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            32,
            32,
            active_mask=jnp.array([True, True, True, False]),
            render_mode="RGB+ED",
            config=config,
        )
        return render.sum(), (render, alpha, info["tile_overflow"])

    (loss, (render, alpha, overflow)), grad = jax.value_and_grad(
        objective, has_aux=True
    )(means)
    assert render.shape == (1, 32, 32, 4)
    assert alpha.shape == (1, 32, 32, 1)
    assert overflow.shape == (1, 4, 4)
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(grad))


def test_reference_chunk_size_does_not_truncate_tile_candidates():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()

    def render(chunk_size):
        return rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            32,
            32,
            active_mask=jnp.asarray([True, True, True, False]),
            config=RasterizationConfig(
                backend="reference",
                tile_size=8,
                max_gaussians_per_tile=chunk_size,
                tile_batch_size=1,
            ),
        )

    chunked_render, chunked_alpha, chunked_info = render(1)
    full_render, full_alpha, _ = render(4)

    assert not jnp.any(chunked_info["tile_overflow"])
    assert jnp.any(chunked_info["candidate_limit_exceeded"])
    assert jnp.allclose(chunked_render, full_render)
    assert jnp.allclose(chunked_alpha, full_alpha)


def test_current_main_metadata_includes_image_and_projected_opacity_fields():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    _, _, info = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        16,
        8,
        packed=False,
        active_mask=jnp.asarray([True, True, True, False]),
        config=RasterizationConfig(
            backend="intersections",
            tile_size=8,
            max_gaussians_per_tile=4,
            max_intersections=32,
        ),
    )

    assert int(info["width"]) == 16
    assert int(info["height"]) == 8
    assert int(info["tile_size"]) == 8
    assert int(info["tile_width"]) == 2
    assert int(info["tile_height"]) == 1
    assert int(info["n_batches"]) == 1
    assert int(info["n_cameras"]) == 1
    assert info["opacities"].shape == (1, 4)
    assert float(info["opacities"][0, 3]) == 0.0


def test_dense_metadata_reuses_multicamera_tile_intersections():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    second_viewmat = viewmats[0].at[0, 3].set(0.1)
    viewmats = jnp.stack((viewmats[0], second_viewmat))
    Ks = jnp.broadcast_to(Ks, (2, 3, 3))
    active_mask = jnp.asarray([True, True, True, False])
    config = RasterizationConfig(
        backend="intersections",
        tile_size=8,
        max_gaussians_per_tile=4,
        max_intersections=64,
    )

    rendered, alphas, info = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        32,
        32,
        packed=False,
        active_mask=active_mask,
        config=config,
    )

    assert rendered.shape == (2, 32, 32, 3)
    assert alphas.shape == (2, 32, 32, 1)
    assert info["batch_ids"] is None
    assert info["camera_ids"] is None
    assert info["gaussian_ids"] is None
    assert info["isect_ids"].shape == (128, 2)
    assert info["isect_ids"].dtype == jnp.int32
    assert info["flatten_ids"].shape == (128,)
    assert info["flatten_ids"].dtype == jnp.int32
    assert info["isect_offsets"].shape == (2, 4, 4)
    assert info["isect_offsets"].dtype == jnp.int32
    assert info["tiles_per_gauss"].shape == (2, 4)
    assert info["tiles_per_gauss"].dtype == jnp.int32
    assert info["isect_valid_count"].shape == ()
    assert info["isect_valid_count"].dtype == jnp.int32

    expected_words = []
    expected_flatten_ids = []
    expected_offsets = []
    expected_tiles_per_gauss = np.zeros((2, 4), dtype=np.int32)
    count_base = 0
    tile_bits = 4
    projected_opacities = info["opacities"]
    for camera_id in range(2):
        intersections = intersect_tiles(
            info["means2d"][camera_id],
            info["radii"][camera_id],
            info["depths"][camera_id],
            info["valid"][camera_id] & (projected_opacities[camera_id] > 0.0),
            tile_size=8,
            tile_width=4,
            tile_height=4,
            max_intersections=64,
            backend=config.intersection_backend,
            sort_backend=config.sort_backend,
            conics=info["conics"][camera_id],
            opacities=projected_opacities[camera_id],
            alpha_threshold=config.alpha_clip,
            mode=config.intersection_mode,
        )
        valid_count = int(intersections.valid_count)
        gaussian_ids = np.asarray(intersections.gaussian_ids[:valid_count])
        tile_ids = np.asarray(intersections.tile_ids[:valid_count])
        depths = np.asarray(info["depths"][camera_id, gaussian_ids], np.float32)
        high_words = ((camera_id << tile_bits) | tile_ids).astype(np.uint32).view(
            np.int32
        )
        depth_words = depths.view(np.int32)
        expected_words.extend(np.stack((high_words, depth_words), axis=-1))
        expected_flatten_ids.extend(camera_id * means.shape[0] + gaussian_ids)
        expected_offsets.append(np.asarray(intersections.offsets) + count_base)
        np.add.at(expected_tiles_per_gauss[camera_id], gaussian_ids, 1)
        count_base += valid_count

    valid_count = int(info["isect_valid_count"])
    assert valid_count == count_base
    np.testing.assert_array_equal(
        np.asarray(info["isect_ids"][:valid_count]), np.asarray(expected_words)
    )
    np.testing.assert_array_equal(
        np.asarray(info["flatten_ids"][:valid_count]),
        np.asarray(expected_flatten_ids),
    )
    np.testing.assert_array_equal(
        np.asarray(info["isect_ids"])[valid_count:], -1
    )
    np.testing.assert_array_equal(
        np.asarray(info["flatten_ids"])[valid_count:], -1
    )
    np.testing.assert_array_equal(
        np.asarray(info["isect_offsets"]), np.asarray(expected_offsets)
    )
    np.testing.assert_array_equal(
        np.asarray(info["tiles_per_gauss"]), expected_tiles_per_gauss
    )


def test_packed_metadata_is_a_stable_projection_prefix_and_remaps_intersections():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    second_viewmat = viewmats[0].at[0, 3].set(0.1)
    viewmats = jnp.stack((viewmats[0], second_viewmat))
    Ks = jnp.broadcast_to(Ks, (2, 3, 3))
    active_mask = jnp.asarray([True, True, True, False])
    config = RasterizationConfig(
        backend="intersections",
        tile_size=8,
        max_gaussians_per_tile=4,
        max_intersections=64,
    )

    def render(current_means, *, packed):
        return rasterization(
            current_means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            32,
            32,
            packed=packed,
            active_mask=active_mask,
            config=config,
        )

    dense_render, dense_alpha, dense_info = render(means, packed=False)
    packed_render, packed_alpha, packed_info = render(means, packed=True)

    np.testing.assert_allclose(packed_render, dense_render, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(packed_alpha, dense_alpha, rtol=0.0, atol=0.0)

    camera_count = viewmats.shape[0]
    gaussian_count = means.shape[0]
    capacity = camera_count * gaussian_count
    dense_valid = np.asarray(dense_info["valid"]).reshape(-1)
    selected = np.flatnonzero(dense_valid)
    valid_count = selected.shape[0]

    assert packed_info["projection_valid_count"].shape == ()
    assert packed_info["projection_valid_count"].dtype == jnp.int32
    assert int(packed_info["projection_valid_count"]) == valid_count
    assert packed_info["projection_capacity"].shape == ()
    assert packed_info["projection_capacity"].dtype == jnp.int32
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
    expected_indptr = np.concatenate(
        (
            np.zeros((1,), np.int32),
            np.cumsum(
                dense_valid.reshape(camera_count, gaussian_count).sum(axis=1),
                dtype=np.int32,
            ),
        )
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["indptr"]), expected_indptr
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
        "conics": (capacity, 3),
        "opacities": (capacity,),
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
    assert packed_info["radii"].dtype == jnp.int32
    assert packed_info["tiles_per_gauss"].dtype == jnp.int32
    for key in ("means2d", "depths", "conics", "opacities"):
        assert packed_info[key].dtype == jnp.float32

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

    dense_gradient = jax.grad(lambda value: render(value, packed=False)[0].sum())(
        means
    )
    packed_gradient = jax.grad(lambda value: render(value, packed=True)[0].sum())(
        means
    )
    np.testing.assert_allclose(
        packed_gradient, dense_gradient, rtol=1.0e-6, atol=1.0e-6
    )


def test_reference_packed_metadata_uses_the_rendered_tile_candidates():
    means = jnp.asarray(
        [
            [0.0, 0.0, 2.0],
            [0.15, 0.0, 2.5],
            [-0.2, 0.1, 3.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=jnp.float32,
    )
    quats = jnp.zeros((4, 4), dtype=jnp.float32).at[:, 0].set(1.0)
    scales = jnp.full((4, 3), 0.1, dtype=jnp.float32)
    opacities = jnp.asarray([0.7, 0.5, 0.8, 0.9], dtype=jnp.float32)
    colors = jnp.asarray(
        [
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.2],
            [0.1, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=jnp.float32,
    )
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.asarray(
        [[[16.0, 0.0, 4.0], [0.0, 16.0, 4.0], [0.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )
    active_mask = jnp.asarray([True, True, True, False])
    config = RasterizationConfig(
        backend="reference",
        tile_size=4,
        max_gaussians_per_tile=2,
        tile_batch_size=1,
    )

    def render(current_means, current_opacities, *, packed):
        return rasterization(
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
            config=config,
        )

    dense_render, dense_alpha, dense_info = render(
        means, opacities, packed=False
    )
    packed_render, packed_alpha, packed_info = render(
        means, opacities, packed=True
    )

    np.testing.assert_array_equal(packed_render, dense_render)
    np.testing.assert_array_equal(packed_alpha, dense_alpha)
    _assert_global_packed_metadata(dense_info, packed_info)

    projected_means = np.asarray(dense_info["means2d"])[0]
    projected_radii = np.asarray(dense_info["radii"])[0]
    projected_depths = np.asarray(dense_info["depths"])[0]
    projected_opacities = np.asarray(dense_info["opacities"])[0]
    projected_valid = np.asarray(dense_info["valid"])[0]
    expected_gaussian_ids = []
    expected_offsets = []
    for tile_id in range(4):
        tile_x = tile_id % 2
        tile_y = tile_id // 2
        x0, y0 = tile_x * 4, tile_y * 4
        overlaps = (
            projected_valid
            & (projected_means[:, 0] + projected_radii[:, 0] > x0)
            & (projected_means[:, 0] - projected_radii[:, 0] < x0 + 4)
            & (projected_means[:, 1] + projected_radii[:, 1] > y0)
            & (projected_means[:, 1] - projected_radii[:, 1] < y0 + 4)
            & (projected_opacities > 0.0)
        )
        tile_gaussian_ids = np.flatnonzero(overlaps)
        tile_gaussian_ids = tile_gaussian_ids[
            np.argsort(projected_depths[tile_gaussian_ids], kind="stable")
        ]
        expected_offsets.append(len(expected_gaussian_ids))
        expected_gaussian_ids.extend(tile_gaussian_ids)

    intersection_count = int(dense_info["isect_valid_count"])
    assert intersection_count == len(expected_gaussian_ids)
    np.testing.assert_array_equal(
        np.asarray(dense_info["flatten_ids"][:intersection_count]),
        expected_gaussian_ids,
    )
    np.testing.assert_array_equal(
        np.asarray(dense_info["isect_offsets"]).reshape(-1),
        expected_offsets,
    )

    def objective(current_means, current_opacities, *, packed):
        rendered, alpha, _ = render(
            current_means, current_opacities, packed=packed
        )
        return jnp.sum(rendered) + 0.1 * jnp.sum(alpha)

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


def test_eval3d_leading_batch_packed_metadata_remaps_visible_candidates():
    batch_count, camera_count, gaussian_count = 2, 1, 3
    means = jnp.asarray(
        [
            [[-0.1, 0.0, 2.4], [0.0, 0.0, 2.0], [0.1, 0.0, 2.8]],
            [[0.0, 0.0, 2.1], [0.1, 0.0, 2.6], [-0.1, 0.0, 2.9]],
        ],
        dtype=jnp.float32,
    )
    quats = jnp.zeros(
        (batch_count, gaussian_count, 4), dtype=jnp.float32
    ).at[..., 0].set(1.0)
    scales = jnp.full(
        (batch_count, gaussian_count, 3), 0.1, dtype=jnp.float32
    )
    opacities = jnp.asarray(
        [[0.4, 0.8, 0.5], [0.7, 0.6, 0.9]], dtype=jnp.float32
    )
    colors = jnp.asarray(
        [
            [[0.4, 0.1, 0.0], [1.0, 0.0, 0.1], [0.1, 0.4, 0.8]],
            [[0.2, 0.9, 0.1], [0.6, 0.2, 0.1], [0.1, 0.2, 1.0]],
        ],
        dtype=jnp.float32,
    )
    viewmats = jnp.broadcast_to(
        jnp.eye(4, dtype=jnp.float32),
        (batch_count, camera_count, 4, 4),
    )
    K = jnp.asarray(
        [[8.0, 0.0, 2.0], [0.0, 8.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    Ks = jnp.broadcast_to(K, (batch_count, camera_count, 3, 3))
    active_mask = jnp.asarray(
        [[False, True, False], [True, False, True]], dtype=jnp.bool_
    )
    config = RasterizationConfig(
        backend="intersections",
        tile_size=4,
        max_gaussians_per_tile=2,
        max_intersections=2,
        tile_batch_size=1,
        ut_chunk_size=1,
    )

    def render(current_means, current_opacities, *, packed):
        return rasterization(
            current_means,
            quats,
            scales,
            current_opacities,
            colors,
            viewmats,
            Ks,
            4,
            4,
            packed=packed,
            active_mask=active_mask,
            with_eval3d=True,
            config=config,
        )

    dense_render, dense_alpha, dense_info = render(
        means, opacities, packed=False
    )
    packed_render, packed_alpha, packed_info = render(
        means, opacities, packed=True
    )

    np.testing.assert_array_equal(packed_render, dense_render)
    np.testing.assert_array_equal(packed_alpha, dense_alpha)
    _assert_global_packed_metadata(dense_info, packed_info)

    for batch_id in range(batch_count):
        intersections = intersect_tiles(
            dense_info["means2d"][batch_id, 0],
            dense_info["radii"][batch_id, 0],
            dense_info["depths"][batch_id, 0],
            dense_info["valid"][batch_id, 0]
            & (dense_info["opacities"][batch_id, 0] > 0.0),
            tile_size=4,
            tile_width=1,
            tile_height=1,
            max_intersections=2,
            backend=config.intersection_backend,
            sort_backend=config.sort_backend,
            mode="aabb",
        )
        intersection_count = int(intersections.valid_count)
        assert int(dense_info["isect_valid_count"][batch_id]) == intersection_count
        np.testing.assert_array_equal(
            np.asarray(
                dense_info["flatten_ids"][batch_id, :intersection_count]
            ),
            np.asarray(intersections.gaussian_ids[:intersection_count]),
        )
        np.testing.assert_array_equal(
            np.asarray(dense_info["isect_offsets"][batch_id, 0]),
            np.asarray(intersections.offsets),
        )

    def objective(current_means, current_opacities, *, packed):
        rendered, alpha, _ = render(
            current_means, current_opacities, packed=packed
        )
        return jnp.sum(rendered) + 0.1 * jnp.sum(alpha)

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


def test_leading_batch_packed_metadata_is_one_global_stable_prefix():
    batch_count, camera_count, gaussian_count = 2, 2, 3
    means = jnp.asarray(
        [
            [[0.0, 0.0, 3.0], [0.2, 0.0, 3.5], [-0.2, 0.1, 2.5]],
            [[0.1, 0.0, 2.8], [-0.15, 0.05, 3.2], [0.25, -0.1, 3.8]],
        ],
        dtype=jnp.float32,
    )
    quats = jnp.zeros((batch_count, gaussian_count, 4), jnp.float32)
    quats = quats.at[..., 0].set(1.0)
    scales = jnp.full((batch_count, gaussian_count, 3), 0.1, jnp.float32)
    opacities = jnp.asarray(
        [[0.7, 0.5, 0.6], [0.4, 0.8, 0.55]], dtype=jnp.float32
    )
    colors = jnp.asarray(
        [
            [[1.0, 0.1, 0.2], [0.2, 1.0, 0.1], [0.1, 0.2, 1.0]],
            [[0.8, 0.2, 0.1], [0.1, 0.7, 0.3], [0.3, 0.1, 0.9]],
        ],
        dtype=jnp.float32,
    )
    second_camera = jnp.eye(4, dtype=jnp.float32).at[0, 3].set(0.1)
    cameras = jnp.stack((jnp.eye(4, dtype=jnp.float32), second_camera))
    viewmats = jnp.broadcast_to(cameras, (batch_count, camera_count, 4, 4))
    K = jnp.asarray(
        [[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    Ks = jnp.broadcast_to(K, (batch_count, camera_count, 3, 3))
    active_mask = jnp.asarray(
        [[True, False, True], [False, True, True]], dtype=jnp.bool_
    )
    config = RasterizationConfig(
        backend="intersections",
        tile_size=8,
        max_gaussians_per_tile=3,
        max_intersections=64,
        tile_batch_size=1,
    )

    def render(current_means, current_opacities, *, packed):
        return rasterization(
            current_means,
            quats,
            scales,
            current_opacities,
            colors,
            viewmats,
            Ks,
            16,
            16,
            packed=packed,
            active_mask=active_mask,
            config=config,
        )

    dense_render, dense_alpha, dense_info = render(
        means, opacities, packed=False
    )
    packed_render, packed_alpha, packed_info = render(
        means, opacities, packed=True
    )

    np.testing.assert_allclose(packed_render, dense_render, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(packed_alpha, dense_alpha, rtol=0.0, atol=0.0)

    capacity = batch_count * camera_count * gaussian_count
    dense_valid = np.asarray(dense_info["valid"]).reshape(
        batch_count, camera_count, gaussian_count
    )
    selected = np.flatnonzero(dense_valid.reshape(-1))
    valid_count = selected.shape[0]
    expected_batch_ids = selected // (camera_count * gaussian_count)
    selected_within_batch = selected % (camera_count * gaussian_count)
    expected_camera_ids = selected_within_batch // gaussian_count
    expected_gaussian_ids = selected_within_batch % gaussian_count

    assert bool(packed_info["packed_requested"])
    assert bool(packed_info["packed_metadata_available"])
    assert int(packed_info["n_batches"]) == batch_count
    assert int(packed_info["n_cameras"]) == camera_count
    assert int(packed_info["projection_capacity"]) == capacity
    assert int(packed_info["projection_valid_count"]) == valid_count
    np.testing.assert_array_equal(
        np.asarray(packed_info["valid"]),
        np.arange(capacity) < valid_count,
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

    counts_per_image = dense_valid.sum(axis=-1).reshape(-1)
    expected_indptr = np.concatenate(
        (np.zeros((1,), np.int32), np.cumsum(counts_per_image, dtype=np.int32))
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["indptr"]), expected_indptr
    )

    projection_shapes = {
        "radii": (capacity, 2),
        "means2d": (capacity, 2),
        "depths": (capacity,),
        "conics": (capacity, 3),
        "opacities": (capacity,),
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

        reconstructed = np.zeros_like(dense_values).reshape(
            (batch_count, camera_count, gaussian_count) + shape[1:]
        )
        reconstructed[
            expected_batch_ids, expected_camera_ids, expected_gaussian_ids
        ] = packed_values[:valid_count]
        dense_projection = dense_values.reshape(reconstructed.shape)
        projection_mask = dense_valid.reshape(
            dense_valid.shape + (1,) * len(shape[1:])
        )
        np.testing.assert_array_equal(
            reconstructed, np.where(projection_mask, dense_projection, 0)
        )

    per_batch_isect_capacity = dense_info["flatten_ids"].shape[-1]
    global_isect_capacity = batch_count * per_batch_isect_capacity
    dense_isect_counts = np.asarray(dense_info["isect_valid_count"]).reshape(-1)
    expected_packed_flatten_ids = []
    dense_to_packed = np.full((capacity,), -1, dtype=np.int32)
    dense_to_packed[selected] = np.arange(valid_count, dtype=np.int32)
    for batch_id, count in enumerate(dense_isect_counts):
        local_dense_ids = np.asarray(dense_info["flatten_ids"])[
            batch_id, :count
        ]
        expected_packed_flatten_ids.extend(
            dense_to_packed[
                batch_id * camera_count * gaussian_count + local_dense_ids
            ]
        )
    global_isect_count = len(expected_packed_flatten_ids)
    assert packed_info["flatten_ids"].shape == (global_isect_capacity,)
    assert packed_info["isect_ids"].shape == (global_isect_capacity, 2)
    assert int(packed_info["isect_valid_count"]) == global_isect_count
    np.testing.assert_array_equal(
        np.asarray(packed_info["flatten_ids"][:global_isect_count]),
        expected_packed_flatten_ids,
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["flatten_ids"])[global_isect_count:], -1
    )
    np.testing.assert_array_equal(
        np.asarray(packed_info["isect_ids"])[global_isect_count:], -1
    )

    packed_flatten_ids = np.asarray(
        packed_info["flatten_ids"][:global_isect_count]
    )
    high_words = np.asarray(
        packed_info["isect_ids"][:global_isect_count, 0], dtype=np.int32
    ).view(np.uint32)
    tile_count = int(packed_info["tile_width"]) * int(
        packed_info["tile_height"]
    )
    tile_bits = max(1, (tile_count - 1).bit_length())
    decoded_image_ids = high_words >> np.uint32(tile_bits)
    expected_image_ids = (
        np.asarray(packed_info["batch_ids"])[packed_flatten_ids] * camera_count
        + np.asarray(packed_info["camera_ids"])[packed_flatten_ids]
    )
    np.testing.assert_array_equal(decoded_image_ids, expected_image_ids)

    dense_offsets = np.asarray(dense_info["isect_offsets"])
    isect_bases = np.concatenate(
        (
            np.zeros((1,), np.int32),
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
        rendered, alphas, _ = render(
            current_means, current_opacities, packed=packed
        )
        return jnp.sum(rendered) + 0.1 * jnp.sum(alphas)

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


def test_zero_means2d_offset_preserves_forward_and_model_gradient():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    second_viewmat = viewmats[0].at[0, 3].set(0.1)
    viewmats = jnp.stack((viewmats[0], second_viewmat))
    Ks = jnp.broadcast_to(Ks, (2, 3, 3))
    active_mask = jnp.asarray([True, True, True, False])
    config = RasterizationConfig(
        backend="intersections",
        tile_size=8,
        max_gaussians_per_tile=4,
        max_intersections=64,
    )
    pixel_weights = jnp.arange(32, dtype=jnp.float32)[None, None, :, None]

    def objective(current_means, offset):
        rendered, _, _ = rasterization(
            current_means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            32,
            32,
            packed=False,
            active_mask=active_mask,
            config=config,
            _means2d_offset=offset,
        )
        return jnp.sum(rendered * pixel_weights), rendered

    baseline_value, baseline_model_grad = jax.value_and_grad(
        lambda current_means: objective(current_means, None)[0]
    )(means)
    offset = jnp.zeros((2, means.shape[0], 2), dtype=means.dtype)
    (probed_value, probed_render), (model_grad, screen_grad) = (
        jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(
            means, offset
        )
    )

    np.testing.assert_allclose(probed_value, baseline_value, rtol=0.0, atol=0.0)
    # The probe reaches the same gradient by a different summation order, so
    # the absolute floor has to match the accumulation rather than the element.
    # These gradients reach a few hundred but include entries near zero, and a
    # 32x32 weighted reduction lands about 30 float32 ulps from the direct
    # path on those; in float64 the two agree to 5.5e-16. Which side rounds
    # where also moves with XLA's fusion choices, so a tighter floor only
    # passes by luck.
    np.testing.assert_allclose(
        model_grad, baseline_model_grad, rtol=1.0e-6, atol=1.0e-4
    )
    assert probed_render.shape == (2, 32, 32, 3)
    assert screen_grad.shape == offset.shape
    assert jnp.all(jnp.isfinite(screen_grad))
    assert jnp.any(jnp.abs(screen_grad) > 0.0)


@pytest.mark.parametrize("backend", ["reference", "intersections"])
def test_high_level_absgrad_probe_sums_before_symmetric_pixel_cancellation(
    backend,
):
    means, quats, scales, opacities, colors, viewmats, Ks = (
        _symmetric_absgrad_scene()
    )
    zero_offset = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    zero_probe = jnp.zeros_like(zero_offset)
    config = RasterizationConfig(
        backend=backend,
        intersection_backend="jax",
        sort_backend="jax",
        intersection_mode="aabb",
        tile_size=2,
        max_gaussians_per_tile=1,
        max_intersections=1,
        tile_batch_size=1,
    )
    packed_results = []

    for packed in (False, True):
        def render(current_means, offset, probe):
            return rasterization(
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
                _means2d_offset=offset,
                **(
                    {}
                    if probe is None
                    else {"_means2d_absgrad_probe": probe}
                ),
            )

        def objective(current_means, offset, probe):
            rendered, alphas, _ = render(current_means, offset, probe)
            return jnp.sum(rendered) + 0.25 * jnp.sum(alphas), (
                rendered,
                alphas,
            )

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


def test_high_level_absgrad_probe_leading_batch_slices_and_jits():
    means, quats, scales, opacities, colors, viewmats, Ks = (
        _symmetric_absgrad_scene()
    )
    batch_shape = (1, 2)
    batched_means = jnp.broadcast_to(means, batch_shape + means.shape)
    batched_quats = jnp.broadcast_to(quats, batch_shape + quats.shape)
    batched_scales = jnp.broadcast_to(scales, batch_shape + scales.shape)
    batched_opacities = jnp.broadcast_to(
        opacities, batch_shape + opacities.shape
    )
    batched_colors = jnp.broadcast_to(colors, batch_shape + colors.shape)
    batched_viewmats = jnp.broadcast_to(
        viewmats, batch_shape + viewmats.shape
    )
    batched_Ks = jnp.broadcast_to(Ks, batch_shape + Ks.shape)
    zero_probes = jnp.zeros(batch_shape + (1, 1, 2), dtype=jnp.float32)
    batch_weights = jnp.asarray([1.0, 2.0], dtype=jnp.float32).reshape(
        1, 2, 1, 1, 1, 1
    )
    config = RasterizationConfig(
        backend="intersections",
        intersection_backend="jax",
        sort_backend="jax",
        intersection_mode="aabb",
        tile_size=2,
        max_gaussians_per_tile=1,
        max_intersections=1,
        tile_batch_size=1,
    )

    def batched_loss(current_means, probes):
        rendered, _, _ = rasterization(
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
            _means2d_absgrad_probe=probes,
        )
        return jnp.sum(rendered * batch_weights)

    batched_gradients = jax.jit(
        jax.grad(batched_loss, argnums=(0, 1))
    )(batched_means, zero_probes)

    def single_loss(current_means, probe, weight):
        rendered, _, _ = rasterization(
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
            _means2d_absgrad_probe=probe,
        )
        return weight * jnp.sum(rendered)

    single_gradients = jax.jit(
        jax.grad(single_loss, argnums=(0, 1))
    )
    expected = [
        single_gradients(means, zero_probes[0, index], index + 1.0)
        for index in range(2)
    ]
    for gradient_index in range(2):
        np.testing.assert_allclose(
            batched_gradients[gradient_index][0],
            jnp.stack([value[gradient_index] for value in expected]),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    assert jnp.all(batched_gradients[1][0, :, 0, 0, 0] > 0.0)


def test_high_level_absgrad_probe_rejects_unsupported_combinations():
    means, quats, scales, opacities, colors, viewmats, Ks = (
        _symmetric_absgrad_scene()
    )
    probe = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    arguments = (means, quats, scales, opacities, colors, viewmats, Ks, 2, 1)

    with pytest.raises(ValueError, match="requires absgrad=True"):
        rasterization(*arguments, _means2d_absgrad_probe=probe)
    with pytest.raises(ValueError, match="with_eval3d"):
        rasterization(
            *arguments,
            absgrad=True,
            with_eval3d=True,
            _means2d_absgrad_probe=probe,
        )
    with pytest.raises(NotImplementedError, match="distributed.*absgrad"):
        rasterization(*arguments, absgrad=True, distributed=True)


def test_sparse_grad_rejects_implicit_ftheta_ut_path():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()

    with pytest.raises(ValueError, match="sparse_grad.*ftheta"):
        rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            32,
            32,
            packed=True,
            sparse_grad=True,
            camera_model="ftheta",
        )


def test_active_mask_value_change_reuses_static_jit_signature():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    config = RasterizationConfig(
        tile_size=8, max_gaussians_per_tile=4, tile_batch_size=2
    )

    @jax.jit
    def render(mask):
        return rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            16,
            16,
            packed=False,
            active_mask=mask,
            config=config,
        )[0]

    one = render(jnp.array([True, False, False, False]))
    cache_size = render._cache_size()
    two = render(jnp.array([True, True, False, False]))
    assert render._cache_size() == cache_size == 1
    assert not jnp.allclose(one, two)


def test_leading_batch_dims_match_individual_calls_and_are_jittable():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    batched_means = jnp.stack((means, means.at[0, 0].set(0.1))).reshape(
        1, 2, 4, 3
    )
    batched_quats = jnp.broadcast_to(quats, (1, 2) + quats.shape)
    batched_scales = jnp.broadcast_to(scales, (1, 2) + scales.shape)
    batched_opacities = jnp.broadcast_to(opacities, (1, 2) + opacities.shape)
    batched_colors = jnp.broadcast_to(colors, (1, 2) + colors.shape)
    batched_viewmats = jnp.broadcast_to(viewmats, (1, 2) + viewmats.shape)
    batched_Ks = jnp.broadcast_to(Ks, (1, 2) + Ks.shape)
    mask = jnp.asarray([True, True, True, False])
    config = RasterizationConfig(
        backend="intersections",
        tile_size=8,
        max_gaussians_per_tile=4,
        max_intersections=64,
        tile_batch_size=1,
    )

    @jax.jit
    def render(current_means):
        return rasterization(
            current_means,
            batched_quats,
            batched_scales,
            batched_opacities,
            batched_colors,
            batched_viewmats,
            batched_Ks,
            16,
            16,
            packed=False,
            active_mask=mask,
            backgrounds=jnp.asarray([[0.1, 0.2, 0.3]], jnp.float32),
            config=config,
            _means2d_offset=jnp.zeros((1, 2, 1, 4, 2), jnp.float32),
        )

    rendered, alphas, info = render(batched_means)
    expected = [
        rasterization(
            batched_means[0, index],
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            16,
            16,
            active_mask=mask,
            backgrounds=jnp.asarray([[0.1, 0.2, 0.3]], jnp.float32),
            config=config,
        )
        for index in range(2)
    ]

    assert rendered.shape == (1, 2, 1, 16, 16, 3)
    assert alphas.shape == (1, 2, 1, 16, 16, 1)
    assert info["valid"].shape == (1, 2, 1, 4)
    assert info["active_count"].shape == (1, 2)
    assert not jnp.any(info["packed_metadata_available"])
    assert jnp.allclose(rendered[0], jnp.stack([value[0] for value in expected]))
    assert jnp.allclose(alphas[0], jnp.stack([value[1] for value in expected]))

    gradient = jax.jit(jax.grad(lambda value: render(value)[0].sum()))(
        batched_means
    )
    assert gradient.shape == batched_means.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_per_camera_sh_coefficients_match_separate_camera_calls():
    means, quats, scales, opacities, _, viewmats, Ks = _scene()
    viewmats = jnp.concatenate((viewmats, viewmats), axis=0)
    viewmats = viewmats.at[1, 0, 3].set(0.1)
    Ks = jnp.concatenate((Ks, Ks), axis=0)
    sh = jnp.stack(
        (
            jnp.full((4, 1, 3), 0.1, jnp.float32),
            jnp.full((4, 1, 3), 0.3, jnp.float32),
        )
    )
    mask = jnp.asarray([True, True, True, False])
    config = RasterizationConfig(
        backend="intersections",
        tile_size=8,
        max_gaussians_per_tile=4,
        max_intersections=64,
        tile_batch_size=1,
    )

    rendered, alphas, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        sh,
        viewmats,
        Ks,
        16,
        16,
        active_mask=mask,
        sh_degree=0,
        config=config,
    )
    separate = [
        rasterization(
            means,
            quats,
            scales,
            opacities,
            sh[index],
            viewmats[index : index + 1],
            Ks[index : index + 1],
            16,
            16,
            active_mask=mask,
            sh_degree=0,
            config=config,
        )
        for index in range(2)
    ]

    assert jnp.allclose(rendered, jnp.concatenate([value[0] for value in separate]))
    assert jnp.allclose(alphas, jnp.concatenate([value[1] for value in separate]))


def test_extra_signals_support_leading_batches_and_match_separate_calls():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    batched_means = jnp.stack((means, means.at[0, 0].set(0.1)))
    batched_quats = jnp.broadcast_to(quats, (2,) + quats.shape)
    batched_scales = jnp.broadcast_to(scales, (2,) + scales.shape)
    batched_opacities = jnp.broadcast_to(opacities, (2,) + opacities.shape)
    batched_colors = jnp.broadcast_to(colors, (2,) + colors.shape)
    batched_viewmats = jnp.broadcast_to(viewmats, (2,) + viewmats.shape)
    batched_Ks = jnp.broadcast_to(Ks, (2,) + Ks.shape)
    extras = jnp.arange(16, dtype=jnp.float32).reshape(2, 4, 2) * 0.1
    config = RasterizationConfig(
        backend="intersections",
        tile_size=8,
        max_gaussians_per_tile=4,
        max_intersections=32,
        tile_batch_size=1,
    )

    @jax.jit
    def render(current_extras):
        return rasterization(
            batched_means,
            batched_quats,
            batched_scales,
            batched_opacities,
            batched_colors,
            batched_viewmats,
            batched_Ks,
            8,
            8,
            extra_signals=current_extras,
            config=config,
        )

    rendered, alphas, info = render(extras)
    separate = [
        rasterization(
            batched_means[index],
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            8,
            8,
            extra_signals=extras[index],
            config=config,
        )
        for index in range(2)
    ]

    assert rendered.shape == (2, 1, 8, 8, 3)
    assert alphas.shape == (2, 1, 8, 8, 1)
    assert info["render_extra_signals"].shape == (2, 1, 8, 8, 2)
    assert jnp.allclose(rendered, jnp.stack([value[0] for value in separate]))
    assert jnp.allclose(alphas, jnp.stack([value[1] for value in separate]))
    assert jnp.allclose(
        info["render_extra_signals"],
        jnp.stack([value[2]["render_extra_signals"] for value in separate]),
    )


def test_intersection_backend_matches_reference_values_and_gradients():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    mask = jnp.array([True, True, True, False])

    def objective(current_means, backend):
        render, alpha, info = rasterization(
            current_means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            32,
            32,
            active_mask=mask,
            backgrounds=jnp.array([[0.1, 0.2, 0.3]], dtype=jnp.float32),
            render_mode="RGB+ED",
            config=RasterizationConfig(
                backend=backend,
                tile_size=8,
                max_gaussians_per_tile=4,
                max_intersections=64,
                tile_batch_size=2,
            ),
        )
        return render.sum() + alpha.sum(), (render, alpha, info)

    (fast_loss, (fast_render, fast_alpha, fast_info)), fast_grad = (
        jax.value_and_grad(objective, has_aux=True)(means, "intersections")
    )
    (ref_loss, (ref_render, ref_alpha, _)), ref_grad = jax.value_and_grad(
        objective, has_aux=True
    )(means, "reference")

    assert jnp.allclose(fast_render, ref_render, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(fast_alpha, ref_alpha, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(fast_loss, ref_loss, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(fast_grad, ref_grad, rtol=2e-4, atol=2e-5)
    assert not bool(fast_info["intersection_overflow"][0])
    assert int(fast_info["intersection_count"][0]) > 0


def test_intersection_capacity_overflow_is_reported():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    _, _, info = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        32,
        32,
        active_mask=jnp.array([True, True, True, False]),
        config=RasterizationConfig(
            backend="intersections",
            tile_size=8,
            max_gaussians_per_tile=4,
            max_intersections=1,
            tile_batch_size=1,
        ),
    )
    assert bool(info["intersection_overflow"][0])
    assert int(info["intersection_count"][0]) == 1
    assert int(info["intersection_capacity"][0]) == 1
    assert bool(info["visible_overflow"][0])
    assert int(info["visible_count"][0]) == 3
    assert int(info["visible_capacity"][0]) == 1


def test_default_intersection_capacity_covers_every_gaussian_tile_pair():
    means = jnp.asarray([[0.0, 0.0, 2.0]], jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], jnp.float32)
    scales = jnp.asarray([[10.0, 10.0, 10.0]], jnp.float32)
    opacities = jnp.asarray([0.9], jnp.float32)
    colors = jnp.asarray([[1.0, 0.0, 0.0]], jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.asarray(
        [[[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )

    _, _, info = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        16,
        16,
        config=RasterizationConfig(
            backend="intersections",
            intersection_mode="aabb",
            tile_size=4,
            max_gaussians_per_tile=1,
            tile_batch_size=1,
        ),
    )

    assert int(info["intersection_capacity"][0]) == 16
    assert int(info["intersection_required_count"][0]) == 16
    assert int(info["intersection_count"][0]) == 16
    assert not bool(info["intersection_overflow"][0])


def test_visible_compaction_preserves_non_overflowing_render_and_gradients():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    padding = 8
    means = jnp.concatenate(
        (means, jnp.tile(jnp.array([[0.0, 0.0, -1.0]], jnp.float32), (padding, 1)))
    )
    quats = jnp.concatenate(
        (
            quats,
            jnp.tile(
                jnp.array([[1.0, 0.0, 0.0, 0.0]], jnp.float32),
                (padding, 1),
            ),
        )
    )
    scales = jnp.concatenate((scales, jnp.full((padding, 3), 0.1, jnp.float32)))
    opacities = jnp.concatenate((opacities, jnp.full((padding,), 0.5, jnp.float32)))
    colors = jnp.concatenate((colors, jnp.zeros((padding, 3), jnp.float32)))
    mask = jnp.arange(means.shape[0]) == 0

    def objective(current_means, max_intersections):
        render, alpha, info = rasterization(
            current_means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            32,
            32,
            active_mask=mask,
            config=RasterizationConfig(
                backend="intersections",
                tile_size=8,
                max_gaussians_per_tile=4,
                max_intersections=max_intersections,
                tile_batch_size=1,
            ),
        )
        return render.sum() + alpha.sum(), (render, alpha, info)

    (compact_loss, (compact_render, compact_alpha, compact_info)), compact_grad = (
        jax.value_and_grad(objective, has_aux=True)(means, 8)
    )
    (dense_loss, (dense_render, dense_alpha, _)), dense_grad = jax.value_and_grad(
        objective, has_aux=True
    )(means, 64)

    assert jnp.allclose(compact_render, dense_render, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(compact_alpha, dense_alpha, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(compact_loss, dense_loss, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(compact_grad, dense_grad, rtol=2e-4, atol=2e-5)
    assert int(compact_info["visible_count"][0]) == 1
    assert int(compact_info["visible_capacity"][0]) == 8
    assert not bool(compact_info["visible_overflow"][0])


def test_split_sh_matches_concatenated_sh_values_and_gradients():
    means, quats, scales, opacities, _, viewmats, Ks = _scene()
    sh0 = jnp.arange(12, dtype=jnp.float32).reshape(4, 1, 3) * 0.01
    sh_rest = jnp.arange(36, dtype=jnp.float32).reshape(4, 3, 3) * 0.001
    mask = jnp.array([True, True, True, False])
    config = RasterizationConfig(
        backend="intersections",
        tile_size=8,
        max_gaussians_per_tile=4,
        max_intersections=64,
        tile_batch_size=1,
    )

    def split_objective(current_sh0, current_sh_rest):
        render, alpha, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            (current_sh0, current_sh_rest),
            viewmats,
            Ks,
            32,
            32,
            active_mask=mask,
            sh_degree=1,
            config=config,
        )
        return render.sum() + alpha.sum(), render

    def joined_objective(current_sh0, current_sh_rest):
        render, alpha, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            jnp.concatenate((current_sh0, current_sh_rest), axis=1),
            viewmats,
            Ks,
            32,
            32,
            active_mask=mask,
            sh_degree=1,
            config=config,
        )
        return render.sum() + alpha.sum(), render

    (split_loss, split_render), split_grads = jax.value_and_grad(
        split_objective, argnums=(0, 1), has_aux=True
    )(sh0, sh_rest)
    (joined_loss, joined_render), joined_grads = jax.value_and_grad(
        joined_objective, argnums=(0, 1), has_aux=True
    )(sh0, sh_rest)

    assert jnp.allclose(split_render, joined_render, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(split_loss, joined_loss, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(split_grads[0], joined_grads[0], rtol=2e-4, atol=2e-5)
    assert jnp.allclose(split_grads[1], joined_grads[1], rtol=2e-4, atol=2e-5)


def test_high_level_radius_clip_filters_small_projected_gaussians():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    render, alpha, info = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        32,
        32,
        active_mask=jnp.array([True, True, True, False]),
        config=RasterizationConfig(
            backend="intersections",
            tile_size=8,
            max_gaussians_per_tile=4,
            max_intersections=64,
            radius_clip=100.0,
        ),
    )
    assert not jnp.any(info["valid"])
    assert not jnp.any(info["radii"])
    assert jnp.all(render == 0.0)
    assert jnp.all(alpha == 0.0)


def test_high_level_unscented_transform_path():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    render, alpha, info = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        16,
        16,
        active_mask=jnp.array([True, True, False, False]),
        with_ut=True,
        config=RasterizationConfig(
            tile_size=8,
            max_gaussians_per_tile=4,
            tile_batch_size=1,
            ut_chunk_size=2,
        ),
    )
    assert render.shape == (1, 16, 16, 3)
    assert alpha.shape == (1, 16, 16, 1)
    assert bool(info["used_unscented_transform"])
    assert jnp.all(jnp.isfinite(render))


def test_inria_wrapper_matches_upstream_signature_and_return_contract():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    config = RasterizationConfig(
        backend="intersections",
        tile_size=8,
        max_gaussians_per_tile=4,
        max_intersections=64,
        tile_batch_size=1,
    )
    wrapped, alpha, info = rasterization_inria_wrapper(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        16,
        16,
        config=config,
    )
    # The four numerical options after width/height are positional upstream.
    expected, _, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        16,
        16,
        0.01,
        100.0,
        0.0,
        0.3,
        config=config,
    )

    assert jnp.allclose(wrapped, expected)
    assert alpha is None
    assert info == {}


def _backward_temp_bytes(
    config: RasterizationConfig, width: int, height: int, **kwargs
) -> int:
    """Compile one backward pass and report its temporary memory."""

    count = 2_000
    k1, k2, k3 = jax.random.split(jax.random.key(11), 3)
    means = jax.random.normal(k1, (count, 3), jnp.float32) * 0.4 + jnp.asarray(
        [0.0, 0.0, 3.0]
    )
    quats = jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0], jnp.float32), (count, 1))
    scales = jnp.exp(
        jax.random.normal(k2, (count, 3), jnp.float32) * 0.2 - 3.0
    )
    opacities = jax.nn.sigmoid(jax.random.normal(k3, (count,), jnp.float32))
    colors = jnp.zeros((count, 1, 3), jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    intrinsics = jnp.asarray(
        [[[96.0, 0.0, width / 2], [0.0, 96.0, height / 2], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )

    def loss(m, q, s, o, c):
        rendered = rasterization(
            m, q, s, o, c, viewmats, intrinsics, width, height,
            sh_degree=0, config=config, **kwargs,
        )[0]
        return jnp.mean(rendered**2)

    compiled = (
        jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4)))
        .lower(means, quats, scales, opacities, colors)
        .compile()
    )
    try:
        return compiled.memory_analysis().temp_size_in_bytes
    except (AttributeError, NotImplementedError) as exc:  # pragma: no cover
        pytest.skip(f"memory analysis is unavailable: {exc}")


# A fixed small tile batch isolates the tile-count scaling from the batch
# workspace, which is what the default tile_batch_size sizes for the device.
@pytest.mark.parametrize(
    ("config", "kwargs"),
    [
        (
            RasterizationConfig(max_intersections=8192, tile_batch_size=4),
            {},
        ),
        (
            RasterizationConfig(
                backend="reference", max_intersections=8192, tile_batch_size=4
            ),
            {},
        ),
        (
            RasterizationConfig(max_intersections=8192, tile_batch_size=4),
            {"with_ut": True, "with_eval3d": True},
        ),
    ],
    ids=["dense", "reference", "eval3d"],
)
def test_backward_memory_does_not_grow_with_the_tile_count(config, kwargs):
    # Per-tile compositing is rematerialized, so reverse mode keeps only one
    # tile batch of [max_gaussians_per_tile, pixel] intermediates alive.
    # Without it, quadrupling the tile count multiplied this by about 3.6.
    small = _backward_temp_bytes(config, 96, 64, **kwargs)
    large = _backward_temp_bytes(config, 192, 128, **kwargs)

    assert large < 1.5 * small


def test_backward_memory_does_not_grow_with_the_intersection_capacity():
    # A tile's chunk loop is rematerialized too, so the backward workspace no
    # longer holds one [max_gaussians_per_tile, pixel] set per chunk.
    small = _backward_temp_bytes(
        RasterizationConfig(max_intersections=8192, tile_batch_size=4), 96, 64
    )
    large = _backward_temp_bytes(
        RasterizationConfig(max_intersections=65_536, tile_batch_size=4), 96, 64
    )

    assert large < 1.5 * small

