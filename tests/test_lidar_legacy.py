import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_gs
from jax_gs.config import RasterizationConfig
from jax_gs.lidar import (
    ANGLE_TO_PIXEL_SCALING_FACTOR,
    LegacyLidarModel,
    RowOffsetStructuredSpinningLidarModelParameters,
    RowOffsetStructuredSpinningLidarModelParametersExt,
    SpinningDirection,
    angles_to_tile_indices,
    compute_angles_to_columns_map,
    compute_tiling,
    generate_lidar_image_points,
    relative_sensor_angles,
    sensor_angles_to_rays,
    valid_sensor_angles,
)
from jax_gs.lidar_intersections import (
    has_any_rays_in_tile,
    isect_tiles_lidar,
    lidar_sample_tileid,
)
from jax_gs.rasterization import rasterization
from jax_gs.three_dgut import (
    _world_rays_from_pixels,
    fully_fused_projection_with_ut,
    rasterize_to_pixels_eval3d,
)


def _parameters():
    return RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=jnp.array([0.2, 0.0, -0.2], dtype=jnp.float32),
        column_azimuths_rad=jnp.array([1.5, 0.75, 0.0, -0.75, -1.5], dtype=jnp.float32),
        row_azimuth_offsets_rad=jnp.array([0.01, 0.0, -0.01], dtype=jnp.float32),
        spinning_frequency_hz=10.0,
        spinning_direction=SpinningDirection.CLOCKWISE,
    )


@pytest.fixture(scope="module")
def lidar():
    parameters = _parameters()
    return RowOffsetStructuredSpinningLidarModelParametersExt(
        parameters,
        compute_angles_to_columns_map(parameters, resolution_factor=1),
        compute_tiling(
            parameters,
            n_bins_elevation=2,
            max_pts_per_tile=4,
            resolution_elevation=20,
            densification_factor_azimuth=2,
        ),
    )


def _decode_images_and_tiles(intersections, tile_count):
    valid_count = int(intersections.valid_count)
    high = np.asarray(intersections.isect_ids[:valid_count, 0], dtype=np.int32)
    high = high.view(np.uint32)
    tile_bits = max(1, (tile_count - 1).bit_length())
    images = (high >> np.uint32(tile_bits)).astype(np.int32)
    tiles = (high & np.uint32((1 << tile_bits) - 1)).astype(np.int32)
    return images, tiles


def _full_fov_gaussians(lidar, depths):
    direction = -1.0 if lidar.spinning_direction == SpinningDirection.CLOCKWISE else 1.0
    center = jnp.array(
        [
            lidar.fov_horiz_rad.start + direction * lidar.fov_horiz_rad.span / 2.0,
            lidar.fov_vert_rad.start - lidar.fov_vert_rad.span / 2.0,
        ],
        dtype=jnp.float32,
    )
    radius = jnp.array(
        [lidar.fov_horiz_rad.span, lidar.fov_vert_rad.span],
        dtype=jnp.float32,
    )
    count = len(depths)
    return (
        jnp.broadcast_to(center * ANGLE_TO_PIXEL_SCALING_FACTOR, (1, count, 2)),
        jnp.broadcast_to(radius * ANGLE_TO_PIXEL_SCALING_FACTOR, (1, count, 2)),
        jnp.asarray(depths, dtype=jnp.float32)[None, :],
    )


def test_root_api_and_parameter_layout_match_current_main():
    parameters = _parameters()

    assert jax_gs.RowOffsetStructuredSpinningLidarModelParameters is type(parameters)
    assert jax_gs.SpinningDirection is SpinningDirection
    assert parameters.n_rows == 3
    assert parameters.n_columns == 5
    assert parameters.fov_vert_rad.start == pytest.approx(0.2)
    assert parameters.fov_vert_rad.span == pytest.approx(0.4)
    assert parameters.fov_horiz_rad.start == pytest.approx(1.51)
    assert parameters.fov_horiz_rad.span == pytest.approx(3.02)

    expected_elements = np.array(
        [[column, row] for row in range(3) for column in range(5)],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(parameters.create_elements(), expected_elements)
    angles = parameters.elements_to_sensor_angles(parameters.create_elements())
    np.testing.assert_allclose(angles[:, 1], np.repeat([0.2, 0.0, -0.2], 5), atol=1e-6)

    copied = _parameters()
    assert copied == parameters
    assert hash(copied) == hash(parameters)


def test_parameters_validate_dtype_shape_and_angular_order():
    with pytest.raises(TypeError, match="dtype float32"):
        RowOffsetStructuredSpinningLidarModelParameters(
            row_elevations_rad=jnp.array([1, 0], dtype=jnp.int32),
            column_azimuths_rad=jnp.array([1.0, 0.0], dtype=jnp.float32),
            row_azimuth_offsets_rad=jnp.zeros((2,), dtype=jnp.float32),
            spinning_frequency_hz=10.0,
            spinning_direction=SpinningDirection.CLOCKWISE,
        )

    with pytest.raises(ValueError, match="descending"):
        RowOffsetStructuredSpinningLidarModelParameters(
            row_elevations_rad=jnp.array([0.2, 0.0, 0.1], dtype=jnp.float32),
            column_azimuths_rad=jnp.array([1.0, 0.0], dtype=jnp.float32),
            row_azimuth_offsets_rad=jnp.zeros((3,), dtype=jnp.float32),
            spinning_frequency_hz=10.0,
            spinning_direction=SpinningDirection.CLOCKWISE,
        )

    with pytest.raises(ValueError, match="spinning direction"):
        RowOffsetStructuredSpinningLidarModelParameters(
            row_elevations_rad=jnp.array([0.2, 0.0], dtype=jnp.float32),
            column_azimuths_rad=jnp.array([1.0, 0.0, 0.5], dtype=jnp.float32),
            row_azimuth_offsets_rad=jnp.zeros((2,), dtype=jnp.float32),
            spinning_frequency_hz=10.0,
            spinning_direction=SpinningDirection.CLOCKWISE,
        )


def test_parameter_and_extended_state_are_jittable_pytrees(lidar):
    eager = lidar.elements_to_sensor_angles(lidar.create_elements())
    compiled = jax.jit(
        lambda value: value.elements_to_sensor_angles(value.create_elements())
    )(lidar)
    np.testing.assert_allclose(compiled, eager, atol=1e-6)

    max_elements = jax.jit(
        lambda value: jnp.asarray(value.tiling.max_elements_per_tile)
    )(lidar)
    assert int(max_elements) == int(
        np.asarray(lidar.tiling.tiles_pack_info[:, 1]).max()
    )


def test_angle_helpers_and_ray_roundtrip(lidar):
    angles = lidar.elements_to_sensor_angles(lidar.create_elements())
    relative = relative_sensor_angles(lidar, angles)
    assert np.all(np.asarray(relative) >= -1e-6)
    assert np.all(np.asarray(valid_sensor_angles(lidar, angles)))

    rays = sensor_angles_to_rays(lidar, angles)
    np.testing.assert_allclose(
        jnp.linalg.norm(rays.sensor_rays, axis=-1), 1.0, atol=1e-6
    )
    assert np.all(np.asarray(rays.valid_flag))

    model = LegacyLidarModel(lidar)
    image_points, valid = model.camera_ray_to_image_point(
        rays.sensor_rays, margin_factor=1.0e-5
    )
    reconstructed = model.image_point_to_camera_ray(image_points)
    assert np.all(np.asarray(valid))
    np.testing.assert_allclose(reconstructed.sensor_rays, rays.sensor_rays, atol=1e-6)


def test_structured_image_points_and_shutter_times(lidar):
    image_points = generate_lidar_image_points(lidar)
    assert image_points.shape == (lidar.n_rows, lidar.n_columns, 2)
    times = jax.jit(
        lambda value: LegacyLidarModel(value).shutter_relative_frame_time(
            generate_lidar_image_points(value)
        )
    )(lidar)
    expected = np.broadcast_to(
        np.linspace(0.0, 1.0, lidar.n_columns, dtype=np.float32),
        (lidar.n_rows, lidar.n_columns),
    )
    np.testing.assert_allclose(times, expected, atol=1e-6)


def test_lidar_world_rays_use_column_lookup_for_scan_pose(lidar):
    image_points = generate_lidar_image_points(lidar).reshape(-1, 2)
    start = jnp.eye(4, dtype=jnp.float32)
    end = start.at[1, 3].set(1.0)
    origins, directions, valid = jax.jit(
        lambda parameters: _world_rays_from_pixels(
            image_points,
            start,
            jnp.eye(3, dtype=jnp.float32),
            parameters.n_columns,
            parameters.n_rows,
            camera_model="lidar",
            radial_coeffs=None,
            tangential_coeffs=None,
            thin_prism_coeffs=None,
            ftheta_coeffs=None,
            rolling_shutter=jax_gs.RollingShutterType.GLOBAL,
            viewmat_rs=end,
            lidar_coeffs=parameters,
        )
    )(lidar)
    expected_times = np.tile(
        np.linspace(0.0, 1.0, lidar.n_columns, dtype=np.float32),
        lidar.n_rows,
    )
    np.testing.assert_allclose(origins[:, 1], -expected_times, atol=1e-6)
    np.testing.assert_allclose(origins[:, [0, 2]], 0.0, atol=1e-6)
    np.testing.assert_allclose(jnp.linalg.norm(directions, axis=-1), 1.0, atol=1e-6)
    assert np.all(np.asarray(valid))


def test_precomputed_column_map_and_tiling_are_consistent(lidar):
    expected_columns = np.broadcast_to(
        np.arange(lidar.n_columns, dtype=np.int32),
        (lidar.n_rows, lidar.n_columns),
    )
    np.testing.assert_array_equal(lidar.angles_to_columns_map, expected_columns)

    tiling = lidar.tiling
    counts = np.asarray(tiling.tiles_pack_info[:, 1])
    starts = np.asarray(tiling.tiles_pack_info[:, 0])
    assert counts.sum() == lidar.n_rows * lidar.n_columns
    assert tiling.max_elements_per_tile == counts.max()
    np.testing.assert_array_equal(starts, np.cumsum(counts) - counts)

    elements = np.asarray(tiling.tiles_to_elements_map)
    expected = np.asarray(lidar.create_elements())
    np.testing.assert_array_equal(
        elements[np.lexsort((elements[:, 0], elements[:, 1]))], expected
    )
    element_tiles = angles_to_tile_indices(
        lidar,
        lidar.elements_to_sensor_angles(tiling.tiles_to_elements_map),
        n_bins_azimuth=tiling.n_bins_azimuth,
        n_bins_elevation=tiling.n_bins_elevation,
        cdf_elevation=tiling.cdf_elevation,
    )
    np.testing.assert_array_equal(
        element_tiles,
        np.repeat(np.arange(counts.size, dtype=np.int32), counts),
    )
    assert bool(
        has_any_rays_in_tile(
            lidar,
            jnp.array([0, 0]),
            jnp.array(
                [
                    tiling.cdf_resolution_azimuth,
                    tiling.cdf_resolution_elevation,
                ]
            ),
        )
    )


def test_lidar_tile_sampling_uses_half_open_bounds(lidar):
    tiling = lidar.tiling
    zero = lidar_sample_tileid(lidar, jnp.zeros((2,)), jnp.floor)
    np.testing.assert_array_equal(zero.idx, [0, 0])
    np.testing.assert_array_equal(zero.idxdense, [0, 0])

    span = (
        jnp.array(
            [lidar.fov_horiz_rad.span, lidar.fov_vert_rad.span], dtype=jnp.float32
        )
        * ANGLE_TO_PIXEL_SCALING_FACTOR
    )
    end = lidar_sample_tileid(lidar, span, jnp.ceil)
    np.testing.assert_array_equal(
        end.idx, [tiling.n_bins_azimuth, tiling.n_bins_elevation]
    )
    np.testing.assert_array_equal(
        end.idxdense,
        [tiling.cdf_resolution_azimuth, tiling.cdf_resolution_elevation],
    )


def test_lidar_intersections_are_complete_depth_sorted_and_jittable(lidar):
    means, radii, depths = _full_fov_gaussians(lidar, [2.0, 1.0])
    tile_count = lidar.tiling.n_bins_azimuth * lidar.tiling.n_bins_elevation
    capacity = tile_count * 2 + 2

    @jax.jit
    def intersect(active_mask):
        return isect_tiles_lidar(
            lidar,
            means,
            radii,
            depths,
            max_intersections=capacity,
            active_mask=active_mask,
        )

    result = intersect(jnp.array([[True, True]]))
    np.testing.assert_array_equal(result.tiles_per_gaussian, [[tile_count, tile_count]])
    assert int(result.valid_count) == tile_count * 2
    assert not bool(result.overflow)
    np.testing.assert_array_equal(
        result.flatten_ids[: result.valid_count],
        np.tile(np.array([1, 0], dtype=np.int32), tile_count),
    )
    images, tiles = _decode_images_and_tiles(result, tile_count)
    np.testing.assert_array_equal(images, np.zeros(tile_count * 2, dtype=np.int32))
    np.testing.assert_array_equal(tiles, np.repeat(np.arange(tile_count), 2))
    np.testing.assert_array_equal(result.flatten_ids[result.valid_count :], [-1, -1])

    cache_size = intersect._cache_size()
    masked = intersect(jnp.array([[True, False]]))
    assert intersect._cache_size() == cache_size == 1
    assert int(masked.valid_count) == tile_count
    np.testing.assert_array_equal(
        masked.flatten_ids[: masked.valid_count], np.zeros(tile_count, dtype=np.int32)
    )

    overflow = isect_tiles_lidar(lidar, means, radii, depths, max_intersections=1)
    assert int(overflow.valid_count) == 1
    assert bool(overflow.overflow)


def test_lidar_intersections_support_dense_leading_images_and_packed(lidar):
    means, radii, depths = _full_fov_gaussians(lidar, [1.0])
    means = jnp.broadcast_to(means, (2, 1, 2))
    radii = jnp.broadcast_to(radii, (2, 1, 2))
    depths = jnp.broadcast_to(depths, (2, 1))
    tile_count = lidar.tiling.n_bins_azimuth * lidar.tiling.n_bins_elevation

    dense = isect_tiles_lidar(
        lidar,
        means,
        radii,
        depths,
        max_intersections=tile_count * 2,
    )
    packed = isect_tiles_lidar(
        lidar,
        means[:, 0],
        radii[:, 0],
        depths[:, 0],
        packed=True,
        n_images=2,
        image_ids=jnp.array([0, 1], dtype=jnp.int32),
        gaussian_ids=jnp.array([4, 8], dtype=jnp.int32),
        max_intersections=tile_count * 2,
    )
    np.testing.assert_array_equal(
        dense.tiles_per_gaussian, [[tile_count], [tile_count]]
    )
    np.testing.assert_array_equal(packed.tiles_per_gaussian, [tile_count, tile_count])
    np.testing.assert_array_equal(dense.isect_ids, packed.isect_ids)
    np.testing.assert_array_equal(dense.flatten_ids, packed.flatten_ids)

    with pytest.raises(ValueError, match="segmented"):
        isect_tiles_lidar(
            lidar,
            means[:, 0],
            radii[:, 0],
            depths[:, 0],
            packed=True,
            segmented=True,
            n_images=2,
            image_ids=jnp.array([0, 1], dtype=jnp.int32),
            gaussian_ids=jnp.array([4, 8], dtype=jnp.int32),
        )


def test_lidar_intersections_reject_zero_extent_and_preserve_padding(lidar):
    point = generate_lidar_image_points(lidar)[1, 2]
    means = point[None, None, :]
    radii = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    depths = jnp.ones((1, 1), dtype=jnp.float32)
    result = isect_tiles_lidar(lidar, means, radii, depths, max_intersections=3)
    np.testing.assert_array_equal(result.tiles_per_gaussian, [[0]])
    assert int(result.valid_count) == 0
    assert not bool(result.overflow)
    np.testing.assert_array_equal(result.isect_ids, np.full((3, 2), -1))
    np.testing.assert_array_equal(result.flatten_ids, np.full((3,), -1))


def test_periodic_lidar_intersection_merges_both_seam_regions():
    parameters = RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=jnp.array([0.1, -0.1], dtype=jnp.float32),
        column_azimuths_rad=jnp.array([3.0, 1.0, -1.0, -3.0], dtype=jnp.float32),
        row_azimuth_offsets_rad=jnp.array([0.2, -0.2], dtype=jnp.float32),
        spinning_frequency_hz=10.0,
        spinning_direction=SpinningDirection.CLOCKWISE,
    )
    assert parameters.fov_horiz_rad.span == pytest.approx(2.0 * math.pi)
    extended = RowOffsetStructuredSpinningLidarModelParametersExt(
        parameters,
        compute_angles_to_columns_map(parameters, resolution_factor=1),
        compute_tiling(
            parameters,
            n_bins_elevation=1,
            max_pts_per_tile=2,
            resolution_elevation=10,
            densification_factor_azimuth=2,
        ),
    )
    mean = (
        jnp.array([[[parameters.fov_horiz_rad.start, 0.0]]], dtype=jnp.float32)
        * ANGLE_TO_PIXEL_SCALING_FACTOR
    )
    radii = jnp.array([[[0.3, 0.2]]], dtype=jnp.float32) * ANGLE_TO_PIXEL_SCALING_FACTOR
    depths = jnp.ones((1, 1), dtype=jnp.float32)
    result = isect_tiles_lidar(extended, mean, radii, depths)
    _, tiles = _decode_images_and_tiles(
        result,
        extended.tiling.n_bins_azimuth * extended.tiling.n_bins_elevation,
    )
    assert 0 in tiles
    assert extended.tiling.n_bins_azimuth - 1 in tiles
    assert len(np.unique(tiles)) == len(tiles)


def _ut_scene(lidar):
    angle = jnp.array([[0.0, 0.1]], dtype=jnp.float32)
    means = 2.0 * sensor_angles_to_rays(lidar, angle).sensor_rays
    quats = jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    scales = jnp.full((1, 3), 0.03, dtype=jnp.float32)
    opacities = jnp.array([0.8], dtype=jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None, ...]
    Ks = jnp.eye(3, dtype=jnp.float32)[None, ...]
    return means, quats, scales, opacities, viewmats, Ks


def test_lidar_ut_projection_is_jittable_differentiable_and_ignores_image_size(lidar):
    means, quats, scales, opacities, viewmats, Ks = _ut_scene(lidar)

    @jax.jit
    def project(parameters, points):
        return fully_fused_projection_with_ut(
            points,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            width=1,
            height=1,
            camera_model="lidar",
            lidar_coeffs=parameters,
            calc_compensations=True,
        )

    radii, means2d, depths, conics, compensations, valid = project(lidar, means)
    assert radii.shape == (1, 1, 2)
    assert means2d.shape == (1, 1, 2)
    assert bool(valid[0, 0])
    assert np.all(np.asarray(radii[0, 0]) > 0)
    np.testing.assert_allclose(means2d[0, 0], [0.0, 102.4], atol=2e-2)
    np.testing.assert_allclose(depths[0, 0], means[0, 2], atol=1e-6)
    assert np.all(np.isfinite(np.asarray(conics)))
    assert np.all(np.isfinite(np.asarray(compensations)))

    gradient = jax.grad(lambda points: jnp.sum(project(lidar, points)[1]))(means)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.any(np.abs(np.asarray(gradient)) > 0.0)


def test_lidar_ut_projection_validates_camera_and_coefficients_pairing(lidar):
    means, quats, scales, opacities, viewmats, Ks = _ut_scene(lidar)
    arguments = (means, quats, scales, opacities, viewmats, Ks, 1, 1)

    with pytest.raises(ValueError, match="requires lidar_coeffs"):
        fully_fused_projection_with_ut(*arguments, camera_model="lidar")
    with pytest.raises(ValueError, match="camera_model='lidar'"):
        fully_fused_projection_with_ut(
            *arguments, camera_model="pinhole", lidar_coeffs=lidar
        )


def _element_gaussian(lidar, row=0, column=2):
    angles = lidar.elements_to_sensor_angles(
        jnp.array([[column, row]], dtype=jnp.int32)
    )
    means = 2.0 * sensor_angles_to_rays(lidar, angles).sensor_rays
    quats = jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    scales = jnp.full((1, 3), 0.08, dtype=jnp.float32)
    opacities = jnp.array([0.8], dtype=jnp.float32)
    colors = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)
    return means, quats, scales, opacities, colors


def test_low_level_lidar_eval3d_uses_angular_tile_elements(lidar):
    row, column = 0, 2
    means, quats, scales, opacities, colors = _element_gaussian(lidar, row, column)
    tile_id = int(
        angles_to_tile_indices(
            lidar,
            lidar.elements_to_sensor_angles(
                jnp.array([[column, row]], dtype=jnp.int32)
            ),
            n_bins_azimuth=lidar.tiling.n_bins_azimuth,
            n_bins_elevation=lidar.tiling.n_bins_elevation,
            cdf_elevation=lidar.tiling.cdf_elevation,
        )[0]
    )
    tile_count = lidar.tiling.n_bins_azimuth * lidar.tiling.n_bins_elevation
    offsets = (jnp.arange(tile_count, dtype=jnp.int32) > tile_id).astype(jnp.int32)
    offsets = offsets.reshape(
        1, lidar.tiling.n_bins_elevation, lidar.tiling.n_bins_azimuth
    )
    viewmats = jnp.eye(4, dtype=jnp.float32)[None, ...]
    Ks = jnp.eye(3, dtype=jnp.float32)[None, ...]

    @jax.jit
    def render(points):
        return rasterize_to_pixels_eval3d(
            points,
            quats,
            scales,
            colors[None, ...],
            opacities[None, ...],
            viewmats,
            Ks,
            image_width=1,
            image_height=1,
            tile_size=4,
            isect_offsets=offsets,
            flatten_ids=jnp.array([0], dtype=jnp.int32),
            camera_model="lidar",
            lidar_coeffs=lidar,
            max_gaussians_per_tile=1,
        )

    rendered, alpha = render(means)
    assert rendered.shape == (1, lidar.n_rows, lidar.n_columns, 3)
    assert alpha.shape == (1, lidar.n_rows, lidar.n_columns, 1)
    np.testing.assert_allclose(alpha[0, row, column, 0], 0.8, atol=1e-6)
    np.testing.assert_allclose(rendered[..., 0], alpha[..., 0], atol=1e-6)
    gradient = jax.grad(lambda points: render(points)[0].sum())(means)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_high_level_lidar_rasterization_overrides_resolution_and_reports_tiles(lidar):
    row, column = 0, 2
    means, quats, scales, opacities, colors = _element_gaussian(lidar, row, column)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None, ...]
    Ks = jnp.eye(3, dtype=jnp.float32)[None, ...]
    config = RasterizationConfig(
        backend="intersections",
        tile_size=4,
        max_gaussians_per_tile=4,
        max_intersections=16,
        tile_batch_size=1,
        ut_chunk_size=1,
    )

    @jax.jit
    def render(points):
        return rasterization(
            points,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width=999,
            height=999,
            camera_model="lidar",
            lidar_coeffs=lidar,
            with_ut=True,
            with_eval3d=True,
            config=config,
        )

    rendered, alpha, info = render(means)
    assert rendered.shape == (1, lidar.n_rows, lidar.n_columns, 3)
    assert alpha.shape == (1, lidar.n_rows, lidar.n_columns, 1)
    np.testing.assert_allclose(alpha[0, row, column, 0], 0.8, atol=2e-5)
    assert int(info["width"]) == lidar.n_columns
    assert int(info["height"]) == lidar.n_rows
    assert int(info["tile_width"]) == lidar.tiling.n_bins_azimuth
    assert int(info["tile_height"]) == lidar.tiling.n_bins_elevation
    assert bool(info["used_unscented_transform"])
    assert bool(info["eval3d_world_space"])
    gradient = jax.grad(lambda points: render(points)[0].sum())(means)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_high_level_lidar_supports_leading_batches_and_default_packed_metadata(
    lidar,
):
    means, quats, scales, opacities, colors = _element_gaussian(lidar)
    means = jnp.broadcast_to(means, (2, 1, 3))
    quats = jnp.broadcast_to(quats, (2, 1, 4))
    scales = jnp.broadcast_to(scales, (2, 1, 3))
    opacities = jnp.broadcast_to(opacities, (2, 1))
    colors = jnp.broadcast_to(colors, (2, 1, 3))
    viewmats = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (2, 2, 4, 4))
    Ks = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (2, 2, 3, 3))
    rendered, alpha, info = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width=1,
        height=1,
        camera_model="lidar",
        lidar_coeffs=lidar,
        with_ut=True,
        with_eval3d=True,
        config=RasterizationConfig(
            backend="intersections",
            tile_size=4,
            max_gaussians_per_tile=2,
            max_intersections=16,
            tile_batch_size=1,
            ut_chunk_size=1,
        ),
    )
    assert rendered.shape == (2, 2, lidar.n_rows, lidar.n_columns, 3)
    assert alpha.shape == (2, 2, lidar.n_rows, lidar.n_columns, 1)
    np.testing.assert_allclose(alpha[:, :, 0, 2, 0], 0.8, atol=2e-5)
    assert bool(info["packed_requested"])
    assert bool(info["packed_metadata_available"])
    assert int(info["projection_capacity"]) == 4
    assert int(info["projection_valid_count"]) == 4
    assert info["means2d"].shape == (4, 2)
    np.testing.assert_array_equal(info["batch_ids"], [0, 0, 1, 1])
    np.testing.assert_array_equal(info["camera_ids"], [0, 1, 0, 1])
    np.testing.assert_array_equal(info["gaussian_ids"], 0)
    np.testing.assert_array_equal(info["indptr"], [0, 1, 2, 3, 4])


def test_high_level_lidar_requires_ut_and_eval3d(lidar):
    means, quats, scales, opacities, colors = _element_gaussian(lidar)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None, ...]
    Ks = jnp.eye(3, dtype=jnp.float32)[None, ...]
    arguments = (means, quats, scales, opacities, colors, viewmats, Ks, 1, 1)

    with pytest.raises(ValueError, match="with_ut=True"):
        rasterization(
            *arguments,
            camera_model="lidar",
            lidar_coeffs=lidar,
            with_eval3d=True,
        )
    with pytest.raises(ValueError, match="with_eval3d=True"):
        rasterization(
            *arguments,
            camera_model="lidar",
            lidar_coeffs=lidar,
            with_ut=True,
        )
