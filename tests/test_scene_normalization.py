from importlib import import_module
from itertools import product
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from jax_gs.training import (
    SceneTransform,
    _training_scene_scale,
    compute_scene_transform,
)


def _normalization_api() -> ModuleType:
    return import_module("jax_gs.data.normalize")


def _cameras_at(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    cameras = np.broadcast_to(
        np.eye(4, dtype=np.float64), (len(centers), 4, 4)
    ).copy()
    cameras[:, :3, 3] = centers
    return cameras


def _focus_scale_cameras() -> np.ndarray:
    # Identity cameras look along +z. Their closest points on the optical axes
    # have median [0, 1, 0], while the recentered camera radii are [2, 1, 3].
    # The non-strict current-main scale is therefore 1 / median(...) = 1 / 2.
    return _cameras_at(
        np.asarray(
            [
                [-2.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [3.0, 1.0, 0.0],
            ]
        )
    )


def _quarter_turn_cameras() -> np.ndarray:
    # Every camera is rotated +90 degrees around world z, so its OpenCV up
    # direction points along +x.  The centers become [-2, 1, 0], [0, 1, 1],
    # and [3, 1, 0] after the analytically required -90-degree alignment.
    return np.asarray(
        [
            [
                [0.0, -1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, -1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, -1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ],
        dtype=np.float64,
    )


def _anisotropic_symmetric_points() -> np.ndarray:
    # Paired points make the component-wise median unambiguous. The three
    # distinct axis variances avoid unstable eigenvector choices in the PCA.
    local_points = np.asarray(
        [
            [5.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.2],
        ],
        dtype=np.float64,
    )
    angle_z = 0.37
    angle_x = -0.23
    rotation_z = np.asarray(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0.0],
            [np.sin(angle_z), np.cos(angle_z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle_x), -np.sin(angle_x)],
            [0.0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )
    rotation = rotation_z @ rotation_x
    translation = np.asarray([1.25, -0.75, 2.5])
    return local_points @ rotation.T + translation


def _upside_down_points() -> np.ndarray:
    # Cartesian products make the covariance diagonal with distinct x/y/z
    # variances. Along z, median=0.45 is greater than mean=0, which triggers
    # current-main's final 180-degree x-axis rotation after PCA centering.
    return np.asarray(
        list(
            product(
                (-4.0, 0.0, 4.0),
                (-2.0, 0.0, 2.0),
                (-1.5, 0.4, 0.5, 0.6),
            )
        ),
        dtype=np.float64,
    )


def _scene(cameras: np.ndarray, points: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(camtoworlds=cameras, points=points)


def _camera_coordinates(world_to_camera: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=points.dtype)), axis=-1
    )
    return homogeneous @ world_to_camera[:3, :].T


def test_similarity_from_cameras_uses_focus_center_and_median_scale():
    normalize = _normalization_api()

    actual = normalize.similarity_from_cameras(
        _focus_scale_cameras(),
        strict_scaling=False,
        center_method="focus",
    )

    expected = np.eye(4, dtype=np.float64)
    expected[:3, :3] *= 0.5
    expected[:3, 3] = np.asarray([0.0, -0.5, 0.0])
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_similarity_aligns_rotated_camera_up_with_pose_strict_scaling():
    normalize = _normalization_api()
    cameras = _quarter_turn_cameras()

    actual = normalize.similarity_from_cameras(
        cameras,
        strict_scaling=True,
        center_method="poses",
    )

    # The -90-degree z rotation aligns camera up (+x in world space) to -y.
    # Pose centering subtracts [0, 1, 0], and strict scaling divides by the
    # largest resulting camera radius, 3.
    expected = np.asarray(
        [
            [0.0, 1.0 / 3.0, 0.0, 0.0],
            [-1.0 / 3.0, 0.0, 0.0, -1.0 / 3.0],
            [0.0, 0.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)

    aligned_up = actual[:3, :3] @ cameras[0, :3, :3] @ np.asarray(
        [0.0, -1.0, 0.0]
    )
    np.testing.assert_allclose(
        aligned_up / np.linalg.norm(aligned_up),
        np.asarray([0.0, -1.0, 0.0]),
        atol=1e-7,
    )


def test_normalize_scene_matches_analytic_rotated_camera_oracle():
    normalize = _normalization_api()
    cameras = _quarter_turn_cameras()
    points = np.asarray(
        [
            [-1.0, 6.0, 0.0],
            [-1.0, -6.0, 0.0],
            [-5.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-1.0, 0.0, 2.0],
            [-1.0, 0.0, -2.0],
        ],
        dtype=np.float64,
    )

    normalized_cameras, normalized_points, transform = normalize.normalize_scene(
        cameras, points
    )

    # Camera focus centering gives [0, -1, 0] after up alignment, then the
    # median camera radius gives scale 1/2.  The input points were chosen so
    # that this transform produces centered axis pairs with distinct x/y/z
    # variances; PCA is therefore identity and no upside-down fix is applied.
    expected_transform = np.asarray(
        [
            [0.0, 0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    expected_points = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    expected_cameras = _cameras_at(
        np.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.5], [1.5, 0.0, 0.0]])
    )

    np.testing.assert_allclose(transform, expected_transform, atol=1e-7)
    np.testing.assert_allclose(normalized_points, expected_points, atol=1e-7)
    np.testing.assert_allclose(normalized_cameras, expected_cameras, atol=1e-7)

    # Independently apply the oracle matrix, including removal of its uniform
    # scale from camera rotations, to keep all three returned values coherent.
    oracle_points = points @ expected_transform[:3, :3].T
    oracle_points += expected_transform[:3, 3]
    oracle_cameras = expected_transform @ cameras
    oracle_cameras[:, :3, :3] /= 0.5
    np.testing.assert_allclose(normalized_points, oracle_points, atol=1e-7)
    np.testing.assert_allclose(normalized_cameras, oracle_cameras, atol=1e-7)


def test_align_principal_axes_orders_unique_axes_and_is_right_handed():
    normalize = _normalization_api()
    points = _anisotropic_symmetric_points()

    matrix = normalize.align_principal_axes(points)
    aligned = normalize.transform_points(matrix, points)
    rotation = matrix[:3, :3]

    np.testing.assert_allclose(
        rotation @ rotation.T, np.eye(3), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.median(aligned, axis=0), 0.0, atol=1e-6)
    covariance = np.cov(aligned, rowvar=False)
    np.testing.assert_allclose(
        covariance - np.diag(np.diag(covariance)), 0.0, atol=1e-6
    )
    variances = np.diag(covariance)
    assert variances[0] > variances[1] > variances[2]


def test_compute_scene_transform_composes_upside_down_fix():
    normalize = _normalization_api()
    cameras = _focus_scale_cameras()
    points = _upside_down_points()
    scene = _scene(cameras, points)

    t1 = normalize.similarity_from_cameras(cameras)
    points_after_t1 = normalize.transform_points(t1, points)
    t2 = normalize.align_principal_axes(points_after_t1)
    points_before_fix = normalize.transform_points(t2, points_after_t1)
    assert np.median(points_before_fix[:, 2]) > np.mean(
        points_before_fix[:, 2]
    )
    t3 = np.diag([1.0, -1.0, -1.0, 1.0])
    expected = t3 @ t2 @ t1

    transform = compute_scene_transform(scene, normalize_world_space=True)

    assert isinstance(transform, SceneTransform)
    np.testing.assert_allclose(
        transform.matrix, expected, rtol=1e-6, atol=1e-6
    )
    normalized_points = transform.points(points)
    np.testing.assert_allclose(
        normalized_points,
        normalize.transform_points(expected, points),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.median(normalized_points[:, 2]) <= np.mean(
        normalized_points[:, 2]
    )


def test_compute_scene_transform_disabled_is_identity():
    cameras = _focus_scale_cameras()
    points = _anisotropic_symmetric_points()
    scene = _scene(cameras, points)

    transform = compute_scene_transform(scene, normalize_world_space=False)

    assert isinstance(transform, SceneTransform)
    np.testing.assert_array_equal(transform.matrix, np.eye(4))
    np.testing.assert_allclose(transform.points(points), points)
    world_to_cameras = np.linalg.inv(cameras)
    np.testing.assert_allclose(
        transform.world_to_camera(world_to_cameras), world_to_cameras
    )


def test_scene_transform_rejects_non_similarity_matrix():
    with pytest.raises(ValueError, match="similarity"):
        SceneTransform(np.diag([1.0, 2.0, 1.0, 1.0]))


def test_scene_transform_preserves_camera_projection_geometry():
    normalize = _normalization_api()
    cameras = _focus_scale_cameras()
    scene = _scene(cameras, _anisotropic_symmetric_points())
    transform = compute_scene_transform(scene, normalize_world_space=True)

    camera_points = np.asarray(
        [[0.25, -0.1, 2.0], [-0.4, 0.3, 4.0]], dtype=np.float64
    )
    probe_points = camera_points + cameras[0, :3, 3]
    normalized_points = transform.points(probe_points)
    world_to_cameras = np.linalg.inv(cameras)
    normalized_world_to_cameras = transform.world_to_camera(world_to_cameras)

    np.testing.assert_allclose(
        normalized_points,
        normalize.transform_points(transform.matrix, probe_points),
        rtol=1e-6,
        atol=1e-6,
    )
    expected_cameras = normalize.transform_cameras(transform.matrix, cameras)
    np.testing.assert_allclose(
        normalized_world_to_cameras,
        np.linalg.inv(expected_cameras),
        rtol=1e-6,
        atol=1e-6,
    )
    nested_cameras = transform.camera_to_world(cameras[None, ...])
    assert nested_cameras.shape == (1,) + cameras.shape
    np.testing.assert_allclose(nested_cameras[0], expected_cameras)

    original_camera_points = _camera_coordinates(
        world_to_cameras[0], probe_points
    )
    transformed_camera_points = _camera_coordinates(
        normalized_world_to_cameras[0], normalized_points
    )
    similarity_scale = np.linalg.norm(transform.matrix[:3, 0])
    np.testing.assert_allclose(
        transformed_camera_points,
        similarity_scale * original_camera_points,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        transformed_camera_points[:, :2]
        / transformed_camera_points[:, 2:3],
        original_camera_points[:, :2] / original_camera_points[:, 2:3],
        rtol=1e-6,
        atol=1e-6,
    )


def test_training_scene_scale_uses_normalized_extent_and_global_scale():
    scene = _scene(_focus_scale_cameras(), _anisotropic_symmetric_points())
    transform = compute_scene_transform(scene)

    np.testing.assert_allclose(
        _training_scene_scale(scene, transform),
        1.478080587188068,
    )
    np.testing.assert_allclose(
        _training_scene_scale(scene, transform, global_scale=2.5),
        3.69520146797017,
    )
