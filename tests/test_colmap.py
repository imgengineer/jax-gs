from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from jax_gs.data.colmap import (
    CAMERA_MODELS,
    ColmapFormatError,
    qvec_to_rotation_matrix,
    read_cameras_binary,
    read_colmap_model,
    read_images_binary,
    read_points3d_binary,
)


def _write_cameras(path: Path) -> None:
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(CAMERA_MODELS)))
        for camera_id, model in enumerate(CAMERA_MODELS, start=1):
            file.write(struct.pack("<iiQQ", camera_id, model.model_id, 100, 80))
            params = tuple(float(index + 1) for index in range(model.num_params))
            file.write(struct.pack("<" + "d" * model.num_params, *params))


def _write_images(path: Path) -> None:
    records = (
        (
            7,
            (1.0, 0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            2,
            "nested/frame-b.png",
            ((10.5, 11.5, 21), (12.5, 13.5, -1)),
        ),
        (
            3,
            (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
            (0.0, 0.0, 0.0),
            1,
            "frame-a.png",
            (),
        ),
    )
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(records)))
        for image_id, qvec, tvec, camera_id, name, observations in records:
            file.write(struct.pack("<i7di", image_id, *qvec, *tvec, camera_id))
            file.write(name.encode("utf-8") + b"\x00")
            file.write(struct.pack("<Q", len(observations)))
            for x, y, point_id in observations:
                file.write(struct.pack("<ddq", x, y, point_id))


def _write_points(path: Path) -> None:
    with path.open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(
            struct.pack(
                "<QdddBBBd", 21, 1.25, -2.5, 3.75, 10, 20, 30, 0.125
            )
        )
        file.write(struct.pack("<Q", 2))
        file.write(struct.pack("<iiii", 7, 0, 3, 5))


@pytest.fixture
def binary_model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True)
    _write_cameras(model_dir / "cameras.bin")
    _write_images(model_dir / "images.bin")
    _write_points(model_dir / "points3D.bin")
    return model_dir


def test_reads_all_current_camera_models(binary_model: Path) -> None:
    cameras = read_cameras_binary(binary_model / "cameras.bin")

    assert len(cameras) == len(CAMERA_MODELS)
    for camera_id, expected_model in enumerate(CAMERA_MODELS, start=1):
        camera = cameras[camera_id]
        assert camera.model_id == expected_model.model_id
        assert camera.model == expected_model.name
        assert camera.params.shape == (expected_model.num_params,)

    pinhole = cameras[2]
    np.testing.assert_allclose(
        pinhole.intrinsic_matrix(width=50, height=20),
        [[0.5, 0.0, 1.5], [0.0, 0.5, 1.0], [0.0, 0.0, 1.0]],
    )
    with pytest.raises(ValueError, match="no pinhole intrinsics"):
        cameras[18].intrinsic_matrix()


def test_reads_images_and_colmap_pose_convention(binary_model: Path) -> None:
    images = read_images_binary(binary_model / "images.bin")

    image = images[7]
    assert image.name == "nested/frame-b.png"
    np.testing.assert_allclose(image.xys, [[10.5, 11.5], [12.5, 13.5]])
    np.testing.assert_array_equal(image.point3D_ids, [21, -1])
    np.testing.assert_allclose(image.w2c[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(image.c2w[:3, 3], [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(image.w2c @ image.c2w, np.eye(4), atol=1e-12)

    rotated = images[3]
    np.testing.assert_allclose(
        rotated.rotation_matrix,
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


def test_reads_points_and_tracks(binary_model: Path) -> None:
    points = read_points3d_binary(binary_model / "points3D.bin")

    point = points[21]
    np.testing.assert_allclose(point.xyz, [1.25, -2.5, 3.75])
    np.testing.assert_array_equal(point.rgb, [10, 20, 30])
    assert point.error == pytest.approx(0.125)
    np.testing.assert_array_equal(point.image_ids, [7, 3])
    np.testing.assert_array_equal(point.point2D_idxs, [0, 5])


def test_reads_complete_model_and_can_skip_points(binary_model: Path) -> None:
    model = read_colmap_model(binary_model)
    assert (len(model.cameras), len(model.images), len(model.points3D)) == (18, 2, 1)

    metadata_only = read_colmap_model(binary_model.parent.parent, load_points=False)
    assert metadata_only.points3D == {}


def test_rejects_truncated_binary(tmp_path: Path) -> None:
    path = tmp_path / "cameras.bin"
    path.write_bytes(struct.pack("<Q", 1) + b"\x00")

    with pytest.raises(ColmapFormatError, match="truncated"):
        read_cameras_binary(path)


def test_qvec_is_normalized_and_validated() -> None:
    np.testing.assert_allclose(qvec_to_rotation_matrix(np.array([2.0, 0, 0, 0])), np.eye(3))
    with pytest.raises(ValueError, match="non-zero"):
        qvec_to_rotation_matrix(np.zeros(4))
