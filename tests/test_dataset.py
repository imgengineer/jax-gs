import struct
from contextlib import closing

import numpy as np
import pytest
from PIL import Image

from jaxgs import Camera
from jaxgs.data import Frame, image_dataset
from jaxgs.io_manager.colmap import (
    load_colmap,
    load_colmap_binary,
    load_colmap_images,
    load_colmap_points,
    load_colmap_text,
)


def test_colmap_text_dataset(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    images = tmp_path / "images"
    images.mkdir()
    (sparse / "cameras.txt").write_text("# cameras\n1 PINHOLE 4 4 4 4 2 2\n")
    (sparse / "images.txt").write_text("# images\n1 1 0 0 0 0 0 0 1 frame.png\n\n")
    (sparse / "points3D.txt").write_text("1 0 0 2 255 0 0 0.1\n")
    Image.fromarray(np.full((4, 4, 3), 255, np.uint8)).save(images / "frame.png")
    frames = load_colmap_text(tmp_path, downsample=2)
    assert len(frames) == 1
    assert frames[0].camera.width == frames[0].camera.height == 2
    assert frames[0].load_image().shape == (2, 2, 3)
    xyz, rgb = load_colmap_points(tmp_path)
    np.testing.assert_allclose(xyz, [[0, 0, 2]])
    np.testing.assert_allclose(rgb, [[1, 0, 0]])


def test_colmap_binary_dataset(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    images = tmp_path / "images"
    images.mkdir()
    (sparse / "cameras.bin").write_bytes(struct.pack("<QiiQQdddd", 1, 1, 1, 4, 4, 4, 4, 2, 2))
    (sparse / "images.bin").write_bytes(
        struct.pack("<Qidddddddi", 1, 1, 1, 0, 0, 0, 0, 0, 0, 1)
        + b"frame.png\0"
        + struct.pack("<Q", 0)
    )
    (sparse / "points3D.bin").write_bytes(
        struct.pack("<QQdddBBBdQ", 1, 1, 0, 0, 2, 255, 0, 0, 0.1, 0)
    )
    Image.fromarray(np.full((4, 4, 3), 255, np.uint8)).save(images / "frame.png")
    frames = load_colmap(tmp_path, downsample=2)
    assert len(frames) == 1
    assert frames[0].camera.width == frames[0].camera.height == 2
    np.testing.assert_allclose(frames[0].camera.fx, 2)
    assert frames[0].load_image().shape == (2, 2, 3)
    xyz, rgb = load_colmap_points(tmp_path)
    np.testing.assert_allclose(xyz, [[0, 0, 2]])
    np.testing.assert_allclose(rgb, [[1, 0, 0]])


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_grain_decode_preserves_rgb_resize_and_normalization(tmp_path, mode):
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    pixels = np.arange(6 * 4 * channels, dtype=np.uint8).reshape(
        (4, 6) if mode == "L" else (4, 6, channels)
    )
    path = tmp_path / "frame.png"
    Image.fromarray(pixels).save(path)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 3, 2, 1.5, 1, 3, 2)
    frame = Frame(path, camera)
    with Image.open(path) as image:
        expected = np.asarray(image.convert("RGB").resize((3, 2), Image.Resampling.LANCZOS))
    with closing(iter(image_dataset([frame]))) as images:
        actual = next(images)
        with pytest.raises(StopIteration):
            next(images)
    assert isinstance(actual, np.ndarray) and actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(frame.load_image(), expected.astype(np.float32) / 255.0)


def test_grain_preserves_frame_order_repeat_and_sample_limit(tmp_path, monkeypatch):
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 2, 2, 1, 1, 2, 2)
    frames = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        Image.fromarray(np.full((2, 2, 3), index, np.uint8)).save(path)
        frames.append(Frame(path, camera))
    decoded = []
    load_rgb = Frame.load_rgb

    def read(frame):
        decoded.append(frame.image_path.name)
        return load_rgb(frame)

    monkeypatch.setattr(Frame, "load_rgb", read)
    with closing(iter(image_dataset(frames, steps=7))) as images:
        values = [int(image[0, 0, 0]) for image in images]
    assert values == [0, 1, 2, 0, 1, 2, 0]
    assert len(decoded) == 7  # Prefetch never reads past the requested steps.
    decoded.clear()
    with closing(iter(image_dataset(frames, steps=0))) as images:
        assert list(images) == []
    assert decoded == []
    with pytest.raises(ValueError, match="nonnegative"):
        image_dataset(frames, steps=-1)


@pytest.mark.parametrize("loader", [load_colmap_text, load_colmap_binary])
def test_colmap_rejects_invalid_downsample(tmp_path, loader):
    with pytest.raises(ValueError, match="downsample"):
        loader(tmp_path, downsample=0)


def test_simple_pinhole_and_sparse_without_subdirectory(tmp_path):
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    (sparse / "cameras.txt").write_text("1 SIMPLE_PINHOLE 9 5 6 4.5 2.5\n")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 image with spaces.png\n\n")
    (sparse / "points3D.txt").write_text("# empty reconstruction\n")
    (frame,) = load_colmap(tmp_path, downsample=2)
    assert frame.image_path.name == "image with spaces.png"
    assert (frame.camera.width, frame.camera.height) == (4, 2)
    np.testing.assert_allclose([frame.camera.fx, frame.camera.fy], [6 * 4 / 9, 6 * 2 / 5])
    assert all(points.shape == (0, 3) for points in load_colmap_points(tmp_path))
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 incomplete.png\n")
    with pytest.raises(ValueError, match="two lines"):
        load_colmap(tmp_path)
    (sparse / "cameras.txt").write_text("1 OPENCV 9 5 6 6 4.5 2.5\n")
    with pytest.raises(ValueError, match="undistorted"):
        load_colmap(tmp_path)


@pytest.mark.parametrize("damage", ["header", "model", "name"])
def test_colmap_rejects_truncated_or_unsupported_binary(tmp_path, damage):
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    cameras = struct.pack("<QiiQQddd", 1, 1, 0, 4, 4, 4, 2, 2)
    images = struct.pack("<Qidddddddi", 1, 1, 1, 0, 0, 0, 0, 0, 0, 1)
    expected = "truncated COLMAP image name"
    if damage == "header":
        cameras = b"\x01"
        expected = "truncated COLMAP binary"
    elif damage == "model":
        cameras = struct.pack("<QiiQQ", 1, 1, 99, 4, 4)
        expected = "unsupported COLMAP model"
    (sparse / "cameras.bin").write_bytes(cameras)
    (sparse / "images.bin").write_bytes(images + b"missing terminator")
    with pytest.raises(ValueError, match=expected):
        load_colmap(tmp_path)


@pytest.mark.parametrize(
    "width,height,resolution,expected",
    [
        (2000, 1000, -1, (1600, 800)),
        (31, 17, -1, (31, 17)),
        (31, 17, 2, (16, 8)),
        (32, 16, 10, (10, 5)),
    ],
)
def test_training_resolution_matches_native_resize(tmp_path, width, height, resolution, expected):
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    (tmp_path / "pyramid").mkdir()
    # Camera dimensions differ from the supplied image pyramid.
    (sparse / "cameras.txt").write_text(
        f"1 PINHOLE {width * 2} {height * 2} {width} {height} {width} {height}\n"
    )
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.png\n\n")
    pixels = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = tmp_path / "pyramid" / "frame.png"
    Image.fromarray(pixels).save(path)
    (frame,) = load_colmap_images(tmp_path, "pyramid", resolution=resolution)
    assert (frame.camera.width, frame.camera.height) == expected
    np.testing.assert_allclose([frame.camera.fx, frame.camera.fy], np.array(expected) / 2)
    with Image.open(path) as original:
        resized = np.asarray(original.resize(expected))  # native PIL default: bicubic
    with closing(iter(image_dataset([frame]))) as images:
        np.testing.assert_array_equal(next(images), resized)
    with pytest.raises(ValueError, match="resolution"):
        load_colmap_images(tmp_path, "pyramid", resolution=0)
