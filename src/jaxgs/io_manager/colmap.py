import struct
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO

import jax.numpy as jnp
import numpy as np
from PIL import Image

from ..data import Frame
from ..scene.camera import Camera


def _records(path: Path):
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _sparse_dir(scene_dir: Path) -> Path:
    sparse = scene_dir / "sparse" / "0"
    return sparse if sparse.exists() else scene_dir / "sparse"


def _scaled_intrinsics(
    model: str, width: int, height: int, params: tuple[float, ...] | list[float], downsample: int
):
    if model == "PINHOLE":
        fx, fy, cx, cy = params
    elif model == "SIMPLE_PINHOLE":
        focal, cx, cy = params
        fx = fy = focal
    else:
        raise ValueError(f"use an undistorted PINHOLE model, got {model}")
    new_width = max(1, round(width / downsample))
    new_height = max(1, round(height / downsample))
    scale_x, scale_y = new_width / width, new_height / height
    return (new_width, new_height, fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y)


def _frame(
    scene_dir: Path, name: str, qvec, tvec, camera_id: int, intrinsics: dict[int, tuple]
) -> Frame:
    width, height, fx, fy, cx, cy = intrinsics[camera_id]
    camera = Camera.from_colmap(qvec, tvec, fx, fy, cx, cy, width, height)
    return Frame(scene_dir / "images" / name, camera)


def _read(fid: BinaryIO, fmt: str):
    record = struct.Struct("<" + fmt)
    data = fid.read(record.size)
    if len(data) != record.size:
        raise ValueError("truncated COLMAP binary file")
    return record.unpack(data)


def _read_name(fid: BinaryIO) -> str:
    chars = bytearray()
    while True:
        char = fid.read(1)
        if not char:
            raise ValueError("truncated COLMAP image name")
        if char == b"\0":
            return chars.decode("utf-8")
        chars.extend(char)


def load_colmap_text(scene_dir: str | Path, downsample: int = 1) -> list[Frame]:
    """Load undistorted COLMAP text cameras/images and their RGB files."""
    if downsample < 1:
        raise ValueError("downsample must be at least 1")
    scene_dir = Path(scene_dir)
    sparse = _sparse_dir(scene_dir)
    intrinsics = {}
    for line in _records(sparse / "cameras.txt"):
        parts = line.split()
        camera_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        params = [float(x) for x in parts[4:]]
        intrinsics[camera_id] = _scaled_intrinsics(model, width, height, params, downsample)
    lines = [
        line.strip()
        for line in (sparse / "images.txt").read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    if len(lines) % 2:
        raise ValueError("COLMAP images.txt must contain two lines per image")
    frames = []
    for line in lines[::2]:
        parts = line.split(maxsplit=9)
        qvec = [float(x) for x in parts[1:5]]
        tvec = [float(x) for x in parts[5:8]]
        frames.append(_frame(scene_dir, parts[9], qvec, tvec, int(parts[8]), intrinsics))
    return sorted(frames, key=lambda frame: frame.image_path.name)


def load_colmap_binary(scene_dir: str | Path, downsample: int = 1) -> list[Frame]:
    """Load undistorted COLMAP binary cameras and registered images."""
    if downsample < 1:
        raise ValueError("downsample must be at least 1")
    scene_dir = Path(scene_dir)
    sparse = _sparse_dir(scene_dir)
    model_info = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4)}
    intrinsics = {}
    with (sparse / "cameras.bin").open("rb") as fid:
        for _ in range(_read(fid, "Q")[0]):
            camera_id, model_id, width, height = _read(fid, "iiQQ")
            if model_id not in model_info:
                raise ValueError(
                    f"camera {camera_id}: unsupported COLMAP model ID {model_id}; undistort first"
                )
            model, n_params = model_info[model_id]
            params = _read(fid, "d" * n_params)
            intrinsics[camera_id] = _scaled_intrinsics(model, width, height, params, downsample)

    frames = []
    with (sparse / "images.bin").open("rb") as fid:
        for _ in range(_read(fid, "Q")[0]):
            record = _read(fid, "idddddddi")
            name = _read_name(fid)
            fid.seek(24 * _read(fid, "Q")[0], 1)
            frames.append(_frame(scene_dir, name, record[1:5], record[5:8], record[8], intrinsics))
    return sorted(frames, key=lambda frame: frame.image_path.name)


def load_colmap(scene_dir: str | Path, downsample: int = 1) -> list[Frame]:
    sparse = _sparse_dir(Path(scene_dir))
    if (sparse / "cameras.bin").exists() and (sparse / "images.bin").exists():
        return load_colmap_binary(scene_dir, downsample)
    return load_colmap_text(scene_dir, downsample)


def load_colmap_images(
    scene_dir: str | Path, images: str = "images", *, resolution: int = 1
) -> list[Frame]:
    """Use image dimensions and the native training resolution convention."""
    if resolution != -1 and resolution <= 0:
        raise ValueError("resolution must be -1 or positive")
    scene_dir = Path(scene_dir)
    result = []
    for frame in load_colmap(scene_dir):
        path = scene_dir / images / frame.image_path.relative_to(scene_dir / "images")
        with Image.open(path) as image:
            width, height = image.size
        if resolution in (1, 2, 4, 8):
            width, height = round(width / resolution), round(height / resolution)
        else:
            scale = (
                (width / 1600 if width > 1600 else 1) if resolution == -1 else width / resolution
            )
            width, height = int(width / scale), int(height / scale)
        width, height = max(1, width), max(1, height)
        camera = frame.camera
        sx, sy = width / camera.width, height / camera.height
        camera = camera.replace(
            width=width,
            height=height,
            fx=camera.fx * sx,
            fy=camera.fy * sy,
            cx=camera.cx * sx,
            cy=camera.cy * sy,
        )
        result.append(
            replace(frame, image_path=path, camera=camera, resample=Image.Resampling.BICUBIC)
        )
    return result


def load_colmap_points(scene_dir: str | Path) -> tuple[jnp.ndarray, jnp.ndarray]:
    scene_dir = Path(scene_dir)
    sparse = _sparse_dir(scene_dir)
    binary = sparse / "points3D.bin"
    xyz, rgb = [], []
    if binary.exists():
        with binary.open("rb") as fid:
            for _ in range(_read(fid, "Q")[0]):
                record = _read(fid, "QdddBBBd")
                xyz.append(record[1:4])
                rgb.append([channel / 255.0 for channel in record[4:7]])
                fid.seek(8 * _read(fid, "Q")[0], 1)
    else:
        for line in _records(sparse / "points3D.txt"):
            parts = line.split()
            xyz.append([float(x) for x in parts[1:4]])
            rgb.append([int(x) / 255.0 for x in parts[4:7]])
    return jnp.asarray(np.asarray(xyz, np.float32).reshape(-1, 3)), jnp.asarray(
        np.asarray(rgb, np.float32).reshape(-1, 3)
    )
