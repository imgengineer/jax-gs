"""Dependency-free readers for COLMAP sparse reconstruction binaries.

COLMAP stores registered image poses as a quaternion and translation mapping
world coordinates to camera coordinates.  This module preserves that
convention and exposes both the world-to-camera matrix and its inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class ColmapFormatError(ValueError):
    """Raised when a COLMAP binary is truncated or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class CameraModel:
    model_id: int
    name: str
    num_params: int
    focal_indices: tuple[int, ...]
    principal_point_indices: tuple[int, ...]


# IDs and parameter counts are part of COLMAP's on-disk format.  Models 11--17
# were added after the original public read_write_model.py helper.
CAMERA_MODELS: tuple[CameraModel, ...] = (
    CameraModel(0, "SIMPLE_PINHOLE", 3, (0,), (1, 2)),
    CameraModel(1, "PINHOLE", 4, (0, 1), (2, 3)),
    CameraModel(2, "SIMPLE_RADIAL", 4, (0,), (1, 2)),
    CameraModel(3, "RADIAL", 5, (0,), (1, 2)),
    CameraModel(4, "OPENCV", 8, (0, 1), (2, 3)),
    CameraModel(5, "OPENCV_FISHEYE", 8, (0, 1), (2, 3)),
    CameraModel(6, "FULL_OPENCV", 12, (0, 1), (2, 3)),
    CameraModel(7, "FOV", 5, (0, 1), (2, 3)),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4, (0,), (1, 2)),
    CameraModel(9, "RADIAL_FISHEYE", 5, (0,), (1, 2)),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12, (0, 1), (2, 3)),
    CameraModel(11, "RAD_TAN_THIN_PRISM_FISHEYE", 16, (0, 1), (2, 3)),
    CameraModel(12, "SIMPLE_DIVISION", 4, (0,), (1, 2)),
    CameraModel(13, "DIVISION", 5, (0, 1), (2, 3)),
    CameraModel(14, "SIMPLE_FISHEYE", 3, (0,), (1, 2)),
    CameraModel(15, "FISHEYE", 4, (0, 1), (2, 3)),
    CameraModel(16, "EUCM", 6, (0, 1), (2, 3)),
    CameraModel(17, "EQUIRECTANGULAR", 2, (), ()),
)
CAMERA_MODEL_BY_ID: Mapping[int, CameraModel] = {
    model.model_id: model for model in CAMERA_MODELS
}
CAMERA_MODEL_BY_NAME: Mapping[str, CameraModel] = {
    model.name: model for model in CAMERA_MODELS
}


@dataclass(frozen=True, slots=True)
class Camera:
    id: int
    model_id: int
    model: str
    width: int
    height: int
    params: FloatArray

    def intrinsic_matrix(
        self, width: int | None = None, height: int | None = None
    ) -> FloatArray:
        """Return a pinhole K, scaled independently to ``width`` and ``height``.

        Distortion parameters remain available in :attr:`params`; this method
        only extracts the focal lengths and principal point used by Gaussian
        rasterization.  Equirectangular cameras do not have a pinhole K.
        """

        model = CAMERA_MODEL_BY_ID[self.model_id]
        if not model.focal_indices:
            raise ValueError(f"camera model {self.model!r} has no pinhole intrinsics")

        if len(model.focal_indices) == 1:
            fx = fy = float(self.params[model.focal_indices[0]])
        else:
            fx = float(self.params[model.focal_indices[0]])
            fy = float(self.params[model.focal_indices[1]])
        cx = float(self.params[model.principal_point_indices[0]])
        cy = float(self.params[model.principal_point_indices[1]])

        target_width = self.width if width is None else width
        target_height = self.height if height is None else height
        if target_width <= 0 or target_height <= 0:
            raise ValueError("target image dimensions must be positive")
        scale_x = target_width / self.width
        scale_y = target_height / self.height
        return np.array(
            [
                [fx * scale_x, 0.0, cx * scale_x],
                [0.0, fy * scale_y, cy * scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def K(self) -> FloatArray:
        """Unscaled pinhole intrinsic matrix."""

        return self.intrinsic_matrix()


@dataclass(frozen=True, slots=True)
class Image:
    id: int
    qvec: FloatArray
    tvec: FloatArray
    camera_id: int
    name: str
    xys: FloatArray
    point3D_ids: IntArray

    @property
    def rotation_matrix(self) -> FloatArray:
        return qvec_to_rotation_matrix(self.qvec)

    def qvec2rotmat(self) -> FloatArray:
        """Compatibility spelling used by COLMAP's Python helper."""

        return self.rotation_matrix

    @property
    def world_to_camera(self) -> FloatArray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation_matrix
        matrix[:3, 3] = self.tvec
        return matrix

    @property
    def camera_to_world(self) -> FloatArray:
        rotation = self.rotation_matrix
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation.T
        matrix[:3, 3] = -(rotation.T @ self.tvec)
        return matrix

    @property
    def w2c(self) -> FloatArray:
        return self.world_to_camera

    @property
    def c2w(self) -> FloatArray:
        return self.camera_to_world


@dataclass(frozen=True, slots=True)
class Point3D:
    id: int
    xyz: FloatArray
    rgb: NDArray[np.uint8]
    error: float
    image_ids: IntArray
    point2D_idxs: IntArray


@dataclass(frozen=True, slots=True)
class ColmapModel:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points3D: dict[int, Point3D]

    @property
    def points3d(self) -> dict[int, Point3D]:
        return self.points3D


def qvec_to_rotation_matrix(qvec: NDArray[np.floating]) -> FloatArray:
    """Convert COLMAP's ``(qw, qx, qy, qz)`` quaternion to a rotation."""

    quaternion = np.asarray(qvec, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"qvec must have shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("qvec must be finite and non-zero")
    qw, qx, qy, qz = quaternion / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
            ],
            [
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qw * qx),
            ],
            [
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _read_exact(file: BinaryIO, size: int, context: str) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ColmapFormatError(
            f"truncated COLMAP binary while reading {context}: "
            f"expected {size} bytes, got {len(data)}"
        )
    return data


def _unpack(file: BinaryIO, fmt: str, context: str) -> tuple[object, ...]:
    binary_format = struct.Struct("<" + fmt)
    return binary_format.unpack(_read_exact(file, binary_format.size, context))


def _read_c_string(file: BinaryIO, context: str) -> str:
    name = bytearray()
    while True:
        char = _read_exact(file, 1, context)
        if char == b"\x00":
            try:
                return name.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ColmapFormatError(f"invalid UTF-8 in {context}") from error
        name.extend(char)


def _insert_unique(records: dict[int, object], key: int, value: object, kind: str) -> None:
    if key in records:
        raise ColmapFormatError(f"duplicate {kind} id {key}")
    records[key] = value


def read_cameras_binary(path: str | Path) -> dict[int, Camera]:
    """Read every camera record from ``cameras.bin``."""

    cameras: dict[int, Camera] = {}
    with Path(path).open("rb") as file:
        (num_cameras,) = _unpack(file, "Q", "camera count")
        for record_index in range(int(num_cameras)):
            camera_id, model_id, width, height = _unpack(
                file, "iiQQ", f"camera {record_index} header"
            )
            camera_model = CAMERA_MODEL_BY_ID.get(int(model_id))
            if camera_model is None:
                raise ColmapFormatError(f"unknown COLMAP camera model id {model_id}")
            params = np.asarray(
                _unpack(
                    file,
                    "d" * camera_model.num_params,
                    f"camera {camera_id} parameters",
                ),
                dtype=np.float64,
            )
            camera = Camera(
                id=int(camera_id),
                model_id=int(model_id),
                model=camera_model.name,
                width=int(width),
                height=int(height),
                params=params,
            )
            if camera.width <= 0 or camera.height <= 0:
                raise ColmapFormatError(
                    f"camera {camera.id} has invalid size {camera.width}x{camera.height}"
                )
            _insert_unique(cameras, camera.id, camera, "camera")
    return cameras


def read_images_binary(path: str | Path) -> dict[int, Image]:
    """Read registered images, including all 2D observations, from ``images.bin``."""

    images: dict[int, Image] = {}
    with Path(path).open("rb") as file:
        (num_images,) = _unpack(file, "Q", "registered image count")
        for record_index in range(int(num_images)):
            values = _unpack(file, "i7di", f"image {record_index} header")
            image_id = int(values[0])
            qvec = np.asarray(values[1:5], dtype=np.float64)
            tvec = np.asarray(values[5:8], dtype=np.float64)
            camera_id = int(values[8])
            name = _read_c_string(file, f"image {image_id} name")
            (num_points2d,) = _unpack(file, "Q", f"image {image_id} point count")
            point_count = int(num_points2d)
            xys = np.empty((point_count, 2), dtype=np.float64)
            point3d_ids = np.empty((point_count,), dtype=np.int64)
            for point_index in range(point_count):
                x, y, point3d_id = _unpack(
                    file, "ddq", f"image {image_id} point {point_index}"
                )
                xys[point_index] = (float(x), float(y))
                point3d_ids[point_index] = int(point3d_id)
            image = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                xys=xys,
                point3D_ids=point3d_ids,
            )
            _insert_unique(images, image.id, image, "image")
    return images


def read_points3d_binary(path: str | Path) -> dict[int, Point3D]:
    """Read every point and its observation track from ``points3D.bin``."""

    points: dict[int, Point3D] = {}
    with Path(path).open("rb") as file:
        (num_points,) = _unpack(file, "Q", "3D point count")
        for record_index in range(int(num_points)):
            values = _unpack(file, "QdddBBBd", f"3D point {record_index} header")
            point_id = int(values[0])
            (track_length,) = _unpack(file, "Q", f"3D point {point_id} track length")
            image_ids = np.empty((int(track_length),), dtype=np.int64)
            point2d_idxs = np.empty((int(track_length),), dtype=np.int64)
            for track_index in range(int(track_length)):
                image_id, point2d_idx = _unpack(
                    file, "ii", f"3D point {point_id} track element {track_index}"
                )
                image_ids[track_index] = int(image_id)
                point2d_idxs[track_index] = int(point2d_idx)
            point = Point3D(
                id=point_id,
                xyz=np.asarray(values[1:4], dtype=np.float64),
                rgb=np.asarray(values[4:7], dtype=np.uint8),
                error=float(values[7]),
                image_ids=image_ids,
                point2D_idxs=point2d_idxs,
            )
            _insert_unique(points, point.id, point, "3D point")
    return points


# Preserve COLMAP's own capitalization for callers porting existing code.
read_points3D_binary = read_points3d_binary


def find_colmap_model_dir(scene_dir: str | Path) -> Path:
    """Find ``sparse/0``, ``sparse``, or a directly supplied model directory."""

    root = Path(scene_dir).expanduser()
    candidates = (root, root / "sparse" / "0", root / "sparse")
    for candidate in candidates:
        if (candidate / "cameras.bin").is_file() and (
            candidate / "images.bin"
        ).is_file():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not find COLMAP cameras.bin/images.bin in {tried}")


def read_colmap_model(
    model_dir: str | Path, *, load_points: bool = True
) -> ColmapModel:
    """Read a COLMAP sparse model directory.

    ``load_points=False`` is useful for inexpensive camera-only inspection of a
    large scene; the three individual readers always consume their full files.
    """

    directory = find_colmap_model_dir(model_dir)
    cameras = read_cameras_binary(directory / "cameras.bin")
    images = read_images_binary(directory / "images.bin")
    missing_camera_ids = sorted(
        {image.camera_id for image in images.values()} - cameras.keys()
    )
    if missing_camera_ids:
        raise ColmapFormatError(
            f"registered images reference missing cameras {missing_camera_ids}"
        )
    points_path = directory / "points3D.bin"
    if load_points:
        if not points_path.is_file():
            raise FileNotFoundError(points_path)
        points = read_points3d_binary(points_path)
    else:
        points = {}
    return ColmapModel(cameras=cameras, images=images, points3D=points)


# Short alias matching common COLMAP helper modules.
read_model = read_colmap_model
