# pyright: reportMissingImports=false

"""COLMAP scene metadata and Grain-compatible image loading."""

from __future__ import annotations

import operator
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from .colmap import ColmapModel, find_colmap_model_dir, read_colmap_model

Split = Literal["train", "test", "val", "all"]
Size = int | tuple[int, int]


@dataclass(frozen=True, slots=True)
class SceneImage:
    """One registered image with intrinsics scaled to its actual file size."""

    index: int
    image_id: int
    camera_id: int
    camera_index: int
    name: str
    path: Path
    width: int
    height: int
    K: NDArray[np.float64]
    w2c: NDArray[np.float64]
    c2w: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ColmapScene:
    root: Path
    model_dir: Path
    image_dir: Path
    model: ColmapModel
    images: tuple[SceneImage, ...]

    def indices(self, split: Split = "all", test_every: int = 8) -> NDArray[np.int64]:
        return mipnerf360_split_indices(len(self.images), split, test_every)

    @property
    def image_names(self) -> tuple[str, ...]:
        return tuple(image.name for image in self.images)

    @property
    def image_paths(self) -> tuple[Path, ...]:
        return tuple(image.path for image in self.images)

    @property
    def Ks(self) -> NDArray[np.float64]:
        return _stack_or_empty((image.K for image in self.images), (0, 3, 3))

    @property
    def worldtocams(self) -> NDArray[np.float64]:
        return _stack_or_empty((image.w2c for image in self.images), (0, 4, 4))

    @property
    def camtoworlds(self) -> NDArray[np.float64]:
        return _stack_or_empty((image.c2w for image in self.images), (0, 4, 4))

    @property
    def points(self) -> NDArray[np.float32]:
        if not self.model.points3D:
            return np.empty((0, 3), dtype=np.float32)
        return np.stack(
            [
                self.model.points3D[point_id].xyz
                for point_id in sorted(self.model.points3D)
            ],
            axis=0,
        ).astype(np.float32)

    @property
    def points_rgb(self) -> NDArray[np.uint8]:
        if not self.model.points3D:
            return np.empty((0, 3), dtype=np.uint8)
        return np.stack(
            [
                self.model.points3D[point_id].rgb
                for point_id in sorted(self.model.points3D)
            ],
            axis=0,
        ).astype(np.uint8)

    @property
    def scene_scale(self) -> float:
        if not self.images:
            return 0.0
        centers = self.camtoworlds[:, :3, 3]
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        return float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max())


def _stack_or_empty(arrays: Any, empty_shape: tuple[int, ...]) -> NDArray[np.float64]:
    values = tuple(arrays)
    if not values:
        return np.empty(empty_shape, dtype=np.float64)
    return np.stack(values, axis=0)


def mipnerf360_split_indices(
    num_images: int, split: Split = "train", test_every: int = 8
) -> NDArray[np.int64]:
    """Return the filename-sorted Mip-NeRF360 train/evaluation split.

    Every ``test_every``-th image, starting with image zero, is held out.  This
    matches the gsplat example loader and the established Mip-NeRF360 metrics.
    """

    if num_images < 0:
        raise ValueError("num_images must be non-negative")
    if test_every <= 0:
        raise ValueError("test_every must be positive")
    if split not in {"train", "test", "val", "all"}:
        raise ValueError(f"unknown split {split!r}")
    indices = np.arange(num_images, dtype=np.int64)
    if split == "all":
        return indices
    is_evaluation = indices % test_every == 0
    return indices[~is_evaluation] if split == "train" else indices[is_evaluation]


def _resolve_image_dir(
    scene_root: Path, image_dir: str | Path | None, factor: int
) -> Path:
    if factor <= 0:
        raise ValueError("factor must be positive")
    if image_dir is None:
        directory = scene_root / ("images" if factor == 1 else f"images_{factor}")
    else:
        directory = Path(image_dir).expanduser()
        if not directory.is_absolute():
            directory = scene_root / directory
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    return directory


def _image_key(name: str) -> tuple[str, str]:
    relative_path = PurePosixPath(name.replace("\\", "/"))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"COLMAP image name must be relative: {name!r}")
    parent = "" if str(relative_path.parent) == "." else str(relative_path.parent)
    return parent.casefold(), relative_path.stem.casefold()


def _index_image_files(
    image_dir: Path,
) -> tuple[dict[str, Path], dict[tuple[str, str], tuple[Path, ...]]]:
    exact: dict[str, Path] = {}
    by_stem_lists: dict[tuple[str, str], list[Path]] = {}
    for path in image_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(image_dir).as_posix()
        exact[relative.casefold()] = path
        by_stem_lists.setdefault(_image_key(relative), []).append(path)
    by_stem = {key: tuple(sorted(paths)) for key, paths in by_stem_lists.items()}
    return exact, by_stem


def _resolve_image_path(
    name: str,
    exact: dict[str, Path],
    by_stem: dict[tuple[str, str], tuple[Path, ...]],
) -> Path:
    normalized = PurePosixPath(name.replace("\\", "/")).as_posix()
    path = exact.get(normalized.casefold())
    if path is not None:
        return path
    candidates = by_stem.get(_image_key(name), ())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no image file matching COLMAP name {name!r}")
    raise ValueError(f"multiple image files match COLMAP name {name!r}: {candidates}")


_JPEG_START_OF_FRAME = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def _jpeg_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        if file.read(2) != b"\xff\xd8":
            raise ValueError(f"not a JPEG file: {path}")
        while True:
            prefix = file.read(1)
            while prefix and prefix != b"\xff":
                prefix = file.read(1)
            if not prefix:
                break
            marker_bytes = file.read(1)
            while marker_bytes == b"\xff":
                marker_bytes = file.read(1)
            if not marker_bytes:
                break
            marker = marker_bytes[0]
            if marker == 0xD9 or marker == 0xDA:
                break
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue
            length_bytes = file.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if segment_length < 2:
                break
            if marker in _JPEG_START_OF_FRAME:
                frame_header = file.read(5)
                if len(frame_header) != 5:
                    break
                height, width = struct.unpack(">HH", frame_header[1:])
                # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
                return int(width), int(height)
            file.seek(segment_length - 2, 1)
    raise ValueError(f"could not read JPEG dimensions: {path}")


def image_size(path: str | Path) -> tuple[int, int]:
    """Read ``(width, height)`` without decoding image pixels when possible."""

    image_path = Path(path)
    with image_path.open("rb") as file:
        header = file.read(32)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        width, height = struct.unpack(">II", header[16:24])
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        return int(width), int(height)
    if header.startswith(b"\xff\xd8"):
        return _jpeg_size(image_path)
    if header[:6] in {b"GIF87a", b"GIF89a"} and len(header) >= 10:
        width, height = struct.unpack("<HH", header[6:10])
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        return int(width), int(height)

    try:
        from PIL import Image as PilImage
    except ImportError as error:  # pragma: no cover - exercised without Pillow
        raise ImportError(
            f"Pillow is required to inspect this image format: {image_path}"
        ) from error
    with PilImage.open(image_path) as image:
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        return int(image.width), int(image.height)


def load_colmap_scene(
    scene_dir: str | Path,
    *,
    image_dir: str | Path | None = None,
    factor: int = 1,
    load_points: bool = True,
) -> ColmapScene:
    """Load sorted COLMAP metadata and scale K for each actual image file."""

    root = Path(scene_dir).expanduser()
    model_dir = find_colmap_model_dir(root)
    # A direct sparse-model path has no image directory; use its scene parent.
    if model_dir == root:
        if root.name == "0" and root.parent.name == "sparse":
            root = root.parent.parent
        elif root.name == "sparse":
            root = root.parent
    resolved_image_dir = _resolve_image_dir(root, image_dir, factor)
    model = read_colmap_model(model_dir, load_points=load_points)
    exact_paths, paths_by_stem = _index_image_files(resolved_image_dir)

    registered_images = sorted(
        model.images.values(), key=lambda image: (image.name, image.id)
    )
    if len({image.name for image in registered_images}) != len(registered_images):
        raise ValueError("COLMAP registered image names must be unique")
    camera_id_to_index = {
        camera_id: index for index, camera_id in enumerate(sorted(model.cameras))
    }

    scene_images: list[SceneImage] = []
    for index, registered_image in enumerate(registered_images):
        path = _resolve_image_path(registered_image.name, exact_paths, paths_by_stem)
        width, height = image_size(path)
        camera = model.cameras[registered_image.camera_id]
        scene_images.append(
            SceneImage(
                index=index,
                image_id=registered_image.id,
                camera_id=registered_image.camera_id,
                camera_index=camera_id_to_index[registered_image.camera_id],
                name=registered_image.name,
                path=path,
                width=width,
                height=height,
                K=camera.intrinsic_matrix(width=width, height=height),
                w2c=registered_image.w2c,
                c2w=registered_image.c2w,
            )
        )
    if not scene_images:
        raise ValueError(f"no registered images found in {model_dir}")
    return ColmapScene(
        root=root,
        model_dir=model_dir,
        image_dir=resolved_image_dir,
        model=model,
        images=tuple(scene_images),
    )


def _normalize_size(value: Size | None, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        size = (value, value)
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or (height, width)")
        size = (operator.index(value[0]), operator.index(value[1]))
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return size


class ColmapDataSource:
    """Deterministic random-access source compatible with Grain MapDataset.

    Images are decoded lazily.  ``crop_size`` applies a center crop and
    ``resize`` is the final ``(height, width)``.  If neither is supplied, all
    selected source images must already share one size, guaranteeing a static
    output shape suitable for JAX batching and JIT compilation.
    """

    def __init__(
        self,
        scene: ColmapScene | str | Path,
        *,
        split: Split = "train",
        test_every: int = 8,
        crop_size: Size | None = None,
        resize: Size | None = None,
        image_dir: str | Path | None = None,
        factor: int = 1,
        cache_images: bool = False,
        uint8: bool = False,
    ) -> None:
        if not isinstance(scene, ColmapScene):
            scene = load_colmap_scene(
                scene, image_dir=image_dir, factor=factor, load_points=False
            )
        self.scene = scene
        self.split = split
        self.test_every = test_every
        self.indices = scene.indices(split, test_every)
        self.crop_size = _normalize_size(crop_size, "crop_size")
        self.resize = _normalize_size(resize, "resize")
        self.cache_images = bool(cache_images)
        self.uint8 = bool(uint8)
        self._cache: dict[int, dict[str, Any]] | None = (
            {} if self.cache_images else None
        )

        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        selected = [scene.images[int(index)] for index in self.indices]
        shape_source = selected if selected else list(scene.images)
        if self.crop_size is not None:
            crop_height, crop_width = self.crop_size
            too_small = [
                image.name
                for image in shape_source
                if image.height < crop_height or image.width < crop_width
            ]
            if too_small:
                raise ValueError(
                    f"crop_size {self.crop_size} exceeds images {too_small[:3]}"
                )
            cropped_shape = self.crop_size
        else:
            source_shapes = {(image.height, image.width) for image in shape_source}
            if self.resize is None and len(source_shapes) != 1:
                raise ValueError(
                    "selected images have different sizes; provide crop_size or resize"
                )
            cropped_shape = next(iter(source_shapes))
        self.image_shape = (*(self.resize or cropped_shape), 3)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        item_index = operator.index(index)
        if item_index < 0 or item_index >= len(self):
            raise IndexError(f"index {item_index} out of range for {len(self)} images")
        if self._cache is not None and item_index in self._cache:
            return self._cache[item_index]
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        scene_index = int(self.indices[item_index])
        metadata = self.scene.images[scene_index]

        try:
            from PIL import Image as PilImage
        except ImportError as error:  # pragma: no cover - dependency is declared
            raise ImportError("Pillow is required to decode training images") from error
        with PilImage.open(metadata.path) as encoded_image:
            image = encoded_image.convert("RGB")
            if image.size != (metadata.width, metadata.height):
                raise ValueError(
                    f"image dimensions changed after scene load: {metadata.path}"
                )
            K = metadata.K.copy()
            if self.crop_size is not None:
                crop_height, crop_width = self.crop_size
                left = (image.width - crop_width) // 2
                top = (image.height - crop_height) // 2
                image = image.crop((left, top, left + crop_width, top + crop_height))
                K[0, 2] -= left
                K[1, 2] -= top
            if self.resize is not None:
                target_height, target_width = self.resize
                scale_x = target_width / image.width
                scale_y = target_height / image.height
                image = image.resize(
                    (target_width, target_height), PilImage.Resampling.BILINEAR
                )
                K[0, :] *= scale_x
                K[1, :] *= scale_y
            if self.uint8:
                pixels = np.asarray(image, dtype=np.uint8)
            else:
                pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)

        if pixels.shape != self.image_shape:
            raise ValueError(
                f"expected static image shape {self.image_shape}, got {pixels.shape}"
            )
        result = {
            "image": pixels,
            "K": K.astype(np.float32),
            "w2c": metadata.w2c.astype(np.float32),
            "c2w": metadata.c2w.astype(np.float32),
            "image_index": np.int32(scene_index),
            "dataset_index": np.int32(item_index),
            "image_id": np.int64(metadata.image_id),
            "image_name": metadata.name,
            "camera_id": np.int64(metadata.camera_id),
            "camera_index": np.int32(metadata.camera_index),
        }
        if self._cache is not None:
            self._cache[item_index] = result
        return result

    def __repr__(self) -> str:
        return (
            f"ColmapDataSource(scene={str(self.scene.root)!r}, split={self.split!r}, "
            f"size={len(self)}, image_shape={self.image_shape})"
        )


def create_grain_dataset(
    scene: ColmapScene | str | Path,
    *,
    split: Split = "train",
    test_every: int = 8,
    crop_size: Size | None = None,
    resize: Size | None = None,
    image_dir: str | Path | None = None,
    factor: int = 1,
    cache_images: bool = False,
    uint8: bool = False,
    shuffle: bool = False,
    seed: int = 0,
    repeat: bool = False,
    batch_size: int | None = None,
    drop_remainder: bool = False,
) -> Any:
    """Create a Grain ``MapDataset`` backed by :class:`ColmapDataSource`."""

    try:
        import grain
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ImportError(
            "install the 'grain' package to create a MapDataset"
        ) from error

    source = ColmapDataSource(
        scene,
        split=split,
        test_every=test_every,
        crop_size=crop_size,
        resize=resize,
        image_dir=image_dir,
        factor=factor,
        cache_images=cache_images,
        uint8=uint8,
    )
    dataset = grain.MapDataset.source(source)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if repeat:
        dataset = dataset.repeat()
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset


create_dataset = create_grain_dataset
