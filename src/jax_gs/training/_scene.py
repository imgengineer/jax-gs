"""Scene normalization: the transform applied to a capture and the render size it implies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TrainConfig
from ..data import ColmapScene
from ..data.normalize import (
    _as_similarity_matrix,
    normalize_scene,
    transform_cameras,
    transform_points,
)


@dataclass(frozen=True)
class SceneTransform:
    """Similarity mapping world coordinates into training coordinates."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _as_similarity_matrix(self.matrix))

    def points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        transformed = transform_points(self.matrix, points)
        if np.issubdtype(points.dtype, np.floating):
            transformed = transformed.astype(points.dtype, copy=False)
        return transformed

    def camera_to_world(self, camera_to_world: np.ndarray) -> np.ndarray:
        cameras = np.asarray(camera_to_world)
        if cameras.ndim < 2 or cameras.shape[-2:] != (4, 4):
            raise ValueError(
                f"camera_to_world must have shape (..., 4, 4), got {cameras.shape}"
            )
        batched = cameras.reshape((-1, 4, 4))
        transformed = transform_cameras(self.matrix, batched)
        if np.issubdtype(cameras.dtype, np.floating):
            transformed = transformed.astype(cameras.dtype, copy=False)
        return transformed.reshape(cameras.shape)

    def world_to_camera(self, world_to_camera: np.ndarray) -> np.ndarray:
        cameras = np.asarray(world_to_camera)
        if cameras.ndim < 2 or cameras.shape[-2:] != (4, 4):
            raise ValueError(
                f"world_to_camera must have shape (..., 4, 4), got {cameras.shape}"
            )
        transformed = self.camera_to_world(np.linalg.inv(cameras))
        result = np.linalg.inv(transformed)
        if np.issubdtype(cameras.dtype, np.floating):
            result = result.astype(cameras.dtype, copy=False)
        return result


def _training_render_size(
    config: TrainConfig,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
) -> tuple[int, int]:
    """Resolve the static training render height and width."""

    if config.data.patch_size is not None:
        patch_size = int(config.data.patch_size)
        return patch_size, patch_size
    if image_height is None or image_width is None:
        raise ValueError(
            "image_height and image_width are required when patch_size=None"
        )
    image_height = int(image_height)
    image_width = int(image_width)
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be positive")
    return image_height, image_width


def _legacy_scene_transform(scene: ColmapScene) -> SceneTransform:
    """Return the mean/max transform used by checkpoints before format v6."""

    centers = scene.camtoworlds[:, :3, 3].astype(np.float32)
    center = np.mean(centers, axis=0)
    scale = float(np.max(np.linalg.norm(centers - center[None, :], axis=-1)))
    scale = max(scale, 1.0e-6)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] /= scale
    matrix[:3, 3] = -center / scale
    return SceneTransform(matrix)


def _legacy_training_scene_scale(scene: ColmapScene) -> float:
    """Reproduce the pre-v6 float32 scene-extent calculation exactly."""

    centers = scene.camtoworlds[:, :3, 3].astype(np.float32)
    center = np.mean(centers, axis=0)
    normalization_scale = float(
        np.max(np.linalg.norm(centers - center[None, :], axis=-1))
    )
    normalized = (centers - center[None, :]) / max(normalization_scale, 1.0e-6)
    normalized_center = np.mean(normalized, axis=0)
    extent = np.max(np.linalg.norm(normalized - normalized_center[None, :], axis=-1))
    return float(extent * 1.1)


def compute_scene_transform(
    scene: ColmapScene, *, normalize_world_space: bool = True
) -> SceneTransform:
    """Build current-main's focus/median and point-PCA world transform."""

    if not normalize_world_space:
        return SceneTransform(np.eye(4, dtype=np.float64))
    _, _, matrix = normalize_scene(scene.camtoworlds, scene.points)
    return SceneTransform(matrix)


def _training_scene_scale(
    scene: ColmapScene,
    transform: SceneTransform,
    *,
    global_scale: float = 1.0,
) -> float:
    """Return current-main's 1.1-margin extent in training coordinates."""

    centers = transform.points(scene.camtoworlds[:, :3, 3].astype(np.float64))
    center = np.mean(centers, axis=0)
    extent = np.max(np.linalg.norm(centers - center[None, :], axis=-1))
    return float(extent * 1.1 * global_scale)


def _scene_training_render_size(
    scene: ColmapScene, config: TrainConfig
) -> tuple[int, int]:
    if config.data.patch_size is not None:
        return _training_render_size(config)
    indices = scene.indices("train", config.data.test_every)
    if len(indices) == 0:
        raise ValueError("training split contains no images")
    sizes = {
        (scene.images[int(index)].height, scene.images[int(index)].width)
        for index in indices
    }
    if len(sizes) != 1:
        raise ValueError(
            "full-image training requires all training images to share one "
            "height and width; set patch_size for mixed-resolution data"
        )
    image_height, image_width = next(iter(sizes))
    return _training_render_size(
        config,
        image_height=image_height,
        image_width=image_width,
    )
