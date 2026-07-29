from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace
import gc
import math
import operator
from pathlib import Path
import time
from typing import Any, Callable, Iterator

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image

from ..capacity import compact_training_state, resize_training_state
from ..checkpoints import (
    load_checkpoint_active_prefix,
    load_checkpoint_config,
    load_checkpoint_intersection_capacity,
    load_checkpoint_scene_transform,
    load_checkpoint_storage_capacity,
    restore_checkpoint,
    save_checkpoint,
)
from ..config import RasterizationConfig, TrainConfig
from ..data import ColmapScene, create_grain_dataset, load_colmap_scene
from ..data.normalize import (
    _as_similarity_matrix,
    normalize_scene,
    transform_cameras,
    transform_points,
)
from ..losses import l1_loss, opacity_reg_loss, psnr, scale_reg_loss, ssim
from ..model import GaussianModel
from ..optimizers import (
    create_optimizer,
    create_row_selective_optimizer,
    create_visible_adam_optimizer,
    mask_inactive_gradients,
)
from ..rasterization import _automatic_intersection_capacity, rasterization
from ..strategy import (
    build_densification_stats,
    DefaultStrategy,
    DensificationStats,
    MCMCStrategy,
    StrategyState,
    reset_opacities,
)
from ..strategy.ops import mcmc_position_perturbation
from ..two_dgs import rasterization_2dgs
from .appearance import (
    APPEARANCE_FEATURE_DIM,
    AppearanceOptModule,
    create_appearance_optimizer,
)
from .pose import CameraOptModule
from .schedulers import TwoStageScheduler


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
                "camera_to_world must have shape (..., 4, 4), "
                f"got {cameras.shape}"
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
                "world_to_camera must have shape (..., 4, 4), "
                f"got {cameras.shape}"
            )
        transformed = self.camera_to_world(np.linalg.inv(cameras))
        result = np.linalg.inv(transformed)
        if np.issubdtype(cameras.dtype, np.floating):
            result = result.astype(cameras.dtype, copy=False)
        return result


@dataclass
class TrainingResult:
    model: GaussianModel
    final_step: int
    output_dir: Path
    checkpoint: Path | None
    metrics: dict[str, float]
    pose_adjust: CameraOptModule | None = None
    appearance: AppearanceOptModule | None = None


class TrainingSafetyState(nnx.Module):
    """Device-resident sticky overflow state for one training run."""

    def __init__(self) -> None:
        self.max_overflow_tiles = nnx.Variable(jnp.array(0, jnp.int32))
        self.intersection_overflow_seen = nnx.Variable(jnp.array(False))


@dataclass
class _PendingTrainStep:
    images: np.ndarray
    intrinsics: np.ndarray
    viewmats: np.ndarray
    camtoworlds: np.ndarray | None
    image_ids: np.ndarray | None
    key: jax.Array
    strategy_key: jax.Array
    sh_degree: jax.Array
    metrics: dict[str, jax.Array]


def _intersection_bucket_capacity(
    required: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Return the smallest configured power-of-two bucket covering required."""

    required = int(required)
    minimum = int(minimum)
    maximum = int(maximum)
    if required < 0:
        raise ValueError("required intersection capacity cannot be negative")
    if minimum <= 0 or minimum & (minimum - 1):
        raise ValueError("minimum intersection bucket must be a power of two")
    if maximum <= 0:
        raise ValueError("maximum intersection capacity must be positive")
    if required > maximum:
        raise RuntimeError(
            f"required intersections {required} exceed configured maximum {maximum}"
        )
    bucket = min(minimum, maximum)
    while bucket < required:
        bucket = min(bucket * 2, maximum)
    return bucket


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


def _training_intersection_limit(
    config: TrainConfig,
    physical_capacity: int,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
) -> int:
    height, width = _training_render_size(
        config, image_height=image_height, image_width=image_width
    )
    tile_width = math.ceil(width / config.rasterizer.tile_size)
    tile_height = math.ceil(height / config.rasterizer.tile_size)
    tile_count = tile_width * tile_height
    return _automatic_intersection_capacity(
        physical_capacity, tile_count, config.rasterizer
    )


def _training_config_with_intersection_capacity(
    config: TrainConfig, capacity: int
) -> TrainConfig:
    return replace(
        config,
        rasterizer=replace(config.rasterizer, max_intersections=int(capacity)),
    )


def _mcmc_required_capacity(active_count: int, config: TrainConfig) -> int:
    """Return the scheduled MCMC population target from the active count."""

    active_count = int(active_count)
    target_count = min(
        config.strategy.cap_max, (active_count * 105) // 100
    )
    births = min(
        max(target_count - active_count, 0),
        config.strategy.max_new_per_refine,
    )
    return active_count + births


def _raise_training_overflow(tile_count: np.ndarray, intersection: np.ndarray) -> None:
    raise RuntimeError(
        "rasterization overflow would produce truncated gradients "
        f"(tiles={int(tile_count)}, intersection={bool(intersection)}); increase "
        "max_intersections or max_gaussians_per_tile"
    )


def _training_overflow_status(
    safety_state: TrainingSafetyState,
) -> tuple[int, bool]:
    max_overflow_tiles, intersection_overflow_seen = jax.device_get(
        (
            safety_state.max_overflow_tiles[...],
            safety_state.intersection_overflow_seen[...],
        )
    )
    return int(max_overflow_tiles), bool(intersection_overflow_seen)


def _pending_overflow_suffix(
    pending: list[_PendingTrainStep],
) -> tuple[int, int]:
    values = jax.device_get(
        tuple(
            (
                record.metrics["overflow_tiles"],
                record.metrics["intersection_overflow"],
                record.metrics["intersection_required_count"],
            )
            for record in pending
        )
    )
    for index, (overflow_tiles, intersection, required) in enumerate(values):
        if int(overflow_tiles) > 0:
            _raise_training_overflow(
                np.asarray(overflow_tiles), np.asarray(intersection)
            )
        if bool(intersection):
            return index, int(required)
    raise RuntimeError("sticky intersection overflow has no matching pending step")


def estimate_rasterization_memory_bytes(
    capacity: int,
    width: int,
    height: int,
    rasterizer: RasterizationConfig,
    *,
    channels: int = 3,
) -> int:
    """Conservative forward-only rasterization workspace estimate."""

    tile_width = math.ceil(width / rasterizer.tile_size)
    tile_height = math.ceil(height / rasterizer.tile_size)
    tile_count = tile_width * tile_height
    projection = capacity * 64
    compositing = (
        min(rasterizer.tile_batch_size, tile_count)
        * rasterizer.max_gaussians_per_tile
        * rasterizer.tile_size**2
        * (32 + channels * 4)
    )
    intersections = 0
    tile_scan = 0
    if rasterizer.backend != "reference":
        intersection_capacity = _automatic_intersection_capacity(
            capacity, tile_count, rasterizer
        )
        intersections = intersection_capacity * 96
    else:
        tile_scan = min(rasterizer.tile_batch_size, tile_count) * capacity * 16
    outputs = width * height * (channels + 1) * 4 * 3
    return int(
        projection
        + compositing
        + intersections
        + tile_scan
        + outputs
        + 256 * 2**20
    )


def _parameter_bytes(config: TrainConfig, physical_capacity: int) -> int:
    color_floats = (
        APPEARANCE_FEATURE_DIM + 3
        if config.app_opt
        else (config.model.sh_degree + 1) ** 2 * 3
    )
    floats_per_gaussian = 3 + 3 + 4 + 1 + color_floats
    return physical_capacity * floats_per_gaussian * 4


def _training_state_bytes(config: TrainConfig, physical_capacity: int) -> int:
    parameter_bytes = _parameter_bytes(config, physical_capacity)
    strategy_bytes = physical_capacity * (3 * 4 + 1)
    # Model parameters, two Adam moments, active mask, and strategy statistics.
    return parameter_bytes * 3 + strategy_bytes


def estimate_training_memory_bytes(
    config: TrainConfig,
    *,
    physical_capacity: int | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
) -> int:
    """Estimate peak memory for one physical Gaussian storage bucket."""

    if physical_capacity is None:
        physical_capacity = config.model.bucket_capacity()
    physical_capacity = int(physical_capacity)
    if not 0 < physical_capacity <= config.model.capacity:
        raise ValueError(
            "physical_capacity must be positive and not exceed the logical maximum"
        )
    render_height, render_width = _training_render_size(
        config, image_height=image_height, image_width=image_width
    )

    parameter_bytes = _parameter_bytes(config, physical_capacity)
    # Parameters, gradients, two Adam moments, updates, and compiler temporaries.
    model_working_set = parameter_bytes * 6
    # Dense projection still visits the complete physical bucket before the
    # bounded visible set is packed. Include forward outputs, reverse-mode
    # residuals, and covariance temporaries explicitly for large buckets.
    projection_working_set = (
        physical_capacity * 192 * config.data.batch_size
    )
    strategy_bytes = physical_capacity * (3 * 4 + 1)
    tile_pixels = config.rasterizer.tile_size**2
    raster_workspace = (
        config.rasterizer.tile_batch_size
        * config.rasterizer.max_gaussians_per_tile
        * tile_pixels
        * 64
    )
    intersection_workspace = 0
    if config.rasterizer.backend != "reference":
        tile_width = math.ceil(render_width / config.rasterizer.tile_size)
        tile_height = math.ceil(render_height / config.rasterizer.tile_size)
        render_tiles = tile_width * tile_height
        intersection_capacity = config.rasterizer.max_intersections
        if intersection_capacity is None:
            intersection_capacity = max(
                1,
                min(
                    render_tiles
                    * (config.rasterizer.max_gaussians_per_tile + 1),
                    max(
                        config.rasterizer.max_gaussians_per_tile + 1,
                        physical_capacity * 8,
                    ),
                ),
            )
        # Padded ids plus conservative temporary storage for lexicographic sort.
        intersection_workspace = intersection_capacity * 96
    ut_workspace = 0
    if config.with_ut or config.with_eval3d:
        ut_workspace = (
            min(physical_capacity, config.rasterizer.ut_chunk_size)
            * 7
            * 3
            * 4
            * 12
            * config.data.batch_size
        )
    return int(
        model_working_set
        + projection_working_set
        + strategy_bytes
        + raster_workspace
        + intersection_workspace
        + ut_workspace
        + 512 * 2**20
    )


def estimate_bucket_transition_memory_bytes(
    config: TrainConfig,
    old_capacity: int,
    new_capacity: int,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
) -> int:
    """Estimate the old/new coexistence peak while growing a storage bucket."""

    old_capacity = int(old_capacity)
    new_capacity = int(new_capacity)
    if not 0 < old_capacity < new_capacity <= config.model.capacity:
        raise ValueError("invalid bucket transition")
    target_peak = estimate_training_memory_bytes(
        config,
        physical_capacity=new_capacity,
        image_height=image_height,
        image_width=image_width,
    )
    # Migration now pads the existing optimizer and strategy state directly;
    # it does not initialize a second all-zero Adam state. Runtime projection
    # and gradient workspaces are not live during the host-controlled resize,
    # so guard the larger of the steady target and old/new persistent overlap.
    migration_peak = (
        _training_state_bytes(config, old_capacity)
        + _training_state_bytes(config, new_capacity)
        + 256 * 2**20
    )
    return int(max(target_peak, migration_peak))


def _check_memory_budget(
    config: TrainConfig,
    *,
    physical_capacity: int,
    label: str = "training",
    image_height: int | None = None,
    image_width: int | None = None,
) -> int:
    estimate = estimate_training_memory_bytes(
        config,
        physical_capacity=physical_capacity,
        image_height=image_height,
        image_width=image_width,
    )
    device = jax.devices()[0]
    stats = device.memory_stats() or {}
    limit = int(stats.get("bytes_limit", 0) or 0)
    print(
        f"estimated_{label}_peak_memory={estimate / 2**30:.2f}GiB "
        f"storage_capacity={physical_capacity} device={device}",
        flush=True,
    )
    if limit and estimate > int(limit * 0.70):
        raise MemoryError(
            "estimated training working set exceeds 70% of JAX's device memory "
            f"limit ({estimate / 2**30:.2f} GiB estimated, "
            f"{limit / 2**30:.2f} GiB limit). Reduce bucket_min_capacity, "
            "tile_batch_size, max_gaussians_per_tile, patch_size, or SH degree."
        )
    return estimate


def _check_bucket_transition_memory_budget(
    config: TrainConfig,
    old_capacity: int,
    new_capacity: int,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
) -> int:
    estimate = estimate_bucket_transition_memory_bytes(
        config,
        old_capacity,
        new_capacity,
        image_height=image_height,
        image_width=image_width,
    )
    bytes_in_use, limit = _device_memory_usage()
    # The optimized resize creates only the new persistent model, Adam moments,
    # mask, and strategy arrays. Include them on top of observed live buffers;
    # old buffers remain live until the replacement state is fully blocked.
    resize_allocation = (
        _training_state_bytes(config, new_capacity) + 256 * 2**20
    )
    projected = max(estimate, bytes_in_use + resize_allocation)
    print(
        f"estimated_capacity_transition_peak={projected / 2**30:.2f}GiB "
        f"capacity_growth={old_capacity}->{new_capacity}",
        flush=True,
    )
    if limit and projected > int(limit * 0.70):
        raise MemoryError(
            "bucket growth was stopped before allocation because old and new "
            "training states would exceed 70% of JAX's device-memory limit "
            f"({projected / 2**30:.2f} GiB projected, "
            f"{limit / 2**30:.2f} GiB limit). Lower the logical capacity, "
            "bucket minimum, SH degree, or rasterizer workspace limits."
        )
    return projected


def _device_memory_usage() -> tuple[int, int]:
    stats = jax.devices()[0].memory_stats() or {}
    return int(stats.get("bytes_in_use", 0) or 0), int(
        stats.get("bytes_limit", 0) or 0
    )


def _check_evaluation_memory_budget(
    config: TrainConfig,
    *,
    physical_capacity: int,
    width: int,
    height: int,
) -> int:
    """Check a full-resolution evaluation against live training allocations."""

    workspace = estimate_rasterization_memory_bytes(
        physical_capacity, width, height, config.rasterizer
    )
    bytes_in_use, limit = _device_memory_usage()
    projected = bytes_in_use + workspace
    print(
        f"estimated_evaluation_peak={projected / 2**30:.2f}GiB "
        f"storage_capacity={physical_capacity} resolution={width}x{height}",
        flush=True,
    )
    if limit and projected > int(limit * 0.70):
        raise MemoryError(
            "full-resolution evaluation was stopped because live training "
            "state plus the render workspace would exceed 70% of JAX's "
            "device-memory limit. Lower eval frequency, image resolution, "
            "max_intersections, max_gaussians_per_tile, or tile_batch_size."
        )
    return projected


def _block_nnx_state(*nodes: Any) -> None:
    for node in nodes:
        state = nnx.as_pure(nnx.state(node))
        for leaf in jax.tree.leaves(state):
            if isinstance(leaf, jax.Array):
                leaf.block_until_ready()


def _initial_storage_capacity(config: TrainConfig, point_count: int) -> int:
    if point_count > config.model.capacity:
        raise ValueError(
            f"point cloud contains {point_count} points, but the logical "
            f"maximum is {config.model.capacity}"
        )
    return config.model.bucket_capacity(point_count)


def _grow_training_state(
    config: TrainConfig,
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    strategy_state: StrategyState,
    new_capacity: int,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
) -> tuple[GaussianModel, nnx.Optimizer, StrategyState]:
    _check_bucket_transition_memory_budget(
        config,
        model.capacity,
        new_capacity,
        image_height=image_height,
        image_width=image_width,
    )
    resized = resize_training_state(
        model,
        optimizer,
        strategy_state,
        new_capacity,
        config.model,
        config.optimizer,
    )
    # Device allocation is asynchronous. Do not drop the valid old state until
    # every new buffer is known to have been allocated and copied successfully.
    _block_nnx_state(*resized)
    return resized


def _save_compacted_training_checkpoint(
    directory: Path,
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    strategy_state: StrategyState,
    *,
    step: int,
    config: TrainConfig,
    intersection_capacity: int,
    scene_transform: SceneTransform,
    scene_scale: float,
    pose_adjust: CameraOptModule | None = None,
    pose_optimizer: nnx.Optimizer | None = None,
    pose_image_names: tuple[str, ...] | None = None,
    appearance_module: AppearanceOptModule | None = None,
    appearance_optimizer: nnx.Optimizer | None = None,
    appearance_image_names: tuple[str, ...] | None = None,
) -> Path:
    compact_count = compact_training_state(model, optimizer, strategy_state)
    compact_count.block_until_ready()
    return save_checkpoint(
        directory,
        model,
        optimizer=optimizer,
        strategy_state=strategy_state,
        step=step,
        config=config,
        intersection_capacity=intersection_capacity,
        scene_transform=scene_transform.matrix,
        scene_scale=scene_scale,
        pose_module=pose_adjust,
        pose_optimizer=pose_optimizer,
        pose_image_names=pose_image_names,
        appearance_module=appearance_module,
        appearance_optimizer=appearance_optimizer,
        appearance_image_names=appearance_image_names,
    )


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
    normalized = (centers - center[None, :]) / max(
        normalization_scale, 1.0e-6
    )
    normalized_center = np.mean(normalized, axis=0)
    extent = np.max(
        np.linalg.norm(normalized - normalized_center[None, :], axis=-1)
    )
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

    centers = transform.points(
        scene.camtoworlds[:, :3, 3].astype(np.float64)
    )
    center = np.mean(centers, axis=0)
    extent = np.max(
        np.linalg.norm(centers - center[None, :], axis=-1)
    )
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


def _grain_iter_dataset(dataset: Any, num_workers: int) -> Any:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if not hasattr(dataset, "to_iter_dataset"):
        return dataset
    import grain

    return dataset.to_iter_dataset(
        read_options=grain.ReadOptions(
            num_threads=num_workers,
            prefetch_buffer_size=8,
        )
    )


def _infinite_batches(
    dataset: Any, *, num_workers: int
) -> Iterator[dict[str, Any]]:
    dataset = _grain_iter_dataset(dataset, num_workers)
    while True:
        yield from iter(dataset)


def _sample_patches(
    images: jax.Array,
    intrinsics: jax.Array,
    key: jax.Array,
    patch_size: int | None,
) -> tuple[jax.Array, jax.Array]:
    if patch_size is None:
        return images, intrinsics
    batch, height, width, channels = images.shape
    if patch_size > height or patch_size > width:
        raise ValueError(
            f"patch_size={patch_size} exceeds image shape {(height, width)}"
        )
    keys = jax.random.split(key, batch * 2).reshape(batch, 2)
    max_y = height - patch_size + 1
    max_x = width - patch_size + 1

    def sample(image, K, sample_keys):
        y = jax.random.randint(sample_keys[0], (), 0, max_y)
        x = jax.random.randint(sample_keys[1], (), 0, max_x)
        patch = jax.lax.dynamic_slice(
            image, (y, x, 0), (patch_size, patch_size, channels)
        )
        adjusted_K = K.at[0, 2].add(-x.astype(K.dtype))
        adjusted_K = adjusted_K.at[1, 2].add(-y.astype(K.dtype))
        return patch, adjusted_K

    return jax.vmap(sample)(images, intrinsics, keys)


def _validate_2dgs_mode(config: TrainConfig) -> None:
    if config.model_type != "2dgs":
        if config.normal_loss or config.dist_loss:
            raise ValueError(
                "normal_loss and dist_loss are available only for 2DGS"
            )
        return
    if config.camera_model != "pinhole":
        raise ValueError("2DGS supports only camera_model='pinhole'")
    if config.with_ut:
        raise ValueError("2DGS does not support with_ut=True")
    if config.with_eval3d:
        raise ValueError("2DGS does not support with_eval3d=True")
    if config.rasterizer.rasterize_mode != "classic":
        raise ValueError("2DGS supports only rasterize_mode='classic'")
    if config.strategy.kind != "default":
        raise ValueError("2DGS supports only the Default strategy")
    if config.packed and config.rasterizer.backend == "reference":
        raise ValueError(
            "2DGS packed training requires projection metadata from the "
            "intersection renderer"
        )


def _create_training_optimizer(
    model: GaussianModel,
    config: TrainConfig,
    *,
    world_size: int = 1,
    scene_scale: float = 1.0,
) -> nnx.Optimizer:
    optimizer_kwargs = {
        "batch_size": config.data.batch_size,
        "world_size": world_size,
        "scene_scale": scene_scale,
    }
    if config.sparse_grad:
        return create_row_selective_optimizer(
            model, config.optimizer, **optimizer_kwargs
        )
    if config.visible_adam:
        return create_visible_adam_optimizer(
            model, config.optimizer, **optimizer_kwargs
        )
    return create_optimizer(model, config.optimizer, **optimizer_kwargs)


def _pose_learning_rate(
    config: TrainConfig, step: int | jax.Array
) -> jax.Array:
    """Current-main pose LR: batch-scaled and exponentially decayed to 1%."""

    step = jnp.asarray(step, dtype=jnp.float32)
    initial = config.pose_opt_lr * math.sqrt(config.data.batch_size)
    progress = step / float(max(config.steps, 1))
    return jnp.asarray(initial, dtype=jnp.float32) * jnp.power(0.01, progress)


def _create_pose_optimizer(
    pose_adjust: CameraOptModule, config: TrainConfig
) -> nnx.Optimizer:
    """Build PyTorch-Adam-equivalent pose optimization with coupled L2 decay."""

    transform = optax.chain(
        optax.add_decayed_weights(config.pose_opt_reg),
        optax.adam(
            lambda count: _pose_learning_rate(config, count),
            eps=1.0e-8,
        ),
    )
    return nnx.Optimizer(pose_adjust, transform, wrt=nnx.Param)


def _validate_camera_module_resume_config(
    config: TrainConfig, checkpoint_path: str | Path
) -> None:
    """Reject resume changes that alter restored optimizer meaning."""

    saved = load_checkpoint_config(checkpoint_path)
    for field in ("normalize_world_space", "global_scale"):
        saved_value = getattr(saved, field)
        current_value = getattr(config, field)
        if saved_value != current_value:
            raise ValueError(
                f"resume requires {field} to match the checkpoint config "
                f"({saved_value} saved, {current_value} requested)"
            )
    if saved.data.batch_size != config.data.batch_size:
        raise ValueError(
            "resume requires data.batch_size to match the checkpoint because "
            "it scales Gaussian Adam "
            f"({saved.data.batch_size} saved, "
            f"{config.data.batch_size} requested)"
        )
    if saved.pose_opt != config.pose_opt:
        raise ValueError(
            "resume requires pose_opt to match the checkpoint config "
            f"({saved.pose_opt} saved, {config.pose_opt} requested)"
        )
    if saved.pose_noise != config.pose_noise:
        raise ValueError(
            "resume requires pose_noise to match the checkpoint config "
            f"({saved.pose_noise} saved, {config.pose_noise} requested)"
        )
    if config.pose_opt and saved.steps != config.steps:
        raise ValueError(
            "resume with pose_opt requires steps to match the checkpoint LR "
            f"horizon ({saved.steps} saved, {config.steps} requested)"
        )
    if config.pose_noise > 0.0 and saved.seed != config.seed:
        raise ValueError(
            "resume with pose_noise requires seed to match the checkpoint "
            f"config ({saved.seed} saved, {config.seed} requested)"
        )
    if saved.app_opt != config.app_opt:
        raise ValueError(
            "resume requires app_opt to match the checkpoint config "
            f"({saved.app_opt} saved, {config.app_opt} requested)"
        )
    if config.app_opt:
        appearance_fields = (
            "app_embed_dim",
            "app_opt_lr",
            "app_opt_reg",
        )
        for field in appearance_fields:
            saved_value = getattr(saved, field)
            current_value = getattr(config, field)
            if saved_value != current_value:
                raise ValueError(
                    f"resume requires {field} to match the checkpoint config "
                    f"({saved_value} saved, {current_value} requested)"
                )
        if saved.model.sh_degree != config.model.sh_degree:
            raise ValueError(
                "resume with app_opt requires model.sh_degree to match the "
                f"checkpoint ({saved.model.sh_degree} saved, "
                f"{config.model.sh_degree} requested)"
            )


def _invert_rigid_transforms(transforms: jax.Array) -> jax.Array:
    """Invert row-major homogeneous rigid transforms without a generic solve."""

    transforms = jnp.asarray(transforms)
    if transforms.ndim < 2 or transforms.shape[-2:] != (4, 4):
        raise ValueError(
            "transforms must have shape (..., 4, 4), "
            f"got {transforms.shape}"
        )
    rotation = transforms[..., :3, :3]
    translation = transforms[..., :3, 3]
    inverse_rotation = jnp.swapaxes(rotation, -1, -2)
    inverse_translation = -jnp.einsum(
        "...ij,...j->...i", inverse_rotation, translation
    )
    result = jnp.broadcast_to(
        jnp.eye(4, dtype=transforms.dtype), transforms.shape
    )
    result = result.at[..., :3, :3].set(inverse_rotation)
    return result.at[..., :3, 3].set(inverse_translation)


def _apply_camera_pose_modules(
    camtoworlds: jax.Array,
    image_ids: jax.Array,
    *,
    pose_adjust: CameraOptModule | None,
    pose_perturb: CameraOptModule | None,
) -> jax.Array:
    """Apply fixed noise followed by trainable local camera-pose deltas."""

    adjusted = jnp.asarray(camtoworlds)
    if pose_perturb is not None:
        adjusted = jax.lax.stop_gradient(pose_perturb(adjusted, image_ids))
    if pose_adjust is not None:
        adjusted = pose_adjust(adjusted, image_ids)
    return adjusted


def _unpack_training_projection_metadata(
    info: dict[str, Any],
    active_mask: jax.Array,
    *,
    camera_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Scatter padded packed projection metadata back to dense camera rows."""

    if camera_count <= 0:
        raise ValueError("packed training requires at least one camera")
    active_mask = jnp.asarray(active_mask, dtype=jnp.bool_)
    if active_mask.ndim != 1 or active_mask.shape[0] == 0:
        raise ValueError("active_mask must have shape [N] with N > 0")
    for key in (
        "camera_ids",
        "gaussian_ids",
        "radii",
        "valid",
        "projection_valid_count",
    ):
        if info.get(key) is None:
            raise ValueError(f"packed training metadata requires {key!r}")

    camera_ids = jnp.asarray(info["camera_ids"], dtype=jnp.int32)
    gaussian_ids = jnp.asarray(info["gaussian_ids"], dtype=jnp.int32)
    radii = jnp.asarray(info["radii"])
    valid = jnp.asarray(info["valid"], dtype=jnp.bool_)
    valid_count = jnp.asarray(
        info["projection_valid_count"], dtype=jnp.int32
    )
    packed_capacity = gaussian_ids.shape[0]
    if camera_ids.shape != (packed_capacity,):
        raise ValueError("packed camera_ids must have shape [P]")
    if radii.shape != (packed_capacity, 2):
        raise ValueError("packed radii must have shape [P, 2]")
    if valid.shape != (packed_capacity,):
        raise ValueError("packed valid must have shape [P]")
    if valid_count.shape != ():
        raise ValueError("projection_valid_count must be scalar")

    gaussian_count = active_mask.shape[0]
    positions = jnp.arange(packed_capacity, dtype=jnp.int32)
    packed_valid = (
        (positions < valid_count)
        & valid
        & (camera_ids >= 0)
        & (camera_ids < camera_count)
        & (gaussian_ids >= 0)
        & (gaussian_ids < gaussian_count)
        & jnp.all(radii > 0, axis=-1)
    )
    safe_camera_ids = jnp.clip(camera_ids, 0, camera_count - 1)
    safe_gaussian_ids = jnp.clip(gaussian_ids, 0, gaussian_count - 1)
    dense_radii = jnp.zeros(
        (camera_count, gaussian_count, 2), dtype=radii.dtype
    ).at[safe_camera_ids, safe_gaussian_ids].max(
        jnp.where(packed_valid[:, None], radii, 0)
    )
    dense_valid = (
        jnp.zeros((camera_count, gaussian_count), dtype=jnp.int32)
        .at[safe_camera_ids, safe_gaussian_ids]
        .max(packed_valid.astype(jnp.int32))
        > 0
    )
    visible_mask = jnp.any(dense_valid, axis=0) & active_mask
    return dense_radii, dense_valid, visible_mask


def _two_dgs_regularization_losses(
    rendered_normals: jax.Array,
    normals_from_depth: jax.Array,
    alphas: jax.Array,
    render_distort: jax.Array,
    step: jax.Array,
    config: TrainConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return weighted 2DGS normal and distortion loss contributions."""

    zero = jnp.zeros((), dtype=rendered_normals.dtype)
    normal_loss_value = zero
    if config.normal_loss:
        normal_weight = jnp.where(
            step > config.normal_start_iter,
            jnp.asarray(config.normal_lambda, rendered_normals.dtype),
            zero,
        )
        alpha_weighted_normals = normals_from_depth * jax.lax.stop_gradient(
            alphas
        )
        normal_error = 1.0 - jnp.sum(
            rendered_normals * alpha_weighted_normals, axis=-1
        )
        normal_loss_value = normal_weight * jnp.mean(normal_error)

    distortion_loss_value = zero
    if config.dist_loss:
        distortion_weight = jnp.where(
            step > config.dist_start_iter,
            jnp.asarray(config.dist_lambda, render_distort.dtype),
            jnp.zeros((), dtype=render_distort.dtype),
        )
        distortion_loss_value = distortion_weight * jnp.mean(render_distort)
    return normal_loss_value, distortion_loss_value


def _make_train_step(
    config: TrainConfig,
    *,
    distributed_world_size: int = 1,
    distributed_axis_name: Hashable | None = None,
    distributed_scene_scale: float = 1.0,
) -> Callable[..., dict[str, jax.Array]]:
    _validate_2dgs_mode(config)
    if config.with_eval3d and config.strategy.kind != "mcmc":
        raise NotImplementedError(
            "screen-space densification statistics do not yet support "
            "with_eval3d=True"
        )
    patch_size = config.data.patch_size
    rasterizer_config = config.rasterizer
    ssim_lambda = config.ssim_lambda
    random_background = config.random_background
    use_absgrad = config.strategy.absgrad
    row_selective_optimizer = config.sparse_grad or config.visible_adam
    distributed = distributed_world_size > 1
    collect_screen_stats = not (
        config.strategy.kind == "mcmc"
        and (config.with_ut or config.with_eval3d)
    )
    mcmc_strategy = (
        MCMCStrategy(config.strategy)
        if config.strategy.kind == "mcmc"
        else None
    )
    # Distributed refinement has no host callback, so the owner-local default
    # strategy must plan inside the step to preflight every shard together.
    distributed_plan_strategy = (
        DefaultStrategy(config.strategy)
        if distributed and config.strategy.kind == "default"
        else None
    )

    donated_nodes = (
        "model",
        "optimizer",
        "strategy_state",
        "safety_state",
    ) + (("pose_adjust", "pose_optimizer") if config.pose_opt else ()) + (
        ("appearance_module", "appearance_optimizer")
        if config.app_opt
        else ()
    )

    @nnx.jit(donate_argnames=donated_nodes)
    def train_step(
        model: GaussianModel,
        optimizer: nnx.Optimizer,
        strategy_state: StrategyState,
        safety_state: TrainingSafetyState,
        images: jax.Array,
        intrinsics: jax.Array,
        viewmats: jax.Array,
        key: jax.Array,
        sh_degree: jax.Array,
        strategy_key: jax.Array | None = None,
        *,
        pose_adjust: CameraOptModule | None = None,
        pose_optimizer: nnx.Optimizer | None = None,
        camtoworlds: jax.Array | None = None,
        image_ids: jax.Array | None = None,
        pose_perturb: CameraOptModule | None = None,
        appearance_module: AppearanceOptModule | None = None,
        appearance_optimizer: nnx.Optimizer | None = None,
    ) -> dict[str, jax.Array]:
        distributed_state_mismatch = jnp.asarray(False)
        if distributed:
            optimizer_batch_size = getattr(
                optimizer, "_jax_gs_batch_size", None
            )
            optimizer_world_size = getattr(
                optimizer, "_jax_gs_world_size", None
            )
            optimizer_scene_scale = getattr(
                optimizer, "_jax_gs_scene_scale", None
            )
            optimizer_config = getattr(
                optimizer, "_jax_gs_optimizer_config", None
            )
            if (
                optimizer_batch_size != config.data.batch_size
                or optimizer_world_size != distributed_world_size
                or optimizer_scene_scale != distributed_scene_scale
                or optimizer_config != config.optimizer
            ):
                raise ValueError(
                    "distributed optimizer must be created with "
                    f"batch_size={config.data.batch_size} and "
                    f"world_size={distributed_world_size}, "
                    f"scene_scale={distributed_scene_scale}, and the train "
                    "step's OptimizerConfig; got "
                    f"batch_size={optimizer_batch_size!r} and "
                    f"world_size={optimizer_world_size!r}, "
                    f"scene_scale={optimizer_scene_scale!r}"
                )
            if images.shape[0] != config.data.batch_size:
                raise ValueError(
                    "distributed rank-local image batch must match "
                    f"config.data.batch_size={config.data.batch_size}"
                )
            assert distributed_axis_name is not None
            optimizer_step = optimizer.step[...]
            minimum_step = jax.lax.pmin(
                optimizer_step, distributed_axis_name
            )
            maximum_step = jax.lax.pmax(
                optimizer_step, distributed_axis_name
            )
            minimum_sh_degree = jax.lax.pmin(
                sh_degree, distributed_axis_name
            )
            maximum_sh_degree = jax.lax.pmax(
                sh_degree, distributed_axis_name
            )
            distributed_state_mismatch = (
                (minimum_step != maximum_step)
                | (minimum_sh_degree != maximum_sh_degree)
            )
        uses_camera_modules = (
            config.pose_opt or config.pose_noise > 0.0 or config.app_opt
        )
        if model.has_appearance != config.app_opt:
            raise ValueError(
                "model color representation must match config.app_opt"
            )
        if config.pose_opt and (pose_adjust is None or pose_optimizer is None):
            raise ValueError(
                "pose_opt=True requires pose_adjust and pose_optimizer"
            )
        if config.app_opt and (
            appearance_module is None or appearance_optimizer is None
        ):
            raise ValueError(
                "app_opt=True requires appearance_module and "
                "appearance_optimizer"
            )
        if uses_camera_modules and (camtoworlds is None or image_ids is None):
            raise ValueError(
                "camera modules require camtoworlds and image_ids"
            )
        if config.pose_noise > 0.0 and pose_perturb is None:
            raise ValueError("pose_noise > 0 requires pose_perturb")
        patch_key, background_key = jax.random.split(key)
        if strategy_key is None:
            strategy_key = jax.random.fold_in(key, 0x53545241)
        refine_key, noise_key = jax.random.split(strategy_key)
        image_height, image_width = images.shape[-3:-1]
        render_height, render_width = _training_render_size(
            config,
            image_height=image_height,
            image_width=image_width,
        )
        targets, patch_intrinsics = _sample_patches(
            images, intrinsics, patch_key, patch_size
        )
        if random_background:
            backgrounds = jax.random.uniform(
                background_key, (images.shape[0], 3), dtype=images.dtype
            )
        else:
            backgrounds = jnp.zeros((images.shape[0], 3), dtype=images.dtype)
        # The host training loop is one-based while optimizer.step is incremented
        # after this loss. Match the host/upstream iteration used by start_iter.
        training_step = optimizer.step[...] + jnp.asarray(
            1, dtype=optimizer.step[...].dtype
        )

        # Keep distributed camera probes independent while spanning the global
        # Gaussian axis. Gathering local probes would sum signed gradients in
        # the gather VJP before the strategy can take each camera's norm.
        screen_gaussian_count = model.means.shape[0] * (
            distributed_world_size if distributed else 1
        )
        screen_probe_shape = (
            (viewmats.shape[0], screen_gaussian_count, 2)
            if collect_screen_stats
            else ()
        )
        screen_probe = jnp.zeros(
            screen_probe_shape, dtype=model.means[...].dtype
        )

        def loss_fn(
            current_model: GaussianModel,
            current_pose_adjust: CameraOptModule | None,
            current_screen_probe: jax.Array,
            current_appearance_module: AppearanceOptModule | None = None,
        ):
            render_viewmats = viewmats
            adjusted_camtoworlds = None
            if uses_camera_modules:
                assert camtoworlds is not None
                assert image_ids is not None
                adjusted_camtoworlds = _apply_camera_pose_modules(
                    camtoworlds,
                    image_ids,
                    pose_adjust=(
                        current_pose_adjust if config.pose_opt else None
                    ),
                    pose_perturb=pose_perturb,
                )
                render_viewmats = _invert_rigid_transforms(
                    adjusted_camtoworlds
                )
            parameters = current_model.activated(
                split_sh=(
                    config.model_type == "3dgs" and not config.app_opt
                )
            )
            if config.app_opt:
                assert current_appearance_module is not None
                assert adjusted_camtoworlds is not None
                assert image_ids is not None
                directions = (
                    parameters["means"][None, :, :]
                    - adjusted_camtoworlds[:, None, :3, 3]
                )
                corrections = current_appearance_module(
                    parameters["features"],
                    image_ids,
                    directions,
                    sh_degree,
                )
                render_colors = jax.nn.sigmoid(
                    parameters["colors"][None, :, :] + corrections
                )
                raster_sh_degree = None
            else:
                render_colors = parameters["sh_coeffs"]
                raster_sh_degree = sh_degree
            if config.model_type == "2dgs":
                (
                    renders,
                    alphas,
                    rendered_normals,
                    normals_from_depth,
                    render_distort,
                    _,
                    info,
                ) = rasterization_2dgs(
                    parameters["means"],
                    parameters["quats"],
                    parameters["scales"],
                    parameters["opacities"],
                    render_colors,
                    render_viewmats,
                    patch_intrinsics,
                    render_width,
                    render_height,
                    packed=config.packed,
                    sparse_grad=config.sparse_grad,
                    absgrad=use_absgrad,
                    active_mask=parameters["active_mask"],
                    sh_degree=raster_sh_degree,
                    backgrounds=backgrounds,
                    render_mode="RGB",
                    distloss=config.dist_loss,
                    config=rasterizer_config,
                    _gradient_2dgs_offset=(
                        None
                        if use_absgrad or not collect_screen_stats
                        else current_screen_probe
                    ),
                    _gradient_2dgs_absgrad_probe=(
                        current_screen_probe
                        if use_absgrad and collect_screen_stats
                        else None
                    ),
                )
                normal_loss_value, distortion_loss_value = (
                    _two_dgs_regularization_losses(
                        rendered_normals,
                        normals_from_depth,
                        alphas,
                        render_distort,
                        training_step,
                        config,
                    )
                )
            else:
                renders, _, info = rasterization(
                    parameters["means"],
                    parameters["quats"],
                    parameters["scales"],
                    parameters["opacities"],
                    render_colors,
                    render_viewmats,
                    patch_intrinsics,
                    render_width,
                    render_height,
                    packed=config.packed,
                    sparse_grad=config.sparse_grad,
                    absgrad=use_absgrad,
                    active_mask=parameters["active_mask"],
                    sh_degree=raster_sh_degree,
                    backgrounds=backgrounds,
                    camera_model=config.camera_model,
                    with_ut=config.with_ut,
                    with_eval3d=config.with_eval3d,
                    distributed=distributed,
                    distributed_world_size=distributed_world_size,
                    distributed_axis_name=distributed_axis_name,
                    config=rasterizer_config,
                    _means2d_offset=(
                        None
                        if use_absgrad or not collect_screen_stats
                        else current_screen_probe
                    ),
                    _means2d_absgrad_probe=(
                        current_screen_probe
                        if use_absgrad and collect_screen_stats
                        else None
                    ),
                )
                normal_loss_value = jnp.zeros((), dtype=renders.dtype)
                distortion_loss_value = jnp.zeros((), dtype=renders.dtype)
            rgb = renders[..., :3]
            l1_value = jnp.mean(l1_loss(rgb, targets))
            ssim_value = ssim(rgb, targets)
            photometric_loss = (1.0 - ssim_lambda) * l1_value + ssim_lambda * (
                1.0 - ssim_value
            )
            opacity_reg_loss_value = jnp.zeros(
                (), dtype=photometric_loss.dtype
            )
            if config.opacity_reg > 0.0:
                opacity_reg_loss_value = jnp.asarray(
                    config.opacity_reg, dtype=photometric_loss.dtype
                ) * opacity_reg_loss(
                    current_model.opacity_logits[...],
                    mask=current_model.active_mask[...],
                )
            scale_reg_loss_value = jnp.zeros(
                (), dtype=photometric_loss.dtype
            )
            if config.scale_reg > 0.0:
                scale_reg_loss_value = jnp.asarray(
                    config.scale_reg, dtype=photometric_loss.dtype
                ) * scale_reg_loss(
                    current_model.log_scales[...],
                    mask=current_model.active_mask[...],
                )
            loss = (
                photometric_loss
                + normal_loss_value
                + distortion_loss_value
                + opacity_reg_loss_value
                + scale_reg_loss_value
            )
            pose_error_value = jnp.zeros((), dtype=loss.dtype)
            if config.pose_opt and config.pose_noise > 0.0:
                assert adjusted_camtoworlds is not None
                assert camtoworlds is not None
                pose_error_value = jnp.mean(
                    jnp.abs(adjusted_camtoworlds - camtoworlds)
                )
            return loss, (
                l1_value,
                ssim_value,
                normal_loss_value,
                distortion_loss_value,
                opacity_reg_loss_value,
                scale_reg_loss_value,
                pose_error_value,
                rgb,
                info,
            )

        with jax.named_scope("loss_and_backward"):
            if config.pose_opt and config.app_opt:
                result, gradients = nnx.value_and_grad(
                    lambda current_model, current_pose, current_appearance,
                    current_screen: loss_fn(
                        current_model,
                        current_pose,
                        current_screen,
                        current_appearance,
                    ),
                    argnums=(0, 1, 2, 3),
                    has_aux=True,
                )(
                    model,
                    pose_adjust,
                    appearance_module,
                    screen_probe,
                )
                grads, pose_grads, appearance_grads, screen_grad = gradients
            elif config.pose_opt:
                result, gradients = nnx.value_and_grad(
                    loss_fn, argnums=(0, 1, 2), has_aux=True
                )(model, pose_adjust, screen_probe)
                grads, pose_grads, screen_grad = gradients
                appearance_grads = None
            elif config.app_opt:
                result, gradients = nnx.value_and_grad(
                    lambda current_model, current_appearance, current_screen: (
                        loss_fn(
                            current_model,
                            None,
                            current_screen,
                            current_appearance,
                        )
                    ),
                    argnums=(0, 1, 2),
                    has_aux=True,
                )(model, appearance_module, screen_probe)
                grads, appearance_grads, screen_grad = gradients
                pose_grads = None
            else:
                result, gradients = nnx.value_and_grad(
                    lambda current_model, current_screen: loss_fn(
                        current_model, None, current_screen
                    ),
                    argnums=(0, 1),
                    has_aux=True,
                )(model, screen_probe)
                grads, screen_grad = gradients
                pose_grads = None
                appearance_grads = None
            loss, (
                l1_value,
                ssim_value,
                normal_loss_value,
                distortion_loss_value,
                opacity_reg_loss_value,
                scale_reg_loss_value,
                pose_error_value,
                rgb,
                info,
            ) = result
        if distributed and config.pose_opt:
            assert distributed_axis_name is not None
            assert pose_grads is not None
            # Current-main wraps the camera-pose module in DDP, which averages
            # the replicated gradient across ranks. Gaussians stay sharded and
            # keep the owner-scattered sum instead.
            pose_grads = jax.lax.pmean(pose_grads, distributed_axis_name)
        active_mask = model.active_mask[...]
        with jax.named_scope("inactive_grad_mask"):
            grads = jax.lax.cond(
                jnp.all(active_mask),
                lambda current: current,
                lambda current: mask_inactive_gradients(
                    current, active_mask
                ),
                grads,
            )
        if config.packed:
            projection_radii, projection_valid, visible = (
                _unpack_training_projection_metadata(
                    info,
                    active_mask,
                    camera_count=viewmats.shape[0],
                )
            )
            stats_projection_radii = projection_radii
            stats_projection_valid = projection_valid
            stats_active_mask = active_mask
        else:
            projection_radii = info["radii"]
            projection_valid = info["valid"]
            stats_projection_radii = projection_radii
            stats_projection_valid = projection_valid
            stats_active_mask = active_mask
            visible = jnp.any(
                jnp.asarray(projection_valid, dtype=jnp.bool_)
                & jnp.all(jnp.asarray(projection_radii) > 0, axis=-1),
                axis=0,
            )
            if distributed:
                assert distributed_axis_name is not None
                visible = (
                    jax.lax.pmax(
                        visible.astype(jnp.int32),
                        distributed_axis_name,
                    )
                    > 0
                )
                owner_start = (
                    jax.lax.axis_index(distributed_axis_name)
                    * model.capacity
                )
                stats_active_mask = jax.lax.all_gather(
                    active_mask,
                    distributed_axis_name,
                    axis=0,
                    tiled=True,
                )
                projection_radii = jax.lax.dynamic_slice_in_dim(
                    projection_radii,
                    owner_start,
                    model.capacity,
                    axis=1,
                )
                projection_valid = jax.lax.dynamic_slice_in_dim(
                    projection_valid,
                    owner_start,
                    model.capacity,
                    axis=1,
                )
                visible = jax.lax.dynamic_slice_in_dim(
                    visible,
                    owner_start,
                    model.capacity,
                    axis=0,
                )
            visible = visible & active_mask
        if collect_screen_stats:
            densification_stats = build_densification_stats(
                screen_grad,
                stats_projection_radii,
                stats_projection_valid,
                stats_active_mask,
                render_width,
                render_height,
            )
            if distributed:
                assert distributed_axis_name is not None
                # The per-camera norm is already in these scalar statistics.
                # Reduce globally before slicing the current Gaussian owner.
                global_stats = DensificationStats(
                    jax.lax.psum(
                        densification_stats.grad_sum,
                        distributed_axis_name,
                    ),
                    jax.lax.psum(
                        densification_stats.count,
                        distributed_axis_name,
                    ),
                    jax.lax.pmax(
                        densification_stats.max_radii,
                        distributed_axis_name,
                    ),
                )

                def owner_slice(value):
                    return jax.lax.dynamic_slice_in_dim(
                        value,
                        owner_start,
                        model.capacity,
                        axis=0,
                    )

                densification_stats = DensificationStats(
                    owner_slice(global_stats.grad_sum),
                    owner_slice(global_stats.count),
                    owner_slice(global_stats.max_radii),
                )
        overflow_tiles = jnp.count_nonzero(info["tile_overflow"])
        intersection_overflow = jnp.any(info["intersection_overflow"])
        previous_max_overflow_tiles = safety_state.max_overflow_tiles[...]
        previous_intersection_overflow_seen = (
            safety_state.intersection_overflow_seen[...]
        )
        if distributed:
            assert distributed_axis_name is not None
            overflow_tiles = jax.lax.psum(
                overflow_tiles, distributed_axis_name
            )
            intersection_overflow = (
                jax.lax.pmax(
                    intersection_overflow.astype(jnp.int32),
                    distributed_axis_name,
                )
                > 0
            )
            previous_max_overflow_tiles = jax.lax.pmax(
                previous_max_overflow_tiles, distributed_axis_name
            )
            previous_intersection_overflow_seen = (
                jax.lax.pmax(
                    previous_intersection_overflow_seen.astype(jnp.int32),
                    distributed_axis_name,
                )
                > 0
            )
        max_overflow_tiles = jnp.maximum(
            previous_max_overflow_tiles, overflow_tiles
        )
        intersection_overflow_seen = (
            previous_intersection_overflow_seen | intersection_overflow
        )
        safety_state.max_overflow_tiles[...] = max_overflow_tiles
        safety_state.intersection_overflow_seen[...] = intersection_overflow_seen
        strategy_capacity_overflow = jnp.asarray(False)
        mcmc_should_refine = jnp.asarray(False)
        plan_metrics: dict[str, jax.Array] = {}
        if config.strategy.kind == "mcmc":
            mcmc_should_refine = (
                (training_step > config.strategy.refine_start)
                & (training_step < config.strategy.refine_stop)
                & (training_step % config.strategy.refine_every == 0)
            )
            assert mcmc_strategy is not None
            refine_plan = mcmc_strategy.plan_refine(
                model,
                strategy_state,
                strategy_state.scene_scale[...],
                step=training_step,
            )
            strategy_capacity_overflow = (
                mcmc_should_refine & refine_plan["capacity_overflow"]
            )
        elif distributed_plan_strategy is not None:
            assert distributed_axis_name is not None
            refine_scheduled = (
                (training_step > config.strategy.refine_start)
                & (training_step < config.strategy.refine_stop)
                & (training_step % config.strategy.refine_every == 0)
                & (
                    training_step % config.strategy.reset_every
                    >= config.strategy.pause_refine_after_reset
                )
            )
            reset_scheduled = (
                (training_step > 0)
                & (training_step < config.strategy.refine_stop)
                & (training_step % config.strategy.reset_every == 0)
            )
            # Every shard decides on its own rows. The step's scene scale is
            # the value already checked against the optimizer, so no rank can
            # score growth or pruning against a different threshold.
            owner_plan = distributed_plan_strategy.plan_refine(
                model,
                strategy_state,
                distributed_scene_scale,
                step=training_step,
            )
            # Only the plan's scalar summaries cross ranks; the owner-local
            # [L] decision arrays must never be reduced.
            planned_new_count = jax.lax.psum(
                owner_plan["planned_new_count"], distributed_axis_name
            )
            planned_pruned_count = jax.lax.psum(
                owner_plan["pruned_count"], distributed_axis_name
            )
            planned_required_capacity = jax.lax.pmax(
                owner_plan["required_capacity"], distributed_axis_name
            )
            any_rank_capacity_overflow = (
                jax.lax.pmax(
                    owner_plan["capacity_overflow"].astype(jnp.int32),
                    distributed_axis_name,
                )
                > 0
            )
            strategy_capacity_overflow = (
                refine_scheduled & any_rank_capacity_overflow
            )
            plan_metrics = {
                "refine_scheduled": refine_scheduled,
                "reset_scheduled": reset_scheduled,
                "refine_planned_new_count": jnp.where(
                    refine_scheduled, planned_new_count, 0
                ),
                "refine_planned_pruned_count": jnp.where(
                    refine_scheduled, planned_pruned_count, 0
                ),
                "refine_required_capacity": jnp.where(
                    refine_scheduled, planned_required_capacity, 0
                ),
                "refine_capacity_overflow": strategy_capacity_overflow,
            }
        has_overflow = (
            intersection_overflow_seen
            | (max_overflow_tiles > 0)
            | strategy_capacity_overflow
            | distributed_state_mismatch
        )

        # Committed topology counters leave both update branches so that their
        # reporting collectives run after the branches rejoin, never inside a
        # conditional.
        uncommitted_refine = (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
        )
        uncommitted_topology = uncommitted_refine + (jnp.asarray(False),)

        def skip_update(
            current_model,
            current_optimizer,
            current_strategy_state,
            current_grads,
            current_visible,
        ):
            del current_model, current_optimizer, current_grads, current_visible
            if config.strategy.kind == "mcmc":
                current_strategy_state.last_new_count[...] = jnp.where(
                    strategy_capacity_overflow,
                    0,
                    current_strategy_state.last_new_count[...],
                )
                current_strategy_state.last_pruned_count[...] = jnp.where(
                    strategy_capacity_overflow,
                    0,
                    current_strategy_state.last_pruned_count[...],
                )
                current_strategy_state.capacity_overflow[...] = (
                    current_strategy_state.capacity_overflow[...]
                    | strategy_capacity_overflow
                )
            return uncommitted_topology

        def apply_update(
            current_model,
            current_optimizer,
            current_strategy_state,
            current_grads,
            current_visible,
        ):
            with jax.named_scope("optimizer_update"):
                if row_selective_optimizer:
                    current_optimizer.update(
                        current_model,
                        current_grads,
                        visible_mask=current_visible,
                    )
                    normalized_quats = current_model.normalized_quats
                    current_model.quats[...] = jnp.where(
                        current_visible[:, None],
                        normalized_quats,
                        current_model.quats[...],
                    )
                else:
                    current_optimizer.update(current_model, current_grads)
                    current_model.normalize_quaternions()
                if config.strategy.kind == "mcmc":
                    step_number = current_optimizer.step[...]

                    def refine(_model, _optimizer, _state):
                        assert mcmc_strategy is not None
                        refine_result = mcmc_strategy.refine(
                            _model,
                            _state,
                            _optimizer,
                            refine_key,
                            _state.scene_scale[...],
                            step=step_number,
                        )
                        return refine_result["capacity_overflow"]

                    def skip_refine(_model, _optimizer, _state):
                        del _model, _optimizer, _state
                        return jnp.asarray(False)

                    refine_overflow = nnx.cond(
                        mcmc_should_refine,
                        refine,
                        skip_refine,
                        current_model,
                        current_optimizer,
                        current_strategy_state,
                    )
                    schedule_progress = step_number.astype(jnp.float32) / float(
                        max(config.optimizer.max_steps, 1)
                    )
                    means_lr = config.optimizer.means_lr * jnp.power(
                        config.optimizer.means_lr_final_scale,
                        schedule_progress,
                    )
                    perturbed_means = mcmc_position_perturbation(
                        current_model.means[...],
                        current_model.quats[...],
                        current_model.log_scales[...],
                        current_model.opacity_logits[...],
                        means_lr * config.strategy.noise_lr,
                        key=noise_key,
                        t=config.strategy.noise_opacity_t,
                        k=config.strategy.noise_opacity_k,
                        active_mask=current_model.active_mask[...],
                    )
                    noise_stop = config.strategy.noise_injection_stop_iter
                    should_inject = (
                        ((noise_stop < 0) | (step_number < noise_stop))
                        & ~refine_overflow
                    )
                    current_model.means[...] = jnp.where(
                        should_inject, perturbed_means, current_model.means[...]
                    )
            if config.strategy.kind == "default" and collect_screen_stats:
                with jax.named_scope("strategy_stats_update"):
                    accumulate_stats = training_step < config.strategy.refine_stop
                    current_strategy_state.grad_accum[...] += jnp.where(
                        accumulate_stats,
                        densification_stats.grad_sum,
                        0.0,
                    )
                    current_strategy_state.visible_count[...] += jnp.where(
                        accumulate_stats,
                        densification_stats.count,
                        0.0,
                    )
                    current_strategy_state.max_radii[...] = jnp.where(
                        accumulate_stats,
                        jnp.maximum(
                            current_strategy_state.max_radii[...],
                            densification_stats.max_radii,
                        ),
                        current_strategy_state.max_radii[...],
                    )
            if distributed_plan_strategy is None:
                return uncommitted_topology
            # current-main refines after the optimizer step and after this
            # step's statistics, so the owner-local commit recomputes its own
            # events here instead of replaying the pre-update preflight.
            with jax.named_scope("topology_commit"):

                def commit_refine(_model, _optimizer, _state):
                    assert distributed_plan_strategy is not None
                    refine_result = distributed_plan_strategy.refine(
                        _model,
                        _state,
                        _optimizer,
                        refine_key,
                        distributed_scene_scale,
                        step=training_step,
                    )
                    return (
                        refine_result["new_count"],
                        refine_result["pruned_count"],
                        refine_result["capacity_overflow"],
                    )

                def skip_commit_refine(_model, _optimizer, _state):
                    del _model, _optimizer, _state
                    return uncommitted_refine

                new_count, pruned_count, commit_overflow = nnx.cond(
                    refine_scheduled,
                    commit_refine,
                    skip_commit_refine,
                    current_model,
                    current_optimizer,
                    current_strategy_state,
                )

                def commit_reset(_model, _optimizer):
                    reset_opacities(
                        _model,
                        _optimizer,
                        maximum_opacity=config.strategy.reset_opacity,
                    )
                    return jnp.asarray(True)

                def skip_commit_reset(_model, _optimizer):
                    del _model, _optimizer
                    return jnp.asarray(False)

                # An owner that could not grow keeps its statistics for the
                # next refinement, so it must not reset opacities either.
                opacity_reset = nnx.cond(
                    reset_scheduled & ~commit_overflow,
                    commit_reset,
                    skip_commit_reset,
                    current_model,
                    current_optimizer,
                )
            return (new_count, pruned_count, commit_overflow, opacity_reset)

        (
            committed_new_count,
            committed_pruned_count,
            committed_overflow,
            committed_opacity_reset,
        ) = nnx.cond(
            has_overflow,
            skip_update,
            apply_update,
            model,
            optimizer,
            strategy_state,
            grads,
            visible,
        )
        if distributed_plan_strategy is not None:
            assert distributed_axis_name is not None
            plan_metrics = {
                **plan_metrics,
                "refine_new_count": jax.lax.psum(
                    committed_new_count, distributed_axis_name
                ),
                "refine_pruned_count": jax.lax.psum(
                    committed_pruned_count, distributed_axis_name
                ),
                "refine_commit_overflow": (
                    jax.lax.pmax(
                        committed_overflow.astype(jnp.int32),
                        distributed_axis_name,
                    )
                    > 0
                ),
                "opacity_reset": (
                    jax.lax.pmax(
                        committed_opacity_reset.astype(jnp.int32),
                        distributed_axis_name,
                    )
                    > 0
                ),
            }
        if config.pose_opt:
            assert pose_adjust is not None
            assert pose_optimizer is not None
            assert pose_grads is not None

            def skip_pose_update(_module, _optimizer, _grads):
                del _module, _optimizer, _grads
                return jnp.asarray(0, dtype=jnp.int32)

            def apply_pose_update(current_module, current_optimizer, current_grads):
                current_optimizer.update(current_module, current_grads)
                return jnp.asarray(0, dtype=jnp.int32)

            nnx.cond(
                has_overflow,
                skip_pose_update,
                apply_pose_update,
                pose_adjust,
                pose_optimizer,
                pose_grads,
            )
        if config.app_opt:
            assert appearance_module is not None
            assert appearance_optimizer is not None
            assert appearance_grads is not None

            def skip_appearance_update(_module, _optimizer, _grads):
                del _module, _optimizer, _grads
                return jnp.asarray(0, dtype=jnp.int32)

            def apply_appearance_update(
                current_module, current_optimizer, current_grads
            ):
                current_optimizer.update(current_module, current_grads)
                return jnp.asarray(0, dtype=jnp.int32)

            nnx.cond(
                has_overflow,
                skip_appearance_update,
                apply_appearance_update,
                appearance_module,
                appearance_optimizer,
                appearance_grads,
            )

        candidate_limit_exceeded_tiles = jnp.count_nonzero(
            info["candidate_limit_exceeded"]
        )
        intersection_count = jnp.sum(info["intersection_count"])
        intersection_required_count = jnp.max(
            info["intersection_required_count"]
        )
        if distributed:
            assert distributed_axis_name is not None
            candidate_limit_exceeded_tiles = jax.lax.psum(
                candidate_limit_exceeded_tiles, distributed_axis_name
            )
            intersection_count = jax.lax.psum(
                intersection_count, distributed_axis_name
            )
            intersection_required_count = jax.lax.pmax(
                intersection_required_count, distributed_axis_name
            )

        return {
            "loss": loss,
            "l1": l1_value,
            "ssim": ssim_value,
            "normal_loss": normal_loss_value,
            "distortion_loss": distortion_loss_value,
            "opacity_reg_loss": opacity_reg_loss_value,
            "scale_reg_loss": scale_reg_loss_value,
            "pose_error": pose_error_value,
            "psnr": psnr(rgb, targets),
            "active_count": model.active_count,
            "visible_count": jnp.count_nonzero(visible),
            "overflow_tiles": overflow_tiles,
            "max_overflow_tiles": max_overflow_tiles,
            "candidate_limit_exceeded_tiles": candidate_limit_exceeded_tiles,
            "intersection_overflow": intersection_overflow,
            "intersection_overflow_seen": intersection_overflow_seen,
            "intersection_count": intersection_count,
            "intersection_required_count": intersection_required_count,
            "distributed_state_mismatch": distributed_state_mismatch,
            **plan_metrics,
        }

    return train_step


def make_train_step(
    config: TrainConfig,
) -> Callable[..., dict[str, jax.Array]]:
    """Create the ordinary single-process training step."""

    return _make_train_step(config)


def make_distributed_train_step(
    config: TrainConfig,
    *,
    world_size: int,
    axis_name: Hashable = "rank",
    scene_scale: float = 1.0,
) -> Callable[..., dict[str, jax.Array]]:
    """Create the first current-main Gaussian-sharded training slice.

    The returned stateful step must run inside ``nnx.pmap`` (or ``nnx.vmap``
    for tests) with ``axis_name`` bound. This slice supports only dense,
    pinhole 3DGS with SH colors. Host data sharding and distributed
    checkpoints remain separate orchestration work, so :func:`train` continues
    to fail fast for multiple JAX processes. ``scene_scale`` must match the
    value passed to the Gaussian optimizer. A rank mismatch in optimizer step
    or SH degree returns ``distributed_state_mismatch=True`` and atomically
    skips the update. Signed screen-space statistics are reduced into each
    Gaussian owner; current-main distributed rendering does not support
    AbsGrad. Camera-pose optimization and pose noise are supported: the
    replicated module's gradient is averaged across ranks, matching the DDP
    wrapper current-main puts around it, while Gaussians keep their sharded
    sum. Appearance stays rejected because its per-view colors have no
    distributed route upstream either.

    Refinement is planned, preflighted, and committed inside the step. Before
    the update every rank plans duplicate/split/prune events for the rows it
    owns; a planned capacity overflow on any single rank atomically skips the
    whole step on every rank so a host can grow all shards and replay. After
    the update each owner commits its own duplicate/split/prune and scheduled
    opacity reset with the ordinary :class:`DefaultStrategy`, matching
    current-main's post-optimizer callback order. Physical shard capacity
    never changes here, so growing a bucket, resharding, and distributed
    checkpoints remain host work. All collectives stay outside conditionals:
    plan summaries are reduced before the update and commit counters after it.
    """

    try:
        world_size = operator.index(world_size)
    except TypeError as exc:
        raise TypeError("world_size must be an integer") from exc
    if world_size <= 1:
        raise ValueError("world_size must be greater than one")
    if axis_name is None or not isinstance(axis_name, Hashable):
        raise TypeError("axis_name must be hashable")
    try:
        scene_scale = float(scene_scale)
    except (TypeError, ValueError) as exc:
        raise TypeError("scene_scale must be a real scalar") from exc
    if not math.isfinite(scene_scale) or scene_scale < 0.0:
        raise ValueError("scene_scale must be finite and non-negative")
    if config.optimizer.max_steps != config.steps:
        raise ValueError(
            "distributed training requires optimizer.max_steps to equal "
            "TrainConfig.steps so every rank uses one schedule horizon"
        )
    if config.model_type != "3dgs":
        raise NotImplementedError(
            "distributed training does not support 2DGS"
        )
    if config.app_opt:
        raise NotImplementedError(
            "distributed appearance training requires gather-before-MLP "
            "camera colors and is not part of the first slice"
        )
    if config.packed or config.sparse_grad or config.visible_adam:
        raise NotImplementedError(
            "the first distributed training slice supports dense Adam only"
        )
    if config.strategy.absgrad:
        raise NotImplementedError(
            "current-main distributed rendering does not support AbsGrad"
        )
    if config.strategy.kind != "default":
        raise NotImplementedError(
            "the first distributed training slice supports DefaultStrategy"
        )
    if (
        config.with_ut
        or config.with_eval3d
        or config.camera_model != "pinhole"
    ):
        raise NotImplementedError(
            "the first distributed training slice supports standard pinhole "
            "EWA rasterization only"
        )
    return _make_train_step(
        config,
        distributed_world_size=world_size,
        distributed_axis_name=axis_name,
        distributed_scene_scale=scene_scale,
    )


def make_render_step(config: TrainConfig, width: int, height: int):
    _validate_2dgs_mode(config)

    @nnx.jit
    def render_step(
        model: GaussianModel,
        viewmat: jax.Array,
        K: jax.Array,
        sh_degree: jax.Array,
        *,
        appearance_module: AppearanceOptModule | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if model.has_appearance != config.app_opt:
            raise ValueError(
                "model color representation must match config.app_opt"
            )
        parameters = model.activated(
            split_sh=config.model_type == "3dgs" and not config.app_opt
        )
        if config.app_opt:
            if appearance_module is None:
                raise ValueError(
                    "app_opt=True requires appearance_module for rendering"
                )
            camtoworld = _invert_rigid_transforms(viewmat[None, ...])
            directions = (
                parameters["means"][None, :, :]
                - camtoworld[:, None, :3, 3]
            )
            corrections = appearance_module(
                parameters["features"], None, directions, sh_degree
            )
            render_colors = jax.nn.sigmoid(
                parameters["colors"][None, :, :] + corrections
            )
            raster_sh_degree = None
        else:
            render_colors = parameters["sh_coeffs"]
            raster_sh_degree = sh_degree
        if config.model_type == "2dgs":
            renders, alphas, _, _, _, _, info = rasterization_2dgs(
                parameters["means"],
                parameters["quats"],
                parameters["scales"],
                parameters["opacities"],
                render_colors,
                viewmat[None, ...],
                K[None, ...],
                width,
                height,
                packed=False,
                active_mask=parameters["active_mask"],
                sh_degree=raster_sh_degree,
                render_mode="RGB",
                config=config.rasterizer,
            )
        else:
            renders, alphas, info = rasterization(
                parameters["means"],
                parameters["quats"],
                parameters["scales"],
                parameters["opacities"],
                render_colors,
                viewmat[None, ...],
                K[None, ...],
                width,
                height,
                active_mask=parameters["active_mask"],
                sh_degree=raster_sh_degree,
                camera_model=config.camera_model,
                with_ut=config.with_ut,
                with_eval3d=config.with_eval3d,
                config=config.rasterizer,
            )
        return (
            renders[0, ..., :3],
            alphas[0],
            info["tile_overflow"][0],
            info["intersection_overflow"][0],
        )

    return render_step


def _save_render(path: Path, image: jax.Array) -> None:
    pixels = np.asarray(jax.device_get(jnp.clip(image, 0.0, 1.0) * 255.0)).astype(
        np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def train(
    config: TrainConfig, *, resume_from: str | Path | None = None
) -> TrainingResult:
    """Train with compact active prefixes and bucketed physical storage."""

    world_size = jax.process_count()
    if world_size != 1:
        raise NotImplementedError(
            "train() currently supports single-process execution only; "
            "distributed rasterization is available, but synchronized "
            "Gaussian, pose, and appearance optimizer/checkpoint state is "
            f"not implemented for process_count={world_size}"
        )
    if resume_from is not None:
        _validate_camera_module_resume_config(config, resume_from)
    output_dir = Path(config.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")
    scene = load_colmap_scene(
        config.data.root,
        image_dir=config.data.image_dir,
        load_points=resume_from is None,
    )
    training_height, training_width = _scene_training_render_size(scene, config)
    saved_scene_transform = (
        load_checkpoint_scene_transform(resume_from)
        if resume_from is not None
        else None
    )
    if saved_scene_transform is not None:
        saved_matrix, scene_scale = saved_scene_transform
        transform = SceneTransform(saved_matrix)
    elif resume_from is not None:
        transform = _legacy_scene_transform(scene)
        scene_scale = _legacy_training_scene_scale(scene)
    else:
        transform = compute_scene_transform(
            scene,
            normalize_world_space=config.normalize_world_space,
        )
        scene_scale = _training_scene_scale(
            scene, transform, global_scale=config.global_scale
        )
    uses_camera_modules = (
        config.pose_opt or config.pose_noise > 0.0 or config.app_opt
    )
    camera_image_names: tuple[str, ...] | None = None
    camera_count = 0
    if uses_camera_modules:
        camera_indices = scene.indices("train", config.data.test_every)
        camera_image_names = tuple(
            scene.images[int(index)].name for index in camera_indices
        )
        camera_count = len(camera_image_names)
        if camera_count == 0:
            raise ValueError(
                "camera-conditioned optimization requires a non-empty "
                "training split"
            )
    if resume_from is None:
        points = transform.points(scene.points)
        storage_capacity = _initial_storage_capacity(config, len(points))
    else:
        storage_capacity = load_checkpoint_storage_capacity(resume_from)
        if storage_capacity > config.model.capacity:
            raise ValueError(
                f"checkpoint physical capacity {storage_capacity} exceeds the "
                f"configured logical maximum {config.model.capacity}"
            )

    intersection_limit = _training_intersection_limit(
        config,
        storage_capacity,
        image_height=training_height,
        image_width=training_width,
    )
    saved_intersection_capacity = (
        load_checkpoint_intersection_capacity(resume_from)
        if resume_from is not None
        else None
    )
    intersection_capacity = _intersection_bucket_capacity(
        saved_intersection_capacity or 1,
        minimum=config.intersection_bucket_min_capacity,
        maximum=intersection_limit,
    )
    runtime_config = _training_config_with_intersection_capacity(
        config, intersection_capacity
    )
    _check_memory_budget(
        runtime_config,
        physical_capacity=storage_capacity,
        label="initial_training" if resume_from is None else "resume_training",
        image_height=training_height,
        image_width=training_width,
    )
    if resume_from is None:
        model = GaussianModel.from_point_cloud(
            points,
            scene.points_rgb,
            config.model,
            physical_capacity=storage_capacity,
            num_workers=config.data.num_workers,
            appearance_feature_dim=(
                APPEARANCE_FEATURE_DIM if config.app_opt else None
            ),
            feature_key=(
                jax.random.fold_in(
                    jax.random.key(config.seed), 0x41505046
                )
                if config.app_opt
                else None
            ),
        )
    else:
        model = GaussianModel.empty(
            config.model,
            physical_capacity=storage_capacity,
            appearance_feature_dim=(
                APPEARANCE_FEATURE_DIM if config.app_opt else None
            ),
        )
    optimizer = _create_training_optimizer(
        model, config, scene_scale=scene_scale
    )
    pose_adjust = None
    pose_optimizer = None
    if config.pose_opt:
        pose_adjust = CameraOptModule(
            camera_count,
            rngs=nnx.Rngs(
                jax.random.fold_in(jax.random.key(config.seed), 0x504F5345)
            ),
        )
        pose_adjust.zero_init()
        pose_optimizer = _create_pose_optimizer(pose_adjust, config)
    pose_perturb = None
    if config.pose_noise > 0.0:
        pose_perturb = CameraOptModule(
            camera_count,
            rngs=nnx.Rngs(
                jax.random.fold_in(jax.random.key(config.seed), 0x4E4F4953)
            ),
        )
        pose_perturb.random_init(config.pose_noise)
    appearance_module = None
    appearance_optimizer = None
    if config.app_opt:
        appearance_module = AppearanceOptModule(
            camera_count,
            APPEARANCE_FEATURE_DIM,
            config.app_embed_dim,
            config.model.sh_degree,
            rngs=nnx.Rngs(
                jax.random.fold_in(
                    jax.random.key(config.seed), 0x4150504D
                )
            ),
        )
        appearance_optimizer = create_appearance_optimizer(
            appearance_module, config
        )
    strategy = (
        MCMCStrategy(config.strategy)
        if config.strategy.kind == "mcmc"
        else DefaultStrategy(config.strategy)
    )
    strategy_state = strategy.initialize_state(model.capacity)
    strategy_state.scene_scale[...] = scene_scale
    start_step = 0
    if resume_from is not None:
        start_step = restore_checkpoint(
            resume_from,
            model,
            optimizer=optimizer,
            strategy_state=strategy_state,
            pose_module=pose_adjust,
            pose_optimizer=pose_optimizer,
            pose_image_names=(
                camera_image_names
                if config.pose_opt or config.pose_noise > 0.0
                else None
            ),
            appearance_module=appearance_module,
            appearance_optimizer=appearance_optimizer,
            appearance_image_names=(
                camera_image_names if config.app_opt else None
            ),
        )
        if start_step > config.steps:
            raise ValueError(
                f"checkpoint step {start_step} exceeds configured training "
                f"steps {config.steps}"
            )
        if not load_checkpoint_active_prefix(resume_from):
            compact_count = compact_training_state(
                model, optimizer, strategy_state
            )
            compact_count.block_until_ready()
            print(
                f"compacted_legacy_checkpoint active={int(compact_count)}",
                flush=True,
            )

    print(
        f"storage_capacity={model.capacity} max_capacity={model.max_capacity} "
        f"active={int(jax.device_get(model.active_count))} "
        f"intersection_capacity={intersection_capacity}/{intersection_limit}",
        flush=True,
    )

    dataset = create_grain_dataset(
        scene,
        split="train",
        test_every=config.data.test_every,
        shuffle=True,
        seed=config.data.shuffle_seed,
        repeat=True,
        batch_size=config.data.batch_size,
        drop_remainder=True,
    )
    batches = _infinite_batches(dataset, num_workers=config.data.num_workers)
    if start_step < config.steps:
        for _ in range(start_step):
            next(batches)
    train_step = make_train_step(runtime_config)
    safety_state = TrainingSafetyState()
    pending_steps: list[_PendingTrainStep] = []
    training_key = jax.random.key(config.seed)
    last_metrics: dict[str, float] = {}
    last_checkpoint: Path | None = None
    last_checkpoint_step: int | None = None
    start_time = time.monotonic()

    evaluation_example = None
    evaluation_render_step = None
    if config.eval_every > 0:
        evaluation_source = create_grain_dataset(
            scene,
            split="test",
            test_every=config.data.test_every,
            shuffle=False,
            batch_size=None,
        )
        evaluation_example = evaluation_source[0]
        eval_height, eval_width = evaluation_example["image"].shape[:2]
        evaluation_render_step = make_render_step(config, eval_width, eval_height)

    def run_device_train_step(
        images: jax.Array,
        intrinsics: jax.Array,
        viewmats: jax.Array,
        step_key: jax.Array,
        sh_degree: jax.Array,
        strategy_key: jax.Array,
        *,
        camtoworlds: jax.Array | None,
        image_ids: jax.Array | None,
    ) -> dict[str, jax.Array]:
        camera_kwargs: dict[str, Any] = {}
        if uses_camera_modules:
            camera_kwargs = {
                "pose_adjust": pose_adjust,
                "pose_optimizer": pose_optimizer,
                "pose_perturb": pose_perturb,
                "camtoworlds": camtoworlds,
                "image_ids": image_ids,
                "appearance_module": appearance_module,
                "appearance_optimizer": appearance_optimizer,
            }
        return train_step(
            model,
            optimizer,
            strategy_state,
            safety_state,
            images,
            intrinsics,
            viewmats,
            step_key,
            sh_degree,
            strategy_key,
            **camera_kwargs,
        )

    def synchronize_pending_steps() -> dict[str, jax.Array] | None:
        """Resolve sticky overflow and replay the uncommitted suffix."""

        nonlocal intersection_capacity
        nonlocal intersection_limit
        nonlocal runtime_config
        nonlocal safety_state
        nonlocal train_step

        if not pending_steps:
            return None
        max_overflow_tiles, intersection_overflow_seen = (
            _training_overflow_status(safety_state)
        )
        if max_overflow_tiles > 0:
            _raise_training_overflow(
                np.asarray(max_overflow_tiles),
                np.asarray(intersection_overflow_seen),
            )
        while intersection_overflow_seen:
            replay_start, required_intersections = _pending_overflow_suffix(
                pending_steps
            )
            intersection_limit = _training_intersection_limit(
                config,
                model.capacity,
                image_height=training_height,
                image_width=training_width,
            )
            next_intersection_capacity = _intersection_bucket_capacity(
                required_intersections,
                minimum=config.intersection_bucket_min_capacity,
                maximum=intersection_limit,
            )
            if next_intersection_capacity <= intersection_capacity:
                raise RuntimeError(
                    "intersection overflow did not request a larger capacity "
                    f"({required_intersections} required, "
                    f"{intersection_capacity} configured)"
                )
            next_runtime_config = _training_config_with_intersection_capacity(
                config, next_intersection_capacity
            )
            _check_memory_budget(
                next_runtime_config,
                physical_capacity=model.capacity,
                label="intersection_bucket_growth",
                image_height=training_height,
                image_width=training_width,
            )
            old_intersection_capacity = intersection_capacity
            del train_step
            jax.clear_caches()
            gc.collect()
            runtime_config = next_runtime_config
            intersection_capacity = next_intersection_capacity
            train_step = make_train_step(runtime_config)
            safety_state = TrainingSafetyState()
            print(
                "intersection_capacity_growth="
                f"{old_intersection_capacity}->{intersection_capacity} "
                f"required={required_intersections} "
                f"replay_steps={len(pending_steps) - replay_start}",
                flush=True,
            )
            for record in pending_steps[replay_start:]:
                record.metrics = run_device_train_step(
                    jax.device_put(record.images),
                    jax.device_put(record.intrinsics),
                    jax.device_put(record.viewmats),
                    record.key,
                    record.sh_degree,
                    record.strategy_key,
                    camtoworlds=(
                        None
                        if record.camtoworlds is None
                        else jax.device_put(record.camtoworlds)
                    ),
                    image_ids=(
                        None
                        if record.image_ids is None
                        else jax.device_put(record.image_ids)
                    ),
                )
            max_overflow_tiles, intersection_overflow_seen = (
                _training_overflow_status(safety_state)
            )
            if max_overflow_tiles > 0:
                _raise_training_overflow(
                    np.asarray(max_overflow_tiles),
                    np.asarray(intersection_overflow_seen),
                )
        latest_metrics = pending_steps[-1].metrics
        pending_steps.clear()
        return latest_metrics

    for step in range(start_step + 1, config.steps + 1):
        batch = next(batches)
        images_np = np.asarray(batch["image"], dtype=np.float32)
        intrinsics_np = np.asarray(batch["K"], dtype=np.float32)
        viewmats_np = transform.world_to_camera(
            np.asarray(batch["w2c"], dtype=np.float32)
        )
        camtoworlds_np = None
        image_ids_np = None
        if uses_camera_modules:
            camtoworlds_np = np.linalg.inv(viewmats_np).astype(np.float32)
            image_ids_np = np.asarray(batch["dataset_index"], dtype=np.int32)
        images = jax.device_put(images_np)
        intrinsics = jax.device_put(intrinsics_np)
        viewmats = jax.device_put(viewmats_np)
        camtoworlds = (
            None if camtoworlds_np is None else jax.device_put(camtoworlds_np)
        )
        image_ids = (
            None if image_ids_np is None else jax.device_put(image_ids_np)
        )
        step_key, strategy_key = jax.random.split(
            jax.random.fold_in(training_key, step), 2
        )
        do_refine = strategy.should_refine(step)
        do_reset = strategy.should_reset(step)
        sh_degree = jnp.minimum(
            config.model.sh_degree,
            step // max(config.sh_degree_interval, 1),
        )
        if config.strategy.kind == "mcmc" and do_refine:
            # Refinement is part of the atomic device commit below. Resolve any
            # older overflow before changing the physical storage bucket.
            synchronize_pending_steps()
            active_count = int(jax.device_get(model.active_count))
            required_capacity = _mcmc_required_capacity(active_count, config)
            bounded_required = min(required_capacity, model.max_capacity)
            target_capacity = max(
                model.capacity,
                config.model.bucket_capacity(bounded_required),
            )
            if target_capacity > model.capacity:
                old_capacity = model.capacity
                model, optimizer, strategy_state = _grow_training_state(
                    runtime_config,
                    model,
                    optimizer,
                    strategy_state,
                    target_capacity,
                    image_height=training_height,
                    image_width=training_width,
                )
                jax.clear_caches()
                gc.collect()
                print(
                    f"capacity_growth={old_capacity}->{target_capacity} "
                    f"required={required_capacity}",
                    flush=True,
                )
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            metrics = run_device_train_step(
                images,
                intrinsics,
                viewmats,
                step_key,
                sh_degree,
                strategy_key,
                camtoworlds=camtoworlds,
                image_ids=image_ids,
            )
        pending_steps.append(
            _PendingTrainStep(
                images=images_np,
                intrinsics=intrinsics_np,
                viewmats=viewmats_np,
                camtoworlds=camtoworlds_np,
                image_ids=image_ids_np,
                key=step_key,
                strategy_key=strategy_key,
                sh_degree=sh_degree,
                metrics=metrics,
            )
        )
        do_log = step == 1 or step % 10 == 0
        do_checkpoint = (
            config.checkpoint_every > 0
            and step % config.checkpoint_every == 0
        )
        do_evaluate = (
            evaluation_example is not None
            and evaluation_render_step is not None
            and config.eval_every > 0
            and step % config.eval_every == 0
        )
        if (
            do_refine
            or do_reset
            or do_log
            or do_checkpoint
            or do_evaluate
            or step == config.steps
        ):
            synchronized_metrics = synchronize_pending_steps()
            if synchronized_metrics is not None:
                metrics = synchronized_metrics

        if do_refine and config.strategy.kind == "default":
            required_capacity = int(
                jax.device_get(
                    strategy.required_capacity(
                        model, strategy_state, scene_scale, step=step
                    )
                )
            )
            bounded_required = min(required_capacity, model.max_capacity)
            target_capacity = max(
                model.capacity,
                config.model.bucket_capacity(bounded_required),
            )
            if target_capacity > model.capacity:
                old_capacity = model.capacity
                model, optimizer, strategy_state = _grow_training_state(
                    runtime_config,
                    model,
                    optimizer,
                    strategy_state,
                    target_capacity,
                    image_height=training_height,
                    image_width=training_width,
                )
                # Bucket shapes only grow, so the old executable will never be
                # reused. Release its compiler cache before compiling the next
                # large shape to avoid cumulative host-memory pressure.
                jax.clear_caches()
                gc.collect()
                print(
                    f"capacity_growth={old_capacity}->{target_capacity} "
                    f"required={required_capacity}",
                    flush=True,
                )
            refine_metrics = strategy.refine(
                model,
                strategy_state,
                optimizer,
                strategy_key,
                scene_scale,
                step=step,
            )
            metrics = {
                **metrics,
                "active_count": model.active_count,
                **{f"refine/{k}": v for k, v in refine_metrics.items()},
            }
        if do_reset:
            reset_opacities(
                model,
                optimizer,
                maximum_opacity=config.strategy.reset_opacity,
            )

        if do_log:
            last_metrics = {
                name: float(jax.device_get(value)) for name, value in metrics.items()
            }
            elapsed = time.monotonic() - start_time
            bytes_in_use, bytes_limit = _device_memory_usage()
            memory_text = (
                f" gpu={bytes_in_use / 2**30:.2f}/{bytes_limit / 2**30:.2f}GiB"
                if bytes_limit
                else ""
            )
            regularization_text = (
                f"normal={last_metrics['normal_loss']:.6f} "
                f"distortion={last_metrics['distortion_loss']:.6f} "
                if config.model_type == "2dgs"
                else ""
            )
            pose_text = (
                f"pose_error={last_metrics['pose_error']:.6f} "
                if config.pose_opt and config.pose_noise > 0.0
                else ""
            )
            print(
                f"step={step:06d} loss={last_metrics['loss']:.6f} "
                f"psnr={last_metrics['psnr']:.2f} "
                f"{regularization_text}"
                f"{pose_text}"
                f"active={int(last_metrics['active_count'])} "
                f"storage={model.capacity}/{model.max_capacity} "
                f"overflow_tiles={int(last_metrics['overflow_tiles'])} "
                f"candidate_limit_exceeded={int(last_metrics['candidate_limit_exceeded_tiles'])} "
                f"intersection_overflow={bool(last_metrics['intersection_overflow'])} "
                f"intersections={int(last_metrics['intersection_required_count'])}/"
                f"{intersection_capacity} "
                f"elapsed={elapsed:.1f}s{memory_text}",
                flush=True,
            )
            if bytes_limit and bytes_in_use > int(bytes_limit * 0.85):
                raise MemoryError(
                    "JAX device memory usage exceeded the 85% safety threshold. "
                    "Training stopped before the next step; reduce capacity or "
                    "rasterizer candidate limits."
                )

        if do_checkpoint:
            last_checkpoint = _save_compacted_training_checkpoint(
                output_dir / "checkpoints",
                model,
                optimizer,
                strategy_state,
                step=step,
                config=config,
                intersection_capacity=intersection_capacity,
                scene_transform=transform,
                scene_scale=scene_scale,
                pose_adjust=pose_adjust,
                pose_optimizer=pose_optimizer,
                pose_image_names=(
                    camera_image_names
                    if config.pose_opt or config.pose_noise > 0.0
                    else None
                ),
                appearance_module=appearance_module,
                appearance_optimizer=appearance_optimizer,
                appearance_image_names=(
                    camera_image_names if config.app_opt else None
                ),
            )
            last_checkpoint_step = step

        if do_evaluate:
            _check_evaluation_memory_budget(
                config,
                physical_capacity=model.capacity,
                width=eval_width,
                height=eval_height,
            )
            normalized_viewmat = transform.world_to_camera(evaluation_example["w2c"])
            rendered, _, overflow, intersection_overflow = evaluation_render_step(
                model,
                jax.device_put(normalized_viewmat),
                jax.device_put(evaluation_example["K"]),
                jnp.asarray(config.model.sh_degree),
                appearance_module=appearance_module,
            )
            _save_render(output_dir / "renders" / f"step_{step:08d}.png", rendered)
            overflow_count = int(jax.device_get(jnp.count_nonzero(overflow)))
            if overflow_count:
                print(f"evaluation tile overflow: {overflow_count}", flush=True)
            if bool(jax.device_get(intersection_overflow)):
                print("evaluation intersection overflow", flush=True)

    if last_checkpoint_step != config.steps:
        last_checkpoint = _save_compacted_training_checkpoint(
            output_dir / "checkpoints",
            model,
            optimizer,
            strategy_state,
            step=config.steps,
            config=config,
            intersection_capacity=intersection_capacity,
            scene_transform=transform,
            scene_scale=scene_scale,
            pose_adjust=pose_adjust,
            pose_optimizer=pose_optimizer,
            pose_image_names=(
                camera_image_names
                if config.pose_opt or config.pose_noise > 0.0
                else None
            ),
            appearance_module=appearance_module,
            appearance_optimizer=appearance_optimizer,
            appearance_image_names=(
                camera_image_names if config.app_opt else None
            ),
        )
    return TrainingResult(
        model=model,
        final_step=config.steps,
        output_dir=output_dir,
        checkpoint=last_checkpoint,
        metrics=last_metrics,
        pose_adjust=pose_adjust,
        appearance=appearance_module,
    )
