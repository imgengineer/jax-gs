"""Training-view selection and fixed-capacity initialization from COLMAP."""

import json
from pathlib import Path

import numpy as np

from ..config import CapacityConfig, ModelConfig
from ..data import Frame
from ..io_manager.colmap import load_colmap_images, load_colmap_points
from ..scene.point import (
    GaussianArrays,
    create_gaussians,
    estimate_initial_scales,
    seed_gaussians,
)
from .optimizer import AdamState, create_adam_state


def load_training_frames(scene: str | Path, model_config: ModelConfig) -> list[Frame]:
    """Load configured views and apply the training split when evaluation is enabled."""
    frames = load_colmap_images(scene, model_config.images, resolution=model_config.resolution)
    if model_config.eval:
        split_path = Path(scene) / "train_test_split.json"
        if split_path.exists():
            names = set(json.loads(split_path.read_text())["train"])
            frames = [
                frame
                for frame in frames
                if frame.image_path.stem in names or frame.image_path.name in names
            ]
        else:
            frames = [frame for index, frame in enumerate(frames) if index % 8 != 0]
    if not frames:
        raise ValueError("no training images selected")
    return frames


def initialize_pool(
    scene: str | Path, config: CapacityConfig
) -> tuple[GaussianArrays, AdamState, int, int]:
    """Create the fixed pool; return sparse and cluster-padded initial counts."""
    xyz, rgb = load_colmap_points(scene)
    initial_point_count = len(xyz)
    if initial_point_count == 0:
        raise ValueError("initial cloud is empty")
    initial_scales = estimate_initial_scales(xyz)
    # LiteGS cluster_points pads its initial cloud with copies of the final
    # points. Gather those same copies into slots without resizing the pool.
    cluster_padding = (-initial_point_count) % config.cluster_size
    point_indices = np.arange(initial_point_count + cluster_padding)
    point_indices[initial_point_count:] = (
        np.arange(initial_point_count - cluster_padding, initial_point_count) % initial_point_count
    )
    if len(point_indices) > config.max_gaussians:
        raise ValueError("initial cloud including cluster padding exceeds capacity")
    pool = seed_gaussians(
        create_gaussians(config),
        xyz[point_indices],
        rgb[point_indices],
        scale=initial_scales[point_indices],
        opacity=0.1,
    )
    adam_state = create_adam_state(pool)
    # Share the initial zeros only during precompilation. The epoch loop
    # separates v after warmup, preserving the existing peak memory budget.
    adam_state = adam_state.replace(v=adam_state.m)
    return pool, adam_state, initial_point_count, len(point_indices)
