"""Dataset iteration and patch sampling for the training loop."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import numpy as np


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


def shard_camera_batch(
    batch: dict[str, Any], world_size: int
) -> dict[str, Any]:
    """Deal one host batch of ``world_size * B`` cameras out to the ranks.

    Upstream's distributed trainer gives every rank its own shuffled loader,
    so one step consumes ``world_size`` independent camera batches and the
    optimizer already scales for that effective batch. This port keeps a
    single host stream sized ``world_size`` times the per-rank batch and
    reshapes each field to ``[world_size, B, ...]`` for the mapped step.
    Within one step the ranks therefore see distinct cameras, where upstream's
    independent loaders may collide; each rank's marginal draw is the same.

    Every field must carry the cameras on its leading axis. Fields that grain
    collates as lists, such as image names, are stacked first.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not batch:
        raise ValueError("cannot shard an empty batch")
    sharded: dict[str, Any] = {}
    per_rank: int | None = None
    for name, value in batch.items():
        if isinstance(value, (list, tuple)):
            value = np.asarray(value)
        leading = value.shape[0] if getattr(value, "ndim", 0) else None
        if not leading or leading % world_size:
            raise ValueError(
                f"batch field {name!r} carries {leading} cameras, which does "
                f"not split across world_size={world_size}"
            )
        if per_rank is None:
            per_rank = leading // world_size
        elif leading != per_rank * world_size:
            raise ValueError(
                f"batch field {name!r} carries {leading} cameras while other "
                f"fields carry {per_rank * world_size}"
            )
        sharded[name] = value.reshape(
            (world_size, per_rank) + tuple(value.shape[1:])
        )
    return sharded
