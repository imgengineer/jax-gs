"""Dataset iteration and patch sampling for the training loop."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax


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
