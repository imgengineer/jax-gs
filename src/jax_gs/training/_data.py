# pyright: reportMissingImports=false

"""Dataset iteration and patch sampling for the training loop."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
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
    dataset: Any, *, num_workers: int, start_batch: int = 0
) -> Iterator[dict[str, Any]]:
    if start_batch:
        import grain

        if isinstance(dataset, grain.MapDataset):
            # Slice after shuffle/repeat/batch so absolute batch indices retain
            # their original random order without decoding discarded images.
            dataset = dataset[start_batch:]
            start_batch = 0
    dataset = _grain_iter_dataset(dataset, num_workers)
    batches = itertools.chain.from_iterable(itertools.repeat(dataset))
    yield from itertools.islice(batches, start_batch, None)


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


def shard_camera_batch(batch: dict[str, Any], world_size: int) -> dict[str, Any]:
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
        assert per_rank is not None
        sharded[name] = value.reshape((world_size, per_rank) + tuple(value.shape[1:]))
    return sharded


class StepKeyGenerator:
    """Vectorized on-device PRNG key generator for single-card and distributed loops."""

    def __init__(
        self,
        base_key: jax.Array,
        *,
        world_size: int = 1,
        chunk_size: int = 1024,
    ) -> None:
        self.base_key = base_key
        self.world_size = world_size
        self.chunk_size = chunk_size
        self._cached_step_keys: jax.Array | None = None
        self._cached_strat_keys: jax.Array | None = None
        self._cached_start: int = -1

    def get(self, step: int) -> tuple[jax.Array, jax.Array]:
        offset = step - self._cached_start
        if self._cached_step_keys is None or offset < 0 or offset >= self.chunk_size:
            self._cached_start = step
            steps_vec = step + jnp.arange(self.chunk_size, dtype=jnp.int32)
            if self.world_size == 1:

                def make_pair(s):
                    pair = jax.random.split(jax.random.fold_in(self.base_key, s), 2)
                    return pair[0], pair[1]

                self._cached_step_keys, self._cached_strat_keys = jax.vmap(make_pair)(
                    steps_vec
                )
            else:
                ranks_vec = jnp.arange(self.world_size, dtype=jnp.int32)

                def make_step(s):
                    def make_rank(r):
                        k = jax.random.fold_in(jax.random.fold_in(self.base_key, s), r)
                        pair = jax.random.split(k, 2)
                        return pair[0], pair[1]

                    return jax.vmap(make_rank)(ranks_vec)

                self._cached_step_keys, self._cached_strat_keys = jax.vmap(make_step)(
                    steps_vec
                )
            offset = 0
        if self._cached_step_keys is None or self._cached_strat_keys is None:
            raise RuntimeError("failed to generate step keys")
        return self._cached_step_keys[offset], self._cached_strat_keys[offset]


class PrecomputedCameraPoses:
    """Precomputed normalized camera matrices for fast O(1) batch slicing."""

    def __init__(
        self,
        scene: Any,
        transform: Any,
        *,
        uses_camera_modules: bool = False,
    ) -> None:
        self.transform = transform
        self.uses_camera_modules = uses_camera_modules
        self.viewmats: np.ndarray | None = None
        self.camtoworlds: np.ndarray | None = None

        if (
            hasattr(scene, "worldtocams")
            and hasattr(scene, "images")
            and len(scene.images) > 0
        ):
            self.viewmats = transform.world_to_camera(scene.worldtocams).astype(
                np.float32
            )
            if uses_camera_modules and hasattr(scene, "camtoworlds"):
                self.camtoworlds = transform.camera_to_world(scene.camtoworlds).astype(
                    np.float32
                )

    def get_poses(
        self,
        raw_w2c: Any,
        image_indices: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        w2c_np = np.asarray(raw_w2c, dtype=np.float32)
        if self.viewmats is not None and image_indices is not None:
            viewmats_np = self.viewmats[np.asarray(image_indices)]
        else:
            viewmats_np = self.transform.world_to_camera(w2c_np)

        camtoworlds_np = None
        if self.uses_camera_modules:
            if self.camtoworlds is not None and image_indices is not None:
                camtoworlds_np = self.camtoworlds[np.asarray(image_indices)]
            else:
                camtoworlds_np = np.linalg.inv(viewmats_np).astype(np.float32)

        return viewmats_np, camtoworlds_np
