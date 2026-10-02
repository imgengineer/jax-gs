"""Compile scheduled training variants while preserving the original NNX buffers."""

from collections.abc import Callable, Iterable
from time import perf_counter

import chex
import jax
import jax.numpy as jnp

from ..config import TrainingConfig
from ..scene.camera import Camera
from ..scene.spatial_refine import reorder_gaussians
from ..scene.types import WorldClusterBounds
from .densify import decay_opacity, densify_step
from .state import TrainingState


def precompile_training(
    training_state: TrainingState,
    train_step_fn: Callable[..., tuple[chex.Array, chex.Array, chex.Array]],
    cluster_bounds: WorldClusterBounds,
    views: Iterable[tuple[Camera, chex.Array]],
    key: chex.Array,
    initial_sparse_count: int,
    scene_radius: chex.Array,
    settings: TrainingConfig,
    training_modes: Iterable[tuple[int, bool]],
) -> float:
    """Compile scheduled SH/statistics modes for each view shape before timing."""
    capacity_config = settings.capacity
    densification = settings.densify
    training_modes = sorted(set(training_modes))
    warmup_start = perf_counter()
    pool = training_state.model.as_arrays()
    adam_state = training_state.adam.get_value()
    fragment_stats = training_state.fragments.get_value()
    # Donation consumes only this working copy; reuse its returned buffers
    # across variants instead of copying the full-capacity state every time.
    warm_pool, warm_adam_state, warm_fragment_stats = jax.tree.map(
        jnp.copy, (pool, adam_state, fragment_stats)
    )
    training_state.model.update_from_arrays(warm_pool)
    training_state.adam.set_value(warm_adam_state)
    training_state.fragments.set_value(warm_fragment_stats)
    try:
        compiled_view_signatures = set()
        for camera, target in views:
            view_signature = (camera.width, camera.height, camera.near, camera.far)
            if view_signature in compiled_view_signatures:
                continue
            compiled_view_signatures.add(view_signature)
            for sh_degree, collect_stats in training_modes:
                step_result = train_step_fn(
                    cluster_bounds,
                    camera,
                    target,
                    jnp.array(0, jnp.int32),
                    scene_radius,
                    active_degree=sh_degree,
                    collect_stats=collect_stats,
                    overflow=jnp.array(False),
                    peak_pairs=jnp.array(0, jnp.int32),
                )
                jax.block_until_ready(
                    (
                        step_result,
                        training_state.adam.get_value(),
                        training_state.fragments.get_value(),
                        training_state.model.as_arrays(),
                    )
                )
    finally:
        training_state.model.update_from_arrays(pool)
        training_state.adam.set_value(adam_state)
        training_state.fragments.set_value(fragment_stats)
    densify_step.lower(
        pool,
        adam_state,
        fragment_stats,
        key,
        jnp.array(initial_sparse_count, jnp.int32),
        scene_radius,
        cluster_size=capacity_config.cluster_size,
        percent_dense=densification.percent_dense,
    ).compile()
    jax.block_until_ready(decay_opacity(pool, adam_state))
    jax.block_until_ready(reorder_gaussians(pool, adam_state))
    return perf_counter() - warmup_start
