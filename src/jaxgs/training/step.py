"""Compiled array and NNX training steps, independent of host orchestration."""

from collections.abc import Callable
from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import nnx

from ..config import CapacityConfig, OptimizationConfig, load_config
from ..render import render_preprocess
from ..render.types import FragmentStatistics
from ..scene.camera import Camera
from ..scene.point import GaussianArrays, GaussianModel
from ..scene.types import WorldClusterBounds
from .muon import MUON_PROGRAM_SHAPE, create_muon_transform
from .optimizer import AdamState, optax_update, sparse_adam_update
from .state import TrainingState

_DEFAULT_OPTIMIZATION = load_config().optimization
_STATIC_ARGUMENTS = (
    "config",
    "active_degree",
    "collect_stats",
    "max_steps",
    "optimizer",
    "optimization",
)


def compute_training_step(
    pool: GaussianArrays,
    state: AdamState,
    stats: FragmentStatistics,
    bounds: WorldClusterBounds,
    camera: Camera,
    target: chex.Array,
    step: int | chex.Array,
    scene_radius: float | chex.Array,
    config: CapacityConfig,
    active_degree: int,
    collect_stats: bool,
    max_steps: int | None = None,
    optimizer: str = "optax",
    optimization: OptimizationConfig = _DEFAULT_OPTIMIZATION,
) -> tuple[GaussianArrays, AdamState, FragmentStatistics, dict[str, chex.Array]]:
    """Pure array computation shared by the JAX and NNX compilation boundaries."""
    from ..kernels.packed_rasterizer import packed_loss_and_grad
    from ..kernels.projector import project_with_compact_pullback
    from ..kernels.sorted_binning import build_sorted_visibility_table_cute
    from ..kernels.sorted_rasterizer import rasterize_loss_and_grad

    visible_clusters, visible_slots, culled_gaussians = render_preprocess(
        bounds, camera, pool, config
    )

    # SH0 already has just three gradients; deferral only helps higher bands.
    defer_sh = optimizer in ("optax", "muon") and active_degree > 0
    projected_gaussians, projection_pullback = project_with_compact_pullback(
        culled_gaussians,
        camera,
        config,
        active_degree,
        visible_clusters,
        rgb_only=True,
        active_sh_only=optimizer in ("optax", "muon"),
        sh_color_only=defer_sh,
        visible_color_only=True,
    )
    visibility_table = build_sorted_visibility_table_cute(
        jax.lax.stop_gradient(projected_gaussians), camera, config
    )
    # LiteGS's half2 kernel needs at least 64 pixels per tile. Small diagnostic
    # scenes retain the float32 path; production uses LiteGS's packed path.
    loss_and_grad = (
        partial(packed_loss_and_grad, symmetric_conic=True, visible_clusters=visible_clusters)
        if config.tile_size in (8, 16)
        else rasterize_loss_and_grad
    )
    loss, projected_gradients, fragment_stats = loss_and_grad(
        projected_gaussians,
        visibility_table,
        camera,
        config,
        target if target.dtype == jnp.uint8 else target.astype(jnp.float32) / 255,
        collect_stats,
    )
    gradients = projection_pullback(projected_gradients)
    if optimizer in ("optax", "muon"):
        muon = optimizer == "muon"
        pool, state = optax_update(
            pool,
            state,
            gradients,
            visible_slots,
            step,
            scene_radius,
            max_steps,
            cluster_size=config.cluster_size,
            compact_gradients=True,
            optimization=optimization,
            active_degree=active_degree,
            compacted_clusters=visible_clusters,
            transform=create_muon_transform((active_degree + 1) ** 2) if muon else None,
            program_shape=MUON_PROGRAM_SHAPE if muon else None,
            sh_pullback_center=camera.center if defer_sh else None,
        )
    elif optimizer == "cute":
        pool, state = sparse_adam_update(
            pool,
            state,
            gradients,
            visible_slots,
            step,
            scene_radius,
            max_steps,
            compacted_clusters=visible_clusters,
            cluster_size=config.cluster_size,
            compact_gradients=True,
            optimization=optimization,
        )
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")
    if collect_stats:
        stats = stats + fragment_stats
    return (
        pool,
        state,
        stats,
        {"loss": loss, "overflow": visibility_table.overflow, "pairs": visibility_table.pair_count},
    )


array_train_step = jax.jit(compute_training_step, static_argnames=_STATIC_ARGUMENTS)


def _update_model(
    state: AdamState,
    stats: FragmentStatistics,
    model: GaussianModel,
    bounds: WorldClusterBounds,
    camera: Camera,
    target: chex.Array,
    step: int | chex.Array,
    scene_radius: float | chex.Array,
    config: CapacityConfig,
    active_degree: int,
    collect_stats: bool,
    max_steps: int | None,
    overflow: chex.Array,
    peak_pairs: chex.Array,
    optimizer: str = "optax",
    optimization: OptimizationConfig = _DEFAULT_OPTIMIZATION,
) -> tuple[AdamState, FragmentStatistics, chex.Array, chex.Array, chex.Array]:
    # The fixed model is a tree with no shared Variables. NNX tree mode avoids
    # graph protocol, while donation keeps the existing in-place GPU updates.
    # NNX appends mutated Variables after explicit outputs. Keep state/stats
    # before the model so XLA pairs each donated buffer with its own output.
    pool, state, stats, metrics = compute_training_step(
        model.as_arrays(),
        state,
        stats,
        bounds,
        camera,
        target,
        step,
        scene_radius,
        config,
        active_degree,
        collect_stats,
        max_steps,
        optimizer,
        optimization,
    )
    model.update_from_arrays(pool)
    return (
        state,
        stats,
        metrics["loss"],
        overflow | metrics["overflow"],
        jnp.maximum(peak_pairs, metrics["pairs"]),
    )


train_step = nnx.jit(
    _update_model, graph=False, static_argnames=_STATIC_ARGUMENTS, donate_argnums=(0, 1, 2)
)


def _update_training_state(
    training: TrainingState,
    bounds: WorldClusterBounds,
    camera: Camera,
    target: chex.Array,
    step: int | chex.Array,
    scene_radius: float | chex.Array,
    *,
    active_degree: int,
    collect_stats: bool,
    overflow: chex.Array,
    peak_pairs: chex.Array,
    config: CapacityConfig,
    max_steps: int | None,
    optimizer: str,
    optimization: OptimizationConfig,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    state, stats, loss, overflow, peak_pairs = _update_model(
        training.adam.get_value(),
        training.fragments.get_value(),
        training.model,
        bounds,
        camera,
        target,
        step,
        scene_radius,
        config,
        active_degree,
        collect_stats,
        max_steps,
        overflow,
        peak_pairs,
        optimizer,
        optimization,
    )
    training.adam.set_value(state)
    training.fragments.set_value(stats)
    return loss, overflow, peak_pairs


def bind_train_step(
    training: TrainingState,
    config: CapacityConfig,
    *,
    max_steps: int | None = None,
    optimizer: str = "optax",
    optimization: OptimizationConfig = _DEFAULT_OPTIMIZATION,
) -> Callable[..., tuple[chex.Array, chex.Array, chex.Array]]:
    """Bind fixed buffers and configuration; SH degree and statistics vary per call."""
    update = partial(
        _update_training_state,
        config=config,
        max_steps=max_steps,
        optimizer=optimizer,
        optimization=optimization,
    )
    return nnx.jit_partial(
        update,
        training,
        graph=False,
        static_argnames=("active_degree", "collect_stats"),
        donate_argnums=(0,),
    )
