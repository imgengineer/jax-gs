"""Compiled array and NNX training steps, independent of host orchestration."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from ..config import load_config
from ..scene.cluster import frustum_cluster_mask
from ..scene.point import GaussianModel
from .optimizer import optax_adam_update, sparse_adam_update

_DEFAULT_OPTIMIZATION = load_config().optimization


@partial(
    jax.jit,
    static_argnames=(
        "config",
        "active_degree",
        "collect_stats",
        "max_steps",
        "optimizer",
        "optimization",
    ),
)
def array_train_step(
    pool,
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
    max_steps=None,
    optimizer="optax",
    optimization=_DEFAULT_OPTIMIZATION,
):
    from ..kernels.cluster_compact import compact_visible_clusters
    from ..kernels.packed_rasterizer import packed_loss_and_grad
    from ..kernels.projector import project_cute_sparse
    from ..kernels.sorted_binning import build_sorted_visibility_table_cute
    from ..kernels.sorted_rasterizer import rasterize_loss_and_grad

    visible = frustum_cluster_mask(bounds, camera, config.cluster_size, config.max_gaussians)

    clusters = compact_visible_clusters(visible, config.cluster_size)

    projected, pullback = project_cute_sparse(
        pool.replace(alive=pool.alive & visible), camera, config, active_degree, clusters
    )
    table = build_sorted_visibility_table_cute(jax.lax.stop_gradient(projected), camera, config)
    # LiteGS's half2 kernel needs at least 64 pixels per tile. Small diagnostic
    # scenes retain the float32 path; production uses LiteGS's packed path.
    render_loss = packed_loss_and_grad if config.tile_size in (8, 16) else rasterize_loss_and_grad
    loss, cotangents, fragments = render_loss(
        projected, table, camera, config, target.astype(jnp.float32) / 255, collect_stats
    )
    gradients = pullback(cotangents)
    if optimizer == "optax":
        pool, state = optax_adam_update(
            pool,
            state,
            gradients,
            visible,
            step,
            scene_radius,
            max_steps,
            cluster_size=config.cluster_size,
            compact_gradients=True,
            optimization=optimization,
        )
    elif optimizer == "cute":
        pool, state = sparse_adam_update(
            pool,
            state,
            gradients,
            visible,
            step,
            scene_radius,
            max_steps,
            compacted_clusters=clusters,
            cluster_size=config.cluster_size,
            compact_gradients=True,
            optimization=optimization,
        )
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")
    if collect_stats:
        stats = stats + fragments
    return pool, state, stats, {"loss": loss, "overflow": table.overflow, "pairs": table.pair_count}


@partial(
    nnx.jit,
    graph=False,
    static_argnames=(
        "config",
        "active_degree",
        "collect_stats",
        "max_steps",
        "optimizer",
        "optimization",
    ),
    donate_argnums=(0, 1, 2),
)
def train_step(
    state,
    stats,
    model: GaussianModel,
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
    optimizer="optax",
    optimization=_DEFAULT_OPTIMIZATION,
):
    # The fixed model is a tree with no shared Variables. NNX tree mode avoids
    # graph protocol, while donation keeps the existing in-place GPU updates.
    # NNX appends mutated Variables after explicit outputs. Keep state/stats
    # before the model so XLA pairs each donated buffer with its own output.
    pool, state, stats, metrics = array_train_step.__wrapped__(
        model.as_pool(),
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
    model.update_from_pool(pool)
    return (
        state,
        stats,
        metrics["loss"],
        overflow | metrics["overflow"],
        jnp.maximum(peak_pairs, metrics["pairs"]),
    )
