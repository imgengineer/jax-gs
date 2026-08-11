from dataclasses import replace
import json
import os
import subprocess
import sys
import tempfile
import warnings
from types import SimpleNamespace
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_gs.training as training_module
from jax_gs.capacity import (
    reshard_distributed_training_state,
    resize_distributed_training_state,
)
from jax_gs.checkpoints import (
    load_checkpoint_intersection_capacity,
    load_checkpoint_scene_transform,
    load_distributed_checkpoint_manifest,
    restore_checkpoint,
    restore_distributed_checkpoint,
    save_checkpoint,
    save_distributed_checkpoint,
)
from jax_gs.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    RasterizationConfig,
    StrategyConfig,
    TrainConfig,
)
from jax_gs.model import GaussianModel, inverse_sigmoid
from jax_gs.optimizers import (
    create_optimizer,
    create_visible_adam_optimizer,
)
from jax_gs.strategy import DefaultStrategy, MCMCStrategy
from jax_gs.training import (
    TrainingSafetyState,
    make_distributed_render_step,
    make_distributed_resize_step,
    make_distributed_train_step,
    make_render_step,
    reduce_distributed_render,
    shard_camera_batch,
    synchronize_distributed_capacity,
)
from jax_gs.training._distributed_loop import (
    _initialize_distributed_training_state,
)
from jax_gs.training.pose import CameraOptModule


def _fixed_topology_config(**overrides) -> TrainConfig:
    values = dict(
        model=ModelConfig(
            capacity=1,
            bucket_min_capacity=1,
            sh_degree=0,
            initial_scale=0.2,
        ),
        optimizer=OptimizerConfig(max_steps=2),
        strategy=StrategyConfig(
            refine_start=3,
            refine_stop=4,
            reset_every=4,
            max_new_per_refine=1,
        ),
        data=DataConfig(root="unused", patch_size=4, batch_size=1),
        rasterizer=RasterizationConfig(
            backend="reference",
            tile_size=4,
            max_gaussians_per_tile=4,
            max_intersections=32,
        ),
        ssim_lambda=0.0,
        steps=2,
        eval_every=0,
        checkpoint_every=0,
    )
    values.update(overrides)
    return TrainConfig(**values)


def _topology_plan_config(
    *,
    capacity: int = 2,
    bucket: int | None = None,
    train: dict | None = None,
    **strategy_overrides,
) -> TrainConfig:
    """Config whose refinement schedule fires on the first training step."""

    strategy = dict(
        refine_start=0,
        refine_stop=4,
        refine_every=1,
        reset_every=4,
        max_new_per_refine=2,
        grow_grad2d=0.5,
        grow_scale3d=1.0,
    )
    strategy.update(strategy_overrides)
    return _fixed_topology_config(
        model=ModelConfig(
            capacity=capacity,
            bucket_min_capacity=capacity if bucket is None else bucket,
            sh_degree=0,
            initial_scale=0.2,
        ),
        strategy=StrategyConfig(**strategy),
        **(train or {}),
    )


def _set_owner_statistics(strategy_state, grad_accum, max_radii):
    """Assign per-rank owner-local densification statistics."""

    grad_accum = np.asarray(grad_accum, np.float32)
    strategy_state.grad_accum[...] = jnp.asarray(grad_accum)
    strategy_state.visible_count[...] = jnp.asarray(
        (grad_accum > 0.0).astype(np.float32)
    )
    strategy_state.max_radii[...] = jnp.asarray(max_radii, jnp.float32)


def _stack_graphs(*graphs):
    graphdef, first_state = nnx.split(graphs[0])
    states = [first_state]
    for graph in graphs[1:]:
        _, state = nnx.split(graph)
        states.append(state)
    stacked_state = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    return nnx.merge(graphdef, stacked_state)


def _rank_bundle(
    config: TrainConfig,
    x: float,
    *,
    optimizer_batch_size: int | None = None,
    optimizer_world_size: int = 2,
    optimizer_scene_scale: float = 1.0,
):
    model = GaussianModel.from_point_cloud(
        np.asarray([[x, 0.0, 3.0]], np.float32),
        np.asarray([[192, 128, 64]], np.uint8),
        config.model,
        appearance_feature_dim=(
            training_module.APPEARANCE_FEATURE_DIM
            if config.app_opt
            else None
        ),
        feature_key=(
            jax.random.key(17) if config.app_opt else None
        ),
    )
    optimizer_factory = (
        create_visible_adam_optimizer
        if config.visible_adam
        else create_optimizer
    )
    optimizer = optimizer_factory(
        model,
        config.optimizer,
        batch_size=(
            config.data.batch_size
            if optimizer_batch_size is None
            else optimizer_batch_size
        ),
        world_size=optimizer_world_size,
        scene_scale=optimizer_scene_scale,
    )
    strategy_state = DefaultStrategy(config.strategy).initialize_state(
        model.capacity
    )
    return model, optimizer, strategy_state, TrainingSafetyState()


def _single_process_model_from_shards(config, stacked_model):
    """Rebuild the whole scene as one unsharded model.

    Every shard holds capacity rows of which the active ones are real, so the
    equivalent single-process scene is the concatenation of the active rows.
    """

    world, capacity = stacked_model.active_mask[...].shape
    active = np.asarray(stacked_model.active_mask[...])
    means = np.asarray(stacked_model.means[...])
    rows = np.concatenate(
        [means[rank][active[rank]] for rank in range(world)], axis=0
    )
    colors = np.tile(np.asarray([[192, 128, 64]], np.uint8), (rows.shape[0], 1))
    model_config = replace(config.model, capacity=world * capacity)
    model = GaussianModel.from_point_cloud(rows, colors, model_config)
    # Copy the sharded values verbatim so the comparison isolates the render.
    count = rows.shape[0]
    for name in ("means", "quats", "log_scales", "opacity_logits", "sh0"):
        source = np.concatenate(
            [
                np.asarray(getattr(stacked_model, name)[...])[rank][active[rank]]
                for rank in range(world)
            ],
            axis=0,
        )
        target = np.asarray(getattr(model, name)[...]).copy()
        target[:count] = source
        getattr(model, name)[...] = jnp.asarray(target)
    return model


def _zero_initialized_pose_module():
    module = CameraOptModule(2, rngs=nnx.Rngs(7))
    module.zero_init()
    return module


def _replicated_pose_training_state(
    config: TrainConfig,
    *,
    camera_count: int = 3,
    world_size: int = 2,
    steps: int = 1,
):
    modules = []
    optimizers = []
    for _ in range(world_size):
        module = CameraOptModule(camera_count, rngs=nnx.Rngs(7))
        module.zero_init()
        optimizer = training_module._create_pose_optimizer(module, config)
        gradients = jax.tree.map(
            jnp.ones_like, nnx.state(module, nnx.Param)
        )
        for _ in range(steps):
            optimizer.update(module, gradients)
        modules.append(module)
        optimizers.append(optimizer)
    return _stack_graphs(*modules), _stack_graphs(*optimizers)


def _replicated_appearance_training_state(
    config: TrainConfig,
    *,
    camera_count: int = 3,
    world_size: int = 2,
    steps: int = 0,
):
    modules = []
    optimizers = []
    for _ in range(world_size):
        module = training_module.AppearanceOptModule(
            camera_count,
            training_module.APPEARANCE_FEATURE_DIM,
            config.app_embed_dim,
            config.model.sh_degree,
            rngs=nnx.Rngs(19),
        )
        optimizer = training_module.create_appearance_optimizer(
            module, config
        )
        gradients = jax.tree.map(
            jnp.ones_like, nnx.state(module, nnx.Param)
        )
        for _ in range(steps):
            optimizer.update(module, gradients)
        modules.append(module)
        optimizers.append(optimizer)
    return _stack_graphs(*modules), _stack_graphs(*optimizers)


def _unstack_graph(graph, index):
    graphdef, state = nnx.split(graph)
    return nnx.merge(graphdef, jax.tree.map(lambda x: x[index], state))


def _adam_means_moments(optimizer):
    moments = []
    leaves, _ = jax.tree_util.tree_flatten_with_path(
        nnx.as_pure(nnx.state(optimizer.opt_state))
    )
    for path, value in leaves:
        keys = tuple(getattr(entry, "key", None) for entry in path)
        if keys[-2:] == ("mu", "means"):
            moments.append(np.asarray(value))
    assert len(moments) == 1
    return moments[0]


def _snapshot_graph_arrays(graph):
    return tuple(
        np.asarray(
            jax.random.key_data(leaf)
            if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
            else leaf
        ).copy()
        for leaf in jax.tree.leaves(nnx.as_pure(nnx.state(graph)))
        if isinstance(leaf, jax.Array)
    )


def _rank_zero_overflow_rasterization(
    means,
    _quats,
    _scales,
    _opacities,
    _colors,
    viewmats,
    _intrinsics,
    width,
    height,
    **kwargs,
):
    assert kwargs["distributed"]
    assert kwargs["distributed_world_size"] > 1
    axis_name = kwargs["distributed_axis_name"]
    gathered_means = jax.lax.all_gather(
        means, axis_name, axis=0, tiled=True
    )
    camera_count = viewmats.shape[0]
    global_capacity = gathered_means.shape[0]
    signal = jnp.asarray(0.25, means.dtype) + 1.0e-3 * jnp.sum(
        gathered_means
    )
    renders = jnp.broadcast_to(
        signal, (camera_count, height, width, 3)
    )
    alphas = jnp.ones(
        (camera_count, height, width, 1), dtype=means.dtype
    )
    rank_zero_overflow = jax.lax.axis_index(axis_name) == 0
    info = {
        "radii": jnp.ones(
            (camera_count, global_capacity, 2), dtype=means.dtype
        ),
        "valid": jnp.ones(
            (camera_count, global_capacity), dtype=jnp.bool_
        ),
        "tile_overflow": jnp.zeros(
            (camera_count, 1, 1), dtype=jnp.bool_
        ),
        "candidate_limit_exceeded": jnp.zeros(
            (camera_count, 1, 1), dtype=jnp.bool_
        ),
        "candidate_counts": jnp.zeros((camera_count, 1, 1), jnp.int32),
        "intersection_overflow": jnp.broadcast_to(
            rank_zero_overflow, (camera_count,)
        ),
        "intersection_count": jnp.ones(
            (camera_count,), dtype=jnp.int32
        ),
        "intersection_required_count": jnp.full(
            (camera_count,),
            jnp.where(rank_zero_overflow, 7, 1),
            dtype=jnp.int32,
        ),
    }
    return renders, alphas, info


def _no_overflow_rasterization(*args, **kwargs):
    renders, alphas, info = _rank_zero_overflow_rasterization(
        *args, **kwargs
    )
    camera_count = info["intersection_overflow"].shape[0]
    info = {
        **info,
        "intersection_overflow": jnp.zeros(
            (camera_count,), dtype=jnp.bool_
        ),
        "intersection_required_count": jnp.ones(
            (camera_count,), dtype=jnp.int32
        ),
    }
    return renders, alphas, info


def _opacity_growth_rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    intrinsics,
    width,
    height,
    **kwargs,
):
    renders, alphas, info = _no_overflow_rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        intrinsics,
        width,
        height,
        **kwargs,
    )
    gathered_opacities = jax.lax.all_gather(
        opacities,
        kwargs["distributed_axis_name"],
        axis=0,
        tiled=True,
    )
    return renders - 1.0e-3 * jnp.sum(gathered_opacities), alphas, info


def _no_statistics_rasterization(*args, **kwargs):
    """Render without a single visible (camera, Gaussian) pair.

    Densification statistics then stay exactly as a test prepared them, so the
    post-update commit decides on the same state as the pre-update plan.
    """

    renders, alphas, info = _no_overflow_rasterization(*args, **kwargs)
    return renders, alphas, {
        **info,
        "valid": jnp.zeros_like(info["valid"]),
    }


def _host_loop_rasterization(
    means,
    _quats,
    _scales,
    _opacities,
    _colors,
    viewmats,
    _intrinsics,
    width,
    height,
    **kwargs,
):
    """Small differentiable distributed render for host-loop integration."""

    axis_name = kwargs["distributed_axis_name"]
    gathered_means = jax.lax.all_gather(
        means, axis_name, axis=0, tiled=True
    )
    gathered_active = jax.lax.all_gather(
        kwargs["active_mask"], axis_name, axis=0, tiled=True
    )
    camera_count = viewmats.shape[0]
    global_capacity = gathered_means.shape[0]
    signal = jnp.asarray(0.25, means.dtype) + 1.0e-3 * jnp.sum(
        gathered_means
    )
    screen_probe = kwargs.get("_means2d_offset")
    if screen_probe is not None:
        assert screen_probe.shape == (camera_count, global_capacity, 2)
        signal = signal + 1.0e-2 * jnp.sum(screen_probe)
    renders = jnp.broadcast_to(
        signal, (camera_count, height, width, 3)
    )
    alphas = jnp.ones(
        (camera_count, height, width, 1), dtype=means.dtype
    )
    visible = jnp.broadcast_to(
        gathered_active[None, :], (camera_count, global_capacity)
    )
    active_count = jnp.count_nonzero(gathered_active).astype(jnp.int32)
    info = {
        "radii": jnp.ones(
            (camera_count, global_capacity, 2), dtype=means.dtype
        ),
        "valid": visible,
        "tile_overflow": jnp.zeros(
            (camera_count, 1, 1), dtype=jnp.bool_
        ),
        "candidate_limit_exceeded": jnp.zeros(
            (camera_count, 1, 1), dtype=jnp.bool_
        ),
        "candidate_counts": jnp.full(
            (camera_count, 1, 1), active_count, dtype=jnp.int32
        ),
        "intersection_overflow": jnp.zeros(
            (camera_count,), dtype=jnp.bool_
        ),
        "intersection_count": jnp.full(
            (camera_count,), active_count, dtype=jnp.int32
        ),
        "intersection_required_count": jnp.full(
            (camera_count,), active_count, dtype=jnp.int32
        ),
    }
    return renders, alphas, info


def _host_loop_overflow_rasterization(*args, **kwargs):
    """Require one intersection growth, then one candidate-bound growth."""

    renders, alphas, info = _host_loop_rasterization(*args, **kwargs)
    config = kwargs["config"]
    camera_count = info["intersection_overflow"].shape[0]
    active_count = info["intersection_required_count"][0]
    intersection_overflow = config.max_intersections < 16
    candidate_overflow = config.max_candidates_per_tile < active_count
    return renders, alphas, {
        **info,
        "tile_overflow": jnp.full(
            (camera_count, 1, 1), candidate_overflow, jnp.bool_
        ),
        "candidate_limit_exceeded": jnp.full(
            (camera_count, 1, 1), candidate_overflow, jnp.bool_
        ),
        "intersection_overflow": jnp.full(
            (camera_count,), intersection_overflow, jnp.bool_
        ),
        "intersection_required_count": jnp.full(
            (camera_count,), 16, jnp.int32
        ),
    }


def _cross_rank_visibility_rasterization(*args, **kwargs):
    renders, alphas, info = _no_overflow_rasterization(*args, **kwargs)
    axis_name = kwargs["distributed_axis_name"]
    rank = jax.lax.axis_index(axis_name)
    valid = jnp.where(
        rank == 0,
        jnp.asarray([[False, True]]),
        jnp.asarray([[True, False]]),
    )
    return renders, alphas, {**info, "valid": valid}


def _screen_stats_rasterization(
    means,
    _quats,
    _scales,
    _opacities,
    _colors,
    viewmats,
    _intrinsics,
    width,
    height,
    **kwargs,
):
    axis_name = kwargs["distributed_axis_name"]
    gathered_means = jax.lax.all_gather(
        means, axis_name, axis=0, tiled=True
    )
    global_capacity = gathered_means.shape[0]
    screen_probe = kwargs["_means2d_offset"]
    assert screen_probe.shape == (viewmats.shape[0], global_capacity, 2)
    rank = jax.lax.axis_index(axis_name)
    screen_coefficients = jnp.where(
        rank == 0,
        jnp.asarray([[[1.0, 0.0], [100.0, 100.0]]]),
        jnp.asarray([[[-3.0, 0.0], [6.0, 8.0]]]),
    ).astype(means.dtype)
    signal = (
        jnp.asarray(0.25, means.dtype)
        + 1.0e-3 * jnp.sum(gathered_means)
        + jnp.sum(screen_probe * screen_coefficients)
    )
    renders = jnp.broadcast_to(signal, (viewmats.shape[0], height, width, 3))
    alphas = jnp.ones(
        (viewmats.shape[0], height, width, 1), dtype=means.dtype
    )
    radii = jnp.where(
        rank == 0,
        jnp.asarray([[[4.0, 2.0], [7.0, 7.0]]]),
        jnp.asarray([[[2.0, 2.0], [1.0, 2.0]]]),
    ).astype(means.dtype)
    valid = jnp.where(
        rank == 0,
        jnp.asarray([[True, False]]),
        jnp.asarray([[True, True]]),
    )
    info = {
        "radii": radii,
        "valid": valid,
        "tile_overflow": jnp.zeros(
            (viewmats.shape[0], 1, 1), dtype=jnp.bool_
        ),
        "candidate_limit_exceeded": jnp.zeros(
            (viewmats.shape[0], 1, 1), dtype=jnp.bool_
        ),
        "candidate_counts": jnp.zeros((viewmats.shape[0], 1, 1), jnp.int32),
        "intersection_overflow": jnp.zeros(
            (viewmats.shape[0],), dtype=jnp.bool_
        ),
        "intersection_count": jnp.ones(
            (viewmats.shape[0],), dtype=jnp.int32
        ),
        "intersection_required_count": jnp.ones(
            (viewmats.shape[0],), dtype=jnp.int32
        ),
    }
    return renders, alphas, info


def test_distributed_train_step_rejects_unsupported_first_slice_modes():
    with pytest.raises(ValueError, match="world_size must be greater than one"):
        make_distributed_train_step(
            _fixed_topology_config(), world_size=1
        )
    with pytest.raises(NotImplementedError, match="2DGS"):
        make_distributed_train_step(
            _fixed_topology_config(model_type="2dgs"), world_size=2
        )
    with pytest.raises(NotImplementedError, match="AbsGrad"):
        make_distributed_train_step(
            _fixed_topology_config(
                strategy=StrategyConfig(
                    refine_start=3,
                    refine_stop=4,
                    reset_every=4,
                    max_new_per_refine=1,
                    absgrad=True,
                )
            ),
            world_size=2,
        )
    with pytest.raises(ValueError, match="optimizer.max_steps.*steps"):
        make_distributed_train_step(
            _fixed_topology_config(
                optimizer=OptimizerConfig(max_steps=3)
            ),
            world_size=2,
        )


def _two_rank_bundles(
    config: TrainConfig,
    *,
    optimizer_world_size: int = 2,
    optimizer_scene_scale: float = 1.0,
):
    return _stack_graphs(
        _rank_bundle(
            config,
            -0.08,
            optimizer_world_size=optimizer_world_size,
            optimizer_scene_scale=optimizer_scene_scale,
        ),
        _rank_bundle(
            config,
            0.08,
            optimizer_world_size=optimizer_world_size,
            optimizer_scene_scale=optimizer_scene_scale,
        ),
    )


def _run_two_rank_update(
    map_transform,
    *,
    config: TrainConfig | None = None,
    bundles=None,
    prepare=None,
    optimizer_world_size: int = 2,
    optimizer_scene_scale: float = 1.0,
    train_scene_scale: float = 1.0,
    initial_optimizer_steps=(0, 0),
    sh_degrees=(0, 0),
    expect_update: bool = True,
):
    config = _fixed_topology_config() if config is None else config
    if bundles is None:
        bundles = _two_rank_bundles(
            config,
            optimizer_world_size=optimizer_world_size,
            optimizer_scene_scale=optimizer_scene_scale,
        )
    model, optimizer, strategy_state, safety_state = bundles
    optimizer.step[...] = jnp.asarray(
        initial_optimizer_steps, dtype=optimizer.step[...].dtype
    )
    if prepare is not None:
        prepare(model, optimizer, strategy_state)
    train_step = make_distributed_train_step(
        config,
        world_size=2,
        axis_name="rank",
        scene_scale=train_scene_scale,
    )

    @map_transform(
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
        axis_name="rank",
    )
    def mapped_step(
        current_model,
        current_optimizer,
        current_strategy_state,
        current_safety_state,
        images,
        intrinsics,
        viewmats,
        key,
        sh_degree,
    ):
        return train_step(
            current_model,
            current_optimizer,
            current_strategy_state,
            current_safety_state,
            images,
            intrinsics,
            viewmats,
            key,
            sh_degree,
        )

    images = jnp.zeros((2, 1, 4, 4, 3), jnp.float32)
    intrinsics = jnp.broadcast_to(
        jnp.asarray(
            [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
            jnp.float32,
        )[None, None],
        (2, 1, 3, 3),
    )
    viewmats = jnp.broadcast_to(
        jnp.eye(4, dtype=jnp.float32)[None, None], (2, 1, 4, 4)
    ).at[1, 0, 0, 3].set(-0.1)
    means_before = np.asarray(model.means[...]).copy()

    metrics = mapped_step(
        model,
        optimizer,
        strategy_state,
        safety_state,
        images,
        intrinsics,
        viewmats,
        jax.random.split(jax.random.key(0), 2),
        jnp.asarray(sh_degrees, jnp.int32),
    )

    assert np.all(np.isfinite(metrics["loss"]))
    assert np.all(np.asarray(metrics["intersection_overflow"]) == 0)
    expected_steps = np.asarray(initial_optimizer_steps) + int(expect_update)
    np.testing.assert_array_equal(optimizer.step[...], expected_steps)
    assert (not np.array_equal(model.means[...], means_before)) == expect_update
    return model, optimizer, strategy_state, safety_state, metrics


def test_named_two_rank_train_step_updates_shards_with_global_rendering():
    _, _, strategy_state, _, _ = _run_two_rank_update(nnx.vmap)

    assert np.all(np.isfinite(strategy_state.grad_accum[...]))
    assert np.any(np.asarray(strategy_state.grad_accum[...]) > 0.0)
    assert np.all(np.asarray(strategy_state.visible_count[...]) > 0.0)


def _run_two_rank_screen_stats(map_transform):
    with mock.patch.object(
        training_module, "rasterization", _screen_stats_rasterization
    ):
        _, _, strategy_state, _, _ = _run_two_rank_update(map_transform)

    np.testing.assert_allclose(
        strategy_state.grad_accum[...],
        [[8.0], [20.0]],
        rtol=2.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        strategy_state.visible_count[...], [[2.0], [1.0]]
    )
    np.testing.assert_allclose(strategy_state.max_radii[...], [[1.0], [0.5]])


def test_named_two_rank_screen_statistics_reduce_before_owner_slice():
    _run_two_rank_screen_stats(nnx.vmap)


def test_distributed_train_step_plans_a_refinement_schedule():
    make_distributed_train_step(_topology_plan_config(), world_size=2)


def _duplicate_only_run(map_transform, *, config=None):
    config = _topology_plan_config() if config is None else config

    def prepare(model, optimizer, strategy_state):
        del model, optimizer
        _set_owner_statistics(
            strategy_state, [[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]
        )

    with mock.patch.object(
        training_module, "rasterization", _no_statistics_rasterization
    ):
        return _run_two_rank_update(
            map_transform, config=config, prepare=prepare
        )


def test_owner_local_duplicate_is_planned_and_committed():
    model, optimizer, strategy_state, _, metrics = _duplicate_only_run(
        nnx.vmap
    )

    np.testing.assert_array_equal(metrics["refine_scheduled"], [True, True])
    np.testing.assert_array_equal(metrics["reset_scheduled"], [False, False])
    np.testing.assert_array_equal(metrics["refine_planned_new_count"], [1, 1])
    np.testing.assert_array_equal(
        metrics["refine_planned_pruned_count"], [0, 0]
    )
    np.testing.assert_array_equal(metrics["refine_required_capacity"], [2, 2])
    np.testing.assert_array_equal(
        metrics["refine_capacity_overflow"], [False, False]
    )
    np.testing.assert_array_equal(metrics["refine_new_count"], [1, 1])
    np.testing.assert_array_equal(metrics["refine_pruned_count"], [0, 0])
    np.testing.assert_array_equal(
        metrics["refine_commit_overflow"], [False, False]
    )
    np.testing.assert_array_equal(metrics["opacity_reset"], [False, False])
    # Only the rank-0 owner grows, and its child is an exact copy.
    np.testing.assert_array_equal(
        model.active_mask[...], [[True, True], [True, False]]
    )
    np.testing.assert_array_equal(model.means[0, 1], model.means[0, 0])
    np.testing.assert_array_equal(model.sh0[0, 1], model.sh0[0, 0])
    np.testing.assert_array_equal(
        model.opacity_logits[0, 1], model.opacity_logits[0, 0]
    )
    # A committed refinement clears the statistics of every owner.
    np.testing.assert_array_equal(strategy_state.grad_accum[...], 0.0)
    np.testing.assert_array_equal(strategy_state.visible_count[...], 0.0)
    np.testing.assert_array_equal(strategy_state.max_radii[...], 0.0)
    means_moments = _adam_means_moments(optimizer)
    np.testing.assert_array_equal(means_moments[0, 1], 0.0)


def _run_two_rank_growth_commit(map_transform):
    """Commit a duplicate and a split for the same rank-0 parent."""

    config = _topology_plan_config(
        capacity=4, refine_scale2d_stop_iter=100, grow_scale2d=0.05
    )

    def prepare(model, optimizer, strategy_state):
        del model, optimizer
        _set_owner_statistics(
            strategy_state,
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        )

    with mock.patch.object(
        training_module, "rasterization", _no_statistics_rasterization
    ):
        model, optimizer, _, _, metrics = _run_two_rank_update(
            map_transform, config=config, prepare=prepare
        )

    np.testing.assert_array_equal(metrics["refine_scheduled"], [True, True])
    np.testing.assert_array_equal(metrics["refine_planned_new_count"], [2, 2])
    np.testing.assert_array_equal(
        metrics["refine_planned_pruned_count"], [0, 0]
    )
    np.testing.assert_array_equal(metrics["refine_required_capacity"], [3, 3])
    np.testing.assert_array_equal(
        metrics["refine_capacity_overflow"], [False, False]
    )
    np.testing.assert_array_equal(metrics["refine_new_count"], [2, 2])
    np.testing.assert_array_equal(
        metrics["refine_commit_overflow"], [False, False]
    )
    np.testing.assert_array_equal(
        np.asarray(model.active_mask[...]).sum(axis=1), [3, 1]
    )
    # The split shrinks the parent and its own child by 1.6, while the
    # duplicate child keeps the parent's original scale.
    log_scales = np.asarray(model.log_scales[...])
    np.testing.assert_allclose(
        log_scales[0, 0], log_scales[0, 2], rtol=1e-6
    )
    np.testing.assert_allclose(
        log_scales[0, 1] - log_scales[0, 0],
        np.full(3, np.log(1.6), np.float32),
        rtol=1e-6,
    )
    # A refined owner restarts its Adam moments; the untouched owner keeps its.
    means_moments = _adam_means_moments(optimizer)
    np.testing.assert_array_equal(means_moments[0, 0], 0.0)
    assert np.any(means_moments[1, 0] != 0.0)


def test_named_two_rank_commits_duplicate_and_split_for_one_parent():
    _run_two_rank_growth_commit(nnx.vmap)


def test_commit_recomputes_from_post_update_statistics():
    # The plan runs before the update, so the extra visible count this step
    # adds drops the average gradient to the threshold and cancels the event.
    config = _topology_plan_config()

    def prepare(model, optimizer, strategy_state):
        del model, optimizer
        _set_owner_statistics(
            strategy_state, [[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]
        )

    with mock.patch.object(
        training_module, "rasterization", _no_overflow_rasterization
    ):
        model, _, _, _, metrics = _run_two_rank_update(
            nnx.vmap, config=config, prepare=prepare
        )

    np.testing.assert_array_equal(metrics["refine_planned_new_count"], [1, 1])
    np.testing.assert_array_equal(metrics["refine_new_count"], [0, 0])
    np.testing.assert_array_equal(
        model.active_mask[...], [[True, False], [True, False]]
    )


def test_owner_commit_matches_single_process_refine():
    committed_model, _, _, _, _ = _duplicate_only_run(nnx.vmap)
    # The same inputs with the schedule disabled leave the post-update state
    # the commit decided on.
    unrefined = _duplicate_only_run(
        nnx.vmap, config=_topology_plan_config(refine_start=4)
    )
    expected_model = _unstack_graph(unrefined[0], 0)
    expected_optimizer = _unstack_graph(unrefined[1], 0)
    expected_state = _unstack_graph(unrefined[2], 0)

    # Duplicate-only refinement consumes no randomness, so any key reproduces
    # the owner-local commit.
    DefaultStrategy(_topology_plan_config().strategy).refine(
        expected_model,
        expected_state,
        expected_optimizer,
        jax.random.key(0),
        1.0,
        step=1,
    )

    np.testing.assert_array_equal(
        committed_model.active_mask[0], expected_model.active_mask[...]
    )
    np.testing.assert_array_equal(
        committed_model.means[0], expected_model.means[...]
    )
    np.testing.assert_array_equal(
        committed_model.log_scales[0], expected_model.log_scales[...]
    )
    np.testing.assert_array_equal(
        committed_model.opacity_logits[0],
        expected_model.opacity_logits[...],
    )


def test_plan_uses_the_train_step_scene_scale():
    # StrategyState keeps its default scene scale of 1.0, which would make the
    # 0.2-scaled parent small enough to duplicate as well as split.
    config = _topology_plan_config(
        capacity=4, refine_scale2d_stop_iter=100, grow_scale2d=0.05
    )

    def prepare(model, optimizer, strategy_state):
        del model, optimizer
        _set_owner_statistics(
            strategy_state,
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        )

    with mock.patch.object(
        training_module, "rasterization", _no_overflow_rasterization
    ):
        _, _, strategy_state, _, metrics = _run_two_rank_update(
            nnx.vmap,
            config=config,
            prepare=prepare,
            optimizer_scene_scale=0.1,
            train_scene_scale=0.1,
        )

    np.testing.assert_array_equal(strategy_state.scene_scale[...], [1.0, 1.0])
    np.testing.assert_array_equal(metrics["refine_planned_new_count"], [1, 1])


def test_inactive_padding_rows_are_never_planned():
    config = _topology_plan_config(
        capacity=4, refine_scale2d_stop_iter=100, grow_scale2d=0.05
    )

    def prepare(model, optimizer, strategy_state):
        del model, optimizer
        _set_owner_statistics(
            strategy_state,
            [[0.0, 5.0, 5.0, 5.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 9.0, 9.0, 9.0], [0.0, 0.0, 0.0, 0.0]],
        )

    with mock.patch.object(
        training_module, "rasterization", _no_overflow_rasterization
    ):
        _, _, _, _, metrics = _run_two_rank_update(
            nnx.vmap, config=config, prepare=prepare
        )

    np.testing.assert_array_equal(metrics["refine_scheduled"], [True, True])
    np.testing.assert_array_equal(metrics["refine_planned_new_count"], [0, 0])
    np.testing.assert_array_equal(
        metrics["refine_planned_pruned_count"], [0, 0]
    )
    np.testing.assert_array_equal(metrics["refine_required_capacity"], [1, 1])


def test_owner_local_prune_is_planned_and_committed():
    config = _topology_plan_config(prune_opacity=0.5)

    with mock.patch.object(
        training_module, "rasterization", _no_statistics_rasterization
    ):
        model, _, _, _, metrics = _run_two_rank_update(nnx.vmap, config=config)

    np.testing.assert_array_equal(metrics["refine_planned_new_count"], [0, 0])
    np.testing.assert_array_equal(
        metrics["refine_planned_pruned_count"], [2, 2]
    )
    np.testing.assert_array_equal(
        metrics["refine_capacity_overflow"], [False, False]
    )
    np.testing.assert_array_equal(metrics["refine_pruned_count"], [2, 2])
    np.testing.assert_array_equal(metrics["refine_new_count"], [0, 0])
    np.testing.assert_array_equal(model.active_mask[...], False)


def test_scheduled_opacity_reset_is_committed():
    config = _topology_plan_config(reset_every=1)

    with mock.patch.object(
        training_module, "rasterization", _no_statistics_rasterization
    ):
        model, _, _, _, metrics = _run_two_rank_update(
            nnx.vmap, config=config
        )

    np.testing.assert_array_equal(metrics["reset_scheduled"], [True, True])
    np.testing.assert_array_equal(metrics["opacity_reset"], [True, True])
    np.testing.assert_allclose(
        model.opacity_logits[:, 0],
        np.full(2, inverse_sigmoid(config.strategy.reset_opacity), np.float32),
        rtol=1e-6,
    )


def test_scheduled_opacity_reset_stops_at_refine_stop():
    config = _topology_plan_config(reset_every=1, refine_stop=1)
    opacity_before = {}

    def prepare(model, optimizer, strategy_state):
        del optimizer, strategy_state
        opacity_before["value"] = np.asarray(model.opacity_logits[...]).copy()

    with mock.patch.object(
        training_module, "rasterization", _no_statistics_rasterization
    ):
        model, _, _, _, metrics = _run_two_rank_update(
            nnx.vmap, config=config, prepare=prepare
        )

    np.testing.assert_array_equal(metrics["refine_scheduled"], [False, False])
    np.testing.assert_array_equal(metrics["reset_scheduled"], [False, False])
    np.testing.assert_array_equal(metrics["opacity_reset"], [False, False])
    np.testing.assert_array_equal(
        model.opacity_logits[...], opacity_before["value"]
    )


def test_single_rank_plan_overflow_atomically_skips_every_rank():
    config = _topology_plan_config(
        refine_scale2d_stop_iter=100, grow_scale2d=0.05
    )
    before = {}

    def prepare(model, optimizer, strategy_state):
        _set_owner_statistics(
            strategy_state, [[1.0, 0.0], [0.0, 0.0]], [[0.5, 0.0], [0.0, 0.0]]
        )
        before["model"] = _snapshot_graph_arrays(model)
        before["optimizer"] = _snapshot_graph_arrays(optimizer)
        before["strategy_state"] = _snapshot_graph_arrays(strategy_state)

    with mock.patch.object(
        training_module, "rasterization", _no_overflow_rasterization
    ):
        model, optimizer, strategy_state, _, metrics = _run_two_rank_update(
            nnx.vmap, config=config, prepare=prepare, expect_update=False
        )

    np.testing.assert_array_equal(metrics["refine_planned_new_count"], [2, 2])
    np.testing.assert_array_equal(
        metrics["refine_capacity_overflow"], [True, True]
    )
    np.testing.assert_array_equal(metrics["refine_new_count"], [0, 0])
    np.testing.assert_array_equal(metrics["refine_pruned_count"], [0, 0])
    np.testing.assert_array_equal(metrics["opacity_reset"], [False, False])
    for graph, name in (
        (model, "model"),
        (optimizer, "optimizer"),
        (strategy_state, "strategy_state"),
    ):
        for old, new in zip(
            before[name], _snapshot_graph_arrays(graph), strict=True
        ):
            np.testing.assert_array_equal(new, old)


def test_owner_commit_overflow_keeps_that_owner_unchanged():
    # This step's statistics push the rank-0 owner from one planned event to a
    # duplicate plus a radius split, which no longer fits its single free slot.
    config = _topology_plan_config(
        capacity=4,
        bucket=2,
        grow_grad2d=0.1,
        refine_scale2d_stop_iter=100,
        grow_scale2d=0.05,
        reset_every=1,
    )

    def prepare(model, optimizer, strategy_state):
        del model, optimizer
        _set_owner_statistics(
            strategy_state, [[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]
        )

    with mock.patch.object(
        training_module, "rasterization", _no_overflow_rasterization
    ):
        model, optimizer, strategy_state, _, metrics = _run_two_rank_update(
            nnx.vmap, config=config, prepare=prepare
        )

    np.testing.assert_array_equal(metrics["refine_planned_new_count"], [1, 1])
    np.testing.assert_array_equal(
        metrics["refine_capacity_overflow"], [False, False]
    )
    np.testing.assert_array_equal(
        metrics["refine_commit_overflow"], [True, True]
    )
    np.testing.assert_array_equal(
        metrics["refine_commit_required_capacity"], [3, 3]
    )
    np.testing.assert_array_equal(metrics["refine_new_count"], [1, 1])
    # Only the owner that fits grows, resets opacities, and clears statistics.
    np.testing.assert_array_equal(
        model.active_mask[...], [[True, False], [True, True]]
    )
    np.testing.assert_array_equal(
        strategy_state.capacity_overflow[...], [True, False]
    )
    np.testing.assert_array_equal(
        strategy_state.grad_accum[...], [[1.0, 0.0], [0.0, 0.0]]
    )
    # Both ranks' cameras see the retained owner, so its count grew by two.
    np.testing.assert_array_equal(
        strategy_state.visible_count[...], [[3.0, 0.0], [0.0, 0.0]]
    )
    np.testing.assert_allclose(
        model.opacity_logits[:, 0],
        [
            inverse_sigmoid(0.1),
            inverse_sigmoid(config.strategy.reset_opacity),
        ],
        rtol=1e-6,
    )

    grown_model, _, _, decision = synchronize_distributed_capacity(
        config, model, optimizer, strategy_state, metrics
    )
    assert decision == (True, False, 2, 4)
    assert grown_model.means[...].shape == (2, 4, 3)


def test_distributed_train_step_rejects_wrong_optimizer_world_size():
    with pytest.raises(ValueError, match="optimizer.*world_size=2"):
        _run_two_rank_update(nnx.vmap, optimizer_world_size=1)


def test_distributed_train_step_rejects_wrong_optimizer_scene_scale():
    with pytest.raises(ValueError, match="optimizer.*scene_scale=2.0"):
        _run_two_rank_update(
            nnx.vmap,
            optimizer_scene_scale=1.0,
            train_scene_scale=2.0,
        )


@pytest.mark.parametrize(
    ("initial_optimizer_steps", "sh_degrees"),
    [
        ((0, 1), (0, 0)),
        ((0, 0), (0, 1)),
    ],
)
def test_rank_state_mismatch_atomically_skips_update(
    monkeypatch, initial_optimizer_steps, sh_degrees
):
    monkeypatch.setattr(
        training_module,
        "rasterization",
        _no_overflow_rasterization,
    )

    model, optimizer, _, _, metrics = _run_two_rank_update(
        nnx.vmap,
        initial_optimizer_steps=initial_optimizer_steps,
        sh_degrees=sh_degrees,
        expect_update=False,
    )

    del model, optimizer
    np.testing.assert_array_equal(
        metrics["distributed_state_mismatch"], [True, True]
    )


def test_gaussian_adam_moments_use_sum_of_rank_local_losses(monkeypatch):
    monkeypatch.setattr(
        training_module,
        "rasterization",
        _no_overflow_rasterization,
    )

    _, optimizer, _, _, _ = _run_two_rank_update(nnx.vmap)

    moment_leaves = []
    leaves, _ = jax.tree_util.tree_flatten_with_path(
        nnx.as_pure(nnx.state(optimizer.opt_state))
    )
    for path, value in leaves:
        keys = tuple(getattr(entry, "key", None) for entry in path)
        if keys[-2:] == ("mu", "means"):
            moment_leaves.append(np.asarray(value))
    assert len(moment_leaves) == 1
    # Each rank-local mean loss contributes dloss/dmean=1e-3. The gather VJP
    # sums both losses, and current-main BS=2 gives beta1=0.8.
    np.testing.assert_allclose(
        moment_leaves[0],
        (1.0 - 0.8) * 2.0e-3,
        rtol=1e-6,
        atol=1e-8,
    )


def test_visibility_is_reduced_before_selecting_the_owner_shard(monkeypatch):
    monkeypatch.setattr(
        training_module,
        "rasterization",
        _cross_rank_visibility_rasterization,
    )

    _, _, _, _, metrics = _run_two_rank_update(nnx.vmap)

    np.testing.assert_array_equal(metrics["visible_count"], [1, 1])


def test_two_virtual_cpu_nnx_pmap_smoke():
    script = "\n".join(
        (
            "import jax",
            "from flax import nnx",
            "from tests.test_training_distributed import _run_two_rank_update",
            "assert jax.local_device_count() == 2",
            "_run_two_rank_update(nnx.pmap)",
            "from tests.test_training_distributed import "
            "_run_two_rank_screen_stats, _run_two_rank_growth_commit, "
            "_run_two_rank_pose_update",
            "_run_two_rank_screen_stats(nnx.pmap)",
            "_run_two_rank_growth_commit(nnx.pmap)",
            "_run_two_rank_pose_update(nnx.pmap)",
            "from tests.test_training_distributed import "
            "_run_two_rank_pmap_resize_lifecycle",
            "_run_two_rank_pmap_resize_lifecycle()",
        )
    )
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _run_two_rank_pmap_resize_lifecycle():
    """A mapped world remains rank-sharded after changing bucket shape."""

    config = _fixed_topology_config(
        model=ModelConfig(
            capacity=4,
            bucket_min_capacity=2,
            sh_degree=0,
            initial_scale=0.2,
        )
    )
    model, optimizer, strategy_state, safety_state = _two_rank_bundles(config)
    train_step = make_distributed_train_step(config, world_size=2)

    @nnx.pmap(
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
        axis_name="rank",
    )
    def mapped_step(*args):
        return train_step(*args)

    images = jnp.zeros((2, 1, 4, 4, 3), jnp.float32)
    intrinsics = jnp.broadcast_to(
        jnp.asarray(
            [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
            jnp.float32,
        )[None, None],
        (2, 1, 3, 3),
    )
    viewmats = jnp.broadcast_to(
        jnp.eye(4, dtype=jnp.float32)[None, None], (2, 1, 4, 4)
    )
    keys = jax.random.split(jax.random.key(0), 2)
    sh_degrees = jnp.zeros((2,), jnp.int32)

    mapped_step(
        model,
        optimizer,
        strategy_state,
        safety_state,
        images,
        intrinsics,
        viewmats,
        keys,
        sh_degrees,
    )
    resize_step = make_distributed_resize_step(config)
    model, optimizer, strategy_state = resize_step(
        model, optimizer, strategy_state, 4
    )
    assert model.means[...].shape == (2, 4, 3)
    assert "P('rank'" in str(model.means[...].sharding)

    metrics = mapped_step(
        model,
        optimizer,
        strategy_state,
        safety_state,
        images,
        intrinsics,
        viewmats,
        keys,
        sh_degrees,
    )
    np.testing.assert_array_equal(
        metrics["distributed_state_mismatch"], [False, False]
    )
    np.testing.assert_array_equal(optimizer.step[...], [2, 2])


def _run_local_device_distributed_host_loop():
    """Exercise grow/replay, eval, checkpoint, and exact resume together."""

    assert jax.process_count() == 1
    assert jax.local_device_count() == 2
    points = np.asarray(
        [
            [-0.20, 0.00, 3.0],
            [-0.10, 0.05, 3.0],
            [0.00, 0.00, 3.0],
            [0.10, -0.05, 3.0],
            [0.20, 0.00, 3.0],
        ],
        np.float32,
    )
    camtoworlds = np.stack(
        (np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))
    )
    camtoworlds[:, 0, 3] = np.asarray([-1.0, 1.0], np.float32)
    scene = SimpleNamespace(
        points=points,
        points_rgb=np.full((len(points), 3), 128, np.uint8),
        camtoworlds=camtoworlds,
    )
    K = np.asarray(
        [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
        np.float32,
    )
    w2c = np.eye(4, dtype=np.float32)
    train_batches = [
        {
            "image": np.full((2, 4, 4, 3), value, np.float32),
            "K": np.tile(K, (2, 1, 1)),
            "w2c": np.tile(w2c, (2, 1, 1)),
        }
        for value in (0.0, 0.05, 0.10)
    ]
    evaluation_example = {
        "image": np.zeros((4, 4, 3), np.float32),
        "K": K,
        "w2c": w2c,
    }
    dataset_batch_sizes = []

    def create_dataset(_scene, *, split, batch_size, **_kwargs):
        if split == "train":
            dataset_batch_sizes.append(batch_size)
            return train_batches
        assert split == "test"
        assert batch_size is None
        return [evaluation_example]

    with tempfile.TemporaryDirectory() as directory:
        config = TrainConfig(
            normalize_world_space=False,
            model=ModelConfig(
                capacity=8,
                bucket_min_capacity=4,
                sh_degree=0,
                initial_scale=0.2,
            ),
            optimizer=OptimizerConfig(max_steps=3),
            strategy=StrategyConfig(
                refine_start=1,
                refine_stop=3,
                refine_every=1,
                reset_every=4,
                max_new_per_refine=2,
                grow_grad2d=0.0,
                grow_scale3d=100.0,
                prune_opacity=1.0e-6,
                prune_scale3d=100.0,
                prune_scale2d=100.0,
            ),
            data=DataConfig(
                root="unused", patch_size=4, batch_size=1, num_workers=1
            ),
            rasterizer=RasterizationConfig(
                backend="reference",
                tile_size=4,
                max_gaussians_per_tile=8,
                max_intersections=128,
            ),
            steps=3,
            checkpoint_every=1,
            eval_every=1,
            intersection_bucket_min_capacity=8,
            output_dir=directory,
            ssim_lambda=0.0,
        )
        with (
            mock.patch.object(
                training_module, "load_colmap_scene", return_value=scene
            ),
            mock.patch.object(
                training_module,
                "create_grain_dataset",
                side_effect=create_dataset,
            ),
            mock.patch.object(
                training_module,
                "rasterization",
                _host_loop_rasterization,
            ),
        ):
            uninterrupted = training_module.train(config, distributed=True)
            step_one = (
                uninterrupted.output_dir
                / "checkpoints"
                / "step_00000001"
            )
            step_one_manifest = load_distributed_checkpoint_manifest(step_one)
            assert step_one_manifest["local_capacity"] == 4
            assert step_one_manifest["intersection_capacity"] == 8
            assert step_one_manifest["candidate_bound"] == 8
            assert "scene" in step_one_manifest["components"]
            uninterrupted_state = _snapshot_graph_arrays(uninterrupted.model)

            resumed = training_module.train(
                config, resume_from=step_one, distributed=True
            )

        assert dataset_batch_sizes == [2, 2]
        assert uninterrupted.model.means[...].shape == (2, 8, 3)
        assert resumed.model.means[...].shape == (2, 8, 3)
        assert "P('rank'" in str(resumed.model.means[...].sharding)
        for expected, actual in zip(
            uninterrupted_state,
            _snapshot_graph_arrays(resumed.model),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        assert (
            resumed.output_dir / "renders" / "step_00000003.png"
        ).is_file()
        final_manifest = load_distributed_checkpoint_manifest(
            resumed.checkpoint
        )
        assert final_manifest["local_capacity"] == 8
        assert final_manifest["active_counts"] == [5, 4]


def _run_local_device_raster_overflow_replay():
    points = np.asarray(
        [
            [-0.20, 0.00, 3.0],
            [-0.10, 0.05, 3.0],
            [0.00, 0.00, 3.0],
            [0.10, -0.05, 3.0],
            [0.20, 0.00, 3.0],
        ],
        np.float32,
    )
    camtoworlds = np.stack(
        (np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))
    )
    camtoworlds[:, 0, 3] = np.asarray([-1.0, 1.0], np.float32)
    scene = SimpleNamespace(
        points=points,
        points_rgb=np.full((len(points), 3), 128, np.uint8),
        camtoworlds=camtoworlds,
    )
    K = np.asarray(
        [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
        np.float32,
    )
    batch = {
        "image": np.zeros((2, 4, 4, 3), np.float32),
        "K": np.tile(K, (2, 1, 1)),
        "w2c": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
    }
    with tempfile.TemporaryDirectory() as directory:
        config = TrainConfig(
            normalize_world_space=False,
            model=ModelConfig(
                capacity=4, bucket_min_capacity=4, sh_degree=0
            ),
            optimizer=OptimizerConfig(max_steps=1),
            strategy=StrategyConfig(
                refine_start=100, max_new_per_refine=1
            ),
            data=DataConfig(
                root="unused", patch_size=4, batch_size=1, num_workers=1
            ),
            rasterizer=RasterizationConfig(
                backend="reference",
                tile_size=4,
                max_gaussians_per_tile=4,
                max_candidates_per_tile=4,
                max_intersections=32,
            ),
            steps=1,
            checkpoint_every=0,
            eval_every=0,
            intersection_bucket_min_capacity=8,
            output_dir=directory,
            ssim_lambda=0.0,
        )
        with (
            mock.patch.object(
                training_module, "load_colmap_scene", return_value=scene
            ),
            mock.patch.object(
                training_module,
                "create_grain_dataset",
                return_value=[batch],
            ),
            mock.patch.object(
                training_module,
                "rasterization",
                _host_loop_overflow_rasterization,
            ),
        ):
            result = training_module.train(config, distributed=True)

        manifest = load_distributed_checkpoint_manifest(result.checkpoint)
        assert manifest["intersection_capacity"] == 16
        assert manifest["candidate_bound"] == 8
        assert manifest["active_counts"] == [3, 2]


def _run_local_device_distributed_pose_host_loop():
    points = np.asarray(
        [[-0.1, 0.0, 3.0], [0.1, 0.0, 3.0]], np.float32
    )
    scene = SimpleNamespace(
        points=points,
        points_rgb=np.full((2, 3), 128, np.uint8),
        camtoworlds=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        images=(
            SimpleNamespace(name="first.png"),
            SimpleNamespace(name="second.png"),
            SimpleNamespace(name="third.png"),
        ),
        indices=lambda split, test_every: np.asarray([0, 1, 2]),
    )
    K = np.asarray(
        [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
        np.float32,
    )
    batches = [
        {
            "image": np.full((2, 4, 4, 3), value, np.float32),
            "K": np.tile(K, (2, 1, 1)),
            "w2c": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
            "dataset_index": indices,
        }
        for value, indices in (
            (0.0, np.asarray([0, 1], np.int32)),
            (0.1, np.asarray([2, 0], np.int32)),
        )
    ]
    evaluation = {
        "image": np.zeros((4, 4, 3), np.float32),
        "K": K,
        "w2c": np.eye(4, dtype=np.float32),
        "dataset_index": np.asarray(0, np.int32),
    }
    dataset_batch_sizes = []

    def create_dataset(_scene, *, split, batch_size, **_kwargs):
        if split == "test":
            assert batch_size is None
            return [evaluation]
        assert split == "train"
        dataset_batch_sizes.append(batch_size)
        return batches

    with tempfile.TemporaryDirectory() as directory:
        config = TrainConfig(
            normalize_world_space=False,
            pose_opt=True,
            pose_opt_lr=0.1,
            pose_opt_reg=0.0,
            pose_noise=0.01,
            app_opt=True,
            app_embed_dim=2,
            app_opt_lr=0.1,
            app_opt_reg=0.0,
            model=ModelConfig(
                capacity=2,
                bucket_min_capacity=2,
                sh_degree=0,
                initial_scale=0.2,
            ),
            optimizer=OptimizerConfig(max_steps=2),
            strategy=StrategyConfig(
                refine_start=100,
                refine_stop=101,
                max_new_per_refine=1,
            ),
            data=DataConfig(
                root="unused", patch_size=4, batch_size=1, num_workers=1
            ),
            rasterizer=RasterizationConfig(
                backend="reference",
                tile_size=4,
                max_gaussians_per_tile=4,
                max_intersections=32,
            ),
            steps=2,
            checkpoint_every=1,
            eval_every=1,
            output_dir=directory,
            ssim_lambda=0.0,
        )
        with (
            mock.patch.object(
                training_module, "load_colmap_scene", return_value=scene
            ),
            mock.patch.object(
                training_module,
                "create_grain_dataset",
                side_effect=create_dataset,
            ),
            mock.patch.object(
                training_module,
                "rasterization",
                _camera_module_sensitive_rasterization,
            ),
        ):
            uninterrupted = training_module.train(config, distributed=True)
            step_one = (
                uninterrupted.output_dir
                / "checkpoints"
                / "step_00000001"
            )
            uninterrupted_model = _snapshot_graph_arrays(
                uninterrupted.model
            )
            uninterrupted_pose = np.asarray(
                uninterrupted.pose_adjust.embeds.embedding[...]
            ).copy()
            uninterrupted_appearance = _snapshot_graph_arrays(
                uninterrupted.appearance
            )
            resumed = training_module.train(
                config, resume_from=step_one, distributed=True
            )
            one_step_optimizer = replace(config.optimizer, max_steps=1)
            opt_only = training_module.train(
                replace(
                    config,
                    pose_noise=0.0,
                    app_opt=False,
                    steps=1,
                    optimizer=one_step_optimizer,
                    output_dir=os.path.join(directory, "opt_only"),
                ),
                distributed=True,
            )
            noise_only = training_module.train(
                replace(
                    config,
                    pose_opt=False,
                    app_opt=False,
                    steps=1,
                    optimizer=one_step_optimizer,
                    output_dir=os.path.join(directory, "noise_only"),
                ),
                distributed=True,
            )
            appearance_only = training_module.train(
                replace(
                    config,
                    pose_opt=False,
                    pose_noise=0.0,
                    steps=1,
                    optimizer=one_step_optimizer,
                    output_dir=os.path.join(directory, "appearance_only"),
                ),
                distributed=True,
            )

        assert dataset_batch_sizes == [2, 2, 2, 2, 2]
        assert opt_only.pose_adjust is not None
        assert noise_only.pose_adjust is None
        assert appearance_only.pose_adjust is None
        assert appearance_only.appearance is not None
        assert uninterrupted_pose.shape == (2, 3, 9)
        np.testing.assert_array_equal(
            uninterrupted_pose[0], uninterrupted_pose[1]
        )
        assert np.any(uninterrupted_pose[0, 0] != 0.0)
        assert np.any(uninterrupted_pose[0, 1] != 0.0)
        assert np.any(uninterrupted_pose[0, 2] != 0.0)
        assert "P('rank'" in str(
            resumed.pose_adjust.embeds.embedding[...].sharding
        )
        np.testing.assert_array_equal(
            resumed.pose_adjust.embeds.embedding[...], uninterrupted_pose
        )
        assert "P('rank'" in str(
            resumed.appearance.embeds.embedding[...].sharding
        )
        for expected, actual in zip(
            uninterrupted_appearance,
            _snapshot_graph_arrays(resumed.appearance),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        for expected, actual in zip(
            uninterrupted_model,
            _snapshot_graph_arrays(resumed.model),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        manifest = load_distributed_checkpoint_manifest(resumed.checkpoint)
        assert "pose" in manifest["components"]
        assert "appearance" in manifest["components"]
        assert manifest["pose_camera_count"] == 3
        assert manifest["appearance_camera_count"] == 3
        assert manifest["pose_image_names"] == [
            "first.png",
            "second.png",
            "third.png",
        ]
        assert manifest["appearance_image_names"] == (
            manifest["pose_image_names"]
        )
        assert (resumed.output_dir / "renders" / "step_00000002.png").is_file()


def test_two_virtual_cpu_distributed_host_loop_smoke():
    script = "\n".join(
        (
            "import jax",
            "from tests.test_training_distributed import "
            "_run_local_device_distributed_host_loop, "
            "_run_local_device_distributed_pose_host_loop, "
            "_run_local_device_raster_overflow_replay",
            "assert jax.local_device_count() == 2",
            "_run_local_device_distributed_host_loop()",
            "_run_local_device_distributed_pose_host_loop()",
            "_run_local_device_raster_overflow_replay()",
        )
    )
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_one_rank_overflow_atomically_skips_both_ranks(monkeypatch):
    monkeypatch.setattr(
        training_module,
        "rasterization",
        _rank_zero_overflow_rasterization,
    )
    config = _fixed_topology_config()
    model, optimizer, strategy_state, safety_state = _stack_graphs(
        _rank_bundle(config, -0.08),
        _rank_bundle(config, 0.08),
    )
    strategy_state.grad_accum[...] = jnp.asarray([[5.0], [7.0]])
    strategy_state.visible_count[...] = jnp.asarray([[11.0], [13.0]])
    strategy_state.max_radii[...] = jnp.asarray([[0.25], [0.75]])
    train_step = make_distributed_train_step(
        config, world_size=2, axis_name="rank"
    )

    @nnx.vmap(
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
        axis_name="rank",
    )
    def mapped_step(
        current_model,
        current_optimizer,
        current_strategy_state,
        current_safety_state,
        images,
        intrinsics,
        viewmats,
        key,
        sh_degree,
    ):
        return train_step(
            current_model,
            current_optimizer,
            current_strategy_state,
            current_safety_state,
            images,
            intrinsics,
            viewmats,
            key,
            sh_degree,
        )

    images = jnp.zeros((2, 1, 4, 4, 3), jnp.float32)
    intrinsics = jnp.broadcast_to(
        jnp.asarray(
            [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
            jnp.float32,
        )[None, None],
        (2, 1, 3, 3),
    )
    viewmats = jnp.broadcast_to(
        jnp.eye(4, dtype=jnp.float32)[None, None],
        (2, 1, 4, 4),
    )
    keys = jax.random.split(jax.random.key(0), 2)
    sh_degrees = jnp.zeros((2,), jnp.int32)
    call_args = (
        model,
        optimizer,
        strategy_state,
        safety_state,
        images,
        intrinsics,
        viewmats,
        keys,
        sh_degrees,
    )
    model_before = _snapshot_graph_arrays(model)
    optimizer_before = _snapshot_graph_arrays(optimizer)
    strategy_before = _snapshot_graph_arrays(strategy_state)
    metrics = mapped_step(*call_args)

    np.testing.assert_array_equal(
        metrics["intersection_overflow"], [True, True]
    )
    np.testing.assert_array_equal(
        metrics["intersection_required_count"], [7, 7]
    )
    np.testing.assert_array_equal(optimizer.step[...], [0, 0])
    np.testing.assert_array_equal(
        safety_state.max_overflow_tiles[...], [0, 0]
    )
    np.testing.assert_array_equal(
        safety_state.intersection_overflow_seen[...], [True, True]
    )
    for before, after in zip(
        model_before, _snapshot_graph_arrays(model), strict=True
    ):
        np.testing.assert_array_equal(after, before)
    for before, after in zip(
        optimizer_before, _snapshot_graph_arrays(optimizer), strict=True
    ):
        np.testing.assert_array_equal(after, before)
    for before, after in zip(
        strategy_before, _snapshot_graph_arrays(strategy_state), strict=True
    ):
        np.testing.assert_array_equal(after, before)

    safety_state.intersection_overflow_seen[...] = jnp.asarray(
        [True, False]
    )
    safety_state.max_overflow_tiles[...] = jnp.asarray([3, 0])
    monkeypatch.setattr(
        training_module,
        "rasterization",
        _no_overflow_rasterization,
    )
    sticky_train_step = make_distributed_train_step(
        config, world_size=2, axis_name="rank"
    )

    @nnx.vmap(
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
        axis_name="rank",
    )
    def sticky_mapped_step(*args):
        return sticky_train_step(*args)

    sticky_metrics = sticky_mapped_step(*call_args)

    np.testing.assert_array_equal(
        sticky_metrics["intersection_overflow"], [False, False]
    )
    np.testing.assert_array_equal(
        safety_state.intersection_overflow_seen[...], [True, True]
    )
    np.testing.assert_array_equal(
        safety_state.max_overflow_tiles[...], [3, 3]
    )
    np.testing.assert_array_equal(optimizer.step[...], [0, 0])
    for before, after in zip(
        strategy_before, _snapshot_graph_arrays(strategy_state), strict=True
    ):
        np.testing.assert_array_equal(after, before)


def test_distributed_checkpoint_round_trip(tmp_path):
    config = _topology_plan_config()
    saved = _two_rank_bundles(config)
    model, optimizer, strategy_state, safety_state = saved
    optimizer.step[...] = jnp.asarray([3, 3], optimizer.step[...].dtype)
    model.active_mask[...] = jnp.asarray([[True, True], [True, False]])
    strategy_state.grad_accum[...] = jnp.asarray([[1.5, 0.25], [0.5, 0.0]])
    safety_state.max_overflow_tiles[...] = jnp.asarray([4, 0])
    safety_state.intersection_overflow_seen[...] = jnp.asarray([True, False])

    scene_transform = np.asarray(
        [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, -1.0],
            [0.0, 0.0, 2.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        np.float32,
    )
    path = save_distributed_checkpoint(
        tmp_path,
        *saved,
        step=3,
        config=config,
        intersection_capacity=64,
        candidate_bound=12,
        scene_transform=scene_transform,
        scene_scale=1.0,
    )

    manifest = load_distributed_checkpoint_manifest(path)
    assert manifest["world_size"] == 2
    assert manifest["local_capacity"] == 2
    assert manifest["global_capacity"] == 4
    assert manifest["active_counts"] == [2, 1]
    assert manifest["active_prefix"] == [True, True]
    assert manifest["optimizer_contract"] == {
        "batch_size": 1,
        "world_size": 2,
        "scene_scale": 1.0,
        "kind": "adam",
        "config": config.to_dict()["optimizer"],
    }
    assert manifest["components"] == [
        "model",
        "optimizer",
        "strategy",
        "safety",
        "scene",
    ]
    assert manifest["intersection_capacity"] == 64
    assert manifest["candidate_bound"] == 12
    restored_transform, restored_scale = load_checkpoint_scene_transform(path)
    np.testing.assert_array_equal(restored_transform, scene_transform)
    assert restored_scale == 1.0
    assert load_checkpoint_intersection_capacity(path) == 64

    restored = _two_rank_bundles(config)
    assert restore_distributed_checkpoint(path, *restored, config=config) == 3
    for original, target in zip(saved, restored, strict=True):
        for before, after in zip(
            _snapshot_graph_arrays(original),
            _snapshot_graph_arrays(target),
            strict=True,
        ):
            np.testing.assert_array_equal(after, before)


def test_distributed_checkpoint_round_trips_canonical_pose_state(tmp_path):
    config = _topology_plan_config(train={"pose_opt": True})
    saved = _two_rank_bundles(config)
    saved[0].active_mask[...] = jnp.ones_like(saved[0].active_mask[...])
    saved[1].step[...] = jnp.asarray([1, 1], saved[1].step[...].dtype)
    pose_module, pose_optimizer = _replicated_pose_training_state(config)
    names = ("first.png", "second.png", "third.png")
    path = save_distributed_checkpoint(
        tmp_path,
        *saved,
        step=1,
        config=config,
        pose_module=pose_module,
        pose_optimizer=pose_optimizer,
        pose_image_names=names,
    )

    manifest = load_distributed_checkpoint_manifest(path)
    assert "pose" in manifest["components"]
    assert manifest["pose_camera_count"] == 3
    assert manifest["pose_image_names"] == list(names)

    core_only = _two_rank_bundles(config)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert (
            restore_distributed_checkpoint(path, *core_only, config=config)
            == 1
        )

    restored = _two_rank_bundles(config)
    restored_pose = _replicated_pose_training_state(config, steps=0)
    assert (
        restore_distributed_checkpoint(
            path,
            *restored,
            config=config,
            pose_module=restored_pose[0],
            pose_optimizer=restored_pose[1],
            pose_image_names=names,
        )
        == 1
    )
    for original, target in (
        (pose_module, restored_pose[0]),
        (pose_optimizer, restored_pose[1]),
    ):
        for before, after in zip(
            _snapshot_graph_arrays(original),
            _snapshot_graph_arrays(target),
            strict=True,
        ):
            np.testing.assert_array_equal(after, before)

    wrong_names_pose = _replicated_pose_training_state(config, steps=0)
    with pytest.raises(ValueError, match="pose image names"):
        restore_distributed_checkpoint(
            path,
            *_two_rank_bundles(config),
            config=config,
            pose_module=wrong_names_pose[0],
            pose_optimizer=wrong_names_pose[1],
            pose_image_names=tuple(reversed(names)),
        )

    wrong_pose_config = replace(config, pose_opt_lr=config.pose_opt_lr * 2.0)
    wrong_pose_optimizer = _replicated_pose_training_state(
        wrong_pose_config, steps=0
    )
    with pytest.raises(ValueError, match="pose optimizer contract"):
        restore_distributed_checkpoint(
            path,
            *_two_rank_bundles(config),
            config=config,
            pose_module=wrong_pose_optimizer[0],
            pose_optimizer=wrong_pose_optimizer[1],
            pose_image_names=names,
        )

    # Canonical pose state is independent of the Gaussian owner count.
    four_rank_pose = _replicated_pose_training_state(
        config, world_size=4, steps=0
    )
    four_rank_core = _stack_graphs(
        *[
            _rank_bundle(config, 0.0, optimizer_world_size=4)
            for _ in range(4)
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert (
            restore_distributed_checkpoint(
                path,
                *four_rank_core,
                config=config,
                model_config=config.model,
                optimizer_config=config.optimizer,
                allow_reshard=True,
            )
            == 1
        )
    restore_distributed_checkpoint(
        path,
        *four_rank_core,
        config=config,
        model_config=config.model,
        optimizer_config=config.optimizer,
        allow_reshard=True,
        pose_module=four_rank_pose[0],
        pose_optimizer=four_rank_pose[1],
        pose_image_names=names,
    )
    expected_embedding = np.asarray(pose_module.embeds.embedding[0])
    np.testing.assert_array_equal(
        four_rank_pose[0].embeds.embedding[...],
        np.broadcast_to(expected_embedding, (4, *expected_embedding.shape)),
    )

    # A failed Gaussian redistribution must not partially restore pose state.
    one_rank_core = _stack_graphs(
        _rank_bundle(config, 0.0, optimizer_world_size=1)
    )
    one_rank_pose = _replicated_pose_training_state(
        config, world_size=1, steps=0
    )
    targets = (*one_rank_core, *one_rank_pose)
    before_failure = tuple(
        _snapshot_graph_arrays(node) for node in targets
    )
    with pytest.raises(ValueError, match="cannot hold the busiest"):
        restore_distributed_checkpoint(
            path,
            *one_rank_core,
            config=config,
            model_config=config.model,
            optimizer_config=config.optimizer,
            allow_reshard=True,
            pose_module=one_rank_pose[0],
            pose_optimizer=one_rank_pose[1],
            pose_image_names=names,
        )
    for node, expected_state in zip(targets, before_failure, strict=True):
        for actual, expected in zip(
            _snapshot_graph_arrays(node), expected_state, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)


def test_distributed_checkpoint_round_trips_canonical_appearance_state(
    tmp_path,
):
    config = _topology_plan_config(
        train={"app_opt": True, "app_embed_dim": 0}
    )
    saved = _two_rank_bundles(config)
    saved[0].active_mask[...] = jnp.ones_like(saved[0].active_mask[...])
    saved[1].step[...] = jnp.asarray([1, 1], saved[1].step[...].dtype)
    appearance, appearance_optimizer = (
        _replicated_appearance_training_state(config, steps=1)
    )
    names = ("first.png", "second.png", "third.png")
    with pytest.raises(ValueError, match="color representation"):
        save_distributed_checkpoint(
            tmp_path / "wrong_color_mode",
            *saved,
            step=1,
            config=replace(config, app_opt=False),
        )
    path = save_distributed_checkpoint(
        tmp_path,
        *saved,
        step=1,
        config=config,
        appearance_module=appearance,
        appearance_optimizer=appearance_optimizer,
        appearance_image_names=names,
    )

    manifest = load_distributed_checkpoint_manifest(path)
    assert "appearance" in manifest["components"]
    assert manifest["appearance_camera_count"] == 3
    assert manifest["appearance_image_names"] == list(names)
    assert (
        manifest["appearance_feature_dim"]
        == training_module.APPEARANCE_FEATURE_DIM
    )

    core_only = _two_rank_bundles(config)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert (
            restore_distributed_checkpoint(path, *core_only, config=config)
            == 1
        )

    restored = _two_rank_bundles(config)
    restored_appearance = _replicated_appearance_training_state(config)
    assert (
        restore_distributed_checkpoint(
            path,
            *restored,
            config=config,
            appearance_module=restored_appearance[0],
            appearance_optimizer=restored_appearance[1],
            appearance_image_names=names,
        )
        == 1
    )
    for original, target in (
        (appearance, restored_appearance[0]),
        (appearance_optimizer, restored_appearance[1]),
    ):
        for before, after in zip(
            _snapshot_graph_arrays(original),
            _snapshot_graph_arrays(target),
            strict=True,
        ):
            np.testing.assert_array_equal(after, before)

    with pytest.raises(ValueError, match="appearance image names"):
        restore_distributed_checkpoint(
            path,
            *_two_rank_bundles(config),
            config=config,
            appearance_module=restored_appearance[0],
            appearance_optimizer=restored_appearance[1],
            appearance_image_names=tuple(reversed(names)),
        )

    four_rank_core = _stack_graphs(
        *[
            _rank_bundle(config, 0.0, optimizer_world_size=4)
            for _ in range(4)
        ]
    )
    four_rank_appearance = _replicated_appearance_training_state(
        config, world_size=4
    )
    restore_distributed_checkpoint(
        path,
        *four_rank_core,
        config=config,
        model_config=config.model,
        optimizer_config=config.optimizer,
        allow_reshard=True,
        appearance_module=four_rank_appearance[0],
        appearance_optimizer=four_rank_appearance[1],
        appearance_image_names=names,
    )
    for source, target in (
        (appearance, four_rank_appearance[0]),
        (appearance_optimizer, four_rank_appearance[1]),
    ):
        expected = _snapshot_graph_arrays(_unstack_graph(source, 0))
        for rank in range(4):
            actual = _snapshot_graph_arrays(_unstack_graph(target, rank))
            for actual_leaf, expected_leaf in zip(
                actual, expected, strict=True
            ):
                np.testing.assert_array_equal(actual_leaf, expected_leaf)

    divergent_appearance = _replicated_appearance_training_state(
        config, steps=1
    )
    divergent_appearance[0].color_head[-1].bias[1, 0] += 1.0
    with pytest.raises(
        ValueError, match="appearance module replicas disagree"
    ):
        save_distributed_checkpoint(
            tmp_path / "divergent",
            *saved,
            step=1,
            config=config,
            appearance_module=divergent_appearance[0],
            appearance_optimizer=divergent_appearance[1],
            appearance_image_names=names,
        )

    divergent_appearance = _replicated_appearance_training_state(
        config, steps=1
    )
    divergent_appearance[1].step[1] = 0
    with pytest.raises(
        ValueError, match="appearance optimizer replicas disagree"
    ):
        save_distributed_checkpoint(
            tmp_path / "divergent_optimizer",
            *saved,
            step=1,
            config=config,
            appearance_module=divergent_appearance[0],
            appearance_optimizer=divergent_appearance[1],
            appearance_image_names=names,
        )

    wrong_contract = _replicated_appearance_training_state(
        replace(config, app_opt_lr=config.app_opt_lr * 2.0), steps=1
    )
    with pytest.raises(ValueError, match="does not match TrainConfig"):
        save_distributed_checkpoint(
            tmp_path / "wrong_contract",
            *saved,
            step=1,
            config=config,
            appearance_module=wrong_contract[0],
            appearance_optimizer=wrong_contract[1],
            appearance_image_names=names,
        )

    # A failed Gaussian redistribution must not partially restore appearance.
    one_rank_core = _stack_graphs(
        _rank_bundle(config, 0.0, optimizer_world_size=1)
    )
    one_rank_appearance = _replicated_appearance_training_state(
        config, world_size=1
    )
    targets = (*one_rank_core, *one_rank_appearance)
    before_failure = tuple(
        _snapshot_graph_arrays(node) for node in targets
    )
    with pytest.raises(ValueError, match="cannot hold the busiest"):
        restore_distributed_checkpoint(
            path,
            *one_rank_core,
            config=config,
            model_config=config.model,
            optimizer_config=config.optimizer,
            allow_reshard=True,
            appearance_module=one_rank_appearance[0],
            appearance_optimizer=one_rank_appearance[1],
            appearance_image_names=names,
        )
    for node, expected_state in zip(targets, before_failure, strict=True):
        for actual, expected in zip(
            _snapshot_graph_arrays(node), expected_state, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)


def test_distributed_pose_checkpoint_validates_replicas_and_noise_names(
    tmp_path,
):
    config = _topology_plan_config(train={"pose_opt": True})
    saved = _two_rank_bundles(config)
    saved[1].step[...] = jnp.asarray([1, 1], saved[1].step[...].dtype)
    pose_module, pose_optimizer = _replicated_pose_training_state(config)
    names = ("first.png", "second.png", "third.png")

    pose_module.embeds.embedding[1, 0, 0] += 1.0
    with pytest.raises(ValueError, match="pose module replicas disagree"):
        save_distributed_checkpoint(
            tmp_path / "module",
            *saved,
            step=1,
            config=config,
            pose_module=pose_module,
            pose_optimizer=pose_optimizer,
            pose_image_names=names,
        )

    pose_module, pose_optimizer = _replicated_pose_training_state(config)
    pose_optimizer.step[...] = jnp.asarray(
        [1, 0], pose_optimizer.step[...].dtype
    )
    with pytest.raises(ValueError, match="pose optimizer replicas disagree"):
        save_distributed_checkpoint(
            tmp_path / "optimizer",
            *saved,
            step=1,
            config=config,
            pose_module=pose_module,
            pose_optimizer=pose_optimizer,
            pose_image_names=names,
        )

    noise_config = _topology_plan_config(train={"pose_noise": 0.01})
    noise_path = save_distributed_checkpoint(
        tmp_path / "noise",
        *_two_rank_bundles(noise_config),
        step=0,
        config=noise_config,
        pose_image_names=names,
    )
    noise_manifest = load_distributed_checkpoint_manifest(noise_path)
    assert "pose" not in noise_manifest["components"]
    assert noise_manifest["pose_camera_count"] == 3
    assert (
        restore_distributed_checkpoint(
            noise_path,
            *_two_rank_bundles(noise_config),
            config=noise_config,
            pose_image_names=names,
        )
        == 0
    )
    with pytest.raises(ValueError, match="pose image names"):
        restore_distributed_checkpoint(
            noise_path,
            *_two_rank_bundles(noise_config),
            config=noise_config,
            pose_image_names=tuple(reversed(names)),
        )


def test_distributed_restore_rejects_resharding(tmp_path):
    config = _topology_plan_config()
    path = save_distributed_checkpoint(
        tmp_path, *_two_rank_bundles(config), step=0
    )
    three_ranks = _stack_graphs(
        _rank_bundle(config, -0.08),
        _rank_bundle(config, 0.0),
        _rank_bundle(config, 0.08),
    )

    # Refusing by default is the point: a width that does not match the
    # checkpoint is far more often a misconfigured run than a move.
    with pytest.raises(ValueError, match="pass allow_reshard=True"):
        restore_distributed_checkpoint(path, *three_ranks)
    with pytest.raises(ValueError, match="requires model_config"):
        restore_distributed_checkpoint(
            path, *three_ranks, allow_reshard=True
        )


def test_distributed_restore_rejects_a_different_shard_capacity(tmp_path):
    path = save_distributed_checkpoint(
        tmp_path,
        *_two_rank_bundles(_topology_plan_config(capacity=2)),
        step=0,
    )

    with pytest.raises(ValueError, match="capacity"):
        restore_distributed_checkpoint(
            path, *_two_rank_bundles(_topology_plan_config(capacity=4))
        )


def test_distributed_restore_rejects_a_different_config(tmp_path):
    config = _topology_plan_config()
    path = save_distributed_checkpoint(
        tmp_path, *_two_rank_bundles(config), step=0, config=config
    )

    with pytest.raises(ValueError, match="different training config"):
        restore_distributed_checkpoint(
            path,
            *_two_rank_bundles(config),
            config=_topology_plan_config(reset_every=3),
        )


def test_distributed_restore_validates_saved_optimizer_contract(tmp_path):
    config = _topology_plan_config()
    path = save_distributed_checkpoint(
        tmp_path / "kind", *_two_rank_bundles(config), step=0
    )
    wrong_kind_ranks = []
    for x in (-0.08, 0.08):
        rank = _rank_bundle(config, x)
        wrong_kind_ranks.append(
            (
                rank[0],
                create_visible_adam_optimizer(
                    rank[0],
                    config.optimizer,
                    batch_size=config.data.batch_size,
                    world_size=2,
                ),
                rank[2],
                rank[3],
            )
        )
    wrong_kind = _stack_graphs(*wrong_kind_ranks)
    with pytest.raises(ValueError, match="optimizer kind"):
        restore_distributed_checkpoint(path, *wrong_kind)

    path = save_distributed_checkpoint(
        tmp_path / "scene",
        *_two_rank_bundles(config, optimizer_scene_scale=1.0),
        step=0,
        config=config,
    )
    wrong_scene_scale = _two_rank_bundles(
        config, optimizer_scene_scale=2.0
    )
    with pytest.raises(ValueError, match="optimizer scene_scale"):
        restore_distributed_checkpoint(
            path, *wrong_scene_scale, config=config
        )


def test_legacy_distributed_optimizer_contract_requires_config(tmp_path):
    config = _topology_plan_config()
    path = save_distributed_checkpoint(
        tmp_path, *_two_rank_bundles(config), step=0, config=config
    )
    metadata_path = path / "jax_gs_checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["optimizer_contract"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy.*requires config"):
        restore_distributed_checkpoint(path, *_two_rank_bundles(config))
    assert (
        restore_distributed_checkpoint(
            path, *_two_rank_bundles(config), config=config
        )
        == 0
    )
    metadata["config_fingerprint"] = None
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert (
        restore_distributed_checkpoint(
            path, *_two_rank_bundles(config), config=config
        )
        == 0
    )


def test_distributed_and_single_checkpoints_reject_each_other(tmp_path):
    config = _topology_plan_config()
    distributed_path = save_distributed_checkpoint(
        tmp_path / "distributed", *_two_rank_bundles(config), step=0
    )
    single_model, single_optimizer, single_state, _ = _rank_bundle(config, 0.0)

    with pytest.raises(ValueError, match="distributed shards"):
        restore_checkpoint(
            distributed_path,
            single_model,
            optimizer=single_optimizer,
            strategy_state=single_state,
        )

    single_path = save_checkpoint(
        tmp_path / "single",
        single_model,
        step=0,
        optimizer=single_optimizer,
        strategy_state=single_state,
    )
    with pytest.raises(ValueError, match="not a distributed shard set"):
        restore_distributed_checkpoint(
            single_path, *_two_rank_bundles(config)
        )


def test_distributed_save_rejects_unsharded_state(tmp_path):
    config = _topology_plan_config()

    with pytest.raises(ValueError, match="unsharded leaf"):
        save_distributed_checkpoint(
            tmp_path, *_rank_bundle(config, 0.0), step=0
        )


def test_distributed_save_rejects_shards_at_different_steps(tmp_path):
    config = _topology_plan_config()
    bundles = _two_rank_bundles(config)
    bundles[1].step[...] = jnp.asarray([1, 2], bundles[1].step[...].dtype)

    with pytest.raises(ValueError, match="disagree on the optimizer step"):
        save_distributed_checkpoint(tmp_path, *bundles, step=1)

    with pytest.raises(ValueError, match="step argument"):
        save_distributed_checkpoint(
            tmp_path, *_two_rank_bundles(config), step=1
        )


def test_restored_shards_continue_distributed_training(tmp_path):
    config = _topology_plan_config()
    trained = _duplicate_only_run(nnx.vmap)[:4]
    path = save_distributed_checkpoint(
        tmp_path, *trained, step=1, config=config
    )

    restored = _two_rank_bundles(config)
    assert restore_distributed_checkpoint(path, *restored, config=config) == 1

    with mock.patch.object(
        training_module, "rasterization", _no_statistics_rasterization
    ):
        model, optimizer, _, _, metrics = _run_two_rank_update(
            nnx.vmap,
            config=config,
            bundles=restored,
            initial_optimizer_steps=(1, 1),
        )

    np.testing.assert_array_equal(
        metrics["distributed_state_mismatch"], [False, False]
    )
    np.testing.assert_array_equal(optimizer.step[...], [2, 2])
    # The restored duplicate keeps training as an ordinary owner-local row.
    np.testing.assert_array_equal(
        model.active_mask[...], [[True, True], [True, False]]
    )


def _resize_world(config, bundles, new_capacity):
    model, optimizer, strategy_state = resize_distributed_training_state(
        bundles[0],
        bundles[1],
        bundles[2],
        new_capacity,
        config.model,
        config.optimizer,
    )
    return model, optimizer, strategy_state, bundles[3]


def test_growing_shards_preserves_every_owner():
    config = _topology_plan_config(capacity=4, bucket=2)
    bundles = _two_rank_bundles(config)
    model, optimizer, strategy_state, _ = bundles
    model.active_mask[...] = jnp.asarray([[True, True], [True, False]])
    _set_owner_statistics(
        strategy_state, [[1.0, 2.0], [3.0, 0.0]], [[0.25, 0.5], [0.75, 0.0]]
    )
    optimizer.step[...] = jnp.asarray([5, 5], optimizer.step[...].dtype)
    means_before = np.asarray(model.means[...]).copy()
    log_scales_before = np.asarray(model.log_scales[...]).copy()

    grown, grown_optimizer, grown_state, _ = _resize_world(
        config, bundles, 4
    )

    assert grown.means[...].shape == (2, 4, 3)
    assert grown.max_capacity == 4
    np.testing.assert_array_equal(grown.means[:, :2], means_before)
    np.testing.assert_array_equal(grown.means[:, 2:], 0.0)
    np.testing.assert_array_equal(grown.log_scales[:, :2], log_scales_before)
    np.testing.assert_allclose(
        grown.log_scales[:, 2:],
        np.full((2, 2, 3), np.log(config.model.initial_scale), np.float32),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        grown.active_mask[...],
        [[True, True, False, False], [True, False, False, False]],
    )
    np.testing.assert_array_equal(
        grown_state.grad_accum[...],
        [[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]],
    )
    np.testing.assert_array_equal(
        grown_state.max_radii[...],
        [[0.25, 0.5, 0.0, 0.0], [0.75, 0.0, 0.0, 0.0]],
    )
    # The schedule and the optimizer contract survive the transition.
    np.testing.assert_array_equal(grown_optimizer.step[...], [5, 5])
    assert _adam_means_moments(grown_optimizer).shape == (2, 4, 3)
    assert getattr(grown_optimizer, "_jax_gs_world_size") == 2
    assert getattr(grown_optimizer, "_jax_gs_scene_scale") == 1.0


def test_growing_shards_rejects_bad_targets():
    config = _topology_plan_config(capacity=4, bucket=2)

    with pytest.raises(ValueError, match="larger than the current capacity"):
        _resize_world(config, _two_rank_bundles(config), 2)
    with pytest.raises(ValueError, match="logical maximum"):
        _resize_world(config, _two_rank_bundles(config), 8)
    with pytest.raises(ValueError, match="unsharded leaf"):
        resize_distributed_training_state(
            *_rank_bundle(config, 0.0)[:3],
            4,
            config.model,
            config.optimizer,
        )


def test_growing_all_shards_lets_the_skipped_step_replay():
    config = _topology_plan_config(
        capacity=4,
        bucket=2,
        refine_scale2d_stop_iter=100,
        grow_scale2d=0.05,
    )
    bundles = _two_rank_bundles(config)
    _set_owner_statistics(
        bundles[2], [[1.0, 0.0], [0.0, 0.0]], [[0.5, 0.0], [0.0, 0.0]]
    )
    train_step = make_distributed_train_step(config, world_size=2)

    @nnx.vmap(
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0, axis_name="rank"
    )
    def mapped_step(*args):
        return train_step(*args)

    images = jnp.zeros((2, 1, 4, 4, 3), jnp.float32)
    intrinsics = jnp.broadcast_to(
        jnp.asarray(
            [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]], jnp.float32
        )[None, None],
        (2, 1, 3, 3),
    )
    viewmats = jnp.broadcast_to(
        jnp.eye(4, dtype=jnp.float32)[None, None], (2, 1, 4, 4)
    )
    keys = jax.random.split(jax.random.key(0), 2)
    sh_degrees = jnp.zeros((2,), jnp.int32)

    with mock.patch.object(
        training_module, "rasterization", _no_statistics_rasterization
    ):
        skipped = mapped_step(
            *bundles, images, intrinsics, viewmats, keys, sh_degrees
        )

        np.testing.assert_array_equal(
            skipped["refine_capacity_overflow"], [True, True]
        )
        np.testing.assert_array_equal(skipped["refine_new_count"], [0, 0])
        np.testing.assert_array_equal(bundles[1].step[...], [0, 0])

        grown = _resize_world(config, bundles, 4)
        replayed = mapped_step(
            *grown, images, intrinsics, viewmats, keys, sh_degrees
        )

    # The replay sees the same schedule and now fits both planned events.
    np.testing.assert_array_equal(
        replayed["refine_capacity_overflow"], [False, False]
    )
    np.testing.assert_array_equal(replayed["refine_planned_new_count"], [2, 2])
    np.testing.assert_array_equal(replayed["refine_new_count"], [2, 2])
    np.testing.assert_array_equal(grown[1].step[...], [1, 1])
    np.testing.assert_array_equal(
        np.asarray(grown[0].active_mask[...]).sum(axis=1), [3, 1]
    )


def _pose_sensitive_rasterization(*args, **kwargs):
    """Render a signal that depends only on the rank's own camera poses."""

    means = args[0]
    viewmats = args[5]
    renders, alphas, info = _no_statistics_rasterization(*args, **kwargs)
    signal = jax.nn.sigmoid(viewmats[:, 0, 3])
    renders = jnp.broadcast_to(
        signal[:, None, None, None] + 0.0 * renders, renders.shape
    ).astype(means.dtype)
    return renders, alphas, info


def _appearance_sensitive_rasterization(
    means,
    _quats,
    _scales,
    _opacities,
    colors,
    viewmats,
    _intrinsics,
    width,
    height,
    **kwargs,
):
    """Render rank-local cameras from already-gathered appearance colors."""

    assert not kwargs["distributed"]
    camera_count = viewmats.shape[0]
    global_capacity = means.shape[0]
    assert colors.shape == (camera_count, global_capacity, 3)
    assert kwargs["active_mask"].shape == (global_capacity,)
    screen_probe = kwargs.get("_means2d_offset")
    if screen_probe is not None:
        assert screen_probe.shape == (camera_count, global_capacity, 2)
    active = kwargs["active_mask"].astype(means.dtype)
    weights = active / jnp.maximum(jnp.sum(active), 1.0)
    signal = jnp.einsum("cnd,n->cd", colors, weights)
    renders = jnp.broadcast_to(
        signal[:, None, None, :],
        (camera_count, height, width, 3),
    )
    alphas = jnp.ones(
        (camera_count, height, width, 1), dtype=means.dtype
    )
    visible = jnp.broadcast_to(
        kwargs["active_mask"][None, :],
        (camera_count, global_capacity),
    )
    info = {
        "radii": jnp.ones(
            (camera_count, global_capacity, 2), dtype=means.dtype
        ),
        "valid": visible,
        "tile_overflow": jnp.zeros(
            (camera_count, 1, 1), dtype=jnp.bool_
        ),
        "candidate_limit_exceeded": jnp.zeros(
            (camera_count, 1, 1), dtype=jnp.bool_
        ),
        "candidate_counts": jnp.zeros(
            (camera_count, 1, 1), dtype=jnp.int32
        ),
        "intersection_overflow": jnp.zeros(
            (camera_count,), dtype=jnp.bool_
        ),
        "intersection_count": jnp.ones(
            (camera_count,), dtype=jnp.int32
        ),
        "intersection_required_count": jnp.ones(
            (camera_count,), dtype=jnp.int32
        ),
    }
    return renders, alphas, info


def _pose_and_appearance_sensitive_rasterization(*args, **kwargs):
    renders, alphas, info = _appearance_sensitive_rasterization(
        *args, **kwargs
    )
    viewmats = args[5]
    pose_signal = jax.nn.sigmoid(viewmats[:, 0, 3])
    return (
        renders + pose_signal[:, None, None, None],
        alphas,
        info,
    )


def _camera_module_sensitive_rasterization(*args, **kwargs):
    if kwargs["distributed"]:
        return _pose_sensitive_rasterization(*args, **kwargs)
    return _pose_and_appearance_sensitive_rasterization(*args, **kwargs)


@pytest.mark.parametrize(
    "initial_appearance_steps", [(0, 0), (0, 1)]
)
def test_named_two_rank_appearance_uses_global_gaussians_and_ddp_gradients(
    initial_appearance_steps,
):
    config = _topology_plan_config(
        refine_start=4,
        train={
            "app_opt": True,
            "app_embed_dim": 1,
            "app_opt_lr": 0.1,
            "app_opt_reg": 0.0,
        },
    )
    bundles = _two_rank_bundles(config)
    appearance, appearance_optimizer = (
        _replicated_appearance_training_state(config)
    )
    appearance_optimizer.step[...] = jnp.asarray(
        initial_appearance_steps, appearance_optimizer.step[...].dtype
    )
    appearance.embeds.embedding[...] = jnp.broadcast_to(
        jnp.asarray([[1.0], [2.0], [3.0]], jnp.float32),
        (2, 3, 1),
    )
    for index in (0, 2, 4):
        appearance.color_head[index].kernel[...] = 0.0
        appearance.color_head[index].bias[...] = 0.0
        appearance.color_head[index].kernel[:, 0, 0] = 1.0
    before_embedding = np.asarray(
        appearance.embeds.embedding[...]
    ).copy()
    before_colors = np.asarray(bundles[0].colors[...]).copy()
    train_step = make_distributed_train_step(config, world_size=2)

    @nnx.vmap(in_axes=(0,) * 13, out_axes=0, axis_name="rank")
    def mapped_step(
        model,
        optimizer,
        strategy_state,
        safety_state,
        images,
        intrinsics,
        viewmats,
        key,
        sh_degree,
        current_appearance,
        current_appearance_optimizer,
        camtoworlds,
        image_ids,
    ):
        return train_step(
            model,
            optimizer,
            strategy_state,
            safety_state,
            images,
            intrinsics,
            viewmats,
            key,
            sh_degree,
            appearance_module=current_appearance,
            appearance_optimizer=current_appearance_optimizer,
            camtoworlds=camtoworlds,
            image_ids=image_ids,
        )

    images = jnp.zeros((2, 1, 4, 4, 3), jnp.float32)
    intrinsics = jnp.broadcast_to(
        jnp.asarray(
            [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
            jnp.float32,
        )[None, None],
        (2, 1, 3, 3),
    )
    camtoworlds = jnp.broadcast_to(
        jnp.eye(4, dtype=jnp.float32)[None, None], (2, 1, 4, 4)
    )
    image_ids = jnp.asarray([[0], [1]], jnp.int32)
    with mock.patch.object(
        training_module,
        "rasterization",
        _appearance_sensitive_rasterization,
    ):
        metrics = mapped_step(
            *bundles,
            images,
            intrinsics,
            camtoworlds,
            jax.random.split(jax.random.key(5), 2),
            jnp.zeros((2,), jnp.int32),
            appearance,
            appearance_optimizer,
            camtoworlds,
            image_ids,
        )

    np.testing.assert_array_equal(
        metrics["distributed_state_mismatch"],
        [initial_appearance_steps == (0, 1)] * 2,
    )
    if initial_appearance_steps == (0, 1):
        np.testing.assert_array_equal(
            appearance.embeds.embedding[...], before_embedding
        )
        np.testing.assert_array_equal(bundles[0].colors[...], before_colors)
        np.testing.assert_array_equal(bundles[1].step[...], [0, 0])
        np.testing.assert_array_equal(
            appearance_optimizer.step[...], [0, 1]
        )
        return
    np.testing.assert_array_equal(
        appearance.embeds.embedding[0], appearance.embeds.embedding[1]
    )
    assert np.any(
        np.asarray(appearance.embeds.embedding[0, :2])
        != before_embedding[0, :2]
    )
    np.testing.assert_array_equal(
        appearance.embeds.embedding[0, 2], before_embedding[0, 2]
    )
    np.testing.assert_array_equal(
        appearance_optimizer.step[...], [1, 1]
    )
    assert np.any(np.asarray(bundles[0].colors[...]) != before_colors)


def _run_two_rank_pose_update(
    map_transform, *, initial_pose_steps=(0, 0)
):
    """Two ranks optimize one replicated camera-pose module."""

    config = _topology_plan_config(refine_start=4, train={"pose_opt": True})
    bundles = _two_rank_bundles(config)
    pose_adjust = _stack_graphs(
        *[_zero_initialized_pose_module() for _ in range(2)]
    )
    pose_optimizer = _stack_graphs(
        *[
            training_module._create_pose_optimizer(
                _zero_initialized_pose_module(), config
            )
            for _ in range(2)
        ]
    )
    pose_optimizer.step[...] = jnp.asarray(
        initial_pose_steps, pose_optimizer.step[...].dtype
    )
    train_step = make_distributed_train_step(config, world_size=2)

    @map_transform(in_axes=(0,) * 13, out_axes=0, axis_name="rank")
    def mapped_step(
        model,
        optimizer,
        strategy_state,
        safety_state,
        images,
        intrinsics,
        viewmats,
        key,
        sh_degree,
        current_pose_adjust,
        current_pose_optimizer,
        camtoworlds,
        image_ids,
    ):
        return train_step(
            model,
            optimizer,
            strategy_state,
            safety_state,
            images,
            intrinsics,
            viewmats,
            key,
            sh_degree,
            pose_adjust=current_pose_adjust,
            pose_optimizer=current_pose_optimizer,
            camtoworlds=camtoworlds,
            image_ids=image_ids,
        )

    images = jnp.zeros((2, 1, 4, 4, 3), jnp.float32)
    intrinsics = jnp.broadcast_to(
        jnp.asarray(
            [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]], jnp.float32
        )[None, None],
        (2, 1, 3, 3),
    )
    camtoworlds = jnp.broadcast_to(
        jnp.eye(4, dtype=jnp.float32)[None, None], (2, 1, 4, 4)
    )
    viewmats = camtoworlds
    # Each rank owns a different training image, so only its own embedding row
    # receives a local gradient.
    image_ids = jnp.asarray([[0], [1]], jnp.int32)

    with mock.patch.object(
        training_module, "rasterization", _pose_sensitive_rasterization
    ):
        metrics = mapped_step(
            *bundles,
            images,
            intrinsics,
            viewmats,
            jax.random.split(jax.random.key(0), 2),
            jnp.zeros((2,), jnp.int32),
            pose_adjust,
            pose_optimizer,
            camtoworlds,
            image_ids,
        )

    embedding = np.asarray(pose_adjust.embeds.embedding[...])
    # DDP keeps the replicas identical and averages both rows across ranks.
    np.testing.assert_array_equal(embedding[0], embedding[1])
    return embedding, pose_optimizer.step[...], metrics


def test_named_two_rank_pose_update_averages_replicated_gradients():
    embedding, pose_steps, _ = _run_two_rank_pose_update(nnx.vmap)
    # Each row's own rank contributed the only non-zero gradient, so DDP's mean
    # halves it and both rows move by the same amount.
    assert np.any(embedding[0, 0] != 0.0)
    assert np.any(embedding[0, 1] != 0.0)
    np.testing.assert_allclose(
        np.abs(embedding[0, 0]), np.abs(embedding[0, 1]), rtol=1e-6
    )
    np.testing.assert_array_equal(pose_steps, [1, 1])


def test_distributed_pose_step_mismatch_atomically_skips_update():
    embedding, pose_steps, metrics = _run_two_rank_pose_update(
        nnx.vmap, initial_pose_steps=(0, 1)
    )

    np.testing.assert_array_equal(
        metrics["distributed_state_mismatch"], [True, True]
    )
    np.testing.assert_array_equal(embedding, 0.0)
    np.testing.assert_array_equal(pose_steps, [0, 1])


def test_distributed_train_step_still_rejects_2dgs():
    with pytest.raises(NotImplementedError, match="2DGS"):
        make_distributed_train_step(
            _topology_plan_config(
                refine_start=4, train={"model_type": "2dgs"}
            ),
            world_size=2,
        )


def test_visible_adam_updates_rows_seen_only_by_another_rank(monkeypatch):
    monkeypatch.setattr(
        training_module,
        "rasterization",
        _cross_rank_visibility_rasterization,
    )
    config = _fixed_topology_config(visible_adam=True)

    _, _, _, _, metrics = _run_two_rank_update(nnx.vmap, config=config)

    # Each owner's row is projected only by the other rank's camera, so the
    # row-selective update depends on the globally reduced visibility.
    np.testing.assert_array_equal(metrics["visible_count"], [1, 1])


def _packed_rasterization(
    means,
    _quats,
    _scales,
    _opacities,
    _colors,
    viewmats,
    _intrinsics,
    width,
    height,
    **kwargs,
):
    axis_name = kwargs["distributed_axis_name"]
    gathered_means = jax.lax.all_gather(means, axis_name, axis=0, tiled=True)
    global_capacity = gathered_means.shape[0]
    camera_count = viewmats.shape[0]
    screen_probe = kwargs["_means2d_offset"]
    assert screen_probe.shape == (camera_count, global_capacity, 2)
    signal = (
        jnp.asarray(0.25, means.dtype)
        + 1.0e-3 * jnp.sum(gathered_means)
        + jnp.sum(screen_probe)
    )
    renders = jnp.broadcast_to(signal, (camera_count, height, width, 3))
    alphas = jnp.ones((camera_count, height, width, 1), dtype=means.dtype)
    info = {
        # One packed entry per global Gaussian, addressed by gathered index.
        "camera_ids": jnp.zeros((global_capacity,), jnp.int32),
        "gaussian_ids": jnp.arange(global_capacity, dtype=jnp.int32),
        "radii": jnp.ones((global_capacity, 2), dtype=means.dtype),
        "valid": jnp.ones((global_capacity,), jnp.bool_),
        "projection_valid_count": jnp.asarray(global_capacity, jnp.int32),
        "tile_overflow": jnp.zeros((camera_count, 1, 1), jnp.bool_),
        "candidate_limit_exceeded": jnp.zeros((camera_count, 1, 1), jnp.bool_),
        "candidate_counts": jnp.zeros((camera_count, 1, 1), jnp.int32),
        "intersection_overflow": jnp.zeros((camera_count,), jnp.bool_),
        "intersection_count": jnp.ones((camera_count,), jnp.int32),
        "intersection_required_count": jnp.ones((camera_count,), jnp.int32),
    }
    return renders, alphas, info


def test_packed_metadata_unpacks_against_the_gathered_scene(monkeypatch):
    monkeypatch.setattr(
        training_module, "rasterization", _packed_rasterization
    )
    config = _fixed_topology_config(packed=True)

    _, _, strategy_state, _, metrics = _run_two_rank_update(
        nnx.vmap, config=config
    )

    np.testing.assert_array_equal(metrics["visible_count"], [1, 1])
    # Both ranks' cameras see every owner, so each owner counts two views.
    np.testing.assert_array_equal(
        strategy_state.visible_count[...], [[2.0], [2.0]]
    )
    assert np.all(np.asarray(strategy_state.grad_accum[...]) > 0.0)


def _mcmc_config(capacity: int, *, bucket: int | None = None) -> TrainConfig:
    return _fixed_topology_config(
        model=ModelConfig(
            capacity=capacity,
            bucket_min_capacity=capacity if bucket is None else bucket,
            sh_degree=0,
            initial_scale=0.2,
        ),
        strategy=StrategyConfig(
            kind="mcmc",
            refine_start=0,
            refine_stop=4,
            refine_every=1,
            reset_every=4,
            max_new_per_refine=2,
            cap_max=1000,
        ),
    )


def _mcmc_rank_bundle(config: TrainConfig, point_count: int):
    points = np.stack(
        [
            np.linspace(-0.1, 0.1, point_count, dtype=np.float32),
            np.zeros(point_count, np.float32),
            np.full(point_count, 3.0, np.float32),
        ],
        axis=1,
    )
    model = GaussianModel.from_point_cloud(
        points,
        np.full((point_count, 3), 128, np.uint8),
        config.model,
        physical_capacity=config.model.bucket_capacity(point_count),
    )
    optimizer = create_optimizer(
        model,
        config.optimizer,
        batch_size=config.data.batch_size,
        world_size=2,
        scene_scale=1.0,
    )
    strategy_state = MCMCStrategy(config.strategy).initialize_state(
        model.capacity
    )
    return model, optimizer, strategy_state, TrainingSafetyState()


def test_mcmc_capacity_overflow_is_reduced_across_ranks(monkeypatch):
    monkeypatch.setattr(
        training_module, "rasterization", _no_overflow_rasterization
    )
    config = _mcmc_config(20)
    # A full rank-0 shard cannot allocate its planned birth; the half-filled
    # rank-1 shard plans nothing and would otherwise advance alone.
    bundles = _stack_graphs(
        _mcmc_rank_bundle(config, 20), _mcmc_rank_bundle(config, 10)
    )

    _, optimizer, _, _, metrics = _run_two_rank_update(
        nnx.vmap, config=config, bundles=bundles, expect_update=False
    )

    np.testing.assert_array_equal(metrics["refine_scheduled"], [True, True])
    np.testing.assert_array_equal(
        metrics["refine_capacity_overflow"], [True, True]
    )
    np.testing.assert_array_equal(metrics["refine_planned_new_count"], [1, 1])
    np.testing.assert_array_equal(optimizer.step[...], [0, 0])


def test_mcmc_shards_train_and_grow_owner_locally(monkeypatch):
    monkeypatch.setattr(
        training_module, "rasterization", _no_overflow_rasterization
    )
    config = _mcmc_config(24)
    bundles = _stack_graphs(
        _mcmc_rank_bundle(config, 20), _mcmc_rank_bundle(config, 20)
    )

    model, optimizer, _, _, metrics = _run_two_rank_update(
        nnx.vmap, config=config, bundles=bundles
    )

    np.testing.assert_array_equal(
        metrics["refine_capacity_overflow"], [False, False]
    )
    np.testing.assert_array_equal(optimizer.step[...], [1, 1])
    # Every shard applies its own five-percent birth budget, as an independent
    # upstream rank does.
    np.testing.assert_array_equal(
        np.asarray(model.active_mask[...]).sum(axis=1), [21, 21]
    )


def test_mcmc_post_update_overflow_reports_capacity_for_host_growth(
    monkeypatch,
):
    monkeypatch.setattr(
        training_module, "rasterization", _opacity_growth_rasterization
    )
    config = _mcmc_config(40, bucket=20)
    bundles = _stack_graphs(
        _mcmc_rank_bundle(config, 20), _mcmc_rank_bundle(config, 20)
    )
    bundles[0].opacity_logits[...] = inverse_sigmoid(0.0049)

    model, optimizer, strategy_state, _, metrics = _run_two_rank_update(
        nnx.vmap, config=config, bundles=bundles
    )

    np.testing.assert_array_equal(
        metrics["refine_capacity_overflow"], [False, False]
    )
    np.testing.assert_array_equal(metrics["refine_required_capacity"], [20, 20])
    np.testing.assert_array_equal(
        metrics["refine_commit_overflow"], [True, True]
    )
    np.testing.assert_array_equal(
        metrics["refine_commit_required_capacity"], [21, 21]
    )
    np.testing.assert_array_equal(optimizer.step[...], [1, 1])
    np.testing.assert_array_equal(
        strategy_state.capacity_overflow[...], [True, True]
    )

    grown_model, _, _, decision = synchronize_distributed_capacity(
        config, model, optimizer, strategy_state, metrics
    )
    assert decision == (True, False, 20, 40)
    assert grown_model.means[...].shape == (2, 40, 3)


def test_host_capacity_synchronizer_grows_and_replays_the_frozen_step():
    # The same chain test_growing_all_shards_lets_the_skipped_step_replay
    # proves by hand, driven through the host half instead: read the step's
    # metrics, grow every shard by the bucket rule, and replay because the
    # frozen step committed nothing.
    config = _topology_plan_config(
        capacity=4,
        bucket=2,
        refine_scale2d_stop_iter=100,
        grow_scale2d=0.05,
    )
    bundles = _two_rank_bundles(config)
    _set_owner_statistics(
        bundles[2], [[1.0, 0.0], [0.0, 0.0]], [[0.5, 0.0], [0.0, 0.0]]
    )
    train_step = make_distributed_train_step(config, world_size=2)

    @nnx.vmap(
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0, axis_name="rank"
    )
    def mapped_step(*args):
        return train_step(*args)

    images = jnp.zeros((2, 1, 4, 4, 3), jnp.float32)
    intrinsics = jnp.broadcast_to(
        jnp.asarray(
            [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]], jnp.float32
        )[None, None],
        (2, 1, 3, 3),
    )
    viewmats = jnp.broadcast_to(
        jnp.eye(4, dtype=jnp.float32)[None, None], (2, 1, 4, 4)
    )
    keys = jax.random.split(jax.random.key(0), 2)
    sh_degrees = jnp.zeros((2,), jnp.int32)

    with mock.patch.object(
        training_module, "rasterization", _no_statistics_rasterization
    ):
        metrics = mapped_step(
            *bundles, images, intrinsics, viewmats, keys, sh_degrees
        )
        model, optimizer, strategy_state, decision = (
            synchronize_distributed_capacity(
                config, bundles[0], bundles[1], bundles[2], metrics
            )
        )
        assert decision.grew and decision.replay_required
        assert (decision.old_capacity, decision.new_capacity) == (2, 4)
        assert model.means[...].shape == (2, 4, 3)

        replayed = mapped_step(
            model,
            optimizer,
            strategy_state,
            bundles[3],
            images,
            intrinsics,
            viewmats,
            keys,
            sh_degrees,
        )

    np.testing.assert_array_equal(
        replayed["refine_capacity_overflow"], [False, False]
    )
    np.testing.assert_array_equal(replayed["refine_new_count"], [2, 2])
    np.testing.assert_array_equal(optimizer.step[...], [1, 1])

    # After growth nothing is pending, so the synchronizer is a no-op.
    _, _, _, after = synchronize_distributed_capacity(
        config, model, optimizer, strategy_state, replayed
    )
    assert after == (False, False, 4, 4)


def test_host_capacity_synchronizer_refuses_beyond_max_capacity():
    config = _topology_plan_config(capacity=4, bucket=2)
    bundles = _two_rank_bundles(config)
    metrics = {
        "refine_capacity_overflow": jnp.asarray([True, True]),
        "refine_required_capacity": jnp.asarray([8, 8]),
    }
    with pytest.raises(RuntimeError, match="max_capacity is 4"):
        synchronize_distributed_capacity(
            config, bundles[0], bundles[1], bundles[2], metrics
        )


def test_host_capacity_synchronizer_grows_without_replay_after_commit_overflow():
    # A commit overflow means the step already committed what fit, so the
    # host only has to grow before the next refine.
    config = _topology_plan_config(capacity=4, bucket=2)
    bundles = _two_rank_bundles(config)
    metrics = {
        "refine_capacity_overflow": jnp.asarray([False, False]),
        "refine_commit_overflow": jnp.asarray([True, False]),
        "refine_required_capacity": jnp.asarray([4, 4]),
    }
    model, optimizer, strategy_state, decision = (
        synchronize_distributed_capacity(
            config, bundles[0], bundles[1], bundles[2], metrics
        )
    )
    assert decision == (True, False, 2, 4)
    assert model.means[...].shape == (2, 4, 3)
    assert optimizer.step[...].shape == (2,)
    assert strategy_state.grad_accum[...].shape == (2, 4)


def test_host_capacity_synchronizer_checks_devices_and_blocks_resize(
    monkeypatch,
):
    config = _topology_plan_config(capacity=4, bucket=2)
    bundles = _two_rank_bundles(config)
    devices = (object(), object())
    observed = {}

    def fake_memory_check(*args, **kwargs):
        observed["devices"] = kwargs["devices"]
        return 0

    def fake_block(*nodes):
        observed["blocked"] = nodes

    monkeypatch.setattr(
        training_module,
        "_check_distributed_bucket_transition_memory_budget",
        fake_memory_check,
    )
    monkeypatch.setattr(training_module, "_block_nnx_state", fake_block)
    metrics = {
        "refine_capacity_overflow": jnp.asarray([False, False]),
        "refine_commit_overflow": jnp.asarray([True, False]),
        "refine_required_capacity": jnp.asarray([4, 4]),
    }

    model, optimizer, strategy_state, decision = (
        synchronize_distributed_capacity(
            config,
            bundles[0],
            bundles[1],
            bundles[2],
            metrics,
            devices=devices,
            resize_step=lambda model, optimizer, strategy_state, capacity: (
                resize_distributed_training_state(
                    model,
                    optimizer,
                    strategy_state,
                    capacity,
                    config.model,
                    config.optimizer,
                )
            ),
        )
    )

    assert decision == (True, False, 2, 4)
    assert observed["devices"] == devices
    assert all(
        blocked is returned
        for blocked, returned in zip(
            observed["blocked"],
            (model, optimizer, strategy_state),
            strict=True,
        )
    )


def test_distributed_memory_preflight_preserves_host_stacked_world_cost(
    monkeypatch,
):
    config = _topology_plan_config(capacity=4, bucket=2)
    monkeypatch.setattr(
        training_module,
        "_device_memory_usage",
        lambda device=None: (0, 0),
    )
    per_device = training_module.estimate_bucket_transition_memory_bytes(
        config,
        2,
        4,
        render_capacity=8,
        image_height=4,
        image_width=4,
    )

    mapped = (
        training_module._check_distributed_bucket_transition_memory_budget(
            config,
            2,
            2,
            4,
            devices=(object(), object()),
            image_height=4,
            image_width=4,
        )
    )
    host_stacked = (
        training_module._check_distributed_bucket_transition_memory_budget(
            config,
            2,
            2,
            4,
            image_height=4,
            image_width=4,
        )
    )

    assert mapped == per_device
    assert host_stacked == 2 * per_device


def test_shard_camera_batch_deals_cameras_to_ranks_in_order():
    batch = {
        "image": np.arange(4 * 2 * 2 * 3, dtype=np.float32).reshape(4, 2, 2, 3),
        "K": np.tile(np.eye(3, dtype=np.float32), (4, 1, 1)),
        "image_id": np.asarray([10, 11, 12, 13], dtype=np.int64),
        "image_name": ["a", "b", "c", "d"],
    }
    sharded = shard_camera_batch(batch, 2)
    assert sharded["image"].shape == (2, 2, 2, 2, 3)
    np.testing.assert_array_equal(sharded["image_id"], [[10, 11], [12, 13]])
    assert sharded["image_name"].tolist() == [["a", "b"], ["c", "d"]]
    np.testing.assert_array_equal(
        sharded["image"][1, 0], batch["image"][2]
    )

    with pytest.raises(ValueError, match="does not split"):
        shard_camera_batch({"image": np.zeros((3, 2, 2, 3))}, 2)
    with pytest.raises(ValueError, match="while other fields carry"):
        shard_camera_batch(
            {"image": np.zeros((4, 2)), "K": np.zeros((2, 3))}, 2
        )


def test_distributed_initialization_uses_global_knn_before_owner_deal():
    config = _topology_plan_config(capacity=8, bucket=4)
    points = np.asarray(
        [
            [-0.4, 0.0, 3.0],
            [-0.1, 0.0, 3.0],
            [0.0, 0.0, 3.0],
            [0.2, 0.0, 3.0],
            [0.8, 0.0, 3.0],
        ],
        np.float32,
    )
    colors = np.full((len(points), 3), 128, np.uint8)

    model, optimizer, _, _ = _initialize_distributed_training_state(
        config, points, colors, world_size=2, scene_scale=1.0
    )
    assert model.means[...].shape == (2, 4, 3)
    np.testing.assert_array_equal(
        np.count_nonzero(np.asarray(model.active_mask[...]), axis=1), [3, 2]
    )
    assert optimizer._jax_gs_world_size == 2

    whole = GaussianModel.from_point_cloud(points, colors, config.model)
    for rank in range(2):
        indices = np.arange(rank, len(points), 2)
        np.testing.assert_allclose(
            np.asarray(model.log_scales[...])[rank, : len(indices)],
            np.asarray(whole.log_scales[...])[indices],
            rtol=0.0,
            atol=0.0,
        )


def test_sharded_camera_batch_drives_the_mapped_step_per_rank():
    config = _topology_plan_config(capacity=4, bucket=2)
    bundles = _two_rank_bundles(config)
    train_step = make_distributed_train_step(config, world_size=2)

    @nnx.vmap(
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0, axis_name="rank"
    )
    def mapped_step(*args):
        return train_step(*args)

    # One host stream of world_size * B cameras; ranks must see their own.
    host_batch = {
        "image": np.stack(
            (
                np.zeros((4, 4, 3), np.float32),
                np.ones((4, 4, 3), np.float32),
            )
        ),
        "K": np.tile(
            np.asarray(
                [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
                np.float32,
            ),
            (2, 1, 1),
        ),
        "w2c": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
    }
    sharded = shard_camera_batch(host_batch, 2)
    keys = jax.random.split(jax.random.key(0), 2)
    sh_degrees = jnp.zeros((2,), jnp.int32)

    with mock.patch.object(
        training_module, "rasterization", _no_statistics_rasterization
    ):
        metrics = mapped_step(
            *bundles,
            jnp.asarray(sharded["image"]),
            jnp.asarray(sharded["K"]),
            jnp.asarray(sharded["w2c"]),
            keys,
            sh_degrees,
        )

    # Rank 0 scored a black target and rank 1 a white one, so the rank-local
    # losses must differ: each rank consumed its own cameras.
    losses = np.asarray(metrics["l1"])
    assert losses.shape == (2,)
    assert abs(float(losses[0]) - float(losses[1])) > 1e-3
    np.testing.assert_array_equal(bundles[1].step[...], [1, 1])


def test_distributed_render_matches_the_single_process_render():
    # Evaluation cameras are replicated, not sharded: every rank renders the
    # same camera against the gathered scene, so all ranks must agree and the
    # result must be what one process rendering the whole scene produces.
    config = _topology_plan_config(capacity=4, bucket=2)
    bundles = _two_rank_bundles(config)
    model = bundles[0]

    render_step = make_distributed_render_step(
        config, 4, 4, world_size=2, axis_name="rank"
    )

    @nnx.vmap(in_axes=(0, None, None, None), out_axes=0, axis_name="rank")
    def mapped_render(shard, viewmat, K, sh_degree):
        return render_step(shard, viewmat, K, sh_degree)

    viewmat = jnp.eye(4, dtype=jnp.float32)
    K = jnp.asarray(
        [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]], jnp.float32
    )
    sh_degree = jnp.asarray(0, jnp.int32)

    rendered, _, _, _ = mapped_render(model, viewmat, K, sh_degree)
    assert rendered.shape == (2, 4, 4, 3)

    # The ranks agree, so the reducer accepts and returns rank 0's image.
    image = reduce_distributed_render(rendered)
    assert image.shape == (4, 4, 3)
    np.testing.assert_array_equal(np.asarray(image), np.asarray(rendered[0]))

    # And that image is the whole scene, not one shard: build the equivalent
    # single-process model by concatenating both shards' rows.
    single = _single_process_model_from_shards(config, model)
    single_render = make_render_step(config, 4, 4)
    reference, _, _, _ = single_render(single, viewmat, K, sh_degree)
    np.testing.assert_allclose(
        np.asarray(image), np.asarray(reference), rtol=0.0, atol=2e-6
    )


def test_reduce_distributed_render_rejects_disagreeing_ranks():
    stacked = jnp.stack(
        (jnp.zeros((2, 2, 3), jnp.float32), jnp.ones((2, 2, 3), jnp.float32))
    )
    with pytest.raises(ValueError, match="disagrees between rank 0 and rank 1"):
        reduce_distributed_render(stacked)
    with pytest.raises(IndexError, match="outside the world"):
        reduce_distributed_render(stacked, rank=2)
    # Rank-selection works, and a tolerance admits reassociation-level drift.
    near = jnp.stack(
        (
            jnp.zeros((2, 2, 3), jnp.float32),
            jnp.full((2, 2, 3), 1e-9, jnp.float32),
        )
    )
    np.testing.assert_array_equal(
        np.asarray(reduce_distributed_render(near, atol=1e-6)),
        np.zeros((2, 2, 3), np.float32),
    )


def test_distributed_render_step_rejects_the_unsupported_combinations():
    config = _topology_plan_config(capacity=4, bucket=2)
    with pytest.raises(ValueError, match="world_size must be greater"):
        make_distributed_render_step(config, 4, 4, world_size=1)
    with pytest.raises(NotImplementedError, match="2DGS"):
        make_distributed_render_step(
            replace(config, model_type="2dgs"), 4, 4, world_size=2
        )
    with pytest.raises(NotImplementedError, match="pinhole"):
        make_distributed_render_step(
            replace(config, with_ut=True), 4, 4, world_size=2
        )


def test_distributed_appearance_render_uses_the_global_scene(monkeypatch):
    config = _topology_plan_config(
        capacity=4,
        bucket=2,
        train={"app_opt": True, "app_embed_dim": 1},
    )
    bundles = _two_rank_bundles(config)
    appearance, _ = _replicated_appearance_training_state(config)
    render_step = make_distributed_render_step(
        config, 4, 4, world_size=2, axis_name="rank"
    )

    @nnx.vmap(
        in_axes=(0, 0, None, None, None),
        out_axes=0,
        axis_name="rank",
    )
    def mapped_render(model, module, viewmat, K, sh_degree):
        return render_step(
            model,
            viewmat,
            K,
            sh_degree,
            appearance_module=module,
        )

    monkeypatch.setattr(
        training_module,
        "rasterization",
        _appearance_sensitive_rasterization,
    )
    viewmat = jnp.eye(4, dtype=jnp.float32)
    K = jnp.asarray(
        [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
        jnp.float32,
    )
    rendered, _, _, _ = mapped_render(
        bundles[0], appearance, viewmat, K, jnp.asarray(0, jnp.int32)
    )

    np.testing.assert_allclose(rendered[0], rendered[1], rtol=1e-6)


def test_distributed_appearance_render_runs_the_reference_rasterizer():
    config = _topology_plan_config(
        capacity=4,
        bucket=2,
        train={"app_opt": True, "app_embed_dim": 1},
    )
    bundles = _two_rank_bundles(config)
    appearance, _ = _replicated_appearance_training_state(config)
    render_step = make_distributed_render_step(
        config, 4, 4, world_size=2, axis_name="rank"
    )

    @nnx.vmap(
        in_axes=(0, 0, None, None, None),
        out_axes=0,
        axis_name="rank",
    )
    def mapped_render(model, module, viewmat, K, sh_degree):
        return render_step(
            model,
            viewmat,
            K,
            sh_degree,
            appearance_module=module,
        )

    viewmat = jnp.eye(4, dtype=jnp.float32)
    K = jnp.asarray(
        [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
        jnp.float32,
    )
    rendered, _, overflow, intersection_overflow = mapped_render(
        bundles[0], appearance, viewmat, K, jnp.asarray(0, jnp.int32)
    )

    assert rendered.shape == (2, 4, 4, 3)
    assert bool(jnp.all(jnp.isfinite(rendered)))
    np.testing.assert_allclose(rendered[0], rendered[1], rtol=1e-6)
    assert not bool(jnp.any(overflow))
    assert not bool(jnp.any(intersection_overflow))


def _global_rows(model):
    """Every active row of a stacked world, in rank order."""

    active = np.asarray(model.active_mask[...])
    means = np.asarray(model.means[...])
    return np.concatenate(
        [means[rank][active[rank]] for rank in range(active.shape[0])], axis=0
    )


def _reshard(config, bundles, world_size, **kwargs):
    return reshard_distributed_training_state(
        bundles[0],
        bundles[1],
        bundles[2],
        bundles[3],
        world_size,
        config.model,
        config.optimizer,
        **kwargs,
    )


def test_resharding_preserves_every_gaussian_and_its_state():
    config = _topology_plan_config(capacity=8, bucket=4)
    bundles = _two_rank_bundles(config)
    model, _, strategy_state, _ = bundles
    # Four Gaussians over two shards, each carrying distinct statistics.
    model.active_mask[...] = jnp.asarray(
        [[True, True, False, False], [True, True, False, False]]
    )
    model.means[...] = jnp.asarray(
        [
            [[0.0, 0, 3], [1.0, 0, 3], [0, 0, 0], [0, 0, 0]],
            [[2.0, 0, 3], [3.0, 0, 3], [0, 0, 0], [0, 0, 0]],
        ],
        jnp.float32,
    )
    _set_owner_statistics(
        strategy_state,
        [[10.0, 11.0, 0.0, 0.0], [12.0, 13.0, 0.0, 0.0]],
        [[0.1, 0.2, 0.0, 0.0], [0.3, 0.4, 0.0, 0.0]],
    )
    before = _global_rows(model)
    assert before.shape == (4, 3)

    grad_before = np.asarray(strategy_state.grad_accum[...])
    owned = {
        float(before[i, 0]): float(
            grad_before[i // 2, i % 2]
        )
        for i in range(4)
    }

    wide = _reshard(config, bundles, 4)
    assert wide[0].means[...].shape[0] == 4
    after = _global_rows(wide[0])
    # Every Gaussian survives, and contiguous blocks keep the global order.
    np.testing.assert_array_equal(after, before)

    # Each Gaussian's statistics travelled with its row, not with its rank.
    active = np.asarray(wide[0].active_mask[...])
    means = np.asarray(wide[0].means[...])
    grads = np.asarray(wide[2].grad_accum[...])
    for rank in range(4):
        for row in np.flatnonzero(active[rank]):
            assert grads[rank, row] == owned[float(means[rank, row, 0])]


@pytest.mark.parametrize("visible_adam", [False, True])
def test_resharded_optimizer_uses_target_world_contract_and_can_train(
    monkeypatch, visible_adam
):
    config = _topology_plan_config(
        capacity=8,
        bucket=4,
        refine_start=3,
        train={"visible_adam": visible_adam},
    )
    actual = _reshard(config, _two_rank_bundles(config), 4)
    expected = _reshard(config, _two_rank_bundles(config), 4)
    optimizer_factory = (
        create_visible_adam_optimizer if visible_adam else create_optimizer
    )
    expected_optimizer = _stack_graphs(
        *[
            optimizer_factory(
                _unstack_graph(expected[0], rank),
                config.optimizer,
                batch_size=config.data.batch_size,
                world_size=4,
            )
            for rank in range(4)
        ]
    )
    expected = (expected[0], expected_optimizer, expected[2], expected[3])
    train_step = make_distributed_train_step(config, world_size=4)

    @nnx.vmap(
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
        axis_name="rank",
    )
    def mapped_step(*args):
        return train_step(*args)

    images = jnp.zeros((4, 1, 4, 4, 3), jnp.float32)
    intrinsics = jnp.broadcast_to(
        jnp.asarray(
            [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
            jnp.float32,
        )[None, None],
        (4, 1, 3, 3),
    )
    viewmats = jnp.broadcast_to(
        jnp.eye(4, dtype=jnp.float32)[None, None], (4, 1, 4, 4)
    )
    keys = jax.random.split(jax.random.key(0), 4)
    sh_degrees = jnp.zeros((4,), jnp.int32)

    monkeypatch.setattr(
        training_module, "rasterization", _no_overflow_rasterization
    )
    actual_metrics = mapped_step(
        *actual, images, intrinsics, viewmats, keys, sh_degrees
    )
    expected_metrics = mapped_step(
        *expected, images, intrinsics, viewmats, keys, sh_degrees
    )

    assert getattr(actual[1], "_jax_gs_world_size") == 4
    np.testing.assert_array_equal(actual[1].step[...], [1, 1, 1, 1])
    np.testing.assert_allclose(
        actual_metrics["loss"], expected_metrics["loss"], rtol=1e-6
    )
    np.testing.assert_allclose(
        actual[0].means[...], expected[0].means[...], rtol=1e-6
    )
    np.testing.assert_allclose(
        _adam_means_moments(actual[1]),
        _adam_means_moments(expected[1]),
        rtol=1e-6,
    )


def test_resharding_rejects_divergent_source_optimizer_steps():
    config = _topology_plan_config(capacity=8, bucket=4)
    bundles = _two_rank_bundles(config)
    bundles[1].step[...] = jnp.asarray([2, 3], bundles[1].step[...].dtype)

    with pytest.raises(ValueError, match="optimizer step"):
        _reshard(config, bundles, 4)


def test_resharding_round_trips_back_to_the_original_assignment():
    # Contiguous blocks preserve the global sequence, so resharding composes:
    # widening and narrowing again restores each rank's own rows.
    config = _topology_plan_config(capacity=8, bucket=4)
    bundles = _two_rank_bundles(config)
    model = bundles[0]
    model.active_mask[...] = jnp.asarray(
        [[True, True, False, False], [True, True, False, False]]
    )
    model.means[...] = jnp.asarray(
        [
            [[0.0, 0, 3], [1.0, 0, 3], [0, 0, 0], [0, 0, 0]],
            [[2.0, 0, 3], [3.0, 0, 3], [0, 0, 0], [0, 0, 0]],
        ],
        jnp.float32,
    )
    before = _global_rows(model)

    wide = _reshard(config, bundles, 4)
    narrow = _reshard(config, wide, 2, local_capacity=4)
    np.testing.assert_array_equal(_global_rows(narrow[0]), before)


def test_resharding_reduces_sticky_overflow_rather_than_dropping_it():
    config = _topology_plan_config(capacity=8, bucket=4)
    bundles = _two_rank_bundles(config)
    safety = bundles[3]
    # Only one rank saw the overflow; the world has still seen it.
    safety.intersection_overflow_seen[...] = jnp.asarray([False, True])
    safety.max_overflow_tiles[...] = jnp.asarray([3, 7], jnp.int32)

    resharded = _reshard(config, bundles, 4)
    np.testing.assert_array_equal(
        resharded[3].intersection_overflow_seen[...], [True] * 4
    )
    np.testing.assert_array_equal(
        resharded[3].max_overflow_tiles[...], [7] * 4
    )


def test_resharding_rejects_a_capacity_that_cannot_hold_a_shard():
    config = _topology_plan_config(capacity=8, bucket=4)
    bundles = _two_rank_bundles(config)
    bundles[0].active_mask[...] = jnp.asarray(
        [[True, True, True, True], [True, True, True, True]]
    )
    with pytest.raises(ValueError, match="cannot hold the busiest"):
        _reshard(config, bundles, 2, local_capacity=1)
    with pytest.raises(ValueError, match="world_size must be positive"):
        _reshard(config, bundles, 0)


def test_resharding_spreads_an_uneven_split_across_the_lowest_ranks():
    config = _topology_plan_config(capacity=8, bucket=4)
    bundles = _two_rank_bundles(config)
    model = bundles[0]
    # Five Gaussians over two shards: 3 and 2.
    model.active_mask[...] = jnp.asarray(
        [[True, True, True, False], [True, True, False, False]]
    )
    model.means[...] = jnp.asarray(
        [
            [[0.0, 0, 3], [1.0, 0, 3], [2.0, 0, 3], [0, 0, 0]],
            [[3.0, 0, 3], [4.0, 0, 3], [0, 0, 0], [0, 0, 0]],
        ],
        jnp.float32,
    )
    before = _global_rows(model)
    assert before.shape[0] == 5

    resharded = _reshard(config, bundles, 3, local_capacity=4)
    counts = np.asarray(
        jnp.count_nonzero(resharded[0].active_mask[...], axis=1)
    )
    # Five across three ranks: sizes differ by at most one, remainder low.
    np.testing.assert_array_equal(counts, [2, 2, 1])
    np.testing.assert_array_equal(_global_rows(resharded[0]), before)

    # Narrowing to one rank puts the whole scene on it, still in order.
    single = _reshard(config, resharded, 1, local_capacity=8)
    np.testing.assert_array_equal(
        np.asarray(jnp.count_nonzero(single[0].active_mask[...], axis=1)), [5]
    )
    np.testing.assert_array_equal(_global_rows(single[0]), before)


def test_distributed_restore_reshards_when_asked(tmp_path):
    config = _topology_plan_config(capacity=8, bucket=4)
    saved = _two_rank_bundles(config)
    model, optimizer, strategy_state, _ = saved
    optimizer.step[...] = jnp.asarray([5, 5], optimizer.step[...].dtype)
    model.active_mask[...] = jnp.asarray(
        [[True, True, False, False], [True, True, False, False]]
    )
    model.means[...] = jnp.asarray(
        [
            [[0.0, 0, 3], [1.0, 0, 3], [0, 0, 0], [0, 0, 0]],
            [[2.0, 0, 3], [3.0, 0, 3], [0, 0, 0], [0, 0, 0]],
        ],
        jnp.float32,
    )
    _set_owner_statistics(
        strategy_state,
        [[10.0, 11.0, 0.0, 0.0], [12.0, 13.0, 0.0, 0.0]],
        [[0.1, 0.2, 0.0, 0.0], [0.3, 0.4, 0.0, 0.0]],
    )
    before = _global_rows(model)
    path = save_distributed_checkpoint(tmp_path, *saved, step=5, config=config)

    # A four-rank target must already own the target optimizer graph/tx;
    # checkpoint state updates cannot replace static NNX graph attributes.
    wrong_target = _stack_graphs(
        *[_rank_bundle(config, 0.0) for _ in range(4)]
    )
    with pytest.raises(ValueError, match="target optimizer.*world_size=4"):
        restore_distributed_checkpoint(
            path,
            *wrong_target,
            config=config,
            model_config=config.model,
            optimizer_config=config.optimizer,
            allow_reshard=True,
        )

    wrong_batch_target = _stack_graphs(
        *[
            _rank_bundle(
                config,
                0.0,
                optimizer_batch_size=2,
                optimizer_world_size=4,
            )
            for _ in range(4)
        ]
    )
    with pytest.raises(ValueError, match="target optimizer.*batch_size=1"):
        restore_distributed_checkpoint(
            path,
            *wrong_batch_target,
            config=config,
            model_config=config.model,
            optimizer_config=config.optimizer,
            allow_reshard=True,
        )

    wrong_kind_ranks = []
    for _ in range(4):
        rank = _rank_bundle(config, 0.0, optimizer_world_size=4)
        wrong_kind_ranks.append(
            (
                rank[0],
                create_visible_adam_optimizer(
                    rank[0],
                    config.optimizer,
                    batch_size=config.data.batch_size,
                    world_size=4,
                ),
                rank[2],
                rank[3],
            )
        )
    wrong_kind_target = _stack_graphs(*wrong_kind_ranks)
    with pytest.raises(ValueError, match="target optimizer.*kind='adam'"):
        restore_distributed_checkpoint(
            path,
            *wrong_kind_target,
            config=config,
            model_config=config.model,
            optimizer_config=config.optimizer,
            allow_reshard=True,
        )

    # A correctly constructed four-rank world reads a two-rank checkpoint.
    target = _stack_graphs(
        *[
            _rank_bundle(config, 0.0, optimizer_world_size=4)
            for _ in range(4)
        ]
    )
    step = restore_distributed_checkpoint(
        path,
        *target,
        config=config,
        model_config=config.model,
        optimizer_config=config.optimizer,
        allow_reshard=True,
    )
    assert step == 5
    assert target[0].means[...].shape[0] == 4
    np.testing.assert_array_equal(_global_rows(target[0]), before)
    # The optimizer step survived the move, on every new rank.
    np.testing.assert_array_equal(target[1].step[...], [5, 5, 5, 5])

    # Reading it back at the saved width restores the original assignment.
    narrowed = _stack_graphs(*[_rank_bundle(config, 0.0) for _ in range(2)])
    restore_distributed_checkpoint(
        path,
        *narrowed,
        config=config,
        model_config=config.model,
        optimizer_config=config.optimizer,
        allow_reshard=True,
    )
    np.testing.assert_array_equal(
        np.asarray(narrowed[0].means[...]), np.asarray(model.means[...])
    )


def test_distributed_restore_refuses_a_target_that_cannot_hold_the_scene(
    tmp_path,
):
    config = _topology_plan_config(capacity=8, bucket=4)
    saved = _two_rank_bundles(config)
    saved[0].active_mask[...] = jnp.asarray(
        [[True, True, True, True], [True, True, True, True]]
    )
    path = save_distributed_checkpoint(tmp_path, *saved, step=0, config=config)

    # Eight Gaussians cannot fit one shard of capacity four.
    target = _stack_graphs(
        _rank_bundle(config, 0.0, optimizer_world_size=1)
    )
    with pytest.raises(ValueError, match="cannot hold the busiest"):
        restore_distributed_checkpoint(
            path,
            *target,
            config=config,
            model_config=config.model,
            optimizer_config=config.optimizer,
            allow_reshard=True,
        )
