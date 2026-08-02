from dataclasses import replace
import os
import subprocess
import sys
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
from jax_gs.optimizers import create_optimizer
from jax_gs.strategy import DefaultStrategy, MCMCStrategy
from jax_gs.training import (
    TrainingSafetyState,
    make_distributed_render_step,
    make_distributed_train_step,
    make_render_step,
    reduce_distributed_render,
    shard_camera_batch,
    synchronize_distributed_capacity,
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
    optimizer_world_size: int = 2,
    optimizer_scene_scale: float = 1.0,
):
    model = GaussianModel.from_point_cloud(
        np.asarray([[x, 0.0, 3.0]], np.float32),
        np.asarray([[192, 128, 64]], np.uint8),
        config.model,
    )
    optimizer = create_optimizer(
        model,
        config.optimizer,
        batch_size=config.data.batch_size,
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
        np.asarray(leaf).copy()
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
    assert kwargs["distributed_world_size"] == 2
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
    with pytest.raises(NotImplementedError, match="appearance"):
        make_distributed_train_step(
            _fixed_topology_config(app_opt=True), world_size=2
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
        model, _, strategy_state, _, metrics = _run_two_rank_update(
            nnx.vmap, config=config, prepare=prepare
        )

    np.testing.assert_array_equal(metrics["refine_planned_new_count"], [1, 1])
    np.testing.assert_array_equal(
        metrics["refine_capacity_overflow"], [False, False]
    )
    np.testing.assert_array_equal(
        metrics["refine_commit_overflow"], [True, True]
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

    path = save_distributed_checkpoint(
        tmp_path, *saved, step=3, config=config
    )

    manifest = load_distributed_checkpoint_manifest(path)
    assert manifest["world_size"] == 2
    assert manifest["local_capacity"] == 2
    assert manifest["global_capacity"] == 4
    assert manifest["active_counts"] == [2, 1]
    assert manifest["active_prefix"] == [True, True]
    assert manifest["components"] == [
        "model",
        "optimizer",
        "strategy",
        "safety",
    ]

    restored = _two_rank_bundles(config)
    assert restore_distributed_checkpoint(path, *restored, config=config) == 3
    for original, target in zip(saved, restored, strict=True):
        for before, after in zip(
            _snapshot_graph_arrays(original),
            _snapshot_graph_arrays(target),
            strict=True,
        ):
            np.testing.assert_array_equal(after, before)


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


def _run_two_rank_pose_update(map_transform):
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
        mapped_step(
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
    assert np.any(embedding[0, 0] != 0.0)
    assert np.any(embedding[0, 1] != 0.0)
    np.testing.assert_array_equal(pose_optimizer.step[...], [1, 1])
    return embedding


def test_named_two_rank_pose_update_averages_replicated_gradients():
    embedding = _run_two_rank_pose_update(nnx.vmap)
    # Each row's own rank contributed the only non-zero gradient, so DDP's mean
    # halves it and both rows move by the same amount.
    np.testing.assert_allclose(
        np.abs(embedding[0, 0]), np.abs(embedding[0, 1]), rtol=1e-6
    )


def test_distributed_train_step_still_rejects_appearance_and_2dgs():
    with pytest.raises(NotImplementedError, match="appearance"):
        make_distributed_train_step(
            _topology_plan_config(refine_start=4, train={"app_opt": True}),
            world_size=2,
        )
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


def _mcmc_config(capacity: int) -> TrainConfig:
    return _fixed_topology_config(
        model=ModelConfig(
            capacity=capacity,
            bucket_min_capacity=capacity,
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
        physical_capacity=config.model.capacity,
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
    with pytest.raises(NotImplementedError, match="appearance"):
        make_distributed_render_step(
            replace(config, app_opt=True), 4, 4, world_size=2
        )
    with pytest.raises(NotImplementedError, match="pinhole"):
        make_distributed_render_step(
            replace(config, with_ut=True), 4, 4, world_size=2
        )


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

    # A four-rank world reads a two-rank checkpoint.
    target = _stack_graphs(*[_rank_bundle(config, 0.0) for _ in range(4)])
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
    target = _stack_graphs(*[_rank_bundle(config, 0.0) for _ in range(1)])
    with pytest.raises(ValueError, match="cannot hold the busiest"):
        restore_distributed_checkpoint(
            path,
            *target,
            config=config,
            model_config=config.model,
            optimizer_config=config.optimizer,
            allow_reshard=True,
        )
