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
from jax_gs.strategy import DefaultStrategy
from jax_gs.training import (
    TrainingSafetyState,
    make_distributed_train_step,
)


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
    *, capacity: int = 2, **strategy_overrides
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
            bucket_min_capacity=capacity,
            sh_degree=0,
            initial_scale=0.2,
        ),
        strategy=StrategyConfig(**strategy),
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
            "_run_two_rank_screen_stats, _run_two_rank_growth_commit",
            "_run_two_rank_screen_stats(nnx.pmap)",
            "_run_two_rank_growth_commit(nnx.pmap)",
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

    with pytest.raises(ValueError, match="resharding is not supported"):
        restore_distributed_checkpoint(path, *three_ranks)


def test_distributed_restore_rejects_a_different_shard_capacity(tmp_path):
    path = save_distributed_checkpoint(
        tmp_path,
        *_two_rank_bundles(_topology_plan_config(capacity=2)),
        step=0,
    )

    with pytest.raises(ValueError, match="shard capacity"):
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
