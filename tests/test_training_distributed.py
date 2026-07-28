import os
import subprocess
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_gs.training as training_module
from jax_gs.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    RasterizationConfig,
    StrategyConfig,
    TrainConfig,
)
from jax_gs.model import GaussianModel
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
    with pytest.raises(NotImplementedError, match="fixed topology"):
        make_distributed_train_step(
            _fixed_topology_config(
                strategy=StrategyConfig(
                    refine_start=1,
                    reset_every=4,
                    max_new_per_refine=1,
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


def _run_two_rank_update(
    map_transform,
    *,
    optimizer_world_size: int = 2,
    optimizer_scene_scale: float = 1.0,
    train_scene_scale: float = 1.0,
    initial_optimizer_steps=(0, 0),
    sh_degrees=(0, 0),
    expect_update: bool = True,
):
    config = _fixed_topology_config()
    bundles = _stack_graphs(
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
    model, optimizer, strategy_state, safety_state = bundles
    optimizer.step[...] = jnp.asarray(
        initial_optimizer_steps, dtype=optimizer.step[...].dtype
    )
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
    np.testing.assert_array_equal(strategy_state.grad_accum[...], 0.0)
    np.testing.assert_array_equal(strategy_state.visible_count[...], 0.0)
    return model, optimizer, strategy_state, safety_state, metrics


def test_named_two_rank_train_step_updates_shards_with_global_rendering():
    _run_two_rank_update(nnx.vmap)


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
