from dataclasses import replace
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import grain
import numpy as np
import pytest

import jax_gs.config as config_module
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
from jax_gs.strategy import (
    build_densification_stats,
    DefaultStrategy,
    MCMCStrategy,
)
from jax_gs.training import (
    _check_evaluation_memory_budget,
    _grow_training_state,
    _grain_iter_dataset,
    _initial_storage_capacity,
    estimate_bucket_transition_memory_bytes,
    estimate_training_memory_bytes,
    make_render_step,
    make_train_step,
    TrainingSafetyState,
)
from jax_gs.training.pose import CameraOptModule
from jax_gs.training.appearance import AppearanceOptModule


def test_sample_patches_none_preserves_rectangular_images_and_intrinsics():
    images = jnp.arange(2 * 5 * 9 * 3, dtype=jnp.float32).reshape(2, 5, 9, 3)
    intrinsics = jnp.asarray(
        [
            [[10.0, 0.0, 4.0], [0.0, 11.0, 2.0], [0.0, 0.0, 1.0]],
            [[12.0, 0.0, 3.0], [0.0, 13.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=jnp.float32,
    )

    sampled, adjusted = training_module._sample_patches(
        images, intrinsics, jax.random.key(0), None
    )

    np.testing.assert_array_equal(sampled, images)
    np.testing.assert_array_equal(adjusted, intrinsics)


def test_sample_patches_integer_keeps_random_square_crop_behavior():
    images = jnp.arange(6 * 9 * 3, dtype=jnp.float32).reshape(1, 6, 9, 3)
    intrinsics = jnp.asarray(
        [[[10.0, 0.0, 4.0], [0.0, 11.0, 3.0], [0.0, 0.0, 1.0]]],
        dtype=jnp.float32,
    )

    sampled, adjusted = training_module._sample_patches(
        images, intrinsics, jax.random.key(7), 4
    )
    x = int(intrinsics[0, 0, 2] - adjusted[0, 0, 2])
    y = int(intrinsics[0, 1, 2] - adjusted[0, 1, 2])

    assert sampled.shape == (1, 4, 4, 3)
    np.testing.assert_array_equal(sampled[0], images[0, y : y + 4, x : x + 4])


def test_synthetic_train_step_updates_optimizer_and_metrics():
    points = np.array([[0, 0, 3], [0.2, 0, 3], [-0.2, 0.1, 3]], np.float32)
    model_config = ModelConfig(capacity=32, sh_degree=1, initial_scale=0.1)
    optimizer_config = OptimizerConfig(max_steps=10)
    strategy_config = StrategyConfig(refine_start=100, max_new_per_refine=4)
    config = TrainConfig(
        model=model_config,
        optimizer=optimizer_config,
        strategy=strategy_config,
        data=DataConfig(root="unused", patch_size=16, batch_size=1),
        rasterizer=RasterizationConfig(
            tile_size=8, max_gaussians_per_tile=1, tile_batch_size=2
        ),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        points, np.eye(3, dtype=np.float32), model_config
    )
    optimizer = create_optimizer(model, optimizer_config)
    strategy_state = DefaultStrategy(strategy_config).initialize_state(32)
    safety_state = TrainingSafetyState()
    step = make_train_step(config)
    metrics = step(
        model,
        optimizer,
        strategy_state,
        safety_state,
        jnp.zeros((1, 32, 32, 3), jnp.float32),
        jnp.array([[[30.0, 0, 16], [0, 30.0, 16], [0, 0, 1]]], jnp.float32),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(1),
    )
    assert int(optimizer.step[...]) == 1
    assert int(metrics["active_count"]) == 3
    assert int(metrics["candidate_limit_exceeded_tiles"]) > 0
    assert int(safety_state.max_overflow_tiles[...]) == 0
    assert not bool(safety_state.intersection_overflow_seen[...])
    assert jnp.isfinite(metrics["loss"])
    assert jnp.isfinite(metrics["psnr"])
    assert float(metrics["normal_loss"]) == 0.0
    assert float(metrics["distortion_loss"]) == 0.0
    assert float(metrics["opacity_reg_loss"]) == 0.0
    assert float(metrics["scale_reg_loss"]) == 0.0


@pytest.mark.parametrize("absgrad", [False, True])
def test_train_step_accumulates_screen_space_densification_stats(absgrad):
    points = np.array([[0, 0, 3], [0.2, 0, 3], [-0.2, 0.1, 3]], np.float32)
    point_colors = np.array(
        [[255, 32, 32], [32, 255, 32], [32, 32, 255]], np.uint8
    )
    config = TrainConfig(
        model=ModelConfig(
            capacity=8,
            bucket_min_capacity=8,
            sh_degree=0,
            initial_scale=0.1,
        ),
        optimizer=OptimizerConfig(max_steps=10),
        strategy=StrategyConfig(
            refine_start=100,
            max_new_per_refine=2,
            absgrad=absgrad,
        ),
        data=DataConfig(root="unused", patch_size=16, batch_size=1),
        rasterizer=RasterizationConfig(
            backend="intersections",
            tile_size=8,
            max_gaussians_per_tile=8,
            max_intersections=128,
        ),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(points, point_colors, config.model)
    optimizer = create_optimizer(model, config.optimizer)
    strategy_state = DefaultStrategy(config.strategy).initialize_state(8)
    safety_state = TrainingSafetyState()
    x = jnp.linspace(0.0, 1.0, 16, dtype=jnp.float32)
    y = jnp.linspace(0.0, 1.0, 16, dtype=jnp.float32)
    image = jnp.stack(
        jnp.meshgrid(x, y, indexing="xy") + [jnp.full((16, 16), 0.25)],
        axis=-1,
    )[None, ...]
    intrinsics = jnp.array(
        [[[30.0, 0.0, 8.0], [0.0, 30.0, 8.0], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )
    viewmats = jnp.eye(4, dtype=jnp.float32)[None, ...]
    key = jax.random.key(17)
    sh_degree = jnp.asarray(0)
    parameters = model.activated(split_sh=True)
    probe = jnp.zeros((1, model.capacity, 2), dtype=jnp.float32)

    def loss_with_probe(current_means, current_probe):
        renders, _, info = training_module.rasterization(
            current_means,
            parameters["quats"],
            parameters["scales"],
            parameters["opacities"],
            parameters["sh_coeffs"],
            viewmats,
            intrinsics,
            16,
            16,
            packed=False,
            active_mask=parameters["active_mask"],
            sh_degree=sh_degree,
            backgrounds=jnp.zeros((1, 3), jnp.float32),
            config=config.rasterizer,
            absgrad=absgrad,
            _means2d_offset=None if absgrad else current_probe,
            _means2d_absgrad_probe=current_probe if absgrad else None,
        )
        rgb = renders[..., :3]
        l1_value = jnp.mean(training_module.l1_loss(rgb, image))
        ssim_value = training_module.ssim(rgb, image)
        loss = (1.0 - config.ssim_lambda) * l1_value + config.ssim_lambda * (
            1.0 - ssim_value
        )
        return loss, info

    (_, info), screen_grad = jax.value_and_grad(
        lambda current_probe: loss_with_probe(
            parameters["means"], current_probe
        ),
        has_aux=True,
    )(probe)
    world_grad = jax.grad(
        lambda current_means: loss_with_probe(current_means, probe)[0]
    )(parameters["means"])
    expected = build_densification_stats(
        screen_grad,
        info["radii"],
        info["valid"],
        parameters["active_mask"],
        16,
        16,
    )
    expected_grad_sum = np.asarray(expected.grad_sum).copy()
    expected_count = np.asarray(expected.count).copy()
    expected_max_radii = np.asarray(expected.max_radii).copy()
    assert np.any(expected_grad_sum > 0.0)
    assert not np.allclose(
        expected_grad_sum, np.asarray(jnp.linalg.norm(world_grad, axis=-1))
    )

    metrics = make_train_step(config)(
        model,
        optimizer,
        strategy_state,
        safety_state,
        image,
        intrinsics,
        viewmats,
        key,
        sh_degree,
    )

    assert not bool(metrics["intersection_overflow"])
    np.testing.assert_allclose(
        strategy_state.grad_accum[...],
        expected_grad_sum,
        rtol=1.0e-3,
        # The joint NNX parameter/probe VJP may reassociate float32 tile
        # reductions relative to the probe-only oracle above.
        atol=2.0e-6,
    )
    np.testing.assert_array_equal(
        strategy_state.visible_count[...], expected_count
    )
    np.testing.assert_allclose(
        strategy_state.max_radii[...], expected_max_radii
    )


def _constant_training_rasterization(
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
    """Small differentiable renderer stub for commit-order tests."""

    camera_count = viewmats.shape[0]
    capacity = means.shape[0]
    screen_probe = kwargs["_means2d_offset"]
    signal = 0.25 + jnp.sum(means) * 0.0 + jnp.sum(screen_probe)
    renders = jnp.full(
        (camera_count, height, width, 3), signal, dtype=means.dtype
    )
    alphas = jnp.ones(
        (camera_count, height, width, 1), dtype=means.dtype
    )
    info = {
        "radii": jnp.ones((camera_count, capacity, 2), means.dtype),
        "valid": jnp.ones((camera_count, capacity), jnp.bool_),
        "tile_overflow": jnp.zeros((camera_count, 1, 1), jnp.bool_),
        "candidate_limit_exceeded": jnp.zeros(
            (camera_count, 1, 1), jnp.bool_
        ),
        "candidate_counts": jnp.zeros((camera_count, 1, 1), jnp.int32),
        "intersection_overflow": jnp.zeros((camera_count,), jnp.bool_),
        "intersection_count": jnp.ones((camera_count,), jnp.int32),
        "intersection_required_count": jnp.ones(
            (camera_count,), jnp.int32
        ),
    }
    return renders, alphas, info


def _snapshot_array_state(*nodes):
    def snapshot(leaf):
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        return np.asarray(leaf).copy()

    return tuple(
        tuple(
            snapshot(leaf)
            for leaf in jax.tree.leaves(nnx.as_pure(nnx.state(node)))
            if isinstance(leaf, jax.Array)
        )
        for node in nodes
    )


def _pose_sensitive_training_rasterization(*, overflow=False):
    def rasterize(
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
        camera_count = viewmats.shape[0]
        capacity = means.shape[0]
        screen_probe = kwargs["_means2d_offset"]
        signal = jax.nn.sigmoid(viewmats[:, 0, 3])
        signal = signal + 0.0 * jnp.sum(means) + 0.0 * jnp.sum(screen_probe)
        renders = jnp.broadcast_to(
            signal[:, None, None, None],
            (camera_count, height, width, 3),
        )
        alphas = jnp.ones(
            (camera_count, height, width, 1), dtype=means.dtype
        )
        info = {
            "radii": jnp.ones((camera_count, capacity, 2), means.dtype),
            "valid": jnp.ones((camera_count, capacity), jnp.bool_),
            "tile_overflow": jnp.zeros((camera_count, 1, 1), jnp.bool_),
            "candidate_limit_exceeded": jnp.zeros(
                (camera_count, 1, 1), jnp.bool_
            ),
            "candidate_counts": jnp.zeros((camera_count, 1, 1), jnp.int32),
            "intersection_overflow": jnp.full(
                (camera_count,), overflow, jnp.bool_
            ),
            "intersection_count": jnp.ones((camera_count,), jnp.int32),
            "intersection_required_count": jnp.full(
                (camera_count,), 2 if overflow else 1, jnp.int32
            ),
        }
        return renders, alphas, info

    return rasterize


def _pose_training_fixture(*, overflow=False, pose_noise=0.0):
    config = TrainConfig(
        pose_opt=True,
        pose_opt_lr=0.1,
        pose_opt_reg=0.0,
        pose_noise=pose_noise,
        model=ModelConfig(
            capacity=2, bucket_min_capacity=2, sh_degree=0
        ),
        optimizer=OptimizerConfig(max_steps=4),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(root="unused", patch_size=4, batch_size=1),
        rasterizer=RasterizationConfig(
            backend="jax", max_intersections=8
        ),
        ssim_lambda=0.0,
        steps=4,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.0, 0.0, 3.0]], np.float32),
        np.asarray([[128, 128, 128]], np.uint8),
        config.model,
    )
    optimizer = create_optimizer(model, config.optimizer)
    strategy_state = DefaultStrategy(config.strategy).initialize_state(
        model.capacity
    )
    pose_adjust = CameraOptModule(3, rngs=nnx.Rngs(7))
    pose_adjust.zero_init()
    pose_optimizer = training_module._create_pose_optimizer(
        pose_adjust, config
    )
    renderer = _pose_sensitive_training_rasterization(overflow=overflow)
    return (
        config,
        model,
        optimizer,
        strategy_state,
        pose_adjust,
        pose_optimizer,
        renderer,
    )


def test_pose_modules_apply_noise_then_adjustment_in_local_camera_frame():
    base_rotation = jnp.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    base = jnp.eye(4, dtype=jnp.float32)
    base = base.at[:3, :3].set(base_rotation)
    base = base.at[:3, 3].set(jnp.asarray([10.0, 20.0, 30.0]))
    pose_perturb = CameraOptModule(2, rngs=nnx.Rngs(1))
    pose_adjust = CameraOptModule(2, rngs=nnx.Rngs(2))
    pose_perturb.zero_init()
    pose_adjust.zero_init()
    pose_perturb.embeds.embedding[1, :3] = jnp.asarray([1.0, 0.0, 0.0])
    pose_adjust.embeds.embedding[1, :3] = jnp.asarray([0.0, 2.0, 0.0])

    adjusted = training_module._apply_camera_pose_modules(
        base[None],
        jnp.asarray([1], dtype=jnp.int32),
        pose_adjust=pose_adjust,
        pose_perturb=pose_perturb,
    )
    noise_delta = jnp.eye(4).at[:3, 3].set(jnp.asarray([1.0, 0.0, 0.0]))
    adjust_delta = jnp.eye(4).at[:3, 3].set(jnp.asarray([0.0, 2.0, 0.0]))
    expected = base @ noise_delta @ adjust_delta

    np.testing.assert_allclose(adjusted[0], expected, atol=1.0e-6)
    np.testing.assert_allclose(
        training_module._invert_rigid_transforms(adjusted)[0],
        jnp.linalg.inv(expected),
        atol=1.0e-6,
    )


def test_pose_optimizer_matches_upstream_lr_schedule_and_coupled_decay():
    config = TrainConfig(
        app_opt=True,
        pose_opt=True,
        pose_opt_lr=1.0e-2,
        pose_opt_reg=0.1,
        steps=2,
        data=DataConfig(batch_size=4),
    )
    pose_adjust = CameraOptModule(1, rngs=nnx.Rngs(3))
    pose_adjust.embeds.embedding[...] = 1.0
    optimizer = training_module._create_pose_optimizer(pose_adjust, config)
    zero_grads = jax.tree.map(
        jnp.zeros_like, nnx.state(pose_adjust, nnx.Param)
    )

    np.testing.assert_allclose(
        training_module._pose_learning_rate(config, jnp.asarray(0)),
        2.0e-2,
    )
    np.testing.assert_allclose(
        training_module._pose_learning_rate(config, jnp.asarray(2)),
        2.0e-4,
    )
    optimizer.update(pose_adjust, zero_grads)

    np.testing.assert_allclose(
        pose_adjust.embeds.embedding[...], 0.98, rtol=1.0e-5, atol=1.0e-6
    )
    assert int(optimizer.step[...]) == 1


def test_pose_train_step_updates_only_selected_embedding(monkeypatch):
    (
        config,
        model,
        optimizer,
        strategy_state,
        pose_adjust,
        pose_optimizer,
        renderer,
    ) = _pose_training_fixture()
    monkeypatch.setattr(training_module, "rasterization", renderer)
    step = make_train_step(config)
    inputs = dict(
        images=jnp.zeros((1, 4, 4, 3), jnp.float32),
        intrinsics=jnp.asarray(
            [[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]]
        ),
        viewmats=jnp.eye(4, dtype=jnp.float32)[None],
        key=jax.random.key(0),
        sh_degree=jnp.asarray(0),
        pose_adjust=pose_adjust,
        pose_optimizer=pose_optimizer,
        camtoworlds=jnp.eye(4, dtype=jnp.float32)[None],
        image_ids=jnp.asarray([2], dtype=jnp.int32),
    )

    first = step(
        model,
        optimizer,
        strategy_state,
        TrainingSafetyState(),
        **inputs,
    )
    second = step(
        model,
        optimizer,
        strategy_state,
        TrainingSafetyState(),
        **{**inputs, "key": jax.random.key(1)},
    )

    np.testing.assert_array_equal(pose_adjust.embeds.embedding[:2], 0.0)
    assert bool(jnp.any(pose_adjust.embeds.embedding[2] != 0.0))
    assert int(pose_optimizer.step[...]) == 2
    assert float(second["loss"]) < float(first["loss"])


def test_pose_train_step_overflow_atomically_skips_pose_update(monkeypatch):
    (
        config,
        model,
        optimizer,
        strategy_state,
        pose_adjust,
        pose_optimizer,
        renderer,
    ) = _pose_training_fixture(overflow=True)
    monkeypatch.setattr(training_module, "rasterization", renderer)
    before = _snapshot_array_state(pose_adjust, pose_optimizer)

    metrics = make_train_step(config)(
        model,
        optimizer,
        strategy_state,
        TrainingSafetyState(),
        jnp.zeros((1, 4, 4, 3), jnp.float32),
        jnp.asarray(
            [[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]]
        ),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(0),
        pose_adjust=pose_adjust,
        pose_optimizer=pose_optimizer,
        camtoworlds=jnp.eye(4, dtype=jnp.float32)[None],
        image_ids=jnp.asarray([1], dtype=jnp.int32),
    )

    assert bool(metrics["intersection_overflow"])
    assert int(pose_optimizer.step[...]) == 0
    after = _snapshot_array_state(pose_adjust, pose_optimizer)
    for before_node, after_node in zip(before, after, strict=True):
        for before_leaf, after_leaf in zip(
            before_node, after_node, strict=True
        ):
            np.testing.assert_array_equal(after_leaf, before_leaf)


def test_pose_train_step_jit_applies_fixed_noise_without_updating_it(
    monkeypatch,
):
    (
        config,
        model,
        optimizer,
        strategy_state,
        pose_adjust,
        pose_optimizer,
        renderer,
    ) = _pose_training_fixture(pose_noise=0.1)
    monkeypatch.setattr(training_module, "rasterization", renderer)
    pose_perturb = CameraOptModule(3, rngs=nnx.Rngs(11))
    pose_perturb.zero_init()
    pose_perturb.embeds.embedding[1, 0] = 0.5
    perturb_before = _snapshot_array_state(pose_perturb)

    metrics = make_train_step(config)(
        model,
        optimizer,
        strategy_state,
        TrainingSafetyState(),
        jnp.zeros((1, 4, 4, 3), jnp.float32),
        jnp.asarray(
            [[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]]
        ),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(0),
        pose_adjust=pose_adjust,
        pose_optimizer=pose_optimizer,
        pose_perturb=pose_perturb,
        camtoworlds=jnp.eye(4, dtype=jnp.float32)[None],
        image_ids=jnp.asarray([1], dtype=jnp.int32),
    )

    assert float(metrics["pose_error"]) > 0.0
    assert int(pose_optimizer.step[...]) == 1
    assert bool(jnp.any(pose_adjust.embeds.embedding[1] != 0.0))
    perturb_after = _snapshot_array_state(pose_perturb)
    for before_node, after_node in zip(
        perturb_before, perturb_after, strict=True
    ):
        for before_leaf, after_leaf in zip(
            before_node, after_node, strict=True
        ):
            np.testing.assert_array_equal(after_leaf, before_leaf)


@pytest.mark.parametrize(
    ("model_type", "gradient_key"),
    [("3dgs", "means2d"), ("2dgs", "gradient_2dgs")],
)
def test_real_rasterizers_produce_finite_nonzero_pose_gradients(
    model_type, gradient_key
):
    config = TrainConfig(
        pose_opt=True,
        pose_opt_lr=1.0e-2,
        pose_opt_reg=0.0,
        model_type=model_type,
        model=ModelConfig(
            capacity=2,
            bucket_min_capacity=2,
            sh_degree=0,
            initial_scale=0.2,
        ),
        optimizer=OptimizerConfig(max_steps=2),
        strategy=StrategyConfig(
            refine_start=100,
            max_new_per_refine=1,
            key_for_gradient=gradient_key,
        ),
        data=DataConfig(root="unused", patch_size=16, batch_size=1),
        rasterizer=RasterizationConfig(
            backend="jax",
            tile_size=8,
            max_gaussians_per_tile=8,
            max_intersections=64,
            near_plane=0.2,
            far_plane=200.0,
        ),
        ssim_lambda=0.0,
        steps=2,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.3, 0.1, 3.0]], np.float32),
        np.asarray([[255, 128, 64]], np.uint8),
        config.model,
    )
    optimizer = create_optimizer(model, config.optimizer)
    strategy_state = DefaultStrategy(config.strategy).initialize_state(
        model.capacity
    )
    pose_adjust = CameraOptModule(1, rngs=nnx.Rngs(0))
    pose_adjust.zero_init()
    pose_optimizer = training_module._create_pose_optimizer(
        pose_adjust, config
    )

    metrics = make_train_step(config)(
        model,
        optimizer,
        strategy_state,
        TrainingSafetyState(),
        jnp.zeros((1, 16, 16, 3), jnp.float32),
        jnp.asarray(
            [[[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]]]
        ),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(0),
        pose_adjust=pose_adjust,
        pose_optimizer=pose_optimizer,
        camtoworlds=jnp.eye(4, dtype=jnp.float32)[None],
        image_ids=jnp.asarray([0], dtype=jnp.int32),
    )

    embedding = np.asarray(pose_adjust.embeds.embedding[...])
    assert np.isfinite(float(metrics["loss"]))
    assert np.all(np.isfinite(embedding))
    assert np.any(embedding != 0.0)
    assert int(pose_optimizer.step[...]) == 1


def test_train_step_adds_active_3d_regularization_contributions(monkeypatch):
    monkeypatch.setattr(
        training_module, "rasterization", _constant_training_rasterization
    )
    model_config = ModelConfig(
        capacity=4,
        bucket_min_capacity=4,
        sh_degree=0,
        initial_scale=1.0,
    )
    config = TrainConfig(
        model=model_config,
        optimizer=OptimizerConfig(max_steps=10),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=2),
        data=DataConfig(root="unused", patch_size=4, batch_size=1),
        opacity_reg=0.4,
        scale_reg=0.1,
        ssim_lambda=0.0,
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.0, 0.0, 3.0], [0.2, 0.0, 3.0]], np.float32),
        np.asarray([[255, 0, 0], [0, 255, 0]], np.uint8),
        model_config,
    )
    model.opacity_logits[...] = jnp.asarray([0.0, jnp.log(3.0), 100.0, 100.0])
    model.log_scales[...] = jnp.log(
        jnp.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0],
            ]
        )
    )
    optimizer = create_optimizer(model, config.optimizer)
    strategy_state = DefaultStrategy(config.strategy).initialize_state(4)

    metrics = make_train_step(config)(
        model,
        optimizer,
        strategy_state,
        TrainingSafetyState(),
        jnp.full((1, 4, 4, 3), 0.25, jnp.float32),
        jnp.asarray(
            [[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]],
            jnp.float32,
        ),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(0),
    )

    np.testing.assert_allclose(metrics["opacity_reg_loss"], 0.25, rtol=1e-6)
    np.testing.assert_allclose(metrics["scale_reg_loss"], 0.35, rtol=1e-6)
    np.testing.assert_allclose(metrics["loss"], 0.6, rtol=1e-6, atol=1e-6)


def _selective_training_rasterization(
    *,
    expected_packed: bool,
    expected_sparse_grad: bool,
    expected_absgrad: bool = False,
    overflow: bool = False,
):
    def rasterize(
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
        assert kwargs["packed"] is expected_packed
        assert kwargs["sparse_grad"] is expected_sparse_grad
        assert kwargs.get("absgrad", False) is expected_absgrad
        camera_count = viewmats.shape[0]
        capacity = means.shape[0]
        if expected_absgrad:
            assert kwargs.get("_means2d_offset") is None
            assert kwargs.get("_gradient_2dgs_offset") is None
            screen_probe = kwargs.get("_means2d_absgrad_probe")
            if screen_probe is None:
                screen_probe = kwargs["_gradient_2dgs_absgrad_probe"]
        else:
            assert kwargs.get("_means2d_absgrad_probe") is None
            assert kwargs.get("_gradient_2dgs_absgrad_probe") is None
            screen_probe = kwargs.get("_means2d_offset")
            if screen_probe is None:
                screen_probe = kwargs["_gradient_2dgs_offset"]
        signal = (
            jnp.asarray(0.25, means.dtype)
            + 1.0e-3 * jnp.sum(means)
            + 1.0e-3 * jnp.sum(screen_probe)
        )
        renders = jnp.broadcast_to(
            signal, (camera_count, height, width, 3)
        )
        alphas = jnp.ones(
            (camera_count, height, width, 1), dtype=means.dtype
        )
        if expected_packed:
            projection_capacity = camera_count * capacity
            prefix = jnp.arange(projection_capacity) < camera_count
            camera_ids = jnp.where(
                prefix,
                jnp.arange(projection_capacity, dtype=jnp.int32),
                -1,
            )
            gaussian_ids = jnp.where(prefix, 0, -1).astype(jnp.int32)
            radii = jnp.where(
                prefix[:, None],
                jnp.ones((projection_capacity, 2), means.dtype),
                0.0,
            )
            valid = prefix
            projection_valid_count = jnp.asarray(
                camera_count, dtype=jnp.int32
            )
        else:
            camera_ids = None
            gaussian_ids = None
            radii = jnp.zeros(
                (camera_count, capacity, 2), dtype=means.dtype
            ).at[:, 0].set(1.0)
            valid = jnp.zeros(
                (camera_count, capacity), dtype=jnp.bool_
            ).at[:, 0].set(True)
            projection_valid_count = None
        info = {
            "camera_ids": camera_ids,
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "valid": valid,
            "projection_valid_count": projection_valid_count,
            "tile_overflow": jnp.zeros(
                (camera_count, 1, 1), dtype=jnp.bool_
            ),
            "candidate_limit_exceeded": jnp.zeros(
                (camera_count, 1, 1), dtype=jnp.bool_
            ),
            "candidate_counts": jnp.zeros((camera_count, 1, 1), jnp.int32),
            "intersection_overflow": jnp.full(
                (camera_count,), overflow, dtype=jnp.bool_
            ),
            "intersection_count": jnp.ones(
                (camera_count,), dtype=jnp.int32
            ),
            "intersection_required_count": jnp.full(
                (camera_count,), 2 if overflow else 1, dtype=jnp.int32
            ),
        }
        return renders, alphas, info

    return rasterize


def _selective_training_rasterization_2dgs(
    *,
    expected_packed: bool,
    expected_sparse_grad: bool,
    expected_absgrad: bool = False,
):
    rasterize_3dgs = _selective_training_rasterization(
        expected_packed=expected_packed,
        expected_sparse_grad=expected_sparse_grad,
        expected_absgrad=expected_absgrad,
    )

    def rasterize(*args, **kwargs):
        renders, alphas, info = rasterize_3dgs(*args, **kwargs)
        rendered_normals = jnp.zeros_like(renders)
        normals_from_depth = jnp.zeros_like(renders)
        render_distort = jnp.zeros_like(alphas)
        render_median = jnp.zeros_like(alphas)
        return (
            renders,
            alphas,
            rendered_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        )

    return rasterize


def _mcmc_ut_training_rasterization(*, with_ut: bool, with_eval3d: bool):
    def rasterize(
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
        assert kwargs["with_ut"] is with_ut
        assert kwargs["with_eval3d"] is with_eval3d
        assert kwargs.get("_means2d_offset") is None
        assert kwargs.get("_means2d_absgrad_probe") is None
        camera_count = viewmats.shape[0]
        capacity = means.shape[0]
        signal = jnp.asarray(0.25, means.dtype) + 0.0 * jnp.sum(means)
        renders = jnp.broadcast_to(
            signal, (camera_count, height, width, 3)
        )
        alphas = jnp.ones(
            (camera_count, height, width, 1), dtype=means.dtype
        )
        active = kwargs["active_mask"]
        radii = jnp.broadcast_to(
            active[None, :, None], (camera_count, capacity, 2)
        ).astype(means.dtype)
        valid = jnp.broadcast_to(active[None, :], (camera_count, capacity))
        info = {
            "radii": radii,
            "valid": valid,
            "tile_overflow": jnp.zeros(
                (camera_count, 1, 1), dtype=jnp.bool_
            ),
            "candidate_limit_exceeded": jnp.zeros(
                (camera_count, 1, 1), dtype=jnp.bool_
            ),
            "candidate_counts": jnp.zeros((camera_count, 1, 1), jnp.int32),
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

    return rasterize


def test_full_image_train_step_uses_rectangular_renderer_and_stats(monkeypatch):
    calls = []
    renderer = _selective_training_rasterization(
        expected_packed=False,
        expected_sparse_grad=False,
    )

    def rectangular_renderer(*args, **kwargs):
        width, height = args[7:9]
        assert (height, width) == (8, 12)
        calls.append(("renderer", height, width))
        return renderer(*args, **kwargs)

    build_stats = training_module.build_densification_stats

    def rectangular_stats(screen_grad, radii, valid, active, width, height):
        assert (height, width) == (8, 12)
        calls.append(("stats", height, width))
        return build_stats(screen_grad, radii, valid, active, width, height)

    monkeypatch.setattr(training_module, "rasterization", rectangular_renderer)
    monkeypatch.setattr(
        training_module, "build_densification_stats", rectangular_stats
    )
    config = TrainConfig(
        model=ModelConfig(capacity=2, bucket_min_capacity=2, sh_degree=0),
        optimizer=OptimizerConfig(max_steps=2),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(root="unused", patch_size=None, batch_size=1),
        rasterizer=RasterizationConfig(backend="jax", max_intersections=64),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.0, 0.0, 3.0]], np.float32),
        np.asarray([[128, 128, 128]], np.uint8),
        config.model,
    )
    optimizer = create_optimizer(model, config.optimizer)
    strategy_state = DefaultStrategy(config.strategy).initialize_state(
        model.capacity
    )

    make_train_step(config)(
        model,
        optimizer,
        strategy_state,
        TrainingSafetyState(),
        jnp.zeros((1, 8, 12, 3), jnp.float32),
        jnp.asarray(
            [[[10.0, 0.0, 6.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]],
            jnp.float32,
        ),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(0),
    )

    assert ("renderer", 8, 12) in calls
    assert ("stats", 8, 12) in calls


@pytest.mark.parametrize(
    ("sparse_grad", "visible_adam", "expected"),
    [
        pytest.param(True, False, "row-selective", id="sparse"),
        pytest.param(False, True, "visible-adam", id="visible-adam"),
    ],
)
def test_training_optimizer_routes_selective_factories(
    monkeypatch, sparse_grad, visible_adam, expected
):
    calls = []

    def make_optimizer(name):
        def factory(_model, _config, **kwargs):
            calls.append((name, kwargs))
            return name

        return factory

    monkeypatch.setattr(
        training_module,
        "create_row_selective_optimizer",
        make_optimizer("row-selective"),
    )
    monkeypatch.setattr(
        training_module,
        "create_visible_adam_optimizer",
        make_optimizer("visible-adam"),
    )
    config = TrainConfig(
        packed=sparse_grad,
        sparse_grad=sparse_grad,
        visible_adam=visible_adam,
        model=ModelConfig(capacity=2, bucket_min_capacity=2, sh_degree=0),
        data=DataConfig(batch_size=4),
    )
    model = GaussianModel.empty(config.model)

    assert (
        training_module._create_training_optimizer(
            model, config, scene_scale=2.5
        )
        == expected
    )
    assert calls == [
        (
            expected,
            {"batch_size": 4, "world_size": 1, "scene_scale": 2.5},
        )
    ]


def test_training_scene_scale_matches_normalized_camera_extent_margin():
    scene = SimpleNamespace(
        camtoworlds=np.asarray(
            [
                np.eye(4, dtype=np.float32),
                np.eye(4, dtype=np.float32),
            ]
        )
    )
    scene.camtoworlds[0, 0, 3] = -2.0
    scene.camtoworlds[1, 0, 3] = 2.0
    transform = training_module._legacy_scene_transform(scene)

    np.testing.assert_allclose(
        training_module._training_scene_scale(scene, transform), 1.1
    )
    np.testing.assert_allclose(
        training_module._training_scene_scale(
            scene, transform, global_scale=2.5
        ),
        2.75,
    )


def test_legacy_scene_scale_reproduces_pre_v6_float32_arithmetic():
    centers = np.asarray(
        [
            [-9.922983624080612, -0.8564410165427665, 5.142267808834161],
            [-9.921047947620284, -0.8569505800387824, 5.142476286631663],
        ],
        dtype=np.float64,
    )
    cameras = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    cameras[:, :3, 3] = centers
    scene = SimpleNamespace(camtoworlds=cameras)

    np.testing.assert_allclose(
        training_module._legacy_training_scene_scale(scene),
        1.0995174646377563,
        rtol=0.0,
        atol=0.0,
    )


def test_packed_training_metadata_deduplicates_ids_and_ignores_padding():
    info = {
        "camera_ids": jnp.asarray([0, 1, 1, -1, 0], jnp.int32),
        "gaussian_ids": jnp.asarray([2, 2, 1, -1, 0], jnp.int32),
        "radii": jnp.asarray(
            [[1, 1], [2, 2], [3, 3], [99, 99], [88, 88]], jnp.float32
        ),
        "valid": jnp.asarray([True, True, True, True, True]),
        "projection_valid_count": jnp.asarray(3, jnp.int32),
    }
    active = jnp.asarray([True, False, True, False])

    radii, valid, visible = training_module._unpack_training_projection_metadata(
        info, active, camera_count=2
    )

    expected_valid = np.zeros((2, 4), dtype=bool)
    expected_valid[0, 2] = True
    expected_valid[1, 2] = True
    expected_valid[1, 1] = True
    np.testing.assert_array_equal(valid, expected_valid)
    np.testing.assert_array_equal(visible, [False, False, True, False])
    np.testing.assert_array_equal(radii[0, 2], [1.0, 1.0])
    np.testing.assert_array_equal(radii[1, 2], [2.0, 2.0])
    np.testing.assert_array_equal(radii[:, 0], 0.0)


@pytest.mark.parametrize(
    ("model_type", "packed", "sparse_grad", "visible_adam"),
    [
        pytest.param("3dgs", True, True, False, id="3dgs-sparse-grad"),
        pytest.param("3dgs", False, False, True, id="3dgs-visible-adam"),
        pytest.param("2dgs", True, True, False, id="2dgs-sparse-grad"),
    ],
)
def test_selective_train_step_updates_only_visible_rows(
    monkeypatch, model_type, packed, sparse_grad, visible_adam
):
    if model_type == "2dgs":
        renderer_name = "rasterization_2dgs"
        renderer = _selective_training_rasterization_2dgs(
            expected_packed=packed,
            expected_sparse_grad=sparse_grad,
        )
    else:
        renderer_name = "rasterization"
        renderer = _selective_training_rasterization(
            expected_packed=packed,
            expected_sparse_grad=sparse_grad,
        )
    monkeypatch.setattr(training_module, renderer_name, renderer)
    config = TrainConfig(
        model_type=model_type,
        packed=packed,
        sparse_grad=sparse_grad,
        visible_adam=visible_adam,
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        optimizer=OptimizerConfig(max_steps=10),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(root="unused", patch_size=8, batch_size=2),
        rasterizer=RasterizationConfig(backend="jax"),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.0, 0.0, 3.0], [0.2, 0.0, 3.0]], np.float32),
        np.asarray([[128, 128, 128], [64, 64, 64]], np.uint8),
        config.model,
    )
    optimizer = training_module._create_training_optimizer(model, config)
    seed_grads = jax.tree.map(
        jnp.ones_like, nnx.state(model, nnx.Param)
    )
    optimizer.update(
        model,
        seed_grads,
        visible_mask=jnp.ones((model.capacity,), dtype=jnp.bool_),
    )
    model.quats[0] = jnp.asarray([2.0, 1.0, 0.0, 0.0])
    model.quats[1] = jnp.asarray([3.0, 4.0, 0.0, 0.0])
    parameter_before = tuple(
        np.asarray(leaf).copy()
        for leaf in jax.tree.leaves(
            nnx.as_pure(nnx.state(model, nnx.Param))
        )
    )
    optimizer_before = _snapshot_array_state(optimizer)[0]
    strategy_state = DefaultStrategy(config.strategy).initialize_state(
        model.capacity
    )

    metrics = make_train_step(config)(
        model,
        optimizer,
        strategy_state,
        TrainingSafetyState(),
        jnp.zeros((2, 8, 8, 3), jnp.float32),
        jnp.broadcast_to(
            jnp.asarray(
                [[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]],
                jnp.float32,
            ),
            (2, 3, 3),
        ),
        jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (2, 4, 4)),
        jax.random.key(0),
        jnp.asarray(0),
    )

    parameter_after = tuple(
        np.asarray(leaf).copy()
        for leaf in jax.tree.leaves(
            nnx.as_pure(nnx.state(model, nnx.Param))
        )
    )
    for before, after in zip(parameter_before, parameter_after, strict=True):
        np.testing.assert_array_equal(after[1:], before[1:])
    optimizer_after = _snapshot_array_state(optimizer)[0]
    for before, after in zip(optimizer_before, optimizer_after, strict=True):
        if before.ndim and before.shape[0] == model.capacity:
            np.testing.assert_array_equal(after[1:], before[1:])
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(model.quats[0])), 1.0, rtol=1.0e-6
    )
    np.testing.assert_array_equal(
        np.asarray(model.quats[1]), [3.0, 4.0, 0.0, 0.0]
    )
    assert int(optimizer.step[...]) == 2
    assert int(metrics["visible_count"]) == 1
    np.testing.assert_array_equal(
        strategy_state.visible_count[...], [2.0, 0.0, 0.0, 0.0]
    )


def test_sparse_train_step_overflow_does_not_advance_optimizer(monkeypatch):
    monkeypatch.setattr(
        training_module,
        "rasterization",
        _selective_training_rasterization(
            expected_packed=True,
            expected_sparse_grad=True,
            overflow=True,
        ),
    )
    config = TrainConfig(
        packed=True,
        sparse_grad=True,
        model=ModelConfig(capacity=2, bucket_min_capacity=2, sh_degree=0),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(root="unused", patch_size=8, batch_size=1),
        rasterizer=RasterizationConfig(backend="jax"),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.0, 0.0, 3.0]], np.float32),
        np.asarray([[128, 128, 128]], np.uint8),
        config.model,
    )
    optimizer = training_module._create_training_optimizer(model, config)
    strategy_state = DefaultStrategy(config.strategy).initialize_state(
        model.capacity
    )
    before = _snapshot_array_state(model, optimizer, strategy_state)

    metrics = make_train_step(config)(
        model,
        optimizer,
        strategy_state,
        TrainingSafetyState(),
        jnp.zeros((1, 8, 8, 3), jnp.float32),
        jnp.asarray(
            [[[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]]
        ),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(0),
    )

    assert bool(metrics["intersection_overflow"])
    assert int(optimizer.step[...]) == 0
    after = _snapshot_array_state(model, optimizer, strategy_state)
    for before_node, after_node in zip(before, after, strict=True):
        for before_leaf, after_leaf in zip(
            before_node, after_node, strict=True
        ):
            np.testing.assert_array_equal(after_leaf, before_leaf)


def test_default_stats_stop_at_refine_stop(monkeypatch):
    monkeypatch.setattr(
        training_module, "rasterization", _constant_training_rasterization
    )
    config = TrainConfig(
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        optimizer=OptimizerConfig(max_steps=10),
        strategy=StrategyConfig(
            refine_start=100,
            refine_stop=2,
            max_new_per_refine=1,
        ),
        data=DataConfig(root="unused", patch_size=8, batch_size=1),
        rasterizer=RasterizationConfig(backend="jax"),
        steps=2,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.0, 0.0, 3.0]], np.float32),
        np.asarray([[128, 128, 128]], np.uint8),
        config.model,
    )
    optimizer = create_optimizer(model, config.optimizer)
    state = DefaultStrategy(config.strategy).initialize_state(model.capacity)
    safety = TrainingSafetyState()
    step = make_train_step(config)
    arguments = (
        model,
        optimizer,
        state,
        safety,
        jnp.zeros((1, 8, 8, 3), jnp.float32),
        jnp.asarray(
            [[[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]]
        ),
        jnp.eye(4, dtype=jnp.float32)[None],
    )

    step(*arguments, jax.random.key(0), jnp.asarray(0), jax.random.key(10))
    first = tuple(
        np.asarray(value).copy()
        for value in (
            state.grad_accum[...],
            state.visible_count[...],
            state.max_radii[...],
        )
    )
    assert np.any(first[0] > 0.0)

    step(*arguments, jax.random.key(1), jnp.asarray(0), jax.random.key(11))
    for expected, actual in zip(
        first,
        (state.grad_accum[...], state.visible_count[...], state.max_radii[...]),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_mcmc_train_step_never_accumulates_screen_stats(monkeypatch):
    monkeypatch.setattr(
        training_module, "rasterization", _constant_training_rasterization
    )
    strategy_config = StrategyConfig(
        kind="mcmc",
        refine_start=100,
        noise_lr=0.0,
        max_new_per_refine=1,
    )
    config = TrainConfig(
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        optimizer=OptimizerConfig(max_steps=10),
        strategy=strategy_config,
        data=DataConfig(root="unused", patch_size=8, batch_size=1),
        rasterizer=RasterizationConfig(backend="jax"),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.0, 0.0, 3.0]], np.float32),
        np.asarray([[128, 128, 128]], np.uint8),
        config.model,
    )
    optimizer = create_optimizer(model, config.optimizer)
    state = MCMCStrategy(strategy_config).initialize_state(model.capacity)

    make_train_step(config)(
        model,
        optimizer,
        state,
        TrainingSafetyState(),
        jnp.zeros((1, 8, 8, 3), jnp.float32),
        jnp.asarray(
            [[[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]]
        ),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(0),
        jax.random.key(10),
    )

    np.testing.assert_array_equal(state.grad_accum[...], 0.0)
    np.testing.assert_array_equal(state.visible_count[...], 0.0)
    np.testing.assert_array_equal(state.max_radii[...], 0.0)


def test_mcmc_capacity_preflight_skips_the_whole_training_step(monkeypatch):
    monkeypatch.setattr(
        training_module,
        "rasterization",
        _selective_training_rasterization(
            expected_packed=False,
            expected_sparse_grad=False,
        ),
    )
    capacity = 20
    strategy_config = StrategyConfig(
        kind="mcmc",
        refine_start=0,
        refine_stop=10,
        refine_every=1,
        prune_opacity=0.01,
        cap_max=40,
        max_new_per_refine=2,
        noise_lr=2.0,
        noise_opacity_t=0.9,
        noise_opacity_k=10.0,
    )
    config = TrainConfig(
        model=ModelConfig(
            capacity=capacity,
            bucket_min_capacity=capacity,
            sh_degree=0,
        ),
        optimizer=OptimizerConfig(max_steps=10),
        strategy=strategy_config,
        data=DataConfig(root="unused", patch_size=8, batch_size=1),
        rasterizer=RasterizationConfig(backend="jax"),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.empty(config.model)
    model.active_mask[:] = True
    model.means[...] = jnp.arange(capacity * 3, dtype=jnp.float32).reshape(
        capacity, 3
    )
    model.log_scales[...] = jnp.log(0.1)
    model.opacity_logits[...] = jnp.log(4.0)
    model.opacity_logits[0] = -20.0
    optimizer = create_optimizer(model, config.optimizer)
    state = MCMCStrategy(strategy_config).initialize_state(capacity)
    state.grad_accum[...] = jnp.arange(capacity, dtype=jnp.float32) + 1.0
    state.visible_count[...] = 2.0
    state.max_radii[...] = 3.0
    model_before, optimizer_before = _snapshot_array_state(model, optimizer)
    stats_before = tuple(
        np.asarray(value).copy()
        for value in (
            state.grad_accum[...],
            state.visible_count[...],
            state.max_radii[...],
        )
    )

    metrics = make_train_step(config)(
        model,
        optimizer,
        state,
        TrainingSafetyState(),
        jnp.zeros((1, 8, 8, 3), jnp.float32),
        jnp.asarray(
            [[[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]]
        ),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(0),
        jax.random.key(10),
    )

    assert not bool(metrics["intersection_overflow"])
    assert int(optimizer.step[...]) == 0
    assert bool(state.capacity_overflow[...])
    assert int(state.last_new_count[...]) == 0
    assert int(state.last_pruned_count[...]) == 0
    model_after, optimizer_after = _snapshot_array_state(model, optimizer)
    for before_node, after_node in (
        (model_before, model_after),
        (optimizer_before, optimizer_after),
    ):
        for before_leaf, after_leaf in zip(
            before_node, after_node, strict=True
        ):
            np.testing.assert_array_equal(after_leaf, before_leaf)
    for before, after in zip(
        stats_before,
        (state.grad_accum[...], state.visible_count[...], state.max_radii[...]),
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize(
    ("with_ut", "with_eval3d"),
    [
        pytest.param(True, False, id="ut"),
        pytest.param(True, True, id="eval3d"),
    ],
)
def test_mcmc_ut_training_skips_screen_densification_stats(
    monkeypatch, with_ut, with_eval3d
):
    monkeypatch.setattr(
        training_module,
        "rasterization",
        _mcmc_ut_training_rasterization(
            with_ut=with_ut,
            with_eval3d=with_eval3d,
        ),
    )

    def unexpected_densification_stats(*_args, **_kwargs):
        raise AssertionError("MCMC UT training must not build screen statistics")

    monkeypatch.setattr(
        training_module,
        "build_densification_stats",
        unexpected_densification_stats,
    )
    strategy_config = StrategyConfig(
        kind="mcmc",
        refine_start=100,
        noise_lr=2.0,
        noise_opacity_t=0.9,
        noise_opacity_k=10.0,
        max_new_per_refine=1,
    )
    config = TrainConfig(
        with_ut=with_ut,
        with_eval3d=with_eval3d,
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        optimizer=OptimizerConfig(means_lr=0.1, max_steps=10),
        strategy=strategy_config,
        data=DataConfig(root="unused", patch_size=8, batch_size=1),
        rasterizer=RasterizationConfig(backend="jax"),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.0, 0.0, 3.0]], np.float32),
        np.asarray([[128, 128, 128]], np.uint8),
        config.model,
    )
    optimizer = create_optimizer(model, config.optimizer)
    state = MCMCStrategy(strategy_config).initialize_state(model.capacity)
    state.grad_accum[...] = 1.0
    state.visible_count[...] = 2.0
    state.max_radii[...] = 3.0
    means_before = np.asarray(model.means[...]).copy()
    stats_before = tuple(
        np.asarray(value).copy()
        for value in (
            state.grad_accum[...],
            state.visible_count[...],
            state.max_radii[...],
        )
    )

    metrics = make_train_step(config)(
        model,
        optimizer,
        state,
        TrainingSafetyState(),
        jnp.zeros((1, 8, 8, 3), jnp.float32),
        jnp.asarray(
            [[[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]]
        ),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(0),
        jax.random.key(10),
    )

    assert int(optimizer.step[...]) == 1
    assert not bool(metrics["intersection_overflow"])
    assert not np.array_equal(np.asarray(model.means[0]), means_before[0])
    np.testing.assert_array_equal(model.means[1:], means_before[1:])
    for before, after in zip(
        stats_before,
        (state.grad_accum[...], state.visible_count[...], state.max_radii[...]),
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("refine_every", [1, 2], ids=["scheduled", "noise-only"])
def test_mcmc_commit_matches_public_post_backward(
    monkeypatch, refine_every
):
    monkeypatch.setattr(
        training_module, "rasterization", _constant_training_rasterization
    )
    strategy_config = StrategyConfig(
        kind="mcmc",
        refine_start=0,
        refine_stop=10,
        refine_every=refine_every,
        prune_opacity=0.01,
        noise_lr=0.05,
        noise_opacity_t=0.9,
        noise_opacity_k=10.0,
        max_new_per_refine=1,
    )
    optimizer_config = OptimizerConfig(
        means_lr=0.1,
        scales_lr=0.0,
        quats_lr=0.0,
        opacities_lr=0.0,
        sh0_lr=0.0,
        sh_rest_lr=0.0,
        means_lr_final_scale=0.1,
        max_steps=10,
    )
    config = TrainConfig(
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        optimizer=optimizer_config,
        strategy=strategy_config,
        data=DataConfig(root="unused", patch_size=8, batch_size=1),
        rasterizer=RasterizationConfig(backend="jax"),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )

    def create_state():
        model = GaussianModel.from_point_cloud(
            np.asarray([[0.0, 0.0, 3.0], [0.3, 0.0, 3.0]], np.float32),
            np.asarray([[128, 128, 128], [64, 64, 64]], np.uint8),
            config.model,
        )
        model.opacity_logits[...] = model.opacity_logits[...].at[:2].set(
            jnp.asarray([-8.0, 0.0], jnp.float32)
        )
        optimizer = create_optimizer(model, optimizer_config)
        strategy = MCMCStrategy(strategy_config)
        return model, optimizer, strategy, strategy.initialize_state(model.capacity)

    model, optimizer, strategy, state = create_state()
    reference_model, reference_optimizer, reference_strategy, reference_state = (
        create_state()
    )
    step_key = jax.random.key(7)
    strategy_key = jax.random.key(19)
    make_train_step(config)(
        model,
        optimizer,
        state,
        TrainingSafetyState(),
        jnp.zeros((1, 8, 8, 3), jnp.float32),
        jnp.asarray(
            [[[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]]
        ),
        jnp.eye(4, dtype=jnp.float32)[None],
        step_key,
        jnp.asarray(0),
        strategy_key,
    )

    zero_grads = jax.tree.map(
        jnp.zeros_like, nnx.state(reference_model, nnx.Param)
    )
    reference_optimizer.update(reference_model, zero_grads)
    reference_model.normalize_quaternions()
    post_update_step = int(reference_optimizer.step[...])
    means_lr = optimizer_config.means_lr * (
        optimizer_config.means_lr_final_scale
        ** (post_update_step / optimizer_config.max_steps)
    )
    reference_strategy.step_post_backward(
        reference_model,
        reference_optimizer,
        reference_state,
        step=post_update_step,
        info={},
        lr=means_lr,
        key=strategy_key,
    )

    actual = _snapshot_array_state(model, optimizer, state)
    expected = _snapshot_array_state(
        reference_model, reference_optimizer, reference_state
    )
    for actual_node, expected_node in zip(actual, expected, strict=True):
        for actual_leaf, expected_leaf in zip(
            actual_node, expected_node, strict=True
        ):
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-6)
    if refine_every == 1:
        assert float(model.opacity_logits[...][0]) != -8.0
    else:
        assert float(model.opacity_logits[...][0]) == -8.0


@pytest.mark.parametrize(
    "packed_sparse", [False, True], ids=["dense", "packed-sparse"]
)
def test_2dgs_train_step_updates_optimizer_and_real_screen_stats(packed_sparse):
    points = np.array(
        [[0.0, 0.0, 3.0], [0.15, 0.0, 2.8], [-0.15, 0.1, 3.2]],
        np.float32,
    )
    colors = np.array(
        [[255, 32, 32], [32, 255, 32], [32, 32, 255]], np.uint8
    )
    config = TrainConfig(
        model_type="2dgs",
        packed=packed_sparse,
        sparse_grad=packed_sparse,
        normal_loss=True,
        normal_start_iter=0,
        dist_loss=True,
        dist_start_iter=0,
        model=ModelConfig(
            capacity=8,
            bucket_min_capacity=8,
            sh_degree=0,
            initial_scale=0.03,
        ),
        optimizer=OptimizerConfig(max_steps=10),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=2),
        data=DataConfig(root="unused", patch_size=16, batch_size=1),
        rasterizer=RasterizationConfig(
            backend="intersections",
            tile_size=8,
            max_gaussians_per_tile=8,
            max_intersections=256,
        ),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(points, colors, config.model)
    # Keep the scene in 2DGS's projective surfel branch. Very small surfels use
    # the screen-space antialiasing fallback, whose upstream densify VJP is
    # intentionally zero.
    model.log_scales[...] = jnp.full_like(model.log_scales[...], jnp.log(0.3))
    optimizer = training_module._create_training_optimizer(model, config)
    strategy_state = DefaultStrategy(config.strategy).initialize_state(8)
    safety_state = TrainingSafetyState()
    x = jnp.linspace(0.0, 1.0, 16, dtype=jnp.float32)
    y = jnp.linspace(0.0, 1.0, 16, dtype=jnp.float32)
    image = jnp.stack(
        jnp.meshgrid(x, y, indexing="xy") + [jnp.full((16, 16), 0.25)],
        axis=-1,
    )[None, ...]
    intrinsics = jnp.array(
        [[[30.0, 0.0, 8.0], [0.0, 30.0, 8.0], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )
    sh0_before = np.asarray(model.sh0[...]).copy()

    metrics = make_train_step(config)(
        model,
        optimizer,
        strategy_state,
        safety_state,
        image,
        intrinsics,
        jnp.eye(4, dtype=jnp.float32)[None, ...],
        jax.random.key(19),
        jnp.asarray(0),
    )

    assert int(optimizer.step[...]) == 1
    assert not np.array_equal(np.asarray(model.sh0[...]), sh0_before)
    grad_accum = np.asarray(strategy_state.grad_accum[...])
    assert np.all(np.isfinite(grad_accum))
    assert np.any(grad_accum > 0.0)
    assert int(metrics["visible_count"]) > 0
    assert float(metrics["normal_loss"]) > 0.0
    assert float(metrics["distortion_loss"]) > 0.0


def test_2dgs_regularizers_start_strictly_after_the_upstream_threshold():
    config = TrainConfig(
        model_type="2dgs",
        normal_loss=True,
        normal_lambda=0.25,
        normal_start_iter=3,
        dist_loss=True,
        dist_lambda=0.5,
        dist_start_iter=3,
    )
    rendered_normals = jnp.array([[[[1.0, 0.0, 0.0]]]], jnp.float32)
    normals_from_depth = jnp.array([[[[0.0, 1.0, 0.0]]]], jnp.float32)
    alphas = jnp.array([[[[0.4]]]], jnp.float32)
    distortion = jnp.array([[[[2.0]]]], jnp.float32)

    at_threshold = training_module._two_dgs_regularization_losses(
        rendered_normals,
        normals_from_depth,
        alphas,
        distortion,
        jnp.asarray(3),
        config,
    )
    after_threshold = training_module._two_dgs_regularization_losses(
        rendered_normals,
        normals_from_depth,
        alphas,
        distortion,
        jnp.asarray(4),
        config,
    )
    disabled = training_module._two_dgs_regularization_losses(
        rendered_normals,
        normals_from_depth,
        alphas,
        distortion,
        jnp.asarray(100),
        TrainConfig(model_type="2dgs"),
    )

    np.testing.assert_allclose(at_threshold, (0.0, 0.0))
    np.testing.assert_allclose(after_threshold, (0.25, 1.0))
    np.testing.assert_allclose(disabled, (0.0, 0.0))


def test_2dgs_render_step_uses_the_2d_rasterizer_contract():
    points = np.array([[0.0, 0.0, 3.0]], np.float32)
    colors = np.array([[255, 64, 32]], np.uint8)
    config = TrainConfig(
        model_type="2dgs",
        model=ModelConfig(
            capacity=2,
            bucket_min_capacity=2,
            sh_degree=0,
            initial_scale=0.2,
        ),
        rasterizer=RasterizationConfig(
            backend="intersections",
            tile_size=8,
            max_gaussians_per_tile=4,
            max_intersections=32,
        ),
    )
    model = GaussianModel.from_point_cloud(points, colors, config.model)

    image, alpha, tile_overflow, intersection_overflow = make_render_step(
        config, 8, 8
    )(
        model,
        jnp.eye(4, dtype=jnp.float32),
        jnp.array([[20.0, 0.0, 4.0], [0.0, 20.0, 4.0], [0.0, 0.0, 1.0]]),
        jnp.asarray(0),
    )

    assert image.shape == (8, 8, 3)
    assert alpha.shape == (8, 8, 1)
    assert tile_overflow.shape == (1, 1)
    assert intersection_overflow.shape == ()
    assert np.all(np.isfinite(np.asarray(image)))
    assert np.any(np.asarray(alpha) > 0.0)


def test_appearance_render_step_uses_zero_embedding_and_direct_rgb(monkeypatch):
    captured = {}

    def fake_rasterization(
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
        captured["sh_degree"] = kwargs["sh_degree"]
        captured["colors"] = colors
        rgb = jnp.broadcast_to(
            colors[:, :1, :], (viewmats.shape[0], height * width, 3)
        ).reshape(viewmats.shape[0], height, width, 3)
        alpha = jnp.ones((viewmats.shape[0], height, width, 1))
        info = {
            "tile_overflow": jnp.zeros(
                (viewmats.shape[0], 1, 1), dtype=jnp.bool_
            ),
            "intersection_overflow": jnp.zeros(
                (viewmats.shape[0],), dtype=jnp.bool_
            ),
        }
        return rgb, alpha, info

    monkeypatch.setattr(training_module, "rasterization", fake_rasterization)
    config = TrainConfig(
        app_opt=True,
        app_embed_dim=1,
        model=ModelConfig(capacity=2, bucket_min_capacity=2, sh_degree=0),
    )
    model = GaussianModel.empty(
        config.model, appearance_feature_dim=32
    )
    model.active_mask[0] = True
    model.colors[0] = jnp.asarray([0.2, -0.3, 0.4])
    appearance = AppearanceOptModule(
        1,
        32,
        embed_dim=1,
        sh_degree=0,
        mlp_width=4,
        mlp_depth=1,
        rngs=nnx.Rngs(31),
    )
    appearance.embeds.embedding[...] = 5.0
    appearance.color_head[0].kernel[...] = 0.0
    appearance.color_head[0].bias[...] = 0.0
    appearance.color_head[0].kernel[0, 0] = 1.0
    appearance.color_head[-1].kernel[...] = 0.0
    appearance.color_head[-1].bias[...] = 0.0
    appearance.color_head[-1].kernel[0, 0] = 1.0

    image, _, _, _ = make_render_step(config, 2, 2)(
        model,
        jnp.eye(4, dtype=jnp.float32),
        jnp.eye(3, dtype=jnp.float32),
        jnp.asarray(0),
        appearance_module=appearance,
    )

    assert captured["sh_degree"] is None
    expected = jnp.broadcast_to(jax.nn.sigmoid(model.colors[0]), image.shape)
    np.testing.assert_allclose(image, expected, rtol=1.0e-6)


@pytest.mark.parametrize(
    "config_factory",
    [
        pytest.param(
            lambda: TrainConfig(model_type="2dgs", camera_model="ortho"),
            id="camera-model",
        ),
        pytest.param(
            lambda: TrainConfig(model_type="2dgs", with_ut=True), id="ut"
        ),
        pytest.param(
            lambda: TrainConfig(model_type="2dgs", with_eval3d=True),
            id="eval3d",
        ),
        pytest.param(
            lambda: TrainConfig(
                model_type="2dgs",
                rasterizer=RasterizationConfig(rasterize_mode="antialiased"),
            ),
            id="antialiased",
        ),
        pytest.param(
            lambda: TrainConfig(
                model_type="2dgs", strategy=StrategyConfig(kind="mcmc")
            ),
            id="mcmc",
        ),
    ],
)
def test_2dgs_train_step_rejects_unsupported_modes(config_factory):
    with pytest.raises(ValueError, match="2DGS"):
        make_train_step(config_factory())


def test_default_train_step_still_rejects_eval3d_screen_statistics():
    with pytest.raises(NotImplementedError, match="screen-space densification"):
        make_train_step(
            TrainConfig(
                with_ut=True,
                with_eval3d=True,
                strategy=StrategyConfig(kind="default"),
            )
        )


def test_train_step_rejects_the_forward_only_pallas_compositor():
    with pytest.raises(NotImplementedError, match="forward-only"):
        make_train_step(
            TrainConfig(
                rasterizer=RasterizationConfig(compositor_backend="pallas")
            )
        )


@pytest.mark.parametrize(
    ("model_type", "packed"),
    [
        pytest.param("3dgs", False, id="3dgs-dense"),
        pytest.param("2dgs", True, id="2dgs-packed"),
    ],
)
def test_train_step_routes_absgrad_probe_to_renderer(
    monkeypatch, model_type, packed
):
    renderer_factory = (
        _selective_training_rasterization_2dgs
        if model_type == "2dgs"
        else _selective_training_rasterization
    )
    renderer_name = (
        "rasterization_2dgs" if model_type == "2dgs" else "rasterization"
    )
    monkeypatch.setattr(
        training_module,
        renderer_name,
        renderer_factory(
            expected_packed=packed,
            expected_sparse_grad=False,
            expected_absgrad=True,
        ),
    )
    config = TrainConfig(
        model_type=model_type,
        packed=packed,
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        optimizer=OptimizerConfig(max_steps=10),
        strategy=StrategyConfig(
            absgrad=True,
            refine_start=100,
            max_new_per_refine=1,
        ),
        data=DataConfig(root="unused", patch_size=8, batch_size=2),
        rasterizer=RasterizationConfig(backend="jax"),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.0, 0.0, 3.0]], np.float32),
        np.asarray([[128, 128, 128]], np.uint8),
        config.model,
    )
    optimizer = training_module._create_training_optimizer(model, config)
    strategy_state = DefaultStrategy(config.strategy).initialize_state(
        model.capacity
    )

    metrics = make_train_step(config)(
        model,
        optimizer,
        strategy_state,
        TrainingSafetyState(),
        jnp.zeros((2, 8, 8, 3), jnp.float32),
        jnp.broadcast_to(
            jnp.asarray(
                [[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]],
                jnp.float32,
            ),
            (2, 3, 3),
        ),
        jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (2, 4, 4)),
        jax.random.key(0),
        jnp.asarray(0),
    )

    assert int(optimizer.step[...]) == 1
    assert int(metrics["visible_count"]) == 1
    assert float(strategy_state.grad_accum[0]) > 0.0


@pytest.mark.parametrize(
    ("model_type", "strategy_kind"),
    [
        pytest.param("3dgs", "default", id="3dgs-default"),
        pytest.param("2dgs", "default", id="2dgs-default"),
        pytest.param("3dgs", "mcmc", id="3dgs-scheduled-mcmc"),
    ],
)
def test_intersection_overflow_skips_all_training_state_updates(
    model_type, strategy_kind
):
    points = np.array(
        [[0, 0, 3], [0.01, 0, 3], [-0.01, 0, 3]], np.float32
    )
    config = TrainConfig(
        model_type=model_type,
        model=ModelConfig(
            capacity=8,
            bucket_min_capacity=8,
            sh_degree=0,
            initial_scale=0.1,
        ),
        optimizer=OptimizerConfig(max_steps=10),
        strategy=StrategyConfig(
            kind=strategy_kind,
            refine_start=0 if strategy_kind == "mcmc" else 100,
            refine_stop=10,
            refine_every=1,
            noise_lr=2.0,
            noise_opacity_t=0.9,
            noise_opacity_k=10.0,
            max_new_per_refine=4,
        ),
        data=DataConfig(
            root="unused", patch_size=8, batch_size=1, num_workers=1
        ),
        rasterizer=RasterizationConfig(
            backend="intersections",
            tile_size=8,
            max_gaussians_per_tile=1,
            max_intersections=1,
            tile_batch_size=1,
        ),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        points, np.eye(3, dtype=np.float32), config.model, num_workers=1
    )
    optimizer = create_optimizer(model, config.optimizer)
    strategy = (
        MCMCStrategy(config.strategy)
        if strategy_kind == "mcmc"
        else DefaultStrategy(config.strategy)
    )
    strategy_state = strategy.initialize_state(8)
    safety_state = TrainingSafetyState()

    def snapshot(node):
        return [
            np.asarray(leaf).copy()
            for leaf in jax.tree.leaves(nnx.as_pure(nnx.state(node)))
            if isinstance(leaf, jax.Array)
        ]

    before = tuple(snapshot(node) for node in (model, optimizer, strategy_state))
    step = make_train_step(config)
    metrics = step(
        model,
        optimizer,
        strategy_state,
        safety_state,
        jnp.zeros((1, 8, 8, 3), jnp.float32),
        jnp.array([[[20.0, 0, 4], [0, 20.0, 4], [0, 0, 1]]], jnp.float32),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(0),
        jax.random.key(10),
    )

    assert int(metrics["overflow_tiles"]) == 0
    assert bool(metrics["intersection_overflow"])
    assert int(metrics["intersection_required_count"]) > 1
    assert int(safety_state.max_overflow_tiles[...]) == 0
    assert bool(safety_state.intersection_overflow_seen[...])
    assert int(optimizer.step[...]) == 0
    after = tuple(snapshot(node) for node in (model, optimizer, strategy_state))
    for before_node, after_node in zip(before, after, strict=True):
        for before_leaf, after_leaf in zip(before_node, after_node, strict=True):
            np.testing.assert_array_equal(after_leaf, before_leaf)

    metrics = step(
        model,
        optimizer,
        strategy_state,
        safety_state,
        jnp.zeros((1, 8, 8, 3), jnp.float32),
        jnp.array([[[20.0, 0, 4], [0, 20.0, 4], [0, 0, 1]]], jnp.float32),
        jnp.eye(4, dtype=jnp.float32)[None].at[0, 2, 3].set(-10.0),
        jax.random.key(1),
        jnp.asarray(0),
        jax.random.key(11),
    )

    assert int(metrics["overflow_tiles"]) == 0
    assert int(metrics["max_overflow_tiles"]) == 0
    assert bool(metrics["intersection_overflow_seen"])
    assert int(optimizer.step[...]) == 0
    after_sticky = tuple(
        snapshot(node) for node in (model, optimizer, strategy_state)
    )
    for before_node, after_node in zip(before, after_sticky, strict=True):
        for before_leaf, after_leaf in zip(before_node, after_node, strict=True):
            np.testing.assert_array_equal(after_leaf, before_leaf)


def test_memory_estimates_follow_physical_bucket_not_logical_maximum():
    config = TrainConfig(
        model=ModelConfig(capacity=100, bucket_min_capacity=16, sh_degree=1),
        strategy=StrategyConfig(max_new_per_refine=4),
        data=DataConfig(root="unused", patch_size=64),
    )
    initial = estimate_training_memory_bytes(config)
    assert initial == estimate_training_memory_bytes(
        config, physical_capacity=16
    )
    maximum = estimate_training_memory_bytes(config, physical_capacity=100)
    assert maximum > initial
    assert estimate_bucket_transition_memory_bytes(config, 16, 32) >= (
        estimate_training_memory_bytes(config, physical_capacity=32)
    )
    assert _initial_storage_capacity(config, 12) == 16
    assert _initial_storage_capacity(config, 16) == 16
    assert _initial_storage_capacity(config, 17) == 32


def test_intersection_capacity_uses_bounded_power_of_two_buckets():
    bucket = training_module._intersection_bucket_capacity
    assert bucket(1, minimum=65_536, maximum=1_000_000) == 65_536
    assert bucket(65_537, minimum=65_536, maximum=1_000_000) == 131_072
    assert bucket(900_000, minimum=65_536, maximum=1_000_000) == 1_000_000
    with pytest.raises(RuntimeError, match="exceed configured maximum"):
        bucket(1_000_001, minimum=65_536, maximum=1_000_000)
    with pytest.raises(ValueError, match="power of two"):
        TrainConfig(intersection_bucket_min_capacity=3)


def test_memory_estimate_accounts_for_dense_projection_camera_batch():
    single = TrainConfig(
        model=ModelConfig(capacity=100, bucket_min_capacity=100, sh_degree=1),
        data=DataConfig(root="unused", patch_size=64, batch_size=1),
    )
    double = TrainConfig(
        model=single.model,
        data=DataConfig(root="unused", patch_size=64, batch_size=2),
    )
    assert estimate_training_memory_bytes(double) > estimate_training_memory_bytes(
        single
    )


def test_memory_estimate_accounts_for_distributed_appearance_mlp_workspace():
    physical_capacity = 8
    render_capacity = 24
    appearance = TrainConfig(
        app_opt=True,
        app_embed_dim=5,
        model=ModelConfig(
            capacity=32, bucket_min_capacity=8, sh_degree=2
        ),
        data=DataConfig(root="unused", patch_size=16, batch_size=3),
    )
    sh = replace(appearance, app_opt=False)

    appearance_estimate = estimate_training_memory_bytes(
        appearance,
        physical_capacity=physical_capacity,
        render_capacity=render_capacity,
    )
    sh_estimate = estimate_training_memory_bytes(
        sh,
        physical_capacity=physical_capacity,
        render_capacity=render_capacity,
    )

    basis_count = (appearance.model.sh_degree + 1) ** 2
    mlp_input_width = appearance.app_embed_dim + 32 + basis_count
    floats_per_camera_gaussian = (
        3 + basis_count + mlp_input_width + 2 * 64 + 3 * 3
    )
    expected_workspace = (
        render_capacity
        * appearance.data.batch_size
        * floats_per_camera_gaussian
        * 4
        * 2
    )
    appearance_color_floats = 32 + 3
    sh_color_floats = basis_count * 3
    expected_model_delta = (
        physical_capacity
        * (appearance_color_floats - sh_color_floats)
        * 4
        * 6
    )

    assert appearance_estimate - sh_estimate == (
        expected_model_delta + expected_workspace
    )


def test_sh_memory_estimate_ignores_appearance_only_configuration():
    config = TrainConfig(
        model=ModelConfig(capacity=16, bucket_min_capacity=16, sh_degree=1),
        data=DataConfig(root="unused", patch_size=16, batch_size=2),
    )
    changed_appearance_fields = replace(
        config,
        app_embed_dim=config.app_embed_dim + 7,
        app_opt_lr=config.app_opt_lr * 2.0,
        app_opt_reg=config.app_opt_reg * 3.0,
    )

    assert estimate_training_memory_bytes(
        changed_appearance_fields,
        physical_capacity=16,
        render_capacity=32,
    ) == estimate_training_memory_bytes(
        config,
        physical_capacity=16,
        render_capacity=32,
    )


def test_evaluation_memory_estimate_accounts_for_appearance_mlp(monkeypatch):
    physical_capacity = 24
    appearance = TrainConfig(
        app_opt=True,
        app_embed_dim=5,
        model=ModelConfig(
            capacity=32, bucket_min_capacity=8, sh_degree=2
        ),
        data=DataConfig(root="unused", patch_size=16),
    )
    sh = replace(appearance, app_opt=False)
    monkeypatch.setattr(
        training_module, "_device_memory_usage", lambda: (0, 0)
    )

    appearance_estimate = _check_evaluation_memory_budget(
        appearance,
        physical_capacity=physical_capacity,
        width=16,
        height=16,
    )
    sh_estimate = _check_evaluation_memory_budget(
        sh,
        physical_capacity=physical_capacity,
        width=16,
        height=16,
    )

    basis_count = (appearance.model.sh_degree + 1) ** 2
    mlp_input_width = appearance.app_embed_dim + 32 + basis_count
    floats_per_gaussian = (
        3 + basis_count + mlp_input_width + 2 * 64 + 3 * 3
    )
    assert appearance_estimate - sh_estimate == (
        physical_capacity * floats_per_gaussian * 4
    )


def test_full_image_memory_estimate_requires_and_uses_rectangular_dimensions():
    config = TrainConfig(
        model=ModelConfig(capacity=100, bucket_min_capacity=100, sh_degree=1),
        data=DataConfig(root="unused", patch_size=None),
        rasterizer=RasterizationConfig(max_gaussians_per_tile=8),
    )

    with pytest.raises(ValueError, match="image_height and image_width"):
        estimate_training_memory_bytes(config)

    wide = estimate_training_memory_bytes(
        config, image_height=17, image_width=33
    )
    tall = estimate_training_memory_bytes(
        config, image_height=33, image_width=17
    )
    square = estimate_training_memory_bytes(
        config, image_height=33, image_width=33
    )

    assert wide == tall
    assert wide < square


def test_full_image_intersection_limit_uses_rectangular_tile_grid():
    config = TrainConfig(
        model=ModelConfig(capacity=10, bucket_min_capacity=10, sh_degree=0),
        data=DataConfig(root="unused", patch_size=None),
        rasterizer=RasterizationConfig(tile_size=16),
    )

    with pytest.raises(ValueError, match="image_height and image_width"):
        training_module._training_intersection_limit(config, 10)

    assert training_module._training_intersection_limit(
        config,
        10,
        image_height=17,
        image_width=33,
    ) == 10 * 2 * 3


def test_scene_full_image_size_uses_training_metadata_and_rejects_mixed_sizes():
    config = TrainConfig(data=DataConfig(root="unused", patch_size=None))
    scene = SimpleNamespace(
        images=(
            SimpleNamespace(height=17, width=33),
            SimpleNamespace(height=17, width=33),
            SimpleNamespace(height=9, width=11),
        ),
        indices=lambda split, test_every: np.asarray([0, 1]),
    )

    assert training_module._scene_training_render_size(scene, config) == (17, 33)

    scene.indices = lambda split, test_every: np.asarray([0, 2])
    with pytest.raises(ValueError, match="full-image training requires"):
        training_module._scene_training_render_size(scene, config)


def test_evaluation_preflight_includes_live_training_allocations(monkeypatch):
    config = TrainConfig(
        model=ModelConfig(capacity=16, bucket_min_capacity=16, sh_degree=0),
        data=DataConfig(root="unused", patch_size=16),
    )
    workspace = training_module.estimate_rasterization_memory_bytes(
        16, 64, 64, config.rasterizer
    )
    monkeypatch.setattr(
        training_module,
        "_device_memory_usage",
        lambda: (workspace, workspace * 2),
    )
    with pytest.raises(MemoryError, match="full-resolution evaluation"):
        _check_evaluation_memory_budget(
            config, physical_capacity=16, width=64, height=64
        )


def test_training_memory_preflight_checks_every_selected_device(monkeypatch):
    config = TrainConfig(
        model=ModelConfig(capacity=16, bucket_min_capacity=16, sh_degree=0),
        data=DataConfig(root="unused", patch_size=16),
    )
    estimate = estimate_training_memory_bytes(config, physical_capacity=16)
    devices = (object(), object())
    seen = []

    def fake_memory_usage(device=None):
        seen.append(device)
        limit = estimate * (2 if device is devices[0] else 1)
        return 0, limit

    monkeypatch.setattr(
        training_module, "_device_memory_usage", fake_memory_usage
    )

    with pytest.raises(MemoryError, match="selected device"):
        training_module._check_memory_budget(
            config,
            physical_capacity=16,
            devices=devices,
        )

    assert seen == list(devices)


def test_render_memory_estimate_matches_exact_default_intersection_capacity():
    capacity = 16
    width = height = 64
    default = RasterizationConfig(tile_size=16, max_intersections=None)
    exact = RasterizationConfig(
        tile_size=16,
        max_intersections=capacity * (width // 16) * (height // 16),
    )

    assert training_module.estimate_rasterization_memory_bytes(
        capacity, width, height, default
    ) == training_module.estimate_rasterization_memory_bytes(
        capacity, width, height, exact
    )


def test_data_workers_default_to_four_and_warn_on_oversubscription(monkeypatch):
    assert DataConfig().num_workers == 4
    monkeypatch.setattr(config_module, "_available_system_workers", lambda: 4)
    with pytest.warns(RuntimeWarning, match="exceeds the 4 workers"):
        config = DataConfig(num_workers=6)
    assert config.num_workers == 6
    with pytest.raises(ValueError, match="must be positive"):
        DataConfig(num_workers=0)

    dataset = _grain_iter_dataset(grain.MapDataset.range(4), 6)
    assert dataset._read_options.num_threads == 6
    assert dataset._read_options.prefetch_buffer_size == 8


def test_growth_preflight_fails_before_allocating_new_state(monkeypatch):
    config = TrainConfig(
        model=ModelConfig(capacity=8, bucket_min_capacity=4, sh_degree=0),
        strategy=StrategyConfig(max_new_per_refine=4),
    )
    model = GaussianModel.empty(config.model, physical_capacity=4)
    optimizer = create_optimizer(model, config.optimizer)
    strategy_state = DefaultStrategy(config.strategy).initialize_state(4)

    monkeypatch.setattr(
        training_module, "_device_memory_usage", lambda: (0, 1024)
    )

    def unexpected_resize(*args, **kwargs):
        raise AssertionError("resize must not run after a failed preflight")

    monkeypatch.setattr(
        training_module, "resize_training_state", unexpected_resize
    )
    with pytest.raises(MemoryError, match="stopped before allocation"):
        _grow_training_state(
            config,
            model,
            optimizer,
            strategy_state,
            new_capacity=8,
            image_height=64,
            image_width=64,
        )
    assert model.capacity == 4


def test_refine_can_grow_then_compact_a_bucket():
    strategy_config = StrategyConfig(
        refine_start=1,
        refine_every=1,
        max_new_per_refine=4,
        grow_grad2d=0.5,
        prune_opacity=1.0e-4,
        prune_scale3d=10.0,
    )
    config = TrainConfig(
        model=ModelConfig(capacity=8, bucket_min_capacity=4, sh_degree=0),
        strategy=strategy_config,
    )
    model = GaussianModel.empty(config.model, physical_capacity=4)
    model.active_mask[...] = True
    optimizer = create_optimizer(model, config.optimizer)
    strategy = DefaultStrategy(strategy_config)
    strategy_state = strategy.initialize_state(4)
    strategy_state.grad_accum[...] = 1.0
    strategy_state.visible_count[...] = 1.0

    required = int(strategy.required_capacity(model, strategy_state, 1.0))
    assert required == 8
    model, optimizer, strategy_state = _grow_training_state(
        config,
        model,
        optimizer,
        strategy_state,
        new_capacity=8,
        image_height=64,
        image_width=64,
    )
    metrics = strategy.refine(
        model,
        strategy_state,
        optimizer,
        jax.random.key(0),
        1.0,
    )
    active_count = training_module.compact_training_state(
        model, optimizer, strategy_state
    )
    assert int(active_count) == 8
    assert int(metrics["new_count"]) == 4
    np.testing.assert_array_equal(np.asarray(model.active_mask[...]), True)


def test_train_rejects_multi_process_before_writing(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    config = TrainConfig(
        data=DataConfig(root="unused"), output_dir=str(output_dir)
    )
    monkeypatch.setattr(training_module.jax, "process_count", lambda: 2)

    with pytest.raises(NotImplementedError, match="single-process"):
        training_module.train(config)

    assert not output_dir.exists()


def test_distributed_train_rejects_unsupported_topology_before_writing(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "output"
    config = TrainConfig(
        data=DataConfig(root="unused"), output_dir=str(output_dir)
    )
    monkeypatch.setattr(training_module.jax, "process_count", lambda: 2)

    with pytest.raises(NotImplementedError, match="one JAX process"):
        training_module.train(config, distributed=True)

    assert not output_dir.exists()

    monkeypatch.setattr(training_module.jax, "process_count", lambda: 1)
    monkeypatch.setattr(
        training_module.jax,
        "local_devices",
        lambda: [jax.devices()[0]],
    )
    with pytest.raises(ValueError, match="at least two local devices"):
        training_module.train(config, distributed=True)

    assert not output_dir.exists()


def test_train_saves_compacted_latest_step(monkeypatch, tmp_path):
    camtoworlds = np.broadcast_to(
        np.eye(4, dtype=np.float32), (3, 4, 4)
    ).copy()
    camtoworlds[:, :3, 3] = np.asarray(
        [[-2.0, 1.0, 0.0], [0.0, 1.0, 1.0], [3.0, 1.0, 0.0]],
        np.float32,
    )
    scene = SimpleNamespace(
        points=np.asarray(
            [
                [-2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, -1.0, -0.25],
                [0.0, 1.0, 1.0],
            ],
            np.float32,
        ),
        points_rgb=np.full((4, 3), 128, np.uint8),
        camtoworlds=camtoworlds,
    )
    batch = {
        "image": np.zeros((4, 4, 3), np.float32),
        "K": np.array([[10.0, 0, 2], [0, 10.0, 2], [0, 0, 1]], np.float32),
        "w2c": np.eye(4, dtype=np.float32),
    }
    config = TrainConfig(
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(
            root="unused", patch_size=4, batch_size=1, num_workers=1
        ),
        steps=3,
        checkpoint_every=2,
        eval_every=0,
        output_dir=str(tmp_path),
    )
    calls = 0
    events = []

    def fake_make_train_step(_config):
        def fake_train_step(model, _optimizer, _strategy_state, safety_state, *_args):
            nonlocal calls
            calls += 1
            if calls == 1:
                model.active_mask[...] = jnp.array(
                    [True, False, False, False]
                )
            elif calls == 3:
                model.active_mask[...] = jnp.array(
                    [False, True, False, False]
                )
            return {
                "loss": jnp.asarray(0.0),
                "l1": jnp.asarray(0.0),
                "ssim": jnp.asarray(1.0),
                "psnr": jnp.asarray(100.0),
                "active_count": model.active_count,
                "visible_count": jnp.asarray(1),
                "overflow_tiles": jnp.asarray(0),
                "max_overflow_tiles": safety_state.max_overflow_tiles[...],
                "candidate_limit_exceeded_tiles": jnp.asarray(0),
                "busiest_tile_candidates": jnp.asarray(0),
                "intersection_overflow": jnp.asarray(False),
                "intersection_overflow_seen": (
                    safety_state.intersection_overflow_seen[...]
                ),
                "intersection_count": jnp.asarray(1),
                "intersection_required_count": jnp.asarray(1),
            }

        return fake_train_step

    def fake_compact(model, _optimizer, _strategy_state):
        events.append(("compact",))
        active_count = int(jax.device_get(model.active_count))
        model.active_mask[...] = jnp.arange(model.capacity) < active_count
        return jnp.asarray(active_count)

    def fake_save(directory, model, *, step, **kwargs):
        mask = np.asarray(model.active_mask[...]).copy()
        events.append(("save", step, mask))
        expected_transform = training_module.compute_scene_transform(scene)
        expected_scale = training_module._training_scene_scale(
            scene, expected_transform
        )
        np.testing.assert_allclose(
            kwargs["scene_transform"], expected_transform.matrix
        )
        np.testing.assert_allclose(kwargs["scene_scale"], expected_scale)
        return directory / f"step_{step:08d}"

    monkeypatch.setattr(training_module, "load_colmap_scene", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        training_module, "create_grain_dataset", lambda *_a, **_k: [batch]
    )
    monkeypatch.setattr(training_module, "make_train_step", fake_make_train_step)
    monkeypatch.setattr(training_module, "compact_training_state", fake_compact)
    monkeypatch.setattr(training_module, "save_checkpoint", fake_save)
    monkeypatch.setattr(training_module, "_check_memory_budget", lambda *_a, **_k: 0)

    result = training_module.train(config)

    assert [event[:2] for event in events] == [
        ("compact",),
        ("save", 2),
        ("compact",),
        ("save", 3),
    ]
    for event in events:
        if event[0] == "save":
            np.testing.assert_array_equal(
                event[2], np.array([True, False, False, False])
            )
    assert result.checkpoint == tmp_path / "checkpoints" / "step_00000003"


def test_train_wires_dataset_index_pose_state_and_checkpoint_manifest(
    monkeypatch, tmp_path
):
    class PoseScene(SimpleNamespace):
        def indices(self, split, _test_every):
            assert split == "train"
            return np.asarray([0, 1], dtype=np.int64)

    scene_camtoworlds = np.repeat(
        np.eye(4, dtype=np.float32)[None], 2, axis=0
    )
    scene_camtoworlds[:, 0, 3] = np.asarray([2.0, 4.0])
    scene = PoseScene(
        points=np.asarray([[0.0, 0.0, 3.0]], np.float32),
        points_rgb=np.asarray([[128, 128, 128]], np.uint8),
        camtoworlds=scene_camtoworlds,
        images=(SimpleNamespace(name="a.png"), SimpleNamespace(name="b.png")),
    )
    batch = {
        "image": np.zeros((1, 4, 4, 3), np.float32),
        "K": np.asarray(
            [[[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]],
            np.float32,
        ),
        "w2c": np.linalg.inv(scene_camtoworlds[1])[None].astype(np.float32),
        "dataset_index": np.asarray([1], np.int32),
        "image_index": np.asarray([99], np.int32),
        "image_id": np.asarray([1234], np.int64),
    }
    config = TrainConfig(
        normalize_world_space=False,
        global_scale=2.5,
        pose_opt=True,
        model=ModelConfig(capacity=2, bucket_min_capacity=2, sh_degree=0),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(
            root="unused", patch_size=4, batch_size=1, num_workers=1
        ),
        steps=1,
        checkpoint_every=0,
        eval_every=0,
        output_dir=str(tmp_path),
    )
    dispatched = {}
    saved = {}

    def fake_make_train_step(_config):
        def fake_train_step(
            model, _optimizer, _strategy, safety_state, *_args, **kwargs
        ):
            dispatched.update(kwargs)
            return {
                "loss": jnp.asarray(0.0),
                "l1": jnp.asarray(0.0),
                "ssim": jnp.asarray(1.0),
                "psnr": jnp.asarray(100.0),
                "active_count": model.active_count,
                "visible_count": jnp.asarray(1),
                "overflow_tiles": jnp.asarray(0),
                "max_overflow_tiles": safety_state.max_overflow_tiles[...],
                "candidate_limit_exceeded_tiles": jnp.asarray(0),
                "busiest_tile_candidates": jnp.asarray(0),
                "intersection_overflow": jnp.asarray(False),
                "intersection_overflow_seen": (
                    safety_state.intersection_overflow_seen[...]
                ),
                "intersection_count": jnp.asarray(1),
                "intersection_required_count": jnp.asarray(1),
                "pose_error": jnp.asarray(0.0),
            }

        return fake_train_step

    def fake_save(directory, *_args, step, **kwargs):
        saved.update(kwargs)
        return directory / f"step_{step:08d}"

    monkeypatch.setattr(training_module, "load_colmap_scene", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        training_module, "create_grain_dataset", lambda *_a, **_k: [batch]
    )
    monkeypatch.setattr(training_module, "make_train_step", fake_make_train_step)
    monkeypatch.setattr(
        training_module, "_save_compacted_training_checkpoint", fake_save
    )
    monkeypatch.setattr(training_module, "_check_memory_budget", lambda *_a, **_k: 0)

    result = training_module.train(config)

    np.testing.assert_array_equal(dispatched["image_ids"], [1])
    expected_camtoworld = np.eye(4, dtype=np.float32)[None]
    expected_camtoworld[0, 0, 3] = 4.0
    np.testing.assert_allclose(
        dispatched["camtoworlds"], expected_camtoworld
    )
    assert dispatched["pose_adjust"] is result.pose_adjust
    assert dispatched["pose_optimizer"] is not None
    assert dispatched["pose_perturb"] is None
    assert saved["pose_adjust"] is result.pose_adjust
    assert saved["pose_image_names"] == ("a.png", "b.png")
    np.testing.assert_array_equal(saved["scene_transform"].matrix, np.eye(4))
    np.testing.assert_allclose(saved["scene_scale"], 2.75)

    dispatched.clear()
    saved.clear()
    noise_result = training_module.train(
        replace(
            config,
            pose_opt=False,
            pose_noise=0.01,
            output_dir=str(tmp_path / "noise"),
        )
    )
    assert noise_result.pose_adjust is None
    assert dispatched["pose_adjust"] is None
    assert dispatched["pose_optimizer"] is None
    assert dispatched["pose_perturb"] is not None
    assert saved["pose_adjust"] is None
    assert saved["pose_optimizer"] is None
    assert saved["pose_image_names"] == ("a.png", "b.png")


def test_train_wires_appearance_state_dataset_index_and_checkpoint_manifest(
    monkeypatch, tmp_path
):
    class AppearanceScene(SimpleNamespace):
        def indices(self, split, _test_every):
            assert split == "train"
            return np.asarray([0, 1], dtype=np.int64)

    scene = AppearanceScene(
        points=np.asarray([[0.0, 0.0, 3.0]], np.float32),
        points_rgb=np.asarray([[128, 64, 32]], np.uint8),
        camtoworlds=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        images=(SimpleNamespace(name="a.png"), SimpleNamespace(name="b.png")),
    )
    batch = {
        "image": np.zeros((1, 4, 4, 3), np.float32),
        "K": np.asarray(
            [[[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]]
        ),
        "w2c": np.eye(4, dtype=np.float32)[None],
        "dataset_index": np.asarray([1], np.int32),
    }
    config = TrainConfig(
        normalize_world_space=False,
        app_opt=True,
        app_embed_dim=4,
        model=ModelConfig(capacity=2, bucket_min_capacity=2, sh_degree=1),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(
            root="unused", patch_size=4, batch_size=1, num_workers=1
        ),
        steps=1,
        checkpoint_every=0,
        eval_every=0,
        output_dir=str(tmp_path),
    )
    dispatched = {}
    saved = {}

    def fake_make_train_step(_config):
        def fake_train_step(
            model, _optimizer, _strategy, safety_state, *_args, **kwargs
        ):
            dispatched.update(kwargs)
            return {
                "loss": jnp.asarray(0.0),
                "l1": jnp.asarray(0.0),
                "ssim": jnp.asarray(1.0),
                "psnr": jnp.asarray(100.0),
                "active_count": model.active_count,
                "visible_count": jnp.asarray(1),
                "overflow_tiles": jnp.asarray(0),
                "max_overflow_tiles": safety_state.max_overflow_tiles[...],
                "candidate_limit_exceeded_tiles": jnp.asarray(0),
                "busiest_tile_candidates": jnp.asarray(0),
                "intersection_overflow": jnp.asarray(False),
                "intersection_overflow_seen": (
                    safety_state.intersection_overflow_seen[...]
                ),
                "intersection_count": jnp.asarray(1),
                "intersection_required_count": jnp.asarray(1),
                "pose_error": jnp.asarray(0.0),
            }

        return fake_train_step

    def fake_save(directory, *_args, step, **kwargs):
        saved.update(kwargs)
        return directory / f"step_{step:08d}"

    monkeypatch.setattr(training_module, "load_colmap_scene", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        training_module, "create_grain_dataset", lambda *_a, **_k: [batch]
    )
    monkeypatch.setattr(training_module, "make_train_step", fake_make_train_step)
    monkeypatch.setattr(
        training_module, "_save_compacted_training_checkpoint", fake_save
    )
    monkeypatch.setattr(training_module, "_check_memory_budget", lambda *_a, **_k: 0)

    result = training_module.train(config)

    assert result.model.has_appearance
    assert result.appearance is dispatched["appearance_module"]
    assert dispatched["appearance_optimizer"] is not None
    np.testing.assert_array_equal(dispatched["image_ids"], [1])
    np.testing.assert_array_equal(dispatched["camtoworlds"], np.eye(4)[None])
    assert saved["appearance_module"] is result.appearance
    assert saved["appearance_optimizer"] is not None
    assert saved["appearance_image_names"] == ("a.png", "b.png")


@pytest.mark.parametrize(
    ("saved", "current", "match"),
    [
        (
            TrainConfig(pose_opt=True),
            TrainConfig(pose_opt=False),
            "pose_opt",
        ),
        (
            TrainConfig(pose_noise=0.01),
            TrainConfig(pose_noise=0.02),
            "pose_noise",
        ),
        (
            TrainConfig(pose_noise=0.01, seed=1),
            TrainConfig(pose_noise=0.01, seed=2),
            "seed",
        ),
        (
            TrainConfig(pose_opt=True, steps=100),
            TrainConfig(pose_opt=True, steps=200),
            "steps",
        ),
        (
            TrainConfig(app_opt=True),
            TrainConfig(app_opt=False),
            "app_opt",
        ),
        (
            TrainConfig(app_opt=True, app_embed_dim=16),
            TrainConfig(app_opt=True, app_embed_dim=8),
            "app_embed_dim",
        ),
        (
            TrainConfig(app_opt=True, app_opt_lr=1.0e-3),
            TrainConfig(app_opt=True, app_opt_lr=2.0e-3),
            "app_opt_lr",
        ),
        (
            TrainConfig(data=DataConfig(batch_size=1)),
            TrainConfig(data=DataConfig(batch_size=2)),
            "batch_size",
        ),
        (
            TrainConfig(normalize_world_space=True),
            TrainConfig(normalize_world_space=False),
            "normalize_world_space",
        ),
        (
            TrainConfig(global_scale=1.0),
            TrainConfig(global_scale=2.0),
            "global_scale",
        ),
    ],
)
def test_camera_module_resume_rejects_structural_config_changes(
    monkeypatch, saved, current, match
):
    monkeypatch.setattr(
        training_module, "load_checkpoint_config", lambda _path: saved
    )

    with pytest.raises(ValueError, match=match):
        training_module._validate_camera_module_resume_config(
            current, "checkpoint"
        )


def test_pose_and_appearance_host_orbax_resume_match_uninterrupted_next_step(
    monkeypatch, tmp_path
):
    class PoseScene(SimpleNamespace):
        def indices(self, split, _test_every):
            assert split == "train"
            return np.asarray([0, 1], dtype=np.int64)

    scene_camtoworlds = np.repeat(
        np.eye(4, dtype=np.float32)[None], 2, axis=0
    )
    scene_camtoworlds[:, 0, 3] = np.asarray([-1.0, 1.0])
    scene = PoseScene(
        points=np.asarray([[0.0, 0.0, 3.0]], np.float32),
        points_rgb=np.asarray([[128, 128, 128]], np.uint8),
        camtoworlds=scene_camtoworlds,
        images=(SimpleNamespace(name="a.png"), SimpleNamespace(name="b.png")),
    )
    batch = {
        "image": np.zeros((1, 4, 4, 3), np.float32),
        "K": np.asarray(
            [[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]],
            np.float32,
        ),
        "w2c": np.linalg.inv(scene_camtoworlds[1])[None].astype(np.float32),
        "dataset_index": np.asarray([1], np.int32),
    }
    config = TrainConfig(
        normalize_world_space=False,
        app_opt=True,
        app_embed_dim=4,
        pose_opt=True,
        pose_opt_lr=0.1,
        pose_opt_reg=0.0,
        model=ModelConfig(capacity=2, bucket_min_capacity=2, sh_degree=0),
        optimizer=OptimizerConfig(max_steps=2),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(
            root="unused", patch_size=4, batch_size=1, num_workers=1
        ),
        rasterizer=RasterizationConfig(
            backend="jax", max_intersections=8
        ),
        ssim_lambda=0.0,
        steps=2,
        checkpoint_every=1,
        eval_every=0,
        output_dir=str(tmp_path / "continuous"),
    )
    snapshots = {}
    pose_optimizers = []
    appearance_optimizers = []
    real_create_pose_optimizer = training_module._create_pose_optimizer
    real_create_appearance_optimizer = (
        training_module.create_appearance_optimizer
    )
    real_save = training_module._save_compacted_training_checkpoint
    real_restore = training_module.restore_checkpoint

    def tracked_create_pose_optimizer(module, current_config):
        optimizer = real_create_pose_optimizer(module, current_config)
        pose_optimizers.append(optimizer)
        return optimizer

    def tracked_create_appearance_optimizer(module, current_config):
        optimizer = real_create_appearance_optimizer(module, current_config)
        appearance_optimizers.append(optimizer)
        return optimizer

    def tracked_save(
        *args,
        step,
        pose_adjust,
        pose_optimizer,
        appearance_module,
        appearance_optimizer,
        **kwargs,
    ):
        if step == 1 and "saved" not in snapshots:
            snapshots["saved"] = _snapshot_array_state(
                pose_adjust,
                pose_optimizer,
                appearance_module,
                appearance_optimizer,
            )
        return real_save(
            *args,
            step=step,
            pose_adjust=pose_adjust,
            pose_optimizer=pose_optimizer,
            appearance_module=appearance_module,
            appearance_optimizer=appearance_optimizer,
            **kwargs,
        )

    def tracked_restore(
        *args,
        pose_module,
        pose_optimizer,
        appearance_module,
        appearance_optimizer,
        **kwargs,
    ):
        step = real_restore(
            *args,
            pose_module=pose_module,
            pose_optimizer=pose_optimizer,
            appearance_module=appearance_module,
            appearance_optimizer=appearance_optimizer,
            **kwargs,
        )
        snapshots["restored"] = _snapshot_array_state(
            pose_module,
            pose_optimizer,
            appearance_module,
            appearance_optimizer,
        )
        snapshots["restored_step"] = step
        return step

    monkeypatch.setattr(training_module, "load_colmap_scene", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        training_module, "create_grain_dataset", lambda *_a, **_k: [batch]
    )
    monkeypatch.setattr(
        training_module,
        "rasterization",
        _pose_sensitive_training_rasterization(),
    )
    monkeypatch.setattr(training_module, "_check_memory_budget", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        training_module, "_create_pose_optimizer", tracked_create_pose_optimizer
    )
    monkeypatch.setattr(
        training_module,
        "create_appearance_optimizer",
        tracked_create_appearance_optimizer,
    )
    monkeypatch.setattr(
        training_module, "_save_compacted_training_checkpoint", tracked_save
    )
    monkeypatch.setattr(training_module, "restore_checkpoint", tracked_restore)

    uninterrupted = training_module.train(config)
    uninterrupted_final = _snapshot_array_state(
        uninterrupted.pose_adjust,
        pose_optimizers[0],
        uninterrupted.appearance,
        appearance_optimizers[0],
    )
    resume_checkpoint = (
        tmp_path / "continuous" / "checkpoints" / "step_00000001"
    )
    resumed = training_module.train(
        replace(config, output_dir=str(tmp_path / "resumed")),
        resume_from=resume_checkpoint,
    )
    resumed_final = _snapshot_array_state(
        resumed.pose_adjust,
        pose_optimizers[1],
        resumed.appearance,
        appearance_optimizers[1],
    )

    assert snapshots["restored_step"] == 1
    assert int(pose_optimizers[1].step[...]) == 2
    assert int(appearance_optimizers[1].step[...]) == 2
    np.testing.assert_array_equal(resumed.pose_adjust.embeds.embedding[0], 0.0)
    assert bool(jnp.any(resumed.pose_adjust.embeds.embedding[1] != 0.0))
    for saved_node, restored_node in zip(
        snapshots["saved"], snapshots["restored"], strict=True
    ):
        for saved_leaf, restored_leaf in zip(
            saved_node, restored_node, strict=True
        ):
            np.testing.assert_array_equal(restored_leaf, saved_leaf)
    for uninterrupted_node, resumed_node in zip(
        uninterrupted_final, resumed_final, strict=True
    ):
        for uninterrupted_leaf, resumed_leaf in zip(
            uninterrupted_node, resumed_node, strict=True
        ):
            np.testing.assert_array_equal(resumed_leaf, uninterrupted_leaf)


@pytest.mark.parametrize("has_scene_metadata", [True, False])
def test_resume_fast_forwards_batches_and_derives_keys_from_absolute_step(
    monkeypatch, tmp_path, has_scene_metadata: bool
):
    scene = SimpleNamespace(
        camtoworlds=np.eye(4, dtype=np.float32)[None],
    )
    batches = [
        {
            "image": np.full((1, 4, 4, 3), value, np.float32),
            "K": np.eye(3, dtype=np.float32)[None],
            "w2c": np.eye(4, dtype=np.float32)[None],
        }
        for value in (0.0, 1.0, 2.0)
    ]
    config = TrainConfig(
        model=ModelConfig(capacity=2, bucket_min_capacity=2, sh_degree=0),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(root="unused", patch_size=4, num_workers=1),
        steps=3,
        checkpoint_every=0,
        eval_every=0,
        output_dir=str(tmp_path),
    )
    dispatched = {}
    scene_matrix = np.eye(4, dtype=np.float64)
    scene_matrix[:3, :3] *= 0.5
    scene_matrix[0, 3] = 1.0
    saved_scene_metadata = []

    def fake_make_train_step(_config):
        def fake_train_step(
            model, _optimizer, _strategy, safety_state, images, *_args
        ):
            dispatched["image"] = np.asarray(images).copy()
            dispatched["viewmats"] = np.asarray(_args[1]).copy()
            dispatched["key"] = np.asarray(
                jax.random.key_data(_args[2])
            ).copy()
            return {
                "loss": jnp.asarray(0.0),
                "l1": jnp.asarray(0.0),
                "ssim": jnp.asarray(1.0),
                "psnr": jnp.asarray(100.0),
                "active_count": model.active_count,
                "visible_count": jnp.asarray(1),
                "overflow_tiles": jnp.asarray(0),
                "max_overflow_tiles": safety_state.max_overflow_tiles[...],
                "candidate_limit_exceeded_tiles": jnp.asarray(0),
                "busiest_tile_candidates": jnp.asarray(0),
                "intersection_overflow": jnp.asarray(False),
                "intersection_overflow_seen": (
                    safety_state.intersection_overflow_seen[...]
                ),
                "intersection_count": jnp.asarray(1),
                "intersection_required_count": jnp.asarray(1),
            }

        return fake_train_step

    monkeypatch.setattr(training_module, "load_colmap_scene", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        training_module, "load_checkpoint_config", lambda _path: config
    )
    monkeypatch.setattr(
        training_module, "load_checkpoint_storage_capacity", lambda _path: 2
    )
    monkeypatch.setattr(
        training_module, "load_checkpoint_intersection_capacity", lambda _path: None
    )
    monkeypatch.setattr(
        training_module,
        "load_checkpoint_scene_transform",
        lambda _path: (scene_matrix, 3.25) if has_scene_metadata else None,
    )
    monkeypatch.setattr(training_module, "restore_checkpoint", lambda *_a, **_k: 2)
    monkeypatch.setattr(
        training_module, "load_checkpoint_active_prefix", lambda _path: True
    )
    yielded = {"count": 0}

    class CountingBatches:
        def __iter__(self):
            for batch in batches:
                yielded["count"] += 1
                yield batch

    monkeypatch.setattr(
        training_module,
        "create_grain_dataset",
        lambda *_a, **_k: CountingBatches(),
    )
    monkeypatch.setattr(training_module, "make_train_step", fake_make_train_step)
    monkeypatch.setattr(training_module, "_check_memory_budget", lambda *_a, **_k: 0)

    def fake_save(directory, *_args, step, **kwargs):
        saved_scene_metadata.append(
            (kwargs["scene_transform"].matrix.copy(), kwargs["scene_scale"])
        )
        return directory / f"step_{step:08d}"

    monkeypatch.setattr(
        training_module, "_save_compacted_training_checkpoint", fake_save
    )

    training_module.train(config, resume_from=tmp_path / "checkpoint")

    np.testing.assert_array_equal(dispatched["image"], batches[2]["image"])
    expected_transform = (
        training_module.SceneTransform(scene_matrix)
        if has_scene_metadata
        else training_module._legacy_scene_transform(scene)
    )
    expected_viewmats = expected_transform.world_to_camera(batches[2]["w2c"])
    np.testing.assert_allclose(dispatched["viewmats"], expected_viewmats)
    np.testing.assert_array_equal(
        saved_scene_metadata[0][0], expected_transform.matrix
    )
    expected_scale = (
        3.25
        if has_scene_metadata
        else training_module._legacy_training_scene_scale(scene)
    )
    assert saved_scene_metadata[0][1] == expected_scale
    expected_key = jax.random.split(
        jax.random.fold_in(jax.random.key(config.seed), 3), 2
    )[0]
    np.testing.assert_array_equal(
        dispatched["key"], np.asarray(jax.random.key_data(expected_key))
    )
    assert yielded["count"] == 3

    yielded["count"] = 0
    monkeypatch.setattr(
        training_module, "restore_checkpoint", lambda *_a, **_k: 3
    )
    training_module.train(config, resume_from=tmp_path / "checkpoint")
    assert yielded["count"] == 0


def test_train_checks_sticky_overflow_before_final_checkpoint(
    monkeypatch, tmp_path
):
    scene = SimpleNamespace(
        points=np.array([[0.0, 0.0, 3.0]], np.float32),
        points_rgb=np.array([[128, 128, 128]], np.uint8),
        camtoworlds=np.eye(4, dtype=np.float32)[None],
    )
    batch = {
        "image": np.zeros((4, 4, 3), np.float32),
        "K": np.array([[10.0, 0, 2], [0, 10.0, 2], [0, 0, 1]], np.float32),
        "w2c": np.eye(4, dtype=np.float32),
    }
    config = TrainConfig(
        normalize_world_space=False,
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(
            root="unused", patch_size=4, batch_size=1, num_workers=1
        ),
        steps=3,
        checkpoint_every=3,
        eval_every=0,
        output_dir=str(tmp_path),
    )
    calls = 0
    save_calls = 0

    def fake_make_train_step(_config):
        def fake_train_step(
            model, _optimizer, _strategy_state, safety_state, *_args
        ):
            nonlocal calls
            calls += 1
            if calls == 2:
                safety_state.max_overflow_tiles[...] = 2
            return {
                "loss": jnp.asarray(0.0),
                "l1": jnp.asarray(0.0),
                "ssim": jnp.asarray(1.0),
                "psnr": jnp.asarray(100.0),
                "active_count": model.active_count,
                "visible_count": jnp.asarray(1),
                "overflow_tiles": jnp.asarray(0),
                "max_overflow_tiles": safety_state.max_overflow_tiles[...],
                "candidate_limit_exceeded_tiles": jnp.asarray(0),
                "busiest_tile_candidates": jnp.asarray(0),
                "intersection_overflow": jnp.asarray(False),
                "intersection_overflow_seen": (
                    safety_state.intersection_overflow_seen[...]
                ),
                "intersection_count": jnp.asarray(1),
                "intersection_required_count": jnp.asarray(1),
            }

        return fake_train_step

    def fake_save(*_args, **_kwargs):
        nonlocal save_calls
        save_calls += 1
        return tmp_path / "unexpected"

    monkeypatch.setattr(training_module, "load_colmap_scene", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        training_module, "create_grain_dataset", lambda *_a, **_k: [batch]
    )
    monkeypatch.setattr(training_module, "make_train_step", fake_make_train_step)
    monkeypatch.setattr(training_module, "save_checkpoint", fake_save)
    monkeypatch.setattr(training_module, "_check_memory_budget", lambda *_a, **_k: 0)

    with pytest.raises(RuntimeError, match="tiles=2"):
        training_module.train(config)

    assert calls == 3
    assert save_calls == 0


def test_train_grows_intersection_bucket_and_replays_uncommitted_suffix(
    monkeypatch, tmp_path
):
    class PoseReplayScene(SimpleNamespace):
        def indices(self, split, _test_every):
            assert split == "train"
            return np.asarray([0], dtype=np.int64)

    scene = PoseReplayScene(
        points=np.array([[0.0, 0.0, 3.0]], np.float32),
        points_rgb=np.array([[128, 128, 128]], np.uint8),
        camtoworlds=np.eye(4, dtype=np.float32)[None],
        images=(SimpleNamespace(name="only.png"),),
    )
    batch = {
        "image": np.zeros((4, 4, 3), np.float32),
        "K": np.array([[10.0, 0, 2], [0, 10.0, 2], [0, 0, 1]], np.float32),
        "w2c": np.eye(4, dtype=np.float32),
        "dataset_index": np.asarray(0, np.int32),
    }
    config = TrainConfig(
        normalize_world_space=False,
        app_opt=True,
        pose_opt=True,
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(
            root="unused", patch_size=4, batch_size=1, num_workers=1
        ),
        rasterizer=RasterizationConfig(max_intersections=2_048),
        steps=4,
        checkpoint_every=0,
        eval_every=0,
        intersection_bucket_min_capacity=512,
        output_dir=str(tmp_path),
    )
    factory_capacities = []
    keys_by_capacity = {512: [], 1_024: []}
    strategy_keys_by_capacity = {512: [], 1_024: []}
    accepted_keys = []
    saved_intersection_capacities = []
    saved_pose_steps = []
    saved_appearance_steps = []
    image_ids_by_capacity = {512: [], 1_024: []}

    def fake_make_train_step(runtime_config):
        capacity = runtime_config.rasterizer.max_intersections
        factory_capacities.append(capacity)

        def fake_train_step(
            model, _optimizer, _strategy_state, safety_state, *_args, **kwargs
        ):
            key_data = np.asarray(jax.random.key_data(_args[-3])).copy()
            strategy_key_data = np.asarray(
                jax.random.key_data(_args[-1])
            ).copy()
            keys_by_capacity[capacity].append(key_data)
            strategy_keys_by_capacity[capacity].append(strategy_key_data)
            image_ids_by_capacity[capacity].append(
                np.asarray(kwargs["image_ids"]).copy()
            )
            low_capacity_call = len(keys_by_capacity[512])
            sticky_before = bool(
                safety_state.intersection_overflow_seen[...]
            )
            overflow = capacity == 512 and low_capacity_call == 3
            if not sticky_before and not overflow:
                accepted_keys.append((key_data, strategy_key_data))
                kwargs["pose_adjust"].embeds.embedding[0, 0] += 1.0
                kwargs["pose_optimizer"].step[...] += 1
                kwargs["appearance_module"].color_head[-1].bias[0] += 1.0
                kwargs["appearance_optimizer"].step[...] += 1
            safety_state.intersection_overflow_seen[...] = (
                safety_state.intersection_overflow_seen[...] | overflow
            )
            return {
                "loss": jnp.asarray(0.0),
                "l1": jnp.asarray(0.0),
                "ssim": jnp.asarray(1.0),
                "psnr": jnp.asarray(100.0),
                "active_count": model.active_count,
                "visible_count": jnp.asarray(1),
                "overflow_tiles": jnp.asarray(0),
                "max_overflow_tiles": safety_state.max_overflow_tiles[...],
                "candidate_limit_exceeded_tiles": jnp.asarray(0),
                "busiest_tile_candidates": jnp.asarray(0),
                "intersection_overflow": jnp.asarray(overflow),
                "intersection_overflow_seen": (
                    safety_state.intersection_overflow_seen[...]
                ),
                "intersection_count": jnp.asarray(1),
                "intersection_required_count": jnp.asarray(
                    700 if overflow else 1
                ),
            }

        return fake_train_step

    monkeypatch.setattr(training_module, "load_colmap_scene", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        training_module, "create_grain_dataset", lambda *_a, **_k: [batch]
    )
    monkeypatch.setattr(training_module, "make_train_step", fake_make_train_step)
    monkeypatch.setattr(training_module, "_check_memory_budget", lambda *_a, **_k: 0)

    def fake_save(directory, *_args, step, intersection_capacity, **kwargs):
        saved_intersection_capacities.append(intersection_capacity)
        saved_pose_steps.append(int(kwargs["pose_optimizer"].step[...]))
        saved_appearance_steps.append(
            int(kwargs["appearance_optimizer"].step[...])
        )
        assert kwargs["pose_image_names"] == ("only.png",)
        assert kwargs["appearance_image_names"] == ("only.png",)
        return directory / f"step_{step:08d}"

    monkeypatch.setattr(training_module, "save_checkpoint", fake_save)

    result = training_module.train(config)

    # The middle rebuild is the per-tile candidate bound being tuned from the
    # first rendered frame, at an intersection capacity that has not moved.
    # It replays nothing, so the key sequence below is unaffected by it.
    assert factory_capacities == [512, 512, 1_024]
    assert len(keys_by_capacity[512]) == 4
    assert len(keys_by_capacity[1_024]) == 2
    np.testing.assert_array_equal(
        keys_by_capacity[1_024][0], keys_by_capacity[512][2]
    )
    np.testing.assert_array_equal(
        keys_by_capacity[1_024][1], keys_by_capacity[512][3]
    )
    np.testing.assert_array_equal(
        strategy_keys_by_capacity[1_024][0],
        strategy_keys_by_capacity[512][2],
    )
    np.testing.assert_array_equal(
        strategy_keys_by_capacity[1_024][1],
        strategy_keys_by_capacity[512][3],
    )
    np.testing.assert_array_equal(
        image_ids_by_capacity[1_024][0], image_ids_by_capacity[512][2]
    )
    np.testing.assert_array_equal(
        image_ids_by_capacity[1_024][1], image_ids_by_capacity[512][3]
    )
    assert len(accepted_keys) == 4
    for accepted, dispatched in zip(
        accepted_keys,
        zip(
            keys_by_capacity[512],
            strategy_keys_by_capacity[512],
            strict=True,
        ),
        strict=True,
    ):
        for accepted_key, dispatched_key in zip(
            accepted, dispatched, strict=True
        ):
            np.testing.assert_array_equal(accepted_key, dispatched_key)
    assert saved_intersection_capacities == [1_024]
    assert saved_pose_steps == [4]
    assert saved_appearance_steps == [4]
    np.testing.assert_array_equal(
        result.pose_adjust.embeds.embedding[0, 0], 4.0
    )
    np.testing.assert_array_equal(
        result.appearance.color_head[-1].bias[0], 4.0
    )
    assert result.final_step == 4


def test_scheduled_mcmc_grows_before_forward_and_skips_host_refine(
    monkeypatch, tmp_path
):
    point_count = 20
    scene = SimpleNamespace(
        points=np.column_stack(
            (
                np.linspace(-0.2, 0.2, point_count, dtype=np.float32),
                np.zeros((point_count,), np.float32),
                np.full((point_count,), 3.0, np.float32),
            )
        ),
        points_rgb=np.full((point_count, 3), 128, np.uint8),
        camtoworlds=np.eye(4, dtype=np.float32)[None],
    )
    batch = {
        "image": np.zeros((4, 4, 3), np.float32),
        "K": np.array([[10.0, 0, 2], [0, 10.0, 2], [0, 0, 1]], np.float32),
        "w2c": np.eye(4, dtype=np.float32),
    }
    config = TrainConfig(
        normalize_world_space=False,
        model=ModelConfig(capacity=40, bucket_min_capacity=20, sh_degree=0),
        optimizer=OptimizerConfig(max_steps=1),
        strategy=StrategyConfig(
            kind="mcmc",
            refine_start=0,
            refine_stop=2,
            refine_every=1,
            cap_max=40,
            max_new_per_refine=1,
            noise_lr=0.0,
        ),
        data=DataConfig(
            root="unused", patch_size=4, batch_size=1, num_workers=1
        ),
        rasterizer=RasterizationConfig(max_intersections=2_048),
        steps=1,
        checkpoint_every=0,
        eval_every=0,
        intersection_bucket_min_capacity=1,
        output_dir=str(tmp_path),
    )
    capacities_seen = []
    growths = []
    original_grow = training_module._grow_training_state

    def tracked_grow(*args, **kwargs):
        growths.append((args[1].capacity, args[-1]))
        return original_grow(*args, **kwargs)

    def fake_make_train_step(_runtime_config):
        def fake_train_step(
            model,
            _optimizer,
            _strategy_state,
            safety_state,
            *_args,
        ):
            capacities_seen.append(model.capacity)
            assert _args[-1] is not None
            return {
                "loss": jnp.asarray(0.0),
                "l1": jnp.asarray(0.0),
                "ssim": jnp.asarray(1.0),
                "psnr": jnp.asarray(100.0),
                "active_count": model.active_count,
                "visible_count": jnp.asarray(1),
                "overflow_tiles": jnp.asarray(0),
                "max_overflow_tiles": safety_state.max_overflow_tiles[...],
                "candidate_limit_exceeded_tiles": jnp.asarray(0),
                "busiest_tile_candidates": jnp.asarray(0),
                "intersection_overflow": jnp.asarray(False),
                "intersection_overflow_seen": (
                    safety_state.intersection_overflow_seen[...]
                ),
                "intersection_count": jnp.asarray(1),
                "intersection_required_count": jnp.asarray(1),
            }

        return fake_train_step

    def unexpected_host_refine(*_args, **_kwargs):
        raise AssertionError("scheduled MCMC refinement must be device-atomic")

    monkeypatch.setattr(training_module, "load_colmap_scene", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        training_module, "create_grain_dataset", lambda *_a, **_k: [batch]
    )
    monkeypatch.setattr(training_module, "make_train_step", fake_make_train_step)
    monkeypatch.setattr(training_module, "_grow_training_state", tracked_grow)
    monkeypatch.setattr(
        training_module,
        "_check_bucket_transition_memory_budget",
        lambda *_a, **_k: 0,
    )
    monkeypatch.setattr(training_module, "_check_memory_budget", lambda *_a, **_k: 0)
    monkeypatch.setattr(MCMCStrategy, "refine", unexpected_host_refine)
    monkeypatch.setattr(
        training_module,
        "_save_compacted_training_checkpoint",
        lambda directory, *_a, step, **_k: directory / f"step_{step:08d}",
    )

    result = training_module.train(config)

    assert growths == [(20, 40)]
    assert capacities_seen == [40]
    assert result.model.capacity == 40


def test_train_resume_reuses_checkpoint_intersection_high_water(
    monkeypatch, tmp_path
):
    scene = SimpleNamespace(
        camtoworlds=np.eye(4, dtype=np.float32)[None],
    )
    config = TrainConfig(
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        data=DataConfig(root="unused", patch_size=4, num_workers=1),
        rasterizer=RasterizationConfig(max_intersections=2_048),
        steps=0,
        eval_every=0,
        intersection_bucket_min_capacity=512,
        output_dir=str(tmp_path / "output"),
    )
    factory_capacities = []
    saved_values = []

    monkeypatch.setattr(training_module, "load_colmap_scene", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        training_module, "load_checkpoint_config", lambda _path: config
    )
    monkeypatch.setattr(
        training_module, "load_checkpoint_storage_capacity", lambda _path: 4
    )
    monkeypatch.setattr(
        training_module,
        "load_checkpoint_intersection_capacity",
        lambda _path: 1_024,
    )
    monkeypatch.setattr(training_module, "restore_checkpoint", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        training_module, "load_checkpoint_active_prefix", lambda _path: True
    )
    monkeypatch.setattr(
        training_module, "create_grain_dataset", lambda *_a, **_k: []
    )
    monkeypatch.setattr(training_module, "_check_memory_budget", lambda *_a, **_k: 0)

    def fake_make_train_step(runtime_config):
        factory_capacities.append(
            runtime_config.rasterizer.max_intersections
        )
        return lambda *_args: {}

    def fake_save(
        directory,
        *_args,
        step,
        config,
        intersection_capacity,
        **_kwargs,
    ):
        saved_values.append(
            (config.rasterizer.max_intersections, intersection_capacity)
        )
        return directory / f"step_{step:08d}"

    monkeypatch.setattr(training_module, "make_train_step", fake_make_train_step)
    monkeypatch.setattr(training_module, "save_checkpoint", fake_save)

    training_module.train(config, resume_from=tmp_path / "checkpoint")

    assert factory_capacities == [1_024]
    assert saved_values == [(2_048, 1_024)]


def test_train_rejects_checkpoint_newer_than_target(monkeypatch, tmp_path):
    scene = SimpleNamespace(
        camtoworlds=np.eye(4, dtype=np.float32)[None],
    )
    config = TrainConfig(
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        data=DataConfig(root="unused", patch_size=4, num_workers=1),
        steps=2,
        eval_every=0,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(training_module, "load_colmap_scene", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        training_module, "load_checkpoint_config", lambda _path: config
    )
    monkeypatch.setattr(
        training_module, "load_checkpoint_storage_capacity", lambda _path: 4
    )
    monkeypatch.setattr(
        training_module, "restore_checkpoint", lambda *_a, **_k: 3
    )
    monkeypatch.setattr(training_module, "_check_memory_budget", lambda *_a, **_k: 0)

    def unexpected_dataset(*_args, **_kwargs):
        raise AssertionError("dataset creation must not run")

    monkeypatch.setattr(
        training_module,
        "create_grain_dataset",
        unexpected_dataset,
    )

    with pytest.raises(ValueError, match="checkpoint step 3"):
        training_module.train(config, resume_from=tmp_path / "checkpoint")


def test_train_step_reports_the_busiest_tile_and_honours_a_candidate_bound():
    # The bound is a static promise the caller makes; the metric is how a
    # caller learns what to promise. Without the metric the only options are
    # guessing or leaving the chunk loop sized for the worst case the shapes
    # allow.
    points = np.array([[0, 0, 3], [0.2, 0, 3], [-0.2, 0.1, 3]], np.float32)
    model_config = ModelConfig(capacity=32, sh_degree=1, initial_scale=0.1)
    optimizer_config = OptimizerConfig(max_steps=10)
    strategy_config = StrategyConfig(refine_start=100, max_new_per_refine=4)

    def run(bound):
        config = TrainConfig(
            model=model_config,
            optimizer=optimizer_config,
            strategy=strategy_config,
            data=DataConfig(root="unused", patch_size=16, batch_size=1),
            rasterizer=RasterizationConfig(
                tile_size=8,
                max_gaussians_per_tile=4,
                tile_batch_size=2,
                max_candidates_per_tile=bound,
            ),
            steps=1,
            eval_every=0,
            checkpoint_every=0,
        )
        model = GaussianModel.from_point_cloud(
            points, np.eye(3, dtype=np.float32), model_config
        )
        optimizer = create_optimizer(model, optimizer_config)
        state = DefaultStrategy(strategy_config).initialize_state(32)
        return make_train_step(config)(
            model,
            optimizer,
            state,
            TrainingSafetyState(),
            jnp.zeros((1, 32, 32, 3), jnp.float32),
            jnp.array([[[30.0, 0, 16], [0, 30.0, 16], [0, 0, 1]]], jnp.float32),
            jnp.eye(4, dtype=jnp.float32)[None],
            jax.random.key(0),
            jnp.asarray(1),
        )

    unbounded = run(None)
    busiest = int(unbounded["busiest_tile_candidates"])
    # Three Gaussians, so no tile can hold more; the metric reports the real
    # occupancy rather than the loop's conservative length.
    assert 0 < busiest <= 3
    assert int(unbounded["overflow_tiles"]) == 0

    # Promising exactly what the frame needs renders the same and stays clean.
    bounded = run(busiest)
    assert int(bounded["busiest_tile_candidates"]) == busiest
    assert int(bounded["overflow_tiles"]) == 0
    np.testing.assert_allclose(
        float(bounded["loss"]), float(unbounded["loss"]), rtol=0.0, atol=1e-6
    )


@pytest.mark.parametrize("bound,shown", [(None, 512), (128, 128)])
def test_train_reports_the_busiest_tile_against_the_bound_in_effect(
    monkeypatch, tmp_path, capsys, bound, shown
):
    # A caller can only choose --max-candidates-per-tile by watching what the
    # scene actually does, so the progress line has to show it -- against the
    # bound the compositor is compiled for, which for an unset one is the
    # value tuned from the first frame (37 candidates round up to one 512
    # chunk), not the None that was configured.
    scene = SimpleNamespace(
        points=np.array([[0.0, 0.0, 3.0]], np.float32),
        points_rgb=np.array([[128, 128, 128]], np.uint8),
        camtoworlds=np.eye(4, dtype=np.float32)[None],
    )
    batch = {
        "image": np.zeros((4, 4, 3), np.float32),
        "K": np.array([[10.0, 0, 2], [0, 10.0, 2], [0, 0, 1]], np.float32),
        "w2c": np.eye(4, dtype=np.float32),
    }
    config = TrainConfig(
        normalize_world_space=False,
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(root="unused", patch_size=4, batch_size=1, num_workers=1),
        rasterizer=RasterizationConfig(max_candidates_per_tile=bound),
        steps=1,
        checkpoint_every=0,
        eval_every=0,
        output_dir=str(tmp_path),
    )

    def fake_make_train_step(_config):
        def fake_train_step(model, _optimizer, _strategy_state, _safety, *_args):
            return {
                "loss": jnp.asarray(0.0),
                "l1": jnp.asarray(0.0),
                "ssim": jnp.asarray(1.0),
                "psnr": jnp.asarray(100.0),
                "active_count": model.active_count,
                "visible_count": jnp.asarray(1),
                "overflow_tiles": jnp.asarray(0),
                "max_overflow_tiles": jnp.asarray(0),
                "candidate_limit_exceeded_tiles": jnp.asarray(0),
                "busiest_tile_candidates": jnp.asarray(37),
                "intersection_overflow": jnp.asarray(False),
                "intersection_overflow_seen": jnp.asarray(False),
                "intersection_count": jnp.asarray(1),
                "intersection_required_count": jnp.asarray(1),
            }

        return fake_train_step

    monkeypatch.setattr(training_module, "load_colmap_scene", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        training_module, "create_grain_dataset", lambda *_a, **_k: [batch]
    )
    monkeypatch.setattr(training_module, "make_train_step", fake_make_train_step)
    monkeypatch.setattr(training_module, "_check_memory_budget", lambda *_a, **_k: 0)

    training_module.train(config)

    expected = f"busiest_tile=37/{shown}"
    line = next(
        text
        for text in capsys.readouterr().out.splitlines()
        if text.startswith("step=")
    )
    assert expected in line
    # The count must stay a standalone field rather than run into the next one.
    assert f"{expected} " in line


def test_candidate_bound_rounds_occupancy_to_a_doubling_chunk_count():
    # Only ceil(bound / max_gaussians_per_tile) is observable, so the bound is
    # a whole number of chunks, and that count is a power of two so a scene
    # that densifies has to double its occupancy before recompiling again.
    bound = training_module._candidate_bound_for_occupancy
    assert bound(2_003, 512) == 2_048
    assert bound(2_049, 512) == 4_096
    assert bound(1, 512) == 512
    assert bound(0, 512) == 512
    # Exactly on a chunk boundary must not round up to a wasted chunk.
    assert bound(1_024, 512) == 1_024


def _bound_growth_scene_and_batch():
    scene = SimpleNamespace(
        points=np.array([[0.0, 0.0, 3.0]], np.float32),
        points_rgb=np.array([[128, 128, 128]], np.uint8),
        camtoworlds=np.eye(4, dtype=np.float32)[None],
    )
    batch = {
        "image": np.zeros((4, 4, 3), np.float32),
        "K": np.array([[10.0, 0, 2], [0, 10.0, 2], [0, 0, 1]], np.float32),
        "w2c": np.eye(4, dtype=np.float32),
    }
    return scene, batch


def _bound_growth_config(tmp_path, bound, *, steps=2):
    return TrainConfig(
        normalize_world_space=False,
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(root="unused", patch_size=4, batch_size=1, num_workers=1),
        rasterizer=RasterizationConfig(
            max_gaussians_per_tile=512, max_candidates_per_tile=bound
        ),
        steps=steps,
        checkpoint_every=0,
        eval_every=0,
        output_dir=str(tmp_path),
    )


def _bound_growth_factory(bounds, busiest, *, replays=None):
    """A train step that overflows until the bound covers ``busiest``."""

    def fake_make_train_step(config):
        bound = config.rasterizer.max_candidates_per_tile
        bounds.append(bound)

        def fake_train_step(model, _optimizer, _strategy_state, safety, *_args):
            if replays is not None:
                replays.append(bound)
            outgrown = bound is not None and bound < busiest
            if outgrown:
                safety.max_overflow_tiles[...] = 3
            return {
                "loss": jnp.asarray(0.0),
                "l1": jnp.asarray(0.0),
                "ssim": jnp.asarray(1.0),
                "psnr": jnp.asarray(100.0),
                "active_count": model.active_count,
                "visible_count": jnp.asarray(1),
                "overflow_tiles": jnp.asarray(3 if outgrown else 0),
                "max_overflow_tiles": safety.max_overflow_tiles[...],
                "candidate_limit_exceeded_tiles": jnp.asarray(0),
                "busiest_tile_candidates": jnp.asarray(busiest),
                "intersection_overflow": jnp.asarray(False),
                "intersection_overflow_seen": jnp.asarray(False),
                "intersection_count": jnp.asarray(1),
                "intersection_required_count": jnp.asarray(1),
            }

        return fake_train_step

    return fake_make_train_step


def _patch_bound_growth_trainer(monkeypatch, scene, batch, factory):
    monkeypatch.setattr(training_module, "load_colmap_scene", lambda *_a, **_k: scene)
    monkeypatch.setattr(
        training_module, "create_grain_dataset", lambda *_a, **_k: [batch]
    )
    monkeypatch.setattr(training_module, "make_train_step", factory)
    monkeypatch.setattr(training_module, "_check_memory_budget", lambda *_a, **_k: 0)


def test_train_grows_an_outgrown_candidate_bound_instead_of_failing(
    monkeypatch, tmp_path
):
    # The bound is a promise about the busiest tile, and a scene that
    # densifies outgrows it. Before this it killed the run; it now costs a
    # recompile and a replay, exactly like an outgrown intersection buffer.
    scene, batch = _bound_growth_scene_and_batch()
    bounds, replays = [], []
    _patch_bound_growth_trainer(
        monkeypatch, scene, batch,
        _bound_growth_factory(bounds, 900, replays=replays),
    )

    training_module.train(_bound_growth_config(tmp_path, 512))

    # 900 candidates need two chunks of 512, so the bound doubles once.
    assert bounds == [512, 1_024]
    # The offending step is re-run under the grown bound rather than lost.
    assert replays.count(1_024) >= 1


def test_train_still_refuses_an_overflow_no_larger_bound_can_fix(
    monkeypatch, tmp_path
):
    # A tile claiming fewer candidates than the bound already covers cannot be
    # rescued by growing it: the input is malformed, and rendering it would
    # truncate gradients silently. That case has to stay fatal.
    scene, batch = _bound_growth_scene_and_batch()
    bounds = []
    factory = _bound_growth_factory(bounds, 100)

    def always_overflowing(config):
        inner = factory(config)

        def step(model, optimizer, strategy_state, safety, *args):
            metrics = inner(model, optimizer, strategy_state, safety, *args)
            safety.max_overflow_tiles[...] = 3
            return {**metrics, "overflow_tiles": jnp.asarray(3),
                    "max_overflow_tiles": safety.max_overflow_tiles[...]}

        return step

    _patch_bound_growth_trainer(monkeypatch, scene, batch, always_overflowing)

    with pytest.raises(RuntimeError, match="tiles=3"):
        training_module.train(_bound_growth_config(tmp_path, 512))
    # It refused rather than growing forever.
    assert bounds == [512]


def test_train_tunes_an_unset_candidate_bound_from_the_first_frame(
    monkeypatch, tmp_path
):
    # Nothing tighter than the shape-derived bound is knowable before a frame
    # has been rendered, so an unset bound starts loose and tightens once.
    scene, batch = _bound_growth_scene_and_batch()
    bounds = []
    _patch_bound_growth_trainer(
        monkeypatch, scene, batch, _bound_growth_factory(bounds, 900)
    )

    training_module.train(_bound_growth_config(tmp_path, None, steps=3))

    # Loose to begin with, then tuned to cover the 900 it actually saw, and
    # not touched again -- a bound that chased occupancy would recompile on
    # every fluctuation.
    assert bounds == [None, 1_024]
