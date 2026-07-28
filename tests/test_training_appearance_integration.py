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
from jax_gs.training import TrainingSafetyState, make_train_step
from jax_gs.training.appearance import (
    APPEARANCE_FEATURE_DIM,
    AppearanceOptModule,
    create_appearance_optimizer,
)


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


def _overflow_rasterization(
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
    signal = 0.25 + 0.0 * jnp.sum(means) + 0.0 * jnp.sum(screen_probe)
    renders = jnp.full(
        (camera_count, height, width, 3), signal, dtype=means.dtype
    )
    alphas = jnp.ones((camera_count, height, width, 1), dtype=means.dtype)
    info = {
        "radii": jnp.ones((camera_count, capacity, 2), means.dtype),
        "valid": jnp.ones((camera_count, capacity), jnp.bool_),
        "tile_overflow": jnp.zeros((camera_count, 1, 1), jnp.bool_),
        "candidate_limit_exceeded": jnp.zeros(
            (camera_count, 1, 1), jnp.bool_
        ),
        "intersection_overflow": jnp.ones((camera_count,), jnp.bool_),
        "intersection_count": jnp.ones((camera_count,), jnp.int32),
        "intersection_required_count": jnp.full(
            (camera_count,), 2, jnp.int32
        ),
    }
    return renders, alphas, info


@pytest.mark.parametrize(
    ("model_type", "gradient_key"),
    [("3dgs", "means2d"), ("2dgs", "gradient_2dgs")],
)
def test_real_rasterizers_train_features_colors_and_selected_camera(
    model_type, gradient_key
):
    config = TrainConfig(
        app_opt=True,
        app_embed_dim=4,
        app_opt_lr=1.0e-2,
        app_opt_reg=0.0,
        model_type=model_type,
        model=ModelConfig(
            capacity=2,
            bucket_min_capacity=2,
            sh_degree=1,
            initial_scale=0.2,
        ),
        optimizer=OptimizerConfig(max_steps=3),
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
        steps=3,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.3, 0.1, 3.0]], np.float32),
        np.asarray([[192, 128, 64]], np.uint8),
        config.model,
        appearance_feature_dim=APPEARANCE_FEATURE_DIM,
        feature_key=jax.random.key(11),
    )
    optimizer = create_optimizer(model, config.optimizer)
    strategy_state = DefaultStrategy(config.strategy).initialize_state(
        model.capacity
    )
    appearance = AppearanceOptModule(
        2,
        APPEARANCE_FEATURE_DIM,
        config.app_embed_dim,
        config.model.sh_degree,
        rngs=nnx.Rngs(12),
    )
    appearance_optimizer = create_appearance_optimizer(appearance, config)
    train_step = make_train_step(config)
    features_before = np.asarray(model.features[...]).copy()
    colors_before = np.asarray(model.colors[...]).copy()
    embedding_before = np.asarray(appearance.embeds.embedding[...]).copy()
    inputs = dict(
        images=jnp.zeros((1, 16, 16, 3), jnp.float32),
        intrinsics=jnp.asarray(
            [[[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]]]
        ),
        viewmats=jnp.eye(4, dtype=jnp.float32)[None],
        sh_degree=jnp.asarray(1),
        appearance_module=appearance,
        appearance_optimizer=appearance_optimizer,
        camtoworlds=jnp.eye(4, dtype=jnp.float32)[None],
        image_ids=jnp.asarray([1], dtype=jnp.int32),
    )

    first = train_step(
        model,
        optimizer,
        strategy_state,
        TrainingSafetyState(),
        key=jax.random.key(0),
        **inputs,
    )
    second = train_step(
        model,
        optimizer,
        strategy_state,
        TrainingSafetyState(),
        key=jax.random.key(1),
        **inputs,
    )

    assert np.isfinite(float(first["loss"]))
    assert np.isfinite(float(second["loss"]))
    assert not np.array_equal(model.colors[...], colors_before)
    assert not np.array_equal(model.features[...], features_before)
    np.testing.assert_array_equal(
        appearance.embeds.embedding[0], embedding_before[0]
    )
    assert not np.array_equal(
        appearance.embeds.embedding[1], embedding_before[1]
    )
    assert int(optimizer.step[...]) == 2
    assert int(appearance_optimizer.step[...]) == 2


def test_overflow_atomically_skips_shared_appearance_module(monkeypatch):
    config = TrainConfig(
        app_opt=True,
        model=ModelConfig(capacity=2, bucket_min_capacity=2, sh_degree=0),
        optimizer=OptimizerConfig(max_steps=2),
        strategy=StrategyConfig(refine_start=100, max_new_per_refine=1),
        data=DataConfig(root="unused", patch_size=4, batch_size=1),
        rasterizer=RasterizationConfig(backend="jax", max_intersections=8),
        ssim_lambda=0.0,
        steps=2,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.0, 0.0, 3.0]], np.float32),
        np.asarray([[128, 128, 128]], np.uint8),
        config.model,
        appearance_feature_dim=APPEARANCE_FEATURE_DIM,
        feature_key=jax.random.key(13),
    )
    optimizer = create_optimizer(model, config.optimizer)
    state = DefaultStrategy(config.strategy).initialize_state(model.capacity)
    appearance = AppearanceOptModule(
        1, APPEARANCE_FEATURE_DIM, rngs=nnx.Rngs(14)
    )
    appearance_optimizer = create_appearance_optimizer(appearance, config)
    monkeypatch.setattr(training_module, "rasterization", _overflow_rasterization)
    before = _snapshot_array_state(appearance, appearance_optimizer)

    metrics = make_train_step(config)(
        model,
        optimizer,
        state,
        TrainingSafetyState(),
        jnp.zeros((1, 4, 4, 3), jnp.float32),
        jnp.asarray(
            [[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]]
        ),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(0),
        appearance_module=appearance,
        appearance_optimizer=appearance_optimizer,
        camtoworlds=jnp.eye(4, dtype=jnp.float32)[None],
        image_ids=jnp.asarray([0], dtype=jnp.int32),
    )

    assert bool(metrics["intersection_overflow"])
    assert int(appearance_optimizer.step[...]) == 0
    after = _snapshot_array_state(appearance, appearance_optimizer)
    for before_node, after_node in zip(before, after, strict=True):
        for before_leaf, after_leaf in zip(before_node, after_node, strict=True):
            np.testing.assert_array_equal(after_leaf, before_leaf)
