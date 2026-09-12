# pyright: reportMissingImports=false

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

import jax_gs.checkpoints as checkpoints_module
from jax_gs.capacity import (
    _shrink_compacted_training_state,
    compact_training_state,
    resize_training_state,
)
from jax_gs.checkpoints import (
    load_checkpoint_appearance_image_names,
    load_checkpoint_candidate_bound,
    load_checkpoint_config,
    load_checkpoint_intersection_capacity,
    load_checkpoint_storage_capacity,
    restore_checkpoint,
    save_checkpoint,
)
from jax_gs.config import (
    MAX_MODEL_CAPACITY,
    ModelConfig,
    OptimizerConfig,
    TrainConfig,
)
from jax_gs.model import GaussianModel, inverse_sigmoid
from jax_gs.optimizers import create_optimizer
from jax_gs.strategy import StrategyState
from jax_gs.training.appearance import (
    AppearanceOptModule,
    create_appearance_optimizer,
)
from jax_gs.training.pose import CameraOptModule


def _snapshot_state(node: object) -> dict[str, np.ndarray]:
    state = nnx.as_pure(nnx.state(node))
    leaves, _ = jax.tree_util.tree_flatten_with_path(state)

    def copy_array(value: jax.Array) -> np.ndarray:
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jax.random.key_data(value)
        return np.asarray(value).copy()

    return {str(path): copy_array(value) for path, value in leaves}


def _seed_adam_moments(model: GaussianModel, optimizer: nnx.Optimizer) -> None:
    parameters = nnx.state(model, nnx.Param)

    def make_gradient(value: jax.Array) -> jax.Array:
        rows = jnp.arange(1, value.shape[0] + 1, dtype=value.dtype)
        rows = rows.reshape((value.shape[0],) + (1,) * (value.ndim - 1))
        return jnp.broadcast_to(rows, value.shape)

    optimizer.update(model, jax.tree.map(make_gradient, parameters))


def _pose_training_state(
    camera_count: int,
) -> tuple[CameraOptModule, nnx.Optimizer]:
    module = CameraOptModule(camera_count, rngs=nnx.Rngs(123))
    module.embeds.embedding[...] = (
        jnp.arange(camera_count * 9, dtype=jnp.float32).reshape(camera_count, 9) / 100.0
    )
    optimizer = nnx.Optimizer(module, optax.adam(1.0e-3), wrt=nnx.Param)
    gradients = jax.tree.map(jnp.ones_like, nnx.state(module, nnx.Param))
    optimizer.update(module, gradients)
    return module, optimizer


def _appearance_training_state(
    camera_count: int, config: TrainConfig
) -> tuple[AppearanceOptModule, nnx.Optimizer]:
    module = AppearanceOptModule(
        camera_count,
        32,
        config.app_embed_dim,
        config.model.sh_degree,
        rngs=nnx.Rngs(321),
    )
    module.embeds.embedding[...] = 0.5
    optimizer = create_appearance_optimizer(module, config)
    gradients = jax.tree.map(jnp.ones_like, nnx.state(module, nnx.Param))
    optimizer.update(module, gradients)
    return module, optimizer


@pytest.mark.parametrize(
    ("required", "expected"),
    [
        (0, 16),
        (16, 16),
        (17, 32),
        (32, 32),
        (33, 64),
        (64, 64),
        (65, 100),
        (100, 100),
    ],
)
def test_bucket_capacity_uses_smallest_covering_bucket(
    required: int, expected: int
) -> None:
    config = ModelConfig(capacity=100, bucket_min_capacity=16)
    assert config.bucket_capacity(required) == expected


def test_bucket_capacity_distinguishes_logical_and_physical_capacity() -> None:
    config = ModelConfig(capacity=100, bucket_min_capacity=16, sh_degree=1)
    model = GaussianModel.empty(config)
    assert model.capacity == 16
    assert model.max_capacity == 100

    points = np.zeros((17, 3), np.float32)
    points[:, 0] = np.arange(17, dtype=np.float32)
    point_model = GaussianModel.from_point_cloud(points, np.zeros_like(points), config)
    assert point_model.capacity == 32
    assert point_model.max_capacity == 100
    assert int(point_model.active_count) == 17

    with pytest.raises(ValueError, match="negative"):
        config.bucket_capacity(-1)
    with pytest.raises(ValueError, match="logical maximum"):
        config.bucket_capacity(101)


def test_logical_capacity_defaults_to_one_million_and_caps_at_ten_million():
    assert ModelConfig().capacity == 1_000_000
    assert ModelConfig(capacity=MAX_MODEL_CAPACITY).capacity == 10_000_000
    with pytest.raises(ValueError, match="cannot exceed"):
        ModelConfig(capacity=MAX_MODEL_CAPACITY + 1)

    model = GaussianModel.empty(
        ModelConfig(capacity=1, bucket_min_capacity=1, sh_degree=0)
    )
    state = model.state_dict()
    with pytest.raises(ValueError, match="max_capacity cannot exceed"):
        GaussianModel(
            **state,
            max_capacity=MAX_MODEL_CAPACITY + 1,
        )


def test_compact_training_state_uses_one_stable_permutation() -> None:
    config = ModelConfig(capacity=16, bucket_min_capacity=4, sh_degree=1)
    optimizer_config = OptimizerConfig(max_steps=10)
    model = GaussianModel.empty(config, physical_capacity=8)
    optimizer = create_optimizer(model, optimizer_config)
    _seed_adam_moments(model, optimizer)

    slots = jnp.arange(model.capacity, dtype=jnp.float32)
    model.means[...] = jnp.stack([slots, slots + 10, slots + 20], axis=-1)
    model.log_scales[...] = jnp.stack([slots + 30, slots + 40, slots + 50], axis=-1)
    model.quats[...] = jnp.stack(
        [slots + 60, slots + 70, slots + 80, slots + 90], axis=-1
    )
    model.opacity_logits[...] = slots + 100
    model.sh0[...] = jnp.broadcast_to(slots[:, None, None], model.sh0.shape)
    model.sh_rest[...] = jnp.broadcast_to(
        (slots + 110)[:, None, None], model.sh_rest.shape
    )
    model.active_mask[...] = jnp.array(
        [False, True, False, True, True, False, False, True]
    )

    strategy_state = StrategyState(model.capacity)
    strategy_state.grad_accum[...] = slots + 200
    strategy_state.visible_count[...] = slots + 300
    strategy_state.max_radii[...] = slots + 400
    strategy_state.last_new_count[...] = 3
    strategy_state.last_pruned_count[...] = 2
    strategy_state.capacity_overflow[...] = True

    order = np.array([1, 3, 4, 7, 0, 2, 5, 6])
    model_before = {
        name: np.asarray(value).copy() for name, value in model.state_dict().items()
    }
    optimizer_before = _snapshot_state(optimizer)
    strategy_before = _snapshot_state(strategy_state)

    active_count = compact_training_state(model, optimizer, strategy_state)
    assert int(active_count) == 4

    for name, before in model_before.items():
        np.testing.assert_array_equal(
            np.asarray(model.state_dict()[name]), before[order]
        )

    optimizer_after = _snapshot_state(optimizer)
    for path, before in optimizer_before.items():
        after = optimizer_after[path]
        expected = before[order] if before.ndim and before.shape[0] == 8 else before
        np.testing.assert_array_equal(after, expected)

    strategy_after = _snapshot_state(strategy_state)
    for path, before in strategy_before.items():
        after = strategy_after[path]
        expected = before[order] if before.ndim and before.shape[0] == 8 else before
        np.testing.assert_array_equal(after, expected)


def test_resize_training_state_preserves_prefix_and_initializes_tail() -> None:
    model_config = ModelConfig(
        capacity=16,
        bucket_min_capacity=4,
        sh_degree=1,
        initial_scale=0.25,
        initial_opacity=0.2,
    )
    optimizer_config = OptimizerConfig(max_steps=10)
    model = GaussianModel.empty(model_config, physical_capacity=4)
    model.active_mask[:2] = True
    optimizer = create_optimizer(model, optimizer_config)
    _seed_adam_moments(model, optimizer)

    slots = jnp.arange(4, dtype=jnp.float32)
    model.means[...] = jnp.stack([slots, slots + 1, slots + 2], axis=-1)
    strategy_state = StrategyState(4)
    strategy_state.grad_accum[...] = slots + 10
    strategy_state.visible_count[...] = slots + 20
    strategy_state.max_radii[...] = slots + 30
    strategy_state.last_new_count[...] = 3
    strategy_state.last_pruned_count[...] = 1
    strategy_state.capacity_overflow[...] = True

    model_before = {
        name: np.asarray(value).copy() for name, value in model.state_dict().items()
    }
    optimizer_before = _snapshot_state(optimizer)
    strategy_before = _snapshot_state(strategy_state)

    new_model, new_optimizer, new_strategy_state = resize_training_state(
        model,
        optimizer,
        strategy_state,
        8,
        model_config,
        optimizer_config,
    )

    assert new_model.capacity == 8
    assert new_model.max_capacity == 16
    for name, before in model_before.items():
        np.testing.assert_array_equal(
            np.asarray(new_model.state_dict()[name])[:4], before
        )
    np.testing.assert_array_equal(np.asarray(new_model.means[4:]), 0.0)
    np.testing.assert_allclose(
        np.asarray(new_model.log_scales[4:]), np.log(model_config.initial_scale)
    )
    np.testing.assert_array_equal(
        np.asarray(new_model.quats[4:]),
        np.array([[1.0, 0.0, 0.0, 0.0]] * 4, np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(new_model.opacity_logits[4:]),
        float(inverse_sigmoid(model_config.initial_opacity)),
    )
    np.testing.assert_array_equal(np.asarray(new_model.sh0[4:]), 0.0)
    np.testing.assert_array_equal(np.asarray(new_model.sh_rest[4:]), 0.0)
    np.testing.assert_array_equal(np.asarray(new_model.active_mask[4:]), False)

    optimizer_after = _snapshot_state(new_optimizer)
    for path, before in optimizer_before.items():
        after = optimizer_after[path]
        if before.ndim and before.shape[0] == 4:
            np.testing.assert_array_equal(after[:4], before)
            np.testing.assert_array_equal(after[4:], 0.0)
        else:
            np.testing.assert_array_equal(after, before)

    strategy_after = _snapshot_state(new_strategy_state)
    for path, before in strategy_before.items():
        after = strategy_after[path]
        if before.ndim and before.shape[0] == 4:
            np.testing.assert_array_equal(after[:4], before)
            np.testing.assert_array_equal(after[4:], 0.0)
        else:
            np.testing.assert_array_equal(after, before)

    previous_step = int(new_optimizer.step[...])
    gradients = jax.tree.map(jnp.zeros_like, nnx.state(new_model, nnx.Param))
    new_optimizer.update(new_model, gradients)
    assert int(new_optimizer.step[...]) == previous_step + 1


def test_shrink_compacted_training_state_preserves_prefix() -> None:
    model_config = ModelConfig(capacity=16, bucket_min_capacity=4, sh_degree=1)
    optimizer_config = OptimizerConfig(max_steps=10)
    model = GaussianModel.empty(model_config, physical_capacity=8)
    optimizer = create_optimizer(model, optimizer_config)
    _seed_adam_moments(model, optimizer)
    strategy_state = StrategyState(8)

    slots = jnp.arange(8, dtype=jnp.float32)
    model.means[...] = jnp.stack([slots, slots + 1, slots + 2], axis=-1)
    model.active_mask[...] = jnp.asarray(
        [False, True, False, True, False, True, False, False]
    )
    strategy_state.grad_accum[...] = slots + 10
    strategy_state.visible_count[...] = slots + 20
    strategy_state.max_radii[...] = slots + 30

    active_count = compact_training_state(model, optimizer, strategy_state)
    active_count.block_until_ready()
    model_before = {
        name: np.asarray(value).copy() for name, value in model.state_dict().items()
    }
    optimizer_before = _snapshot_state(optimizer)
    strategy_before = _snapshot_state(strategy_state)

    with pytest.raises(ValueError, match="active rows"):
        _shrink_compacted_training_state(model, optimizer, strategy_state, 4, 5)

    model, optimizer, strategy_state = _shrink_compacted_training_state(
        model, optimizer, strategy_state, 4, int(active_count)
    )

    assert model.capacity == 4
    assert model.max_capacity == 16
    for name, before in model_before.items():
        np.testing.assert_array_equal(model.state_dict()[name], before[:4])
    for before, after in (
        (optimizer_before, _snapshot_state(optimizer)),
        (strategy_before, _snapshot_state(strategy_state)),
    ):
        for path, before_value in before.items():
            after_value = after[path]
            expected = (
                before_value[:4]
                if before_value.ndim and before_value.shape[0] == 8
                else before_value
            )
            np.testing.assert_array_equal(after_value, expected)

    previous_step = int(optimizer.step[...])
    gradients = jax.tree.map(jnp.zeros_like, nnx.state(model, nnx.Param))
    optimizer.update(model, gradients)
    assert int(optimizer.step[...]) == previous_step + 1


def test_appearance_capacity_resize_and_compaction_keep_rows_and_moments_aligned():
    model_config = ModelConfig(capacity=8, bucket_min_capacity=4)
    optimizer_config = OptimizerConfig(max_steps=10)
    model = GaussianModel.empty(
        model_config,
        physical_capacity=4,
        appearance_feature_dim=32,
    )
    slots = jnp.arange(4, dtype=jnp.float32)
    model.features[...] = jnp.broadcast_to(slots[:, None], model.features.shape)
    model.colors[...] = jnp.stack((slots, slots + 10, slots + 20), axis=-1)
    model.active_mask[...] = jnp.asarray([False, True, False, True])
    optimizer = create_optimizer(model, optimizer_config)
    _seed_adam_moments(model, optimizer)
    strategy_state = StrategyState(4)
    features_before = np.asarray(model.features[...]).copy()
    colors_before = np.asarray(model.colors[...]).copy()
    optimizer_before = _snapshot_state(optimizer)

    active_count = compact_training_state(model, optimizer, strategy_state)

    assert int(active_count) == 2
    order = np.asarray([1, 3, 0, 2])
    np.testing.assert_array_equal(model.features[:, 0], features_before[order, 0])
    np.testing.assert_array_equal(model.colors[:, 0], colors_before[order, 0])
    for path, before in optimizer_before.items():
        after = _snapshot_state(optimizer)[path]
        expected = before[order] if before.ndim and before.shape[0] == 4 else before
        np.testing.assert_array_equal(after, expected)

    grown, grown_optimizer, _ = resize_training_state(
        model,
        optimizer,
        strategy_state,
        8,
        model_config,
        optimizer_config,
    )

    assert grown.has_appearance
    np.testing.assert_array_equal(grown.features[:4], model.features[...])
    np.testing.assert_array_equal(grown.colors[:4], model.colors[...])
    np.testing.assert_array_equal(grown.features[4:], 0.0)
    np.testing.assert_array_equal(grown.colors[4:], 0.0)
    for path, before in _snapshot_state(optimizer).items():
        after = _snapshot_state(grown_optimizer)[path]
        if before.ndim and before.shape[0] == 4:
            np.testing.assert_array_equal(after[:4], before)
            np.testing.assert_array_equal(after[4:], 0.0)
        else:
            np.testing.assert_array_equal(after, before)

    shrunk, _, _ = _shrink_compacted_training_state(
        grown, grown_optimizer, strategy_state, 4, 2
    )
    assert shrunk.has_appearance
    np.testing.assert_array_equal(shrunk.features, grown.features[:4])
    np.testing.assert_array_equal(shrunk.colors, grown.colors[:4])


def test_checkpoint_records_physical_capacity_separately_from_logical_maximum(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(capacity=16, bucket_min_capacity=4, sh_degree=1)
    config = TrainConfig(model=model_config)
    model = GaussianModel.empty(model_config, physical_capacity=8)
    model.active_mask[:2] = True
    model.means[:2] = jnp.array([[1, 2, 3], [4, 5, 6]], jnp.float32)

    checkpoint = save_checkpoint(
        tmp_path,
        model,
        step=7,
        config=config,
        intersection_capacity=131_072,
        candidate_bound=2_048,
    )
    metadata = json.loads(
        (checkpoint / "jax_gs_checkpoint.json").read_text(encoding="utf-8")
    )
    assert metadata["storage_capacity"] == 8
    assert metadata["max_capacity"] == 16
    assert metadata["intersection_capacity"] == 131_072
    assert metadata["candidate_bound"] == 2_048
    assert load_checkpoint_storage_capacity(checkpoint) == 8
    assert load_checkpoint_intersection_capacity(checkpoint) == 131_072
    assert load_checkpoint_candidate_bound(checkpoint) == 2_048
    assert load_checkpoint_config(checkpoint).model.capacity == 16

    wrong_shape = GaussianModel.empty(model_config, physical_capacity=4)
    with pytest.raises(ValueError, match="storage capacity is 8"):
        restore_checkpoint(checkpoint, wrong_shape)

    restored = GaussianModel.empty(model_config, physical_capacity=8)
    assert restore_checkpoint(checkpoint, restored) == 7
    assert restored.capacity == 8
    assert restored.max_capacity == 16
    assert int(restored.active_count) == 2
    np.testing.assert_array_equal(
        np.asarray(restored.means[:2]), np.array([[1, 2, 3], [4, 5, 6]])
    )


def test_checkpoint_round_trip_supports_degree_zero_empty_sh_rest(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(capacity=8, bucket_min_capacity=4, sh_degree=0)
    config = TrainConfig(model=model_config)
    model = GaussianModel.empty(model_config, physical_capacity=4)
    model.active_mask[:2] = True
    optimizer = create_optimizer(model, config.optimizer)
    strategy_state = StrategyState(4)
    optimizer.step[...] = 3
    strategy_state.last_new_count[...] = 2

    checkpoint = save_checkpoint(
        tmp_path,
        model,
        step=3,
        optimizer=optimizer,
        strategy_state=strategy_state,
        config=config,
    )
    restored_model = GaussianModel.empty(model_config, physical_capacity=4)
    restored_optimizer = create_optimizer(restored_model, config.optimizer)
    restored_strategy = StrategyState(4)
    step = restore_checkpoint(
        checkpoint,
        restored_model,
        optimizer=restored_optimizer,
        strategy_state=restored_strategy,
    )

    assert step == 3
    assert restored_model.sh_rest.shape == (4, 0, 3)
    assert int(restored_optimizer.step[...]) == 3
    assert int(restored_strategy.last_new_count[...]) == 2


def test_checkpoint_round_trip_restores_pose_module_optimizer_and_manifest(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(capacity=8, bucket_min_capacity=4, sh_degree=0)
    config = TrainConfig(model=model_config)
    model = GaussianModel.empty(model_config, physical_capacity=4)
    optimizer = create_optimizer(model, config.optimizer)
    strategy_state = StrategyState(4)
    pose_module, pose_optimizer = _pose_training_state(3)
    pose_image_names = ("images/a.png", "images/b.png", "images/c.png")
    expected_pose = _snapshot_state(pose_module)
    expected_pose_optimizer = _snapshot_state(pose_optimizer)

    checkpoint = save_checkpoint(
        tmp_path,
        model,
        step=5,
        optimizer=optimizer,
        strategy_state=strategy_state,
        config=config,
        pose_module=pose_module,
        pose_optimizer=pose_optimizer,
        pose_image_names=pose_image_names,
    )

    metadata = json.loads(
        (checkpoint / "jax_gs_checkpoint.json").read_text(encoding="utf-8")
    )
    assert metadata["format_version"] == 6
    assert metadata["components"] == [
        "model",
        "optimizer",
        "strategy",
        "pose",
    ]
    assert metadata["pose_camera_count"] == 3
    assert metadata["pose_image_names"] == list(pose_image_names)

    restored_model = GaussianModel.empty(model_config, physical_capacity=4)
    restored_optimizer = create_optimizer(restored_model, config.optimizer)
    restored_strategy = StrategyState(4)
    restored_pose = CameraOptModule(3, rngs=nnx.Rngs(999))
    restored_pose.zero_init()
    restored_pose_optimizer = nnx.Optimizer(
        restored_pose, optax.adam(1.0e-3), wrt=nnx.Param
    )

    step = restore_checkpoint(
        checkpoint,
        restored_model,
        optimizer=restored_optimizer,
        strategy_state=restored_strategy,
        pose_module=restored_pose,
        pose_optimizer=restored_pose_optimizer,
        pose_image_names=pose_image_names,
    )

    assert step == 5
    for path, expected in expected_pose.items():
        np.testing.assert_array_equal(_snapshot_state(restored_pose)[path], expected)
    for path, expected in expected_pose_optimizer.items():
        np.testing.assert_array_equal(
            _snapshot_state(restored_pose_optimizer)[path], expected
        )

    gradients = jax.tree.map(
        lambda value: (
            jnp.arange(value.size, dtype=value.dtype).reshape(value.shape) / 10.0
        ),
        nnx.state(pose_module, nnx.Param),
    )
    pose_optimizer.update(pose_module, gradients)
    restored_pose_optimizer.update(restored_pose, gradients)
    actual_pose = _snapshot_state(restored_pose)
    actual_pose_optimizer = _snapshot_state(restored_pose_optimizer)
    expected_pose = _snapshot_state(pose_module)
    expected_pose_optimizer = _snapshot_state(pose_optimizer)
    assert actual_pose.keys() == expected_pose.keys()
    assert actual_pose_optimizer.keys() == expected_pose_optimizer.keys()
    for path, expected in expected_pose.items():
        np.testing.assert_array_equal(actual_pose[path], expected)
    for path, expected in expected_pose_optimizer.items():
        np.testing.assert_array_equal(actual_pose_optimizer[path], expected)


def test_v6_checkpoint_restores_appearance_module_optimizer_and_manifest(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(capacity=8, bucket_min_capacity=4, sh_degree=2)
    config = TrainConfig(
        model=model_config,
        app_opt=True,
        app_embed_dim=8,
        data=TrainConfig().data,
    )
    model = GaussianModel.empty(
        model_config,
        physical_capacity=4,
        appearance_feature_dim=32,
    )
    model.active_mask[:2] = True
    model.features[:2] = jnp.arange(64, dtype=jnp.float32).reshape(2, 32)
    model.colors[:2] = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    optimizer = create_optimizer(model, config.optimizer)
    strategy_state = StrategyState(4)
    appearance, appearance_optimizer = _appearance_training_state(3, config)
    names = ("images/a.png", "images/b.png", "images/c.png")
    expected_module = _snapshot_state(appearance)
    expected_optimizer = _snapshot_state(appearance_optimizer)

    checkpoint = save_checkpoint(
        tmp_path,
        model,
        step=4,
        optimizer=optimizer,
        strategy_state=strategy_state,
        config=config,
        appearance_module=appearance,
        appearance_optimizer=appearance_optimizer,
        appearance_image_names=names,
    )

    metadata = json.loads(
        (checkpoint / "jax_gs_checkpoint.json").read_text(encoding="utf-8")
    )
    assert metadata["format_version"] == 6
    assert metadata["model_color_mode"] == "appearance"
    assert metadata["appearance_feature_dim"] == 32
    assert metadata["appearance_camera_count"] == 3
    assert metadata["appearance_image_names"] == list(names)
    assert metadata["components"] == [
        "model",
        "optimizer",
        "strategy",
        "appearance",
    ]
    assert load_checkpoint_appearance_image_names(checkpoint) == names

    restored_model = GaussianModel.empty(
        model_config,
        physical_capacity=4,
        appearance_feature_dim=32,
    )
    restored_optimizer = create_optimizer(restored_model, config.optimizer)
    restored_strategy = StrategyState(4)
    restored_appearance, restored_appearance_optimizer = _appearance_training_state(
        3, config
    )
    assert (
        restore_checkpoint(
            checkpoint,
            restored_model,
            optimizer=restored_optimizer,
            strategy_state=restored_strategy,
            appearance_module=restored_appearance,
            appearance_optimizer=restored_appearance_optimizer,
            appearance_image_names=names,
        )
        == 4
    )
    np.testing.assert_array_equal(restored_model.features, model.features)
    np.testing.assert_array_equal(restored_model.colors, model.colors)
    for path, expected in expected_module.items():
        np.testing.assert_array_equal(
            _snapshot_state(restored_appearance)[path], expected
        )
    for path, expected in expected_optimizer.items():
        np.testing.assert_array_equal(
            _snapshot_state(restored_appearance_optimizer)[path], expected
        )

    model_only = GaussianModel.empty(
        model_config,
        physical_capacity=4,
        appearance_feature_dim=32,
    )
    assert restore_checkpoint(checkpoint, model_only) == 4
    np.testing.assert_array_equal(model_only.features, model.features)
    np.testing.assert_array_equal(model_only.colors, model.colors)

    with pytest.raises(ValueError, match="appearance image names"):
        restore_checkpoint(
            checkpoint,
            restored_model,
            appearance_module=restored_appearance,
            appearance_optimizer=restored_appearance_optimizer,
            appearance_image_names=tuple(reversed(names)),
        )


def test_v6_pose_checkpoint_supports_model_only_restore_and_rejects_reordered_names(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(capacity=8, bucket_min_capacity=4, sh_degree=0)
    model = GaussianModel.empty(model_config, physical_capacity=4)
    model.means[0] = jnp.asarray([1.0, 2.0, 3.0])
    pose_module, pose_optimizer = _pose_training_state(2)
    names = ("first.png", "second.png")
    checkpoint = save_checkpoint(
        tmp_path,
        model,
        step=2,
        pose_module=pose_module,
        pose_optimizer=pose_optimizer,
        pose_image_names=names,
    )

    model_only = GaussianModel.empty(model_config, physical_capacity=4)
    assert restore_checkpoint(checkpoint, model_only) == 2
    np.testing.assert_array_equal(model_only.means[0], [1.0, 2.0, 3.0])

    expected_pose = _snapshot_state(pose_module)
    expected_pose_optimizer = _snapshot_state(pose_optimizer)
    partially_restored_pose = CameraOptModule(2, rngs=nnx.Rngs(999))
    partially_restored_pose.zero_init()
    partially_restored_pose_optimizer = nnx.Optimizer(
        partially_restored_pose, optax.adam(1.0e-3), wrt=nnx.Param
    )
    assert (
        restore_checkpoint(
            checkpoint,
            model_only,
            pose_module=partially_restored_pose,
            pose_optimizer=partially_restored_pose_optimizer,
            pose_image_names=names,
        )
        == 2
    )
    for path, expected in expected_pose.items():
        np.testing.assert_array_equal(
            _snapshot_state(partially_restored_pose)[path], expected
        )
    for path, expected in expected_pose_optimizer.items():
        np.testing.assert_array_equal(
            _snapshot_state(partially_restored_pose_optimizer)[path], expected
        )

    restored_pose, restored_pose_optimizer = _pose_training_state(2)
    with pytest.raises(ValueError, match="pose image names"):
        restore_checkpoint(
            checkpoint,
            model_only,
            pose_module=restored_pose,
            pose_optimizer=restored_pose_optimizer,
            pose_image_names=tuple(reversed(names)),
        )


def test_checkpoint_camera_manifest_supports_fixed_pose_noise_without_optimizer(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0)
    model = GaussianModel.empty(model_config, physical_capacity=4)
    names = ("first.png", "second.png")

    checkpoint = save_checkpoint(
        tmp_path,
        model,
        step=1,
        pose_image_names=names,
    )
    metadata = json.loads(
        (checkpoint / "jax_gs_checkpoint.json").read_text(encoding="utf-8")
    )
    assert metadata["components"] == ["model"]
    assert metadata["pose_camera_count"] == 2
    assert metadata["pose_image_names"] == list(names)
    assert (
        restore_checkpoint(
            checkpoint,
            model,
            pose_image_names=names,
        )
        == 1
    )
    with pytest.raises(ValueError, match="pose image names"):
        restore_checkpoint(
            checkpoint,
            model,
            pose_image_names=tuple(reversed(names)),
        )


def test_legacy_v3_core_restore_remains_supported_but_pose_restore_is_rejected(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(capacity=8, bucket_min_capacity=4, sh_degree=0)
    config = TrainConfig(model=model_config)
    model = GaussianModel.empty(model_config, physical_capacity=4)
    model.active_mask[0] = True
    optimizer = create_optimizer(model, config.optimizer)
    optimizer.step[...] = 6
    strategy_state = StrategyState(4)
    strategy_state.last_new_count[...] = 2
    checkpoint = save_checkpoint(
        tmp_path,
        model,
        step=6,
        optimizer=optimizer,
        strategy_state=strategy_state,
        config=config,
    )
    metadata_path = checkpoint / "jax_gs_checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format_version"] = 3
    metadata.pop("components", None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    restored_model = GaussianModel.empty(model_config, physical_capacity=4)
    restored_optimizer = create_optimizer(restored_model, config.optimizer)
    restored_strategy = StrategyState(4)
    assert (
        restore_checkpoint(
            checkpoint,
            restored_model,
            optimizer=restored_optimizer,
            strategy_state=restored_strategy,
        )
        == 6
    )
    assert int(restored_model.active_count) == 1
    assert int(restored_optimizer.step[...]) == 6
    assert int(restored_strategy.last_new_count[...]) == 2

    pose_module, pose_optimizer = _pose_training_state(1)
    with pytest.raises(ValueError, match="does not contain pose"):
        restore_checkpoint(
            checkpoint,
            restored_model,
            pose_module=pose_module,
            pose_optimizer=pose_optimizer,
            pose_image_names=("first.png",),
        )


def test_checkpoint_pose_arguments_must_be_provided_together(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0)
    model = GaussianModel.empty(model_config, physical_capacity=4)
    pose_module, pose_optimizer = _pose_training_state(1)

    with pytest.raises(ValueError, match="provided together"):
        save_checkpoint(
            tmp_path,
            model,
            step=1,
            pose_module=pose_module,
        )

    checkpoint = save_checkpoint(tmp_path, model, step=1)
    with pytest.raises(ValueError, match="provided together"):
        restore_checkpoint(
            checkpoint,
            model,
            pose_module=pose_module,
            pose_optimizer=pose_optimizer,
        )

    with pytest.raises(TypeError, match="sequence of image names"):
        save_checkpoint(
            tmp_path,
            model,
            step=1,
            pose_module=pose_module,
            pose_optimizer=pose_optimizer,
            pose_image_names="a.png",
        )


def test_typed_prng_key_restore_preserves_implementation() -> None:
    target = jax.random.key(7, impl="rbg")
    restored = checkpoints_module._restore_empty_arrays(
        jax.random.key_data(target), target
    )

    assert jax.random.key_impl(restored) == jax.random.key_impl(target)
    np.testing.assert_array_equal(
        jax.random.key_data(restored), jax.random.key_data(target)
    )
