from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.capacity import compact_training_state, resize_training_state
from jax_gs.config import ModelConfig, OptimizerConfig
from jax_gs.model import GaussianModel
from jax_gs.optimizers import create_optimizer
from jax_gs.scene import GaussianScene
from jax_gs.strategy import StrategyState


def _scene(capacity: int) -> GaussianScene:
    quats = jnp.zeros((capacity, 4), jnp.float32).at[:, 0].set(1.0)
    scene = GaussianScene.from_splats(
        nnx.Dict(
            {
                "means": nnx.Param(jnp.arange(capacity * 3).reshape(capacity, 3)),
                "scales": nnx.Param(jnp.zeros((capacity, 3), jnp.float32)),
                "quats": nnx.Param(quats),
                "opacities": nnx.Param(jnp.zeros((capacity,), jnp.float32)),
            }
        ),
        id="world",
        signal={
            "label": 10 * (jnp.arange(capacity, dtype=jnp.int32) + 1),
            "vector": jnp.arange(capacity * 2, dtype=jnp.float32).reshape(
                capacity, 2
            ),
        },
    )
    scene.component_names = [f"component_{slot}" for slot in range(capacity)]
    scene.component_index = jnp.arange(capacity, dtype=jnp.int32)
    return scene


def _training_state(capacity: int, maximum: int | None = None):
    if maximum is None:
        maximum = capacity
    model_config = ModelConfig(
        capacity=maximum,
        bucket_min_capacity=capacity,
        sh_degree=0,
    )
    optimizer_config = OptimizerConfig(max_steps=10)
    model = GaussianModel.empty(model_config, physical_capacity=capacity)
    slots = jnp.arange(capacity, dtype=jnp.float32)
    model.means[...] = jnp.stack((slots, slots + 10, slots + 20), axis=-1)
    optimizer = create_optimizer(model, optimizer_config)
    optimizer.update(
        model,
        jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param)),
    )
    state = StrategyState(capacity)
    state.grad_accum[...] = slots + 100
    return model_config, optimizer_config, model, optimizer, state


def _snapshot(node: object):
    return jax.tree.map(
        lambda value: np.asarray(value).copy(),
        nnx.as_pure(nnx.state(node)),
    )


def _assert_snapshot_equal(actual: object, expected: object) -> None:
    actual_leaves = jax.tree.leaves(nnx.as_pure(nnx.state(actual)))
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_value, expected_value in zip(actual_leaves, expected_leaves):
        np.testing.assert_array_equal(np.asarray(actual_value), expected_value)


def test_resize_training_state_pads_scene_rows_with_inactive_values():
    model_config, optimizer_config, model, optimizer, state = _training_state(
        4, maximum=8
    )
    model.active_mask[:2] = True
    scene = _scene(4)
    splats_before = {
        name: np.asarray(value[...]).copy() for name, value in scene.splats.items()
    }

    new_model, _, _ = resize_training_state(
        model,
        optimizer,
        state,
        8,
        model_config,
        optimizer_config,
        scene=scene,
    )

    np.testing.assert_array_equal(
        np.asarray(scene.component_index),
        np.asarray([0, 1, 2, 3, 0, 0, 0, 0], np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(scene.signal["label"]),
        np.asarray([10, 20, 30, 40, 0, 0, 0, 0], np.int32),
    )
    np.testing.assert_array_equal(np.asarray(scene.signal["vector"][4:]), 0.0)
    np.testing.assert_array_equal(np.asarray(new_model.active_mask[4:]), False)
    scene.validate_slot_capacity(8)
    scene.validate()

    # Scene rows are padded mechanically. They are not copied from the
    # independently stored GaussianModel.
    assert scene.num_gaussians() == 8
    for name, before in splats_before.items():
        after = np.asarray(scene.splats[name][...])
        np.testing.assert_array_equal(after[:4], before)
        np.testing.assert_array_equal(after[4:], 0)
    restored = GaussianScene.from_state_dict(scene.state_dict())
    restored.validate()
    np.testing.assert_array_equal(
        np.asarray(restored.signal["label"]), np.asarray(scene.signal["label"])
    )


def test_compact_training_state_applies_the_model_stable_order_to_scene_sidecars():
    _, _, model, optimizer, state = _training_state(6)
    model.active_mask[...] = jnp.asarray(
        [False, True, False, True, True, False]
    )
    scene = _scene(6)
    order = np.asarray([1, 3, 4, 0, 2, 5])
    components_before = np.asarray(scene.component_index).copy()
    labels_before = np.asarray(scene.signal["label"]).copy()
    vectors_before = np.asarray(scene.signal["vector"]).copy()
    model_means_before = np.asarray(model.means[...]).copy()
    splats_before = {
        name: np.asarray(value[...]).copy() for name, value in scene.splats.items()
    }

    active_count = compact_training_state(model, optimizer, state, scene=scene)

    assert int(active_count) == 3
    np.testing.assert_array_equal(
        np.asarray(scene.component_index), components_before[order]
    )
    np.testing.assert_array_equal(
        np.asarray(scene.signal["label"]), labels_before[order]
    )
    np.testing.assert_array_equal(
        np.asarray(scene.signal["vector"]), vectors_before[order]
    )
    np.testing.assert_array_equal(
        np.asarray(model.means[...]), model_means_before[order]
    )
    for name, before in splats_before.items():
        np.testing.assert_array_equal(
            np.asarray(scene.splats[name][...]), before[order]
        )
    scene.validate()


@pytest.mark.parametrize("operation", ["resize", "compact"])
def test_scene_capacity_mismatch_fails_before_training_state_mutation(operation):
    model_config, optimizer_config, model, optimizer, state = _training_state(
        4, maximum=8
    )
    model.active_mask[:2] = True
    scene = _scene(3)
    model_before = _snapshot(model)
    optimizer_before = _snapshot(optimizer)
    state_before = _snapshot(state)

    with pytest.raises(ValueError, match="scene slot capacity"):
        if operation == "resize":
            resize_training_state(
                model,
                optimizer,
                state,
                8,
                model_config,
                optimizer_config,
                scene=scene,
            )
        else:
            compact_training_state(model, optimizer, state, scene=scene)

    _assert_snapshot_equal(model, model_before)
    _assert_snapshot_equal(optimizer, optimizer_before)
    _assert_snapshot_equal(state, state_before)
