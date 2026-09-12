import math

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jax_gs.config import ModelConfig, OptimizerConfig, StrategyConfig
from jax_gs.contrib.dynamic import DynamicStrategy
from jax_gs.model import GaussianModel
from jax_gs.optimizers import create_optimizer
from jax_gs.scene import GaussianScene
from jax_gs.strategy import DefaultStrategy, MCMCStrategy, StrategyState
from jax_gs.strategy.ops import duplicate, relocate, remove, sample_add, split


def _model(capacity: int, active: int):
    model = GaussianModel.empty(
        ModelConfig(
            capacity=capacity,
            bucket_min_capacity=capacity,
            sh_degree=0,
        )
    )
    model.active_mask[:active] = True
    slots = jnp.arange(capacity, dtype=jnp.float32)
    model.means[...] = jnp.stack((slots, slots + 100.0, slots + 200.0), -1)
    model.log_scales[...] = jnp.log(0.01)
    model.opacity_logits[...] = jnp.log(4.0)
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    state = StrategyState(capacity)
    return model, optimizer, state


def _scene(capacity: int) -> GaussianScene:
    quats = jnp.zeros((capacity, 4), jnp.float32).at[:, 0].set(1.0)
    scene = GaussianScene.from_splats(
        nnx.Dict(
            {
                "means": nnx.Param(jnp.zeros((capacity, 3), jnp.float32)),
                "scales": nnx.Param(jnp.zeros((capacity, 3), jnp.float32)),
                "quats": nnx.Param(quats),
                "opacities": nnx.Param(jnp.zeros((capacity,), jnp.float32)),
                "colors": nnx.Param(jnp.zeros((capacity, 3), jnp.float32)),
            }
        ),
        id="world",
        signal={
            "label": 10 * (jnp.arange(capacity, dtype=jnp.int32) + 1),
        },
    )
    scene.component_names = [f"component_{slot}" for slot in range(capacity)]
    scene.component_index = jnp.arange(capacity, dtype=jnp.int32)
    return scene


def _binomials(size: int = 8) -> jax.Array:
    return jnp.asarray(
        [
            [math.comb(row, column) if column <= row else 0 for column in range(size)]
            for row in range(size)
        ],
        jnp.float32,
    )


def test_eager_fixed_slot_ops_preserve_scene_lineage_across_reused_holes():
    model, optimizer, state = _model(8, 2)
    scene = _scene(8)

    duplicate_targets = duplicate(
        model,
        optimizer,
        state,
        jnp.asarray([True, False, False, False, False, False, False, False]),
        scene=scene,
    )
    assert duplicate_targets.tolist() == [2]
    assert int(scene.signal["label"][2]) == 10
    assert int(scene.component_index[2]) == 0

    split_targets = split(
        model,
        optimizer,
        state,
        jnp.asarray([False, True, False, False, False, False, False, False]),
        scene=scene,
        key=jax.random.key(1),
    )
    assert split_targets.tolist() == [3]
    assert int(scene.signal["label"][3]) == 20
    assert int(scene.component_index[3]) == 1

    remove(
        model,
        optimizer,
        state,
        jnp.asarray([True, False, False, False, False, False, False, False]),
        scene=scene,
    )
    assert not bool(model.active_mask[0])
    assert int(scene.signal["label"][0]) == 10

    before_relocate_labels = jnp.array(scene.signal["label"])
    before_relocate_components = jnp.array(scene.component_index)
    dead, donors = relocate(
        model,
        optimizer,
        state,
        jnp.asarray([False, False, True, False, False, False, False, False]),
        _binomials(),
        scene=scene,
        key=jax.random.key(2),
    )
    assert dead.tolist() == [2]
    assert int(scene.signal["label"][dead[0]]) == int(before_relocate_labels[donors[0]])
    assert int(scene.component_index[dead[0]]) == int(
        before_relocate_components[donors[0]]
    )

    before_add_labels = jnp.array(scene.signal["label"])
    before_add_components = jnp.array(scene.component_index)
    targets, sampled = sample_add(
        model,
        optimizer,
        state,
        1,
        _binomials(),
        scene=scene,
        key=jax.random.key(3),
    )
    assert targets.tolist() == [0]
    assert int(scene.signal["label"][targets[0]]) == int(before_add_labels[sampled[0]])
    assert int(scene.component_index[targets[0]]) == int(
        before_add_components[sampled[0]]
    )

    active_labels = scene.signal["label"][model.active_mask[...]]
    active_components = scene.component_index[model.active_mask[...]]
    assert jnp.array_equal(active_labels, jnp.full((4,), 20, jnp.int32))
    assert jnp.array_equal(active_components, jnp.ones((4,), jnp.int32))


def test_eager_scene_capacity_mismatch_fails_before_model_mutation():
    model, optimizer, state = _model(4, 1)
    scene = _scene(3)
    active_before = jnp.array(model.active_mask[...])
    means_before = jnp.array(model.means[...])

    with pytest.raises(ValueError, match="scene slot capacity"):
        duplicate(
            model,
            optimizer,
            state,
            jnp.asarray([True, False, False, False]),
            scene=scene,
        )

    assert jnp.array_equal(model.active_mask[...], active_before)
    assert jnp.array_equal(model.means[...], means_before)


def test_default_hook_commits_scene_transaction_without_private_result_keys():
    model, optimizer, state = _model(4, 1)
    scene = _scene(4)
    strategy = DefaultStrategy(
        StrategyConfig(
            refine_start=0,
            refine_stop=10,
            refine_every=1,
            reset_every=20,
            grow_grad2d=0.1,
            grow_scale3d=100.0,
            prune_scale3d=100.0,
            max_new_per_refine=1,
        )
    )

    result = strategy.step_post_backward(
        model,
        optimizer,
        state,
        1,
        {
            "means_gradient": jnp.ones((4, 3), jnp.float32),
            "visible": jnp.asarray([True, False, False, False]),
            "radii": jnp.asarray([0.1, 0.0, 0.0, 0.0]),
        },
        scene=scene,
        key=jax.random.key(4),
    )

    new_slots = jnp.nonzero(model.active_mask[...].at[0].set(False), size=1)[0]
    assert int(result["new_count"]) == 1
    assert int(scene.signal["label"][new_slots[0]]) == 10
    assert int(scene.component_index[new_slots[0]]) == 0
    assert not any(key.startswith("_slot_") for key in result)


def test_dynamic_hook_scheduled_refine_updates_scene_and_mask_lineage():
    model, optimizer, _ = _model(4, 1)
    scene = _scene(4)
    strategy = DynamicStrategy(
        StrategyConfig(
            refine_start=0,
            refine_stop=10,
            refine_every=1,
            reset_every=20,
            grow_grad2d=0.1,
            grow_scale3d=100.0,
            prune_scale3d=100.0,
            max_new_per_refine=1,
        )
    )
    state = strategy.initialize_state(4, init_dynamic=False)
    state.dynamic_mask[0] = True

    result = strategy.step_post_backward(
        model,
        optimizer,
        state,
        1,
        {
            "means_gradient": jnp.ones((4, 3), jnp.float32),
            "visible": jnp.asarray([True, False, False, False]),
            "radii": jnp.asarray([0.1, 0.0, 0.0, 0.0]),
        },
        scene=scene,
        key=jax.random.key(8),
    )

    new_slots = jnp.nonzero(model.active_mask[...].at[0].set(False), size=1)[0]
    new_slot = new_slots[0]
    assert int(result["new_count"]) == 1
    assert bool(state.dynamic_mask[new_slot])
    assert int(scene.signal["label"][new_slot]) == 10
    assert int(scene.component_index[new_slot]) == 0
    assert not any(key.startswith("_slot_") for key in result)


def test_default_combined_growth_copies_both_scene_children_from_parent():
    model, optimizer, state = _model(3, 1)
    scene = _scene(3)
    strategy = DefaultStrategy(
        StrategyConfig(
            max_new_per_refine=2,
            grow_grad2d=0.1,
            grow_scale3d=1.0,
            grow_scale2d=0.05,
            refine_scale2d_stop_iter=10,
            prune_scale3d=100.0,
        )
    )
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0
    state.max_radii[0] = 0.2

    result = strategy.refine(
        model,
        state,
        optimizer,
        jax.random.key(6),
        1.0,
        step=1,
        scene=scene,
    )

    assert int(result["new_count"]) == 2
    assert jnp.array_equal(
        scene.signal["label"][model.active_mask[...]],
        jnp.asarray([10, 10, 10], jnp.int32),
    )
    assert jnp.array_equal(
        scene.component_index[model.active_mask[...]],
        jnp.zeros((3,), jnp.int32),
    )
    assert not any(key.startswith("_slot_") for key in result)


def test_default_combined_growth_overflow_keeps_scene_lineage_unchanged():
    model, optimizer, state = _model(2, 1)
    scene = _scene(2)
    strategy = DefaultStrategy(
        StrategyConfig(
            max_new_per_refine=2,
            grow_grad2d=0.1,
            grow_scale3d=1.0,
            grow_scale2d=0.05,
            refine_scale2d_stop_iter=10,
            prune_scale3d=100.0,
        )
    )
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0
    state.max_radii[0] = 0.2
    labels_before = jnp.array(scene.signal["label"])
    components_before = jnp.array(scene.component_index)

    result = strategy.refine(
        model,
        state,
        optimizer,
        jax.random.key(7),
        1.0,
        step=1,
        scene=scene,
    )

    assert bool(result["capacity_overflow"])
    assert int(result["new_count"]) == 0
    assert jnp.array_equal(scene.signal["label"], labels_before)
    assert jnp.array_equal(scene.component_index, components_before)


def test_mcmc_overflow_keeps_all_scene_slots_unchanged():
    model, optimizer, state = _model(20, 20)
    scene = _scene(20)
    model.opacity_logits[0] = -20.0
    strategy = MCMCStrategy(
        StrategyConfig(
            kind="mcmc",
            prune_opacity=0.01,
            cap_max=40,
            max_new_per_refine=2,
        )
    )
    scene_before = scene.state_dict()

    result = strategy.refine(
        model,
        state,
        optimizer,
        jax.random.key(9),
        1.0,
        scene=scene,
    )

    scene_after = scene.state_dict()
    assert bool(result["capacity_overflow"])
    assert int(result["new_count"]) == 0
    assert int(result["relocated_count"]) == 0
    assert scene_after["id"] == scene_before["id"]
    assert scene_after["component_names"] == scene_before["component_names"]
    assert jnp.array_equal(
        scene_after["component_index"], scene_before["component_index"]
    )
    for group in ("splats", "signal"):
        for name, before in scene_before[group].items():
            assert jnp.array_equal(scene_after[group][name], before)


def test_mcmc_hook_commits_relocation_and_birth_scene_transactions():
    model, optimizer, state = _model(24, 20)
    scene = _scene(24)
    model.opacity_logits[0] = -20.0
    strategy = MCMCStrategy(
        StrategyConfig(
            kind="mcmc",
            refine_start=0,
            refine_stop=10,
            refine_every=1,
            prune_opacity=0.01,
            cap_max=24,
            max_new_per_refine=24,
            noise_lr=0.0,
        )
    )

    result = strategy.step_post_backward(
        model,
        optimizer,
        state,
        1,
        {},
        lr=0.0,
        scene=scene,
        key=jax.random.key(5),
    )

    active = model.active_mask[...]
    expected_labels = 10 * (jnp.rint(model.means[:, 0]).astype(jnp.int32) + 1)
    assert int(model.active_count) == 21
    assert jnp.array_equal(scene.signal["label"][active], expected_labels[active])
    assert jnp.array_equal(
        scene.component_index[active],
        jnp.rint(model.means[active, 0]).astype(jnp.int32),
    )
    assert not any(key.startswith("_slot_") for key in result)
