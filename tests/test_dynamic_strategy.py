import jax
import jax.numpy as jnp
import pytest

from jax_gs.capacity import compact_training_state, resize_training_state
from jax_gs.config import ModelConfig, OptimizerConfig, StrategyConfig
from jax_gs.contrib.dynamic import DynamicStrategy
from jax_gs.model import GaussianModel
from jax_gs.optimizers import create_optimizer
from jax_gs.strategy import StrategyState


def _training_state(capacity=6, active=2):
    model_config = ModelConfig(
        capacity=capacity,
        bucket_min_capacity=capacity,
        sh_degree=0,
    )
    model = GaussianModel.empty(model_config)
    model.active_mask[:active] = True
    model.opacity_logits[:active] = 2.0
    optimizer_config = OptimizerConfig(max_steps=10)
    optimizer = create_optimizer(model, optimizer_config)
    return model_config, optimizer_config, model, optimizer


def test_dynamic_strategy_initializes_boolean_mask_and_default_statistics():
    state = DynamicStrategy().initialize_state(5)
    assert state.dynamic_mask.shape == (5,)
    assert state.dynamic_mask[...].dtype == jnp.bool_
    assert bool(jnp.all(state.dynamic_mask[...]))
    assert state.grad_accum.shape == (5,)
    assert state.visible_count.shape == (5,)
    assert state.max_radii.shape == (5,)


def test_dynamic_strategy_honors_static_initialization():
    state = DynamicStrategy().initialize_state(4, init_dynamic=False)
    assert not bool(jnp.any(state.dynamic_mask[...]))


def test_dynamic_strategy_accepts_the_upstream_initialize_state_surface():
    # Upstream's calling convention: scene_scale, num_gaussians, device,
    # init_dynamic. The port sizes state by num_gaussians when no trailing
    # capacity is given, ignores device, and stores scene_scale like the
    # parent state does.
    state = DynamicStrategy().initialize_state(2.0, 6, None, False)
    assert state.dynamic_mask.shape == (6,)
    assert not bool(jnp.any(state.dynamic_mask[...]))
    assert float(state.scene_scale[...]) == 2.0

    # The trailing JAX extension wins over num_gaussians when both are given.
    state = DynamicStrategy().initialize_state(1.5, 3, capacity=7)
    assert state.dynamic_mask.shape == (7,)
    assert float(state.scene_scale[...]) == 1.5

    # A first positional integer keeps meaning the capacity, matching
    # DefaultStrategy's documented legacy form.
    state = DynamicStrategy().initialize_state(5)
    assert state.dynamic_mask.shape == (5,)
    assert float(state.scene_scale[...]) == 1.0


def test_dynamic_strategy_new_slots_inherit_dynamic_and_static_parents():
    _, _, model, optimizer = _training_state(capacity=6, active=2)
    model.means[:2] = jnp.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    strategy = DynamicStrategy(
        StrategyConfig(
            max_new_per_refine=2,
            grow_grad2d=0.1,
            grow_scale3d=100.0,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(6, init_dynamic=False)
    state.dynamic_mask[0] = True
    state.grad_accum[:2] = jnp.asarray([2.0, 1.0])
    state.visible_count[:2] = 1.0

    statistics = strategy.refine(
        model,
        state,
        optimizer,
        jax.random.key(0),
        1.0,
    )

    assert int(statistics["new_count"]) == 2
    assert jnp.array_equal(
        state.dynamic_mask[...],
        jnp.asarray([True, False, True, False, False, False]),
    )
    assert jnp.allclose(model.means[2], model.means[0])
    assert jnp.allclose(model.means[3], model.means[1])


def test_dynamic_strategy_clears_pruned_slot_flags():
    _, _, model, optimizer = _training_state(capacity=5, active=3)
    model.opacity_logits[0] = -20.0
    strategy = DynamicStrategy(
        StrategyConfig(
            max_new_per_refine=1,
            grow_grad2d=100.0,
            prune_opacity=0.01,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(5, init_dynamic=False)
    state.dynamic_mask[:3] = jnp.asarray([True, False, True])
    statistics = strategy.refine(
        model,
        state,
        optimizer,
        jax.random.key(1),
        1.0,
    )
    assert int(statistics["pruned_count"]) == 1
    assert not bool(state.dynamic_mask[0])
    assert bool(state.dynamic_mask[2])


def test_dynamic_strategy_split_child_inherits_parent_before_post_grow_prune():
    _, _, model, optimizer = _training_state(capacity=2, active=1)
    model.log_scales[0] = jnp.log(1.5)
    strategy = DynamicStrategy(
        StrategyConfig(
            reset_every=1,
            max_new_per_refine=1,
            grow_grad2d=0.1,
            grow_scale3d=0.5,
            prune_scale3d=1.0,
        )
    )
    state = strategy.initialize_state(2, init_dynamic=False)
    state.dynamic_mask[0] = True
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0

    statistics = strategy.refine(
        model,
        state,
        optimizer,
        jax.random.key(3),
        1.0,
        step=2,
    )

    assert int(statistics["new_count"]) == 1
    assert int(statistics["pruned_count"]) == 0
    assert jnp.array_equal(model.active_mask[...], jnp.asarray([True, True]))
    assert jnp.array_equal(state.dynamic_mask[...], jnp.asarray([True, True]))


def test_dynamic_strategy_combined_growth_marks_duplicate_and_split_children():
    _, _, model, optimizer = _training_state(capacity=3, active=1)
    model.log_scales[0] = jnp.log(0.1)
    strategy = DynamicStrategy(
        StrategyConfig(
            max_new_per_refine=2,
            grow_grad2d=0.1,
            grow_scale3d=1.0,
            grow_scale2d=0.05,
            refine_scale2d_stop_iter=10,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(3, init_dynamic=False)
    state.dynamic_mask[0] = True
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0
    state.max_radii[0] = 0.2

    statistics = strategy.refine(
        model,
        state,
        optimizer,
        jax.random.key(5),
        1.0,
        step=1,
    )

    assert int(statistics["new_count"]) == 2
    assert jnp.array_equal(model.active_mask[...], jnp.ones((3,), jnp.bool_))
    assert jnp.array_equal(state.dynamic_mask[...], jnp.ones((3,), jnp.bool_))


def test_dynamic_strategy_combined_growth_overflow_keeps_lineage_unchanged():
    _, _, model, optimizer = _training_state(capacity=2, active=1)
    model.log_scales[0] = jnp.log(0.1)
    strategy = DynamicStrategy(
        StrategyConfig(
            max_new_per_refine=2,
            grow_grad2d=0.1,
            grow_scale3d=1.0,
            grow_scale2d=0.05,
            refine_scale2d_stop_iter=10,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(2, init_dynamic=False)
    state.dynamic_mask[0] = True
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0
    state.max_radii[0] = 0.2
    mask_before = jnp.array(state.dynamic_mask[...])

    statistics = strategy.refine(
        model,
        state,
        optimizer,
        jax.random.key(6),
        1.0,
        step=1,
    )

    assert bool(statistics["capacity_overflow"])
    assert int(statistics["new_count"]) == 0
    assert jnp.array_equal(state.dynamic_mask[...], mask_before)


def test_dynamic_strategy_overflow_preserves_inactive_true_mask_bits():
    _, _, model, optimizer = _training_state(capacity=2, active=1)
    model.log_scales[0] = jnp.log(0.1)
    strategy = DynamicStrategy(
        StrategyConfig(
            max_new_per_refine=2,
            grow_grad2d=0.1,
            grow_scale3d=1.0,
            grow_scale2d=0.05,
            refine_scale2d_stop_iter=10,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(2, init_dynamic=True)
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0
    state.max_radii[0] = 0.2
    mask_before = jnp.array(state.dynamic_mask[...])

    statistics = strategy.refine(
        model,
        state,
        optimizer,
        jax.random.key(7),
        1.0,
        step=1,
    )

    assert bool(statistics["capacity_overflow"])
    assert jnp.array_equal(state.dynamic_mask[...], mask_before)


def test_dynamic_strategy_success_preserves_unrelated_inactive_mask_bits():
    _, _, model, optimizer = _training_state(capacity=5, active=1)
    strategy = DynamicStrategy(
        StrategyConfig(
            max_new_per_refine=1,
            grow_grad2d=0.1,
            grow_scale3d=100.0,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(5, init_dynamic=False)
    state.dynamic_mask[0] = True
    state.dynamic_mask[4] = True
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0

    statistics = strategy.refine(
        model,
        state,
        optimizer,
        jax.random.key(9),
        1.0,
    )

    assert int(statistics["new_count"]) == 1
    assert jnp.array_equal(
        state.dynamic_mask[...],
        jnp.asarray([True, True, False, False, True]),
    )


def test_dynamic_strategy_clears_parent_and_child_pruned_after_growth():
    _, _, model, optimizer = _training_state(capacity=3, active=2)
    model.log_scales[:2] = jnp.log(0.1)
    model.opacity_logits[:2] = jnp.asarray([-20.0, jnp.log(4.0)])
    strategy = DynamicStrategy(
        StrategyConfig(
            reset_every=100,
            max_new_per_refine=1,
            grow_grad2d=0.5,
            grow_scale3d=1.0,
            prune_opacity=0.01,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(3, init_dynamic=False)
    state.dynamic_mask[0] = True
    state.grad_accum[:2] = jnp.asarray([1.0, 0.0])
    state.visible_count[:2] = 1.0

    statistics = strategy.refine(
        model,
        state,
        optimizer,
        jax.random.key(4),
        1.0,
        step=1,
    )

    assert int(statistics["new_count"]) == 1
    assert int(statistics["pruned_count"]) == 2
    assert jnp.array_equal(model.active_mask[...], jnp.asarray([False, True, False]))
    assert not bool(jnp.any(state.dynamic_mask[...]))


def test_dynamic_strategy_requires_its_own_state():
    _, _, model, optimizer = _training_state(capacity=4, active=1)
    with pytest.raises(RuntimeError, match="initialize_state"):
        DynamicStrategy().refine(
            model,
            StrategyState(4),
            optimizer,
            jax.random.key(2),
            1.0,
        )


def test_dynamic_mask_stays_aligned_during_compaction():
    _, _, model, optimizer = _training_state(capacity=4, active=0)
    model.active_mask[:] = jnp.asarray([False, True, False, True])
    state = DynamicStrategy().initialize_state(4, init_dynamic=False)
    state.dynamic_mask[:] = jnp.asarray([False, True, False, False])
    active_count = compact_training_state(model, optimizer, state)
    assert int(active_count) == 2
    assert jnp.array_equal(
        state.dynamic_mask[...],
        jnp.asarray([True, False, False, False]),
    )


def test_dynamic_mask_resizes_with_physical_capacity():
    model_config = ModelConfig(capacity=8, bucket_min_capacity=4, sh_degree=0)
    optimizer_config = OptimizerConfig(max_steps=10)
    model = GaussianModel.empty(model_config, physical_capacity=4)
    optimizer = create_optimizer(model, optimizer_config)
    state = DynamicStrategy().initialize_state(4, init_dynamic=False)
    state.dynamic_mask[:2] = jnp.asarray([True, False])

    model, optimizer, state = resize_training_state(
        model,
        optimizer,
        state,
        8,
        model_config,
        optimizer_config,
    )
    assert model.capacity == 8
    assert state.dynamic_mask.shape == (8,)
    assert jnp.array_equal(
        state.dynamic_mask[...],
        jnp.asarray([True, False, False, False, False, False, False, False]),
    )
