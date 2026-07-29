import math

from flax import nnx
import jax
import jax.numpy as jnp

from jax_gs.config import ModelConfig, OptimizerConfig, StrategyConfig
from jax_gs.model import GaussianModel
from jax_gs.optimizers import create_optimizer
from jax_gs.strategy import DefaultStrategy, MCMCStrategy, _sample_weighted_ids


def _state_leaves(node):
    return [
        jnp.array(value)
        for value in jax.tree.leaves(nnx.as_pure(nnx.state(node)))
    ]


def test_default_strategy_allocates_without_resizing():
    model = GaussianModel.empty(ModelConfig(capacity=16, sh_degree=1))
    model.active_mask[:4] = True
    model.opacity_logits[:4] = 2.0
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = DefaultStrategy(
        StrategyConfig(
            kind="default",
            max_new_per_refine=4,
            grow_grad2d=0.1,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(16)
    state.grad_accum[:4] = 1.0
    state.visible_count[:4] = 1.0
    shapes_before = jax.tree.map(lambda x: x.shape, model.state_dict())
    stats = strategy.refine(model, state, optimizer, jax.random.key(0), 1.0)
    shapes_after = jax.tree.map(lambda x: x.shape, model.state_dict())
    assert shapes_after == shapes_before
    assert int(stats["new_count"]) == 4
    assert int(model.active_count) == 8
    assert not bool(stats["capacity_overflow"])


def test_compiled_default_strategy_copies_appearance_rows_to_children():
    model = GaussianModel.empty(
        ModelConfig(capacity=2, bucket_min_capacity=2),
        appearance_feature_dim=32,
    )
    model.active_mask[0] = True
    model.features[0] = jnp.arange(32, dtype=jnp.float32)
    model.colors[0] = jnp.asarray([1.0, 2.0, 3.0])
    model.log_scales[0] = jnp.log(0.1)
    model.opacity_logits[0] = jnp.log(4.0)
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = DefaultStrategy(
        StrategyConfig(
            max_new_per_refine=1,
            grow_grad2d=0.1,
            grow_scale3d=100.0,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(2)
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0

    result = strategy.refine(model, state, optimizer, jax.random.key(0), 1.0)

    assert int(result["new_count"]) == 1
    assert jnp.array_equal(model.features[1], model.features[0])
    assert jnp.array_equal(model.colors[1], model.colors[0])


def test_default_strategy_plan_does_not_grow_at_equal_gradient_threshold():
    model = GaussianModel.empty(ModelConfig(capacity=2, sh_degree=0))
    model.active_mask[0] = True
    model.opacity_logits[0] = jnp.log(4.0)
    strategy = DefaultStrategy(
        StrategyConfig(
            max_new_per_refine=1,
            grow_grad2d=0.125,
            grow_scale3d=100.0,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(2)
    state.grad_accum[0] = 0.125
    state.visible_count[0] = 1.0

    plan = strategy.plan_refine(model, state, 1.0)

    assert int(plan["planned_new_count"]) == 0
    assert int(plan["required_capacity"]) == 1


def test_default_strategy_hook_does_not_grow_at_equal_gradient_threshold():
    model = GaussianModel.empty(ModelConfig(capacity=2, sh_degree=0))
    model.active_mask[0] = True
    model.opacity_logits[0] = jnp.log(4.0)
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = DefaultStrategy(
        StrategyConfig(
            refine_start=0,
            refine_stop=10,
            refine_every=1,
            reset_every=20,
            max_new_per_refine=1,
            grow_grad2d=0.125,
            grow_scale3d=100.0,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(2)

    result = strategy.step_post_backward(
        model,
        optimizer,
        state,
        1,
        {
            "means2d_gradient": jnp.asarray(
                [[[0.125, 0.0], [0.0, 0.0]]], jnp.float32
            ),
            "radii": jnp.asarray([[[1.0, 1.0], [0.0, 0.0]]], jnp.float32),
            "valid": jnp.asarray([[True, False]]),
            "width": 2,
            "height": 2,
        },
        key=jax.random.key(1),
    )

    assert int(result["new_count"]) == 0
    assert int(model.active_count) == 1


def test_default_strategy_compiled_split_samples_independent_children():
    model = GaussianModel.empty(ModelConfig(capacity=2, sh_degree=0))
    model.active_mask[0] = True
    model.means[0] = jnp.asarray([2.0, -1.0, 0.5])
    model.log_scales[0] = jnp.log(jnp.asarray([0.5, 1.0, 2.0]))
    model.opacity_logits[0] = jnp.log(4.0)
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = DefaultStrategy(
        StrategyConfig(
            max_new_per_refine=1,
            grow_grad2d=0.1,
            grow_scale3d=0.1,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(2)
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0
    key = jax.random.key(23)
    parent_mean = model.means[0].copy()
    parent_scale = jnp.exp(model.log_scales[0])
    standard_noise = jax.random.normal(key, (2, 1, 3), dtype=jnp.float32)
    expected_means = parent_mean + standard_noise[:, 0] * parent_scale

    result = strategy.refine(model, state, optimizer, key, 1.0)

    assert int(result["new_count"]) == 1
    assert jnp.allclose(model.means[0], expected_means[0])
    assert jnp.allclose(model.means[1], expected_means[1])


def test_default_strategy_small_screen_large_parent_duplicates_then_splits():
    model = GaussianModel.empty(ModelConfig(capacity=3, sh_degree=0))
    model.active_mask[0] = True
    model.means[0] = jnp.asarray([1.0, 2.0, 3.0])
    model.log_scales[0] = jnp.log(0.1)
    model.opacity_logits[0] = jnp.log(4.0)
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    optimizer.update(
        model,
        jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param)),
    )
    model.means[0] = jnp.asarray([1.0, 2.0, 3.0])
    model.log_scales[0] = jnp.log(0.1)
    model.opacity_logits[0] = jnp.log(4.0)
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
    state = strategy.initialize_state(3)
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0
    state.max_radii[0] = 0.2

    plan = strategy.plan_refine(model, state, 1.0, step=1)

    assert int(plan["planned_new_count"]) == 2
    assert int(plan["required_capacity"]) == 3
    assert not bool(plan["capacity_overflow"])

    result = strategy.refine(
        model, state, optimizer, jax.random.key(41), 1.0, step=1
    )

    active_scales = jnp.sort(
        jnp.max(jnp.exp(model.log_scales[model.active_mask[...]]), axis=-1)
    )
    assert int(result["new_count"]) == 2
    assert int(model.active_count) == 3
    assert jnp.allclose(
        active_scales,
        jnp.asarray([0.1 / 1.6, 0.1 / 1.6, 0.1]),
    )
    for value in _state_leaves(optimizer):
        if value.ndim > 0 and value.shape[0] == model.capacity:
            assert jnp.array_equal(value, jnp.zeros_like(value))


def test_default_strategy_combined_growth_event_limit_prefers_duplicate():
    model = GaussianModel.empty(ModelConfig(capacity=2, sh_degree=0))
    model.active_mask[0] = True
    model.log_scales[0] = jnp.log(0.1)
    model.opacity_logits[0] = jnp.log(4.0)
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = DefaultStrategy(
        StrategyConfig(
            max_new_per_refine=1,
            grow_grad2d=0.1,
            grow_scale3d=1.0,
            grow_scale2d=0.05,
            refine_scale2d_stop_iter=10,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(2)
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0
    state.max_radii[0] = 0.2

    result = strategy.refine(
        model, state, optimizer, jax.random.key(42), 1.0, step=1
    )

    assert int(result["new_count"]) == 1
    assert jnp.allclose(jnp.exp(model.log_scales[:2]), 0.1)


def test_default_strategy_combined_growth_revises_and_prunes_only_split_rows():
    model = GaussianModel.empty(ModelConfig(capacity=3, sh_degree=0))
    model.active_mask[0] = True
    model.log_scales[0] = jnp.log(0.1)
    original_opacity = 0.36
    model.opacity_logits[0] = jnp.log(
        original_opacity / (1.0 - original_opacity)
    )
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = DefaultStrategy(
        StrategyConfig(
            reset_every=100,
            max_new_per_refine=2,
            grow_grad2d=0.1,
            grow_scale3d=1.0,
            grow_scale2d=0.05,
            refine_scale2d_stop_iter=10,
            prune_opacity=0.25,
            prune_scale3d=100.0,
            revised_opacity=True,
        )
    )
    state = strategy.initialize_state(3)
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0
    state.max_radii[0] = 0.2

    plan = strategy.plan_refine(model, state, 1.0, step=1)
    result = strategy.refine(
        model, state, optimizer, jax.random.key(44), 1.0, step=1
    )

    assert int(plan["planned_new_count"]) == 2
    assert int(plan["pruned_count"]) == 2
    assert int(plan["active_after_prune_count"]) == 1
    assert int(result["new_count"]) == 2
    assert int(result["pruned_count"]) == 2
    assert int(model.active_count) == 1
    assert jnp.allclose(
        jax.nn.sigmoid(model.opacity_logits[model.active_mask[...]]),
        original_opacity,
    )
    assert jnp.allclose(
        jnp.exp(model.log_scales[model.active_mask[...]]), 0.1
    )


def test_default_strategy_combined_growth_overflow_is_atomic():
    model = GaussianModel.empty(ModelConfig(capacity=2, sh_degree=0))
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    optimizer.update(
        model,
        jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param)),
    )
    model.active_mask[0] = True
    model.means[0] = jnp.asarray([1.0, 2.0, 3.0])
    model.log_scales[0] = jnp.log(0.1)
    model.opacity_logits[0] = jnp.log(4.0)
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
    state = strategy.initialize_state(2)
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0
    state.max_radii[0] = 0.2
    model_before = {
        name: jnp.array(value) for name, value in model.state_dict().items()
    }
    optimizer_before = _state_leaves(optimizer)
    stats_before = (
        jnp.array(state.grad_accum[...]),
        jnp.array(state.visible_count[...]),
        jnp.array(state.max_radii[...]),
    )

    plan = strategy.plan_refine(model, state, 1.0, step=1)
    assert int(plan["planned_new_count"]) == 2
    assert int(plan["free_count"]) == 1
    assert int(plan["required_capacity"]) == 3
    assert bool(plan["capacity_overflow"])

    result = strategy.refine(
        model, state, optimizer, jax.random.key(43), 1.0, step=1
    )

    assert int(result["new_count"]) == 0
    assert int(result["pruned_count"]) == 0
    assert bool(result["capacity_overflow"])
    assert int(model.active_count) == 1
    for name, before in model_before.items():
        assert jnp.array_equal(model.state_dict()[name], before)
    for after, before in zip(_state_leaves(optimizer), optimizer_before, strict=True):
        assert jnp.array_equal(after, before)
    assert jnp.array_equal(state.grad_accum[...], stats_before[0])
    assert jnp.array_equal(state.visible_count[...], stats_before[1])
    assert jnp.array_equal(state.max_radii[...], stats_before[2])


def test_default_hook_overflow_skips_same_step_opacity_reset():
    model = GaussianModel.empty(ModelConfig(capacity=3, sh_degree=0))
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    optimizer.update(
        model,
        jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param)),
    )
    model.active_mask[:2] = True
    model.means[:2] = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    model.log_scales[:2] = jnp.log(0.1)
    model.opacity_logits[:2] = jnp.log(4.0)
    strategy = DefaultStrategy(
        StrategyConfig(
            refine_start=0,
            refine_stop=10,
            refine_every=1,
            reset_every=1,
            reset_opacity=0.1,
            max_new_per_refine=2,
            grow_grad2d=0.1,
            grow_scale3d=1.0,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(3)
    model_before = {
        name: jnp.array(value) for name, value in model.state_dict().items()
    }
    optimizer_before = _state_leaves(optimizer)

    result = strategy.step_post_backward(
        model,
        optimizer,
        state,
        1,
        {
            "means_gradient": jnp.ones((3, 3), jnp.float32),
            "visible": jnp.asarray([True, True, False]),
            "radii": jnp.asarray([0.1, 0.1, 0.0]),
        },
        key=jax.random.key(44),
    )

    assert bool(result["capacity_overflow"])
    assert not bool(result["opacity_reset"])
    assert int(result["new_count"]) == 0
    assert int(result["pruned_count"]) == 0
    for name, before in model_before.items():
        assert jnp.array_equal(model.state_dict()[name], before)
    for after, before in zip(_state_leaves(optimizer), optimizer_before, strict=True):
        assert jnp.array_equal(after, before)


def test_default_strategy_stops_resetting_opacity_at_refine_stop():
    strategy = DefaultStrategy(
        StrategyConfig(
            refine_start=0,
            refine_stop=6,
            refine_every=1,
            reset_every=3,
        )
    )

    assert strategy.should_reset(3)
    assert not strategy.should_reset(0)
    # current-main returns from the post-backward hook before its reset once
    # refinement has stopped.
    assert not strategy.should_reset(6)
    assert not strategy.should_reset(9)


def test_default_strategy_splits_before_pruning_large_gaussians():
    model = GaussianModel.empty(ModelConfig(capacity=2, sh_degree=0))
    model.active_mask[0] = True
    model.log_scales[0] = jnp.log(1.5)
    model.opacity_logits[0] = jnp.log(4.0)
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = DefaultStrategy(
        StrategyConfig(
            reset_every=1,
            max_new_per_refine=1,
            grow_grad2d=0.1,
            grow_scale3d=0.5,
            prune_scale3d=1.0,
        )
    )
    state = strategy.initialize_state(2)
    state.grad_accum[0] = 1.0
    state.visible_count[0] = 1.0

    plan = strategy.plan_refine(model, state, 1.0, step=2)

    assert int(plan["planned_new_count"]) == 1
    assert int(plan["required_capacity"]) == 2
    assert int(plan["pruned_count"]) == 0
    assert int(plan["active_after_prune_count"]) == 2

    result = strategy.refine(
        model, state, optimizer, jax.random.key(29), 1.0, step=2
    )

    assert int(result["new_count"]) == 1
    assert int(result["pruned_count"]) == 0
    assert int(model.active_count) == 2
    assert jnp.allclose(jnp.exp(model.log_scales[:2]), 1.5 / 1.6)


def test_default_strategy_revised_opacity_matches_split_and_prune_plan():
    model = GaussianModel.empty(ModelConfig(capacity=4, sh_degree=0))
    model.active_mask[:2] = True
    model.log_scales[:2] = jnp.log(jnp.asarray([[1.0] * 3, [0.1] * 3]))
    original_opacity = 0.36
    original_logit = jnp.log(original_opacity / (1.0 - original_opacity))
    model.opacity_logits[:2] = original_logit
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = DefaultStrategy(
        StrategyConfig(
            reset_every=100,
            max_new_per_refine=2,
            grow_grad2d=0.5,
            grow_scale3d=0.5,
            prune_opacity=0.25,
            prune_scale3d=100.0,
            revised_opacity=True,
        )
    )
    state = strategy.initialize_state(4)
    state.grad_accum[:2] = jnp.asarray([2.0, 1.0])
    state.visible_count[:2] = 1.0

    plan = strategy.plan_refine(model, state, 1.0, step=1)

    assert int(plan["planned_new_count"]) == 2
    assert int(plan["pruned_count"]) == 2
    assert int(plan["active_after_prune_count"]) == 2

    result = strategy.refine(
        model, state, optimizer, jax.random.key(37), 1.0, step=1
    )

    revised_opacity = 1.0 - math.sqrt(1.0 - original_opacity)
    stored_opacities = jnp.sort(jax.nn.sigmoid(model.opacity_logits[...]))
    assert int(result["new_count"]) == 2
    assert int(result["pruned_count"]) == int(plan["pruned_count"])
    assert int(model.active_count) == int(plan["active_after_prune_count"])
    assert jnp.allclose(
        stored_opacities,
        jnp.asarray(
            [revised_opacity, revised_opacity, original_opacity, original_opacity]
        ),
    )
    assert jnp.allclose(
        jax.nn.sigmoid(model.opacity_logits[...][model.active_mask[...]]),
        original_opacity,
    )


def test_default_strategy_prunes_low_opacity_parent_and_duplicate_after_growth():
    model = GaussianModel.empty(ModelConfig(capacity=3, sh_degree=0))
    model.active_mask[:2] = True
    model.log_scales[:2] = jnp.log(0.1)
    model.opacity_logits[:2] = jnp.asarray([-20.0, jnp.log(4.0)])
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = DefaultStrategy(
        StrategyConfig(
            reset_every=100,
            max_new_per_refine=1,
            grow_grad2d=0.5,
            grow_scale3d=1.0,
            prune_opacity=0.01,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(3)
    state.grad_accum[:2] = jnp.asarray([1.0, 0.0])
    state.visible_count[:2] = 1.0

    plan = strategy.plan_refine(model, state, 1.0, step=1)

    assert int(plan["planned_new_count"]) == 1
    assert int(plan["free_count"]) == 1
    assert int(plan["required_capacity"]) == 3
    assert int(plan["pruned_count"]) == 2
    assert int(plan["active_after_prune_count"]) == 1
    assert not bool(plan["capacity_overflow"])

    result = strategy.refine(
        model, state, optimizer, jax.random.key(31), 1.0, step=1
    )

    assert int(result["new_count"]) == 1
    assert int(result["pruned_count"]) == 2
    assert int(model.active_count) == 1
    assert jnp.array_equal(
        model.active_mask[...], jnp.asarray([False, True, False])
    )


def test_default_strategy_prunes_low_opacity_slots():
    model = GaussianModel.empty(ModelConfig(capacity=8, sh_degree=0))
    model.active_mask[:2] = True
    model.opacity_logits[:2] = jnp.array([-20.0, 2.0])
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = DefaultStrategy(
        StrategyConfig(
            max_new_per_refine=2,
            grow_grad2d=100.0,
            prune_opacity=0.01,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(8)
    stats = strategy.refine(model, state, optimizer, jax.random.key(0), 1.0)
    assert int(stats["pruned_count"]) == 1
    assert int(model.active_count) == 1


def test_default_strategy_overflow_uses_bounded_plan():
    model = GaussianModel.empty(ModelConfig(capacity=8, sh_degree=0))
    model.active_mask[:7] = True
    model.opacity_logits[:7] = 2.0
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = DefaultStrategy(
        StrategyConfig(
            max_new_per_refine=1,
            grow_grad2d=0.1,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(8)
    state.grad_accum[:7] = 1.0
    state.visible_count[:7] = 1.0

    plan = strategy.plan_refine(model, state, 1.0)
    assert int(plan["planned_new_count"]) == 1
    assert int(plan["required_capacity"]) == 8
    assert int(strategy.required_capacity(model, state, 1.0)) == 8
    assert not bool(plan["capacity_overflow"])
    assert int(model.active_count) == 7

    stats = strategy.refine(model, state, optimizer, jax.random.key(0), 1.0)
    assert int(stats["new_count"]) == 1
    assert not bool(stats["capacity_overflow"])


def test_mcmc_strategy_overflow_uses_bounded_plan():
    model = GaussianModel.empty(ModelConfig(capacity=42, sh_degree=0))
    model.active_mask[:41] = True
    model.opacity_logits[:41] = 2.0
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = MCMCStrategy(
        StrategyConfig(
            kind="mcmc",
            max_new_per_refine=1,
            prune_scale3d=100.0,
        )
    )
    state = strategy.initialize_state(42)

    plan = strategy.plan_refine(model, state, 1.0)
    assert int(plan["planned_new_count"]) == 1
    assert int(plan["required_capacity"]) == 42
    assert int(strategy.required_capacity(model, state, 1.0)) == 42
    assert not bool(plan["capacity_overflow"])

    stats = strategy.refine(model, state, optimizer, jax.random.key(0), 1.0)
    assert int(stats["new_count"]) == 1
    assert not bool(stats["capacity_overflow"])


def test_compiled_mcmc_strategy_copies_appearance_rows_for_birth_and_relocation():
    birth_model = GaussianModel.empty(
        ModelConfig(capacity=21, bucket_min_capacity=21),
        appearance_feature_dim=32,
    )
    birth_model.active_mask[:20] = True
    birth_model.opacity_logits[:20] = jnp.log(4.0)
    birth_model.features[:20] = jnp.broadcast_to(
        jnp.arange(20, dtype=jnp.float32)[:, None], (20, 32)
    )
    birth_model.colors[:20] = jnp.broadcast_to(
        jnp.arange(20, dtype=jnp.float32)[:, None], (20, 3)
    )
    birth_optimizer = create_optimizer(
        birth_model, OptimizerConfig(max_steps=10)
    )
    birth_strategy = MCMCStrategy(
        StrategyConfig(kind="mcmc", cap_max=21, max_new_per_refine=2)
    )
    birth_state = birth_strategy.initialize_state(21)

    birth = birth_strategy.refine(
        birth_model,
        birth_state,
        birth_optimizer,
        jax.random.key(7),
        1.0,
    )
    assert int(birth["new_count"]) == 1
    birth_target = 20
    birth_source = int(birth_model.features[birth_target, 0])
    assert jnp.array_equal(
        birth_model.features[birth_target], birth_model.features[birth_source]
    )
    assert jnp.array_equal(
        birth_model.colors[birth_target], birth_model.colors[birth_source]
    )

    relocate_model = GaussianModel.empty(
        ModelConfig(capacity=2, bucket_min_capacity=2),
        appearance_feature_dim=32,
    )
    relocate_model.active_mask[...] = True
    relocate_model.opacity_logits[...] = jnp.asarray([-20.0, jnp.log(4.0)])
    relocate_model.features[1] = jnp.arange(32, dtype=jnp.float32)
    relocate_model.colors[1] = jnp.asarray([1.0, 2.0, 3.0])
    relocate_optimizer = create_optimizer(
        relocate_model, OptimizerConfig(max_steps=10)
    )
    relocate_strategy = MCMCStrategy(
        StrategyConfig(
            kind="mcmc",
            cap_max=2,
            max_new_per_refine=2,
            prune_opacity=0.01,
        )
    )
    relocate_state = relocate_strategy.initialize_state(2)

    relocation = relocate_strategy.refine(
        relocate_model,
        relocate_state,
        relocate_optimizer,
        jax.random.key(8),
        1.0,
    )
    assert int(relocation["relocated_count"]) == 1
    relocate_source = 1
    relocate_target = 0
    assert jnp.array_equal(
        relocate_model.features[relocate_target],
        relocate_model.features[relocate_source],
    )
    assert jnp.array_equal(
        relocate_model.colors[relocate_target],
        relocate_model.colors[relocate_source],
    )


def test_mcmc_strategy_capacity_overflow_is_atomic():
    capacity = 20
    model = GaussianModel.empty(ModelConfig(capacity=capacity, sh_degree=0))
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    optimizer.update(
        model,
        jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param)),
    )
    model.active_mask[:] = True
    model.means[...] = jnp.arange(capacity * 3, dtype=jnp.float32).reshape(
        capacity, 3
    )
    model.log_scales[...] = jnp.log(0.1)
    model.quats[...] = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    model.opacity_logits[...] = jnp.log(4.0)
    model.opacity_logits[0] = -20.0
    strategy = MCMCStrategy(
        StrategyConfig(
            kind="mcmc",
            max_new_per_refine=2,
            prune_opacity=0.01,
            cap_max=40,
        )
    )
    state = strategy.initialize_state(capacity)
    state.grad_accum[...] = jnp.arange(capacity, dtype=jnp.float32) + 1.0
    state.visible_count[...] = 2.0
    state.max_radii[...] = 3.0
    model_before = {
        name: jnp.array(value) for name, value in model.state_dict().items()
    }
    optimizer_before = _state_leaves(optimizer)
    stats_before = (
        jnp.array(state.grad_accum[...]),
        jnp.array(state.visible_count[...]),
        jnp.array(state.max_radii[...]),
    )

    plan = strategy.plan_refine(model, state, 1.0)
    assert int(plan["planned_new_count"]) == 1
    assert int(plan["planned_relocate_count"]) == 1
    assert int(plan["free_count"]) == 0
    assert bool(plan["capacity_overflow"])

    result = strategy.refine(
        model, state, optimizer, jax.random.key(45), 1.0
    )

    assert int(result["new_count"]) == 0
    assert int(result["pruned_count"]) == 0
    assert int(result["relocated_count"]) == 0
    assert bool(result["capacity_overflow"])
    assert int(result["active_count"]) == capacity
    for name, before in model_before.items():
        assert jnp.array_equal(model.state_dict()[name], before)
    for after, before in zip(_state_leaves(optimizer), optimizer_before, strict=True):
        assert jnp.array_equal(after, before)
    assert jnp.array_equal(state.grad_accum[...], stats_before[0])
    assert jnp.array_equal(state.visible_count[...], stats_before[1])
    assert jnp.array_equal(state.max_radii[...], stats_before[2])


def test_mcmc_strategy_treats_equal_min_opacity_as_dead():
    model = GaussianModel.empty(ModelConfig(capacity=3, sh_degree=0))
    model.active_mask[:2] = True
    model.opacity_logits[:2] = jnp.asarray([0.0, jnp.log(4.0)])
    strategy = MCMCStrategy(
        StrategyConfig(
            kind="mcmc",
            max_new_per_refine=2,
            prune_opacity=0.5,
            cap_max=2,
        )
    )
    state = strategy.initialize_state(3)

    plan = strategy.plan_refine(model, state, 1.0)

    assert int(plan["planned_relocate_count"]) == 1


def test_mcmc_strategy_does_not_force_a_birth_below_twenty_points():
    model = GaussianModel.empty(ModelConfig(capacity=20, sh_degree=0))
    model.active_mask[:19] = True
    model.opacity_logits[:19] = jnp.log(4.0)
    strategy = MCMCStrategy(
        StrategyConfig(
            kind="mcmc",
            max_new_per_refine=4,
            cap_max=20,
        )
    )
    state = strategy.initialize_state(20)

    plan = strategy.plan_refine(model, state, 1.0)

    assert int(plan["planned_new_count"]) == 0
    assert int(plan["required_capacity"]) == 19


def test_mcmc_weighted_sampling_has_linear_intermediate_shapes():
    abstract_weights = jax.ShapeDtypeStruct((1_000_000,), jnp.float32)
    output_shapes = jax.eval_shape(
        lambda weights: _sample_weighted_ids(
            jax.random.key(0), weights, 8_192
        ),
        abstract_weights,
    )
    assert output_shapes[0].shape == (8_192,)
    assert output_shapes[1].shape == ()

    jaxpr = str(
        jax.make_jaxpr(
            lambda weights: _sample_weighted_ids(
                jax.random.key(0), weights, 8_192
            )
        )(abstract_weights)
    )
    assert "f32[8192,1000000]" not in jaxpr


def test_mcmc_refine_is_noop_when_no_live_donor_exists():
    model = GaussianModel.empty(ModelConfig(capacity=8, sh_degree=0))
    model.active_mask[:2] = True
    model.opacity_logits[:2] = -20.0
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = MCMCStrategy(
        StrategyConfig(kind="mcmc", max_new_per_refine=2, prune_opacity=0.01)
    )
    state = strategy.initialize_state(8)
    means_before = model.means[...].copy()
    plan = strategy.plan_refine(model, state, 1.0)

    assert int(plan["planned_new_count"]) == 0
    assert int(plan["planned_relocate_count"]) == 0
    assert int(plan["required_capacity"]) == 2
    assert not bool(plan["capacity_overflow"])

    stats = strategy.refine(model, state, optimizer, jax.random.key(0), 1.0)

    assert int(stats["new_count"]) == 0
    assert int(model.active_count) == 2
    assert jnp.array_equal(model.means[...], means_before)


def test_mcmc_refine_uses_equation_nine_for_donor_and_relocated_slot():
    model = GaussianModel.empty(ModelConfig(capacity=2, sh_degree=0))
    model.active_mask[:] = True
    model.means[:] = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    model.log_scales[:] = jnp.log(jnp.asarray([[0.5] * 3, [2.0] * 3]))
    model.opacity_logits[:] = jnp.asarray([-20.0, jnp.log(4.0)])
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    strategy = MCMCStrategy(
        StrategyConfig(kind="mcmc", max_new_per_refine=2, prune_opacity=0.01)
    )
    state = strategy.initialize_state(2)
    stats = strategy.refine(model, state, optimizer, jax.random.key(0), 1.0)

    original_opacity = 0.8
    expected_opacity = 1.0 - math.sqrt(1.0 - original_opacity)
    denominator = 2 * expected_opacity - expected_opacity**2 / math.sqrt(2)
    expected_scale = 2.0 * original_opacity / denominator
    opacities = jax.nn.sigmoid(model.opacity_logits[...])
    scales = jnp.exp(model.log_scales[...])

    assert int(stats["new_count"]) == 0
    assert not bool(stats["capacity_overflow"])
    assert jnp.allclose(model.means[0], model.means[1])
    assert jnp.allclose(opacities, expected_opacity, rtol=1e-6)
    assert jnp.allclose(scales, expected_scale, rtol=1e-6)
