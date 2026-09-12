import math

import jax
import jax.numpy as jnp
import pytest

from jax_gs.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    RasterizationConfig,
    StrategyConfig,
    TrainConfig,
)
from jax_gs.math import quat_scale_to_covar_preci
from jax_gs.model import GaussianModel
from jax_gs.optimizers import SelectiveAdam, create_optimizer
from jax_gs.optimizers.selective_adam import SelectiveAdam as SelectiveAdamPath
from jax_gs.strategy import (
    DefaultStrategy,
    MCMCStrategy,
    Strategy,
    StrategyState,
    accumulate_densification_stats,
    build_densification_stats,
    update_strategy_state,
)
from jax_gs.strategy.base import Strategy as StrategyPath
from jax_gs.strategy.default import DefaultStrategy as DefaultStrategyPath
from jax_gs.strategy.mcmc import MCMCStrategy as MCMCStrategyPath
from jax_gs.strategy.ops import (
    _cuda_fused_mcmc_perturb,
    _resolve_noise_scale,
    duplicate,
    inject_noise_to_position,
    mcmc_position_perturbation,
    relocate,
    remove,
    sample_add,
    split,
)
from jax_gs.training import TrainingSafetyState, make_train_step


def _stateful_model(capacity=6, active=2):
    model = GaussianModel.empty(
        ModelConfig(capacity=capacity, bucket_min_capacity=capacity, sh_degree=0)
    )
    model.active_mask[:active] = True
    model.means[:active] = jnp.arange(active * 3, dtype=jnp.float32).reshape(active, 3)
    model.log_scales[:active] = jnp.log(0.1)
    model.opacity_logits[:active] = jnp.log(4.0)
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    state = StrategyState(capacity)
    state.grad_accum[:active] = jnp.arange(1, active + 1, dtype=jnp.float32)
    state.visible_count[:active] = 1.0
    return model, optimizer, state


def _binomials(size=5):
    return jnp.asarray(
        [
            [math.comb(row, column) if column <= row else 0 for column in range(size)]
            for row in range(size)
        ],
        jnp.float32,
    )


def test_current_main_strategy_and_optimizer_import_paths_are_live():
    assert StrategyPath is Strategy
    assert DefaultStrategyPath is DefaultStrategy
    assert MCMCStrategyPath is MCMCStrategy
    assert SelectiveAdamPath is SelectiveAdam
    assert issubclass(DefaultStrategy, Strategy)


def test_current_main_constructor_fields_map_to_strategy_config():
    default = DefaultStrategy(
        prune_opa=0.02,
        grow_scale2d=0.07,
        refine_start_iter=7,
        pause_refine_after_reset=3,
        key_for_gradient="gradient_2dgs",
    )
    assert default.config.prune_opacity == 0.02
    assert default.config.grow_scale2d == 0.07
    assert default.config.refine_start == 7
    assert default.config.pause_refine_after_reset == 3
    assert default.config.key_for_gradient == "gradient_2dgs"

    mcmc = MCMCStrategy(
        cap_max=123,
        noise_lr=4.0,
        noise_injection_stop_iter=9,
        min_opacity=0.02,
    )
    assert mcmc.refine_stop_iter == 25_000
    assert mcmc.config.cap_max == 123
    assert mcmc.config.noise_lr == 4.0
    assert mcmc.config.prune_opacity == 0.02


def test_default_strategy_accepts_current_main_positional_constructor():
    strategy = DefaultStrategy(
        0.02,
        0.0003,
        0.03,
        0.04,
        0.2,
        0.25,
        1_000,
        10,
        20_000,
        2_000,
        50,
        25,
        True,
        True,
        True,
        "gradient_2dgs",
    )

    assert strategy.prune_opa == 0.02
    assert strategy.grow_grad2d == 0.0003
    assert strategy.refine_start_iter == 10
    assert strategy.refine_stop_iter == 20_000
    assert strategy.key_for_gradient == "gradient_2dgs"
    assert strategy.config.prune_opacity == 0.02
    assert strategy.config.absgrad


def test_mcmc_strategy_accepts_current_main_positional_constructor():
    strategy = MCMCStrategy(
        123_456,
        4.0e5,
        20,
        21_000,
        19_000,
        50,
        0.02,
        True,
        0.03,
        80.0,
    )

    assert strategy.cap_max == 123_456
    assert strategy.noise_lr == 4.0e5
    assert strategy.refine_start_iter == 20
    assert strategy.refine_stop_iter == 21_000
    assert strategy.noise_injection_stop_iter == 19_000
    assert strategy.refine_every == 50
    assert strategy.min_opacity == 0.02
    assert strategy.verbose
    assert strategy.noise_opacity_t == 0.03
    assert strategy.noise_opacity_k == 80.0
    assert strategy.config.kind == "mcmc"


def test_current_main_strategy_state_initializers_are_callable_without_capacity():
    default_state = DefaultStrategy().initialize_state()
    mcmc_state = MCMCStrategy().initialize_state()

    assert isinstance(default_state, StrategyState)
    assert isinstance(mcmc_state, StrategyState)
    assert default_state.grad_accum.shape == (0,)
    assert mcmc_state.grad_accum.shape == (0,)
    assert float(default_state.scene_scale[...]) == 1.0


def test_default_strategy_lazy_state_materializes_at_first_model_callback():
    model, optimizer, _ = _stateful_model(capacity=4, active=1)
    strategy = DefaultStrategy()
    state = strategy.initialize_state(scene_scale=2.5)

    strategy.step_pre_backward(
        model,
        optimizer,
        state,
        1,
        {"means2d": jnp.zeros((4, 2), dtype=jnp.float32)},
    )

    assert state.grad_accum.shape == (4,)
    assert state.visible_count.shape == (4,)
    assert state.max_radii.shape == (4,)
    assert float(state.scene_scale[...]) == 2.5


def test_dense_densification_stats_match_upstream_multicamera_oracle():
    screen_grad = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[-1.0, 1.0], [2.0, -2.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        jnp.float32,
    )
    radii = jnp.asarray(
        [
            [[2.0, 4.0], [1.0, 2.0], [7.0, 8.0], [0.0, 4.0]],
            [[6.0, 1.0], [5.0, 6.0], [9.0, 10.0], [3.0, 0.0]],
        ],
        jnp.float32,
    )
    valid = jnp.asarray([[True, True, True, True], [True, False, True, True]])
    active = jnp.asarray([True, True, False, True])

    stats = jax.jit(build_densification_stats)(
        screen_grad,
        radii,
        valid,
        active,
        8,
        4,
    )

    expected_grad_sum = jnp.asarray(
        [
            math.hypot(8.0, 8.0) + math.hypot(-8.0, 4.0),
            math.hypot(24.0, 16.0),
            0.0,
            0.0,
        ],
        jnp.float32,
    )
    assert jnp.allclose(stats.grad_sum, expected_grad_sum)
    assert jnp.array_equal(stats.count, jnp.asarray([2.0, 1.0, 0.0, 0.0], jnp.float32))
    assert jnp.allclose(
        stats.max_radii, jnp.asarray([0.75, 0.25, 0.0, 0.0], jnp.float32)
    )


def test_densification_stats_accumulator_adds_counts_and_takes_radius_maximum():
    state = StrategyState(3)
    state.grad_accum[...] = jnp.asarray([1.0, 2.0, 3.0])
    state.visible_count[...] = jnp.asarray([4.0, 5.0, 6.0])
    state.max_radii[...] = jnp.asarray([0.1, 0.8, 0.3])
    stats = build_densification_stats(
        jnp.asarray([[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]]),
        jnp.asarray([[[2.0, 1.0], [4.0, 2.0], [6.0, 3.0]]]),
        jnp.asarray([[True, True, False]]),
        jnp.asarray([True, True, True]),
        4,
        2,
    )

    accumulate_densification_stats(state, stats)

    assert jnp.allclose(state.grad_accum[...], jnp.asarray([3.0, 4.0, 3.0]))
    assert jnp.array_equal(state.visible_count[...], jnp.asarray([5.0, 6.0, 6.0]))
    assert jnp.allclose(state.max_radii[...], jnp.asarray([0.5, 1.0, 0.3]))


def test_legacy_update_strategy_state_keeps_one_gradient_one_count_semantics():
    model, _, state = _stateful_model(capacity=3, active=2)
    state.grad_accum[...] = jnp.asarray([1.0, 2.0, 3.0])
    state.visible_count[...] = jnp.asarray([4.0, 5.0, 6.0])
    state.max_radii[...] = jnp.asarray([0.1, 0.8, 0.3])

    update_strategy_state(
        state,
        model,
        jnp.asarray([[3.0, 4.0], [0.0, 2.0], [6.0, 8.0]]),
        jnp.asarray([True, True, True]),
        jnp.asarray([0.5, 0.4, 0.9]),
    )

    assert jnp.allclose(state.grad_accum[...], jnp.asarray([6.0, 4.0, 3.0]))
    assert jnp.array_equal(state.visible_count[...], jnp.asarray([5.0, 6.0, 6.0]))
    assert jnp.allclose(state.max_radii[...], jnp.asarray([0.5, 0.8, 0.3]))


def test_default_strategy_uses_2d_scale_only_inside_configured_window():
    model, _, state = _stateful_model(capacity=4, active=1)
    state.grad_accum[...] = 0.0
    state.max_radii[0] = 0.1
    strategy = DefaultStrategy(
        StrategyConfig(
            refine_scale2d_stop_iter=10,
            grow_scale2d=0.05,
            prune_scale2d=1.0,
            grow_grad2d=100.0,
            prune_scale3d=100.0,
            max_new_per_refine=1,
        )
    )
    early = strategy.plan_refine(model, state, 1.0, step=1)
    late = strategy.plan_refine(model, state, 1.0, step=11)
    assert int(early["planned_new_count"]) == 1
    assert int(late["planned_new_count"]) == 0


def test_fixed_capacity_duplicate_split_and_remove_keep_rows_aligned():
    model, optimizer, state = _stateful_model()
    source_mean = model.means[0].copy()
    duplicated = duplicate(
        model,
        optimizer,
        state,
        jnp.asarray([True, False, False, False, False, False]),
    )
    assert duplicated.tolist() == [2]
    assert jnp.array_equal(model.means[2], source_mean)
    assert float(state.grad_accum[2]) == 1.0

    removed = remove(
        model,
        optimizer,
        state,
        jnp.asarray([False, True, False, False, False, False]),
    )
    assert removed.tolist() == [1]
    assert not bool(model.active_mask[1])
    assert float(state.grad_accum[1]) == 0.0

    old_scale = model.log_scales[0].copy()
    children = split(
        model,
        optimizer,
        state,
        jnp.asarray([True, False, False, False, False, False]),
        key=jax.random.key(3),
    )
    assert children.tolist() == [1]
    assert bool(model.active_mask[1])
    assert jnp.allclose(model.log_scales[0], old_scale - jnp.log(1.6))
    assert jnp.allclose(model.log_scales[1], model.log_scales[0])


def test_appearance_rows_follow_duplicate_split_sample_and_relocation():
    model = GaussianModel.empty(
        ModelConfig(capacity=6, bucket_min_capacity=6),
        appearance_feature_dim=32,
    )
    model.active_mask[:2] = True
    model.features[:2] = jnp.stack(
        (jnp.arange(32, dtype=jnp.float32), jnp.arange(32) + 100)
    )
    model.colors[:2] = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    model.opacity_logits[:2] = jnp.log(4.0)
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    state = StrategyState(6)
    state.grad_accum[:2] = 1.0
    state.visible_count[:2] = 1.0

    duplicate_target = int(
        duplicate(
            model,
            optimizer,
            state,
            jnp.asarray([True, False, False, False, False, False]),
        )[0]
    )
    assert jnp.array_equal(model.features[duplicate_target], model.features[0])
    assert jnp.array_equal(model.colors[duplicate_target], model.colors[0])

    remove(
        model,
        optimizer,
        state,
        jnp.asarray([False, True, False, False, False, False]),
    )
    split_target = int(
        split(
            model,
            optimizer,
            state,
            jnp.asarray([True, False, False, False, False, False]),
            key=jax.random.key(3),
        )[0]
    )
    assert jnp.array_equal(model.features[split_target], model.features[0])
    assert jnp.array_equal(model.colors[split_target], model.colors[0])

    targets, donors = sample_add(
        model,
        optimizer,
        state,
        1,
        _binomials(),
        key=jax.random.key(4),
    )
    assert jnp.array_equal(model.features[targets[0]], model.features[donors[0]])
    assert jnp.array_equal(model.colors[targets[0]], model.colors[donors[0]])

    model.opacity_logits[0] = -20.0
    dead, donors = relocate(
        model,
        optimizer,
        state,
        jnp.asarray([True, False, False, False, False, False]),
        _binomials(),
        min_opacity=0.01,
        key=jax.random.key(5),
    )
    assert jnp.array_equal(model.features[dead[0]], model.features[donors[0]])
    assert jnp.array_equal(model.colors[dead[0]], model.colors[donors[0]])


def test_fixed_capacity_mcmc_sample_and_relocation_use_live_donors():
    model, optimizer, state = _stateful_model(capacity=5, active=2)
    targets, sampled = sample_add(
        model,
        optimizer,
        state,
        1,
        _binomials(),
        key=jax.random.key(2),
    )
    assert targets.tolist() == [2]
    assert sampled.shape == (1,)
    assert bool(model.active_mask[2])
    assert jnp.allclose(model.means[2], model.means[sampled[0]])

    model.opacity_logits[0] = -20.0
    dead, donors = relocate(
        model,
        optimizer,
        state,
        jnp.asarray([True, False, False, False, False]),
        _binomials(),
        min_opacity=0.01,
        key=jax.random.key(5),
    )
    assert dead.tolist() == [0]
    assert donors.shape == (1,)
    assert jnp.allclose(model.means[0], model.means[donors[0]])
    assert float(jax.nn.sigmoid(model.opacity_logits[0])) >= 0.01 - 1.0e-6


def test_mcmc_noise_matches_covariance_reference_and_alias_validation():
    positions = jnp.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2)
    log_scales = jnp.log(jnp.asarray([[0.5, 1.0, 2.0], [1.0, 1.0, 1.0]]))
    logits = jnp.asarray([-2.0, 5.0])
    key = jax.random.key(7)
    result = mcmc_position_perturbation(
        positions,
        quats,
        log_scales,
        logits,
        0.25,
        key=key,
        t=0.25,
        k=8.0,
    )
    covariance, _ = quat_scale_to_covar_preci(
        quats,
        jnp.exp(log_scales),
        compute_covar=True,
        compute_preci=False,
    )
    noise = jax.random.normal(key, positions.shape)
    gate = jax.nn.sigmoid(-8.0 * (jax.nn.sigmoid(logits) - 0.25))
    expected = positions + jnp.einsum(
        "nij,nj->ni", covariance, noise * gate[:, None] * 0.25
    )
    assert jnp.allclose(result, expected)
    assert _resolve_noise_scale(None, 0.25) == 0.25
    with pytest.raises(ValueError, match="different values"):
        _resolve_noise_scale(0.2, 0.3)
    assert not _cuda_fused_mcmc_perturb(positions, quats, log_scales, logits, 0.1)


def test_explicit_jax_strategy_hook_accumulates_and_refines():
    model, optimizer, state = _stateful_model(capacity=4, active=1)
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
    strategy.step_pre_backward(
        model,
        optimizer,
        state,
        1,
        {"means2d": jnp.zeros((4, 2))},
    )
    result = strategy.step_post_backward(
        model,
        optimizer,
        state,
        1,
        {
            "means_gradient": jnp.ones((4, 3)),
            "visible": jnp.asarray([True, False, False, False]),
            "radii": jnp.asarray([0.1, 0.0, 0.0, 0.0]),
        },
        key=jax.random.key(9),
    )
    assert int(result["new_count"]) == 1
    assert int(model.active_count) == 2


def test_default_strategy_hook_accumulates_dense_screen_space_statistics():
    model, optimizer, _ = _stateful_model(capacity=3, active=2)
    state = StrategyState(3)
    strategy = DefaultStrategy(refine_start_iter=100, refine_stop_iter=200)
    screen_grad = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, 1.0], [2.0, -2.0], [7.0, 8.0]],
        ],
        jnp.float32,
    )
    radii = jnp.asarray(
        [
            [[2.0, 4.0], [1.0, 2.0], [7.0, 8.0]],
            [[6.0, 1.0], [5.0, 6.0], [9.0, 10.0]],
        ],
        jnp.float32,
    )
    valid = jnp.asarray([[True, True, True], [True, False, True]])

    strategy.step_post_backward(
        model,
        optimizer,
        state,
        1,
        {
            "means2d_gradient": screen_grad,
            "radii": radii,
            "valid": valid,
            "width": 8,
            "height": 4,
            "n_cameras": 2,
        },
    )

    expected = build_densification_stats(
        screen_grad,
        radii,
        valid,
        model.active_mask[...],
        8,
        4,
    )
    assert jnp.allclose(state.grad_accum[...], expected.grad_sum)
    assert jnp.array_equal(state.visible_count[...], expected.count)
    assert jnp.allclose(state.max_radii[...], expected.max_radii)


def test_default_strategy_hook_scatter_accumulates_padded_packed_statistics():
    model, optimizer, _ = _stateful_model(capacity=3, active=2)
    state = StrategyState(3)
    strategy = DefaultStrategy(refine_start_iter=100, refine_stop_iter=200)

    strategy.step_post_backward(
        model,
        optimizer,
        state,
        1,
        {
            "means2d_gradient": jnp.asarray(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [100.0, 100.0],
                    [100.0, 100.0],
                    [100.0, 100.0],
                ],
                jnp.float32,
            ),
            "radii": jnp.asarray(
                [
                    [2.0, 4.0],
                    [6.0, 1.0],
                    [5.0, 6.0],
                    [9.0, 9.0],
                    [9.0, 9.0],
                    [9.0, 9.0],
                ],
                jnp.float32,
            ),
            "valid": jnp.asarray([True, True, True, False, False, False]),
            "gaussian_ids": jnp.asarray([1, 0, 1, -1, -1, -1], jnp.int32),
            "projection_valid_count": jnp.asarray(3, jnp.int32),
            "width": 8,
            "height": 4,
            "n_cameras": 2,
        },
        packed=True,
    )

    assert jnp.allclose(
        state.grad_accum[...],
        jnp.asarray([8.0, 4.0 + math.sqrt(80.0), 0.0]),
    )
    assert jnp.array_equal(state.visible_count[...], jnp.asarray([1.0, 2.0, 0.0]))
    assert jnp.allclose(state.max_radii[...], jnp.asarray([0.75, 0.75, 0.0]))


def test_default_strategy_absgrad_hook_uses_explicit_dense_statistic():
    model, optimizer, _ = _stateful_model(capacity=3, active=2)
    state = StrategyState(3)
    strategy = DefaultStrategy(
        absgrad=True,
        refine_start_iter=100,
        refine_stop_iter=200,
    )
    abs_gradient = jnp.asarray(
        [[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]],
        jnp.float32,
    )
    radii = jnp.asarray(
        [[[2.0, 4.0], [6.0, 2.0], [8.0, 8.0]]],
        jnp.float32,
    )
    valid = jnp.asarray([[True, True, True]])

    strategy.step_pre_backward(
        model,
        optimizer,
        state,
        1,
        {"means2d": jnp.zeros_like(abs_gradient)},
    )
    strategy.step_post_backward(
        model,
        optimizer,
        state,
        1,
        {
            "means2d_gradient": -100.0 * jnp.ones_like(abs_gradient),
            "means2d_absgrad": abs_gradient,
            "radii": radii,
            "valid": valid,
            "width": 8,
            "height": 4,
            "n_cameras": 1,
        },
    )

    expected = build_densification_stats(
        abs_gradient,
        radii,
        valid,
        model.active_mask[...],
        8,
        4,
    )
    assert jnp.allclose(state.grad_accum[...], expected.grad_sum)
    assert jnp.array_equal(state.visible_count[...], expected.count)
    assert jnp.allclose(state.max_radii[...], expected.max_radii)


def test_default_strategy_absgrad_hook_requires_explicit_statistic():
    model, optimizer, state = _stateful_model(capacity=3, active=1)
    strategy = DefaultStrategy(absgrad=True)

    with pytest.raises(ValueError, match="means2d_absgrad"):
        strategy.step_post_backward(
            model,
            optimizer,
            state,
            1,
            {
                "means2d_gradient": jnp.ones((1, 3, 2), jnp.float32),
                "radii": jnp.ones((1, 3, 2), jnp.float32),
                "valid": jnp.ones((1, 3), jnp.bool_),
                "width": 8,
                "height": 4,
                "n_cameras": 1,
            },
        )


def test_default_strategy_absgrad_hook_scatter_accumulates_packed_statistic():
    model, optimizer, _ = _stateful_model(capacity=3, active=2)
    state = StrategyState(3)
    strategy = DefaultStrategy(
        absgrad=True,
        refine_start_iter=100,
        refine_stop_iter=200,
    )

    strategy.step_post_backward(
        model,
        optimizer,
        state,
        1,
        {
            "means2d_absgrad": jnp.asarray(
                [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
                jnp.float32,
            ),
            "radii": jnp.asarray(
                [[2.0, 4.0], [6.0, 2.0], [9.0, 9.0]],
                jnp.float32,
            ),
            "valid": jnp.asarray([True, True, False]),
            "gaussian_ids": jnp.asarray([1, 0, -1], jnp.int32),
            "projection_valid_count": jnp.asarray(2, jnp.int32),
            "width": 8,
            "height": 4,
            "n_cameras": 1,
        },
        packed=True,
    )

    assert jnp.allclose(
        state.grad_accum[...],
        jnp.asarray([4.0 * jnp.sqrt(13.0), 4.0 * jnp.sqrt(2.0), 0.0]),
    )
    assert jnp.array_equal(state.visible_count[...], jnp.asarray([1.0, 1.0, 0.0]))


def test_mcmc_hook_injects_noise_only_into_active_slots():
    model, optimizer, state = _stateful_model(capacity=4, active=1)
    before = model.means[...].copy()
    strategy = MCMCStrategy(
        StrategyConfig(
            kind="mcmc",
            refine_start=100,
            noise_lr=2.0,
            noise_opacity_t=0.9,
            noise_opacity_k=10.0,
        )
    )
    result = strategy.step_post_backward(
        model,
        optimizer,
        state,
        1,
        {},
        lr=0.1,
        key=jax.random.key(4),
    )
    assert bool(result["noise_injected"])
    assert not jnp.array_equal(model.means[0], before[0])
    assert jnp.array_equal(model.means[1:], before[1:])


def test_mcmc_hook_overflow_skips_noise_injection():
    model, optimizer, state = _stateful_model(capacity=20, active=20)
    model.opacity_logits[0] = -20.0
    before = model.means[...].copy()
    strategy = MCMCStrategy(
        StrategyConfig(
            kind="mcmc",
            refine_start=0,
            refine_stop=10,
            refine_every=1,
            max_new_per_refine=2,
            cap_max=40,
            prune_opacity=0.01,
            noise_lr=2.0,
            noise_opacity_t=0.9,
            noise_opacity_k=10.0,
        )
    )

    result = strategy.step_post_backward(
        model,
        optimizer,
        state,
        1,
        {},
        lr=0.1,
        key=jax.random.key(12),
    )

    assert bool(result["capacity_overflow"])
    assert not bool(result["noise_injected"])
    assert int(result["new_count"]) == 0
    assert int(result["relocated_count"]) == 0
    assert jnp.array_equal(model.means[...], before)


def test_noise_mapping_compatibility_path_mutates_means():
    params = {
        "means": jnp.zeros((1, 3)),
        "quats": jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
        "scales": jnp.zeros((1, 3)),
        "opacities": jnp.asarray([-10.0]),
    }
    updated = inject_noise_to_position(
        params,
        optimizers={},
        state={},
        noise_scale=0.1,
        key=jax.random.key(0),
    )
    assert jnp.array_equal(params["means"], updated)
    assert not jnp.array_equal(updated, jnp.zeros_like(updated))


def test_training_step_executes_mcmc_commit_branch():
    strategy_config = StrategyConfig(
        kind="mcmc",
        refine_start=100,
        noise_lr=2.0,
        noise_opacity_t=0.9,
        noise_opacity_k=10.0,
        max_new_per_refine=1,
    )
    config = TrainConfig(
        model=ModelConfig(capacity=8, bucket_min_capacity=8, sh_degree=0),
        optimizer=OptimizerConfig(max_steps=10),
        strategy=strategy_config,
        data=DataConfig(root="unused", patch_size=8, batch_size=1),
        rasterizer=RasterizationConfig(
            tile_size=8,
            max_gaussians_per_tile=8,
            tile_batch_size=1,
            backend="jax",
        ),
        steps=1,
        eval_every=0,
        checkpoint_every=0,
    )
    model = GaussianModel.from_point_cloud(
        jnp.asarray([[0.0, 0.0, 3.0]], jnp.float32),
        jnp.asarray([[0.5, 0.5, 0.5]], jnp.float32),
        config.model,
    )
    optimizer = create_optimizer(model, config.optimizer)
    state = MCMCStrategy(strategy_config).initialize_state(model.capacity)
    safety = TrainingSafetyState()
    inactive_before = model.means[1:].copy()
    metrics = make_train_step(config)(
        model,
        optimizer,
        state,
        safety,
        jnp.zeros((1, 8, 8, 3), jnp.float32),
        jnp.asarray([[[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]]),
        jnp.eye(4, dtype=jnp.float32)[None],
        jax.random.key(0),
        jnp.asarray(0),
    )
    assert int(optimizer.step[...]) == 1
    assert not bool(metrics["intersection_overflow"])
    assert jnp.array_equal(model.means[1:], inactive_before)
