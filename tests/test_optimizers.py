import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_gs.config import ModelConfig, OptimizerConfig
from jax_gs.model import GaussianModel
from jax_gs.optimizers import (
    SelectiveAdam,
    create_optimizer,
    create_row_selective_optimizer,
    create_visible_adam_optimizer,
    reset_optimizer_indices,
)
from jax_gs.strategy import reset_opacities


def _seed_adam_moments(model: GaussianModel, optimizer: nnx.Optimizer) -> None:
    parameters = nnx.state(model, nnx.Param)

    def make_gradient(value: jax.Array) -> jax.Array:
        rows = jnp.arange(1, value.shape[0] + 1, dtype=value.dtype)
        rows = rows.reshape((value.shape[0],) + (1,) * (value.ndim - 1))
        return jnp.broadcast_to(rows, value.shape)

    optimizer.update(model, jax.tree.map(make_gradient, parameters))


def _snapshot(optimizer: nnx.Optimizer) -> dict[str, np.ndarray]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(
        nnx.as_pure(nnx.state(optimizer.opt_state))
    )
    return {str(path): np.asarray(value).copy() for path, value in leaves}


def _gradient_tree(model: GaussianModel, scale: float = 1.0):
    parameters = nnx.state(model, nnx.Param)

    def make_gradient(value: jax.Array) -> jax.Array:
        rows = jnp.arange(1, value.shape[0] + 1, dtype=value.dtype)
        rows = rows.reshape((value.shape[0],) + (1,) * (value.ndim - 1))
        return jnp.broadcast_to(rows * scale, value.shape)

    return jax.tree.map(make_gradient, parameters)


def _array_leaves(tree) -> list[np.ndarray]:
    return [np.asarray(value) for value in jax.tree.leaves(nnx.as_pure(tree))]


def _assert_trees_equal(actual, expected) -> None:
    actual_leaves = _array_leaves(actual)
    expected_leaves = _array_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_value, expected_value in zip(
        actual_leaves, expected_leaves, strict=True
    ):
        np.testing.assert_array_equal(actual_value, expected_value)


def _parameter_state(model: GaussianModel):
    return nnx.state(model, nnx.Param)


def test_indexed_reset_only_clears_valid_slot_rows():
    model = GaussianModel.empty(
        ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=1)
    )
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    _seed_adam_moments(model, optimizer)
    before = _snapshot(optimizer)
    step_before = int(optimizer.step[...])

    reset_optimizer_indices(
        optimizer,
        jnp.array([1, 0, 3], jnp.int32),
        jnp.array([True, False, True]),
        capacity=4,
    )

    after = _snapshot(optimizer)
    assert int(optimizer.step[...]) == step_before
    for path, previous in before.items():
        expected = previous.copy()
        if previous.ndim and previous.shape[0] == 4:
            expected[[1, 3]] = 0
        np.testing.assert_array_equal(after[path], expected)


def test_appearance_features_and_colors_use_independent_sh0_rate_transforms():
    model = GaussianModel.empty(
        ModelConfig(capacity=3, bucket_min_capacity=3),
        appearance_feature_dim=32,
    )
    model.active_mask[...] = True
    config = OptimizerConfig(
        means_lr=0.0,
        scales_lr=0.0,
        quats_lr=0.0,
        opacities_lr=0.0,
        sh0_lr=1.0e-2,
        sh_rest_lr=0.0,
        max_steps=10,
    )
    optimizer = create_optimizer(model, config)
    features_before = np.asarray(model.features[...]).copy()
    colors_before = np.asarray(model.colors[...]).copy()

    optimizer.update(
        model,
        jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param)),
    )

    assert not np.array_equal(model.features[...], features_before)
    assert not np.array_equal(model.colors[...], colors_before)
    state_paths = "\n".join(_snapshot(optimizer))
    assert "features" in state_paths
    assert "colors" in state_paths


def test_gaussian_adam_uses_current_main_global_batch_scaling():
    model = GaussianModel.empty(
        ModelConfig(capacity=1, bucket_min_capacity=1, sh_degree=0)
    )
    model.active_mask[...] = True
    config = OptimizerConfig(
        means_lr=1.0e-2,
        scales_lr=1.0e-2,
        quats_lr=1.0e-2,
        opacities_lr=1.0e-2,
        sh0_lr=1.0e-2,
        sh_rest_lr=1.0e-2,
        means_lr_final_scale=1.0,
        max_steps=10,
        eps=1.0e-6,
    )
    optimizer = create_optimizer(model, config, batch_size=2, world_size=2)
    before = [value.copy() for value in _array_leaves(_parameter_state(model))]

    optimizer.update(
        model,
        jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param)),
    )

    expected_delta = 2.0e-2 / (1.0 + 0.5e-6)
    for initial, current in zip(
        before, _array_leaves(_parameter_state(model)), strict=True
    ):
        np.testing.assert_allclose(
            current, initial - expected_delta, rtol=2e-6, atol=1e-7
        )


def test_gaussian_adam_scales_betas_and_means_schedule_across_steps():
    model = GaussianModel.empty(
        ModelConfig(capacity=1, bucket_min_capacity=1, sh_degree=0)
    )
    config = OptimizerConfig(
        means_lr=1.0e-2,
        means_lr_final_scale=0.25,
        max_steps=2,
        eps=1.0e-6,
    )
    optimizer = create_optimizer(model, config, batch_size=2, world_size=2)
    parameters = nnx.state(model, nnx.Param)

    optimizer.update(model, jax.tree.map(jnp.ones_like, parameters))
    optimizer.update(
        model,
        jax.tree.map(lambda value: jnp.full_like(value, 3.0), parameters),
    )

    beta1 = 0.6
    beta2 = 0.996
    scaled_eps = 0.5e-6
    first_moment = (1.0 - beta1) * 1.0
    first_variance = (1.0 - beta2) * 1.0
    first_delta = (
        2.0e-2
        * (first_moment / (1.0 - beta1))
        / (np.sqrt(first_variance / (1.0 - beta2)) + scaled_eps)
    )
    second_moment = beta1 * first_moment + (1.0 - beta1) * 3.0
    second_variance = beta2 * first_variance + (1.0 - beta2) * 9.0
    second_delta = (
        1.0e-2
        * (second_moment / (1.0 - beta1**2))
        / (np.sqrt(second_variance / (1.0 - beta2**2)) + scaled_eps)
    )
    np.testing.assert_allclose(
        model.means[...],
        -(first_delta + second_delta),
        rtol=2e-6,
        atol=1e-7,
    )


def test_gaussian_adam_applies_scene_scale_only_to_means_rate():
    model = GaussianModel.empty(
        ModelConfig(capacity=1, bucket_min_capacity=1, sh_degree=0)
    )
    config = OptimizerConfig(
        means_lr=1.0e-2,
        scales_lr=1.0e-2,
        means_lr_final_scale=1.0,
        eps=1.0e-6,
    )
    optimizer = create_optimizer(model, config, scene_scale=3.0)
    means_before = np.asarray(model.means[...]).copy()
    scales_before = np.asarray(model.log_scales[...]).copy()

    optimizer.update(
        model,
        jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param)),
    )

    means_update = np.asarray(model.means[...]) - means_before
    scales_update = np.asarray(model.log_scales[...]) - scales_before
    np.testing.assert_allclose(
        means_update,
        3.0 * scales_update,
        rtol=1e-6,
        atol=1e-8,
    )


def test_gaussian_adam_rejects_current_main_invalid_effective_batch_beta():
    model = GaussianModel.empty(
        ModelConfig(capacity=1, bucket_min_capacity=1, sh_degree=0)
    )

    create_optimizer(model, batch_size=5, world_size=2)
    with pytest.raises(ValueError, match="effective batch size.*10"):
        create_optimizer(model, batch_size=11, world_size=1)


def test_opacity_reset_only_clears_opacity_moments():
    model = GaussianModel.empty(
        ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=1)
    )
    model.active_mask[...] = True
    model.opacity_logits[...] = jnp.array([2.0, -20.0, 2.0, -20.0])
    optimizer = create_optimizer(model, OptimizerConfig(max_steps=10))
    _seed_adam_moments(model, optimizer)
    before = _snapshot(optimizer)

    reset_opacities(model, optimizer, maximum_opacity=0.1)

    after = _snapshot(optimizer)
    for path, previous in before.items():
        expected = previous.copy()
        if "opacity_logits" in path:
            expected[[0, 2]] = 0
        np.testing.assert_array_equal(after[path], expected)


def test_row_selective_adam_matches_adam_on_visible_rows_and_freezes_others():
    config = ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=1)
    optimizer_config = OptimizerConfig(max_steps=10)
    regular_model = GaussianModel.empty(config)
    selective_model = GaussianModel.from_state_dict(
        regular_model.state_dict(), max_capacity=regular_model.max_capacity
    )
    regular = create_optimizer(regular_model, optimizer_config)
    selective = create_row_selective_optimizer(selective_model, optimizer_config)
    regular_before = _array_leaves(_parameter_state(regular_model))
    selective_state_before = _snapshot(selective)
    visible_mask = jnp.asarray([True, False, True, False])

    regular_updates = regular.update(regular_model, _gradient_tree(regular_model))
    selective_updates = selective.update(
        selective_model,
        _gradient_tree(selective_model),
        visible_mask=visible_mask,
    )

    for regular_value, selective_value, initial_value in zip(
        _array_leaves(_parameter_state(regular_model)),
        _array_leaves(_parameter_state(selective_model)),
        regular_before,
        strict=True,
    ):
        np.testing.assert_array_equal(
            selective_value[visible_mask], regular_value[visible_mask]
        )
        np.testing.assert_array_equal(
            selective_value[~visible_mask], initial_value[~visible_mask]
        )
    for update in _array_leaves(selective_updates):
        np.testing.assert_array_equal(update[~visible_mask], 0.0)

    regular_state = _snapshot(regular)
    selective_state = _snapshot(selective)
    assert regular_state.keys() == selective_state.keys()
    for path, selective_value in selective_state.items():
        regular_value = regular_state[path]
        initial_value = selective_state_before[path]
        if selective_value.ndim and selective_value.shape[0] == 4:
            np.testing.assert_array_equal(
                selective_value[visible_mask], regular_value[visible_mask]
            )
            np.testing.assert_array_equal(
                selective_value[~visible_mask], initial_value[~visible_mask]
            )
        else:
            np.testing.assert_array_equal(selective_value, regular_value)
    for regular_update, selective_update in zip(
        _array_leaves(regular_updates),
        _array_leaves(selective_updates),
        strict=True,
    ):
        np.testing.assert_array_equal(
            selective_update[visible_mask], regular_update[visible_mask]
        )
        np.testing.assert_array_equal(selective_update[~visible_mask], 0.0)


def test_row_selective_adam_all_visible_is_exactly_stock_adam():
    model_config = ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=1)
    optimizer_config = OptimizerConfig(max_steps=10)
    regular_model = GaussianModel.empty(model_config)
    selective_model = GaussianModel.from_state_dict(
        regular_model.state_dict(), max_capacity=regular_model.max_capacity
    )
    regular = create_optimizer(regular_model, optimizer_config)
    selective = create_row_selective_optimizer(selective_model, optimizer_config)
    visible_mask = jnp.ones((4,), dtype=jnp.bool_)

    for scale in (1.0, -0.25, 0.5):
        regular_updates = regular.update(
            regular_model, _gradient_tree(regular_model, scale)
        )
        selective_updates = selective.update(
            selective_model,
            _gradient_tree(selective_model, scale),
            visible_mask=visible_mask,
        )
        _assert_trees_equal(selective_updates, regular_updates)
        _assert_trees_equal(
            _parameter_state(selective_model), _parameter_state(regular_model)
        )
        selective_state = _snapshot(selective)
        regular_state = _snapshot(regular)
        assert selective_state.keys() == regular_state.keys()
        for path, expected in regular_state.items():
            np.testing.assert_array_equal(selective_state[path], expected)
        assert int(selective.step[...]) == int(regular.step[...])


def test_row_selective_adam_preserves_hidden_history_while_count_advances():
    model = GaussianModel.empty(
        ModelConfig(capacity=3, bucket_min_capacity=3, sh_degree=1)
    )
    optimizer = create_row_selective_optimizer(model, OptimizerConfig(max_steps=10))
    optimizer.update(
        model,
        _gradient_tree(model),
        visible_mask=jnp.asarray([True, True, True]),
    )
    state_before = _snapshot(optimizer)
    parameters_before = _array_leaves(_parameter_state(model))

    updates = optimizer.update(
        model,
        _gradient_tree(model, scale=100.0),
        visible_mask=jnp.asarray([False, False, False]),
    )

    state_after = _snapshot(optimizer)
    for path, previous in state_before.items():
        current = state_after[path]
        if previous.ndim and previous.shape[0] == 3:
            np.testing.assert_array_equal(current, previous)
    for before, after, update in zip(
        parameters_before,
        _array_leaves(_parameter_state(model)),
        _array_leaves(updates),
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)
        np.testing.assert_array_equal(update, 0.0)
    assert int(optimizer.step[...]) == 2
    scalar_counts = [
        value
        for path, value in state_after.items()
        if value.ndim == 0 and "count" in path
    ]
    assert scalar_counts
    for count in scalar_counts:
        np.testing.assert_array_equal(count, 2)


def test_row_selective_adam_validates_mask_before_mutating_state():
    model = GaussianModel.empty(
        ModelConfig(capacity=3, bucket_min_capacity=3, sh_degree=0)
    )
    optimizer = create_row_selective_optimizer(model)
    gradients = _gradient_tree(model)

    with pytest.raises(ValueError, match=r"visible_mask must have shape \[N\]"):
        optimizer.update(
            model,
            gradients,
            visible_mask=jnp.ones((3, 1), dtype=jnp.bool_),
        )
    with pytest.raises(TypeError, match="visible_mask must be boolean"):
        optimizer.update(
            model,
            gradients,
            visible_mask=jnp.ones((3,), dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="does not match parameter rows"):
        optimizer.update(
            model,
            gradients,
            visible_mask=jnp.ones((2,), dtype=jnp.bool_),
        )
    assert int(optimizer.step[...]) == 0


def test_row_selective_adam_is_nnx_jittable():
    model = GaussianModel.empty(
        ModelConfig(capacity=3, bucket_min_capacity=3, sh_degree=0)
    )
    optimizer = create_row_selective_optimizer(model, OptimizerConfig(max_steps=10))
    means_before = np.asarray(model.means[...]).copy()

    @nnx.jit
    def update(current_model, current_optimizer, gradients, visible_mask):
        return current_optimizer.update(
            current_model, gradients, visible_mask=visible_mask
        )

    updates = update(
        model,
        optimizer,
        _gradient_tree(model),
        jnp.asarray([True, False, False]),
    )

    assert int(optimizer.step[...]) == 1
    assert not np.array_equal(np.asarray(model.means[0]), means_before[0])
    np.testing.assert_array_equal(np.asarray(model.means[1:]), means_before[1:])
    for value in _array_leaves(updates):
        np.testing.assert_array_equal(value[1:], 0.0)


def test_visible_adam_matches_upstream_uncorrected_first_step():
    model = GaussianModel.empty(
        ModelConfig(capacity=3, bucket_min_capacity=3, sh_degree=0)
    )
    config = OptimizerConfig(means_lr=0.1, max_steps=10)
    optimizer = create_visible_adam_optimizer(model, config)
    parameters_before = _array_leaves(_parameter_state(model))
    state_before = _snapshot(optimizer)
    visible_mask = jnp.asarray([True, False, True])
    gradients = _gradient_tree(model)

    updates = optimizer.update(
        model,
        gradients,
        visible_mask=visible_mask,
    )

    means_gradient = np.asarray(gradients["means"][...])
    expected_means_update = (
        -config.means_lr
        * (1.0 - 0.9)
        * means_gradient
        / (np.sqrt((1.0 - 0.999) * np.square(means_gradient)) + config.eps)
    )
    np.testing.assert_allclose(
        np.asarray(model.means[visible_mask]),
        expected_means_update[visible_mask],
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(np.asarray(model.means[1]), 0.0)
    assert not np.allclose(
        np.asarray(model.means[0]),
        -config.means_lr,
    )

    for before, after, update in zip(
        parameters_before,
        _array_leaves(_parameter_state(model)),
        _array_leaves(updates),
        strict=True,
    ):
        np.testing.assert_array_equal(after[~visible_mask], before[~visible_mask])
        np.testing.assert_array_equal(update[~visible_mask], 0.0)

    state_after = _snapshot(optimizer)
    for path, previous in state_before.items():
        current = state_after[path]
        if previous.ndim and previous.shape[0] == 3:
            np.testing.assert_array_equal(
                current[~np.asarray(visible_mask)],
                previous[~np.asarray(visible_mask)],
            )
    assert int(optimizer.step[...]) == 1


def test_selective_adam_intersects_explicit_visibility_with_active_mask():
    model = GaussianModel.empty(
        ModelConfig(capacity=3, bucket_min_capacity=3, sh_degree=0)
    )
    model.active_mask[...] = jnp.asarray([True, False, True])
    model.quats[1] = jnp.asarray([2.0, 0.0, 0.0, 0.0])
    model.quats[2] = jnp.asarray([3.0, 4.0, 0.0, 0.0])
    adapter = SelectiveAdam(model, OptimizerConfig(max_steps=10))
    means_before = np.asarray(model.means[...]).copy()
    quats_before = np.asarray(model.quats[...]).copy()

    updates = adapter.update(
        model,
        _gradient_tree(model),
        visible_mask=jnp.asarray([True, True, False]),
    )

    assert int(adapter.step) == 1
    assert not np.array_equal(np.asarray(model.means[0]), means_before[0])
    np.testing.assert_array_equal(np.asarray(model.means[1:]), means_before[1:])
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(model.quats[0])), 1.0, rtol=1.0e-6
    )
    np.testing.assert_array_equal(np.asarray(model.quats[1:]), quats_before[1:])
    for value in _array_leaves(updates):
        np.testing.assert_array_equal(value[1:], 0.0)
