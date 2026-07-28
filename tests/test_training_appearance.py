from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jax_gs.config import DataConfig, TrainConfig
from jax_gs.model import sh_to_rgb
from jax_gs.training.appearance import (
    AppearanceOptModule,
    bake_appearance_sh,
    create_appearance_optimizer,
)


def _inputs():
    features = jnp.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=jnp.float32)
    embed_ids = jnp.asarray([2, 0], dtype=jnp.int32)
    dirs = jnp.asarray(
        [
            [[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            [[0.0, 0.0, 4.0], [0.0, 0.0, 5.0]],
        ],
        dtype=jnp.float32,
    )
    return features, embed_ids, dirs


def _probe_module() -> AppearanceOptModule:
    module = AppearanceOptModule(
        n=3,
        feature_dim=1,
        embed_dim=1,
        sh_degree=1,
        mlp_width=6,
        mlp_depth=1,
        rngs=nnx.Rngs(7),
    )
    module.embeds.embedding[...] = jnp.asarray([[1.0], [2.0], [3.0]])
    module.color_head[0].kernel[...] = jnp.eye(6, dtype=jnp.float32)
    module.color_head[0].bias[...] = 0.0
    output_kernel = jnp.zeros((6, 3), dtype=jnp.float32)
    output_kernel = output_kernel.at[0, 0].set(1.0)
    output_kernel = output_kernel.at[1, 1].set(1.0)
    output_kernel = output_kernel.at[4, 2].set(1.0)
    module.color_head[-1].kernel[...] = output_kernel
    module.color_head[-1].bias[...] = 0.0
    return module


def test_initialization_matches_trainer_zero_color_correction():
    module = AppearanceOptModule(
        n=5,
        feature_dim=4,
        embed_dim=3,
        sh_degree=2,
        mlp_width=7,
        mlp_depth=3,
        rngs=nnx.Rngs(11),
    )
    same_seed = AppearanceOptModule(
        n=5,
        feature_dim=4,
        embed_dim=3,
        sh_degree=2,
        mlp_width=7,
        mlp_depth=3,
        rngs=nnx.Rngs(11),
    )

    assert module.embeds.embedding.shape == (5, 3)
    assert len(module.color_head) == 7
    assert module.color_head[0].kernel.shape == (16, 7)
    assert module.color_head[2].kernel.shape == (7, 7)
    assert module.color_head[4].kernel.shape == (7, 7)
    assert module.color_head[-1].kernel.shape == (7, 3)
    np.testing.assert_array_equal(module.color_head[-1].kernel[...], 0.0)
    np.testing.assert_array_equal(module.color_head[-1].bias[...], 0.0)
    np.testing.assert_array_equal(
        module.embeds.embedding[...], same_seed.embeds.embedding[...]
    )
    np.testing.assert_array_equal(
        module.color_head[0].kernel[...], same_seed.color_head[0].kernel[...]
    )

    features = jnp.ones((2, 4), dtype=jnp.float32)
    dirs = jnp.ones((3, 2, 3), dtype=jnp.float32)
    corrections = module(features, jnp.asarray([0, 1, 2]), dirs, 2)
    np.testing.assert_array_equal(corrections, jnp.zeros((3, 2, 3)))

    base_color_logits = jnp.asarray([[0.2, -0.4, 0.7], [-0.3, 0.1, 0.5]])
    colors = jax.nn.sigmoid(corrections + base_color_logits[None, :, :])
    expected = jnp.broadcast_to(jax.nn.sigmoid(base_color_logits), colors.shape)
    np.testing.assert_array_equal(colors, expected)


def test_linear_initialization_is_symmetric_pytorch_uniform():
    module = AppearanceOptModule(
        n=5,
        feature_dim=4,
        embed_dim=3,
        sh_degree=2,
        mlp_width=64,
        mlp_depth=1,
        rngs=nnx.Rngs(23),
    )
    input_dim = 3 + 4 + 9
    bound = 1.0 / np.sqrt(input_dim)

    for values in (
        np.asarray(module.color_head[0].kernel[...]),
        np.asarray(module.color_head[0].bias[...]),
    ):
        assert np.all(values >= -bound)
        assert np.all(values <= bound)
        assert np.any(values < 0.0)
        assert np.any(values > 0.0)


def test_embedding_features_and_sh_bases_follow_upstream_concatenation():
    module = _probe_module()
    features = jnp.asarray([[4.0], [5.0]], dtype=jnp.float32)
    embed_ids = jnp.asarray([2, 0], dtype=jnp.int32)
    dirs = jnp.asarray(
        [
            [[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            [[0.0, 0.0, 4.0], [0.0, 0.0, 5.0]],
        ],
        dtype=jnp.float32,
    )

    actual = module(features, embed_ids, dirs, 1)

    z_basis = np.float32(0.48860251190292)
    expected = np.asarray(
        [
            [[3.0, 4.0, z_basis], [3.0, 5.0, z_basis]],
            [[1.0, 4.0, z_basis], [1.0, 5.0, z_basis]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6)

    no_embeddings = module(features, None, dirs, 1)
    np.testing.assert_array_equal(no_embeddings[..., 0], 0.0)
    np.testing.assert_allclose(no_embeddings[..., 1:], expected[..., 1:])

    degree_zero = module(features, embed_ids, dirs, 0)
    np.testing.assert_array_equal(degree_zero[..., 2], 0.0)


def test_forward_is_nnx_jittable_with_dynamic_sh_degree():
    module = _probe_module()
    features = jnp.asarray([[4.0], [5.0]], dtype=jnp.float32)
    _, embed_ids, dirs = _inputs()

    compiled = nnx.jit(
        lambda current, degree: current(features, embed_ids, dirs, degree)
    )
    for degree in (jnp.asarray(0), jnp.asarray(1)):
        np.testing.assert_array_equal(
            compiled(module, degree), module.forward(features, embed_ids, dirs, degree)
        )


def test_view_direction_normalization_matches_upstream_epsilon():
    module = _probe_module()
    features = jnp.asarray([[4.0], [5.0]], dtype=jnp.float32)
    embed_ids = jnp.asarray([2], dtype=jnp.int32)
    dirs = jnp.asarray(
        [[[0.0, 0.0, 1.0e-10], [0.0, 0.0, 0.0]]], dtype=jnp.float32
    )

    corrections = module(features, embed_ids, dirs, 1)

    np.testing.assert_allclose(
        corrections[0, :, 2],
        np.asarray([0.48860251190292, 0.0], dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_gradients_follow_zero_output_initialization_then_reach_all_layers():
    module = AppearanceOptModule(
        n=3,
        feature_dim=2,
        embed_dim=2,
        sh_degree=1,
        mlp_width=8,
        mlp_depth=1,
        rngs=nnx.Rngs(13),
    )
    features, embed_ids, dirs = _inputs()
    target = jnp.full((2, 2, 3), 0.5, dtype=jnp.float32)

    def loss(current):
        return jnp.mean((current(features, embed_ids, dirs, 1) - target) ** 2)

    initial_gradients = nnx.grad(loss)(module)
    np.testing.assert_array_equal(initial_gradients.embeds.embedding[...], 0.0)
    np.testing.assert_array_equal(initial_gradients.color_head[0].kernel[...], 0.0)
    output_index = len(module.color_head) - 1
    assert jnp.any(
        initial_gradients.color_head[output_index].kernel[...] != 0.0
    )
    assert jnp.any(
        initial_gradients.color_head[output_index].bias[...] != 0.0
    )

    module.embeds.embedding[...] = (
        jnp.abs(module.embeds.embedding[...]) + 0.1
    )
    module.color_head[0].kernel[...] = (
        jnp.abs(module.color_head[0].kernel[...]) + 0.1
    )
    module.color_head[0].bias[...] = 0.1
    module.color_head[-1].kernel[...] = 0.1
    gradients = nnx.grad(loss)(module)
    assert jnp.any(gradients.embeds.embedding[...] != 0.0)
    assert jnp.any(gradients.color_head[0].kernel[...] != 0.0)
    assert jnp.any(gradients.color_head[output_index].kernel[...] != 0.0)


def test_nnx_optimizer_trains_embeddings_and_color_head_under_jit():
    module = AppearanceOptModule(
        n=3,
        feature_dim=2,
        embed_dim=2,
        sh_degree=1,
        mlp_width=8,
        mlp_depth=1,
        rngs=nnx.Rngs(17),
    )
    module.embeds.embedding[...] = (
        jnp.abs(module.embeds.embedding[...]) + 0.1
    )
    module.color_head[0].kernel[...] = (
        jnp.abs(module.color_head[0].kernel[...]) + 0.05
    )
    module.color_head[0].bias[...] = 0.1
    features, embed_ids, dirs = _inputs()
    target = jnp.asarray(
        [
            [[0.8, -0.4, 0.3], [0.2, 0.5, -0.7]],
            [[-0.6, 0.1, 0.7], [0.4, -0.8, 0.2]],
        ],
        dtype=jnp.float32,
    )
    embedding_before = np.asarray(module.embeds.embedding[...]).copy()
    head_before = np.asarray(module.color_head[-1].kernel[...]).copy()
    optimizer = nnx.Optimizer(module, optax.adam(2.0e-2), wrt=nnx.Param)

    def objective(current):
        prediction = current(features, embed_ids, dirs, 1)
        return jnp.mean((prediction - target) ** 2)

    @nnx.jit
    def train_step(current, current_optimizer):
        loss, gradients = nnx.value_and_grad(objective)(current)
        current_optimizer.update(current, gradients)
        return loss

    initial_loss = float(objective(module))
    for _ in range(30):
        train_step(module, optimizer)
    final_loss = float(objective(module))

    assert final_loss < initial_loss * 0.6
    assert not np.array_equal(module.embeds.embedding[...], embedding_before)
    assert not np.array_equal(module.color_head[-1].kernel[...], head_before)
    assert int(optimizer.step[...]) == 30


def test_zero_embedding_dimension_and_input_shape_contracts():
    module = AppearanceOptModule(
        n=2,
        feature_dim=2,
        embed_dim=0,
        sh_degree=0,
        mlp_width=4,
        rngs=nnx.Rngs(19),
    )
    features = jnp.ones((3, 2), dtype=jnp.float32)
    dirs = jnp.ones((2, 3, 3), dtype=jnp.float32)
    assert module(features, jnp.asarray([0, 1]), dirs, 0).shape == (2, 3, 3)

    with pytest.raises(ValueError, match="features must have shape"):
        module(features[:, :1], jnp.asarray([0, 1]), dirs, 0)
    with pytest.raises(ValueError, match="Gaussian counts"):
        module(features[:2], jnp.asarray([0, 1]), dirs, 0)
    with pytest.raises(ValueError, match="embed_ids must have shape"):
        module(features, jnp.asarray([0]), dirs, 0)
    with pytest.raises(ValueError, match="sh_degree"):
        module(features, jnp.asarray([0, 1]), dirs, -1)
    with pytest.raises(ValueError, match="sh_degree"):
        module(features, jnp.asarray([0, 1]), dirs, 1)


def test_appearance_optimizer_applies_decay_only_to_embeddings():
    module = AppearanceOptModule(
        n=3,
        feature_dim=2,
        embed_dim=2,
        sh_degree=1,
        mlp_width=4,
        mlp_depth=1,
        rngs=nnx.Rngs(23),
    )
    module.embeds.embedding[...] = 1.0
    module.color_head[0].kernel[...] = 1.0
    config = TrainConfig(
        app_opt=True,
        app_opt_lr=1.0e-2,
        app_opt_reg=0.1,
        data=DataConfig(batch_size=4),
    )
    optimizer = create_appearance_optimizer(module, config)
    embedding_before = np.asarray(module.embeds.embedding[...]).copy()
    head_before = np.asarray(module.color_head[0].kernel[...]).copy()
    zero_gradients = jax.tree.map(
        jnp.zeros_like, nnx.state(module, nnx.Param)
    )

    optimizer.update(module, zero_gradients)

    assert not np.array_equal(module.embeds.embedding[...], embedding_before)
    np.testing.assert_array_equal(module.color_head[0].kernel[...], head_before)
    assert int(optimizer.step[...]) == 1


def test_canonical_appearance_bake_uses_zero_embedding_and_direction():
    module = _probe_module()
    features = jnp.asarray([[4.0], [5.0]], dtype=jnp.float32)
    color_logits = jnp.asarray(
        [[0.2, -0.3, 0.4], [-0.5, 0.6, -0.7]], dtype=jnp.float32
    )

    sh0, sh_rest = bake_appearance_sh(
        module, features, color_logits, sh_degree=1
    )

    expected_rgb = jax.nn.sigmoid(
        module(
            features,
            None,
            jnp.zeros((1, 2, 3), dtype=jnp.float32),
            1,
        )[0]
        + color_logits
    )
    assert sh0.shape == (2, 1, 3)
    assert sh_rest.shape == (2, 0, 3)
    np.testing.assert_allclose(sh_to_rgb(sh0[:, 0]), expected_rgb, rtol=1.0e-6)
