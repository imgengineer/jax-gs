import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig, create_pool, seed_pool
from jaxgs.training.densify import decay_opacity, densify_step, fragment_scores
from jaxgs.training.optimizer import create_adam_state, sparse_adam_update


def test_fragment_score_and_append_only_split():
    config = CapacityConfig(4, 1, 4, 4, 0)
    pool = seed_pool(
        create_pool(config),
        jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        jnp.ones((2, 3)),
        scale=0.2,
        opacity=0.5,
    )
    state = create_adam_state(pool)
    state = state.replace(m=jax.tree.map(lambda x: jnp.ones_like(x), state.m))
    # Only the first parent has nonzero variance; the second has zero weight
    # and must be pruned.
    stats = jnp.array(
        [[3.0, 1.0, 2.0, 4.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    score, prune = fragment_scores(pool, stats)
    np.testing.assert_allclose(score[0], (4 / 4 - (2 / 4) ** 2) * 3 * 0.5**2)
    np.testing.assert_array_equal(prune, [False, True, False, False])
    # LiteGS's minimum growth budget is one, plus replacement of pruned points.
    next_pool, next_state, born, removed = densify_step(
        pool,
        state,
        stats,
        jax.random.key(1),
        jnp.array(1),
        jnp.array(1.0),
        cluster_size=1,
        allocator="jax",
    )
    assert int(born) == 2 and int(removed) == 1
    np.testing.assert_array_equal(next_pool.xyz[0], pool.xyz[0])
    np.testing.assert_array_equal(next_pool.log_scale[0], pool.log_scale[0])
    np.testing.assert_allclose(next_pool.log_scale[1], pool.log_scale[0] - np.log(1.6))
    for field in ("xyz", "log_scale", "rotation", "opacity", "sh"):
        np.testing.assert_array_equal(getattr(next_state.m, field)[1:3], 0)
    assert next_pool.xyz.shape == pool.xyz.shape
    assert bool(jnp.all(next_pool.free_mask == ~next_pool.alive))


def test_percent_dense_controls_clone_versus_split():
    pool = seed_pool(
        create_pool(CapacityConfig(2, 1, 2, 4, 0)),
        jnp.array([[1.0, 2.0, 3.0]]),
        jnp.ones((1, 3)),
        scale=0.2,
    )
    stats = jnp.array([[3.0, 1.0, 2.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
    for percent_dense, split in ((0.01, True), (1.0, False)):
        updated, _, born, _ = densify_step(
            pool,
            create_adam_state(pool),
            stats,
            jax.random.key(5),
            jnp.array(2),
            jnp.array(1.0),
            cluster_size=1,
            allocator="jax",
            percent_dense=percent_dense,
        )
        assert int(born) == 1
        np.testing.assert_allclose(
            updated.log_scale[1], pool.log_scale[0] - (np.log(1.6) if split else 0)
        )
        if not split:
            np.testing.assert_array_equal(updated.xyz[1], pool.xyz[0])
        else:
            assert not np.array_equal(updated.xyz[1], pool.xyz[0])


def test_adam_rates_mask_and_opacity_decay():
    config = CapacityConfig(2, 1, 2, 4, 1)
    pool = seed_pool(create_pool(config), jnp.ones((2, 3)), jnp.ones((2, 3)))
    state = create_adam_state(pool)
    fields = ("xyz", "log_scale", "rotation", "opacity", "sh")
    gradients = tuple(jnp.ones_like(getattr(pool, name)) for name in fields)
    updated, state = sparse_adam_update(
        pool, state, gradients, jnp.array([True, False]), jnp.array(0), jnp.array(2.0)
    )
    rates = [0.00032, 0.005, 0.001, 0.025]
    for field, rate in zip(fields[:4], rates, strict=True):
        np.testing.assert_allclose(
            getattr(updated, field)[0],
            getattr(pool, field)[0] - rate * 0.1 / np.sqrt(0.001),
            atol=1e-6,
        )
        np.testing.assert_array_equal(getattr(updated, field)[1], getattr(pool, field)[1])
    np.testing.assert_allclose(updated.sh[0, 0], pool.sh[0, 0] - 0.0025 * np.sqrt(10), atol=1e-6)
    np.testing.assert_allclose(updated.sh[0, 1], pool.sh[0, 1] - 0.00025 * np.sqrt(10), atol=1e-6)
    decayed, cleared = decay_opacity(updated, state)
    np.testing.assert_allclose(
        jax.nn.sigmoid(decayed.opacity),
        jnp.maximum(jax.nn.sigmoid(updated.opacity) * 0.5, 1 / 128),
        atol=1e-7,
    )
    for field in fields:
        np.testing.assert_array_equal(getattr(cleared.m, field), 0)
        np.testing.assert_array_equal(getattr(cleared.v, field), 0)


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe protocol requires JAX CUDA",
)
def test_fragment_statistics_and_full_step():
    from jaxgs.kernels.projector import project_cute_vjp
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
    from jaxgs.kernels.sorted_rasterizer import rasterize_loss_and_grad, rasterize_sorted_cute_vjp
    from jaxgs.reference.loss import photometric_loss
    from jaxgs.scene.cluster import world_cluster_bounds
    from jaxgs.training.trainer import array_train_step

    config = CapacityConfig(2, 1, 2, 4, 1, 16)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 8, 8, 4, 4, 8, 8)
    pool = seed_pool(
        create_pool(config),
        jnp.array([[0.1, 0.0, 2.0]]),
        jnp.array([[0.4, 0.3, 0.2]]),
        scale=0.2,
        opacity=0.5,
    )
    projected = project_cute_vjp(pool, camera, config, 0)
    table = build_sorted_visibility_table_cute(projected, camera, config)
    target = jnp.zeros((8, 8, 3))
    loss, grads, stats = rasterize_loss_and_grad(projected, table, camera, config, target, True)
    expected_loss, expected_grad = jax.value_and_grad(
        lambda p: photometric_loss(rasterize_sorted_cute_vjp(p, table, camera, config).rgb, target),
        allow_int=True,
    )(projected)
    np.testing.assert_allclose(loss, expected_loss, atol=1e-6)
    np.testing.assert_allclose(grads.alpha, expected_grad.alpha, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(stats[:, 2], expected_grad.alpha, rtol=1e-4, atol=1e-5)
    image = rasterize_sorted_cute_vjp(projected, table, camera, config).rgb
    image_grad = jax.grad(photometric_loss)(image, target)
    yy, xx = jnp.meshgrid(jnp.arange(8), jnp.arange(8), indexing="ij")
    delta = jnp.stack((xx + 0.5, yy + 0.5), axis=-1) - projected.mean[0]
    exponent = -0.5 * jnp.einsum(
        "hwi,ij,hwj->hw", delta, projected.conic[0], delta, precision=jax.lax.Precision.HIGHEST
    )
    valid = (exponent >= -4.5) & (projected.alpha[0] * jnp.exp(exponent) >= 1 / 256)
    errors = jnp.where(
        valid, jnp.sum(image_grad * projected.color[0], axis=-1) * jnp.exp(exponent), 0
    )
    np.testing.assert_allclose(stats[0, 3], jnp.sum(errors**2), rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(stats[0, 0], jnp.sum(valid))
    assert float(stats[0, 0]) > 0 and float(stats[0, 1]) > 0 and float(stats[0, 3]) > 0
    np.testing.assert_array_equal(stats[1], 0)
    state = create_adam_state(pool)
    bounds = world_cluster_bounds(pool, config.cluster_size)
    next_pool, _, accumulated, metrics = array_train_step(
        pool,
        state,
        jnp.zeros((2, 4)),
        bounds,
        camera,
        target.astype(jnp.uint8),
        jnp.array(0),
        jnp.array(1.0),
        config,
        0,
        True,
    )
    assert not bool(metrics["overflow"])
    np.testing.assert_allclose(accumulated, stats, rtol=1e-4, atol=1e-5)
    assert bool(jnp.all(jnp.isfinite(next_pool.xyz)))
    np.testing.assert_array_equal(next_pool.sh[:, 1:], pool.sh[:, 1:])
