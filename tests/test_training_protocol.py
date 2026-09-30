import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig, create_gaussians, seed_gaussians
from jaxgs.training.densify import compute_densification_scores, decay_opacity, densify_step
from jaxgs.training.optimizer import create_adam_state, sparse_adam_update


def test_fragment_score_and_append_only_split():
    config = CapacityConfig(4, 1, 4, 4, 0)
    pool = seed_gaussians(
        create_gaussians(config),
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
    score, prune = compute_densification_scores(pool, stats)
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
    pool = seed_gaussians(
        create_gaussians(CapacityConfig(2, 1, 2, 4, 0)),
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


@pytest.mark.parametrize("pattern", ["split", "clone", "mixed"])
@pytest.mark.parametrize(
    "capacity,active_count,target_count,cluster_size",
    [(1, 0, 0, 1), (9, 9, 9, 2), (17, 7, 10, 1), (257, 129, 257, 128)],
)
def test_densification_preserves_sampled_parent_order(
    pattern, capacity, active_count, target_count, cluster_size
):
    config = CapacityConfig(capacity, cluster_size, 4, 4, 0)
    pool = seed_gaussians(
        create_gaussians(config),
        jnp.zeros((active_count, 3)),
        jnp.full((active_count, 3), 0.5),
    )
    indices = np.arange(capacity)
    split = (indices % 2 == 0) if pattern == "mixed" else np.full(capacity, pattern == "split")
    pool = pool.replace(
        log_scale=jnp.asarray(np.repeat(np.log(np.where(split, 0.2, 0.001))[:, None], 3, axis=1)),
        sh=pool.sh.at[:, 0, 0].set(jnp.arange(capacity)),
    )
    stats = jnp.tile(jnp.array([3.0, 1.0, 2.0, 4.0]), (capacity, 1))
    key = jax.random.key(23)
    scores, _ = compute_densification_scores(pool, stats)
    priorities = jnp.where(
        pool.alive,
        jnp.log(jnp.maximum(scores, 1e-30))
        + jax.random.gumbel(jax.random.split(key)[0], (capacity,)),
        -jnp.inf,
    )
    candidate_count = min(max(target_count - active_count, 1), active_count)
    sampled = np.argsort(-np.asarray(priorities), kind="stable")[:candidate_count]
    parents = np.concatenate((sampled[split[sampled]], sampled[~split[sampled]]))
    birth_count = min(
        candidate_count // cluster_size * cluster_size,
        (capacity - active_count) // cluster_size * cluster_size,
    )
    state = create_adam_state(pool)
    state = state.replace(
        m=jax.tree.map(jnp.ones_like, state.m), v=jax.tree.map(jnp.ones_like, state.v)
    )
    updated, next_state, born, pruned = densify_step(
        pool,
        state,
        stats,
        key,
        jnp.array(target_count),
        jnp.array(1.0),
        cluster_size=cluster_size,
        allocator="jax",
    )
    assert int(born) == birth_count and int(pruned) == 0
    child_slots = slice(active_count, active_count + birth_count)
    np.testing.assert_array_equal(updated.sh[child_slots, 0, 0], parents[:birth_count])
    np.testing.assert_allclose(
        updated.log_scale[child_slots],
        np.asarray(pool.log_scale)[parents[:birth_count]]
        - np.where(split[parents[:birth_count], None], np.log(1.6), 0),
        atol=2e-7,
    )
    np.testing.assert_array_equal(updated.xyz[:active_count], pool.xyz[:active_count])
    for values in jax.tree.leaves((next_state.m, next_state.v, next_state.step)):
        np.testing.assert_array_equal(values[child_slots], 0)
    assert updated.xyz.shape == pool.xyz.shape
    np.testing.assert_array_equal(updated.free_mask, ~updated.alive)


@pytest.mark.parametrize("partitionable", [True, False])
def test_densification_preserves_normal_random_prefix_and_compilation(partitionable):
    config = CapacityConfig(257, 1, 4, 4, 0)
    pool = seed_gaussians(
        create_gaussians(config), jnp.zeros((129, 3)), jnp.full((129, 3), 0.5), scale=0.2
    )
    state = create_adam_state(pool)
    stats = jnp.tile(jnp.array([3.0, 1.0, 2.0, 4.0]), (257, 1))
    key = jax.random.key(37)
    with jax.threefry_partitionable(partitionable):
        expected = jax.random.normal(jax.random.split(key)[1], (257, 3)) * 0.2
        cache_sizes = []
        for births in (1, 16, 64, 128):
            updated, _, born, _ = densify_step(
                pool,
                state,
                stats,
                key,
                jnp.array(129 + births),
                jnp.array(1.0),
                cluster_size=1,
                allocator="jax",
            )
            assert int(born) == births
            np.testing.assert_allclose(
                updated.xyz[129 : 129 + births], expected[:births], atol=1e-7
            )
            cache_sizes.append(densify_step._cache_size())
        assert len(set(cache_sizes)) == 1


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe partition requires JAX CUDA",
)
@pytest.mark.parametrize("capacity", [1, 257, 1025])
@pytest.mark.parametrize("pattern", ["split", "clone", "mixed"])
def test_cute_partition_preserves_stable_groups_and_unused_tail(capacity, pattern):
    from jaxgs.kernels.partitioner import stable_split_partition_cute

    indices = np.arange(capacity)
    split = (indices % 3 == 0) if pattern == "mixed" else np.full(capacity, pattern == "split")
    partition = jax.jit(lambda mask, count: stable_split_partition_cute(mask, count))
    for candidate_count in sorted({0, min(31, capacity), min(256, capacity), capacity}):
        selected = indices[:candidate_count]
        expected = np.concatenate(
            (
                selected[split[:candidate_count]],
                selected[~split[:candidate_count]],
                indices[candidate_count:],
            )
        )
        actual = partition(jnp.asarray(split), jnp.array(candidate_count))
        np.testing.assert_array_equal(actual, expected)
        assert partition._cache_size() == 1


@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe is not installed")
def test_cute_partition_requires_cuda(monkeypatch):
    from jaxgs.kernels.partitioner import stable_split_partition_cute

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="CuTe partition requires"):
        stable_split_partition_cute(jnp.array([True]), jnp.array(1))


def test_adam_rates_mask_and_opacity_decay():
    config = CapacityConfig(2, 1, 2, 4, 1)
    pool = seed_gaussians(create_gaussians(config), jnp.ones((2, 3)), jnp.ones((2, 3)))
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
    pool = seed_gaussians(
        create_gaussians(config),
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
    valid = (exponent <= 0) & (projected.alpha[0] * jnp.exp(exponent) >= 1 / 256)
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
