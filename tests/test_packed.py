import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig, create_pool, seed_pool

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="Packed rasterizer requires JAX CUDA and CuTe",
)


@pytest.mark.parametrize("tile_height,tile_width", [(8, 8), (8, 16), (12, 16), (16, 16)])
def test_packed_partial_tiles_and_parameter_pullback(tile_height, tile_width):
    from jaxgs.kernels.packed_rasterizer import (
        packed_backward,
        packed_forward,
        rasterize_packed_cute_vjp,
    )
    from jaxgs.kernels.projector import project_cute_vjp
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
    from jaxgs.reference.rasterizer_jax import rasterize_jax
    from jaxgs.render.projection import project
    from jaxgs.render.visibility_table import build_visibility_table

    config = CapacityConfig(4, 1, 4, tile_width, 1, 128, tile_height=tile_height)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 9, 8, 19, 17)
    pool = seed_pool(
        create_pool(config),
        jnp.array([[0.15, 0.1, 2.0], [-0.2, -0.1, 3.0]]),
        jnp.array([[0.3, 0.6, 0.4], [0.7, 0.3, 0.2]]),
        scale=0.25,
        opacity=0.25,
    )
    projected = project_cute_vjp(pool, camera, config)
    table = build_sorted_visibility_table_cute(projected, camera, config)
    # A single reference tile containing every point avoids binning dependencies.
    reference_config = CapacityConfig(4, 1, 4, 32, 1)
    reference_table = build_visibility_table(projected, camera, reference_config)
    result, cache, _ = packed_forward(projected, table, camera, config)
    expected = rasterize_jax(projected, reference_table, camera, reference_config).rgb
    np.testing.assert_allclose(result, expected, atol=3e-4, rtol=5e-3)
    # Evaluation invokes the custom_vjp primal, while jax.grad uses its fwd rule.
    inference = jax.jit(lambda p: rasterize_packed_cute_vjp(p, table, camera, config))(projected)
    np.testing.assert_array_equal(inference, result)
    weights = jax.random.normal(jax.random.key(8), result.shape) * 0.01

    def render_loss(p, packed):
        if packed:
            projected = project_cute_vjp(p, camera, config)
            image = rasterize_packed_cute_vjp(projected, table, camera, config)
        else:
            projected = project(p, camera, reference_config)
            image = rasterize_jax(projected, reference_table, camera, reference_config).rgb
        return jnp.sum(image * weights)

    with jax.default_matmul_precision("highest"):
        actual = jax.grad(lambda p: render_loss(p, True), allow_int=True)(pool)
        expected = jax.grad(lambda p: render_loss(p, False), allow_int=True)(pool)
    for field in ("xyz", "log_scale", "rotation", "opacity", "sh"):
        a, b = np.asarray(getattr(actual, field)), np.asarray(getattr(expected, field))
        np.testing.assert_allclose(a[:2], b[:2], rtol=0.025, atol=2e-5)
        np.testing.assert_array_equal(a[2:], 0)
    zero, _ = packed_backward(projected, table, cache, jnp.zeros_like(result), camera, config)
    for field in ("mean", "conic", "color", "alpha"):
        np.testing.assert_array_equal(getattr(zero, field), 0)


def test_packed_empty_tile_and_early_termination():
    from jaxgs.kernels.packed_rasterizer import packed_backward, packed_forward
    from jaxgs.render.projection import ProjectedGaussians
    from jaxgs.render.visibility_table import SortedVisibilityTable

    config = CapacityConfig(12, 1, 12, 16, 0, 12, tile_height=8)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 8, 4, 32, 8)
    projected = ProjectedGaussians(
        jnp.full((12, 2), 4.0),
        jnp.ones(12),
        jnp.tile(jnp.eye(2)[None] * 0.0001, (12, 1, 1)),
        jnp.ones(12),
        jnp.full((12, 3), 0.5),
        jnp.full(12, 0.99),
        jnp.ones(12, bool),
    )
    table = SortedVisibilityTable(
        jnp.arange(12),
        jnp.array([0, 12, 12]),
        jnp.ones(12, jnp.int32),
        jnp.array(12),
        jnp.array(False),
    )
    image, cache, stats = packed_forward(projected, table, camera, config, True)
    np.testing.assert_array_equal(image[:, 16:], 0)
    np.testing.assert_allclose(image[:, :16], 0.5, atol=1e-3)
    assert np.max(np.asarray(cache[2]).reshape(8, 32)[:, :16]) < 12
    grads, square = packed_backward(
        projected, table, cache, jnp.ones_like(image), camera, config, True
    )
    np.testing.assert_array_equal(grads.color[3:], 0)
    np.testing.assert_array_equal(stats.reshape(12, 2)[3:], 0)
    assert np.isfinite(np.asarray(square)).all()


def test_compacted_projection_and_sparse_adam_preserve_invisible_slots():
    from jaxgs.kernels.cluster_compact import compact_visible_clusters
    from jaxgs.kernels.projector import project_cute_vjp
    from jaxgs.training.optimizer import create_adam_state, sparse_adam_update

    config = CapacityConfig(10, 2, 16, 16, 2, 128, tile_height=8)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 9, 8, 19, 17)
    pool = seed_pool(create_pool(config), jnp.ones((9, 3)) * 2.0, jnp.ones((9, 3)) * 0.4)
    visible = jnp.repeat(jnp.array([False, True, False, True, True]), 2)
    clusters = compact_visible_clusters(visible, 2)
    np.testing.assert_array_equal(clusters[0][:3], [1, 3, 4])
    np.testing.assert_array_equal(clusters[1], [3])
    masked = pool.replace(alive=pool.alive & visible)
    full = project_cute_vjp(masked, camera, config)
    compact = project_cute_vjp(masked, camera, config, compacted_clusters=clusters)
    np.testing.assert_allclose(compact.mean[visible], full.mean[visible])
    np.testing.assert_array_equal(compact.visible, full.visible)
    weights = jnp.arange(10, dtype=jnp.float32)[:, None] * visible[:, None]

    def loss(p, ids):
        projected = project_cute_vjp(p, camera, config, compacted_clusters=ids)
        return jnp.sum(projected.color * weights) + jnp.sum(projected.mean * weights)

    actual = jax.grad(lambda p: loss(p, clusters), allow_int=True)(masked)
    expected = jax.grad(lambda p: loss(p, None), allow_int=True)(masked)
    fields = ("xyz", "log_scale", "rotation", "opacity", "sh")
    for name in fields:
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=1e-5, atol=1e-6
        )

    state = create_adam_state(pool)
    state = state.replace(
        m=jax.tree.map(lambda x: jnp.full_like(x, 0.2), state.m),
        v=jax.tree.map(lambda x: jnp.full_like(x, 0.1), state.v),
    )
    gradients = tuple(jnp.ones_like(getattr(pool, name)) * 0.3 for name in fields)
    expected_pool, expected_state = sparse_adam_update(pool, state, gradients, visible, 123, 2.0)
    update = jax.jit(
        lambda p, s: sparse_adam_update(
            p, s, gradients, visible, 123, 2.0, compacted_clusters=clusters, cluster_size=2
        )
    )
    actual_pool, actual_state = update(pool, state)
    for name in fields:
        np.testing.assert_allclose(
            getattr(actual_pool, name), getattr(expected_pool, name), atol=1e-6
        )
        np.testing.assert_allclose(
            getattr(actual_state.m, name), getattr(expected_state.m, name), atol=1e-6
        )
        np.testing.assert_allclose(
            getattr(actual_state.v, name), getattr(expected_state.v, name), atol=1e-6
        )
        np.testing.assert_array_equal(
            getattr(actual_pool, name)[~visible], getattr(pool, name)[~visible]
        )
    # Functional calls must not mutate inputs even though the FFI uses aliases.
    np.testing.assert_array_equal(state.m.xyz, np.full((10, 3), 0.2, np.float32))


@pytest.mark.parametrize("cluster_size,capacity", [(2, 9), (128, 257)])
@pytest.mark.parametrize("optimizer", ["cute", "optax"])
def test_compact_gradient_prefix_matches_dense_pullback(cluster_size, capacity, optimizer):
    from jaxgs.kernels.cluster_compact import compact_visible_clusters
    from jaxgs.kernels.projector import project_cute_sparse, project_cute_vjp
    from jaxgs.training.optimizer import create_adam_state, optax_adam_update, sparse_adam_update

    config = CapacityConfig(capacity, cluster_size, 16, 16, 3, 4096, tile_height=8)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 9, 8, 19, 17)
    pool = seed_pool(
        create_pool(config), jnp.full((capacity, 3), 2.0), jnp.full((capacity, 3), 0.4)
    )
    pool = pool.replace(sh=jax.random.normal(jax.random.key(5), pool.sh.shape) * 0.2)
    visible = (jnp.arange(capacity) // cluster_size) % 2 == 0
    clusters = compact_visible_clusters(visible, cluster_size)
    slots = np.flatnonzero(np.asarray(visible))
    weights = jax.random.normal(jax.random.key(9), (capacity, 3)) * visible[:, None]
    masked = pool.replace(alive=visible)

    def objective(p):
        return (
            jnp.sum(p.color * weights) + jnp.sum(p.mean * weights[:, :2]) + jnp.sum(p.conic) * 0.01
        )

    full = jax.grad(
        lambda p: objective(project_cute_vjp(p, camera, config, compacted_clusters=clusters)),
        allow_int=True,
    )(masked)
    projected, pullback = project_cute_sparse(masked, camera, config, 3, clusters)
    compact = pullback(jax.grad(objective, allow_int=True)(projected))
    names = ("xyz", "log_scale", "rotation", "opacity", "sh")
    for value, name in zip(compact, names, strict=True):
        np.testing.assert_allclose(
            value[: len(slots)], getattr(full, name)[slots], rtol=1e-5, atol=1e-6
        )
    # Poison the unused capacity. Sparse Adam must only read the compact prefix.
    compact = tuple(value.at[len(slots) :].set(jnp.nan) for value in compact)
    state = create_adam_state(pool)
    expected, expected_state = sparse_adam_update(
        pool, state, tuple(getattr(full, name) for name in names), visible, 0, 1.0
    )
    if optimizer == "optax":
        actual, actual_state = optax_adam_update(
            pool, state, compact, visible, 0, 1.0, cluster_size=cluster_size, compact_gradients=True
        )
    else:
        actual, actual_state = sparse_adam_update(
            pool,
            state,
            compact,
            visible,
            0,
            1.0,
            compacted_clusters=clusters,
            cluster_size=cluster_size,
            compact_gradients=True,
        )
    for name in names:
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            getattr(actual_state.m, name), getattr(expected_state.m, name), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            getattr(actual_state.v, name), getattr(expected_state.v, name), rtol=1e-5, atol=1e-6
        )
