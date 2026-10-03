import importlib.util
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig, create_gaussians, seed_gaussians

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="Packed rasterizer requires JAX CUDA and CuTe",
)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("cluster_size", [65, 128])
def test_active_sh_prefix_pullback_matches_full_coefficients(degree, empty, cluster_size):
    from jaxgs.kernels.cluster_compact import compact_visible_clusters
    from jaxgs.kernels.projector import project_with_compact_pullback

    capacity = 257
    config = CapacityConfig(capacity, cluster_size, 16, 16, 3, 4096, tile_height=8)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 9, 8, 19, 17)
    visible = (
        jnp.zeros(capacity, jnp.bool_) if empty else (jnp.arange(capacity) // cluster_size) % 2 == 0
    )
    pool = seed_gaussians(
        create_gaussians(config),
        jnp.tile(jnp.array([[0.1, -0.2, 2.0]]), (capacity, 1)),
        jnp.full((capacity, 3), 0.4),
        scale=0.1,
    ).replace(alive=visible)
    pool = pool.replace(sh=jax.random.normal(jax.random.key(5), pool.sh.shape) * 0.2)
    clusters = compact_visible_clusters(visible, cluster_size)
    projected, full_pullback = project_with_compact_pullback(
        pool, camera, config, degree, clusters, rgb_only=True
    )
    cotangents = projected.replace(
        mean=jnp.ones_like(projected.mean),
        conic=jnp.ones_like(projected.conic) * 0.01,
        color=jnp.ones_like(projected.color),
        alpha=jnp.ones_like(projected.alpha),
        depth=jnp.zeros_like(projected.depth),
        radius=jnp.zeros_like(projected.radius),
    )
    _, narrow_pullback = project_with_compact_pullback(
        pool,
        camera,
        config,
        degree,
        clusters,
        rgb_only=True,
        active_sh_only=True,
    )
    actual, expected = narrow_pullback(cotangents), full_pullback(cotangents)
    valid_count = int(jnp.sum(visible))
    sh_dim = (degree + 1) ** 2
    assert actual[-1].shape == (capacity, sh_dim, 3)
    for index, (result, reference) in enumerate(zip(actual, expected, strict=True)):
        if index == 4:
            reference = reference[:, :sh_dim]
        np.testing.assert_allclose(
            result[:valid_count], reference[:valid_count], rtol=2e-5, atol=2e-6
        )
    _, color_pullback = project_with_compact_pullback(
        pool, camera, config, degree, clusters, rgb_only=True, sh_color_only=True
    )
    color_gradients = color_pullback(cotangents)
    assert color_gradients[-1].shape == (capacity, 1, 3)
    for index, (result, reference) in enumerate(zip(color_gradients, expected, strict=True)):
        if index == 4:
            # DC reconstructs the masked color cotangent with one constant.
            result = result * 0.28209479177387814
            reference = reference[:, :1]
        np.testing.assert_allclose(
            result[:valid_count], reference[:valid_count], rtol=2e-5, atol=2e-6
        )


@pytest.mark.parametrize("degree", [0, 3])
@pytest.mark.parametrize("collect_stats", [False, True])
def test_symmetric_conic_accumulation_preserves_parameter_pullback(degree, collect_stats):
    from jaxgs.kernels.cluster_compact import compact_visible_clusters
    from jaxgs.kernels.packed_rasterizer import packed_backward, packed_forward
    from jaxgs.kernels.projector import project_with_compact_pullback
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute

    config = CapacityConfig(17, 2, 16, 16, 3, 1024, tile_height=8)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 9, 8, 19, 17)
    pool = seed_gaussians(
        create_gaussians(config),
        jnp.stack(
            (jnp.linspace(-0.3, 0.3, 17), jnp.linspace(0.1, -0.1, 17), jnp.linspace(2.0, 3.0, 17)),
            axis=1,
        ),
        jnp.full((17, 3), 0.4),
        scale=jnp.tile(jnp.array([0.1, 0.2, 0.3]), (17, 1)),
        opacity=0.3,
    )
    pool = pool.replace(
        sh=jax.random.normal(jax.random.key(5), pool.sh.shape) * 0.2,
        rotation=jnp.tile(jnp.array([0.9, 0.1, 0.2, 0.05]), (17, 1)),
    )
    clusters = compact_visible_clusters(pool.alive, config.cluster_size)
    projected, pullback = project_with_compact_pullback(
        pool, camera, config, degree, clusters, rgb_only=True
    )
    table = build_sorted_visibility_table_cute(projected, camera, config)
    image, cache, _ = packed_forward(projected, table, camera, config, collect_stats)
    image_grad = jax.random.normal(jax.random.key(11), image.shape) * 0.1
    matrix, matrix_stats = packed_backward(
        projected, table, cache, image_grad, camera, config, collect_stats
    )
    symmetric, symmetric_stats = packed_backward(
        projected,
        table,
        cache,
        image_grad,
        camera,
        config,
        collect_stats,
        symmetric_conic=True,
    )
    np.testing.assert_array_equal(symmetric.conic[:, 1, 0], 0)
    np.testing.assert_allclose(
        symmetric.conic[:, 0, 1],
        matrix.conic[:, 0, 1] + matrix.conic[:, 1, 0],
        rtol=2e-5,
        atol=2e-6,
    )
    for field in ("mean", "depth", "color", "alpha"):
        np.testing.assert_allclose(
            getattr(symmetric, field), getattr(matrix, field), rtol=2e-5, atol=2e-6
        )
    if collect_stats:
        np.testing.assert_allclose(symmetric_stats, matrix_stats, rtol=2e-5, atol=2e-6)
    for actual, expected in zip(pullback(symmetric), pullback(matrix), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("collect_stats", [False, True])
def test_cluster_backward_defines_the_cotangents_training_reads(collect_stats):
    from jaxgs.kernels.cluster_compact import compact_visible_clusters
    from jaxgs.kernels.packed_rasterizer import packed_backward, packed_forward
    from jaxgs.kernels.projector import project_with_compact_pullback
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute

    config = CapacityConfig(17, 2, 16, 16, 3, 1024, tile_height=8)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 9, 8, 19, 17)
    pool = seed_gaussians(
        create_gaussians(config),
        jnp.stack(
            (jnp.linspace(-0.3, 0.3, 17), jnp.linspace(0.1, -0.1, 17), jnp.linspace(2.0, 3.0, 17)),
            axis=1,
        ),
        jnp.full((17, 3), 0.4),
        scale=0.15,
        opacity=0.3,
    )
    visible = jnp.arange(17) // 2 % 3 != 1
    clusters = compact_visible_clusters(visible, config.cluster_size)
    projected, _ = project_with_compact_pullback(
        pool.replace(alive=pool.alive & visible), camera, config, 3, clusters, rgb_only=True
    )
    table = build_sorted_visibility_table_cute(projected, camera, config)
    image, cache, _ = packed_forward(projected, table, camera, config, collect_stats)
    image_grad = jax.random.normal(jax.random.key(3), image.shape) * 0.1
    args = (projected, table, cache, image_grad, camera, config, collect_stats)
    full, full_stats = packed_backward(*args, symmetric_conic=True)
    restricted, restricted_stats = packed_backward(
        *args, symmetric_conic=True, visible_clusters=clusters
    )
    assert np.any(np.asarray(full.alpha)[np.asarray(visible)] != 0)
    for field in ("mean", "conic", "color", "alpha"):
        np.testing.assert_allclose(
            np.asarray(getattr(restricted, field))[np.asarray(visible)],
            np.asarray(getattr(full, field))[np.asarray(visible)],
            rtol=1e-6,
            atol=1e-9,
        )
    if collect_stats:
        np.testing.assert_allclose(restricted.alpha, full.alpha, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(restricted_stats, full_stats, rtol=1e-6, atol=1e-9)


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
    pool = seed_gaussians(
        create_gaussians(config),
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
    from jaxgs.render.types import ProjectedGaussians, SortedVisibilityTable

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
    pool = seed_gaussians(create_gaussians(config), jnp.ones((9, 3)) * 2.0, jnp.ones((9, 3)) * 0.4)
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
    from jaxgs.kernels.projector import project_cute_vjp, project_with_compact_pullback
    from jaxgs.training.optimizer import create_adam_state, optax_update, sparse_adam_update

    config = CapacityConfig(capacity, cluster_size, 16, 16, 3, 4096, tile_height=8)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 9, 8, 19, 17)
    pool = seed_gaussians(
        create_gaussians(config), jnp.full((capacity, 3), 2.0), jnp.full((capacity, 3), 0.4)
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
    projected, pullback = project_with_compact_pullback(masked, camera, config, 3, clusters)
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
        actual, actual_state = optax_update(
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


@pytest.mark.parametrize("cluster_size", [2, 65, 128])
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_rgb_projection_pullback_matches_general_and_reference(cluster_size, degree):
    from jaxgs.kernels.cluster_compact import compact_visible_clusters
    from jaxgs.kernels.projector import project_with_compact_pullback
    from jaxgs.render.projection import project

    capacity = 257
    config = CapacityConfig(capacity, cluster_size, 16, 16, 3, 4096, tile_height=8)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 9, 8, 19, 17)
    pool = seed_gaussians(
        create_gaussians(config),
        jnp.tile(jnp.array([[0.1, -0.2, 2.0]]), (capacity, 1)),
        jnp.full((capacity, 3), 0.4),
        scale=jnp.tile(jnp.array([0.1, 0.2, 0.3]), (capacity, 1)),
        opacity=0.3,
    )
    pool = pool.replace(
        sh=jax.random.normal(jax.random.key(5), pool.sh.shape) * 0.2,
        rotation=jnp.tile(jnp.array([0.9, 0.1, 0.2, 0.05]), (capacity, 1)),
    )
    visible = (jnp.arange(capacity) // cluster_size) % 2 == 0
    clusters = compact_visible_clusters(visible, cluster_size)
    masked = pool.replace(alive=visible)
    weights = jax.random.normal(jax.random.key(9), (capacity, 3)) * visible[:, None]

    def objective(p):
        return (
            jnp.sum(p.mean * weights[:, :2])
            + jnp.sum(p.conic * weights[:, :2, None]) * 0.01
            + jnp.sum(p.color * weights)
            + jnp.sum(p.alpha * weights[:, 0])
        )

    projected, general_pullback = project_with_compact_pullback(
        masked, camera, config, degree, clusters
    )
    cotangents = jax.grad(objective, allow_int=True)(projected)
    general = general_pullback(cotangents)

    @jax.jit
    def rgb_pullback(p):
        _, pullback = project_with_compact_pullback(
            p, camera, config, degree, clusters, rgb_only=True
        )
        return pullback(cotangents)

    actual = rgb_pullback(masked)
    with jax.default_matmul_precision("highest"):
        reference = jax.grad(
            lambda p: objective(project(p, camera, replace(config, sh_degree=degree))),
            allow_int=True,
        )(masked)
    slots = np.flatnonzero(np.asarray(visible))
    for rgb, full, name in zip(
        actual, general, ("xyz", "log_scale", "rotation", "opacity", "sh"), strict=True
    ):
        np.testing.assert_allclose(rgb[: len(slots)], full[: len(slots)], rtol=1e-5, atol=2e-6)
        np.testing.assert_allclose(
            rgb[: len(slots)], getattr(reference, name)[slots], rtol=2e-4, atol=2e-5
        )


@pytest.mark.parametrize("rgb_only", [False, True])
def test_empty_compact_projection_pullback_preserves_parameters(rgb_only):
    from jaxgs.kernels.cluster_compact import compact_visible_clusters
    from jaxgs.kernels.projector import project_with_compact_pullback
    from jaxgs.training.optimizer import create_adam_state, optax_update

    config = CapacityConfig(257, 65, 16, 16, 3, 4096, tile_height=8)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 9, 8, 19, 17)
    pool = create_gaussians(config)
    state = create_adam_state(pool)
    visible = jnp.zeros(257, bool)
    clusters = compact_visible_clusters(visible, config.cluster_size)

    @jax.jit
    def update(p, s):
        projected, pullback = project_with_compact_pullback(
            p, camera, config, 3, clusters, rgb_only=rgb_only
        )
        cotangents = jax.grad(lambda x: jnp.sum(x.color), allow_int=True)(projected)
        gradients = pullback(cotangents)
        return optax_update(
            p, s, gradients, visible, 0, 1.0, cluster_size=65, compact_gradients=True
        )

    actual = update(pool, state)
    for result, expected in zip(
        jax.tree.leaves(actual), jax.tree.leaves((pool, state)), strict=True
    ):
        np.testing.assert_array_equal(result, expected)
