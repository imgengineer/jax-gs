import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig, create_gaussians, estimate_initial_scales, seed_gaussians
from jaxgs.reference.rasterizer_jax import rasterize_jax
from jaxgs.render.cluster_compact import compact_clusters
from jaxgs.render.cluster_culling import build_cluster_tile_mask
from jaxgs.render.projection import project
from jaxgs.render.types import ProjectedGaussians, VisibilityTable
from jaxgs.render.visibility_table import build_visibility_table
from jaxgs.scene.spatial_refine import reorder_gaussians
from jaxgs.training.optimizer import create_adam_state, masked_adam_update
from jaxgs.training.pool_ops import prune_step
from jaxgs.training.reference_densify import densify_step, reset_opacity
from jaxgs.training.reference_trainer import train_step


def test_reference_compaction_keeps_stable_visible_prefix_under_jit():
    compact = jax.jit(compact_clusters)
    for mask, expected in [
        ([[False, False], [True, False], [False, False], [False, True]], [1, 3]),
        ([[False, False]] * 4, []),
        ([[True, False]] * 4, [0, 1, 2, 3]),
    ]:
        result = compact(jnp.array(mask))
        assert result.ids.shape == result.valid.shape == (4,)
        assert int(result.count) == len(expected)
        np.testing.assert_array_equal(np.asarray(result.ids)[np.asarray(result.valid)], expected)
    assert compact._cache_size() == 1


def test_seed_validation_preserves_fixed_capacity():
    pool = create_gaussians(CapacityConfig(2))
    with pytest.raises(ValueError, match="capacity or RGB"):
        seed_gaussians(pool, jnp.zeros((3, 3)), jnp.zeros((3, 3)))
    with pytest.raises(ValueError, match="capacity or RGB"):
        seed_gaussians(pool, jnp.zeros((2, 3)), jnp.zeros((2, 4)))
    with pytest.raises(ValueError, match="scale must be"):
        seed_gaussians(pool, jnp.zeros((2, 3)), jnp.zeros((2, 3)), scale=jnp.ones((1, 3)))
    with pytest.raises(ValueError, match="xyz must have shape"):
        estimate_initial_scales(jnp.ones((2, 4)))


@pytest.mark.parametrize("max_new", [0, 5])
def test_densification_rejects_invalid_growth_budget(max_new):
    _, _, pool = scene()
    with pytest.raises(ValueError, match="max_new must be within pool capacity"):
        densify_step(
            pool,
            create_adam_state(pool),
            jnp.ones(4),
            jax.random.key(0),
            max_new=max_new,
            threshold=0.5,
        )


def test_densification_and_training_reject_unknown_backends():
    config, camera, pool = scene()
    state = create_adam_state(pool)
    with pytest.raises(ValueError, match="unknown allocator: typo"):
        densify_step(
            pool, state, jnp.ones(4), jax.random.key(0), max_new=1, threshold=0.5, allocator="typo"
        )
    with pytest.raises(ValueError, match="unknown backend: typo"):
        train_step(pool, state, camera, jnp.zeros((8, 8, 3)), config, backend="typo")


def scene(capacity=4, k=2):
    config = CapacityConfig(
        capacity, cluster_size=2, max_gaussians_per_tile=k, tile_size=4, sh_degree=0
    )
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 8, 8, 4, 4, 8, 8)
    pool = seed_gaussians(
        create_gaussians(config),
        jnp.array([[0.1, -0.1, 2.0]], jnp.float32),
        jnp.array([[0.8, 0.2, 0.1]], jnp.float32),
        scale=0.2,
        opacity=0.5,
    )
    return config, camera, pool


def test_point_spacing_initializes_per_gaussian_scale():
    xyz = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], jnp.float32
    )
    scales = estimate_initial_scales(xyz)
    np.testing.assert_allclose(scales, [np.sqrt(14 / 3), np.sqrt(2), np.sqrt(2), np.sqrt(14 / 3)])
    pool = seed_gaussians(
        create_gaussians(CapacityConfig(4, 2, 2, 4, 0)), xyz, jnp.full((4, 3), 0.5), scale=scales
    )
    np.testing.assert_allclose(jnp.exp(pool.log_scale[:, 0]), scales)
    np.testing.assert_allclose(jnp.exp(pool.log_scale[:, 1]), scales)


def test_spatial_refine_keeps_adam_slots_attached():
    config = CapacityConfig(4, 2, 2, 4, 0)
    xyz = jnp.array([[2.0, 0.0, 2.0], [0.0, 0.0, 2.0], [3.0, 0.0, 2.0]])
    pool = seed_gaussians(create_gaussians(config), xyz, jnp.full((3, 3), 0.5))
    state = create_adam_state(pool)
    state = state.replace(step=jnp.array([1, 2, 3, 0]))
    next_pool, next_state = reorder_gaussians(pool, state)
    np.testing.assert_array_equal(next_pool.xyz[:3, 0], [0.0, 2.0, 3.0])
    np.testing.assert_array_equal(next_state.step, [2, 1, 3, 0])
    np.testing.assert_array_equal(next_pool.alive, [True, True, True, False])
    assert next_pool.xyz.shape == pool.xyz.shape
    assert int(next_pool.n_active) == 3


def test_reference_render_and_train():
    config, camera, pool = scene()
    projected = project(pool, camera, config)
    table = build_visibility_table(
        projected, camera, config, build_cluster_tile_mask(projected, camera, config)
    )
    assert table.tile_gaussian_ids.shape == (4, 2)
    assert not np.any(table.overflow)
    rendered = rasterize_jax(projected, table, camera, config)
    assert rendered.rgb.shape == (8, 8, 3)
    assert float(rendered.alpha.max()) > 0
    next_pool, state, metrics = train_step(
        pool, create_adam_state(pool), camera, jnp.zeros((8, 8, 3)), config, backend="reference"
    )
    assert float(metrics["loss"]) > 0
    assert not bool(metrics["overflow"])
    np.testing.assert_array_equal(state.step, [1, 0, 0, 0])
    np.testing.assert_array_equal(next_pool.alive, pool.alive)
    assert next_pool.xyz.shape == pool.xyz.shape


@pytest.mark.parametrize("tile_height", [8, 12])
def test_reference_rectangular_tiles_preserve_separate_rows(tile_height):
    config = CapacityConfig(2, 1, 1, 16, 0, tile_height=tile_height)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 16, 16, 8, 8, 17, 2 * tile_height + 1)
    projected = ProjectedGaussians(
        mean=jnp.array([[8.0, 4.0], [8.0, tile_height + 4.0]]),
        depth=jnp.array([1.0, 2.0]),
        conic=jnp.broadcast_to(jnp.eye(2) * 2, (2, 2, 2)),
        radius=jnp.full((2,), 3 / np.sqrt(2)),
        color=jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        alpha=jnp.full((2,), 0.5),
        visible=jnp.ones((2,), jnp.bool_),
    )
    mask = build_cluster_tile_mask(projected, camera, config)
    np.testing.assert_array_equal(
        mask, [[True, False, False, False, False, False], [False, False, True, False, False, False]]
    )
    table = build_visibility_table(projected, camera, config, mask)
    assert table.tile_gaussian_ids.shape == (6, 1)
    np.testing.assert_array_equal(table.tile_count, [1, 0, 1, 0, 0, 0])
    assert not np.any(table.overflow)
    render = jax.jit(lambda p: rasterize_jax(p, table, camera, config))
    image = render(projected).rgb
    expected = 0.5 * np.exp(-0.5)
    np.testing.assert_allclose(image[4, 8], [expected, 0, 0], atol=1e-7)
    np.testing.assert_allclose(image[tile_height + 4, 8], [0, expected, 0], atol=1e-7)
    np.testing.assert_array_equal(image[-1, -1], 0)
    gradient = jax.grad(
        lambda alpha: render(projected.replace(alpha=alpha)).rgb[tile_height + 4, 8, 1]
    )(projected.alpha)
    np.testing.assert_allclose(gradient, [0, np.exp(-0.5)], atol=1e-7)


@pytest.mark.parametrize("has_visible", [False, True])
def test_reference_depth_masks_infinite_culled_slots(has_visible):
    config = CapacityConfig(2, 1, 4, 1, 0)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 1, 1, 0.5, 0.5, 1, 1)
    projected = ProjectedGaussians(
        mean=jnp.full((2, 2), 0.5),
        depth=jnp.array([2.0 if has_visible else jnp.inf, jnp.inf]),
        conic=jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
        radius=jnp.full((2,), 3.0),
        color=jnp.full((2, 3), 0.4),
        alpha=jnp.array([0.5, 0.0]),
        visible=jnp.array([has_visible, False]),
    )
    table = build_visibility_table(projected, camera, config)

    def objective(value):
        rendered = rasterize_jax(value, table, camera, config)
        return rendered.depth.sum(), rendered

    (loss, rendered), gradient = jax.jit(
        jax.value_and_grad(objective, allow_int=True, has_aux=True)
    )(projected)
    np.testing.assert_allclose(loss, 1.0 if has_visible else 0.0)
    np.testing.assert_allclose(rendered.rgb, 0.2 if has_visible else 0.0)
    np.testing.assert_allclose(rendered.alpha, 0.5 if has_visible else 0.0)
    for name in ("mean", "depth", "conic", "radius", "color", "alpha"):
        assert np.isfinite(np.asarray(getattr(gradient, name))).all(), name
    np.testing.assert_allclose(gradient.alpha, [2.0 if has_visible else 0.0, 0.0])
    np.testing.assert_allclose(gradient.depth, [0.5 if has_visible else 0.0, 0.0])


def test_raster_alpha_cutoff_and_cap():
    config = CapacityConfig(1, 1, 1, 1, 0)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 1, 1, 0.5, 0.5, 1, 1)
    projected = ProjectedGaussians(
        mean=jnp.array([[0.5, 0.5]]),
        depth=jnp.array([1.0]),
        conic=jnp.eye(2)[None],
        radius=jnp.array([1.0]),
        color=jnp.array([[1.0, 0.0, 0.0]]),
        alpha=jnp.array([0.1]),
        visible=jnp.array([True]),
    )
    table = VisibilityTable(
        jnp.array([[0]]),
        jnp.array([[1.0]]),
        jnp.array([[True]]),
        jnp.array([1]),
        jnp.array([False]),
    )
    renderers = [rasterize_jax]
    if jax.default_backend() == "gpu" and importlib.util.find_spec("cutlass") is not None:
        from jaxgs.kernels.rasterizer import rasterize_cute_vjp

        renderers.append(rasterize_cute_vjp)
    for renderer in renderers:
        for alpha, expected in ((0.001, 0.0), (0.1, 0.1), (1.0, 255.0 / 256)):
            result = renderer(projected.replace(alpha=jnp.array([alpha])), table, camera, config)
            np.testing.assert_allclose(result.rgb[0, 0, 0], expected, atol=1e-6)
            np.testing.assert_allclose(result.alpha[0, 0], expected, atol=1e-6)


def test_projection_culls_near_and_transparent_points():
    config = CapacityConfig(2, 2, 2, 4, 0)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 8, 8, 4, 4, 8, 8)
    pool = seed_gaussians(
        create_gaussians(config),
        jnp.array([[0.0, 0.0, 0.1], [0.0, 0.0, 2.0]]),
        jnp.full((2, 3), 0.5),
        opacity=0.5,
    )
    pool = pool.replace(opacity=pool.opacity.at[1, 0].set(jnp.log(0.001 / 0.999)))
    projectors = [project]
    if jax.default_backend() == "gpu" and importlib.util.find_spec("cutlass") is not None:
        from jaxgs.kernels.projector import project_cute

        projectors.append(project_cute)
    for projector in projectors:
        np.testing.assert_array_equal(projector(pool, camera, config).visible, [False, False])


def test_overflow_is_reported():
    config, camera, pool = scene(capacity=3, k=1)
    pool = seed_gaussians(
        create_gaussians(config),
        jnp.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]),
        jnp.ones((2, 3)) * 0.5,
        scale=0.2,
        opacity=0.5,
    )
    projected = project(pool, camera, config)
    table = build_visibility_table(projected, camera, config)
    assert np.any(table.overflow)
    assert table.tile_gaussian_ids.shape == (4, 1)


def test_slot_reuse_resets_moments():
    config, _, pool = scene()
    state = create_adam_state(pool)
    gradients = tuple(
        jnp.ones_like(getattr(pool, field))
        for field in ("xyz", "log_scale", "rotation", "opacity", "sh")
    )
    _, state = masked_adam_update(pool, state, gradients)
    pool, state = prune_step(pool, state, jnp.array([True, False, False, False]))
    np.testing.assert_array_equal(state.step, [0, 0, 0, 0])
    pool = seed_gaussians(pool, jnp.array([[0.1, 0.0, 2.0]]), jnp.ones((1, 3)) * 0.5)
    pool, state, count = densify_step(
        pool, state, jnp.array([1.0, 0.0, 0.0, 0.0]), jax.random.key(1), max_new=1, threshold=0.5
    )
    assert int(count) == 1
    assert int(pool.n_active) == 2
    assert bool(pool.alive[1])
    np.testing.assert_array_equal(state.m.xyz[1], 0)
    np.testing.assert_array_equal(state.v.xyz[1], 0)
    assert int(state.step[1]) == 0
    pool, state = reset_opacity(pool, state)
    assert np.all(np.asarray(jax.nn.sigmoid(pool.opacity[pool.alive])) <= 0.01001)


def test_split_keeps_capacity_and_shrinks_both_gaussians():
    config, _, pool = scene()
    pool = pool.replace(log_scale=pool.log_scale.at[0].set(jnp.log(0.1)))
    original_xyz = pool.xyz[0]
    child_pool, _, count = densify_step(
        pool,
        create_adam_state(pool),
        jnp.array([1.0, 0.0, 0.0, 0.0]),
        jax.random.key(2),
        max_new=1,
        threshold=0.5,
    )
    assert int(count) == 1
    assert child_pool.xyz.shape == pool.xyz.shape
    assert int(child_pool.n_active) == 2
    np.testing.assert_allclose(
        child_pool.xyz[0] + child_pool.xyz[1], 2 * original_xyz, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(child_pool.log_scale[0], jnp.log(0.1 / 1.6))
    np.testing.assert_allclose(child_pool.log_scale[1], jnp.log(0.1 / 1.6))


def test_split_offset_follows_gaussian_rotation():
    from jaxgs.render.projection import quaternion_to_matrix

    config, _, pool = scene()
    rotation = jnp.array([jnp.sqrt(0.5), 0.0, 0.0, jnp.sqrt(0.5)])
    scales = jnp.array([0.2, 0.1, 0.1])
    pool = pool.replace(
        rotation=pool.rotation.at[0].set(rotation),
        log_scale=pool.log_scale.at[0].set(jnp.log(scales)),
    )
    key = jax.random.key(3)
    child_pool, _, _ = densify_step(
        pool,
        create_adam_state(pool),
        jnp.array([1.0, 0.0, 0.0, 0.0]),
        key,
        max_new=1,
        threshold=0.5,
    )
    expected = quaternion_to_matrix(rotation) @ (jax.random.normal(key, (1, 3))[0] * scales / 1.6)
    np.testing.assert_allclose(
        (child_pool.xyz[1] - child_pool.xyz[0]) / 2, expected, rtol=1e-5, atol=1e-5
    )


def test_reference_gradient_matches_finite_difference():
    config, camera, pool = scene(capacity=1, k=1)
    pool = pool.replace(
        log_scale=pool.log_scale.at[0].set(jnp.log(jnp.array([0.18, 0.12, 0.23]))),
        rotation=pool.rotation.at[0].set(jnp.array([0.95, 0.1, 0.2, -0.1])),
    )
    projected = project(pool, camera, config)
    table = build_visibility_table(projected, camera, config)

    def loss(xyz, log_scale, rotation, opacity, sh):
        p = pool.replace(xyz=xyz, log_scale=log_scale, rotation=rotation, opacity=opacity, sh=sh)
        image = rasterize_jax(project(p, camera, config), table, camera, config).rgb
        return (
            jnp.sum(image * jnp.arange(image.size, dtype=jnp.float32).reshape(image.shape))
            / image.size
        )

    params = (pool.xyz, pool.log_scale, pool.rotation, pool.opacity, pool.sh)
    gradients = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*params)
    for arg, index in enumerate(((0, 0), (0, 1), (0, 2), (0, 0), (0, 0, 0))):
        delta = jnp.zeros_like(params[arg]).at[index].set(1e-3)
        plus = list(params)
        minus = list(params)
        plus[arg] += delta
        minus[arg] -= delta
        numeric = (loss(*plus) - loss(*minus)) / 2e-3
        np.testing.assert_allclose(gradients[arg][index], numeric, rtol=0.03, atol=0.003)
