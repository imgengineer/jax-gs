import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig, create_gaussians, seed_gaussians
from jaxgs.reference.rasterizer_jax import rasterize_jax
from jaxgs.render.cluster_culling import build_cluster_tile_mask
from jaxgs.render.projection import project
from jaxgs.render.visibility_table import build_visibility_table

requires_cute = pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe edge regression requires JAX CUDA",
)


def edge_scene(edge="left", anisotropic=False):
    vertical = edge in ("top", "bottom")
    width, height = (128, 256) if vertical else (256, 128)
    camera = Camera.from_colmap(
        [1, 0, 0, 0], [0, 0, 0], 128, 128, width / 2, height / 2, width, height
    )
    config = CapacityConfig(128, 128, 1, 16, 0, 512)
    axis = 1 if vertical else 0
    mean = np.array([width / 2 + 0.5, height / 2 + 0.5])
    mean[axis] = -24.1 if edge in ("left", "top") else 256 + 24.1
    pixel = mean.astype(int)
    pixel[axis] = 0 if edge in ("left", "top") else 255
    scale = np.sqrt(np.array([64.0, 36.0 if anisotropic else 64.0]) - 0.3) / 64
    pool = seed_gaussians(
        create_gaussians(config),
        jnp.array([[(mean[0] - width / 2) / 64, (mean[1] - height / 2) / 64, 2.0]]),
        jnp.array([[0.3, 0.5, 0.7]]),
        scale=jnp.array([[scale[0], scale[1], 1e-6]]),
        opacity=0.99,
    )
    angle = (np.pi / 2 if vertical else 0) + (0.2 if anisotropic else 0)
    pool = pool.replace(
        rotation=pool.rotation.at[0].set(jnp.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]))
    )
    return config, camera, pool, tuple(pixel)


def pixel_contribution(projected, pixel):
    delta = jnp.array(pixel, jnp.float32) + 0.5 - projected.mean[0]
    alpha = projected.alpha[0] * jnp.exp(-0.5 * delta @ projected.conic[0] @ delta)
    return alpha * projected.color[0]


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
@pytest.mark.parametrize("anisotropic", [False, True])
def test_reference_preserves_edge_support(edge, anisotropic):
    config, camera, pool, pixel = edge_scene(edge, anisotropic)
    projected = project(pool, camera, config)
    axis = 1 if edge in ("top", "bottom") else 0
    boundary = 0 if edge in ("left", "top") else 256
    assert abs(float(projected.mean[0, axis]) - boundary) > float(projected.radius[0])
    expected = pixel_contribution(projected, pixel)
    assert float(expected[0] / projected.color[0, 0]) > 1 / 256
    assert bool(projected.visible[0])
    table = build_visibility_table(
        projected, camera, config, build_cluster_tile_mask(projected, camera, config)
    )
    image = rasterize_jax(projected, table, camera, config).rgb
    np.testing.assert_allclose(image[pixel[1], pixel[0]], expected, rtol=1e-5, atol=1e-7)
    assert not np.any(table.overflow)


@pytest.mark.parametrize("backend", ["jax", pytest.param("cute", marks=requires_cute)])
def test_projection_uses_native_coarse_bounds(backend):
    config, camera, pool, _ = edge_scene()
    means = np.array([[-38.3, 64], [-38.5, 64], [294.3, 64], [294.5, 64], [128, 64]])
    xyz = np.c_[(means - [128, 64]) / 64, np.full(5, 2)]
    pool = seed_gaussians(create_gaussians(config), jnp.array(xyz), jnp.full((5, 3), 0.5), scale=2)
    pool = pool.replace(opacity=pool.opacity.at[4, 0].set(jnp.log(0.001 / 0.999)))
    projector = project
    if backend == "cute":
        from jaxgs.kernels.projector import project_cute

        projector = project_cute
    np.testing.assert_array_equal(
        projector(pool, camera, config).visible[:5], [True, False, True, False, False]
    )


@requires_cute
@pytest.mark.parametrize("backend", ["bounded", "sorted", "packed"])
def test_cute_edge_support_and_parameter_pullback(backend):
    from jaxgs.kernels.binning import build_visibility_table_cute
    from jaxgs.kernels.packed_rasterizer import rasterize_packed_cute_vjp
    from jaxgs.kernels.projector import project_cute_vjp
    from jaxgs.kernels.rasterizer import rasterize_cute_vjp
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
    from jaxgs.kernels.sorted_rasterizer import rasterize_sorted_cute_vjp

    config, camera, pool, pixel = edge_scene(anisotropic=True)
    projected = project_cute_vjp(pool, camera, config)
    if backend == "bounded":
        table = build_visibility_table_cute(projected, camera, config)
    else:
        table = build_sorted_visibility_table_cute(projected, camera, config)
    renderers = {
        "bounded": lambda p: rasterize_cute_vjp(p, table, camera, config).rgb,
        "sorted": lambda p: rasterize_sorted_cute_vjp(p, table, camera, config).rgb,
        "packed": lambda p: rasterize_packed_cute_vjp(p, table, camera, config),
    }
    renderer = renderers[backend]
    assert bool(projected.visible[0])
    assert not np.any(table.overflow)
    with jax.default_matmul_precision("highest"):
        expected_pixel = pixel_contribution(project(pool, camera, config), pixel)
    np.testing.assert_allclose(
        renderer(projected)[pixel[1], pixel[0]], expected_pixel, rtol=0.005, atol=1e-5
    )

    def loss(p):
        return jnp.sum(renderer(project_cute_vjp(p, camera, config))[pixel[1], pixel[0]])

    with jax.default_matmul_precision("highest"):
        actual = jax.jit(jax.grad(loss, allow_int=True))(pool)
        expected = jax.grad(
            lambda p: jnp.sum(pixel_contribution(project(p, camera, config), pixel)), allow_int=True
        )(pool)
    for name in ("xyz", "log_scale", "rotation", "opacity", "sh"):
        a, b = np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name))
        assert np.linalg.norm(b[0]) > 0, name
        np.testing.assert_allclose(a[0], b[0], rtol=0.025, atol=2e-5)
        np.testing.assert_array_equal(a[1:], 0)


@requires_cute
def test_production_step_updates_edge_gaussian():
    from jaxgs.scene.cluster import world_cluster_bounds
    from jaxgs.training.optimizer import create_adam_state
    from jaxgs.training.step import array_train_step

    config, camera, pool, _ = edge_scene()
    config = CapacityConfig(128, 128, 1, 16, 0, 512, tile_height=8)
    updated, state, stats, metrics = array_train_step(
        pool,
        create_adam_state(pool),
        jnp.zeros((128, 4)),
        world_cluster_bounds(pool, 128),
        camera,
        jnp.zeros((camera.height, camera.width, 3), jnp.uint8),
        jnp.array(0),
        jnp.array(1.0),
        config,
        0,
        True,
    )
    assert int(metrics["pairs"]) == 4
    assert not bool(metrics["overflow"])
    assert float(metrics["loss"]) > 0
    assert float(stats[0, 0]) > 0
    for name in ("xyz", "log_scale", "opacity", "sh"):
        assert not np.array_equal(getattr(updated, name)[0], getattr(pool, name)[0]), name
        np.testing.assert_array_equal(getattr(updated, name)[1:], getattr(pool, name)[1:])
        np.testing.assert_array_equal(getattr(state.m, name)[1:], 0)
