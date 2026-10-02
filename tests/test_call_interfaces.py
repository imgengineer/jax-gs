"""LiteGS-style calls retain JAX shapes, NNX updates and rendering contracts."""

import importlib.util
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxgs import Camera, CapacityConfig, GaussianModel, create_gaussians, seed_gaussians
from jaxgs.reference.rasterizer_jax import rasterize_jax
from jaxgs.render import render, render_preprocess
from jaxgs.render.projection import project
from jaxgs.render.visibility_table import build_visibility_table
from jaxgs.scene.cluster import world_cluster_bounds

requires_gpu = pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="requires JAX CUDA and CuTe",
)
backends = ["reference", pytest.param("cute", marks=requires_gpu)]


def scene(tile_size=4, tile_height=None):
    config = CapacityConfig(5, 2, 5, tile_size, 1, 256, tile_height=tile_height)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 24, 24, 16, 8, 32, 16)
    pool = seed_gaussians(
        create_gaussians(config),
        jnp.array([[0.1, 0.0, 2.0], [-0.1, 0.1, 2.5], [25.0, 0.0, 2.0]]),
        jnp.array([[0.4, 0.5, 0.3], [0.6, 0.4, 0.5], [0.3, 0.2, 0.1]]),
        scale=0.2,
        opacity=0.5,
    )
    pool = pool.replace(sh=pool.sh.at[:, 1:].set(0.05).at[0, 0, 0].add(8.0))
    return config, camera, pool


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("empty", [False, True])
def test_preprocess_keeps_fixed_shapes_and_parameter_buffers(backend, empty):
    config, camera, pool = scene()
    if empty:
        pool = pool.replace(alive=jnp.zeros_like(pool.alive))
    clusters, mask, culled = render_preprocess(
        world_cluster_bounds(pool, config.cluster_size), camera, pool, config, backend=backend
    )
    ids, count = clusters
    assert ids.shape == (3,) and count.shape == (1,) and mask.shape == (5,)
    np.testing.assert_array_equal(ids[: int(count[0])], [] if empty else [0])
    np.testing.assert_array_equal(culled.alive, pool.alive & mask)
    for name in pool.__dataclass_fields__:
        if name != "alive":
            assert getattr(culled, name) is getattr(pool, name)
    np.testing.assert_array_equal(pool.alive, [False] * 5 if empty else [True] * 3 + [False] * 2)


@pytest.mark.parametrize(
    "backend,tile_size,tile_height",
    [
        ("reference", 4, 4),
        pytest.param("cute", 4, 4, marks=requires_gpu),
        pytest.param("cute", 16, 8, marks=requires_gpu),
    ],
)
@pytest.mark.parametrize("degree", [0, 1])
def test_render_calls_match_existing_backend_and_reuse_nnx_compilation(
    backend, tile_size, tile_height, degree
):
    config, camera, pool = scene(tile_size, tile_height)
    bounds = world_cluster_bounds(pool, config.cluster_size)
    model = GaussianModel(pool)

    @nnx.jit(graph=False)
    def render_view(model, camera):
        clusters, _, culled = render_preprocess(
            bounds, camera, model.as_arrays(), config, backend=backend
        )
        return render(camera, culled, clusters, degree, config, backend=backend)

    actual = render_view(model, camera)
    if backend == "reference":
        projected = project(pool, camera, replace(config, sh_degree=degree))
        table = build_visibility_table(projected, camera, config)
        image = rasterize_jax(projected, table, camera, config).rgb
    else:
        from jaxgs.kernels.packed_rasterizer import packed_forward
        from jaxgs.kernels.projector import project_cute_vjp
        from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
        from jaxgs.kernels.sorted_rasterizer import rasterize_sorted_cute_vjp

        projected = project_cute_vjp(pool, camera, config, degree)
        table = build_sorted_visibility_table_cute(projected, camera, config)
        image = (
            packed_forward(projected, table, camera, config)[0]
            if tile_size == 16
            else rasterize_sorted_cute_vjp(projected, table, camera, config).rgb
        )
    np.testing.assert_allclose(actual.image, jnp.clip(image, 0, 1), rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(actual.primitive_visible, [True, True, False, False, False])
    assert actual.overflow.shape == () and not bool(actual.overflow)
    assert float(actual.image.max()) == 1.0
    model.sh.set_value(pool.sh.at[0, 0, 1].add(0.02))
    updated = render_view(model, camera)
    assert float(updated.image[:, :, 1].sum()) > float(actual.image[:, :, 1].sum())
    assert render_view.jitted_fn._cache_size() == 1

    def image_sum(sh):
        clusters, _, culled = render_preprocess(
            bounds, camera, pool.replace(sh=sh), config, backend=backend
        )
        return render(camera, culled, clusters, degree, config, backend=backend).image.sum()

    gradient = jax.jit(jax.grad(image_sum))(pool.sh)
    assert np.all(np.isfinite(gradient)) and np.any(np.asarray(gradient[:2]) != 0)
    np.testing.assert_array_equal(gradient[2:], 0)
    if degree == 0:
        np.testing.assert_array_equal(gradient[:, 1:], 0)


@pytest.mark.parametrize("operation", ["preprocess", "render"])
def test_render_interface_rejects_unknown_backend(operation):
    config, camera, pool = scene()
    with pytest.raises(ValueError, match="unknown backend: typo"):
        if operation == "preprocess":
            render_preprocess(world_cluster_bounds(pool, 2), camera, pool, config, backend="typo")
        else:
            render(camera, pool, (jnp.array([0]), jnp.array([1])), 0, config, backend="typo")


@pytest.mark.parametrize("degree", [-1, 2])
def test_render_interface_rejects_invalid_sh_degree(degree):
    config, camera, pool = scene()
    with pytest.raises(ValueError, match="actived_sh_degree"):
        render(camera, pool, (jnp.array([0]), jnp.array([1])), degree, config)


def test_reference_render_exposes_overflow():
    config, camera, pool = scene()
    config = replace(config, max_gaussians_per_tile=1)
    clusters, _, culled = render_preprocess(
        world_cluster_bounds(pool, 2), camera, pool, config, backend="reference"
    )
    result = render(camera, culled, clusters, 0, config, backend="reference")
    assert bool(result.overflow) and result.image.shape == (16, 32, 3)
