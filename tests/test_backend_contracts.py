"""Public rendering contracts and errors that must precede GPU launches."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig, create_pool, seed_pool
from jaxgs.render.projection import project
from jaxgs.render.rasterizer import rasterize, rasterize_forward
from jaxgs.render.visibility_table import build_visibility_table

has_cute = importlib.util.find_spec("cutlass") is not None
requires_cute = pytest.mark.skipif(not has_cute, reason="requires the cute extra")
requires_gpu = pytest.mark.skipif(
    not has_cute or jax.default_backend() != "gpu", reason="requires JAX CUDA and CuTe"
)


@pytest.fixture
def scene():
    config = CapacityConfig(4, 2, 4, 4, 1, 64)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 8, 8, 4, 4, 8, 8)
    pool = seed_pool(
        create_pool(config),
        jnp.array([[0.1, -0.1, 2.0], [-0.2, 0.2, 3.0]]),
        jnp.array([[0.7, 0.2, 0.3], [0.2, 0.5, 0.1]]),
        scale=0.2,
        opacity=0.4,
    )
    projected = project(pool, camera, config)
    table = build_visibility_table(projected, camera, config)
    return config, camera, pool, projected, table


@pytest.mark.parametrize("backend", ["reference", pytest.param("cute", marks=requires_gpu)])
def test_forward_dispatch_preserves_background_depth_and_alpha(scene, backend):
    from jaxgs.reference.rasterizer_jax import rasterize_jax

    config, camera, _, projected, table = scene
    background = jnp.array([0.15, 0.25, 0.35])
    expected = rasterize_jax(projected, table, camera, config, background)
    actual = rasterize_forward(
        projected, table, camera, config, backend=backend, background=background
    )
    differentiable = rasterize(
        projected, table, camera, config, backend=backend, background=background
    )
    for result in (actual, differentiable):
        for value, reference in zip(
            jax.tree.leaves(result), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(value, reference, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("render", [rasterize_forward, rasterize])
def test_render_rejects_unknown_backend(scene, render):
    config, camera, _, projected, table = scene
    with pytest.raises(ValueError, match="unknown backend: typo"):
        render(projected, table, camera, config, backend="typo")


@requires_cute
@pytest.mark.parametrize(
    "operation", ["allocate", "cluster", "project", "bin", "render", "sorted_render"]
)
def test_cute_rejects_cpu_before_launch(scene, monkeypatch, operation):
    import cutlass.jax

    from jaxgs.kernels.allocator import allocate_free_slots_cute
    from jaxgs.kernels.binning import compact_clusters_cute
    from jaxgs.kernels.projector import project_cute
    from jaxgs.kernels.rasterizer import rasterize_cute
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
    from jaxgs.kernels.sorted_rasterizer import rasterize_sorted_cute_vjp

    config, camera, pool, projected, table = scene
    # Simulate the device boundary, while retaining real inputs and wrappers.
    # A regression must not silently dispatch a CUDA call on a CPU backend.
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")

    def unexpected_launch(*args, **kwargs):
        pytest.fail("CuTe launch requested on a CPU backend")

    monkeypatch.setattr(cutlass.jax, "cutlass_call", unexpected_launch)
    calls = {
        "allocate": lambda: allocate_free_slots_cute(pool.free_mask, 2),
        "cluster": lambda: compact_clusters_cute(projected, config),
        "project": lambda: project_cute(pool, camera, config),
        "bin": lambda: build_sorted_visibility_table_cute(projected, camera, config),
        "render": lambda: rasterize_cute(projected, table, camera, config),
        # The device check precedes access to the sorted table representation.
        "sorted_render": lambda: rasterize_sorted_cute_vjp(projected, None, camera, config),
    }
    with pytest.raises(RuntimeError, match="requires a JAX CUDA device"):
        calls[operation]()


@requires_cute
@pytest.mark.parametrize("degree", [-1, 2])
@pytest.mark.parametrize("sparse", [False, True])
def test_projection_rejects_degree_outside_allocated_sh(scene, degree, sparse):
    from jaxgs.kernels.projector import project_cute_vjp, project_with_compact_pullback

    config, camera, pool, _, _ = scene
    with pytest.raises(ValueError, match="active_degree must fit"):
        if sparse:
            project_with_compact_pullback(pool, camera, config, degree, None)
        else:
            project_cute_vjp(pool, camera, config, degree)


@requires_cute
@pytest.mark.parametrize("shape,target_shape", [((4, 4, 1), (4, 4, 1)), ((4, 4, 3), (4, 3, 3))])
def test_fused_loss_rejects_non_rgb_or_mismatched_images(shape, target_shape):
    from jaxgs.kernels.fused_loss import fused_loss_and_grad

    with pytest.raises(ValueError, match="matching HWC RGB"):
        fused_loss_and_grad(jnp.zeros(shape), jnp.zeros(target_shape))


@requires_gpu
def test_renderers_reject_unsupported_tile_shapes(scene):
    from jaxgs.kernels.packed_rasterizer import packed_forward
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
    from jaxgs.kernels.sorted_rasterizer import rasterize_sorted_cute_vjp

    config, camera, _, projected, _ = scene
    table = build_sorted_visibility_table_cute(projected, camera, config)
    with pytest.raises(ValueError, match="Packed rasterizer supports tiles"):
        packed_forward(projected, table, camera, config)  # 4x4 is too small for half2.
    rectangular = CapacityConfig(4, 2, 4, 16, 1, 64, tile_height=8)
    with pytest.raises(ValueError, match="requires square tiles"):
        rasterize_sorted_cute_vjp(projected, table, camera, rectangular)


@requires_gpu
def test_array_training_rejects_unknown_optimizer(scene):
    from jaxgs.scene.cluster import world_cluster_bounds
    from jaxgs.training.optimizer import create_adam_state
    from jaxgs.training.step import array_train_step

    config, camera, pool, _, _ = scene
    with pytest.raises(ValueError, match="unknown optimizer: typo"):
        array_train_step(
            pool,
            create_adam_state(pool),
            jnp.zeros((4, 4)),
            world_cluster_bounds(pool, config.cluster_size),
            camera,
            jnp.zeros((8, 8, 3), jnp.uint8),
            jnp.array(0),
            jnp.array(1.0),
            config,
            0,
            False,
            optimizer="typo",
        )


@requires_cute
def test_bounded_cute_interfaces_reject_rectangles_before_launch(scene, monkeypatch):
    import cutlass.jax

    from jaxgs.kernels.binning import build_visibility_table_cute
    from jaxgs.kernels.rasterizer import rasterize_cute, rasterize_cute_vjp

    _, camera, _, projected, table = scene
    rectangular = CapacityConfig(4, 2, 4, 16, 1, 64, tile_height=8)

    def unexpected_launch(*args, **kwargs):
        pytest.fail("unsupported tile shape reached a CuTe launch")

    monkeypatch.setattr(cutlass.jax, "cutlass_call", unexpected_launch)
    calls = (
        lambda: build_visibility_table_cute(projected, camera, rectangular),
        lambda: rasterize_cute(projected, table, camera, rectangular),
        lambda: rasterize_cute_vjp(projected, table, camera, rectangular),
    )
    for call in calls:
        with pytest.raises(ValueError, match="requires square tiles"):
            call()
