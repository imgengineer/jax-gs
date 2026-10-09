"""Training may skip culled SH values without changing rendering or its pullback."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig, create_gaussians, seed_gaussians

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="Projection visibility requires JAX CUDA and CuTe",
)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("empty", [False, True])
def test_visible_color_preserves_render_and_parameter_gradients(degree, empty):
    from jaxgs.kernels.cluster_compact import compact_visible_clusters
    from jaxgs.kernels.packed_rasterizer import packed_backward, packed_forward
    from jaxgs.kernels.projector import project_with_compact_pullback
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute

    capacity, cluster_size = 131, 65
    config = CapacityConfig(capacity, cluster_size, 16, 16, 3, 4096, tile_height=8)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 9, 8, 19, 17, far=5.0)
    rng = np.random.default_rng(912)
    xyz = rng.uniform([-0.3, -0.2, 2.0], [0.3, 0.2, 3.0], (capacity, 3)).astype(np.float32)
    # A listed cluster contains points rejected by every projection predicate.
    xyz[1:6] = [[8, 0, 2], [0, -8, 2], [0, 0, -2], [0, 0, 0.1], [0, 0, 6]]
    pool = seed_gaussians(
        create_gaussians(config), jnp.asarray(xyz), jnp.full((capacity, 3), 0.4), scale=0.12
    )
    slots = np.arange(capacity) // cluster_size != 1
    if empty:
        slots[:] = False
    pool = pool.replace(
        alive=jnp.asarray(slots).at[7].set(False),
        opacity=pool.opacity.at[6].set(-20),
        sh=jnp.asarray(rng.normal(0, 0.2, pool.sh.shape), jnp.float32).at[8, 0, 0].set(-4),
    )
    clusters = compact_visible_clusters(jnp.asarray(slots), cluster_size)

    def project(visible_color_only):
        return project_with_compact_pullback(
            pool,
            camera,
            config,
            degree,
            clusters,
            rgb_only=True,
            visible_color_only=visible_color_only,
        )

    expected, expected_pullback = project(False)
    actual, actual_pullback = project(True)
    np.testing.assert_array_equal(actual.visible, expected.visible)
    visible = np.asarray(expected.visible)
    for name in ("mean", "depth", "conic", "radius", "alpha"):
        np.testing.assert_array_equal(
            np.asarray(getattr(actual, name))[slots], np.asarray(getattr(expected, name))[slots]
        )
    np.testing.assert_array_equal(
        np.asarray(actual.color)[visible], np.asarray(expected.color)[visible]
    )
    np.testing.assert_array_equal(np.asarray(actual.color)[slots & ~visible], 0)
    if not empty:
        assert visible[0] and visible[-1] and not visible[1:8].any()
        assert np.any(np.asarray(expected.color)[slots & ~visible] > 0)
        assert actual.color[8, 0] == 0  # Keep the original clamp and its derivative.

    table = build_sorted_visibility_table_cute(expected, camera, config)
    image_grad = jnp.asarray(rng.normal(0, 0.01, (camera.height, camera.width, 3)), jnp.float32)
    for collect_stats in (False, True):
        image, cache, fragments = packed_forward(actual, table, camera, config, collect_stats)
        reference, reference_cache, reference_fragments = packed_forward(
            expected, table, camera, config, collect_stats
        )
        np.testing.assert_array_equal(image, reference)
        np.testing.assert_array_equal(cache[1], reference_cache[1])
        if collect_stats:
            np.testing.assert_allclose(fragments, reference_fragments, rtol=1e-5, atol=1e-7)
        gradients, _ = packed_backward(
            actual,
            table,
            cache,
            image_grad,
            camera,
            config,
            collect_stats,
            symmetric_conic=True,
            visible_clusters=clusters,
        )
        # Use identical upstream gradients to isolate projection from atomic order.
        for result, ref in zip(
            actual_pullback(gradients), expected_pullback(gradients), strict=True
        ):
            np.testing.assert_array_equal(result[: slots.sum()], ref[: slots.sum()])
