import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig, create_pool, seed_pool
from jaxgs.reference.rasterizer_jax import rasterize_jax
from jaxgs.render.projection import project
from jaxgs.render.visibility_table import build_visibility_table
from jaxgs.training.optimizer import create_adam_state
from jaxgs.training.reference_densify import densify_step
from jaxgs.training.reference_trainer import train_step


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_cute_forward_and_backward_match_reference():
    from jaxgs.kernels.binning import build_visibility_table_cute
    from jaxgs.render.rasterize_backward import rasterize
    from jaxgs.render.rasterize_forward import rasterize_forward

    config = CapacityConfig(2, 2, 2, 4, 0)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 8, 8, 4, 4, 8, 8)
    pool = seed_pool(
        create_pool(config),
        jnp.array([[0.1, 0.0, 2.0], [-0.2, 0.1, 3.0]], jnp.float32),
        jnp.array([[1.0, 0.2, 0.1], [0.1, 0.5, 0.9]], jnp.float32),
        scale=0.2,
        opacity=0.5,
    )
    projected = project(pool, camera, config)
    table = build_visibility_table(projected, camera, config)
    forward = rasterize_forward(projected, table, camera, config)
    reference = rasterize_jax(projected, table, camera, config)
    narrow_table = build_visibility_table_cute(projected, camera, CapacityConfig(2, 2, 1, 4, 0))
    assert np.any(narrow_table.overflow)
    for name in ("rgb", "depth", "alpha"):
        np.testing.assert_allclose(
            getattr(forward, name), getattr(reference, name), rtol=1e-5, atol=1e-6
        )

    def loss(p, renderer):
        result = renderer(p, table, camera, config)
        return jnp.sum(result.rgb**2) + 0.1 * jnp.sum(result.depth) + 0.2 * jnp.sum(result.alpha)

    cute_grads = jax.grad(lambda p: loss(p, rasterize), allow_int=True)(projected)
    ref_grads = jax.grad(lambda p: loss(p, rasterize_jax), allow_int=True)(projected)
    for name in ("mean", "conic", "depth", "color", "alpha"):
        np.testing.assert_allclose(
            getattr(cute_grads, name), getattr(ref_grads, name), rtol=2e-3, atol=2e-4
        )


@pytest.mark.parametrize(
    "capacity,num_tiles,dtype",
    [
        (1, 35, np.int32),
        (257, 35, np.int32),
        (1024, 8034, np.int32),
        (257, 65535, np.uint16),
        (257, 65536, np.uint32),
    ],
)
@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_sorted_tile_ranges_cover_gaps_padding_and_full_capacity(capacity, num_tiles, dtype):
    from cutlass.jax import cutlass_call

    from jaxgs.kernels.sorted_visibility import launch_tile_ranges

    ranges = jax.jit(
        cutlass_call(
            launch_tile_ranges,
            output_shape_dtype=jax.ShapeDtypeStruct((num_tiles + 1,), jnp.int32),
            use_static_tensors=True,
            max_pairs=capacity,
            num_tiles=num_tiles,
        )
    )
    rng = np.random.default_rng(71)
    cases = [
        np.full(capacity, num_tiles, np.int32),
        np.zeros(capacity, np.int32),
        np.full(capacity, num_tiles // 2, np.int32),
        np.full(capacity, num_tiles - 1, np.int32),
        rng.choice([0, num_tiles // 2, num_tiles - 1, num_tiles], capacity),
        rng.integers(0, num_tiles, capacity),
    ]
    for keys in cases:
        keys = np.sort(keys.astype(dtype))
        expected = np.searchsorted(keys, np.arange(num_tiles + 1), side="left")
        np.testing.assert_array_equal(ranges(jnp.asarray(keys)), expected)


@pytest.mark.parametrize("tiles_x", [255, 256])
@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_sorted_binning_tile_key_width_preserves_padding_sentinel(tiles_x):
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute

    config = CapacityConfig(1, 1, 1, 8, 0, 4)
    width, height = tiles_x * 8, 256 * 8
    camera = Camera.from_colmap(
        [1, 0, 0, 0], [0, 0, 0], 20, 20, width - 4, height - 4, width, height
    )
    projected = project(create_pool(config), camera, config)
    table = build_sorted_visibility_table_cute(projected, camera, config)
    np.testing.assert_array_equal(table.tile_offsets, 0)
    assert int(table.pair_count) == 0
    assert not bool(table.overflow)
    # A live point in the last tile exercises IDs above the signed-16 limit,
    # as well as the uint32 fallback when the padding sentinel is 65536.
    pool = seed_pool(
        create_pool(config),
        jnp.array([[0.0, 0.0, 2.0]]),
        jnp.full((1, 3), 0.5),
        scale=0.02,
        opacity=0.5,
    )
    table = build_sorted_visibility_table_cute(project(pool, camera, config), camera, config)
    assert int(table.pair_count) == 1
    assert not bool(table.overflow)
    np.testing.assert_array_equal(table.tile_offsets[:-1], 0)
    assert int(table.tile_offsets[-1]) == 1
    assert int(table.gaussian_ids[0]) == 0


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_sorted_binning_keeps_more_than_one_gaussian_per_tile():
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
    from jaxgs.kernels.sorted_rasterizer import rasterize_sorted_cute_vjp

    config = CapacityConfig(4, 2, 1, 4, 0, 16)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 8, 8, 4, 4, 8, 8)
    pool = seed_pool(
        create_pool(config),
        jnp.array([[0.0, 0.0, 3.0], [0.1, 0.0, 2.0], [0.0, 0.1, 2.0]]),
        jnp.ones((3, 3)) * 0.5,
        scale=0.2,
        opacity=0.5,
    )
    projected = project(pool, camera, config)
    table = build_sorted_visibility_table_cute(projected, camera, config)
    np.testing.assert_array_equal(np.diff(table.tile_offsets), [3, 3, 3, 3])
    np.testing.assert_array_equal(table.gaussian_ids[:12].reshape(4, 3), np.tile([1, 2, 0], (4, 1)))
    assert int(table.pair_count) == 12
    assert not bool(table.overflow)

    reference_config = CapacityConfig(4, 2, 3, 4, 0)
    reference_table = build_visibility_table(projected, camera, reference_config)
    result = rasterize_sorted_cute_vjp(projected, table, camera, config)
    reference = rasterize_jax(projected, reference_table, camera, reference_config)
    for name in ("rgb", "depth", "alpha"):
        np.testing.assert_allclose(
            getattr(result, name), getattr(reference, name), rtol=1e-5, atol=1e-6
        )

    def objective(p, renderer, visibility, settings):
        image = renderer(p, visibility, camera, settings)
        return jnp.sum(image.rgb**2) + 0.1 * jnp.sum(image.depth) + 0.2 * jnp.sum(image.alpha)

    actual = jax.grad(
        lambda p: objective(p, rasterize_sorted_cute_vjp, table, config), allow_int=True
    )(projected)
    expected = jax.grad(
        lambda p: objective(p, rasterize_jax, reference_table, reference_config), allow_int=True
    )(projected)
    for name in ("mean", "conic", "depth", "color", "alpha"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=3e-3, atol=3e-4
        )

    narrow = build_sorted_visibility_table_cute(projected, camera, CapacityConfig(4, 2, 1, 4, 0, 4))
    assert bool(narrow.overflow)


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_sorted_binning_keeps_large_ellipses_and_handles_empty_scene():
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
    from jaxgs.kernels.sorted_rasterizer import rasterize_sorted_cute_vjp

    config = CapacityConfig(1, 1, 1, 16, 0, 4)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 8, 9, 17, 19)
    pool = seed_pool(
        create_pool(config),
        jnp.array([[0.0, 0.0, 2.0]]),
        jnp.ones((1, 3)) * 0.5,
        scale=0.2,
        opacity=0.5,
    )
    projected = project(pool, camera, config).replace(conic=jnp.array([[[1e-5, 0.0], [0.0, 1e-5]]]))
    table = build_sorted_visibility_table_cute(projected, camera, config)
    assert int(table.pair_count) == 4  # positive determinant < 1e-8 is valid
    assert not bool(table.overflow)
    projected = projected.replace(visible=jnp.array([False]))
    empty = build_sorted_visibility_table_cute(projected, camera, config)
    np.testing.assert_array_equal(empty.tile_offsets, 0)
    assert int(empty.pair_count) == 0
    background = jnp.array([0.1, 0.2, 0.3])
    result = rasterize_sorted_cute_vjp(projected, empty, camera, config, background)
    np.testing.assert_allclose(result.rgb, jnp.broadcast_to(background, result.rgb.shape))
    np.testing.assert_array_equal(result.alpha, 0)


@pytest.mark.parametrize("tile_size", [16, 32])
@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_sorted_rasterizer_partial_tiles_and_early_termination(tile_size):
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
    from jaxgs.kernels.sorted_rasterizer import rasterize_sorted_cute_vjp
    from jaxgs.render.projection import ProjectedGaussians

    count, width, height = 24, 23, 19
    config = CapacityConfig(count, 4, 1, tile_size, 0, count * 4)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 11, 9, width, height)
    key = jax.random.key(41)
    projected = ProjectedGaussians(
        mean=jax.random.uniform(key, (count, 2), minval=8, maxval=12),
        conic=jnp.broadcast_to(jnp.array([[0.014, 0.003], [0.003, 0.025]]), (count, 2, 2)),
        depth=jnp.linspace(1.5, 4.0, count),
        color=jax.random.uniform(key, (count, 3)),
        alpha=jnp.linspace(0.72, 0.97, count),
        radius=jnp.full((count,), 30.0),
        visible=jnp.ones((count,), bool),
    )
    table = build_sorted_visibility_table_cute(projected, camera, config)
    assert not bool(table.overflow)
    # Every ellipse covers every tile, so the independent reference can use
    # one depth-ordered dense list without depending on the binning code.
    np.testing.assert_array_equal(np.diff(table.tile_offsets), count)
    yy, xx = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    pixels = jnp.stack((xx + 0.5, yy + 0.5), axis=-1)
    background = jnp.array([0.12, 0.24, 0.36])

    def reference(p):
        delta = pixels[:, :, None, :] - p.mean
        exponent = -0.5 * jnp.einsum(
            "hwgi,gij,hwgj->hwg", delta, p.conic, delta, precision=jax.lax.Precision.HIGHEST
        )
        raw_alpha = p.alpha * jnp.exp(exponent)
        alpha = jnp.where(
            (exponent >= -4.5) & (raw_alpha >= 1 / 256), jnp.minimum(raw_alpha, 255 / 256), 0
        )
        prefix = jnp.cumprod(1 - alpha, axis=-1)
        before = jnp.roll(prefix, 1, axis=-1).at[:, :, 0].set(1)
        active = before >= 1 / 8192
        alpha = jnp.where(active, alpha, 0)
        weight = before * alpha
        final_t = jnp.prod(1 - alpha, axis=-1)
        return (
            jnp.einsum("hwg,gc->hwc", weight, p.color, precision=jax.lax.Precision.HIGHEST)
            + final_t[:, :, None] * background,
            jnp.sum(weight * p.depth, axis=-1),
            1 - final_t,
        )

    def actual(p):
        result = rasterize_sorted_cute_vjp(p, table, camera, config, background)
        return result.rgb, result.depth, result.alpha

    for expected, value in zip(reference(projected), actual(projected), strict=True):
        np.testing.assert_allclose(value, expected, rtol=2e-4, atol=2e-5)

    def objective(p, renderer):
        rgb, depth, alpha = renderer(p)
        return jnp.sum(rgb**2) + 0.1 * jnp.sum(depth) + 0.2 * jnp.sum(alpha)

    expected = jax.grad(objective, allow_int=True)(projected, reference)
    value = jax.grad(objective, allow_int=True)(projected, actual)
    for name in ("mean", "conic", "depth", "color", "alpha"):
        np.testing.assert_allclose(
            getattr(value, name), getattr(expected, name), rtol=3e-3, atol=3e-4, err_msg=name
        )


@pytest.mark.parametrize("pipeline", ["bounded", "sorted"])
@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_cute_projection_binning_and_parameter_gradients(pipeline):
    from jaxgs.kernels.binning import build_visibility_table_cute
    from jaxgs.kernels.projector import project_cute, project_cute_vjp
    from jaxgs.kernels.rasterizer import rasterize_cute, rasterize_cute_vjp

    config = CapacityConfig(2, 2, 2, 4, 3)
    camera = Camera.from_colmap([0.98, 0.02, -0.1, 0.15], [0.01, 0.02, 0.0], 8, 8, 4, 4, 8, 8)
    pool = seed_pool(
        create_pool(config),
        jnp.array([[0.1, -0.1, 2.0]], jnp.float32),
        jnp.array([[0.8, 0.2, 0.1]], jnp.float32),
        scale=0.2,
        opacity=0.5,
    )
    pool = pool.replace(
        log_scale=pool.log_scale.at[0].set(jnp.log(jnp.array([0.18, 0.12, 0.23]))),
        rotation=pool.rotation.at[0].set(jnp.array([0.95, 0.1, 0.2, -0.1])),
        sh=pool.sh.at[0, 1, 0].set(0.25),
    )
    ref_projection = project(pool, camera, config)
    cute_projection = project_cute(pool, camera, config)
    for name in ("mean", "depth", "conic", "radius", "color", "alpha", "visible"):
        np.testing.assert_allclose(
            getattr(cute_projection, name), getattr(ref_projection, name), rtol=1e-4, atol=1e-5
        )
    ref_table = build_visibility_table(ref_projection, camera, config)
    cute_table = build_visibility_table_cute(cute_projection, camera, config)
    for name in ("tile_depths", "tile_valid", "tile_count", "overflow"):
        np.testing.assert_allclose(getattr(cute_table, name), getattr(ref_table, name))
    np.testing.assert_array_equal(
        cute_table.tile_gaussian_ids[cute_table.tile_valid],
        ref_table.tile_gaussian_ids[ref_table.tile_valid],
    )

    render_forward, render_gradient = rasterize_cute, rasterize_cute_vjp
    if pipeline == "sorted":
        from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
        from jaxgs.kernels.sorted_rasterizer import rasterize_sorted_cute_vjp

        cute_table = build_sorted_visibility_table_cute(cute_projection, camera, config)
        render_forward = render_gradient = rasterize_sorted_cute_vjp

    def loss(xyz, log_scale, rotation, opacity, sh, gradient):
        p = pool.replace(xyz=xyz, log_scale=log_scale, rotation=rotation, opacity=opacity, sh=sh)
        projected = (
            project_cute_vjp(p, camera, config) if gradient else project_cute(p, camera, config)
        )
        result = (
            render_gradient(projected, cute_table, camera, config)
            if gradient
            else render_forward(projected, cute_table, camera, config)
        )
        weights = jnp.arange(result.rgb.size, dtype=jnp.float32).reshape(result.rgb.shape)
        return jnp.sum(result.rgb * weights) / result.rgb.size

    params = (pool.xyz, pool.log_scale, pool.rotation, pool.opacity, pool.sh)
    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*params, True)
    for arg, index in enumerate(((0, 0), (0, 1), (0, 2), (0, 0), (0, 1, 0))):
        delta = jnp.zeros_like(params[arg]).at[index].set(1e-3)
        plus, minus = list(params), list(params)
        plus[arg] += delta
        minus[arg] -= delta
        numeric = (loss(*plus, False) - loss(*minus, False)) / 2e-3
        np.testing.assert_allclose(
            grads[arg][index],
            numeric,
            rtol=0.04,
            atol=0.003,
            err_msg=f"parameter {arg}, index {index}",
        )


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_projection_pullback_zero_and_individual_cotangents():
    from jaxgs.kernels.projector import project_cute_vjp

    config = CapacityConfig(9, 1, 16, 16, 3)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 20, 20, 16, 16, 32, 32)
    pool = seed_pool(
        create_pool(config),
        jnp.tile(jnp.array([[0.1, 0.2, 2.0]]), (9, 1)),
        jnp.full((9, 3), 0.4),
        scale=jnp.tile(jnp.array([[0.1, 0.2, 0.3]]), (9, 1)),
        opacity=0.3,
    )
    pool = pool.replace(sh=pool.sh.at[:, 1:].set(0.02))

    def loss(projector, pool):
        value = projector(pool, camera, config)
        return (
            value.depth[1]
            + value.radius[2] * 0.01
            + value.mean[3].sum()
            + value.conic[4].sum()
            + value.color[5].sum()
            + value.alpha[6]
        )

    with jax.default_matmul_precision("highest"):
        expected = jax.grad(lambda p: loss(project, p), allow_int=True)(pool)
        actual = jax.grad(lambda p: loss(project_cute_vjp, p), allow_int=True)(pool)
    for name in ("xyz", "log_scale", "rotation", "opacity", "sh"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=2e-4, atol=2e-5
        )
        np.testing.assert_array_equal(np.asarray(getattr(actual, name))[[0, 7, 8]], 0)


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_cute_projection_pullback_matches_reference_all_fields():
    from jaxgs.kernels.projector import project_cute_vjp

    config = CapacityConfig(4, 2, 4, 4, 3)
    camera = Camera.from_colmap([0.97, 0.1, -0.08, 0.18], [0.02, -0.03, 0.01], 9, 8, 4, 4, 8, 8)
    keys = jax.random.split(jax.random.key(9), 7)
    pool = create_pool(config).replace(
        xyz=jax.random.uniform(keys[0], (4, 3), minval=-0.3, maxval=0.3)
        .at[:, 2]
        .set(jax.random.uniform(keys[1], (4,), minval=1.5, maxval=3.0)),
        log_scale=jnp.log(jax.random.uniform(keys[2], (4, 3), minval=0.08, maxval=0.3)),
        rotation=jax.random.normal(keys[3], (4, 4)),
        opacity=jax.random.normal(keys[4], (4, 1)),
        sh=jax.random.normal(keys[5], (4, 16, 3)) * 0.15,
        alive=jnp.array([True, True, False, True]),
    )
    projected = project(pool, camera, config)
    weights = jax.tree_util.tree_map(
        lambda value: (
            jax.random.normal(keys[6], value.shape)
            if jnp.issubdtype(value.dtype, jnp.floating)
            else value
        ),
        projected,
    )

    def loss(projector, xyz, log_scale, rotation, opacity, sh):
        current = pool.replace(
            xyz=xyz, log_scale=log_scale, rotation=rotation, opacity=opacity, sh=sh
        )
        value = projector(current, camera, config)
        return sum(
            jnp.sum(getattr(value, name) * getattr(weights, name))
            for name in ("mean", "depth", "conic", "radius", "color", "alpha")
        )

    params = (pool.xyz, pool.log_scale, pool.rotation, pool.opacity, pool.sh)
    reference = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(project, *params)
    actual = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(project_cute_vjp, *params)
    for name, expected, value in zip(
        ("xyz", "log_scale", "rotation", "opacity", "sh"), reference, actual, strict=True
    ):
        np.testing.assert_allclose(value, expected, rtol=3e-3, atol=3e-4, err_msg=name)


@pytest.mark.parametrize("degree", [0, 2])
@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_cute_projection_pullback_clipped_color_and_near_plane(degree):
    from jaxgs.kernels.projector import project_cute_vjp

    config = CapacityConfig(2, 2, 2, 4, degree)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 8, 8, 4, 4, 8, 8)
    pool = create_pool(config).replace(
        xyz=jnp.array([[0.005, -0.003, 0.1], [0.1, 0.2, 2.0]], jnp.float32),
        log_scale=jnp.log(jnp.array([[0.1, 0.2, 0.3], [0.2, 0.15, 0.1]])),
        rotation=jnp.array([[0.9, 0.2, -0.1, 0.15], [0.8, -0.1, 0.25, 0.2]]),
        opacity=jnp.array([[0.1], [-0.2]]),
        sh=jnp.full((2, config.sh_dim, 3), 0.1).at[0, 0, 0].set(-3.0),
    )
    assert float(project(pool, camera, config).color[0, 0]) == 0.0

    def loss(projector, xyz, log_scale, rotation, opacity, sh):
        current = pool.replace(
            xyz=xyz, log_scale=log_scale, rotation=rotation, opacity=opacity, sh=sh
        )
        value = projector(current, camera, config)
        return (
            jnp.sum(value.mean**2)
            + jnp.sum(value.conic**2)
            + jnp.sum(value.radius)
            + jnp.sum(value.depth)
            + jnp.sum(value.color)
            + jnp.sum(value.alpha)
        )

    params = (pool.xyz, pool.log_scale, pool.rotation, pool.opacity, pool.sh)
    expected = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(project, *params)
    actual = jax.grad(loss, argnums=(1, 2, 3, 4, 5))(project_cute_vjp, *params)
    for name, reference, value in zip(
        ("xyz", "log_scale", "rotation", "opacity", "sh"),
        expected,
        actual,
        strict=True,
    ):
        np.testing.assert_allclose(value, reference, rtol=3e-3, atol=3e-4, err_msg=name)


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_cute_training_with_symmetric_covariance_stays_finite():
    config = CapacityConfig(2, 2, 2, 4, 0)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 8, 8, 4, 4, 8, 8)
    pool = seed_pool(
        create_pool(config),
        jnp.array([[0.0, 0.0, 2.0]], jnp.float32),
        jnp.array([[1.0, 0.0, 0.0]], jnp.float32),
        scale=0.01,
        opacity=0.1,
    )
    state = create_adam_state(pool)
    losses = []
    for _ in range(2):
        pool, state, metrics = train_step(pool, state, camera, jnp.zeros((8, 8, 3)), config)
        losses.append(float(metrics["loss"]))
    assert losses[0] > 0 and losses[1] > 0
    assert np.all(np.isfinite(np.asarray(pool.xyz)))
    assert np.all(np.isfinite(np.asarray(pool.log_scale)))


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_cute_allocator_resets_reused_slot_with_padded_indices():
    config = CapacityConfig(3, 2, 2, 4, 0)
    pool = create_pool(config).replace(
        xyz=jnp.zeros((3, 3)).at[1, 2].set(2.0),
        alive=jnp.array([False, True, False]),
        free_mask=jnp.array([True, False, True]),
        n_active=jnp.array(1, jnp.int32),
    )
    state = create_adam_state(pool)
    state = state.replace(
        m=state.m.replace(xyz=state.m.xyz.at[0].set(1.0)), step=state.step.at[0].set(7)
    )
    next_pool, next_state, count = densify_step(
        pool,
        state,
        jnp.array([0.0, 1.0, 0.0]),
        jax.random.key(0),
        max_new=3,
        threshold=0.5,
        allocator="cute",
    )
    assert int(count) == 1
    assert bool(next_pool.alive[0])
    np.testing.assert_array_equal(next_state.m.xyz[0], 0)
    assert int(next_state.step[0]) == 0


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe test requires JAX CUDA and the cute extra",
)
def test_cute_pipeline_matches_reference_with_multiple_clusters():
    from jaxgs.kernels.binning import build_visibility_table_cute
    from jaxgs.kernels.projector import project_cute
    from jaxgs.kernels.rasterizer import rasterize_cute_vjp

    config = CapacityConfig(16, 4, 16, 4, 3)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 12, 11, 8, 8, 16, 16)
    keys = jax.random.split(jax.random.key(7), 5)
    alive = jnp.arange(16) < 12
    pool = create_pool(config).replace(
        xyz=jax.random.normal(keys[0], (16, 3))
        .at[:, 2]
        .set(jax.random.uniform(keys[1], (16,), minval=1.5, maxval=3.5)),
        log_scale=jnp.log(jax.random.uniform(keys[2], (16, 3), minval=0.05, maxval=0.25)),
        rotation=jax.random.normal(keys[3], (16, 4)),
        opacity=jax.random.normal(keys[4], (16, 1)),
        alive=alive,
        free_mask=~alive,
        n_active=jnp.array(12, jnp.int32),
    )
    reference = project(pool, camera, config)
    projected = project_cute(pool, camera, config)
    np.testing.assert_allclose(projected.mean, reference.mean, rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(projected.conic, reference.conic, rtol=2e-4, atol=1e-3)
    np.testing.assert_allclose(projected.radius, reference.radius, rtol=2e-4, atol=1e-3)
    reference_table = build_visibility_table(reference, camera, config)
    table = build_visibility_table_cute(projected, camera, config)
    np.testing.assert_array_equal(table.tile_count, reference_table.tile_count)
    np.testing.assert_array_equal(
        table.tile_gaussian_ids[table.tile_valid],
        reference_table.tile_gaussian_ids[reference_table.tile_valid],
    )
    actual = rasterize_cute_vjp(projected, table, camera, config)
    expected = rasterize_jax(reference, reference_table, camera, config)
    np.testing.assert_allclose(actual.rgb, expected.rgb, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(actual.alpha, expected.alpha, rtol=1e-3, atol=1e-4)

    def loss(p, renderer):
        image = renderer(p, table, camera, config)
        return jnp.sum(image.rgb**2) + 0.1 * jnp.sum(image.depth) + 0.2 * jnp.sum(image.alpha)

    cute_grads = jax.grad(lambda p: loss(p, rasterize_cute_vjp), allow_int=True)(projected)
    ref_grads = jax.grad(lambda p: loss(p, rasterize_jax), allow_int=True)(projected)
    for field in ("mean", "conic", "depth", "color", "alpha"):
        np.testing.assert_allclose(
            getattr(cute_grads, field), getattr(ref_grads, field), rtol=2e-3, atol=2e-3
        )
