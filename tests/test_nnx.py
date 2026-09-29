import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxgs import Camera, CapacityConfig, GaussianModel, create_pool, seed_pool


def test_model_parameters_pool_views_and_fixed_shape_updates():
    pool = seed_pool(
        create_pool(CapacityConfig(4, 1, 4, 4, 1)), jnp.array([[0.0, 0.0, 2.0]]), jnp.ones((1, 3))
    )
    model = GaussianModel(pool)
    assert set(nnx.state(model, nnx.Param)) == {"xyz", "log_scale", "rotation", "opacity", "sh"}
    for name in pool.__dataclass_fields__:
        assert getattr(model.as_pool(), name) is getattr(pool, name)
    graph, state = nnx.split(model)
    restored = nnx.merge(graph, state)
    for actual, expected in zip(
        jax.tree.leaves(restored.as_pool()), jax.tree.leaves(pool), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)

    @nnx.jit(graph=False)
    def activate(model, slot):
        pool = model.as_pool()
        model.update_from_pool(
            pool.replace(
                alive=pool.alive.at[slot].set(True),
                free_mask=pool.free_mask.at[slot].set(False),
                n_active=pool.n_active + 1,
            )
        )

    variables = {name: getattr(model, name) for name in pool.__dataclass_fields__}
    for slot in (1, 2, 3):
        activate(model, jnp.array(slot, jnp.int32))
        assert activate.jitted_fn._cache_size() == 1
        assert model.xyz.get_value().shape == (4, 3)
    assert int(model.n_active.get_value()) == 4
    np.testing.assert_array_equal(model.free_mask.get_value(), ~model.alive.get_value())
    for name, variable in variables.items():
        assert getattr(model, name) is variable


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="CuTe protocol requires JAX CUDA",
)
@pytest.mark.parametrize("degree,collect", [(0, False), (3, True)])
@pytest.mark.parametrize("optimizer", ["cute", "optax"])
def test_nnx_donated_training_matches_array_step(degree, collect, optimizer):
    from jaxgs.scene.cluster import world_cluster_bounds
    from jaxgs.training.optimizer import create_adam_state
    from jaxgs.training.trainer import array_train_step, train_step

    config = CapacityConfig(128, 128, 128, 16, 3, 512, tile_height=8)
    camera = Camera.from_colmap([1, 0, 0, 0], [0, 0, 0], 24, 24, 16, 8, 32, 16)
    pool = seed_pool(
        create_pool(config),
        jnp.array([[0.1, 0.0, 2.0], [0.2, 0.1, 3.0]]),
        jnp.array([[0.4, 0.3, 0.2], [0.2, 0.5, 0.6]]),
        scale=jnp.array([[0.15, 0.2, 0.3], [0.18, 0.25, 0.22]]),
        opacity=0.5,
    )
    # Anisotropic splats exercise rotation gradients without Adam amplifying
    # rounding noise around the zero derivative of a spherical Gaussian.
    pool = pool.replace(
        sh=pool.sh.at[:2, 1:].set(0.01),
        rotation=pool.rotation.at[:2].set(
            jnp.array([[0.9, 0.1, 0.2, 0.05], [0.8, -0.2, 0.1, 0.15]])
        ),
    )
    state = create_adam_state(pool)
    stats = jnp.zeros((128, 4), jnp.float32)
    bounds = world_cluster_bounds(pool, config.cluster_size)
    target = jnp.full((16, 32, 3), 32, jnp.uint8)
    model = GaussianModel(jax.tree.map(lambda x: x.copy(), pool))
    nnx_state, nnx_stats = jax.tree.map(lambda x: x.copy(), (state, stats))
    overflow, peak = jnp.array(False), jnp.array(0, jnp.int32)
    old_xyz = model.xyz.get_value()
    fields = ("xyz", "log_scale", "rotation", "opacity", "sh")
    buffers = [getattr(model.as_pool(), name) for name in fields]
    buffers += [getattr(moment, name) for moment in (nnx_state.m, nnx_state.v) for name in fields]
    pointers = [value.unsafe_buffer_pointer() for value in buffers]
    for step in range(2):
        pool, state, stats, metrics = array_train_step(
            pool,
            state,
            stats,
            bounds,
            camera,
            target,
            jnp.array(step),
            jnp.array(1.0),
            config,
            degree,
            collect,
            optimizer=optimizer,
        )
        nnx_state, nnx_stats, loss, overflow, peak = train_step(
            nnx_state,
            nnx_stats,
            model,
            bounds,
            camera,
            target,
            jnp.array(step),
            jnp.array(1.0),
            config,
            degree,
            collect,
            10000,
            overflow,
            peak,
            optimizer,
        )
        actual = (model.as_pool(), nnx_state, nnx_stats, loss)
        expected = (pool, state, stats, metrics["loss"])
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
        assert not bool(overflow)
        buffers = [getattr(model.as_pool(), name) for name in fields]
        buffers += [
            getattr(moment, name) for moment in (nnx_state.m, nnx_state.v) for name in fields
        ]
        assert [value.unsafe_buffer_pointer() for value in buffers] == pointers
        assert int(peak) >= int(metrics["pairs"])
        if step == 0:
            cache_size = train_step.jitted_fn._cache_size()
        else:
            assert train_step.jitted_fn._cache_size() == cache_size
    assert old_xyz.is_deleted()
