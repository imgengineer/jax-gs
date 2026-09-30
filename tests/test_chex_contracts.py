import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import Camera, CapacityConfig, GaussianModel, create_pool, seed_pool
from jaxgs.training.optimizer import create_adam_state, optax_adam_update


@pytest.mark.parametrize(
    "field,shape,dtype",
    [
        ("xyz", (2, 2), jnp.float32),
        ("log_scale", (3, 3), jnp.float32),
        ("rotation", (2, 3), jnp.float32),
        ("opacity", (2,), jnp.float32),
        ("sh", (2, 1, 4), jnp.float32),
        ("alive", (2, 1), jnp.bool_),
        ("free_mask", (2,), jnp.float32),
        ("n_active", (1,), jnp.int32),
    ],
)
def test_model_rejects_inconsistent_pool_layout(field, shape, dtype):
    pool = create_pool(CapacityConfig(2, sh_degree=0))
    pool = pool.replace(**{field: jnp.zeros(shape, dtype)})
    with pytest.raises(AssertionError, match="Chex"):
        GaussianModel(pool)


@pytest.mark.parametrize("qvec,tvec", [([1, 0, 0], [0, 0, 0]), ([1, 0, 0, 0], [0, 0])])
def test_camera_rejects_malformed_extrinsics(qvec, tvec):
    with pytest.raises(AssertionError, match="assert_shape"):
        Camera.from_colmap(qvec, tvec, 4, 4, 2, 2, 4, 4)


def test_camera_numeric_input_types_share_one_compilation():
    traces = []

    @jax.jit
    def camera_center_and_intrinsics(camera):
        traces.append(True)
        return camera.center, jnp.stack((camera.fx, camera.fy, camera.cx, camera.cy))

    for scalar_type in (int, float, np.float32, np.float64):
        camera = Camera.from_colmap(
            [1, 0, 0, 0], [0, 0, -2], *(scalar_type(x) for x in (8, 8, 4, 4)), 8, 8
        )
        for value in (camera.fx, camera.fy, camera.cx, camera.cy):
            assert value.dtype == jnp.float32 and not value.weak_type
        center, intrinsics = camera_center_and_intrinsics(camera)
        np.testing.assert_array_equal(center, [0, 0, 2])
        np.testing.assert_array_equal(intrinsics, [8, 8, 4, 4])
    assert traces == [True]


@pytest.mark.parametrize("invalid", ["mask", "moment_shape", "gradient_dtype", "step_rank"])
def test_optimizer_rejects_incompatible_state_at_trace_time(invalid):
    pool = seed_pool(
        create_pool(CapacityConfig(2, sh_degree=0)), jnp.ones((1, 3)), jnp.full((1, 3), 0.5)
    )
    state = create_adam_state(pool)
    fields = ("xyz", "log_scale", "rotation", "opacity", "sh")
    gradients = tuple(jnp.zeros_like(getattr(pool, name)) for name in fields)
    visible = pool.alive
    if invalid == "mask":
        visible = visible[:, None]
    elif invalid == "moment_shape":
        state = state.replace(m=state.m.replace(xyz=jnp.zeros((1, 3))))
    elif invalid == "gradient_dtype":
        gradients = (gradients[0].astype(jnp.float16), *gradients[1:])
    else:
        state = state.replace(step=jnp.array(0, jnp.int32))
    update = jax.jit(lambda p, s, g, v: optax_adam_update(p, s, g, v, 0, 1.0))
    with pytest.raises(AssertionError, match="Chex"):
        update.lower(pool, state, gradients, visible)


def test_static_checks_do_not_add_device_work_or_retrace(monkeypatch):
    pool = seed_pool(
        create_pool(CapacityConfig(2, sh_degree=0)), jnp.ones((1, 3)), jnp.full((1, 3), 0.5)
    )
    state = create_adam_state(pool)
    gradients = tuple(
        jnp.ones_like(getattr(pool, name))
        for name in ("xyz", "log_scale", "rotation", "opacity", "sh")
    )
    calls = []
    check = chex.assert_trees_all_equal_shapes_and_dtypes

    def checked(*args):
        calls.append(True)
        return check(*args)

    monkeypatch.setattr(chex, "assert_trees_all_equal_shapes_and_dtypes", checked)
    update = jax.jit(lambda p, s, g, v, t: optax_adam_update(p, s, g, v, t, 1.0))
    args = (pool, state, gradients, pool.alive, jnp.array(0))
    lowered = update.lower(*args)
    compiled = lowered.compile()
    for step in range(3):
        updated, next_state = compiled(*args[:-1], jnp.array(step))
        chex.assert_tree_all_finite((updated, next_state))
    assert calls == [True]
    assert "callback" not in lowered.as_text().lower()
