import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from jax_gs.training.pose import CameraOptModule, rotation_6d_to_matrix


def _camera_to_world(
    rotation: jax.Array | None = None,
    translation: jax.Array | None = None,
) -> jax.Array:
    matrix = jnp.eye(4, dtype=jnp.float32)
    if rotation is not None:
        matrix = matrix.at[:3, :3].set(rotation)
    if translation is not None:
        matrix = matrix.at[:3, 3].set(translation)
    return matrix


def test_rotation_6d_to_matrix_matches_upstream_row_convention():
    representations = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 1.0, 3.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    rotations = rotation_6d_to_matrix(representations)

    expected = jnp.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(rotations, expected, atol=1e-6)
    np.testing.assert_allclose(
        rotations @ jnp.swapaxes(rotations, -1, -2),
        jnp.broadcast_to(jnp.eye(3), rotations.shape),
        atol=1e-6,
    )
    np.testing.assert_allclose(jnp.linalg.det(rotations), 1.0, atol=1e-6)


def test_camera_opt_zero_init_is_identity_and_forward_is_nnx_jittable():
    module = CameraOptModule(3, rngs=nnx.Rngs(7))
    assert bool(jnp.any(module.embeds.embedding[...] != 0.0))
    module.zero_init()
    poses = jnp.stack(
        (
            _camera_to_world(translation=jnp.asarray([1.0, 2.0, 3.0])),
            _camera_to_world(translation=jnp.asarray([-1.0, 0.5, 4.0])),
        )
    )
    image_ids = jnp.asarray([0, 2], dtype=jnp.int32)

    eager = module(poses, image_ids)
    forwarded = module.forward(poses, image_ids)
    compiled = nnx.jit(
        lambda current, current_poses, current_ids: current(current_poses, current_ids)
    )(module, poses, image_ids)

    np.testing.assert_array_equal(module.embeds.embedding[...], 0.0)
    np.testing.assert_array_equal(module.identity[...], [1, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(eager, poses, atol=1e-6)
    np.testing.assert_allclose(forwarded, poses, atol=1e-6)
    np.testing.assert_allclose(compiled, poses, atol=1e-6)


def test_camera_opt_right_multiplies_local_pose_delta():
    module = CameraOptModule(2, rngs=nnx.Rngs(8))
    module.zero_init()
    base_rotation = jnp.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    base = _camera_to_world(base_rotation, jnp.asarray([10.0, 20.0, 30.0]))

    # A local +x translation becomes world +y under the base rotation.
    module.embeds.embedding[1, :3] = jnp.asarray([1.0, 0.0, 0.0])
    translated = module(base[None], jnp.asarray([1], dtype=jnp.int32))[0]
    np.testing.assert_allclose(
        translated[:3, 3], jnp.asarray([10.0, 21.0, 30.0]), atol=1e-6
    )

    desired_rotation_6d = jnp.asarray(
        [0.0, 1.0, 0.0, -1.0, 0.0, 0.0], dtype=jnp.float32
    )
    module.embeds.embedding[0, 3:] = desired_rotation_6d - module.identity[...]
    delta = _camera_to_world(rotation_6d_to_matrix(desired_rotation_6d))
    rotated = module(base[None], jnp.asarray([0], dtype=jnp.int32))[0]
    np.testing.assert_allclose(rotated, base @ delta, atol=1e-6)


def test_camera_opt_supports_arbitrary_batch_dims_and_validates_shapes():
    module = CameraOptModule(4, rngs=nnx.Rngs(9))
    module.zero_init()
    poses = jnp.broadcast_to(jnp.eye(4), (2, 2, 4, 4))
    image_ids = jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32)
    module.embeds.embedding[:, :3] = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)

    adjusted = module(poses, image_ids)

    assert adjusted.shape == poses.shape
    np.testing.assert_array_equal(
        adjusted[..., :3, 3], module.embeds.embedding[:, :3].reshape(2, 2, 3)
    )
    with pytest.raises(ValueError, match="batch dimensions"):
        module(poses, image_ids.reshape(-1))
    with pytest.raises(ValueError, match="4, 4"):
        module(jnp.zeros((2, 2, 3, 4)), image_ids)


def test_camera_opt_random_and_zero_initializers_are_reproducible():
    explicit_a = CameraOptModule(3, rngs=nnx.Rngs(1))
    explicit_b = CameraOptModule(3, rngs=nnx.Rngs(2))
    explicit_a.random_init(0.02, rngs=nnx.Rngs(91))
    explicit_b.random_init(0.02, rngs=nnx.Rngs(91))

    np.testing.assert_array_equal(
        explicit_a.embeds.embedding[...], explicit_b.embeds.embedding[...]
    )
    assert bool(jnp.any(explicit_a.embeds.embedding[...] != 0.0))

    default_a = CameraOptModule(2)
    default_b = CameraOptModule(2)
    np.testing.assert_array_equal(
        default_a.embeds.embedding[...], default_b.embeds.embedding[...]
    )
    first_default_draw = np.asarray(default_a.embeds.embedding[...]).copy()
    default_a.random_init(0.01)
    second_default_draw = np.asarray(default_a.embeds.embedding[...]).copy()
    default_a.random_init(0.01)
    third_default_draw = np.asarray(default_a.embeds.embedding[...]).copy()
    assert not np.array_equal(first_default_draw, second_default_draw)
    assert not np.array_equal(second_default_draw, third_default_draw)

    default_b.random_init(0.01)
    np.testing.assert_array_equal(second_default_draw, default_b.embeds.embedding[...])

    explicit_a.zero_init()
    np.testing.assert_array_equal(explicit_a.embeds.embedding[...], 0.0)
    with pytest.raises(ValueError, match="std"):
        explicit_a.random_init(-0.1)


def test_camera_opt_has_gradients_and_trains_with_jitted_nnx_optimizer():
    module = CameraOptModule(2, rngs=nnx.Rngs(10))
    module.zero_init()
    optimizer = nnx.Optimizer(module, optax.adam(0.05), wrt=nnx.Param)
    poses = jnp.broadcast_to(jnp.eye(4), (2, 4, 4))
    image_ids = jnp.asarray([0, 1], dtype=jnp.int32)
    target_translations = jnp.asarray(
        [[0.5, -0.25, 0.1], [-0.4, 0.2, 0.3]], dtype=jnp.float32
    )
    target_rotations = rotation_6d_to_matrix(
        jnp.asarray(
            [
                [1.0, 0.25, 0.0, -0.2, 1.0, 0.1],
                [1.0, -0.2, 0.1, 0.15, 1.0, -0.1],
            ],
            dtype=jnp.float32,
        )
    )

    def objective(current: CameraOptModule) -> jax.Array:
        adjusted = current(poses, image_ids)
        translation_loss = jnp.mean(
            jnp.square(adjusted[..., :3, 3] - target_translations)
        )
        rotation_loss = jnp.mean(jnp.square(adjusted[..., :3, :3] - target_rotations))
        return translation_loss + rotation_loss

    gradients = nnx.grad(objective)(module)
    embedding_gradient = gradients.embeds.embedding[...]
    assert not hasattr(gradients, "identity")
    assert bool(jnp.all(jnp.isfinite(embedding_gradient)))
    assert bool(jnp.any(embedding_gradient[:, :3] != 0.0))
    assert bool(jnp.any(embedding_gradient[:, 3:] != 0.0))

    @nnx.jit
    def train_step(
        current: CameraOptModule, current_optimizer: nnx.Optimizer
    ) -> jax.Array:
        loss, current_gradients = nnx.value_and_grad(objective)(current)
        current_optimizer.update(current, current_gradients)
        return loss

    initial_loss = objective(module)
    identity_before = np.asarray(module.identity[...]).copy()
    for _ in range(60):
        train_step(module, optimizer)
    final_loss = objective(module)

    assert float(final_loss) < float(initial_loss) * 0.01
    np.testing.assert_array_equal(module.identity[...], identity_before)
