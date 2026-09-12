import jax
import jax.numpy as jnp
import pytest

import jax_gs.geometry.functional as geometry
from jax_gs.geometry.kernels.quaternion_ops import (
    SLERP_SMALL_ANGLE_DOT_THRESHOLD,
)


def _normalized(values):
    values = jnp.asarray(values, dtype=jnp.float32)
    return values / jnp.linalg.norm(values, axis=-1, keepdims=True)


def _assert_same_rotation(actual, expected, atol=2.0e-5):
    same = jnp.max(jnp.abs(actual - expected), axis=-1) < atol
    opposite = jnp.max(jnp.abs(actual + expected), axis=-1) < atol
    assert bool(jnp.all(same | opposite))


def test_geometry_public_quaternion_surface():
    expected = {
        "quat_angular_distance",
        "quat_conjugate",
        "quat_from_axis_angle",
        "quat_identity",
        "quat_inverse",
        "quat_lerp",
        "quat_manifold_interp",
        "quat_multiply",
        "quat_normalize_safe",
        "quat_rotate_vector",
        "quat_slerp",
        "quat_to_matrix",
    }
    assert expected <= set(geometry.__all__)
    assert SLERP_SMALL_ANGLE_DOT_THRESHOLD == 0.9995


def test_normalize_safe_degenerate_rows_and_zero_gradient():
    quaternions = jnp.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1.0e-10] * 4, [1.0, 2.0, 3.0, 4.0]],
        dtype=jnp.float32,
    )
    actual = geometry.quat_normalize_safe(quaternions)
    identity = jnp.asarray([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    assert jnp.allclose(actual[:2], identity)
    assert jnp.allclose(jnp.linalg.norm(actual[2]), 1.0)

    gradient = jax.grad(lambda q: jnp.sum(geometry.quat_normalize_safe(q)))(
        jnp.zeros((3, 4), dtype=jnp.float32)
    )
    assert jnp.array_equal(gradient, jnp.zeros_like(gradient))


def test_conjugate_inverse_and_hamilton_product():
    q = _normalized([[1.0, -2.0, 3.0, 4.0]])
    conjugate = geometry.quat_conjugate(q)
    assert jnp.allclose(conjugate, -q.at[:, 3].set(-q[:, 3]))
    assert jnp.array_equal(geometry.quat_inverse(q), conjugate)
    product = geometry.quat_multiply(q, conjugate)
    assert jnp.allclose(product, geometry.quat_identity((1,)), atol=1.0e-6)


def test_multiply_composition_matches_rotation_matrices():
    q1 = _normalized([[0.3, -0.4, 0.1, 0.8], [0.1, 0.2, 0.3, -0.9]])
    q2 = _normalized([[0.2, 0.1, -0.5, 0.7], [-0.4, 0.2, 0.1, 0.6]])
    product_matrix = geometry.quat_to_matrix(geometry.quat_multiply(q1, q2))
    expected = jnp.einsum(
        "nij,njk->nik", geometry.quat_to_matrix(q1), geometry.quat_to_matrix(q2)
    )
    assert jnp.allclose(product_matrix, expected, atol=2.0e-6)


def test_rotate_vector_matches_matrix_and_preserves_leading_shape():
    q = _normalized(jnp.arange(1, 1 + 2 * 3 * 4, dtype=jnp.float32).reshape((2, 3, 4)))
    vectors = jnp.linspace(-1.0, 2.0, 2 * 3 * 3).reshape((2, 3, 3))
    actual = geometry.quat_rotate_vector(q, vectors)
    expected = jnp.einsum("...ij,...j->...i", geometry.quat_to_matrix(q), vectors)
    assert actual.shape == (2, 3, 3)
    assert jnp.allclose(actual, expected, atol=2.0e-6)


def test_to_matrix_safely_maps_zero_quaternion_to_identity():
    actual = geometry.quat_to_matrix(jnp.zeros((2, 4), dtype=jnp.float32))
    expected = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (2, 3, 3))
    assert jnp.array_equal(actual, expected)


def test_axis_angle_uses_xyzw_and_does_not_normalize_axis():
    axis = jnp.asarray([[0.0, 0.0, 1.0]], dtype=jnp.float32)
    angle = jnp.asarray([jnp.pi / 2], dtype=jnp.float32)
    actual = geometry.quat_from_axis_angle(axis, angle)
    expected = jnp.asarray(
        [[0.0, 0.0, jnp.sqrt(0.5), jnp.sqrt(0.5)]], dtype=jnp.float32
    )
    assert jnp.allclose(actual, expected, atol=1.0e-6)

    non_unit = geometry.quat_from_axis_angle(
        jnp.asarray([[2.0, 0.0, 0.0]], dtype=jnp.float32),
        jnp.asarray([jnp.pi], dtype=jnp.float32),
    )
    assert jnp.allclose(non_unit[0, 0], 2.0, atol=1.0e-6)


def test_lerp_uses_shortest_hemisphere_and_normalizes():
    q1 = geometry.quat_identity((2,))
    q2 = jnp.asarray(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, -1.0, -0.01]],
        dtype=jnp.float32,
    )
    q2 = geometry.quat_normalize_safe(q2)
    actual = geometry.quat_lerp(q1, q2, 0.5)
    assert jnp.allclose(jnp.linalg.norm(actual, axis=-1), 1.0, atol=1.0e-6)
    assert jnp.all(jnp.sum(q1 * actual, axis=-1) >= 0)


def test_slerp_spherical_and_small_angle_paths():
    q1 = geometry.quat_identity((2,))
    q2 = jnp.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0e-3, jnp.sqrt(1.0 - 1.0e-6)],
        ],
        dtype=jnp.float32,
    )
    actual = geometry.quat_slerp(q1, q2, jnp.asarray([0.5, 0.25]))
    expected_first = jnp.asarray(
        [0.0, 0.0, jnp.sqrt(0.5), jnp.sqrt(0.5)], dtype=jnp.float32
    )
    assert jnp.allclose(actual[0], expected_first, atol=2.0e-6)
    assert jnp.allclose(jnp.linalg.norm(actual, axis=-1), 1.0, atol=2.0e-6)

    raw = 0.75 * q1[1] + 0.25 * q2[1]
    expected_small = raw / jnp.linalg.norm(raw)
    assert jnp.allclose(actual[1], expected_small, atol=2.0e-6)


def test_slerp_accepts_flattened_batch_and_singleton_t():
    q1 = geometry.quat_identity((2, 3))
    angles = jnp.linspace(0.1, 1.2, 6, dtype=jnp.float32)
    axes = jnp.broadcast_to(jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32), (6, 3))
    q2 = geometry.quat_from_axis_angle(axes, angles).reshape((2, 3, 4))
    per_row = geometry.quat_slerp(q1, q2, jnp.full((2, 3), 0.4))
    singleton = geometry.quat_slerp(q1, q2, jnp.asarray([0.4]))
    scalar = geometry.quat_slerp(q1, q2, 0.4)
    assert jnp.allclose(per_row, singleton, atol=2.0e-6)
    assert jnp.allclose(singleton, scalar, atol=2.0e-6)


def test_slerp_is_jittable_and_differentiable_in_quaternions_and_time():
    q1 = _normalized([[0.2, 0.3, -0.1, 0.8], [0.1, -0.4, 0.3, 0.7]])
    q2 = _normalized([[-0.3, 0.2, 0.4, 0.6], [0.5, 0.1, -0.2, 0.7]])
    t = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    eager = geometry.quat_slerp(q1, q2, t)
    compiled = jax.jit(geometry.quat_slerp)(q1, q2, t)
    assert jnp.allclose(compiled, eager, atol=1.0e-6)

    gradients = jax.grad(
        lambda a, b, time: jnp.sum(geometry.quat_slerp(a, b, time)),
        argnums=(0, 1, 2),
    )(q1, q2, t)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients)
    assert bool(jnp.any(jnp.abs(gradients[2]) > 0))


def test_manifold_interpolation_matches_unit_quaternion_geodesic():
    q1 = geometry.quat_identity((3,))
    axes = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    q2 = geometry.quat_from_axis_angle(
        axes, jnp.asarray([0.4, 1.2, 2.4], dtype=jnp.float32)
    )
    t = jnp.asarray([0.2, 0.5, 0.8], dtype=jnp.float32)
    manifold = geometry.quat_manifold_interp(q1, q2, t)
    slerp = geometry.quat_slerp(q1, q2, t)
    _assert_same_rotation(manifold, slerp, atol=3.0e-5)
    _assert_same_rotation(geometry.quat_manifold_interp(q1, q2, 0.0), q1)
    _assert_same_rotation(geometry.quat_manifold_interp(q1, q2, 1.0), q2)


def test_manifold_interpolation_is_jittable_and_differentiable():
    q1 = _normalized([[0.2, 0.4, 0.1, 0.8], [-0.2, 0.1, 0.5, 0.7]])
    q2 = _normalized([[0.5, -0.2, 0.3, 0.6], [0.4, 0.3, -0.1, 0.8]])
    t = jnp.asarray([0.3, 0.7], dtype=jnp.float32)
    expected = geometry.quat_manifold_interp(q1, q2, t)
    actual = jax.jit(geometry.quat_manifold_interp)(q1, q2, t)
    assert jnp.allclose(actual, expected, atol=2.0e-6)
    gradients = jax.grad(
        lambda a, b, time: jnp.sum(geometry.quat_manifold_interp(a, b, time)),
        argnums=(0, 1, 2),
    )(q1, q2, t)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients)


def test_angular_distance_is_sign_invariant_and_bounded():
    identity = geometry.quat_identity((3,))
    axes = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    angles = jnp.asarray([0.2, 1.0, jnp.pi], dtype=jnp.float32)
    rotations = geometry.quat_from_axis_angle(axes, angles)
    actual = geometry.quat_angular_distance(identity, rotations)
    opposite = geometry.quat_angular_distance(identity, -rotations)
    assert jnp.allclose(actual, angles, atol=5.0e-4)
    assert jnp.allclose(opposite, actual, atol=1.0e-6)
    assert bool(jnp.all((actual >= 0) & (actual <= jnp.pi)))


def test_identity_supports_shape_dtype_and_explicit_jax_device():
    identity = geometry.quat_identity(
        (2, 3), dtype=jnp.float32, device=jax.devices()[0]
    )
    assert identity.shape == (2, 3, 4)
    assert identity.dtype == jnp.float32
    assert jnp.array_equal(identity[..., :3], jnp.zeros((2, 3, 3)))
    assert jnp.array_equal(identity[..., 3], jnp.ones((2, 3)))


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: geometry.quat_conjugate([0.0, 0.0, 0.0, 1.0]), TypeError),
        (lambda: geometry.quat_conjugate(jnp.ones((2, 3))), ValueError),
        (lambda: geometry.quat_conjugate(jnp.ones((2, 4), dtype=jnp.int32)), TypeError),
        (
            lambda: geometry.quat_multiply(
                jnp.ones((1, 4), dtype=jnp.float32),
                jnp.ones((2, 4), dtype=jnp.float32),
            ),
            ValueError,
        ),
        (
            lambda: geometry.quat_rotate_vector(
                jnp.ones((1, 4), dtype=jnp.float32),
                jnp.ones((2, 3), dtype=jnp.float32),
            ),
            ValueError,
        ),
        (
            lambda: geometry.quat_slerp(
                geometry.quat_identity((2,)),
                geometry.quat_identity((2,)),
                jnp.ones((3,), dtype=jnp.float32),
            ),
            ValueError,
        ),
        (
            lambda: geometry.quat_slerp(
                geometry.quat_identity((2,)),
                geometry.quat_identity((2,)),
                jnp.ones((2,), dtype=jnp.int32),
            ),
            TypeError,
        ),
        (
            lambda: geometry.quat_lerp(
                geometry.quat_identity((1,)),
                geometry.quat_identity((1,)),
                jnp.asarray(0.5),
            ),
            TypeError,
        ),
        (
            lambda: geometry.quat_from_axis_angle(
                jnp.ones((2, 3), dtype=jnp.float32),
                jnp.asarray(0.5, dtype=jnp.float32),
            ),
            ValueError,
        ),
        (lambda: geometry.quat_identity(dtype=jnp.int32), TypeError),
    ],
)
def test_quaternion_validation(call, error):
    with pytest.raises(error):
        call()
