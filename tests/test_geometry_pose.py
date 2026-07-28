import jax
import jax.numpy as jnp
import pytest

import jax_gs.geometry.functional as geometry


def _z_rotation(angle):
    axis = jnp.asarray([[0.0, 0.0, 1.0]], dtype=jnp.float32)
    return geometry.quat_from_axis_angle(
        axis, jnp.asarray([angle], dtype=jnp.float32)
    )[0]


def _normalized(values):
    values = jnp.asarray(values, dtype=jnp.float32)
    return values / jnp.linalg.norm(values, axis=-1, keepdims=True)


def _assert_same_rotation(actual, expected, atol=2.0e-5):
    same = jnp.max(jnp.abs(actual - expected), axis=-1) < atol
    opposite = jnp.max(jnp.abs(actual + expected), axis=-1) < atol
    assert bool(jnp.all(same | opposite))


def test_geometry_public_pose_surface():
    expected = {
        "frame_transform_poses_tquat",
        "se3_interpolate_tracks",
        "se3pose_compose",
        "se3pose_from_matrix",
        "se3pose_inverse_transform_direction",
        "se3pose_inverse_transform_point",
        "se3pose_to_inverse_matrix",
        "se3pose_to_matrix",
        "se3pose_transform_direction",
        "se3pose_transform_point",
        "trajectory_get_rotation_2poses",
        "trajectory_transform_point_1pose",
        "trajectory_transform_point_2poses",
    }
    assert expected <= set(geometry.__all__)


def test_se3_point_and_direction_forward_inverse():
    translation = jnp.asarray([[2.0, -1.0, 0.5]], dtype=jnp.float32)
    rotation = _z_rotation(jnp.pi / 2)[None]
    point = jnp.asarray([[1.0, 0.0, 2.0]], dtype=jnp.float32)
    direction = jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float32)

    transformed_point = geometry.se3pose_transform_point(
        translation, rotation, point
    )
    transformed_direction = geometry.se3pose_transform_direction(
        translation, rotation, direction
    )
    assert jnp.allclose(
        transformed_point, jnp.asarray([[2.0, 0.0, 2.5]]), atol=1.0e-6
    )
    assert jnp.allclose(
        transformed_direction, jnp.asarray([[0.0, 1.0, 0.0]]), atol=1.0e-6
    )
    assert jnp.allclose(
        geometry.se3pose_inverse_transform_point(
            translation, rotation, transformed_point
        ),
        point,
        atol=1.0e-6,
    )
    assert jnp.allclose(
        geometry.se3pose_inverse_transform_direction(
            translation, rotation, transformed_direction
        ),
        direction,
        atol=1.0e-6,
    )


def test_se3_transform_is_jittable_and_differentiable():
    translation = jnp.asarray(
        [[0.2, -0.1, 0.3], [1.0, 2.0, -1.0]], dtype=jnp.float32
    )
    rotation = _normalized(
        [[0.1, 0.2, 0.3, 0.8], [-0.2, 0.4, 0.1, 0.7]]
    )
    point = jnp.asarray(
        [[1.0, -2.0, 0.5], [0.1, 0.2, 0.3]], dtype=jnp.float32
    )
    expected = geometry.se3pose_transform_point(translation, rotation, point)
    actual = jax.jit(geometry.se3pose_transform_point)(
        translation, rotation, point
    )
    assert jnp.allclose(actual, expected, atol=1.0e-6)
    gradients = jax.grad(
        lambda t, q, p: jnp.sum(geometry.se3pose_transform_point(t, q, p)),
        argnums=(0, 1, 2),
    )(translation, rotation, point)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients)
    assert jnp.array_equal(gradients[0], jnp.ones_like(translation))


def test_pose_matrix_roundtrip_and_flattened_input():
    translation = jnp.asarray(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    rotation = _normalized(
        [[0.1, 0.2, 0.3, 0.9], [0.7, -0.1, 0.2, 0.4], [-0.2, 0.8, 0.1, 0.3]]
    )
    matrices = geometry.se3pose_to_matrix(translation, rotation)
    actual_translation, actual_rotation = geometry.se3pose_from_matrix(matrices)
    flat_translation, flat_rotation = geometry.se3pose_from_matrix(
        matrices.reshape((3, 16))
    )
    assert matrices.shape == (3, 4, 4)
    assert jnp.allclose(actual_translation, translation, atol=1.0e-6)
    assert jnp.array_equal(flat_translation, actual_translation)
    _assert_same_rotation(actual_rotation, rotation)
    _assert_same_rotation(flat_rotation, rotation)


def test_pose_from_matrix_covers_all_shepperd_branches():
    axes = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    half_turns = geometry.quat_from_axis_angle(
        axes, jnp.full((3,), jnp.pi, dtype=jnp.float32)
    )
    rotations = jnp.concatenate((geometry.quat_identity((1,)), half_turns), axis=0)
    translations = jnp.zeros((4, 3), dtype=jnp.float32)
    _, recovered = geometry.se3pose_from_matrix(
        geometry.se3pose_to_matrix(translations, rotations)
    )
    _assert_same_rotation(recovered, rotations, atol=3.0e-5)


def test_pose_from_matrix_is_jittable_with_finite_gradient():
    translation = jnp.asarray([[0.2, -0.1, 0.4]], dtype=jnp.float32)
    rotation = _normalized([[0.2, -0.3, 0.1, 0.8]])
    matrix = geometry.se3pose_to_matrix(translation, rotation)
    eager = geometry.se3pose_from_matrix(matrix)
    compiled = jax.jit(geometry.se3pose_from_matrix)(matrix)
    assert jnp.allclose(compiled[0], eager[0])
    _assert_same_rotation(compiled[1], eager[1])
    gradient = jax.grad(
        lambda value: sum(
            jnp.sum(part) for part in geometry.se3pose_from_matrix(value)
        )
    )(matrix)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_inverse_matrix_xyzw_and_wxyz_formats():
    translation = jnp.asarray([[1.0, 2.0, -3.0]], dtype=jnp.float32)
    xyzw = _z_rotation(0.7)[None]
    wxyz = jnp.concatenate((xyzw[:, 3:], xyzw[:, :3]), axis=-1)
    forward = geometry.se3pose_to_matrix(translation, xyzw)
    inverse_xyzw = geometry.se3pose_to_inverse_matrix(translation, xyzw)
    inverse_wxyz = geometry.se3pose_to_inverse_matrix(
        translation, wxyz, wxyz_format=True
    )
    identity = jnp.eye(4, dtype=jnp.float32)[None]
    assert jnp.allclose(inverse_xyzw, inverse_wxyz, atol=1.0e-6)
    assert jnp.allclose(
        jnp.einsum("nij,njk->nik", inverse_xyzw, forward),
        identity,
        atol=2.0e-6,
    )


def test_pose_composition_matches_sequential_application():
    parent_translation = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=jnp.float32
    )
    parent_rotation = jnp.stack((_z_rotation(0.5), _z_rotation(-0.3)))
    child_translation = jnp.asarray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32
    )
    child_rotation = jnp.stack((_z_rotation(0.2), _z_rotation(0.8)))
    point = jnp.asarray(
        [[0.2, -0.4, 1.0], [1.0, 2.0, 3.0]], dtype=jnp.float32
    )
    composed = geometry.se3pose_compose(
        parent_translation,
        parent_rotation,
        child_translation,
        child_rotation,
    )
    sequential = geometry.se3pose_transform_point(
        parent_translation,
        parent_rotation,
        geometry.se3pose_transform_point(
            child_translation, child_rotation, point
        ),
    )
    direct = geometry.se3pose_transform_point(*composed, point)
    assert jnp.allclose(direct, sequential, atol=2.0e-6)


def test_interpolate_tracks_variable_counts_and_per_track_queries():
    q0 = _z_rotation(0.0)
    q90 = _z_rotation(jnp.pi / 2)
    q180 = _z_rotation(jnp.pi)
    translations = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 6.0, 7.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    rotations = jnp.stack((q0, q90, q180, q90, q0, q180))
    times = jnp.asarray([0.0, 1.0, 2.0, 4.0, 0.0, 10.0], dtype=jnp.float32)
    offsets = jnp.asarray([0, 3, 4], dtype=jnp.int32)
    counts = jnp.asarray([3, 1, 2], dtype=jnp.int32)
    query = jnp.asarray([0.5, 123.0, -5.0], dtype=jnp.float32)
    out_translation, out_rotation = geometry.se3_interpolate_tracks(
        translations, rotations, times, offsets, counts, query
    )
    assert jnp.allclose(
        out_translation,
        jnp.asarray([[1.0, 0.0, 0.0], [5.0, 6.0, 7.0], [10.0, 0.0, 0.0]]),
        atol=1.0e-6,
    )
    expected_first = geometry.quat_slerp(q0[None], q90[None], 0.5)[0]
    _assert_same_rotation(out_rotation[0], expected_first)
    _assert_same_rotation(out_rotation[1], q90)
    _assert_same_rotation(out_rotation[2], q0)


def test_interpolate_tracks_duplicate_timestamps_follow_lower_bound_semantics():
    identity = geometry.quat_identity((4,))
    translations = jnp.asarray(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [31.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    times = jnp.asarray([0.0, 1.0, 1.0, 3.0], dtype=jnp.float32)
    offsets = jnp.asarray([0, 0], dtype=jnp.int32)
    counts = jnp.asarray([4, 4], dtype=jnp.int32)
    query = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    actual, _ = geometry.se3_interpolate_tracks(
        translations, identity, times, offsets, counts, query
    )
    assert jnp.allclose(
        actual[:, 0], jnp.asarray([10.0, 21.0], dtype=jnp.float32), atol=1.0e-6
    )


def test_interpolate_tracks_invalid_ranges_are_identity_noops():
    translations = jnp.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32
    )
    rotations = jnp.stack((_z_rotation(0.0), _z_rotation(1.0)))
    times = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    offsets = jnp.asarray([-1, 0, 999], dtype=jnp.int32)
    counts = jnp.asarray([2, 0, 1], dtype=jnp.int32)
    out_translation, out_rotation = geometry.se3_interpolate_tracks(
        translations,
        rotations,
        times,
        offsets,
        counts,
        jnp.full((3,), 0.5, dtype=jnp.float32),
    )
    assert jnp.array_equal(out_translation, jnp.zeros((3, 3)))
    assert jnp.array_equal(out_rotation, geometry.quat_identity((3,)))


def test_interpolate_tracks_accepts_column_metadata_and_empty_tracks():
    translations = jnp.asarray(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=jnp.float32
    )
    rotations = geometry.quat_identity((2,))
    actual, _ = geometry.se3_interpolate_tracks(
        translations,
        rotations,
        jnp.asarray([[0.0], [2.0]], dtype=jnp.float32),
        jnp.asarray([[0]], dtype=jnp.int32),
        jnp.asarray([[2]], dtype=jnp.int32),
        1.0,
    )
    assert jnp.allclose(actual, jnp.asarray([[2.0, 0.0, 0.0]]))

    empty_translation, empty_rotation = geometry.se3_interpolate_tracks(
        jnp.empty((0, 3), dtype=jnp.float32),
        jnp.empty((0, 4), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.int32),
        0.0,
    )
    assert empty_translation.shape == (0, 3)
    assert empty_rotation.shape == (0, 4)


def test_interpolate_tracks_jit_integer_times_and_gradients():
    translations = jnp.asarray(
        [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=jnp.float32
    )
    rotations = jnp.stack((_z_rotation(0.0), _z_rotation(jnp.pi / 2)))
    times = jnp.asarray([100, 108], dtype=jnp.int32)
    offsets = jnp.asarray([0], dtype=jnp.int32)
    counts = jnp.asarray([2], dtype=jnp.int32)
    query = jnp.asarray([102], dtype=jnp.int32)
    compiled = jax.jit(geometry.se3_interpolate_tracks)(
        translations, rotations, times, offsets, counts, query
    )
    assert jnp.allclose(compiled[0][0, 0], 2.0, atol=1.0e-6)

    float_times = times.astype(jnp.float32)
    float_query = query.astype(jnp.float32)
    gradients = jax.grad(
        lambda t, q, key_times, query_times: sum(
            jnp.sum(value)
            for value in geometry.se3_interpolate_tracks(
                t, q, key_times, offsets, counts, query_times
            )
        ),
        argnums=(0, 1, 2, 3),
    )(translations, rotations, float_times, float_query)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients)
    assert jnp.allclose(
        gradients[0][:, 0], jnp.asarray([0.75, 0.25]), atol=1.0e-6
    )


def test_two_pose_trajectory_interpolation_swapping_and_extrapolation():
    trans0 = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32)
    trans1 = jnp.asarray([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    rot0 = geometry.quat_identity((2,))
    rot1 = geometry.quat_identity((2,))
    time0 = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    time1 = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    point = jnp.zeros((2, 3), dtype=jnp.float32)
    query = jnp.asarray([0.5, 2.0], dtype=jnp.float32)
    result = geometry.trajectory_transform_point_2poses(
        trans0, rot0, time0, trans1, rot1, time1, point, query
    )
    assert jnp.allclose(
        result["point"],
        jnp.asarray([[1.0, 1.0, 1.0], [4.0, 0.0, 0.0]]),
        atol=1.0e-6,
    )
    assert jnp.array_equal(result["out_of_bounds"], jnp.asarray([False, True]))


def test_two_pose_trajectory_rotation_and_jit():
    trans = jnp.zeros((2, 3), dtype=jnp.float32)
    rot0 = geometry.quat_identity((2,))
    rot1 = jnp.stack((_z_rotation(jnp.pi / 2), _z_rotation(jnp.pi)))
    time0 = jnp.zeros((2,), dtype=jnp.float32)
    time1 = jnp.ones((2,), dtype=jnp.float32)
    query = jnp.asarray([0.5, 0.25], dtype=jnp.float32)
    expected = geometry.quat_slerp(rot0, rot1, query)
    result = jax.jit(geometry.trajectory_get_rotation_2poses)(
        trans, rot0, time0, trans, rot1, time1, query
    )
    _assert_same_rotation(result["quat"], expected)
    assert not bool(jnp.any(result["out_of_bounds"]))


def test_equal_trajectory_times_use_pose_zero_and_have_zero_time_gradients():
    trans0 = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    trans1 = jnp.asarray([[9.0, 8.0, 7.0]], dtype=jnp.float32)
    rot0 = _z_rotation(0.3)[None]
    rot1 = _z_rotation(1.2)[None]
    point = jnp.asarray([[0.2, -0.4, 0.1]], dtype=jnp.float32)
    times = jnp.zeros((1,), dtype=jnp.float32)

    def objective(time0, time1, query):
        return jnp.sum(
            geometry.trajectory_transform_point_2poses(
                trans0,
                rot0,
                time0,
                trans1,
                rot1,
                time1,
                point,
                query,
            )["point"]
        )

    actual = geometry.trajectory_transform_point_2poses(
        trans0, rot0, times, trans1, rot1, times, point, times
    )
    expected = geometry.se3pose_transform_point(trans0, rot0, point)
    assert jnp.allclose(actual["point"], expected, atol=1.0e-6)
    assert not bool(actual["out_of_bounds"][0])
    gradients = jax.grad(objective, argnums=(0, 1, 2))(times, times, times)
    assert all(jnp.array_equal(value, jnp.zeros_like(value)) for value in gradients)


def test_one_pose_trajectory_always_transforms_and_marks_exact_time():
    trans = jnp.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    rot = geometry.quat_identity((2,))
    point = jnp.asarray([[0.5, 1.0, -1.0], [1.0, 2.0, 3.0]], dtype=jnp.float32)
    time = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    query = jnp.asarray([0.0, 2.0], dtype=jnp.float32)
    result = geometry.trajectory_transform_point_1pose(
        trans, rot, time, point, query
    )
    assert jnp.allclose(result["point"], trans + point)
    assert jnp.array_equal(result["out_of_bounds"], jnp.asarray([False, True]))


def test_frame_transform_poses_tquat_matches_definition():
    input_poses = jnp.asarray(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32
    )
    frame_rotation = tuple(float(x) for x in _z_rotation(jnp.pi / 2))
    actual = geometry.frame_transform_poses_tquat(
        input_poses,
        frame_rotation,
        (1.0, 2.0, 3.0),
        2.0,
    )
    assert jnp.allclose(
        actual[:, :3], jnp.asarray([[2.0, 6.0, 6.0]]), atol=1.0e-6
    )
    _assert_same_rotation(actual[:, 3:], jnp.asarray(frame_rotation)[None])


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (
            lambda: geometry.se3pose_transform_point(
                jnp.zeros((1, 3)), jnp.zeros((2, 4)), jnp.zeros((1, 3))
            ),
            ValueError,
        ),
        (lambda: geometry.se3pose_from_matrix(jnp.zeros((2, 4))), ValueError),
        (
            lambda: geometry.se3pose_compose(
                jnp.zeros((1, 3)),
                geometry.quat_identity((1,)),
                jnp.zeros((2, 3)),
                geometry.quat_identity((2,)),
            ),
            ValueError,
        ),
        (
            lambda: geometry.se3_interpolate_tracks(
                jnp.zeros((2, 3)),
                geometry.quat_identity((2,)),
                jnp.zeros((1,), dtype=jnp.float32),
                jnp.asarray([0], dtype=jnp.int32),
                jnp.asarray([2], dtype=jnp.int32),
                0.5,
            ),
            ValueError,
        ),
        (
            lambda: geometry.se3_interpolate_tracks(
                jnp.zeros((2, 3)),
                geometry.quat_identity((2,)),
                jnp.zeros((2,), dtype=jnp.float32),
                jnp.asarray([0], dtype=jnp.float32),
                jnp.asarray([2], dtype=jnp.int32),
                0.5,
            ),
            TypeError,
        ),
        (
            lambda: geometry.se3_interpolate_tracks(
                jnp.zeros((2, 3)),
                geometry.quat_identity((2,)),
                jnp.zeros((2,), dtype=jnp.float32),
                jnp.asarray([0], dtype=jnp.int32),
                jnp.asarray([2], dtype=jnp.int32),
                jnp.asarray([0.2, 0.8], dtype=jnp.float32),
            ),
            ValueError,
        ),
        (
            lambda: geometry.trajectory_transform_point_1pose(
                jnp.zeros((1, 3), dtype=jnp.float16),
                jnp.zeros((1, 4), dtype=jnp.float16),
                jnp.zeros((1,), dtype=jnp.float16),
                jnp.zeros((1, 3), dtype=jnp.float16),
                jnp.zeros((1,), dtype=jnp.float16),
            ),
            TypeError,
        ),
        (
            lambda: geometry.frame_transform_poses_tquat(
                jnp.zeros((1, 6), dtype=jnp.float32),
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0),
                1.0,
            ),
            ValueError,
        ),
    ],
)
def test_pose_validation(call, error):
    with pytest.raises(error):
        call()
