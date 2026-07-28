"""Pure-JAX SE(3), packed-track, and trajectory operators."""

from __future__ import annotations

from numbers import Integral, Real

import jax
import jax.numpy as jnp

from .quaternion_ops import (
    _expect_array,
    _quat_multiply_raw,
    _quat_rotate_vector_raw,
    _require_float,
    _slerp_rows,
    quat_normalize_safe,
    quat_slerp,
    quat_to_matrix,
)


def _validate_pose_pair(
    translation: object,
    rotation: object,
    *,
    translation_name: str = "translation",
    rotation_name: str = "rotation",
) -> tuple[jax.Array, jax.Array]:
    translation = _expect_array(translation_name, translation)
    rotation = _expect_array(rotation_name, rotation)
    _require_float(translation_name, translation)
    _require_float(rotation_name, rotation)
    if translation.dtype != rotation.dtype:
        raise TypeError(
            f"{translation_name} and {rotation_name} must have the same dtype; "
            f"got {translation.dtype} vs {rotation.dtype}"
        )
    if translation.ndim != 2 or translation.shape[1] != 3:
        raise ValueError(
            f"{translation_name} must have shape (N, 3); got {translation.shape}"
        )
    if rotation.ndim != 2 or rotation.shape[1] != 4:
        raise ValueError(
            f"{rotation_name} must have shape (N, 4) xyzw; got {rotation.shape}"
        )
    if translation.shape[0] != rotation.shape[0]:
        raise ValueError(
            f"{translation_name} and {rotation_name} must share batch size N; "
            f"got {translation.shape[0]} vs {rotation.shape[0]}"
        )
    return translation, rotation


def _validate_pose_vector(
    translation: object,
    rotation: object,
    vector: object,
    vector_name: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    translation, rotation = _validate_pose_pair(translation, rotation)
    vector = _expect_array(vector_name, vector)
    _require_float(vector_name, vector)
    if vector.dtype != translation.dtype:
        raise TypeError(
            f"translation, rotation, and {vector_name} must have the same dtype; "
            f"got {translation.dtype}, {rotation.dtype}, {vector.dtype}"
        )
    if vector.ndim != 2 or vector.shape[1] != 3:
        raise ValueError(f"{vector_name} must have shape (N, 3); got {vector.shape}")
    if vector.shape[0] != translation.shape[0]:
        raise ValueError(
            f"translation, rotation, and {vector_name} must share batch size N; "
            f"got {translation.shape[0]} and {vector.shape[0]}"
        )
    return translation, rotation, vector


def se3pose_transform_point(
    translation: jax.Array, rotation: jax.Array, point: jax.Array
) -> jax.Array:
    """Apply ``R(rotation) @ point + translation`` row-wise."""

    translation, rotation, point = _validate_pose_vector(
        translation, rotation, point, "point"
    )
    return _quat_rotate_vector_raw(rotation, point) + translation


def se3pose_transform_direction(
    translation: jax.Array, rotation: jax.Array, direction: jax.Array
) -> jax.Array:
    """Transform directions row-wise, ignoring the validated translation."""

    _, rotation, direction = _validate_pose_vector(
        translation, rotation, direction, "direction"
    )
    return _quat_rotate_vector_raw(rotation, direction)


def se3pose_inverse_transform_point(
    translation: jax.Array, rotation: jax.Array, point: jax.Array
) -> jax.Array:
    """Apply ``R(rotation).T @ (point - translation)`` row-wise."""

    translation, rotation, point = _validate_pose_vector(
        translation, rotation, point, "point"
    )
    inverse_rotation = jnp.concatenate(
        (-rotation[:, :3], rotation[:, 3:4]), axis=-1
    )
    return _quat_rotate_vector_raw(inverse_rotation, point - translation)


def se3pose_inverse_transform_direction(
    translation: jax.Array, rotation: jax.Array, direction: jax.Array
) -> jax.Array:
    """Apply the inverse pose rotation to directions row-wise."""

    _, rotation, direction = _validate_pose_vector(
        translation, rotation, direction, "direction"
    )
    inverse_rotation = jnp.concatenate(
        (-rotation[:, :3], rotation[:, 3:4]), axis=-1
    )
    return _quat_rotate_vector_raw(inverse_rotation, direction)


def _pack_transform_matrix(
    rotation_matrix: jax.Array, translation: jax.Array
) -> jax.Array:
    upper = jnp.concatenate((rotation_matrix, translation[..., None]), axis=-1)
    bottom = jnp.broadcast_to(
        jnp.asarray((0, 0, 0, 1), dtype=translation.dtype),
        (translation.shape[0], 1, 4),
    )
    return jnp.concatenate((upper, bottom), axis=-2)


def se3pose_to_matrix(
    translation: jax.Array, rotation: jax.Array
) -> jax.Array:
    """Convert batched ``xyzw`` poses to homogeneous matrices."""

    translation, rotation = _validate_pose_pair(translation, rotation)
    return _pack_transform_matrix(quat_to_matrix(rotation), translation)


def _safe_shepperd_scale(value: jax.Array) -> jax.Array:
    # Inactive branches also execute under JAX. Keeping them finite does not
    # change the selected Shepperd branch for a valid rotation matrix.
    return 2 * jnp.sqrt(jnp.maximum(value, jnp.finfo(value.dtype).tiny))


def _matrix_to_quaternion_shepperd(matrix: jax.Array) -> jax.Array:
    r00, r01, r02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    r10, r11, r12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    r20, r21, r22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]
    trace = r00 + r11 + r22

    scalar_branch = (trace > r00) & (trace > r11) & (trace > r22)
    x_branch = (~scalar_branch) & (r00 > r11) & (r00 > r22)
    y_branch = (~scalar_branch) & (~x_branch) & (r11 > r22)

    scalar_scale = _safe_shepperd_scale(1 + trace)
    scalar_candidate = jnp.stack(
        (
            (r21 - r12) / scalar_scale,
            (r02 - r20) / scalar_scale,
            (r10 - r01) / scalar_scale,
            0.25 * scalar_scale,
        ),
        axis=-1,
    )

    x_scale = _safe_shepperd_scale(1 + r00 - r11 - r22)
    x_candidate = jnp.stack(
        (
            0.25 * x_scale,
            (r01 + r10) / x_scale,
            (r02 + r20) / x_scale,
            (r21 - r12) / x_scale,
        ),
        axis=-1,
    )

    y_scale = _safe_shepperd_scale(1 + r11 - r00 - r22)
    y_candidate = jnp.stack(
        (
            (r01 + r10) / y_scale,
            0.25 * y_scale,
            (r12 + r21) / y_scale,
            (r02 - r20) / y_scale,
        ),
        axis=-1,
    )

    z_scale = _safe_shepperd_scale(1 + r22 - r00 - r11)
    z_candidate = jnp.stack(
        (
            (r02 + r20) / z_scale,
            (r12 + r21) / z_scale,
            0.25 * z_scale,
            (r10 - r01) / z_scale,
        ),
        axis=-1,
    )

    quaternion = jnp.where(
        scalar_branch[:, None],
        scalar_candidate,
        jnp.where(
            x_branch[:, None],
            x_candidate,
            jnp.where(y_branch[:, None], y_candidate, z_candidate),
        ),
    )
    return quat_normalize_safe(quaternion)


def se3pose_from_matrix(matrix: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Convert ``(N, 4, 4)`` or flattened ``(N, 16)`` matrices to poses."""

    matrix = _expect_array("matrix", matrix)
    _require_float("matrix", matrix)
    if matrix.ndim == 3 and matrix.shape[1:] == (4, 4):
        matrices = matrix
    elif matrix.ndim == 2 and matrix.shape[1] == 16:
        matrices = matrix.reshape((-1, 4, 4))
    else:
        raise ValueError(
            "matrix must have shape (N, 4, 4) or (N, 16); "
            f"got {matrix.shape}"
        )
    translation = matrices[:, :3, 3]
    rotation = _matrix_to_quaternion_shepperd(matrices[:, :3, :3])
    return translation, rotation


def se3pose_compose(
    parent_translation: jax.Array,
    parent_rotation: jax.Array,
    child_translation: jax.Array,
    child_rotation: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compose poses as ``parent * child`` row-wise."""

    parent_translation, parent_rotation = _validate_pose_pair(
        parent_translation,
        parent_rotation,
        translation_name="parent_translation",
        rotation_name="parent_rotation",
    )
    child_translation, child_rotation = _validate_pose_pair(
        child_translation,
        child_rotation,
        translation_name="child_translation",
        rotation_name="child_rotation",
    )
    if parent_translation.dtype != child_translation.dtype:
        raise TypeError(
            "parent and child poses must have the same dtype; got "
            f"{parent_translation.dtype} vs {child_translation.dtype}"
        )
    if parent_translation.shape[0] != child_translation.shape[0]:
        raise ValueError(
            "parent and child poses must share batch size N; got "
            f"{parent_translation.shape[0]} vs {child_translation.shape[0]}"
        )
    translation = (
        _quat_rotate_vector_raw(parent_rotation, child_translation)
        + parent_translation
    )
    rotation = _quat_multiply_raw(parent_rotation, child_rotation)
    return translation, rotation


def _validate_track_arrays(
    pose_translations: object,
    pose_rotations: object,
    pose_times: object,
    pose_offsets: object,
    pose_counts: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    pose_translations = _expect_array("pose_translations", pose_translations)
    pose_rotations = _expect_array("pose_rotations", pose_rotations)
    pose_times = _expect_array("pose_times", pose_times)
    pose_offsets = _expect_array("pose_offsets", pose_offsets)
    pose_counts = _expect_array("pose_counts", pose_counts)
    _require_float("pose_translations", pose_translations)
    _require_float("pose_rotations", pose_rotations)
    if pose_translations.dtype != pose_rotations.dtype:
        raise TypeError(
            "pose_translations and pose_rotations must have the same dtype; got "
            f"{pose_translations.dtype} vs {pose_rotations.dtype}"
        )
    if not (
        pose_times.dtype in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64))
        or jnp.issubdtype(pose_times.dtype, jnp.integer)
    ):
        raise TypeError(
            "pose_times must have dtype float32, float64, or integer; got "
            f"{pose_times.dtype}"
        )
    if not jnp.issubdtype(pose_offsets.dtype, jnp.integer):
        raise TypeError("pose_offsets must have an integer dtype")
    if not jnp.issubdtype(pose_counts.dtype, jnp.integer):
        raise TypeError("pose_counts must have an integer dtype")

    if pose_translations.ndim != 2 or pose_translations.shape[1] != 3:
        raise ValueError(
            "pose_translations must have shape (M, 3); "
            f"got {pose_translations.shape}"
        )
    if pose_rotations.ndim != 2 or pose_rotations.shape[1] != 4:
        raise ValueError(
            "pose_rotations must have shape (M, 4) xyzw; "
            f"got {pose_rotations.shape}"
        )
    pose_count = pose_translations.shape[0]
    if pose_rotations.shape[0] != pose_count:
        raise ValueError(
            "pose_translations and pose_rotations must share M rows; got "
            f"{pose_count} vs {pose_rotations.shape[0]}"
        )
    if not (
        (pose_times.ndim == 1 and pose_times.shape[0] == pose_count)
        or (pose_times.ndim == 2 and pose_times.shape == (pose_count, 1))
    ):
        raise ValueError(
            "pose_times must have shape (M,) or (M, 1) matching poses; "
            f"got {pose_times.shape} for M={pose_count}"
        )
    if pose_offsets.ndim == 1:
        track_count = pose_offsets.shape[0]
    elif pose_offsets.ndim == 2 and pose_offsets.shape[1] == 1:
        track_count = pose_offsets.shape[0]
    else:
        raise ValueError(
            "pose_offsets must have shape (C,) or (C, 1); "
            f"got {pose_offsets.shape}"
        )
    if not (
        (pose_counts.ndim == 1 and pose_counts.shape[0] == track_count)
        or (pose_counts.ndim == 2 and pose_counts.shape == (track_count, 1))
    ):
        raise ValueError(
            "pose_counts must have shape (C,) or (C, 1) matching "
            f"pose_offsets; got {pose_counts.shape} for C={track_count}"
        )
    return (
        pose_translations,
        pose_rotations,
        pose_times,
        pose_offsets,
        pose_counts,
    )


def _track_time_dtype(
    pose_times: jax.Array, query_time: float | jax.Array
) -> jnp.dtype:
    if jnp.issubdtype(pose_times.dtype, jnp.floating):
        if isinstance(query_time, jax.Array) and jnp.issubdtype(
            query_time.dtype, jnp.floating
        ):
            return jnp.result_type(pose_times.dtype, query_time.dtype)
        return pose_times.dtype
    query_is_integer = isinstance(query_time, Integral) or (
        isinstance(query_time, jax.Array)
        and jnp.issubdtype(query_time.dtype, jnp.integer)
    )
    if query_is_integer:
        return jnp.dtype(jnp.int64 if jax.config.x64_enabled else jnp.int32)
    return jnp.dtype(jnp.float64 if jax.config.x64_enabled else jnp.float32)


def _track_query_times(
    query_time: float | jax.Array,
    track_count: int,
    dtype: jnp.dtype,
) -> jax.Array:
    if isinstance(query_time, Real):
        return jnp.full((track_count,), query_time, dtype=dtype)
    query_time = _expect_array("query_time", query_time)
    if not (
        jnp.issubdtype(query_time.dtype, jnp.floating)
        or jnp.issubdtype(query_time.dtype, jnp.integer)
    ):
        raise TypeError(
            "query_time must have a floating-point or integer dtype; got "
            f"{query_time.dtype}"
        )
    query_time = query_time.astype(dtype).reshape((-1,))
    if query_time.size == 1:
        return jnp.broadcast_to(query_time, (track_count,))
    if query_time.size != track_count:
        raise ValueError(
            "query_time must be a scalar or flatten to one value per track; "
            f"got {query_time.size} values for C={track_count}"
        )
    return query_time


def se3_interpolate_tracks(
    pose_translations: jax.Array,
    pose_rotations: jax.Array,
    pose_times: jax.Array,
    pose_offsets: jax.Array,
    pose_counts: jax.Array,
    query_time: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Interpolate packed SE(3) tracks, clamping queries to each track span."""

    (
        pose_translations,
        pose_rotations,
        pose_times,
        pose_offsets,
        pose_counts,
    ) = _validate_track_arrays(
        pose_translations,
        pose_rotations,
        pose_times,
        pose_offsets,
        pose_counts,
    )
    index_dtype = jnp.int64 if jax.config.x64_enabled else jnp.int32
    # Upstream widens every accepted integer offset/count tensor to int64
    # before indexing. int32 is the corresponding safe JAX index type when
    # x64 is disabled.
    offsets = pose_offsets.reshape((-1,)).astype(index_dtype)
    counts = pose_counts.reshape((-1,)).astype(index_dtype)
    track_count = offsets.shape[0]
    if track_count == 0:
        return pose_translations[:0], pose_rotations[:0]

    time_dtype = _track_time_dtype(pose_times, query_time)
    times = pose_times.reshape((-1,)).astype(time_dtype)
    query_times = _track_query_times(query_time, track_count, time_dtype)
    packed_pose_count = pose_translations.shape[0]
    if packed_pose_count == 0:
        translations = jnp.zeros(
            (track_count, 3), dtype=pose_translations.dtype
        )
        rotations = jnp.zeros((track_count, 4), dtype=pose_rotations.dtype)
        return translations, rotations.at[:, 3].set(1)

    indices = jnp.arange(packed_pose_count, dtype=offsets.dtype)
    identity = jnp.asarray((0, 0, 0, 1), dtype=pose_rotations.dtype)

    def interpolate_one(start, count, query):
        valid_range = (
            (start >= 0)
            & (count > 0)
            & (start <= packed_pose_count)
            & (count <= packed_pose_count - start)
        )
        safe_start = jnp.clip(start, 0, packed_pose_count - 1)
        safe_last = jnp.clip(start + jnp.maximum(count, 1) - 1, 0, packed_pose_count - 1)
        first_time = times[safe_start]
        last_time = times[safe_last]
        clamped_query = jnp.clip(query, first_time, last_time)

        in_track = (indices >= start) & (indices < start + count)
        lower_candidates = in_track & (times >= clamped_query)
        lower_bound = jnp.min(
            jnp.where(lower_candidates, indices, packed_pose_count)
        )
        right = jnp.where(
            query <= first_time,
            safe_start,
            jnp.where(
                query > last_time,
                safe_last,
                jnp.minimum(lower_bound, safe_last),
            ),
        )
        right_time = times[right]
        exact_or_first = (right == safe_start) | (right_time == clamped_query)
        left = jnp.where(exact_or_first, right, right - 1)
        left_time = times[left]
        same_pose = left == right
        valid_interval = (~same_pose) & (right_time != left_time)

        if jnp.issubdtype(time_dtype, jnp.integer):
            alpha_dtype = jnp.dtype(
                jnp.float64 if jax.config.x64_enabled else pose_translations.dtype
            )
        else:
            alpha_dtype = time_dtype
        numerator = clamped_query.astype(alpha_dtype) - left_time.astype(alpha_dtype)
        denominator = right_time.astype(alpha_dtype) - left_time.astype(alpha_dtype)
        safe_denominator = jnp.where(valid_interval, denominator, 1)
        alpha = jnp.where(
            valid_interval, numerator / safe_denominator, 0
        ).astype(pose_translations.dtype)

        left_translation = pose_translations[left]
        right_translation = pose_translations[right]
        translation = left_translation + alpha * (
            right_translation - left_translation
        )
        left_rotation = pose_rotations[left]
        right_rotation = pose_rotations[right]
        slerp_left = jnp.where(same_pose, identity, left_rotation)
        slerp_right = jnp.where(same_pose, identity, right_rotation)
        interpolated_rotation = _slerp_rows(
            slerp_left[None], slerp_right[None], alpha[None]
        )[0]
        rotation = jnp.where(same_pose, left_rotation, interpolated_rotation)
        return (
            jnp.where(valid_range, translation, jnp.zeros_like(translation)),
            jnp.where(valid_range, rotation, identity),
        )

    return jax.vmap(interpolate_one)(offsets, counts, query_times)


def se3pose_to_inverse_matrix(
    translation: jax.Array,
    rotation: jax.Array,
    wxyz_format: bool = False,
) -> jax.Array:
    """Convert poses to homogeneous inverse matrices."""

    translation, rotation = _validate_pose_pair(translation, rotation)
    if wxyz_format:
        rotation = jnp.concatenate((rotation[:, 1:], rotation[:, :1]), axis=-1)
    rotation_matrix = quat_to_matrix(rotation)
    inverse_rotation = jnp.swapaxes(rotation_matrix, -1, -2)
    inverse_translation = -jnp.einsum(
        "nij,nj->ni", inverse_rotation, translation
    )
    return _pack_transform_matrix(inverse_rotation, inverse_translation)


def _require_float32(name: str, value: jax.Array) -> None:
    if value.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"{name} must have dtype float32, got {value.dtype}")


def _validate_trajectory_time(
    name: str, value: object, batch_size: int
) -> jax.Array:
    value = _expect_array(name, value)
    _require_float32(name, value)
    if not (
        (value.ndim == 1 and value.shape[0] == batch_size)
        or (value.ndim == 2 and value.shape == (batch_size, 1))
    ):
        raise ValueError(
            f"{name} must have shape (N,) or (N, 1) with N={batch_size}; "
            f"got {value.shape}"
        )
    return value.reshape((-1,))


def _validate_trajectory_pose(
    translation_name: str,
    translation: object,
    rotation_name: str,
    rotation: object,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    translation = _expect_array(translation_name, translation)
    rotation = _expect_array(rotation_name, rotation)
    _require_float32(translation_name, translation)
    _require_float32(rotation_name, rotation)
    if translation.shape != (batch_size, 3):
        raise ValueError(
            f"{translation_name} must be (N, 3) with N={batch_size}; "
            f"got {translation.shape}"
        )
    if rotation.shape != (batch_size, 4):
        raise ValueError(
            f"{rotation_name} must be (N, 4) with N={batch_size}; "
            f"got {rotation.shape}"
        )
    return translation, rotation


def _interpolate_two_pose_trajectory(
    trans0: jax.Array,
    rot0: jax.Array,
    time0: jax.Array,
    trans1: jax.Array,
    rot1: jax.Array,
    time1: jax.Array,
    query_time: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    first_is_earlier = time0 <= time1
    low_time = jnp.minimum(time0, time1)
    high_time = jnp.maximum(time0, time1)
    duration = high_time - low_time
    safe_duration = jnp.where(duration > 0, duration, 1)
    alpha = jnp.where(duration > 0, (query_time - low_time) / safe_duration, 0)
    low_translation = jnp.where(first_is_earlier[:, None], trans0, trans1)
    high_translation = jnp.where(first_is_earlier[:, None], trans1, trans0)
    low_rotation = jnp.where(first_is_earlier[:, None], rot0, rot1)
    high_rotation = jnp.where(first_is_earlier[:, None], rot1, rot0)
    translation = low_translation + alpha[:, None] * (
        high_translation - low_translation
    )
    rotation = quat_slerp(low_rotation, high_rotation, alpha)
    out_of_bounds = (query_time < low_time) | (query_time > high_time)
    return translation, rotation, out_of_bounds


def trajectory_transform_point_2poses(
    trans0: jax.Array,
    rot0: jax.Array,
    time0: jax.Array,
    trans1: jax.Array,
    rot1: jax.Array,
    time1: jax.Array,
    point: jax.Array,
    query_time: jax.Array,
) -> dict[str, jax.Array]:
    """Transform points with unordered two-keyframe trajectories."""

    point = _expect_array("point", point)
    _require_float32("point", point)
    if point.ndim != 2 or point.shape[1] != 3:
        raise ValueError(f"point must be (N, 3); got {point.shape}")
    batch_size = point.shape[0]
    trans0, rot0 = _validate_trajectory_pose(
        "trans0", trans0, "rot0", rot0, batch_size
    )
    trans1, rot1 = _validate_trajectory_pose(
        "trans1", trans1, "rot1", rot1, batch_size
    )
    time0 = _validate_trajectory_time("time0", time0, batch_size)
    time1 = _validate_trajectory_time("time1", time1, batch_size)
    query_time = _validate_trajectory_time(
        "query_time", query_time, batch_size
    )
    translation, rotation, out_of_bounds = _interpolate_two_pose_trajectory(
        trans0, rot0, time0, trans1, rot1, time1, query_time
    )
    return {
        "point": _quat_rotate_vector_raw(rotation, point) + translation,
        "out_of_bounds": out_of_bounds,
    }


def trajectory_get_rotation_2poses(
    trans0: jax.Array,
    rot0: jax.Array,
    time0: jax.Array,
    trans1: jax.Array,
    rot1: jax.Array,
    time1: jax.Array,
    query_time: jax.Array,
) -> dict[str, jax.Array]:
    """Get rotations from unordered two-keyframe trajectories."""

    trans0 = _expect_array("trans0", trans0)
    if trans0.ndim != 2 or trans0.shape[1] != 3:
        raise ValueError(f"trans0 must be (N, 3); got {trans0.shape}")
    batch_size = trans0.shape[0]
    trans0, rot0 = _validate_trajectory_pose(
        "trans0", trans0, "rot0", rot0, batch_size
    )
    trans1, rot1 = _validate_trajectory_pose(
        "trans1", trans1, "rot1", rot1, batch_size
    )
    time0 = _validate_trajectory_time("time0", time0, batch_size)
    time1 = _validate_trajectory_time("time1", time1, batch_size)
    query_time = _validate_trajectory_time(
        "query_time", query_time, batch_size
    )
    _, rotation, out_of_bounds = _interpolate_two_pose_trajectory(
        trans0, rot0, time0, trans1, rot1, time1, query_time
    )
    return {"quat": rotation, "out_of_bounds": out_of_bounds}


def trajectory_transform_point_1pose(
    trans: jax.Array,
    rot: jax.Array,
    time: jax.Array,
    point: jax.Array,
    query_time: jax.Array,
) -> dict[str, jax.Array]:
    """Transform points with a single-keyframe trajectory."""

    point = _expect_array("point", point)
    _require_float32("point", point)
    if point.ndim != 2 or point.shape[1] != 3:
        raise ValueError(f"point must be (N, 3); got {point.shape}")
    batch_size = point.shape[0]
    trans, rot = _validate_trajectory_pose(
        "trans", trans, "rot", rot, batch_size
    )
    time = _validate_trajectory_time("time", time, batch_size)
    query_time = _validate_trajectory_time(
        "query_time", query_time, batch_size
    )
    return {
        "point": _quat_rotate_vector_raw(rot, point) + trans,
        "out_of_bounds": query_time != time,
    }


def frame_transform_poses_tquat(
    tquat_poses: jax.Array,
    rotation: tuple[float, float, float, float],
    translation: tuple[float, float, float],
    scale: float,
) -> jax.Array:
    """Apply a fixed frame transform to ``[t, q_xyzw]`` pose rows."""

    tquat_poses = _expect_array("tquat_poses", tquat_poses)
    _require_float32("tquat_poses", tquat_poses)
    if tquat_poses.ndim != 2 or tquat_poses.shape[1] != 7:
        raise ValueError(
            f"tquat_poses must have shape (N, 7); got {tquat_poses.shape}"
        )
    if len(rotation) != 4:
        raise ValueError("rotation must contain four xyzw values")
    if len(translation) != 3:
        raise ValueError("translation must contain three values")
    frame_rotation = jnp.asarray(rotation, dtype=tquat_poses.dtype)
    frame_translation = jnp.asarray(translation, dtype=tquat_poses.dtype)
    frame_rotations = jnp.broadcast_to(
        frame_rotation, (tquat_poses.shape[0], 4)
    )
    output_translation = float(scale) * (
        _quat_rotate_vector_raw(frame_rotations, tquat_poses[:, :3])
        + frame_translation
    )
    output_rotation = _quat_multiply_raw(
        frame_rotations, tquat_poses[:, 3:]
    )
    return jnp.concatenate((output_translation, output_rotation), axis=-1)


__all__ = [
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
]
