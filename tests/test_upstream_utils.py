import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.relocation import compute_relocation
from jax_gs.utils import (
    depth_to_normal,
    depth_to_points,
    get_projection_matrix,
    inverse_log_transform,
    log_transform,
    normalized_quat_to_rotmat,
    save_ply,
)


def _binomial_table(size: int) -> jax.Array:
    table = np.zeros((size, size), dtype=np.float32)
    for row in range(size):
        for column in range(row + 1):
            table[row, column] = math.comb(row, column)
    return jnp.asarray(table)


def test_compute_relocation_matches_equation_and_is_differentiable():
    opacities = jnp.asarray([0.36, 0.64], jnp.float32)
    scales = jnp.asarray([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]], jnp.float32)
    ratios = jnp.asarray([1, 2], jnp.int32)
    binoms = _binomial_table(4)
    new_opacities, new_scales = jax.jit(compute_relocation)(
        opacities, scales, ratios, binoms
    )

    second_opacity = 1.0 - math.sqrt(1.0 - float(opacities[1]))
    denominator = 2 * second_opacity - second_opacity**2 / math.sqrt(2)
    expected_factor = float(opacities[1]) / denominator
    np.testing.assert_allclose(new_opacities[0], opacities[0], rtol=1e-6)
    np.testing.assert_allclose(new_scales[0], scales[0], rtol=1e-6)
    np.testing.assert_allclose(new_opacities[1], second_opacity, rtol=1e-6)
    np.testing.assert_allclose(new_scales[1], scales[1] * expected_factor, rtol=1e-6)

    def objective(values):
        relocated_opacities, relocated_scales = compute_relocation(
            values, scales, ratios, binoms
        )
        return relocated_opacities.sum() + relocated_scales.sum()

    gradients = jax.grad(objective)(opacities)
    assert jnp.all(jnp.isfinite(gradients))


def test_signed_log_transform_round_trip_and_projection_matrix():
    values = jnp.asarray([-100.0, -0.5, 0.0, 0.5, 100.0], jnp.float32)
    restored = jax.jit(inverse_log_transform)(jax.jit(log_transform)(values))
    np.testing.assert_allclose(restored, values, rtol=1e-6, atol=1e-6)

    matrix = jax.jit(get_projection_matrix)(0.1, 100.0, math.pi / 2, math.pi / 2)
    expected = np.zeros((4, 4), np.float32)
    expected[0, 0] = expected[1, 1] = 1.0
    expected[3, 2] = 1.0
    expected[2, 2] = 100.0 / 99.9
    expected[2, 3] = -10.0 / 99.9
    np.testing.assert_allclose(matrix, expected, rtol=1e-6)


def test_signed_log_transform_accepts_current_main_keyword_names():
    values = jnp.asarray([-2.0, 0.0, 3.0], jnp.float32)
    transformed = log_transform(x=values)
    restored = inverse_log_transform(y=transformed)
    np.testing.assert_allclose(restored, values, rtol=1e-6, atol=1e-6)

    quaternion = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], jnp.float32)
    np.testing.assert_allclose(
        normalized_quat_to_rotmat(quat=quaternion), jnp.eye(3)[None]
    )


def test_utils_geometry_aliases_support_batches():
    quaternion = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], jnp.float32)
    np.testing.assert_allclose(normalized_quat_to_rotmat(quaternion), jnp.eye(3)[None])
    depths = jnp.ones((1, 3, 3, 1), jnp.float32)
    camtoworlds = jnp.eye(4, dtype=jnp.float32)[None]
    intrinsics = jnp.asarray([[[1.0, 0.0, 1.5], [0.0, 1.0, 1.5], [0, 0, 1]]])
    points = depth_to_points(depths, camtoworlds, intrinsics)
    normals = depth_to_normal(depths, camtoworlds, intrinsics)
    assert points.shape == (1, 3, 3, 3)
    assert normals.shape == (1, 3, 3, 3)
    assert jnp.all(jnp.isfinite(points))
    assert jnp.all(jnp.isfinite(normals))


def test_deprecated_save_ply_filters_invalid_rows(tmp_path: Path):
    splats = {
        "means": jnp.asarray([[0.0, 0.0, 1.0], [jnp.nan, 0.0, 2.0]]),
        "scales": jnp.zeros((2, 3)),
        "quats": jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        "opacities": jnp.zeros((2,)),
        "sh0": jnp.zeros((2, 1, 3)),
        "shN": jnp.zeros((2, 0, 3)),
    }
    with pytest.warns(DeprecationWarning):
        path = save_ply(splats, tmp_path / "model.ply")
    data = path.read_bytes()
    assert b"element vertex 1\n" in data.split(b"end_header\n", maxsplit=1)[0]
