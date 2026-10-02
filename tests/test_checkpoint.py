import jax.numpy as jnp
import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from jaxgs import CapacityConfig, create_gaussians
from jaxgs.io_manager.checkpoint import load_gaussians, save_gaussians
from jaxgs.scene.types import PARAMETER_NAMES


@pytest.mark.parametrize("degree", range(4))
@pytest.mark.parametrize("empty", [False, True])
def test_ply_preserves_active_parameters_and_standard_sh_layout(tmp_path, degree, empty):
    pool = create_gaussians(CapacityConfig(7, sh_degree=degree))
    alive = np.zeros(7, bool) if empty else np.array([True, False, False, True, False, True, False])
    rng = np.random.default_rng(8)
    values = {}
    for name in PARAMETER_NAMES:
        value = rng.normal(size=getattr(pool, name).shape).astype(np.float32)
        value[~alive] = np.nan  # Unused slots must never reach the exported model.
        values[name] = jnp.asarray(value)
    pool = pool.replace(
        **values,
        alive=jnp.asarray(alive),
        free_mask=jnp.asarray(~alive),
        n_active=jnp.array(alive.sum(), jnp.int32),
    )
    output = tmp_path / "nested" / "model.ply"
    save_gaussians(output, pool)
    assert output.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")
    vertices = PlyData.read(output)["vertex"]
    count = int(alive.sum())
    assert len(vertices) == count
    for name in ("nx", "ny", "nz"):
        np.testing.assert_array_equal(vertices[name], 0)
    for i in range(3):
        np.testing.assert_array_equal(vertices[f"f_dc_{i}"], np.asarray(pool.sh)[alive, 0, i])
        np.testing.assert_array_equal(vertices[f"scale_{i}"], np.asarray(pool.log_scale)[alive, i])
    np.testing.assert_array_equal(vertices["opacity"], np.asarray(pool.opacity)[alive, 0])
    rest_count = 3 * (pool.sh.shape[1] - 1)
    expected_rest = np.asarray(pool.sh)[alive, 1:].transpose(0, 2, 1).reshape(count, rest_count)
    for i in range(rest_count):
        np.testing.assert_array_equal(vertices[f"f_rest_{i}"], expected_rest[:, i])
    restored = load_gaussians(output)
    for name in PARAMETER_NAMES:
        np.testing.assert_array_equal(
            getattr(restored, name), np.asarray(getattr(pool, name))[alive]
        )
    assert int(restored.n_active) == count
    np.testing.assert_array_equal(restored.alive, np.ones(count, bool))
    np.testing.assert_array_equal(restored.free_mask, np.zeros(count, bool))


@pytest.mark.parametrize("rest_count", [3, 48])
def test_ply_rejects_unsupported_sh_dimension(tmp_path, rest_count):
    vertices = np.zeros(1, dtype=[(f"f_rest_{i}", "f4") for i in range(rest_count)])
    path = tmp_path / "invalid.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)
    with pytest.raises(ValueError, match="PLY SH dimension"):
        load_gaussians(path)
