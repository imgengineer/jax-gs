import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from jaxgs import Camera, CapacityConfig, create_gaussians, seed_gaussians
from jaxgs.io_manager.checkpoint import save_gaussians
from jaxgs.reference.rasterizer_jax import rasterize_jax
from jaxgs.render.projection import project
from jaxgs.render.visibility_table import build_visibility_table


@pytest.fixture
def evaluation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("evaluate_protocol")


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="evaluation requires JAX CUDA and CuTe",
)
@pytest.mark.parametrize(
    "degree,width,height,expected_shape",
    [(degree, 16, 8, (8, 16)) for degree in range(4)] + [(0, 2000, 24, (19, 1600))],
)
def test_evaluation_matches_reference_for_checkpoint_sh_and_resolution(
    evaluation, tmp_path, degree, width, height, expected_shape
):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (tmp_path / "images").mkdir()
    (sparse / "cameras.txt").write_text(
        f"1 PINHOLE {width} {height} 24 24 {width / 2} {height / 2}\n"
    )
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.png\n\n")
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (height, width, 3), np.uint8))
    image.save(tmp_path / "images" / "frame.png")
    config = CapacityConfig(2, 1, 4, 16, degree, 512, tile_height=8)
    pool = seed_gaussians(
        create_gaussians(config),
        jnp.array([[0.1, 0.0, 2.0], [-0.1, 0.1, 2.5]]),
        jnp.array([[0.4, 0.5, 0.3], [0.6, 0.4, 0.5]]),
        scale=0.2,
        opacity=0.5,
    )
    pool = pool.replace(sh=pool.sh.at[:, 1:].set(0.05))
    checkpoint = tmp_path / "model.npz"
    save_gaussians(checkpoint, pool)
    scaled_height, scaled_width = expected_shape
    camera = Camera.from_colmap(
        [1, 0, 0, 0],
        [0, 0, 0],
        24 * scaled_width / width,
        24 * scaled_height / height,
        scaled_width / 2,
        scaled_height / 2,
        scaled_width,
        scaled_height,
    )
    projected = project(pool, camera, config)
    table = build_visibility_table(projected, camera, config)
    expected_image = rasterize_jax(projected, table, camera, config).rgb
    target = np.asarray(image.resize((scaled_width, scaled_height)), np.float32) / 255
    expected_psnr = -10 * np.log10(np.mean((np.asarray(expected_image) - target) ** 2))
    rows = evaluation.evaluate_jax(
        SimpleNamespace(model=checkpoint, scene=tmp_path, images="images")
    )
    assert len(rows) == 1 and rows[0]["view"] == "frame.png"
    # The production renderer packs intermediate values into float16.
    np.testing.assert_allclose(rows[0]["psnr"], expected_psnr, rtol=0, atol=0.02)


@pytest.mark.parametrize("sh_dim", [0, 2, 25])
def test_evaluation_rejects_invalid_sh_dimension_before_render(evaluation, tmp_path, sh_dim):
    pool = create_gaussians(CapacityConfig(2, sh_degree=0))
    pool = pool.replace(sh=jnp.zeros((2, sh_dim, 3), jnp.float32))
    checkpoint = tmp_path / "invalid.npz"
    save_gaussians(checkpoint, pool)
    with pytest.raises(ValueError, match="SH dimension"):
        evaluation.evaluate_jax(SimpleNamespace(model=checkpoint, scene=tmp_path, images="images"))
