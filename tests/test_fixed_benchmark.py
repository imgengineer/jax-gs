import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("fixed_model")


@pytest.mark.parametrize("count", [1, 65, 128, 129, 1_000_000])
def test_benchmark_cluster_padding_matches_native_tail_copies(benchmark, count):
    indices = benchmark._cluster_indices(count, 128)
    np.testing.assert_array_equal(indices[:count], np.arange(count))
    expected_count = {1: 128, 65: 128, 128: 128, 129: 256, 1_000_000: 1_000_064}[count]
    assert len(indices) == expected_count
    if count == 1:
        np.testing.assert_array_equal(indices, 0)
    elif count != 128:
        # LiteGS duplicates these original tail slots before reshaping.
        first_tail = {65: 2, 129: 2, 1_000_000: 999_936}[count]
        np.testing.assert_array_equal(indices[count:], np.arange(first_tail, count))


def test_benchmark_rejects_empty_gaussian_cloud(benchmark):
    with pytest.raises(ValueError, match="PLY is empty"):
        benchmark._cluster_indices(0, 128)


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="fixed training benchmark requires JAX CUDA and CuTe",
)
@pytest.mark.parametrize(
    "count,width,height,expected_shape",
    [(128, 32, 16, (16, 32)), (129, 32, 16, (16, 32)), (129, 2000, 24, (19, 1600))],
)
def test_fixed_benchmark_pads_parameters_before_training(
    benchmark, tmp_path, monkeypatch, count, width, height, expected_shape
):
    from jaxgs.scene.types import PARAMETER_NAMES
    from jaxgs.training import step

    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    images = tmp_path / "images"
    images.mkdir()
    (sparse / "cameras.txt").write_text(
        f"1 PINHOLE {width} {height} 24 24 {width / 2} {height / 2}\n"
    )
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.png\n\n")
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (height, width, 3), np.uint8))
    image.save(images / "frame.png")
    # Native ImageFrame.load_image(-1) uses Pillow's default bicubic resize.
    expected_target = np.asarray(image.resize(expected_shape[::-1]), np.uint8)
    fields = ["x", "y", "z", "opacity"]
    fields += [
        f"{prefix}_{i}"
        for prefix, size in (("scale", 3), ("rot", 4), ("f_dc", 3), ("f_rest", 45))
        for i in range(size)
    ]
    vertices = np.zeros(count, dtype=[(name, "f4") for name in fields])
    vertices["x"] = np.linspace(-0.2, 0.2, count)
    vertices["z"] = 2
    vertices["opacity"] = np.log(0.1 / 0.9)
    vertices["rot_0"] = 1
    for i in range(3):
        vertices[f"scale_{i}"] = np.log(0.03 + 0.01 * i)
        vertices[f"f_dc_{i}"] = -0.3 + 0.1 * i
    ply = tmp_path / "gaussians.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(ply)
    captured = {}
    original_step = step.array_train_step

    def capture_initial_pool(pool, state, stats, bounds, camera, target, *args, **kwargs):
        captured.update({name: np.asarray(getattr(pool, name)).copy() for name in PARAMETER_NAMES})
        np.testing.assert_array_equal(pool.alive, True)
        np.testing.assert_array_equal(pool.free_mask, False)
        assert int(pool.n_active) == pool.xyz.shape[0]
        assert (camera.height, camera.width) == expected_shape
        np.testing.assert_allclose(
            [camera.fx, camera.fy, camera.cx, camera.cy],
            [
                24 * camera.width / width,
                24 * camera.height / height,
                camera.width / 2,
                camera.height / 2,
            ],
        )
        np.testing.assert_array_equal(target, expected_target)
        return original_step(pool, state, stats, bounds, camera, target, *args, **kwargs)

    monkeypatch.setattr(step, "array_train_step", capture_initial_pool)
    result = benchmark.benchmark_jax(
        SimpleNamespace(
            ply=ply,
            scene=tmp_path,
            images="images",
            pairs=4096,
            view=0,
            optimizer="optax",
            warmup=2,
            steps=1,
            trace=None,
        )
    )
    assert result["input_count"] == count
    assert result["count"] == (128 if count == 128 else 256)
    assert result["shape"] == list(expected_shape)
    assert result["jit_cache_size"] == 1
    assert np.isfinite(result["final_loss"])
    np.testing.assert_array_equal(captured["xyz"][:count, 0], vertices["x"])
    if count == 129:
        for value in captured.values():
            np.testing.assert_array_equal(value[129:], value[2:129])
