import importlib.util
import runpy
import sys

import jax
import numpy as np
import pytest
from PIL import Image

from jaxgs.io_manager.checkpoint import load_gaussians
from jaxgs.training.reference_trainer import train_colmap


@pytest.fixture
def small_scene(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    images = tmp_path / "images"
    images.mkdir()
    (sparse / "cameras.txt").write_text("1 PINHOLE 4 4 4 4 2 2\n")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.png\n\n")
    (sparse / "points3D.txt").write_text("1 0 0 2 128 64 32 0.1\n2 0.1 0 2 64 128 32 0.1\n")
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(images / "frame.png")
    return tmp_path


@pytest.mark.parametrize("missing", ["images", "points"])
def test_training_requires_registered_images_and_points(small_scene, missing):
    name = "images.txt" if missing == "images" else "points3D.txt"
    (small_scene / "sparse" / "0" / name).write_text("")
    output = small_scene / "out.npz"
    with pytest.raises(ValueError, match="scene must contain images and sparse points"):
        train_colmap(small_scene, output, steps=0, capacity=2, backend="reference")
    assert not output.exists()


@pytest.mark.parametrize("capacity,initial_points", [(2, 0), (1, 2), (4, 3)])
def test_initial_points_must_fit_cloud_and_capacity(small_scene, capacity, initial_points):
    with pytest.raises(ValueError, match="initial_points must fit"):
        train_colmap(
            small_scene,
            small_scene / "out.npz",
            steps=0,
            capacity=capacity,
            initial_points=initial_points,
            backend="reference",
        )


@pytest.mark.parametrize(
    "backend,error",
    [("reference", "tile capacity overflow"), ("cute", "visibility pair capacity overflow")],
)
def test_reference_trainer_aborts_on_backend_overflow(small_scene, backend, error):
    if backend == "cute" and (
        jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None
    ):
        pytest.skip("CuTe test requires JAX CUDA and NVIDIA CUTLASS DSL")
    output = small_scene / "overflow.npz"
    with pytest.raises(RuntimeError, match=error):
        train_colmap(
            small_scene,
            output,
            steps=1,
            capacity=2,
            initial_points=2,
            downsample=1,
            backend=backend,
            densify_every=0,
            cluster_size=2,
            tile_size=4,
            sh_degree=0,
            max_gaussians_per_tile=1,
            max_visibility_pairs=1,
        )
    assert not output.exists()


def test_opacity_reset_runs_at_step_3000_without_densification(small_scene, monkeypatch):
    from jaxgs.training import reference_trainer

    resets = []
    original = reference_trainer.reset_opacity

    def reset(pool, state):
        # Learning rate zero keeps alpha at its seeded value until this reset;
        # moments still accumulate, so the scheduled cleanup is observable.
        np.testing.assert_allclose(jax.nn.sigmoid(pool.opacity[pool.alive]), 0.1, atol=1e-7)
        assert np.any(np.asarray(state.v.opacity[pool.alive]) > 0)
        updated, cleared = original(pool, state)
        np.testing.assert_array_equal(cleared.m.opacity[pool.alive], 0)
        np.testing.assert_array_equal(cleared.v.opacity[pool.alive], 0)
        for field in ("xyz", "log_scale", "rotation", "sh"):
            np.testing.assert_array_equal(getattr(cleared.m, field), getattr(state.m, field))
            np.testing.assert_array_equal(getattr(cleared.v, field), getattr(state.v, field))
        resets.append(int(pool.n_active))
        return updated, cleared

    monkeypatch.setattr(reference_trainer, "reset_opacity", reset)
    output = small_scene / "reset.npz"
    pool = train_colmap(
        small_scene,
        output,
        steps=3001,
        capacity=2,
        initial_points=1,
        downsample=1,
        backend="reference",
        densify_every=0,
        learning_rate=0.0,
        cluster_size=2,
        tile_size=4,
        sh_degree=0,
    )
    assert resets == [1] and int(pool.n_active) == 1
    np.testing.assert_allclose(jax.nn.sigmoid(pool.opacity[pool.alive]), 0.01, atol=1e-7)
    with np.load(output) as saved:
        np.testing.assert_array_equal(saved["opacity"], pool.opacity)


@pytest.mark.filterwarnings(
    "ignore:.*found in sys.modules after import of package.*:RuntimeWarning"
)
def test_reference_module_cli_trains_and_saves(small_scene, monkeypatch, capsys):
    output = small_scene / "gaussians.ply"
    monkeypatch.chdir(small_scene)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jaxgs-train-reference",
            str(small_scene),
            "--steps",
            "3",
            "--capacity",
            "4",
            "--initial-points",
            "2",
            "--downsample",
            "1",
            "--backend",
            "reference",
            "--densify-every",
            "0",
            "--cluster-size",
            "2",
            "--max-gaussians-per-tile",
            "4",
            "--max-visibility-pairs",
            "16",
            "--tile-size",
            "4",
            "--sh-degree",
            "1",
        ],
    )
    runpy.run_module("jaxgs.training.reference_trainer", run_name="__main__")
    saved = load_gaussians(output)
    assert saved.xyz.shape == (2, 3) and saved.sh.shape == (2, 4, 3)
    assert int(saved.n_active) == 2
    assert np.isfinite(saved.opacity).all()
    printed = capsys.readouterr().out
    assert "step 1:" in printed and "step 3:" in printed and "step 2:" not in printed


def test_training_reserves_free_slots_when_points_exceed_capacity(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (tmp_path / "images").mkdir()
    (sparse / "cameras.txt").write_text("1 PINHOLE 4 4 4 4 2 2\n")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.png\n\n")
    (sparse / "points3D.txt").write_text(
        "1 0 0 2 255 0 0 0.1\n2 1 0 2 255 0 0 0.1\n3 0 1 2 255 0 0 0.1\n4 1 1 2 255 0 0 0.1\n"
    )
    pool = train_colmap(tmp_path, tmp_path / "seeded.npz", steps=0, capacity=2, sh_degree=0)
    assert int(pool.n_active) == 1
    np.testing.assert_array_equal(pool.free_mask, [False, True])


@pytest.mark.parametrize("backend", ["reference", "cute"])
def test_colmap_training_densifies_and_saves(tmp_path, backend):
    if backend == "cute" and (
        jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None
    ):
        pytest.skip("CuTe test requires JAX CUDA and NVIDIA CUTLASS DSL")
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    images = tmp_path / "images"
    images.mkdir()
    (sparse / "cameras.txt").write_text("1 PINHOLE 4 4 4 4 2 2\n")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.png\n\n")
    (sparse / "points3D.txt").write_text("1 0 0 2 255 0 0 0.1\n")
    target = np.zeros((4, 4, 3), np.uint8)
    target[1, 2] = [255, 255, 255]
    Image.fromarray(target).save(images / "frame.png")

    output = tmp_path / "gaussians.npz"
    pool = train_colmap(
        tmp_path,
        output,
        steps=2,
        capacity=2,
        downsample=1,
        backend=backend,
        densify_every=1,
        max_new=1,
        cluster_size=2,
        max_gaussians_per_tile=2,
        tile_size=4,
        sh_degree=0,
    )
    assert int(pool.n_active) == 2
    assert output.exists()
    with np.load(output) as saved:
        assert saved["xyz"].shape == (2, 3)
        assert saved["sh"].shape == (2, 1, 3)
        assert np.all(np.isfinite(saved["xyz"]))
        np.testing.assert_array_equal(saved["alive"], [True, True])
        np.testing.assert_array_equal(saved["free_mask"], [False, False])
        assert int(saved["n_active"]) == 2
