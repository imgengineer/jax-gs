import importlib.util
import json
import runpy
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from jaxgs.config import load_config
from jaxgs.io_manager.checkpoint import load_pool, save_pool
from jaxgs.training.trainer import load_training_frames, train


@pytest.fixture
def scene(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (tmp_path / "images").mkdir()
    (sparse / "cameras.txt").write_text("1 PINHOLE 32 16 24 24 16 8\n")
    records = []
    for i in range(3):
        records.append(f"{i + 1} 1 0 0 0 {(i - 1) * 0.1} 0 0 1 frame{i}.png\n\n")
        image = np.full((16, 32, 3), 32 + i * 16, np.uint8)
        Image.fromarray(image).save(tmp_path / "images" / f"frame{i}.png")
    (sparse / "images.txt").write_text("".join(records))
    rng = np.random.default_rng(5)
    points = rng.uniform([-0.3, -0.2, 2], [0.3, 0.2, 3], (129, 3))
    (sparse / "points3D.txt").write_text(
        "".join(f"{i + 1} {x} {y} {z} 96 64 32 0.1\n" for i, (x, y, z) in enumerate(points))
    )
    return tmp_path


def test_training_split_only_applies_in_eval_mode(scene):
    model = load_config().model
    assert len(load_training_frames(scene, model)) == 3
    evaluated = replace(model, eval=True)
    assert [f.image_path.stem for f in load_training_frames(scene, evaluated)] == [
        "frame1",
        "frame2",
    ]
    split = scene / "train_test_split.json"
    split.write_text(json.dumps({"train": ["frame0", "frame2.png"]}))
    frames = load_training_frames(scene, evaluated)
    assert [f.image_path.stem for f in frames] == ["frame0", "frame2"]
    assert all(f.camera.near == 0.2 for f in frames)
    assert len(load_training_frames(scene, model)) == 3
    split.write_text('{"train": []}')
    with pytest.raises(ValueError, match="no training images"):
        load_training_frames(scene, evaluated)


def test_training_rejects_incomplete_epoch_and_invalid_cloud(scene):
    output = scene / "out.npz"
    with pytest.raises(ValueError, match="one complete epoch"):
        train(scene, output, iterations=1)
    with pytest.raises(ValueError, match="cluster padding exceeds capacity"):
        train(scene, output, iterations=3, target_points=129)
    (scene / "sparse" / "0" / "points3D.txt").write_text("")
    with pytest.raises(ValueError, match="initial cloud is empty"):
        train(scene, output, iterations=3)
    assert not output.exists()


@pytest.mark.parametrize(
    "filename,report_name",
    [
        ("model.npz", "model.json"),
        ("model.json", "model.report.json"),
        ("model.checkpoint", "model.json"),
    ],
)
def test_training_report_preserves_checkpoint(tmp_path, filename, report_name):
    from jaxgs.scene.point import create_pool, seed_pool
    from jaxgs.training.trainer import _write_training_report

    settings = load_config()
    settings = replace(
        settings,
        densify=replace(settings.densify, target_primitives=2),
        runtime=replace(settings.runtime, max_gaussians=2),
    )
    pool = seed_pool(
        create_pool(settings.capacity), jnp.array([[0.1, 0.2, 2.0]]), jnp.full((1, 3), 0.4)
    )
    metrics = {
        "scene": str(tmp_path),
        "image_shape": [8, 16],
        "training_images": 1,
        "actual_updates": 1,
        "initial_sparse_gaussians": 1,
        "initial_padded_gaussians": 1,
        "warmup_seconds": 0.0,
        "image_load_seconds": 0.0,
        "training_seconds": 0.0,
        "densify_until": 0,
        "history": [],
        "jit_cache_size": 1,
    }
    output = tmp_path / "output" / filename
    report = _write_training_report(output, pool, settings, metrics)
    for actual, expected in zip(
        jax.tree.leaves(load_pool(output)), jax.tree.leaves(pool), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert json.loads((output.parent / report_name).read_text()) == json.loads(json.dumps(report))
    assert sorted(path.name for path in output.parent.iterdir()) == sorted([filename, report_name])


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="production training requires JAX CUDA and CuTe",
)
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("mixed_resolution", [False, True])
def test_precompilation_preserves_original_training_buffers(scene, fail, mixed_resolution):
    from jaxgs.scene.cluster import world_cluster_bounds
    from jaxgs.scene.point import GaussianModel
    from jaxgs.training import trainer
    from jaxgs.training.state import TrainingState
    from jaxgs.training.step import bind_train_step

    settings = load_config()
    settings = replace(
        settings,
        model=replace(settings.model, sh_degree=1),
        densify=replace(settings.densify, target_primitives=256),
        runtime=replace(settings.runtime, max_gaussians=256, max_visibility_pairs=4096),
    )
    config = settings.capacity
    pool, state, initial_count, _ = trainer._initialize_pool(scene, config)
    stats = jnp.zeros((config.max_gaussians, 4), jnp.float32)
    training = TrainingState(GaussianModel(pool), state, stats)
    update = bind_train_step(training)
    originals = jax.tree.leaves((pool, state, stats))
    snapshots = [np.asarray(value).copy() for value in originals]
    pointers = [value.unsafe_buffer_pointer() for value in originals]
    frames = load_training_frames(scene, settings.model)
    views = [(frame.camera, jnp.asarray(frame.load_rgb())) for frame in frames]
    if mixed_resolution:
        camera = views[1][0]
        views[1] = (
            camera.replace(
                width=48,
                height=24,
                fx=camera.fx * 1.5,
                fy=camera.fy * 1.5,
                cx=camera.cx * 1.5,
                cy=camera.cy * 1.5,
            ),
            jnp.full((24, 48, 3), 64, jnp.uint8),
        )

    args = (
        world_cluster_bounds(pool, config.cluster_size),
        views,
        jax.random.key(0),
        initial_count,
        jnp.array(1.0, jnp.float32),
        settings,
    )
    if fail:

        def interrupted_update(*args):
            jax.block_until_ready(update(*args))
            raise RuntimeError("warmup interrupted")

        with pytest.raises(RuntimeError, match="warmup interrupted"):
            trainer._precompile_training(training, interrupted_update, *args)
    else:
        trainer._precompile_training(training, update, *args)

    assert all(not value.is_deleted() for value in originals)
    assert [value.unsafe_buffer_pointer() for value in originals] == pointers
    for value, snapshot in zip(originals, snapshots, strict=True):
        np.testing.assert_array_equal(value, snapshot)
    restored = jax.tree.leaves(
        (training.model.as_pool(), training.adam.get_value(), training.fragments.get_value())
    )
    assert all(value is original for value, original in zip(restored, originals, strict=True))
    if not fail:
        cache_size = update.jitted_fn._cache_size()
        assert cache_size == 4 * (2 if mixed_resolution else 1)
        training.adam.set_value(state.replace(v=jax.tree.map(jnp.copy, state.v)))
        for camera, target in views:
            jax.block_until_ready(
                update(
                    args[0],
                    camera,
                    target,
                    jnp.array(1, jnp.int32),
                    args[4],
                    config,
                    0,
                    False,
                    settings.optimization.position_lr_max_steps,
                    jnp.array(False),
                    jnp.array(0, jnp.int32),
                    settings.runtime.optimizer,
                    settings.optimization,
                )
            )
        assert update.jitted_fn._cache_size() == cache_size


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="production training requires JAX CUDA and CuTe",
)
def test_complete_training_schedule_and_checkpoint(scene, monkeypatch):
    from jaxgs.training import trainer

    config = load_config()
    config = replace(
        config,
        model=replace(config.model, sh_degree=1),
        densify=replace(config.densify, target_primitives=512),
        runtime=replace(config.runtime, max_gaussians=512, max_visibility_pairs=4096),
    )
    resets = []
    real_decay = trainer.decay_opacity

    def decay(pool, state):
        result = real_decay(pool, state)
        resets.append(int(pool.n_active))
        for value in jax.tree.leaves(result[1]):
            np.testing.assert_array_equal(value, 0)
        return result

    monkeypatch.setattr(trainer, "decay_opacity", decay)
    report = train(
        scene,
        scene / "output" / "pool.npz",
        settings=config,
        iterations=63,
        images="images",
        seed=7,
        optimizer="optax",
        pair_capacity=4096,
    )
    assert report["actual_updates"] == 63
    assert report["training_images"] == 3 and report["densify_until"] == 11
    assert report["initial_sparse_gaussians"] == 129 and report["initial_padded_gaussians"] == 256
    assert report["config"]["optimization"]["position_lr_max_steps"] == 30000
    assert report["config"]["densify"]["densification_interval"] == 5
    assert len(resets) == 2  # warmup and epoch 10
    history = report["history"]
    assert any(row["born"] > 0 for row in history)
    assert all(row["born"] == 0 for row in history if row["epoch"] not in (5, 10))
    assert all(np.isfinite(row["loss"]) and row["peak_pairs"] <= 4096 for row in history)
    pool = load_pool(scene / "output" / "pool.npz")
    assert pool.xyz.shape == (512, 3) and pool.sh.shape == (512, 4, 3)
    assert int(pool.n_active) == report["final_gaussians"]
    np.testing.assert_array_equal(pool.free_mask, ~pool.alive)
    assert int(pool.alive.sum()) == int(pool.n_active)
    # Shared checkpoint helpers preserve occupancy and every parameter, even
    # when the caller chooses a suffix other than .npz.
    copy = scene / "copy.checkpoint"
    save_pool(copy, pool)
    for actual, expected in zip(
        jax.tree.leaves(load_pool(copy)), jax.tree.leaves(pool), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert json.loads((scene / "output" / "pool.json").read_text())["config"] == json.loads(
        json.dumps(report["config"])
    )


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="production training requires JAX CUDA and CuTe",
)
def test_training_overflow_aborts_before_saving(scene):
    config = load_config()
    config = replace(
        config,
        model=replace(config.model, sh_degree=0),
        optimization=replace(config.optimization, iterations=3),
        densify=replace(config.densify, target_primitives=256),
        runtime=replace(config.runtime, max_gaussians=256, max_visibility_pairs=1),
    )
    output = scene / "overflow.npz"
    with pytest.raises(RuntimeError, match="pairs exceed 1"):
        train(scene, output, settings=config)
    assert not output.exists() and not output.with_suffix(".json").exists()


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="production training requires JAX CUDA and CuTe",
)
@pytest.mark.parametrize("loss", [float("nan"), float("inf")])
def test_nonfinite_loss_preserves_existing_checkpoint(scene, monkeypatch, loss):
    from jaxgs.training import trainer

    config = load_config()
    config = replace(
        config,
        model=replace(config.model, sh_degree=0),
        optimization=replace(config.optimization, iterations=3),
        densify=replace(config.densify, target_primitives=256),
        runtime=replace(config.runtime, max_gaussians=256, max_visibility_pairs=4096),
    )
    original_bind = trainer.bind_train_step

    def bind_nonfinite_step(training):
        original_step = original_bind(training)

        def nonfinite_step(*args, **kwargs):
            _, overflow, peak = original_step(*args, **kwargs)
            return jnp.array(loss), overflow, peak

        return nonfinite_step

    monkeypatch.setattr(trainer, "bind_train_step", bind_nonfinite_step)
    output = scene / "existing.npz"
    output.write_bytes(b"previous checkpoint")
    with pytest.raises(RuntimeError, match="non-finite training loss"):
        train(scene, output, settings=config)
    assert output.read_bytes() == b"previous checkpoint"
    assert not output.with_suffix(".json").exists()


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="production training requires JAX CUDA and CuTe",
)
@pytest.mark.filterwarnings(
    "ignore:.*found in sys.modules after import of package.*:RuntimeWarning"
)
def test_production_module_cli_applies_config_and_overrides(scene, monkeypatch):
    path = scene / "cli.toml"
    path.write_text(
        "[model]\nsh_degree=0\n[optimization]\niterations=6\n"
        "[densify]\ntarget_primitives=256\n"
        "[runtime]\nmax_gaussians=256\nmax_visibility_pairs=4096\n"
    )
    output = scene / "cli.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jaxgs-train",
            str(scene),
            "--config",
            str(path),
            "--output",
            str(output),
            "--iterations",
            "3",
        ],
    )
    runpy.run_module("jaxgs.training.trainer", run_name="__main__")
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["actual_updates"] == 3
    assert report["config"]["optimization"]["iterations"] == 3
    assert report["config"]["optimization"]["position_lr_max_steps"] == 30000
    assert report["config"]["model"]["sh_degree"] == 0
    pool = load_pool(output)
    assert pool.xyz.shape == (256, 3) and pool.sh.shape == (256, 1, 3)
    assert int(pool.n_active) == 256
