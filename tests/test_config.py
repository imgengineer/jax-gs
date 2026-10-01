import ast
import os
from dataclasses import asdict, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxgs import create_gaussians, seed_gaussians
from jaxgs.config import CapacityConfig, load_config
from jaxgs.training.optimizer import create_adam_state, optax_update


def test_packaged_defaults_match_native_source():
    source = Path(os.environ.get("LITEGS_ROOT", Path(__file__).resolve().parents[2] / "LiteGS"))
    source = source / "litegs" / "arguments.py"
    if not source.exists():
        pytest.skip("set LITEGS_ROOT to compare against a native checkout")
    classes = {
        node.name: node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef)
    }
    config = load_config()
    for section, native_name in (
        ("model", "ModelParams"),
        ("pipeline", "PipelineParams"),
        ("optimization", "OptimizationParams"),
        ("densify", "DensifyParams"),
    ):
        native = {
            node.targets[0].id.lstrip("_"): ast.literal_eval(node.value)
            for node in classes[native_name].body
            if isinstance(node, ast.Assign)
        }
        if section == "model":
            del native["source_path"], native["model_path"]  # CLI positional paths
        assert asdict(getattr(config, section)) == native


def test_default_schedule_and_capacity():
    config = load_config()
    assert config.model.images == "images" and config.model.resolution == -1
    assert not config.model.eval and config.runtime.optimizer == "optax"
    assert config.optimization.iterations == config.optimization.position_lr_max_steps == 30000
    assert config.densify.densification_interval == 5
    assert config.densify.end_epoch(155) == 121
    assert replace(config.densify, densify_until=17).end_epoch(155) == 17
    capacity = config.capacity
    assert (capacity.cluster_size, capacity.raster_tile_height, capacity.tile_size) == (128, 8, 16)
    assert capacity.sh_dim == 16 and capacity.visibility_capacity == 8000000
    assert capacity.max_gaussians == config.densify.target_primitives == 1000000


def test_toml_overrides_and_cli_forwarding(tmp_path, monkeypatch):
    from jaxgs.training import trainer

    path = tmp_path / "config.toml"
    path.write_text(
        '[model]\nimages="pyramid"\n[optimization]\niterations=120\n'
        "position_lr_max_steps=300\n[densify]\ndensification_interval=2\n"
    )
    config = load_config(path)
    assert config.model.images == "pyramid"
    assert config.optimization.iterations == 120
    assert config.optimization.position_lr_max_steps == 300
    assert config.optimization.position_lr_final == 1.6e-6
    calls = []
    monkeypatch.setattr(trainer, "train", lambda *a, **kw: calls.append((a, kw)))
    trainer.main(
        [str(tmp_path), "--config", str(path), "--iterations", "42", "--optimizer", "cute"]
    )
    args, kwargs = calls[0]
    assert args == (tmp_path, Path("gaussians.npz"))
    assert kwargs["settings"] == config
    assert kwargs["iterations"] == 42 and kwargs["optimizer"] == "cute"
    assert kwargs["images"] is None  # CLI defaults do not clobber TOML values.
    path.write_text('[runtime]\noptimizer="muon"\n')
    assert load_config(path).runtime.optimizer == "muon"
    trainer.main([str(tmp_path), "--optimizer", "muon"])
    assert calls[1][1]["optimizer"] == "muon"


@pytest.mark.parametrize(
    "text,match",
    [
        ("[unknown]\nx=1", "unknown configuration section"),
        ("[optimization]\niteratons=1", "unknown optimization settings"),
        ("[optimization]\niterations=0", "must be positive"),
        ("[optimization]\nposition_lr_final=0", "both be positive or both zero"),
        ("[optimization]\nfeature_lr=-1", "finite and nonnegative"),
        ("[optimization]\nposition_lr_init=nan", "finite and nonnegative"),
        ("[densify]\ndensify_from=-2", "invalid densification"),
        ("[model]\nresolution=0", "resolution"),
        ("[runtime]\nmax_gaussians=128", "exceeds"),
        ("[pipeline]\ntile_size=[4,4]", "production tile_size"),
        ('[runtime]\noptimizer="adam"', "optimizer must"),
        ("[pipeline]\nenable_depth=true", "requires enable_depth=False"),
    ],
)
def test_invalid_training_settings_fail_before_compilation(tmp_path, text, match):
    path = tmp_path / "invalid.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_config(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_gaussians": 0},
        {"sh_degree": 4},
        {"max_visibility_pairs": 0},
        {"tile_height": 0},
    ],
)
def test_invalid_static_capacity(overrides):
    with pytest.raises(ValueError):
        CapacityConfig(**overrides)


@pytest.mark.parametrize("step", [0, 15000, 30000, 45000])
def test_default_position_schedule_reaches_native_endpoints(step):
    pool = seed_gaussians(
        create_gaussians(CapacityConfig(1, sh_degree=1)), jnp.ones((1, 3)), jnp.full((1, 3), 0.5)
    )
    grads = tuple(
        jnp.ones_like(getattr(pool, name))
        for name in ("xyz", "log_scale", "rotation", "opacity", "sh")
    )
    updated, _ = optax_update(pool, create_adam_state(pool), grads, pool.alive, step, 2.0)
    # Independent closed-form first sparse Adam step at four schedule positions.
    expected_rate = [1.6e-4, 1.6e-5, 1.6e-6, 1.6e-6][step // 15000]
    np.testing.assert_allclose(updated.xyz, 1 - 2 * expected_rate * np.sqrt(10), atol=1e-7)


def test_each_property_uses_configured_rate_and_zero_xyz_schedule():
    pool = seed_gaussians(
        create_gaussians(CapacityConfig(1, sh_degree=1)), jnp.ones((1, 3)), jnp.full((1, 3), 0.5)
    )
    op = replace(
        load_config().optimization,
        position_lr_init=0.0,
        position_lr_final=0.0,
        feature_lr=0.004,
        opacity_lr=0.03,
        scaling_lr=0.002,
        rotation_lr=0.007,
    )
    fields = ("xyz", "log_scale", "rotation", "opacity", "sh")
    grads = tuple(jnp.ones_like(getattr(pool, name)) for name in fields)
    updated, _ = optax_update(
        pool, create_adam_state(pool), grads, pool.alive, 15000, 2.0, optimization=op
    )
    for name, rate in zip(fields[:4], [0.0, 0.002, 0.007, 0.03], strict=True):
        np.testing.assert_allclose(
            getattr(updated, name), getattr(pool, name) - rate * np.sqrt(10), atol=2e-7
        )
    np.testing.assert_allclose(updated.sh[:, 0], -0.004 * np.sqrt(10), atol=1e-7)
    np.testing.assert_allclose(updated.sh[:, 1:], -0.0004 * np.sqrt(10), atol=1e-7)
