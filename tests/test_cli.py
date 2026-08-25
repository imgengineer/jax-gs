import argparse
from types import SimpleNamespace
from typing import cast

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_gs.cli as cli_module
from jax_gs.checkpoints import save_distributed_checkpoint
from jax_gs.config import ModelConfig, RasterizationConfig, TrainConfig
from jax_gs.data import ColmapScene
from jax_gs.exporter import load_ply_to_splats
from jax_gs.optimizers import create_optimizer
from jax_gs.strategy import DefaultStrategy
from jax_gs.training import TrainingSafetyState


def _stack_graphs(*graphs):
    graphdef, first_state = nnx.split(graphs[0])
    states = [first_state, *(nnx.split(graph)[1] for graph in graphs[1:])]
    return nnx.merge(
        graphdef, jax.tree.map(lambda *values: jnp.stack(values), *states)
    )


def _write_distributed_cli_checkpoint(
    tmp_path, config, *, include_scene: bool = True
):
    models = []
    optimizers = []
    strategies = []
    safety_states = []
    masks = np.asarray([[False, True, False], [True, False, True]])
    for rank in range(2):
        model = cli_module.GaussianModel.empty(
            config.model,
            physical_capacity=3,
            appearance_feature_dim=32 if config.app_opt else None,
        )
        model.active_mask[...] = jnp.asarray(masks[rank])
        model.means[...] = jnp.asarray(
            [
                [rank * 10.0 + slot, rank + slot, 3.0]
                for slot in range(3)
            ],
            jnp.float32,
        )
        if config.app_opt:
            model.features[...] = rank + jnp.arange(
                3 * 32, dtype=jnp.float32
            ).reshape(3, 32)
            model.colors[...] = jnp.asarray(
                [[-1.0, 0.0, 1.0], [0.25, -0.5, 0.75], [1.0, 0.5, -1.0]]
            ) + rank
        else:
            model.sh0[...] = rank + jnp.arange(
                9, dtype=jnp.float32
            ).reshape(3, 1, 3)
            model.sh_rest[...] = rank + jnp.arange(
                model.sh_rest[...].size, dtype=jnp.float32
            ).reshape(model.sh_rest[...].shape)
        models.append(model)
        optimizers.append(
            create_optimizer(
                model,
                config.optimizer,
                batch_size=config.data.batch_size,
                world_size=2,
                scene_scale=1.0,
            )
        )
        strategies.append(
            DefaultStrategy(config.strategy).initialize_state(model.capacity)
        )
        safety_states.append(TrainingSafetyState())

    stacked_model = _stack_graphs(*models)
    save_kwargs = {}
    if config.app_opt:
        names = ("a.png", "b.png", "c.png")
        modules = [
            cli_module.AppearanceOptModule(
                len(names),
                32,
                config.app_embed_dim,
                config.model.sh_degree,
                rngs=nnx.Rngs(7),
            )
            for _ in range(2)
        ]
        appearance_optimizers = [
            cli_module.create_appearance_optimizer(module, config)
            for module in modules
        ]
        save_kwargs = {
            "appearance_module": _stack_graphs(*modules),
            "appearance_optimizer": _stack_graphs(*appearance_optimizers),
            "appearance_image_names": names,
        }
    checkpoint = save_distributed_checkpoint(
        tmp_path,
        stacked_model,
        _stack_graphs(*optimizers),
        _stack_graphs(*strategies),
        _stack_graphs(*safety_states),
        step=0,
        config=config,
        scene_transform=(
            np.eye(4, dtype=np.float32) if include_scene else None
        ),
        scene_scale=1.0 if include_scene else None,
        **save_kwargs,
    )
    means = np.asarray(stacked_model.means[...])
    expected_means = np.concatenate(
        [means[rank][masks[rank]] for rank in range(2)], axis=0
    )
    return checkpoint, expected_means


def test_train_cli_exposes_2dgs_and_upstream_regularizers(monkeypatch):
    captured = {}

    def fake_train(config, *, resume_from=None):
        captured["config"] = config
        captured["resume_from"] = resume_from
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    args = cli_module.build_parser().parse_args(
        [
            "train",
            "--model-type",
            "2dgs",
            "--normal-loss",
            "--normal-lambda",
            "0.1",
            "--normal-start-iter",
            "11",
            "--dist-loss",
            "--dist-lambda",
            "0.2",
            "--dist-start-iter",
            "7",
        ]
    )

    args.func(args)

    config = captured["config"]
    assert config.model_type == "2dgs"
    assert config.model.initial_scale == 1.0
    assert config.rasterizer.near_plane == 0.2
    assert config.rasterizer.far_plane == 200.0
    assert config.strategy.prune_opacity == 0.05
    assert config.strategy.key_for_gradient == "gradient_2dgs"
    assert config.normal_loss
    assert config.normal_lambda == 0.1
    assert config.normal_start_iter == 11
    assert config.dist_loss
    assert config.dist_lambda == 0.2
    assert config.dist_start_iter == 7
    assert captured["resume_from"] is None


def test_train_cli_uses_native_defaults_when_supported(monkeypatch):
    captured = {}

    def fake_train(config, *, resume_from=None):
        captured["config"] = config
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    monkeypatch.setattr(
        cli_module, "_native_training_defaults_available", lambda: True
    )

    args = cli_module.build_parser().parse_args(["train"])
    args.func(args)

    rasterizer = captured["config"].rasterizer
    assert rasterizer.backend == "intersections"
    assert rasterizer.projection_backend == "cute"
    assert rasterizer.compositor_backend == "cute"
    assert rasterizer.intersection_backend == "cute"
    assert rasterizer.intersection_mode == "accutile"


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--model-type", "2dgs"],
        ["train", "--with-ut"],
        ["train", "--distributed"],
        ["train", "--intersection-backend", "jax"],
        ["train", "--config", "scene.json"],
        ["train", "--resume", "checkpoint"],
    ],
)
def test_train_cli_keeps_jax_defaults_when_native_mode_does_not_apply(
    monkeypatch, argv
):
    captured = {}

    def fake_train(config, *, resume_from=None, distributed=False):
        captured["config"] = config
        captured["distributed"] = distributed
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    monkeypatch.setattr(
        cli_module, "_native_training_defaults_available", lambda: True
    )
    monkeypatch.setattr(
        cli_module,
        "load_checkpoint_config",
        lambda _path: TrainConfig(
            rasterizer=RasterizationConfig(intersection_backend="jax")
        ),
    )
    monkeypatch.setattr(
        TrainConfig,
        "load",
        lambda _path: TrainConfig(
            rasterizer=RasterizationConfig(intersection_backend="jax")
        ),
    )

    args = cli_module.build_parser().parse_args(argv)
    args.func(args)

    rasterizer = captured["config"].rasterizer
    assert rasterizer.compositor_backend == "jax"
    assert rasterizer.intersection_backend in {"auto", "jax"}


def test_train_cli_keeps_jax_defaults_when_native_mode_is_unavailable(
    monkeypatch,
):
    captured = {}

    def fake_train(config, *, resume_from=None):
        captured["config"] = config
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    monkeypatch.setattr(
        cli_module, "_native_training_defaults_available", lambda: False
    )

    args = cli_module.build_parser().parse_args(["train"])
    args.func(args)

    rasterizer = captured["config"].rasterizer
    assert rasterizer.compositor_backend == "jax"
    assert rasterizer.intersection_backend == "auto"


def test_train_cli_exposes_packed_sparse_and_visible_adam(monkeypatch):
    captured = []

    def fake_train(config, *, resume_from=None):
        captured.append((config, resume_from))
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    parser = cli_module.build_parser()

    sparse_args = parser.parse_args(["train", "--packed", "--sparse-grad"])
    sparse_args.func(sparse_args)
    visible_args = parser.parse_args(["train", "--visible-adam"])
    visible_args.func(visible_args)
    pallas_args = parser.parse_args(
        ["train", "--compositor-backend", "pallas"]
    )
    pallas_args.func(pallas_args)
    cute_compositor_args = parser.parse_args(
        ["train", "--compositor-backend", "cute"]
    )
    cute_compositor_args.func(cute_compositor_args)
    cute_intersection_args = parser.parse_args(
        ["train", "--intersection-backend", "cute"]
    )
    cute_intersection_args.func(cute_intersection_args)

    sparse_config, sparse_resume = captured[0]
    assert sparse_config.packed
    assert sparse_config.sparse_grad
    assert not sparse_config.visible_adam
    assert sparse_resume is None
    visible_config, visible_resume = captured[1]
    assert not visible_config.packed
    assert not visible_config.sparse_grad
    assert visible_config.visible_adam
    assert visible_resume is None
    pallas_config, pallas_resume = captured[2]
    assert pallas_config.rasterizer.compositor_backend == "pallas"
    assert pallas_resume is None
    cute_compositor_config, cute_compositor_resume = captured[3]
    assert cute_compositor_config.rasterizer.compositor_backend == "cute"
    assert cute_compositor_resume is None
    cute_intersection_config, cute_intersection_resume = captured[4]
    assert cute_intersection_config.rasterizer.intersection_backend == "cute"
    assert cute_intersection_resume is None


def test_train_cli_exposes_target_primitives(monkeypatch):
    captured = {}

    def fake_train(config, *, resume_from=None):
        captured["config"] = config
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    args = cli_module.build_parser().parse_args(
        ["train", "--capacity", "32", "--target-primitives", "24"]
    )

    args.func(args)

    assert captured["config"].strategy.target_primitives == 24


def test_train_cli_rejects_distributed_target_primitives(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "train",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("train must not be called")
        ),
    )
    args = cli_module.build_parser().parse_args(
        [
            "train",
            "--capacity",
            "32",
            "--target-primitives",
            "24",
            "--distributed",
        ]
    )

    with pytest.raises(NotImplementedError, match="target_primitives"):
        args.func(args)


def test_train_cli_forwards_local_device_distributed_mode(monkeypatch):
    captured = {}

    def fake_train(config, *, resume_from=None, distributed=False):
        captured["config"] = config
        captured["resume_from"] = resume_from
        captured["distributed"] = distributed
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    args = cli_module.build_parser().parse_args(["train", "--distributed"])

    args.func(args)

    assert captured["resume_from"] is None
    assert captured["distributed"] is True


def test_train_cli_uses_current_main_mcmc_profile(monkeypatch):
    captured = {}

    def fake_train(config, *, resume_from=None):
        captured["config"] = config
        captured["resume_from"] = resume_from
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    args = cli_module.build_parser().parse_args(["train", "--strategy", "mcmc"])

    args.func(args)

    config = captured["config"]
    assert config.model.initial_opacity == 0.5
    assert config.model.initial_scale == 0.1
    assert config.opacity_reg == 0.01
    assert config.scale_reg == 0.01
    assert config.strategy.kind == "mcmc"
    assert config.strategy.verbose
    assert captured["resume_from"] is None


def test_train_cli_exposes_3d_regularization_weights(monkeypatch):
    captured = {}

    def fake_train(config, *, resume_from=None):
        captured["config"] = config
        captured["resume_from"] = resume_from
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    args = cli_module.build_parser().parse_args(
        [
            "train",
            "--opacity-reg",
            "0.01",
            "--scale-reg",
            "0.02",
        ]
    )

    args.func(args)

    assert captured["config"].opacity_reg == 0.01
    assert captured["config"].scale_reg == 0.02
    assert captured["resume_from"] is None


def test_train_cli_exposes_current_main_scene_normalization(monkeypatch):
    captured = {}

    def fake_train(config, *, resume_from=None):
        captured["config"] = config
        captured["resume_from"] = resume_from
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    args = cli_module.build_parser().parse_args(
        ["train", "--global-scale", "2.5", "--no-normalize-world-space"]
    )

    args.func(args)

    assert captured["config"].global_scale == 2.5
    assert not captured["config"].normalize_world_space
    assert captured["resume_from"] is None


def test_train_cli_exposes_pose_optimization_overrides(monkeypatch):
    captured = {}

    def fake_train(config, *, resume_from=None):
        captured["config"] = config
        captured["resume_from"] = resume_from
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    args = cli_module.build_parser().parse_args(
        [
            "train",
            "--pose-opt",
            "--pose-opt-lr",
            "2e-5",
            "--pose-opt-reg",
            "3e-6",
            "--pose-noise",
            "0.01",
        ]
    )

    args.func(args)

    config = captured["config"]
    assert config.pose_opt
    assert config.pose_opt_lr == 2.0e-5
    assert config.pose_opt_reg == 3.0e-6
    assert config.pose_noise == 0.01
    assert captured["resume_from"] is None


def test_train_cli_exposes_appearance_optimization_overrides(monkeypatch):
    captured = {}

    def fake_train(config, *, resume_from=None):
        captured["config"] = config
        captured["resume_from"] = resume_from
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    args = cli_module.build_parser().parse_args(
        [
            "train",
            "--app-opt",
            "--app-embed-dim",
            "24",
            "--app-opt-lr",
            "2e-3",
            "--app-opt-reg",
            "3e-6",
        ]
    )

    args.func(args)

    config = captured["config"]
    assert config.app_opt
    assert config.app_embed_dim == 24
    assert config.app_opt_lr == 2.0e-3
    assert config.app_opt_reg == 3.0e-6
    assert captured["resume_from"] is None


def test_checkpoint_loader_reconstructs_appearance_graph_and_manifest(
    monkeypatch, tmp_path
):
    config = TrainConfig(
        app_opt=True,
        app_embed_dim=4,
        model=ModelConfig(capacity=4, bucket_min_capacity=2, sh_degree=2),
    )
    captured = {}
    monkeypatch.setattr(cli_module, "load_checkpoint_config", lambda _p: config)
    monkeypatch.setattr(cli_module, "load_checkpoint_storage_capacity", lambda _p: 2)
    monkeypatch.setattr(
        cli_module,
        "load_checkpoint_appearance_image_names",
        lambda _p: ("a.png", "b.png"),
    )

    def fake_restore(_checkpoint, model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return 7

    monkeypatch.setattr(cli_module, "restore_checkpoint", fake_restore)

    loaded_config, model, appearance, step = cli_module._load_training_objects(
        tmp_path
    )

    assert loaded_config is config
    assert model.has_appearance
    assert appearance is captured["appearance_module"]
    assert appearance is not None
    assert isinstance(captured["appearance_optimizer"], nnx.Optimizer)
    assert captured["appearance_image_names"] == ("a.png", "b.png")
    assert appearance.embeds.embedding.shape == (2, 4)
    assert appearance.feature_dim == 32
    assert appearance.sh_degree == 2
    assert step == 7


def test_checkpoint_loader_materializes_distributed_inference_model(
    monkeypatch, tmp_path
):
    config = TrainConfig(
        app_opt=True,
        app_embed_dim=4,
        model=ModelConfig(capacity=3, bucket_min_capacity=3, sh_degree=2),
    )
    inference_config = ModelConfig(
        capacity=4, bucket_min_capacity=4, sh_degree=2
    )
    inference_model = cli_module.GaussianModel.empty(
        inference_config, appearance_feature_dim=32
    )
    captured = {}
    monkeypatch.setattr(cli_module, "load_checkpoint_config", lambda _p: config)
    monkeypatch.setattr(cli_module, "is_distributed_checkpoint", lambda _p: True)
    monkeypatch.setattr(
        cli_module,
        "load_checkpoint_appearance_image_names",
        lambda _p: ("a.png", "b.png"),
    )

    def fake_load(_checkpoint, source_config, **kwargs):
        captured["config"] = source_config
        captured.update(kwargs)
        return inference_model, 9

    monkeypatch.setattr(
        cli_module, "load_distributed_inference_checkpoint", fake_load
    )
    monkeypatch.setattr(
        cli_module,
        "load_checkpoint_storage_capacity",
        lambda _p: pytest.fail("generic capacity loader must not run"),
    )
    monkeypatch.setattr(
        cli_module,
        "restore_checkpoint",
        lambda *_a, **_k: pytest.fail("generic restore must not run"),
    )

    loaded_config, model, appearance, step = cli_module._load_training_objects(
        tmp_path
    )

    assert captured["config"] is config
    assert captured["appearance_module"] is appearance
    assert captured["appearance_image_names"] == ("a.png", "b.png")
    assert loaded_config.model.capacity == 4
    assert model is inference_model
    assert step == 9


@pytest.mark.parametrize("has_scene_metadata", [True, False])
def test_render_cli_uses_checkpoint_or_legacy_scene_transform(
    monkeypatch, tmp_path, has_scene_metadata: bool
):
    config = TrainConfig(
        model=ModelConfig(capacity=2, bucket_min_capacity=2, sh_degree=0)
    )
    model = cli_module.GaussianModel.empty(config.model)
    camtoworlds = np.broadcast_to(
        np.eye(4, dtype=np.float32), (2, 4, 4)
    ).copy()
    camtoworlds[:, 0, 3] = np.asarray([-2.0, 2.0])
    scene = SimpleNamespace(camtoworlds=camtoworlds)
    example = {
        "image": np.zeros((2, 3, 3), dtype=np.float32),
        "K": np.eye(3, dtype=np.float32),
        "w2c": np.linalg.inv(camtoworlds[1]),
        "image_name": "frame.png",
    }
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] *= 0.5
    matrix[0, 3] = 1.0
    captured = {}

    monkeypatch.setattr(
        cli_module,
        "_load_training_objects",
        lambda _path: (config, model, None, 7),
    )
    monkeypatch.setattr(
        cli_module, "load_colmap_scene", lambda *_args, **_kwargs: scene
    )
    monkeypatch.setattr(
        cli_module, "create_grain_dataset", lambda *_args, **_kwargs: [example]
    )
    monkeypatch.setattr(
        cli_module,
        "load_checkpoint_scene_transform",
        lambda _path: (matrix, 1.25) if has_scene_metadata else None,
    )
    monkeypatch.setattr(
        cli_module, "_check_evaluation_memory_budget", lambda *_args, **_kwargs: 0
    )

    def fake_render_step(_config, _width, _height):
        captured["compositor_backend"] = (
            _config.rasterizer.compositor_backend
        )

        def render(_model, viewmat, _K, _degree, **_kwargs):
            captured["viewmat"] = np.asarray(viewmat)
            return (
                jnp.zeros((2, 3, 3), jnp.float32),
                jnp.zeros((2, 3, 1), jnp.float32),
                jnp.zeros((1,), jnp.bool_),
                jnp.asarray(False),
            )

        return render

    monkeypatch.setattr(cli_module, "make_render_step", fake_render_step)
    monkeypatch.setattr(cli_module, "_save_render", lambda *_args: None)
    args = cli_module.build_parser().parse_args(
        [
            "render",
            str(tmp_path / "checkpoint"),
            "--data",
            "unused",
            "--compositor-backend",
            "pallas",
        ]
    )

    args.func(args)

    expected_transform = (
        cli_module.SceneTransform(matrix)
        if has_scene_metadata
        else cli_module._legacy_scene_transform(cast(ColmapScene, scene))
    )
    expected = expected_transform.world_to_camera(example["w2c"])
    np.testing.assert_allclose(captured["viewmat"], expected)
    assert captured["compositor_backend"] == "pallas"


@pytest.mark.parametrize("app_opt", [False, True])
def test_render_cli_materializes_real_distributed_checkpoint(
    monkeypatch, tmp_path, app_opt: bool
):
    config = TrainConfig(
        app_opt=app_opt,
        app_embed_dim=0,
        model=ModelConfig(capacity=3, bucket_min_capacity=3, sh_degree=1),
    )
    checkpoint, expected_means = _write_distributed_cli_checkpoint(
        tmp_path / ("render_appearance" if app_opt else "render_sh"), config
    )
    camtoworlds = np.broadcast_to(
        np.eye(4, dtype=np.float32), (2, 4, 4)
    ).copy()
    scene = SimpleNamespace(camtoworlds=camtoworlds)
    example = {
        "image": np.zeros((2, 3, 3), dtype=np.float32),
        "K": np.eye(3, dtype=np.float32),
        "w2c": np.eye(4, dtype=np.float32),
        "image_name": "frame.png",
    }
    captured = {}
    monkeypatch.setattr(
        cli_module, "load_colmap_scene", lambda *_args, **_kwargs: scene
    )
    monkeypatch.setattr(
        cli_module, "create_grain_dataset", lambda *_args, **_kwargs: [example]
    )

    def fake_memory_budget(_config, **kwargs):
        captured["memory_capacity"] = kwargs["physical_capacity"]
        return 0

    monkeypatch.setattr(
        cli_module, "_check_evaluation_memory_budget", fake_memory_budget
    )
    monkeypatch.setattr(cli_module, "_save_render", lambda *_args: None)

    def fake_render_step(_config, _width, _height):
        def render(model, _viewmat, _K, _degree, *, appearance_module=None):
            captured["model"] = model
            captured["appearance"] = appearance_module
            return (
                jnp.zeros((2, 3, 3), jnp.float32),
                jnp.zeros((2, 3, 1), jnp.float32),
                jnp.zeros((1,), jnp.bool_),
                jnp.asarray(False),
            )

        return render

    monkeypatch.setattr(cli_module, "make_render_step", fake_render_step)
    args = cli_module.build_parser().parse_args(
        ["render", str(checkpoint), "--data", "unused"]
    )

    args.func(args)

    assert captured["model"].capacity == 3
    assert captured["memory_capacity"] == 3
    np.testing.assert_array_equal(
        captured["model"].means[...], expected_means
    )
    assert (captured["appearance"] is not None) is app_opt


def test_render_cli_rejects_distributed_checkpoint_without_scene(
    monkeypatch, tmp_path
):
    config = TrainConfig(
        model=ModelConfig(capacity=3, bucket_min_capacity=3, sh_degree=0)
    )
    checkpoint, _ = _write_distributed_cli_checkpoint(
        tmp_path / "missing_scene", config, include_scene=False
    )
    scene = SimpleNamespace(
        camtoworlds=np.eye(4, dtype=np.float32)[None, ...]
    )
    example = {
        "image": np.zeros((2, 3, 3), dtype=np.float32),
        "K": np.eye(3, dtype=np.float32),
        "w2c": np.eye(4, dtype=np.float32),
        "image_name": "frame.png",
    }
    monkeypatch.setattr(
        cli_module, "load_colmap_scene", lambda *_args, **_kwargs: scene
    )
    monkeypatch.setattr(
        cli_module, "create_grain_dataset", lambda *_args, **_kwargs: [example]
    )
    args = cli_module.build_parser().parse_args(
        ["render", str(checkpoint), "--data", "unused"]
    )

    with pytest.raises(ValueError, match="requires checkpoint scene metadata"):
        args.func(args)


def test_cli_export_bakes_appearance_to_degree_zero_sh(monkeypatch, tmp_path):
    config = TrainConfig(
        app_opt=True,
        model=ModelConfig(capacity=2, bucket_min_capacity=2, sh_degree=0),
    )
    model = cli_module.GaussianModel.empty(
        config.model, appearance_feature_dim=32
    )
    model.active_mask[0] = True
    model.colors[0] = jnp.asarray([0.2, -0.3, 0.4])
    appearance = cli_module.AppearanceOptModule(
        1, 32, config.app_embed_dim, config.model.sh_degree, rngs=nnx.Rngs(0)
    )
    captured = {}
    monkeypatch.setattr(
        cli_module,
        "_load_training_objects",
        lambda _p: (config, model, appearance, 3),
    )

    def fake_export(splats, output):
        captured["splats"] = splats
        return output

    monkeypatch.setattr(cli_module, "export_splats", fake_export)

    cli_module._export_command(
        argparse.Namespace(
            checkpoint=str(tmp_path / "checkpoint"),
            output=tmp_path / "out.ply",
        )
    )

    assert "features" not in captured["splats"]
    assert "colors" not in captured["splats"]
    assert captured["splats"]["sh0"].shape == (2, 1, 3)
    assert captured["splats"]["sh_rest"].shape == (2, 0, 3)


@pytest.mark.parametrize("app_opt", [False, True])
def test_cli_export_materializes_real_distributed_checkpoint(
    tmp_path, app_opt: bool
):
    config = TrainConfig(
        app_opt=app_opt,
        app_embed_dim=0,
        model=ModelConfig(capacity=3, bucket_min_capacity=3, sh_degree=1),
    )
    checkpoint, expected_means = _write_distributed_cli_checkpoint(
        tmp_path / ("appearance" if app_opt else "sh"), config
    )
    output = tmp_path / ("appearance.ply" if app_opt else "sh.ply")

    cli_module._export_command(
        argparse.Namespace(checkpoint=str(checkpoint), output=output)
    )

    restored = load_ply_to_splats(output)
    np.testing.assert_array_equal(restored["means"], expected_means)
    assert restored["means"].shape == (3, 3)
    assert restored["shN"].shape == (3, 0 if app_opt else 3, 3)


def test_train_cli_defaults_to_full_images_and_preserves_explicit_patch(monkeypatch):
    captured = []

    def fake_train(config, *, resume_from=None):
        captured.append((config, resume_from))
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    parser = cli_module.build_parser()

    default_args = parser.parse_args(["train"])
    default_args.func(default_args)
    patch_args = parser.parse_args(["train", "--patch-size", "24"])
    patch_args.func(patch_args)

    assert captured[0][0].data.patch_size is None
    assert captured[0][1] is None
    assert captured[1][0].data.patch_size == 24
    assert captured[1][1] is None


def test_estimate_memory_cli_forwards_full_image_dimensions(monkeypatch):
    calls = []

    def fake_estimate(
        _config,
        *,
        physical_capacity,
        image_height=None,
        image_width=None,
    ):
        calls.append((physical_capacity, image_height, image_width))
        return 1

    monkeypatch.setattr(
        cli_module, "estimate_training_memory_bytes", fake_estimate
    )
    args = cli_module.build_parser().parse_args(
        [
            "estimate-memory",
            "--image-height",
            "480",
            "--image-width",
            "640",
        ]
    )
    args.func(args)

    assert args.image_height == 480
    assert args.image_width == 640
    assert len(calls) == 2
    assert all(call[1:] == (480, 640) for call in calls)


def test_train_cli_exposes_the_per_tile_candidate_bound(monkeypatch):
    captured = {}

    def fake_train(config, *, resume_from=None):
        captured["config"] = config
        return SimpleNamespace(checkpoint="unused")

    monkeypatch.setattr(cli_module, "train", fake_train)
    args = cli_module.build_parser().parse_args(
        ["train", "--max-candidates-per-tile", "2048"]
    )
    args.func(args)
    assert captured["config"].rasterizer.max_candidates_per_tile == 2048

    # Omitting it keeps the conservative default rather than guessing a bound.
    args = cli_module.build_parser().parse_args(["train"])
    args.func(args)
    assert captured["config"].rasterizer.max_candidates_per_tile is None
