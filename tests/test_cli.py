from types import SimpleNamespace

from flax import nnx
import jax.numpy as jnp

import jax_gs.cli as cli_module
from jax_gs.config import ModelConfig, TrainConfig


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
    assert isinstance(captured["appearance_optimizer"], nnx.Optimizer)
    assert captured["appearance_image_names"] == ("a.png", "b.png")
    assert appearance.embeds.embedding.shape == (2, 4)
    assert appearance.feature_dim == 32
    assert appearance.sh_degree == 2
    assert step == 7


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
        SimpleNamespace(checkpoint=str(tmp_path / "checkpoint"), output=tmp_path / "out.ply")
    )

    assert "features" not in captured["splats"]
    assert "colors" not in captured["splats"]
    assert captured["splats"]["sh0"].shape == (2, 1, 3)
    assert captured["splats"]["sh_rest"].shape == (2, 0, 3)


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
