import pytest

from jax_gs.config import (
    ModelConfig,
    RasterizationConfig,
    StrategyConfig,
    TrainConfig,
)


def test_legacy_gpu_backends_migrate_without_rewriting_current_pallas():
    with pytest.warns(UserWarning, match="migrated legacy rasterizer settings"):
        config = TrainConfig.from_dict(
            {
                "rasterizer": {
                    "backend": "cuda_ffi",
                    "intersection_backend": "pallas",
                    "sort_backend": "cuda_ffi",
                }
            }
        )

    assert config.rasterizer.backend == "jax"
    assert config.rasterizer.intersection_backend == "pallas"
    assert config.rasterizer.sort_backend == "jax"


def test_removed_cutile_backend_values_map_to_pure_jax():
    with pytest.warns(DeprecationWarning, match="pure JAX"):
        config = RasterizationConfig(
            backend="cutile",
            intersection_backend="cutile",
            sort_backend="cutile",
        )

    assert config.backend == "jax"
    assert config.intersection_backend == "jax"
    assert config.sort_backend == "jax"


def test_compositor_backend_accepts_explicit_gpu_paths():
    assert RasterizationConfig().compositor_backend == "jax"
    pallas = RasterizationConfig(compositor_backend="pallas")
    cuda_ffi = RasterizationConfig(compositor_backend="cuda_ffi")
    assert pallas.compositor_backend == "pallas"
    assert cuda_ffi.compositor_backend == "cuda_ffi"
    with pytest.raises(ValueError, match="compositor_backend"):
        RasterizationConfig(compositor_backend="invalid")
    for compositor_backend in ("pallas", "cuda_ffi"):
        with pytest.raises(ValueError, match="intersections backend"):
            RasterizationConfig(
                backend="reference", compositor_backend=compositor_backend
            )


def test_projection_backend_accepts_explicit_strict_cuda_path():
    assert RasterizationConfig().projection_backend == "jax"
    config = RasterizationConfig(projection_backend="cuda_ffi_strict")
    assert config.projection_backend == "cuda_ffi_strict"
    with pytest.raises(ValueError, match="projection_backend"):
        RasterizationConfig(projection_backend="invalid")
    with pytest.raises(ValueError, match="intersections backend"):
        RasterizationConfig(
            backend="reference", projection_backend="cuda_ffi_strict"
        )


@pytest.mark.parametrize(
    "backend", ["pallas", "cuda_tile", "cuda_tile_cub"]
)
def test_intersection_backend_accepts_explicit_gpu_paths(backend):
    config = RasterizationConfig(intersection_backend=backend)
    assert config.intersection_backend == backend


def test_2dgs_training_config_round_trips_upstream_regularizers(tmp_path):
    config = TrainConfig(
        model_type="2dgs",
        normal_loss=True,
        normal_lambda=0.05,
        normal_start_iter=7_000,
        dist_loss=True,
        dist_lambda=0.01,
        dist_start_iter=3_000,
    )

    path = tmp_path / "config.json"
    config.save(path)
    restored = TrainConfig.load(path)

    assert restored == config
    assert restored.densification_gradient_key == "gradient_2dgs"
    assert restored.strategy.key_for_gradient == "means2d"


def test_3dgs_remains_the_default_training_model():
    config = TrainConfig()

    assert config.model_type == "3dgs"
    assert not config.app_opt
    assert config.app_embed_dim == 16
    assert config.app_opt_lr == 1.0e-3
    assert config.app_opt_reg == 1.0e-6
    assert not config.pose_opt
    assert config.pose_opt_lr == 1.0e-5
    assert config.pose_opt_reg == 1.0e-6
    assert config.pose_noise == 0.0
    assert config.data.patch_size is None
    assert not config.packed
    assert not config.sparse_grad
    assert not config.visible_adam
    assert not config.normal_loss
    assert not config.dist_loss
    assert config.opacity_reg == 0.0
    assert config.scale_reg == 0.0
    assert config.densification_gradient_key == "means2d"


def test_3dgs_regularization_weights_round_trip(tmp_path):
    config = TrainConfig(opacity_reg=0.01, scale_reg=0.02)
    path = tmp_path / "regularizers.json"

    config.save(path)

    assert TrainConfig.load(path) == config


def test_pose_optimization_config_round_trips_through_json(tmp_path):
    config = TrainConfig(
        pose_opt=True,
        pose_opt_lr=2.0e-5,
        pose_opt_reg=3.0e-6,
        pose_noise=0.01,
    )
    path = tmp_path / "pose_opt.json"

    config.save(path)

    assert TrainConfig.load(path) == config


def test_appearance_optimization_config_round_trips_through_json(tmp_path):
    config = TrainConfig(
        app_opt=True,
        app_embed_dim=24,
        app_opt_lr=2.0e-3,
        app_opt_reg=3.0e-6,
    )
    path = tmp_path / "appearance_opt.json"

    config.save(path)

    assert TrainConfig.load(path) == config


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("app_embed_dim", -1, "app_embed_dim"),
        ("app_opt_lr", -1.0e-3, "app_opt_lr"),
        ("app_opt_reg", -1.0e-6, "app_opt_reg"),
    ],
)
def test_appearance_optimization_config_rejects_invalid_values(
    name, value, match
):
    with pytest.raises(ValueError, match=match):
        TrainConfig(**{name: value})


def test_pose_optimizer_accepts_zero_learning_rate_like_current_main():
    assert TrainConfig(pose_opt_lr=0.0).pose_opt_lr == 0.0


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("pose_opt_lr", -1.0e-5, "pose_opt_lr"),
        ("pose_opt_reg", -1.0e-6, "pose_opt_reg"),
        ("pose_noise", -0.01, "pose_noise"),
    ],
)
def test_pose_optimization_config_rejects_invalid_values(name, value, match):
    with pytest.raises(ValueError, match=match):
        TrainConfig(**{name: value})


@pytest.mark.parametrize("name", ["opacity_reg", "scale_reg"])
def test_3dgs_regularization_weights_must_be_non_negative(name):
    with pytest.raises(ValueError, match="regularization weights"):
        TrainConfig(**{name: -0.01})


@pytest.mark.parametrize("name", ["opacity_reg", "scale_reg"])
def test_2dgs_rejects_3d_regularization_weights(name):
    with pytest.raises(ValueError, match="3DGS"):
        TrainConfig(model_type="2dgs", **{name: 0.01})


def test_target_primitives_is_optional_validated_and_round_trips(tmp_path):
    assert TrainConfig().strategy.target_primitives is None
    config = TrainConfig(
        model=ModelConfig(capacity=16),
        strategy=StrategyConfig(target_primitives=12),
    )
    path = tmp_path / "target.json"
    config.save(path)
    assert TrainConfig.load(path) == config

    with pytest.raises(TypeError, match="integer"):
        TrainConfig.from_dict({"strategy": {"target_primitives": 1.5}})
    with pytest.raises(ValueError, match="positive"):
        StrategyConfig(target_primitives=0)
    with pytest.raises(ValueError, match="default strategy"):
        StrategyConfig(kind="mcmc", target_primitives=8)
    with pytest.raises(ValueError, match="logical model capacity"):
        TrainConfig(
            model=ModelConfig(capacity=8),
            strategy=StrategyConfig(target_primitives=9),
        )


def test_current_main_point_cloud_scale_multiplier_defaults_to_one():
    assert ModelConfig().initial_scale == 1.0


def test_current_main_scene_normalization_defaults_and_validation():
    config = TrainConfig()

    assert config.normalize_world_space
    assert config.global_scale == 1.0
    for invalid_scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="global_scale"):
            TrainConfig(global_scale=invalid_scale)


def test_2dgs_training_preset_matches_current_main_example_profile():
    config = TrainConfig.for_model_type("2dgs")

    assert config.model_type == "2dgs"
    assert config.model.initial_scale == 1.0
    assert config.rasterizer.near_plane == 0.2
    assert config.rasterizer.far_plane == 200.0
    assert config.strategy.prune_opacity == 0.05
    assert config.strategy.key_for_gradient == "gradient_2dgs"


def test_mcmc_training_preset_matches_current_main_example_profile():
    config = TrainConfig.for_model_type("3dgs", strategy_kind="mcmc")

    assert config.model_type == "3dgs"
    assert config.model.initial_opacity == 0.5
    assert config.model.initial_scale == 0.1
    assert config.opacity_reg == 0.01
    assert config.scale_reg == 0.01
    assert config.strategy.kind == "mcmc"
    assert config.strategy.verbose


def test_full_image_default_round_trips_through_json(tmp_path):
    config = TrainConfig()
    path = tmp_path / "full_image.json"

    config.save(path)
    restored = TrainConfig.load(path)

    assert restored == config
    assert restored.data.patch_size is None


def test_selective_training_flags_round_trip(tmp_path):
    config = TrainConfig(packed=True, sparse_grad=True)
    path = tmp_path / "selective.json"

    config.save(path)

    assert TrainConfig.load(path) == config


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sparse_grad": True}, "requires packed=True"),
        (
            {"packed": True, "sparse_grad": True, "visible_adam": True},
            "mutually exclusive",
        ),
        (
            {"model_type": "2dgs", "visible_adam": True},
            "2DGS.*visible_adam",
        ),
        (
            {"packed": True, "sparse_grad": True, "with_ut": True},
            "sparse_grad.*with_ut",
        ),
        (
            {
                "packed": True,
                "sparse_grad": True,
                "camera_model": "ftheta",
            },
            "sparse_grad.*ftheta",
        ),
    ],
)
def test_selective_training_config_rejects_unsupported_combinations(
    kwargs, match
):
    with pytest.raises(ValueError, match=match):
        TrainConfig(**kwargs)


def test_2dgs_accepts_packed_sparse_training():
    config = TrainConfig(model_type="2dgs", packed=True, sparse_grad=True)

    assert config.packed
    assert config.sparse_grad
