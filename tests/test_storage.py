from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.checkpoints import (
    _config_fingerprint,
    load_checkpoint_scene_transform,
    restore_checkpoint,
    save_checkpoint,
)
from jax_gs.compression import PngCompression
from jax_gs.compression.png_compression import PngCompression as CurrentPngCompression
from jax_gs.compression.sort import sort_splats
from jax_gs.config import ModelConfig, OptimizerConfig, StrategyConfig, TrainConfig
from jax_gs.exporter import (
    export_ply,
    export_splat,
    export_splats,
    load_ply_to_splats,
    pack_rotation,
    part1by2_vec,
)
from jax_gs.model import GaussianModel
from jax_gs.optimizers import create_optimizer
from jax_gs.strategy import DefaultStrategy


def test_inference_compositor_does_not_change_the_training_fingerprint():
    config = TrainConfig()
    pallas = replace(
        config,
        rasterizer=replace(config.rasterizer, compositor_backend="pallas"),
    )

    assert _config_fingerprint(pallas) == _config_fingerprint(config)


def test_target_primitives_preserves_legacy_fingerprint_when_disabled():
    config = TrainConfig()
    legacy_values = config.to_dict()
    legacy_values["rasterizer"].pop("compositor_backend", None)
    legacy_values["strategy"].pop("target_primitives", None)
    legacy_payload = json.dumps(
        legacy_values, sort_keys=True, separators=(",", ":")
    )
    assert _config_fingerprint(config) == hashlib.sha256(
        legacy_payload.encode("utf-8")
    ).hexdigest()


def test_target_primitives_changes_the_training_fingerprint():
    config = TrainConfig()
    targeted = replace(
        config,
        strategy=replace(config.strategy, target_primitives=250_000),
    )

    assert _config_fingerprint(targeted) != _config_fingerprint(config)


def test_orbax_checkpoint_round_trip(tmp_path: Path):
    model_config = ModelConfig(capacity=8, sh_degree=1)
    optimizer_config = OptimizerConfig(max_steps=10)
    strategy_config = StrategyConfig(
        max_new_per_refine=2,
        refine_start=0,
        refine_stop=2,
        refine_every=1,
        target_primitives=4,
        grow_grad2d=0.1,
        grow_scale3d=100.0,
        prune_scale3d=100.0,
    )
    model = GaussianModel.empty(model_config)
    model.active_mask[:2] = True
    model.means[:2] = jnp.array([[1, 2, 3], [4, 5, 6]], jnp.float32)
    optimizer = create_optimizer(model, optimizer_config)
    state = DefaultStrategy(strategy_config).initialize_state(8)
    state.target_births_remaining[...] = 1
    checkpoint = save_checkpoint(
        tmp_path,
        model,
        step=3,
        optimizer=optimizer,
        strategy_state=state,
        config=TrainConfig(
            model=model_config,
            optimizer=optimizer_config,
            strategy=strategy_config,
        ),
    )
    state.target_births_remaining[...] = 0
    model.means[:] = 0.0
    model.active_mask[:] = False
    step = restore_checkpoint(
        checkpoint, model, optimizer=optimizer, strategy_state=state
    )
    assert step == 3
    assert int(model.active_count) == 2
    assert int(state.target_births_remaining[...]) == 1
    assert jnp.allclose(model.means[:2], jnp.array([[1, 2, 3], [4, 5, 6]]))
    assert load_checkpoint_scene_transform(checkpoint) is None


@pytest.mark.parametrize("scene_scale", [2.75, 0.0])
def test_checkpoint_scene_metadata_round_trip(tmp_path: Path, scene_scale: float):
    model_config = ModelConfig(
        capacity=2, bucket_min_capacity=2, sh_degree=0
    )
    model = GaussianModel.empty(model_config)
    matrix = np.asarray(
        [
            [0.0, -0.5, 0.0, 1.25],
            [0.5, 0.0, 0.0, -2.5],
            [0.0, 0.0, 0.5, 3.75],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    checkpoint = save_checkpoint(
        tmp_path,
        model,
        step=4,
        scene_transform=matrix,
        scene_scale=jnp.asarray(scene_scale, jnp.float32),
    )

    restored = load_checkpoint_scene_transform(checkpoint)
    assert restored is not None
    restored_matrix, restored_scale = restored
    np.testing.assert_array_equal(restored_matrix, matrix.astype(np.float64))
    assert restored_scale == scene_scale
    metadata = json.loads(
        (checkpoint / "jax_gs_checkpoint.json").read_text(encoding="utf-8")
    )
    assert metadata["format_version"] == 6
    assert "scene" in metadata["components"]
    assert set(metadata["scene"]) == {"matrix", "scene_scale"}


@pytest.mark.parametrize(
    ("scene_transform", "scene_scale", "match"),
    [
        (np.eye(4), None, "provided together"),
        (None, 1.0, "provided together"),
        (np.zeros((3, 4)), 1.0, "shape"),
        (np.eye(4) * np.nan, 1.0, "finite"),
        (np.diag([1.0, 1.0, 1.0, 2.0]), 1.0, "homogeneous"),
        (np.diag([1.0, 2.0, 1.0, 1.0]), 1.0, "similarity"),
        (np.eye(4), -1.0, "finite non-negative scalar"),
        (np.eye(4), np.inf, "finite non-negative scalar"),
        (np.eye(4), np.ones(1), "finite non-negative scalar"),
    ],
)
def test_save_checkpoint_rejects_invalid_scene_metadata(
    tmp_path: Path,
    scene_transform,
    scene_scale,
    match: str,
):
    model_config = ModelConfig(
        capacity=2, bucket_min_capacity=2, sh_degree=0
    )
    model = GaussianModel.empty(model_config)

    with pytest.raises(ValueError, match=match):
        save_checkpoint(
            tmp_path,
            model,
            step=1,
            scene_transform=scene_transform,
            scene_scale=scene_scale,
        )


@pytest.mark.parametrize("format_version", range(1, 7))
def test_checkpoint_without_scene_component_returns_none(
    tmp_path: Path, format_version: int
):
    checkpoint = tmp_path / f"v{format_version}"
    checkpoint.mkdir()
    (checkpoint / "jax_gs_checkpoint.json").write_text(
        json.dumps(
            {"format_version": format_version, "components": ["model"]}
        ),
        encoding="utf-8",
    )

    assert load_checkpoint_scene_transform(checkpoint) is None


@pytest.mark.parametrize(
    ("format_version", "scene", "match"),
    [
        (
            5,
            {
                "matrix": np.eye(4).tolist(),
                "scene_scale": 1.0,
            },
            "version 6",
        ),
        (6, {"matrix": np.eye(4).tolist()}, "incomplete"),
        (
            6,
            {
                "matrix": np.zeros((3, 4)).tolist(),
                "scene_scale": 1.0,
            },
            "shape",
        ),
        (
            6,
            {
                "matrix": (np.eye(4) * np.nan).tolist(),
                "scene_scale": 1.0,
            },
            "finite",
        ),
        (
            6,
            {
                "matrix": np.eye(4).tolist(),
                "scene_scale": -1.0,
            },
            "finite non-negative scalar",
        ),
        (
            6,
            {
                "matrix": np.diag([1.0, 1.0, 1.0, 2.0]).tolist(),
                "scene_scale": 1.0,
            },
            "homogeneous",
        ),
        (
            6,
            {
                "matrix": np.diag([1.0, 2.0, 1.0, 1.0]).tolist(),
                "scene_scale": 1.0,
            },
            "similarity",
        ),
    ],
)
def test_load_checkpoint_scene_transform_rejects_invalid_metadata(
    tmp_path: Path, format_version: int, scene: dict, match: str
):
    checkpoint = tmp_path / "invalid"
    checkpoint.mkdir()
    (checkpoint / "jax_gs_checkpoint.json").write_text(
        json.dumps(
            {
                "format_version": format_version,
                "components": ["model", "scene"],
                "scene": scene,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        load_checkpoint_scene_transform(checkpoint)


def test_png_compression_is_lossless_and_exports_have_expected_size(tmp_path: Path):
    model = GaussianModel.empty(ModelConfig(capacity=8, sh_degree=1))
    model.active_mask[:2] = True
    compression = PngCompression(image_width=8)
    compression.compress(tmp_path / "png", model)
    restored = compression.decompress(tmp_path / "png", as_jax=False)
    assert np.array_equal(restored["active_mask"], np.asarray(model.active_mask[...]))
    assert np.array_equal(restored["means"], np.asarray(model.means[...]))
    ply_path = export_ply(model, tmp_path / "model.ply")
    splat_path = export_splat(model, tmp_path / "model.splat")
    assert ply_path.stat().st_size > 0
    assert splat_path.stat().st_size == 2 * 32

    dispatched_path = export_splats(model, tmp_path / "dispatched.splat")
    assert dispatched_path == tmp_path / "dispatched.splat"
    assert dispatched_path.stat().st_size == 2 * 32


def test_generic_export_rejects_unbaked_appearance_model(tmp_path: Path):
    model = GaussianModel.empty(
        ModelConfig(capacity=2, bucket_min_capacity=2),
        appearance_feature_dim=32,
    )

    with pytest.raises(ValueError, match="bake"):
        export_splats(model, tmp_path / "appearance.ply")


def _upstream_export_arrays():
    means = np.array(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [np.nan, 7.0, 8.0]],
        dtype=np.float32,
    )
    scales = np.zeros((3, 3), dtype=np.float32)
    quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (3, 1))
    opacities = np.array([0.0, -10.0, 0.0], dtype=np.float32)
    sh0 = np.zeros((3, 1, 3), dtype=np.float32)
    shN = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 32
    return means, scales, quats, opacities, sh0, shN


def test_exporter_helpers_accept_current_main_keyword_names():
    morton_bits = part1by2_vec(x=np.asarray([0, 1, 2], np.uint32))
    np.testing.assert_array_equal(morton_bits, np.asarray([0, 1, 8], np.uint32))

    packed = pack_rotation(q=np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32))
    assert packed.shape == (1,)
    assert packed.dtype == np.uint32


def test_upstream_export_splats_ply_filters_invalid_rows_and_saves(tmp_path: Path):
    arrays = _upstream_export_arrays()
    output_path = tmp_path / "nested" / "upstream.ply"
    data = export_splats(*arrays, format="ply", save_to=output_path)
    header, payload = data.split(b"end_header\n", maxsplit=1)

    assert b"element vertex 2\n" in header
    assert b"property float f_rest_5\n" in header
    assert output_path.read_bytes() == data
    records = np.frombuffer(payload, dtype="<f4").reshape(2, 20)
    np.testing.assert_array_equal(records[:, :3], arrays[0][:2])


def test_upstream_export_splats_compact_formats():
    arrays = _upstream_export_arrays()
    splat = export_splats(*arrays, format="splat")
    assert len(splat) == 2 * 32

    compressed = export_splats(*arrays, format="ply_compressed")
    header, payload = compressed.split(b"end_header\n", maxsplit=1)
    # The second finite row is below the compressed format's opacity threshold.
    assert b"element chunk 1\n" in header
    assert b"element vertex 1\n" in header
    assert b"element sh 1\n" in header
    assert len(payload) == 18 * 4 + 4 * 4 + 6


def test_upstream_export_splats_validates_shapes_and_format():
    arrays = _upstream_export_arrays()
    with np.testing.assert_raises_regex(ValueError, "sh0 must have shape"):
        export_splats(*arrays[:4], arrays[4][:, 0], arrays[5])
    with np.testing.assert_raises_regex(ValueError, "unsupported format"):
        export_splats(*arrays, format="obj")


def test_upstream_export_splats_supports_empty_models():
    arrays = (
        np.zeros((0, 3), np.float32),
        np.zeros((0, 3), np.float32),
        np.zeros((0, 4), np.float32),
        np.zeros((0,), np.float32),
        np.zeros((0, 1, 3), np.float32),
        np.zeros((0, 2, 3), np.float32),
    )
    ply = export_splats(*arrays, format="ply")
    splat = export_splats(*arrays, format="splat")
    compressed = export_splats(*arrays, format="ply_compressed")
    assert b"element vertex 0\n" in ply
    assert splat == b""
    assert b"element chunk 0\n" in compressed
    assert b"element vertex 0\n" in compressed


def test_load_ply_to_splats_round_trip_preserves_sh_layout(tmp_path: Path):
    arrays = list(_upstream_export_arrays())
    arrays[0] = arrays[0][:2]
    arrays[1] = arrays[1][:2]
    arrays[2] = arrays[2][:2]
    arrays[3] = arrays[3][:2]
    arrays[4] = arrays[4][:2]
    arrays[5] = arrays[5][:2]
    path = tmp_path / "round_trip.ply"
    export_splats(*arrays, format="ply", save_to=path)

    restored = load_ply_to_splats(path)

    assert set(restored) == {"means", "scales", "quats", "opacities", "sh0", "shN"}
    for name, expected in zip(
        ("means", "scales", "quats", "opacities", "sh0", "shN"), arrays
    ):
        assert restored[name].dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(restored[name]), expected)


def test_load_ply_to_splats_supports_degree_zero(tmp_path: Path):
    arrays = list(_upstream_export_arrays())
    arrays = [values[:2] for values in arrays]
    arrays[-1] = np.zeros((2, 0, 3), np.float32)
    path = tmp_path / "degree_zero.ply"
    export_splats(*arrays, format="ply", save_to=path)

    restored = load_ply_to_splats(path)

    assert restored["shN"].shape == (2, 0, 3)


def test_load_ply_to_splats_rejects_incomplete_rgb_sh_properties(tmp_path: Path):
    path = tmp_path / "invalid_rest.ply"
    path.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property float f_dc_0",
                "property float f_dc_1",
                "property float f_dc_2",
                "property float f_rest_0",
                "property float f_rest_1",
                "property float opacity",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "property float rot_0",
                "property float rot_1",
                "property float rot_2",
                "property float rot_3",
                "end_header",
                "0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0",
            )
        )
    )

    with np.testing.assert_raises_regex(ValueError, "not a multiple of 3"):
        load_ply_to_splats(path)


def test_current_compression_import_path_and_sort_contract():
    assert CurrentPngCompression is PngCompression
    splats = {
        "means": jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]
        ),
        "ids": np.array([10, 20, 30, 40]),
    }

    result = sort_splats(splats, verbose=False)

    assert result is splats
    np.testing.assert_array_equal(splats["ids"], [40, 30, 20, 10])
    np.testing.assert_array_equal(
        np.asarray(splats["means"]),
        [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]],
    )


def test_sort_splats_requires_a_square_grid():
    with np.testing.assert_raises_regex(AssertionError, "perfect square"):
        sort_splats({"means": jnp.zeros((3, 3))}, verbose=False)


def test_upstream_png_compression_schema_round_trip(tmp_path: Path):
    splats = {
        "means": np.array(
            [[-2.0, 0.0, 1.0], [-0.5, 1.0, 2.0], [0.5, 2.0, 3.0], [2.0, 3.0, 4.0]],
            np.float32,
        ),
        "scales": np.arange(12, dtype=np.float32).reshape(4, 3) / 10,
        "quats": np.array(
            [[2.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]] * 2,
            np.float32,
        ),
        "opacities": np.array([-2.0, -1.0, 0.0, 1.0], np.float32),
        "sh0": np.arange(12, dtype=np.float32).reshape(4, 1, 3) / 20,
        "shN": np.arange(24, dtype=np.float32).reshape(4, 2, 3) / 30,
        "features": np.arange(20, dtype=np.float32).reshape(4, 5),
    }
    directory = tmp_path / "upstream_png"
    compression = PngCompression(
        use_sort=False,
        verbose=False,
        kmeans_clusters=4,
        kmeans_iterations=2,
    )
    assert compression.compress(compress_dir=directory, splats=splats) is None
    restored = compression.decompress(compress_dir=directory, as_jax=False)

    assert (directory / "meta.json").is_file()
    assert (directory / "means_l.png").is_file()
    assert (directory / "means_u.png").is_file()
    assert (directory / "shN.npz").is_file()
    assert set(restored) == set(splats)
    for name in splats:
        assert restored[name].shape == splats[name].shape
        assert restored[name].dtype == splats[name].dtype
    np.testing.assert_allclose(restored["means"], splats["means"], atol=1e-4)
    np.testing.assert_allclose(restored["scales"], splats["scales"], atol=5e-3)
    expected_quats = splats["quats"] / np.linalg.norm(
        splats["quats"], axis=-1, keepdims=True
    )
    np.testing.assert_allclose(restored["quats"], expected_quats, atol=5e-3)
    np.testing.assert_allclose(restored["opacities"], splats["opacities"], atol=0.02)
    np.testing.assert_allclose(restored["sh0"], splats["sh0"], atol=5e-3)
    np.testing.assert_allclose(restored["shN"], splats["shN"], atol=0.02)
    np.testing.assert_array_equal(restored["features"], splats["features"])


def test_upstream_png_compression_crops_lowest_opacity_to_square(tmp_path: Path):
    count = 5
    splats = {
        "means": np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        "scales": np.zeros((count, 3), np.float32),
        "quats": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (count, 1)),
        "opacities": np.arange(count, dtype=np.float32),
        "sh0": np.zeros((count, 1, 3), np.float32),
        "shN": np.zeros((count, 0, 3), np.float32),
    }
    compression = PngCompression(use_sort=False, verbose=False)
    directory = tmp_path / "cropped"
    compression.compress(directory, splats)
    restored = compression.decompress(directory, as_jax=False)

    assert restored["means"].shape == (4, 3)
    # Stable descending opacity keeps source rows 4, 3, 2, and 1.
    np.testing.assert_allclose(
        restored["means"], splats["means"][[4, 3, 2, 1]], atol=1e-3
    )
