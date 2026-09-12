import jax
import jax.numpy as jnp
import pytest

import jax_gs
import jax_gs.data as data_api
import jax_gs.data.normalize as normalize_implementation
from jax_gs import PaddedProjection, fully_fused_projection
from jax_gs.checkpoints import (
    is_distributed_checkpoint,
    load_checkpoint_scene_transform,
    load_distributed_inference_checkpoint,
)


def _inputs():
    means = jnp.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]], jnp.float32)
    quats = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, jnp.float32)
    scales = jnp.full((2, 3), 0.1, jnp.float32)
    viewmats = jnp.eye(4, dtype=jnp.float32)[None]
    Ks = jnp.asarray(
        [[[20.0, 0.0, 4.0], [0.0, 20.0, 4.0], [0.0, 0.0, 1.0]]],
        jnp.float32,
    )
    return means, quats, scales, viewmats, Ks


def test_public_scene_normalization_and_checkpoint_exports():
    assert jax_gs.load_checkpoint_scene_transform is load_checkpoint_scene_transform
    assert "load_checkpoint_scene_transform" in jax_gs.__all__
    assert jax_gs.is_distributed_checkpoint is is_distributed_checkpoint
    assert (
        jax_gs.load_distributed_inference_checkpoint
        is load_distributed_inference_checkpoint
    )
    assert "is_distributed_checkpoint" in jax_gs.__all__
    assert "load_distributed_inference_checkpoint" in jax_gs.__all__

    normalization_helpers = (
        "align_principal_axes",
        "normalize_scene",
        "similarity_from_cameras",
        "transform_cameras",
        "transform_points",
    )
    for name in normalization_helpers:
        assert getattr(data_api, name) is getattr(normalize_implementation, name)
        assert name in data_api.__all__


def test_public_fully_fused_projection_dense_signature():
    means, quats, scales, viewmats, Ks = _inputs()
    result = fully_fused_projection(means, None, quats, scales, viewmats, Ks, 8, 8)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert result[0].shape == (1, 2, 2)
    assert jnp.all(result[0][0, 1] == 0)


def test_public_fully_fused_projection_static_packed_result():
    means, quats, scales, viewmats, Ks = _inputs()
    result = fully_fused_projection(
        means, None, quats, scales, viewmats, Ks, 8, 8, packed=True
    )
    assert isinstance(result, PaddedProjection)
    unpacked = tuple(result)
    assert len(unpacked) == 9
    assert int(result.valid_count) == 1
    assert result.radii.shape == (2, 2)
    assert int(result.gaussian_ids[0]) == 0
    assert int(result.gaussian_ids[1]) == -1

    packed_call = jax.jit(
        lambda current_means: fully_fused_projection(
            current_means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            8,
            8,
            packed=True,
        )
    )
    assert int(packed_call(means).valid_count) == 1


def test_public_fully_fused_projection_sparse_contract():
    means, quats, scales, viewmats, Ks = _inputs()
    sparse = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        8,
        8,
        packed=True,
        sparse_grad=True,
    )
    assert isinstance(sparse, PaddedProjection)

    with pytest.raises(ValueError, match="packed=True"):
        fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            8,
            8,
            sparse_grad=True,
        )
    with pytest.raises(ValueError, match="batch dimensions"):
        fully_fused_projection(
            means[None],
            None,
            quats[None],
            scales[None],
            viewmats[None],
            Ks[None],
            8,
            8,
            packed=True,
            sparse_grad=True,
        )


def test_gsplat_api_and_submodule_alignment():
    import importlib

    for module_name in (
        "jax_gs.rendering",
        "jax_gs.strategy",
        "jax_gs.distributed",
        "jax_gs.exporter",
        "jax_gs.color_correct",
        "jax_gs.compression",
        "jax_gs.losses",
        "jax_gs.optimizers",
        "jax_gs.cameras",
        "jax_gs.camera_wrappers",
        "jax_gs.relocation",
        "jax_gs.cuda",
    ):
        importlib.import_module(module_name)
    _C = importlib.import_module("jax_gs.cuda._wrapper")

    root_expected = [
        "rasterization",
        "rasterization_2dgs",
        "fully_fused_projection",
        "isect_tiles",
        "rasterize_to_pixels",
        "DefaultStrategy",
        "MCMCStrategy",
        "export_splats",
        "export_ply",
        "export_splat",
        "quat_scale_to_covar_preci",
        "spherical_harmonics",
        "world_to_cam",
    ]
    for symbol in root_expected:
        assert hasattr(jax_gs, symbol), f"Missing root export: {symbol}"
        assert getattr(jax_gs, symbol) is not None

    cuda_wrapper_expected = [
        "fully_fused_projection",
        "isect_tiles",
        "rasterize_to_pixels",
        "spherical_harmonics",
        "world_to_cam",
        "quat_scale_to_covar_preci",
        "rasterize_to_pixels_2dgs",
    ]
    for symbol in cuda_wrapper_expected:
        assert hasattr(_C, symbol), f"Missing cuda._wrapper export: {symbol}"
        assert getattr(_C, symbol) is not None
