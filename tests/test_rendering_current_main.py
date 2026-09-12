import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_gs
from jax_gs.config import RasterizationConfig
from jax_gs.rendering import (
    RendererConfig,
    RendererConfig_MixedBatch,
    RendererConfig_ParallelBatch,
    render_mode_has_color,
    render_mode_has_depth,
    render_mode_has_depth_channel,
    render_mode_has_expected_depth,
    render_mode_has_hit_distance,
    render_mode_has_only_color,
    render_mode_has_only_depth_channel,
)
from jax_gs.rendering_types import resolve_tile_size


def _minimal_rasterization_args():
    return {
        "means": jnp.asarray([[0.0, 0.0, 2.0]], jnp.float32),
        "quats": jnp.asarray([[1.0, 0.0, 0.0, 0.0]], jnp.float32),
        "scales": jnp.asarray([[0.1, 0.1, 0.1]], jnp.float32),
        "opacities": jnp.asarray([0.8], jnp.float32),
        "colors": jnp.asarray([[1.0, 0.0, 0.0]], jnp.float32),
        "viewmats": jnp.eye(4, dtype=jnp.float32)[None],
        "Ks": jnp.asarray(
            [[[20.0, 0.0, 4.5], [0.0, 20.0, 4.5], [0.0, 0.0, 1.0]]],
            jnp.float32,
        ),
        "width": 8,
        "height": 8,
    }


def _minimal_2dgs_rasterization_args():
    args = _minimal_rasterization_args()
    args["scales"] = jnp.asarray([[0.2, 0.2, 0.01]], jnp.float32)
    return args


def test_renderer_config_public_api_and_default():
    assert jax_gs.RendererConfig is RendererConfig
    assert jax_gs.RendererConfig_MixedBatch is RendererConfig_MixedBatch
    assert jax_gs.RendererConfig_ParallelBatch is RendererConfig_ParallelBatch
    with pytest.raises(TypeError, match="RendererConfig_MixedBatch"):
        RendererConfig()
    assert isinstance(RendererConfig_MixedBatch(), RendererConfig)
    assert isinstance(RendererConfig_ParallelBatch(), RendererConfig)
    assert (
        inspect.signature(jax_gs.rasterization).parameters["renderer_config"].default
        is None
    )


def test_pure_jax_capability_queries_report_migrated_subsystems():
    assert jax_gs.has_2dgs()
    assert jax_gs.has_3dgs()
    assert jax_gs.has_3dgut()
    assert jax_gs.has_adam()
    assert jax_gs.has_reloc()
    assert jax_gs.has_camera_wrappers()
    assert jax_gs.has_losses()


def test_current_main_sparse_visibility_symbols_are_public():
    for name in (
        "build_sparse_tile_layout",
        "isect_tiles_sparse",
        "rasterize_to_pixels_sparse",
        "rasterize_num_contributing_gaussians",
        "rasterize_num_contributing_gaussians_sparse",
        "rasterize_contributing_gaussian_ids",
        "rasterize_contributing_gaussian_ids_sparse",
        "rasterize_top_contributing_gaussian_ids",
        "rasterize_top_contributing_gaussian_ids_sparse",
    ):
        assert callable(getattr(jax_gs, name))


def test_high_level_sparse_gradient_adaptation_is_explicit():
    rendered, alpha, info = jax_gs.rasterization(
        **_minimal_rasterization_args(),
        packed=True,
        sparse_grad=True,
        absgrad=True,
        config=RasterizationConfig(
            backend="reference",
            tile_size=4,
            max_gaussians_per_tile=2,
            max_intersections=8,
        ),
    )
    assert rendered.shape == (1, 8, 8, 3)
    assert alpha.shape == (1, 8, 8, 1)
    assert bool(info["packed_requested"])
    assert bool(info["sparse_grad_requested"])
    assert bool(info["sparse_grad_is_dense"])
    assert bool(info["absgrad_requested"])
    assert not bool(info["absgrad_available"])


@pytest.mark.parametrize(
    ("updates", "error_type", "match"),
    [
        ({"packed": False}, ValueError, "packed=True"),
        ({"packed": True, "with_ut": True}, ValueError, "with_ut"),
        (
            {"packed": True, "with_eval3d": True},
            ValueError,
            "with_eval3d",
        ),
        (
            {"packed": True, "distributed": True},
            NotImplementedError,
            "distributed",
        ),
    ],
)
def test_high_level_sparse_gradient_rejects_unsupported_3d_modes(
    updates, error_type, match
):
    kwargs = _minimal_rasterization_args()
    kwargs.update(updates)
    with pytest.raises(error_type, match=match):
        jax_gs.rasterization(**kwargs, sparse_grad=True)


def test_high_level_sparse_gradient_rejects_3d_leading_batch():
    kwargs = _minimal_rasterization_args()
    for key in ("means", "quats", "scales", "opacities", "colors"):
        kwargs[key] = jnp.broadcast_to(kwargs[key], (2,) + kwargs[key].shape)
    for key in ("viewmats", "Ks"):
        kwargs[key] = jnp.broadcast_to(kwargs[key], (2,) + kwargs[key].shape)
    with pytest.raises(ValueError, match="batch dimensions"):
        jax_gs.rasterization(**kwargs, packed=True, sparse_grad=True)


def test_high_level_2dgs_sparse_gradient_adaptation_is_explicit():
    *outputs, info = jax_gs.rasterization_2dgs(
        **_minimal_2dgs_rasterization_args(),
        packed=True,
        sparse_grad=True,
        config=RasterizationConfig(
            backend="intersections",
            tile_size=4,
            max_gaussians_per_tile=2,
            max_intersections=8,
        ),
    )
    assert outputs[0].shape == (1, 8, 8, 3)
    assert bool(info["packed_requested"])
    assert bool(info["sparse_grad_requested"])
    assert bool(info["sparse_grad_is_dense"])


@pytest.mark.parametrize("leading_batch", [False, True])
def test_high_level_2dgs_sparse_gradient_requires_unbatched_packed(leading_batch):
    kwargs = _minimal_2dgs_rasterization_args()
    packed = False
    match = "packed=True"
    if leading_batch:
        packed = True
        match = "batch dimensions"
        for key in ("means", "quats", "scales", "opacities", "colors"):
            kwargs[key] = jnp.broadcast_to(kwargs[key], (2,) + kwargs[key].shape)
        for key in ("viewmats", "Ks"):
            kwargs[key] = jnp.broadcast_to(kwargs[key], (2,) + kwargs[key].shape)
    with pytest.raises(ValueError, match=match):
        jax_gs.rasterization_2dgs(
            **kwargs,
            packed=packed,
            sparse_grad=True,
        )


class RendererConfig_Future(RendererConfig):
    pass


@pytest.mark.parametrize(
    ("renderer_config", "error_type", "match"),
    [
        (object(), TypeError, "renderer_config"),
        (RendererConfig_Future(), NotImplementedError, "RendererConfig_Future"),
    ],
)
def test_rasterization_rejects_unsupported_renderer_config(
    renderer_config, error_type, match
):
    with pytest.raises(error_type, match=match):
        jax_gs.rasterization(
            **_minimal_rasterization_args(),
            renderer_config=renderer_config,
        )


def test_parallel_renderer_requires_eval3d():
    with pytest.raises(ValueError, match="with_eval3d=True"):
        jax_gs.rasterization(
            **_minimal_rasterization_args(),
            renderer_config=RendererConfig_ParallelBatch(),
        )


def test_distributed_flag_single_rank_matches_local_forward_metadata_and_gradients():
    arguments = _minimal_rasterization_args()
    arguments["extra_signals"] = jnp.asarray([[0.25, 0.75]], jnp.float32)
    differentiable_names = ("means", "quats", "scales", "opacities", "colors")
    differentiable_values = tuple(arguments.pop(name) for name in differentiable_names)
    rasterization_config = RasterizationConfig(
        backend="reference",
        tile_size=4,
        max_gaussians_per_tile=2,
        max_intersections=8,
    )

    def render_and_loss(distributed, *values):
        dynamic = dict(zip(differentiable_names, values, strict=True))
        rendered, alpha, info = jax_gs.rasterization(
            **dynamic,
            **arguments,
            packed=True,
            distributed=distributed,
            config=rasterization_config,
        )
        return rendered.sum() + alpha.sum(), (rendered, alpha, info)

    local_value_and_grad = jax.value_and_grad(
        lambda *values: render_and_loss(False, *values),
        argnums=tuple(range(len(differentiable_values))),
        has_aux=True,
    )
    distributed_value_and_grad = jax.value_and_grad(
        lambda *values: render_and_loss(True, *values),
        argnums=tuple(range(len(differentiable_values))),
        has_aux=True,
    )
    (_, local_outputs), local_gradients = local_value_and_grad(*differentiable_values)
    (_, distributed_outputs), distributed_gradients = distributed_value_and_grad(
        *differentiable_values
    )

    local_rendered, local_alpha, local_info = local_outputs
    distributed_rendered, distributed_alpha, distributed_info = distributed_outputs
    np.testing.assert_allclose(distributed_rendered, local_rendered, rtol=1e-6)
    np.testing.assert_allclose(distributed_alpha, local_alpha, rtol=1e-6)
    for distributed_gradient, local_gradient in zip(
        distributed_gradients, local_gradients, strict=True
    ):
        np.testing.assert_allclose(
            distributed_gradient, local_gradient, rtol=1e-6, atol=1e-7
        )

    assert set(distributed_info) == set(local_info)
    for name in local_info.keys() - {
        "distributed_requested",
        "distributed_world_size",
    }:
        local_value = local_info[name]
        distributed_value = distributed_info[name]
        if local_value is None:
            assert distributed_value is None
        else:
            np.testing.assert_allclose(
                distributed_value, local_value, rtol=1e-6, atol=1e-7
            )
    assert bool(distributed_info["distributed_requested"])
    assert int(distributed_info["distributed_world_size"]) == 1


@pytest.mark.parametrize(
    ("mode", "color", "hit", "depth", "expected"),
    [
        ("RGB", True, False, False, False),
        ("d", False, True, False, False),
        ("Ed", False, True, False, True),
        ("D", False, False, True, False),
        ("ED", False, False, True, True),
        ("RGB-d", True, True, False, False),
        ("RGB-Ed", True, True, False, True),
        ("RGB+D", True, False, True, False),
        ("RGB+ED", True, False, True, True),
    ],
)
def test_render_mode_queries(mode, color, hit, depth, expected):
    assert render_mode_has_color(mode) is color
    assert render_mode_has_hit_distance(mode) is hit
    assert render_mode_has_depth(mode) is depth
    assert render_mode_has_expected_depth(mode) is expected
    assert render_mode_has_depth_channel(mode) is (hit or depth)
    assert render_mode_has_only_depth_channel(mode) is ((hit or depth) and not color)
    assert render_mode_has_only_color(mode) is (color and not (hit or depth))


def test_current_main_tile_size_resolution():
    assert resolve_tile_size(None, with_eval3d=False, width=640, height=480) == 16
    assert resolve_tile_size(None, with_eval3d=True, width=1920, height=1079) == 8
    assert resolve_tile_size(None, with_eval3d=True, width=1920, height=1080) == 16
    assert resolve_tile_size(4, with_eval3d=True, width=1920, height=1080) == 4


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rays": jnp.zeros((1, 8, 8, 6), jnp.float32)}, "with_eval3d=True"),
        ({"return_normals": True}, "return_normals=True"),
        ({"render_mode": "d", "colors": None}, "hit-distance"),
    ],
)
def test_eval3d_only_arguments_fail_at_public_boundary(kwargs, match):
    arguments = _minimal_rasterization_args()
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        jax_gs.rasterization(**arguments)


def test_external_distortion_rejects_the_wrong_parameter_type():
    with pytest.raises(TypeError, match="BivariateWindshieldModelParameters"):
        jax_gs.rasterization(
            **_minimal_rasterization_args(),
            external_distortion_coeffs=object(),
        )


def test_lidar_coefficients_require_the_lidar_camera_model():
    with pytest.raises(ValueError, match="camera_model='lidar'"):
        jax_gs.rasterization(
            **_minimal_rasterization_args(),
            lidar_coeffs=object(),
        )
