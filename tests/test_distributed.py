import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jax_gs
from jax_gs.config import RasterizationConfig
from jax_gs.distributed import (
    all_gather_int32,
    all_gather_tensor_list,
    all_to_all_int32,
    all_to_all_tensor_list,
    cli,
)


_DISTRIBUTED_RENDER_CONFIG = RasterizationConfig(
    backend="reference",
    tile_size=4,
    max_gaussians_per_tile=2,
    max_intersections=8,
)


def _distributed_render_inputs(sh_degree):
    means = jnp.asarray(
        [[[-0.08, 0.01, 2.0]], [[0.09, -0.02, 2.2]]], jnp.float32
    )
    quats = jnp.asarray(
        [[[1.0, 0.1, 0.0, 0.0]], [[1.0, 0.0, 0.1, 0.0]]], jnp.float32
    )
    scales = jnp.asarray(
        [[[0.08, 0.11, 0.09]], [[0.10, 0.07, 0.12]]], jnp.float32
    )
    opacities = jnp.asarray([[0.7], [0.6]], jnp.float32)
    if sh_degree is None:
        colors = jnp.asarray(
            [[[0.8, 0.2, 0.1]], [[0.1, 0.4, 0.9]]], jnp.float32
        )
        extra_signals = jnp.asarray(
            [[[0.2, 0.7]], [[0.8, 0.3]]], jnp.float32
        )
    else:
        basis_count = (sh_degree + 1) ** 2
        weights = jnp.where(
            jnp.arange(basis_count) == 0,
            1.0,
            0.01 * (jnp.arange(basis_count) + 1),
        )
        colors = jnp.asarray(
            [[[0.8, 0.2, 0.1]], [[0.1, 0.4, 0.9]]], jnp.float32
        )[:, :, None, :] * weights[None, None, :, None]
        extra_signals = jnp.asarray(
            [[[0.2, 0.7]], [[0.8, 0.3]]], jnp.float32
        )[:, :, None, :] * weights[None, None, :, None]
    viewmats = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (2, 1, 4, 4))
    viewmats = viewmats.at[1, 0, 0, 3].set(0.04)
    Ks = jnp.broadcast_to(
        jnp.asarray(
            [[20.0, 0.0, 4.5], [0.0, 20.0, 4.5], [0.0, 0.0, 1.0]],
            jnp.float32,
        ),
        (2, 1, 3, 3),
    )
    return means, quats, scales, opacities, colors, extra_signals, viewmats, Ks


def _render_distributed_shards(
    means,
    quats,
    scales,
    opacities,
    colors,
    extra_signals,
    viewmats,
    Ks,
    *,
    sh_degree,
    active_masks=None,
):
    if active_masks is None:
        active_masks = jnp.ones(means.shape[:2], dtype=jnp.bool_)

    def render_rank(
        local_means,
        local_quats,
        local_scales,
        local_opacities,
        local_colors,
        local_extra_signals,
        local_viewmats,
        local_Ks,
        local_active_mask,
    ):
        rendered, alpha, info = jax_gs.rasterization(
            local_means,
            local_quats,
            local_scales,
            local_opacities,
            local_colors,
            local_viewmats,
            local_Ks,
            8,
            8,
            sh_degree=sh_degree,
            packed=False,
            active_mask=local_active_mask,
            distributed=True,
            extra_signals=local_extra_signals,
            extra_signals_sh_degree=sh_degree,
            config=_DISTRIBUTED_RENDER_CONFIG,
            distributed_world_size=2,
            distributed_axis_name="rank",
        )
        return (
            rendered,
            alpha,
            info.get(
                "render_extra_signals",
                jnp.zeros(rendered.shape[:-1] + (0,), dtype=rendered.dtype),
            ),
            info["distributed_requested"],
            info["distributed_world_size"],
            info["distributed_active_mask"],
            info.get("distributed_feature_exchange", jnp.asarray(False)),
        )

    return jax.vmap(
        render_rank,
        in_axes=(
            0,
            0,
            0,
            0,
            0,
            None if extra_signals is None else 0,
            0,
            0,
            0,
        ),
        axis_name="rank",
    )(
        means,
        quats,
        scales,
        opacities,
        colors,
        extra_signals,
        viewmats,
        Ks,
        active_masks,
    )


def _render_concatenated_baseline(
    means,
    quats,
    scales,
    opacities,
    colors,
    extra_signals,
    viewmats,
    Ks,
    *,
    sh_degree,
    active_masks=None,
):
    global_means = means.reshape((-1,) + means.shape[2:])
    global_quats = quats.reshape((-1,) + quats.shape[2:])
    global_scales = scales.reshape((-1,) + scales.shape[2:])
    global_opacities = opacities.reshape(-1)
    global_colors = (
        tuple(value.reshape((-1,) + value.shape[2:]) for value in colors)
        if isinstance(colors, tuple)
        else colors.reshape((-1,) + colors.shape[2:])
    )
    global_extra_signals = (
        None
        if extra_signals is None
        else extra_signals.reshape((-1,) + extra_signals.shape[2:])
    )
    if active_masks is None:
        active_masks = jnp.ones(means.shape[:2], dtype=jnp.bool_)
    rendered, alpha, info = jax_gs.rasterization(
        global_means,
        global_quats,
        global_scales,
        global_opacities,
        global_colors,
        viewmats.reshape(-1, 4, 4),
        Ks.reshape(-1, 3, 3),
        8,
        8,
        sh_degree=sh_degree,
        packed=False,
        active_mask=active_masks.reshape(-1),
        extra_signals=global_extra_signals,
        extra_signals_sh_degree=sh_degree,
        config=_DISTRIBUTED_RENDER_CONFIG,
    )
    camera_shape = viewmats.shape[:2]
    return (
        rendered.reshape(camera_shape + rendered.shape[1:]),
        alpha.reshape(camera_shape + alpha.shape[1:]),
        (
            info["render_extra_signals"].reshape(
                camera_shape + info["render_extra_signals"].shape[1:]
            )
            if global_extra_signals is not None
            else jnp.zeros(
                camera_shape + rendered.shape[1:-1] + (0,),
                dtype=rendered.dtype,
            )
        ),
    )


def test_single_rank_fast_paths_work_inside_plain_jit():
    @jax.jit
    def collect(value):
        gathered_int = all_gather_int32(1, value)
        exchanged_int = all_to_all_int32(1, value[None])
        gathered_tensor = all_gather_tensor_list(1, [value[None, None]])[0]
        exchanged_tensor = all_to_all_tensor_list(1, [value[None, None]])[0]
        return gathered_int, exchanged_int, gathered_tensor, exchanged_tensor

    outputs = collect(jnp.asarray(7, dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(outputs[0]), [7])
    np.testing.assert_array_equal(np.asarray(outputs[1]), [7])
    np.testing.assert_array_equal(np.asarray(outputs[2]), [[7]])
    np.testing.assert_array_equal(np.asarray(outputs[3]), [[7]])


def test_named_vmap_static_collectives():
    def collect_int(rank):
        gathered = all_gather_int32(2, rank, axis_name="rank")
        sent = jnp.array([rank * 10, rank * 10 + 1], dtype=jnp.int32)
        exchanged = all_to_all_int32(2, sent, axis_name="rank")
        return gathered, exchanged

    gathered, exchanged = jax.jit(jax.vmap(collect_int, axis_name="rank"))(
        jnp.arange(2, dtype=jnp.int32)
    )
    np.testing.assert_array_equal(np.asarray(gathered), [[0, 1], [0, 1]])
    np.testing.assert_array_equal(np.asarray(exchanged), [[0, 10], [1, 11]])

    def collect_tensor(value):
        gathered_leaf = all_gather_tensor_list(
            2, [value], axis_name="rank"
        )[0]
        exchanged_leaf = all_to_all_tensor_list(
            2, [value], axis_name="rank"
        )[0]
        return gathered_leaf, exchanged_leaf

    values = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    gathered_leaf, exchanged_leaf = jax.jit(
        jax.vmap(collect_tensor, axis_name="rank")
    )(values)
    np.testing.assert_array_equal(
        np.asarray(gathered_leaf),
        [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]],
    )
    np.testing.assert_array_equal(
        np.asarray(exchanged_leaf), [[0, 1, 4, 5], [2, 3, 6, 7]]
    )


@pytest.mark.parametrize(
    "sh_degree", [None, 0, 3], ids=["features", "sh0", "sh3"]
)
def test_root_distributed_flag_matches_concatenated_scene_and_shard_gradients(
    sh_degree,
):
    inputs = _distributed_render_inputs(sh_degree)
    model_inputs = inputs[:6]
    viewmats, Ks = inputs[6:]

    distributed_outputs = _render_distributed_shards(
        *model_inputs, viewmats, Ks, sh_degree=sh_degree
    )
    baseline_outputs = _render_concatenated_baseline(
        *model_inputs, viewmats, Ks, sh_degree=sh_degree
    )
    for distributed_value, baseline_value in zip(
        distributed_outputs[:3], baseline_outputs, strict=True
    ):
        np.testing.assert_allclose(
            distributed_value, baseline_value, rtol=1e-5, atol=1e-6
        )
    np.testing.assert_array_equal(
        np.asarray(distributed_outputs[3]), [True, True]
    )
    np.testing.assert_array_equal(np.asarray(distributed_outputs[4]), [2, 2])
    np.testing.assert_array_equal(
        np.asarray(distributed_outputs[5]),
        np.ones((2, 2), dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        np.asarray(distributed_outputs[6]),
        [sh_degree == 3, sh_degree == 3],
    )

    def distributed_loss(*values):
        outputs = _render_distributed_shards(
            *values, sh_degree=sh_degree
        )
        return sum(jnp.sum(value) for value in outputs[:3])

    def baseline_loss(*values):
        outputs = _render_concatenated_baseline(
            *values, sh_degree=sh_degree
        )
        return sum(jnp.sum(value) for value in outputs)

    argument_numbers = tuple(range(len(inputs)))
    distributed_gradients = jax.grad(
        distributed_loss, argnums=argument_numbers
    )(*inputs)
    baseline_gradients = jax.grad(baseline_loss, argnums=argument_numbers)(
        *inputs
    )
    for distributed_gradient, baseline_gradient in zip(
        distributed_gradients, baseline_gradients, strict=True
    ):
        np.testing.assert_allclose(
            distributed_gradient, baseline_gradient, rtol=2e-4, atol=2e-5
        )


def test_distributed_feature_exchange_supports_split_sh_and_multiple_cameras():
    inputs = list(_distributed_render_inputs(3))
    colors = inputs[4]
    inputs[4] = (colors[:, :, :1], colors[:, :, 1:])
    inputs[5] = None
    second_view = inputs[6].at[:, :, 1, 3].add(0.03)
    inputs[6] = jnp.concatenate((inputs[6], second_view), axis=1)
    inputs[7] = jnp.concatenate((inputs[7], inputs[7]), axis=1)

    distributed_outputs = _render_distributed_shards(
        *inputs, sh_degree=3
    )
    baseline_outputs = _render_concatenated_baseline(
        *inputs, sh_degree=3
    )
    for distributed_value, baseline_value in zip(
        distributed_outputs[:2], baseline_outputs[:2], strict=True
    ):
        np.testing.assert_allclose(
            distributed_value, baseline_value, rtol=1e-5, atol=1e-6
        )
    assert distributed_outputs[2].shape[-1] == 0
    np.testing.assert_array_equal(distributed_outputs[6], [True, True])


def test_distributed_feature_exchange_preserves_split_input_validation():
    inputs = list(_distributed_render_inputs(3))
    colors = inputs[4]
    inputs[4] = (colors[:, :, :2], colors[:, :, 2:])
    with pytest.raises(ValueError, match="sh0"):
        _render_distributed_shards(*inputs, sh_degree=3)

    inputs = list(_distributed_render_inputs(3))
    extra = inputs[5]
    inputs[5] = (extra[:, :, :1], extra[:, :, 1:])
    with pytest.raises(TypeError, match="extra_signals"):
        _render_distributed_shards(*inputs, sh_degree=3)


def test_distributed_feature_exchange_masks_inactive_nan_rows():
    inputs = list(_distributed_render_inputs(3))
    inputs[0] = inputs[0].at[1, 0].set(jnp.nan)
    inputs[4] = inputs[4].at[1, 0].set(jnp.nan)
    inputs[5] = inputs[5].at[1, 0].set(jnp.nan)
    active_masks = jnp.asarray([[True], [False]])

    distributed_outputs = _render_distributed_shards(
        *inputs, sh_degree=3, active_masks=active_masks
    )
    baseline_outputs = _render_concatenated_baseline(
        *inputs, sh_degree=3, active_masks=active_masks
    )
    for distributed_value, baseline_value in zip(
        distributed_outputs[:3], baseline_outputs, strict=True
    ):
        assert np.all(np.isfinite(distributed_value))
        np.testing.assert_allclose(
            distributed_value, baseline_value, rtol=1e-5, atol=1e-6
        )

    def loss(render, means, colors, extra_signals):
        values = list(inputs)
        values[0] = means
        values[4] = colors
        values[5] = extra_signals
        outputs = render(
            *values, sh_degree=3, active_masks=active_masks
        )
        return sum(jnp.sum(value) for value in outputs[:3])

    gradient_inputs = (inputs[0], inputs[4], inputs[5])
    distributed_gradients = jax.grad(
        lambda *values: loss(_render_distributed_shards, *values),
        argnums=(0, 1, 2),
    )(*gradient_inputs)
    baseline_gradients = jax.grad(
        lambda *values: loss(_render_concatenated_baseline, *values),
        argnums=(0, 1, 2),
    )(*gradient_inputs)
    for distributed_gradient, baseline_gradient in zip(
        distributed_gradients, baseline_gradients, strict=True
    ):
        assert np.all(np.isfinite(distributed_gradient[0]))
        np.testing.assert_allclose(
            distributed_gradient,
            baseline_gradient,
            rtol=2e-5,
            atol=2e-6,
            equal_nan=True,
        )


def test_distributed_active_mask_preserves_rank_major_inactive_slots():
    outputs = _render_distributed_shards(
        *_distributed_render_inputs(None),
        sh_degree=None,
        active_masks=jnp.asarray([[True], [False]]),
    )
    np.testing.assert_array_equal(
        np.asarray(outputs[5]),
        np.asarray([[True, False], [True, False]]),
    )


def test_root_distributed_flag_requires_named_axis_for_multiple_ranks():
    inputs = _distributed_render_inputs(None)
    with pytest.raises(ValueError, match="axis_name"):
        jax_gs.rasterization(
            inputs[0][0],
            inputs[1][0],
            inputs[2][0],
            inputs[3][0],
            inputs[4][0],
            inputs[6][0],
            inputs[7][0],
            8,
            8,
            distributed=True,
            distributed_world_size=2,
            config=_DISTRIBUTED_RENDER_CONFIG,
        )


def test_all_to_all_rejects_runtime_ragged_layouts():
    with pytest.raises(ValueError, match="equal padded slots"):
        all_to_all_tensor_list(
            2,
            [jnp.arange(4)],
            splits=(1, 3),
            output_splits=(1, 3),
            axis_name="rank",
        )

    with pytest.raises(ValueError, match="axis_name"):
        all_gather_int32(2, 1)


def _supports_cuda() -> bool:
    device = jax.devices()[0]
    return device.platform == "gpu" and "cuda" in str(device).lower()


@pytest.mark.parametrize(
    "config",
    [
        RasterizationConfig(
            backend="intersections",
            compositor_backend="cute",
            intersection_backend="jax",
            intersection_mode="accutile",
            tile_size=16,
            max_intersections=64,
            max_candidates_per_tile=32,
        ),
        RasterizationConfig(
            backend="intersections",
            compositor_backend="jax",
            intersection_backend="cute",
            intersection_mode="accutile",
            tile_size=16,
            max_intersections=64,
            max_candidates_per_tile=32,
        ),
    ],
    ids=("compositor", "intersections"),
)
def test_distributed_rasterization_rejects_cute_backends(config):
    if not _supports_cuda():
        pytest.skip("CuTe backends require an NVIDIA CUDA GPU")

    inputs = _distributed_render_inputs(None)
    with pytest.raises(NotImplementedError, match="distributed"):
        jax_gs.rasterization(
            inputs[0][0],
            inputs[1][0],
            inputs[2][0],
            inputs[3][0],
            inputs[4][0],
            inputs[6][0],
            inputs[7][0],
            16,
            16,
            sh_degree=None,
            packed=False,
            distributed=True,
            distributed_world_size=2,
            distributed_axis_name="rank",
            config=config,
        )


def test_cli_runs_one_worker_for_the_current_jax_process(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    calls = []

    def worker(local_rank, world_rank, world_size, args):
        calls.append((local_rank, world_rank, world_size, args))

    assert cli(worker, {"value": 3})
    assert calls == [(0, jax.process_index(), jax.process_count(), {"value": 3})]
    with pytest.raises(TypeError, match="callable"):
        cli(None, None)
