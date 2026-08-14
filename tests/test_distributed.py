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
        colors = jnp.asarray(
            [[[[0.8, 0.2, 0.1]]], [[[0.1, 0.4, 0.9]]]], jnp.float32
        )
        extra_signals = jnp.asarray(
            [[[[0.2, 0.7]]], [[[0.8, 0.3]]]], jnp.float32
        )
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
            info["render_extra_signals"],
            info["distributed_requested"],
            info["distributed_world_size"],
            info["distributed_active_mask"],
        )

    return jax.vmap(render_rank, axis_name="rank")(
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
):
    global_means = means.reshape((-1,) + means.shape[2:])
    global_quats = quats.reshape((-1,) + quats.shape[2:])
    global_scales = scales.reshape((-1,) + scales.shape[2:])
    global_opacities = opacities.reshape(-1)
    global_colors = colors.reshape((-1,) + colors.shape[2:])
    global_extra_signals = extra_signals.reshape(
        (-1,) + extra_signals.shape[2:]
    )
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
        extra_signals=global_extra_signals,
        extra_signals_sh_degree=sh_degree,
        config=_DISTRIBUTED_RENDER_CONFIG,
    )
    camera_shape = viewmats.shape[:2]
    return (
        rendered.reshape(camera_shape + rendered.shape[1:]),
        alpha.reshape(camera_shape + alpha.shape[1:]),
        info["render_extra_signals"].reshape(
            camera_shape + info["render_extra_signals"].shape[1:]
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


@pytest.mark.parametrize("sh_degree", [None, 0], ids=["features", "sh"])
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

    def distributed_loss(*values):
        outputs = _render_distributed_shards(
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            viewmats,
            Ks,
            sh_degree=sh_degree,
        )
        return sum(jnp.sum(value) for value in outputs[:3])

    def baseline_loss(*values):
        outputs = _render_concatenated_baseline(
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            viewmats,
            Ks,
            sh_degree=sh_degree,
        )
        return sum(jnp.sum(value) for value in outputs)

    argument_numbers = tuple(range(len(model_inputs)))
    distributed_gradients = jax.grad(
        distributed_loss, argnums=argument_numbers
    )(*model_inputs)
    baseline_gradients = jax.grad(baseline_loss, argnums=argument_numbers)(
        *model_inputs
    )
    for distributed_gradient, baseline_gradient in zip(
        distributed_gradients, baseline_gradients, strict=True
    ):
        np.testing.assert_allclose(
            distributed_gradient, baseline_gradient, rtol=2e-5, atol=2e-6
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


def test_distributed_rasterization_cuda_ffi_parity():
    if not _supports_cuda():
        pytest.skip("CUDA FFI compositor requires an NVIDIA CUDA GPU")

    inputs = _distributed_render_inputs(None)
    model_inputs = inputs[:6]
    viewmats, Ks = inputs[6:]

    cfg_jax = RasterizationConfig(
        backend="intersections",
        compositor_backend="jax",
        intersection_backend="jax",
        intersection_mode="accutile",
        tile_size=16,
        max_intersections=64,
        max_candidates_per_tile=32,
    )
    cfg_ffi = RasterizationConfig(
        backend="intersections",
        compositor_backend="cuda_ffi",
        intersection_backend="jax",
        intersection_mode="accutile",
        tile_size=16,
        max_intersections=64,
        max_candidates_per_tile=32,
    )

    def render_distributed_cfg(
        cfg,
        m_in,
        q_in,
        s_in,
        o_in,
        c_in,
        e_in,
    ):
        def render_rank(
            local_means,
            local_quats,
            local_scales,
            local_opacities,
            local_colors,
            local_extra_signals,
            local_viewmats,
            local_Ks,
        ):
            rendered, alpha, info = jax_gs.rasterization(
                local_means,
                local_quats,
                local_scales,
                local_opacities,
                local_colors,
                local_viewmats,
                local_Ks,
                16,
                16,
                sh_degree=None,
                packed=False,
                distributed=True,
                distributed_world_size=2,
                distributed_axis_name="rank",
                config=cfg,
            )
            return rendered, alpha

        return jax.vmap(render_rank, axis_name="rank")(
            m_in,
            q_in,
            s_in,
            o_in,
            c_in,
            e_in,
            viewmats,
            Ks,
        )

    out_jax = render_distributed_cfg(cfg_jax, *model_inputs)
    out_ffi = render_distributed_cfg(cfg_ffi, *model_inputs)

    np.testing.assert_allclose(out_jax[0], out_ffi[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_jax[1], out_ffi[1], rtol=1e-4, atol=1e-4)

    def distributed_loss(cfg, m, q, s, o, c, e):
        renders, alphas = render_distributed_cfg(cfg, m, q, s, o, c, e)
        return jnp.sum(renders) + 0.01 * jnp.sum(alphas)

    argnums = (0, 1, 2, 3, 4)
    grad_jax = jax.grad(
        lambda m, q, s, o, c, e: distributed_loss(cfg_jax, m, q, s, o, c, e),
        argnums=argnums,
    )(*model_inputs)
    grad_ffi = jax.grad(
        lambda m, q, s, o, c, e: distributed_loss(cfg_ffi, m, q, s, o, c, e),
        argnums=argnums,
    )(*model_inputs)

    for g_jax, g_ffi in zip(grad_jax, grad_ffi, strict=True):
        np.testing.assert_allclose(g_jax, g_ffi, rtol=1e-4, atol=1e-4)


def test_cli_runs_one_worker_for_the_current_jax_process(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    calls = []

    def worker(local_rank, world_rank, world_size, args):
        calls.append((local_rank, world_rank, world_size, args))

    assert cli(worker, {"value": 3})
    assert calls == [(0, jax.process_index(), jax.process_count(), {"value": 3})]
    with pytest.raises(TypeError, match="callable"):
        cli(None, None)
