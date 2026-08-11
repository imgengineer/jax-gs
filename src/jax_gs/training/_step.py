"""The jitted training step, single-process and distributed."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from functools import partial
import math
import operator
import sys
from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from ..config import RasterizationConfig, TrainConfig
from ..losses import l1_loss, opacity_reg_loss, psnr, scale_reg_loss, ssim
from ..model import GaussianModel
from ..optimizers import (
    mask_inactive_gradients,
)
from ..strategy import (
    DefaultStrategy,
    DensificationStats,
    MCMCStrategy,
    StrategyState,
    reset_opacities,
)
from ..strategy.ops import mcmc_position_perturbation
from ._data import _sample_patches
from ._scene import _training_render_size
from ._setup import (
    _apply_camera_pose_modules,
    _invert_rigid_transforms,
    _validate_2dgs_mode,
)
from .appearance import (
    AppearanceOptModule,
)
from .pose import CameraOptModule

# Tests drive the trainer by patching seams on the package, for example
# monkeypatch.setattr(jax_gs.training, "rasterization", fake). Calls resolve
# through the package namespace at run time so those seams keep working now
# that the implementation lives in submodules.
_training = sys.modules[__package__]


class TrainingSafetyState(nnx.Module):
    """Device-resident sticky overflow state for one training run."""

    def __init__(self) -> None:
        self.max_overflow_tiles = nnx.Variable(jnp.array(0, jnp.int32))
        self.intersection_overflow_seen = nnx.Variable(jnp.array(False))


@dataclass
class _PendingTrainStep:
    images: np.ndarray
    intrinsics: np.ndarray
    viewmats: np.ndarray
    camtoworlds: np.ndarray | None
    image_ids: np.ndarray | None
    key: jax.Array
    strategy_key: jax.Array
    sh_degree: jax.Array
    metrics: dict[str, jax.Array]


def _unpack_training_projection_metadata(
    info: dict[str, Any],
    active_mask: jax.Array,
    *,
    camera_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Scatter padded packed projection metadata back to dense camera rows."""

    if camera_count <= 0:
        raise ValueError("packed training requires at least one camera")
    active_mask = jnp.asarray(active_mask, dtype=jnp.bool_)
    if active_mask.ndim != 1 or active_mask.shape[0] == 0:
        raise ValueError("active_mask must have shape [N] with N > 0")
    for key in (
        "camera_ids",
        "gaussian_ids",
        "radii",
        "valid",
        "projection_valid_count",
    ):
        if info.get(key) is None:
            raise ValueError(f"packed training metadata requires {key!r}")

    camera_ids = jnp.asarray(info["camera_ids"], dtype=jnp.int32)
    gaussian_ids = jnp.asarray(info["gaussian_ids"], dtype=jnp.int32)
    radii = jnp.asarray(info["radii"])
    valid = jnp.asarray(info["valid"], dtype=jnp.bool_)
    valid_count = jnp.asarray(
        info["projection_valid_count"], dtype=jnp.int32
    )
    packed_capacity = gaussian_ids.shape[0]
    if camera_ids.shape != (packed_capacity,):
        raise ValueError("packed camera_ids must have shape [P]")
    if radii.shape != (packed_capacity, 2):
        raise ValueError("packed radii must have shape [P, 2]")
    if valid.shape != (packed_capacity,):
        raise ValueError("packed valid must have shape [P]")
    if valid_count.shape != ():
        raise ValueError("projection_valid_count must be scalar")

    gaussian_count = active_mask.shape[0]
    positions = jnp.arange(packed_capacity, dtype=jnp.int32)
    packed_valid = (
        (positions < valid_count)
        & valid
        & (camera_ids >= 0)
        & (camera_ids < camera_count)
        & (gaussian_ids >= 0)
        & (gaussian_ids < gaussian_count)
        & jnp.all(radii > 0, axis=-1)
    )
    safe_camera_ids = jnp.clip(camera_ids, 0, camera_count - 1)
    safe_gaussian_ids = jnp.clip(gaussian_ids, 0, gaussian_count - 1)
    dense_radii = jnp.zeros(
        (camera_count, gaussian_count, 2), dtype=radii.dtype
    ).at[safe_camera_ids, safe_gaussian_ids].max(
        jnp.where(packed_valid[:, None], radii, 0)
    )
    dense_valid = (
        jnp.zeros((camera_count, gaussian_count), dtype=jnp.int32)
        .at[safe_camera_ids, safe_gaussian_ids]
        .max(packed_valid.astype(jnp.int32))
        > 0
    )
    visible_mask = jnp.any(dense_valid, axis=0) & active_mask
    return dense_radii, dense_valid, visible_mask


def _two_dgs_regularization_losses(
    rendered_normals: jax.Array,
    normals_from_depth: jax.Array,
    alphas: jax.Array,
    render_distort: jax.Array,
    step: jax.Array,
    config: TrainConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return weighted 2DGS normal and distortion loss contributions."""

    zero = jnp.zeros((), dtype=rendered_normals.dtype)
    normal_loss_value = zero
    if config.normal_loss:
        normal_weight = jnp.where(
            step > config.normal_start_iter,
            jnp.asarray(config.normal_lambda, rendered_normals.dtype),
            zero,
        )
        alpha_weighted_normals = normals_from_depth * jax.lax.stop_gradient(
            alphas
        )
        normal_error = 1.0 - jnp.sum(
            rendered_normals * alpha_weighted_normals, axis=-1
        )
        normal_loss_value = normal_weight * jnp.mean(normal_error)

    distortion_loss_value = zero
    if config.dist_loss:
        distortion_weight = jnp.where(
            step > config.dist_start_iter,
            jnp.asarray(config.dist_lambda, render_distort.dtype),
            jnp.zeros((), dtype=render_distort.dtype),
        )
        distortion_loss_value = distortion_weight * jnp.mean(render_distort)
    return normal_loss_value, distortion_loss_value


def _skip_topology_update(
    current_model,
    current_optimizer,
    current_strategy_state,
    current_grads,
    current_visible,
    *,
    config: TrainConfig,
    strategy_capacity_overflow: jax.Array,
    uncommitted_topology: tuple[jax.Array, ...],
):
    """Leave the model alone, because committing would truncate a gradient.

    The counterpart to :func:`_apply_topology_update` under the same
    conditional, so it takes the same five traced arguments and returns the
    same shape of result. MCMC still has to record that its own refinement was
    dropped, since its state carries the counts across steps.
    """

    del current_model, current_optimizer, current_grads, current_visible
    if config.strategy.kind == "mcmc":
        current_strategy_state.last_new_count[...] = jnp.where(
            strategy_capacity_overflow,
            0,
            current_strategy_state.last_new_count[...],
        )
        current_strategy_state.last_pruned_count[...] = jnp.where(
            strategy_capacity_overflow,
            0,
            current_strategy_state.last_pruned_count[...],
        )
        current_strategy_state.capacity_overflow[...] = (
            current_strategy_state.capacity_overflow[...]
            | strategy_capacity_overflow
        )
    return uncommitted_topology


def _apply_topology_update(
    current_model,
    current_optimizer,
    current_strategy_state,
    current_grads,
    current_visible,
    *,
    plan: _TrainStepPlan,
    config: TrainConfig,
    distributed_scene_scale: float,
    densification_stats,
    mcmc_should_refine,
    noise_key,
    refine_key,
    refine_scheduled,
    reset_scheduled,
    training_step,
    uncommitted_refine,
    uncommitted_topology,
):
    """Commit the step: optimizer update, then whatever refinement is due.

    The taken branch of the conditional :func:`_skip_topology_update` guards,
    so it carries the same five traced arguments. Everything else the step
    settled before reaching here arrives by keyword.
    """
    mcmc_commit_overflow = jnp.asarray(False)
    mcmc_commit_required = jnp.asarray(0, dtype=jnp.int32)
    with jax.named_scope("optimizer_update"):
        if plan.row_selective_optimizer:
            current_optimizer.update(
                current_model,
                current_grads,
                visible_mask=current_visible,
            )
            normalized_quats = current_model.normalized_quats
            current_model.quats[...] = jnp.where(
                current_visible[:, None],
                normalized_quats,
                current_model.quats[...],
            )
        else:
            current_optimizer.update(current_model, current_grads)
            current_model.normalize_quaternions()
        if config.strategy.kind == "mcmc":
            step_number = current_optimizer.step[...]
            def refine(_model, _optimizer, _state):
                assert plan.mcmc_strategy is not None
                commit_required = jnp.asarray(0, dtype=jnp.int32)
                if plan.distributed:
                    commit_required = plan.mcmc_strategy.required_capacity(
                        _model,
                        _state,
                        _state.scene_scale[...],
                        step=step_number,
                    )
                refine_result = plan.mcmc_strategy.refine(
                    _model,
                    _state,
                    _optimizer,
                    refine_key,
                    _state.scene_scale[...],
                    step=step_number,
                )
                return (
                    refine_result["capacity_overflow"],
                    commit_required,
                )
            def skip_refine(_model, _optimizer, _state):
                del _model, _optimizer, _state
                return (
                    jnp.asarray(False),
                    jnp.asarray(0, dtype=jnp.int32),
                )
            mcmc_commit_overflow, mcmc_commit_required = nnx.cond(
                mcmc_should_refine,
                refine,
                skip_refine,
                current_model,
                current_optimizer,
                current_strategy_state,
            )
            schedule_progress = step_number.astype(jnp.float32) / float(
                max(config.optimizer.max_steps, 1)
            )
            means_lr = config.optimizer.means_lr * jnp.power(
                config.optimizer.means_lr_final_scale,
                schedule_progress,
            )
            perturbed_means = mcmc_position_perturbation(
                current_model.means[...],
                current_model.quats[...],
                current_model.log_scales[...],
                current_model.opacity_logits[...],
                means_lr * config.strategy.noise_lr,
                key=noise_key,
                t=config.strategy.noise_opacity_t,
                k=config.strategy.noise_opacity_k,
                active_mask=current_model.active_mask[...],
            )
            noise_stop = config.strategy.noise_injection_stop_iter
            should_inject = (
                ((noise_stop < 0) | (step_number < noise_stop))
                & ~mcmc_commit_overflow
            )
            current_model.means[...] = jnp.where(
                should_inject, perturbed_means, current_model.means[...]
            )
    if config.strategy.kind == "default" and plan.collect_screen_stats:
        with jax.named_scope("strategy_stats_update"):
            accumulate_stats = training_step < config.strategy.refine_stop
            current_strategy_state.grad_accum[...] += jnp.where(
                accumulate_stats,
                densification_stats.grad_sum,
                0.0,
            )
            current_strategy_state.visible_count[...] += jnp.where(
                accumulate_stats,
                densification_stats.count,
                0.0,
            )
            current_strategy_state.max_radii[...] = jnp.where(
                accumulate_stats,
                jnp.maximum(
                    current_strategy_state.max_radii[...],
                    densification_stats.max_radii,
                ),
                current_strategy_state.max_radii[...],
            )
    if plan.distributed_plan_strategy is None:
        if plan.distributed and config.strategy.kind == "mcmc":
            return (
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                mcmc_commit_overflow,
                mcmc_commit_required,
                jnp.asarray(False),
            )
        return uncommitted_topology
    # current-main refines after the optimizer step and after this
    # step's statistics, so the owner-local commit recomputes its own
    # events here instead of replaying the pre-update preflight.
    with jax.named_scope("topology_commit"):
        def commit_refine(_model, _optimizer, _state):
            assert plan.distributed_plan_strategy is not None
            commit_plan = plan.distributed_plan_strategy.plan_refine(
                _model,
                _state,
                distributed_scene_scale,
                step=training_step,
            )
            refine_result = plan.distributed_plan_strategy.refine(
                _model,
                _state,
                _optimizer,
                refine_key,
                distributed_scene_scale,
                step=training_step,
            )
            return (
                refine_result["new_count"],
                refine_result["pruned_count"],
                refine_result["capacity_overflow"],
                commit_plan["required_capacity"],
            )
        def skip_commit_refine(_model, _optimizer, _state):
            del _model, _optimizer, _state
            return uncommitted_refine
        new_count, pruned_count, commit_overflow, commit_required = nnx.cond(
            refine_scheduled,
            commit_refine,
            skip_commit_refine,
            current_model,
            current_optimizer,
            current_strategy_state,
        )
        def commit_reset(_model, _optimizer):
            reset_opacities(
                _model,
                _optimizer,
                maximum_opacity=config.strategy.reset_opacity,
            )
            return jnp.asarray(True)
        def skip_commit_reset(_model, _optimizer):
            del _model, _optimizer
            return jnp.asarray(False)
        # An owner that could not grow keeps its statistics for the
        # next refinement, so it must not reset opacities either.
        opacity_reset = nnx.cond(
            reset_scheduled & ~commit_overflow,
            commit_reset,
            skip_commit_reset,
            current_model,
            current_optimizer,
        )
    return (
        new_count,
        pruned_count,
        commit_overflow,
        commit_required,
        opacity_reset,
    )


class _ResolvedBatch(NamedTuple):
    """One training batch after sampling, posing and sizing.

    The step resolves these before it can render: the crop actually being
    scored, the cameras to render it from, and the sizes and step number
    that follow. Grouping them keeps the loss's own arguments to the ones
    it is differentiated with respect to.
    """

    targets: jax.Array
    viewmats: jax.Array
    camtoworlds: jax.Array | None
    image_ids: jax.Array | None
    patch_intrinsics: jax.Array
    backgrounds: jax.Array
    render_height: int
    render_width: int
    sh_degree: jax.Array
    training_step: jax.Array
    uses_camera_modules: bool
    pose_perturb: CameraOptModule | None


def _training_loss(
    current_model: GaussianModel,
    current_pose_adjust: CameraOptModule | None,
    current_screen_probe: jax.Array,
    current_appearance_module: AppearanceOptModule | None = None,
    *,
    batch: _ResolvedBatch,
    plan: _TrainStepPlan,
    config: TrainConfig,
    distributed_axis_name: Hashable | None,
    distributed_world_size: int,
):
    """Render the batch and score it.

    The differentiated arguments stay positional, because the step takes their
    gradients by index and which ones are live depends on whether pose and
    appearance are being optimized. Everything the batch already resolved, and
    everything the step was compiled around, arrives by keyword.
    """
    render_viewmats = batch.viewmats
    adjusted_camtoworlds = None
    if batch.uses_camera_modules:
        assert batch.camtoworlds is not None
        assert batch.image_ids is not None
        adjusted_camtoworlds = _apply_camera_pose_modules(
            batch.camtoworlds,
            batch.image_ids,
            pose_adjust=(
                current_pose_adjust if config.pose_opt else None
            ),
            pose_perturb=batch.pose_perturb,
        )
        render_viewmats = _invert_rigid_transforms(
            adjusted_camtoworlds
        )
    parameters = current_model.activated(
        split_sh=(
            config.model_type == "3dgs" and not config.app_opt
        )
    )
    if config.app_opt:
        assert current_appearance_module is not None
        assert adjusted_camtoworlds is not None
        assert batch.image_ids is not None
        directions = (
            parameters["means"][None, :, :]
            - adjusted_camtoworlds[:, None, :3, 3]
        )
        corrections = current_appearance_module(
            parameters["features"],
            batch.image_ids,
            directions,
            batch.sh_degree,
        )
        render_colors = jax.nn.sigmoid(
            parameters["colors"][None, :, :] + corrections
        )
        raster_sh_degree = None
    else:
        render_colors = parameters["sh_coeffs"]
        raster_sh_degree = batch.sh_degree
    if config.model_type == "2dgs":
        (
            renders,
            alphas,
            rendered_normals,
            normals_from_depth,
            render_distort,
            _,
            info,
        ) = _training.rasterization_2dgs(
            parameters["means"],
            parameters["quats"],
            parameters["scales"],
            parameters["opacities"],
            render_colors,
            render_viewmats,
            batch.patch_intrinsics,
            batch.render_width,
            batch.render_height,
            packed=config.packed,
            sparse_grad=config.sparse_grad,
            absgrad=plan.use_absgrad,
            active_mask=parameters["active_mask"],
            sh_degree=raster_sh_degree,
            backgrounds=batch.backgrounds,
            render_mode="RGB",
            distloss=config.dist_loss,
            config=plan.rasterizer_config,
            _gradient_2dgs_offset=(
                None
                if plan.use_absgrad or not plan.collect_screen_stats
                else current_screen_probe
            ),
            _gradient_2dgs_absgrad_probe=(
                current_screen_probe
                if plan.use_absgrad and plan.collect_screen_stats
                else None
            ),
        )
        normal_loss_value, distortion_loss_value = (
            _two_dgs_regularization_losses(
                rendered_normals,
                normals_from_depth,
                alphas,
                render_distort,
                batch.training_step,
                config,
            )
        )
    else:
        renders, _, info = _training.rasterization(
            parameters["means"],
            parameters["quats"],
            parameters["scales"],
            parameters["opacities"],
            render_colors,
            render_viewmats,
            batch.patch_intrinsics,
            batch.render_width,
            batch.render_height,
            packed=config.packed,
            sparse_grad=config.sparse_grad,
            absgrad=plan.use_absgrad,
            active_mask=parameters["active_mask"],
            sh_degree=raster_sh_degree,
            backgrounds=batch.backgrounds,
            camera_model=config.camera_model,
            with_ut=config.with_ut,
            with_eval3d=config.with_eval3d,
            distributed=plan.distributed,
            distributed_world_size=distributed_world_size,
            distributed_axis_name=distributed_axis_name,
            config=plan.rasterizer_config,
            _means2d_offset=(
                None
                if plan.use_absgrad or not plan.collect_screen_stats
                else current_screen_probe
            ),
            _means2d_absgrad_probe=(
                current_screen_probe
                if plan.use_absgrad and plan.collect_screen_stats
                else None
            ),
        )
        normal_loss_value = jnp.zeros((), dtype=renders.dtype)
        distortion_loss_value = jnp.zeros((), dtype=renders.dtype)
    rgb = renders[..., :3]
    l1_value = jnp.mean(l1_loss(rgb, batch.targets))
    ssim_value = ssim(rgb, batch.targets)
    photometric_loss = (1.0 - plan.ssim_lambda) * l1_value + plan.ssim_lambda * (
        1.0 - ssim_value
    )
    opacity_reg_loss_value = jnp.zeros(
        (), dtype=photometric_loss.dtype
    )
    if config.opacity_reg > 0.0:
        opacity_reg_loss_value = jnp.asarray(
            config.opacity_reg, dtype=photometric_loss.dtype
        ) * opacity_reg_loss(
            current_model.opacity_logits[...],
            mask=current_model.active_mask[...],
        )
    scale_reg_loss_value = jnp.zeros(
        (), dtype=photometric_loss.dtype
    )
    if config.scale_reg > 0.0:
        scale_reg_loss_value = jnp.asarray(
            config.scale_reg, dtype=photometric_loss.dtype
        ) * scale_reg_loss(
            current_model.log_scales[...],
            mask=current_model.active_mask[...],
        )
    loss = (
        photometric_loss
        + normal_loss_value
        + distortion_loss_value
        + opacity_reg_loss_value
        + scale_reg_loss_value
    )
    pose_error_value = jnp.zeros((), dtype=loss.dtype)
    if config.pose_opt and config.pose_noise > 0.0:
        assert adjusted_camtoworlds is not None
        assert batch.camtoworlds is not None
        pose_error_value = jnp.mean(
            jnp.abs(adjusted_camtoworlds - batch.camtoworlds)
        )
    return loss, (
        l1_value,
        ssim_value,
        normal_loss_value,
        distortion_loss_value,
        opacity_reg_loss_value,
        scale_reg_loss_value,
        pose_error_value,
        rgb,
        info,
    )


class _TrainStepPlan(NamedTuple):
    """What a training step's configuration settles before it can be traced.

    These are the values the step closes over: fixed for the life of the
    compiled step, and derived only from the configuration and the distributed
    topology. Naming them together separates deciding what the step will do
    from doing it, and makes the decisions checkable on their own.
    """

    patch_size: int | None
    rasterizer_config: RasterizationConfig
    ssim_lambda: float
    random_background: bool
    use_absgrad: bool
    row_selective_optimizer: bool
    distributed: bool
    collect_screen_stats: bool
    mcmc_strategy: MCMCStrategy | None
    distributed_plan_strategy: DefaultStrategy | None
    donated_nodes: tuple[str, ...]


def _plan_train_step(
    config: TrainConfig, *, distributed_world_size: int = 1
) -> _TrainStepPlan:
    """Settle the fixed decisions a training step is compiled around."""

    _validate_2dgs_mode(config)
    if config.with_eval3d and config.strategy.kind != "mcmc":
        raise NotImplementedError(
            "screen-space densification statistics do not yet support "
            "with_eval3d=True"
        )
    distributed = distributed_world_size > 1
    return _TrainStepPlan(
        patch_size=config.data.patch_size,
        rasterizer_config=config.rasterizer,
        ssim_lambda=config.ssim_lambda,
        random_background=config.random_background,
        use_absgrad=config.strategy.absgrad,
        row_selective_optimizer=config.sparse_grad or config.visible_adam,
        distributed=distributed,
        # MCMC densifies from its own state, so the screen-space statistics are
        # only collected when something will read them.
        collect_screen_stats=not (
            config.strategy.kind == "mcmc"
            and (config.with_ut or config.with_eval3d)
        ),
        mcmc_strategy=(
            MCMCStrategy(config.strategy)
            if config.strategy.kind == "mcmc"
            else None
        ),
        # Distributed refinement has no host callback, so the owner-local
        # default strategy must plan inside the step to preflight every shard
        # together.
        distributed_plan_strategy=(
            DefaultStrategy(config.strategy)
            if distributed and config.strategy.kind == "default"
            else None
        ),
        donated_nodes=(
            "model",
            "optimizer",
            "strategy_state",
            "safety_state",
        )
        + (("pose_adjust", "pose_optimizer") if config.pose_opt else ())
        + (
            ("appearance_module", "appearance_optimizer")
            if config.app_opt
            else ()
        ),
    )


def _make_train_step(
    config: TrainConfig,
    *,
    distributed_world_size: int = 1,
    distributed_axis_name: Hashable | None = None,
    distributed_scene_scale: float = 1.0,
) -> Callable[..., dict[str, jax.Array]]:
    plan = _plan_train_step(
        config, distributed_world_size=distributed_world_size
    )
    # Bound by name rather than unpacked, so the step body reads the same as
    # before the plan existed and adding a field cannot silently shift these.
    # The fields only the loss and the commit stage read stay on the plan and
    # travel to them by keyword.
    patch_size = plan.patch_size
    random_background = plan.random_background
    distributed = plan.distributed
    collect_screen_stats = plan.collect_screen_stats
    mcmc_strategy = plan.mcmc_strategy
    distributed_plan_strategy = plan.distributed_plan_strategy

    @nnx.jit(donate_argnames=plan.donated_nodes)
    def train_step(
        model: GaussianModel,
        optimizer: nnx.Optimizer,
        strategy_state: StrategyState,
        safety_state: TrainingSafetyState,
        images: jax.Array,
        intrinsics: jax.Array,
        viewmats: jax.Array,
        key: jax.Array,
        sh_degree: jax.Array,
        strategy_key: jax.Array | None = None,
        *,
        pose_adjust: CameraOptModule | None = None,
        pose_optimizer: nnx.Optimizer | None = None,
        camtoworlds: jax.Array | None = None,
        image_ids: jax.Array | None = None,
        pose_perturb: CameraOptModule | None = None,
        appearance_module: AppearanceOptModule | None = None,
        appearance_optimizer: nnx.Optimizer | None = None,
    ) -> dict[str, jax.Array]:
        distributed_state_mismatch = jnp.asarray(False)
        if distributed:
            optimizer_batch_size = getattr(
                optimizer, "_jax_gs_batch_size", None
            )
            optimizer_world_size = getattr(
                optimizer, "_jax_gs_world_size", None
            )
            optimizer_scene_scale = getattr(
                optimizer, "_jax_gs_scene_scale", None
            )
            optimizer_config = getattr(
                optimizer, "_jax_gs_optimizer_config", None
            )
            if (
                optimizer_batch_size != config.data.batch_size
                or optimizer_world_size != distributed_world_size
                or optimizer_scene_scale != distributed_scene_scale
                or optimizer_config != config.optimizer
            ):
                raise ValueError(
                    "distributed optimizer must be created with "
                    f"batch_size={config.data.batch_size} and "
                    f"world_size={distributed_world_size}, "
                    f"scene_scale={distributed_scene_scale}, and the train "
                    "step's OptimizerConfig; got "
                    f"batch_size={optimizer_batch_size!r} and "
                    f"world_size={optimizer_world_size!r}, "
                    f"scene_scale={optimizer_scene_scale!r}"
                )
            if images.shape[0] != config.data.batch_size:
                raise ValueError(
                    "distributed rank-local image batch must match "
                    f"config.data.batch_size={config.data.batch_size}"
                )
            assert distributed_axis_name is not None
            optimizer_step = optimizer.step[...]
            minimum_step = jax.lax.pmin(
                optimizer_step, distributed_axis_name
            )
            maximum_step = jax.lax.pmax(
                optimizer_step, distributed_axis_name
            )
            minimum_sh_degree = jax.lax.pmin(
                sh_degree, distributed_axis_name
            )
            maximum_sh_degree = jax.lax.pmax(
                sh_degree, distributed_axis_name
            )
            distributed_state_mismatch = (
                (minimum_step != maximum_step)
                | (minimum_sh_degree != maximum_sh_degree)
            )
        uses_camera_modules = (
            config.pose_opt or config.pose_noise > 0.0 or config.app_opt
        )
        if model.has_appearance != config.app_opt:
            raise ValueError(
                "model color representation must match config.app_opt"
            )
        if config.pose_opt and (pose_adjust is None or pose_optimizer is None):
            raise ValueError(
                "pose_opt=True requires pose_adjust and pose_optimizer"
            )
        if config.app_opt and (
            appearance_module is None or appearance_optimizer is None
        ):
            raise ValueError(
                "app_opt=True requires appearance_module and "
                "appearance_optimizer"
            )
        if uses_camera_modules and (camtoworlds is None or image_ids is None):
            raise ValueError(
                "camera modules require camtoworlds and image_ids"
            )
        if config.pose_noise > 0.0 and pose_perturb is None:
            raise ValueError("pose_noise > 0 requires pose_perturb")
        if distributed and config.pose_opt:
            assert distributed_axis_name is not None
            assert pose_optimizer is not None
            expected_pose_contract = (
                config.data.batch_size,
                config.steps,
                float(config.pose_opt_lr),
                float(config.pose_opt_reg),
            )
            if (
                getattr(
                    pose_optimizer, "_jax_gs_pose_contract", None
                )
                != expected_pose_contract
            ):
                raise ValueError(
                    "distributed pose optimizer does not match the train "
                    "step's batch size, schedule, learning rate, or "
                    "regularization"
                )
            pose_optimizer_step = pose_optimizer.step[...]
            minimum_pose_step = jax.lax.pmin(
                pose_optimizer_step, distributed_axis_name
            )
            maximum_pose_step = jax.lax.pmax(
                pose_optimizer_step, distributed_axis_name
            )
            distributed_state_mismatch = (
                distributed_state_mismatch
                | (minimum_pose_step != maximum_pose_step)
                | (pose_optimizer_step != optimizer_step)
            )
        patch_key, background_key = jax.random.split(key)
        if strategy_key is None:
            strategy_key = jax.random.fold_in(key, 0x53545241)
        refine_key, noise_key = jax.random.split(strategy_key)
        image_height, image_width = images.shape[-3:-1]
        render_height, render_width = _training_render_size(
            config,
            image_height=image_height,
            image_width=image_width,
        )
        targets, patch_intrinsics = _sample_patches(
            images, intrinsics, patch_key, patch_size
        )
        if random_background:
            backgrounds = jax.random.uniform(
                background_key, (images.shape[0], 3), dtype=images.dtype
            )
        else:
            backgrounds = jnp.zeros((images.shape[0], 3), dtype=images.dtype)
        # The host training loop is one-based while optimizer.step is incremented
        # after this loss. Match the host/upstream iteration used by start_iter.
        training_step = optimizer.step[...] + jnp.asarray(
            1, dtype=optimizer.step[...].dtype
        )

        # Keep distributed camera probes independent while spanning the global
        # Gaussian axis. Gathering local probes would sum signed gradients in
        # the gather VJP before the strategy can take each camera's norm.
        screen_gaussian_count = model.means.shape[0] * (
            distributed_world_size if distributed else 1
        )
        screen_probe_shape = (
            (viewmats.shape[0], screen_gaussian_count, 2)
            if collect_screen_stats
            else ()
        )
        screen_probe = jnp.zeros(
            screen_probe_shape, dtype=model.means[...].dtype
        )

        batch = _ResolvedBatch(
            targets=targets,
            viewmats=viewmats,
            camtoworlds=camtoworlds,
            image_ids=image_ids,
            patch_intrinsics=patch_intrinsics,
            backgrounds=backgrounds,
            render_height=render_height,
            render_width=render_width,
            sh_degree=sh_degree,
            training_step=training_step,
            uses_camera_modules=uses_camera_modules,
            pose_perturb=pose_perturb,
        )
        loss_fn = partial(
            _training_loss,
            batch=batch,
            plan=plan,
            config=config,
            distributed_axis_name=distributed_axis_name,
            distributed_world_size=distributed_world_size,
        )

        with jax.named_scope("loss_and_backward"):
            if config.pose_opt and config.app_opt:
                result, gradients = nnx.value_and_grad(
                    lambda current_model, current_pose, current_appearance,
                    current_screen: loss_fn(
                        current_model,
                        current_pose,
                        current_screen,
                        current_appearance,
                    ),
                    argnums=(0, 1, 2, 3),
                    has_aux=True,
                )(
                    model,
                    pose_adjust,
                    appearance_module,
                    screen_probe,
                )
                grads, pose_grads, appearance_grads, screen_grad = gradients
            elif config.pose_opt:
                result, gradients = nnx.value_and_grad(
                    loss_fn, argnums=(0, 1, 2), has_aux=True
                )(model, pose_adjust, screen_probe)
                grads, pose_grads, screen_grad = gradients
                appearance_grads = None
            elif config.app_opt:
                result, gradients = nnx.value_and_grad(
                    lambda current_model, current_appearance, current_screen: (
                        loss_fn(
                            current_model,
                            None,
                            current_screen,
                            current_appearance,
                        )
                    ),
                    argnums=(0, 1, 2),
                    has_aux=True,
                )(model, appearance_module, screen_probe)
                grads, appearance_grads, screen_grad = gradients
                pose_grads = None
            else:
                result, gradients = nnx.value_and_grad(
                    lambda current_model, current_screen: loss_fn(
                        current_model, None, current_screen
                    ),
                    argnums=(0, 1),
                    has_aux=True,
                )(model, screen_probe)
                grads, screen_grad = gradients
                pose_grads = None
                appearance_grads = None
            loss, (
                l1_value,
                ssim_value,
                normal_loss_value,
                distortion_loss_value,
                opacity_reg_loss_value,
                scale_reg_loss_value,
                pose_error_value,
                rgb,
                info,
            ) = result
        if distributed and config.pose_opt:
            assert distributed_axis_name is not None
            assert pose_grads is not None
            # Current-main wraps the camera-pose module in DDP, which averages
            # the replicated gradient across ranks. Gaussians stay sharded and
            # keep the owner-scattered sum instead.
            pose_grads = jax.lax.pmean(pose_grads, distributed_axis_name)
        active_mask = model.active_mask[...]
        with jax.named_scope("inactive_grad_mask"):
            grads = jax.lax.cond(
                jnp.all(active_mask),
                lambda current: current,
                lambda current: mask_inactive_gradients(
                    current, active_mask
                ),
                grads,
            )
        owner_start = None
        if distributed:
            assert distributed_axis_name is not None
            owner_start = (
                jax.lax.axis_index(distributed_axis_name) * model.capacity
            )
        if config.packed:
            packed_active_mask = active_mask
            if distributed:
                assert distributed_axis_name is not None
                # Packed ids index the gathered scene, so the metadata unpacks
                # against the global active mask and is sliced afterwards.
                packed_active_mask = jax.lax.all_gather(
                    active_mask,
                    distributed_axis_name,
                    axis=0,
                    tiled=True,
                )
            stats_projection_radii, stats_projection_valid, visible = (
                _unpack_training_projection_metadata(
                    info,
                    packed_active_mask,
                    camera_count=viewmats.shape[0],
                )
            )
            stats_active_mask = packed_active_mask
            if distributed:
                assert distributed_axis_name is not None
                visible = (
                    jax.lax.pmax(
                        visible.astype(jnp.int32), distributed_axis_name
                    )
                    > 0
                )
                visible = jax.lax.dynamic_slice_in_dim(
                    visible,
                    owner_start,
                    model.capacity,
                    axis=0,
                )
                visible = visible & active_mask
        else:
            stats_projection_radii = info["radii"]
            stats_projection_valid = info["valid"]
            stats_active_mask = active_mask
            visible = jnp.any(
                jnp.asarray(stats_projection_valid, dtype=jnp.bool_)
                & jnp.all(jnp.asarray(stats_projection_radii) > 0, axis=-1),
                axis=0,
            )
            if distributed:
                assert distributed_axis_name is not None
                visible = (
                    jax.lax.pmax(
                        visible.astype(jnp.int32),
                        distributed_axis_name,
                    )
                    > 0
                )
                stats_active_mask = jax.lax.all_gather(
                    active_mask,
                    distributed_axis_name,
                    axis=0,
                    tiled=True,
                )
                visible = jax.lax.dynamic_slice_in_dim(
                    visible,
                    owner_start,
                    model.capacity,
                    axis=0,
                )
            visible = visible & active_mask
        # Only the branches that produce these read them again, but the commit
        # stage now receives them by argument rather than closing over them, so
        # they have to be bound on every path.
        densification_stats = None
        if collect_screen_stats:
            densification_stats = _training.build_densification_stats(
                screen_grad,
                stats_projection_radii,
                stats_projection_valid,
                stats_active_mask,
                render_width,
                render_height,
            )
            if distributed:
                assert distributed_axis_name is not None
                # The per-camera norm is already in these scalar statistics.
                # Reduce globally before slicing the current Gaussian owner.
                global_stats = DensificationStats(
                    jax.lax.psum(
                        densification_stats.grad_sum,
                        distributed_axis_name,
                    ),
                    jax.lax.psum(
                        densification_stats.count,
                        distributed_axis_name,
                    ),
                    jax.lax.pmax(
                        densification_stats.max_radii,
                        distributed_axis_name,
                    ),
                )

                def owner_slice(value):
                    return jax.lax.dynamic_slice_in_dim(
                        value,
                        owner_start,
                        model.capacity,
                        axis=0,
                    )

                densification_stats = DensificationStats(
                    owner_slice(global_stats.grad_sum),
                    owner_slice(global_stats.count),
                    owner_slice(global_stats.max_radii),
                )
        overflow_tiles = jnp.count_nonzero(info["tile_overflow"])
        intersection_overflow = jnp.any(info["intersection_overflow"])
        previous_max_overflow_tiles = safety_state.max_overflow_tiles[...]
        previous_intersection_overflow_seen = (
            safety_state.intersection_overflow_seen[...]
        )
        if distributed:
            assert distributed_axis_name is not None
            overflow_tiles = jax.lax.psum(
                overflow_tiles, distributed_axis_name
            )
            intersection_overflow = (
                jax.lax.pmax(
                    intersection_overflow.astype(jnp.int32),
                    distributed_axis_name,
                )
                > 0
            )
            previous_max_overflow_tiles = jax.lax.pmax(
                previous_max_overflow_tiles, distributed_axis_name
            )
            previous_intersection_overflow_seen = (
                jax.lax.pmax(
                    previous_intersection_overflow_seen.astype(jnp.int32),
                    distributed_axis_name,
                )
                > 0
            )
        max_overflow_tiles = jnp.maximum(
            previous_max_overflow_tiles, overflow_tiles
        )
        intersection_overflow_seen = (
            previous_intersection_overflow_seen | intersection_overflow
        )
        safety_state.max_overflow_tiles[...] = max_overflow_tiles
        safety_state.intersection_overflow_seen[...] = intersection_overflow_seen
        strategy_capacity_overflow = jnp.asarray(False)
        mcmc_should_refine = jnp.asarray(False)
        refine_scheduled = None
        reset_scheduled = None
        plan_metrics: dict[str, jax.Array] = {}
        if config.strategy.kind == "mcmc":
            mcmc_should_refine = (
                (training_step > config.strategy.refine_start)
                & (training_step < config.strategy.refine_stop)
                & (training_step % config.strategy.refine_every == 0)
            )
            assert mcmc_strategy is not None
            refine_plan = mcmc_strategy.plan_refine(
                model,
                strategy_state,
                strategy_state.scene_scale[...],
                step=training_step,
            )
            strategy_capacity_overflow = (
                mcmc_should_refine & refine_plan["capacity_overflow"]
            )
            if distributed:
                assert distributed_axis_name is not None
                # Every shard caps and grows its own rows exactly as an
                # independent upstream rank does, but one shard that cannot
                # allocate must not advance its schedule alone.
                strategy_capacity_overflow = (
                    jax.lax.pmax(
                        strategy_capacity_overflow.astype(jnp.int32),
                        distributed_axis_name,
                    )
                    > 0
                )
                plan_metrics = {
                    "refine_scheduled": mcmc_should_refine,
                    "refine_planned_new_count": jnp.where(
                        mcmc_should_refine,
                        jax.lax.psum(
                            refine_plan["planned_new_count"],
                            distributed_axis_name,
                        ),
                        0,
                    ),
                    "refine_required_capacity": jnp.where(
                        mcmc_should_refine,
                        jax.lax.pmax(
                            refine_plan["required_capacity"],
                            distributed_axis_name,
                        ),
                        0,
                    ),
                    "refine_capacity_overflow": strategy_capacity_overflow,
                }
        elif distributed_plan_strategy is not None:
            assert distributed_axis_name is not None
            refine_scheduled = (
                (training_step > config.strategy.refine_start)
                & (training_step < config.strategy.refine_stop)
                & (training_step % config.strategy.refine_every == 0)
                & (
                    training_step % config.strategy.reset_every
                    >= config.strategy.pause_refine_after_reset
                )
            )
            reset_scheduled = (
                (training_step > 0)
                & (training_step < config.strategy.refine_stop)
                & (training_step % config.strategy.reset_every == 0)
            )
            # Every shard decides on its own rows. The step's scene scale is
            # the value already checked against the optimizer, so no rank can
            # score growth or pruning against a different threshold.
            owner_plan = distributed_plan_strategy.plan_refine(
                model,
                strategy_state,
                distributed_scene_scale,
                step=training_step,
            )
            # Only the plan's scalar summaries cross ranks; the owner-local
            # [L] decision arrays must never be reduced.
            planned_new_count = jax.lax.psum(
                owner_plan["planned_new_count"], distributed_axis_name
            )
            planned_pruned_count = jax.lax.psum(
                owner_plan["pruned_count"], distributed_axis_name
            )
            planned_required_capacity = jax.lax.pmax(
                owner_plan["required_capacity"], distributed_axis_name
            )
            any_rank_capacity_overflow = (
                jax.lax.pmax(
                    owner_plan["capacity_overflow"].astype(jnp.int32),
                    distributed_axis_name,
                )
                > 0
            )
            strategy_capacity_overflow = (
                refine_scheduled & any_rank_capacity_overflow
            )
            plan_metrics = {
                "refine_scheduled": refine_scheduled,
                "reset_scheduled": reset_scheduled,
                "refine_planned_new_count": jnp.where(
                    refine_scheduled, planned_new_count, 0
                ),
                "refine_planned_pruned_count": jnp.where(
                    refine_scheduled, planned_pruned_count, 0
                ),
                "refine_required_capacity": jnp.where(
                    refine_scheduled, planned_required_capacity, 0
                ),
                "refine_capacity_overflow": strategy_capacity_overflow,
            }
        has_overflow = (
            intersection_overflow_seen
            | (max_overflow_tiles > 0)
            | strategy_capacity_overflow
            | distributed_state_mismatch
        )

        # Committed topology counters leave both update branches so that their
        # reporting collectives run after the branches rejoin, never inside a
        # conditional.
        uncommitted_refine = (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int32),
        )
        uncommitted_topology = uncommitted_refine + (jnp.asarray(False),)

        skip_update = partial(
            _skip_topology_update,
            config=config,
            strategy_capacity_overflow=strategy_capacity_overflow,
            uncommitted_topology=uncommitted_topology,
        )

        apply_update = partial(
            _apply_topology_update,
            plan=plan,
            config=config,
            distributed_scene_scale=distributed_scene_scale,
            densification_stats=densification_stats,
            mcmc_should_refine=mcmc_should_refine,
            noise_key=noise_key,
            refine_key=refine_key,
            refine_scheduled=refine_scheduled,
            reset_scheduled=reset_scheduled,
            training_step=training_step,
            uncommitted_refine=uncommitted_refine,
            uncommitted_topology=uncommitted_topology,
        )

        (
            committed_new_count,
            committed_pruned_count,
            committed_overflow,
            committed_required_capacity,
            committed_opacity_reset,
        ) = nnx.cond(
            has_overflow,
            skip_update,
            apply_update,
            model,
            optimizer,
            strategy_state,
            grads,
            visible,
        )
        if distributed_plan_strategy is not None:
            assert distributed_axis_name is not None
            global_commit_overflow = (
                jax.lax.pmax(
                    committed_overflow.astype(jnp.int32),
                    distributed_axis_name,
                )
                > 0
            )
            plan_metrics = {
                **plan_metrics,
                "refine_new_count": jax.lax.psum(
                    committed_new_count, distributed_axis_name
                ),
                "refine_pruned_count": jax.lax.psum(
                    committed_pruned_count, distributed_axis_name
                ),
                "refine_commit_overflow": global_commit_overflow,
                "refine_commit_required_capacity": jnp.where(
                    global_commit_overflow,
                    jax.lax.pmax(
                        committed_required_capacity,
                        distributed_axis_name,
                    ),
                    0,
                ),
                "opacity_reset": (
                    jax.lax.pmax(
                        committed_opacity_reset.astype(jnp.int32),
                        distributed_axis_name,
                    )
                    > 0
                ),
            }
        elif distributed and mcmc_strategy is not None:
            assert distributed_axis_name is not None
            global_commit_overflow = (
                jax.lax.pmax(
                    committed_overflow.astype(jnp.int32),
                    distributed_axis_name,
                )
                > 0
            )
            plan_metrics = {
                **plan_metrics,
                "refine_commit_overflow": global_commit_overflow,
                "refine_commit_required_capacity": jnp.where(
                    global_commit_overflow,
                    jax.lax.pmax(
                        committed_required_capacity,
                        distributed_axis_name,
                    ),
                    0,
                ),
            }
        if config.pose_opt:
            assert pose_adjust is not None
            assert pose_optimizer is not None
            assert pose_grads is not None

            def skip_pose_update(_module, _optimizer, _grads):
                del _module, _optimizer, _grads
                return jnp.asarray(0, dtype=jnp.int32)

            def apply_pose_update(current_module, current_optimizer, current_grads):
                current_optimizer.update(current_module, current_grads)
                return jnp.asarray(0, dtype=jnp.int32)

            nnx.cond(
                has_overflow,
                skip_pose_update,
                apply_pose_update,
                pose_adjust,
                pose_optimizer,
                pose_grads,
            )
        if config.app_opt:
            assert appearance_module is not None
            assert appearance_optimizer is not None
            assert appearance_grads is not None

            def skip_appearance_update(_module, _optimizer, _grads):
                del _module, _optimizer, _grads
                return jnp.asarray(0, dtype=jnp.int32)

            def apply_appearance_update(
                current_module, current_optimizer, current_grads
            ):
                current_optimizer.update(current_module, current_grads)
                return jnp.asarray(0, dtype=jnp.int32)

            nnx.cond(
                has_overflow,
                skip_appearance_update,
                apply_appearance_update,
                appearance_module,
                appearance_optimizer,
                appearance_grads,
            )

        candidate_limit_exceeded_tiles = jnp.count_nonzero(
            info["candidate_limit_exceeded"]
        )
        # What the compositor's chunk loop would have to cover. Without a
        # RasterizationConfig.max_candidates_per_tile the loop is sized for
        # the worst tile the shapes allow, which is far above what a real
        # frame holds; reporting the real figure is what lets a caller choose
        # that bound instead of guessing at it.
        busiest_tile_candidates = jnp.max(info["candidate_counts"])
        intersection_count = jnp.sum(info["intersection_count"])
        intersection_required_count = jnp.max(
            info["intersection_required_count"]
        )
        if distributed:
            assert distributed_axis_name is not None
            candidate_limit_exceeded_tiles = jax.lax.psum(
                candidate_limit_exceeded_tiles, distributed_axis_name
            )
            # The bound is one static number for the world, so the world's
            # busiest tile is what has to fit, not each rank's own.
            busiest_tile_candidates = jax.lax.pmax(
                busiest_tile_candidates, distributed_axis_name
            )
            intersection_count = jax.lax.psum(
                intersection_count, distributed_axis_name
            )
            intersection_required_count = jax.lax.pmax(
                intersection_required_count, distributed_axis_name
            )

        return {
            "loss": loss,
            "l1": l1_value,
            "ssim": ssim_value,
            "normal_loss": normal_loss_value,
            "distortion_loss": distortion_loss_value,
            "opacity_reg_loss": opacity_reg_loss_value,
            "scale_reg_loss": scale_reg_loss_value,
            "pose_error": pose_error_value,
            "psnr": psnr(rgb, targets),
            "active_count": model.active_count,
            "visible_count": jnp.count_nonzero(visible),
            "overflow_tiles": overflow_tiles,
            "max_overflow_tiles": max_overflow_tiles,
            "candidate_limit_exceeded_tiles": candidate_limit_exceeded_tiles,
            "busiest_tile_candidates": busiest_tile_candidates,
            "intersection_overflow": intersection_overflow,
            "intersection_overflow_seen": intersection_overflow_seen,
            "intersection_count": intersection_count,
            "intersection_required_count": intersection_required_count,
            "distributed_state_mismatch": distributed_state_mismatch,
            **plan_metrics,
        }

    return train_step


def make_train_step(
    config: TrainConfig,
) -> Callable[..., dict[str, jax.Array]]:
    """Create the ordinary single-process training step."""

    return _make_train_step(config)


def make_distributed_train_step(
    config: TrainConfig,
    *,
    world_size: int,
    axis_name: Hashable = "rank",
    scene_scale: float = 1.0,
) -> Callable[..., dict[str, jax.Array]]:
    """Create the current-main Gaussian-sharded training step.

    The returned stateful step must run inside ``nnx.pmap`` (or ``nnx.vmap``
    for tests) with ``axis_name`` bound. It supports dense or packed pinhole
    3DGS with SH colors. Host camera sharding, capacity synchronization,
    checkpoint/reshard, and eval are separate primitives; the local-device
    :func:`train_distributed` loop wires them together, while multiple JAX
    processes still fail fast.
    ``scene_scale`` must match the value passed to the Gaussian optimizer. A
    rank mismatch in optimizer step or SH degree returns
    ``distributed_state_mismatch=True`` and atomically skips the update.
    Signed screen-space statistics are reduced into each Gaussian owner;
    current-main distributed rendering does not support AbsGrad. Camera-pose
    optimization and pose noise are supported: the
    replicated module's gradient is averaged across ranks, matching the DDP
    wrapper current-main puts around it, while Gaussians keep their sharded
    sum. Packed projection, ``visible_adam``, and MCMC are supported too;
    packed metadata indexes the gathered scene, so it unpacks globally before
    the owner slice, and every MCMC shard caps and grows its own rows the way
    an independent upstream rank does. Appearance and ``sparse_grad`` remain
    conservative implementation boundaries of this JAX slice; current-main's
    trainer can orchestrate more distributed combinations than this step.

    Refinement is planned, preflighted, and committed inside the step. Before
    the update every rank plans duplicate/split/prune events for the rows it
    owns; a planned capacity overflow on any single rank atomically skips the
    whole step on every rank so a host can grow all shards and replay. After
    the update each owner commits its own duplicate/split/prune and scheduled
    opacity reset with the ordinary :class:`DefaultStrategy`, matching
    current-main's post-optimizer callback order. Physical shard capacity
    never changes here; a commit that outgrows it reports the recomputed
    ``refine_commit_required_capacity`` so the host can grow before the next
    refine. Growing a bucket, resharding, and distributed checkpoints remain
    host work; :func:`train_distributed` wires those pieces for one process's
    local devices. All collectives stay outside conditionals: plan summaries
    are reduced before the update and commit counters after it.
    """

    try:
        world_size = operator.index(world_size)
    except TypeError as exc:
        raise TypeError("world_size must be an integer") from exc
    if world_size <= 1:
        raise ValueError("world_size must be greater than one")
    if axis_name is None or not isinstance(axis_name, Hashable):
        raise TypeError("axis_name must be hashable")
    try:
        scene_scale = float(scene_scale)
    except (TypeError, ValueError) as exc:
        raise TypeError("scene_scale must be a real scalar") from exc
    if not math.isfinite(scene_scale) or scene_scale < 0.0:
        raise ValueError("scene_scale must be finite and non-negative")
    if config.optimizer.max_steps != config.steps:
        raise ValueError(
            "distributed training requires optimizer.max_steps to equal "
            "TrainConfig.steps so every rank uses one schedule horizon"
        )
    if config.model_type != "3dgs":
        raise NotImplementedError(
            "distributed training does not support 2DGS"
        )
    if config.app_opt:
        raise NotImplementedError(
            "distributed appearance training requires gather-before-MLP "
            "camera colors and has no upstream distributed route"
        )
    if config.sparse_grad:
        raise NotImplementedError(
            "current-main distributed rendering does not support sparse_grad"
        )
    if config.strategy.absgrad:
        raise NotImplementedError(
            "current-main distributed rendering does not support AbsGrad"
        )
    if (
        config.with_ut
        or config.with_eval3d
        or config.camera_model != "pinhole"
    ):
        raise NotImplementedError(
            "the first distributed training slice supports standard pinhole "
            "EWA rasterization only"
        )
    return _make_train_step(
        config,
        distributed_world_size=world_size,
        distributed_axis_name=axis_name,
        distributed_scene_scale=scene_scale,
    )
