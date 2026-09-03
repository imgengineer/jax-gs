# pyright: reportMissingImports=false

"""The host training loop, and the render step it evaluates with."""

from __future__ import annotations

import concurrent.futures
import gc
import operator
import sys
import time
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from ..capacity import _shrink_compacted_training_state
from ..config import TrainConfig
from ..model import GaussianModel
from ..strategy import (
    DefaultStrategy,
    MCMCStrategy,
    reset_opacities,
)
from ._data import (
    PrecomputedCameraPoses,
    StepKeyGenerator,
    _infinite_batches,
)
from ._memory import (
    _candidate_bound_for_occupancy,
    _check_evaluation_memory_budget,
    _intersection_bucket_capacity,
    _mcmc_required_capacity,
    _pending_overflow_suffix,
    _raise_training_overflow,
    _training_config_with_candidate_bound,
    _training_config_with_intersection_capacity,
    _training_intersection_limit,
    _training_overflow_status,
)
from ._scene import (
    SceneTransform,
    _legacy_scene_transform,
    _legacy_training_scene_scale,
    _scene_training_render_size,
    _training_scene_scale,
    compute_scene_transform,
)
from ._setup import (
    _create_training_optimizer,
    _invert_rigid_transforms,
    _validate_2dgs_mode,
    _validate_camera_module_resume_config,
)
from ._state import _initial_storage_capacity
from ._step import TrainingSafetyState, _PendingTrainStep
from .appearance import (
    APPEARANCE_FEATURE_DIM,
    AppearanceOptModule,
)
from .pose import CameraOptModule

# Tests drive the trainer by patching seams on the package, for example
# monkeypatch.setattr(jax_gs.training, "rasterization", fake). Calls resolve
# through the package namespace at run time so those seams keep working now
# that the implementation lives in submodules.
if __package__ is None:
    raise RuntimeError("training package context is unavailable")
_training = sys.modules[__package__]


@dataclass
class TrainingResult:
    model: GaussianModel
    final_step: int
    output_dir: Path
    checkpoint: Path | None
    metrics: dict[str, float]
    pose_adjust: CameraOptModule | None = None
    appearance: AppearanceOptModule | None = None


def _clear_obsolete_train_step_cache(train_step: Any) -> None:
    """Evict only an obsolete train-step plan, never unrelated JAX caches."""

    clear_cache = getattr(getattr(train_step, "jitted_fn", None), "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


def _prewarm_train_step(
    train_step: Any,
    *args: Any,
    reason: str,
    signature: str = "",
    **kwargs: Any,
) -> None:
    """Compile one concrete train-step shape before executing it."""

    lower = getattr(train_step, "lower", None)
    if not callable(lower):
        return
    started = time.perf_counter()
    lowered = lower(*args, **kwargs)
    lowered_at = time.perf_counter()
    compile_fn = getattr(lowered, "compile", None)
    if callable(compile_fn):
        compile_fn()
    finished = time.perf_counter()
    signature_text = f" {signature}" if signature else ""
    print(
        f"jit_prewarm reason={reason}{signature_text} "
        f"lower_ms={(lowered_at - started) * 1e3:.1f} "
        f"compile_ms={(finished - lowered_at) * 1e3:.1f} "
        f"total_ms={(finished - started) * 1e3:.1f}",
        flush=True,
    )


def make_render_step(config: TrainConfig, width: int, height: int):
    _validate_2dgs_mode(config)

    @nnx.jit
    def render_step(
        model: GaussianModel,
        viewmat: jax.Array,
        K: jax.Array,
        sh_degree: jax.Array,
        *,
        appearance_module: AppearanceOptModule | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if model.has_appearance != config.app_opt:
            raise ValueError("model color representation must match config.app_opt")
        parameters = model.activated(
            split_sh=config.model_type == "3dgs" and not config.app_opt
        )
        if config.app_opt:
            if appearance_module is None:
                raise ValueError(
                    "app_opt=True requires appearance_module for rendering"
                )
            camtoworld = _invert_rigid_transforms(viewmat[None, ...])
            directions = parameters["means"][None, :, :] - camtoworld[:, None, :3, 3]
            corrections = appearance_module(
                parameters["features"], None, directions, sh_degree
            )
            render_colors = jax.nn.sigmoid(
                parameters["colors"][None, :, :] + corrections
            )
            raster_sh_degree = None
        else:
            render_colors = parameters["sh_coeffs"]
            raster_sh_degree = sh_degree
        if config.model_type == "2dgs":
            renders, alphas, _, _, _, _, info = _training.rasterization_2dgs(
                parameters["means"],
                parameters["quats"],
                parameters["scales"],
                parameters["opacities"],
                render_colors,
                viewmat[None, ...],
                K[None, ...],
                width,
                height,
                packed=False,
                active_mask=parameters["active_mask"],
                sh_degree=raster_sh_degree,
                render_mode="RGB",
                config=config.rasterizer,
            )
        else:
            renders, alphas, info = _training.rasterization(
                parameters["means"],
                parameters["quats"],
                parameters["scales"],
                parameters["opacities"],
                render_colors,
                viewmat[None, ...],
                K[None, ...],
                width,
                height,
                active_mask=parameters["active_mask"],
                sh_degree=raster_sh_degree,
                camera_model=config.camera_model,
                with_ut=config.with_ut,
                with_eval3d=config.with_eval3d,
                config=config.rasterizer,
            )
        return (
            renders[0, ..., :3],
            alphas[0],
            info["tile_overflow"][0],
            info["intersection_overflow"][0],
        )

    return render_step


def make_distributed_render_step(
    config: TrainConfig,
    width: int,
    height: int,
    *,
    world_size: int,
    axis_name: Hashable = "rank",
):
    """Create the evaluation counterpart of :func:`make_distributed_train_step`.

    Upstream does not shard evaluation cameras: every rank iterates the whole
    validation set and renders it, because with Gaussians sharded the render
    is itself the collective, and only rank zero keeps the metrics and writes
    the images. This mirrors that. The returned step runs inside the same
    ``nnx.pmap`` or ``nnx.vmap`` as training with ``axis_name`` bound, and is
    given the same camera on every rank; the renderer gathers the shards, so
    every rank comes back with the whole image and they must agree.

    SH colors use the distributed rasterizer directly. Learned appearance
    gathers the Gaussian inputs first, evaluates the replicated MLP with the
    held-out camera and a zero embedding, then runs the ordinary local
    rasterizer on the resulting global scene. Other distributed restrictions
    are the training slice's.
    """

    try:
        world_size = operator.index(world_size)
    except TypeError as exc:
        raise TypeError("world_size must be an integer") from exc
    if world_size <= 1:
        raise ValueError("world_size must be greater than one")
    if axis_name is None or not isinstance(axis_name, Hashable):
        raise TypeError("axis_name must be hashable")
    _validate_2dgs_mode(config)
    if config.with_ut or config.with_eval3d or config.camera_model != "pinhole":
        raise NotImplementedError(
            "distributed rendering supports standard pinhole EWA rasterization only"
        )
    if config.model_type == "2dgs":

        @nnx.jit
        def render_step_2dgs(
            model: GaussianModel,
            viewmat: jax.Array,
            K: jax.Array,
            sh_degree: jax.Array,
            *,
            appearance_module: AppearanceOptModule | None = None,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            del appearance_module
            parameters = model.activated(split_sh=True)
            parameters = jax.tree.map(
                lambda value: jax.lax.all_gather(value, axis_name, axis=0, tiled=True),
                parameters,
            )
            (
                renders,
                alphas,
                _normals,
                _normals_depth,
                _distort,
                _,
                info,
            ) = _training.rasterization_2dgs(
                parameters["means"],
                parameters["quats"],
                parameters["scales"],
                parameters["opacities"],
                parameters["sh_coeffs"],
                viewmat[None, ...],
                K[None, ...],
                width,
                height,
                active_mask=parameters["active_mask"],
                sh_degree=sh_degree,
                render_mode="RGB",
                config=config.rasterizer,
            )
            return (
                renders[0, ..., :3],
                alphas[0],
                info["tile_overflow"][0],
                info["intersection_overflow"][0],
            )

        return render_step_2dgs

    @nnx.jit
    def render_step(
        model: GaussianModel,
        viewmat: jax.Array,
        K: jax.Array,
        sh_degree: jax.Array,
        *,
        appearance_module: AppearanceOptModule | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if model.has_appearance != config.app_opt:
            raise ValueError("model color representation must match config.app_opt")
        parameters = model.activated(split_sh=not config.app_opt)
        raster_distributed = True
        if config.app_opt:
            if appearance_module is None:
                raise ValueError(
                    "app_opt=True requires appearance_module for rendering"
                )
            parameters = jax.tree.map(
                lambda value: jax.lax.all_gather(value, axis_name, axis=0, tiled=True),
                parameters,
            )
            camtoworld = _invert_rigid_transforms(viewmat[None, ...])
            directions = parameters["means"][None, :, :] - camtoworld[:, None, :3, 3]
            corrections = appearance_module(
                parameters["features"], None, directions, sh_degree
            )
            render_colors = jax.nn.sigmoid(
                parameters["colors"][None, :, :] + corrections
            )
            raster_sh_degree = None
            raster_distributed = False
        else:
            render_colors = parameters["sh_coeffs"]
            raster_sh_degree = sh_degree
        renders, alphas, info = _training.rasterization(
            parameters["means"],
            parameters["quats"],
            parameters["scales"],
            parameters["opacities"],
            render_colors,
            viewmat[None, ...],
            K[None, ...],
            width,
            height,
            active_mask=parameters["active_mask"],
            sh_degree=raster_sh_degree,
            camera_model=config.camera_model,
            distributed=raster_distributed,
            distributed_world_size=world_size,
            distributed_axis_name=axis_name,
            config=config.rasterizer,
        )
        return (
            renders[0, ..., :3],
            alphas[0],
            info["tile_overflow"][0],
            info["intersection_overflow"][0],
        )

    return render_step


def reduce_distributed_render(
    rendered: jax.Array,
    *,
    rank: int = 0,
    atol: float = 0.0,
) -> jax.Array:
    """Take one rank's image out of a mapped render, checking the ranks agree.

    Every rank renders the same camera against the same gathered scene, so
    their images carry the same value and any disagreement beyond float
    reassociation means the collective did not deliver the whole scene
    somewhere. Checking that is cheap next to the render it follows, and a
    silently rank-dependent evaluation is worth failing loudly for.
    """

    stacked = np.asarray(jax.device_get(rendered))
    if stacked.ndim < 1 or stacked.shape[0] < 1:
        raise ValueError("expected a rank-major stack of renders")
    if not 0 <= rank < stacked.shape[0]:
        raise IndexError(f"rank {rank} is outside the world of {stacked.shape[0]}")
    reference = stacked[rank]
    for other in range(stacked.shape[0]):
        if other == rank:
            continue
        if not np.allclose(stacked[other], reference, rtol=0.0, atol=atol):
            worst = np.max(np.abs(stacked[other] - reference)).item()
            raise ValueError(
                f"distributed render disagrees between rank {rank} and rank "
                f"{other} by {worst:.3e}; every rank renders the same camera "
                "against the gathered scene, so they must match"
            )
    return jnp.asarray(reference)


def _save_render(path: Path, image: jax.Array) -> None:
    pixels = np.asarray(jax.device_get(jnp.clip(image, 0.0, 1.0) * 255.0)).astype(
        np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def train(
    config: TrainConfig,
    *,
    resume_from: str | Path | None = None,
    distributed: bool = False,
) -> TrainingResult:
    """Train with compact active prefixes and bucketed physical storage."""

    if distributed:
        return _training.train_distributed(config, resume_from=resume_from)

    world_size = jax.process_count()
    if world_size != 1:
        raise NotImplementedError(
            "train() currently supports single-process execution only; "
            "distributed rasterization is available, but synchronized "
            "Gaussian, pose, and appearance optimizer/checkpoint state is "
            f"not implemented for process_count={world_size}"
        )
    if resume_from is not None:
        _validate_camera_module_resume_config(config, resume_from)
    output_dir = Path(config.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")
    scene = _training.load_colmap_scene(
        config.data.root,
        image_dir=config.data.image_dir,
        load_points=resume_from is None,
    )
    training_height, training_width = _scene_training_render_size(scene, config)
    saved_scene_transform = (
        _training.load_checkpoint_scene_transform(resume_from)
        if resume_from is not None
        else None
    )
    if saved_scene_transform is not None:
        saved_matrix, scene_scale = saved_scene_transform
        transform = SceneTransform(saved_matrix)
    elif resume_from is not None:
        transform = _legacy_scene_transform(scene)
        scene_scale = _legacy_training_scene_scale(scene)
    else:
        transform = compute_scene_transform(
            scene,
            normalize_world_space=config.normalize_world_space,
        )
        scene_scale = _training_scene_scale(
            scene, transform, global_scale=config.global_scale
        )
    uses_camera_modules = config.pose_opt or config.pose_noise > 0.0 or config.app_opt
    camera_poses = PrecomputedCameraPoses(
        scene, transform, uses_camera_modules=uses_camera_modules
    )
    camera_image_names: tuple[str, ...] | None = None
    camera_count = 0
    if uses_camera_modules:
        camera_indices = scene.indices("train", config.data.test_every)
        camera_image_names = tuple(
            scene.images[np.asarray(index).item()].name for index in camera_indices
        )
        camera_count = len(camera_image_names)
        if camera_count == 0:
            raise ValueError(
                "camera-conditioned optimization requires a non-empty training split"
            )
    points: np.ndarray | None = None
    if resume_from is None:
        points = transform.points(scene.points)
        assert points is not None
        storage_capacity = _initial_storage_capacity(config, len(points))
    else:
        storage_capacity = _training.load_checkpoint_storage_capacity(resume_from)
        if storage_capacity > config.model.capacity:
            raise ValueError(
                f"checkpoint physical capacity {storage_capacity} exceeds the "
                f"configured logical maximum {config.model.capacity}"
            )

    intersection_limit = _training_intersection_limit(
        config,
        storage_capacity,
        image_height=training_height,
        image_width=training_width,
    )
    saved_intersection_capacity = (
        _training.load_checkpoint_intersection_capacity(resume_from)
        if resume_from is not None
        else None
    )
    intersection_capacity = _intersection_bucket_capacity(
        saved_intersection_capacity or 1,
        minimum=config.intersection_bucket_min_capacity,
        maximum=intersection_limit,
    )
    candidate_bound = (
        config.rasterizer.max_candidates_per_tile
        if resume_from is None
        else (
            _training.load_checkpoint_candidate_bound(resume_from)
            or config.rasterizer.max_candidates_per_tile
        )
    )

    def _runtime_training_config(
        intersection_capacity: int, candidate_bound: int | None
    ) -> TrainConfig:
        """Apply both grown quantities to the configuration as given.

        Each is derived from ``config`` rather than from the last runtime
        configuration, so growing one never silently discards the other.
        """

        runtime = _training_config_with_intersection_capacity(
            config, intersection_capacity
        )
        if candidate_bound is None:
            return runtime
        return _training_config_with_candidate_bound(runtime, candidate_bound)

    runtime_config = _runtime_training_config(intersection_capacity, candidate_bound)
    _training._check_memory_budget(
        runtime_config,
        physical_capacity=storage_capacity,
        label="initial_training" if resume_from is None else "resume_training",
        image_height=training_height,
        image_width=training_width,
    )
    if resume_from is None:
        assert points is not None
        model = GaussianModel.from_point_cloud(
            points,
            scene.points_rgb,
            config.model,
            physical_capacity=storage_capacity,
            num_workers=config.data.num_workers,
            appearance_feature_dim=(APPEARANCE_FEATURE_DIM if config.app_opt else None),
            feature_key=(
                jax.random.fold_in(jax.random.key(config.seed), 0x41505046)
                if config.app_opt
                else None
            ),
        )
    else:
        model = GaussianModel.empty(
            config.model,
            physical_capacity=storage_capacity,
            appearance_feature_dim=(APPEARANCE_FEATURE_DIM if config.app_opt else None),
        )
    optimizer = _create_training_optimizer(model, config, scene_scale=scene_scale)
    pose_adjust = None
    pose_optimizer = None
    if config.pose_opt:
        pose_adjust = CameraOptModule(
            camera_count,
            rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(config.seed), 0x504F5345)),
        )
        pose_adjust.zero_init()
        pose_optimizer = _training._create_pose_optimizer(pose_adjust, config)
    pose_perturb = None
    if config.pose_noise > 0.0:
        pose_perturb = CameraOptModule(
            camera_count,
            rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(config.seed), 0x4E4F4953)),
        )
        pose_perturb.random_init(config.pose_noise)
    appearance_module = None
    appearance_optimizer = None
    if config.app_opt:
        appearance_module = AppearanceOptModule(
            camera_count,
            APPEARANCE_FEATURE_DIM,
            config.app_embed_dim,
            config.model.sh_degree,
            rngs=nnx.Rngs(jax.random.fold_in(jax.random.key(config.seed), 0x4150504D)),
        )
        appearance_optimizer = _training.create_appearance_optimizer(
            appearance_module, config
        )
    strategy = (
        MCMCStrategy(config.strategy)
        if config.strategy.kind == "mcmc"
        else DefaultStrategy(config.strategy)
    )
    strategy_state = strategy.initialize_state(model.capacity)
    strategy_state.scene_scale[...] = scene_scale
    start_step = 0
    if resume_from is not None:
        start_step = _training.restore_checkpoint(
            resume_from,
            model,
            optimizer=optimizer,
            strategy_state=strategy_state,
            pose_module=pose_adjust,
            pose_optimizer=pose_optimizer,
            pose_image_names=(
                camera_image_names
                if config.pose_opt or config.pose_noise > 0.0
                else None
            ),
            appearance_module=appearance_module,
            appearance_optimizer=appearance_optimizer,
            appearance_image_names=(camera_image_names if config.app_opt else None),
        )
        if start_step > config.steps:
            raise ValueError(
                f"checkpoint step {start_step} exceeds configured training "
                f"steps {config.steps}"
            )
        if not _training.load_checkpoint_active_prefix(resume_from):
            compact_count = _training.compact_training_state(
                model, optimizer, strategy_state
            )
            compact_count.block_until_ready()
            print(
                "compacted_legacy_checkpoint "
                f"active={np.asarray(compact_count).item()}",
                flush=True,
            )

    print(
        f"storage_capacity={model.capacity} max_capacity={model.max_capacity} "
        f"active={np.asarray(jax.device_get(model.active_count)).item()} "
        f"intersection_capacity={intersection_capacity}/{intersection_limit}",
        flush=True,
    )

    dataset = _training.create_grain_dataset(
        scene,
        split="train",
        test_every=config.data.test_every,
        image_dir=config.data.image_dir,
        cache_images=config.data.cache_images,
        uint8=config.data.uint8,
        shuffle=True,
        seed=config.data.shuffle_seed,
        repeat=True,
        batch_size=config.data.batch_size,
        drop_remainder=True,
    )
    batches = _infinite_batches(dataset, num_workers=config.data.num_workers)
    if start_step < config.steps:
        for _ in range(start_step):
            next(batches)
    train_step = _training.make_train_step(runtime_config)
    train_step_prewarm_reason: str | None = "initial"
    safety_state = TrainingSafetyState()
    pending_steps: list[_PendingTrainStep] = []
    training_key = jax.random.key(config.seed)
    last_metrics: dict[str, float] = {}
    last_checkpoint: Path | None = None
    last_checkpoint_step: int | None = None
    checkpoint_executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=1)
        if config.async_checkpoint
        else None
    )
    pending_checkpoint_future: concurrent.futures.Future[Path] | None = None
    start_time = time.monotonic()

    evaluation_example = None
    evaluation_render_step = None
    eval_height = None
    eval_width = None
    if config.eval_every > 0:
        evaluation_source = _training.create_grain_dataset(
            scene,
            split="test",
            test_every=config.data.test_every,
            shuffle=False,
            batch_size=None,
        )
        evaluation_example = evaluation_source[0]
        eval_height, eval_width = evaluation_example["image"].shape[:2]
        evaluation_render_step = make_render_step(config, eval_width, eval_height)

    def run_device_train_step(
        images: jax.Array,
        intrinsics: jax.Array,
        viewmats: jax.Array,
        step_key: jax.Array,
        sh_degree: jax.Array,
        strategy_key: jax.Array,
        *,
        camtoworlds: jax.Array | None,
        image_ids: jax.Array | None,
    ) -> dict[str, jax.Array]:
        nonlocal train_step_prewarm_reason
        camera_kwargs: dict[str, Any] = {}
        if uses_camera_modules:
            camera_kwargs = {
                "pose_adjust": pose_adjust,
                "pose_optimizer": pose_optimizer,
                "pose_perturb": pose_perturb,
                "camtoworlds": camtoworlds,
                "image_ids": image_ids,
                "appearance_module": appearance_module,
                "appearance_optimizer": appearance_optimizer,
            }
        train_args = (
            model,
            optimizer,
            strategy_state,
            safety_state,
            images,
            intrinsics,
            viewmats,
            step_key,
            sh_degree,
            strategy_key,
        )
        if train_step_prewarm_reason is not None:
            image_shape = "x".join(str(size) for size in images.shape)
            _training._prewarm_train_step(
                train_step,
                *train_args,
                reason=train_step_prewarm_reason,
                signature=(
                    f"model_capacity={model.capacity} "
                    f"intersection_capacity={intersection_capacity} "
                    f"candidate_bound={candidate_bound} "
                    f"images={image_shape}"
                ),
                **camera_kwargs,
            )
            train_step_prewarm_reason = None
        return train_step(*train_args, **camera_kwargs)

    def synchronize_pending_steps() -> dict[str, jax.Array] | None:
        """Resolve sticky overflow and replay the uncommitted suffix."""

        nonlocal candidate_bound
        nonlocal intersection_capacity
        nonlocal intersection_limit
        nonlocal runtime_config
        nonlocal safety_state
        nonlocal train_step
        nonlocal train_step_prewarm_reason

        if not pending_steps:
            return None
        tuned_bound = None
        if candidate_bound is None:
            # Nothing tighter than the shape-derived bound is knowable before
            # a frame has been rendered, so a run starts loose and tightens
            # once it has seen one. Only this first observation tightens;
            # afterwards the bound only grows, because a scene that densifies
            # gets busier and a bound chasing occupancy downward would
            # recompile on every fluctuation.
            tuned_bound = _candidate_bound_for_occupancy(
                max(
                    np.asarray(value).item()
                    for value in jax.device_get(
                        tuple(
                            record.metrics["busiest_tile_candidates"]
                            for record in pending_steps
                        )
                    )
                ),
                config.rasterizer.max_gaussians_per_tile,
            )
            candidate_bound = tuned_bound
        max_overflow_tiles, intersection_overflow_seen = _training_overflow_status(
            safety_state
        )
        while max_overflow_tiles > 0 or intersection_overflow_seen:
            # Whatever the overflow turns out to be, its rebuild carries the
            # freshly tuned bound too, so tuning never costs its own compile.
            tuned_bound = None
            overflow = _pending_overflow_suffix(pending_steps)
            if overflow is None:
                _raise_training_overflow(
                    np.asarray(max_overflow_tiles),
                    np.asarray(intersection_overflow_seen),
                )
            assert overflow is not None
            replay_start = overflow.replay_start
            if overflow.kind == "candidates":
                # The per-tile bound is a promise the run made about its own
                # busiest tile, and a scene that densifies eventually outgrows
                # it. Growing and replaying is the same answer the
                # intersection buffer already gets, so an outgrown promise
                # costs a recompile rather than the run.
                next_candidate_bound = _candidate_bound_for_occupancy(
                    overflow.required, config.rasterizer.max_gaussians_per_tile
                )
                if candidate_bound is None or next_candidate_bound <= candidate_bound:
                    # Nothing larger to ask for: the loop was already sized
                    # for the worst tile the shapes admit, so the input claims
                    # a tile holds more candidates than it has slots to
                    # address, and no bound can render it whole.
                    _raise_training_overflow(
                        np.asarray(max_overflow_tiles),
                        np.asarray(intersection_overflow_seen),
                    )
                next_runtime_config = _runtime_training_config(
                    intersection_capacity, next_candidate_bound
                )
                growth_text = (
                    f"candidate_bound_growth={candidate_bound}"
                    f"->{next_candidate_bound} "
                    f"required={overflow.required} "
                )
            else:
                intersection_limit = _training_intersection_limit(
                    config,
                    model.capacity,
                    image_height=training_height,
                    image_width=training_width,
                )
                next_intersection_capacity = _intersection_bucket_capacity(
                    overflow.required,
                    minimum=config.intersection_bucket_min_capacity,
                    maximum=intersection_limit,
                )
                if next_intersection_capacity <= intersection_capacity:
                    raise RuntimeError(
                        "intersection overflow did not request a larger "
                        f"capacity ({overflow.required} required, "
                        f"{intersection_capacity} configured)"
                    )
                next_candidate_bound = candidate_bound
                next_runtime_config = _runtime_training_config(
                    next_intersection_capacity, candidate_bound
                )
                growth_text = (
                    "intersection_capacity_growth="
                    f"{intersection_capacity}->{next_intersection_capacity} "
                    f"required={overflow.required} "
                )
                intersection_capacity = next_intersection_capacity
            _training._check_memory_budget(
                next_runtime_config,
                physical_capacity=model.capacity,
                label=f"{overflow.kind}_growth",
                image_height=training_height,
                image_width=training_width,
            )
            _clear_obsolete_train_step_cache(train_step)
            del train_step
            gc.collect()
            runtime_config = next_runtime_config
            candidate_bound = next_candidate_bound
            train_step = _training.make_train_step(runtime_config)
            train_step_prewarm_reason = (
                "intersection_capacity_growth"
                if overflow.kind == "intersection"
                else "candidate_bound_growth"
            )
            safety_state = TrainingSafetyState()
            print(
                growth_text + f"replay_steps={len(pending_steps) - replay_start}",
                flush=True,
            )
            for record in pending_steps[replay_start:]:
                record.metrics = run_device_train_step(
                    jax.device_put(record.images),
                    jax.device_put(record.intrinsics),
                    jax.device_put(record.viewmats),
                    record.key,
                    record.sh_degree,
                    record.strategy_key,
                    camtoworlds=(
                        None
                        if record.camtoworlds is None
                        else jax.device_put(record.camtoworlds)
                    ),
                    image_ids=(
                        None
                        if record.image_ids is None
                        else jax.device_put(record.image_ids)
                    ),
                )
            max_overflow_tiles, intersection_overflow_seen = _training_overflow_status(
                safety_state
            )
        if tuned_bound is not None:
            # The pending steps ran with the looser bound, which renders them
            # correctly and only slowly, so tightening needs no replay -- only
            # a step compiled for the tighter loop from here on.
            runtime_config = _runtime_training_config(
                intersection_capacity, tuned_bound
            )
            _clear_obsolete_train_step_cache(train_step)
            del train_step
            gc.collect()
            train_step = _training.make_train_step(runtime_config)
            train_step_prewarm_reason = "candidate_bound_tuned"
            print(f"candidate_bound_tuned={tuned_bound}", flush=True)
        latest_metrics = pending_steps[-1].metrics
        pending_steps.clear()
        return latest_metrics

    def prepare_batch(raw_batch):
        images_raw = raw_batch["image"]
        if getattr(images_raw, "dtype", None) == np.uint8:
            images_np = np.asarray(images_raw, dtype=np.uint8)
        else:
            images_np = np.asarray(images_raw, dtype=np.float32)
        intrinsics_np = np.asarray(raw_batch["K"], dtype=np.float32)
        viewmats_np, camtoworlds_np = camera_poses.get_poses(
            raw_batch["w2c"], raw_batch.get("image_index")
        )
        image_ids_np = (
            np.asarray(raw_batch["dataset_index"], dtype=np.int32)
            if uses_camera_modules
            else None
        )
        images = jax.device_put(images_np)
        intrinsics = jax.device_put(intrinsics_np)
        viewmats = jax.device_put(viewmats_np)
        camtoworlds = None if camtoworlds_np is None else jax.device_put(camtoworlds_np)
        image_ids = None if image_ids_np is None else jax.device_put(image_ids_np)
        return (
            images,
            intrinsics,
            viewmats,
            camtoworlds,
            image_ids,
            images_np,
            intrinsics_np,
            viewmats_np,
            camtoworlds_np,
            image_ids_np,
        )

    key_generator = StepKeyGenerator(training_key)
    next_batch_data = (
        prepare_batch(next(batches)) if start_step < config.steps else None
    )

    for step in range(start_step + 1, config.steps + 1):
        if next_batch_data is None:
            raise RuntimeError("missing prepared batch data")
        (
            images,
            intrinsics,
            viewmats,
            camtoworlds,
            image_ids,
            images_np,
            intrinsics_np,
            viewmats_np,
            camtoworlds_np,
            image_ids_np,
        ) = next_batch_data
        if step < config.steps:
            next_batch_data = prepare_batch(next(batches))
        else:
            next_batch_data = None

        step_key, strategy_key = key_generator.get(step)
        do_refine = strategy.should_refine(step)
        do_reset = strategy.should_reset(step)
        sh_degree = jnp.minimum(
            config.model.sh_degree,
            step // max(config.sh_degree_interval, 1),
        )
        if config.strategy.kind == "mcmc" and do_refine:
            # Refinement is part of the atomic device commit below. Resolve any
            # older overflow before changing the physical storage bucket.
            synchronize_pending_steps()
            active_count = np.asarray(jax.device_get(model.active_count)).item()
            required_capacity = _mcmc_required_capacity(active_count, config)
            bounded_required = min(required_capacity, model.max_capacity)
            target_capacity = max(
                model.capacity,
                config.model.bucket_capacity(bounded_required),
            )
            if target_capacity > model.capacity:
                old_capacity = model.capacity
                model, optimizer, strategy_state = _training._grow_training_state(
                    runtime_config,
                    model,
                    optimizer,
                    strategy_state,
                    target_capacity,
                    image_height=training_height,
                    image_width=training_width,
                )
                train_step_prewarm_reason = "capacity_growth"
                print(
                    f"capacity_growth={old_capacity}->{target_capacity} "
                    f"required={required_capacity}",
                    flush=True,
                )
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            metrics = run_device_train_step(
                images,
                intrinsics,
                viewmats,
                step_key,
                sh_degree,
                strategy_key,
                camtoworlds=camtoworlds,
                image_ids=image_ids,
            )
        pending_steps.append(
            _PendingTrainStep(
                images=images_np,
                intrinsics=intrinsics_np,
                viewmats=viewmats_np,
                camtoworlds=camtoworlds_np,
                image_ids=image_ids_np,
                key=step_key,
                strategy_key=strategy_key,
                sh_degree=sh_degree,
                metrics=metrics,
            )
        )
        do_log = step == 1 or step % 10 == 0
        do_checkpoint = (
            config.checkpoint_every > 0 and step % config.checkpoint_every == 0
        )
        do_evaluate = (
            evaluation_example is not None
            and evaluation_render_step is not None
            and config.eval_every > 0
            and step % config.eval_every == 0
        )
        if (
            do_refine
            or do_reset
            or do_log
            or do_checkpoint
            or do_evaluate
            or step == config.steps
        ):
            synchronized_metrics = synchronize_pending_steps()
            if synchronized_metrics is not None:
                metrics = synchronized_metrics

        if do_refine and config.strategy.kind == "default":
            required_capacity = np.asarray(
                jax.device_get(
                    strategy.required_capacity(
                        model, strategy_state, scene_scale, step=step
                    )
                )
            ).item()
            bounded_required = min(required_capacity, model.max_capacity)
            target_capacity = max(
                model.capacity,
                config.model.bucket_capacity(bounded_required),
            )
            if target_capacity > model.capacity:
                old_capacity = model.capacity
                model, optimizer, strategy_state = _training._grow_training_state(
                    runtime_config,
                    model,
                    optimizer,
                    strategy_state,
                    target_capacity,
                    image_height=training_height,
                    image_width=training_width,
                )
                # Keep prior bucket executables cached: compaction can revisit
                # them, and a global clear would invalidate unrelated JITs.
                train_step_prewarm_reason = "capacity_growth"
                print(
                    f"capacity_growth={old_capacity}->{target_capacity} "
                    f"required={required_capacity}",
                    flush=True,
                )
            refine_metrics = strategy.refine(
                model,
                strategy_state,
                optimizer,
                strategy_key,
                scene_scale,
                step=step,
            )
            metrics = {
                **metrics,
                "active_count": model.active_count,
                **{f"refine/{k}": v for k, v in refine_metrics.items()},
            }
        if do_reset:
            reset_opacities(
                model,
                optimizer,
                maximum_opacity=config.strategy.reset_opacity,
            )

        if do_refine:
            active_count = np.asarray(jax.device_get(model.active_count)).item()
            compaction_required = min(
                model.max_capacity,
                max(
                    active_count * 2,
                    active_count + config.strategy.max_new_per_refine,
                ),
            )
            compact_capacity = config.model.bucket_capacity(compaction_required)
            # ponytail: require a 4x drop to amortize migration and one
            # recompilation; revisit after multi-scene end-to-end profiles.
            if compact_capacity * 4 <= model.capacity:
                old_capacity = model.capacity
                # The existing transition check requires an increasing pair;
                # reversing the capacities is conservative for a shrink and
                # preserves the old/new coexistence guard.
                _training._check_bucket_transition_memory_budget(
                    runtime_config,
                    compact_capacity,
                    old_capacity,
                    image_height=training_height,
                    image_width=training_width,
                )
                compact_count = _training.compact_training_state(
                    model, optimizer, strategy_state
                )
                compact_count.block_until_ready()
                compacted_count = np.asarray(compact_count).item()
                if compacted_count != active_count:
                    raise RuntimeError(
                        "active count changed while compacting the training bucket"
                    )
                model, optimizer, strategy_state = _shrink_compacted_training_state(
                    model,
                    optimizer,
                    strategy_state,
                    compact_capacity,
                    compacted_count,
                )
                _training._block_nnx_state(model, optimizer, strategy_state)
                intersection_limit = _training_intersection_limit(
                    config,
                    model.capacity,
                    image_height=training_height,
                    image_width=training_width,
                )
                if intersection_capacity > intersection_limit:
                    intersection_capacity = _intersection_bucket_capacity(
                        intersection_limit,
                        minimum=config.intersection_bucket_min_capacity,
                        maximum=intersection_limit,
                    )
                next_runtime_config = _runtime_training_config(
                    intersection_capacity, candidate_bound
                )
                if next_runtime_config != runtime_config:
                    runtime_config = next_runtime_config
                    _clear_obsolete_train_step_cache(train_step)
                    del train_step
                    gc.collect()
                    train_step = _training.make_train_step(runtime_config)
                train_step_prewarm_reason = "capacity_compaction"
                print(
                    f"capacity_compaction={old_capacity}->{model.capacity} "
                    f"active={compacted_count}",
                    flush=True,
                )

        if do_log:
            last_metrics = {
                name: np.asarray(jax.device_get(value)).item()
                for name, value in metrics.items()
            }
            elapsed = time.monotonic() - start_time
            bytes_in_use, bytes_limit = _training._device_memory_usage()
            memory_text = (
                f" gpu={bytes_in_use / 2**30:.2f}/{bytes_limit / 2**30:.2f}GiB"
                if bytes_limit
                else ""
            )
            regularization_text = (
                f"normal={last_metrics['normal_loss']:.6f} "
                f"distortion={last_metrics['distortion_loss']:.6f} "
                if config.model_type == "2dgs"
                else ""
            )
            pose_text = (
                f"pose_error={last_metrics['pose_error']:.6f} "
                if config.pose_opt and config.pose_noise > 0.0
                else ""
            )
            # How busy the worst tile actually got, against the bound the
            # compositor's chunk loop is currently sized for -- the tuned or
            # grown one, not the configured one, since that is the promise
            # the number has to be read against.
            busiest_tile_text = (
                f"busiest_tile={last_metrics['busiest_tile_candidates']:.0f}"
                f"/{candidate_bound} "
            )
            print(
                f"step={step:06d} loss={last_metrics['loss']:.6f} "
                f"psnr={last_metrics['psnr']:.2f} "
                f"{regularization_text}"
                f"{pose_text}"
                f"active={last_metrics['active_count']:.0f} "
                f"storage={model.capacity}/{model.max_capacity} "
                f"overflow_tiles={last_metrics['overflow_tiles']:.0f} "
                f"{busiest_tile_text}"
                f"candidate_limit_exceeded="
                f"{last_metrics['candidate_limit_exceeded_tiles']:.0f} "
                f"intersection_overflow={bool(last_metrics['intersection_overflow'])} "
                f"intersections={last_metrics['intersection_required_count']:.0f}/"
                f"{intersection_capacity} "
                f"elapsed={elapsed:.1f}s{memory_text}",
                flush=True,
            )
            if bytes_limit and bytes_in_use > bytes_limit * 0.85:
                raise MemoryError(
                    "JAX device memory usage exceeded the 85% safety threshold. "
                    "Training stopped before the next step; reduce capacity or "
                    "rasterizer candidate limits."
                )

        if do_checkpoint:
            if pending_checkpoint_future is not None:
                last_checkpoint = pending_checkpoint_future.result()
                pending_checkpoint_future = None
            save_args = (
                output_dir / "checkpoints",
                model,
                optimizer,
                strategy_state,
            )
            save_kwargs = {
                "step": step,
                "config": config,
                "intersection_capacity": intersection_capacity,
                "candidate_bound": candidate_bound,
                "scene_transform": transform,
                "scene_scale": scene_scale,
                "pose_adjust": pose_adjust,
                "pose_optimizer": pose_optimizer,
                "pose_image_names": (
                    camera_image_names
                    if config.pose_opt or config.pose_noise > 0.0
                    else None
                ),
                "appearance_module": appearance_module,
                "appearance_optimizer": appearance_optimizer,
                "appearance_image_names": (
                    camera_image_names if config.app_opt else None
                ),
            }
            if checkpoint_executor is not None:
                pending_checkpoint_future = checkpoint_executor.submit(
                    _training._save_compacted_training_checkpoint,
                    *save_args,
                    **save_kwargs,
                )
            else:
                last_checkpoint = _training._save_compacted_training_checkpoint(
                    *save_args,
                    **save_kwargs,
                )
            last_checkpoint_step = step

        if do_evaluate:
            assert eval_width is not None and eval_height is not None
            assert evaluation_render_step is not None and evaluation_example is not None
            _check_evaluation_memory_budget(
                config,
                physical_capacity=model.capacity,
                width=eval_width,
                height=eval_height,
            )
            normalized_viewmat = transform.world_to_camera(evaluation_example["w2c"])
            rendered, _, overflow, intersection_overflow = evaluation_render_step(
                model,
                jax.device_put(normalized_viewmat),
                jax.device_put(evaluation_example["K"]),
                jnp.asarray(config.model.sh_degree),
                appearance_module=appearance_module,
            )
            _save_render(output_dir / "renders" / f"step_{step:08d}.png", rendered)
            overflow_count = np.asarray(
                jax.device_get(jnp.count_nonzero(overflow))
            ).item()
            if overflow_count:
                print(f"evaluation tile overflow: {overflow_count}", flush=True)
            if bool(jax.device_get(intersection_overflow)):
                print("evaluation intersection overflow", flush=True)

    if last_checkpoint_step != config.steps:
        if pending_checkpoint_future is not None:
            last_checkpoint = pending_checkpoint_future.result()
            pending_checkpoint_future = None
        last_checkpoint = _training._save_compacted_training_checkpoint(
            output_dir / "checkpoints",
            model,
            optimizer,
            strategy_state,
            step=config.steps,
            config=config,
            intersection_capacity=intersection_capacity,
            candidate_bound=candidate_bound,
            scene_transform=transform,
            scene_scale=scene_scale,
            pose_adjust=pose_adjust,
            pose_optimizer=pose_optimizer,
            pose_image_names=(
                camera_image_names
                if config.pose_opt or config.pose_noise > 0.0
                else None
            ),
            appearance_module=appearance_module,
            appearance_optimizer=appearance_optimizer,
            appearance_image_names=(camera_image_names if config.app_opt else None),
        )
    if pending_checkpoint_future is not None:
        last_checkpoint = pending_checkpoint_future.result()
        pending_checkpoint_future = None
    if checkpoint_executor is not None:
        checkpoint_executor.shutdown(wait=True)
    return TrainingResult(
        model=model,
        final_step=config.steps,
        output_dir=output_dir,
        checkpoint=last_checkpoint,
        metrics=last_metrics,
        pose_adjust=pose_adjust,
        appearance=appearance_module,
    )
