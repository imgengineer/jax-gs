"""The host training loop, and the render step it evaluates with."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
import gc
import operator
from pathlib import Path
import sys
import time
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from ..config import TrainConfig
from ..model import GaussianModel
from ..strategy import (
    DefaultStrategy,
    MCMCStrategy,
    reset_opacities,
)
from ._data import _infinite_batches
from ._memory import (
    _check_evaluation_memory_budget,
    _intersection_bucket_capacity,
    _mcmc_required_capacity,
    _pending_overflow_suffix,
    _raise_training_overflow,
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
            raise ValueError(
                "model color representation must match config.app_opt"
            )
        parameters = model.activated(
            split_sh=config.model_type == "3dgs" and not config.app_opt
        )
        if config.app_opt:
            if appearance_module is None:
                raise ValueError(
                    "app_opt=True requires appearance_module for rendering"
                )
            camtoworld = _invert_rigid_transforms(viewmat[None, ...])
            directions = (
                parameters["means"][None, :, :]
                - camtoworld[:, None, :3, 3]
            )
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

    The distributed restrictions are the training slice's, so this rejects the
    same combinations for the same reasons.
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
    if config.model_type != "3dgs":
        raise NotImplementedError("distributed rendering does not support 2DGS")
    if config.app_opt:
        raise NotImplementedError(
            "distributed appearance rendering requires gather-before-MLP "
            "camera colors and is not part of this slice"
        )
    if (
        config.with_ut
        or config.with_eval3d
        or config.camera_model != "pinhole"
    ):
        raise NotImplementedError(
            "distributed rendering supports standard pinhole EWA "
            "rasterization only"
        )

    @nnx.jit
    def render_step(
        model: GaussianModel,
        viewmat: jax.Array,
        K: jax.Array,
        sh_degree: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if model.has_appearance != config.app_opt:
            raise ValueError(
                "model color representation must match config.app_opt"
            )
        parameters = model.activated(split_sh=True)
        renders, alphas, info = _training.rasterization(
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
            camera_model=config.camera_model,
            distributed=True,
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
        raise IndexError(
            f"rank {rank} is outside the world of {stacked.shape[0]}"
        )
    reference = stacked[rank]
    for other in range(stacked.shape[0]):
        if other == rank:
            continue
        if not np.allclose(stacked[other], reference, rtol=0.0, atol=atol):
            worst = float(np.max(np.abs(stacked[other] - reference)))
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
    config: TrainConfig, *, resume_from: str | Path | None = None
) -> TrainingResult:
    """Train with compact active prefixes and bucketed physical storage."""

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
    uses_camera_modules = (
        config.pose_opt or config.pose_noise > 0.0 or config.app_opt
    )
    camera_image_names: tuple[str, ...] | None = None
    camera_count = 0
    if uses_camera_modules:
        camera_indices = scene.indices("train", config.data.test_every)
        camera_image_names = tuple(
            scene.images[int(index)].name for index in camera_indices
        )
        camera_count = len(camera_image_names)
        if camera_count == 0:
            raise ValueError(
                "camera-conditioned optimization requires a non-empty "
                "training split"
            )
    if resume_from is None:
        points = transform.points(scene.points)
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
    runtime_config = _training_config_with_intersection_capacity(
        config, intersection_capacity
    )
    _training._check_memory_budget(
        runtime_config,
        physical_capacity=storage_capacity,
        label="initial_training" if resume_from is None else "resume_training",
        image_height=training_height,
        image_width=training_width,
    )
    if resume_from is None:
        model = GaussianModel.from_point_cloud(
            points,
            scene.points_rgb,
            config.model,
            physical_capacity=storage_capacity,
            num_workers=config.data.num_workers,
            appearance_feature_dim=(
                APPEARANCE_FEATURE_DIM if config.app_opt else None
            ),
            feature_key=(
                jax.random.fold_in(
                    jax.random.key(config.seed), 0x41505046
                )
                if config.app_opt
                else None
            ),
        )
    else:
        model = GaussianModel.empty(
            config.model,
            physical_capacity=storage_capacity,
            appearance_feature_dim=(
                APPEARANCE_FEATURE_DIM if config.app_opt else None
            ),
        )
    optimizer = _create_training_optimizer(
        model, config, scene_scale=scene_scale
    )
    pose_adjust = None
    pose_optimizer = None
    if config.pose_opt:
        pose_adjust = CameraOptModule(
            camera_count,
            rngs=nnx.Rngs(
                jax.random.fold_in(jax.random.key(config.seed), 0x504F5345)
            ),
        )
        pose_adjust.zero_init()
        pose_optimizer = _training._create_pose_optimizer(pose_adjust, config)
    pose_perturb = None
    if config.pose_noise > 0.0:
        pose_perturb = CameraOptModule(
            camera_count,
            rngs=nnx.Rngs(
                jax.random.fold_in(jax.random.key(config.seed), 0x4E4F4953)
            ),
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
            rngs=nnx.Rngs(
                jax.random.fold_in(
                    jax.random.key(config.seed), 0x4150504D
                )
            ),
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
            appearance_image_names=(
                camera_image_names if config.app_opt else None
            ),
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
                f"compacted_legacy_checkpoint active={int(compact_count)}",
                flush=True,
            )

    print(
        f"storage_capacity={model.capacity} max_capacity={model.max_capacity} "
        f"active={int(jax.device_get(model.active_count))} "
        f"intersection_capacity={intersection_capacity}/{intersection_limit}",
        flush=True,
    )

    dataset = _training.create_grain_dataset(
        scene,
        split="train",
        test_every=config.data.test_every,
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
    safety_state = TrainingSafetyState()
    pending_steps: list[_PendingTrainStep] = []
    training_key = jax.random.key(config.seed)
    last_metrics: dict[str, float] = {}
    last_checkpoint: Path | None = None
    last_checkpoint_step: int | None = None
    start_time = time.monotonic()

    evaluation_example = None
    evaluation_render_step = None
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
        return train_step(
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
            **camera_kwargs,
        )

    def synchronize_pending_steps() -> dict[str, jax.Array] | None:
        """Resolve sticky overflow and replay the uncommitted suffix."""

        nonlocal intersection_capacity
        nonlocal intersection_limit
        nonlocal runtime_config
        nonlocal safety_state
        nonlocal train_step

        if not pending_steps:
            return None
        max_overflow_tiles, intersection_overflow_seen = (
            _training_overflow_status(safety_state)
        )
        if max_overflow_tiles > 0:
            _raise_training_overflow(
                np.asarray(max_overflow_tiles),
                np.asarray(intersection_overflow_seen),
            )
        while intersection_overflow_seen:
            replay_start, required_intersections = _pending_overflow_suffix(
                pending_steps
            )
            intersection_limit = _training_intersection_limit(
                config,
                model.capacity,
                image_height=training_height,
                image_width=training_width,
            )
            next_intersection_capacity = _intersection_bucket_capacity(
                required_intersections,
                minimum=config.intersection_bucket_min_capacity,
                maximum=intersection_limit,
            )
            if next_intersection_capacity <= intersection_capacity:
                raise RuntimeError(
                    "intersection overflow did not request a larger capacity "
                    f"({required_intersections} required, "
                    f"{intersection_capacity} configured)"
                )
            next_runtime_config = _training_config_with_intersection_capacity(
                config, next_intersection_capacity
            )
            _training._check_memory_budget(
                next_runtime_config,
                physical_capacity=model.capacity,
                label="intersection_bucket_growth",
                image_height=training_height,
                image_width=training_width,
            )
            old_intersection_capacity = intersection_capacity
            del train_step
            jax.clear_caches()
            gc.collect()
            runtime_config = next_runtime_config
            intersection_capacity = next_intersection_capacity
            train_step = _training.make_train_step(runtime_config)
            safety_state = TrainingSafetyState()
            print(
                "intersection_capacity_growth="
                f"{old_intersection_capacity}->{intersection_capacity} "
                f"required={required_intersections} "
                f"replay_steps={len(pending_steps) - replay_start}",
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
            max_overflow_tiles, intersection_overflow_seen = (
                _training_overflow_status(safety_state)
            )
            if max_overflow_tiles > 0:
                _raise_training_overflow(
                    np.asarray(max_overflow_tiles),
                    np.asarray(intersection_overflow_seen),
                )
        latest_metrics = pending_steps[-1].metrics
        pending_steps.clear()
        return latest_metrics

    for step in range(start_step + 1, config.steps + 1):
        batch = next(batches)
        images_np = np.asarray(batch["image"], dtype=np.float32)
        intrinsics_np = np.asarray(batch["K"], dtype=np.float32)
        viewmats_np = transform.world_to_camera(
            np.asarray(batch["w2c"], dtype=np.float32)
        )
        camtoworlds_np = None
        image_ids_np = None
        if uses_camera_modules:
            camtoworlds_np = np.linalg.inv(viewmats_np).astype(np.float32)
            image_ids_np = np.asarray(batch["dataset_index"], dtype=np.int32)
        images = jax.device_put(images_np)
        intrinsics = jax.device_put(intrinsics_np)
        viewmats = jax.device_put(viewmats_np)
        camtoworlds = (
            None if camtoworlds_np is None else jax.device_put(camtoworlds_np)
        )
        image_ids = (
            None if image_ids_np is None else jax.device_put(image_ids_np)
        )
        step_key, strategy_key = jax.random.split(
            jax.random.fold_in(training_key, step), 2
        )
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
            active_count = int(jax.device_get(model.active_count))
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
                jax.clear_caches()
                gc.collect()
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
            config.checkpoint_every > 0
            and step % config.checkpoint_every == 0
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
            required_capacity = int(
                jax.device_get(
                    strategy.required_capacity(
                        model, strategy_state, scene_scale, step=step
                    )
                )
            )
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
                # Bucket shapes only grow, so the old executable will never be
                # reused. Release its compiler cache before compiling the next
                # large shape to avoid cumulative host-memory pressure.
                jax.clear_caches()
                gc.collect()
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

        if do_log:
            last_metrics = {
                name: float(jax.device_get(value)) for name, value in metrics.items()
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
            print(
                f"step={step:06d} loss={last_metrics['loss']:.6f} "
                f"psnr={last_metrics['psnr']:.2f} "
                f"{regularization_text}"
                f"{pose_text}"
                f"active={int(last_metrics['active_count'])} "
                f"storage={model.capacity}/{model.max_capacity} "
                f"overflow_tiles={int(last_metrics['overflow_tiles'])} "
                f"candidate_limit_exceeded={int(last_metrics['candidate_limit_exceeded_tiles'])} "
                f"intersection_overflow={bool(last_metrics['intersection_overflow'])} "
                f"intersections={int(last_metrics['intersection_required_count'])}/"
                f"{intersection_capacity} "
                f"elapsed={elapsed:.1f}s{memory_text}",
                flush=True,
            )
            if bytes_limit and bytes_in_use > int(bytes_limit * 0.85):
                raise MemoryError(
                    "JAX device memory usage exceeded the 85% safety threshold. "
                    "Training stopped before the next step; reduce capacity or "
                    "rasterizer candidate limits."
                )

        if do_checkpoint:
            last_checkpoint = _training._save_compacted_training_checkpoint(
                output_dir / "checkpoints",
                model,
                optimizer,
                strategy_state,
                step=step,
                config=config,
                intersection_capacity=intersection_capacity,
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
                appearance_image_names=(
                    camera_image_names if config.app_opt else None
                ),
            )
            last_checkpoint_step = step

        if do_evaluate:
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
            overflow_count = int(jax.device_get(jnp.count_nonzero(overflow)))
            if overflow_count:
                print(f"evaluation tile overflow: {overflow_count}", flush=True)
            if bool(jax.device_get(intersection_overflow)):
                print("evaluation intersection overflow", flush=True)

    if last_checkpoint_step != config.steps:
        last_checkpoint = _training._save_compacted_training_checkpoint(
            output_dir / "checkpoints",
            model,
            optimizer,
            strategy_state,
            step=config.steps,
            config=config,
            intersection_capacity=intersection_capacity,
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
            appearance_image_names=(
                camera_image_names if config.app_opt else None
            ),
        )
    return TrainingResult(
        model=model,
        final_step=config.steps,
        output_dir=output_dir,
        checkpoint=last_checkpoint,
        metrics=last_metrics,
        pose_adjust=pose_adjust,
        appearance=appearance_module,
    )
