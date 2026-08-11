"""Single-process, multi-device host orchestration for sharded training."""

from __future__ import annotations

from collections.abc import Sequence
import gc
from pathlib import Path
import sys
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from ..capacity import (
    _distributed_local_capacity,
    _stack_graphs,
)
from ..checkpoints import (
    load_distributed_checkpoint_manifest,
    restore_distributed_checkpoint,
    save_distributed_checkpoint,
)
from ..config import TrainConfig
from ..init_utils import knn_scale_init
from ..model import GaussianModel
from ..strategy import DefaultStrategy, MCMCStrategy
from ._data import _infinite_batches, shard_camera_batch
from ._loop import (
    TrainingResult,
    _save_render,
    make_distributed_render_step,
    reduce_distributed_render,
)
from ._memory import (
    _candidate_bound_for_occupancy,
    _check_evaluation_memory_budget,
    _intersection_bucket_capacity,
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
from ._setup import _create_pose_optimizer, _create_training_optimizer
from ._state import (
    make_distributed_resize_step,
    synchronize_distributed_capacity,
)
from ._step import TrainingSafetyState
from .pose import CameraOptModule


_training = sys.modules[__package__]
_RANK_AXIS = "rank"


def _resolve_local_distributed_devices(
    devices: Sequence[jax.Device] | None,
) -> tuple[jax.Device, ...]:
    """Resolve the fully addressable pmap world before any side effect."""

    if jax.process_count() != 1:
        raise NotImplementedError(
            "train_distributed() currently supports one JAX process with "
            "multiple local devices; multi-process host state and checkpoint "
            "ownership are not implemented"
        )
    resolved = tuple(jax.local_devices() if devices is None else devices)
    if len(resolved) < 2:
        raise ValueError(
            "distributed training requires at least two local devices"
        )
    if len({device.id for device in resolved}) != len(resolved):
        raise ValueError("distributed training devices must be unique")
    process_index = jax.process_index()
    if any(device.process_index != process_index for device in resolved):
        raise ValueError(
            "distributed training devices must belong to the local process"
        )
    return resolved


def _point_cloud_log_scales(
    points: np.ndarray, initial_scale: float
) -> np.ndarray:
    """Compute KNN scales before the point cloud is divided among owners."""

    count = len(points)
    result = np.full((count, 3), np.log(initial_scale), np.float32)
    if count > 1:
        neighbor_count = min(3, count - 1)
        values = np.asarray(
            knn_scale_init(jnp.asarray(points), k=neighbor_count)
        ) + np.float32(np.log(initial_scale))
        result = np.repeat(values[:, None], 3, axis=1)
    return result


def _strategy_state(config: TrainConfig, capacity: int, scene_scale: float):
    strategy = (
        MCMCStrategy(config.strategy)
        if config.strategy.kind == "mcmc"
        else DefaultStrategy(config.strategy)
    )
    state = strategy.initialize_state(capacity)
    state.scene_scale[...] = scene_scale
    return state


def _stack_bundles(bundles):
    return tuple(
        _stack_graphs([bundle[index] for bundle in bundles])
        for index in range(4)
    )


def _place_distributed_state(nodes, devices: Sequence[jax.Device]):
    """Put every stacked Variable on the pmap rank partition explicitly."""

    mesh = jax.sharding.Mesh(np.asarray(devices), (_RANK_AXIS,))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(_RANK_AXIS)
    )
    placed = []
    for node in nodes:
        graphdef, state = nnx.split(node)
        state = jax.tree.map(
            lambda value: (
                jax.device_put(value, sharding)
                if isinstance(value, jax.Array)
                else value
            ),
            state,
        )
        placed.append(nnx.merge(graphdef, state))
    return tuple(placed)


def _initialize_distributed_training_state(
    config: TrainConfig,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    world_size: int,
    scene_scale: float,
):
    """Build equal-capacity owner shards from one globally initialized cloud."""

    owner_indices = [
        np.arange(rank, len(points), world_size, dtype=np.int64)
        for rank in range(world_size)
    ]
    busiest = max((len(indices) for indices in owner_indices), default=0)
    if busiest > config.model.capacity:
        raise ValueError(
            f"the busiest initial shard has {busiest} points, but the "
            f"per-shard logical capacity is {config.model.capacity}"
        )
    local_capacity = config.model.bucket_capacity(busiest)
    initial_log_scales = _point_cloud_log_scales(
        points, config.model.initial_scale
    )
    bundles = []
    for indices in owner_indices:
        model = GaussianModel.from_point_cloud(
            points[indices],
            colors[indices],
            config.model,
            physical_capacity=local_capacity,
            num_workers=config.data.num_workers,
            initial_log_scales=initial_log_scales[indices],
        )
        optimizer = _create_training_optimizer(
            model,
            config,
            world_size=world_size,
            scene_scale=scene_scale,
        )
        bundles.append(
            (
                model,
                optimizer,
                _strategy_state(config, local_capacity, scene_scale),
                TrainingSafetyState(),
            )
        )
    return _stack_bundles(bundles)


def _empty_distributed_training_state(
    config: TrainConfig,
    *,
    world_size: int,
    local_capacity: int,
    scene_scale: float,
):
    bundles = []
    for _ in range(world_size):
        model = GaussianModel.empty(
            config.model, physical_capacity=local_capacity
        )
        optimizer = _create_training_optimizer(
            model,
            config,
            world_size=world_size,
            scene_scale=scene_scale,
        )
        bundles.append(
            (
                model,
                optimizer,
                _strategy_state(config, local_capacity, scene_scale),
                TrainingSafetyState(),
            )
        )
    return _stack_bundles(bundles)


def _initialize_distributed_pose_state(
    config: TrainConfig,
    *,
    world_size: int,
    camera_count: int,
):
    pose_adjust = None
    pose_optimizer = None
    if config.pose_opt:
        pose_pairs = []
        for _ in range(world_size):
            module = CameraOptModule(
                camera_count,
                rngs=nnx.Rngs(
                    jax.random.fold_in(
                        jax.random.key(config.seed), 0x504F5345
                    )
                ),
            )
            module.zero_init()
            pose_pairs.append((module, _create_pose_optimizer(module, config)))
        pose_adjust = _stack_graphs([pair[0] for pair in pose_pairs])
        pose_optimizer = _stack_graphs([pair[1] for pair in pose_pairs])

    pose_perturb = None
    if config.pose_noise > 0.0:
        perturbations = []
        for _ in range(world_size):
            module = CameraOptModule(
                camera_count,
                rngs=nnx.Rngs(
                    jax.random.fold_in(
                        jax.random.key(config.seed), 0x4E4F4953
                    )
                ),
            )
            module.random_init(config.pose_noise)
            perturbations.append(module)
        pose_perturb = _stack_graphs(perturbations)
    return pose_adjust, pose_optimizer, pose_perturb


def _runtime_training_config(
    config: TrainConfig,
    intersection_capacity: int,
    candidate_bound: int | None,
) -> TrainConfig:
    runtime = _training_config_with_intersection_capacity(
        config, intersection_capacity
    )
    if candidate_bound is not None:
        runtime = _training_config_with_candidate_bound(
            runtime, candidate_bound
        )
    return runtime


def _mapped_overflow(metrics: dict[str, jax.Array]):
    values = jax.device_get(
        (
            metrics["overflow_tiles"],
            metrics["intersection_overflow"],
            metrics["intersection_required_count"],
            metrics["busiest_tile_candidates"],
        )
    )
    overflow_tiles, intersection, required, busiest = map(np.asarray, values)
    if np.any(intersection):
        return "intersection", int(np.max(required))
    if np.max(overflow_tiles) > 0:
        return "candidates", int(np.max(busiest))
    return None


def _distributed_metric_summary(
    metrics: dict[str, jax.Array], model: GaussianModel
) -> dict[str, float]:
    """Reduce rank-local quality and owner-local counts for host reporting."""

    mean_metrics = {
        "loss",
        "l1",
        "ssim",
        "normal_loss",
        "distortion_loss",
        "opacity_reg_loss",
        "scale_reg_loss",
        "pose_error",
        "psnr",
    }
    count_metrics = {"active_count", "visible_count"}
    result: dict[str, float] = {}
    for name, value in metrics.items():
        array = np.asarray(jax.device_get(value))
        if name in mean_metrics:
            result[name] = float(np.mean(array))
        elif name in count_metrics:
            result[name] = float(np.sum(array))
        else:
            result[name] = float(np.ravel(array)[0])
    result["active_count"] = float(
        np.count_nonzero(np.asarray(jax.device_get(model.active_mask[...])))
    )
    return result


def train_distributed(
    config: TrainConfig,
    *,
    resume_from: str | Path | None = None,
    devices: Sequence[jax.Device] | None = None,
) -> TrainingResult:
    """Train Gaussian owner shards on one process's local devices.

    The per-rank batch size remains ``config.data.batch_size``. One host batch
    contains ``world_size`` such batches and is dealt across a named ``pmap``
    axis. Multi-process execution needs different host ownership and checkpoint
    semantics and is deliberately rejected before creating the output folder.
    """

    devices = _resolve_local_distributed_devices(devices)
    world_size = len(devices)
    uses_pose_modules = config.pose_opt or config.pose_noise > 0.0
    if config.data.batch_size * world_size > 10:
        raise ValueError(
            "current-main Adam requires distributed effective batch size <= 10"
        )
    # Validate every device-step restriction before filesystem or data work.
    _training.make_distributed_train_step(
        config, world_size=world_size, axis_name=_RANK_AXIS
    )
    if resume_from is not None:
        _training._validate_camera_module_resume_config(config, resume_from)
        resume_manifest = load_distributed_checkpoint_manifest(resume_from)
        if int(resume_manifest["world_size"]) != world_size:
            raise ValueError(
                "distributed train resume requires the checkpoint world size "
                f"{resume_manifest['world_size']}; got {world_size} local devices"
            )
    else:
        resume_manifest = None

    output_dir = Path(config.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")
    scene = _training.load_colmap_scene(
        config.data.root,
        image_dir=config.data.image_dir,
        load_points=resume_from is None,
    )
    camera_image_names: tuple[str, ...] | None = None
    camera_count = 0
    if uses_pose_modules:
        camera_indices = scene.indices("train", config.data.test_every)
        camera_image_names = tuple(
            scene.images[int(index)].name for index in camera_indices
        )
        camera_count = len(camera_image_names)
        if camera_count == 0:
            raise ValueError(
                "camera-pose training requires a non-empty training split"
            )
    training_height, training_width = _scene_training_render_size(scene, config)
    saved_scene = (
        _training.load_checkpoint_scene_transform(resume_from)
        if resume_from is not None
        else None
    )
    if saved_scene is not None:
        saved_matrix, scene_scale = saved_scene
        transform = SceneTransform(saved_matrix)
    elif resume_from is not None:
        transform = _legacy_scene_transform(scene)
        scene_scale = _legacy_training_scene_scale(scene)
    else:
        transform = compute_scene_transform(
            scene, normalize_world_space=config.normalize_world_space
        )
        scene_scale = _training_scene_scale(
            scene, transform, global_scale=config.global_scale
        )

    pose_adjust, pose_optimizer, pose_perturb = (
        _initialize_distributed_pose_state(
            config,
            world_size=world_size,
            camera_count=camera_count,
        )
    )
    if resume_manifest is None:
        model, optimizer, strategy_state, safety_state = (
            _initialize_distributed_training_state(
                config,
                transform.points(scene.points),
                scene.points_rgb,
                world_size=world_size,
                scene_scale=scene_scale,
            )
        )
        start_step = 0
    else:
        local_capacity = int(resume_manifest["local_capacity"])
        model, optimizer, strategy_state, safety_state = (
            _empty_distributed_training_state(
                config,
                world_size=world_size,
                local_capacity=local_capacity,
                scene_scale=scene_scale,
            )
        )
        start_step = restore_distributed_checkpoint(
            resume_from,
            model,
            optimizer,
            strategy_state,
            safety_state,
            config=config,
            pose_module=pose_adjust,
            pose_optimizer=pose_optimizer,
            pose_image_names=(
                camera_image_names if uses_pose_modules else None
            ),
        )
        if start_step > config.steps:
            raise ValueError(
                f"checkpoint step {start_step} exceeds configured training "
                f"steps {config.steps}"
            )

    model, optimizer, strategy_state, safety_state = (
        _place_distributed_state(
            (model, optimizer, strategy_state, safety_state), devices
        )
    )
    if pose_adjust is not None:
        pose_adjust, pose_optimizer = _place_distributed_state(
            (pose_adjust, pose_optimizer), devices
        )
    if pose_perturb is not None:
        (pose_perturb,) = _place_distributed_state(
            (pose_perturb,), devices
        )

    local_capacity = _distributed_local_capacity(model, world_size)
    global_capacity = world_size * local_capacity
    intersection_limit = _training_intersection_limit(
        config,
        global_capacity,
        image_height=training_height,
        image_width=training_width,
    )
    saved_intersection_capacity = (
        None
        if resume_manifest is None
        else resume_manifest.get("intersection_capacity")
    )
    intersection_capacity = _intersection_bucket_capacity(
        saved_intersection_capacity or 1,
        minimum=config.intersection_bucket_min_capacity,
        maximum=intersection_limit,
    )
    candidate_bound = (
        config.rasterizer.max_candidates_per_tile
        if resume_manifest is None
        else resume_manifest.get(
            "candidate_bound", config.rasterizer.max_candidates_per_tile
        )
    )
    runtime_config = _runtime_training_config(
        config, intersection_capacity, candidate_bound
    )
    _training._check_memory_budget(
        runtime_config,
        physical_capacity=local_capacity,
        render_capacity=global_capacity,
        devices=devices,
        label="initial_distributed_training",
        image_height=training_height,
        image_width=training_width,
    )
    print(
        f"distributed_world={world_size} local_capacity={local_capacity} "
        f"global_capacity={global_capacity} "
        f"active={int(np.count_nonzero(np.asarray(model.active_mask[...])))} "
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
        batch_size=world_size * config.data.batch_size,
        drop_remainder=True,
    )
    batches = _infinite_batches(dataset, num_workers=config.data.num_workers)
    if start_step < config.steps:
        for _ in range(start_step):
            next(batches)

    def make_mapped_train_step(current_config: TrainConfig):
        device_step = _training.make_distributed_train_step(
            current_config,
            world_size=world_size,
            axis_name=_RANK_AXIS,
            scene_scale=scene_scale,
        )
        camera_in_axes = ()
        if uses_pose_modules:
            camera_in_axes = (
                0 if config.pose_opt else None,
                0 if config.pose_opt else None,
                0 if config.pose_noise > 0.0 else None,
                0,
                0,
            )

        @nnx.pmap(
            in_axes=(0,) * 10 + camera_in_axes,
            out_axes=0,
            axis_name=_RANK_AXIS,
            devices=devices,
        )
        def mapped_step(*args):
            if not uses_pose_modules:
                return device_step(*args)
            return device_step(
                *args[:10],
                pose_adjust=args[10],
                pose_optimizer=args[11],
                pose_perturb=args[12],
                camtoworlds=args[13],
                image_ids=args[14],
            )

        return mapped_step

    mapped_train_step = make_mapped_train_step(runtime_config)
    mapped_resize_step = make_distributed_resize_step(
        config, axis_name=_RANK_AXIS, devices=devices
    )
    training_key = jax.random.key(config.seed)
    last_metrics: dict[str, float] = {}
    last_checkpoint: Path | None = None
    last_checkpoint_step: int | None = None
    start_time = time.monotonic()

    evaluation_example = None
    mapped_render_step = None
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
        render_step = make_distributed_render_step(
            config,
            eval_width,
            eval_height,
            world_size=world_size,
            axis_name=_RANK_AXIS,
        )

        @nnx.pmap(
            in_axes=(0, None, None, None),
            out_axes=0,
            axis_name=_RANK_AXIS,
            devices=devices,
        )
        def mapped_render_step(current_model, viewmat, K, sh_degree):
            return render_step(current_model, viewmat, K, sh_degree)

    def reset_safety_state() -> None:
        safety_state.max_overflow_tiles[...] = jnp.zeros_like(
            safety_state.max_overflow_tiles[...]
        )
        safety_state.intersection_overflow_seen[...] = jnp.zeros_like(
            safety_state.intersection_overflow_seen[...]
        )

    def resolve_raster_overflow(metrics, replay):
        nonlocal candidate_bound
        nonlocal intersection_capacity
        nonlocal mapped_train_step
        nonlocal runtime_config

        while True:
            if bool(
                np.any(
                    np.asarray(
                        jax.device_get(
                            metrics["distributed_state_mismatch"]
                        )
                    )
                )
            ):
                raise RuntimeError(
                    "distributed optimizer step or SH degree differs across "
                    "ranks"
                )
            overflow_tiles, intersection_seen = _training_overflow_status(
                safety_state
            )
            if overflow_tiles == 0 and not intersection_seen:
                return metrics
            overflow = _mapped_overflow(metrics)
            if overflow is None:
                _raise_training_overflow(
                    np.asarray(overflow_tiles), np.asarray(intersection_seen)
                )
            kind, required = overflow
            if kind == "intersection":
                next_intersection_capacity = _intersection_bucket_capacity(
                    required,
                    minimum=config.intersection_bucket_min_capacity,
                    maximum=intersection_limit,
                )
                if next_intersection_capacity <= intersection_capacity:
                    raise RuntimeError(
                        "intersection overflow did not request a larger "
                        f"capacity ({required} required, "
                        f"{intersection_capacity} configured)"
                    )
                print(
                    "distributed_intersection_capacity_growth="
                    f"{intersection_capacity}->{next_intersection_capacity} "
                    f"required={required}",
                    flush=True,
                )
                intersection_capacity = next_intersection_capacity
            else:
                next_candidate_bound = _candidate_bound_for_occupancy(
                    required, config.rasterizer.max_gaussians_per_tile
                )
                if (
                    candidate_bound is not None
                    and next_candidate_bound <= candidate_bound
                ):
                    _raise_training_overflow(
                        np.asarray(overflow_tiles), np.asarray(intersection_seen)
                    )
                print(
                    f"distributed_candidate_bound_growth={candidate_bound}"
                    f"->{next_candidate_bound} required={required}",
                    flush=True,
                )
                candidate_bound = next_candidate_bound
            runtime_config = _runtime_training_config(
                config, intersection_capacity, candidate_bound
            )
            _training._check_memory_budget(
                runtime_config,
                physical_capacity=local_capacity,
                render_capacity=world_size * local_capacity,
                devices=devices,
                label=f"distributed_{kind}_growth",
                image_height=training_height,
                image_width=training_width,
            )
            mapped_train_step = make_mapped_train_step(runtime_config)
            reset_safety_state()
            metrics = replay()

    for step in range(start_step + 1, config.steps + 1):
        sharded = shard_camera_batch(next(batches), world_size)
        images_np = np.asarray(sharded["image"], dtype=np.float32)
        intrinsics_np = np.asarray(sharded["K"], dtype=np.float32)
        viewmats_np = transform.world_to_camera(
            np.asarray(sharded["w2c"], dtype=np.float32)
        )
        camtoworlds_np = None
        image_ids_np = None
        if uses_pose_modules:
            camtoworlds_np = np.linalg.inv(viewmats_np).astype(np.float32)
            image_ids_np = np.asarray(
                sharded["dataset_index"], dtype=np.int32
            )
        rank_keys = [
            jax.random.fold_in(
                jax.random.fold_in(training_key, step), rank
            )
            for rank in range(world_size)
        ]
        split_keys = [jax.random.split(key, 2) for key in rank_keys]
        step_keys = jnp.stack([keys[0] for keys in split_keys])
        strategy_keys = jnp.stack([keys[1] for keys in split_keys])
        sh_degrees = jnp.full(
            (world_size,),
            min(
                config.model.sh_degree,
                step // max(config.sh_degree_interval, 1),
            ),
            jnp.int32,
        )
        device_inputs = (
            jnp.asarray(images_np),
            jnp.asarray(intrinsics_np),
            jnp.asarray(viewmats_np),
            step_keys,
            sh_degrees,
            strategy_keys,
        )
        camera_inputs = ()
        if uses_pose_modules:
            camera_inputs = (
                pose_adjust,
                pose_optimizer,
                pose_perturb,
                jnp.asarray(camtoworlds_np),
                jnp.asarray(image_ids_np),
            )

        def run_step():
            return mapped_train_step(
                model,
                optimizer,
                strategy_state,
                safety_state,
                *device_inputs,
                *camera_inputs,
            )

        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            metrics = run_step()
        metrics = resolve_raster_overflow(metrics, run_step)

        if candidate_bound is None:
            candidate_bound = _candidate_bound_for_occupancy(
                int(
                    np.max(
                        np.asarray(
                            jax.device_get(
                                metrics["busiest_tile_candidates"]
                            )
                        )
                    )
                ),
                config.rasterizer.max_gaussians_per_tile,
            )
            runtime_config = _runtime_training_config(
                config, intersection_capacity, candidate_bound
            )
            mapped_train_step = make_mapped_train_step(runtime_config)
            print(
                f"distributed_candidate_bound_tuned={candidate_bound}",
                flush=True,
            )

        model, optimizer, strategy_state, capacity_decision = (
            synchronize_distributed_capacity(
                runtime_config,
                model,
                optimizer,
                strategy_state,
                metrics,
                image_height=training_height,
                image_width=training_width,
                devices=devices,
                resize_step=mapped_resize_step,
            )
        )
        if capacity_decision.grew:
            local_capacity = capacity_decision.new_capacity
            global_capacity = world_size * local_capacity
            intersection_limit = _training_intersection_limit(
                config,
                global_capacity,
                image_height=training_height,
                image_width=training_width,
            )
            print(
                "distributed_capacity_growth="
                f"{capacity_decision.old_capacity}"
                f"->{capacity_decision.new_capacity} "
                f"replay={capacity_decision.replay_required}",
                flush=True,
            )
        if capacity_decision.replay_required:
            metrics = resolve_raster_overflow(run_step(), run_step)
            model, optimizer, strategy_state, replay_decision = (
                synchronize_distributed_capacity(
                    runtime_config,
                    model,
                    optimizer,
                    strategy_state,
                    metrics,
                    image_height=training_height,
                    image_width=training_width,
                    devices=devices,
                    resize_step=mapped_resize_step,
                )
            )
            if replay_decision.grew:
                local_capacity = replay_decision.new_capacity
                global_capacity = world_size * local_capacity
                intersection_limit = _training_intersection_limit(
                    config,
                    global_capacity,
                    image_height=training_height,
                    image_width=training_width,
                )
            if replay_decision.replay_required:
                raise RuntimeError(
                    "distributed refinement still overflows after bucket growth"
                )

        do_log = step == 1 or step % 10 == 0
        do_checkpoint = (
            config.checkpoint_every > 0
            and step % config.checkpoint_every == 0
        )
        do_evaluate = (
            evaluation_example is not None
            and mapped_render_step is not None
            and config.eval_every > 0
            and step % config.eval_every == 0
        )
        if do_log:
            last_metrics = _distributed_metric_summary(metrics, model)
            elapsed = time.monotonic() - start_time
            print(
                f"step={step:06d} loss={last_metrics['loss']:.6f} "
                f"psnr={last_metrics['psnr']:.2f} "
                f"active={int(last_metrics['active_count'])} "
                f"local_storage={local_capacity}/{model.max_capacity} "
                f"world={world_size} "
                f"overflow_tiles={int(last_metrics['overflow_tiles'])} "
                f"busiest_tile={int(last_metrics['busiest_tile_candidates'])}"
                f"/{candidate_bound} "
                f"intersections="
                f"{int(last_metrics['intersection_required_count'])}/"
                f"{intersection_capacity} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if do_checkpoint:
            last_checkpoint = save_distributed_checkpoint(
                output_dir / "checkpoints",
                model,
                optimizer,
                strategy_state,
                safety_state,
                step=step,
                config=config,
                intersection_capacity=intersection_capacity,
                candidate_bound=candidate_bound,
                scene_transform=transform.matrix,
                scene_scale=scene_scale,
                pose_module=pose_adjust,
                pose_optimizer=pose_optimizer,
                pose_image_names=(
                    camera_image_names if uses_pose_modules else None
                ),
            )
            last_checkpoint_step = step

        if do_evaluate:
            _check_evaluation_memory_budget(
                config,
                physical_capacity=world_size * local_capacity,
                width=eval_width,
                height=eval_height,
                devices=devices,
            )
            viewmat = transform.world_to_camera(evaluation_example["w2c"])
            rendered, _, overflow, intersection_overflow = mapped_render_step(
                model,
                jnp.asarray(viewmat),
                jnp.asarray(evaluation_example["K"]),
                jnp.asarray(config.model.sh_degree, jnp.int32),
            )
            image = reduce_distributed_render(rendered)
            _save_render(
                output_dir / "renders" / f"step_{step:08d}.png", image
            )
            overflow_count = int(
                np.count_nonzero(np.asarray(jax.device_get(overflow)))
            )
            if overflow_count:
                print(
                    f"distributed evaluation tile overflow: {overflow_count}",
                    flush=True,
                )
            if bool(
                np.any(np.asarray(jax.device_get(intersection_overflow)))
            ):
                print("distributed evaluation intersection overflow", flush=True)

    if last_checkpoint_step != config.steps:
        last_checkpoint = save_distributed_checkpoint(
            output_dir / "checkpoints",
            model,
            optimizer,
            strategy_state,
            safety_state,
            step=config.steps,
            config=config,
            intersection_capacity=intersection_capacity,
            candidate_bound=candidate_bound,
            scene_transform=transform.matrix,
            scene_scale=scene_scale,
            pose_module=pose_adjust,
            pose_optimizer=pose_optimizer,
            pose_image_names=(
                camera_image_names if uses_pose_modules else None
            ),
        )

    # Large mapped executables are not reusable once this world leaves scope.
    jax.effects_barrier()
    gc.collect()
    return TrainingResult(
        model=model,
        final_step=config.steps,
        output_dir=output_dir,
        checkpoint=last_checkpoint,
        metrics=last_metrics,
        pose_adjust=pose_adjust,
    )
