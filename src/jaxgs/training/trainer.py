"""Fixed-capacity Gaussian training with Optax and CuTe rendering."""

import argparse
import json
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import chex
import jax
import jax.numpy as jnp
import numpy as np

from ..config import (
    CapacityConfig,
    DensifyConfig,
    ModelConfig,
    OptimizationConfig,
    PipelineConfig,
    RuntimeConfig,
    TrainingConfig,
    load_config,
)
from ..data import Frame, image_dataset
from ..io_manager.checkpoint import save_gaussians
from ..io_manager.colmap import load_colmap_images, load_colmap_points
from ..scene.camera import Camera
from ..scene.cluster import world_cluster_bounds
from ..scene.point import (
    GaussianArrays,
    GaussianModel,
    create_gaussians,
    estimate_initial_scales,
    seed_gaussians,
)
from ..scene.spatial_refine import reorder_gaussians
from ..scene.types import WorldClusterBounds
from .densify import decay_opacity, densify_step
from .optimizer import AdamState, create_adam_state
from .state import TrainingState
from .step import array_train_step as array_train_step  # public compatibility export
from .step import bind_train_step
from .step import train_step as train_step  # public compatibility export


def load_training_frames(scene: str | Path, model_config: ModelConfig) -> list[Frame]:
    """Load configured views and apply the training split when evaluation is enabled."""
    frames = load_colmap_images(scene, model_config.images, resolution=model_config.resolution)
    if model_config.eval:
        split_path = Path(scene) / "train_test_split.json"
        if split_path.exists():
            names = set(json.loads(split_path.read_text())["train"])
            frames = [
                frame
                for frame in frames
                if frame.image_path.stem in names or frame.image_path.name in names
            ]
        else:
            frames = [frame for index, frame in enumerate(frames) if index % 8 != 0]
    if not frames:
        raise ValueError("no training images selected")
    return frames


def _initialize_pool(
    scene: str | Path, config: CapacityConfig
) -> tuple[GaussianArrays, AdamState, int, int]:
    """Create the fixed pool; return sparse and cluster-padded initial counts."""
    xyz, rgb = load_colmap_points(scene)
    initial_point_count = len(xyz)
    if initial_point_count == 0:
        raise ValueError("initial cloud is empty")
    initial_scales = estimate_initial_scales(xyz)
    # LiteGS cluster_points pads its initial cloud with copies of the final
    # points. Gather those same copies into slots without resizing the pool.
    cluster_padding = (-initial_point_count) % config.cluster_size
    point_indices = np.arange(initial_point_count + cluster_padding)
    point_indices[initial_point_count:] = (
        np.arange(initial_point_count - cluster_padding, initial_point_count) % initial_point_count
    )
    if len(point_indices) > config.max_gaussians:
        raise ValueError("initial cloud including cluster padding exceeds capacity")
    pool = seed_gaussians(
        create_gaussians(config),
        xyz[point_indices],
        rgb[point_indices],
        scale=initial_scales[point_indices],
        opacity=0.1,
    )
    adam_state = create_adam_state(pool)
    # Share the initial zeros only during precompilation. The epoch loop
    # separates v after warmup, preserving the existing peak memory budget.
    adam_state = adam_state.replace(v=adam_state.m)
    return pool, adam_state, initial_point_count, len(point_indices)


def _precompile_training(
    training_state: TrainingState,
    train_step_fn: Callable[..., tuple[chex.Array, chex.Array, chex.Array]],
    cluster_bounds: WorldClusterBounds,
    views: Iterable[tuple[Camera, chex.Array]],
    key: chex.Array,
    initial_sparse_count: int,
    scene_radius: chex.Array,
    settings: TrainingConfig,
) -> float:
    """Compile SH/statistics variants for each view shape before timing."""
    capacity_config = settings.capacity
    densification = settings.densify
    warmup_start = perf_counter()
    pool = training_state.model.as_arrays()
    adam_state = training_state.adam.get_value()
    fragment_stats = training_state.fragments.get_value()
    # Donation consumes only this working copy; reuse its returned buffers
    # across variants instead of copying the full-capacity state every time.
    warm_pool, warm_adam_state, warm_fragment_stats = jax.tree.map(
        jnp.copy, (pool, adam_state, fragment_stats)
    )
    training_state.model.update_from_arrays(warm_pool)
    training_state.adam.set_value(warm_adam_state)
    training_state.fragments.set_value(warm_fragment_stats)
    try:
        compiled_view_signatures = set()
        for camera, target in views:
            view_signature = (camera.width, camera.height, camera.near, camera.far)
            if view_signature in compiled_view_signatures:
                continue
            compiled_view_signatures.add(view_signature)
            for sh_degree in range(capacity_config.sh_degree + 1):
                for collect_stats in (False, True):
                    step_result = train_step_fn(
                        cluster_bounds,
                        camera,
                        target,
                        jnp.array(0, jnp.int32),
                        scene_radius,
                        active_degree=sh_degree,
                        collect_stats=collect_stats,
                        overflow=jnp.array(False),
                        peak_pairs=jnp.array(0, jnp.int32),
                    )
                    jax.block_until_ready(
                        (
                            step_result,
                            training_state.adam.get_value(),
                            training_state.fragments.get_value(),
                            training_state.model.as_arrays(),
                        )
                    )
    finally:
        training_state.model.update_from_arrays(pool)
        training_state.adam.set_value(adam_state)
        training_state.fragments.set_value(fragment_stats)
    densify_step.lower(
        pool,
        adam_state,
        fragment_stats,
        key,
        jnp.array(initial_sparse_count, jnp.int32),
        scene_radius,
        cluster_size=capacity_config.cluster_size,
        percent_dense=densification.percent_dense,
    ).compile()
    jax.block_until_ready(decay_opacity(pool, adam_state))
    jax.block_until_ready(reorder_gaussians(pool, adam_state))
    return perf_counter() - warmup_start


def _write_training_report(
    output: str | Path,
    pool: GaussianArrays,
    settings: TrainingConfig,
    metrics: dict[str, object],
) -> dict[str, object]:
    """Save the final pool and preserve the training report schema."""
    output = Path(output)
    save_gaussians(output, pool)
    capacity_config = settings.capacity
    report = {
        "scene": metrics["scene"],
        "gpu": jax.devices()[0].device_kind,
        "images": settings.model.images,
        "image_shape": metrics["image_shape"],
        "training_images": metrics["training_images"],
        "actual_updates": metrics["actual_updates"],
        "initial_sparse_gaussians": metrics["initial_sparse_gaussians"],
        "initial_padded_gaussians": metrics["initial_padded_gaussians"],
        "target_gaussians": settings.densify.target_primitives,
        "final_gaussians": int(pool.n_active),
        "pair_capacity": settings.runtime.max_visibility_pairs,
        "cluster_size": capacity_config.cluster_size,
        "tile_size": capacity_config.tile_size,
        "tile_height": capacity_config.raster_tile_height,
        "warmup_seconds": metrics["warmup_seconds"],
        "image_load_seconds": metrics["image_load_seconds"],
        "data_loader": "grain.MapDataset, 4 decode threads, 8 prefetched images; GPU uint8 preload",
        "training_seconds": metrics["training_seconds"],
        "densify_until": metrics["densify_until"],
        "seed": settings.runtime.seed,
        "history": metrics["history"],
        "config": asdict(settings),
        "model": "flax.nnx.Module",
        "jit": "nnx.jit_partial(graph=False)",
        "optimizer": settings.runtime.optimizer,
        "jit_cache_size": metrics["jit_cache_size"],
        "donation": True,
        "timing_scope": "epoch loop including densification, pruning, opacity decay and spatial refinement; excludes preload, warmup and final checkpoint write",
        "remaining_differences": [
            "JAX world cluster bounds and frustum mask",
            "fixed-capacity intermediates with valid compact prefixes",
            "XLA prefix sums and stable sorting",
            "reduction rounding and partial-tile masks",
            "analytic SH view-direction gradient for xyz",
            "independent RNG implementations",
        ],
    }
    report_path = output.with_suffix(".json")
    if report_path == output:
        report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"training: {metrics['training_seconds']:.3f}s; report: {report_path}", flush=True)
    return report


def train(
    scene: str | Path,
    output: str | Path,
    *,
    settings: TrainingConfig | None = None,
    images: str | None = None,
    iterations: int | None = None,
    target_points: int | None = None,
    pair_capacity: int | None = None,
    seed: int | None = None,
    optimizer: str | None = None,
) -> dict[str, object]:
    settings = settings or load_config()
    if images is not None:
        settings = replace(settings, model=replace(settings.model, images=images))
    if iterations is not None:
        settings = replace(
            settings, optimization=replace(settings.optimization, iterations=iterations)
        )
    if target_points is not None:
        settings = replace(
            settings,
            densify=replace(settings.densify, target_primitives=target_points),
            runtime=replace(settings.runtime, max_gaussians=target_points),
        )
    overrides = {
        name: value
        for name, value in (
            ("max_visibility_pairs", pair_capacity),
            ("seed", seed),
            ("optimizer", optimizer),
        )
        if value is not None
    }
    settings = replace(settings, runtime=replace(settings.runtime, **overrides))
    return start(
        settings.model,
        settings.optimization,
        settings.pipeline,
        settings.densify,
        source_path=scene,
        model_path=output,
        runtime=settings.runtime,
    )


def start(
    lp: ModelConfig,
    op: OptimizationConfig,
    pp: PipelineConfig,
    dp: DensifyConfig,
    *,
    source_path: str | Path,
    model_path: str | Path,
    runtime: RuntimeConfig | None = None,
) -> dict[str, object]:
    """Train with LiteGS's model, optimization, pipeline and densification groups."""
    scene, output = source_path, model_path
    runtime = runtime or replace(load_config().runtime, max_gaussians=dp.target_primitives)
    settings = TrainingConfig(model=lp, optimization=op, pipeline=pp, densify=dp, runtime=runtime)
    settings.validate()
    optimization, densification, runtime = settings.optimization, settings.densify, settings.runtime
    iterations = optimization.iterations
    target_points, pair_capacity = densification.target_primitives, runtime.max_visibility_pairs
    seed, optimizer = runtime.seed, runtime.optimizer
    frames = load_training_frames(scene, settings.model)
    num_epochs = iterations // len(frames)
    if num_epochs < 1:
        raise ValueError("iterations must cover at least one complete epoch")
    densify_from = densification.densify_from
    densify_interval, opacity_reset_interval = (
        densification.densification_interval,
        densification.opacity_reset_interval,
    )
    densify_until = densification.end_epoch(num_epochs)
    capacity_config = settings.capacity
    pool, adam_state, initial_sparse_count, initial_padded_count = _initialize_pool(
        scene, capacity_config
    )
    cluster_bounds = world_cluster_bounds(pool, capacity_config.cluster_size)
    fragment_stats = jnp.zeros((capacity_config.max_gaussians, 4), jnp.float32)
    camera_centers = np.stack([np.asarray(frame.camera.center) for frame in frames])
    scene_radius = jnp.array(
        1.1 * np.max(np.linalg.norm(camera_centers - camera_centers.mean(0), axis=1)), jnp.float32
    )
    image_load_start = perf_counter()
    with closing(iter(image_dataset(frames))) as images_iter:
        targets = [jax.device_put(image) for image in images_iter]
    jax.block_until_ready(targets)
    image_load_seconds = perf_counter() - image_load_start
    key = jax.random.key(seed)
    order_rng = np.random.default_rng(seed)
    jax.block_until_ready((pool, adam_state, cluster_bounds, targets))

    model = GaussianModel(pool)
    training_state = TrainingState(model, adam_state, fragment_stats)
    train_step_fn = bind_train_step(
        training_state,
        capacity_config,
        max_steps=optimization.position_lr_max_steps,
        optimizer=optimizer,
        optimization=optimization,
    )
    warmup_seconds = _precompile_training(
        training_state,
        train_step_fn,
        cluster_bounds,
        zip((frame.camera for frame in frames), targets, strict=True),
        key,
        initial_sparse_count,
        scene_radius,
        settings,
    )
    print(f"warmup: {warmup_seconds:.3f}s; {len(frames)} views x {num_epochs} epochs", flush=True)
    # Keep the initial shared zeros during warmup to avoid another full moment
    # copy at peak memory. Donation needs distinct m/v for actual training.
    adam_state = adam_state.replace(v=jax.tree.map(jnp.copy, adam_state.v))
    training_state.adam.set_value(adam_state)
    jax.block_until_ready(adam_state)

    training_start = perf_counter()
    history = []
    step = 0
    for epoch in range(num_epochs):
        if (epoch - 1) % densify_interval == 0:
            pool, adam_state = reorder_gaussians(model.as_arrays(), adam_state)
            model.update_from_arrays(pool)
            training_state.adam.set_value(adam_state)
            cluster_bounds = world_cluster_bounds(pool, capacity_config.cluster_size)
        collect_stats = densify_from <= epoch < densify_until and epoch % densify_interval == 0
        overflow, peak_pairs = jnp.array(False), jnp.array(0, jnp.int32)
        for frame_index in order_rng.permutation(len(frames)):
            loss, overflow, peak_pairs = train_step_fn(
                cluster_bounds,
                frames[frame_index].camera,
                targets[frame_index],
                jnp.array(step, jnp.int32),
                scene_radius,
                active_degree=min(epoch // 5, capacity_config.sh_degree),
                collect_stats=collect_stats,
                overflow=overflow,
                peak_pairs=peak_pairs,
            )
            step += 1
        adam_state = training_state.adam.get_value()
        fragment_stats = training_state.fragments.get_value()
        if bool(overflow):
            raise RuntimeError(f"epoch {epoch}: {int(peak_pairs)} pairs exceed {pair_capacity}")
        born_count, pruned_count = 0, 0
        pool = model.as_arrays()
        if collect_stats:
            target_count = int(
                (target_points - initial_sparse_count)
                / (densify_until - densify_from)
                * (epoch - densify_from)
                + initial_sparse_count
            )
            key, densify_key = jax.random.split(key)
            pool, adam_state, born_count, pruned_count = densify_step(
                pool,
                adam_state,
                fragment_stats,
                densify_key,
                jnp.array(target_count, jnp.int32),
                scene_radius,
                cluster_size=capacity_config.cluster_size,
                percent_dense=densification.percent_dense,
            )
            fragment_stats = jnp.zeros_like(fragment_stats)
        if densify_from <= epoch < densify_until and epoch % opacity_reset_interval == 0:
            pool, adam_state = decay_opacity(pool, adam_state)
            adam_state = adam_state.replace(v=jax.tree.map(jnp.copy, adam_state.v))
        model.update_from_arrays(pool)
        training_state.adam.set_value(adam_state)
        training_state.fragments.set_value(fragment_stats)
        row = {
            "epoch": epoch,
            "step": step,
            "active": int(pool.n_active),
            "loss": float(loss),
            "born": int(born_count),
            "pruned": int(pruned_count),
            "peak_pairs": int(peak_pairs),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if not np.isfinite(row["loss"]):
            raise RuntimeError("non-finite training loss")
    jax.block_until_ready((pool, adam_state))
    training_seconds = perf_counter() - training_start
    return _write_training_report(
        output,
        pool,
        settings,
        {
            "scene": str(scene),
            "image_shape": list(targets[0].shape[:2]),
            "training_images": len(frames),
            "actual_updates": step,
            "initial_sparse_gaussians": initial_sparse_count,
            "initial_padded_gaussians": initial_padded_count,
            "warmup_seconds": warmup_seconds,
            "image_load_seconds": image_load_seconds,
            "training_seconds": training_seconds,
            "densify_until": densify_until,
            "history": history,
            "jit_cache_size": train_step_fn.jitted_fn._cache_size(),
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--config", type=Path, help="TOML overrides for packaged default.toml")
    parser.add_argument("--images", help="override model.images")
    parser.add_argument("--output", type=Path, default=Path("gaussians.npz"))
    parser.add_argument("--iterations", type=int, help="override optimization.iterations")
    parser.add_argument(
        "--target-points", type=int, help="override growth target and pool capacity"
    )
    parser.add_argument("--max-visibility-pairs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--optimizer",
        choices=("optax", "muon", "cute"),
        help="optimizer backend (default: optax; muon uses Muon for SH colors)",
    )
    args = parser.parse_args(argv)
    train(
        args.scene,
        args.output,
        settings=load_config(args.config),
        images=args.images,
        iterations=args.iterations,
        target_points=args.target_points,
        pair_capacity=args.max_visibility_pairs,
        seed=args.seed,
        optimizer=args.optimizer,
    )


if __name__ == "__main__":
    main()
