"""Fixed-capacity Gaussian training with Optax and CuTe rendering."""

import argparse
import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from ..config import (
    DensifyConfig,
    ModelConfig,
    OptimizationConfig,
    PipelineConfig,
    RuntimeConfig,
    TrainingConfig,
    load_config,
)
from ..data import image_dataset
from ..io_manager.report import write_training_report
from ..scene.cluster import world_cluster_bounds
from ..scene.point import GaussianModel
from ..scene.spatial_refine import reorder_gaussians
from .densify import decay_opacity, densify_step
from .initialization import initialize_pool, load_training_frames
from .state import TrainingState
from .step import array_train_step as array_train_step  # public compatibility export
from .step import bind_train_step
from .step import train_step as train_step  # public compatibility export
from .warmup import precompile_training


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
    training_modes = [
        (
            min(epoch // 5, settings.model.sh_degree),
            densify_from <= epoch < densify_until and epoch % densify_interval == 0,
        )
        for epoch in range(num_epochs)
    ]
    capacity_config = settings.capacity
    pool, adam_state, initial_sparse_count, initial_padded_count = initialize_pool(
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
    warmup_seconds = precompile_training(
        training_state,
        train_step_fn,
        cluster_bounds,
        zip((frame.camera for frame in frames), targets, strict=True),
        key,
        initial_sparse_count,
        scene_radius,
        settings,
        training_modes,
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
    for epoch, (active_degree, collect_stats) in enumerate(training_modes):
        if (epoch - 1) % densify_interval == 0:
            pool, adam_state = reorder_gaussians(model.as_arrays(), adam_state)
            model.update_from_arrays(pool)
            training_state.adam.set_value(adam_state)
            cluster_bounds = world_cluster_bounds(pool, capacity_config.cluster_size)
        overflow, peak_pairs = jnp.array(False), jnp.array(0, jnp.int32)
        for frame_index in order_rng.permutation(len(frames)):
            loss, overflow, peak_pairs = train_step_fn(
                cluster_bounds,
                frames[frame_index].camera,
                targets[frame_index],
                jnp.array(step, jnp.int32),
                scene_radius,
                active_degree=active_degree,
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
    return write_training_report(
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
    parser.add_argument("--output", type=Path, default=Path("gaussians.ply"))
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
