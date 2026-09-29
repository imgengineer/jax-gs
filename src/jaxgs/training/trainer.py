"""Fixed-capacity Gaussian training with Optax and CuTe rendering."""

import argparse
import json
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from ..config import load_config
from ..data import image_dataset
from ..io_manager.checkpoint import save_pool
from ..io_manager.colmap import load_colmap_images, load_colmap_points
from ..scene.cluster import world_cluster_bounds
from ..scene.point import GaussianModel, create_pool, estimate_initial_scales, seed_pool
from ..scene.spatial_refine import spatial_refine
from .densify import decay_opacity, densify_step
from .optimizer import create_adam_state
from .step import array_train_step as array_train_step  # public compatibility export
from .step import train_step


def training_frames(scene, model_config):
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


def train(
    scene,
    output,
    *,
    settings=None,
    images=None,
    iterations=None,
    target_points=None,
    pair_capacity=None,
    seed=None,
    optimizer=None,
):
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
    settings.validate()
    op, dp, runtime = settings.optimization, settings.densify, settings.runtime
    images, iterations = settings.model.images, op.iterations
    target_points, pair_capacity = dp.target_primitives, runtime.max_visibility_pairs
    seed, optimizer = runtime.seed, runtime.optimizer
    frames = training_frames(scene, settings.model)
    epochs = iterations // len(frames)
    if epochs < 1:
        raise ValueError("iterations must cover at least one complete epoch")
    densify_from = dp.densify_from
    interval, reset_interval = dp.densification_interval, dp.opacity_reset_interval
    densify_until = dp.end_epoch(epochs)
    config = settings.capacity
    xyz, rgb = load_colmap_points(scene)
    initial_count = len(xyz)
    if initial_count == 0:
        raise ValueError("initial cloud is empty")
    scale = estimate_initial_scales(xyz)
    # LiteGS cluster_points pads its initial cloud with copies of the final
    # points. Gather those same copies into slots without resizing the pool.
    padding = (-initial_count) % config.cluster_size
    indices = np.arange(initial_count + padding)
    indices[initial_count:] = np.arange(initial_count - padding, initial_count) % initial_count
    if len(indices) > config.max_gaussians:
        raise ValueError("initial cloud including cluster padding exceeds capacity")
    pool = seed_pool(
        create_pool(config), xyz[indices], rgb[indices], scale=scale[indices], opacity=0.1
    )
    state = create_adam_state(pool)
    bounds = world_cluster_bounds(pool, config.cluster_size)
    stats = jnp.zeros((config.max_gaussians, 4), jnp.float32)
    centers = np.stack([np.asarray(frame.camera.center) for frame in frames])
    scene_radius = jnp.array(
        1.1 * np.max(np.linalg.norm(centers - centers.mean(0), axis=1)), jnp.float32
    )
    image_load_start = perf_counter()
    with closing(iter(image_dataset(frames))) as images_iter:
        targets = [jax.device_put(image) for image in images_iter]
    jax.block_until_ready(targets)
    image_load_seconds = perf_counter() - image_load_start
    key = jax.random.key(seed)
    order_rng = np.random.default_rng(seed)
    jax.block_until_ready((pool, state, bounds, targets))

    warmup_start = perf_counter()
    for degree in range(config.sh_degree + 1):
        for collect in (False, True):
            warm_pool, warm_state, warm_stats = jax.tree.map(
                lambda x: x.copy(), (pool, state, stats)
            )
            jax.block_until_ready(
                train_step(
                    warm_state,
                    warm_stats,
                    GaussianModel(warm_pool),
                    bounds,
                    frames[0].camera,
                    targets[0],
                    jnp.array(0, jnp.int32),
                    scene_radius,
                    config,
                    degree,
                    collect,
                    op.position_lr_max_steps,
                    jnp.array(False),
                    jnp.array(0, jnp.int32),
                    optimizer,
                    op,
                )
            )
    densify_step.lower(
        pool,
        state,
        stats,
        key,
        jnp.array(initial_count, jnp.int32),
        scene_radius,
        cluster_size=config.cluster_size,
        percent_dense=dp.percent_dense,
    ).compile()
    jax.block_until_ready(decay_opacity(pool, state))
    jax.block_until_ready(spatial_refine(pool, state))
    warmup_seconds = perf_counter() - warmup_start
    print(f"warmup: {warmup_seconds:.3f}s; {len(frames)} views x {epochs} epochs", flush=True)
    # create_adam_state shares its initial zero buffers. Donation needs distinct
    # buffers for m and v; subsequent optimizer outputs already have them.
    state = state.replace(v=jax.tree.map(lambda x: x.copy(), state.v))
    jax.block_until_ready(state)
    model = GaussianModel(pool)

    start = perf_counter()
    history = []
    step = 0
    for epoch in range(epochs):
        if (epoch - 1) % interval == 0:
            pool, state = spatial_refine(model.as_pool(), state)
            model.update_from_pool(pool)
            bounds = world_cluster_bounds(pool, config.cluster_size)
        collect = densify_from <= epoch < densify_until and epoch % interval == 0
        overflow, peak_pairs = jnp.array(False), jnp.array(0, jnp.int32)
        for frame_index in order_rng.permutation(len(frames)):
            state, stats, loss, overflow, peak_pairs = train_step(
                state,
                stats,
                model,
                bounds,
                frames[frame_index].camera,
                targets[frame_index],
                jnp.array(step, jnp.int32),
                scene_radius,
                config,
                min(epoch // 5, config.sh_degree),
                collect,
                op.position_lr_max_steps,
                overflow,
                peak_pairs,
                optimizer,
                op,
            )
            step += 1
        if bool(overflow):
            raise RuntimeError(f"epoch {epoch}: {int(peak_pairs)} pairs exceed {pair_capacity}")
        born, pruned = 0, 0
        pool = model.as_pool()
        if collect:
            target = int(
                (target_points - initial_count)
                / (densify_until - densify_from)
                * (epoch - densify_from)
                + initial_count
            )
            key, subkey = jax.random.split(key)
            pool, state, born, pruned = densify_step(
                pool,
                state,
                stats,
                subkey,
                jnp.array(target, jnp.int32),
                scene_radius,
                cluster_size=config.cluster_size,
                percent_dense=dp.percent_dense,
            )
            stats = jnp.zeros_like(stats)
        if densify_from <= epoch < densify_until and epoch % reset_interval == 0:
            pool, state = decay_opacity(pool, state)
            state = state.replace(v=jax.tree.map(lambda x: x.copy(), state.v))
        model.update_from_pool(pool)
        row = {
            "epoch": epoch,
            "step": step,
            "active": int(pool.n_active),
            "loss": float(loss),
            "born": int(born),
            "pruned": int(pruned),
            "peak_pairs": int(peak_pairs),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if not np.isfinite(row["loss"]):
            raise RuntimeError("non-finite training loss")
    jax.block_until_ready((pool, state))
    training_seconds = perf_counter() - start
    output = Path(output)
    save_pool(output, pool)
    report = {
        "scene": str(scene),
        "gpu": jax.devices()[0].device_kind,
        "images": images,
        "image_shape": list(targets[0].shape[:2]),
        "training_images": len(frames),
        "actual_updates": step,
        "initial_sparse_gaussians": initial_count,
        "initial_padded_gaussians": len(indices),
        "target_gaussians": target_points,
        "final_gaussians": int(pool.n_active),
        "pair_capacity": pair_capacity,
        "cluster_size": config.cluster_size,
        "tile_size": config.tile_size,
        "tile_height": config.raster_tile_height,
        "warmup_seconds": warmup_seconds,
        "image_load_seconds": image_load_seconds,
        "data_loader": "grain.MapDataset, 4 decode threads, 8 prefetched images; GPU uint8 preload",
        "training_seconds": training_seconds,
        "densify_until": densify_until,
        "seed": seed,
        "history": history,
        "config": asdict(settings),
        "model": "flax.nnx.Module",
        "jit": "nnx.jit(graph=False)",
        "optimizer": optimizer,
        "jit_cache_size": train_step.jitted_fn._cache_size(),
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
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"training: {training_seconds:.3f}s; report: {report_path}", flush=True)
    return report


def main(argv=None):
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
        "--optimizer", choices=("optax", "cute"), help="optimizer backend (default: optax)"
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
