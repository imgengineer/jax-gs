"""Fixed-count training from the same LiteGS PLY, without densification."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


def _cluster_indices(count: int, cluster_size: int) -> np.ndarray:
    """Keep the input order and repeat tail points to complete the last cluster."""
    if count < 1:
        raise ValueError("Gaussian PLY is empty")
    padding = (-count) % cluster_size
    indices = np.arange(count + padding)
    indices[count:] = np.arange(count - padding, count) % count
    return indices


def benchmark_jax(args):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from plyfile import PlyData
    from sorted_pipeline import load_gaussian_ply

    from jaxgs import CapacityConfig, GaussianModel
    from jaxgs.config import load_config
    from jaxgs.io_manager.colmap import load_colmap_images
    from jaxgs.scene.cluster import world_cluster_bounds
    from jaxgs.scene.types import PARAMETER_NAMES
    from jaxgs.training.optimizer import create_adam_state
    from jaxgs.training.state import TrainingState
    from jaxgs.training.step import array_train_step, compute_training_step

    optimization = load_config(Path(__file__).parent / "configs" / "bicycle_10k.toml").optimization
    input_count = len(PlyData.read(args.ply)["vertex"])
    indices = _cluster_indices(input_count, 128)
    count = len(indices)
    config = CapacityConfig(count, 128, 128, 16, 3, args.pairs, tile_height=8)
    pool = load_gaussian_ply(args.ply, config)
    if count != input_count:
        pool = pool.replace(
            **{name: getattr(pool, name)[indices] for name in PARAMETER_NAMES},
            alive=jnp.ones((count,), jnp.bool_),
            free_mask=jnp.zeros((count,), jnp.bool_),
            n_active=jnp.array(count, jnp.int32),
        )
    state = create_adam_state(pool)
    bounds = world_cluster_bounds(pool, 128)
    frame = load_colmap_images(args.scene, args.images, resolution=-1)[args.view]
    target = jnp.asarray(frame.load_rgb())
    stats = jnp.zeros((count, 4), jnp.float32)
    radius = jnp.array(1.0, jnp.float32)

    # Compile an ordinary warmup before binding the donated timed step.
    compile_start = perf_counter()
    pool, state, stats, metrics = jax.block_until_ready(
        array_train_step(
            pool,
            state,
            stats,
            bounds,
            frame.camera,
            target,
            jnp.array(0),
            radius,
            config,
            3,
            False,
            optimizer=args.optimizer,
            optimization=optimization,
        )
    )

    model = GaussianModel(pool)
    training = TrainingState(model, state, stats)

    def update(training, seen_overflow, step):
        pool, state, stats, metrics = compute_training_step(
            training.model.as_arrays(),
            training.adam.get_value(),
            training.fragments.get_value(),
            bounds,
            frame.camera,
            target,
            step,
            radius,
            config,
            3,
            False,
            optimizer=args.optimizer,
            optimization=optimization,
        )
        training.model.update_from_arrays(pool)
        training.adam.set_value(state)
        training.fragments.set_value(stats)
        return seen_overflow | metrics["overflow"], metrics

    update = nnx.jit_partial(update, training, graph=False, donate_argnums=(0,))
    overflow = metrics["overflow"]
    for step in range(1, args.warmup):
        overflow, metrics = update(overflow, jnp.array(step))
    jax.block_until_ready((training, metrics))
    warmup_seconds = perf_counter() - compile_start
    first_loss = float(metrics["loss"])
    start = perf_counter()
    for step in range(args.warmup, args.warmup + args.steps):
        overflow, metrics = update(overflow, jnp.array(step))
    jax.block_until_ready((training, metrics))
    seconds = perf_counter() - start
    if bool(overflow):
        raise RuntimeError("fixed-count benchmark overflowed")
    result = {
        "input_count": input_count,
        "count": count,
        "image": frame.image_path.name,
        "shape": list(target.shape[:2]),
        "seconds": seconds,
        "warmup_seconds": warmup_seconds,
        "initial_loss": first_loss,
        "final_loss": float(metrics["loss"]),
        "final_pairs": int(metrics["pairs"]),
        "jit_cache_size": update.jitted_fn._cache_size(),
        "donation": True,
        "model": "flax.nnx.Module",
        "jit": "nnx.jit_partial(graph=False)",
        "optimizer": args.optimizer,
    }
    if args.trace:
        with jax.profiler.trace(str(args.trace)):
            for step in range(10):
                overflow, metrics = update(overflow, jnp.array(args.steps + args.warmup + step))
            jax.block_until_ready((training, metrics))
    return result


def benchmark_litegs(args):
    sys.path.insert(0, str(args.litegs_root))
    import fused_ssim
    import litegs.config
    import torch
    from litegs import data, io_manager, render, scene
    from litegs.training.optimizer import get_optimizer

    cameras, frames, _, _ = io_manager.load_colmap_result(str(args.scene), args.images)
    frame = frames[args.view]
    frame.load_image(-1)
    dataset = data.CameraFrameDataset(cameras, [frame], -1, True)
    feedback = data.FramesBuffer(dataset)
    _, optimizer_settings, pipeline, _ = litegs.config.get_default_arg()
    optimizer_settings.position_lr_final = 0.000016
    optimizer_settings.position_lr_max_steps = 10000
    arrays = io_manager.load_ply(str(args.ply), 3)
    input_count = arrays[0].shape[-1]
    indices = _cluster_indices(input_count, 128)
    count = len(indices)
    if count != input_count:
        arrays = tuple(value[..., indices] for value in arrays)
    params = tuple(
        torch.as_tensor(value, dtype=torch.float32, device="cuda").contiguous() for value in arrays
    )
    params = tuple(
        torch.nn.Parameter(value) for value in scene.cluster.cluster_points(128, *params)
    )
    xyz, scale, rot, sh0, shrest, opacity = params
    optimizer, scheduler = get_optimizer(*params, 1.0, optimizer_settings, pipeline)
    with torch.no_grad():
        origin, extent = scene.cluster.get_cluster_AABB(
            xyz, scale.exp(), torch.nn.functional.normalize(rot, dim=0)
        )
    view, projection, planes, target, _ = dataset[0]
    view, projection, planes = (value[None].contiguous() for value in (view, projection, planes))
    target = target[None].float() / 255
    index = torch.tensor([0], dtype=torch.int64)

    def step():
        ids, visible_count, px, ps, pr, color, po = render.render_preprocess(
            origin,
            extent,
            planes,
            view,
            xyz,
            scale,
            rot,
            sh0,
            shrest,
            opacity,
            feedback.feedback_visible_chunks_num,
            index,
            pipeline,
            3,
        )
        image, _, _, _, visible = render.render(
            view,
            projection,
            px,
            ps,
            pr,
            color,
            po,
            visible_count * 128,
            feedback.feedback_binning_allocate_size,
            index,
            3,
            target.shape[2:],
            pipeline,
        )
        loss = fused_ssim.fused_l1_ssim_loss(image, target)
        loss.backward()
        optimizer.step(ids, visible_count, visible)
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        return loss

    torch.cuda.synchronize()
    warmup_start = perf_counter()
    for _ in range(args.warmup):
        loss = step()
    torch.cuda.synchronize()
    warmup_seconds = perf_counter() - warmup_start
    first_loss = float(loss.detach())
    start = perf_counter()
    for _ in range(args.steps):
        loss = step()
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    result = {
        "input_count": input_count,
        "count": count,
        "image": frame.name,
        "shape": list(target.shape[2:]),
        "seconds": seconds,
        "warmup_seconds": warmup_seconds,
        "initial_loss": first_loss,
        "final_loss": float(loss.detach()),
    }
    if args.trace:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            for _ in range(10):
                step()
            torch.cuda.synchronize()
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.trace))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("ply", type=Path)
    parser.add_argument("--backend", choices=("jaxgs", "litegs"), required=True)
    parser.add_argument(
        "--optimizer",
        choices=("optax", "muon", "cute"),
        help="jaxgs optimizer backend (default: optax)",
    )
    parser.add_argument("--litegs-root", type=Path)
    parser.add_argument("--images", default="images_4")
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=4_000_000)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 2:
        parser.error("steps must be positive and warmup must be at least two")
    if args.backend != "jaxgs" and args.optimizer is not None:
        parser.error("--optimizer requires --backend jaxgs")
    if args.backend == "jaxgs":
        args.optimizer = args.optimizer or "optax"
    result = benchmark_jax(args) if args.backend == "jaxgs" else benchmark_litegs(args)
    result.update(
        backend=args.backend,
        steps=args.steps,
        warmup_steps=args.warmup,
        mean_step_ms=result["seconds"] * 1000 / args.steps,
        scope="fixed-count updates of one common PLY, one view, SH=3, spatial LR scale=1, no densification",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
