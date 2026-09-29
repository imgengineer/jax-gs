"""Fixed-count training from the same LiteGS PLY, without densification."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


def benchmark_jax(args):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from PIL import Image
    from plyfile import PlyData
    from sorted_pipeline import load_gaussian_ply

    from jaxgs import CapacityConfig, GaussianModel
    from jaxgs.config import load_config
    from jaxgs.io_manager.colmap import load_colmap_images
    from jaxgs.scene.cluster import world_cluster_bounds
    from jaxgs.training.optimizer import create_adam_state
    from jaxgs.training.trainer import array_train_step

    optimization = load_config(Path(__file__).parent / "configs" / "bicycle_10k.toml").optimization
    count = len(PlyData.read(args.ply)["vertex"])
    config = CapacityConfig(count, 128, 128, 16, 3, args.pairs, tile_height=8)
    pool = load_gaussian_ply(args.ply, config)
    state = create_adam_state(pool)
    bounds = world_cluster_bounds(pool, 128)
    frame = load_colmap_images(args.scene, args.images)[args.view]
    with Image.open(frame.image_path) as image:
        target = jnp.asarray(np.asarray(image.convert("RGB"), np.uint8))
    stats = jnp.zeros((count, 4), jnp.float32)
    radius = jnp.array(1.0, jnp.float32)

    # One ordinary warmup produces distinct m/v buffers; the initial Adam
    # state intentionally shares its zero arrays. Subsequent steps donate.
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

    # Match the production argument order: NNX appends model updates after
    # explicit outputs, so state/stats precede model for XLA buffer donation.
    def update(state, stats, model, seen_overflow, step):
        pool, state, stats, metrics = array_train_step.__wrapped__(
            model.as_pool(),
            state,
            stats,
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
        model.update_from_pool(pool)
        return state, stats, seen_overflow | metrics["overflow"], metrics

    update = nnx.jit(update, graph=False, donate_argnums=(0, 1, 2))
    overflow = metrics["overflow"]
    for step in range(1, args.warmup):
        state, stats, overflow, metrics = update(state, stats, model, overflow, jnp.array(step))
    jax.block_until_ready((model.as_pool(), state, stats, metrics))
    warmup_seconds = perf_counter() - compile_start
    first_loss = float(metrics["loss"])
    start = perf_counter()
    for step in range(args.warmup, args.warmup + args.steps):
        state, stats, overflow, metrics = update(state, stats, model, overflow, jnp.array(step))
    jax.block_until_ready((model.as_pool(), state, stats, metrics))
    seconds = perf_counter() - start
    if bool(overflow):
        raise RuntimeError("fixed-count benchmark overflowed")
    result = {
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
        "jit": "nnx.jit(graph=False)",
        "optimizer": args.optimizer,
    }
    if args.trace:
        with jax.profiler.trace(str(args.trace)):
            for step in range(10):
                state, stats, overflow, metrics = update(
                    state, stats, model, overflow, jnp.array(args.steps + args.warmup + step)
                )
            jax.block_until_ready((model.as_pool(), state, stats, metrics))
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
    params = tuple(
        torch.as_tensor(value, dtype=torch.float32, device="cuda").contiguous() for value in arrays
    )
    count = params[0].shape[-1]
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
        "--optimizer", choices=("optax", "cute"), help="jaxgs optimizer backend (default: optax)"
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
