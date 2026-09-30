"""Reproduce global-pair occupancy and fixed-state training-step timings.

This measures repeated updates from the same input state, not a training run.
LiteGS PLY ordering follows litegs/io_manager/ply.py::load_ply.
"""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from plyfile import PlyData

from jaxgs import CapacityConfig, create_gaussians, estimate_initial_scales, seed_gaussians
from jaxgs.io_manager.colmap import load_colmap_images, load_colmap_points
from jaxgs.kernels.projector import project_cute
from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
from jaxgs.kernels.sorted_rasterizer import rasterize_sorted_cute_vjp
from jaxgs.training.optimizer import create_adam_state
from jaxgs.training.reference_trainer import train_step


def load_gaussian_ply(path: Path, config: CapacityConfig):
    vertices = PlyData.read(path)["vertex"]
    count = len(vertices)
    if count > config.max_gaussians:
        raise ValueError("PLY exceeds Gaussian pool capacity")

    def fields(names):
        return np.stack([vertices[name] for name in names], axis=-1).astype(np.float32)

    pool = create_gaussians(config)
    sh = np.zeros((count, config.sh_dim, 3), np.float32)
    sh[:, 0] = fields([f"f_dc_{i}" for i in range(3)])
    if config.sh_dim > 1:
        # PLY stores [point, channel, coefficient]; the pool uses
        # [point, coefficient, channel]. A direct (N, 15, 3) reshape is wrong.
        rest = fields([f"f_rest_{i}" for i in range(3 * (config.sh_dim - 1))])
        sh[:, 1:] = rest.reshape(count, 3, config.sh_dim - 1).transpose(0, 2, 1)
    values = dict(
        xyz=fields(["x", "y", "z"]),
        log_scale=fields([f"scale_{i}" for i in range(3)]),
        rotation=fields([f"rot_{i}" for i in range(4)]),
        opacity=fields(["opacity"]),
        sh=sh,
    )
    alive = jnp.arange(config.max_gaussians) < count
    return pool.replace(
        **{name: getattr(pool, name).at[:count].set(value) for name, value in values.items()},
        alive=alive,
        free_mask=~alive,
        n_active=jnp.array(count, jnp.int32),
    )


def measure(fn, repeats):
    jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        jax.block_until_ready(fn())
        samples.append(1000 * (perf_counter() - start))
    return {"median_ms": median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--images", default="images_4")
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--capacity", type=int, default=1_000_000)
    parser.add_argument("--max-visibility-pairs", type=int, default=4_000_000)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--all-views", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    config = CapacityConfig(args.capacity, 128, 128, 16, 3, args.max_visibility_pairs)
    frames = load_colmap_images(args.scene, args.images)
    if args.ply:
        pool = load_gaussian_ply(args.ply, config)
    else:
        xyz, rgb = load_colmap_points(args.scene)
        if len(xyz) > args.capacity:
            parser.error("sparse point cloud exceeds --capacity")
        pool = seed_gaussians(
            create_gaussians(config), xyz, rgb, scale=estimate_initial_scales(xyz), opacity=0.1
        )
    state = create_adam_state(pool)
    frame = frames[0]
    target = frame.load_image()
    project = jax.jit(lambda p, c: project_cute(p, c, config))
    binning = jax.jit(lambda p, c: build_sorted_visibility_table_cute(p, c, config))
    render = jax.jit(lambda p, t, c: rasterize_sorted_cute_vjp(p, t, c, config))

    occupancy = []
    for current in frames if args.all_views else frames[:1]:
        table = binning(project(pool, current.camera), current.camera)
        overflow = bool(table.overflow)
        occupancy.append(
            {
                "view": current.image_path.name,
                "pairs": int(table.pair_count),
                "max_tile_count": int(jnp.max(jnp.diff(table.tile_offsets))),
                "overflow": overflow,
            }
        )
        if overflow:
            raise RuntimeError(
                f"{current.image_path.name}: {int(table.pair_count)} pairs "
                f"exceed capacity {config.visibility_capacity}"
            )
    projected = project(pool, frame.camera)
    table = binning(projected, frame.camera)

    def step():
        return train_step(pool, state, frame.camera, target, config)

    compile_start = perf_counter()
    next_pool, _, metrics = jax.block_until_ready(step())
    first_step_seconds = perf_counter() - compile_start
    if bool(metrics["overflow"]) or not np.isfinite(float(metrics["loss"])):
        raise RuntimeError("training probe overflowed or produced a non-finite loss")
    for name in ("xyz", "log_scale", "rotation", "opacity", "sh"):
        if not bool(jnp.all(jnp.isfinite(getattr(next_pool, name)))):
            raise RuntimeError(f"non-finite updated {name}")
    result = {
        "scope": "fixed-state step probe, not full training",
        "gpu": jax.devices()[0].device_kind,
        "scene": str(args.scene),
        "images": args.images,
        "ply": str(args.ply) if args.ply else None,
        "capacity": args.capacity,
        "active": int(pool.n_active),
        "pair_capacity": config.visibility_capacity,
        "image_shape": [frame.camera.height, frame.camera.width],
        "sh_degree": config.sh_degree,
        "tile_size": config.tile_size,
        "first_step_including_compile_seconds": first_step_seconds,
        "loss": float(metrics["loss"]),
        "occupancy": occupancy,
        "max_pairs_across_views": max(row["pairs"] for row in occupancy),
        "projection": measure(lambda: project(pool, frame.camera), args.repeats),
        "binning": measure(lambda: binning(projected, frame.camera), args.repeats),
        "forward": measure(lambda: render(projected, table, frame.camera), args.repeats),
        "train_step": measure(step, args.repeats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("active", "image_shape", "max_pairs_across_views", "loss")
            },
            indent=2,
        )
    )
    for stage in ("projection", "binning", "forward", "train_step"):
        print(f"{stage}: {result[stage]['median_ms']:.3f} ms")


if __name__ == "__main__":
    main()
