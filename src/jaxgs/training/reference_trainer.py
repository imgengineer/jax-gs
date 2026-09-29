import argparse
from contextlib import closing
from functools import partial
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..config import CapacityConfig
from ..data import image_dataset
from ..io_manager.checkpoint import save_pool
from ..io_manager.colmap import load_colmap, load_colmap_points
from ..reference.loss import photometric_loss
from ..render.cluster_culling import cluster_culling
from ..render.projection import project
from ..render.rasterize_backward import rasterize
from ..render.visibility_table import build_visibility_table
from ..scene.camera import Camera
from ..scene.point import (
    GaussianModel,
    GaussianPool,
    create_pool,
    estimate_initial_scales,
    seed_pool,
)
from ..scene.spatial_refine import spatial_refine
from .optimizer import AdamState, create_adam_state, masked_adam_update
from .reference_densify import densify_step, prune_step, reset_opacity


@partial(jax.jit, static_argnames=("config", "learning_rate", "backend"))
def train_step(
    pool: GaussianPool,
    state: AdamState,
    camera: Camera,
    target: chex.Array,
    config: CapacityConfig,
    learning_rate: float = 1e-3,
    backend: str = "cute",
):
    """Fixed-capacity step; JAX binning remains a small-scene reference path."""
    if backend not in ("reference", "cute"):
        raise ValueError(f"unknown backend: {backend}")
    if backend == "cute":
        from ..kernels.projector import project_cute_vjp
        from ..kernels.sorted_binning import build_sorted_visibility_table_cute
        from ..kernels.sorted_rasterizer import rasterize_sorted_cute_vjp

    def loss_fn(xyz, log_scale, rotation, opacity, sh, mean_delta):
        current = pool.replace(
            xyz=xyz, log_scale=log_scale, rotation=rotation, opacity=opacity, sh=sh
        )
        projected_current = (
            project_cute_vjp(current, camera, config)
            if backend == "cute"
            else project(current, camera, config)
        )
        projected = jax.lax.stop_gradient(projected_current)
        if backend == "cute":
            table = build_sorted_visibility_table_cute(projected, camera, config)
            binned_visible = table.point_counts > 0
            image = rasterize_sorted_cute_vjp(
                projected_current.replace(mean=projected_current.mean + mean_delta),
                table,
                camera,
                config,
            ).rgb
        else:
            clusters = cluster_culling(projected, camera, config)
            table = build_visibility_table(projected, camera, config, clusters)
            binned_visible = (
                jnp.zeros((config.max_gaussians,), jnp.int32)
                .at[table.tile_gaussian_ids.reshape(-1)]
                .max(table.tile_valid.reshape(-1).astype(jnp.int32))
                > 0
            )
            image = rasterize(
                projected_current.replace(mean=projected_current.mean + mean_delta),
                table,
                camera,
                config,
                backend=backend,
            ).rgb
        return photometric_loss(image, target), (jnp.any(table.overflow), binned_visible)

    parameters = (
        pool.xyz,
        pool.log_scale,
        pool.rotation,
        pool.opacity,
        pool.sh,
        jnp.zeros_like(pool.xyz[:, :2]),
    )
    (loss, (overflow, binned_visible)), gradients = jax.value_and_grad(
        loss_fn, argnums=(0, 1, 2, 3, 4, 5), has_aux=True
    )(*parameters)
    next_pool, next_state = masked_adam_update(pool, state, gradients[:5], learning_rate)
    stats = jax.lax.stop_gradient(jnp.linalg.norm(gradients[5], axis=1))
    return (
        next_pool,
        next_state,
        {
            "loss": loss,
            "overflow": overflow,
            "gradient_stats": stats,
            "visible": jax.lax.stop_gradient(binned_visible),
        },
    )


@partial(nnx.jit, graph=False, static_argnames=("config", "learning_rate", "backend"))
def nnx_train_step(
    model: GaussianModel,
    state: AdamState,
    camera: Camera,
    target: chex.Array,
    config: CapacityConfig,
    learning_rate: float = 1e-3,
    backend: str = "cute",
):
    pool, state, metrics = train_step.__wrapped__(
        model.as_pool(), state, camera, target, config, learning_rate, backend
    )
    model.update_from_pool(pool)
    return state, metrics


def train_colmap(
    scene_dir: str | Path,
    output: str | Path,
    *,
    steps: int = 1000,
    capacity: int = 1024,
    downsample: int = 8,
    backend: str = "cute",
    densify_every: int = 100,
    max_new: int = 16,
    densify_threshold: float = 1e-6,
    learning_rate: float = 1e-3,
    cluster_size: int = 128,
    max_gaussians_per_tile: int = 128,
    tile_size: int = 16,
    sh_degree: int = 3,
    initial_points: int | None = None,
    max_visibility_pairs: int | None = None,
) -> GaussianPool:
    """Small-scene training loop for undistorted COLMAP reconstructions."""
    frames = load_colmap(scene_dir, downsample)
    xyz, rgb = load_colmap_points(scene_dir)
    if not frames or xyz.shape[0] == 0:
        raise ValueError("scene must contain images and sparse points")
    config = CapacityConfig(
        capacity, cluster_size, max_gaussians_per_tile, tile_size, sh_degree, max_visibility_pairs
    )
    if initial_points is None:
        initial_points = (
            min(capacity, xyz.shape[0]) if xyz.shape[0] < capacity else max(1, capacity // 2)
        )
    if not 0 < initial_points <= min(capacity, xyz.shape[0]):
        raise ValueError("initial_points must fit the scene and pool capacity")
    indices = np.linspace(0, xyz.shape[0] - 1, initial_points, dtype=np.int32)
    initial_xyz, initial_rgb = xyz[indices], rgb[indices]
    initial_scales = estimate_initial_scales(xyz)[indices]
    pool = seed_pool(
        create_pool(config), initial_xyz, initial_rgb, scale=initial_scales, opacity=0.1
    )
    state = create_adam_state(pool)
    pool, state = spatial_refine(pool, state)
    model = GaussianModel(pool)
    gradient_sum = jnp.zeros((capacity,), jnp.float32)
    visible_count = jnp.zeros((capacity,), jnp.int32)
    key = jax.random.key(0)

    with closing(iter(image_dataset(frames, steps=steps))) as images:
        for step, image in enumerate(images, start=1):
            frame = frames[(step - 1) % len(frames)]
            target = jnp.asarray(image.astype(np.float32) / 255.0)
            state, metrics = nnx_train_step(
                model, state, frame.camera, target, config, learning_rate, backend
            )
            pool = model.as_pool()
            if bool(metrics["overflow"]):
                if backend == "cute":
                    raise RuntimeError(
                        "visibility pair capacity overflow; increase --max-visibility-pairs"
                    )
                raise RuntimeError("tile capacity overflow; increase --max-gaussians-per-tile")
            visible = metrics["visible"] & pool.alive
            gradient_sum = gradient_sum + jnp.where(visible, metrics["gradient_stats"], 0)
            visible_count = visible_count + visible.astype(jnp.int32)
            if densify_every > 0 and step % densify_every == 0:
                mean_gradient = gradient_sum / jnp.maximum(visible_count, 1)
                key, subkey = jax.random.split(key)
                pool, state, _ = densify_step(
                    pool,
                    state,
                    mean_gradient,
                    subkey,
                    max_new=min(max_new, capacity),
                    threshold=densify_threshold,
                    allocator="cute" if backend == "cute" else "jax",
                )
                gradient_sum = jnp.zeros_like(gradient_sum)
                visible_count = jnp.zeros_like(visible_count)
                pool, state = prune_step(pool, state, jax.nn.sigmoid(pool.opacity[:, 0]) < 0.005)
                pool, state = spatial_refine(pool, state)
            if step % 3000 == 0:
                pool, state = reset_opacity(pool, state)
            model.update_from_pool(pool)
            if step == 1 or step % 100 == 0 or step == steps:
                print(
                    f"step {step}: loss={float(metrics['loss']):.6f}, active={int(pool.n_active)}"
                )

    output = Path(output)
    save_pool(output, pool)
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small fixed-capacity Gaussian scene")
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("gaussians.npz"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--initial-points", type=int)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--backend", choices=("reference", "cute"), default="cute")
    parser.add_argument("--densify-every", type=int, default=100)
    parser.add_argument("--densify-threshold", type=float, default=1e-6)
    parser.add_argument("--cluster-size", type=int, default=128)
    parser.add_argument("--max-gaussians-per-tile", type=int, default=128)
    parser.add_argument("--max-visibility-pairs", type=int)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--sh-degree", type=int, default=3)
    args = parser.parse_args()
    train_colmap(
        args.scene_dir,
        args.output,
        steps=args.steps,
        capacity=args.capacity,
        downsample=args.downsample,
        backend=args.backend,
        densify_every=args.densify_every,
        densify_threshold=args.densify_threshold,
        cluster_size=args.cluster_size,
        max_gaussians_per_tile=args.max_gaussians_per_tile,
        tile_size=args.tile_size,
        sh_degree=args.sh_degree,
        max_visibility_pairs=args.max_visibility_pairs,
        initial_points=args.initial_points,
    )


if __name__ == "__main__":
    main()
