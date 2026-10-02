"""Small-scene training for reference and CuTe correctness checks."""

import argparse
from contextlib import closing
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..config import CapacityConfig
from ..data import image_dataset
from ..io_manager.checkpoint import save_gaussians
from ..io_manager.colmap import load_colmap, load_colmap_points
from ..reference.loss import photometric_loss
from ..render.cluster_culling import build_cluster_tile_mask
from ..render.projection import project
from ..render.rasterizer import rasterize
from ..render.visibility_table import build_visibility_table
from ..scene.camera import Camera
from ..scene.point import (
    GaussianArrays,
    GaussianModel,
    create_gaussians,
    estimate_initial_scales,
    seed_gaussians,
)
from ..scene.spatial_refine import reorder_gaussians
from .optimizer import AdamState, create_adam_state, masked_adam_update
from .pool_ops import prune_step
from .reference_densify import densify_step, reset_opacity


def _compute_training_step(
    pool: GaussianArrays,
    state: AdamState,
    camera: Camera,
    target: chex.Array,
    config: CapacityConfig,
    learning_rate: float = 1e-3,
    backend: str = "cute",
) -> tuple[GaussianArrays, AdamState, dict[str, chex.Array]]:
    """Fixed-capacity step; JAX binning remains a small-scene reference path."""
    if backend not in ("reference", "cute"):
        raise ValueError(f"unknown backend: {backend}")
    if backend == "cute":
        from ..kernels.projector import project_cute_vjp
        from ..kernels.sorted_binning import build_sorted_visibility_table_cute
        from ..kernels.sorted_rasterizer import rasterize_sorted_cute_vjp

    def render_loss(xyz, log_scale, rotation, opacity, sh, mean_delta):
        current_pool = pool.replace(
            xyz=xyz, log_scale=log_scale, rotation=rotation, opacity=opacity, sh=sh
        )
        projected_gaussians = (
            project_cute_vjp(current_pool, camera, config)
            if backend == "cute"
            else project(current_pool, camera, config)
        )
        binning_gaussians = jax.lax.stop_gradient(projected_gaussians)
        if backend == "cute":
            visibility_table = build_sorted_visibility_table_cute(binning_gaussians, camera, config)
            binned_slots = visibility_table.point_counts > 0
            image = rasterize_sorted_cute_vjp(
                projected_gaussians.replace(mean=projected_gaussians.mean + mean_delta),
                visibility_table,
                camera,
                config,
            ).rgb
        else:
            cluster_tile_mask = build_cluster_tile_mask(binning_gaussians, camera, config)
            visibility_table = build_visibility_table(
                binning_gaussians, camera, config, cluster_tile_mask
            )
            binned_slots = (
                jnp.zeros((config.max_gaussians,), jnp.int32)
                .at[visibility_table.tile_gaussian_ids.reshape(-1)]
                .max(visibility_table.tile_valid.reshape(-1).astype(jnp.int32))
                > 0
            )
            image = rasterize(
                projected_gaussians.replace(mean=projected_gaussians.mean + mean_delta),
                visibility_table,
                camera,
                config,
                backend=backend,
            ).rgb
        return photometric_loss(image, target), (jnp.any(visibility_table.overflow), binned_slots)

    parameters = (
        pool.xyz,
        pool.log_scale,
        pool.rotation,
        pool.opacity,
        pool.sh,
        jnp.zeros_like(pool.xyz[:, :2]),
    )
    (loss, (overflow, binned_slots)), gradients = jax.value_and_grad(
        render_loss, argnums=(0, 1, 2, 3, 4, 5), has_aux=True
    )(*parameters)
    next_pool, next_adam_state = masked_adam_update(pool, state, gradients[:5], learning_rate)
    gradient_stats = jax.lax.stop_gradient(jnp.linalg.norm(gradients[5], axis=1))
    return (
        next_pool,
        next_adam_state,
        {
            "loss": loss,
            "overflow": overflow,
            "gradient_stats": gradient_stats,
            "visible": jax.lax.stop_gradient(binned_slots),
        },
    )


_STATIC_ARGUMENTS = ("config", "learning_rate", "backend")
train_step = jax.jit(_compute_training_step, static_argnames=_STATIC_ARGUMENTS)


@nnx.jit(graph=False, static_argnames=_STATIC_ARGUMENTS)
def nnx_train_step(
    model: GaussianModel,
    state: AdamState,
    camera: Camera,
    target: chex.Array,
    config: CapacityConfig,
    learning_rate: float = 1e-3,
    backend: str = "cute",
) -> tuple[AdamState, dict[str, chex.Array]]:
    pool, state, metrics = _compute_training_step(
        model.as_arrays(), state, camera, target, config, learning_rate, backend
    )
    model.update_from_arrays(pool)
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
) -> GaussianArrays:
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
    seed_indices = np.linspace(0, xyz.shape[0] - 1, initial_points, dtype=np.int32)
    initial_xyz, initial_rgb = xyz[seed_indices], rgb[seed_indices]
    initial_scales = estimate_initial_scales(xyz)[seed_indices]
    pool = seed_gaussians(
        create_gaussians(config), initial_xyz, initial_rgb, scale=initial_scales, opacity=0.1
    )
    adam_state = create_adam_state(pool)
    pool, adam_state = reorder_gaussians(pool, adam_state)
    model = GaussianModel(pool)
    gradient_sums = jnp.zeros((capacity,), jnp.float32)
    visibility_counts = jnp.zeros((capacity,), jnp.int32)
    random_key = jax.random.key(0)

    with closing(iter(image_dataset(frames, steps=steps))) as images:
        for step, image_rgb in enumerate(images, start=1):
            frame = frames[(step - 1) % len(frames)]
            target = jnp.asarray(image_rgb.astype(np.float32) / 255.0)
            adam_state, metrics = nnx_train_step(
                model, adam_state, frame.camera, target, config, learning_rate, backend
            )
            pool = model.as_arrays()
            if bool(metrics["overflow"]):
                if backend == "cute":
                    raise RuntimeError(
                        "visibility pair capacity overflow; increase --max-visibility-pairs"
                    )
                raise RuntimeError("tile capacity overflow; increase --max-gaussians-per-tile")
            visible = metrics["visible"] & pool.alive
            gradient_sums = gradient_sums + jnp.where(visible, metrics["gradient_stats"], 0)
            visibility_counts = visibility_counts + visible.astype(jnp.int32)
            if densify_every > 0 and step % densify_every == 0:
                mean_gradients = gradient_sums / jnp.maximum(visibility_counts, 1)
                random_key, densify_key = jax.random.split(random_key)
                pool, adam_state, _ = densify_step(
                    pool,
                    adam_state,
                    mean_gradients,
                    densify_key,
                    max_new=min(max_new, capacity),
                    threshold=densify_threshold,
                    allocator="cute" if backend == "cute" else "jax",
                )
                gradient_sums = jnp.zeros_like(gradient_sums)
                visibility_counts = jnp.zeros_like(visibility_counts)
                pool, adam_state = prune_step(
                    pool, adam_state, jax.nn.sigmoid(pool.opacity[:, 0]) < 0.005
                )
                pool, adam_state = reorder_gaussians(pool, adam_state)
            if step % 3000 == 0:
                pool, adam_state = reset_opacity(pool, adam_state)
            model.update_from_arrays(pool)
            if step == 1 or step % 100 == 0 or step == steps:
                print(
                    f"step {step}: loss={float(metrics['loss']):.6f}, active={int(pool.n_active)}"
                )

    output = Path(output)
    save_gaussians(output, pool)
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small fixed-capacity Gaussian scene")
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("gaussians.ply"))
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
