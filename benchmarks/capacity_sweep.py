"""Legacy bounded-tile CuTe sweep; does not measure the production train_step."""

import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from jaxgs import Camera, CapacityConfig, create_pool, estimate_initial_scales, seed_pool
from jaxgs.io_manager.colmap import load_colmap, load_colmap_points
from jaxgs.kernels.binning import build_visibility_table_cute
from jaxgs.kernels.projector import project_cute, project_cute_vjp
from jaxgs.kernels.rasterizer import rasterize_cute, rasterize_cute_vjp
from jaxgs.scene.spatial_refine import reorder_pool
from jaxgs.training.optimizer import create_adam_state


def milliseconds(fn, repeats: int) -> float:
    jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        jax.block_until_ready(fn())
        samples.append((perf_counter() - start) * 1000)
    return median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--include-backward", action="store_true")
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--sh-degree", type=int, default=3)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("capacity sweep requires JAX CUDA")

    if args.scene:
        frames = load_colmap(args.scene, args.downsample)
        points, colors = load_colmap_points(args.scene)
        if not frames or len(points) == 0:
            raise ValueError("scene must contain images and sparse points")
        camera = frames[0].camera
        indices = np.linspace(0, len(points) - 1, min(args.capacity, len(points)), dtype=np.int32)
        xyz, rgb = points[indices], colors[indices]
        scales = estimate_initial_scales(points)[indices]
    else:
        side = math.ceil(math.sqrt(args.capacity))
        grid_x, grid_y = jnp.meshgrid(jnp.linspace(-1, 1, side), jnp.linspace(-1, 1, side))
        xyz = jnp.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), jnp.full((side * side,), 2.0)], axis=-1
        )[: args.capacity]
        rgb = jnp.full((args.capacity, 3), 0.5)
        scales = 0.012
        size = args.image_size
        camera = Camera.from_colmap(
            [1, 0, 0, 0], [0, 0, 0], size, size, size / 2, size / 2, size, size
        )
    results = []
    reference_signature = None
    reference_image = None
    for cluster_size in (128, 256, 512):
        for k_max in (64, 128, 256):
            config = CapacityConfig(args.capacity, cluster_size, k_max, 16, args.sh_degree)
            pool = seed_pool(create_pool(config), xyz, rgb, scale=scales, opacity=0.1)
            if args.scene:
                pool, _ = reorder_pool(pool, create_adam_state(pool))
            projection = jax.jit(lambda: project_cute(pool, camera, config))
            projected = projection()
            binning = jax.jit(lambda: build_visibility_table_cute(projected, camera, config))
            table = binning()
            rendering = jax.jit(lambda: rasterize_cute(projected, table, camera, config))
            overflow_tiles = int(jnp.sum(table.overflow))
            if overflow_tiles:
                results.append(
                    {
                        "cluster_size": cluster_size,
                        "max_gaussians_per_tile": k_max,
                        "overflow_tiles": overflow_tiles,
                    }
                )
                continue
            counts = np.asarray(table.tile_count)
            ids = np.asarray(table.tile_gaussian_ids)
            signature = tuple(tuple(row[:count]) for row, count in zip(ids, counts, strict=True))
            if reference_signature is None:
                reference_signature = signature
            elif signature != reference_signature:
                raise RuntimeError("visibility tables differ across capacity combinations")
            image = np.asarray(rendering().rgb)
            if reference_image is None:
                reference_image = image
            else:
                np.testing.assert_allclose(image, reference_image, rtol=1e-5, atol=1e-6)
            projection_ms = milliseconds(lambda: projection().mean, args.repeats)
            binning_ms = milliseconds(lambda: binning().tile_count, args.repeats)
            render_ms = milliseconds(lambda: rendering().rgb, args.repeats)
            result = {
                "cluster_size": cluster_size,
                "max_gaussians_per_tile": k_max,
                "projection_ms": projection_ms,
                "binning_ms": binning_ms,
                "render_ms": render_ms,
                "total_ms": projection_ms + binning_ms + render_ms,
            }
            if args.include_backward:
                training_render = jax.jit(
                    jax.grad(
                        lambda mean: jnp.sum(
                            rasterize_cute_vjp(
                                projected.replace(mean=mean), table, camera, config
                            ).rgb
                        )
                    )
                )
                forward_backward_ms = milliseconds(
                    lambda: training_render(projected.mean), args.repeats
                )
                result["forward_backward_ms"] = forward_backward_ms

                def parameter_loss(xyz, log_scale, rotation, opacity, sh):
                    current = pool.replace(
                        xyz=xyz, log_scale=log_scale, rotation=rotation, opacity=opacity, sh=sh
                    )
                    current_projection = project_cute_vjp(current, camera, config)
                    return jnp.sum(
                        rasterize_cute_vjp(current_projection, table, camera, config).rgb
                    )

                parameter_grad = jax.jit(jax.grad(parameter_loss, argnums=(0, 1, 2, 3, 4)))
                parameter_backward_ms = milliseconds(
                    lambda: parameter_grad(
                        pool.xyz, pool.log_scale, pool.rotation, pool.opacity, pool.sh
                    ),
                    args.repeats,
                )
                result["parameter_backward_ms"] = parameter_backward_ms
            results.append(result)
    print(json.dumps(sorted(results, key=lambda row: row.get("total_ms", float("inf"))), indent=2))


if __name__ == "__main__":
    main()
