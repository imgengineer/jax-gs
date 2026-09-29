"""Evaluate every eighth COLMAP view using each model's own renderer."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def evaluate_jax(args):
    import jax.numpy as jnp
    from flax import nnx

    from jaxgs import CapacityConfig, GaussianModel
    from jaxgs.io_manager.checkpoint import load_pool
    from jaxgs.io_manager.colmap import load_colmap_images
    from jaxgs.kernels.packed_rasterizer import packed_forward
    from jaxgs.kernels.projector import project_cute
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute

    pool = load_pool(args.model)
    config = CapacityConfig(pool.xyz.shape[0], 128, 128, 16, 3, 8_000_000, tile_height=8)
    model = GaussianModel(pool)

    @nnx.jit(graph=False)
    def render(model, camera):
        projected = project_cute(model.as_pool(), camera, config)
        table = build_sorted_visibility_table_cute(projected, camera, config)
        image, _, _ = packed_forward(projected, table, camera, config)
        return jnp.clip(image, 0, 1), table.overflow

    rows = []
    for frame in load_colmap_images(args.scene, args.images)[::8]:
        image, overflow = render(model, frame.camera)
        if bool(overflow):
            raise RuntimeError(f"overflow in evaluation view {frame.image_path.name}")
        mse = float(jnp.mean((image - frame.load_image()) ** 2))
        rows.append({"view": frame.image_path.name, "psnr": -10 * np.log10(mse)})
    return rows


def evaluate_litegs(args):
    sys.path.insert(0, str(args.litegs_root))
    import litegs.config
    import torch
    from litegs import data, io_manager, render, scene

    cameras, frames, _, _ = io_manager.load_colmap_result(str(args.scene), args.images)
    frames = frames[::8]
    for frame in frames:
        frame.load_image(-1)
    dataset = data.CameraFrameDataset(cameras, frames, -1, True)
    _, _, pipeline, _ = litegs.config.get_default_arg()
    arrays = io_manager.load_ply(str(args.model), 3)
    params = tuple(
        torch.as_tensor(value, dtype=torch.float32, device="cuda").contiguous() for value in arrays
    )
    xyz, scale, rot, sh0, shrest, opacity = scene.cluster.cluster_points(128, *params)
    with torch.no_grad():
        feedback = data.FramesBuffer(dataset)
        origin, extent = scene.cluster.get_cluster_AABB(
            xyz, scale.exp(), torch.nn.functional.normalize(rot, dim=0)
        )
        rows = []
        for index, frame in enumerate(frames):
            view, projection, planes, target, _ = dataset[index]
            # Native kernels receive contiguous batches from the training
            # DataLoader; adding a batch axis alone preserves transposed strides.
            view, projection, planes = (
                value[None].contiguous() for value in (view, projection, planes)
            )
            index_tensor = torch.tensor([index], dtype=torch.int64)
            ids, count, px, ps, pr, color, po = render.render_preprocess(
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
                index_tensor,
                pipeline,
                3,
            )
            image, *_ = render.render(
                view,
                projection,
                px,
                ps,
                pr,
                color,
                po,
                count * 128,
                feedback.feedback_binning_allocate_size,
                index_tensor,
                3,
                target.shape[1:],
                pipeline,
            )
            mse = float(torch.mean((image[0] - target / 255.0) ** 2))
            rows.append({"view": frame.name, "psnr": -10 * np.log10(mse)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--backend", choices=("jaxgs", "litegs"), required=True)
    parser.add_argument("--litegs-root", type=Path)
    parser.add_argument("--images", default="images_4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = evaluate_jax(args) if args.backend == "jaxgs" else evaluate_litegs(args)
    result = {
        "backend": args.backend,
        "model": str(args.model),
        "views": len(rows),
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("backend", "views", "mean_psnr")}))


if __name__ == "__main__":
    main()
