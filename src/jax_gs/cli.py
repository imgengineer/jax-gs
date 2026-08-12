from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from flax import nnx
import jax
import jax.numpy as jnp

from .checkpoints import (
    is_distributed_checkpoint,
    load_checkpoint_appearance_image_names,
    load_checkpoint_config,
    load_checkpoint_scene_transform,
    load_checkpoint_storage_capacity,
    load_distributed_inference_checkpoint,
    restore_checkpoint,
)
from .config import TrainConfig
from .data import create_grain_dataset, load_colmap_scene
from .exporter import export_splats
from .model import GaussianModel
from .training import (
    SceneTransform,
    _check_evaluation_memory_budget,
    _legacy_scene_transform,
    _save_render,
    estimate_training_memory_bytes,
    make_render_step,
    train,
)
from .training.appearance import (
    APPEARANCE_FEATURE_DIM,
    AppearanceOptModule,
    bake_appearance_sh,
    create_appearance_optimizer,
)


def _load_training_objects(checkpoint: Path):
    config = load_checkpoint_config(checkpoint)
    appearance = None
    image_names = None
    if config.app_opt:
        image_names = load_checkpoint_appearance_image_names(checkpoint)
        if image_names is None:
            raise ValueError("checkpoint does not contain appearance state")
        appearance = AppearanceOptModule(
            len(image_names),
            APPEARANCE_FEATURE_DIM,
            config.app_embed_dim,
            config.model.sh_degree,
            rngs=nnx.Rngs(0),
        )

    if is_distributed_checkpoint(checkpoint):
        model, step = load_distributed_inference_checkpoint(
            checkpoint,
            config,
            appearance_module=appearance,
            appearance_image_names=image_names,
        )
        config = replace(
            config,
            model=replace(config.model, capacity=model.capacity),
        )
        return config, model, appearance, step

    storage_capacity = load_checkpoint_storage_capacity(checkpoint)
    if storage_capacity > config.model.capacity:
        raise ValueError(
            "checkpoint storage capacity exceeds the configured logical maximum"
        )
    model = GaussianModel.empty(
        config.model,
        physical_capacity=storage_capacity,
        appearance_feature_dim=(
            APPEARANCE_FEATURE_DIM if config.app_opt else None
        ),
    )
    restore_kwargs = {}
    if appearance is not None:
        appearance_optimizer = create_appearance_optimizer(
            appearance, config
        )
        restore_kwargs = {
            "appearance_module": appearance,
            "appearance_optimizer": appearance_optimizer,
            "appearance_image_names": image_names,
        }
    step = restore_checkpoint(checkpoint, model, **restore_kwargs)
    return config, model, appearance, step


def _train_command(args: argparse.Namespace) -> None:
    if args.config:
        config = TrainConfig.load(args.config)
    elif args.resume:
        config = load_checkpoint_config(args.resume)
    else:
        config = TrainConfig.for_model_type(
            args.model_type or "3dgs",
            strategy_kind=args.strategy or "default",
        )
    training_overrides = {
        name: getattr(args, name)
        for name in (
            "model_type",
            "global_scale",
            "normalize_world_space",
            "opacity_reg",
            "scale_reg",
            "app_embed_dim",
            "app_opt_lr",
            "app_opt_reg",
            "pose_opt_lr",
            "pose_opt_reg",
            "pose_noise",
            "normal_lambda",
            "normal_start_iter",
            "dist_lambda",
            "dist_start_iter",
        )
        if getattr(args, name) is not None
    }
    if args.normal_loss:
        training_overrides["normal_loss"] = True
    if args.dist_loss:
        training_overrides["dist_loss"] = True
    if args.pose_opt:
        training_overrides["pose_opt"] = True
    if args.app_opt:
        training_overrides["app_opt"] = True
    for name in ("packed", "sparse_grad", "visible_adam"):
        if getattr(args, name):
            training_overrides[name] = True
    if training_overrides:
        config = replace(config, **training_overrides)
    if args.data is not None:
        config = replace(config, data=replace(config.data, root=args.data))
    if args.image_dir is not None:
        config = replace(
            config, data=replace(config.data, image_dir=args.image_dir)
        )
    if args.capacity is not None:
        config = replace(
            config, model=replace(config.model, capacity=args.capacity)
        )
    if args.bucket_min_capacity is not None:
        config = replace(
            config,
            model=replace(
                config.model, bucket_min_capacity=args.bucket_min_capacity
            ),
        )
    if args.patch_size is not None:
        config = replace(
            config, data=replace(config.data, patch_size=args.patch_size)
        )
    if args.num_workers is not None:
        config = replace(
            config, data=replace(config.data, num_workers=args.num_workers)
        )
    if args.max_gaussians_per_tile is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer,
                max_gaussians_per_tile=args.max_gaussians_per_tile,
            ),
        )
    if args.max_candidates_per_tile is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer,
                max_candidates_per_tile=args.max_candidates_per_tile,
            ),
        )
    if args.max_intersections is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer, max_intersections=args.max_intersections
            ),
        )
    if args.intersection_bucket_min_capacity is not None:
        config = replace(
            config,
            intersection_bucket_min_capacity=(
                args.intersection_bucket_min_capacity
            ),
        )
    if args.rasterizer_backend is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer, backend=args.rasterizer_backend
            ),
        )
    if args.compositor_backend is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer,
                compositor_backend=args.compositor_backend,
            ),
        )
    if args.intersection_backend is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer,
                intersection_backend=args.intersection_backend,
            ),
        )
    if args.intersection_mode is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer, intersection_mode=args.intersection_mode
            ),
        )
    if args.sort_backend is not None:
        config = replace(
            config,
            rasterizer=replace(config.rasterizer, sort_backend=args.sort_backend),
        )
    if args.tile_batch_size is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer, tile_batch_size=args.tile_batch_size
            ),
        )
    if args.steps is not None:
        config = replace(
            config,
            steps=args.steps,
            optimizer=replace(config.optimizer, max_steps=args.steps),
        )
    if args.output is not None:
        config = replace(config, output_dir=args.output)
    if args.strategy is not None:
        config = replace(
            config, strategy=replace(config.strategy, kind=args.strategy)
        )
    strategy_overrides = {
        name: getattr(args, name)
        for name in (
            "max_new_per_refine",
            "refine_every",
            "refine_stop",
        )
        if getattr(args, name) is not None
    }
    if strategy_overrides:
        config = replace(
            config,
            strategy=replace(config.strategy, **strategy_overrides),
        )
    if args.camera_model is not None:
        config = replace(config, camera_model=args.camera_model)
    if args.with_ut:
        config = replace(config, with_ut=True)
    if args.with_eval3d:
        config = replace(config, with_eval3d=True, with_ut=True)
    train_kwargs = {"resume_from": args.resume}
    if args.distributed:
        train_kwargs["distributed"] = True
    result = train(config, **train_kwargs)
    print(f"checkpoint={result.checkpoint}")


def _render_command(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).absolute()
    distributed_checkpoint = is_distributed_checkpoint(checkpoint)
    saved_scene_transform = load_checkpoint_scene_transform(checkpoint)
    if saved_scene_transform is None and distributed_checkpoint:
        raise ValueError(
            "distributed render requires checkpoint scene metadata; "
            "the camera transform cannot be inferred from a shard set"
        )
    config, model, appearance, step = _load_training_objects(checkpoint)
    if args.max_gaussians_per_tile is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer,
                max_gaussians_per_tile=args.max_gaussians_per_tile,
            ),
        )
    if args.max_candidates_per_tile is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer,
                max_candidates_per_tile=args.max_candidates_per_tile,
            ),
        )
    if args.max_intersections is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer, max_intersections=args.max_intersections
            ),
        )
    if args.rasterizer_backend is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer, backend=args.rasterizer_backend
            ),
        )
    if args.compositor_backend is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer,
                compositor_backend=args.compositor_backend,
            ),
        )
    if args.intersection_backend is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer,
                intersection_backend=args.intersection_backend,
            ),
        )
    if args.intersection_mode is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer, intersection_mode=args.intersection_mode
            ),
        )
    if args.sort_backend is not None:
        config = replace(
            config,
            rasterizer=replace(config.rasterizer, sort_backend=args.sort_backend),
        )
    if args.tile_batch_size is not None:
        config = replace(
            config,
            rasterizer=replace(
                config.rasterizer, tile_batch_size=args.tile_batch_size
            ),
        )
    scene = load_colmap_scene(
        args.data or config.data.root,
        image_dir=args.image_dir or config.data.image_dir,
        load_points=False,
    )
    split = create_grain_dataset(
        scene,
        split=args.split,
        test_every=config.data.test_every,
        shuffle=False,
    )
    example = split[args.index]
    if saved_scene_transform is None:
        transform = _legacy_scene_transform(scene)
    else:
        matrix, _ = saved_scene_transform
        transform = SceneTransform(matrix)
    viewmat = transform.world_to_camera(example["w2c"])
    height, width = example["image"].shape[:2]
    _check_evaluation_memory_budget(
        config,
        physical_capacity=model.capacity,
        width=width,
        height=height,
    )
    render_step = make_render_step(config, width, height)
    image, alpha, overflow, intersection_overflow = render_step(
        model,
        jax.device_put(viewmat),
        jax.device_put(example["K"]),
        jnp.asarray(config.model.sh_degree),
        appearance_module=appearance,
    )
    overflow_count = int(jnp.count_nonzero(overflow))
    has_intersection_overflow = bool(intersection_overflow)
    if args.strict_overflow and (overflow_count or has_intersection_overflow):
        raise RuntimeError(
            f"render had {overflow_count} overflowing tiles and "
            f"intersection_overflow={has_intersection_overflow}; raise "
            "--max-gaussians-per-tile or --max-intersections before final evaluation"
        )
    output = Path(args.output)
    _save_render(output, image)
    if args.alpha is not None:
        alpha_rgb = jnp.broadcast_to(alpha, alpha.shape[:-1] + (3,))
        _save_render(Path(args.alpha), alpha_rgb)
    print(
        f"rendered step={step} image={example['image_name']} "
        f"overflow_tiles={overflow_count} "
        f"intersection_overflow={has_intersection_overflow} output={output}"
    )


def _export_command(args: argparse.Namespace) -> None:
    config, model, appearance, step = _load_training_objects(
        Path(args.checkpoint).absolute()
    )
    export_state: GaussianModel | dict[str, jax.Array] = model
    if config.app_opt:
        assert appearance is not None
        sh0, sh_rest = bake_appearance_sh(
            appearance,
            model.features[...],
            model.colors[...],
            sh_degree=config.model.sh_degree,
        )
        export_state = model.state_dict()
        del export_state["features"]
        del export_state["colors"]
        export_state["sh0"] = sh0
        export_state["sh_rest"] = sh_rest
    output = export_splats(export_state, args.output)
    print(f"exported step={step} active={int(model.active_count)} output={output}")


def _inspect_data_command(args: argparse.Namespace) -> None:
    scene = load_colmap_scene(
        args.data, image_dir=args.image_dir, load_points=not args.no_points
    )
    shapes = sorted({(image.height, image.width) for image in scene.images})
    summary = {
        "root": str(scene.root),
        "images": len(scene.images),
        "image_shapes": shapes,
        "points": len(scene.model.points3D),
        "cameras": len(scene.model.cameras),
        "scene_scale": scene.scene_scale,
        "train_images": len(scene.indices("train", args.test_every)),
        "test_images": len(scene.indices("test", args.test_every)),
    }
    print(json.dumps(summary, indent=2))


def _init_config_command(args: argparse.Namespace) -> None:
    config = TrainConfig()
    if args.data is not None:
        config = replace(config, data=replace(config.data, root=args.data))
    config.save(args.output)
    print(args.output)


def _estimate_command(args: argparse.Namespace) -> None:
    config = TrainConfig.load(args.config) if args.config else TrainConfig()
    if args.capacity is not None:
        config = replace(
            config, model=replace(config.model, capacity=args.capacity)
        )
    if args.bucket_min_capacity is not None:
        config = replace(
            config,
            model=replace(
                config.model, bucket_min_capacity=args.bucket_min_capacity
            ),
        )
    active_target = 0 if args.active_target is None else int(args.active_target)
    storage_capacity = config.model.bucket_capacity(active_target)
    estimate = estimate_training_memory_bytes(
        config,
        physical_capacity=storage_capacity,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    maximum_estimate = estimate_training_memory_bytes(
        config,
        physical_capacity=config.model.capacity,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    print(
        json.dumps(
            {
                "logical_max_capacity": config.model.capacity,
                "active_target": active_target,
                "minimum_bucket_capacity": storage_capacity,
                "estimated_storage_capacity": storage_capacity,
                "sh_degree": config.model.sh_degree,
                "estimated_peak_gib": estimate / 2**30,
                "maximum_bucket_peak_gib": maximum_estimate / 2**30,
                "tile_batch_size": config.rasterizer.tile_batch_size,
                "max_gaussians_per_tile": config.rasterizer.max_gaussians_per_tile,
                "max_intersections": config.rasterizer.max_intersections,
                "rasterizer_backend": config.rasterizer.backend,
                "intersection_backend": config.rasterizer.intersection_backend,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jax-gs", description="Flax NNX/JAX Gaussian splatting"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a COLMAP scene")
    train_parser.add_argument("--config", type=Path)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument(
        "--model-type",
        choices=("3dgs", "2dgs"),
        help="training renderer; 2dgs selects gradient_2dgs automatically",
    )
    train_parser.add_argument("--opacity-reg", type=float)
    train_parser.add_argument("--scale-reg", type=float)
    train_parser.add_argument("--app-opt", action="store_true")
    train_parser.add_argument("--app-embed-dim", type=int)
    train_parser.add_argument("--app-opt-lr", type=float)
    train_parser.add_argument("--app-opt-reg", type=float)
    train_parser.add_argument("--pose-opt", action="store_true")
    train_parser.add_argument("--pose-opt-lr", type=float)
    train_parser.add_argument("--pose-opt-reg", type=float)
    train_parser.add_argument("--pose-noise", type=float)
    train_parser.add_argument("--normal-loss", action="store_true")
    train_parser.add_argument("--normal-lambda", type=float)
    train_parser.add_argument("--normal-start-iter", type=int)
    train_parser.add_argument("--dist-loss", action="store_true")
    train_parser.add_argument("--dist-lambda", type=float)
    train_parser.add_argument("--dist-start-iter", type=int)
    train_parser.add_argument(
        "--packed",
        action="store_true",
        help="use fixed-capacity packed projection metadata during training",
    )
    train_parser.add_argument(
        "--sparse-grad",
        action="store_true",
        help="update only packed visible Gaussian rows; requires --packed",
    )
    train_parser.add_argument(
        "--visible-adam",
        action="store_true",
        help="update Adam state and parameters only for visible Gaussian rows",
    )
    train_parser.add_argument("--data", type=str)
    train_parser.add_argument("--image-dir", type=str)
    train_parser.add_argument("--global-scale", type=float)
    train_parser.add_argument(
        "--normalize-world-space",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    train_parser.add_argument(
        "--capacity",
        type=int,
        help="logical maximum Gaussian count; not fully preallocated",
    )
    train_parser.add_argument(
        "--bucket-min-capacity",
        type=int,
        help="minimum physical bucket; buckets double up to --capacity",
    )
    train_parser.add_argument("--patch-size", type=int)
    train_parser.add_argument(
        "--num-workers",
        type=int,
        help=(
            "Grain/KD-tree worker count (default: 4); values above available "
            "CPU affinity emit a warning"
        ),
    )
    train_parser.add_argument("--max-gaussians-per-tile", type=int)
    train_parser.add_argument(
        "--max-candidates-per-tile",
        type=int,
        help=(
            "static promise about the busiest tile's candidate count, "
            "which sets the compositor chunk loop length. The training "
            "metric busiest_tile_candidates reports what a run actually "
            "needs. Training grows a too-small bound and replays the "
            "uncommitted step"
        ),
    )
    train_parser.add_argument("--max-intersections", type=int)
    train_parser.add_argument("--intersection-bucket-min-capacity", type=int)
    train_parser.add_argument(
        "--rasterizer-backend",
        choices=("auto", "jax", "intersections", "reference"),
    )
    train_parser.add_argument(
        "--compositor-backend",
        choices=("jax", "pallas", "cuda_ffi"),
        help=(
            "compositor used for forward and reverse mode; Pallas requires "
            "Hopper-or-newer, CUDA FFI requires an NVIDIA GPU and nvcc or "
            "a prebuilt JAX_GS_CUDA_FFI_LIBRARY"
        ),
    )
    train_parser.add_argument(
        "--intersection-backend",
        choices=("auto", "jax", "pallas"),
        help="Pallas accelerates the AccuTile count and pair-emission scans",
    )
    train_parser.add_argument(
        "--intersection-mode",
        choices=("auto", "aabb", "accutile"),
    )
    train_parser.add_argument(
        "--sort-backend",
        choices=("auto", "jax"),
    )
    train_parser.add_argument("--tile-batch-size", type=int)
    train_parser.add_argument("--steps", type=int)
    train_parser.add_argument("--output", type=str)
    train_parser.add_argument("--strategy", choices=("default", "mcmc"))
    train_parser.add_argument("--max-new-per-refine", type=int)
    train_parser.add_argument("--refine-every", type=int)
    train_parser.add_argument("--refine-stop", type=int)
    train_parser.add_argument(
        "--camera-model", choices=("pinhole", "ortho", "fisheye", "ftheta")
    )
    train_parser.add_argument("--with-ut", action="store_true")
    train_parser.add_argument("--with-eval3d", action="store_true")
    train_parser.add_argument(
        "--distributed",
        action="store_true",
        help=(
            "shard Gaussians across all local devices with nnx.pmap; "
            "currently requires one JAX process and at least two devices"
        ),
    )
    train_parser.set_defaults(func=_train_command)

    render_parser = subparsers.add_parser("render", help="render a checkpoint")
    render_parser.add_argument("checkpoint", type=str)
    render_parser.add_argument("--data", type=str)
    render_parser.add_argument("--image-dir", type=str)
    render_parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    render_parser.add_argument("--index", type=int, default=0)
    render_parser.add_argument("--output", type=str, default="render.png")
    render_parser.add_argument("--alpha", type=str)
    render_parser.add_argument("--max-gaussians-per-tile", type=int)
    render_parser.add_argument("--max-candidates-per-tile", type=int)
    render_parser.add_argument("--max-intersections", type=int)
    render_parser.add_argument(
        "--rasterizer-backend",
        choices=("auto", "jax", "intersections", "reference"),
    )
    render_parser.add_argument(
        "--compositor-backend",
        choices=("jax", "pallas", "cuda_ffi"),
        help=(
            "experimental compositor; Pallas requires Hopper-or-newer, "
            "CUDA FFI requires an NVIDIA GPU and nvcc or a prebuilt library"
        ),
    )
    render_parser.add_argument(
        "--intersection-backend",
        choices=("auto", "jax", "pallas"),
        help="Pallas accelerates the AccuTile count and pair-emission scans",
    )
    render_parser.add_argument(
        "--intersection-mode",
        choices=("auto", "aabb", "accutile"),
    )
    render_parser.add_argument(
        "--sort-backend",
        choices=("auto", "jax"),
    )
    render_parser.add_argument("--tile-batch-size", type=int)
    render_parser.add_argument("--strict-overflow", action="store_true")
    render_parser.set_defaults(func=_render_command)

    export_parser = subparsers.add_parser("export", help="export PLY or .splat")
    export_parser.add_argument("checkpoint", type=str)
    export_parser.add_argument("output", type=str)
    export_parser.set_defaults(func=_export_command)

    inspect_parser = subparsers.add_parser(
        "inspect-data", help="inspect a COLMAP dataset"
    )
    inspect_parser.add_argument("data", type=str)
    inspect_parser.add_argument("--image-dir", default="images_8")
    inspect_parser.add_argument("--test-every", type=int, default=8)
    inspect_parser.add_argument("--no-points", action="store_true")
    inspect_parser.set_defaults(func=_inspect_data_command)

    config_parser = subparsers.add_parser(
        "init-config", help="write the default JSON config"
    )
    config_parser.add_argument("--output", default="jax_gs.json")
    config_parser.add_argument("--data", type=str)
    config_parser.set_defaults(func=_init_config_command)

    estimate_parser = subparsers.add_parser(
        "estimate-memory", help="estimate peak training memory without allocating a model"
    )
    estimate_parser.add_argument("--config", type=Path)
    estimate_parser.add_argument(
        "--capacity",
        type=int,
        help="logical maximum Gaussian count; not fully preallocated",
    )
    estimate_parser.add_argument(
        "--bucket-min-capacity",
        type=int,
        help="minimum physical bucket; buckets double up to --capacity",
    )
    estimate_parser.add_argument(
        "--active-target",
        type=int,
        help="estimate the physical bucket required for this active count",
    )
    estimate_parser.add_argument(
        "--image-height",
        type=int,
        help="full-image training height; required when patch_size is null",
    )
    estimate_parser.add_argument(
        "--image-width",
        type=int,
        help="full-image training width; required when patch_size is null",
    )
    estimate_parser.set_defaults(func=_estimate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
