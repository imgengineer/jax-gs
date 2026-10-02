"""Checkpoint and JSON report output for production training."""

import json
from dataclasses import asdict
from pathlib import Path

import jax

from ..config import TrainingConfig
from ..scene.point import GaussianArrays
from .checkpoint import save_gaussians


def write_training_report(
    output: str | Path,
    pool: GaussianArrays,
    settings: TrainingConfig,
    metrics: dict[str, object],
) -> dict[str, object]:
    """Save the final pool and preserve the training report schema."""
    output = Path(output)
    save_gaussians(output, pool)
    capacity_config = settings.capacity
    report = {
        "scene": metrics["scene"],
        "gpu": jax.devices()[0].device_kind,
        "images": settings.model.images,
        "image_shape": metrics["image_shape"],
        "training_images": metrics["training_images"],
        "actual_updates": metrics["actual_updates"],
        "initial_sparse_gaussians": metrics["initial_sparse_gaussians"],
        "initial_padded_gaussians": metrics["initial_padded_gaussians"],
        "target_gaussians": settings.densify.target_primitives,
        "final_gaussians": int(pool.n_active),
        "pair_capacity": settings.runtime.max_visibility_pairs,
        "cluster_size": capacity_config.cluster_size,
        "tile_size": capacity_config.tile_size,
        "tile_height": capacity_config.raster_tile_height,
        "warmup_seconds": metrics["warmup_seconds"],
        "image_load_seconds": metrics["image_load_seconds"],
        "data_loader": "grain.MapDataset, 4 decode threads, 8 prefetched images; GPU uint8 preload",
        "training_seconds": metrics["training_seconds"],
        "densify_until": metrics["densify_until"],
        "seed": settings.runtime.seed,
        "history": metrics["history"],
        "config": asdict(settings),
        "model": "flax.nnx.Module",
        "jit": "nnx.jit_partial(graph=False)",
        "optimizer": settings.runtime.optimizer,
        "jit_cache_size": metrics["jit_cache_size"],
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
    if report_path == output:
        report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"training: {metrics['training_seconds']:.3f}s; report: {report_path}", flush=True)
    return report
