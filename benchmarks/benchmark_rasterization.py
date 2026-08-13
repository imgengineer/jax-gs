#!/usr/bin/env python3
"""Reproducible, memory-conscious benchmark for the JAX rasterizer.

The default invocation uses a small synthetic scene and only benchmarks the
forward pass.  Backward benchmarking is opt-in and guarded by conservative
limits because compiling a large reverse-mode rasterizer can exhaust device or
host memory.  In this script, ``capacity`` always denotes the physical Gaussian
array length, not the logical maximum accepted by the training CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable

# This must be set before importing JAX.  Assignment (rather than setdefault)
# makes the benchmark safe even when a shell profile enables preallocation.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "4")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_force_compilation_parallelism=1",
)

import jax
import jax.numpy as jnp
import numpy as np

from jax_gs.config import RasterizationConfig
from jax_gs.rasterization import rasterization


@dataclass(frozen=True)
class Scene:
    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray
    active_mask: np.ndarray
    viewmats: np.ndarray
    Ks: np.ndarray
    source: str


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _resolution(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use WIDTHxHEIGHT, for example 320x180") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def _parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI; capacity denotes the physical input shape."""

    parser = argparse.ArgumentParser(
        description=(
            "Measure cold/hot JAX rasterization latency with a statically shaped "
            "physical Gaussian buffer and active_mask. Synthetic data is used "
            "unless --npz is given."
        )
    )
    parser.add_argument(
        "--npz",
        type=Path,
        help=(
            "garden or other scene NPZ containing means, colors, viewmats and Ks; "
            "missing quats/scales/opacities are generated deterministically"
        ),
    )
    parser.add_argument(
        "--capacity",
        type=_positive_int,
        default=10_000,
        help=(
            "physical Gaussian array length used by the benchmark; unlike "
            "jax-gs train --capacity, this is not a logical maximum"
        ),
    )
    parser.add_argument(
        "--active",
        type=_positive_int,
        default=10_000,
        help="number of active entries inside the physical Gaussian array",
    )
    parser.add_argument("--resolution", type=_resolution, default=(320, 180))
    parser.add_argument(
        "--k",
        "--max-gaussians-per-tile",
        dest="k",
        type=_positive_int,
        default=128,
        help="pure-JAX per-tile compositing chunk size",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "jax", "intersections", "reference"),
        default="auto",
    )
    parser.add_argument(
        "--compositor-backend",
        choices=("jax", "pallas", "cuda_ffi"),
        default="jax",
        help="forward/reverse compositor implementation",
    )
    parser.add_argument(
        "--intersection-backend",
        choices=("auto", "jax", "pallas", "cuda_tile", "cuda_tile_cub"),
        default="auto",
        help=(
            "intersection backend; cuda_tile_cub additionally uses CUDA "
            "FFI+CUB for prefix, sorting, and offsets"
        ),
    )
    parser.add_argument(
        "--intersection-mode",
        choices=("auto", "aabb", "accutile"),
        default="auto",
        help="AABB or opacity-aware AccuTile geometry",
    )
    parser.add_argument(
        "--sort-backend",
        choices=("auto", "jax"),
        default="auto",
        help="fixed-capacity intersection sort backend",
    )
    parser.add_argument(
        "--max-intersections",
        type=_positive_int,
        help="static global intersection capacity; default uses the renderer heuristic",
    )
    parser.add_argument(
        "--max-candidates-per-tile",
        type=_positive_int,
        help=(
            "static promise about the busiest tile's candidate count; sets the "
            "compositor chunk loop length. Too small only sets tile_overflow"
        ),
    )
    parser.add_argument(
        "--tile-batch",
        type=_positive_int,
        default=1,
        help="number of tiles evaluated together by lax.map",
    )
    parser.add_argument("--tile-size", type=_positive_int, default=16)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--hot-iters", type=_positive_int, default=10)
    parser.add_argument("--warmup-iters", type=_nonnegative_int, default=5)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="write warm JAX/Perfetto traces under forward/ and backward/",
    )
    parser.add_argument(
        "--profile-iters",
        type=_positive_int,
        default=3,
        help="number of already-warm iterations captured per trace",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--gsplat-v153-garden-profile",
        action="store_true",
        help=(
            "match gsplat v1.5.3's 10k garden shape/crop setup while using "
            "deterministic NumPy-generated Gaussian attributes"
        ),
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="also time value_and_grad; disabled by default and safety-limited",
    )
    parser.add_argument("--backward-iters", type=_positive_int, default=3)
    parser.add_argument(
        "--allow-unsafe",
        action="store_true",
        help="override memory estimate and backward safety limits",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="also write the complete result to this JSON file",
    )
    return parser


def _pick(data: np.lib.npyio.NpzFile, *names: str) -> tuple[np.ndarray, str] | None:
    for name in names:
        if name in data:
            return np.asarray(data[name]), name
    return None


def _required(
    data: np.lib.npyio.NpzFile, label: str, *names: str
) -> tuple[np.ndarray, str]:
    result = _pick(data, *names)
    if result is None:
        choices = ", ".join(names)
        available = ", ".join(data.files)
        raise ValueError(
            f"NPZ is missing {label}; expected one of [{choices}]. "
            f"Available keys: [{available}]"
        )
    return result


def _camera_array(array: np.ndarray, shape: tuple[int, int], index: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.shape == shape:
        return array
    if array.ndim == 3 and array.shape[1:] == shape:
        return array[index % array.shape[0]]
    raise ValueError(f"camera array must have shape {shape} or [C, {shape[0]}, {shape[1]}]")


def _source_resolution(
    data: np.lib.npyio.NpzFile, K: np.ndarray
) -> tuple[float, float]:
    width = _pick(data, "width", "image_width", "W")
    height = _pick(data, "height", "image_height", "H")
    if width is not None and height is not None:
        return float(np.asarray(width[0]).reshape(-1)[0]), float(
            np.asarray(height[0]).reshape(-1)[0]
        )
    # Principal points in COLMAP-style intrinsics are normally at image center.
    return max(2.0 * float(K[0, 2]), 1.0), max(2.0 * float(K[1, 2]), 1.0)


def _pad_scene(
    means: np.ndarray,
    quats: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
    indices: np.ndarray,
    *,
    capacity: int,
    viewmat: np.ndarray,
    K: np.ndarray,
    source: str,
) -> Scene:
    active = len(indices)
    output_means = np.zeros((capacity, 3), dtype=np.float32)
    output_quats = np.zeros((capacity, 4), dtype=np.float32)
    output_quats[:, 0] = 1.0
    output_scales = np.full((capacity, 3), 0.01, dtype=np.float32)
    output_opacities = np.zeros((capacity,), dtype=np.float32)
    output_colors = np.zeros((capacity, 3), dtype=np.float32)
    active_mask = np.zeros((capacity,), dtype=np.bool_)

    output_means[:active] = means[indices]
    output_quats[:active] = quats[indices]
    output_scales[:active] = scales[indices]
    output_opacities[:active] = opacities[indices]
    output_colors[:active] = colors[indices]
    active_mask[:active] = True
    return Scene(
        output_means,
        output_quats,
        output_scales,
        output_opacities,
        output_colors,
        active_mask,
        viewmat[None],
        K[None],
        source,
    )


def _load_npz(
    path: Path,
    *,
    capacity: int,
    active: int,
    width: int,
    height: int,
    camera_index: int,
    seed: int,
    gsplat_v153_garden_profile: bool = False,
) -> Scene:
    with np.load(path, allow_pickle=False) as data:
        means, _ = _required(data, "means", "means", "means3d", "xyz")
        quat_result = _pick(data, "quats", "rotations")
        quats = None if quat_result is None else quat_result[0]

        scale_result = _pick(data, "scales")
        log_scale_result = _pick(data, "log_scales")
        if scale_result is None and log_scale_result is not None:
            log_scales = log_scale_result[0]
            scales = np.exp(log_scales)
        elif scale_result is not None:
            scales = scale_result[0]
        else:
            scales = None

        opacity_result = _pick(data, "opacities", "opacity")
        opacity_logit_result = _pick(data, "opacity_logits")
        if opacity_result is None and opacity_logit_result is not None:
            opacity_logits = opacity_logit_result[0]
            opacities = 1.0 / (1.0 + np.exp(-opacity_logits))
        elif opacity_result is not None:
            opacities = opacity_result[0]
        else:
            opacities = None

        colors, color_key = _required(
            data, "colors", "colors", "rgb", "sh_coeffs", "sh0", "features_dc"
        )
        viewmats, _ = _required(
            data, "view matrices", "viewmats", "viewmat", "world_to_camera"
        )
        Ks, _ = _required(data, "intrinsics", "Ks", "K", "intrinsics")
        viewmat = _camera_array(viewmats, (4, 4), camera_index)
        K = _camera_array(Ks, (3, 3), camera_index)
        source_width, source_height = _source_resolution(data, K)

        source_mask_result = _pick(data, "active_mask")
        source_mask = None if source_mask_result is None else source_mask_result[0]

    means = np.asarray(means, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors)
    if gsplat_v153_garden_profile:
        crop = np.all((means >= -2.0) & (means <= 2.0), axis=-1)
        means = means[crop]
        colors = colors[crop]
        quats = None
        scales = None
        opacities = None
        source_mask = None
    attribute_rng = np.random.default_rng(seed)
    if gsplat_v153_garden_profile:
        scales = attribute_rng.uniform(1.0e-4, 0.02, (means.shape[0], 3)).astype(
            np.float32
        )
        quats = attribute_rng.normal(size=(means.shape[0], 4)).astype(np.float32)
        opacities = attribute_rng.random(means.shape[0]).astype(np.float32)
    else:
        if quats is None:
            quats = attribute_rng.normal(size=(means.shape[0], 4)).astype(np.float32)
        if scales is None:
            scales = (
                attribute_rng.random((means.shape[0], 3), dtype=np.float32) * 0.02
            )
        if opacities is None:
            opacities = attribute_rng.random(means.shape[0], dtype=np.float32)
    quats = np.asarray(quats, dtype=np.float32).reshape(-1, 4)
    scales = np.asarray(scales, dtype=np.float32).reshape(-1, 3)
    opacities = np.asarray(opacities, dtype=np.float32).reshape(-1)
    colors = np.asarray(colors, dtype=np.float32)
    sh_colors = color_key in {"sh_coeffs", "sh0", "features_dc"}
    if colors.ndim == 3 and colors.shape[0] == means.shape[0]:
        # Model NPZ files commonly store SH coefficients. Convert the DC term
        # to direct RGB so the benchmark measures the same RGB raster path for
        # both synthetic and loaded scenes.
        colors = colors[:, 0, :]
        sh_colors = True
    colors = colors.reshape(-1, 3)
    if sh_colors:
        colors = colors * 0.28209479177387814 + 0.5
    if np.issubdtype(colors.dtype, np.integer) or float(colors.max(initial=0.0)) > 1.5:
        colors = colors / 255.0

    count = means.shape[0]
    for label, array in (
        ("quats", quats),
        ("scales", scales),
        ("opacities", opacities),
        ("colors", colors),
    ):
        if array.shape[0] != count:
            raise ValueError(f"{label} has {array.shape[0]} rows, expected {count}")

    quat_norm = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = np.divide(
        quats,
        quat_norm,
        out=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (count, 1)),
        where=quat_norm > 1.0e-12,
    )
    scales = np.maximum(scales, 1.0e-7)
    opacities = np.clip(opacities, 0.0, 0.999)
    colors = np.clip(colors, 0.0, 1.0)

    candidates = np.arange(count)
    if source_mask is not None:
        source_mask = np.asarray(source_mask, dtype=np.bool_).reshape(-1)
        if source_mask.shape[0] != count:
            raise ValueError("active_mask length does not match means")
        candidates = candidates[source_mask]
    if active > candidates.size:
        raise ValueError(
            f"requested {active} active Gaussians, but NPZ contains only "
            f"{candidates.size} usable rows"
        )
    if gsplat_v153_garden_profile:
        indices = candidates[:active]
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(candidates, active, replace=False))

    K = K.copy()
    K[0, :] *= width / source_width
    K[1, :] *= height / source_height
    return _pad_scene(
        means,
        quats,
        scales,
        opacities,
        colors,
        indices,
        capacity=capacity,
        viewmat=viewmat,
        K=K,
        source=(
            f"{path.resolve()} (gsplat-v1.5.3-garden-profile)"
            if gsplat_v153_garden_profile
            else str(path.resolve())
        ),
    )


def _synthetic_scene(
    *, capacity: int, active: int, width: int, height: int, seed: int
) -> Scene:
    rng = np.random.default_rng(seed)
    z = rng.uniform(2.0, 6.0, size=active).astype(np.float32)
    fx = 0.8 * width
    fy = 0.8 * width
    pixel_x = rng.uniform(0.05 * width, 0.95 * width, size=active)
    pixel_y = rng.uniform(0.05 * height, 0.95 * height, size=active)
    means = np.stack(
        ((pixel_x - width / 2) * z / fx, (pixel_y - height / 2) * z / fy, z),
        axis=-1,
    ).astype(np.float32)
    quats = np.zeros((active, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    base_scale = rng.lognormal(mean=math.log(0.025), sigma=0.35, size=(active, 1))
    anisotropy = rng.uniform(0.7, 1.3, size=(active, 3))
    scales = (base_scale * anisotropy).astype(np.float32)
    opacities = rng.uniform(0.1, 0.9, size=active).astype(np.float32)
    colors = rng.uniform(0.0, 1.0, size=(active, 3)).astype(np.float32)
    K = np.array(
        [[fx, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    indices = np.arange(active)
    return _pad_scene(
        means,
        quats,
        scales,
        opacities,
        colors,
        indices,
        capacity=capacity,
        viewmat=np.eye(4, dtype=np.float32),
        K=K,
        source=f"synthetic(seed={seed})",
    )


def _estimated_peak_bytes(
    *,
    capacity: int,
    width: int,
    height: int,
    tile_size: int,
    k: int,
    tile_batch: int,
    backend: str,
    compositor_backend: str,
    max_intersections: int | None,
    backward: bool,
) -> int:
    # Inputs + projection buffers + per-tile score/top-k storage + compositing
    # workspace. The multiplier for reverse mode intentionally overestimates a
    # small benchmark; it is a guard, not a profiler measurement.
    inputs = capacity * 57
    projection = capacity * 64
    tile_scan = tile_batch * capacity * 16 if backend == "reference" else 0
    intersection_workspace = 0
    intersection_capacity = 0
    tile_count = math.ceil(width / tile_size) * math.ceil(height / tile_size)
    if backend != "reference":
        intersection_capacity = max_intersections
        if intersection_capacity is None:
            intersection_capacity = max(
                1,
                min(tile_count * (k + 1), max(k + 1, capacity * 8)),
            )
        intersection_workspace = intersection_capacity * 96
    if compositor_backend == "pallas":
        pixel_count = math.ceil(tile_size**2 / 128) * 128
        # RGB plus alpha, pre-update transmittance, last accepted slot, and
        # accepted transmittance.
        compositing = tile_count * pixel_count * 7 * 4
    elif compositor_backend == "cuda_ffi":
        # Image outputs plus final-transmittance/last-id residuals; Gaussian
        # gradients are allocated directly by the backward FFI call.
        compositing = width * height * 8
    else:
        compositing = tile_batch * k * tile_size * tile_size * 32
    outputs = width * height * 20
    forward = (
        inputs
        + projection
        + tile_scan
        + intersection_workspace
        + compositing
        + outputs
    )
    pallas_backward = (
        intersection_capacity * 9 * 4
        if backward and compositor_backend == "pallas"
        else 0
    )
    return forward * (5 if backward else 2) + pallas_backward


def _memory_stats(device: jax.Device) -> dict[str, int]:
    stats = device.memory_stats() or {}
    wanted = (
        "bytes_in_use",
        "peak_bytes_in_use",
        "bytes_limit",
        "pool_bytes",
        "largest_free_block_bytes",
    )
    return {key: int(stats[key]) for key in wanted if key in stats}


def _peak_rss_bytes() -> int:
    # Linux reports KiB. This project and its CUDA13 environment target Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _block(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


def _time_once(function: Callable[..., Any], arguments: tuple[Any, ...]) -> tuple[Any, float]:
    started = time.perf_counter()
    output = _block(function(*arguments))
    return output, time.perf_counter() - started


def _profile_warm_iterations(
    function: Callable[..., Any],
    arguments: tuple[Any, ...],
    directory: Path,
    *,
    label: str,
    iterations: int,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(
        directory,
        create_perfetto_link=False,
        create_perfetto_trace=True,
    ):
        for step in range(iterations):
            with jax.profiler.StepTraceAnnotation(label, step_num=step):
                _block(function(*arguments))
    return directory.resolve()


def _latency_summary(seconds: list[float]) -> dict[str, Any]:
    milliseconds = np.asarray(seconds, dtype=np.float64) * 1_000.0
    return {
        "iterations": len(seconds),
        "milliseconds": [round(float(value), 6) for value in milliseconds],
        "mean_ms": float(milliseconds.mean()),
        "median_ms": float(np.median(milliseconds)),
        "min_ms": float(milliseconds.min()),
        "p90_ms": float(np.percentile(milliseconds, 90)),
        "fps_from_mean": float(1_000.0 / milliseconds.mean()),
    }


def _validate_safety(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    width, height = args.resolution
    if args.active > args.capacity:
        parser.error("--active cannot exceed --capacity")
    if args.radius_clip < 0.0:
        parser.error("--radius-clip cannot be negative")
    if args.gsplat_v153_garden_profile:
        if args.npz is None:
            parser.error("--gsplat-v153-garden-profile requires --npz")
        if args.capacity != args.active:
            parser.error(
                "--gsplat-v153-garden-profile requires --capacity == --active"
            )
        if (width, height) != (640, 360):
            parser.error(
                "--gsplat-v153-garden-profile requires --resolution 640x360"
            )
        if not math.isclose(args.radius_clip, 3.0):
            parser.error(
                "--gsplat-v153-garden-profile requires --radius-clip 3"
            )
    if args.backward and not args.allow_unsafe:
        violations = []
        if width * height > 128 * 128:
            violations.append("resolution exceeds 128x128")
        if args.capacity > 20_000:
            violations.append("capacity exceeds 20,000")
        if args.k > 128:
            violations.append("K exceeds 128")
        if args.tile_batch > 1:
            violations.append("tile batch exceeds 1")
        if violations:
            parser.error(
                "safe --backward limits violated: "
                + ", ".join(violations)
                + ". Lower the settings or explicitly pass --allow-unsafe."
            )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_safety(args, parser)
    width, height = args.resolution

    device = jax.devices()[0]
    estimate = _estimated_peak_bytes(
        capacity=args.capacity,
        width=width,
        height=height,
        tile_size=args.tile_size,
        k=args.k,
        tile_batch=args.tile_batch,
        backend=args.backend,
        compositor_backend=args.compositor_backend,
        max_intersections=args.max_intersections,
        backward=args.backward,
    )
    initial_memory = _memory_stats(device)
    memory_limit = initial_memory.get("bytes_limit")
    if not args.allow_unsafe and estimate > 8 * 2**30:
        parser.error(
            f"estimated working set {estimate / 2**30:.2f} GiB exceeds the "
            "benchmark's 8 GiB absolute safety limit; lower the settings or "
            "explicitly pass --allow-unsafe"
        )
    if (
        not args.allow_unsafe
        and memory_limit is not None
        and estimate > int(memory_limit * 0.6)
    ):
        parser.error(
            f"estimated working set {estimate / 2**30:.2f} GiB exceeds 60% "
            f"of the device limit {memory_limit / 2**30:.2f} GiB; lower the "
            "settings or explicitly pass --allow-unsafe"
        )

    if args.npz is None:
        scene = _synthetic_scene(
            capacity=args.capacity,
            active=args.active,
            width=width,
            height=height,
            seed=args.seed,
        )
    else:
        scene = _load_npz(
            args.npz,
            capacity=args.capacity,
            active=args.active,
            width=width,
            height=height,
            camera_index=args.camera_index,
            seed=args.seed,
            gsplat_v153_garden_profile=args.gsplat_v153_garden_profile,
        )

    arrays = tuple(
        jax.device_put(value, device)
        for value in (
            scene.means,
            scene.quats,
            scene.scales,
            scene.opacities,
            scene.colors,
            scene.active_mask,
            scene.viewmats,
            scene.Ks,
        )
    )
    _block(arrays)
    config = RasterizationConfig(
        backend=args.backend,
        compositor_backend=args.compositor_backend,
        intersection_backend=args.intersection_backend,
        intersection_mode=args.intersection_mode,
        sort_backend=args.sort_backend,
        tile_size=args.tile_size,
        max_gaussians_per_tile=args.k,
        max_intersections=args.max_intersections,
        max_candidates_per_tile=args.max_candidates_per_tile,
        tile_batch_size=args.tile_batch,
        radius_clip=args.radius_clip,
    )

    def forward(*values: jax.Array):
        means, quats, scales, opacities, colors, mask, viewmats, Ks = values
        renders, alphas, info = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
            active_mask=mask,
            config=config,
        )
        return (
            renders,
            alphas,
            jnp.count_nonzero(info["tile_overflow"]),
            jnp.max(info["candidate_counts"]),
            jnp.any(info["intersection_overflow"]),
            jnp.sum(info["intersection_count"]),
            jnp.max(info["intersection_required_count"]),
            jnp.max(info["intersection_capacity"]),
            jnp.count_nonzero(info["candidate_limit_exceeded"]),
            jnp.max(info["visible_count"]),
            jnp.max(info["visible_capacity"]),
            jnp.any(info["visible_overflow"]),
        )

    jitted_forward = jax.jit(forward)
    forward_output, cold_seconds = _time_once(jitted_forward, arrays)
    for _ in range(args.warmup_iters):
        forward_output = _block(jitted_forward(*arrays))
    hot_seconds = []
    for _ in range(args.hot_iters):
        forward_output, elapsed = _time_once(jitted_forward, arrays)
        hot_seconds.append(elapsed)
    profile_paths: dict[str, str] = {}
    if args.profile_dir is not None:
        profile_paths["forward"] = str(
            _profile_warm_iterations(
                jitted_forward,
                arrays,
                args.profile_dir / "forward",
                label="raster_forward",
                iterations=args.profile_iters,
            )
        )

    overflow_tiles = int(jax.device_get(forward_output[2]))
    max_candidates = int(jax.device_get(forward_output[3]))
    intersection_overflow = bool(jax.device_get(forward_output[4]))
    intersection_count = int(jax.device_get(forward_output[5]))
    intersection_required_count = int(jax.device_get(forward_output[6]))
    intersection_capacity = int(jax.device_get(forward_output[7]))
    candidate_limit_exceeded = int(jax.device_get(forward_output[8]))
    visible_count = int(jax.device_get(forward_output[9]))
    visible_capacity = int(jax.device_get(forward_output[10]))
    visible_overflow = bool(jax.device_get(forward_output[11]))
    tile_count = math.ceil(width / args.tile_size) * math.ceil(height / args.tile_size)
    result: dict[str, Any] = {
        "backend": jax.default_backend(),
        "device": str(device),
        "jax_version": jax.__version__,
        "preallocate": os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"],
        "scene": scene.source,
        "timing": {
            "cold_includes_jit_compile": True,
            "inputs_preloaded_on_device": True,
            "calls_explicitly_synchronized": True,
            "warmup_iterations_after_compile": args.warmup_iters,
        },
        "config": {
            "capacity": args.capacity,
            "active": args.active,
            "width": width,
            "height": height,
            "tile_size": args.tile_size,
            "max_gaussians_per_tile": args.k,
            "rasterizer_backend": args.backend,
            "compositor_backend": args.compositor_backend,
            "intersection_backend": args.intersection_backend,
            "intersection_mode": args.intersection_mode,
            "sort_backend": args.sort_backend,
            "max_intersections": args.max_intersections,
            "max_candidates_per_tile": args.max_candidates_per_tile,
            "tile_batch": args.tile_batch,
            "radius_clip": args.radius_clip,
            "seed": args.seed,
            "gsplat_v153_garden_profile": args.gsplat_v153_garden_profile,
        },
        "estimated_peak_bytes": estimate,
        "forward": {
            "cold_ms": cold_seconds * 1_000.0,
            "hot": _latency_summary(hot_seconds),
            "render_shape": list(forward_output[0].shape),
        },
        "tiles": {
            "count": tile_count,
            "overflow_count": overflow_tiles,
            "max_candidates": max_candidates,
            "candidate_limit": args.k,
            "candidate_limit_exceeded": candidate_limit_exceeded,
        },
        "intersections": {
            "count": intersection_count,
            "required_count": intersection_required_count,
            "capacity": intersection_capacity,
            "fill_ratio": intersection_required_count / intersection_capacity,
            "overflow": intersection_overflow,
        },
        "visible": {
            "count": visible_count,
            "capacity": visible_capacity,
            "overflow": visible_overflow,
        },
        "memory": {
            "device_before": initial_memory,
            "device_after_forward": _memory_stats(device),
            "process_peak_rss_bytes": _peak_rss_bytes(),
        },
    }

    if args.backward:
        def loss_function(
            means: jax.Array,
            quats: jax.Array,
            scales: jax.Array,
            opacities: jax.Array,
            colors: jax.Array,
            mask: jax.Array,
            viewmats: jax.Array,
            Ks: jax.Array,
        ):
            renders, alphas, info = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats,
                Ks,
                width,
                height,
                active_mask=mask,
                config=config,
            )
            loss = jnp.mean(renders) + 0.01 * jnp.mean(alphas)
            auxiliary = (
                jnp.count_nonzero(info["tile_overflow"]),
                jnp.max(info["candidate_counts"]),
            )
            return loss, auxiliary

        backward = jax.jit(
            jax.value_and_grad(
                loss_function,
                argnums=(0, 1, 2, 3, 4),
                has_aux=True,
            )
        )
        backward_output, backward_cold = _time_once(backward, arrays)
        for _ in range(args.warmup_iters):
            backward_output = _block(backward(*arrays))
        backward_hot = []
        for _ in range(args.backward_iters):
            backward_output, elapsed = _time_once(backward, arrays)
            backward_hot.append(elapsed)
        loss_value = float(jax.device_get(backward_output[0][0]))
        result["backward"] = {
            "cold_ms": backward_cold * 1_000.0,
            "hot": _latency_summary(backward_hot),
            "loss": loss_value,
        }
        result["memory"]["device_after_backward"] = _memory_stats(device)
        result["memory"]["process_peak_rss_bytes"] = _peak_rss_bytes()
        if args.profile_dir is not None:
            profile_paths["backward"] = str(
                _profile_warm_iterations(
                    backward,
                    arrays,
                    args.profile_dir / "backward",
                    label="raster_value_and_grad",
                    iterations=args.profile_iters,
                )
            )

    if profile_paths:
        result["profiles"] = profile_paths

    if overflow_tiles or intersection_overflow:
        result["warning"] = (
            f"{overflow_tiles}/{tile_count} tiles exceeded K={args.k}; "
            f"intersection_overflow={intersection_overflow}. Increase --k or "
            "--max-intersections before using the latency as a quality-equivalent result"
        )
        print(result["warning"], file=sys.stderr)

    rendered_json = json.dumps(result, indent=2, sort_keys=True)
    print(rendered_json)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered_json + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
