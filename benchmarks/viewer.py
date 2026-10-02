"""Compare eager and JIT viewer latency for identical fixed models and camera paths."""

import argparse
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import viser
import viser.transforms as tf
from viser._scene_api import _encode_image_binary

from jaxgs.io_manager.checkpoint import load_gaussians
from jaxgs.render.view import RESOLUTIONS, ViewRenderer, resize_camera
from jaxgs.viewer import initial_view, view_camera


def _measure(call, cameras, steps):
    samples = []
    for index in range(steps):
        start = perf_counter()
        result = jax.block_until_ready(call(cameras[index % len(cameras)]))
        samples.append((perf_counter() - start) * 1000)
        if bool(result[1]):
            raise RuntimeError("visibility overflow during measurement")
    return {
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--images", default="images")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-visibility-pairs", type=int, default=8_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    snapshots = []
    stopped = threading.Event()

    def monitor():
        while not stopped.is_set():
            value = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            snapshots.append(
                [
                    line.strip()
                    for line in value.splitlines()
                    if line.strip() and int(line.split(",", 1)[0]) != os.getpid()
                ]
            )
            stopped.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        pool = load_gaussians(args.model)
        renderer = ViewRenderer(pool, pair_capacity=args.max_visibility_pairs)
        position, look_at, up, fov = initial_view(pool, args.scene, args.images)
        forward = look_at - position
        distance = np.linalg.norm(forward)
        forward = forward / distance
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        rotation = tf.SO3.from_matrix(np.stack((right, np.cross(forward, right), forward), axis=1))
        camera = view_camera(rotation.wxyz, position, fov, 16 / 9, 1280, 720)
        warmup = renderer.warmup(camera)
        rows = []
        for name, (width, height) in RESOLUTIONS.items():
            poses = [
                view_camera(
                    rotation.wxyz,
                    position + right * (0.01 * distance * np.sin(index)),
                    fov,
                    16 / 9,
                    width,
                    height,
                )
                for index in range(8)
            ]
            fixed = resize_camera(camera, width, height)
            parity = []
            for pose in poses:
                eager = jax.block_until_ready(renderer.eager(pose))
                compiled = jax.block_until_ready(renderer(pose))
                if bool(eager[1]) or bool(compiled[1]):
                    raise RuntimeError("visibility overflow during warmup")
                # Packed half-precision rendering can cross a uint8 rounding boundary.
                np.testing.assert_allclose(eager[0], compiled[0], rtol=0, atol=1)
                difference = np.abs(
                    np.asarray(eager[0], dtype=np.int16) - np.asarray(compiled[0], dtype=np.int16)
                )
                parity.append(
                    {
                        "max_abs_difference": int(difference.max()),
                        "changed_channels": int(np.count_nonzero(difference)),
                        "compared_channels": difference.size,
                    }
                )
            # Alternate order between resolutions. Both paths use the same moving cameras.
            calls = [("eager", renderer.eager), ("jit", renderer)]
            if len(rows) % 2:
                calls.reverse()
            row = {
                "resolution": name,
                "parity_u8": parity,
                "moving": {key: _measure(call, poses, args.steps) for key, call in calls},
            }
            row["fixed_jit"] = _measure(renderer, [fixed], args.steps)
            frame_samples, sizes = [], []
            for index in range(args.steps):
                start = perf_counter()
                image, overflow = jax.device_get(renderer(poses[index % len(poses)]))
                _, encoded = _encode_image_binary(image, "jpeg", jpeg_quality=90)
                frame_samples.append((perf_counter() - start) * 1000)
                sizes.append(len(encoded))
                if overflow:
                    raise RuntimeError("visibility overflow during image encoding")
            row["render_readback_jpeg"] = {
                "median_ms": float(np.median(frame_samples)),
                "p95_ms": float(np.percentile(frame_samples, 95)),
                "samples_ms": frame_samples,
                "median_bytes": float(np.median(sizes)),
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "resolution": name,
                        "eager_ms": row["moving"]["eager"]["median_ms"],
                        "jit_ms": row["moving"]["jit"]["median_ms"],
                        "frame_ms": row["render_readback_jpeg"]["median_ms"],
                    }
                ),
                flush=True,
            )
        assert renderer.jitted._cache_size() == len(RESOLUTIONS)
    finally:
        stopped.set()
        thread.join()
    with args.model.open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    root = Path(__file__).resolve().parents[1]
    result = {
        "model": str(args.model),
        "model_sha256": model_hash,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("src/jaxgs/viewer.py", "src/jaxgs/render/view.py", "benchmarks/viewer.py")
        },
        "gaussians": int(pool.n_active),
        "gpu": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "viser": viser.__version__,
        "pair_capacity": args.max_visibility_pairs,
        "camera_source": str(args.scene) if args.scene else "automatic bounds fit",
        "warmup_seconds": warmup,
        "jit_cache_size": renderer.jitted._cache_size(),
        "gpu_exclusive_observed": bool(snapshots) and not any(snapshots),
        "gpu_observations": snapshots,
        "results": rows,
        "timing_scope": "Synchronized host-call latency; frame time includes device readback and viser JPEG encoding, excludes network and browser display. Fixed model arrays remain dynamic JIT arguments. Warmup excluded.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
