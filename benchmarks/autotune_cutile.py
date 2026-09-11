"""Autotune semantic-equivalent cuTile variants through full renderer timing."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_PROFILES = ("default", "small", "wide", "low_occupancy")
_TUNING_OVERRIDE_ENV = (
    "JAX_GS_CUTILE_COUNT_BLOCK_SIZE",
    "JAX_GS_CUTILE_EMIT_BLOCK_SIZE",
    "JAX_GS_CUTILE_COUNT_OCCUPANCY",
    "JAX_GS_CUTILE_EMIT_OCCUPANCY",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark cuTile topology variants inside the complete "
            "cuda_tile renderer and cache the fastest profile"
        )
    )
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--active", type=int, required=True)
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--max-intersections", type=int, required=True)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--hot-iters", type=int, default=100)
    parser.add_argument("--backward-iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--profiles", nargs="+", choices=_PROFILES, default=_PROFILES
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "jax-gs" / "cutile-autotune",
    )
    return parser


def _benchmark_command(args: argparse.Namespace, output: Path) -> list[str]:
    benchmark = Path(__file__).with_name("benchmark_rasterization.py")
    return [
        sys.executable,
        str(benchmark),
        "--npz",
        str(args.npz),
        "--capacity",
        str(args.capacity),
        "--active",
        str(args.active),
        "--resolution",
        args.resolution,
        "--backend",
        "intersections",
        "--compositor-backend",
        "cuda_tile",
        "--projection-backend",
        "cuda_tile",
        "--intersection-backend",
        "cuda_tile",
        "--intersection-mode",
        "accutile",
        "--sort-backend",
        "cuda_tile",
        "--max-intersections",
        str(args.max_intersections),
        "--max-candidates-per-tile",
        "2048",
        "--k",
        "512",
        "--tile-size",
        "16",
        "--radius-clip",
        str(args.radius_clip),
        "--camera-index",
        str(args.camera_index),
        "--warmup-iters",
        str(args.warmup_iters),
        "--hot-iters",
        str(args.hot_iters),
        "--backward",
        "--backward-iters",
        str(args.backward_iters),
        "--allow-unsafe",
        "--json-output",
        str(output),
    ]


def _run_profile(
    args: argparse.Namespace, profile: str, directory: Path
) -> dict[str, Any]:
    forward = []
    backward = []
    for repeat in range(args.repeats):
        output = directory / f"{profile}-{repeat}.json"
        env = os.environ.copy()
        for name in _TUNING_OVERRIDE_ENV:
            env.pop(name, None)
        env["JAX_GS_CUTILE_TUNING"] = profile
        result = subprocess.run(
            _benchmark_command(args, output),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"cuTile profile {profile!r} failed:\n{result.stdout[-12000:]}"
            )
        try:
            measurement = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"failed to read benchmark output for profile {profile!r}"
            ) from exc
        intersections = measurement["intersections"]
        if intersections["overflow"]:
            raise RuntimeError(
                f"cuTile profile {profile!r} overflowed max_intersections"
            )
        forward.append(measurement["forward"]["hot"]["median_ms"])
        backward.append(measurement["backward"]["hot"]["median_ms"])
    return {
        "profile": profile,
        "forward_median_ms": statistics.median(forward),
        "value_and_grad_median_ms": statistics.median(backward),
        "forward_runs_ms": forward,
        "value_and_grad_runs_ms": backward,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jax-gs-cutile-tune-") as temporary:
        directory = Path(temporary)
        measurements = [
            _run_profile(args, profile, directory)
            for profile in args.profiles
        ]
    best = min(
        measurements,
        key=lambda value: (
            value["value_and_grad_median_ms"],
            value["forward_median_ms"],
        ),
    )
    cache_key = (
        f"{args.npz.stem}-{args.capacity}-{args.active}-"
        f"{args.resolution}-camera{args.camera_index}-"
        f"max{args.max_intersections}-radius{args.radius_clip:g}"
    )
    record = {
        "cache_key": cache_key,
        "selection": best["profile"],
        "measurements": measurements,
        "environment": {
            "JAX_GS_CUTILE_TUNING": best["profile"],
        },
    }
    path = args.cache_dir / f"{cache_key}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    print(json.dumps(record, indent=2, sort_keys=True))
    print(
        f"export JAX_GS_CUTILE_TUNING={best['profile']}  # cached in {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
