"""Pure-JAX timing, input capture, and replay profiling helpers."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from .trace import trace_range

_F = TypeVar("_F", bound=Callable[..., Any])
_GRAD_PARAMS = {
    "means",
    "quats",
    "scales",
    "opacities",
    "colors",
    "extra_signals",
}


def _detach_for_capture(value: Any) -> Any:
    if isinstance(value, jax.Array):
        return np.asarray(jax.device_get(value))
    return value


profiler: dict[str, float] = {}


@dataclass
class ProfileWorkload:
    """One callable and its resolved inputs for a replay run."""

    name: str
    operator_name: str
    operator: Callable[..., Any]
    replay_inputs: dict[str, Any]
    losses: list[str]
    loss_contribution: Callable[[tuple[Any, ...]], jax.Array]
    expected_kernel_families: list[str]
    notes: list[str]


def _parse_override_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "nan":
        return float("nan")
    if lowered == "inf":
        return float("inf")
    if lowered == "-inf":
        return float("-inf")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _parse_input_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"expected NAME=VALUE, got {raw!r}")
    name, raw_value = raw.split("=", 1)
    name = name.strip()
    if not name or any(token in name for token in (".", "[", "]")):
        raise ValueError(
            f"override name must be a top-level operator argument, got {name!r}"
        )
    return name, _parse_override_value(raw_value)


def _parse_renderer_config_override(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.lower().replace("_", "")
    factories = {
        "mixedbatch": "RendererConfig_MixedBatch",
        "parallelbatch": "RendererConfig_ParallelBatch",
    }
    if normalized not in factories:
        allowed = ", ".join(sorted(factories))
        raise ValueError(
            f"renderer_config override must be one of: {allowed}; got {value!r}"
        )
    from . import rendering

    return getattr(rendering, factories[normalized])()


class timeit:
    """Accumulate synchronized wall time when ``TIMEIT=1``."""

    def __init__(self, name: str = "unnamed") -> None:
        self.name = name
        self.start_time: float | None = None
        self.enabled = os.environ.get("TIMEIT", "0") == "1"

    def __enter__(self) -> None:
        if self.enabled:
            jax.effects_barrier()
            self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_type, exc_val, exc_tb
        if self.enabled:
            jax.effects_barrier()
            if self.start_time is None:
                raise RuntimeError("timeit context exited before it entered")
            profiler[self.name] = profiler.get(self.name, 0.0) + (
                time.perf_counter() - self.start_time
            )

    def __call__(self, f: _F) -> _F:
        @wraps(f)
        def decorated(*args, **kwargs):
            if self.name == "unnamed":
                self.name = f.__name__
            with self:
                return f(*args, **kwargs)

        return decorated  # type: ignore[return-value]


_pending_captures: set[int] = set()
_next_capture_id = 0


def _parse_capture_specs(envvar: str, raw_specs: str) -> list[tuple[Path, range]]:
    capture_dir = os.environ.get("GSPLAT_INPUT_CAPTURE_DIR")
    specs: list[tuple[Path, range]] = []
    for raw_spec in raw_specs.split(","):
        raw_spec = raw_spec.strip()
        if not raw_spec:
            continue
        parts = raw_spec.split(":")
        range_values: list[int] = []
        path_end = len(parts)
        for index in range(len(parts) - 1, -1, -1):
            try:
                range_values.insert(0, int(parts[index]))
                path_end = index
            except ValueError:
                break
        if not 1 <= len(range_values) <= 3:
            raise ValueError(
                f"{envvar}: expected <path>:<stop>, <path>:<start>:<stop>, "
                f"or <path>:<start>:<stop>:<step> in each spec, got {raw_spec!r}"
            )
        if any(value < 0 for value in range_values):
            raise ValueError(
                f"{envvar}: negative values are not supported, got {raw_spec!r}"
            )
        if len(range_values) >= 2 and range_values[1] < range_values[0]:
            raise ValueError(
                f"{envvar}: stop ({range_values[1]}) must be >= start "
                f"({range_values[0]}), got {raw_spec!r}"
            )
        try:
            capture_range = range(*range_values)
        except ValueError as exc:
            raise ValueError(
                f"{envvar}: invalid capture range in {raw_spec!r}"
            ) from exc
        if not capture_range:
            raise ValueError(
                f"{envvar}: empty range (nothing to capture), got {raw_spec!r}"
            )
        output_text = ":".join(parts[:path_end])
        if not output_text:
            raise ValueError(f"{envvar}: capture output path cannot be empty")
        output_path = Path(output_text)
        if capture_dir and not output_path.is_absolute():
            output_path = Path(capture_dir) / output_path
        if not output_path.suffix:
            output_path = output_path.with_suffix(".pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        specs.append((output_path, capture_range))
    if not specs:
        raise ValueError(f"{envvar}: no specs provided, got {raw_specs!r}")

    claimed: dict[int, int] = {}
    for spec_index, (_, capture_range) in enumerate(specs):
        for call_index in capture_range:
            if call_index in claimed:
                raise ValueError(
                    f"{envvar}: call index {call_index} is claimed by multiple "
                    f"specs ({claimed[call_index]} and {spec_index}), got "
                    f"{raw_specs!r}"
                )
            claimed[call_index] = spec_index
    return specs


def _capture_path(path: Path, worker_tag: str, index: int, digits: int) -> Path:
    return path.with_name(f"{path.stem}_{worker_tag}_{index:0{digits}d}{path.suffix}")


def capture_inputs(*, envvar: str) -> Callable[[_F], _F]:
    """Capture selected calls as pickle/NumPy pytrees when ``envvar`` is set.

    The range syntax and multi-decorator exit behavior match current main.
    Files use the upstream ``.pt`` default name for discoverability, but are
    standard Python pickle rather than ``torch.save`` payloads.
    """

    def decorator(function: _F) -> _F:
        global _next_capture_id

        raw_specs = os.environ.get(envvar)
        if not raw_specs:
            return function
        specs = _parse_capture_specs(envvar, raw_specs)
        all_indices = [index for _, capture_range in specs for index in capture_range]
        digits = len(str(max(all_indices)))
        total_captures = len(all_indices)
        capture_id = _next_capture_id
        _next_capture_id += 1
        _pending_captures.add(capture_id)
        signature = inspect.signature(function)
        call_count = 0
        captures_done = 0

        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count, captures_done

            matching_path = next(
                (path for path, capture_range in specs if call_count in capture_range),
                None,
            )
            if matching_path is not None:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                captured = jax.tree.map(_detach_for_capture, dict(bound.arguments))
                rank = os.environ.get("RANK")
                worker_tag = f"r{rank}" if rank is not None else f"p{os.getpid()}"
                save_path = _capture_path(matching_path, worker_tag, call_count, digits)
                with save_path.open("wb") as handle:
                    pickle.dump(captured, handle, protocol=pickle.HIGHEST_PROTOCOL)
                captures_done += 1
                print(
                    f"[jax_gs.profile] captured {function.__name__} "
                    f"({captures_done}/{total_captures}, call {call_count}) "
                    f"to {save_path}",
                    flush=True,
                )
                if captures_done >= total_captures:
                    _pending_captures.discard(capture_id)
                    if not _pending_captures:
                        raise SystemExit("[jax_gs.profile] all captures done, exiting")
            call_count += 1
            return function(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def load_capture(path: str | Path) -> dict[str, Any]:
    """Load a :func:`capture_inputs` payload and place NumPy arrays on JAX."""

    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("captured payload must be a dictionary")
    return jax.tree.map(
        lambda value: jax.device_put(value) if isinstance(value, np.ndarray) else value,
        payload,
    )


def _resize_color_last_dim(value: jax.Array, target: int) -> jax.Array:
    value = jnp.asarray(value)
    current = value.shape[-1]
    if current <= 0:
        raise ValueError("source color array must have at least one channel")
    if target <= current:
        return value[..., :target]
    repeats = (target + current - 1) // current
    tiled = jnp.concatenate([value] * repeats, axis=-1)
    return tiled[..., :target]


def _apply_channel_override(replay_inputs: dict[str, Any], channels: int) -> list[str]:
    """Resize captured color/background channels for a replay specialization."""

    if channels < 1:
        raise ValueError("channels must be positive")
    colors = replay_inputs.get("colors")
    if colors is None:
        raise ValueError("colors must be present for a channel override")
    colors = jnp.asarray(colors)
    sh_degree = replay_inputs.get("sh_degree")
    notes: list[str] = []
    if sh_degree is not None:
        expected_bases = (int(sh_degree) + 1) ** 2
        if colors.ndim < 3 or colors.shape[-2] != expected_bases:
            raise ValueError("colors shape is inconsistent with sh_degree")
        colors = colors[..., 0, :]
        replay_inputs["sh_degree"] = None
        notes.append("collapsed spherical harmonics to the DC band")
    render_mode = replay_inputs.get("render_mode", "RGB")
    if render_mode in {"D", "ED"}:
        raise ValueError("channel override is undefined for depth-only rendering")
    depth_channels = int("D" in str(render_mode))
    extra = replay_inputs.get("extra_signals")
    extra_channels = 0 if extra is None else int(jnp.asarray(extra).shape[-1])
    color_channels = channels - depth_channels - extra_channels
    if color_channels < 1:
        raise ValueError("channels leaves no room for a color channel")
    replay_inputs["colors"] = _resize_color_last_dim(colors, color_channels)
    if replay_inputs.get("backgrounds") is not None:
        replay_inputs["backgrounds"] = _resize_color_last_dim(
            replay_inputs["backgrounds"], color_channels
        )
    notes.append(f"resized rendered channel specialization to {channels}")
    return notes


def _select_replay_inputs(
    inputs: dict[str, Any], operator: Callable[..., Any]
) -> dict[str, Any]:
    parameters = inspect.signature(operator).parameters
    selected = {name: inputs[name] for name in parameters if name in inputs}
    missing = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and name not in selected
    ]
    if missing:
        raise ValueError(f"capture is missing required inputs: {missing}")
    return selected


def _block_tree(value: Any) -> Any:
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready() if isinstance(leaf, jax.Array) else leaf,
        value,
    )


def _make_replay(
    operator: Callable[..., Any],
    replay_inputs: dict[str, Any],
    *,
    with_grad: bool,
) -> tuple[Callable[[tuple[jax.Array, ...]], Any], tuple[jax.Array, ...]]:
    array_names = [
        name for name, value in replay_inputs.items() if isinstance(value, jax.Array)
    ]
    array_values = tuple(replay_inputs[name] for name in array_names)

    def inputs_from(values: tuple[jax.Array, ...]) -> dict[str, Any]:
        current = dict(replay_inputs)
        current.update(zip(array_names, values, strict=True))
        return current

    if not with_grad:
        return jax.jit(lambda values: operator(**inputs_from(values))), array_values

    grad_positions = tuple(
        index
        for index, name in enumerate(array_names)
        if name in _GRAD_PARAMS
        and jnp.issubdtype(array_values[index].dtype, jnp.inexact)
    )
    if not grad_positions:
        raise ValueError("gradient replay has no differentiable captured inputs")

    def loss(values: tuple[jax.Array, ...]) -> jax.Array:
        differentiable_values = tuple(
            value if index in grad_positions else jax.lax.stop_gradient(value)
            for index, value in enumerate(values)
        )
        outputs = operator(**inputs_from(differentiable_values))
        leaves = [
            leaf
            for leaf in jax.tree.leaves(outputs)
            if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.inexact)
        ]
        if not leaves:
            raise ValueError("replay operator returned no floating-point arrays")
        return sum(jnp.sum(leaf) for leaf in leaves)

    return jax.jit(jax.value_and_grad(loss, argnums=0, allow_int=True)), array_values


def main() -> None:
    """Replay captured JAX inputs through rasterization for profiler traces."""

    from .rendering import rasterization, rasterization_2dgs

    parser = argparse.ArgumentParser(description="JAX-GS profiling harness")
    parser.add_argument("--input", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--nograd", dest="grad", action="store_false")
    parser.add_argument("--grad", dest="grad", action="store_true", default=True)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--2dgs", dest="use_2dgs", action="store_true")
    parser.add_argument("--channels", type=int)
    parser.add_argument(
        "--set", "--override", dest="overrides", action="append", default=[]
    )
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.iterations < 1:
        parser.error("warmup must be non-negative and iterations must be positive")

    inputs = load_capture(arguments.input)
    operator = rasterization_2dgs if arguments.use_2dgs else rasterization
    replay_inputs = _select_replay_inputs(inputs, operator)
    parameters = inspect.signature(operator).parameters
    for raw_override in arguments.overrides:
        try:
            name, value = _parse_input_override(raw_override)
        except ValueError as exc:
            parser.error(str(exc))
        if name not in parameters:
            parser.error(f"unknown operator argument {name!r}")
        if name == "renderer_config":
            value = _parse_renderer_config_override(value)
        replay_inputs[name] = value
    if arguments.channels is not None:
        try:
            _apply_channel_override(replay_inputs, arguments.channels)
        except ValueError as exc:
            parser.error(str(exc))

    if arguments.describe:
        print(f"operator={operator.__name__} grad={arguments.grad}")
        for name, value in replay_inputs.items():
            description = (
                f"shape={value.shape}, dtype={value.dtype}"
                if isinstance(value, jax.Array)
                else repr(value)
            )
            print(f"  {name}: {description}")
        return

    replay, array_values = _make_replay(
        operator, replay_inputs, with_grad=arguments.grad
    )
    for _ in range(arguments.warmup):
        _block_tree(replay(array_values))
    start = time.perf_counter()
    for iteration in range(arguments.iterations):
        with trace_range("profile_iteration", iteration=iteration):
            output = replay(array_values)
            if arguments.sync:
                _block_tree(output)
    _block_tree(output)
    elapsed = time.perf_counter() - start
    print(
        f"operator={operator.__name__} iterations={arguments.iterations} "
        f"average_ms={elapsed * 1000.0 / arguments.iterations:.3f}",
        flush=True,
    )


__all__ = [
    "ProfileWorkload",
    "capture_inputs",
    "load_capture",
    "main",
    "profiler",
    "timeit",
]


if __name__ == "__main__":
    main()
