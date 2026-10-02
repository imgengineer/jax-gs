"""Compare fused target normalization or gradient scaling at the bicycle image size."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from jaxgs.kernels.fused_loss import _fused_loss_and_grad_with_scale, fused_loss_and_grad


def _separate_scale(prediction, target):
    loss, gradient = fused_loss_and_grad(prediction, target)
    return loss, gradient, jnp.maximum(jnp.max(jnp.abs(gradient)), 1e-12).reshape(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", choices=("target", "scale"), default="target")
    args = parser.parse_args()
    rng = np.random.default_rng(41)
    shape = (822, 1237, 3)
    prediction = jnp.asarray(rng.uniform(0.02, 0.98, shape), jnp.float32)
    target = jnp.asarray(rng.integers(0, 256, shape, dtype=np.uint8))
    functions = {
        "separate": jax.jit(lambda p, t: fused_loss_and_grad(p, t.astype(jnp.float32) / 255)),
        "fused": jax.jit(fused_loss_and_grad),
    }
    if args.compare == "scale":
        functions = {
            "separate": jax.jit(_separate_scale),
            "fused": jax.jit(_fused_loss_and_grad_with_scale),
        }
    samples = {name: [] for name in functions}
    outputs, memory = {}, {}
    for name, fn in functions.items():
        for _ in range(30):
            outputs[name] = fn(prediction, target)
        jax.block_until_ready(outputs[name])
        stats = fn.lower(prediction, target).compile().memory_analysis()
        memory[name] = {
            field: getattr(stats, field)
            for field in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
        }
    for actual, expected in zip(outputs["fused"], outputs["separate"], strict=True):
        np.testing.assert_array_equal(actual, expected)
    for trial in range(8):
        order = list(functions) if trial % 2 == 0 else list(reversed(functions))
        for name in order:
            start = perf_counter()
            for _ in range(1000):
                output = functions[name](prediction, target)
            jax.block_until_ready(output)
            samples[name].append(1e6 * (perf_counter() - start) / 1000)
    report = {
        "comparison": args.compare,
        "shape": shape,
        "gpu": jax.devices()[0].device_kind,
        "protocol": "30 warmup calls per path, 8 alternating rounds of 1000 calls; includes host dispatch",
        "loss_and_gradient_bitwise_equal": True,
        "samples_us": samples,
        "median_us": {name: median(values) for name, values in samples.items()},
        "memory": memory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
