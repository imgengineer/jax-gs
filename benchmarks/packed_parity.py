"""Compare CuTe half2 kernels directly with LiteGS, using identical projected inputs.

Run with uv's environment; --litegs-python selects the native LiteGS environment.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def native(args):
    sys.path.insert(0, str(args.litegs_root))
    import torch
    from litegs.utils.wrapper import litegs_fused

    data = np.load(args.native)
    height, width = data["image_grad"].shape[:2]
    tile_size = int(data["tile_size"])
    tile_height = int(data["tile_height"])

    def tensor(x):
        return torch.as_tensor(np.ascontiguousarray(x), device="cuda")

    ndc = np.zeros((1, 4, len(data["mean"])), np.float32)
    ndc[0, :2] = (data["mean"] * (2 / np.array([width, height])) - 1).T
    ids = tensor(data["ids"][None])
    offsets = tensor(np.r_[-1, data["offsets"]].astype(np.int32)[None])
    result = litegs_fused.rasterize_forward(
        ids,
        offsets,
        tensor(ndc),
        tensor(data["conic"].transpose(1, 2, 0)[None]),
        tensor(data["color"].T[None]),
        tensor(data["alpha"][None]),
        None,
        height,
        width,
        tile_height,
        tile_size,
        True,
        False,
        False,
    )
    _, _, pair_counts = litegs_fused.get_allocate_size(
        tensor(ndc),
        tensor(data["depth"][None]),
        tensor(data["conic"].transpose(1, 2, 0)[None]),
        tensor(data["alpha"][None]),
        height,
        width,
        tile_height,
        tile_size,
        None,
    )
    depth_order = torch.argsort(tensor(data["depth"][None]), stable=True)
    pair_offsets = torch.cumsum(torch.gather(pair_counts, 1, depth_order), 1).to(torch.int32)
    tile_ids, gaussian_ids = litegs_fused.create_table(
        tensor(ndc),
        tensor(data["conic"].transpose(1, 2, 0)[None]),
        tensor(data["alpha"][None]),
        pair_offsets,
        depth_order,
        None,
        None,
        height,
        width,
        tile_height,
        tile_size,
    )
    image, trans, _, last, params, count, weight = result
    image_grad = tensor(data["image_grad"].transpose(2, 0, 1)[None])
    scale = image_grad.abs().max()
    grads = litegs_fused.rasterize_backward(
        ids,
        offsets,
        params,
        None,
        trans,
        last,
        image_grad / scale,
        None,
        None,
        scale,
        height,
        width,
        tile_height,
        tile_size,
        True,
    )
    gm, gc, grgb, ga, _, square = grads

    def array(t):
        return t.detach().cpu().numpy()

    np.savez(
        args.native.with_suffix(".native.npz"),
        pair_counts=array(pair_counts).reshape(-1),
        tile_ids=array(tile_ids).reshape(-1) - 1,
        gaussian_ids=array(gaussian_ids).reshape(-1),
        rgb=array(image)[0].transpose(1, 2, 0),
        trans=array(trans).reshape(height, width),
        mean=array(gm)[0, :2].T * (2 / np.array([width, height])),
        conic=array(gc)[0].transpose(2, 0, 1),
        color=array(grgb)[0].T,
        alpha=array(ga).reshape(-1),
        count=array(count).reshape(-1),
        weight=array(weight).reshape(-1),
        square=array(square).reshape(-1) * float(scale) ** 2,
    )


def compare(args):
    import jax
    import jax.numpy as jnp

    from jaxgs import Camera, CapacityConfig
    from jaxgs.kernels.packed_rasterizer import packed_backward, packed_forward
    from jaxgs.kernels.sorted_binning import build_sorted_visibility_table_cute
    from jaxgs.render.types import ProjectedGaussians, SortedVisibilityTable

    rng = np.random.default_rng(47)
    width, height, capacity = 48, 32, 48
    camera = Camera.from_colmap(
        [1, 0, 0, 0], [0, 0, 0], 32, 32, width / 2, height / 2, width, height
    )
    report = {}
    for tile_height, tile_size in ((8, 8), (8, 16), (16, 16)):
        config = CapacityConfig(
            capacity, 1, 128, tile_size, 0, capacity * 24, tile_height=tile_height
        )
        mean = rng.uniform((0, 0), (width, height), (capacity, 2)).astype(np.float32)
        factors = rng.normal(0, 0.2, (capacity, 2, 2)).astype(np.float32)
        conic = factors @ factors.transpose(0, 2, 1) + np.eye(2, dtype=np.float32) * 0.02
        projected = ProjectedGaussians(
            jnp.array(mean),
            jnp.arange(capacity, dtype=jnp.float32) + 1,
            jnp.array(conic, jnp.float32),
            jnp.ones(capacity),
            jnp.array(rng.uniform(0.05, 0.95, (capacity, 3)), jnp.float32),
            jnp.array(rng.uniform(0.1, 0.9, capacity), jnp.float32),
            jnp.ones(capacity, bool),
        )
        tiles = width * height // (tile_size * tile_height)
        table = SortedVisibilityTable(
            jnp.tile(jnp.arange(capacity), tiles),
            jnp.arange(tiles + 1) * capacity,
            jnp.full(capacity, tiles),
            jnp.array(capacity * tiles),
            jnp.array(False),
        )
        binned = build_sorted_visibility_table_cute(projected, camera, config)
        image_grad = jnp.array(rng.normal(0, 0.001, (height, width, 3)), jnp.float32)
        rgb, cache, fragments = jax.jit(lambda p: packed_forward(p, table, camera, config, True))(
            projected
        )
        grads, square = jax.jit(
            lambda g: packed_backward(projected, table, cache, g, camera, config, True)
        )(image_grad)
        with tempfile.TemporaryDirectory(prefix="jaxgs-packed-parity-") as directory:
            path = Path(directory) / "inputs.npz"
            np.savez(
                path,
                mean=mean,
                conic=projected.conic,
                color=projected.color,
                alpha=projected.alpha,
                ids=table.gaussian_ids,
                offsets=table.tile_offsets,
                image_grad=image_grad,
                tile_size=tile_size,
                tile_height=tile_height,
                depth=projected.depth,
            )
            subprocess.run(
                [
                    str(args.litegs_python),
                    __file__,
                    "--litegs-root",
                    str(args.litegs_root),
                    "--native",
                    str(path),
                ],
                check=True,
            )
            expected = np.load(path.with_suffix(".native.npz"))
            np.testing.assert_array_equal(binned.point_counts, expected["pair_counts"])
            np.testing.assert_array_equal(
                binned.gaussian_ids[: int(binned.pair_count)], expected["gaussian_ids"]
            )
            np.testing.assert_array_equal(
                np.diff(binned.tile_offsets), np.bincount(expected["tile_ids"], minlength=tiles)
            )
            actual = dict(
                rgb=rgb,
                trans=cache[1].reshape(height, width),
                mean=grads.mean,
                conic=grads.conic,
                color=grads.color,
                alpha=grads.alpha,
                count=fragments[0::2],
                weight=fragments[1::2],
                square=square,
            )
            errors = {}
            for name, value in actual.items():
                value, reference = np.asarray(value), expected[name]
                diff = value - reference
                relative = float(np.linalg.norm(diff) / max(np.linalg.norm(reference), 1e-20))
                errors[name] = dict(relative_l2=relative, max_abs=float(np.max(np.abs(diff))))
                if name == "count":
                    np.testing.assert_array_equal(value, reference)
                else:
                    assert relative < 0.005, (tile_size, name, errors[name])
            report[f"{tile_height}x{tile_size}"] = errors
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--litegs-root", type=Path, default=Path("/home/lzc/Documents/LiteGS"))
    parser.add_argument(
        "--litegs-python", type=Path, default=Path("/home/lzc/Documents/LiteGS/.venv/bin/python")
    )
    parser.add_argument("--native", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/packed_parity.json")
    )
    arguments = parser.parse_args()
    native(arguments) if arguments.native else compare(arguments)
