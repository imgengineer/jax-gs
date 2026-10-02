"""Gaussian Splatting PLY models and fixed-capacity NPZ checkpoints."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from plyfile import PlyData, PlyElement

from ..scene.point import GaussianArrays
from ..scene.types import PARAMETER_NAMES


def save_gaussians(path: str | Path, pool: GaussianArrays) -> None:
    """Write active Gaussians to PLY, or preserve the entire pool in NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".ply":
        _save_ply(path, pool)
        return
    # A file handle preserves the exact requested path, including custom suffixes.
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            **{
                name: np.asarray(getattr(pool, name))
                for name in GaussianArrays.__dataclass_fields__
            },
        )


def load_gaussians(path: str | Path) -> GaussianArrays:
    """Load PLY into a compact, all-active pool, or restore an NPZ pool."""
    if Path(path).suffix.lower() == ".ply":
        return _load_ply(path)
    with np.load(path, allow_pickle=False) as saved:
        return GaussianArrays(
            **{name: jnp.asarray(saved[name]) for name in GaussianArrays.__dataclass_fields__}
        )


def _save_ply(path: Path, pool: GaussianArrays) -> None:
    alive = np.asarray(pool.alive)
    xyz, scale, rotation, opacity, sh = (
        np.asarray(getattr(pool, name))[alive] for name in PARAMETER_NAMES
    )
    attributes = {name: xyz[:, i] for i, name in enumerate(("x", "y", "z"))}
    attributes.update({name: np.zeros(len(xyz), np.float32) for name in ("nx", "ny", "nz")})
    attributes.update({f"f_dc_{i}": sh[:, 0, i] for i in range(3)})
    # LiteGS/3DGS stores channel-major SH: all red coefficients, then green and blue.
    rest = sh.shape[1] - 1
    attributes.update(
        {
            f"f_rest_{channel * rest + coefficient}": sh[:, coefficient + 1, channel]
            for channel in range(3)
            for coefficient in range(rest)
        }
    )
    attributes["opacity"] = opacity[:, 0]
    attributes.update({f"scale_{i}": scale[:, i] for i in range(3)})
    attributes.update({f"rot_{i}": rotation[:, i] for i in range(4)})
    vertices = np.empty(len(xyz), dtype=[(name, "<f4") for name in attributes])
    for name, values in attributes.items():
        vertices[name] = values
    PlyData([PlyElement.describe(vertices, "vertex")], byte_order="<").write(path)


def _load_ply(path: str | Path) -> GaussianArrays:
    vertices = PlyData.read(path)["vertex"]
    count = len(vertices)
    rest_count = sum(name.startswith("f_rest_") for name in vertices.data.dtype.names)
    if rest_count not in (0, 9, 24, 45):
        raise ValueError("PLY SH dimension must be 1, 4, 9 or 16")

    def fields(names):
        return np.stack([vertices[name] for name in names], axis=-1).astype(np.float32)

    sh = np.empty((count, rest_count // 3 + 1, 3), np.float32)
    sh[:, 0] = fields([f"f_dc_{i}" for i in range(3)])
    if rest_count:
        sh[:, 1:] = (
            fields([f"f_rest_{i}" for i in range(rest_count)])
            .reshape(count, 3, rest_count // 3)
            .transpose(0, 2, 1)
        )
    return GaussianArrays(
        xyz=jnp.asarray(fields(["x", "y", "z"])),
        log_scale=jnp.asarray(fields([f"scale_{i}" for i in range(3)])),
        rotation=jnp.asarray(fields([f"rot_{i}" for i in range(4)])),
        opacity=jnp.asarray(fields(["opacity"])),
        sh=jnp.asarray(sh),
        alive=jnp.ones(count, jnp.bool_),
        free_mask=jnp.zeros(count, jnp.bool_),
        n_active=jnp.array(count, jnp.int32),
    )
