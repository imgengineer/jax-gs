"""Fixed-capacity NPZ checkpoints shared by training and evaluation."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..scene.point import GaussianArrays


def save_gaussians(path: str | Path, pool: GaussianArrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with np.load(path, allow_pickle=False) as saved:
        return GaussianArrays(
            **{name: jnp.asarray(saved[name]) for name in GaussianArrays.__dataclass_fields__}
        )
