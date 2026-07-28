from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def sort_splats(
    splats: dict[str, Any], verbose: bool = True
) -> dict[str, Any]:
    """Sort aligned splat arrays into a deterministic spatial order.

    Upstream uses the optional PLAS package to optimize the resulting image
    grid. This pure JAX port uses a lexicographic xyz order instead: it is less
    compression-efficient, but deterministic and dependency-free. Like the
    upstream helper, this function mutates and returns ``splats``.
    """

    del verbose
    means = np.asarray(jax.device_get(splats["means"]))
    count = len(means)
    side = int(count**0.5)
    assert side * side == count, "Must be a perfect square"

    order = np.lexsort((means[:, 2], means[:, 1], means[:, 0]))
    jax_order = jnp.asarray(order)
    for name, values in splats.items():
        splats[name] = values[order] if isinstance(values, np.ndarray) else values[jax_order]
    return splats


__all__ = ["sort_splats"]
