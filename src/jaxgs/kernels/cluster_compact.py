"""Fixed-capacity visible-cluster list consumed by projection kernels."""

import chex
import cuda.bindings.driver as cuda
import cutlass.cute as cute
import jax
import jax.numpy as jnp

from ..scene.types import VisibleClusters


@cute.kernel
def _scatter(visible: cute.Tensor, ranks: cute.Tensor, ids: cute.Tensor, size: int):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    i = block * 256 + tid
    if i < size:
        if visible[i] != 0:
            ids[ranks[i] - 1] = i


@cute.jit
def _launch(
    stream: cuda.CUstream, visible: cute.Tensor, ranks: cute.Tensor, ids: cute.Tensor, *, size: int
):
    _scatter(visible, ranks, ids, size).launch(
        grid=[(size + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
    )


def compact_visible_clusters(point_mask: chex.Array, cluster_size: int) -> VisibleClusters:
    """Only ids[:count] is consumed; capacity never depends on visibility."""
    from cutlass.jax import cutlass_call

    visible = point_mask[::cluster_size].astype(jnp.int32)
    ranks = jnp.cumsum(visible, dtype=jnp.int32)
    call = cutlass_call(
        _launch,
        compile_key=_launch,
        output_shape_dtype=jax.ShapeDtypeStruct(visible.shape, jnp.int32),
        use_static_tensors=True,
        size=visible.size,
    )
    return call(visible, ranks), ranks[-1:]
