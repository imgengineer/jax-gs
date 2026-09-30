"""Fixed-capacity visible-cluster list consumed by projection kernels."""

import chex
import cuda.bindings.driver as cuda
import cutlass
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
        output_shape_dtype=jax.ShapeDtypeStruct(visible.shape, jnp.int32),
        use_static_tensors=True,
        size=visible.size,
    )
    return call(visible, ranks), ranks[-1:]


@cute.kernel
def _clear_projected(
    mean: cute.Tensor,
    depth: cute.Tensor,
    conic: cute.Tensor,
    radius: cute.Tensor,
    color: cute.Tensor,
    alpha: cute.Tensor,
    visible: cute.Tensor,
    capacity: int,
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    i = block * 256 + tid
    if i < capacity * 4:
        conic[i] = cute.Float32(0)
    if i < capacity * 3:
        color[i] = cute.Float32(0)
    if i < capacity * 2:
        mean[i] = cute.Float32(0)
    if i < capacity:
        depth[i] = cute.Float32(float("inf"))
        radius[i], alpha[i] = cute.Float32(0), cute.Float32(0)
        visible[i] = cutlass.Int8(0)


@cute.kernel
def _clear_parameter_grads(
    xyz: cute.Tensor,
    scale: cute.Tensor,
    rotation: cute.Tensor,
    opacity: cute.Tensor,
    sh: cute.Tensor,
    capacity: int,
    sh_dim: int,
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    i = block * 256 + tid
    if i < capacity * sh_dim * 3:
        sh[i] = cute.Float32(0)
    if i < capacity * 4:
        rotation[i] = cute.Float32(0)
    if i < capacity * 3:
        xyz[i], scale[i] = cute.Float32(0), cute.Float32(0)
    if i < capacity:
        opacity[i] = cute.Float32(0)
