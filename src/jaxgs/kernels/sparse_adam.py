"""LiteGS compact.cu::sparse_chunk_adam_kernel on fixed-capacity JAX buffers.

Adapted from LiteGS; see LICENSE.LiteGS. Buffer aliases preserve untouched slots.
"""

import chex
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import jax
import jax.numpy as jnp

from ..scene.types import VisibleClusters


@cute.kernel
def _update(
    value: cute.Tensor,
    mean: cute.Tensor,
    variance: cute.Tensor,
    gradient: cute.Tensor,
    alive: cute.Tensor,
    ids: cute.Tensor,
    count: cute.Tensor,
    rate: cute.Tensor,
    capacity: int,
    channels: cutlass.Constexpr,
    cluster_size: cutlass.Constexpr,
    is_sh: cutlass.Constexpr,
    compact_gradients: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    cluster, part, _ = cute.arch.block_idx()
    if cluster < count[0]:
        local = part * 256 + tid
        if local < cluster_size * channels:
            gid = ids[cluster] * cluster_size + local // channels
            if gid < capacity and alive[gid] != 0:
                i = gid * channels + local % channels
                gradient_index = i
                if cutlass.const_expr(compact_gradients):
                    gradient_index = cluster * cluster_size * channels + local
                g = gradient[gradient_index]
                m = 0.9 * mean[i] + 0.1 * g
                v = 0.999 * variance[i] + 0.001 * g * g
                lr = rate[0]
                if cutlass.const_expr(is_sh):
                    if local % channels < 3:
                        lr *= 10.0
                value[i] = value[i] - lr * m / (cute.sqrt(v) + 1e-15)
                mean[i], variance[i] = m, v


@cute.jit
def _launch(
    stream: cuda.CUstream,
    value: cute.Tensor,
    mean: cute.Tensor,
    variance: cute.Tensor,
    gradient: cute.Tensor,
    alive: cute.Tensor,
    ids: cute.Tensor,
    count: cute.Tensor,
    rate: cute.Tensor,
    *,
    capacity: int,
    channels: int,
    cluster_size: int,
    is_sh: bool,
    compact_gradients: bool,
):
    _update(
        value,
        mean,
        variance,
        gradient,
        alive,
        ids,
        count,
        rate,
        capacity,
        channels,
        cluster_size,
        is_sh,
        compact_gradients,
    ).launch(
        grid=[
            (capacity + cluster_size - 1) // cluster_size,
            (cluster_size * channels + 255) // 256,
            1,
        ],
        block=[256, 1, 1],
        stream=stream,
    )


def update_field(
    value: chex.Array,
    mean: chex.Array,
    variance: chex.Array,
    gradient: chex.Array,
    alive: chex.Array,
    clusters: VisibleClusters,
    rate: float | chex.Array,
    cluster_size: int,
    is_sh: bool,
    compact_gradients: bool,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    from cutlass.jax import cutlass_call

    call = cutlass_call(
        _launch,
        output_shape_dtype=tuple(
            jax.ShapeDtypeStruct((x.size,), x.dtype) for x in (value, mean, variance)
        ),
        input_output_aliases={0: 0, 1: 1, 2: 2},
        use_static_tensors=True,
        capacity=alive.size,
        channels=value.size // alive.size,
        cluster_size=cluster_size,
        is_sh=is_sh,
        compact_gradients=compact_gradients,
    )
    return tuple(
        x.reshape(value.shape)
        for x in call(
            value.reshape(-1),
            mean.reshape(-1),
            variance.reshape(-1),
            gradient.reshape(-1),
            alive.astype(jnp.int8),
            *clusters,
            jnp.asarray(rate, jnp.float32).reshape(1),
        )
    )
