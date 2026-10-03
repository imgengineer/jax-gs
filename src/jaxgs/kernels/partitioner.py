"""JAX binding for stable split/clone candidate grouping."""

import chex
import jax
import jax.numpy as jnp


def stable_split_partition_cute(split_mask: chex.Array, candidate_count: chex.Array) -> chex.Array:
    """Group the selected prefix as splits then clones, preserving both orders."""
    from cutlass.jax import cutlass_call

    from .partition import launch_count_splits, launch_partition_order

    if jax.default_backend() != "gpu":
        raise RuntimeError("CuTe partition requires a JAX CUDA device")
    chex.assert_rank(split_mask, 1)
    capacity = split_mask.shape[0]
    split = split_mask.astype(jnp.int8)
    candidate_count = jnp.asarray(candidate_count, jnp.int32).reshape(1)
    count = cutlass_call(
        launch_count_splits,
        compile_key=launch_count_splits,
        output_shape_dtype=jax.ShapeDtypeStruct(((capacity + 255) // 256,), jnp.int32),
        use_static_tensors=True,
        capacity=capacity,
    )
    block_prefix = jnp.cumsum(count(split, candidate_count), dtype=jnp.int32)
    partition = cutlass_call(
        launch_partition_order,
        compile_key=launch_partition_order,
        output_shape_dtype=jax.ShapeDtypeStruct((capacity,), jnp.int32),
        use_static_tensors=True,
        capacity=capacity,
    )
    return partition(split, candidate_count, block_prefix)
