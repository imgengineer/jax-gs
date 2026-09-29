import chex
import jax
import jax.numpy as jnp


def allocate_free_slots_cute(free_mask: chex.Array, max_new: int) -> tuple[chex.Array, chex.Array]:
    from cutlass.jax import cutlass_call

    from .allocation import launch_allocate_free

    if jax.default_backend() != "gpu":
        raise RuntimeError("CuTe allocation requires a JAX CUDA device")
    capacity = free_mask.shape[0]
    shapes = (
        jax.ShapeDtypeStruct((max_new,), jnp.int32),
        jax.ShapeDtypeStruct((max_new,), jnp.int8),
    )
    call = cutlass_call(
        launch_allocate_free,
        output_shape_dtype=shapes,
        use_static_tensors=True,
        capacity=capacity,
        max_new=max_new,
    )
    rank = jnp.cumsum(free_mask, dtype=jnp.int32) - 1
    slots, valid = call(free_mask.astype(jnp.int8), rank)
    return slots, valid.astype(jnp.bool_)
