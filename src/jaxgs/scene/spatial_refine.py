import chex
import jax
import jax.numpy as jnp

from ..training.optimizer import AdamState
from .point import GaussianPool


def _spread_bits(value: chex.Array) -> chex.Array:
    value = (value | (value << 16)) & jnp.uint32(0x030000FF)
    value = (value | (value << 8)) & jnp.uint32(0x0300F00F)
    value = (value | (value << 4)) & jnp.uint32(0x030C30C3)
    return (value | (value << 2)) & jnp.uint32(0x09249249)


@jax.jit
def spatial_refine(pool: GaussianPool, state: AdamState) -> tuple[GaussianPool, AdamState]:
    """Group live slots by 3D Morton order without changing pool or Adam shapes."""
    lower = jnp.min(jnp.where(pool.alive[:, None], pool.xyz, jnp.inf), axis=0)
    upper = jnp.max(jnp.where(pool.alive[:, None], pool.xyz, -jnp.inf), axis=0)
    lower = jnp.where(jnp.isfinite(lower), lower, 0)
    upper = jnp.where(jnp.isfinite(upper), upper, 1)
    quantized = jnp.asarray(
        jnp.clip((pool.xyz - lower) / jnp.maximum(upper - lower, 1e-12), 0, 1) * 1023, jnp.uint32
    )
    code = (
        _spread_bits(quantized[:, 0])
        | (_spread_bits(quantized[:, 1]) << 1)
        | (_spread_bits(quantized[:, 2]) << 2)
    )
    order = jnp.argsort(jnp.where(pool.alive, code, jnp.uint32(0xFFFFFFFF)), stable=True)
    return (
        jax.tree_util.tree_map(lambda value: value[order] if value.ndim else value, pool),
        jax.tree_util.tree_map(lambda value: value[order], state),
    )
