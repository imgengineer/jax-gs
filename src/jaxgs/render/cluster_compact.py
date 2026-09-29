import chex
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class CompactedClusters:
    ids: chex.Array
    valid: chex.Array
    count: chex.Array


def cluster_compact(cluster_tile_mask: chex.Array) -> CompactedClusters:
    """Fixed-size list of clusters touching any tile."""
    active = jnp.any(cluster_tile_mask, axis=1)
    count = jnp.sum(active, dtype=jnp.int32)
    ids = jnp.argsort(~active, stable=True).astype(jnp.int32)
    valid = jnp.arange(active.shape[0]) < count
    return CompactedClusters(ids, valid, count)
