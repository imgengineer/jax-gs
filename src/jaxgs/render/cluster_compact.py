"""Fixed-capacity reference compaction of visible clusters."""

import chex
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class CompactedClusters:
    """Stable cluster IDs with masked padding and a scalar valid count."""

    ids: chex.Array  # [num_clusters]
    valid: chex.Array  # [num_clusters]
    count: chex.Array  # scalar


def compact_clusters(cluster_tile_mask: chex.Array) -> CompactedClusters:
    """Compact clusters touching any tile into a stable valid prefix."""
    active_clusters = jnp.any(cluster_tile_mask, axis=1)
    cluster_count = jnp.sum(active_clusters, dtype=jnp.int32)
    cluster_ids = jnp.argsort(~active_clusters, stable=True).astype(jnp.int32)
    valid_mask = jnp.arange(active_clusters.shape[0]) < cluster_count
    return CompactedClusters(cluster_ids, valid_mask, cluster_count)
