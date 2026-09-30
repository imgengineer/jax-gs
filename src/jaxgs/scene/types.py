"""Parameter order and fixed-shape cluster layouts shared across stages."""

import chex
from flax import struct

PARAMETER_NAMES = ("xyz", "log_scale", "rotation", "opacity", "sh")


@struct.dataclass
class ParameterArrays:
    """Five parameter leaves, also used for gradients, moments and learning rates."""

    xyz: chex.Array | float
    log_scale: chex.Array | float
    rotation: chex.Array | float
    opacity: chex.Array | float
    sh: chex.Array | float


# Gradient buffers preserve PARAMETER_NAMES order. Compact gradients use the
# visible-cluster prefix; inactive tails must not be read.
type ParameterGradients = tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]

# Stable cluster IDs [num_clusters] and valid count [1]. IDs beyond count[0]
# are undefined; each cluster contributes cluster_size consecutive pool slots.
type VisibleClusters = tuple[chex.Array, chex.Array]

# Cluster center [num_clusters, 3], half extent [num_clusters, 3], valid mask
# [num_clusters], in that order.
type WorldClusterBounds = tuple[chex.Array, chex.Array, chex.Array]
