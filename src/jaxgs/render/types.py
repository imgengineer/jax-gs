"""Fixed-shape data shared by production and reference renderers."""

import chex
from flax import struct


@struct.dataclass
class ProjectedGaussians:
    mean: chex.Array  # [C, 2], pixel coordinates
    depth: chex.Array  # [C]
    conic: chex.Array  # [C, 2, 2], inverse covariance
    radius: chex.Array  # [C], 3 sigma
    color: chex.Array  # [C, 3]
    alpha: chex.Array  # [C]
    visible: chex.Array  # [C]


@struct.dataclass
class VisibilityTable:
    """Diagnostic table with K entries per tile and masked padding."""

    tile_gaussian_ids: chex.Array  # [T, K]
    tile_depths: chex.Array  # [T, K]
    tile_valid: chex.Array  # [T, K]
    tile_count: chex.Array  # [T]
    overflow: chex.Array  # [T]


@struct.dataclass
class SortedVisibilityTable:
    """Global depth-ordered tile/Gaussian pairs in a fixed-capacity arena."""

    gaussian_ids: chex.Array  # [P_MAX], valid prefix ends at pair_count; the rest is undefined
    tile_offsets: chex.Array  # [T + 1]
    point_counts: chex.Array  # [C]
    pair_count: chex.Array  # scalar
    overflow: chex.Array  # scalar


@struct.dataclass
class RenderResult:
    rgb: chex.Array  # [H, W, 3]
    depth: chex.Array  # [H, W]
    alpha: chex.Array  # [H, W]


@struct.dataclass
class RenderOutput:
    """RGB pipeline output, binned primitive mask and capacity status."""

    image: chex.Array  # [H, W, 3], clamped to [0, 1]
    primitive_visible: chex.Array  # [C], appears in at least one tile
    overflow: chex.Array  # scalar


# Packed parameters [C * 8] uint32, final transmittance [H * W] float32,
# last processed pair [H * W] int32, backward pair counts [T] int32 and contribution
# bits uint32. Tile t starts at floor(tile_offsets[t] / 32) + t; only words before
# ceil(backward_pair_count / 32) are initialized. Remaining words must not be read.
type PackedRasterCache = tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]

# [C, 4]: fragment count, compositing weight, sum(dL/dalpha), sum((dL/dalpha)^2).
type FragmentStatistics = chex.Array
