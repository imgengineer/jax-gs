"""Inference-scene spherical-harmonic packing modes."""

from enum import IntEnum


class SHCompressionMode(IntEnum):
    """Layout transforms used by the current-main inference scene packer."""

    NONE = 0
    PACKED_32B = 1
    PACKED_16B = 2


SH_COMPRESSION_MAP = {
    "none": SHCompressionMode.NONE,
    "32b": SHCompressionMode.PACKED_32B,
    "16b": SHCompressionMode.PACKED_16B,
}
SH_COMPRESSION_MODE_VALUES = frozenset(SHCompressionMode)

__all__ = [  # noqa: RUF022 - preserve the public compatibility order
    "SHCompressionMode",
    "SH_COMPRESSION_MAP",
    "SH_COMPRESSION_MODE_VALUES",
]
