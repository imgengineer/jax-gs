"""Shared experimental render API types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax


@dataclass
class RenderReturn:
    """Frame plus renderer-specific metadata."""

    frame: jax.Array
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["RenderReturn"]
