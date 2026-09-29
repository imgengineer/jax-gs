from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityConfig:
    max_gaussians: int = 1 << 20
    cluster_size: int = 128
    max_gaussians_per_tile: int = 128
    tile_size: int = 16
    sh_degree: int = 3
    max_visibility_pairs: int | None = None
    tile_height: int | None = None

    def __post_init__(self) -> None:
        if (
            min(self.max_gaussians, self.cluster_size, self.max_gaussians_per_tile, self.tile_size)
            <= 0
        ):
            raise ValueError("capacities and tile_size must be positive")
        if not 0 <= self.sh_degree <= 3:
            raise ValueError("sh_degree must be between 0 and 3")
        if self.max_visibility_pairs is not None and self.max_visibility_pairs <= 0:
            raise ValueError("max_visibility_pairs must be positive")
        if self.tile_height is not None and self.tile_height <= 0:
            raise ValueError("tile_height must be positive")

    @property
    def raster_tile_height(self) -> int:
        return self.tile_height or self.tile_size

    @property
    def sh_dim(self) -> int:
        return (self.sh_degree + 1) ** 2

    @property
    def num_clusters(self) -> int:
        return (self.max_gaussians + self.cluster_size - 1) // self.cluster_size

    @property
    def visibility_capacity(self) -> int:
        return self.max_visibility_pairs or self.max_gaussians * 16
