"""LiteGS-style preprocessing and RGB rendering with fixed-shape JAX data."""

from dataclasses import replace

import chex
import jax
import jax.numpy as jnp

from ..config import CapacityConfig
from ..scene.camera import Camera
from ..scene.point import GaussianArrays
from ..scene.types import VisibleClusters, WorldClusterBounds
from .types import RenderOutput

__all__ = ["render_preprocess", "render", "RenderOutput"]


def render_preprocess(
    cluster_bounds: WorldClusterBounds,
    camera: Camera,
    gaussians: GaussianArrays,
    pp: CapacityConfig,
    *,
    backend: str = "cute",
) -> tuple[VisibleClusters, chex.Array, GaussianArrays]:
    """Return visible cluster IDs/count, the slot mask and culled array views."""
    if backend not in ("cute", "reference"):
        raise ValueError(f"unknown backend: {backend}")
    from ..scene.cluster import frustum_cluster_mask

    visible_slots = frustum_cluster_mask(cluster_bounds, camera, pp.cluster_size, pp.max_gaussians)
    if backend == "cute":
        from ..kernels.cluster_compact import compact_visible_clusters

        visible_clusters = compact_visible_clusters(visible_slots, pp.cluster_size)
    else:
        from .cluster_compact import compact_clusters

        padding = pp.num_clusters * pp.cluster_size - pp.max_gaussians
        compacted = compact_clusters(
            jnp.pad(visible_slots, (0, padding)).reshape(pp.num_clusters, pp.cluster_size)
        )
        visible_clusters = compacted.ids, compacted.count.reshape(1)
    return visible_clusters, visible_slots, gaussians.replace(alive=gaussians.alive & visible_slots)


def render(
    camera: Camera,
    gaussians: GaussianArrays,
    visible_clusters: VisibleClusters,
    actived_sh_degree: int,
    pp: CapacityConfig,
    *,
    backend: str = "cute",
) -> RenderOutput:
    """Project, bin and render preprocessed Gaussians; image gradients use VJPs."""
    if backend not in ("cute", "reference"):
        raise ValueError(f"unknown backend: {backend}")
    if not 0 <= actived_sh_degree <= pp.sh_degree:
        raise ValueError("actived_sh_degree must fit the pool's SH coefficients")
    if backend == "cute":
        from ..kernels.packed_rasterizer import rasterize_packed_cute_vjp
        from ..kernels.projector import project_cute_vjp
        from ..kernels.sorted_binning import build_sorted_visibility_table_cute
        from ..kernels.sorted_rasterizer import rasterize_sorted_cute_vjp

        projected = project_cute_vjp(gaussians, camera, pp, actived_sh_degree, visible_clusters)
        table = build_sorted_visibility_table_cute(jax.lax.stop_gradient(projected), camera, pp)
        image = (
            rasterize_packed_cute_vjp(projected, table, camera, pp)
            if pp.tile_size in (8, 16)
            else rasterize_sorted_cute_vjp(projected, table, camera, pp).rgb
        )
        primitive_visible = table.point_counts > 0
    else:
        from .cluster_culling import build_cluster_tile_mask
        from .projection import project
        from .rasterizer import rasterize
        from .visibility_table import build_visibility_table

        projected = project(gaussians, camera, replace(pp, sh_degree=actived_sh_degree))
        binned = jax.lax.stop_gradient(projected)
        table = build_visibility_table(
            binned, camera, pp, build_cluster_tile_mask(binned, camera, pp)
        )
        image = rasterize(projected, table, camera, pp, backend="reference").rgb
        primitive_visible = (
            jnp.zeros((pp.max_gaussians,), jnp.int32)
            .at[table.tile_gaussian_ids.reshape(-1)]
            .max(table.tile_valid.reshape(-1).astype(jnp.int32))
            > 0
        )
    return RenderOutput(jnp.clip(image, 0, 1), primitive_visible, jnp.any(table.overflow))
