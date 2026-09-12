# SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level primitive wrapper matching gsplat.cuda._wrapper."""

from ..camera_wrappers import create_camera_model
from ..cameras import fully_fused_projection, persp_proj, proj, world_to_cam
from ..capabilities import (
    has_2dgs,
    has_3dgs,
    has_3dgut,
    has_adam,
    has_camera_wrappers,
    has_losses,
    has_reloc,
)
from ..lidar_intersections import isect_tiles_lidar
from ..low_level import (
    accumulate,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_indices_in_range,
    rasterize_to_pixels,
)
from ..math import quat_scale_to_covar_preci
from ..sparse import (
    build_sparse_tile_layout,
    isect_tiles_sparse,
    rasterize_to_pixels_sparse,
)
from ..spherical_harmonics import spherical_harmonics
from ..three_dgut import (
    fully_fused_projection_with_ut,
    rasterize_to_pixels_eval3d,
    rasterize_to_pixels_eval3d_extra,
)
from ..two_dgs import (
    accumulate_2dgs,
    fully_fused_projection_2dgs,
    rasterize_to_indices_in_range_2dgs,
    rasterize_to_pixels_2dgs,
)
from ..visibility import (
    rasterize_contributing_gaussian_ids,
    rasterize_contributing_gaussian_ids_sparse,
    rasterize_num_contributing_gaussians,
    rasterize_num_contributing_gaussians_sparse,
    rasterize_top_contributing_gaussian_ids,
    rasterize_top_contributing_gaussian_ids_sparse,
)

__all__ = [
    "accumulate",
    "accumulate_2dgs",
    "build_sparse_tile_layout",
    "create_camera_model",
    "fully_fused_projection",
    "fully_fused_projection_2dgs",
    "fully_fused_projection_with_ut",
    "has_2dgs",
    "has_3dgs",
    "has_3dgut",
    "has_adam",
    "has_camera_wrappers",
    "has_losses",
    "has_reloc",
    "isect_offset_encode",
    "isect_tiles",
    "isect_tiles_lidar",
    "isect_tiles_sparse",
    "persp_proj",
    "proj",
    "quat_scale_to_covar_preci",
    "rasterize_contributing_gaussian_ids",
    "rasterize_contributing_gaussian_ids_sparse",
    "rasterize_num_contributing_gaussians",
    "rasterize_num_contributing_gaussians_sparse",
    "rasterize_to_indices_in_range",
    "rasterize_to_indices_in_range_2dgs",
    "rasterize_to_pixels",
    "rasterize_to_pixels_2dgs",
    "rasterize_to_pixels_eval3d",
    "rasterize_to_pixels_eval3d_extra",
    "rasterize_to_pixels_sparse",
    "rasterize_top_contributing_gaussian_ids",
    "rasterize_top_contributing_gaussian_ids_sparse",
    "spherical_harmonics",
    "world_to_cam",
]
