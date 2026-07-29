"""JAX/Flax NNX Gaussian splatting.

The public surface is migrating by subsystem to a pinned gsplat ``main``
commit. Dynamic packed buffers are represented by padded arrays plus valid
counts and overflow flags where JAX requires static shapes.
"""

import os

from ._runtime_cache import configure_persistent_compilation_cache

configure_persistent_compilation_cache()

# Large bucketed models are supported up to ten million logical slots, but
# JAX's default GPU preallocation can make
# concurrent tools or desktop workloads run out of memory before training even
# starts. Serial GPU code generation also avoids launching several memory-heavy
# ptxas workers at once. Users can override either default in their environment.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_force_compilation_parallelism=1",
)

from .api import PaddedProjection, fully_fused_projection
from .cameras import proj, world_to_cam
from .capabilities import (
    has_2dgs,
    has_3dgs,
    has_3dgut,
    has_adam,
    has_camera_wrappers,
    has_losses,
    has_reloc,
)
from .checkpoints import (
    load_checkpoint_appearance_image_names,
    load_checkpoint_config,
    load_checkpoint_intersection_capacity,
    load_checkpoint_scene_transform,
    load_checkpoint_storage_capacity,
    load_distributed_checkpoint_manifest,
    restore_checkpoint,
    restore_distributed_checkpoint,
    save_checkpoint,
    save_distributed_checkpoint,
)
from .camera_wrappers import RootCameraModel, create_camera_model
from .color_correct import color_correct_affine, color_correct_quadratic
from .compression import PngCompression
from .config import (
    DataConfig,
    MAX_MODEL_CAPACITY,
    ModelConfig,
    OptimizerConfig,
    RasterizationConfig,
    StrategyConfig,
    TrainConfig,
)
from .distributed import distributed_rasterization
from .external_distortion import (
    BivariateWindshieldModelParameters,
    ExternalDistortionModelMeta,
    ExternalDistortionModelParameters,
    ExternalDistortionReferencePolynomial,
)
from .exporter import export_splat, export_splats, export_ply
from .math import quat_scale_to_covar_preci, quat_to_rotmat
from .low_level import (
    PaddedIntersections,
    PaddedOffsets,
    PaddedRasterizationIndices,
    accumulate,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_indices_in_range,
    rasterize_to_pixels,
)
from .lidar import (
    LidarTiling,
    RowOffsetStructuredSpinningLidarModelParameters,
    RowOffsetStructuredSpinningLidarModelParametersExt,
    SpinningDirection,
    compute_angles_to_columns_map as compute_lidar_angles_to_columns_map,
    compute_tiling as compute_lidar_tiling,
)
from .lidar_intersections import isect_tiles_lidar
from .losses import (
    create_ssim_window,
    depth_l1_loss,
    gaussian_density_reg,
    gaussian_scale_reg,
    gaussian_z_scale_reg,
    l1_loss,
    lidar_background_loss,
    lidar_distance_loss,
    lidar_intensity_loss,
    lidar_raydrop_loss,
    mse_loss,
    opacity_reg_loss,
    out_of_bound_loss,
    scale_reg_loss,
    ssim_loss,
    torch_ssim_loss,
    total_variation_loss,
)
from .losses_fused import FusedGaussianLosses
from .model import GaussianModel
from .optimizers import SelectiveAdam, create_optimizer
from .rasterization import rasterization, rasterization_inria_wrapper
from .rendering_types import (
    CameraModel,
    RasterizeMode,
    RendererConfig,
    RendererConfig_MixedBatch,
    RendererConfig_ParallelBatch,
    RenderMode,
)
from .sparse import (
    PaddedSparseIntersections,
    PaddedSparseTileLayout,
    build_sparse_tile_layout,
    isect_tiles_sparse,
    rasterize_to_pixels_sparse,
)
from .spherical_harmonics import spherical_harmonics
from .strategy import DefaultStrategy, MCMCStrategy, Strategy, StrategyState
from .three_dgut import (
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    RollingShutterType,
    UnscentedTransformParameters,
    fully_fused_projection_with_ut,
    rasterize_to_pixels_eval3d,
)
from .two_dgs import (
    PaddedProjection2DGS,
    accumulate_2dgs,
    fully_fused_projection_2dgs,
    rasterization_2dgs,
    rasterization_2dgs_inria_wrapper,
    rasterize_to_indices_in_range_2dgs,
    rasterize_to_pixels_2dgs,
)
from .visibility import (
    PaddedContributors,
    rasterize_contributing_gaussian_ids,
    rasterize_contributing_gaussian_ids_sparse,
    rasterize_num_contributing_gaussians,
    rasterize_num_contributing_gaussians_sparse,
    rasterize_top_contributing_gaussian_ids,
    rasterize_top_contributing_gaussian_ids_sparse,
)
from .version import (
    GSPLAT_BASELINE_SHA,
    GSPLAT_BASELINE_VERSION,
    GSPLAT_TARGET_BRANCH,
    GSPLAT_TARGET_COMMIT_DATE,
    GSPLAT_TARGET_SHA,
    __version__,
)

__all__ = [
    "DataConfig",
    "DefaultStrategy",
    "FusedGaussianLosses",
    "CameraModel",
    "BivariateWindshieldModelParameters",
    "ExternalDistortionModelMeta",
    "ExternalDistortionModelParameters",
    "ExternalDistortionReferencePolynomial",
    "FThetaCameraDistortionParameters",
    "FThetaPolynomialType",
    "GaussianModel",
    "GSPLAT_BASELINE_SHA",
    "GSPLAT_BASELINE_VERSION",
    "GSPLAT_TARGET_BRANCH",
    "GSPLAT_TARGET_COMMIT_DATE",
    "GSPLAT_TARGET_SHA",
    "MCMCStrategy",
    "MAX_MODEL_CAPACITY",
    "ModelConfig",
    "OptimizerConfig",
    "PngCompression",
    "PaddedIntersections",
    "PaddedOffsets",
    "PaddedProjection",
    "PaddedProjection2DGS",
    "PaddedRasterizationIndices",
    "PaddedContributors",
    "PaddedSparseIntersections",
    "PaddedSparseTileLayout",
    "LidarTiling",
    "RasterizationConfig",
    "RasterizeMode",
    "RendererConfig",
    "RendererConfig_MixedBatch",
    "RendererConfig_ParallelBatch",
    "RenderMode",
    "SelectiveAdam",
    "StrategyConfig",
    "Strategy",
    "StrategyState",
    "TrainConfig",
    "UnscentedTransformParameters",
    "RollingShutterType",
    "RootCameraModel",
    "RowOffsetStructuredSpinningLidarModelParameters",
    "RowOffsetStructuredSpinningLidarModelParametersExt",
    "SpinningDirection",
    "accumulate",
    "accumulate_2dgs",
    "build_sparse_tile_layout",
    "color_correct_affine",
    "color_correct_quadratic",
    "create_optimizer",
    "create_camera_model",
    "compute_lidar_angles_to_columns_map",
    "compute_lidar_tiling",
    "distributed_rasterization",
    "export_ply",
    "export_splat",
    "export_splats",
    "fully_fused_projection",
    "fully_fused_projection_2dgs",
    "fully_fused_projection_with_ut",
    "gaussian_density_reg",
    "gaussian_scale_reg",
    "gaussian_z_scale_reg",
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
    "load_checkpoint_appearance_image_names",
    "load_checkpoint_config",
    "load_checkpoint_intersection_capacity",
    "load_checkpoint_scene_transform",
    "load_checkpoint_storage_capacity",
    "load_distributed_checkpoint_manifest",
    "out_of_bound_loss",
    "proj",
    "quat_scale_to_covar_preci",
    "quat_to_rotmat",
    "rasterization",
    "rasterization_2dgs",
    "rasterization_2dgs_inria_wrapper",
    "rasterization_inria_wrapper",
    "rasterize_contributing_gaussian_ids",
    "rasterize_contributing_gaussian_ids_sparse",
    "rasterize_num_contributing_gaussians",
    "rasterize_num_contributing_gaussians_sparse",
    "rasterize_to_indices_in_range",
    "rasterize_to_indices_in_range_2dgs",
    "rasterize_to_pixels",
    "rasterize_to_pixels_2dgs",
    "rasterize_to_pixels_eval3d",
    "rasterize_to_pixels_sparse",
    "rasterize_top_contributing_gaussian_ids",
    "rasterize_top_contributing_gaussian_ids_sparse",
    "restore_checkpoint",
    "restore_distributed_checkpoint",
    "save_checkpoint",
    "save_distributed_checkpoint",
    "spherical_harmonics",
    "world_to_cam",
    "__version__",
]
