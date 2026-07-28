"""Current-main-compatible high-level rendering surface."""

from .external_distortion import (
    BivariateWindshieldModelParameters,
    ExternalDistortionModelMeta,
    ExternalDistortionModelParameters,
    ExternalDistortionReferencePolynomial,
)
from .rasterization import rasterization, rasterization_inria_wrapper
from .rendering_types import (
    CameraModel,
    RasterizeMode,
    RendererConfig,
    RendererConfig_MixedBatch,
    RendererConfig_ParallelBatch,
    RenderMode,
    render_mode_has_color,
    render_mode_has_depth,
    render_mode_has_depth_channel,
    render_mode_has_expected_depth,
    render_mode_has_hit_distance,
    render_mode_has_only_color,
    render_mode_has_only_depth_channel,
)
from .three_dgut import (
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    RollingShutterType,
    UnscentedTransformParameters,
)
from .two_dgs import (
    rasterization_2dgs,
    rasterization_2dgs_inria_wrapper,
)


__all__ = [
    "CameraModel",
    "BivariateWindshieldModelParameters",
    "ExternalDistortionModelMeta",
    "ExternalDistortionModelParameters",
    "ExternalDistortionReferencePolynomial",
    "FThetaCameraDistortionParameters",
    "FThetaPolynomialType",
    "RasterizeMode",
    "RendererConfig",
    "RendererConfig_MixedBatch",
    "RendererConfig_ParallelBatch",
    "RenderMode",
    "RollingShutterType",
    "UnscentedTransformParameters",
    "rasterization",
    "rasterization_2dgs",
    "rasterization_2dgs_inria_wrapper",
    "rasterization_inria_wrapper",
    "render_mode_has_color",
    "render_mode_has_depth",
    "render_mode_has_depth_channel",
    "render_mode_has_expected_depth",
    "render_mode_has_hit_distance",
    "render_mode_has_only_color",
    "render_mode_has_only_depth_channel",
]
