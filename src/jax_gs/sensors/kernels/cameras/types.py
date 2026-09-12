"""Pure-JAX parameter types for current-main camera sensor kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import TypeAlias

import jax
import jax.numpy as jnp

FTHETA_MAX_POLYNOMIAL_TERMS = 6
FISHEYE_MAX_FORWARD_POLY_TERMS = 4
_MAX_NEWTON_ITERATIONS = 32


class ShutterType(IntEnum):
    """Sensorlib shutter values, matching the current-main C++ enum."""

    ROLLING_TOP_TO_BOTTOM = 1
    ROLLING_LEFT_TO_RIGHT = 2
    ROLLING_BOTTOM_TO_TOP = 3
    ROLLING_RIGHT_TO_LEFT = 4
    GLOBAL = 5


class ReferencePolynomial(IntEnum):
    """Which direction of a paired polynomial is authoritative."""

    FORWARD = 0
    BACKWARD = 1


def _array_with_shape(value, shape: tuple[int, ...], name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating-point dtype")
    return array


def _resolution(value: tuple[int, int], *, positive: bool) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("resolution must be a (width, height) pair")
    resolution = (int(value[0]), int(value[1]))
    minimum = 1 if positive else 0
    if resolution[0] < minimum or resolution[1] < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"resolution must be {qualifier}")
    return resolution


def _scale_offset(
    scale: tuple[float, float], offset: tuple[float, float]
) -> tuple[jax.Array, jax.Array]:
    scale_array = jnp.asarray(scale)
    offset_array = jnp.asarray(offset)
    if scale_array.shape != (2,) or offset_array.shape != (2,):
        raise ValueError("scale and offset must each contain two values")
    if bool(jnp.any(scale_array <= 0.0)):
        raise ValueError("scale values must be positive")
    return scale_array, offset_array


@dataclass(frozen=True)
class OpenCVPinholeProjection:
    """OpenCV rational pinhole intrinsics and image resolution."""

    focal_length: jax.Array
    principal_point: jax.Array
    radial_coeffs: jax.Array
    tangential_coeffs: jax.Array
    thin_prism_coeffs: jax.Array
    resolution: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "focal_length",
            _array_with_shape(self.focal_length, (2,), "focal_length"),
        )
        object.__setattr__(
            self,
            "principal_point",
            _array_with_shape(self.principal_point, (2,), "principal_point"),
        )
        object.__setattr__(
            self,
            "radial_coeffs",
            _array_with_shape(self.radial_coeffs, (6,), "radial_coeffs"),
        )
        object.__setattr__(
            self,
            "tangential_coeffs",
            _array_with_shape(self.tangential_coeffs, (2,), "tangential_coeffs"),
        )
        object.__setattr__(
            self,
            "thin_prism_coeffs",
            _array_with_shape(self.thin_prism_coeffs, (4,), "thin_prism_coeffs"),
        )
        object.__setattr__(
            self, "resolution", _resolution(self.resolution, positive=True)
        )

    def transform(
        self,
        scale: tuple[float, float],
        offset: tuple[float, float],
        new_resolution: tuple[int, int],
    ) -> OpenCVPinholeProjection:
        """Return intrinsics transformed into a scaled and cropped image."""

        scale_array, offset_array = _scale_offset(scale, offset)
        return replace(
            self,
            focal_length=self.focal_length * scale_array,
            principal_point=self.principal_point * scale_array - offset_array,
            resolution=_resolution(new_resolution, positive=True),
        )


@dataclass(frozen=True)
class FThetaProjection:
    """Paired-polynomial F-Theta projection parameters."""

    principal_point: jax.Array
    fw_poly: jax.Array
    bw_poly: jax.Array
    A: jax.Array
    resolution: tuple[int, int]
    reference_polynomial: int
    fw_poly_degree: int
    bw_poly_degree: int
    newton_iterations: int
    max_angle: float
    min_2d_norm: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "principal_point",
            _array_with_shape(self.principal_point, (2,), "principal_point"),
        )
        object.__setattr__(
            self,
            "fw_poly",
            _array_with_shape(self.fw_poly, (FTHETA_MAX_POLYNOMIAL_TERMS,), "fw_poly"),
        )
        object.__setattr__(
            self,
            "bw_poly",
            _array_with_shape(self.bw_poly, (FTHETA_MAX_POLYNOMIAL_TERMS,), "bw_poly"),
        )
        object.__setattr__(self, "A", _array_with_shape(self.A, (4,), "A"))
        object.__setattr__(
            self, "resolution", _resolution(self.resolution, positive=False)
        )
        reference = int(self.reference_polynomial)
        if reference not in (0, 1):
            raise ValueError("reference_polynomial must be FORWARD (0) or BACKWARD (1)")
        object.__setattr__(self, "reference_polynomial", reference)
        for name in ("fw_poly_degree", "bw_poly_degree"):
            degree = int(getattr(self, name))
            if not 0 <= degree < FTHETA_MAX_POLYNOMIAL_TERMS:
                raise ValueError(
                    f"{name} must be in [0, {FTHETA_MAX_POLYNOMIAL_TERMS})"
                )
            object.__setattr__(self, name, degree)
        iterations = int(self.newton_iterations)
        if not 0 <= iterations <= _MAX_NEWTON_ITERATIONS:
            raise ValueError(
                f"newton_iterations must be in [0, {_MAX_NEWTON_ITERATIONS}]"
            )
        object.__setattr__(self, "newton_iterations", iterations)
        max_angle = float(self.max_angle)
        if not 0.0 <= max_angle <= math.pi:
            raise ValueError("max_angle must be finite and in [0, pi]")
        object.__setattr__(self, "max_angle", max_angle)
        min_2d_norm = float(self.min_2d_norm)
        if not math.isfinite(min_2d_norm) or min_2d_norm <= 0.0:
            raise ValueError("min_2d_norm must be finite and strictly positive")
        object.__setattr__(self, "min_2d_norm", min_2d_norm)

    @staticmethod
    def get_max_polynomial_terms() -> int:
        return FTHETA_MAX_POLYNOMIAL_TERMS

    @property
    def Ainv(self) -> jax.Array:
        """Closed-form row-major inverse of the 2x2 affine matrix ``A``."""

        a, b, c, d = self.A
        determinant = a * d - b * c
        return jnp.stack((d, -b, -c, a)) / determinant

    def transform(
        self,
        scale: tuple[float, float],
        offset: tuple[float, float],
        new_resolution: tuple[int, int],
    ) -> FThetaProjection:
        """Return the exact current-main image-domain intrinsic transform."""

        scale_array, offset_array = _scale_offset(scale, offset)
        scale_u, scale_v = scale_array
        half = jnp.asarray((0.5, 0.5), dtype=self.principal_point.dtype)
        powers = jnp.arange(FTHETA_MAX_POLYNOMIAL_TERMS, dtype=self.bw_poly.dtype)
        ratio = scale_u / scale_v
        return replace(
            self,
            principal_point=(self.principal_point + half) * scale_array
            - half
            - offset_array,
            fw_poly=self.fw_poly * scale_v,
            bw_poly=self.bw_poly * jnp.power(1.0 / scale_v, powers),
            A=self.A
            * jnp.stack((ratio, ratio, jnp.ones_like(ratio), jnp.ones_like(ratio))),
            resolution=_resolution(new_resolution, positive=False),
        )


@dataclass(frozen=True)
class OpenCVFisheyeProjection:
    """OpenCV equidistant-fisheye projection parameters."""

    principal_point: jax.Array
    focal_length: jax.Array
    forward_poly: jax.Array
    approx_backward_factor: jax.Array
    resolution: tuple[int, int]
    newton_iterations: int
    max_angle: float
    min_2d_norm: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "principal_point",
            _array_with_shape(self.principal_point, (2,), "principal_point"),
        )
        object.__setattr__(
            self,
            "focal_length",
            _array_with_shape(self.focal_length, (2,), "focal_length"),
        )
        object.__setattr__(
            self,
            "forward_poly",
            _array_with_shape(
                self.forward_poly,
                (FISHEYE_MAX_FORWARD_POLY_TERMS,),
                "forward_poly",
            ),
        )
        object.__setattr__(
            self,
            "approx_backward_factor",
            _array_with_shape(
                self.approx_backward_factor, (1,), "approx_backward_factor"
            ),
        )
        object.__setattr__(
            self, "resolution", _resolution(self.resolution, positive=False)
        )
        iterations = int(self.newton_iterations)
        if not 0 <= iterations <= _MAX_NEWTON_ITERATIONS:
            raise ValueError(
                f"newton_iterations must be in [0, {_MAX_NEWTON_ITERATIONS}]"
            )
        object.__setattr__(self, "newton_iterations", iterations)
        max_angle = float(self.max_angle)
        if not 0.0 <= max_angle <= math.pi:
            raise ValueError("max_angle must be finite and in [0, pi]")
        object.__setattr__(self, "max_angle", max_angle)
        min_2d_norm = float(self.min_2d_norm)
        if not math.isfinite(min_2d_norm) or min_2d_norm <= 0.0:
            raise ValueError("min_2d_norm must be finite and strictly positive")
        object.__setattr__(self, "min_2d_norm", min_2d_norm)

    @staticmethod
    def get_max_forward_poly_terms() -> int:
        return FISHEYE_MAX_FORWARD_POLY_TERMS

    def transform(
        self,
        scale: tuple[float, float],
        offset: tuple[float, float],
        new_resolution: tuple[int, int],
    ) -> OpenCVFisheyeProjection:
        """Return transformed fisheye intrinsics without mutating this object."""

        scale_array, offset_array = _scale_offset(scale, offset)
        resolution = _resolution(new_resolution, positive=False)
        half = jnp.asarray((0.5, 0.5), dtype=self.principal_point.dtype)
        focal_length = self.focal_length * scale_array
        resolution_array = jnp.asarray(resolution, dtype=focal_length.dtype)
        max_normalized_radius = jnp.max(resolution_array / (2.0 * focal_length))
        backward_factor = jnp.asarray(
            [self.max_angle / max_normalized_radius], dtype=focal_length.dtype
        )
        return replace(
            self,
            principal_point=(self.principal_point + half) * scale_array
            - half
            - offset_array,
            focal_length=focal_length,
            approx_backward_factor=backward_factor,
            resolution=resolution,
        )


@dataclass(frozen=True)
class NoExternalDistortion:
    """Identity external distortion marker."""


@dataclass(frozen=True)
class BivariateWindshieldDistortion:
    """Packed bivariate windshield polynomial parameters."""

    distortion_coeffs: jax.Array
    reference_polynomial: int
    h_poly_degree: int
    v_poly_degree: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "distortion_coeffs",
            _array_with_shape(self.distortion_coeffs, (42,), "distortion_coeffs"),
        )
        reference = int(self.reference_polynomial)
        if reference not in (0, 1):
            raise ValueError("reference_polynomial must be FORWARD (0) or BACKWARD (1)")
        object.__setattr__(self, "reference_polynomial", reference)
        h_degree = int(self.h_poly_degree)
        v_degree = int(self.v_poly_degree)
        if not 0 <= h_degree <= 2:
            raise ValueError("h_poly_degree must be in [0, 2]")
        if not 0 <= v_degree <= 4:
            raise ValueError("v_poly_degree must be in [0, 4]")
        object.__setattr__(self, "h_poly_degree", h_degree)
        object.__setattr__(self, "v_poly_degree", v_degree)


CameraProjection: TypeAlias = (  # noqa: UP040 - runtime alias API
    OpenCVPinholeProjection | FThetaProjection | OpenCVFisheyeProjection
)
ExternalDistortion: TypeAlias = (  # noqa: UP040 - runtime alias API
    NoExternalDistortion | BivariateWindshieldDistortion
)

REGISTERED_CAMERA_PROJECTIONS = (
    OpenCVPinholeProjection,
    FThetaProjection,
    OpenCVFisheyeProjection,
)
REGISTERED_DISTORTIONS = (NoExternalDistortion, BivariateWindshieldDistortion)
REGISTERED_CAMERA_PROJECTION_NAMES = tuple(
    projection.__name__ for projection in REGISTERED_CAMERA_PROJECTIONS
)
REGISTERED_DISTORTION_NAMES = tuple(
    distortion.__name__ for distortion in REGISTERED_DISTORTIONS
)


def script_class_name(obj: object) -> str:
    """Compatibility helper used by current-main dispatch code."""

    return type(obj).__name__


jax.tree_util.register_dataclass(
    OpenCVPinholeProjection,
    data_fields=(
        "focal_length",
        "principal_point",
        "radial_coeffs",
        "tangential_coeffs",
        "thin_prism_coeffs",
    ),
    meta_fields=("resolution",),
)
jax.tree_util.register_dataclass(
    FThetaProjection,
    data_fields=("principal_point", "fw_poly", "bw_poly", "A"),
    meta_fields=(
        "resolution",
        "reference_polynomial",
        "fw_poly_degree",
        "bw_poly_degree",
        "newton_iterations",
        "max_angle",
        "min_2d_norm",
    ),
)
jax.tree_util.register_dataclass(
    OpenCVFisheyeProjection,
    data_fields=(
        "principal_point",
        "focal_length",
        "forward_poly",
        "approx_backward_factor",
    ),
    meta_fields=(
        "resolution",
        "newton_iterations",
        "max_angle",
        "min_2d_norm",
    ),
)
jax.tree_util.register_dataclass(NoExternalDistortion, data_fields=(), meta_fields=())
jax.tree_util.register_dataclass(
    BivariateWindshieldDistortion,
    data_fields=("distortion_coeffs",),
    meta_fields=("reference_polynomial", "h_poly_degree", "v_poly_degree"),
)


__all__ = [  # noqa: RUF022 - preserve the public compatibility order
    "BivariateWindshieldDistortion",
    "CameraProjection",
    "ExternalDistortion",
    "FISHEYE_MAX_FORWARD_POLY_TERMS",
    "FTHETA_MAX_POLYNOMIAL_TERMS",
    "FThetaProjection",
    "NoExternalDistortion",
    "OpenCVFisheyeProjection",
    "OpenCVPinholeProjection",
    "ReferencePolynomial",
    "REGISTERED_CAMERA_PROJECTIONS",
    "REGISTERED_CAMERA_PROJECTION_NAMES",
    "REGISTERED_DISTORTIONS",
    "REGISTERED_DISTORTION_NAMES",
    "script_class_name",
    "ShutterType",
]
