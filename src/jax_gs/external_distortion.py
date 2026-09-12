"""Root current-main external-distortion parameters and pure-JAX math.

This module mirrors the bivariate windshield model used by gsplat's root
camera wrappers.  It is intentionally separate from
``jax_gs.sensors.kernels.cameras.BivariateWindshieldDistortion``: upstream
defines different coefficient layouts and degree limits for those two APIs.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

ExternalDistortionModelMeta: TypeAlias = Literal[  # noqa: UP040 - runtime alias API
    "bivariate-windshield"
]


class ExternalDistortionModelParameters(ABC):
    """Base class matching gsplat's type-erased rendering argument."""


class ExternalDistortionReferencePolynomial(IntEnum):
    """Which fitted polynomial pair is considered authoritative."""

    FORWARD = 1
    BACKWARD = 2


_VALID_COEFFICIENT_COUNTS = (1, 3, 6, 10, 15, 21)


def _coefficient_array(value, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim != 1 or array.shape[0] not in _VALID_COEFFICIENT_COUNTS:
        raise ValueError(f"{name} must have 1, 3, 6, 10, 15, or 21 coefficients")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating-point dtype")
    return array


@dataclass(eq=False)
class BivariateWindshieldModelParameters(ExternalDistortionModelParameters):
    """Four triangular polynomials used by the root windshield model.

    The no-argument form is retained for gsplat compatibility: callers may
    construct an empty object and assign its four coefficient arrays before
    use.  Passing arrays to the constructor is the clearer JAX-native form.
    Polynomial lengths encode orders zero through five.
    """

    horizontal_poly: jax.Array | None = None
    vertical_poly: jax.Array | None = None
    horizontal_poly_inverse: jax.Array | None = None
    vertical_poly_inverse: jax.Array | None = None
    reference_poly: ExternalDistortionReferencePolynomial = (
        ExternalDistortionReferencePolynomial.FORWARD
    )

    MAX_ORDER = 5
    MAX_COEFFS = 21

    def __setattr__(self, name: str, value) -> None:
        if (
            name
            in {
                "horizontal_poly",
                "vertical_poly",
                "horizontal_poly_inverse",
                "vertical_poly_inverse",
            }
            and value is not None
        ):
            value = _coefficient_array(value, name)
        elif name == "reference_poly":
            value = ExternalDistortionReferencePolynomial(value)
        object.__setattr__(self, name, value)


def validate_external_distortion(
    parameters: BivariateWindshieldModelParameters,
) -> None:
    """Validate that a parameter object is complete and internally consistent."""

    if not isinstance(parameters, BivariateWindshieldModelParameters):
        raise TypeError(
            "external_distortion_coeffs must be BivariateWindshieldModelParameters"
        )
    names = (
        "horizontal_poly",
        "vertical_poly",
        "horizontal_poly_inverse",
        "vertical_poly_inverse",
    )
    arrays = []
    for name in names:
        value = getattr(parameters, name)
        if value is None:
            raise ValueError(f"external distortion is missing {name}")
        arrays.append(_coefficient_array(value, name))
    if any(array.dtype != arrays[0].dtype for array in arrays[1:]):
        raise ValueError("external-distortion polynomials must share one dtype")


def coefficient_order(coefficient_count: int) -> int:
    """Return the triangular polynomial order encoded by a coefficient count."""

    try:
        return _VALID_COEFFICIENT_COUNTS.index(int(coefficient_count))
    except ValueError as error:
        raise ValueError("coefficient_count must be 1, 3, 6, 10, 15, or 21") from error


def eval_bivariate_polynomial(
    x: jax.Array, y: jax.Array, coefficients: jax.Array
) -> jax.Array:
    """Evaluate gsplat's triangular, ascending-in-``x`` coefficient layout."""

    x = jnp.asarray(x)
    y = jnp.asarray(y)
    coefficients = _coefficient_array(coefficients, "coefficients")
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes")
    order = coefficient_order(coefficients.shape[0])
    outer_values = []
    start = 0
    for y_power in range(order + 1):
        group_size = order - y_power + 1
        group = coefficients[start : start + group_size]
        value = jnp.zeros_like(x, dtype=jnp.result_type(x, coefficients))
        for coefficient in reversed(group):
            value = value * x + coefficient
        outer_values.append(value)
        start += group_size
    result = jnp.zeros_like(outer_values[0])
    for value in reversed(outer_values):
        result = result * y + value
    return result


def distort_camera_rays(
    camera_rays: jax.Array,
    parameters: BivariateWindshieldModelParameters,
    *,
    inverse: bool = False,
) -> jax.Array:
    """Apply the forward or inverse bivariate windshield mapping to rays."""

    validate_external_distortion(parameters)
    rays = jnp.asarray(camera_rays)
    if rays.shape[-1] != 3:
        raise ValueError("camera_rays must end in 3 coordinates")
    if not jnp.issubdtype(rays.dtype, jnp.floating):
        raise TypeError("camera_rays must have a floating-point dtype")
    horizontal = (
        parameters.horizontal_poly_inverse if inverse else parameters.horizontal_poly
    )
    vertical = parameters.vertical_poly_inverse if inverse else parameters.vertical_poly
    assert horizontal is not None and vertical is not None
    horizontal = jnp.asarray(horizontal, dtype=rays.dtype)
    vertical = jnp.asarray(vertical, dtype=rays.dtype)

    ray_length = jnp.linalg.norm(rays, axis=-1)
    safe_length = jnp.maximum(ray_length, jnp.asarray(1.0e-6, rays.dtype))
    normalized_x = jnp.clip(rays[..., 0] / safe_length, -1.0, 1.0)
    normalized_y = jnp.clip(rays[..., 1] / safe_length, -1.0, 1.0)
    phi = jnp.arcsin(normalized_x)
    theta = jnp.arcsin(normalized_y)
    x = jnp.sin(eval_bivariate_polynomial(phi, theta, horizontal))
    y = jnp.sin(eval_bivariate_polynomial(phi, theta, vertical))
    z = jnp.sqrt(jnp.maximum(1.0 - jnp.minimum(x * x + y * y, 1.0), 0.0))
    z = z * jnp.where(rays[..., 2] < 0.0, -1.0, 1.0)
    distorted = jnp.stack((x, y, z), axis=-1)
    return jnp.where((ray_length < 1.0e-6)[..., None], rays, distorted)


jax.tree_util.register_dataclass(
    BivariateWindshieldModelParameters,
    data_fields=(
        "horizontal_poly",
        "vertical_poly",
        "horizontal_poly_inverse",
        "vertical_poly_inverse",
    ),
    meta_fields=("reference_poly",),
)


__all__ = [
    "BivariateWindshieldModelParameters",
    "ExternalDistortionModelMeta",
    "ExternalDistortionModelParameters",
    "ExternalDistortionReferencePolynomial",
    "coefficient_order",
    "distort_camera_rays",
    "eval_bivariate_polynomial",
    "validate_external_distortion",
]
