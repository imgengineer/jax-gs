"""Factories for packed bivariate windshield distortion parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import BivariateWindshieldDistortion, ReferencePolynomial

MAX_H_POLYNOMIAL_TERMS = 6
MAX_V_POLYNOMIAL_TERMS = 15


def _compute_poly_order(poly_coeffs: jax.Array) -> int:
    if poly_coeffs.ndim != 1:
        raise ValueError("windshield polynomial coefficients must be 1D")
    term_count = poly_coeffs.shape[0]
    running = 0
    for order in range(term_count):
        running += order + 1
        if running == term_count:
            return order
        if running > term_count:
            break
    raise ValueError(
        "The input length of the windshield distortion coefficients is not "
        "consistent with a triangular bivariate polynomial layout "
        f"(got {term_count} terms; valid sizes: 1, 3, 6, 10, 15, ...)."
    )


def _pad_poly_to_max_terms(poly: jax.Array, max_terms: int, name: str) -> jax.Array:
    if poly.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if poly.shape[0] > max_terms:
        raise ValueError(f"{name} must have at most {max_terms} coefficients")
    return jnp.pad(poly, (0, max_terms - poly.shape[0]))


def from_components(
    h_poly: jax.Array,
    v_poly: jax.Array,
    h_poly_inv: jax.Array,
    v_poly_inv: jax.Array,
    reference_polynomial: ReferencePolynomial | int,
) -> BivariateWindshieldDistortion:
    """Pack four triangular polynomials into current-main's 42-value layout."""

    h_poly = jnp.asarray(h_poly)
    v_poly = jnp.asarray(v_poly)
    h_poly_inv = jnp.asarray(h_poly_inv)
    v_poly_inv = jnp.asarray(v_poly_inv)
    arrays = (h_poly, v_poly, h_poly_inv, v_poly_inv)
    if not all(jnp.issubdtype(array.dtype, jnp.floating) for array in arrays):
        raise TypeError("windshield coefficients must have floating-point dtypes")
    if any(array.dtype != h_poly.dtype for array in arrays[1:]):
        raise ValueError("all windshield polynomial tensors must have matching dtype")

    h_degree = _compute_poly_order(h_poly)
    v_degree = _compute_poly_order(v_poly)
    h_inv_degree = _compute_poly_order(h_poly_inv)
    v_inv_degree = _compute_poly_order(v_poly_inv)
    if h_degree != h_inv_degree:
        raise ValueError("h_poly and h_poly_inv must have matching triangular degree")
    if v_degree != v_inv_degree:
        raise ValueError("v_poly and v_poly_inv must have matching triangular degree")
    if h_degree > 2:
        raise ValueError("h_poly degree must be <= 2")
    if v_degree > 4:
        raise ValueError("v_poly degree must be <= 4")

    distortion_coeffs = jnp.concatenate(
        (
            _pad_poly_to_max_terms(h_poly, MAX_H_POLYNOMIAL_TERMS, "h_poly"),
            _pad_poly_to_max_terms(v_poly, MAX_V_POLYNOMIAL_TERMS, "v_poly"),
            _pad_poly_to_max_terms(h_poly_inv, MAX_H_POLYNOMIAL_TERMS, "h_poly_inv"),
            _pad_poly_to_max_terms(v_poly_inv, MAX_V_POLYNOMIAL_TERMS, "v_poly_inv"),
        )
    )
    return BivariateWindshieldDistortion(
        distortion_coeffs=distortion_coeffs,
        reference_polynomial=int(reference_polynomial),
        h_poly_degree=h_degree,
        v_poly_degree=v_degree,
    )


__all__ = [  # noqa: RUF022 - preserve the public compatibility order
    "BivariateWindshieldDistortion",
    "MAX_H_POLYNOMIAL_TERMS",
    "MAX_V_POLYNOMIAL_TERMS",
    "from_components",
]
