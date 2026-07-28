"""Real spherical harmonics in the ordering used by gsplat."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import Array

from .math import safe_normalize


MAX_SH_DEGREE = 4
MAX_SH_BASES = 25


def _all_sh_bases(dirs: Array) -> Array:
    """Evaluate all real SH bases through degree four."""

    x, y, z = jnp.moveaxis(dirs, -1, 0)
    z2 = z * z

    f_tmp_a1 = -0.48860251190292
    degree0 = (jnp.full_like(x, 0.2820947917738781),)
    degree1 = (
        f_tmp_a1 * y,
        -f_tmp_a1 * z,
        f_tmp_a1 * x,
    )

    f_tmp_b2 = -1.092548430592079 * z
    f_tmp_a2 = 0.5462742152960395
    f_c1 = x * x - y * y
    f_s1 = 2.0 * x * y
    degree2 = (
        f_tmp_a2 * f_s1,
        f_tmp_b2 * y,
        0.9461746957575601 * z2 - 0.3153915652525201,
        f_tmp_b2 * x,
        f_tmp_a2 * f_c1,
    )

    f_tmp_c3 = -2.285228997322329 * z2 + 0.4570457994644658
    f_tmp_b3 = 1.445305721320277 * z
    f_tmp_a3 = -0.5900435899266435
    f_c2 = x * f_c1 - y * f_s1
    f_s2 = x * f_s1 + y * f_c1
    degree3 = (
        f_tmp_a3 * f_s2,
        f_tmp_b3 * f_s1,
        f_tmp_c3 * y,
        z * (1.865881662950577 * z2 - 1.119528997770346),
        f_tmp_c3 * x,
        f_tmp_b3 * f_c1,
        f_tmp_a3 * f_c2,
    )

    f_tmp_d4 = z * (-4.683325804901025 * z2 + 2.007139630671868)
    f_tmp_c4 = 3.31161143515146 * z2 - 0.47308734787878
    f_tmp_b4 = -1.770130769779931 * z
    f_tmp_a4 = 0.6258357354491763
    f_c3 = x * f_c2 - y * f_s2
    f_s3 = x * f_s2 + y * f_c2
    degree4 = (
        f_tmp_a4 * f_s3,
        f_tmp_b4 * f_s2,
        f_tmp_c4 * f_s1,
        f_tmp_d4 * y,
        1.984313483298443
        * z2
        * (1.865881662950577 * z2 - 1.119528997770346)
        - 1.006230589874905
        * (0.9461746957575601 * z2 - 0.3153915652525201),
        f_tmp_d4 * x,
        f_tmp_c4 * f_c1,
        f_tmp_b4 * f_c2,
        f_tmp_a4 * f_c3,
    )
    return jnp.stack(degree0 + degree1 + degree2 + degree3 + degree4, axis=-1)


def eval_sh_bases(degree: int, dirs: Array) -> Array:
    """Evaluate real SH bases through a static degree in ``[0, 4]``."""

    if degree < 0 or degree > MAX_SH_DEGREE:
        raise ValueError(f"degree must be in [0, {MAX_SH_DEGREE}], got {degree}")
    dirs = safe_normalize(jnp.asarray(dirs), axis=-1)
    return _all_sh_bases(dirs)[..., : (degree + 1) ** 2]


def spherical_harmonics(
    degrees_to_use: int | Array,
    dirs: Array,
    coeffs: Array,
    masks: Optional[Array] = None,
) -> Array:
    """Evaluate RGB spherical harmonics with gsplat-compatible ordering.

    Unlike a dynamic slice, the degree mask preserves a fixed output shape and
    lets ``degrees_to_use`` itself be traced by JAX.  ``coeffs`` may contain 1,
    4, 9, 16, or 25 (or any intermediate number of) bases.
    """

    dirs = safe_normalize(jnp.asarray(dirs), axis=-1)
    coeffs = jnp.asarray(coeffs)
    basis_count = coeffs.shape[-2]
    if basis_count > MAX_SH_BASES:
        raise ValueError(f"At most {MAX_SH_BASES} SH coefficients are supported")
    bases = _all_sh_bases(dirs)[..., :basis_count]
    requested_count = (jnp.asarray(degrees_to_use) + 1) ** 2
    degree_mask = jnp.arange(basis_count) < requested_count
    result = jnp.sum(bases[..., :, None] * coeffs * degree_mask[:, None], axis=-2)
    if masks is not None:
        result = jnp.where(jnp.asarray(masks)[..., None], result, 0.0)
    return result


eval_sh = spherical_harmonics
