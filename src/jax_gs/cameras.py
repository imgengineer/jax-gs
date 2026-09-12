"""Camera transforms and EWA projection for 3D Gaussian splatting."""

from __future__ import annotations

import warnings
from typing import Literal

import jax.numpy as jnp
from jax import Array

from .math import quat_scale_to_covar_preci, quat_to_rotmat, triu_to_full

CameraModel = Literal["pinhole", "ortho", "fisheye"]
_FACTOR_PROJECTION_MIN_GAUSSIANS = 262_144


def _safe_denominator(x: Array, eps: float = 1e-8) -> Array:
    eps_array = jnp.asarray(eps, dtype=x.dtype)
    sign = jnp.where(x < 0.0, -jnp.ones_like(x), jnp.ones_like(x))
    return jnp.where(jnp.abs(x) < eps_array, sign * eps_array, x)


def world_to_cam(
    means: Array,
    covars: Array,
    viewmats: Array,
) -> tuple[Array, Array]:
    """Transform Gaussian means/covariances from world to camera space.

    Args:
        means: World-space means with shape ``[..., N, 3]``.
        covars: World-space covariance matrices ``[..., N, 3, 3]``.
        viewmats: World-to-camera matrices ``[..., C, 4, 4]``.
    """

    means = jnp.asarray(means)
    covars = jnp.asarray(covars)
    viewmats = jnp.asarray(viewmats)
    rotation = viewmats[..., :3, :3]
    translation = viewmats[..., :3, 3]
    means_c = jnp.einsum("...cij,...nj->...cni", rotation, means)
    means_c = means_c + translation[..., None, :]
    covars_c = jnp.einsum("...cij,...njk,...clk->...cnil", rotation, covars, rotation)
    return means_c, covars_c


def _pinhole_mean_and_jacobian_components(
    means: Array,
    Ks: Array,
    width: int,
    height: int,
    eps: float = 1e-8,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return pinhole means and the four nonzero Jacobian components."""

    means = jnp.asarray(means)
    Ks = jnp.asarray(Ks)
    x, y, z = jnp.moveaxis(means, -1, 0)
    z_safe = _safe_denominator(z, eps)
    inv_z = jnp.reciprocal(z_safe)
    inv_z2 = inv_z * inv_z

    fx = Ks[..., 0, 0, None]
    fy = Ks[..., 1, 1, None]
    cx = Ks[..., 0, 2, None]
    cy = Ks[..., 1, 2, None]
    fx_safe = _safe_denominator(fx, eps)
    fy_safe = _safe_denominator(fy, eps)

    tan_fovx = 0.5 * width / fx_safe
    tan_fovy = 0.5 * height / fy_safe
    lim_x_pos = (width - cx) / fx_safe + 0.3 * tan_fovx
    lim_x_neg = cx / fx_safe + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy_safe + 0.3 * tan_fovy
    lim_y_neg = cy / fy_safe + 0.3 * tan_fovy
    tx = z * jnp.clip(x * inv_z, -lim_x_neg, lim_x_pos)
    ty = z * jnp.clip(y * inv_z, -lim_y_neg, lim_y_pos)

    j00 = fx * inv_z
    j02 = -fx * tx * inv_z2
    j11 = fy * inv_z
    j12 = -fy * ty * inv_z2
    means2d = jnp.stack((fx * x * inv_z + cx, fy * y * inv_z + cy), axis=-1)
    return means2d, j00, j02, j11, j12


def _pinhole_mean_and_jacobian(
    means: Array,
    Ks: Array,
    width: int,
    height: int,
    eps: float = 1e-8,
) -> tuple[Array, Array]:
    """Return pinhole means and the projection Jacobian."""

    means2d, j00, j02, j11, j12 = _pinhole_mean_and_jacobian_components(
        means, Ks, width, height, eps
    )
    zeros = jnp.zeros_like(j00)
    jacobian = jnp.stack((j00, zeros, j02, zeros, j11, j12), axis=-1).reshape(
        means.shape[:-1] + (2, 3)
    )
    return means2d, jacobian


def pinhole_proj(
    means: Array,
    covars: Array,
    Ks: Array,
    width: int,
    height: int,
    eps: float = 1e-8,
) -> tuple[Array, Array]:
    """Project camera-space Gaussians with gsplat's pinhole EWA model."""

    means2d, jacobian = _pinhole_mean_and_jacobian(means, Ks, width, height, eps)
    covars = jnp.asarray(covars)
    covars2d = jnp.einsum("...ij,...jk,...lk->...il", jacobian, covars, jacobian)
    return means2d, covars2d


def _pinhole_proj_factors(
    means: Array,
    factors: Array,
    Ks: Array,
    width: int,
    height: int,
) -> tuple[Array, Array]:
    """Project camera-space covariance factors without forming 3D covariances."""

    means2d, j00, j02, j11, j12 = _pinhole_mean_and_jacobian_components(
        means, Ks, width, height
    )
    factors = jnp.asarray(factors)
    projected_x = (
        j00[..., None] * factors[..., 0, :] + j02[..., None] * factors[..., 2, :]
    )
    projected_y = (
        j11[..., None] * factors[..., 1, :] + j12[..., None] * factors[..., 2, :]
    )
    covar_xx = jnp.sum(projected_x * projected_x, axis=-1)
    covar_xy = jnp.sum(projected_x * projected_y, axis=-1)
    covar_yy = jnp.sum(projected_y * projected_y, axis=-1)
    covars2d = jnp.stack((covar_xx, covar_xy, covar_xy, covar_yy), axis=-1).reshape(
        means.shape[:-1] + (2, 2)
    )
    return means2d, covars2d


def _pinhole_proj_world_factors(
    means: Array,
    world_factors: Array,
    camera_rotation: Array,
    Ks: Array,
    width: int,
    height: int,
) -> tuple[Array, Array]:
    """Project world factors without materializing camera-space 3x3 factors."""

    means2d, j00, j02, j11, j12 = _pinhole_mean_and_jacobian_components(
        means, Ks, width, height
    )
    row_x = (
        j00[..., None] * camera_rotation[..., 0, :][..., None, :]
        + j02[..., None] * camera_rotation[..., 2, :][..., None, :]
    )
    row_y = (
        j11[..., None] * camera_rotation[..., 1, :][..., None, :]
        + j12[..., None] * camera_rotation[..., 2, :][..., None, :]
    )
    projected_x = jnp.einsum("...cni,...nik->...cnk", row_x, world_factors)
    projected_y = jnp.einsum("...cni,...nik->...cnk", row_y, world_factors)
    covar_xx = jnp.sum(projected_x * projected_x, axis=-1)
    covar_xy = jnp.sum(projected_x * projected_y, axis=-1)
    covar_yy = jnp.sum(projected_y * projected_y, axis=-1)
    covars2d = jnp.stack((covar_xx, covar_xy, covar_xy, covar_yy), axis=-1).reshape(
        means.shape[:-1] + (2, 2)
    )
    return means2d, covars2d


pinhole_projection = pinhole_proj


def ortho_proj(
    means: Array,
    covars: Array,
    Ks: Array,
    width: int,
    height: int,
) -> tuple[Array, Array]:
    """Project camera-space Gaussians with an orthographic camera."""

    del width, height
    means = jnp.asarray(means)
    covars = jnp.asarray(covars)
    Ks = jnp.asarray(Ks)
    fx = Ks[..., 0, 0, None]
    fy = Ks[..., 1, 1, None]
    cx = Ks[..., 0, 2, None]
    cy = Ks[..., 1, 2, None]
    zeros = jnp.zeros_like(means[..., 0])
    jacobian = jnp.stack(
        (fx + zeros, zeros, zeros, zeros, fy + zeros, zeros), axis=-1
    ).reshape(means.shape[:-1] + (2, 3))
    covars2d = jnp.einsum("...ij,...jk,...lk->...il", jacobian, covars, jacobian)
    means2d = jnp.stack((fx * means[..., 0] + cx, fy * means[..., 1] + cy), axis=-1)
    return means2d, covars2d


ortho_projection = ortho_proj


def fisheye_proj(
    means: Array,
    covars: Array,
    Ks: Array,
    width: int,
    height: int,
    eps: float = 1e-7,
) -> tuple[Array, Array]:
    """Project Gaussians with gsplat's equidistant fisheye camera model.

    The optical-axis limit is evaluated explicitly, avoiding the singular
    ``atan(r / z) / r`` expression used by a literal implementation.
    """

    del width, height
    means = jnp.asarray(means)
    covars = jnp.asarray(covars)
    Ks = jnp.asarray(Ks)
    x, y, z = jnp.moveaxis(means, -1, 0)
    fx = Ks[..., 0, 0, None]
    fy = Ks[..., 1, 1, None]
    cx = Ks[..., 0, 2, None]
    cy = Ks[..., 1, 2, None]

    r2 = x * x + y * y
    eps_array = jnp.asarray(eps, means.dtype)
    # Keeping epsilon inside the square root also makes the derivative at the
    # optical axis well-defined (sqrt(x**2 + y**2) alone has a NaN gradient).
    r = jnp.sqrt(r2 + eps_array * eps_array)
    r_safe = r
    rho2 = r2 + z * z
    rho2_safe = jnp.maximum(rho2, jnp.asarray(eps * eps, means.dtype))
    theta = jnp.arctan2(r, z)
    z_safe = _safe_denominator(z, eps)
    # gsplat regularizes x^2 with 1e-7 in the Jacobian.  Using the same scale
    # (rather than eps^2) prevents enormous inactive-branch tangents in XLA.
    axis = r2 < eps_array
    scale = jnp.where(axis, jnp.reciprocal(z_safe), theta / r_safe)
    means2d = jnp.stack((fx * x * scale + cx, fy * y * scale + cy), axis=-1)

    # Analytical Jacobian away from the optical axis, plus its pinhole limit.
    r2_safe = jnp.maximum(r2, eps_array)
    theta_jacobian = jnp.where(axis, r_safe / z_safe, theta)
    b = theta_jacobian / (r_safe * r2_safe)
    a = z / (rho2_safe * r2_safe)
    xy = x * y
    j00 = fx * (x * x * a + y * y * b)
    j01 = fx * xy * (a - b)
    j02 = -fx * x / rho2_safe
    j10 = fy * xy * (a - b)
    j11 = fy * (y * y * a + x * x * b)
    j12 = -fy * y / rho2_safe
    inv_z = jnp.reciprocal(z_safe)
    j00 = jnp.where(axis, fx * inv_z, j00)
    j01 = jnp.where(axis, 0.0, j01)
    j02 = jnp.where(axis, 0.0, j02)
    j10 = jnp.where(axis, 0.0, j10)
    j11 = jnp.where(axis, fy * inv_z, j11)
    j12 = jnp.where(axis, 0.0, j12)
    jacobian = jnp.stack((j00, j01, j02, j10, j11, j12), axis=-1)
    jacobian = jacobian.reshape(means.shape[:-1] + (2, 3))
    covars2d = jnp.einsum("...ij,...jk,...lk->...il", jacobian, covars, jacobian)
    return means2d, covars2d


fisheye_projection = fisheye_proj


def persp_proj(
    means: Array,
    covars: Array,
    Ks: Array,
    width: int,
    height: int,
) -> tuple[Array, Array]:
    """Deprecated perspective-projection alias retained by current main."""

    warnings.warn(
        "persp_proj is deprecated and will be removed in a future release. "
        "Use proj with camera_model='pinhole' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return pinhole_proj(means, covars, Ks, width, height)


def proj(
    means: Array,
    covars: Array,
    Ks: Array,
    width: int,
    height: int,
    camera_model: CameraModel = "pinhole",
) -> tuple[Array, Array]:
    """Dispatch to a supported camera projection model."""

    if camera_model == "pinhole":
        return pinhole_proj(means, covars, Ks, width, height)
    if camera_model == "ortho":
        return ortho_proj(means, covars, Ks, width, height)
    if camera_model == "fisheye":
        return fisheye_proj(means, covars, Ks, width, height)
    raise ValueError(f"Unsupported camera model: {camera_model}")


def fully_fused_projection(
    means: Array,
    viewmats: Array,
    Ks: Array,
    width: int,
    height: int,
    *,
    quats: Array | None = None,
    scales: Array | None = None,
    covars: Array | None = None,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
    opacities: Array | None = None,
    active_mask: Array | None = None,
    alpha_threshold: float = 1.0 / 255.0,
) -> tuple[Array, Array, Array, Array, Array | None, Array]:
    """Transform and project a fixed-size Gaussian buffer.

    This is the dense, fixed-shape counterpart of gsplat v1.5.3's fused CUDA
    projection.  ``active_mask`` disables unused slots (for example in a
    one-million-element capacity buffer) without changing shapes or retracing.

    Returns ``(radii, means2d, depths, conics, compensations, valid)``.  Radii
    have dtype ``int32`` and invalid entries are zero.  Compensation is ``None``
    unless ``calc_compensations`` is enabled, matching gsplat.
    """

    means = jnp.asarray(means)
    if (
        covars is None
        and camera_model == "pinhole"
        and means.shape[-2] >= _FACTOR_PROJECTION_MIN_GAUSSIANS
    ):
        if quats is None or scales is None:
            raise ValueError("Provide covars or both quats and scales")
        viewmats = jnp.asarray(viewmats)
        camera_rotation = viewmats[..., :3, :3]
        camera_translation = viewmats[..., :3, 3]
        means_c = jnp.einsum("...cij,...nj->...cni", camera_rotation, means)
        means_c = means_c + camera_translation[..., None, :]
        world_factors = quat_to_rotmat(quats) * jnp.asarray(scales)[..., None, :]
        means2d, covars2d_orig = _pinhole_proj_world_factors(
            means_c,
            world_factors,
            camera_rotation,
            Ks,
            width,
            height,
        )
    else:
        if covars is None:
            if quats is None or scales is None:
                raise ValueError("Provide covars or both quats and scales")
            covars, _ = quat_scale_to_covar_preci(
                quats, scales, compute_covar=True, compute_preci=False
            )
        else:
            covars = jnp.asarray(covars)
            if covars.shape[-1] == 6 and covars.ndim == means.ndim:
                covars = triu_to_full(covars)

        means_c, covars_c = world_to_cam(means, covars, viewmats)
        means2d, covars2d_orig = proj(
            means_c, covars_c, Ks, width, height, camera_model=camera_model
        )

    det_orig = (
        covars2d_orig[..., 0, 0] * covars2d_orig[..., 1, 1]
        - covars2d_orig[..., 0, 1] * covars2d_orig[..., 1, 0]
    )
    identity = jnp.eye(2, dtype=covars2d_orig.dtype)
    covars2d = covars2d_orig + jnp.asarray(eps2d, covars2d_orig.dtype) * identity
    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    safe_det = jnp.maximum(det, jnp.asarray(1e-10, det.dtype))
    compensation_values = jnp.sqrt(jnp.maximum(det_orig / safe_det, 0.0))
    conics = jnp.stack(
        (
            covars2d[..., 1, 1] / safe_det,
            -0.5 * (covars2d[..., 0, 1] + covars2d[..., 1, 0]) / safe_det,
            covars2d[..., 0, 0] / safe_det,
        ),
        axis=-1,
    )

    extend = jnp.full_like(det, 3.33)
    opacity_valid = jnp.ones_like(det, dtype=jnp.bool_)
    if opacities is not None:
        opacity = jnp.asarray(opacities)[..., None, :]
        if calc_compensations:
            opacity = opacity * compensation_values
        alpha_threshold_array = jnp.asarray(alpha_threshold, opacity.dtype)
        opacity_valid = opacity >= alpha_threshold_array
        opacity_ratio = jnp.maximum(opacity / alpha_threshold_array, 1.0)
        opacity_extend = jnp.sqrt(jnp.maximum(2.0 * jnp.log(opacity_ratio), 0.0))
        extend = jnp.minimum(extend, opacity_extend)

    radius_x = jnp.ceil(extend * jnp.sqrt(jnp.maximum(covars2d[..., 0, 0], 0.0)))
    radius_y = jnp.ceil(extend * jnp.sqrt(jnp.maximum(covars2d[..., 1, 1], 0.0)))
    radius = jnp.stack((radius_x, radius_y), axis=-1)
    depths = means_c[..., 2]

    valid = (det > 0.0) & (depths > near_plane) & (depths < far_plane)
    valid = valid & opacity_valid
    valid = valid & ((radius_x > radius_clip) | (radius_y > radius_clip))
    inside = (
        (means2d[..., 0] + radius_x > 0.0)
        & (means2d[..., 0] - radius_x < width)
        & (means2d[..., 1] + radius_y > 0.0)
        & (means2d[..., 1] - radius_y < height)
    )
    valid = valid & inside
    if active_mask is not None:
        valid = valid & jnp.asarray(active_mask, dtype=jnp.bool_)[..., None, :]
    finite = (
        jnp.all(jnp.isfinite(means2d), axis=-1)
        & jnp.isfinite(depths)
        & jnp.all(jnp.isfinite(conics), axis=-1)
    )
    valid = valid & finite

    radii = jnp.where(valid[..., None], radius, 0.0).astype(jnp.int32)
    compensations = compensation_values if calc_compensations else None
    return radii, means2d, depths, conics, compensations, valid
