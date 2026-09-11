from __future__ import annotations

"""Strict float32 pinhole projection through NVIDIA cuTile Python."""

import operator
from functools import partial

import cuda.tile as ct
import cuda.tile.jax as ctj
import jax
import jax.numpy as jnp

from .cameras import (
    _pinhole_mean_and_jacobian,
    fully_fused_projection,
    world_to_cam,
)
from .math import quat_scale_to_covar_preci

_BLOCK_SIZE = 128


def _static_int(name: str, value: int, *, minimum: int = 0) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a static integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _static_float(name: str, value: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a static float") from exc


def _require_cuda_tile_device() -> jax.Device:
    try:
        devices = jax.local_devices(backend="gpu")
    except RuntimeError as exc:
        raise RuntimeError(
            "projection_backend='cuda_tile' requires an NVIDIA CUDA GPU"
        ) from exc
    if not devices or any("cuda" not in str(device).lower() for device in devices):
        raise RuntimeError(
            "projection_backend='cuda_tile' requires NVIDIA CUDA devices"
        )
    capabilities = {getattr(device, "compute_capability", None) for device in devices}
    if len(capabilities) != 1 or None in capabilities:
        raise RuntimeError(
            "projection_backend='cuda_tile' requires one shared compute capability"
        )
    return devices[0]


@ct.kernel
def _projection_forward_kernel(
    means_camera,
    means2d,
    covariances_camera,
    opacities,
    active_mask,
    intrinsics,
    reference_covariances_2d,
    reference_determinants,
    radii,
    conics_out,
    compensations_out,
    valid_out,
    camera_count: ct.Constant[int],
    gaussian_count: ct.Constant[int],
    image_width: ct.Constant[int],
    image_height: ct.Constant[int],
    eps2d: ct.Constant[float],
    near_plane: ct.Constant[float],
    far_plane: ct.Constant[float],
    radius_clip: ct.Constant[float],
    alpha_threshold: ct.Constant[float],
    calc_compensations: ct.Constant[bool],
):
    rank = ct.arange(_BLOCK_SIZE, start=ct.bid(0) * _BLOCK_SIZE, dtype=ct.int32)
    count = camera_count * gaussian_count
    in_bounds = rank < count
    safe_rank = ct.minimum(rank, count - 1)
    camera_id = safe_rank // gaussian_count
    gaussian_id = safe_rank - camera_id * gaussian_count

    mean_x = ct.gather(means_camera, (camera_id, gaussian_id, 0), mask=in_bounds)
    mean_y = ct.gather(means_camera, (camera_id, gaussian_id, 1), mask=in_bounds)
    mean_z = ct.gather(means_camera, (camera_id, gaussian_id, 2), mask=in_bounds)
    z_safe = ct.where(
        ct.abs(mean_z) < 1.0e-8,
        ct.where(mean_z < 0.0, -1.0e-8, 1.0e-8),
        mean_z,
    )
    inverse_z = 1.0 / z_safe
    inverse_z_squared = inverse_z * inverse_z

    fx = ct.gather(intrinsics, (camera_id, 0, 0), mask=in_bounds)
    fy = ct.gather(intrinsics, (camera_id, 1, 1), mask=in_bounds)
    cx = ct.gather(intrinsics, (camera_id, 0, 2), mask=in_bounds)
    cy = ct.gather(intrinsics, (camera_id, 1, 2), mask=in_bounds)
    fx_safe = ct.where(ct.abs(fx) < 1.0e-8, ct.where(fx < 0.0, -1.0e-8, 1.0e-8), fx)
    fy_safe = ct.where(ct.abs(fy) < 1.0e-8, ct.where(fy < 0.0, -1.0e-8, 1.0e-8), fy)
    inverse_fx = 1.0 / fx_safe
    inverse_fy = 1.0 / fy_safe
    tan_fov_x = 0.5 * image_width * inverse_fx
    tan_fov_y = 0.5 * image_height * inverse_fy
    limit_x_positive = (image_width - cx) * inverse_fx + 0.3 * tan_fov_x
    limit_x_negative = cx * inverse_fx + 0.3 * tan_fov_x
    limit_y_positive = (image_height - cy) * inverse_fy + 0.3 * tan_fov_y
    limit_y_negative = cy * inverse_fy + 0.3 * tan_fov_y
    tx = mean_z * ct.minimum(
        ct.maximum(mean_x * inverse_z, -limit_x_negative, propagate_nan=True),
        limit_x_positive,
        propagate_nan=True,
    )
    ty = mean_z * ct.minimum(
        ct.maximum(mean_y * inverse_z, -limit_y_negative, propagate_nan=True),
        limit_y_positive,
        propagate_nan=True,
    )

    j00 = fx * inverse_z
    j02 = -fx * tx * inverse_z_squared
    j11 = fy * inverse_z
    j12 = -fy * ty * inverse_z_squared

    c00 = ct.gather(covariances_camera, (camera_id, gaussian_id, 0, 0), mask=in_bounds)
    c01 = ct.gather(covariances_camera, (camera_id, gaussian_id, 0, 1), mask=in_bounds)
    c02 = ct.gather(covariances_camera, (camera_id, gaussian_id, 0, 2), mask=in_bounds)
    c10 = ct.gather(covariances_camera, (camera_id, gaussian_id, 1, 0), mask=in_bounds)
    c11 = ct.gather(covariances_camera, (camera_id, gaussian_id, 1, 1), mask=in_bounds)
    c12 = ct.gather(covariances_camera, (camera_id, gaussian_id, 1, 2), mask=in_bounds)
    c20 = ct.gather(covariances_camera, (camera_id, gaussian_id, 2, 0), mask=in_bounds)
    c21 = ct.gather(covariances_camera, (camera_id, gaussian_id, 2, 1), mask=in_bounds)
    c22 = ct.gather(covariances_camera, (camera_id, gaussian_id, 2, 2), mask=in_bounds)

    p00 = j00 * c00 + j02 * c20
    p01 = j00 * c01 + j02 * c21
    p02 = j00 * c02 + j02 * c22
    p10 = j11 * c10 + j12 * c20
    p11 = j11 * c11 + j12 * c21
    p12 = j11 * c12 + j12 * c22

    covariance_xx = p00 * j00 + p02 * j02 + eps2d
    covariance_xy = p01 * j11 + p02 * j12
    covariance_yx = p10 * j00 + p12 * j02
    covariance_yy = p11 * j11 + p12 * j12 + eps2d
    determinant = covariance_xx * covariance_yy - covariance_xy * covariance_yx

    reference_xx = ct.gather(
        reference_covariances_2d,
        (camera_id, gaussian_id, 0, 0),
        mask=in_bounds,
    )
    reference_xy = ct.gather(
        reference_covariances_2d,
        (camera_id, gaussian_id, 0, 1),
        mask=in_bounds,
    )
    reference_yx = ct.gather(
        reference_covariances_2d,
        (camera_id, gaussian_id, 1, 0),
        mask=in_bounds,
    )
    reference_yy = ct.gather(
        reference_covariances_2d,
        (camera_id, gaussian_id, 1, 1),
        mask=in_bounds,
    )
    reference_determinant = ct.gather(
        reference_determinants,
        (camera_id, gaussian_id),
        mask=in_bounds,
    )
    safe_determinant = ct.maximum(reference_determinant, 1.0e-10, propagate_nan=True)
    conic_xx = reference_yy / safe_determinant
    conic_xy = -0.5 * (reference_xy + reference_yx) / safe_determinant
    conic_yy = reference_xx / safe_determinant
    ct.scatter(conics_out, (camera_id, gaussian_id, 0), conic_xx, mask=in_bounds)
    ct.scatter(conics_out, (camera_id, gaussian_id, 1), conic_xy, mask=in_bounds)
    ct.scatter(conics_out, (camera_id, gaussian_id, 2), conic_yy, mask=in_bounds)

    original_determinant = (reference_xx - eps2d) * (
        reference_yy - eps2d
    ) - reference_xy * reference_yx
    compensation = ct.sqrt(
        ct.maximum(
            original_determinant / safe_determinant,
            0.0,
            propagate_nan=True,
        )
    )
    ct.scatter(
        compensations_out,
        (camera_id, gaussian_id),
        compensation,
        mask=in_bounds,
    )

    opacity = ct.gather(opacities, gaussian_id, mask=in_bounds)
    opacity = ct.where(calc_compensations, opacity * compensation, opacity)
    opacity_valid = opacity >= alpha_threshold
    opacity_ratio = ct.maximum(opacity / alpha_threshold, 1.0, propagate_nan=True)
    opacity_extend = ct.sqrt(
        ct.maximum(2.0 * ct.log(opacity_ratio), 0.0, propagate_nan=True)
    )
    extend = ct.minimum(3.33, opacity_extend, propagate_nan=True)
    radius_x = ct.ceil(
        extend * ct.sqrt(ct.maximum(covariance_xx, 0.0, propagate_nan=True))
    )
    radius_y = ct.ceil(
        extend * ct.sqrt(ct.maximum(covariance_yy, 0.0, propagate_nan=True))
    )

    mean2d_x = ct.gather(means2d, (camera_id, gaussian_id, 0), mask=in_bounds)
    mean2d_y = ct.gather(means2d, (camera_id, gaussian_id, 1), mask=in_bounds)
    active = ct.gather(active_mask, gaussian_id, mask=in_bounds) != 0
    finite = (
        (~ct.isnan(mean2d_x))
        & (~ct.isnan(mean2d_y))
        & (~ct.isnan(mean_z))
        & (~ct.isnan(conic_xx))
        & (~ct.isnan(conic_xy))
        & (~ct.isnan(conic_yy))
        & (ct.abs(mean2d_x) != float("inf"))
        & (ct.abs(mean2d_y) != float("inf"))
        & (ct.abs(mean_z) != float("inf"))
        & (ct.abs(conic_xx) != float("inf"))
        & (ct.abs(conic_xy) != float("inf"))
        & (ct.abs(conic_yy) != float("inf"))
    )
    valid = (
        in_bounds
        & (determinant > 0.0)
        & (mean_z > near_plane)
        & (mean_z < far_plane)
        & opacity_valid
        & ((radius_x > radius_clip) | (radius_y > radius_clip))
        & (mean2d_x + radius_x > 0.0)
        & (mean2d_x - radius_x < image_width)
        & (mean2d_y + radius_y > 0.0)
        & (mean2d_y - radius_y < image_height)
        & active
        & finite
    )
    zero = ct.zeros((_BLOCK_SIZE,), ct.int32)
    ct.scatter(
        radii,
        (camera_id, gaussian_id, 0),
        ct.where(valid, ct.astype(radius_x, ct.int32), zero),
        mask=in_bounds,
    )
    ct.scatter(
        radii,
        (camera_id, gaussian_id, 1),
        ct.where(valid, ct.astype(radius_y, ct.int32), zero),
        mask=in_bounds,
    )
    ct.scatter(
        valid_out,
        (camera_id, gaussian_id),
        ct.astype(valid, ct.uint8),
        mask=in_bounds,
    )


def _validate_inputs(
    means: jax.Array,
    quaternions: jax.Array,
    scales: jax.Array,
    opacities: jax.Array,
    active_mask: jax.Array,
    viewmats: jax.Array,
    intrinsics: jax.Array,
) -> int:
    arrays = (means, quaternions, scales, opacities, viewmats, intrinsics)
    if any(array.dtype != jnp.float32 for array in arrays):
        raise ValueError("strict cuTile projection inputs must use float32")
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape [N, 3]")
    gaussian_count = means.shape[0]
    if gaussian_count == 0:
        raise ValueError("means must be non-empty")
    if quaternions.shape != (gaussian_count, 4):
        raise ValueError("quaternions must have shape [N, 4]")
    if scales.shape != (gaussian_count, 3):
        raise ValueError("scales must have shape [N, 3]")
    if opacities.shape != (gaussian_count,):
        raise ValueError("opacities must have shape [N]")
    if active_mask.dtype != jnp.bool_ or active_mask.shape != (gaussian_count,):
        raise ValueError("active_mask must be a bool array with shape [N]")
    if viewmats.ndim != 3 or viewmats.shape[1:] != (4, 4):
        raise ValueError("viewmats must have shape [C, 4, 4]")
    camera_count = viewmats.shape[0]
    if camera_count == 0:
        raise ValueError("viewmats must be non-empty")
    if intrinsics.shape != (camera_count, 3, 3):
        raise ValueError("intrinsics must have shape [C, 3, 3]")
    return camera_count


def _run_forward(
    means,
    quaternions,
    scales,
    opacities,
    active_mask,
    viewmats,
    intrinsics,
    image_width: int,
    image_height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    alpha_threshold: float,
    calc_compensations: bool,
):
    means = jnp.asarray(means)
    quaternions = jnp.asarray(quaternions)
    scales = jnp.asarray(scales)
    opacities = jnp.asarray(opacities)
    active_mask = jnp.asarray(active_mask)
    viewmats = jnp.asarray(viewmats)
    intrinsics = jnp.asarray(intrinsics)
    camera_count = _validate_inputs(
        means,
        quaternions,
        scales,
        opacities,
        active_mask,
        viewmats,
        intrinsics,
    )
    _require_cuda_tile_device()
    gaussian_count = means.shape[0]
    # Keep the contractions in the topology oracle: a separately compiled
    # cuTile contraction can differ by one ULP and cross ceil(radius).  The
    # cuTile kernel below still owns every discrete projection decision.
    covariances, _ = quat_scale_to_covar_preci(
        quaternions,
        scales,
        compute_covar=True,
        compute_preci=False,
    )
    assert covariances is not None
    means_camera, covariances_camera = world_to_cam(means, covariances, viewmats)
    means2d, jacobian = _pinhole_mean_and_jacobian(
        means_camera,
        intrinsics,
        image_width,
        image_height,
    )
    covariances_2d = jnp.einsum(
        "...ij,...jk,...lk->...il",
        jacobian,
        covariances_camera,
        jacobian,
    )
    covariances_2d = covariances_2d + jnp.asarray(
        eps2d, covariances_2d.dtype
    ) * jnp.eye(2, dtype=covariances_2d.dtype)
    determinant = (
        covariances_2d[..., 0, 0] * covariances_2d[..., 1, 1]
        - covariances_2d[..., 0, 1] * covariances_2d[..., 1, 0]
    )
    depths = means_camera[..., 2]
    radii_output = ctj.OutputPlaceholder((camera_count, gaussian_count, 2), jnp.int32)
    conics_output = ctj.OutputPlaceholder(
        (camera_count, gaussian_count, 3), jnp.float32
    )
    compensations_output = ctj.OutputPlaceholder(
        (camera_count, gaussian_count), jnp.float32
    )
    valid_output = ctj.OutputPlaceholder((camera_count, gaussian_count), jnp.uint8)
    radii, native_conics, compensations, valid = ctj.cutile_call(
        ((camera_count * gaussian_count + _BLOCK_SIZE - 1) // _BLOCK_SIZE,),
        _projection_forward_kernel,
        (
            means_camera,
            means2d,
            covariances_camera,
            opacities,
            active_mask.astype(jnp.uint8),
            intrinsics,
            covariances_2d,
            determinant,
            radii_output,
            conics_output,
            compensations_output,
            valid_output,
            camera_count,
            gaussian_count,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            alpha_threshold,
            calc_compensations,
        ),
    )
    return (
        radii,
        means2d,
        depths,
        native_conics,
        compensations,
        valid.astype(jnp.bool_),
    )


def _reference_continuous_outputs(
    means,
    quaternions,
    scales,
    opacities,
    active_mask,
    viewmats,
    intrinsics,
    image_width: int,
    image_height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    alpha_threshold: float,
    calc_compensations: bool,
):
    _, means2d, depths, conics, compensations, _ = fully_fused_projection(
        means,
        viewmats,
        intrinsics,
        image_width,
        image_height,
        quats=quaternions,
        scales=scales,
        opacities=opacities,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        calc_compensations=calc_compensations,
        alpha_threshold=alpha_threshold,
        active_mask=active_mask,
    )
    if compensations is None:
        compensations = jnp.zeros_like(depths)
    return means2d, depths, conics, compensations


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(7, 15)))
def _projection(
    means,
    quaternions,
    scales,
    opacities,
    active_mask,
    viewmats,
    intrinsics,
    image_width: int,
    image_height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    alpha_threshold: float,
    calc_compensations: bool,
):
    return _run_forward(
        means,
        quaternions,
        scales,
        opacities,
        active_mask,
        viewmats,
        intrinsics,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        alpha_threshold,
        calc_compensations,
    )


def _projection_fwd(
    means,
    quaternions,
    scales,
    opacities,
    active_mask,
    viewmats,
    intrinsics,
    image_width: int,
    image_height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    alpha_threshold: float,
    calc_compensations: bool,
):
    outputs = _run_forward(
        means,
        quaternions,
        scales,
        opacities,
        active_mask,
        viewmats,
        intrinsics,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        alpha_threshold,
        calc_compensations,
    )
    return outputs, (
        means,
        quaternions,
        scales,
        opacities,
        active_mask,
        viewmats,
        intrinsics,
    )


def _projection_bwd(
    image_width: int,
    image_height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    alpha_threshold: float,
    calc_compensations: bool,
    residuals,
    cotangents,
):
    means, quaternions, scales, opacities, active_mask, viewmats, intrinsics = residuals
    (
        _,
        means2d_cotangent,
        depths_cotangent,
        conics_cotangent,
        compensation_cotangent,
        _,
    ) = cotangents

    def reference(means_, quaternions_, scales_, opacities_, viewmats_, intrinsics_):
        return _reference_continuous_outputs(
            means_,
            quaternions_,
            scales_,
            opacities_,
            active_mask,
            viewmats_,
            intrinsics_,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            alpha_threshold,
            calc_compensations,
        )

    _, pullback = jax.vjp(
        reference,
        means,
        quaternions,
        scales,
        opacities,
        viewmats,
        intrinsics,
    )
    gradients = pullback(
        (
            means2d_cotangent,
            depths_cotangent,
            conics_cotangent,
            compensation_cotangent,
        )
    )
    return (*gradients[:4], None, *gradients[4:])


_projection.defvjp(_projection_fwd, _projection_bwd)  # type: ignore


def fully_fused_projection_cutile(
    means: jax.Array,
    viewmats: jax.Array,
    intrinsics: jax.Array,
    image_width: int,
    image_height: int,
    *,
    quats: jax.Array,
    scales: jax.Array,
    opacities: jax.Array,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1.0e10,
    radius_clip: float = 0.0,
    calc_compensations: bool = False,
    alpha_threshold: float = 1.0 / 255.0,
    active_mask: jax.Array,
):
    """Project float32 pinhole 3DGS inputs with NVIDIA cuTile Python."""

    image_width = _static_int("image_width", image_width, minimum=1)
    image_height = _static_int("image_height", image_height, minimum=1)
    eps2d = _static_float("eps2d", eps2d)
    near_plane = _static_float("near_plane", near_plane)
    far_plane = _static_float("far_plane", far_plane)
    radius_clip = _static_float("radius_clip", radius_clip)
    alpha_threshold = _static_float("alpha_threshold", alpha_threshold)
    calc_compensations = bool(calc_compensations)
    means = jnp.asarray(means)
    quats = jnp.asarray(quats)
    scales = jnp.asarray(scales)
    opacities = jnp.asarray(opacities)
    active_mask = jnp.asarray(active_mask)
    viewmats = jnp.asarray(viewmats)
    intrinsics = jnp.asarray(intrinsics)
    _validate_inputs(
        means,
        quats,
        scales,
        opacities,
        active_mask,
        viewmats,
        intrinsics,
    )
    radii, means2d, depths, conics, compensations, valid = _projection(
        means,
        quats,
        scales,
        opacities,
        active_mask,
        viewmats,
        intrinsics,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        alpha_threshold,
        calc_compensations,
    )
    return (
        radii,
        means2d,
        depths,
        conics,
        compensations if calc_compensations else None,
        valid,
    )


__all__ = ["fully_fused_projection_cutile"]
