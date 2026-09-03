# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

"""Strict float32 pinhole projection through CuTe DSL."""

from __future__ import annotations

import operator
from functools import partial
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp

from .cameras import (
    _pinhole_mean_and_jacobian,
    fully_fused_projection,
    world_to_cam,
)
from .math import quat_scale_to_covar_preci

_BLOCK_SIZE = 256
_FACTOR_PROJECTION_MIN_GAUSSIANS = 262_144


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


def _cute_device() -> Any:
    try:
        devices = jax.local_devices(backend="gpu")
    except RuntimeError as exc:
        raise RuntimeError(
            "projection_backend='cute' requires an NVIDIA CUDA GPU"
        ) from exc
    if not devices or any("cuda" not in str(device).lower() for device in devices):
        raise RuntimeError("projection_backend='cute' requires NVIDIA CUDA devices")
    capabilities = {getattr(device, "compute_capability", None) for device in devices}
    if len(capabilities) != 1 or None in capabilities:
        raise RuntimeError(
            "projection_backend='cute' requires one shared compute capability"
        )
    return devices[0]


@cute.jit
def _jax_max(left, right):
    result = left if left > right else right
    if cute.math.isnan(left) or cute.math.isnan(right):
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        result = cutlass.Float32(float("nan"))
    return result


@cute.jit
def _jax_min(left, right):
    result = left if left < right else right
    if cute.math.isnan(left) or cute.math.isnan(right):
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        result = cutlass.Float32(float("nan"))
    return result


@cute.jit
def _safe_denominator(value):
    result = value
    if cute.math.absf(value) < 1.0e-8:
        result = cutlass.Float32(-1.0e-8 if value < 0.0 else 1.0e-8)
    return result


@cute.kernel
def _projection_forward_kernel(
    means_camera: cute.Tensor,
    means2d: cute.Tensor,
    covariances_camera: cute.Tensor,
    opacities: cute.Tensor,
    active_mask: cute.Tensor,
    intrinsics: cute.Tensor,
    reference_covariances_2d: cute.Tensor,
    reference_determinants: cute.Tensor,
    radii: cute.Tensor,
    conics_out: cute.Tensor,
    valid_out: cute.Tensor,
    camera_count: int,
    gaussian_count: int,
    image_width: int,
    image_height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    alpha_threshold: float,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * _BLOCK_SIZE + thread
    count = camera_count * gaussian_count
    if index < count:
        camera_id = index // gaussian_count
        gaussian_id = index - camera_id * gaussian_count

        mean_x = means_camera[camera_id, gaussian_id, 0]
        mean_y = means_camera[camera_id, gaussian_id, 1]
        mean_z = means_camera[camera_id, gaussian_id, 2]
        z_safe = _safe_denominator(mean_z)
        inverse_z = cute.math.div(cutlass.Float32(1.0), z_safe, approx=True)
        inverse_z_squared = inverse_z * inverse_z

        fx = intrinsics[camera_id, 0, 0]
        fy = intrinsics[camera_id, 1, 1]
        cx = intrinsics[camera_id, 0, 2]
        cy = intrinsics[camera_id, 1, 2]
        fx_safe = _safe_denominator(fx)
        fy_safe = _safe_denominator(fy)
        inverse_fx = cute.math.div(cutlass.Float32(1.0), fx_safe, approx=True)
        inverse_fy = cute.math.div(cutlass.Float32(1.0), fy_safe, approx=True)
        tan_fov_x = 0.5 * image_width * inverse_fx
        tan_fov_y = 0.5 * image_height * inverse_fy
        limit_x_positive = (image_width - cx) * inverse_fx + 0.3 * tan_fov_x
        limit_x_negative = cx * inverse_fx + 0.3 * tan_fov_x
        limit_y_positive = (image_height - cy) * inverse_fy + 0.3 * tan_fov_y
        limit_y_negative = cy * inverse_fy + 0.3 * tan_fov_y
        tx = mean_z * _jax_min(
            _jax_max(mean_x * inverse_z, -limit_x_negative),
            limit_x_positive,
        )
        ty = mean_z * _jax_min(
            _jax_max(mean_y * inverse_z, -limit_y_negative),
            limit_y_positive,
        )

        j00 = fx * inverse_z
        j02 = -fx * tx * inverse_z_squared
        j11 = fy * inverse_z
        j12 = -fy * ty * inverse_z_squared

        c00 = covariances_camera[camera_id, gaussian_id, 0, 0]
        c01 = covariances_camera[camera_id, gaussian_id, 0, 1]
        c02 = covariances_camera[camera_id, gaussian_id, 0, 2]
        c10 = covariances_camera[camera_id, gaussian_id, 1, 0]
        c11 = covariances_camera[camera_id, gaussian_id, 1, 1]
        c12 = covariances_camera[camera_id, gaussian_id, 1, 2]
        c20 = covariances_camera[camera_id, gaussian_id, 2, 0]
        c21 = covariances_camera[camera_id, gaussian_id, 2, 1]
        c22 = covariances_camera[camera_id, gaussian_id, 2, 2]

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
        reference_xx = reference_covariances_2d[camera_id, gaussian_id, 0, 0]
        reference_xy = reference_covariances_2d[camera_id, gaussian_id, 0, 1]
        reference_yx = reference_covariances_2d[camera_id, gaussian_id, 1, 0]
        reference_yy = reference_covariances_2d[camera_id, gaussian_id, 1, 1]
        reference_determinant = reference_determinants[camera_id, gaussian_id]
        safe_determinant = _jax_max(reference_determinant, 1.0e-10)
        conic_xx = cute.math.div(reference_yy, safe_determinant, full=True)
        conic_xy = cute.math.div(
            -0.5 * (reference_xy + reference_yx),
            safe_determinant,
            full=True,
        )
        conic_yy = cute.math.div(reference_xx, safe_determinant, full=True)
        conics_out[camera_id, gaussian_id, 0] = conic_xx
        conics_out[camera_id, gaussian_id, 1] = conic_xy
        conics_out[camera_id, gaussian_id, 2] = conic_yy

        opacity = opacities[gaussian_id]
        opacity_valid = opacity >= alpha_threshold
        opacity_ratio = _jax_max(
            cute.math.div(opacity, alpha_threshold, approx=True), 1.0
        )
        opacity_extend = cute.math.sqrt(
            _jax_max(2.0 * cute.math.log(opacity_ratio), 0.0)
        )
        extend = _jax_min(3.33, opacity_extend)
        radius_x = cute.math.ceil(extend * cute.math.sqrt(_jax_max(covariance_xx, 0.0)))
        radius_y = cute.math.ceil(extend * cute.math.sqrt(_jax_max(covariance_yy, 0.0)))

        mean2d_x = means2d[camera_id, gaussian_id, 0]
        mean2d_y = means2d[camera_id, gaussian_id, 1]
        valid = (
            determinant > 0.0
            and mean_z > near_plane
            and mean_z < far_plane
            and opacity_valid
        )
        valid = valid and (radius_x > radius_clip or radius_y > radius_clip)
        valid = (
            valid
            and mean2d_x + radius_x > 0.0
            and mean2d_x - radius_x < image_width
            and mean2d_y + radius_y > 0.0
            and mean2d_y - radius_y < image_height
        )
        valid = valid and active_mask[gaussian_id] != 0
        valid = (
            valid
            and cute.math.isfinite(mean2d_x)
            and cute.math.isfinite(mean2d_y)
            and cute.math.isfinite(mean_z)
            and cute.math.isfinite(conic_xx)
            and cute.math.isfinite(conic_xy)
            and cute.math.isfinite(conic_yy)
        )
        valid_out[camera_id, gaussian_id] = cutlass.Uint8(1 if valid else 0)
        radii[camera_id, gaussian_id, 0] = (
            cutlass.Int32(radius_x) if valid else cutlass.Int32(0)
        )
        radii[camera_id, gaussian_id, 1] = (
            cutlass.Int32(radius_y) if valid else cutlass.Int32(0)
        )


@cute.jit
def _launch_projection_forward(
    stream: cuda.CUstream,
    means_camera: cute.Tensor,
    means2d: cute.Tensor,
    covariances_camera: cute.Tensor,
    opacities: cute.Tensor,
    active_mask: cute.Tensor,
    intrinsics: cute.Tensor,
    reference_covariances_2d: cute.Tensor,
    reference_determinants: cute.Tensor,
    radii: cute.Tensor,
    conics: cute.Tensor,
    valid: cute.Tensor,
    *,
    camera_count: int,
    gaussian_count: int,
    image_width: int,
    image_height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    alpha_threshold: float,
):
    blocks = (camera_count * gaussian_count + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    _projection_forward_kernel(
        means_camera,
        means2d,
        covariances_camera,
        opacities,
        active_mask,
        intrinsics,
        reference_covariances_2d,
        reference_determinants,
        radii,
        conics,
        valid,
        camera_count,
        gaussian_count,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        alpha_threshold,
    ).launch(
        grid=[blocks, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
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
        raise ValueError("strict CuTe projection inputs must use float32")
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
    _cute_device()
    gaussian_count = means.shape[0]
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
    compensations = jnp.zeros_like(depths)
    call = cjax.cutlass_call(
        _launch_projection_forward,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((camera_count, gaussian_count, 2), jnp.int32),
            jax.ShapeDtypeStruct((camera_count, gaussian_count, 3), jnp.float32),
            jax.ShapeDtypeStruct((camera_count, gaussian_count), jnp.uint8),
        ),
        use_static_tensors=True,
        camera_count=camera_count,
        gaussian_count=gaussian_count,
        image_width=image_width,
        image_height=image_height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        alpha_threshold=alpha_threshold,
    )
    radii, native_conics, valid = call(
        means_camera,
        means2d,
        covariances_camera,
        opacities,
        active_mask.astype(jnp.uint8),
        intrinsics,
        covariances_2d,
        determinant,
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


def fully_fused_projection_cute(
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
    """Project float32 pinhole 3DGS inputs with CuTe DSL."""

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
    if calc_compensations or means.shape[0] >= _FACTOR_PROJECTION_MIN_GAUSSIANS:
        return fully_fused_projection(
            means,
            viewmats,
            intrinsics,
            image_width,
            image_height,
            quats=quats,
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


__all__ = ["fully_fused_projection_cute"]
