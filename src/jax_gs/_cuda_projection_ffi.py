# pyright: reportMissingImports=false

"""Optional strict float32 pinhole projection through CUDA/XLA FFI."""

from __future__ import annotations

import ctypes
from functools import partial
import hashlib
import json
import operator
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import numpy as np

import jax
import jax.numpy as jnp

from .cameras import (
    _pinhole_mean_and_jacobian_components,
    fully_fused_projection,
    world_to_cam,
)
from .math import quat_scale_to_covar_preci

_TARGET = "jax_gs_projection_forward"
_FACTOR_PROJECTION_MIN_GAUSSIANS = 262_144
_LIBRARY_ENV = "JAX_GS_CUDA_PROJECTION_FFI_LIBRARY"
_CACHE_ENV = "JAX_GS_CUDA_FFI_CACHE_DIR"
_NVCC_ENV = "JAX_GS_NVCC"
_SOURCE = Path(__file__).with_name("cuda_ffi") / "projection.cu"

_LOAD_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_REGISTERED = False


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


def _compute_capability(device: jax.Device) -> str:
    value = getattr(device, "compute_capability", None)
    if value is None:
        raise RuntimeError("the CUDA device does not report a compute capability")
    try:
        if isinstance(value, tuple):
            major, minor = value
            return f"{int(major)}{int(minor)}"
        numeric = float(value)
        major = int(numeric)
        minor = int(round((numeric - major) * 10))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"invalid CUDA compute capability reported by {device}: {value!r}"
        ) from exc
    return f"{major}{minor}"


def _cuda_device() -> jax.Device:
    try:
        devices = jax.local_devices(backend="gpu")
    except RuntimeError as exc:
        raise RuntimeError(
            "projection_backend='cuda_ffi_strict' requires a CUDA GPU; use "
            "projection_backend='jax' on this device"
        ) from exc
    if not devices or any("cuda" not in str(device).lower() for device in devices):
        raise RuntimeError(
            "projection_backend='cuda_ffi_strict' requires NVIDIA CUDA devices"
        )
    capabilities = {_compute_capability(device) for device in devices}
    if len(capabilities) != 1:
        raise RuntimeError(
            "projection_backend='cuda_ffi_strict' currently requires all "
            "local CUDA devices to share one compute capability"
        )
    return devices[0]


def _nvcc_path() -> str:
    configured = os.environ.get(_NVCC_ENV)
    if configured:
        path = shutil.which(configured) if os.path.sep not in configured else configured
    else:
        path = shutil.which("nvcc")
    if not path:
        raise RuntimeError(
            "projection_backend='cuda_ffi_strict' needs nvcc for the first "
            f"build. Set {_NVCC_ENV} or provide a prebuilt library via "
            f"{_LIBRARY_ENV}."
        )
    return path


def _cache_root() -> Path:
    configured = os.environ.get(_CACHE_ENV)
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "jax-gs" / "cuda-ffi"
    )
    return root / "projection"


def _build_key(nvcc: str, architecture: str) -> str:
    version = subprocess.run(
        [nvcc, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    payload = {
        "source": hashlib.sha256(_SOURCE.read_bytes()).hexdigest(),
        "jax": jax.__version__,
        "jaxlib_include": jax.ffi.include_dir(),
        "nvcc": version,
        "architecture": architecture,
        "compile_flags": ["--fmad=false"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def _compile_library(device: jax.Device) -> Path:
    if not _SOURCE.is_file():
        raise RuntimeError(
            f"CUDA FFI source is missing from the installation: {_SOURCE}"
        )
    architecture = _compute_capability(device)
    nvcc = _nvcc_path()
    cache_dir = _cache_root() / _build_key(nvcc, architecture)
    library = cache_dir / "libjax_gs_projection.so"
    if library.is_file():
        return library
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / "build.lock"
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Linux is the CUDA host.
        raise RuntimeError(
            "CUDA FFI runtime compilation requires POSIX file locks"
        ) from exc
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if library.is_file():
            return library
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="projection-", suffix=".so", dir=cache_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        command = [
            nvcc,
            "-std=c++17",
            "-O3",
            "--fmad=false",
            "-shared",
            "-Xcompiler=-fPIC",
            (
                f"-gencode=arch=compute_{architecture},"
                f"code=[sm_{architecture},compute_{architecture}]"
            ),
            f"-I{jax.ffi.include_dir()}",
            str(_SOURCE),
            "-o",
            str(temporary),
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(
                    "failed to compile the strict CUDA projection FFI:\n"
                    + result.stdout[-12000:]
                )
            os.replace(temporary, library)
        finally:
            temporary.unlink(missing_ok=True)
    return library


def _library_path(device: jax.Device) -> Path:
    configured = os.environ.get(_LIBRARY_ENV)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise RuntimeError(f"{_LIBRARY_ENV} does not name a file: {path}")
        return path
    return _compile_library(device)


def _ensure_registered(device: jax.Device) -> None:
    global _LIBRARY, _REGISTERED
    if _REGISTERED:
        return
    with _LOAD_LOCK:
        if _REGISTERED:
            return
        library = ctypes.CDLL(str(_library_path(device)))
        jax.ffi.register_ffi_target(
            _TARGET,
            jax.ffi.pycapsule(library.JaxGsProjectionForward),
            platform="CUDA",
        )
        _LIBRARY = library
        _REGISTERED = True


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
        raise ValueError("strict CUDA projection inputs must use float32")
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
    gaussian_count = means.shape[0]
    covariances, _ = quat_scale_to_covar_preci(
        quaternions,
        scales,
        compute_covar=True,
        compute_preci=False,
    )
    assert covariances is not None
    means_camera, covariances_camera = world_to_cam(
        means, covariances, viewmats
    )
    means2d, _, _, _, _ = _pinhole_mean_and_jacobian_components(
        means_camera,
        intrinsics,
        image_width,
        image_height,
    )
    depths = means_camera[..., 2]
    compensations = jnp.zeros_like(depths)
    _ensure_registered(_cuda_device())
    output_types = (
        jax.ShapeDtypeStruct((camera_count, gaussian_count, 2), jnp.int32),
        jax.ShapeDtypeStruct((camera_count, gaussian_count, 3), jnp.float32),
        jax.ShapeDtypeStruct((camera_count, gaussian_count), jnp.bool_),
    )
    radii, conics, valid = jax.ffi.ffi_call(
        _TARGET,
        output_types,
        vmap_method="sequential",
    )(
        means_camera,
        means2d,
        covariances_camera,
        opacities,
        active_mask,
        intrinsics,
        image_width=np.int64(image_width),
        image_height=np.int64(image_height),
        eps2d=np.float32(eps2d),
        near_plane=np.float32(near_plane),
        far_plane=np.float32(far_plane),
        radius_clip=np.float32(radius_clip),
        alpha_threshold=np.float32(alpha_threshold),
    )
    return radii, means2d, depths, conics, compensations, valid


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
    residuals = (
        means,
        quaternions,
        scales,
        opacities,
        active_mask,
        viewmats,
        intrinsics,
    )
    return outputs, residuals


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
    (
        means,
        quaternions,
        scales,
        opacities,
        active_mask,
        viewmats,
        intrinsics,
    ) = residuals
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


getattr(_projection, "defvjp")(_projection_fwd, _projection_bwd)


def fully_fused_projection_cuda_ffi(
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
    """Project float32 pinhole 3DGS inputs with strict native topology.

    Camera-space preparation stays in authoritative JAX and dense pinhole
    projection runs natively. Reverse mode recomputes the complete JAX
    projection. Antialiased rendering and the large-shape factor route remain
    fully JAX because they use different topology-sensitive contraction paths.
    """

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
    if (
        calc_compensations
        or means.shape[0] >= _FACTOR_PROJECTION_MIN_GAUSSIANS
    ):
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
