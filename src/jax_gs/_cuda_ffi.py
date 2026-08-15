"""Explicit CUDA/XLA FFI compositor for single-camera float32 3DGS."""

from __future__ import annotations

import ctypes
from functools import partial
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import numpy as np  # pyright: ignore[reportMissingImports]

import jax  # pyright: ignore[reportMissingImports]
import jax.numpy as jnp  # pyright: ignore[reportMissingImports]

from .low_level import (
    DEFAULT_ALPHA_THRESHOLD,
    DEFAULT_TRANSMITTANCE_THRESHOLD,
)

# Direct atomics intentionally make this explicit backend reduction-order
# nondeterministic at float32 rounding scale. The default JAX/Pallas paths are
# unchanged and retain their existing reduction behavior.
_SUPPORTED_CHANNELS = frozenset((1, 2, 3, 4, 8, 16, 32))
_FORWARD_TARGET = "jax_gs_compositor_forward"
_BACKWARD_TARGET = "jax_gs_compositor_backward"
_LIBRARY_ENV = "JAX_GS_CUDA_FFI_LIBRARY"
_CACHE_ENV = "JAX_GS_CUDA_FFI_CACHE_DIR"
_NVCC_ENV = "JAX_GS_NVCC"
_SOURCE = Path(__file__).with_name("cuda_ffi") / "compositor.cu"

_LOAD_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_REGISTERED = False


def _static_int(name: str, value: int, *, minimum: int = 0) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a static integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _static_float(name: str, value: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a static real scalar") from exc


def _cuda_device() -> jax.Device:
    try:
        devices = jax.local_devices(backend="gpu")
    except RuntimeError as exc:
        raise RuntimeError(
            "compositor_backend='cuda_ffi' requires a CUDA GPU; use "
            "compositor_backend='jax' on this device"
        ) from exc
    if not devices:
        raise RuntimeError(
            "compositor_backend='cuda_ffi' requires a CUDA GPU; use "
            "compositor_backend='jax' on this device"
        )
    if any("cuda" not in str(device).lower() for device in devices):
        raise RuntimeError(
            "compositor_backend='cuda_ffi' only supports NVIDIA CUDA devices"
        )
    capabilities = {_compute_capability(device) for device in devices}
    if len(capabilities) != 1:
        raise RuntimeError(
            "compositor_backend='cuda_ffi' currently requires all local CUDA "
            "devices to share one compute capability"
        )
    return devices[0]


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


def _nvcc_path() -> str:
    configured = os.environ.get(_NVCC_ENV)
    if configured:
        path = shutil.which(configured) if os.path.sep not in configured else configured
    else:
        path = shutil.which("nvcc")
    if not path:
        raise RuntimeError(
            "compositor_backend='cuda_ffi' needs nvcc for the first build. "
            f"Set {_NVCC_ENV} or provide a prebuilt library via {_LIBRARY_ENV}."
        )
    return path


def _cache_root() -> Path:
    configured = os.environ.get(_CACHE_ENV)
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "jax-gs" / "cuda-ffi"
    )


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
    library = cache_dir / "libjax_gs_compositor.so"
    if library.is_file():
        return library
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / "build.lock"
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Linux is the supported CUDA host.
        raise RuntimeError("CUDA FFI runtime compilation requires POSIX file locks") from exc
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if library.is_file():
            return library
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="compositor-", suffix=".so", dir=cache_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        command = [
            nvcc,
            "-std=c++17",
            "-O3",
            "-shared",
            "-Xcompiler=-fPIC",
            f"-gencode=arch=compute_{architecture},code=[sm_{architecture},compute_{architecture}]",
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
                    "failed to compile the CUDA FFI compositor:\n"
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
            _FORWARD_TARGET,
            jax.ffi.pycapsule(library.JaxGsCompositorForward),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(
            _BACKWARD_TARGET,
            jax.ffi.pycapsule(library.JaxGsCompositorBackward),
            platform="CUDA",
        )
        _LIBRARY = library
        _REGISTERED = True


def _call_attributes(
    *,
    image_width: int,
    image_height: int,
    tile_width: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
) -> dict[str, np.generic]:
    return {
        "image_width": np.int64(image_width),
        "image_height": np.int64(image_height),
        "tile_width": np.int64(tile_width),
        "per_tile_bound": np.int64(per_tile_bound),
        "alpha_threshold": np.float32(alpha_threshold),
        "transmittance_threshold": np.float32(transmittance_threshold),
    }


def _run_forward(
    means2d,
    conics,
    colors,
    opacities,
    offsets,
    flatten_ids,
    valid_count,
    *,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    tile_height, tile_width = offsets.shape
    channels = colors.shape[-1]
    outputs = (
        jax.ShapeDtypeStruct(
            (image_height, image_width, channels), jnp.float32
        ),
        jax.ShapeDtypeStruct((image_height, image_width), jnp.float32),
        jax.ShapeDtypeStruct((image_height, image_width), jnp.float32),
        jax.ShapeDtypeStruct((image_height, image_width), jnp.int32),
        jax.ShapeDtypeStruct((tile_height * tile_width,), jnp.bool_),
    )
    attributes = _call_attributes(
        image_width=image_width,
        image_height=image_height,
        tile_width=tile_width,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    return jax.ffi.ffi_call(
        _FORWARD_TARGET,
        outputs,
        vmap_method="sequential",
    )(
        means2d,
        conics,
        colors,
        opacities,
        offsets.reshape(-1),
        flatten_ids,
        valid_count,
        **attributes,
    )


def _run_backward(
    means2d,
    conics,
    colors,
    opacities,
    offsets,
    flatten_ids,
    valid_count,
    accepted_final_transmittance,
    last_ids,
    rendered_cotangent,
    alpha_cotangent,
    *,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
    _target: str = _BACKWARD_TARGET,
):
    tile_width = offsets.shape[1]
    outputs = (
        jax.ShapeDtypeStruct(means2d.shape, jnp.float32),
        jax.ShapeDtypeStruct(conics.shape, jnp.float32),
        jax.ShapeDtypeStruct(colors.shape, jnp.float32),
        jax.ShapeDtypeStruct(opacities.shape, jnp.float32),
    )
    attributes = _call_attributes(
        image_width=image_width,
        image_height=image_height,
        tile_width=tile_width,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    return jax.ffi.ffi_call(
        _target,
        outputs,
        vmap_method="sequential",
    )(
        means2d,
        conics,
        colors,
        opacities,
        offsets.reshape(-1),
        flatten_ids,
        valid_count,
        accepted_final_transmittance,
        last_ids,
        rendered_cotangent,
        alpha_cotangent,
        **attributes,
    )


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11))
def _composite_foreground(
    means2d,
    conics,
    colors,
    opacities,
    offsets,
    flatten_ids,
    valid_count,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    foreground, alpha, _, _, tile_overflow = _run_forward(
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    return foreground, alpha[..., None], tile_overflow


def _composite_foreground_fwd(
    means2d,
    conics,
    colors,
    opacities,
    offsets,
    flatten_ids,
    valid_count,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    (
        foreground,
        alpha,
        accepted_final_transmittance,
        last_ids,
        tile_overflow,
    ) = _run_forward(
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    residuals = (
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        accepted_final_transmittance,
        last_ids,
    )
    return (foreground, alpha[..., None], tile_overflow), residuals


def _composite_foreground_bwd(
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
    residuals,
    cotangents,
):
    (
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        accepted_final_transmittance,
        last_ids,
    ) = residuals
    rendered_cotangent, alpha_cotangent, _ = cotangents
    gradients = _run_backward(
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count,
        accepted_final_transmittance,
        last_ids,
        rendered_cotangent,
        alpha_cotangent[..., 0],
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    return (*gradients, None, None, None)


getattr(_composite_foreground, "defvjp")(
    _composite_foreground_fwd, _composite_foreground_bwd
)


def rasterize_to_pixels_cuda_ffi(
    means2d: jax.Array,
    conics: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: jax.Array,
    flatten_ids: jax.Array,
    backgrounds: jax.Array | None = None,
    *,
    valid_count: jax.Array | int,
    overflow: jax.Array | bool = False,
    max_gaussians_per_tile: int = 512,
    max_candidates_per_tile: int | None = None,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Composite one camera using an explicit, native CUDA/XLA FFI backend."""

    image_width = _static_int("image_width", image_width, minimum=1)
    image_height = _static_int("image_height", image_height, minimum=1)
    tile_size = _static_int("tile_size", tile_size, minimum=1)
    if tile_size != 16:
        raise ValueError("the CUDA FFI compositor currently requires tile_size=16")
    max_gaussians_per_tile = _static_int(
        "max_gaussians_per_tile", max_gaussians_per_tile, minimum=1
    )
    if max_candidates_per_tile is not None:
        max_candidates_per_tile = _static_int(
            "max_candidates_per_tile", max_candidates_per_tile, minimum=1
        )

    means2d = jnp.asarray(means2d)
    conics = jnp.asarray(conics)
    colors = jnp.asarray(colors)
    opacities = jnp.asarray(opacities)
    offsets = jnp.asarray(isect_offsets, jnp.int32)
    flatten_ids = jnp.asarray(flatten_ids, jnp.int32)
    valid_count_array = jnp.asarray(valid_count, jnp.int32)
    gaussian_count = means2d.shape[0]
    channels = colors.shape[-1]
    input_capacity = flatten_ids.shape[0]

    if gaussian_count == 0 or input_capacity == 0:
        raise ValueError(
            "the CUDA FFI compositor requires non-empty fixed-capacity inputs"
        )
    if means2d.shape != (gaussian_count, 2):
        raise ValueError("means2d must have shape [N, 2]")
    if conics.shape != (gaussian_count, 3):
        raise ValueError("conics must have shape [N, 3]")
    if colors.shape != (gaussian_count, channels):
        raise ValueError("colors must have shape [N, channels]")
    if channels not in _SUPPORTED_CHANNELS:
        raise ValueError(
            "the CUDA FFI compositor supports channel counts "
            f"{sorted(_SUPPORTED_CHANNELS)}, got {channels}"
        )
    if opacities.shape != (gaussian_count,):
        raise ValueError("opacities must have shape [N]")
    if offsets.ndim != 2:
        raise ValueError("isect_offsets must have shape [tile_height, tile_width]")
    if flatten_ids.ndim != 1:
        raise ValueError("flatten_ids must be one-dimensional")
    if valid_count_array.shape != ():
        raise ValueError("valid_count must be a scalar")
    if any(
        value.dtype != jnp.float32
        for value in (means2d, conics, colors, opacities)
    ):
        raise TypeError("the CUDA FFI compositor requires float32 inputs")
    if offsets.dtype != jnp.int32 or flatten_ids.dtype != jnp.int32:
        raise TypeError("the CUDA FFI compositor requires int32 metadata")

    tile_height, tile_width = offsets.shape
    if (
        tile_width * tile_size < image_width
        or tile_height * tile_size < image_height
    ):
        raise ValueError("isect_offsets tile grid does not cover the image")
    if tile_height * tile_size >= image_height + tile_size:
        raise ValueError("isect_offsets has extra tile rows beyond the image")
    if tile_width * tile_size >= image_width + tile_size:
        raise ValueError("isect_offsets has extra tile columns beyond the image")
    if backgrounds is None:
        background_array = jnp.zeros((channels,), jnp.float32)
    else:
        background_array = jnp.asarray(backgrounds)
        if background_array.shape != (channels,):
            raise ValueError("backgrounds must have shape [channels]")
        if background_array.dtype != jnp.float32:
            raise TypeError("the CUDA FFI compositor requires float32 inputs")

    device = _cuda_device()
    _ensure_registered(device)
    per_tile_bound = min(gaussian_count, input_capacity)
    if max_candidates_per_tile is not None:
        per_tile_bound = min(per_tile_bound, max_candidates_per_tile)
    per_tile_bound = (
        math.ceil(per_tile_bound / max_gaussians_per_tile)
        * max_gaussians_per_tile
    )
    foreground, alphas, tile_overflow = _composite_foreground(
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        flatten_ids,
        valid_count_array,
        image_width,
        image_height,
        per_tile_bound,
        _static_float("alpha_threshold", alpha_threshold),
        _static_float("transmittance_threshold", transmittance_threshold),
    )
    rendered = foreground + background_array[None, None, :] * (1.0 - alphas)
    tile_overflow = tile_overflow.reshape(offsets.shape)
    return rendered, alphas, {
        "tile_overflow": tile_overflow,
        "overflow": jnp.asarray(overflow, jnp.bool_) | jnp.any(tile_overflow),
    }


__all__ = ["rasterize_to_pixels_cuda_ffi"]
