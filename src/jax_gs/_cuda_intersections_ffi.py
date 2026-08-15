"""Explicit CUDA/XLA FFI+CUB fixed-capacity intersection stages."""

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

_PREFIX_TARGET = "jax_gs_intersection_prefix"
_SORT_OFFSETS_TARGET = "jax_gs_intersection_sort_offsets"
_FUSED_TARGET = "jax_gs_intersection_compositor_pipeline"
_FUSED_BACKWARD_TARGET = "jax_gs_fused_compositor_backward"
_LIBRARY_ENV = "JAX_GS_CUDA_INTERSECTIONS_FFI_LIBRARY"
_CACHE_ENV = "JAX_GS_CUDA_FFI_CACHE_DIR"
_NVCC_ENV = "JAX_GS_NVCC"
_SOURCE = Path(__file__).with_name("cuda_ffi") / "intersections.cu"
_COMPOSITOR_SOURCE = Path(__file__).with_name("cuda_ffi") / "compositor.cu"

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
        raise TypeError(f"{name} must be a static real scalar") from exc


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
            "intersection_backend='cuda_tile_cub' requires a CUDA GPU; use "
            "intersection_backend='jax' on this device"
        ) from exc
    if not devices or any("cuda" not in str(device).lower() for device in devices):
        raise RuntimeError(
            "intersection_backend='cuda_tile_cub' requires NVIDIA CUDA devices"
        )
    capabilities = {_compute_capability(device) for device in devices}
    if len(capabilities) != 1:
        raise RuntimeError(
            "intersection_backend='cuda_tile_cub' currently requires all local "
            "CUDA devices to share one compute capability"
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
            "intersection_backend='cuda_tile_cub' needs nvcc for the first "
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
    return root / "intersections"


def _build_key(nvcc: str, architecture: str) -> str:
    version = subprocess.run(
        [nvcc, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    payload = {
        "sources": [
            hashlib.sha256(source.read_bytes()).hexdigest()
            for source in (_SOURCE, _COMPOSITOR_SOURCE)
        ],
        "jax": jax.__version__,
        "jaxlib_include": jax.ffi.include_dir(),
        "nvcc": version,
        "architecture": architecture,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def _compile_library(device: jax.Device) -> Path:
    for source in (_SOURCE, _COMPOSITOR_SOURCE):
        if not source.is_file():
            raise RuntimeError(
                f"CUDA FFI source is missing from the installation: {source}"
            )
    architecture = _compute_capability(device)
    nvcc = _nvcc_path()
    cache_dir = _cache_root() / _build_key(nvcc, architecture)
    library = cache_dir / "libjax_gs_intersections.so"
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
            prefix="intersections-", suffix=".so", dir=cache_dir
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
            (
                f"-gencode=arch=compute_{architecture},"
                f"code=[sm_{architecture},compute_{architecture}]"
            ),
            f"-I{jax.ffi.include_dir()}",
            str(_SOURCE),
            str(_COMPOSITOR_SOURCE),
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
                    "failed to compile the CUDA FFI+CUB intersection stages:\n"
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
        library.JaxGsIntersectionPrefixWorkspaceBytes.argtypes = [
            ctypes.c_int64
        ]
        library.JaxGsIntersectionPrefixWorkspaceBytes.restype = (
            ctypes.c_uint64
        )
        library.JaxGsIntersectionSortWorkspaceBytes.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        library.JaxGsIntersectionSortWorkspaceBytes.restype = ctypes.c_uint64
        library.JaxGsIntersectionPipelineWorkspaceBytes.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        library.JaxGsIntersectionPipelineWorkspaceBytes.restype = (
            ctypes.c_uint64
        )
        jax.ffi.register_ffi_target(
            _PREFIX_TARGET,
            jax.ffi.pycapsule(library.JaxGsIntersectionPrefix),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(
            _SORT_OFFSETS_TARGET,
            jax.ffi.pycapsule(library.JaxGsIntersectionSortOffsets),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(
            _FUSED_TARGET,
            jax.ffi.pycapsule(library.JaxGsIntersectionCompositorPipeline),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(
            _FUSED_BACKWARD_TARGET,
            jax.ffi.pycapsule(library.JaxGsCompositorBackward),
            platform="CUDA",
        )
        _LIBRARY = library
        _REGISTERED = True


def _checked_workspace_bytes(function_name: str, *arguments: int) -> int:
    if _LIBRARY is None:
        raise RuntimeError("the CUDA intersection FFI library is not loaded")
    try:
        function = getattr(_LIBRARY, function_name)
        workspace_bytes = int(function(*arguments))
    except (
        AttributeError,
        ctypes.ArgumentError,
        OSError,
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise RuntimeError(
            f"failed to query CUDA workspace size with {function_name}"
        ) from exc
    if workspace_bytes == 2**64 - 1:
        raise RuntimeError(
            f"failed to query CUDA workspace size with {function_name}"
        )
    return max(workspace_bytes, 1)


def intersection_prefix_cuda_ffi(
    counts: jax.Array, *, capacity: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute the exact saturated prefix and overflow metadata with CUB."""

    capacity = _static_int("capacity", capacity)
    counts = jnp.asarray(counts)
    if counts.ndim != 1 or counts.dtype != jnp.int32:
        raise ValueError("counts must be a rank-one int32 array")
    if counts.shape[0] == 0:
        raise ValueError("counts must be non-empty")
    _ensure_registered(_cuda_device())
    workspace_bytes = _checked_workspace_bytes(
        "JaxGsIntersectionPrefixWorkspaceBytes", counts.shape[0]
    )
    # XLA FFI ScratchAllocator is unavailable in JAX 0.11's CUDA execution
    # context on this supported stack. Declaring handler-owned workspace as
    # an unused output lets XLA plan/reuse the memory without stream-local
    # cudaMallocAsync calls or hidden persistent device pointers.
    outputs = (
        jax.ShapeDtypeStruct(counts.shape, jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.bool_),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((workspace_bytes,), jnp.uint8),
    )
    cumulative, valid_count, overflow, required_count, _ = jax.ffi.ffi_call(
        _PREFIX_TARGET,
        outputs,
        vmap_method="sequential",
    )(counts, capacity=np.int64(capacity))
    return cumulative, valid_count, overflow, required_count


_STATE_FIELDS = (
    ("valid", jnp.bool_),
    ("is_y", jnp.bool_),
    ("b", jnp.float32),
    ("disc", jnp.float32),
    ("t", jnp.float32),
    ("p_u", jnp.float32),
    ("p_v", jnp.float32),
    ("coefficient", jnp.float32),
    ("outer_min", jnp.int32),
    ("outer_max", jnp.int32),
    ("cross_min", jnp.int32),
    ("cross_max", jnp.int32),
    ("outer_bbox_min", jnp.float32),
    ("outer_bbox_max", jnp.float32),
    ("cross_bbox_min", jnp.float32),
    ("cross_bbox_max", jnp.float32),
    ("argmin_outer", jnp.float32),
    ("argmax_outer", jnp.float32),
)
_SUPPORTED_CHANNELS = frozenset((1, 2, 3, 4, 8, 16, 32))


def _state_arrays(state):
    arrays = tuple(jnp.asarray(getattr(state, name)) for name, _ in _STATE_FIELDS)
    gaussian_count = arrays[0].shape[0]
    if gaussian_count == 0:
        raise ValueError("prepared AccuTile state must be non-empty")
    for (name, dtype), array in zip(_STATE_FIELDS, arrays, strict=True):
        if array.shape != (gaussian_count,) or array.dtype != dtype:
            raise ValueError(
                f"state.{name} must have shape ({gaussian_count},) and dtype {dtype}"
            )
    return arrays


def _run_intersection_compositor_forward(
    state,
    depths,
    means2d,
    conics,
    colors,
    opacities,
    *,
    capacity: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    arrays = _state_arrays(state)
    gaussian_count = arrays[0].shape[0]
    depths = jnp.asarray(depths)
    means2d = jnp.asarray(means2d)
    conics = jnp.asarray(conics)
    colors = jnp.asarray(colors)
    opacities = jnp.asarray(opacities)
    channels = colors.shape[-1]
    if depths.shape != (gaussian_count,) or depths.dtype != jnp.float32:
        raise ValueError("depths must match prepared state and use float32")
    if means2d.shape != (gaussian_count, 2):
        raise ValueError("means2d must have shape [N, 2]")
    if conics.shape != (gaussian_count, 3):
        raise ValueError("conics must have shape [N, 3]")
    if (
        colors.shape != (gaussian_count, channels)
        or channels not in _SUPPORTED_CHANNELS
    ):
        raise ValueError("unsupported fused compositor color shape")
    if opacities.shape != (gaussian_count,):
        raise ValueError("opacities must have shape [N]")
    if any(
        value.dtype != jnp.float32
        for value in (means2d, conics, colors, opacities)
    ):
        raise TypeError("fused intersection/compositor inputs must be float32")

    tile_count = tile_width * tile_height
    _ensure_registered(_cuda_device())
    workspace_bytes = _checked_workspace_bytes(
        "JaxGsIntersectionPipelineWorkspaceBytes",
        gaussian_count,
        capacity,
        tile_count,
    )
    outputs = (
        jax.ShapeDtypeStruct((capacity,), jnp.int32),
        jax.ShapeDtypeStruct((capacity,), jnp.int32),
        jax.ShapeDtypeStruct((tile_count,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.bool_),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((image_height, image_width, channels), jnp.float32),
        jax.ShapeDtypeStruct((image_height, image_width), jnp.float32),
        jax.ShapeDtypeStruct((image_height, image_width), jnp.float32),
        jax.ShapeDtypeStruct((image_height, image_width), jnp.int32),
        jax.ShapeDtypeStruct((tile_count,), jnp.bool_),
        jax.ShapeDtypeStruct((gaussian_count,), jnp.int32),
        jax.ShapeDtypeStruct((16 * capacity,), jnp.uint8),
        jax.ShapeDtypeStruct((workspace_bytes,), jnp.uint8),
    )
    result = jax.ffi.ffi_call(
        _FUSED_TARGET,
        outputs,
        vmap_method="sequential",
    )(
        *arrays,
        depths,
        means2d,
        conics,
        colors,
        opacities,
        capacity=np.int64(capacity),
        tile_size=np.int64(tile_size),
        tile_width=np.int64(tile_width),
        tile_height=np.int64(tile_height),
        image_width=np.int64(image_width),
        image_height=np.int64(image_height),
        per_tile_bound=np.int64(per_tile_bound),
        alpha_threshold=np.float32(alpha_threshold),
        transmittance_threshold=np.float32(transmittance_threshold),
    )
    return result[:11]


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(6, 15)))
def _fused_accutile_composite(
    state,
    depths,
    means2d,
    conics,
    colors,
    opacities,
    capacity: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    (
        g,
        t,
        offsets,
        count,
        overflow,
        required,
        foreground,
        alpha,
        _,
        _,
        tile_overflow,
    ) = _run_intersection_compositor_forward(
        state,
        depths,
        means2d,
        conics,
        colors,
        opacities,
        capacity=capacity,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    return (
        foreground,
        alpha[..., None],
        tile_overflow,
        g,
        t,
        offsets,
        count,
        overflow,
        required,
    )


def _fused_accutile_composite_fwd(
    state,
    depths,
    means2d,
    conics,
    colors,
    opacities,
    capacity,
    tile_size,
    tile_width,
    tile_height,
    image_width,
    image_height,
    per_tile_bound,
    alpha_threshold,
    transmittance_threshold,
):
    result = _run_intersection_compositor_forward(
        state,
        depths,
        means2d,
        conics,
        colors,
        opacities,
        capacity=capacity,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    (
        g,
        t,
        offsets,
        count,
        overflow,
        required,
        foreground,
        alpha,
        accepted,
        last_ids,
        tile_overflow,
    ) = result
    output = (
        foreground,
        alpha[..., None],
        tile_overflow,
        g,
        t,
        offsets,
        count,
        overflow,
        required,
    )
    residuals = (
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        g,
        count,
        accepted,
        last_ids,
    )
    return output, residuals


def _fused_accutile_composite_bwd(
    capacity,
    tile_size,
    tile_width,
    tile_height,
    image_width,
    image_height,
    per_tile_bound,
    alpha_threshold,
    transmittance_threshold,
    residuals,
    cotangents,
):
    del capacity, tile_size, tile_height
    (
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        ids,
        count,
        accepted,
        last_ids,
    ) = residuals
    rendered_cotangent, alpha_cotangent, *_ = cotangents
    from ._cuda_ffi import _run_backward

    gradients = _run_backward(
        means2d,
        conics,
        colors,
        opacities,
        offsets.reshape((-1, tile_width)),
        ids,
        count,
        accepted,
        last_ids,
        rendered_cotangent,
        alpha_cotangent[..., 0],
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
        _target=_FUSED_BACKWARD_TARGET,
    )
    return None, None, *gradients


getattr(_fused_accutile_composite, "defvjp")(
    _fused_accutile_composite_fwd, _fused_accutile_composite_bwd
)


def rasterize_accutile_cuda_ffi_fused(
    state,
    depths,
    means2d,
    conics,
    colors,
    opacities,
    *,
    capacity: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    background,
    max_gaussians_per_tile: int,
    max_candidates_per_tile: int | None,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    capacity = _static_int("capacity", capacity, minimum=1)
    tile_size = _static_int("tile_size", tile_size, minimum=1)
    tile_width = _static_int("tile_width", tile_width, minimum=1)
    tile_height = _static_int("tile_height", tile_height, minimum=1)
    if tile_size != 16:
        raise ValueError("the fused CUDA compositor requires tile_size=16")
    image_width = _static_int("image_width", image_width, minimum=1)
    image_height = _static_int("image_height", image_height, minimum=1)
    max_gaussians_per_tile = _static_int(
        "max_gaussians_per_tile", max_gaussians_per_tile, minimum=1
    )
    if (
        tile_width * tile_size < image_width
        or tile_height * tile_size < image_height
        or tile_width * tile_size >= image_width + tile_size
        or tile_height * tile_size >= image_height + tile_size
    ):
        raise ValueError("the fused tile grid must cover the image exactly")
    gaussian_count = means2d.shape[0]
    per_tile_bound = min(gaussian_count, capacity)
    if max_candidates_per_tile is not None:
        per_tile_bound = min(
            per_tile_bound,
            _static_int(
                "max_candidates_per_tile",
                max_candidates_per_tile,
                minimum=1,
            ),
        )
    per_tile_bound = (
        math.ceil(per_tile_bound / max_gaussians_per_tile)
        * max_gaussians_per_tile
    )
    alpha_threshold = _static_float("alpha_threshold", alpha_threshold)
    transmittance_threshold = _static_float(
        "transmittance_threshold", transmittance_threshold
    )
    result: object = _fused_accutile_composite(
        state,
        depths,
        means2d,
        conics,
        colors,
        opacities,
        capacity,
        tile_size,
        tile_width,
        tile_height,
        image_width,
        image_height,
        per_tile_bound,
        alpha_threshold,
        transmittance_threshold,
    )
    (
        foreground,
        alpha,
        tile_overflow,
        g,
        t,
        offsets,
        count,
        overflow,
        required,
    ) = result  # pyright: ignore[reportGeneralTypeIssues]
    background = jnp.asarray(background)
    if (
        background.shape != (colors.shape[-1],)
        or background.dtype != jnp.float32
    ):
        raise ValueError("background must match the float32 color channels")
    rendered = foreground + background[None, None, :] * (1.0 - alpha)
    return rendered, alpha, {
        "tile_overflow": tile_overflow.reshape((tile_height, tile_width)),
        "gaussian_ids": g,
        "tile_ids": t,
        "offsets": offsets.reshape((tile_height, tile_width)),
        "valid_count": count,
        "overflow": overflow,
        "required_count": required,
    }


def intersection_sort_offsets_cuda_ffi(
    gaussian_ids: jax.Array,
    tile_ids: jax.Array,
    depths: jax.Array,
    valid_count: jax.Array,
    *,
    tile_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sort pairs, build offsets, and return the effective valid prefix."""

    tile_count = _static_int("tile_count", tile_count, minimum=1)
    gaussian_ids = jnp.asarray(gaussian_ids)
    tile_ids = jnp.asarray(tile_ids)
    depths = jnp.asarray(depths)
    valid_count = jnp.asarray(valid_count)
    if gaussian_ids.ndim != 1 or gaussian_ids.dtype != jnp.int32:
        raise ValueError("gaussian_ids must be a rank-one int32 array")
    if tile_ids.shape != gaussian_ids.shape or tile_ids.dtype != jnp.int32:
        raise ValueError("tile_ids must match gaussian_ids and use int32")
    if gaussian_ids.shape[0] == 0:
        raise ValueError("the fixed intersection capacity must be non-zero")
    if depths.ndim != 1 or depths.dtype != jnp.float32:
        raise ValueError("depths must be a rank-one float32 array")
    if depths.shape[0] == 0:
        raise ValueError("depths must be non-empty")
    if valid_count.shape != () or valid_count.dtype != jnp.int32:
        raise ValueError("valid_count must be an int32 scalar")
    _ensure_registered(_cuda_device())
    cub_workspace_bytes = _checked_workspace_bytes(
        "JaxGsIntersectionSortWorkspaceBytes",
        gaussian_ids.shape[0],
        tile_count,
    )
    outputs = (
        jax.ShapeDtypeStruct(gaussian_ids.shape, jnp.int32),
        jax.ShapeDtypeStruct(tile_ids.shape, jnp.int32),
        jax.ShapeDtypeStruct((tile_count,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((16 * gaussian_ids.shape[0],), jnp.uint8),
        jax.ShapeDtypeStruct((cub_workspace_bytes,), jnp.uint8),
    )
    (
        sorted_gaussian_ids,
        sorted_tile_ids,
        offsets,
        effective_valid_count,
        _,
        _,
    ) = jax.ffi.ffi_call(
        _SORT_OFFSETS_TARGET,
        outputs,
        vmap_method="sequential",
    )(
        gaussian_ids,
        tile_ids,
        depths,
        valid_count,
        tile_count=np.int64(tile_count),
    )
    return (
        sorted_gaussian_ids,
        sorted_tile_ids,
        offsets,
        effective_valid_count,
    )
