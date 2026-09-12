from __future__ import annotations

import math
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, overload

import jax
import jax.numpy as jnp
import numpy as np

from .model import GaussianModel

ExportFormat = Literal["ply", "splat", "ply_compressed"]
_SH_C0 = 0.28209479177387814


def _active_splats(
    splats: GaussianModel | Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if isinstance(splats, GaussianModel):
        state = splats.state_dict()
    else:
        state = dict(splats)
    if "features" in state or "colors" in state:
        raise ValueError(
            "appearance splats must bake features/colors to degree-zero SH "
            "before export"
        )
    host = {name: np.asarray(jax.device_get(value)) for name, value in state.items()}
    active = host.pop("active_mask", np.ones((host["means"].shape[0],), bool)).astype(
        bool
    )
    return {name: value[active] for name, value in host.items()}


def _as_numpy(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _sigmoid(values: Any) -> np.ndarray:
    values = _as_numpy(values)
    result = np.empty_like(values, dtype=np.result_type(values, np.float32))
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def sh2rgb(sh: Any) -> np.ndarray:
    """Convert degree-zero spherical harmonics coefficients to RGB."""

    return _as_numpy(sh) * _SH_C0 + 0.5


def part1by2_vec(x: Any) -> np.ndarray:
    """Interleave the low ten bits of each integer with two zero bits."""

    x = _as_numpy(x).astype(np.uint32, copy=False) & np.uint32(0x000003FF)
    x = (x ^ (x << np.uint32(16))) & np.uint32(0xFF0000FF)
    x = (x ^ (x << np.uint32(8))) & np.uint32(0x0300F00F)
    x = (x ^ (x << np.uint32(4))) & np.uint32(0x030C30C3)
    x = (x ^ (x << np.uint32(2))) & np.uint32(0x09249249)
    return x


def encode_morton3_vec(x: Any, y: Any, z: Any) -> np.ndarray:
    """Return 30-bit Morton codes for integer 3D coordinates."""

    return (
        (part1by2_vec(z) << np.uint32(2))
        + (part1by2_vec(y) << np.uint32(1))
        + part1by2_vec(x)
    )


def sort_centers(centers: Any, indices: Any) -> np.ndarray:
    """Order indices by the Morton codes of their corresponding centers."""

    centers = _as_numpy(centers)
    indices = _as_numpy(indices)
    if centers.shape[0] == 0:
        return indices.copy()
    minimum = centers.min(axis=0)
    lengths = centers.max(axis=0) - minimum
    lengths = np.where(lengths == 0, 1, lengths)
    coordinates = np.floor((centers - minimum) / lengths * 1024).astype(np.int32)
    morton = encode_morton3_vec(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    return indices[np.argsort(morton, kind="stable")]


def pack_unorm(value: Any, bits: int) -> np.ndarray:
    """Quantize normalized values into unsigned integers of ``bits`` width."""

    maximum = (1 << bits) - 1
    packed = np.floor(_as_numpy(value) * maximum + 0.5)
    return np.clip(packed, 0, maximum).astype(np.uint32)


def pack_111011(x: Any, y: Any, z: Any) -> np.ndarray:
    """Pack three normalized values into 11, 10, and 11 bits."""

    return (
        (pack_unorm(x, 11) << np.uint32(21))
        | (pack_unorm(y, 10) << np.uint32(11))
        | pack_unorm(z, 11)
    )


def pack_8888(x: Any, y: Any, z: Any, w: Any) -> np.ndarray:
    """Pack four normalized values into a 32-bit integer."""

    return (
        (pack_unorm(x, 8) << np.uint32(24))
        | (pack_unorm(y, 8) << np.uint32(16))
        | (pack_unorm(z, 8) << np.uint32(8))
        | pack_unorm(w, 8)
    )


def pack_rotation(q: Any) -> np.ndarray:
    """Pack normalized quaternions using the Supersplat 2+10+10+10 layout."""

    q = _as_numpy(q).astype(np.float32, copy=True)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    q /= np.maximum(norms, np.finfo(np.float32).tiny)
    largest = np.argmax(np.abs(q), axis=-1)
    rows = np.arange(q.shape[0])
    q[q[rows, largest] < 0] *= -1
    component_indices = np.array(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.intp
    )
    components = q[rows[:, None], component_indices[largest]]
    packed = pack_unorm(components * (math.sqrt(2) * 0.5) + 0.5, 10)
    return (
        (largest.astype(np.uint32) << np.uint32(30))
        | (packed[:, 0] << np.uint32(20))
        | (packed[:, 1] << np.uint32(10))
        | packed[:, 2]
    )


def _normalize(
    values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> np.ndarray:
    """Normalize a chunk, mapping constant components to zero."""

    span = maximum - minimum
    return np.divide(
        values - minimum,
        span,
        out=np.zeros_like(values, dtype=np.float32),
        where=span != 0,
    )


def splat2ply_bytes(
    means: Any,
    scales: Any,
    quats: Any,
    opacities: Any,
    sh0: Any,
    shN: Any,
) -> bytes:
    """Serialize pre-activated Gaussian parameters as standard binary PLY."""

    means = _as_numpy(means)
    scales = _as_numpy(scales)
    quats = _as_numpy(quats)
    opacities = _as_numpy(opacities)
    sh0 = _as_numpy(sh0)
    shN = _as_numpy(shN)
    buffer = BytesIO()
    buffer.write(b"ply\nformat binary_little_endian 1.0\n")
    buffer.write(f"element vertex {means.shape[0]}\n".encode())
    for name in ("x", "y", "z"):
        buffer.write(f"property float {name}\n".encode())
    for prefix, values in (("f_dc", sh0), ("f_rest", shN)):
        for index in range(values.shape[1]):
            buffer.write(f"property float {prefix}_{index}\n".encode())
    buffer.write(b"property float opacity\n")
    for index in range(scales.shape[1]):
        buffer.write(f"property float scale_{index}\n".encode())
    for index in range(quats.shape[1]):
        buffer.write(f"property float rot_{index}\n".encode())
    buffer.write(b"end_header\n")
    data = np.concatenate([means, sh0, shN, opacities[:, None], scales, quats], axis=1)
    buffer.write(data.astype("<f4", copy=False).tobytes())
    return buffer.getvalue()


def load_ply_to_splats(path: str | Path) -> dict[str, jax.Array]:
    """Load a standard 3DGS PLY file into float32 JAX arrays.

    Higher-order SH properties follow the INRIA channel-major PLY convention
    and are returned in the basis-major ``(N, K - 1, 3)`` layout used by
    gsplat.
    """

    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError(
            "load_ply_to_splats requires the 'plyfile' package. "
            "Install it with `uv add plyfile`."
        ) from exc

    ply = PlyData.read(str(path))
    vertex = ply.elements[0]
    count = len(vertex)

    means = np.stack([np.asarray(vertex[name]) for name in ("x", "y", "z")], axis=1)
    opacities = np.asarray(vertex["opacity"])
    sh0 = np.stack([np.asarray(vertex[f"f_dc_{index}"]) for index in range(3)], axis=1)

    rest_names = sorted(
        (prop.name for prop in vertex.properties if prop.name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", maxsplit=1)[1]),
    )
    if rest_names:
        if len(rest_names) % 3 != 0:
            raise ValueError(
                f"f_rest property count ({len(rest_names)}) is not a multiple "
                "of 3 (RGB channels); cannot reshape SH coefficients."
            )
        rest = np.stack([np.asarray(vertex[name]) for name in rest_names], axis=1)
        rest = rest.reshape(count, 3, len(rest_names) // 3).swapaxes(1, 2)
    else:
        rest = np.zeros((count, 0, 3), dtype=np.float32)

    def numbered_properties(prefix: str) -> list[str]:
        return sorted(
            (prop.name for prop in vertex.properties if prop.name.startswith(prefix)),
            key=lambda name: int(name.rsplit("_", maxsplit=1)[1]),
        )

    scale_names = numbered_properties("scale_")
    rotation_names = numbered_properties("rot_")
    scales = np.stack([np.asarray(vertex[name]) for name in scale_names], axis=1)
    quats = np.stack([np.asarray(vertex[name]) for name in rotation_names], axis=1)

    def as_float32(values: np.ndarray) -> jax.Array:
        return jnp.asarray(np.ascontiguousarray(values), dtype=jnp.float32)

    return {
        "means": as_float32(means),
        "scales": as_float32(scales),
        "quats": as_float32(quats),
        "opacities": as_float32(opacities),
        "sh0": as_float32(sh0[:, None, :]),
        "shN": as_float32(rest),
    }


def splat2splat_bytes(
    means: Any,
    scales: Any,
    quats: Any,
    opacities: Any,
    sh0: Any,
) -> bytes:
    """Serialize pre-activated parameters in the 32-byte ``.splat`` layout."""

    means = _as_numpy(means).astype(np.float32, copy=False)
    scales = np.exp(_as_numpy(scales)).astype(np.float32, copy=False)
    opacities = _as_numpy(opacities)
    colors = np.concatenate([sh2rgb(sh0), _sigmoid(opacities)[:, None]], axis=1)
    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
    quaternions = _as_numpy(quats).astype(np.float32, copy=False)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    rotations = quaternions / np.maximum(norms, np.finfo(np.float32).tiny)
    rotations = np.clip(rotations * 128 + 128, 0, 255).astype(np.uint8)
    order = sort_centers(means, np.arange(means.shape[0]))

    buffer = BytesIO()
    for index in order:
        buffer.write(means[index].astype("<f4", copy=False).tobytes())
        buffer.write(scales[index].astype("<f4", copy=False).tobytes())
        buffer.write(colors[index].tobytes())
        buffer.write(rotations[index].tobytes())
    return buffer.getvalue()


def splat2ply_bytes_compressed(
    means: Any,
    scales: Any,
    quats: Any,
    opacities: Any,
    sh0: Any,
    shN: Any,
    chunk_max_size: int = 256,
    opacity_threshold: float = 1 / 255,
) -> bytes:
    """Serialize parameters in Supersplat's chunk-compressed PLY layout."""

    if chunk_max_size <= 0:
        raise ValueError("chunk_max_size must be positive")

    means = _as_numpy(means)
    scales = _as_numpy(scales)
    quats = _as_numpy(quats)
    opacities = _as_numpy(opacities)
    sh0 = _as_numpy(sh0)
    shN = _as_numpy(shN)
    keep = _sigmoid(opacities) > opacity_threshold
    means, scales, quats, opacities, shN = (
        values[keep] for values in (means, scales, quats, opacities, shN)
    )
    colors = sh2rgb(sh0)[keep]

    count = means.shape[0]
    chunk_count = (count + chunk_max_size - 1) // chunk_max_size
    order = sort_centers(means, np.arange(count))
    float_properties = (
        "min_x",
        "min_y",
        "min_z",
        "max_x",
        "max_y",
        "max_z",
        "min_scale_x",
        "min_scale_y",
        "min_scale_z",
        "max_scale_x",
        "max_scale_y",
        "max_scale_z",
        "min_r",
        "min_g",
        "min_b",
        "max_r",
        "max_g",
        "max_b",
    )
    uint_properties = (
        "packed_position",
        "packed_rotation",
        "packed_scale",
        "packed_color",
    )
    buffer = BytesIO()
    buffer.write(b"ply\nformat binary_little_endian 1.0\n")
    buffer.write(f"element chunk {chunk_count}\n".encode())
    for name in float_properties:
        buffer.write(f"property float {name}\n".encode())
    buffer.write(f"element vertex {count}\n".encode())
    for name in uint_properties:
        buffer.write(f"property uint {name}\n".encode())
    buffer.write(f"element sh {count}\n".encode())
    for index in range(shN.shape[1]):
        buffer.write(f"property uchar f_rest_{index}\n".encode())
    buffer.write(b"end_header\n")

    chunk_records: list[np.ndarray] = []
    splat_records: list[np.ndarray] = []
    sh_records: list[np.ndarray] = []
    for start in range(0, count, chunk_max_size):
        chunk_indices = order[start : start + chunk_max_size]
        chunk_means = means[chunk_indices]
        min_means = chunk_means.min(axis=0)
        max_means = chunk_means.max(axis=0)
        chunk_scales = scales[chunk_indices]
        min_scales = np.clip(chunk_scales.min(axis=0), -20, 20)
        max_scales = np.clip(chunk_scales.max(axis=0), -20, 20)
        chunk_colors = colors[chunk_indices]
        min_colors = chunk_colors.min(axis=0)
        max_colors = chunk_colors.max(axis=0)
        chunk_records.extend(
            (
                np.concatenate((min_means, max_means)),
                np.concatenate((min_scales, max_scales)),
                np.concatenate((min_colors, max_colors)),
            )
        )

        normalized_means = _normalize(chunk_means, min_means, max_means)
        packed_means = pack_111011(*normalized_means.T)
        packed_quats = pack_rotation(quats[chunk_indices])
        normalized_scales = _normalize(chunk_scales, min_scales, max_scales)
        packed_scales = pack_111011(*normalized_scales.T)
        normalized_colors = _normalize(chunk_colors, min_colors, max_colors)
        alpha = _sigmoid(opacities[chunk_indices])
        packed_colors = pack_8888(*normalized_colors.T, alpha)
        splat_records.append(
            np.stack(
                (packed_means, packed_quats, packed_scales, packed_colors), axis=1
            ).ravel()
        )

        quantized_sh = (shN[chunk_indices] / 8 + 0.5) * 256
        sh_records.append(np.clip(np.trunc(quantized_sh), 0, 255).astype(np.uint8))

    if chunk_records:
        buffer.write(np.concatenate(chunk_records).astype("<f4").tobytes())
        buffer.write(np.concatenate(splat_records).astype("<u4").tobytes())
        buffer.write(np.concatenate(sh_records).astype(np.uint8).tobytes())
    return buffer.getvalue()


def _prepare_export_arrays(
    means: Any,
    scales: Any,
    quats: Any,
    opacities: Any,
    sh0: Any,
    shN: Any,
) -> tuple[np.ndarray, ...]:
    means = _as_numpy(means)
    if means.ndim != 2:
        raise ValueError(f"means must have shape (N, 3), got {means.shape}")
    scales = _as_numpy(scales)
    quats = _as_numpy(quats)
    opacities = _as_numpy(opacities)
    sh0 = _as_numpy(sh0)
    shN = _as_numpy(shN)
    count = means.shape[0]
    expected_shapes = (
        (means, (count, 3), "means"),
        (scales, (count, 3), "scales"),
        (quats, (count, 4), "quats"),
        (opacities, (count,), "opacities"),
        (sh0, (count, 1, 3), "sh0"),
    )
    for values, expected, name in expected_shapes:
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
    if shN.ndim != 3 or shN.shape[0] != count or shN.shape[2] != 3:
        raise ValueError(f"shN must have shape ({count}, K, 3), got {shN.shape}")

    sh0 = sh0[:, 0]
    shN = shN.transpose(0, 2, 1).reshape(count, shN.shape[1] * 3)
    valid = np.ones((count,), dtype=bool)
    for values in (means, scales, quats, opacities, sh0, shN):
        finite = np.isfinite(values)
        if values.ndim == 1:
            valid &= finite
        else:
            valid &= finite.all(axis=tuple(range(1, values.ndim)))
    return tuple(
        values[valid] for values in (means, scales, quats, opacities, sh0, shN)
    )


def export_ply(splats: GaussianModel | Mapping[str, Any], path: str | Path) -> Path:
    """Export active splats in the conventional 3DGS binary PLY layout."""

    arrays = _active_splats(splats)
    means = arrays["means"].astype("<f4")
    sh0 = arrays["sh0"].reshape(len(means), 3).astype("<f4")
    # The reference 3DGS PLY layout stores all coefficients for channel R,
    # then G, then B (not basis-major as used by the model tensor).
    sh_rest = arrays["sh_rest"].transpose(0, 2, 1).reshape(len(means), -1).astype("<f4")
    opacity = arrays["opacity_logits"].reshape(len(means), 1).astype("<f4")
    scales = arrays["log_scales"].astype("<f4")
    quats = arrays["quats"].astype("<f4")
    normals = np.zeros_like(means)
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{index}" for index in range(3)]
    names += [f"f_rest_{index}" for index in range(sh_rest.shape[1])]
    names += ["opacity"]
    names += [f"scale_{index}" for index in range(3)]
    names += [f"rot_{index}" for index in range(4)]
    matrix = np.concatenate(
        [means, normals, sh0, sh_rest, opacity, scales, quats], axis=1
    ).astype("<f4", copy=False)
    dtype = np.dtype([(name, "<f4") for name in names])
    records = np.empty((len(means),), dtype=dtype)
    for index, name in enumerate(names):
        records[name] = matrix[:, index]

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(means)}",
        *[f"property float {name}" for name in names],
        "end_header",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        records.tofile(handle)
    return path


def export_splat(splats: GaussianModel | Mapping[str, Any], path: str | Path) -> Path:
    """Export active splats in gsplat's 32-byte web-viewer layout."""

    arrays = _active_splats(splats)
    data = splat2splat_bytes(
        arrays["means"],
        arrays["log_scales"],
        arrays["quats"],
        arrays["opacity_logits"],
        arrays["sh0"][:, 0],
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@overload
def export_splats(
    means: GaussianModel | Mapping[str, Any],
    scales: str | Path,
    /,
) -> Path: ...


@overload
def export_splats(
    means: Any,
    scales: Any,
    quats: Any,
    opacities: Any,
    sh0: Any,
    shN: Any,
    format: ExportFormat = "ply",
    save_to: str | Path | None = None,
) -> bytes: ...


def export_splats(
    means: Any,
    scales: Any,
    quats: Any | None = None,
    opacities: Any | None = None,
    sh0: Any | None = None,
    shN: Any | None = None,
    format: ExportFormat = "ply",
    save_to: str | Path | None = None,
) -> bytes | Path:
    """Export arrays like gsplat v1.5.3, or export a model to a path.

    The upstream-compatible array form returns bytes and accepts pre-activated
    parameters. The legacy ``(model, path)`` form is retained for this project's
    command-line interface.
    """

    if isinstance(means, (GaussianModel, Mapping)):
        if any(value is not None for value in (quats, opacities, sh0, shN)):
            raise TypeError("the model export form accepts only (model, path)")
        path = Path(scales)
        if path.suffix.lower() == ".ply":
            return export_ply(means, path)
        if path.suffix.lower() == ".splat":
            return export_splat(means, path)
        raise ValueError("export path must end in .ply or .splat")

    if any(value is None for value in (quats, opacities, sh0, shN)):
        raise TypeError("the array export form requires six parameter arrays")
    arrays = _prepare_export_arrays(means, scales, quats, opacities, sh0, shN)
    if format == "ply":
        data = splat2ply_bytes(*arrays)
    elif format == "splat":
        data = splat2splat_bytes(*arrays[:5])
    elif format == "ply_compressed":
        data = splat2ply_bytes_compressed(*arrays)
    else:
        raise ValueError(f"unsupported format: {format}")
    if save_to is not None:
        path = Path(save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data
