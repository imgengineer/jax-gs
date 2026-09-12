from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from ..model import GaussianModel
from .sort import sort_splats

_UPSTREAM_FIELDS = {"means", "scales", "quats", "opacities", "sh0", "shN"}


def _host_array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _safe_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values),
        where=norms != 0,
    )


def _log_transform(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _inverse_log_transform(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.expm1(np.abs(values))


def _normalize_range(
    values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> np.ndarray:
    span = maximum - minimum
    return np.divide(
        values - minimum,
        span,
        out=np.zeros_like(values, dtype=np.float64),
        where=span != 0,
    )


def _write_png(path: Path, values: np.ndarray) -> None:
    image = values.squeeze(axis=2) if values.shape[2] == 1 else values
    Image.fromarray(image).save(path, optimize=True)


def _read_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).copy()


def _spatial_sort(splats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply a deterministic spatial order suitable for PNG grids.

    Upstream uses optional PLAS to improve compression ratio. Ordering does not
    form part of the on-disk schema, so a lexicographic spatial order keeps this
    port dependency-free while preserving decoded Gaussian tuples.
    """

    return sort_splats(splats, verbose=False)


def _crop_to_square(
    splats: dict[str, np.ndarray], *, verbose: bool
) -> tuple[dict[str, np.ndarray], int]:
    count = splats["means"].shape[0]
    side = math.isqrt(count)
    crop_count = count - side * side
    if crop_count == 0:
        return splats, side
    keep_count = side * side
    order = np.argsort(-splats["opacities"].reshape(count), kind="stable")
    keep = order[:keep_count]
    if verbose:
        warnings.warn(
            f"number of Gaussians was not square; removed {crop_count} "
            "lowest-opacity entries",
            stacklevel=3,
        )
    return {name: values[keep] for name, values in splats.items()}, side


def _compress_png(
    directory: Path,
    name: str,
    values: np.ndarray,
    side: int,
    *,
    bits: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "shape": list(values.shape),
        "dtype": values.dtype.name,
    }
    if values.size == 0:
        return metadata
    grid = values.reshape(side, side, -1)
    minimum = grid.min(axis=(0, 1))
    maximum = grid.max(axis=(0, 1))
    normalized = _normalize_range(grid, minimum, maximum)
    quantized = np.rint(normalized * ((1 << bits) - 1)).astype(np.uint16)
    if bits == 8:
        _write_png(directory / f"{name}.png", quantized.astype(np.uint8))
    else:
        _write_png(directory / f"{name}_l.png", (quantized & 0xFF).astype(np.uint8))
        _write_png(directory / f"{name}_u.png", (quantized >> 8).astype(np.uint8))
    metadata.update({"mins": minimum.tolist(), "maxs": maximum.tolist()})
    return metadata


def _decompress_png(
    directory: Path,
    name: str,
    metadata: Mapping[str, Any],
    *,
    bits: int,
) -> np.ndarray:
    shape = tuple(metadata["shape"])
    dtype = np.dtype(metadata["dtype"])
    if math.prod(shape) == 0:
        return np.zeros(shape, dtype=dtype)
    if bits == 8:
        quantized = _read_png(directory / f"{name}.png").astype(np.uint16)
    else:
        lower = _read_png(directory / f"{name}_l.png").astype(np.uint16)
        upper = _read_png(directory / f"{name}_u.png").astype(np.uint16)
        quantized = (upper << 8) + lower
    normalized = quantized / ((1 << bits) - 1)
    minimum = np.asarray(metadata["mins"])
    maximum = np.asarray(metadata["maxs"])
    values = normalized.reshape(-1, minimum.size) * (maximum - minimum) + minimum
    return values.reshape(shape).astype(dtype)


def _compress_npz(directory: Path, name: str, values: np.ndarray) -> dict[str, Any]:
    np.savez_compressed(directory / f"{name}.npz", arr=values)
    return {"shape": list(values.shape), "dtype": values.dtype.name}


def _decompress_npz(
    directory: Path, name: str, metadata: Mapping[str, Any]
) -> np.ndarray:
    with np.load(directory / f"{name}.npz") as archive:
        values = archive["arr"]
    return values.reshape(metadata["shape"]).astype(np.dtype(metadata["dtype"]))


def _nearest_centroids(
    values: np.ndarray, centroids: np.ndarray, *, target_elements: int = 8_000_000
) -> np.ndarray:
    labels = np.empty((values.shape[0],), dtype=np.intp)
    rows_per_chunk = max(1, target_elements // centroids.shape[0])
    centroid_norms = np.sum(centroids * centroids, axis=1)
    for start in range(0, values.shape[0], rows_per_chunk):
        chunk = values[start : start + rows_per_chunk]
        distances = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            + centroid_norms[None, :]
            - 2 * chunk @ centroids.T
        )
        labels[start : start + chunk.shape[0]] = np.argmin(distances, axis=1)
    return labels


def _kmeans(
    values: np.ndarray, cluster_count: int, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run bounded-memory deterministic Lloyd iterations on host arrays."""

    sample_indices = np.linspace(0, values.shape[0] - 1, cluster_count, dtype=np.intp)
    centroids = values[sample_indices].astype(np.float32, copy=True)
    previous_labels: np.ndarray | None = None
    for _ in range(iterations):
        labels = _nearest_centroids(values, centroids)
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            break
        totals = np.zeros_like(centroids)
        np.add.at(totals, labels, values)
        counts = np.bincount(labels, minlength=cluster_count)
        occupied = counts > 0
        centroids[occupied] = totals[occupied] / counts[occupied, None]
        previous_labels = labels
    return centroids, labels


def _compress_kmeans(
    directory: Path,
    name: str,
    values: np.ndarray,
    *,
    maximum_clusters: int,
    iterations: int,
    quantization_bits: int = 6,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "shape": list(values.shape),
        "dtype": values.dtype.name,
    }
    if values.size == 0:
        return metadata
    vectors = values.reshape(values.shape[0], -1).astype(np.float32)
    cluster_count = min(maximum_clusters, vectors.shape[0])
    centroids, labels = _kmeans(vectors, cluster_count, iterations)
    minimum = centroids.min()
    maximum = centroids.max()
    normalized = _normalize_range(centroids, minimum, maximum)
    quantized = np.rint(normalized * ((1 << quantization_bits) - 1)).astype(np.uint8)
    np.savez_compressed(
        directory / f"{name}.npz",
        centroids=quantized,
        labels=labels.astype(np.uint16),
    )
    metadata.update(
        {
            "mins": float(minimum),
            "maxs": float(maximum),
            "quantization": quantization_bits,
        }
    )
    return metadata


def _decompress_kmeans(
    directory: Path, name: str, metadata: Mapping[str, Any]
) -> np.ndarray:
    shape = tuple(metadata["shape"])
    dtype = np.dtype(metadata["dtype"])
    if math.prod(shape) == 0:
        return np.zeros(shape, dtype=dtype)
    with np.load(directory / f"{name}.npz") as archive:
        quantized = archive["centroids"]
        labels = archive["labels"]
    normalized = quantized / ((1 << metadata["quantization"]) - 1)
    centroids = normalized * (metadata["maxs"] - metadata["mins"]) + metadata["mins"]
    return centroids[labels].reshape(shape).astype(dtype)


class PngCompression:
    """Encode Gaussian parameters using gsplat's PNG directory schema.

    A mapping with the six gsplat v1.5.3 parameter names uses its lossy codec:
    16-bit means, 8-bit parameter PNGs, and a quantized SH codebook. PLAS and
    TorchPQ are replaced by deterministic host implementations so the package
    remains free of PyTorch/CUDA dependencies.

    Passing a :class:`GaussianModel`, a non-gsplat mapping, or an explicit
    ``image_width`` retains jax-gs's byte-exact transport codec.
    """

    def __init__(
        self,
        use_sort: bool = True,
        verbose: bool = True,
        *,
        image_width: int | None = None,
        kmeans_clusters: int = 4096,
        kmeans_iterations: int = 4,
    ) -> None:
        if image_width is not None and image_width <= 0:
            raise ValueError("image_width must be positive")
        if not 1 <= kmeans_clusters <= 65536:
            raise ValueError("kmeans_clusters must be between 1 and 65536")
        if kmeans_iterations <= 0:
            raise ValueError("kmeans_iterations must be positive")
        self.use_sort = use_sort
        self.verbose = verbose
        self.image_width = image_width
        self.kmeans_clusters = kmeans_clusters
        self.kmeans_iterations = kmeans_iterations

    def compress(
        self,
        compress_dir: str | Path,
        splats: GaussianModel | Mapping[str, Any],
    ) -> None:
        """Compress splats into ``compress_dir``."""

        directory = Path(compress_dir)
        directory.mkdir(parents=True, exist_ok=True)
        is_upstream_mapping = not isinstance(
            splats, GaussianModel
        ) and _UPSTREAM_FIELDS.issubset(splats.keys())
        if is_upstream_mapping and self.image_width is None:
            self._compress_upstream(directory, splats)
            (directory / "metadata.json").unlink(missing_ok=True)
        else:
            self._compress_lossless(directory, splats)
            (directory / "meta.json").unlink(missing_ok=True)

    def decompress(
        self, compress_dir: str | Path, *, as_jax: bool = True
    ) -> dict[str, jax.Array] | dict[str, np.ndarray]:
        """Decode either the upstream-compatible or byte-exact directory form."""

        directory = Path(compress_dir)
        if (directory / "meta.json").is_file():
            result = self._decompress_upstream(directory)
        elif (directory / "metadata.json").is_file():
            result = self._decompress_lossless(directory)
        else:
            raise FileNotFoundError(
                f"no meta.json or metadata.json found in {directory}"
            )
        if as_jax:
            return {name: jnp.asarray(values) for name, values in result.items()}
        return result

    def _compress_upstream(self, directory: Path, splats: Mapping[str, Any]) -> None:
        host = {
            name: np.asarray(_host_array(value)).copy()
            for name, value in splats.items()
        }
        count = host["means"].shape[0]
        for name, values in host.items():
            if values.ndim == 0 or values.shape[0] != count:
                raise ValueError(f"{name} must have leading Gaussian dimension {count}")
        if host["means"].shape != (count, 3):
            raise ValueError("means must have shape (N, 3)")
        if host["quats"].shape != (count, 4):
            raise ValueError("quats must have shape (N, 4)")
        if host["opacities"].shape not in {(count,), (count, 1)}:
            raise ValueError("opacities must have shape (N,) or (N, 1)")

        host["means"] = _log_transform(host["means"])
        host["quats"] = _safe_normalize(host["quats"])
        host, side = _crop_to_square(host, verbose=self.verbose)
        if self.use_sort:
            host = _spatial_sort(host)

        metadata: dict[str, Any] = {}
        for name, values in host.items():
            if name == "means":
                metadata[name] = _compress_png(directory, name, values, side, bits=16)
            elif name in {"scales", "quats", "opacities", "sh0"}:
                metadata[name] = _compress_png(directory, name, values, side, bits=8)
            elif name == "shN":
                metadata[name] = _compress_kmeans(
                    directory,
                    name,
                    values,
                    maximum_clusters=self.kmeans_clusters,
                    iterations=self.kmeans_iterations,
                )
            else:
                metadata[name] = _compress_npz(directory, name, values)
        (directory / "meta.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def _decompress_upstream(self, directory: Path) -> dict[str, np.ndarray]:
        metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        result: dict[str, np.ndarray] = {}
        for name, specification in metadata.items():
            if name == "means":
                values = _decompress_png(directory, name, specification, bits=16)
            elif name in {"scales", "quats", "opacities", "sh0"}:
                values = _decompress_png(directory, name, specification, bits=8)
            elif name == "shN":
                values = _decompress_kmeans(directory, name, specification)
            else:
                values = _decompress_npz(directory, name, specification)
            result[name] = values
        result["means"] = _inverse_log_transform(result["means"])
        return result

    def _compress_lossless(
        self,
        directory: Path,
        splats: GaussianModel | Mapping[str, Any],
    ) -> None:
        state = splats.state_dict() if isinstance(splats, GaussianModel) else splats
        image_width = self.image_width or 2048
        metadata: dict[str, Any] = {"version": 1, "arrays": {}}
        for name, value in state.items():
            array = np.ascontiguousarray(_host_array(value))
            byte_values = array.view(np.uint8).reshape(-1)
            pixel_count = math.ceil(byte_values.size / 4)
            height = max(1, math.ceil(pixel_count / image_width))
            packed = np.zeros((height * image_width * 4,), dtype=np.uint8)
            packed[: byte_values.size] = byte_values
            image = packed.reshape(height, image_width, 4)
            Image.fromarray(image).save(directory / f"{name}.png", optimize=True)
            metadata["arrays"][name] = {
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "byte_count": int(byte_values.size),
            }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _decompress_lossless(directory: Path) -> dict[str, np.ndarray]:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        result: dict[str, np.ndarray] = {}
        for name, specification in metadata["arrays"].items():
            with Image.open(directory / f"{name}.png") as image:
                packed = np.asarray(image.convert("RGBA"))
            raw = packed.reshape(-1)[: specification["byte_count"]].tobytes()
            values = np.frombuffer(raw, dtype=np.dtype(specification["dtype"])).copy()
            result[name] = values.reshape(specification["shape"])
        return result
