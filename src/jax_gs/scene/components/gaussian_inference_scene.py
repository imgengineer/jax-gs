"""Activated, packed Gaussian scene container for inference."""

from __future__ import annotations

import math
import warnings
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp

from ..sh_compression import SH_COMPRESSION_MAP, SHCompressionMode
from .base import Scene


def _array(value: Any) -> jax.Array:
    if isinstance(value, nnx.Variable):
        return value[...]
    return jnp.asarray(value)


def _host_bool(value: jax.Array) -> bool:
    return bool(jax.device_get(value))


class GaussianInferenceScene(Scene):
    """Packed viewer layout built through the provided class methods."""

    def __init__(self, id: str) -> None:
        super().__init__(id)
        self.means_planar: jax.Array | None = None
        self.qso_packed: jax.Array | None = None
        self.colors_packed: jax.Array | None = None
        self.sh_degree: int | None = None
        self.sh_compression_mode: SHCompressionMode | None = None
        self.num_gaussians = 0
        self.component_names: list[str] = []
        self.component_index = jnp.zeros((0,), dtype=jnp.int32)

    def is_empty(self) -> bool:
        return (
            self.means_planar is None
            or self.qso_packed is None
            or self.colors_packed is None
            or self.num_gaussians == 0
        )

    def release(self) -> None:
        self.means_planar = None
        self.qso_packed = None
        self.colors_packed = None
        self.sh_degree = None
        self.sh_compression_mode = None
        self.num_gaussians = 0
        self.component_names = []
        self.component_index = jnp.zeros((0,), dtype=jnp.int32)

    def put(self, name: str, component: dict[str, Any]) -> None:
        if not name:
            raise ValueError("component name must not be empty")
        scene_empty = self.is_empty()
        if not scene_empty and name in self.component_names:
            raise ValueError(
                f"component '{name}' already present in GaussianInferenceScene"
            )

        means_planar = component["means_planar"]
        qso_packed = component["qso_packed"]
        colors_packed = component["colors_packed"]
        sh_degree = component["sh_degree"]
        sh_compression_mode = component["sh_compression_mode"]
        local_count = self._validate_packed_component(
            means_planar,
            qso_packed,
            colors_packed,
            sh_degree,
            sh_compression_mode,
        )

        if scene_empty:
            self.means_planar = means_planar
            self.qso_packed = qso_packed
            self.colors_packed = colors_packed
            self.sh_degree = sh_degree
            self.sh_compression_mode = sh_compression_mode
            self.num_gaussians = local_count
            self.component_names = [name]
            self.component_index = jnp.zeros((local_count,), dtype=jnp.int32)
            return

        if sh_degree != self.sh_degree:
            raise ValueError(
                f"sh_degree mismatch: scene has {self.sh_degree}, "
                f"component has {sh_degree}"
            )
        if sh_compression_mode != self.sh_compression_mode:
            raise ValueError(
                "sh_compression_mode mismatch: scene has "
                f"{self.sh_compression_mode}, component has "
                f"{sh_compression_mode}"
            )
        self.means_planar = jnp.concatenate(
            (self.means_planar, means_planar), axis=1
        )
        self.qso_packed = jnp.concatenate(
            (self.qso_packed, qso_packed), axis=0
        )
        self.colors_packed = jnp.concatenate(
            (self.colors_packed, colors_packed), axis=0
        )
        self.component_names.append(name)
        self.component_index = jnp.concatenate(
            (
                self.component_index,
                jnp.full(
                    (local_count,),
                    len(self.component_names) - 1,
                    dtype=self.component_index.dtype,
                ),
            )
        )
        self.num_gaussians += local_count

    @staticmethod
    def _validate_packed_component(
        means_planar: jax.Array,
        qso_packed: jax.Array,
        colors_packed: jax.Array,
        sh_degree: int,
        sh_compression_mode: SHCompressionMode,
    ) -> int:
        for name, value in (
            ("means_planar", means_planar),
            ("qso_packed", qso_packed),
            ("colors_packed", colors_packed),
        ):
            if not isinstance(value, jax.Array):
                raise TypeError(f"{name} must be a JAX array")
        if means_planar.ndim != 2 or means_planar.shape[0] != 3:
            raise ValueError(
                f"means_planar must have shape [3, N]; got {means_planar.shape}"
            )
        if qso_packed.ndim != 2 or qso_packed.shape[1] != 8:
            raise ValueError(
                f"qso_packed must have shape [N, 8]; got {qso_packed.shape}"
            )
        local_count = qso_packed.shape[0]
        if means_planar.shape[1] != local_count:
            raise ValueError(
                f"means_planar.shape[1] ({means_planar.shape[1]}) must match "
                f"qso_packed.shape[0] ({local_count})"
            )
        if colors_packed.ndim < 2 or colors_packed.shape[0] != local_count:
            raise ValueError(
                f"colors_packed.shape[0] must match N={local_count}; "
                f"got shape {colors_packed.shape}"
            )
        if means_planar.dtype != jnp.dtype(jnp.float32):
            raise TypeError(
                "means_planar must have dtype float32; got "
                f"{means_planar.dtype}"
            )
        if qso_packed.dtype != jnp.dtype(jnp.float16):
            raise TypeError(
                f"qso_packed must have dtype float16; got {qso_packed.dtype}"
            )
        if not isinstance(sh_compression_mode, SHCompressionMode):
            raise TypeError(
                "sh_compression_mode must be an SHCompressionMode; got "
                f"{type(sh_compression_mode).__name__}"
            )
        if sh_compression_mode is not SHCompressionMode.NONE and sh_degree != 3:
            raise ValueError(
                f"sh_compression_mode={sh_compression_mode} requires "
                f"sh_degree=3; got sh_degree={sh_degree}"
            )

        if sh_degree == -1:
            expected_shape = (local_count, 4)
            expected_dtype = jnp.dtype(jnp.float16)
        elif sh_degree in (0, 1, 2):
            expected_shape = (local_count, (sh_degree + 1) ** 2, 3)
            expected_dtype = jnp.dtype(jnp.float32)
        elif sh_degree == 3 and sh_compression_mode is SHCompressionMode.NONE:
            expected_shape = (local_count, 16, 3)
            expected_dtype = jnp.dtype(jnp.float16)
        elif sh_degree == 3 and sh_compression_mode in (
            SHCompressionMode.PACKED_32B,
            SHCompressionMode.PACKED_16B,
        ):
            expected_shape = (local_count, 48)
            expected_dtype = jnp.dtype(jnp.float16)
        else:
            raise ValueError(
                f"sh_degree must be one of [-1, 0, 1, 2, 3]; got {sh_degree}"
            )
        if colors_packed.shape != expected_shape:
            raise ValueError(
                f"colors_packed must have shape {expected_shape} for "
                f"sh_degree={sh_degree}, "
                f"sh_compression_mode={sh_compression_mode}; got "
                f"{colors_packed.shape}"
            )
        if colors_packed.dtype != expected_dtype:
            raise TypeError(
                f"colors_packed must have dtype {expected_dtype}; got "
                f"{colors_packed.dtype}"
            )
        return local_count

    def get(self, component: str | int) -> dict[str, Any]:
        if self.is_empty():
            raise RuntimeError(
                "GaussianInferenceScene has been released and contains no "
                "packed arrays"
            )
        if isinstance(component, int):
            if component < 0 or component >= len(self.component_names):
                raise KeyError(f"'{component}'")
            component_id = component
        else:
            if component not in self.component_names:
                raise KeyError(f"'{component}'")
            component_id = self.component_names.index(component)
        mask = self.component_index == component_id
        return {
            "name": self.component_names[component_id],
            "index": component_id,
            "mask": mask,
            "means_planar": self.means_planar[:, mask],
            "qso_packed": self.qso_packed[mask],
            "colors_packed": self.colors_packed[mask],
            "sh_degree": self.sh_degree,
            "sh_compression_mode": self.sh_compression_mode,
        }

    @classmethod
    def from_gaussian_scene(
        cls,
        scene,
        *,
        id: str,
        sh_compression: str = "none",
    ) -> "GaussianInferenceScene":
        if hasattr(scene, "component_names") and len(scene.component_names) > 1:
            raise ValueError(
                "from_gaussian_scene does not support multi-component scenes; "
                "convert each component individually via from_gaussian_tensors"
            )
        if jax.process_count() > 1:
            raise RuntimeError(
                "GaussianInferenceScene.from_gaussian_scene is not supported "
                "with multiple JAX processes; gather first and use "
                "from_gaussian_tensors"
            )
        splats = scene.splats
        if "features" in splats:
            raise ValueError(
                "from_gaussian_scene does not support appearance-optimized "
                "scenes (splats contain 'features'). Bake RGB first."
            )

        means = _array(splats["means"])
        raw_quats = _array(splats["quats"])
        quat_norm = jnp.linalg.norm(raw_quats, axis=-1, keepdims=True)
        quats = raw_quats / jnp.maximum(quat_norm, 1.0e-12)
        scales = jnp.exp(_array(splats["scales"]))
        opacities = jax.nn.sigmoid(_array(splats["opacities"]))
        for name, value in (
            ("quats (after normalize)", quats),
            ("scales (after exp)", scales),
            ("opacities (after sigmoid)", opacities),
        ):
            if not _host_bool(jnp.all(jnp.isfinite(value))):
                raise ValueError(
                    f"from_gaussian_scene: {name} contains NaN or Inf after "
                    "activation"
                )

        colors_value = splats.get("colors", None)
        colors = None if colors_value is None else _array(colors_value)
        if colors is None:
            sh0_value = splats.get("sh0", None)
            sh0 = None if sh0_value is None else _array(sh0_value)
            if sh0 is not None and "shN" in splats:
                colors = jnp.concatenate((sh0, _array(splats["shN"])), axis=1)
            elif sh0 is not None:
                warnings.warn(
                    "GaussianScene has sh0 but no shN; degrading to SH degree 0",
                    UserWarning,
                    stacklevel=2,
                )
                colors = sh0
        if colors is None:
            raise ValueError("GaussianScene must contain 'colors' or 'sh0' in splats")

        if colors.ndim == 3:
            basis_count = colors.shape[1]
            basis_width = math.isqrt(basis_count)
            if basis_width * basis_width != basis_count:
                raise ValueError(
                    "colors SH basis dimension must be a perfect square; got "
                    f"{basis_count}"
                )
            sh_degree = basis_width - 1
        else:
            sh_degree = None
        return cls._build(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            sh_degree=sh_degree,
            sh_compression=sh_compression,
            id=id,
            skip_activation_checks=True,
        )

    @classmethod
    def from_gaussian_tensors(
        cls,
        means: jax.Array,
        quats: jax.Array,
        scales: jax.Array,
        opacities: jax.Array,
        colors: jax.Array,
        sh_degree: int | None,
        sh_compression: str,
        *,
        id: str,
    ) -> "GaussianInferenceScene":
        return cls._build(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            sh_degree=sh_degree,
            sh_compression=sh_compression,
            id=id,
            skip_activation_checks=False,
        )

    @classmethod
    def _build(
        cls,
        *,
        means: jax.Array,
        quats: jax.Array,
        scales: jax.Array,
        opacities: jax.Array,
        colors: jax.Array,
        sh_degree: int | None,
        sh_compression: str,
        id: str,
        skip_activation_checks: bool,
    ) -> "GaussianInferenceScene":
        if sh_compression not in SH_COMPRESSION_MAP:
            raise ValueError(
                "sh_compression must be one of {'none', '32b', '16b'}; "
                f"got '{sh_compression}'"
            )
        compression_mode = SH_COMPRESSION_MAP[sh_compression]
        if not isinstance(means, jax.Array):
            raise TypeError("means must be a JAX array")
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError(f"means must have shape [N, 3]; got {means.shape}")
        if sh_degree is not None and sh_degree >= 0:
            if colors.ndim != 3:
                raise ValueError(
                    f"sh_degree={sh_degree} requires colors to be 3-D "
                    f"[N, K, 3]; got {colors.ndim}-D shape {colors.shape}"
                )
            expected_basis_count = (sh_degree + 1) ** 2
            if colors.shape[1] != expected_basis_count:
                raise ValueError(
                    f"sh_degree={sh_degree} requires colors.shape[1]="
                    f"{expected_basis_count} (got {colors.shape[1]})"
                )
        elif colors.ndim != 2:
            raise ValueError(
                "sh_degree=None (pre-activated RGB) requires colors to be "
                f"2-D [N, 3]; got {colors.ndim}-D shape {colors.shape}"
            )
        if compression_mode is not SHCompressionMode.NONE and sh_degree != 3:
            raise ValueError(
                f"sh_compression='{sh_compression}' requires sh_degree=3; "
                f"got sh_degree={sh_degree}"
            )
        if not skip_activation_checks:
            _check_activation_contract(means, quats, scales, opacities, colors)

        clamp_warnings: list[tuple[str, int, float, float, int]] = []
        _check_fp16_range(scales, "scales", clamp_warnings)
        _check_fp16_range(colors, "colors", clamp_warnings)
        if clamp_warnings:
            lines = ["GaussianInferenceScene: fp16 clamping applied:"]
            for name, count, minimum, maximum, nonfinite_count in clamp_warnings:
                line = (
                    f"  {name}: clamped {count} elements "
                    f"(original range [{minimum:.4g}, {maximum:.4g}])"
                )
                if nonfinite_count:
                    line += f" ({nonfinite_count} non-finite)"
                lines.append(line)
            warnings.warn("\n".join(lines), RuntimeWarning, stacklevel=3)

        from ..kernels.gaussian_inference_ops import (
            pack_gaussian_inference_scene,
        )

        encoded_degree = -1 if sh_degree is None else sh_degree
        packed = pack_gaussian_inference_scene(
            means,
            quats,
            scales,
            opacities,
            colors,
            encoded_degree,
            compression_mode,
        )
        scene = cls(id)
        scene.put(
            id,
            {
                "means_planar": packed[0],
                "qso_packed": packed[1],
                "colors_packed": packed[2],
                "sh_degree": encoded_degree,
                "sh_compression_mode": compression_mode,
            },
        )
        return scene


def _mask_indices(mask: jax.Array) -> tuple[list[int], int]:
    count = int(jax.device_get(jnp.count_nonzero(mask)))
    if count == 0:
        return [], 0
    indices = jnp.nonzero(mask.reshape((-1,)), size=min(count, 10))[0]
    return [int(value) for value in jax.device_get(indices)], count


def _check_activation_contract(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    opacities: jax.Array,
    colors: jax.Array,
) -> None:
    for name, value in (
        ("means", means),
        ("quats", quats),
        ("scales", scales),
        ("opacities", opacities),
        ("colors", colors),
    ):
        if not isinstance(value, jax.Array):
            raise TypeError(f"{name} must be a JAX array")
        bad = ~jnp.isfinite(value)
        if _host_bool(jnp.any(bad)):
            indices, count = _mask_indices(bad)
            suffix = f"... ({count} total)" if count > 10 else ""
            raise ValueError(
                f"tensor '{name}' contains NaN or Inf at indices "
                f"{indices}{suffix}"
            )

    bad_scales = scales <= 0
    if _host_bool(jnp.any(bad_scales)):
        rows, count = _mask_indices(jnp.any(bad_scales, axis=-1))
        suffix = f"... ({count} total)" if count > 10 else ""
        raise ValueError(
            f"scales contain non-positive values at indices {rows}{suffix}; "
            "did you forget to call exp()?"
        )
    bad_opacities = (opacities < 0) | (opacities > 1)
    if _host_bool(jnp.any(bad_opacities)):
        rows, count = _mask_indices(bad_opacities)
        suffix = f"... ({count} total)" if count > 10 else ""
        raise ValueError(
            f"opacities outside [0, 1] at indices {rows}{suffix}; "
            "did you forget to call sigmoid()?"
        )
    norms = jnp.linalg.norm(quats, axis=-1)
    bad_norms = jnp.abs(norms - 1) > 1.0e-3
    if _host_bool(jnp.any(bad_norms)):
        rows, count = _mask_indices(bad_norms)
        suffix = f"... ({count} total)" if count > 10 else ""
        raise ValueError(
            f"quats are not unit-norm at indices {rows}{suffix}; "
            "did you forget to normalize quats?"
        )


def _check_fp16_range(
    tensor: jax.Array,
    name: str,
    warnings_list: list[tuple[str, int, float, float, int]],
) -> None:
    finite = jnp.isfinite(tensor)
    exceeds = (jnp.abs(tensor) > 65504.0) | (~finite)
    count = int(jax.device_get(jnp.count_nonzero(exceeds)))
    if count == 0:
        return
    finite_count = int(jax.device_get(jnp.count_nonzero(finite)))
    if finite_count:
        minimum = float(
            jax.device_get(jnp.min(jnp.where(finite, tensor, jnp.inf)))
        )
        maximum = float(
            jax.device_get(jnp.max(jnp.where(finite, tensor, -jnp.inf)))
        )
    else:
        minimum = maximum = float("nan")
    nonfinite_count = int(jax.device_get(jnp.count_nonzero(~finite)))
    warnings_list.append((name, count, minimum, maximum, nonfinite_count))


__all__ = ["GaussianInferenceScene"]
