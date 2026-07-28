"""Pure-JAX packing for the current-main inference-scene layout."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..sh_compression import SH_COMPRESSION_MODE_VALUES, SHCompressionMode


_VALID_SH_DEGREES = frozenset({-1, 0, 1, 2, 3})
_FP16_MAX = 65504.0


def _expect_array(name: str, value: object) -> jax.Array:
    if not isinstance(value, jax.Array):
        raise TypeError(f"{name} must be a JAX array; got {type(value).__name__}")
    return value


def _require_float32(name: str, value: jax.Array) -> None:
    if value.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"{name} must have dtype float32; got {value.dtype}")


def pack_gaussian_inference_scene(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    opacities: jax.Array,
    colors: jax.Array,
    sh_degree: int,
    sh_compression_mode: SHCompressionMode,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pack activated Gaussian arrays into the inference-internal layout."""

    if not isinstance(sh_degree, int):
        raise TypeError(
            f"sh_degree must be an int; got {type(sh_degree).__name__}"
        )
    if sh_degree not in _VALID_SH_DEGREES:
        raise ValueError(
            f"sh_degree must be one of {sorted(_VALID_SH_DEGREES)}; "
            f"got {sh_degree}"
        )
    if not isinstance(sh_compression_mode, SHCompressionMode):
        raise TypeError(
            "sh_compression_mode must be an SHCompressionMode; got "
            f"{type(sh_compression_mode).__name__}"
        )
    if sh_compression_mode not in SH_COMPRESSION_MODE_VALUES:
        raise ValueError(f"invalid sh_compression_mode: {sh_compression_mode}")
    if sh_compression_mode is not SHCompressionMode.NONE and sh_degree != 3:
        raise ValueError(
            f"sh_compression_mode={sh_compression_mode} requires sh_degree=3; "
            f"got sh_degree={sh_degree}"
        )

    arrays = {
        "means": _expect_array("means", means),
        "quats": _expect_array("quats", quats),
        "scales": _expect_array("scales", scales),
        "opacities": _expect_array("opacities", opacities),
        "colors": _expect_array("colors", colors),
    }
    for name, value in arrays.items():
        _require_float32(name, value)

    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"means must have shape [N, 3]; got {means.shape}")
    count = means.shape[0]
    if quats.shape != (count, 4):
        raise ValueError(f"quats must have shape [{count}, 4]; got {quats.shape}")
    if scales.shape != (count, 3):
        raise ValueError(f"scales must have shape [{count}, 3]; got {scales.shape}")
    if opacities.shape != (count,):
        raise ValueError(
            f"opacities must have shape [{count}]; got {opacities.shape}"
        )
    if sh_degree >= 0:
        basis_count = (sh_degree + 1) ** 2
        if colors.shape != (count, basis_count, 3):
            raise ValueError(
                f"sh_degree={sh_degree} requires colors shape "
                f"({count}, {basis_count}, 3); got {colors.shape}"
            )
    elif colors.shape != (count, 3):
        raise ValueError(
            f"RGB mode requires colors shape ({count}, 3); got {colors.shape}"
        )

    means_planar = jnp.swapaxes(means, 0, 1)
    scales_half = jnp.clip(scales, -_FP16_MAX, _FP16_MAX).astype(jnp.float16)
    qso_packed = jnp.concatenate(
        (
            quats.astype(jnp.float16),
            scales_half,
            opacities[:, None].astype(jnp.float16),
        ),
        axis=-1,
    )
    if sh_degree == 3:
        colors_half = jnp.clip(colors, -_FP16_MAX, _FP16_MAX).astype(
            jnp.float16
        )
        colors_packed = (
            colors_half
            if sh_compression_mode is SHCompressionMode.NONE
            else colors_half.reshape((count, 48))
        )
    elif sh_degree >= 0:
        colors_packed = colors
    else:
        rgb_half = jnp.clip(colors, -_FP16_MAX, _FP16_MAX).astype(jnp.float16)
        colors_packed = jnp.pad(rgb_half, ((0, 0), (0, 1)))

    # Upstream's C++ packer is explicitly no-grad. Preserve that inference
    # boundary even though the implementation consists of JAX primitives.
    return jax.lax.stop_gradient(
        (means_planar, qso_packed, colors_packed)
    )


__all__ = ["pack_gaussian_inference_scene"]
