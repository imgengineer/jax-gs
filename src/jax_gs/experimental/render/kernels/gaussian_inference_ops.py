"""Pure-JAX packed Gaussian inference render operators."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ....config import RasterizationConfig
from ....rasterization import rasterization
from ....scene.sh_compression import SHCompressionMode

_SH_C0 = 0.28209479177387814
_SH_BIAS = 0.5


def _require_array(name: str, value: Any) -> jax.Array:
    if not isinstance(value, jax.Array):
        raise TypeError(f"{name} must be a JAX array; got {type(value).__name__}")
    return value


def _round_away_from_zero(values: jax.Array) -> jax.Array:
    return jnp.sign(values) * jnp.floor(jnp.abs(values) + 0.5)


def _round_half_mantissa(values: jax.Array, retained_bits: int) -> jax.Array:
    """Round float16 values while retaining ``retained_bits`` mantissa bits."""

    half = values.astype(jnp.float16)
    dropped_bits = 10 - retained_bits
    bits = jax.lax.bitcast_convert_type(half, jnp.uint16)
    rounding = jnp.asarray(1 << (dropped_bits - 1), dtype=jnp.uint16)
    mask = jnp.asarray(~((1 << dropped_bits) - 1) & 0xFFFF, dtype=jnp.uint16)
    rounded = (bits + rounding) & mask
    return jax.lax.bitcast_convert_type(rounded, jnp.float16).astype(jnp.float32)


def _rgb_to_ycocg(colors: jax.Array) -> jax.Array:
    red, green, blue = jnp.moveaxis(colors, -1, 0)
    return jnp.stack(
        (
            0.25 * red + 0.5 * green + 0.25 * blue,
            0.5 * red - 0.5 * blue,
            -0.25 * red + 0.5 * green - 0.25 * blue,
        ),
        axis=-1,
    )


def _ycocg_to_rgb(colors: jax.Array) -> jax.Array:
    luminance, orange, green = jnp.moveaxis(colors, -1, 0)
    return jnp.stack(
        (
            luminance - green + orange,
            luminance + green,
            luminance - green - orange,
        ),
        axis=-1,
    )


def _visible_percentile_scales(values: jax.Array, opacities: jax.Array) -> jax.Array:
    """Compute upstream's per-basis p99.99 absolute scale without ragged arrays."""

    visible = opacities >= 0.005
    visible_count = jnp.count_nonzero(visible)
    masked = jnp.where(visible[:, None, None], jnp.abs(values), jnp.inf)
    ordered = jnp.sort(masked, axis=0)
    index = jnp.minimum((visible_count * 9999) // 10000, visible_count - 1)
    index = jnp.clip(index, 0, values.shape[0] - 1)
    scales = ordered[index]
    return jnp.where(visible_count == 0, jnp.ones_like(scales), scales)


def _quantize_then_decode(
    values: jax.Array, scales: jax.Array, maximum: float
) -> jax.Array:
    normalized = jnp.where(scales > 0, values / scales, 0.0)
    gamma = jnp.sign(normalized) * jnp.sqrt(jnp.abs(normalized))
    quantized = _round_away_from_zero(gamma * maximum)
    quantized = jnp.clip(quantized, -(maximum + 1.0), maximum)
    decoded = quantized / maximum
    return jnp.sign(decoded) * jnp.square(decoded) * scales


def _simulate_sh_codec(
    colors: jax.Array,
    opacities: jax.Array,
    mode: SHCompressionMode,
) -> jax.Array:
    """Apply the 32B/16B codec numerics without materializing its bitstream."""

    colors = colors.astype(jnp.float16).astype(jnp.float32)
    dc_rgb = colors[:, 0] * _SH_C0 + _SH_BIAS
    dc_ycocg = _rgb_to_ycocg(dc_rgb)
    retained_bits = 9 if mode is SHCompressionMode.PACKED_32B else 5
    dc_y = dc_ycocg[:, :1].astype(jnp.float16).astype(jnp.float32)
    dc_chroma = _round_half_mantissa(dc_ycocg[:, 1:], retained_bits)
    decoded_dc_rgb = _ycocg_to_rgb(jnp.concatenate((dc_y, dc_chroma), axis=-1))
    decoded_dc = (decoded_dc_rgb - _SH_BIAS) / _SH_C0

    higher_ycocg = _rgb_to_ycocg(colors[:, 1:])
    scales = _visible_percentile_scales(higher_ycocg, opacities)
    scales = scales.astype(jnp.float16).astype(jnp.float32)
    decoded_y = _quantize_then_decode(higher_ycocg[..., :1], scales[..., :1], 31.0)
    if mode is SHCompressionMode.PACKED_32B:
        decoded_chroma = _quantize_then_decode(
            higher_ycocg[..., 1:], scales[..., 1:], 7.0
        )
    else:
        decoded_chroma = jnp.zeros_like(higher_ycocg[..., 1:])
    decoded_higher = _ycocg_to_rgb(
        jnp.concatenate((decoded_y, decoded_chroma), axis=-1)
    )
    return jnp.concatenate((decoded_dc[:, None, :], decoded_higher), axis=1)


def _unpack_scene(
    means_planar: jax.Array,
    qso_packed: jax.Array,
    colors_packed: jax.Array,
    sh_degree: int,
    sh_compression_mode: int | SHCompressionMode,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int | None]:
    means_planar = _require_array("means_planar", means_planar)
    qso_packed = _require_array("qso_packed", qso_packed)
    colors_packed = _require_array("colors_packed", colors_packed)
    try:
        mode = SHCompressionMode(sh_compression_mode)
    except ValueError as exc:
        raise ValueError(f"invalid sh_compression_mode: {sh_compression_mode}") from exc

    if means_planar.ndim != 2 or means_planar.shape[0] != 3:
        raise ValueError(
            f"means_planar must have shape [3, N]; got {means_planar.shape}"
        )
    count = means_planar.shape[1]
    if count == 0:
        raise ValueError("packed inference scenes must contain at least one Gaussian")
    if means_planar.dtype != jnp.float32:
        raise TypeError("means_planar must have dtype float32")
    if qso_packed.shape != (count, 8) or qso_packed.dtype != jnp.float16:
        raise ValueError(f"qso_packed must have shape ({count}, 8) and dtype float16")
    if sh_degree not in {-1, 0, 1, 2, 3}:
        raise ValueError(f"sh_degree must be one of [-1, 0, 1, 2, 3]; got {sh_degree}")
    if mode is not SHCompressionMode.NONE and sh_degree != 3:
        raise ValueError(f"sh_compression_mode={mode} requires sh_degree=3")

    if sh_degree == -1:
        expected_shape = (count, 4)
        expected_dtype = jnp.dtype(jnp.float16)
    elif sh_degree < 3:
        expected_shape = (count, (sh_degree + 1) ** 2, 3)
        expected_dtype = jnp.dtype(jnp.float32)
    elif mode is SHCompressionMode.NONE:
        expected_shape = (count, 16, 3)
        expected_dtype = jnp.dtype(jnp.float16)
    else:
        expected_shape = (count, 48)
        expected_dtype = jnp.dtype(jnp.float16)
    if colors_packed.shape != expected_shape or colors_packed.dtype != expected_dtype:
        raise ValueError(
            f"colors_packed must have shape {expected_shape} and dtype "
            f"{expected_dtype}; got {colors_packed.shape} and {colors_packed.dtype}"
        )

    qso = qso_packed.astype(jnp.float32)
    means = jnp.swapaxes(means_planar, 0, 1)
    quats = qso[:, :4]
    scales = qso[:, 4:7]
    opacities = qso[:, 7]
    if sh_degree == -1:
        colors = colors_packed[:, :3].astype(jnp.float32)
        render_degree = None
    else:
        colors = colors_packed.reshape((count, (sh_degree + 1) ** 2, 3)).astype(
            jnp.float32
        )
        render_degree = sh_degree
        if mode is not SHCompressionMode.NONE:
            colors = _simulate_sh_codec(colors, opacities, mode)
    return means, quats, scales, opacities, colors, render_degree


def _render_unpacked(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    opacities: jax.Array,
    colors: jax.Array,
    viewmat: jax.Array,
    K: jax.Array,
    width: int,
    height: int,
    sh_degree: int | None,
    tile_size: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    eps2d: float,
    background: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    if viewmat.shape != (4, 4) or viewmat.dtype != jnp.float32:
        raise ValueError("viewmat must have shape (4, 4) and dtype float32")
    if K.shape != (3, 3) or K.dtype != jnp.float32:
        raise ValueError("K must have shape (3, 3) and dtype float32")
    if not isinstance(width, int) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    if not isinstance(height, int) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")
    if tile_size not in (8, 16):
        raise TypeError(
            f"Inference branch supports tile_size in {{8, 16}}; got {tile_size}"
        )
    if background is not None:
        if background.shape != (3,) or background.dtype != jnp.float32:
            raise ValueError("background must have shape (3,) and dtype float32")
        backgrounds = background[None]
    else:
        backgrounds = None

    config = RasterizationConfig(
        tile_size=tile_size,
        max_gaussians_per_tile=max(means.shape[0], 1),
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        eps2d=eps2d,
        backend="reference",
    )
    renders, alphas, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmat[None],
        K[None],
        width,
        height,
        sh_degree=sh_degree,
        tile_size=tile_size,
        backgrounds=backgrounds,
        render_mode="RGB",
        camera_model="pinhole",
        config=config,
    )
    return renders[0], alphas[0]


def gaussian_render_inference_only(
    means_planar: jax.Array,
    qso_packed: jax.Array,
    colors_packed: jax.Array,
    viewmat: jax.Array,
    K: jax.Array,
    width: int,
    height: int,
    sh_degree: int,
    tile_size: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    eps2d: float,
    sh_compression_mode: int | SHCompressionMode,
    background: jax.Array | None,
    *,
    out_renders: jax.Array | None = None,
    out_alphas: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Render packed inference arrays and return ``[H,W,3]`` RGB plus alpha.

    ``out_renders`` and ``out_alphas`` are validated as destination templates.
    JAX arrays are immutable, so the returned arrays replace rather than mutate
    those buffers.
    """

    viewmat = _require_array("viewmat", viewmat)
    K = _require_array("K", K)
    unpacked = _unpack_scene(
        means_planar,
        qso_packed,
        colors_packed,
        sh_degree,
        sh_compression_mode,
    )
    renders, alphas = _render_unpacked(
        *unpacked[:5],
        viewmat,
        K,
        width,
        height,
        unpacked[5],
        tile_size,
        near_plane,
        far_plane,
        radius_clip,
        eps2d,
        background,
    )
    for name, destination, result in (
        ("out_renders", out_renders, renders),
        ("out_alphas", out_alphas, alphas),
    ):
        if destination is not None:
            _require_array(name, destination)
            if destination.shape != result.shape or destination.dtype != result.dtype:
                raise RuntimeError(
                    f"{name} expected shape {result.shape} and dtype {result.dtype}; "
                    f"got {destination.shape} and {destination.dtype}"
                )
    return jax.lax.stop_gradient((renders, alphas))


class _PureJaxGaussianInferenceRenderer:
    def __init__(
        self,
        means_planar: jax.Array,
        qso_packed: jax.Array,
        colors_packed: jax.Array,
        sh_degree: int,
        sh_compression_mode: int | SHCompressionMode,
    ) -> None:
        unpacked = _unpack_scene(
            means_planar,
            qso_packed,
            colors_packed,
            sh_degree,
            sh_compression_mode,
        )
        self._colors = unpacked[4]
        self._render_degree = unpacked[5]
        self._count = means_planar.shape[1]
        self._mode = SHCompressionMode(sh_compression_mode)
        self._released = False

    def render(
        self,
        means_planar: jax.Array,
        qso_packed: jax.Array,
        colors_packed: jax.Array,
        viewmat: jax.Array,
        K: jax.Array,
        width: int,
        height: int,
        tile_size: int,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        eps2d: float,
        sh_degree: int,
        sh_compression_mode: int | SHCompressionMode,
        background: jax.Array | None,
        out_rgbt: jax.Array | None,
    ) -> jax.Array:
        if self._released:
            raise RuntimeError("GaussianInferenceRenderer has been released")
        if means_planar.shape[1] != self._count:
            raise RuntimeError("scene Gaussian count changed after renderer creation")
        if SHCompressionMode(sh_compression_mode) is not self._mode:
            raise ValueError("render sh_compression_mode differs from renderer scene")
        del colors_packed
        qso = qso_packed.astype(jnp.float32)
        effective_degree = self._render_degree
        if effective_degree is not None:
            effective_degree = min(sh_degree, effective_degree)
        renders, alphas = _render_unpacked(
            jnp.swapaxes(means_planar, 0, 1),
            qso[:, :4],
            qso[:, 4:7],
            qso[:, 7],
            self._colors,
            viewmat,
            K,
            width,
            height,
            effective_degree,
            tile_size,
            near_plane,
            far_plane,
            radius_clip,
            eps2d,
            background,
        )
        rgbt = jnp.concatenate((renders, 1.0 - alphas), axis=-1)[None].astype(
            jnp.float16
        )
        if out_rgbt is not None and (
            out_rgbt.shape != rgbt.shape or out_rgbt.dtype != rgbt.dtype
        ):
            raise RuntimeError(
                f"out_rgbt expected shape {rgbt.shape} and dtype {rgbt.dtype}"
            )
        return jax.lax.stop_gradient(rgbt)

    def release(self) -> None:
        self._colors = None
        self._released = True

    def is_released(self) -> bool:
        return self._released

    def num_gaussians(self) -> int:
        return 0 if self._released else self._count


def create_native_gaussian_inference_renderer(
    means_planar: jax.Array,
    qso_packed: jax.Array,
    colors_packed: jax.Array,
    sh_degree: int,
    sh_compression_mode: int | SHCompressionMode,
) -> _PureJaxGaussianInferenceRenderer:
    """Create the reusable pure-JAX renderer backing the public component."""

    return _PureJaxGaussianInferenceRenderer(
        means_planar,
        qso_packed,
        colors_packed,
        sh_degree,
        sh_compression_mode,
    )


__all__ = [
    "create_native_gaussian_inference_renderer",
    "gaussian_render_inference_only",
]
