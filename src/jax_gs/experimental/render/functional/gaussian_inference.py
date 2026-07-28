"""Stateless packed Gaussian inference rendering."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ....scene import GaussianInferenceScene
from .._common import check_inference_grad_mode
from ..kernels.gaussian_inference_ops import gaussian_render_inference_only
from ..types import RenderReturn


_INFERENCE_UNSUPPORTED_FEATURES = frozenset(
    {
        "with_ut",
        "with_eval3d",
        "absgrad",
        "sparse_grad",
        "distributed",
        "packed",
        "segmented",
        "return_normals",
        "covars",
        "rays",
        "radial_coeffs",
        "tangential_coeffs",
        "thin_prism_coeffs",
        "ftheta_coeffs",
        "lidar_coeffs",
        "external_distortion_coeffs",
        "rolling_shutter",
        "viewmats_rs",
        "extra_signals",
        "extra_signals_sh_degree",
        "rasterize_mode",
        "channel_chunk",
        "global_z_order",
        "ut_params",
        "colors",
    }
)

_INFERENCE_ACCEPTED_KWARGS = frozenset(
    {
        "viewmat",
        "viewmats",
        "K",
        "Ks",
        "width",
        "height",
        "tile_size",
        "near_plane",
        "far_plane",
        "radius_clip",
        "eps2d",
        "background",
        "render_mode",
        "camera_model",
    }
)


def _validate_device_consistency(
    scene: GaussianInferenceScene,
    request: dict[str, Any],
    out: RenderReturn | None,
) -> None:
    scene_device = scene.means_planar.device
    for name in ("viewmat", "viewmats", "K", "Ks", "background"):
        value = request.get(name)
        if isinstance(value, jax.Array) and value.device != scene_device:
            raise ValueError(
                f"{name} is on {value.device} but scene is on {scene_device}; "
                "all arrays must be on the same device"
            )
    if out is not None and out.frame.device != scene_device:
        raise ValueError(
            f"out buffer is on {out.frame.device} but scene is on "
            f"{scene_device}; all arrays must be on the same device"
        )


def _validate_inference_request(request: dict[str, Any]) -> dict[str, Any]:
    for key in ("sh_degree", "sh_compression_mode"):
        if key in request:
            raise TypeError(
                "sh_degree/sh_compression_mode are read from scene; "
                "cannot be overridden via request kwargs"
            )
    if "backgrounds" in request:
        raise TypeError(
            "rasterize_gaussian_inference_scene got unexpected keyword "
            "argument 'backgrounds'"
        )
    for key in _INFERENCE_UNSUPPORTED_FEATURES:
        if key in request:
            raise TypeError(f"Inference branch does not support {key}")
    for key in request:
        if key not in _INFERENCE_ACCEPTED_KWARGS:
            raise TypeError(
                "rasterize_gaussian_inference_scene got unexpected keyword "
                f"argument '{key}'"
            )

    render_mode = request.get("render_mode", "RGB")
    if render_mode != "RGB":
        raise TypeError(
            "Inference branch supports render_mode='RGB' only; "
            f"got '{render_mode}'"
        )
    camera_model = request.get("camera_model", "pinhole")
    if camera_model != "pinhole":
        raise TypeError(
            "Inference branch supports camera_model='pinhole' only; "
            f"got '{camera_model}'"
        )
    tile_size = request.get("tile_size", 8)
    if tile_size not in (8, 16):
        raise TypeError(
            f"Inference branch supports tile_size in {{8, 16}}; got {tile_size}"
        )
    return {
        "viewmat": request.get("viewmat"),
        "viewmats": request.get("viewmats"),
        "K": request.get("K"),
        "Ks": request.get("Ks"),
        "width": request.get("width"),
        "height": request.get("height"),
        "tile_size": tile_size,
        "near_plane": request.get("near_plane", 0.01),
        "far_plane": request.get("far_plane", 1.0e10),
        "radius_clip": request.get("radius_clip", 0.0),
        "eps2d": request.get("eps2d", 0.3),
        "background": request.get("background"),
    }


def _normalize_single(
    singular: jax.Array | None,
    plural: jax.Array | None,
    *,
    singular_name: str,
    plural_name: str,
    trailing_shape: tuple[int, int],
) -> jax.Array:
    if singular is not None and plural is not None:
        raise RuntimeError(
            f"pass exactly one of {singular_name} or {plural_name}, not both"
        )
    if singular is None and plural is None:
        raise RuntimeError(f"pass exactly one of {singular_name} or {plural_name}")
    if plural is not None:
        if plural.ndim != 3 or plural.shape[0] != 1:
            leading = plural.shape[0] if plural.ndim else 0
            raise RuntimeError(
                "Inference branch supports single camera (leading dim == 1); "
                f"got {plural_name} with leading dim {leading}"
            )
        value = plural[0]
    else:
        value = singular
    if not isinstance(value, jax.Array) or value.shape != trailing_shape:
        raise ValueError(f"{singular_name} must have shape {trailing_shape}")
    return value


def _normalize_cameras(kwargs: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
    viewmat = _normalize_single(
        kwargs["viewmat"],
        kwargs["viewmats"],
        singular_name="viewmat",
        plural_name="viewmats",
        trailing_shape=(4, 4),
    )
    K = _normalize_single(
        kwargs["K"],
        kwargs["Ks"],
        singular_name="K",
        plural_name="Ks",
        trailing_shape=(3, 3),
    )
    return viewmat, K


def _validate_out_buffer(
    out: RenderReturn, height: int, width: int, device: jax.Device
) -> None:
    if not isinstance(out, RenderReturn):
        raise TypeError("out must be a RenderReturn")
    if (
        out.frame.shape != (1, height, width, 3)
        or out.frame.dtype != jnp.float32
        or out.frame.device != device
    ):
        raise RuntimeError(
            f"out.frame expected shape [1, {height}, {width}, 3], dtype "
            f"float32, device {device}; got shape {list(out.frame.shape)}, "
            f"dtype {out.frame.dtype}, device {out.frame.device}"
        )
    alpha = out.metadata.get("alpha")
    if alpha is not None and (
        not isinstance(alpha, jax.Array)
        or alpha.shape != (1, height, width, 1)
        or alpha.dtype != jnp.float32
        or alpha.device != device
    ):
        raise RuntimeError(
            "out.metadata['alpha'] expected shape "
            f"[1, {height}, {width}, 1], dtype float32, device {device}"
        )


def rasterize_gaussian_inference_scene(
    scene: GaussianInferenceScene,
    *,
    out: RenderReturn | None = None,
    **request: Any,
) -> RenderReturn:
    """Render a packed ``GaussianInferenceScene`` through the pure-JAX path."""

    if not isinstance(scene, GaussianInferenceScene):
        raise TypeError(
            "rasterize_gaussian_inference_scene requires a "
            f"GaussianInferenceScene; got {type(scene).__name__}"
        )
    if scene.is_empty():
        raise ValueError(
            "GaussianInferenceScene has been released and contains no packed "
            "tensors. Did you forget to rebuild the snapshot?"
        )
    check_inference_grad_mode()
    _validate_device_consistency(scene, request, out)
    validated = _validate_inference_request(request)
    viewmat, K = _normalize_cameras(validated)
    height = validated["height"]
    width = validated["width"]
    if not isinstance(height, int) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")
    if not isinstance(width, int) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    if out is not None:
        _validate_out_buffer(out, height, width, viewmat.device)

    renders, alphas = gaussian_render_inference_only(
        scene.means_planar,
        scene.qso_packed,
        scene.colors_packed,
        viewmat,
        K,
        width,
        height,
        scene.sh_degree,
        validated["tile_size"],
        validated["near_plane"],
        validated["far_plane"],
        validated["radius_clip"],
        validated["eps2d"],
        scene.sh_compression_mode,
        validated["background"],
        out_renders=None if out is None else out.frame[0],
        out_alphas=(
            None
            if out is None or out.metadata.get("alpha") is None
            else out.metadata["alpha"][0]
        ),
    )
    frame = renders[None]
    alpha = alphas[None]
    if out is None:
        return RenderReturn(frame=frame, metadata={"alpha": alpha})
    out.frame = frame
    out.metadata.clear()
    out.metadata["alpha"] = alpha
    return out


__all__ = ["rasterize_gaussian_inference_scene"]
