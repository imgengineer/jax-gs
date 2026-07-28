"""Reusable Flax NNX packed-scene inference renderer."""

from __future__ import annotations

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp

from ....scene import GaussianInferenceScene
from .._common import check_inference_grad_mode
from ..kernels.gaussian_inference_ops import (
    create_native_gaussian_inference_renderer,
)
from ..types import RenderReturn


_RENDERER_UNSUPPORTED_KWARGS = frozenset(
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
        "render_mode",
        "camera_model",
        "backgrounds",
        "sh_compression_mode",
    }
)


class GaussianInferenceRenderer(nnx.Module):
    """Stateful inference facade with lifecycle and frame-cache management.

    The packed SH codec is prepared once at construction. JAX's compilation
    cache reuses the numerical program for subsequent frames of the same shape.
    """

    def __init__(self, scene: Any, *, tile_size: int = 8) -> None:
        if not isinstance(scene, GaussianInferenceScene):
            raise TypeError(
                "GaussianInferenceRenderer requires a GaussianInferenceScene; "
                f"got {type(scene).__name__}"
            )
        if scene.is_empty():
            raise ValueError(
                "GaussianInferenceScene has been released and contains no packed "
                "tensors. Did you forget to rebuild the snapshot?"
            )
        if tile_size not in (8, 16):
            raise ValueError(f"tile_size must be 8 or 16; got {tile_size}")
        self._native = create_native_gaussian_inference_renderer(
            scene.means_planar,
            scene.qso_packed,
            scene.colors_packed,
            scene.sh_degree,
            scene.sh_compression_mode,
        )
        self._scene = scene
        self._tile_size = tile_size
        self._frame_buffer = nnx.Cache(jnp.zeros((0,), dtype=jnp.float16))
        self._frame_buffer_shape: tuple[int, int, int, int] | None = None

    def render(
        self,
        *,
        viewmat: jax.Array | None = None,
        viewmats: jax.Array | None = None,
        K: jax.Array | None = None,
        Ks: jax.Array | None = None,
        width: int,
        height: int,
        tile_size: int | None = None,
        near_plane: float = 0.01,
        far_plane: float = 1.0e10,
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
        background: jax.Array | None = None,
        sh_degree: int | None = None,
        out: RenderReturn | None = None,
        **kwargs: Any,
    ) -> RenderReturn:
        """Render one float16 ``[1,H,W,4]`` RGBT frame."""

        if self.is_released:
            raise RuntimeError(
                "GaussianInferenceRenderer has been released; cannot render"
            )
        if self._scene.num_gaussians != self._native.num_gaussians():
            raise RuntimeError(
                "Scene was mutated after renderer creation "
                f"(scene has {self._scene.num_gaussians} Gaussians, renderer "
                f"was built for {self._native.num_gaussians()}). Create a new "
                "GaussianInferenceRenderer after modifying the scene."
            )
        for key in kwargs:
            if key in _RENDERER_UNSUPPORTED_KWARGS:
                raise TypeError(
                    f"GaussianInferenceRenderer.render() does not support {key}"
                )
            raise TypeError(
                "GaussianInferenceRenderer.render() got unexpected keyword "
                f"argument '{key}'"
            )
        check_inference_grad_mode()
        viewmat_array = self._normalize_camera(
            viewmat, viewmats, singular="viewmat", plural="viewmats", shape=(4, 4)
        )
        K_array = self._normalize_camera(
            K, Ks, singular="K", plural="Ks", shape=(3, 3)
        )
        if not isinstance(width, int) or width <= 0:
            raise ValueError(f"width must be a positive integer, got {width!r}")
        if not isinstance(height, int) or height <= 0:
            raise ValueError(f"height must be a positive integer, got {height!r}")

        scene_device = self._scene.means_planar.device
        for name, value in (("viewmat", viewmat_array), ("K", K_array)):
            if value.device != scene_device:
                raise ValueError(
                    f"{name} must be a JAX array on {scene_device}; "
                    f"got device={value.device}"
                )
        if background is not None:
            if (
                not isinstance(background, jax.Array)
                or background.device != scene_device
            ):
                raise ValueError(
                    f"background must be a JAX array on {scene_device}"
                )

        effective_tile_size = self._tile_size if tile_size is None else tile_size
        if effective_tile_size not in (8, 16):
            raise ValueError(
                f"tile_size must be 8 or 16; got {effective_tile_size}"
            )
        effective_sh_degree = (
            self._scene.sh_degree if sh_degree is None else sh_degree
        )
        if not isinstance(effective_sh_degree, int):
            raise TypeError("sh_degree must be an int")
        if out is not None:
            self._validate_out(out, height, width, scene_device)

        destination = None if out is None else out.frame
        rgbt = self._native.render(
            self._scene.means_planar,
            self._scene.qso_packed,
            self._scene.colors_packed,
            viewmat_array,
            K_array,
            width,
            height,
            effective_tile_size,
            near_plane,
            far_plane,
            radius_clip,
            eps2d,
            effective_sh_degree,
            self._scene.sh_compression_mode,
            background,
            destination,
        )
        metadata = {"format": "RGBT", "channels": "RGBT"}
        if out is not None:
            out.frame = rgbt
            out.metadata.clear()
            out.metadata.update(metadata)
            return out
        self._frame_buffer.set_value(rgbt)
        self._frame_buffer_shape = rgbt.shape
        return RenderReturn(frame=self._frame_buffer[...], metadata=metadata)

    def release(self) -> None:
        """Release codec state and the cached frame."""

        if self._native is not None and not self._native.is_released():
            self._native.release()
        self._native = None
        self._frame_buffer.set_value(jnp.zeros((0,), dtype=jnp.float16))
        self._frame_buffer_shape = None

    @property
    def is_released(self) -> bool:
        return self._native is None or self._native.is_released()

    @property
    def num_gaussians(self) -> int:
        return 0 if self.is_released else self._native.num_gaussians()

    def __enter__(self) -> "GaussianInferenceRenderer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    @staticmethod
    def _normalize_camera(
        singular_value: jax.Array | None,
        plural_value: jax.Array | None,
        *,
        singular: str,
        plural: str,
        shape: tuple[int, int],
    ) -> jax.Array:
        if singular_value is not None and plural_value is not None:
            raise RuntimeError(
                f"pass exactly one of {singular} or {plural}, not both"
            )
        if singular_value is None and plural_value is None:
            raise RuntimeError(f"pass exactly one of {singular} or {plural}")
        if plural_value is not None:
            if plural_value.ndim != 3 or plural_value.shape[0] != 1:
                leading = plural_value.shape[0] if plural_value.ndim else 0
                raise RuntimeError(
                    "Inference branch supports single camera (leading dim == 1); "
                    f"got {plural} with leading dim {leading}"
                )
            result = plural_value[0]
        else:
            result = singular_value
        if not isinstance(result, jax.Array) or result.shape != shape:
            raise ValueError(f"{singular} must have shape {shape}")
        return result

    @staticmethod
    def _validate_out(
        out: RenderReturn, height: int, width: int, device: jax.Device
    ) -> None:
        if not isinstance(out, RenderReturn):
            raise TypeError("out must be a RenderReturn")
        if (
            out.frame.shape != (1, height, width, 4)
            or out.frame.dtype != jnp.float16
            or out.frame.device != device
        ):
            raise RuntimeError(
                "out.frame expected shape "
                f"[1, {height}, {width}, 4], dtype float16, device {device}; "
                f"got shape {list(out.frame.shape)}, dtype {out.frame.dtype}, "
                f"device {out.frame.device}"
            )


__all__ = ["GaussianInferenceRenderer"]
