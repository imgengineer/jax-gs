"""JIT rendering of a fixed Gaussian model at reusable image resolutions."""

from dataclasses import replace
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from ..config import CapacityConfig
from ..scene.camera import Camera
from ..scene.cluster import world_cluster_bounds
from ..scene.point import GaussianArrays
from . import render, render_preprocess

RESOLUTIONS = {
    "640x360": (640, 360),
    "640x480": (640, 480),
    "1280x720": (1280, 720),
    "1920x1080": (1920, 1080),
}


def resize_camera(camera: Camera, width: int, height: int) -> Camera:
    """Keep the same viewing rays while changing image sampling resolution."""
    return camera.replace(
        width=width,
        height=height,
        fx=np.float32(camera.fx) * np.float32(width / camera.width),
        fy=np.float32(camera.fy) * np.float32(height / camera.height),
        cx=np.float32(camera.cx) * np.float32(width / camera.width),
        cy=np.float32(camera.cy) * np.float32(height / camera.height),
    )


class ViewRenderer:
    """Keep model buffers on device; only camera values change between frames."""

    def __init__(
        self, pool: GaussianArrays, *, pair_capacity: int = 8_000_000, backend: str = "cute"
    ):
        sh_dims = (1, 4, 9, 16)
        if pool.sh.shape[1] not in sh_dims:
            raise ValueError("model SH dimension must be 1, 4, 9 or 16")
        if int(pool.n_active) == 0:
            raise ValueError("model has no active Gaussians")
        self.pool = pool
        self.config = replace(
            CapacityConfig(),
            max_gaussians=pool.xyz.shape[0],
            sh_degree=sh_dims.index(pool.sh.shape[1]),
            max_visibility_pairs=pair_capacity,
            tile_height=8,
        )
        self.bounds = jax.block_until_ready(world_cluster_bounds(pool, self.config.cluster_size))
        config = self.config

        def frame(pool, bounds, camera):
            clusters, _, culled = render_preprocess(bounds, camera, pool, config, backend=backend)
            output = render(camera, culled, clusters, config.sh_degree, config, backend=backend)
            return jnp.rint(output.image * 255).astype(jnp.uint8), output.overflow

        # Large model arrays remain arguments, avoiding embedding them as XLA constants.
        self.jitted = jax.jit(frame)
        self.eager = partial(frame, pool, self.bounds)

    def __call__(self, camera: Camera):
        return self.jitted(self.pool, self.bounds, camera)

    def warmup(self, camera: Camera) -> dict[str, float]:
        """Compile and execute all presets once; return first-call seconds."""
        times = {}
        for name, (width, height) in RESOLUTIONS.items():
            start = perf_counter()
            _, overflow = jax.block_until_ready(self(resize_camera(camera, width, height)))
            if bool(overflow):
                raise RuntimeError(f"{name}: visibility overflow; increase --max-visibility-pairs")
            times[name] = perf_counter() - start
        return times
