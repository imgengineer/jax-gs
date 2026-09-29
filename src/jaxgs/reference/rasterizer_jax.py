import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..config import CapacityConfig
from ..render.projection import ProjectedGaussians
from ..render.visibility_table import VisibilityTable
from ..scene.camera import Camera


@struct.dataclass
class RenderResult:
    rgb: chex.Array
    depth: chex.Array
    alpha: chex.Array


def rasterize_jax(
    projected: ProjectedGaussians,
    table: VisibilityTable,
    camera: Camera,
    config: CapacityConfig,
    background: chex.Array | None = None,
) -> RenderResult:
    """Front-to-back alpha compositing with fixed K, for tests and tiny scenes."""
    background = jnp.zeros((3,), jnp.float32) if background is None else background
    yy, xx = jnp.meshgrid(jnp.arange(camera.height), jnp.arange(camera.width), indexing="ij")
    pixel = jnp.stack([xx.reshape(-1) + 0.5, yy.reshape(-1) + 0.5], axis=-1)
    tiles_x = (camera.width + config.tile_size - 1) // config.tile_size
    tile_ids = (yy.reshape(-1) // config.tile_size) * tiles_x + xx.reshape(-1) // config.tile_size
    ids = table.tile_gaussian_ids[tile_ids]
    valid = table.tile_valid[tile_ids]

    def composite(carry, i):
        rgb, depth, transmittance = carry
        gid = ids[:, i]
        delta = pixel - projected.mean[gid]
        exponent = -0.5 * jnp.einsum("pi,pij,pj->p", delta, projected.conic[gid], delta)
        raw_alpha = projected.alpha[gid] * jnp.exp(exponent)
        alpha = jnp.where(
            valid[:, i] & (exponent >= -4.5) & (raw_alpha >= 1.0 / 256),
            jnp.minimum(255.0 / 256, raw_alpha),
            0.0,
        )
        weight = alpha * transmittance
        rgb = rgb + weight[:, None] * projected.color[gid]
        depth = depth + weight * projected.depth[gid]
        return (rgb, depth, transmittance * (1.0 - alpha)), None

    count = pixel.shape[0]
    initial = (
        jnp.zeros((count, 3), jnp.float32),
        jnp.zeros((count,), jnp.float32),
        jnp.ones((count,), jnp.float32),
    )
    (rgb, depth, transmittance), _ = jax.lax.scan(
        composite, initial, jnp.arange(config.max_gaussians_per_tile)
    )
    rgb = rgb + transmittance[:, None] * background
    return RenderResult(
        rgb.reshape(camera.height, camera.width, 3),
        depth.reshape(camera.height, camera.width),
        (1 - transmittance).reshape(camera.height, camera.width),
    )
