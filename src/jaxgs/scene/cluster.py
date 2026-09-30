"""LiteGS world-space AABBs and frustum tests for sparse optimizer masks."""

from functools import partial

import chex
import jax
import jax.numpy as jnp

from ..render.projection import quaternion_to_matrix
from .camera import Camera
from .point import GaussianPool
from .types import WorldClusterBounds


@partial(jax.jit, static_argnames=("cluster_size",))
def world_cluster_bounds(pool: GaussianPool, cluster_size: int) -> WorldClusterBounds:
    transform = quaternion_to_matrix(pool.rotation) * jnp.exp(pool.log_scale)[:, None, :]
    extent = jnp.sum(jnp.abs(transform), axis=-1) * jnp.sqrt(2 * jnp.log(255.0))
    padding = (-pool.xyz.shape[0]) % cluster_size
    lower = jnp.pad(
        jnp.where(pool.alive[:, None], pool.xyz - extent, jnp.inf),
        ((0, padding), (0, 0)),
        constant_values=jnp.inf,
    )
    upper = jnp.pad(
        jnp.where(pool.alive[:, None], pool.xyz + extent, -jnp.inf),
        ((0, padding), (0, 0)),
        constant_values=-jnp.inf,
    )
    lower = lower.reshape(-1, cluster_size, 3).min(axis=1)
    upper = upper.reshape(-1, cluster_size, 3).max(axis=1)
    valid = jnp.isfinite(lower[:, 0])
    return (
        jnp.where(valid[:, None], (lower + upper) / 2, 0),
        jnp.where(valid[:, None], (upper - lower) / 2, 0),
        valid,
    )


def frustum_cluster_mask(
    bounds: WorldClusterBounds, camera: Camera, cluster_size: int, capacity: int
) -> chex.Array:
    center, extent, valid = bounds
    near, far = 0.01, 5000.0  # LiteGS PinHoleCameraInfo projection defaults
    projection = jnp.array(
        [
            [2 * camera.fx / camera.width, 0, 2 * camera.cx / camera.width - 1, 0],
            [0, 2 * camera.fy / camera.height, 2 * camera.cy / camera.height - 1, 0],
            [0, 0, far / (far - near), -far * near / (far - near)],
            [0, 0, 1, 0],
        ]
    )
    matrix = projection @ camera.world_to_camera
    planes = jnp.stack(
        (
            matrix[3] + matrix[0],
            matrix[3] - matrix[0],
            matrix[3] + matrix[1],
            matrix[3] - matrix[1],
            matrix[2],
            matrix[3] - matrix[2],
        )
    )
    distance = center @ planes[:, :3].T + planes[:, 3]
    radius = extent @ jnp.abs(planes[:, :3]).T
    visible = valid & jnp.all(distance + radius >= 0, axis=1)
    return jnp.repeat(visible, cluster_size)[:capacity]
