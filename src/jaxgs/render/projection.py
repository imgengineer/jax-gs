import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..config import CapacityConfig
from ..reference.sh import eval_sh
from ..scene.camera import Camera
from ..scene.point import GaussianPool


@struct.dataclass
class ProjectedGaussians:
    mean: chex.Array  # [C, 2], pixel coordinates
    depth: chex.Array  # [C]
    conic: chex.Array  # [C, 2, 2], inverse covariance
    radius: chex.Array  # [C], 3 sigma
    color: chex.Array  # [C, 3]
    alpha: chex.Array  # [C]
    visible: chex.Array  # [C]


def quaternion_to_matrix(q: chex.Array) -> chex.Array:
    q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    w, x, y, z = jnp.moveaxis(q, -1, 0)
    return jnp.stack(
        [
            jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def project(pool: GaussianPool, camera: Camera, config: CapacityConfig) -> ProjectedGaussians:
    rotation = camera.world_to_camera[:3, :3]
    points = pool.xyz @ rotation.T + camera.world_to_camera[:3, 3]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    safe_z = jnp.maximum(z, camera.near)
    mean = jnp.stack(
        [camera.fx * x / safe_z + camera.cx, camera.fy * y / safe_z + camera.cy], axis=-1
    )
    scales = jnp.exp(pool.log_scale)
    local = quaternion_to_matrix(pool.rotation) * scales[:, None, :]
    world_cov = local @ jnp.swapaxes(local, -1, -2)
    view_cov = rotation[None] @ world_cov @ rotation.T
    zeros = jnp.zeros_like(z)
    jacobian = jnp.stack(
        [
            jnp.stack([camera.fx / safe_z, zeros, -camera.fx * x / safe_z**2], axis=-1),
            jnp.stack([zeros, camera.fy / safe_z, -camera.fy * y / safe_z**2], axis=-1),
        ],
        axis=-2,
    )
    covariance = jacobian @ view_cov @ jnp.swapaxes(jacobian, -1, -2)
    covariance = covariance + 0.3 * jnp.eye(2, dtype=covariance.dtype)
    a, b, d = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 1, 1]
    det = jnp.maximum(a * d - b * b, 1e-8)
    conic = jnp.stack([jnp.stack([d, -b], -1), jnp.stack([-b, a], -1)], -2) / det[:, None, None]
    max_eigenvalue = 0.5 * (a + d + jnp.sqrt(jnp.maximum((a - d) ** 2 + 4 * b * b, 1e-12)))
    radius = 3.0 * jnp.sqrt(max_eigenvalue)
    direction = pool.xyz - camera.center
    direction = direction / jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
    color = eval_sh(pool.sh, direction, config.sh_degree)
    alpha = jax.nn.sigmoid(pool.opacity[:, 0])
    visible = (
        pool.alive
        & (z > camera.near)
        & (z < camera.far)
        & (alpha >= 1.0 / 255)
        & (mean[:, 0] + radius >= 0)
        & (mean[:, 0] - radius < camera.width)
        & (mean[:, 1] + radius >= 0)
        & (mean[:, 1] - radius < camera.height)
    )
    return ProjectedGaussians(mean, z, conic, radius, color, alpha, visible)
