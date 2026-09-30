"""Fixed-capacity Gaussian parameters and their NNX model."""

import chex
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct
from scipy.spatial import cKDTree

from ..config import CapacityConfig


@struct.dataclass
class GaussianPool:
    """Gaussian parameters, live slots and free slots in one fixed-capacity tree."""

    xyz: chex.Array  # [C, 3]
    log_scale: chex.Array  # [C, 3]
    rotation: chex.Array  # [C, 4], wxyz; normalized during projection
    opacity: chex.Array  # [C, 1], logits
    sh: chex.Array  # [C, SH_DIM, 3]
    alive: chex.Array  # [C], boolean
    free_mask: chex.Array  # [C], boolean
    n_active: chex.Array  # scalar


class GaussianModel(nnx.Module):
    """Fixed-capacity NNX parameters and occupancy; pool views share buffers."""

    def __init__(self, pool: GaussianPool):
        capacity = pool.xyz.shape[0]
        chex.assert_shape([pool.xyz, pool.log_scale], (capacity, 3))
        chex.assert_shape(pool.rotation, (capacity, 4))
        chex.assert_shape(pool.opacity, (capacity, 1))
        chex.assert_shape(pool.sh, (capacity, None, 3))
        chex.assert_shape([pool.alive, pool.free_mask], (capacity,))
        chex.assert_shape(pool.n_active, ())
        chex.assert_type([pool.alive, pool.free_mask], jnp.bool_)
        self.xyz = nnx.Param(pool.xyz)
        self.log_scale = nnx.Param(pool.log_scale)
        self.rotation = nnx.Param(pool.rotation)
        self.opacity = nnx.Param(pool.opacity)
        self.sh = nnx.Param(pool.sh)
        self.alive = nnx.Variable(pool.alive)
        self.free_mask = nnx.Variable(pool.free_mask)
        self.n_active = nnx.Variable(pool.n_active)

    def as_pool(self) -> GaussianPool:
        """Expose arrays to the CuTe/custom-VJP kernels without copying them."""
        return GaussianPool(
            **{name: getattr(self, name).get_value() for name in GaussianPool.__dataclass_fields__}
        )

    def update_from_pool(self, pool: GaussianPool) -> None:
        for name in GaussianPool.__dataclass_fields__:
            getattr(self, name).set_value(getattr(pool, name))


def create_pool(config: CapacityConfig) -> GaussianPool:
    capacity = config.max_gaussians
    return GaussianPool(
        xyz=jnp.zeros((capacity, 3), jnp.float32),
        log_scale=jnp.zeros((capacity, 3), jnp.float32),
        rotation=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0], jnp.float32), (capacity, 1)),
        opacity=jnp.zeros((capacity, 1), jnp.float32),
        sh=jnp.zeros((capacity, config.sh_dim, 3), jnp.float32),
        alive=jnp.zeros((capacity,), jnp.bool_),
        free_mask=jnp.ones((capacity,), jnp.bool_),
        n_active=jnp.array(0, jnp.int32),
    )


def seed_pool(
    pool: GaussianPool,
    xyz: chex.Array,
    rgb: chex.Array,
    scale: float | chex.Array = 0.01,
    opacity: float = 0.1,
) -> GaussianPool:
    """Seed a new pool once; subsequent births use fixed-size slot allocation."""
    from ..reference.sh import C0

    point_count = xyz.shape[0]
    if point_count > pool.xyz.shape[0] or rgb.shape != (point_count, 3):
        raise ValueError("seed data exceeds capacity or RGB shape is invalid")
    scale = jnp.asarray(scale, jnp.float32)
    if scale.ndim == 0:
        scale = jnp.broadcast_to(scale, (point_count, 3))
    elif scale.shape == (point_count,):
        scale = jnp.broadcast_to(scale[:, None], (point_count, 3))
    elif scale.shape != (point_count, 3):
        raise ValueError("scale must be scalar, [N], or [N, 3]")
    sh = pool.sh.at[:point_count, 0, :].set((jnp.asarray(rgb) - 0.5) / C0)
    return pool.replace(
        xyz=pool.xyz.at[:point_count].set(xyz),
        log_scale=pool.log_scale.at[:point_count].set(jnp.log(scale)),
        opacity=pool.opacity.at[:point_count].set(jnp.log(opacity / (1 - opacity))),
        sh=sh,
        alive=pool.alive.at[:point_count].set(True),
        free_mask=pool.free_mask.at[:point_count].set(False),
        n_active=jnp.array(point_count, jnp.int32),
    )


def estimate_initial_scales(xyz: chex.Array) -> chex.Array:
    """RMS distance to the three nearest other points, as in LiteGS seeding."""
    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if len(points) < 2:
        return jnp.full((len(points),), 0.01, jnp.float32)
    distances, _ = cKDTree(points).query(points, k=min(4, len(points)))
    mean_distance_sq = np.mean(np.square(distances[:, 1:]), axis=1)
    return jnp.asarray(np.sqrt(np.maximum(mean_distance_sq, 1e-7)), jnp.float32)
