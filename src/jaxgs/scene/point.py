import chex
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct
from scipy.spatial import cKDTree

from ..config import CapacityConfig


@struct.dataclass
class GaussianPool:
    xyz: chex.Array
    log_scale: chex.Array
    rotation: chex.Array  # wxyz
    opacity: chex.Array  # logit
    sh: chex.Array
    alive: chex.Array
    free_mask: chex.Array
    n_active: chex.Array


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
    c = config.max_gaussians
    return GaussianPool(
        xyz=jnp.zeros((c, 3), jnp.float32),
        log_scale=jnp.zeros((c, 3), jnp.float32),
        rotation=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0], jnp.float32), (c, 1)),
        opacity=jnp.zeros((c, 1), jnp.float32),
        sh=jnp.zeros((c, config.sh_dim, 3), jnp.float32),
        alive=jnp.zeros((c,), jnp.bool_),
        free_mask=jnp.ones((c,), jnp.bool_),
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

    n = xyz.shape[0]
    if n > pool.xyz.shape[0] or rgb.shape != (n, 3):
        raise ValueError("seed data exceeds capacity or RGB shape is invalid")
    scale = jnp.asarray(scale, jnp.float32)
    if scale.ndim == 0:
        scale = jnp.broadcast_to(scale, (n, 3))
    elif scale.shape == (n,):
        scale = jnp.broadcast_to(scale[:, None], (n, 3))
    elif scale.shape != (n, 3):
        raise ValueError("scale must be scalar, [N], or [N, 3]")
    sh = pool.sh.at[:n, 0, :].set((jnp.asarray(rgb) - 0.5) / C0)
    return pool.replace(
        xyz=pool.xyz.at[:n].set(xyz),
        log_scale=pool.log_scale.at[:n].set(jnp.log(scale)),
        opacity=pool.opacity.at[:n].set(jnp.log(opacity / (1 - opacity))),
        sh=sh,
        alive=pool.alive.at[:n].set(True),
        free_mask=pool.free_mask.at[:n].set(False),
        n_active=jnp.array(n, jnp.int32),
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
