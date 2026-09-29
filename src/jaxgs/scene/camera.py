import chex
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Camera:
    """Column-vector world-to-camera transform; camera looks along +Z."""

    world_to_camera: chex.Array  # [4, 4]
    fx: chex.Array
    fy: chex.Array
    cx: chex.Array
    cy: chex.Array
    width: int = struct.field(pytree_node=False)
    height: int = struct.field(pytree_node=False)
    near: float = struct.field(pytree_node=False, default=0.2)
    far: float = struct.field(pytree_node=False, default=1000.0)

    @classmethod
    def from_colmap(
        cls,
        qvec,
        tvec,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        near: float = 0.2,
        far: float = 1000.0,
    ) -> "Camera":
        q = jnp.asarray(qvec, dtype=jnp.float32)
        translation = jnp.asarray(tvec, dtype=jnp.float32)
        chex.assert_shape(q, (4,))
        chex.assert_shape(translation, (3,))
        q = q / jnp.linalg.norm(q)
        w, x, y, z = q
        rotation = jnp.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        transform = jnp.eye(4, dtype=jnp.float32)
        transform = transform.at[:3, :3].set(rotation)
        transform = transform.at[:3, 3].set(translation)
        return cls(
            transform,
            jnp.asarray(fx),
            jnp.asarray(fy),
            jnp.asarray(cx),
            jnp.asarray(cy),
            width,
            height,
            near,
            far,
        )

    @property
    def center(self) -> chex.Array:
        r = self.world_to_camera[:3, :3]
        return -(r.T @ self.world_to_camera[:3, 3])
