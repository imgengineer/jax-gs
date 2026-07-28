"""Pure-JAX camera-pose refinement matching gsplat's training utility."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp


_IDENTITY_ROTATION_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _normalize(value: jax.Array, epsilon: float = 1.0e-12) -> jax.Array:
    denominator = jnp.linalg.norm(value, axis=-1, keepdims=True)
    denominator = jnp.maximum(
        denominator, jnp.asarray(epsilon, dtype=denominator.dtype)
    )
    return value / denominator


def rotation_6d_to_matrix(d6: jax.Array) -> jax.Array:
    """Convert Zhou et al.'s 6D representation with Gram--Schmidt.

    The normalized vectors are stacked as matrix rows. This is the exact
    convention used by current-main gsplat's ``examples/utils.py`` helper.
    """

    d6 = jnp.asarray(d6)
    if d6.ndim < 1 or d6.shape[-1] != 6:
        raise ValueError(
            "rotation_6d_to_matrix: expected shape (..., 6), "
            f"got {d6.shape}."
        )
    first, second = d6[..., :3], d6[..., 3:]
    row_1 = _normalize(first)
    row_2 = second - jnp.sum(row_1 * second, axis=-1, keepdims=True) * row_1
    row_2 = _normalize(row_2)
    row_3 = jnp.cross(row_1, row_2, axis=-1)
    return jnp.stack((row_1, row_2, row_3), axis=-2)


class CameraOptModule(nnx.Module):
    """Per-image camera-to-world pose deltas.

    Each embedding stores translation ``[3]`` followed by a 6D rotation
    delta. Calling :meth:`zero_init` makes every embedding an identity
    transform. As in gsplat, the delta transform is right-multiplied, so
    translations are expressed in the local camera frame.

    ``rngs`` is optional for source compatibility. Passing an explicit
    ``nnx.Rngs`` is recommended when initialization must be controlled.
    """

    def __init__(self, n: int, *, rngs: nnx.Rngs | None = None) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.embeds = nnx.Embed(
            n,
            9,
            embedding_init=jax.nn.initializers.normal(stddev=1.0),
            rngs=rngs,
        )
        self.rngs = rngs
        self.identity = nnx.Variable(
            jnp.asarray(_IDENTITY_ROTATION_6D, dtype=jnp.float32)
        )

    def zero_init(self) -> None:
        """Reset every pose delta to the identity transform."""

        self.embeds.embedding[...] = jnp.zeros_like(self.embeds.embedding[...])

    def random_init(
        self,
        std: float,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize pose deltas from a zero-mean normal distribution."""

        if std < 0.0:
            raise ValueError(
                "CameraOptModule.random_init: std must be >= 0, "
                f"got {std}."
            )
        rngs = self.rngs if rngs is None else rngs
        current = self.embeds.embedding[...]
        values = jax.random.normal(
            rngs.params(), current.shape, dtype=current.dtype
        ) * jnp.asarray(std, dtype=current.dtype)
        self.embeds.embedding[...] = values

    def __call__(
        self,
        camtoworlds: jax.Array,
        embed_ids: jax.Array,
    ) -> jax.Array:
        """Apply indexed local deltas to ``(..., 4, 4)`` camera poses."""

        camtoworlds = jnp.asarray(camtoworlds)
        embed_ids = jnp.asarray(embed_ids)
        if camtoworlds.ndim < 2 or camtoworlds.shape[-2:] != (4, 4):
            raise ValueError(
                "CameraOptModule: camtoworlds must have shape (..., 4, 4), "
                f"got {camtoworlds.shape}."
            )
        batch_dims = camtoworlds.shape[:-2]
        if batch_dims != embed_ids.shape:
            raise ValueError(
                "CameraOptModule: camtoworlds batch dimensions must match "
                f"embed_ids; got {batch_dims} and {embed_ids.shape}."
            )

        pose_deltas = self.embeds(embed_ids)
        translation = pose_deltas[..., :3]
        identity = self.identity[...].astype(pose_deltas.dtype)
        rotation = rotation_6d_to_matrix(pose_deltas[..., 3:] + identity)

        transform = jnp.broadcast_to(
            jnp.eye(4, dtype=pose_deltas.dtype), batch_dims + (4, 4)
        )
        transform = transform.at[..., :3, :3].set(rotation)
        transform = transform.at[..., :3, 3].set(translation)
        return jnp.matmul(camtoworlds, transform)

    forward = __call__


__all__ = ["CameraOptModule", "rotation_6d_to_matrix"]
