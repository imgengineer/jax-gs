from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .cameras import fully_fused_projection as _dense_projection


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PaddedProjection:
    """Static packed projection that unpacks like gsplat's nine-value tuple."""

    batch_ids: jax.Array
    camera_ids: jax.Array
    gaussian_ids: jax.Array
    indptr: jax.Array
    radii: jax.Array
    means2d: jax.Array
    depths: jax.Array
    conics: jax.Array
    compensations: jax.Array | None
    valid_count: jax.Array
    overflow: jax.Array

    def __iter__(self) -> Iterator[jax.Array | None]:
        yield self.batch_ids
        yield self.camera_ids
        yield self.gaussian_ids
        yield self.indptr
        yield self.radii
        yield self.means2d
        yield self.depths
        yield self.conics
        yield self.compensations

    def tree_flatten(self):
        return (
            (
                self.batch_ids,
                self.camera_ids,
                self.gaussian_ids,
                self.indptr,
                self.radii,
                self.means2d,
                self.depths,
                self.conics,
                self.compensations,
                self.valid_count,
                self.overflow,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


def fully_fused_projection(
    means: jax.Array,
    covars: jax.Array | None,
    quats: jax.Array | None,
    scales: jax.Array | None,
    viewmats: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1.0e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: str = "pinhole",
    opacities: jax.Array | None = None,
    *,
    active_mask: jax.Array | None = None,
):
    """gsplat-compatible projection signature with a static packed variant.

    ``sparse_grad=True`` retains current-main's unbatched ``packed=True``
    contract. JAX gradients still use dense fixed-capacity storage.
    """

    means = jnp.asarray(means)
    if sparse_grad:
        if not packed:
            raise ValueError("sparse_grad=True requires packed=True")
        if means.ndim != 2:
            raise ValueError("sparse_grad does not support batch dimensions")
    viewmats = jnp.asarray(viewmats)
    radii, means2d, depths, conics, compensations, valid = _dense_projection(
        means,
        viewmats,
        Ks,
        width,
        height,
        quats=quats,
        scales=scales,
        covars=covars,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        calc_compensations=calc_compensations,
        camera_model=camera_model,
        opacities=opacities,
        active_mask=active_mask,
    )
    if not packed:
        return radii, means2d, depths, conics, compensations

    batch_shape = means.shape[:-2]
    batch_count = math.prod(batch_shape) if batch_shape else 1
    camera_count = viewmats.shape[-3]
    gaussian_count = means.shape[-2]
    capacity = batch_count * camera_count * gaussian_count
    flat_valid = valid.reshape(-1)
    selected = jnp.nonzero(flat_valid, size=capacity, fill_value=0)[0]
    valid_count = jnp.count_nonzero(flat_valid)
    output_valid = jnp.arange(capacity) < valid_count
    gaussian_ids = selected % gaussian_count
    camera_ids = (selected // gaussian_count) % camera_count
    batch_ids = selected // (gaussian_count * camera_count)
    batch_camera = batch_ids * camera_count + camera_ids
    counts = (
        jnp.zeros((batch_count * camera_count,), jnp.int32)
        .at[batch_camera]
        .add(output_valid.astype(jnp.int32))
    )
    indptr = jnp.concatenate(
        (jnp.zeros((1,), jnp.int32), jnp.cumsum(counts, dtype=jnp.int32))
    )

    # Projection outputs all have the same leading [..., C, N] dimensions.
    packed_radii = jnp.where(
        output_valid[:, None], radii.reshape(capacity, 2)[selected], 0
    )
    packed_means = jnp.where(
        output_valid[:, None], means2d.reshape(capacity, 2)[selected], 0.0
    )
    packed_depths = jnp.where(output_valid, depths.reshape(capacity)[selected], 0.0)
    packed_conics = jnp.where(
        output_valid[:, None], conics.reshape(capacity, 3)[selected], 0.0
    )
    packed_compensations = None
    if compensations is not None:
        packed_compensations = jnp.where(
            output_valid, compensations.reshape(capacity)[selected], 0.0
        )
    return PaddedProjection(
        jnp.where(output_valid, batch_ids, -1).astype(jnp.int32),
        jnp.where(output_valid, camera_ids, -1).astype(jnp.int32),
        jnp.where(output_valid, gaussian_ids, -1).astype(jnp.int32),
        indptr,
        packed_radii,
        packed_means,
        packed_depths,
        packed_conics,
        packed_compensations,
        valid_count.astype(jnp.int32),
        jnp.asarray(False),
    )


__all__ = ["PaddedProjection", "fully_fused_projection"]
