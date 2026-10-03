"""Reconstruct SH parameter cotangents inside a row-local Optax update."""

import jax
import jax.numpy as jnp
import optax


def with_sh_pullback(
    tx: optax.GradientTransformationExtraArgs, degree: int
) -> optax.GradientTransformationExtraArgs:
    """Expand masked color cotangents before running an unchanged Optax transform.

    Reductions select coordinates and color lanes on both ordinary JAX arrays
    and Pallas's padded row blocks. Higher, inactive SH bands receive zeros
    while the transform still handles their existing momentum.
    """

    def update(gradients, state, params, *, sh_center, **kwargs):
        delta = params.xyz - sh_center
        lane = jax.lax.broadcasted_iota(jnp.int32, delta.shape, 1)
        vx, vy, vz = (
            jnp.sum(jnp.where(lane == i, delta, 0), axis=1, keepdims=True) for i in range(3)
        )
        norm = jnp.sqrt(jnp.maximum(vx * vx + vy * vy + vz * vz, 1e-16))
        dx, dy, dz = vx / norm, vy / norm, vz / norm
        xx, yy, zz = dx * dx, dy * dy, dz * dz
        coefficient = jax.lax.broadcasted_iota(jnp.int32, gradients.sh.shape, 1)
        gc = jnp.sum(jnp.where(coefficient == 0, gradients.sh, 0), axis=1)
        coefficients = [
            gc * 0.28209479177387814,
            -gc * 0.4886025119029199 * dy,
            gc * 0.4886025119029199 * dz,
            -gc * 0.4886025119029199 * dx,
            gc * 1.0925484305920792 * dx * dy,
            -gc * 1.0925484305920792 * dy * dz,
            gc * 0.31539156525252005 * (2 * zz - xx - yy),
            -gc * 1.0925484305920792 * dx * dz,
            gc * 0.5462742152960396 * (xx - yy),
            -gc * 0.5900435899266435 * dy * (3 * xx - yy),
            gc * 2.890611442640554 * dx * dy * dz,
            -gc * 0.4570457994644658 * dy * (4 * zz - xx - yy),
            gc * 0.3731763325901154 * dz * (2 * zz - 3 * xx - 3 * yy),
            -gc * 0.4570457994644658 * dx * (4 * zz - xx - yy),
            gc * 1.445305721320277 * dz * (xx - yy),
            -gc * 0.5900435899266435 * dx * (xx - 3 * yy),
        ]
        active = (degree + 1) ** 2
        coefficient = jax.lax.broadcasted_iota(jnp.int32, params.sh.shape, 1)
        sh = jnp.zeros_like(params.sh)
        for i, value in enumerate(coefficients[:active]):
            # Disjoint selection avoids adding a full padded block per coefficient.
            sh = jnp.where(coefficient == i, value[:, None, :], sh)
        return tx.update(gradients.replace(sh=sh), state, params, **kwargs)

    return optax.GradientTransformationExtraArgs(tx.init, update)
