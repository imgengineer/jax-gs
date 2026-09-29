import chex
import jax.numpy as jnp

C0 = 0.28209479177387814
C1 = 0.4886025119029199


def eval_sh(sh: chex.Array, direction: chex.Array, degree: int) -> chex.Array:
    """LiteGS/3DGS real SH order, with the standard +0.5 RGB offset."""
    x, y, z = direction[..., 0:1], direction[..., 1:2], direction[..., 2:3]
    result = C0 * sh[..., 0, :]
    if degree >= 1:
        result = result - C1 * y * sh[..., 1, :] + C1 * z * sh[..., 2, :] - C1 * x * sh[..., 3, :]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        result = (
            result
            + 1.0925484305920792 * x * y * sh[..., 4, :]
            - 1.0925484305920792 * y * z * sh[..., 5, :]
            + 0.31539156525252005 * (2 * zz - xx - yy) * sh[..., 6, :]
            - 1.0925484305920792 * x * z * sh[..., 7, :]
            + 0.5462742152960396 * (xx - yy) * sh[..., 8, :]
        )
    if degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        result = (
            result
            - 0.5900435899266435 * y * (3 * xx - yy) * sh[..., 9, :]
            + 2.890611442640554 * x * y * z * sh[..., 10, :]
            - 0.4570457994644658 * y * (4 * zz - xx - yy) * sh[..., 11, :]
            + 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12, :]
            - 0.4570457994644658 * x * (4 * zz - xx - yy) * sh[..., 13, :]
            + 1.445305721320277 * z * (xx - yy) * sh[..., 14, :]
            - 0.5900435899266435 * x * (xx - 3 * yy) * sh[..., 15, :]
        )
    return jnp.maximum(result + 0.5, 0.0)
