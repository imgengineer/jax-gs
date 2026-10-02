"""Muon for each Gaussian's SH color map, with LiteGS Adam for the other fields."""

import chex
import jax
import jax.numpy as jnp
import optax

from ..kernels.visible_optax import ProgramShape
from .optimizer import AdamState, create_adam_transform

# optax.contrib.muon defaults: quintic Newton-Schulz coefficients, five
# iterations, Nesterov momentum 0.95 and the Frobenius pre-normalization epsilon.
_NEWTON_SCHULZ_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_NEWTON_SCHULZ_STEPS = 5
_MOMENTUM = 0.95
_EPS = 1e-8
# RMS of Muon's SH updates. Like optax.contrib.muon's consistent_rms (0.2 to
# match AdamW in language models), it lets Muon reuse LiteGS's Adam learning
# rates. LiteGS Adam's normalized updates m / sqrt(v) settle at an RMS of
# 0.42-0.50 on a converged bicycle model; in 30k bicycle training an update
# RMS of 0.4 also gave the best held-out PSNR of 0.1-0.8.
MUON_UPDATE_RMS = 0.4
# Single-warp programs keep Muon's per-Gaussian reductions inside a warp.
MUON_PROGRAM_SHAPE = ProgramShape(rows=4, warps=1, blocks=8)
_SH_CHANNELS = 3  # RGB; the Optax executor pads SH rows to four channels


def _orthogonalize(x: chex.Array, columns: int) -> chex.Array:
    """Muon's Newton-Schulz orthogonalization of each matrix in [B, R, C].

    Only the first `columns` columns may be nonzero. optax.contrib.muon
    iterates X <- aX + (bA + cA^2)X with A = XX^T, on the transpose of tall
    matrices; X <- aX + X(bG + cG^2) with G = X^TX is the same iteration
    without transposes. With few columns, G is a handful of per-matrix scalars,
    so every step stays elementwise over [B, R]. Zero rows stay zero, so
    padded or inactive rows do not change the result.
    """
    a, b, c = _NEWTON_SCHULZ_COEFFICIENTS
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)
    cols = [jnp.sum(jnp.where(lane == j, x, 0), axis=-1) for j in range(columns)]
    norm = jnp.sqrt(sum(jnp.sum(col * col, axis=-1) for col in cols))[:, None] + _EPS
    pairs = [(i, j) for i in range(columns) for j in range(i, columns)]

    def entry(table, i, j):
        return table[min(i, j), max(i, j)]

    def iterate(_, cols):
        gram = {(i, j): jnp.sum(cols[i] * cols[j], axis=-1) for i, j in pairs}
        square = {
            (i, j): sum(entry(gram, i, k) * entry(gram, k, j) for k in range(columns))
            for i, j in pairs
        }
        poly = {key: b * gram[key] + c * square[key] for key in pairs}
        return tuple(
            a * cols[j] + sum(cols[i] * entry(poly, i, j)[:, None] for i in range(columns))
            for j in range(columns)
        )

    # A rolled loop compiles in a third of the time of an unrolled one.
    cols = jax.lax.fori_loop(0, _NEWTON_SCHULZ_STEPS, iterate, tuple(col / norm for col in cols))
    return sum(jnp.where(lane == j, cols[j][..., None], 0) for j in range(columns))


def _orthogonalize_rank_one(x: chex.Array) -> chex.Array:
    """_orthogonalize for matrices of rank at most one, such as one SH row.

    Newton-Schulz scales a rank-one matrix by the same polynomial in its norm.
    """
    a, b, c = _NEWTON_SCHULZ_COEFFICIENTS
    norm = jnp.sqrt(jnp.sum(x * x, axis=(1, 2), keepdims=True))
    x = x / (norm + _EPS)
    norm = norm / (norm + _EPS)
    for _ in range(_NEWTON_SCHULZ_STEPS):
        factor = a + b * norm**2 + c * norm**4
        x, norm = x * factor, norm * factor
    return x


def create_muon_transform(
    active_sh_dim: int, update_rms: float = MUON_UPDATE_RMS
) -> optax.GradientTransformationExtraArgs:
    """Muon for each Gaussian's SH color map; LiteGS Adam for the other fields.

    Muon orthogonalizes updates of linear maps. A Gaussian's SH coefficients
    map the view-dependent SH basis to RGB, so its DC row and its higher
    coefficients ([active_sh_dim - 1, 3]) are two matrices, orthogonalized
    per Gaussian with optax.contrib.muon's Newton-Schulz iteration after
    Nesterov momentum. Position, scale, rotation and opacity are per-Gaussian
    vectors and keep LiteGS Adam, as optax.contrib.muon keeps Adam for
    parameters that are not matrices.

    Like optax.contrib.muon's consistent_rms, each orthogonalized matrix is
    scaled by sqrt(max(rows, columns)) * update_rms, so its update RMS is about
    update_rms and LiteGS's per-field learning rates apply unchanged. Momentum
    is bias-corrected per slot by its own update count. The state layout,
    inactive-slot freezing and extra arguments match create_adam_transform;
    the SH second moments stay zero.
    """
    adam = create_adam_transform()
    rest_scale = update_rms * (max(active_sh_dim - 1, _SH_CHANNELS) ** 0.5)
    dc_scale = update_rms * _SH_CHANNELS**0.5

    def update_fn(gradients, state, params=None, *, active, rates):
        updates, adam_state = adam.update(gradients, state, params, active=active, rates=rates)
        # Optax's Nesterov form with per-slot counts: this is update count + 1.
        count = (state.step + 1).astype(jnp.float32).reshape(-1, 1, 1)
        gradient = gradients.sh
        momentum = optax.tree.update_moment(gradient, state.m.sh, _MOMENTUM, 1)
        log_beta = jnp.log(_MOMENTUM)
        nesterov = _MOMENTUM * momentum / (1 - jnp.exp(log_beta * (count + 1))) + (
            1 - _MOMENTUM
        ) * gradient / (1 - jnp.exp(log_beta * count))
        row = jax.lax.broadcasted_iota(jnp.int32, nesterov.shape, 1)
        dc = _orthogonalize_rank_one(jnp.where(row == 0, nesterov, 0))
        rest = _orthogonalize(
            jnp.where((row > 0) & (row < active_sh_dim), nesterov, 0), _SH_CHANNELS
        )
        active_mask = active.reshape(-1, 1, 1)
        sh_update = jnp.where(active_mask, -rates.sh * (dc_scale * dc + rest_scale * rest), 0)
        return updates.replace(sh=sh_update), AdamState(
            adam_state.m.replace(sh=jnp.where(active_mask, momentum, state.m.sh)),
            adam_state.v.replace(sh=state.v.sh),
            adam_state.step,
        )

    return optax.GradientTransformationExtraArgs(adam.init, update_fn)
