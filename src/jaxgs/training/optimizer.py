"""Masked Adam updates with fixed-capacity moments and per-parameter rates."""

import chex
import jax
import jax.numpy as jnp
import optax
from flax import struct

from ..config import OptimizationConfig, load_config
from ..kernels.visible_optax import ProgramShape
from ..scene.point import GaussianArrays
from ..scene.types import PARAMETER_NAMES, ParameterArrays, ParameterGradients, VisibleClusters

_DEFAULT_OPTIMIZATION = load_config().optimization


@struct.dataclass
class AdamState:
    m: ParameterArrays
    v: ParameterArrays
    step: chex.Array  # [C], reset when a slot is reused


def create_adam_state(pool: GaussianArrays) -> AdamState:
    """Initialize independent first and second moments for every pool slot."""
    parameters = ParameterArrays(*(getattr(pool, name) for name in PARAMETER_NAMES))
    return _ADAM_TRANSFORM.init(parameters)


def masked_adam_update(
    pool: GaussianArrays,
    state: AdamState,
    gradients: ParameterGradients,
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[GaussianArrays, AdamState]:
    """Reference Adam with bias correction, restricted to live pool slots."""
    update_counts = state.step + pool.alive.astype(jnp.int32)
    updated_pool = pool
    first_moments, second_moments = {}, {}
    for name, gradient in zip(PARAMETER_NAMES, gradients, strict=True):
        parameter = getattr(pool, name)
        active_mask = pool.alive.reshape((-1,) + (1,) * (parameter.ndim - 1))
        previous_first_moment, previous_second_moment = (
            getattr(state.m, name),
            getattr(state.v, name),
        )
        first_moment = jnp.where(
            active_mask,
            beta1 * previous_first_moment + (1 - beta1) * gradient,
            previous_first_moment,
        )
        second_moment = jnp.where(
            active_mask,
            beta2 * previous_second_moment + (1 - beta2) * gradient**2,
            previous_second_moment,
        )
        first_bias_correction = (1 - beta1 ** jnp.maximum(update_counts, 1)).reshape(
            active_mask.shape
        )
        second_bias_correction = (1 - beta2 ** jnp.maximum(update_counts, 1)).reshape(
            active_mask.shape
        )
        updated_parameter = parameter - learning_rate * (first_moment / first_bias_correction) / (
            jnp.sqrt(second_moment / second_bias_correction) + eps
        )
        updated_pool = updated_pool.replace(
            **{name: jnp.where(active_mask, updated_parameter, parameter)}
        )
        first_moments[name], second_moments[name] = first_moment, second_moment
    return updated_pool, AdamState(
        ParameterArrays(**first_moments), ParameterArrays(**second_moments), update_counts
    )


def reset_adam_slots(state: AdamState, slot_mask: chex.Array) -> AdamState:
    """Reset moments and update counts where the fixed-shape slot mask is true."""
    first_moments, second_moments = {}, {}
    for name in PARAMETER_NAMES:
        moment = getattr(state.m, name)
        reset_mask = slot_mask.reshape((-1,) + (1,) * (moment.ndim - 1))
        first_moments[name] = jnp.where(reset_mask, 0, moment)
        second_moments[name] = jnp.where(reset_mask, 0, getattr(state.v, name))
    return AdamState(
        ParameterArrays(**first_moments),
        ParameterArrays(**second_moments),
        jnp.where(slot_mask, 0, state.step),
    )


def _parameter_learning_rates(
    sh_dim: int,
    step: int | chex.Array,
    spatial_scale: float | chex.Array,
    max_steps: int | None,
    optimization: OptimizationConfig,
) -> ParameterArrays:
    """Build broadcastable field rates, including the position schedule and SH rates."""
    schedule_fraction = jnp.clip(step / (max_steps or optimization.position_lr_max_steps), 0, 1)
    position_rate = (
        0.0
        if optimization.position_lr_init == optimization.position_lr_final == 0
        else spatial_scale
        * jnp.exp(
            jnp.log(optimization.position_lr_init) * (1 - schedule_fraction)
            + jnp.log(optimization.position_lr_final) * schedule_fraction
        )
    )
    sh_rates = (
        jnp.full((1, sh_dim, 1), optimization.feature_lr / 10).at[:, 0].set(optimization.feature_lr)
    )
    return ParameterArrays(
        position_rate,
        optimization.scaling_lr,
        optimization.rotation_lr,
        optimization.opacity_lr,
        sh_rates,
    )


def create_adam_transform() -> optax.GradientTransformationExtraArgs:
    """Optax transformation matching LiteGS Adam (no bias correction).

    Optax's built-in Adam always corrects the moments. Use its moment helpers
    with LiteGS's normalization and freeze both moments in inactive slots.
    The existing fixed-capacity state also supports slot reuse and reordering.
    """

    def init_fn(params):
        zero_moments = optax.tree.zeros_like(params)
        # Eager initialization must provide distinct buffers for donation.
        return AdamState(
            zero_moments, optax.tree.zeros_like(params), jnp.zeros(params.xyz.shape[0], jnp.int32)
        )

    def update_fn(gradients, state, params=None, *, active, rates):
        del params
        # Static checks run during tracing; no device values or callbacks enter
        # the compiled update. In particular, mask broadcasting must be exact.
        chex.assert_rank(state.step, 1)
        chex.assert_shape(active, state.step.shape)
        chex.assert_type(active, jnp.bool_)
        chex.assert_trees_all_equal_shapes_and_dtypes(gradients, state.m, state.v)
        first_moment = optax.tree.update_moment(gradients, state.m, 0.9, 1)
        second_moment = optax.tree.update_moment_per_elem_norm(gradients, state.v, 0.999, 2)
        updates = jax.tree.map(
            lambda first_moment, second_moment, learning_rate: (
                -learning_rate * first_moment / (jnp.sqrt(second_moment) + 1e-15)
            ),
            first_moment,
            second_moment,
            rates,
        )

        def select_active_slots(updated, previous):
            active_mask = active.reshape((-1,) + (1,) * (updated.ndim - 1))
            return jnp.where(active_mask, updated, previous)

        updates = jax.tree.map(lambda value: select_active_slots(value, 0), updates)
        return updates, AdamState(
            jax.tree.map(select_active_slots, first_moment, state.m),
            jax.tree.map(select_active_slots, second_moment, state.v),
            state.step + active.astype(jnp.int32),
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


_ADAM_TRANSFORM = create_adam_transform()

# optax.contrib.muon defaults: quintic Newton-Schulz coefficients, five
# iterations, Nesterov momentum 0.95 and the Frobenius pre-normalization epsilon.
_MUON_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_MUON_STEPS = 5
_MUON_MOMENTUM = 0.95
_MUON_EPS = 1e-8
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
    a, b, c = _MUON_COEFFICIENTS
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)
    cols = [jnp.sum(jnp.where(lane == j, x, 0), axis=-1) for j in range(columns)]
    norm = jnp.sqrt(sum(jnp.sum(col * col, axis=-1) for col in cols))[:, None] + _MUON_EPS
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
    cols = jax.lax.fori_loop(0, _MUON_STEPS, iterate, tuple(col / norm for col in cols))
    return sum(jnp.where(lane == j, cols[j][..., None], 0) for j in range(columns))


def _orthogonalize_rank_one(x: chex.Array) -> chex.Array:
    """_orthogonalize for matrices of rank at most one, such as one SH row.

    Newton-Schulz scales a rank-one matrix by the same polynomial in its norm.
    """
    a, b, c = _MUON_COEFFICIENTS
    norm = jnp.sqrt(jnp.sum(x * x, axis=(1, 2), keepdims=True))
    x = x / (norm + _MUON_EPS)
    norm = norm / (norm + _MUON_EPS)
    for _ in range(_MUON_STEPS):
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
        momentum = optax.tree.update_moment(gradient, state.m.sh, _MUON_MOMENTUM, 1)
        log_beta = jnp.log(_MUON_MOMENTUM)
        nesterov = _MUON_MOMENTUM * momentum / (1 - jnp.exp(log_beta * (count + 1))) + (
            1 - _MUON_MOMENTUM
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


def _visible_kernel_available() -> bool:
    from ..kernels.visible_optax import available

    return available()


def optax_update(
    pool: GaussianArrays,
    state: AdamState,
    gradients: ParameterGradients,
    visible: chex.Array,
    step: int | chex.Array,
    spatial_scale: float | chex.Array,
    max_steps: int | None = None,
    *,
    cluster_size: int = 128,
    compact_gradients: bool = False,
    optimization: OptimizationConfig = _DEFAULT_OPTIMIZATION,
    active_degree: int | None = None,
    compacted_clusters: VisibleClusters | None = None,
    transform: optax.GradientTransformationExtraArgs | None = None,
    program_shape: ProgramShape | None = None,
) -> tuple[GaussianArrays, AdamState]:
    """Apply an Optax transformation to fixed pool slots, preserving momentum in every SH band.

    transform defaults to LiteGS Adam (create_adam_transform); it receives the
    active slots and per-field learning rates as extra arguments.
    program_shape tunes the visible-cluster kernel for the transformation.
    active_degree declares that higher-order projection gradients are zero.
    Their moments still decay and update parameters when slots are visible.
    With compacted_clusters and compact gradients, the same transformation
    runs on the GPU only over visible clusters, instead of every pool slot.
    """
    transform = _ADAM_TRANSFORM if transform is None else transform
    sh_dim = pool.sh.shape[1]
    active_sh_dim = sh_dim if active_degree is None else (active_degree + 1) ** 2
    if active_degree is not None and (active_degree < 0 or active_sh_dim > sh_dim):
        raise ValueError("active_degree must fit the pool's SH coefficients")
    if active_sh_dim < sh_dim:
        gradients = (*gradients[:-1], gradients[-1][:, :active_sh_dim])
    learning_rates = _parameter_learning_rates(
        pool.sh.shape[1], step, spatial_scale, max_steps, optimization
    )
    parameters = ParameterArrays(*(getattr(pool, name) for name in PARAMETER_NAMES))
    if compacted_clusters is not None and compact_gradients and _visible_kernel_available():
        from ..kernels.visible_optax import update_visible_clusters

        # The compact SH gradient prefix is read as is; omitted coefficients are zero.
        parameters, state = update_visible_clusters(
            transform,
            parameters,
            state,
            ParameterArrays(*gradients),
            pool.alive,
            compacted_clusters,
            cluster_size=cluster_size,
            program_shape=program_shape or ProgramShape(),
            rates=learning_rates,
        )
        return pool.replace(**{name: getattr(parameters, name) for name in PARAMETER_NAMES}), state
    active_slots = visible & pool.alive
    if compact_gradients:
        # compact_visible_clusters preserves cluster order. Invert that order
        # without touching its undefined IDs or gradient tail. Out-of-bounds
        # gathers produce zero, including when no clusters are visible.
        capacity = pool.xyz.shape[0]
        visible_cluster_ranks = jnp.cumsum(visible[::cluster_size], dtype=jnp.int32) - 1
        pool_slots = jnp.arange(capacity, dtype=jnp.int32)
        compact_indices = (
            visible_cluster_ranks[pool_slots // cluster_size] * cluster_size
            + pool_slots % cluster_size
        )
        compact_indices = jnp.where(active_slots, compact_indices, capacity)
        gradients = tuple(
            jnp.take(gradient, compact_indices, axis=0, mode="fill", fill_value=0)
            for gradient in gradients
        )
    if active_sh_dim < sh_dim:
        # XLA broadcasts the constant tail in the update fusion. No full-size
        # compact-gradient gather is needed for unused SH coefficients.
        gradients = (
            *gradients[:-1],
            jnp.pad(gradients[-1], ((0, 0), (0, sh_dim - active_sh_dim), (0, 0))),
        )
    parameter_updates, state = transform.update(
        ParameterArrays(*gradients), state, parameters, active=active_slots, rates=learning_rates
    )
    parameters = optax.apply_updates(parameters, parameter_updates)
    return pool.replace(**{name: getattr(parameters, name) for name in PARAMETER_NAMES}), state


def sparse_adam_update(
    pool: GaussianArrays,
    state: AdamState,
    gradients: ParameterGradients,
    visible: chex.Array,
    step: int | chex.Array,
    spatial_scale: float | chex.Array,
    max_steps: int | None = None,
    *,
    compacted_clusters: VisibleClusters | None = None,
    cluster_size: int = 128,
    compact_gradients: bool = False,
    optimization: OptimizationConfig = _DEFAULT_OPTIMIZATION,
) -> tuple[GaussianArrays, AdamState]:
    """LiteGS sparse Adam: per-field rates, no bias correction, eps=1e-15.

    Matches litegs/training/optimizer.py and compact.cu::adamUpdate. The
    visibility mask selects complete visible clusters, including zero grads.
    compact_gradients reads only the valid prefix in visible-cluster order;
    parameter and moment writes still address their original pool slots.
    """
    learning_rates = _parameter_learning_rates(
        pool.sh.shape[1], step, spatial_scale, max_steps, optimization
    )
    updated_parameters, first_moments, second_moments = {}, {}, {}
    active_slots = visible & pool.alive
    for name, gradient in zip(PARAMETER_NAMES, gradients, strict=True):
        learning_rate = getattr(learning_rates, name)
        parameter = getattr(pool, name)
        active_mask = active_slots.reshape((-1,) + (1,) * (parameter.ndim - 1))
        previous_first_moment, previous_second_moment = (
            getattr(state.m, name),
            getattr(state.v, name),
        )
        if compacted_clusters is not None:
            from ..kernels.sparse_adam import update_field

            updated_parameters[name], first_moments[name], second_moments[name] = update_field(
                parameter,
                previous_first_moment,
                previous_second_moment,
                gradient,
                pool.alive,
                compacted_clusters,
                optimization.feature_lr / 10 if name == "sh" else learning_rate,
                cluster_size,
                name == "sh",
                compact_gradients,
            )
            continue
        first_moment = 0.9 * previous_first_moment + 0.1 * gradient
        second_moment = 0.999 * previous_second_moment + 0.001 * gradient**2
        updated_parameters[name] = jnp.where(
            active_mask,
            parameter - learning_rate * first_moment / (jnp.sqrt(second_moment) + 1e-15),
            parameter,
        )
        first_moments[name], second_moments[name] = (
            jnp.where(active_mask, first_moment, previous_first_moment),
            jnp.where(active_mask, second_moment, previous_second_moment),
        )
    return pool.replace(**updated_parameters), AdamState(
        ParameterArrays(**first_moments),
        ParameterArrays(**second_moments),
        state.step + active_slots.astype(jnp.int32),
    )
