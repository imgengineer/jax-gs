"""Masked Adam updates with fixed-capacity moments and per-parameter rates."""

import chex
import jax
import jax.numpy as jnp
import optax
from flax import struct

from ..config import OptimizationConfig, load_config
from ..scene.point import GaussianPool
from ..scene.types import PARAMETER_NAMES, ParameterArrays, ParameterGradients, VisibleClusters

_DEFAULT_OPTIMIZATION = load_config().optimization


@struct.dataclass
class AdamState:
    m: ParameterArrays
    v: ParameterArrays
    step: chex.Array  # [C], reset when a slot is reused


def create_adam_state(pool: GaussianPool) -> AdamState:
    """Initialize independent first and second moments for every pool slot."""
    parameters = ParameterArrays(*(getattr(pool, name) for name in PARAMETER_NAMES))
    return _ADAM_TRANSFORM.init(parameters)


def masked_adam_update(
    pool: GaussianPool,
    state: AdamState,
    gradients: ParameterGradients,
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[GaussianPool, AdamState]:
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


def optax_adam_update(
    pool: GaussianPool,
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
) -> tuple[GaussianPool, AdamState]:
    """Apply the Optax transformation to fixed pool slots under nnx.jit."""
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
    parameters = ParameterArrays(*(getattr(pool, name) for name in PARAMETER_NAMES))
    learning_rates = _parameter_learning_rates(
        pool.sh.shape[1], step, spatial_scale, max_steps, optimization
    )
    parameter_updates, state = _ADAM_TRANSFORM.update(
        ParameterArrays(*gradients), state, parameters, active=active_slots, rates=learning_rates
    )
    parameters = optax.apply_updates(parameters, parameter_updates)
    return pool.replace(**{name: getattr(parameters, name) for name in PARAMETER_NAMES}), state


def sparse_adam_update(
    pool: GaussianPool,
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
) -> tuple[GaussianPool, AdamState]:
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
