"""Fixed-capacity Adam state, LiteGS learning rates and Optax or CuTe updates."""

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
    sh_pullback_center: chex.Array | None = None,
) -> tuple[GaussianArrays, AdamState]:
    """Apply an Optax transformation to fixed pool slots, preserving momentum in every SH band.

    transform defaults to LiteGS Adam (create_adam_transform); it receives the
    active slots and per-field learning rates as extra arguments.
    program_shape tunes the visible-cluster kernel for the transformation.
    active_degree declares that higher-order projection gradients are zero.
    Their moments still decay and update parameters when slots are visible.
    With compacted_clusters and compact gradients, the same transformation
    runs on the GPU only over visible clusters, instead of every pool slot.
    sh_pullback_center declares that the SH gradient holds masked RGB
    cotangents; reconstruct its coefficients from the original positions
    inside the transformation, including on the dense fallback path.
    """
    transform = _ADAM_TRANSFORM if transform is None else transform
    sh_dim = pool.sh.shape[1]
    active_sh_dim = sh_dim if active_degree is None else (active_degree + 1) ** 2
    if active_degree is not None and (active_degree < 0 or active_sh_dim > sh_dim):
        raise ValueError("active_degree must fit the pool's SH coefficients")
    if active_sh_dim < sh_dim and sh_pullback_center is None:
        gradients = (*gradients[:-1], gradients[-1][:, :active_sh_dim])
    learning_rates = _parameter_learning_rates(
        pool.sh.shape[1], step, spatial_scale, max_steps, optimization
    )
    transform_kwargs = {"rates": learning_rates}
    if sh_pullback_center is not None:
        from .sh_pullback import with_sh_pullback

        degree = int(sh_dim**0.5) - 1 if active_degree is None else active_degree
        transform = with_sh_pullback(transform, degree)
        transform_kwargs["sh_center"] = sh_pullback_center[None, :]
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
            **transform_kwargs,
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
    if active_sh_dim < sh_dim and sh_pullback_center is None:
        # XLA broadcasts the constant tail in the update fusion. No full-size
        # compact-gradient gather is needed for unused SH coefficients.
        gradients = (
            *gradients[:-1],
            jnp.pad(gradients[-1], ((0, 0), (0, sh_dim - active_sh_dim), (0, 0))),
        )
    parameter_updates, state = transform.update(
        ParameterArrays(*gradients), state, parameters, active=active_slots, **transform_kwargs
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
