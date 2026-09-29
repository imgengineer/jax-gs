import chex
import jax
import jax.numpy as jnp
import optax
from flax import struct

from ..config import load_config
from ..scene.point import GaussianPool

_DEFAULT_OPTIMIZATION = load_config().optimization


@struct.dataclass
class Moments:
    xyz: chex.Array
    log_scale: chex.Array
    rotation: chex.Array
    opacity: chex.Array
    sh: chex.Array


@struct.dataclass
class AdamState:
    m: Moments
    v: Moments
    step: chex.Array  # [C], reset when a slot is reused


def create_adam_state(pool: GaussianPool) -> AdamState:
    params = Moments(
        *(getattr(pool, name) for name in ("xyz", "log_scale", "rotation", "opacity", "sh"))
    )
    return _ADAM.init(params)


def masked_adam_update(
    pool: GaussianPool,
    state: AdamState,
    gradients: tuple[chex.Array, ...],
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[GaussianPool, AdamState]:
    names = ("xyz", "log_scale", "rotation", "opacity", "sh")
    step = state.step + pool.alive.astype(jnp.int32)
    next_pool = pool
    m_fields, v_fields = {}, {}
    for name, gradient in zip(names, gradients, strict=True):
        value = getattr(pool, name)
        mask = pool.alive.reshape((-1,) + (1,) * (value.ndim - 1))
        old_m, old_v = getattr(state.m, name), getattr(state.v, name)
        m = jnp.where(mask, beta1 * old_m + (1 - beta1) * gradient, old_m)
        v = jnp.where(mask, beta2 * old_v + (1 - beta2) * gradient**2, old_v)
        correction1 = (1 - beta1 ** jnp.maximum(step, 1)).reshape(mask.shape)
        correction2 = (1 - beta2 ** jnp.maximum(step, 1)).reshape(mask.shape)
        updated = value - learning_rate * (m / correction1) / (jnp.sqrt(v / correction2) + eps)
        next_pool = next_pool.replace(**{name: jnp.where(mask, updated, value)})
        m_fields[name], v_fields[name] = m, v
    return next_pool, AdamState(Moments(**m_fields), Moments(**v_fields), step)


def clear_slots(state: AdamState, slots: chex.Array) -> AdamState:
    """Reset only newly allocated slots; slots is a fixed-shape boolean mask."""
    m_fields, v_fields = {}, {}
    for name in ("xyz", "log_scale", "rotation", "opacity", "sh"):
        value = getattr(state.m, name)
        mask = slots.reshape((-1,) + (1,) * (value.ndim - 1))
        m_fields[name] = jnp.where(mask, 0, value)
        v_fields[name] = jnp.where(mask, 0, getattr(state.v, name))
    return AdamState(Moments(**m_fields), Moments(**v_fields), jnp.where(slots, 0, state.step))


def _parameter_rates(sh_dim, step, spatial_scale, max_steps, optimization):
    op = optimization
    t = jnp.clip(step / (max_steps or op.position_lr_max_steps), 0, 1)
    xyz_lr = (
        0.0
        if op.position_lr_init == op.position_lr_final == 0
        else spatial_scale
        * jnp.exp(jnp.log(op.position_lr_init) * (1 - t) + jnp.log(op.position_lr_final) * t)
    )
    sh_lr = jnp.full((1, sh_dim, 1), op.feature_lr / 10).at[:, 0].set(op.feature_lr)
    return Moments(xyz_lr, op.scaling_lr, op.rotation_lr, op.opacity_lr, sh_lr)


def adam_transform():
    """Optax transformation matching LiteGS Adam (no bias correction).

    Optax's built-in Adam always corrects the moments. Use its moment helpers
    with LiteGS's normalization and freeze both moments in inactive slots.
    The existing fixed-capacity state also supports slot reuse and reordering.
    """

    def init_fn(params):
        zero = optax.tree.zeros_like(params)
        return AdamState(zero, zero, jnp.zeros(params.xyz.shape[0], jnp.int32))

    def update_fn(gradients, state, params=None, *, active, rates):
        del params
        # Static checks run during tracing; no device values or callbacks enter
        # the compiled update. In particular, mask broadcasting must be exact.
        chex.assert_rank(state.step, 1)
        chex.assert_shape(active, state.step.shape)
        chex.assert_type(active, jnp.bool_)
        chex.assert_trees_all_equal_shapes_and_dtypes(gradients, state.m, state.v)
        m = optax.tree.update_moment(gradients, state.m, 0.9, 1)
        v = optax.tree.update_moment_per_elem_norm(gradients, state.v, 0.999, 2)
        updates = jax.tree.map(
            lambda mean, variance, rate: -rate * mean / (jnp.sqrt(variance) + 1e-15), m, v, rates
        )

        def select(new, old):
            mask = active.reshape((-1,) + (1,) * (new.ndim - 1))
            return jnp.where(mask, new, old)

        updates = jax.tree.map(lambda value: select(value, 0), updates)
        return updates, AdamState(
            jax.tree.map(select, m, state.m),
            jax.tree.map(select, v, state.v),
            state.step + active.astype(jnp.int32),
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


_ADAM = adam_transform()


def optax_adam_update(
    pool,
    state,
    gradients,
    visible,
    step,
    spatial_scale,
    max_steps=None,
    *,
    cluster_size=128,
    compact_gradients=False,
    optimization=_DEFAULT_OPTIMIZATION,
):
    """Apply the Optax transformation to fixed pool slots under nnx.jit."""
    active = visible & pool.alive
    if compact_gradients:
        # compact_visible_clusters preserves cluster order. Invert that order
        # without touching its undefined IDs or gradient tail. Out-of-bounds
        # gathers produce zero, including when no clusters are visible.
        capacity = pool.xyz.shape[0]
        ranks = jnp.cumsum(visible[::cluster_size], dtype=jnp.int32) - 1
        slots = jnp.arange(capacity, dtype=jnp.int32)
        indices = ranks[slots // cluster_size] * cluster_size + slots % cluster_size
        indices = jnp.where(active, indices, capacity)
        gradients = tuple(
            jnp.take(g, indices, axis=0, mode="fill", fill_value=0) for g in gradients
        )
    names = ("xyz", "log_scale", "rotation", "opacity", "sh")
    params = Moments(*(getattr(pool, name) for name in names))
    rates = _parameter_rates(pool.sh.shape[1], step, spatial_scale, max_steps, optimization)
    updates, state = _ADAM.update(Moments(*gradients), state, params, active=active, rates=rates)
    params = optax.apply_updates(params, updates)
    return pool.replace(**{name: getattr(params, name) for name in names}), state


def sparse_adam_update(
    pool,
    state,
    gradients,
    visible,
    step,
    spatial_scale,
    max_steps=None,
    *,
    compacted_clusters=None,
    cluster_size=128,
    compact_gradients=False,
    optimization=_DEFAULT_OPTIMIZATION,
):
    """LiteGS sparse Adam: per-field rates, no bias correction, eps=1e-15.

    Matches litegs/training/optimizer.py and compact.cu::adamUpdate. The
    visibility mask selects complete visible clusters, including zero grads.
    compact_gradients reads only the valid prefix in visible-cluster order;
    parameter and moment writes still address their original pool slots.
    """
    rates = _parameter_rates(pool.sh.shape[1], step, spatial_scale, max_steps, optimization)
    values, means, variances = {}, {}, {}
    active = visible & pool.alive
    for name, grad in zip(
        ("xyz", "log_scale", "rotation", "opacity", "sh"), gradients, strict=True
    ):
        rate = getattr(rates, name)
        value = getattr(pool, name)
        mask = active.reshape((-1,) + (1,) * (value.ndim - 1))
        old_m, old_v = getattr(state.m, name), getattr(state.v, name)
        if compacted_clusters is not None:
            from ..kernels.sparse_adam import update_field

            values[name], means[name], variances[name] = update_field(
                value,
                old_m,
                old_v,
                grad,
                pool.alive,
                compacted_clusters,
                optimization.feature_lr / 10 if name == "sh" else rate,
                cluster_size,
                name == "sh",
                compact_gradients,
            )
            continue
        m = 0.9 * old_m + 0.1 * grad
        v = 0.999 * old_v + 0.001 * grad**2
        values[name] = jnp.where(mask, value - rate * m / (jnp.sqrt(v) + 1e-15), value)
        means[name], variances[name] = jnp.where(mask, m, old_m), jnp.where(mask, v, old_v)
    return pool.replace(**values), AdamState(
        Moments(**means), Moments(**variances), state.step + active.astype(jnp.int32)
    )
