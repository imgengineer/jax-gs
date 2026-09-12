"""Readable pure-JAX topology operations for fixed-capacity Gaussian models.

The upstream operations resize PyTorch parameters and their optimizer objects.
This port keeps physical shapes fixed: duplicate/split/sample operations fill
inactive slots, removal marks slots inactive, and insufficient capacity raises
before mutation. These functions are eager topology boundaries rather than
JIT-compatible training steps; the strategy classes provide compiled variants.
Optional scenes receive fixed-slot lineage transactions without being resized.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from ..math import quat_scale_to_covar_preci, quat_to_rotmat
from ..model import GaussianModel, inverse_sigmoid
from ..optimizers import reset_optimizer_indices, reset_optimizer_slots
from ..relocation import compute_relocation

DEFAULT_MCMC_OPACITY_T = 0.005
DEFAULT_MCMC_OPACITY_K = 100.0


def _resolve_noise_scale(noise_scale: float | None, scaler: float | None) -> float:
    if noise_scale is None:
        if scaler is None:
            raise TypeError("noise_scale must be provided")
        return float(scaler)
    if scaler is not None and float(noise_scale) != float(scaler):
        raise ValueError(
            "noise_scale and scaler aliases were both provided with different values"
        )
    return float(noise_scale)


def _multinomial_sample(
    weights: jax.Array,
    n: int,
    replacement: bool = True,
    *,
    key: jax.Array | None = None,
) -> jax.Array:
    """Sample weighted indices with an explicit, reproducible JAX RNG."""

    weights = jnp.asarray(weights)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    finite_weights = jnp.where(jnp.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if n > 0 and float(jax.device_get(jnp.sum(finite_weights))) <= 0.0:
        raise ValueError("weights must contain at least one positive value")
    if not replacement and n > int(jax.device_get(jnp.count_nonzero(finite_weights))):
        raise ValueError(
            "cannot sample more positive-weight entries without replacement"
        )
    if key is None:
        key = jax.random.key(0)
    probabilities = finite_weights / jnp.maximum(
        jnp.sum(finite_weights), jnp.asarray(1.0, finite_weights.dtype)
    )
    return jax.lax.stop_gradient(
        jax.random.choice(
            key,
            weights.shape[0],
            shape=(n,),
            replace=replacement,
            p=probabilities,
        ).astype(jnp.int32)
    )


def _update_param_with_optimizer(
    param_fn: Callable[[str, jax.Array], jax.Array],
    optimizer_fn: Callable[[str, jax.Array], jax.Array],
    params: MutableMapping[str, jax.Array],
    optimizers: Mapping[str, Any],
    names: Sequence[str] | None = None,
) -> None:
    """Apply a host topology transform to plain arrays without optimizer state.

    NNX optimizer topology is handled by the fixed-capacity operations below.
    A non-empty per-parameter optimizer mapping is rejected rather than leaving
    moments silently misaligned.
    """

    del optimizer_fn
    if optimizers:
        raise TypeError(
            "dynamic per-parameter optimizer resizing is not supported; use "
            "GaussianModel with one NNX Optimizer"
        )
    for name in list(params) if names is None else names:
        params[name] = param_fn(name, jnp.asarray(params[name]))


def _indices(mask: jax.Array) -> jax.Array:
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    count = int(jax.device_get(jnp.count_nonzero(mask)))
    return jnp.nonzero(mask, size=count)[0].astype(jnp.int32)


def _check_topology_inputs(
    params: GaussianModel,
    optimizers: nnx.Optimizer,
    state: Any,
    mask: jax.Array,
    scene: Any,
) -> jax.Array:
    if not isinstance(params, GaussianModel):
        raise TypeError("params must be a GaussianModel")
    if not isinstance(optimizers, nnx.Optimizer):
        raise TypeError("optimizers must be one Flax NNX Optimizer")
    if scene is not None:
        validate_capacity = getattr(scene, "validate_slot_capacity", None)
        apply_transaction = getattr(scene, "apply_slot_transaction", None)
        if not callable(validate_capacity) or not callable(apply_transaction):
            raise TypeError("scene must implement the fixed-slot transaction interface")
        validate_capacity(params.capacity)
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if mask.shape != (params.capacity,):
        raise ValueError(f"mask must have shape ({params.capacity},)")
    for name in ("grad_accum", "visible_count", "max_radii"):
        if not hasattr(state, name):
            raise TypeError("state must be a StrategyState-compatible NNX module")
    return mask


def _commit_scene_slot_transaction(
    scene: Any,
    *,
    source_slots: jax.Array,
    target_slots: jax.Array,
    valid: jax.Array | None = None,
) -> None:
    if scene is None:
        return
    scene.apply_slot_transaction(
        source_slots=source_slots,
        target_slots=target_slots,
        valid=valid,
    )


def _copy_model_rows(
    model: GaussianModel, targets: jax.Array, sources: jax.Array
) -> None:
    names = (
        "means",
        "log_scales",
        "quats",
        "opacity_logits",
    ) + (("features", "colors") if model.has_appearance else ("sh0", "sh_rest"))
    for name in names:
        variable = getattr(model, name)
        variable[...] = variable[...].at[targets].set(variable[...][sources])


def _copy_state_rows(state: Any, targets: jax.Array, sources: jax.Array) -> None:
    for name in ("grad_accum", "visible_count", "max_radii", "dynamic_mask"):
        if hasattr(state, name):
            variable = getattr(state, name)
            variable[...] = variable[...].at[targets].set(variable[...][sources])


def _zero_state_rows(state: Any, mask: jax.Array) -> None:
    for name in ("grad_accum", "visible_count", "max_radii"):
        variable = getattr(state, name)
        variable[...] = jnp.where(mask, 0.0, variable[...])
    if hasattr(state, "dynamic_mask"):
        state.dynamic_mask[...] = jnp.where(mask, False, state.dynamic_mask[...])


def _free_slots(model: GaussianModel, count: int) -> jax.Array:
    free = _indices(~model.active_mask[...])
    if free.shape[0] < count:
        raise RuntimeError(
            f"topology operation needs {count} inactive slots but only "
            f"{free.shape[0]} are available"
        )
    return free[:count]


def duplicate(
    params: GaussianModel,
    optimizers: nnx.Optimizer,
    state: Any,
    mask: jax.Array,
    scene: Any = None,
) -> jax.Array:
    """Copy selected active rows into inactive slots and return target IDs."""

    mask = _check_topology_inputs(params, optimizers, state, mask, scene)
    sources = _indices(mask & params.active_mask[...])
    targets = _free_slots(params, sources.shape[0])
    _copy_model_rows(params, targets, sources)
    _copy_state_rows(state, targets, sources)
    params.active_mask[...] = params.active_mask[...].at[targets].set(True)
    if targets.shape[0] > 0:
        reset_optimizer_indices(
            optimizers,
            targets,
            jnp.ones(targets.shape, dtype=jnp.bool_),
            capacity=params.capacity,
        )
    _commit_scene_slot_transaction(scene, source_slots=sources, target_slots=targets)
    return targets


def split(
    params: GaussianModel,
    optimizers: nnx.Optimizer,
    state: Any,
    mask: jax.Array,
    revised_opacity: bool = False,
    scene: Any = None,
    *,
    key: jax.Array | None = None,
) -> jax.Array:
    """Replace selected rows with one child and place the second in free slots."""

    mask = _check_topology_inputs(params, optimizers, state, mask, scene)
    sources = _indices(mask & params.active_mask[...])
    targets = _free_slots(params, sources.shape[0])
    if sources.shape[0] == 0:
        return targets
    if key is None:
        key = jax.random.key(0)
    scales = jnp.exp(params.log_scales[...][sources])
    rotations = quat_to_rotmat(params.quats[...][sources])
    standard_noise = jax.random.normal(
        key, (2, sources.shape[0], 3), dtype=scales.dtype
    )
    offsets = jnp.einsum("nij,nj,bnj->bni", rotations, scales, standard_noise)
    source_means = params.means[...][sources]
    source_log_scales = params.log_scales[...][sources]
    child_means = source_means[None, ...] + offsets
    child_log_scales = source_log_scales - jnp.log(1.6)
    child_opacities = params.opacity_logits[...][sources]
    if revised_opacity:
        revised = 1.0 - jnp.sqrt(1.0 - jax.nn.sigmoid(child_opacities))
        child_opacities = inverse_sigmoid(revised)

    _copy_model_rows(params, targets, sources)
    _copy_state_rows(state, targets, sources)
    params.means[...] = params.means[...].at[sources].set(child_means[0])
    params.means[...] = params.means[...].at[targets].set(child_means[1])
    params.log_scales[...] = params.log_scales[...].at[sources].set(child_log_scales)
    params.log_scales[...] = params.log_scales[...].at[targets].set(child_log_scales)
    params.opacity_logits[...] = (
        params.opacity_logits[...].at[sources].set(child_opacities)
    )
    params.opacity_logits[...] = (
        params.opacity_logits[...].at[targets].set(child_opacities)
    )
    params.active_mask[...] = params.active_mask[...].at[targets].set(True)
    changed = jnp.zeros((params.capacity,), dtype=jnp.bool_)
    changed = changed.at[sources].set(True).at[targets].set(True)
    reset_optimizer_slots(optimizers, changed)
    _commit_scene_slot_transaction(scene, source_slots=sources, target_slots=targets)
    return targets


def remove(
    params: GaussianModel,
    optimizers: nnx.Optimizer,
    state: Any,
    mask: jax.Array,
    scene: Any = None,
) -> jax.Array:
    """Mark selected active rows inactive and clear aligned runtime state."""

    mask = _check_topology_inputs(params, optimizers, state, mask, scene)
    removed = mask & params.active_mask[...]
    params.active_mask[...] &= ~removed
    _zero_state_rows(state, removed)
    reset_optimizer_slots(optimizers, removed)
    return _indices(removed)


def reset_opa(
    params: GaussianModel,
    optimizers: nnx.Optimizer,
    state: Any,
    value: float,
) -> None:
    """Clamp activated opacity and clear the changed opacity moments."""

    del state
    from . import reset_opacities

    reset_opacities(params, optimizers, maximum_opacity=value)


def relocate(
    params: GaussianModel,
    optimizers: nnx.Optimizer,
    state: Any,
    mask: jax.Array,
    binoms: jax.Array,
    min_opacity: float = 0.005,
    scene: Any = None,
    *,
    key: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Relocate selected live slots from opacity-weighted active donors."""

    mask = _check_topology_inputs(params, optimizers, state, mask, scene)
    dead = mask & params.active_mask[...]
    dead_indices = _indices(dead)
    donor_indices = _indices(params.active_mask[...] & ~dead)
    if dead_indices.shape[0] == 0:
        return dead_indices, dead_indices
    if donor_indices.shape[0] == 0:
        raise RuntimeError("cannot relocate without an active donor")
    if key is None:
        key = jax.random.key(0)
    donor_local = _multinomial_sample(
        jax.nn.sigmoid(params.opacity_logits[...][donor_indices]),
        dead_indices.shape[0],
        key=key,
    )
    sampled = donor_indices[donor_local]
    counts = jnp.zeros((params.capacity,), dtype=jnp.int32).at[sampled].add(1)
    new_opacity, new_scales = compute_relocation(
        jax.nn.sigmoid(params.opacity_logits[...][sampled]),
        jnp.exp(params.log_scales[...][sampled]),
        counts[sampled] + 1,
        binoms,
        min_opacity=min_opacity,
    )
    params.opacity_logits[...] = (
        params.opacity_logits[...].at[sampled].set(inverse_sigmoid(new_opacity))
    )
    params.log_scales[...] = params.log_scales[...].at[sampled].set(jnp.log(new_scales))
    _copy_model_rows(params, dead_indices, sampled)
    _copy_state_rows(state, dead_indices, sampled)
    changed = jnp.zeros((params.capacity,), dtype=jnp.bool_)
    changed = changed.at[sampled].set(True).at[dead_indices].set(True)
    _zero_state_rows(state, changed)
    reset_optimizer_slots(optimizers, changed)
    _commit_scene_slot_transaction(
        scene, source_slots=sampled, target_slots=dead_indices
    )
    return dead_indices, sampled


def sample_add(
    params: GaussianModel,
    optimizers: nnx.Optimizer,
    state: Any,
    n: int,
    binoms: jax.Array,
    min_opacity: float = 0.005,
    scene: Any = None,
    *,
    key: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Sample active donors into ``n`` inactive slots."""

    empty_mask = jnp.zeros((params.capacity,), dtype=jnp.bool_)
    _check_topology_inputs(params, optimizers, state, empty_mask, scene)
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    targets = _free_slots(params, n)
    donors = _indices(params.active_mask[...])
    if n == 0:
        return targets, targets
    if donors.shape[0] == 0:
        raise RuntimeError("cannot sample new slots without an active donor")
    if key is None:
        key = jax.random.key(0)
    donor_local = _multinomial_sample(
        jax.nn.sigmoid(params.opacity_logits[...][donors]), n, key=key
    )
    sampled = donors[donor_local]
    counts = jnp.zeros((params.capacity,), dtype=jnp.int32).at[sampled].add(1)
    new_opacity, new_scales = compute_relocation(
        jax.nn.sigmoid(params.opacity_logits[...][sampled]),
        jnp.exp(params.log_scales[...][sampled]),
        counts[sampled] + 1,
        binoms,
        min_opacity=min_opacity,
    )
    params.opacity_logits[...] = (
        params.opacity_logits[...].at[sampled].set(inverse_sigmoid(new_opacity))
    )
    params.log_scales[...] = params.log_scales[...].at[sampled].set(jnp.log(new_scales))
    _copy_model_rows(params, targets, sampled)
    _copy_state_rows(state, targets, sampled)
    params.active_mask[...] = params.active_mask[...].at[targets].set(True)
    changed = jnp.zeros((params.capacity,), dtype=jnp.bool_)
    changed = changed.at[sampled].set(True).at[targets].set(True)
    _zero_state_rows(state, changed)
    reset_optimizer_slots(optimizers, changed)
    _commit_scene_slot_transaction(scene, source_slots=sampled, target_slots=targets)
    return targets, sampled


def mcmc_position_perturbation(
    positions: jax.Array,
    quats: jax.Array,
    log_scales: jax.Array,
    opacity_logits: jax.Array,
    noise_scale: float | jax.Array,
    *,
    key: jax.Array,
    t: float = DEFAULT_MCMC_OPACITY_T,
    k: float = DEFAULT_MCMC_OPACITY_K,
    active_mask: jax.Array | None = None,
) -> jax.Array:
    """Return covariance- and opacity-weighted MCMC position perturbations."""

    positions = jnp.asarray(positions)
    covariance, _ = quat_scale_to_covar_preci(
        quats,
        jnp.exp(log_scales),
        compute_covar=True,
        compute_preci=False,
        triu=False,
    )
    opacity = jax.nn.sigmoid(jnp.asarray(opacity_logits).reshape(-1))
    gate = jax.nn.sigmoid(-float(k) * (opacity - float(t)))
    noise = jax.random.normal(key, positions.shape, dtype=positions.dtype)
    weighted_noise = (
        noise * gate[:, None] * jnp.asarray(noise_scale, dtype=positions.dtype)
    )
    delta = jnp.einsum("nij,nj->ni", covariance, weighted_noise)
    if active_mask is not None:
        delta = jnp.where(jnp.asarray(active_mask)[:, None], delta, 0.0)
    return positions + delta


def _cuda_fused_mcmc_perturb(
    positions: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    opacities: jax.Array,
    noise_scale: float | None = None,
    *,
    scaler: float | None = None,
    t: float = DEFAULT_MCMC_OPACITY_T,
    k: float = DEFAULT_MCMC_OPACITY_K,
) -> bool:
    """Report that the CUDA-only in-place backend is not used by pure JAX."""

    del positions, quats, scales, opacities, t, k
    _resolve_noise_scale(noise_scale, scaler)
    return False


def inject_noise_to_position(
    params: GaussianModel | MutableMapping[str, jax.Array],
    optimizers: Any,
    state: Any,
    noise_scale: float | None = None,
    *,
    scaler: float | None = None,
    t: float = DEFAULT_MCMC_OPACITY_T,
    k: float = DEFAULT_MCMC_OPACITY_K,
    key: jax.Array | None = None,
) -> jax.Array:
    """Apply the pure-JAX MCMC perturbation and return updated positions."""

    del optimizers, state
    resolved = _resolve_noise_scale(noise_scale, scaler)
    if key is None:
        key = jax.random.key(0)
    if isinstance(params, GaussianModel):
        updated = mcmc_position_perturbation(
            params.means[...],
            params.quats[...],
            params.log_scales[...],
            params.opacity_logits[...],
            resolved,
            key=key,
            t=t,
            k=k,
            active_mask=params.active_mask[...],
        )
        params.means[...] = updated
        return updated
    required = {"means", "quats", "scales", "opacities"}
    if not required.issubset(params):
        missing = sorted(required - set(params))
        raise KeyError(f"params missing required keys: {missing}")
    updated = mcmc_position_perturbation(
        params["means"],
        params["quats"],
        params["scales"],
        params["opacities"],
        resolved,
        key=key,
        t=t,
        k=k,
    )
    params["means"] = updated
    return updated


__all__ = [
    "DEFAULT_MCMC_OPACITY_K",
    "DEFAULT_MCMC_OPACITY_T",
    "duplicate",
    "inject_noise_to_position",
    "mcmc_position_perturbation",
    "relocate",
    "remove",
    "reset_opa",
    "sample_add",
    "split",
]
