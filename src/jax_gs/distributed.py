from __future__ import annotations

import operator
from collections.abc import Hashable, Sequence
import os
from typing import Any

import jax
import jax.numpy as jnp


def _static_world_size(world_size: int) -> int:
    try:
        world_size = operator.index(world_size)
    except TypeError as exc:
        raise TypeError("world_size must be a static integer") from exc
    if world_size < 1:
        raise ValueError("world_size must be positive")
    return world_size


def _check_collective_axis(world_size: int, axis_name: Hashable | None) -> Hashable:
    if axis_name is None:
        raise ValueError("axis_name is required when world_size is greater than one")
    actual_size = jax.lax.axis_size(axis_name)
    if actual_size != world_size:
        raise ValueError(
            f"axis {axis_name!r} has size {actual_size}, expected {world_size}"
        )
    return axis_name


def all_gather_int32(
    world_size: int,
    value: int | jax.Array,
    device: Any | None = None,
    *,
    axis_name: Hashable | None = None,
) -> jax.Array:
    """Gather one int32 scalar from every rank into a static ``[W]`` array.

    ``device`` is accepted only as a migration aid for gsplat's PyTorch API and
    is ignored. The ``world_size == 1`` branch does not require a bound JAX
    collective axis and is safe inside an ordinary :func:`jax.jit`.
    """

    del device
    world_size = _static_world_size(world_size)
    value = jnp.asarray(value, dtype=jnp.int32)
    if value.shape != ():
        raise ValueError("value must be an int32 scalar")
    if world_size == 1:
        return value[None]
    axis_name = _check_collective_axis(world_size, axis_name)
    return jax.lax.all_gather(value, axis_name, axis=0, tiled=False)


def all_to_all_int32(
    world_size: int,
    values: Sequence[int | jax.Array] | jax.Array,
    device: Any | None = None,
    *,
    axis_name: Hashable | None = None,
) -> jax.Array:
    """Exchange one int32 value per destination using a static ``[W]`` array."""

    del device
    world_size = _static_world_size(world_size)
    values = jnp.asarray(values, dtype=jnp.int32)
    if values.shape != (world_size,):
        raise ValueError(f"values must have shape ({world_size},)")
    if world_size == 1:
        return values
    axis_name = _check_collective_axis(world_size, axis_name)
    return jax.lax.all_to_all(
        values, axis_name, split_axis=0, concat_axis=0, tiled=False
    )


def all_gather_tensor_list(
    world_size: int,
    tensor_list: Sequence[jax.Array],
    *,
    axis_name: Hashable | None = None,
) -> list[jax.Array]:
    """Gather fixed-size tensor leaves along their leading dimension.

    Leaves are gathered separately, preserving heterogeneous dtypes and
    avoiding the large concatenation temporary used by gsplat's PyTorch helper.
    """

    world_size = _static_world_size(world_size)
    tensors = [jnp.asarray(tensor) for tensor in tensor_list]
    if not tensors:
        return []
    leading_size = tensors[0].shape[0] if tensors[0].ndim else None
    if leading_size is None:
        raise ValueError("tensor leaves must have rank at least one")
    for tensor in tensors:
        if tensor.ndim < 1 or tensor.shape[0] != leading_size:
            raise ValueError("all tensor leaves must share their leading size")
    if world_size == 1:
        return tensors
    axis_name = _check_collective_axis(world_size, axis_name)
    return [
        jax.lax.all_gather(tensor, axis_name, axis=0, tiled=True)
        for tensor in tensors
    ]


def _static_equal_splits(
    name: str,
    splits: Sequence[int | jax.Array] | None,
    world_size: int,
    leading_size: int,
) -> None:
    if splits is None:
        return
    try:
        static_splits = tuple(operator.index(value) for value in splits)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a static sequence; runtime ragged splits are not JIT-safe"
        ) from exc
    if len(static_splits) != world_size or sum(static_splits) != leading_size:
        raise ValueError(
            f"{name} must contain {world_size} values summing to {leading_size}"
        )
    expected = leading_size // world_size
    if any(value != expected for value in static_splits):
        raise ValueError(
            "JAX all_to_all_tensor_list supports equal padded slots only; "
            "carry runtime counts or masks separately"
        )


def all_to_all_tensor_list(
    world_size: int,
    tensor_list: Sequence[jax.Array],
    splits: Sequence[int | jax.Array] | None = None,
    output_splits: Sequence[int | jax.Array] | None = None,
    *,
    axis_name: Hashable | None = None,
) -> list[jax.Array]:
    """Exchange equal-capacity destination slots for each tensor leaf.

    Each input leading dimension is laid out as ``[dest0 slots, ..., destW
    slots]`` and must be divisible by ``world_size``. Variable gsplat/PyTorch
    splits cannot be represented by a JIT-compiled JAX array shape; pad every
    destination to the same capacity and exchange int32 counts with
    :func:`all_to_all_int32` when a validity mask is needed.
    """

    world_size = _static_world_size(world_size)
    tensors = [jnp.asarray(tensor) for tensor in tensor_list]
    if not tensors:
        return []
    if tensors[0].ndim < 1:
        raise ValueError("tensor leaves must have rank at least one")
    leading_size = tensors[0].shape[0]
    for tensor in tensors:
        if tensor.ndim < 1 or tensor.shape[0] != leading_size:
            raise ValueError("all tensor leaves must share their leading size")
    if leading_size % world_size:
        raise ValueError("the leading size must be divisible by world_size")
    _static_equal_splits("splits", splits, world_size, leading_size)
    _static_equal_splits("output_splits", output_splits, world_size, leading_size)
    if world_size == 1:
        return tensors
    axis_name = _check_collective_axis(world_size, axis_name)
    return [
        jax.lax.all_to_all(
            tensor,
            axis_name,
            split_axis=0,
            concat_axis=0,
            tiled=True,
        )
        for tensor in tensors
    ]


def _all_gather_axis(
    value: jax.Array, axis: int, axis_name: Hashable
) -> jax.Array:
    return jax.lax.all_gather(value, axis_name, axis=axis, tiled=True)


def _all_gather_feature_shards(
    value: jax.Array | tuple[jax.Array, jax.Array],
    *,
    name: str,
    local_capacity: int,
    camera_count: int,
    sh_degree: int | None,
    axis_name: Hashable,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"split {name} must contain exactly two leaves")
        gathered = []
        for leaf in value:
            leaf = jnp.asarray(leaf)
            if leaf.ndim < 1 or leaf.shape[0] != local_capacity:
                raise ValueError(
                    f"split {name} leaves must have shape [local_capacity, ...]"
                )
            gathered.append(_all_gather_axis(leaf, 0, axis_name))
        return gathered[0], gathered[1]

    value = jnp.asarray(value)
    per_camera_rank = 4 if sh_degree is not None else 3
    per_camera = (
        value.ndim == per_camera_rank
        and value.shape[0] == camera_count
    )
    gaussian_axis = 1 if per_camera else 0
    if value.ndim <= gaussian_axis or value.shape[gaussian_axis] != local_capacity:
        layout = "[C, local_capacity, ...] or [local_capacity, ...]"
        raise ValueError(f"{name} must have shape {layout}")
    return _all_gather_axis(value, gaussian_axis, axis_name)


def rasterization(
    means: jax.Array,
    quats: jax.Array,
    scales: jax.Array,
    opacities: jax.Array,
    colors: jax.Array | tuple[jax.Array, jax.Array] | None,
    viewmats: jax.Array,
    Ks: jax.Array,
    width: int,
    height: int,
    *,
    world_size: int = 1,
    axis_name: Hashable | None = None,
    active_mask: jax.Array | None = None,
    **kwargs: Any,
):
    """Static-capacity distributed wrapper around :mod:`jax_gs.rasterization`.

    The single-rank path is a zero-collective delegation. For multiple ranks,
    equally sized Gaussian shards are gathered on the named mapped axis before
    rendering the local camera batch. This compatibility path replicates the
    global Gaussian storage on every rank; for million-slot models prefer a
    fixed-slot all-to-all projection pipeline to avoid the ``world_size`` memory
    multiplier.
    """

    from .rasterization import rasterization as local_rasterization

    world_size = _static_world_size(world_size)
    kwargs.pop("distributed", None)
    if world_size == 1:
        renders, alphas, info = local_rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
            active_mask=active_mask,
            distributed=False,
            **kwargs,
        )
        info = dict(info)
        info["distributed_requested"] = jnp.asarray(True)
        info["distributed_world_size"] = jnp.asarray(1, dtype=jnp.int32)
        return renders, alphas, info

    axis_name = _check_collective_axis(world_size, axis_name)
    means = jnp.asarray(means)
    viewmats = jnp.asarray(viewmats)
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(
            "multi-rank distributed rasterization requires means with shape "
            "[local_capacity, 3]; leading batch dimensions are unsupported"
        )
    if viewmats.ndim != 3 or viewmats.shape[-2:] != (4, 4):
        raise ValueError(
            "multi-rank distributed rasterization requires viewmats with shape "
            "[C, 4, 4]; leading batch dimensions are unsupported"
        )
    local_capacity = means.shape[0]
    camera_count = viewmats.shape[0]
    if active_mask is None:
        active_mask = jnp.ones((local_capacity,), dtype=jnp.bool_)
    active_mask = jnp.asarray(active_mask, dtype=jnp.bool_)
    if active_mask.shape != (local_capacity,):
        raise ValueError("active_mask must have shape [local_capacity]")

    means = _all_gather_axis(means, 0, axis_name)
    quats = _all_gather_axis(jnp.asarray(quats), 0, axis_name)
    scales = _all_gather_axis(jnp.asarray(scales), 0, axis_name)
    opacities = _all_gather_axis(jnp.asarray(opacities), 0, axis_name)
    active_mask = _all_gather_axis(active_mask, 0, axis_name)

    sh_degree = kwargs.get("sh_degree")
    if colors is not None:
        colors = _all_gather_feature_shards(
            colors,
            name="colors",
            local_capacity=local_capacity,
            camera_count=camera_count,
            sh_degree=sh_degree,
            axis_name=axis_name,
        )
    if kwargs.get("covars") is not None:
        kwargs["covars"] = _all_gather_axis(
            jnp.asarray(kwargs["covars"]), 0, axis_name
        )
    if kwargs.get("extra_signals") is not None:
        kwargs["extra_signals"] = _all_gather_feature_shards(
            kwargs["extra_signals"],
            name="extra_signals",
            local_capacity=local_capacity,
            camera_count=camera_count,
            sh_degree=kwargs.get("extra_signals_sh_degree"),
            axis_name=axis_name,
        )
    if kwargs.get("_means2d_offset") is not None:
        means2d_offset = jnp.asarray(kwargs["_means2d_offset"])
        local_shape = (camera_count, local_capacity, 2)
        global_shape = (camera_count, local_capacity * world_size, 2)
        if means2d_offset.shape == local_shape:
            means2d_offset = _all_gather_axis(
                means2d_offset, 1, axis_name
            )
        # Training uses an already-global probe per local camera batch so its
        # screen gradient stays separate until after the per-camera norm.
        elif means2d_offset.shape != global_shape:
            raise ValueError(
                "_means2d_offset must have rank-local shape "
                f"{local_shape} or global-Gaussian shape {global_shape}"
            )
        kwargs["_means2d_offset"] = means2d_offset

    renders, alphas, info = local_rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
        active_mask=active_mask,
        distributed=False,
        **kwargs,
    )
    info = dict(info)
    info["distributed_world_size"] = jnp.asarray(world_size, dtype=jnp.int32)
    info["distributed_requested"] = jnp.asarray(True)
    info["distributed_active_mask"] = active_mask
    return renders, alphas, info


distributed_rasterization = rasterization


def cli(fn, args: Any, verbose: bool = False) -> bool:
    """Run a gsplat-style worker once in each externally launched JAX process.

    Unlike the PyTorch helper, this function does not spawn one process per
    visible GPU. JAX multi-host jobs must be launched by ``jax.distributed``,
    MPI, Slurm, or another process manager before entering this function;
    single-process multi-device work belongs inside ``pmap``/``shard_map``.
    """

    if not callable(fn):
        raise TypeError("fn must be callable")
    world_rank = jax.process_index()
    world_size = jax.process_count()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if verbose:
        print(
            f"JAX distributed worker: {world_rank + 1} / {world_size}",
            flush=True,
        )
    fn(local_rank, world_rank, world_size, args)
    jax.effects_barrier()
    if verbose:
        print(
            f"JAX worker done: {world_rank + 1} / {world_size}",
            flush=True,
        )
    return True


__all__ = [
    "all_gather_int32",
    "all_gather_tensor_list",
    "all_to_all_int32",
    "all_to_all_tensor_list",
    "cli",
    "distributed_rasterization",
    "rasterization",
]
