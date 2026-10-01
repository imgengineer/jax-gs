"""Apply a row-local Optax transformation only to the pool rows of visible clusters.

A dense Optax update traverses every pool slot, including free slots and
clusters outside the view. This Pallas (Triton) kernel instead traces the
transformation's own update code on blocks of rows from visible clusters;
other rows keep their parameters and state. Each program loads a block of
rows from every parameter, state and compact gradient leaf, calls
``tx.update`` and ``optax.apply_updates`` on those blocks, and stores the
rows that are alive.

The transformation must be row-local: a slot's update may depend only on
that slot's gradient, state and parameters, on scalar arguments and on
constants broadcast against a parameter row. Every state array leaf must
have the pool capacity as its leading dimension. Triton requires power-of-two
block sizes, so each row is padded to power-of-two dimensions; padded
elements are masked on load (as zeros) and on store.

On this GPU generation Pallas lowers only through its Triton backend, which
JAX 0.11 marks as deprecated (Mosaic GPU does not support sm_120). When that
backend is unavailable, ``available()`` is false and callers keep the dense
Optax update.
"""

import functools
import math
from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..scene.types import VisibleClusters

# Rows per program; each visible cluster is covered by ceil(cluster_size / _ROWS) programs.
_ROWS = 32
_NUM_WARPS = 4


@functools.cache
def available() -> bool:
    """Whether this kernel can run: a JAX CUDA device and Pallas's Triton backend."""
    if jax.default_backend() != "gpu":
        return False
    try:
        from jax.experimental.pallas import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _power_of_two(size: int) -> int:
    return 1 << max(0, (size - 1).bit_length())


def _offsets(first_row, rows: int | None, row_shape: Sequence[int], padded: Sequence[int]):
    """Flat element offsets of a [rows, *padded] block (rows=None: one unindexed row)."""
    lead = () if rows is None else (rows,)
    shape = (*lead, *padded)
    size = math.prod(row_shape)
    offset = jnp.zeros(shape, jnp.int32)
    if rows is not None:
        offset = (first_row + jax.lax.broadcasted_iota(jnp.int32, shape, 0)) * size
    mask = None
    stride = size
    for axis, (dim, padded_dim) in enumerate(zip(row_shape, padded, strict=True)):
        stride //= dim
        index = jax.lax.broadcasted_iota(jnp.int32, shape, axis + len(lead))
        offset = offset + index * stride
        if padded_dim != dim:
            mask = index < dim if mask is None else mask & (index < dim)
    return offset, mask


def _with_rows(mask, rows_valid):
    return rows_valid if mask is None else mask & rows_valid


def _load(ref, mask):
    from jax.experimental.pallas import triton as plt

    return plt.load(ref) if mask is None else plt.load(ref, mask=mask, other=0)


def update_visible_clusters(
    tx: optax.GradientTransformationExtraArgs,
    params: chex.ArrayTree,
    state: chex.ArrayTree,
    gradients: chex.ArrayTree,
    alive: chex.Array,
    clusters: VisibleClusters,
    *,
    cluster_size: int,
    **extra_args: chex.ArrayTree,
) -> tuple[chex.ArrayTree, chex.ArrayTree]:
    """Run ``tx`` on the alive rows of visible clusters; return new params and state.

    ``gradients`` match ``params`` but are compact: row ``i`` of visible cluster
    ``k`` is row ``k * cluster_size + i``. A gradient leaf may cover only a
    prefix of its parameter's trailing dimensions; missing entries are zero.
    ``tx.update`` receives ``active`` (alive rows of the block) and
    ``extra_args``, whose leaves are Python scalars, 0-d arrays or arrays
    broadcastable against a parameter row with a leading unit dimension.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    ids, count = clusters
    capacity = alive.shape[0]
    param_leaves, param_tree = jax.tree.flatten(params)
    state_leaves, state_tree = jax.tree.flatten(state)
    grad_leaves = param_tree.flatten_up_to(gradients)
    extra_leaves, extra_tree = jax.tree.flatten(extra_args)
    extra_leaves = [jnp.asarray(x) if isinstance(x, np.ndarray) else x for x in extra_leaves]
    for leaf in (*param_leaves, *state_leaves):
        if leaf.ndim == 0 or leaf.shape[0] != capacity:
            raise ValueError("parameter and state leaves must have one row per pool slot")
    for gradient, parameter in zip(grad_leaves, param_leaves, strict=True):
        if gradient.shape[0] != capacity or any(
            g > p for g, p in zip(gradient.shape[1:], parameter.shape[1:], strict=True)
        ):
            raise ValueError("compact gradients must fit their parameters")
    pooled = param_leaves + state_leaves
    row_shapes = [leaf.shape[1:] for leaf in pooled]
    padded_shapes = [tuple(_power_of_two(d) for d in shape) for shape in row_shapes]
    array_extras = [i for i, leaf in enumerate(extra_leaves) if isinstance(leaf, jax.Array)]
    parts = -(-cluster_size // _ROWS)
    inputs = [ids, count, alive.astype(jnp.int8)]
    inputs += [leaf.reshape(-1) for leaf in pooled]
    inputs += [leaf.reshape(-1) for leaf in grad_leaves]
    inputs += [extra_leaves[i].reshape(-1) for i in array_extras]
    first_pooled = 3
    aliases = {first_pooled + k: k for k in range(len(pooled))}

    def kernel(*refs):
        ids_ref, count_ref, alive_ref = refs[:3]
        pooled_refs = refs[3 : 3 + len(pooled)]
        k = 3 + len(pooled)
        grad_refs = refs[k : k + len(grad_leaves)]
        k += len(grad_leaves)
        extra_refs = refs[k : k + len(array_extras)]
        out_refs = refs[k + len(array_extras) :]
        program = pl.program_id(0)
        visible = program // parts

        @pl.when(visible < count_ref[0])
        def _():
            local = jax.lax.broadcasted_iota(jnp.int32, (_ROWS,), 0)
            row0 = ids_ref[visible] * cluster_size + program % parts * _ROWS
            compact0 = visible * cluster_size + program % parts * _ROWS
            row = row0 + local
            rows_valid = (program % parts * _ROWS + local < cluster_size) & (row < capacity)
            active = rows_valid & (plt.load(alive_ref.at[row], mask=rows_valid, other=0) != 0)
            blocks, offsets, store_masks = [], [], []
            for ref, shape, padded in zip(pooled_refs, row_shapes, padded_shapes, strict=True):
                expand = (_ROWS,) + (1,) * len(padded)
                offset, mask = _offsets(row0, _ROWS, shape, padded)
                blocks.append(_load(ref.at[offset], _with_rows(mask, rows_valid.reshape(expand))))
                offsets.append(offset)
                store_masks.append(_with_rows(mask, active.reshape(expand)))
            gradient_blocks = []
            for ref, gradient, padded in zip(
                grad_refs, grad_leaves, padded_shapes[: len(param_leaves)], strict=True
            ):
                expand = (_ROWS,) + (1,) * len(padded)
                offset, mask = _offsets(compact0, _ROWS, gradient.shape[1:], padded)
                gradient_blocks.append(
                    _load(ref.at[offset], _with_rows(mask, rows_valid.reshape(expand)))
                )
            extras = list(extra_leaves)
            for ref, i in zip(extra_refs, array_extras, strict=True):
                shape = extra_leaves[i].shape
                if not shape:
                    extras[i] = ref[0]
                    continue
                offset, mask = _offsets(0, None, shape, tuple(_power_of_two(d) for d in shape))
                extras[i] = _load(ref.at[offset], mask)
            parameters = param_tree.unflatten(blocks[: len(param_leaves)])
            updates, new_state = tx.update(
                param_tree.unflatten(gradient_blocks),
                state_tree.unflatten(blocks[len(param_leaves) :]),
                parameters,
                active=active,
                **extra_tree.unflatten(extras),
            )
            results = jax.tree.leaves(optax.apply_updates(parameters, updates))
            results += state_tree.flatten_up_to(new_state)
            for out, value, offset, mask in zip(
                out_refs, results, offsets, store_masks, strict=True
            ):
                plt.store(out.at[offset], value.astype(out.dtype), mask=mask)

    outputs = pl.pallas_call(
        kernel,
        out_shape=[jax.ShapeDtypeStruct((leaf.size,), leaf.dtype) for leaf in pooled],
        grid=(ids.shape[0] * parts,),
        input_output_aliases=aliases,
        compiler_params=plt.CompilerParams(num_warps=_NUM_WARPS, num_stages=1),
    )(*inputs)
    outputs = [out.reshape(leaf.shape) for out, leaf in zip(outputs, pooled, strict=True)]
    return (
        param_tree.unflatten(outputs[: len(param_leaves)]),
        state_tree.unflatten(outputs[len(param_leaves) :]),
    )
