# pyright: reportMissingImports=false

"""Fixed-capacity intersection topology kernels through CuTe DSL."""

from __future__ import annotations

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import cuda.bindings.driver as cuda

from ._cute_compositor import (
    _SUPPORTED_CHANNELS,
    _composite_tile_forward,
    _cute_device,
    _run_backward,
    _run_forward,
    _static_float,
    _static_int,
)


_BLOCK_SIZE = 256
_COUNT_LIMIT = (1 << 30) - 1
_RADIX_BITS = 8
_RADIX_SIZE = 1 << _RADIX_BITS
_TILE_RADIX_BITS = 5
_TILE_RADIX_SIZE = 1 << _TILE_RADIX_BITS
_WARPS_PER_BLOCK = _BLOCK_SIZE // 32
_UINT64_MAX = (1 << 64) - 1
# Uint8 warp-local radix counts leave enough shared memory for two CTAs per SM
# with a 256-candidate production RGB compositor batch.
_MEGA_COMPOSITOR_BATCH_CAPACITY = 256


@cute.jit
def _saturating_add(left, right):
    return (
        cutlass.Int32(_COUNT_LIMIT)
        if left >= _COUNT_LIMIT - right
        else left + right
    )


@cute.kernel
def _scan_blocks_kernel(
    values: cute.Tensor,
    scanned: cute.Tensor,
    block_sums: cute.Tensor,
    count: int,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * _BLOCK_SIZE + thread
    shared = cutlass.utils.SmemAllocator().allocate_tensor(
        cutlass.Int32, _BLOCK_SIZE
    )
    value = cutlass.Int32(0)
    if index < count:
        value = cutlass.min(
            cutlass.max(values[index], cutlass.Int32(0)),
            cutlass.Int32(_COUNT_LIMIT),
        )
    shared[thread] = value
    cute.arch.sync_threads()
    for offset in (1, 2, 4, 8, 16, 32, 64, 128):
        addend = cutlass.Int32(0)
        if thread >= offset:
            addend = shared[thread - offset]
        cute.arch.sync_threads()
        shared[thread] = _saturating_add(shared[thread], addend)
        cute.arch.sync_threads()
    if index < count:
        scanned[index] = shared[thread]
    active = cutlass.min(_BLOCK_SIZE, count - block * _BLOCK_SIZE)
    if thread == active - 1:
        block_sums[block] = shared[thread]


@cute.jit
def _launch_scan_blocks(
    stream: cuda.CUstream,
    values: cute.Tensor,
    scanned: cute.Tensor,
    block_sums: cute.Tensor,
    *,
    count: cutlass.Constexpr[int],
    block_count: cutlass.Constexpr[int],
):
    _scan_blocks_kernel(values, scanned, block_sums, count).launch(
        grid=[block_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def _scan_block_sums_kernel(
    block_sums: cute.Tensor,
    scanned_block_sums: cute.Tensor,
    block_count: int,
):
    thread, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    shared = smem.allocate_tensor(cutlass.Int32, _BLOCK_SIZE)
    carry = smem.allocate_tensor(cutlass.Int32, 1)
    if thread == 0:
        carry[0] = 0
    cute.arch.sync_threads()
    for chunk_start in cutlass.range(
        cutlass.Int32(0), block_count, _BLOCK_SIZE
    ):
        index = chunk_start + thread
        value = cutlass.Int32(0)
        if index < block_count:
            value = block_sums[index]
        shared[thread] = value
        cute.arch.sync_threads()
        for offset in (1, 2, 4, 8, 16, 32, 64, 128):
            addend = cutlass.Int32(0)
            if thread >= offset:
                addend = shared[thread - offset]
            cute.arch.sync_threads()
            shared[thread] = _saturating_add(shared[thread], addend)
            cute.arch.sync_threads()
        if index < block_count:
            scanned_block_sums[index] = _saturating_add(
                carry[0], shared[thread]
            )
        active = cutlass.min(_BLOCK_SIZE, block_count - chunk_start)
        cute.arch.sync_threads()
        if thread == 0:
            carry[0] = _saturating_add(
                carry[0], shared[active - 1]
            )
        cute.arch.sync_threads()


@cute.kernel
def _finalize_hierarchical_prefix_kernel(
    cumulative: cute.Tensor,
    scanned_block_sums: cute.Tensor,
    valid_count: cute.Tensor,
    overflow: cute.Tensor,
    required_count: cute.Tensor,
    count: int,
    capacity: int,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * _BLOCK_SIZE + thread
    if index < count:
        value = cumulative[index]
        if block > 0:
            value = _saturating_add(value, scanned_block_sums[block - 1])
        cumulative[index] = cutlass.min(value, capacity + 1)
        if index == count - 1:
            required_count[0] = value
            valid_count[0] = cutlass.min(value, capacity)
            overflow[0] = cutlass.Uint8(1 if value > capacity else 0)


@cute.jit
def _launch_hierarchical_prefix(
    stream: cuda.CUstream,
    counts: cute.Tensor,
    cumulative: cute.Tensor,
    block_sums: cute.Tensor,
    scanned_block_sums: cute.Tensor,
    valid_count: cute.Tensor,
    overflow: cute.Tensor,
    required_count: cute.Tensor,
    *,
    count: cutlass.Constexpr[int],
    capacity: int,
    block_count: cutlass.Constexpr[int],
):
    _scan_blocks_kernel(counts, cumulative, block_sums, count).launch(
        grid=[block_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )
    _scan_block_sums_kernel(
        block_sums, scanned_block_sums, block_count
    ).launch(
        grid=[1, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )
    _finalize_hierarchical_prefix_kernel(
        cumulative,
        scanned_block_sums,
        valid_count,
        overflow,
        required_count,
        count,
        capacity,
    ).launch(
        grid=[block_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


def intersection_prefix_cute(
    counts: jax.Array, *, capacity: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute a saturated inclusive prefix and fixed-capacity metadata."""

    capacity = _static_int("capacity", capacity)
    counts = jnp.asarray(counts)
    if counts.ndim != 1 or counts.dtype != jnp.int32:
        raise ValueError("counts must be a rank-one int32 array")
    if counts.shape[0] == 0:
        raise ValueError("counts must be non-empty")
    _cute_device()
    count = counts.shape[0]
    block_count = (count + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    call = cjax.cutlass_call(
        _launch_hierarchical_prefix,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(counts.shape, jnp.int32),
            jax.ShapeDtypeStruct((block_count,), jnp.int32),
            jax.ShapeDtypeStruct((block_count,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.uint8),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ),
        use_static_tensors=True,
        count=count,
        capacity=capacity,
        block_count=block_count,
    )
    capped, _, _, valid_count, overflow, required_count = call(counts)
    return (
        capped,
        valid_count[0],
        overflow[0].astype(jnp.bool_),
        required_count[0],
    )


@cute.kernel
def _prepare_sort_keys_kernel(
    gaussian_ids: cute.Tensor,
    tile_ids: cute.Tensor,
    depths: cute.Tensor,
    valid_count: cute.Tensor,
    keys: cute.Tensor,
    values: cute.Tensor,
    capacity: int,
    gaussian_count: int,
    tile_count: int,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * _BLOCK_SIZE + thread
    if index < capacity:
        count = cutlass.min(
            cutlass.max(valid_count[0], cutlass.Int32(0)), capacity
        )
        gaussian_id = gaussian_ids[index]
        tile_id = tile_ids[index]
        values[index] = gaussian_id
        key = cutlass.Uint64(_UINT64_MAX)
        valid = (
            index < count
            and gaussian_id >= 0
            and gaussian_id < gaussian_count
            and tile_id >= 0
            and tile_id < tile_count
        )
        if valid:
            depth = depths[gaussian_id]
            valid = cute.math.isfinite(depth)
            if valid:
                if depth == 0.0:
                    depth = cutlass.Float32(0.0)
                bits = depth.bitcast(cutlass.Uint32)
                ordered = bits ^ cutlass.Uint32(0x80000000)
                if (bits & cutlass.Uint32(0x80000000)) != 0:
                    ordered = ~bits
                key = (
                    cutlass.Uint64(cutlass.Uint32(tile_id)) << 32
                ) | cutlass.Uint64(ordered)
        keys[index] = key


@cute.jit
def _launch_prepare_sort_keys(
    stream: cuda.CUstream,
    gaussian_ids: cute.Tensor,
    tile_ids: cute.Tensor,
    depths: cute.Tensor,
    valid_count: cute.Tensor,
    keys: cute.Tensor,
    values: cute.Tensor,
    *,
    capacity: cutlass.Constexpr[int],
    gaussian_count: int,
    tile_count: int,
    block_count: cutlass.Constexpr[int],
):
    _prepare_sort_keys_kernel(
        gaussian_ids,
        tile_ids,
        depths,
        valid_count,
        keys,
        values,
        capacity,
        gaussian_count,
        tile_count,
    ).launch(
        grid=[block_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def _prepare_sort_keys_tile_histogram_kernel(
    gaussian_ids: cute.Tensor,
    tile_ids: cute.Tensor,
    depths: cute.Tensor,
    valid_count: cute.Tensor,
    keys: cute.Tensor,
    values: cute.Tensor,
    histogram: cute.Tensor,
    local_ranks: cute.Tensor,
    capacity: int,
    gaussian_count: int,
    tile_count: int,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * _BLOCK_SIZE + thread
    warp = thread >> 5
    lane = thread & 31
    smem = cutlass.utils.SmemAllocator()
    warp_histogram = smem.allocate_tensor(
        cutlass.Int32,
        cute.make_layout(
            (_WARPS_PER_BLOCK, _TILE_RADIX_SIZE),
            stride=(_TILE_RADIX_SIZE, 1),
        ),
    )
    if thread < _TILE_RADIX_SIZE:
        for current_warp in cutlass.range_constexpr(_WARPS_PER_BLOCK):
            warp_histogram[current_warp, thread] = 0
    cute.arch.sync_threads()

    active = index < capacity
    key = cutlass.Uint64(_UINT64_MAX)
    gaussian_id = cutlass.Int32(-1)
    if active:
        count = cutlass.min(
            cutlass.max(valid_count[0], cutlass.Int32(0)), capacity
        )
        gaussian_id = gaussian_ids[index]
        tile_id = tile_ids[index]
        item_valid = index < count
        item_valid = item_valid & (gaussian_id >= 0)
        item_valid = item_valid & (gaussian_id < gaussian_count)
        item_valid = item_valid & (tile_id >= 0)
        item_valid = item_valid & (tile_id < tile_count)
        if item_valid:
            depth = depths[gaussian_id]
            item_valid = cute.math.isfinite(depth)
            if item_valid:
                if depth == 0.0:
                    depth = cutlass.Float32(0.0)
                bits = depth.bitcast(cutlass.Uint32)
                ordered = bits ^ cutlass.Uint32(0x80000000)
                if (bits & cutlass.Uint32(0x80000000)) != 0:
                    ordered = ~bits
                key = (
                    cutlass.Uint64(cutlass.Uint32(tile_id)) << 32
                ) | cutlass.Uint64(ordered)
        keys[index] = key
        values[index] = gaussian_id

    digit = cutlass.Int32(
        (key >> 32) & cutlass.Uint64(_TILE_RADIX_SIZE - 1)
    )
    if active:
        cute.arch.atomic_add(
            warp_histogram.iterator + warp * _TILE_RADIX_SIZE + digit,
            cutlass.Int32(1),
            sem="relaxed",
            scope="cta",
        )
    active_mask = cutlass.Uint32(cute.arch.vote_ballot_sync(active))
    matches = cutlass.Uint32(0)
    if active_mask != 0:
        matches = cute.arch.match_sync(active_mask, digit)
    lower_lanes = (cutlass.Uint32(1) << lane) - cutlass.Uint32(1)
    rank = cutlass.Int32(cute.arch.popc(matches & lower_lanes))
    cute.arch.sync_threads()
    if active:
        for previous_warp in cutlass.range_constexpr(_WARPS_PER_BLOCK):
            if previous_warp < warp:
                rank = rank + warp_histogram[previous_warp, digit]
        local_ranks[index] = rank
    if thread < _TILE_RADIX_SIZE:
        total = cutlass.Int32(0)
        for current_warp in cutlass.range_constexpr(_WARPS_PER_BLOCK):
            total = total + warp_histogram[current_warp, thread]
        histogram[thread, block] = total


@cute.jit
def _launch_prepare_sort_keys_tile_histogram(
    stream: cuda.CUstream,
    gaussian_ids: cute.Tensor,
    tile_ids: cute.Tensor,
    depths: cute.Tensor,
    valid_count: cute.Tensor,
    keys: cute.Tensor,
    values: cute.Tensor,
    histogram: cute.Tensor,
    local_ranks: cute.Tensor,
    *,
    capacity: cutlass.Constexpr[int],
    gaussian_count: int,
    tile_count: int,
    block_count: cutlass.Constexpr[int],
):
    _prepare_sort_keys_tile_histogram_kernel(
        gaussian_ids,
        tile_ids,
        depths,
        valid_count,
        keys,
        values,
        histogram,
        local_ranks,
        capacity,
        gaussian_count,
        tile_count,
    ).launch(
        grid=[block_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def _radix_histogram_kernel(
    keys: cute.Tensor,
    histogram: cute.Tensor,
    local_ranks: cute.Tensor,
    count: int,
    shift: cutlass.Constexpr[int],
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * _BLOCK_SIZE + thread
    warp = thread >> 5
    lane = thread & 31
    smem = cutlass.utils.SmemAllocator()
    warp_histogram = smem.allocate_tensor(
        cutlass.Int32,
        cute.make_layout(
            (_WARPS_PER_BLOCK, _RADIX_SIZE),
            stride=(_RADIX_SIZE, 1),
        ),
    )
    for current_warp in cutlass.range_constexpr(_WARPS_PER_BLOCK):
        warp_histogram[current_warp, thread] = 0
    cute.arch.sync_threads()

    active = index < count
    digit = cutlass.Int32(0)
    if active:
        digit = cutlass.Int32(
            (keys[index] >> shift) & cutlass.Uint64(_RADIX_SIZE - 1)
        )
        cute.arch.atomic_add(
            warp_histogram.iterator + warp * _RADIX_SIZE + digit,
            cutlass.Int32(1),
            sem="relaxed",
            scope="cta",
        )
    active_mask = cutlass.Uint32(cute.arch.vote_ballot_sync(active))
    matches = cutlass.Uint32(0)
    if active_mask != 0:
        matches = cute.arch.match_sync(active_mask, digit)
    lower_lanes = (cutlass.Uint32(1) << lane) - cutlass.Uint32(1)
    rank = cutlass.Int32(cute.arch.popc(matches & lower_lanes))
    cute.arch.sync_threads()

    if active:
        for previous_warp in cutlass.range_constexpr(_WARPS_PER_BLOCK):
            if previous_warp < warp:
                rank = rank + warp_histogram[previous_warp, digit]
        local_ranks[index] = rank
    total = cutlass.Int32(0)
    for current_warp in cutlass.range_constexpr(_WARPS_PER_BLOCK):
        total = total + warp_histogram[current_warp, thread]
    histogram[thread, block] = total


@cute.jit
def _launch_radix_histogram(
    stream: cuda.CUstream,
    keys: cute.Tensor,
    histogram: cute.Tensor,
    local_ranks: cute.Tensor,
    *,
    count: cutlass.Constexpr[int],
    block_count: cutlass.Constexpr[int],
    shift: cutlass.Constexpr[int],
):
    _radix_histogram_kernel(
        keys, histogram, local_ranks, count, shift
    ).launch(
        grid=[block_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def _scan_radix_histogram_blocks_kernel(
    histogram: cute.Tensor,
    block_prefix: cute.Tensor,
    bucket_totals: cute.Tensor,
    block_count: int,
):
    thread, _, _ = cute.arch.thread_idx()
    bucket, _, _ = cute.arch.block_idx()
    warp = thread >> 5
    lane = thread & 31
    smem = cutlass.utils.SmemAllocator()
    warp_totals = smem.allocate_tensor(cutlass.Int32, _WARPS_PER_BLOCK)
    carry = smem.allocate_tensor(cutlass.Int32, 1)
    if thread == 0:
        carry[0] = 0
    cute.arch.sync_threads()

    for chunk_start in cutlass.range(
        cutlass.Int32(0), block_count, _BLOCK_SIZE
    ):
        block = chunk_start + thread
        value = cutlass.Int32(0)
        if block < block_count:
            value = histogram[bucket, block]
        inclusive = value
        for offset in (1, 2, 4, 8, 16):
            addend = cute.arch.shuffle_sync_up(inclusive, offset)
            if lane >= offset:
                inclusive = inclusive + addend
        if lane == 31:
            warp_totals[warp] = inclusive
        cute.arch.sync_threads()

        if warp == 0:
            warp_value = cutlass.Int32(0)
            if lane < _WARPS_PER_BLOCK:
                warp_value = warp_totals[lane]
            for offset in (1, 2, 4):
                addend = cute.arch.shuffle_sync_up(warp_value, offset)
                if lane >= offset:
                    warp_value = warp_value + addend
            if lane < _WARPS_PER_BLOCK:
                warp_totals[lane] = warp_value
        cute.arch.sync_threads()

        preceding_warps = cutlass.Int32(0)
        if warp > 0:
            preceding_warps = warp_totals[warp - 1]
        if block < block_count:
            block_prefix[bucket, block] = (
                carry[0] + preceding_warps + inclusive - value
            )
        cute.arch.sync_threads()
        if thread == 0:
            carry[0] = carry[0] + warp_totals[_WARPS_PER_BLOCK - 1]
        cute.arch.sync_threads()

    if thread == 0:
        bucket_totals[bucket] = carry[0]


@cute.kernel
def _scan_radix_bucket_totals_kernel(
    bucket_totals: cute.Tensor,
    bucket_base: cute.Tensor,
):
    bucket, _, _ = cute.arch.thread_idx()
    shared = cutlass.utils.SmemAllocator().allocate_tensor(
        cutlass.Int32, _RADIX_SIZE
    )
    shared[bucket] = bucket_totals[bucket]
    cute.arch.sync_threads()
    for offset in (1, 2, 4, 8, 16, 32, 64, 128):
        addend = cutlass.Int32(0)
        if bucket >= offset:
            addend = shared[bucket - offset]
        cute.arch.sync_threads()
        shared[bucket] = shared[bucket] + addend
        cute.arch.sync_threads()
    bucket_base[bucket] = 0 if bucket == 0 else shared[bucket - 1]


@cute.jit
def _launch_scan_radix_histogram(
    stream: cuda.CUstream,
    histogram: cute.Tensor,
    block_prefix: cute.Tensor,
    bucket_totals: cute.Tensor,
    bucket_base: cute.Tensor,
    *,
    block_count: int,
):
    _scan_radix_histogram_blocks_kernel(
        histogram, block_prefix, bucket_totals, block_count
    ).launch(
        grid=[_RADIX_SIZE, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )
    _scan_radix_bucket_totals_kernel(
        bucket_totals, bucket_base
    ).launch(
        grid=[1, 1, 1],
        block=[_RADIX_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def _radix_scatter_kernel(
    keys: cute.Tensor,
    values: cute.Tensor,
    block_prefix: cute.Tensor,
    bucket_base: cute.Tensor,
    local_ranks: cute.Tensor,
    output_keys: cute.Tensor,
    output_values: cute.Tensor,
    count: int,
    shift: cutlass.Constexpr[int],
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * _BLOCK_SIZE + thread
    if index < count:
        key = keys[index]
        digit = cutlass.Int32(
            (key >> shift) & cutlass.Uint64(_RADIX_SIZE - 1)
        )
        target = (
            bucket_base[digit]
            + block_prefix[digit, block]
            + local_ranks[index]
        )
        output_keys[target] = key
        output_values[target] = values[index]


@cute.jit
def _launch_radix_scatter(
    stream: cuda.CUstream,
    keys: cute.Tensor,
    values: cute.Tensor,
    block_prefix: cute.Tensor,
    bucket_base: cute.Tensor,
    local_ranks: cute.Tensor,
    output_keys: cute.Tensor,
    output_values: cute.Tensor,
    *,
    count: cutlass.Constexpr[int],
    block_count: cutlass.Constexpr[int],
    shift: cutlass.Constexpr[int],
):
    _radix_scatter_kernel(
        keys,
        values,
        block_prefix,
        bucket_base,
        local_ranks,
        output_keys,
        output_values,
        count,
        shift,
    ).launch(
        grid=[block_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def _tile_radix_histogram_kernel(
    keys: cute.Tensor,
    histogram: cute.Tensor,
    local_ranks: cute.Tensor,
    count: int,
    shift: cutlass.Constexpr[int],
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * _BLOCK_SIZE + thread
    warp = thread >> 5
    lane = thread & 31
    smem = cutlass.utils.SmemAllocator()
    warp_histogram = smem.allocate_tensor(
        cutlass.Int32,
        cute.make_layout(
            (_WARPS_PER_BLOCK, _TILE_RADIX_SIZE),
            stride=(_TILE_RADIX_SIZE, 1),
        ),
    )
    if thread < _TILE_RADIX_SIZE:
        for current_warp in cutlass.range_constexpr(_WARPS_PER_BLOCK):
            warp_histogram[current_warp, thread] = 0
    cute.arch.sync_threads()

    active = index < count
    digit = cutlass.Int32(0)
    if active:
        digit = cutlass.Int32(
            (keys[index] >> shift) & cutlass.Uint64(_TILE_RADIX_SIZE - 1)
        )
        cute.arch.atomic_add(
            warp_histogram.iterator + warp * _TILE_RADIX_SIZE + digit,
            cutlass.Int32(1),
            sem="relaxed",
            scope="cta",
        )
    active_mask = cutlass.Uint32(cute.arch.vote_ballot_sync(active))
    matches = cutlass.Uint32(0)
    if active_mask != 0:
        matches = cute.arch.match_sync(active_mask, digit)
    lower_lanes = (cutlass.Uint32(1) << lane) - cutlass.Uint32(1)
    rank = cutlass.Int32(cute.arch.popc(matches & lower_lanes))
    cute.arch.sync_threads()

    if active:
        for previous_warp in cutlass.range_constexpr(_WARPS_PER_BLOCK):
            if previous_warp < warp:
                rank = rank + warp_histogram[previous_warp, digit]
        local_ranks[index] = rank
    if thread < _TILE_RADIX_SIZE:
        total = cutlass.Int32(0)
        for current_warp in cutlass.range_constexpr(_WARPS_PER_BLOCK):
            total = total + warp_histogram[current_warp, thread]
        histogram[thread, block] = total


@cute.jit
def _launch_tile_radix_histogram(
    stream: cuda.CUstream,
    keys: cute.Tensor,
    histogram: cute.Tensor,
    local_ranks: cute.Tensor,
    *,
    count: cutlass.Constexpr[int],
    block_count: cutlass.Constexpr[int],
    shift: cutlass.Constexpr[int],
):
    _tile_radix_histogram_kernel(
        keys, histogram, local_ranks, count, shift
    ).launch(
        grid=[block_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.jit
def _launch_scan_tile_radix_histogram(
    stream: cuda.CUstream,
    histogram: cute.Tensor,
    block_prefix: cute.Tensor,
    bucket_totals: cute.Tensor,
    *,
    block_count: int,
):
    _scan_radix_histogram_blocks_kernel(
        histogram, block_prefix, bucket_totals, block_count
    ).launch(
        grid=[_TILE_RADIX_SIZE, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def _tile_radix_scatter_kernel(
    keys: cute.Tensor,
    values: cute.Tensor,
    block_prefix: cute.Tensor,
    bucket_totals: cute.Tensor,
    local_ranks: cute.Tensor,
    output_keys: cute.Tensor,
    output_values: cute.Tensor,
    count: int,
    shift: cutlass.Constexpr[int],
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    bucket_base = cutlass.utils.SmemAllocator().allocate_tensor(
        cutlass.Int32, _TILE_RADIX_SIZE
    )
    if thread < _TILE_RADIX_SIZE:
        value = bucket_totals[thread]
        inclusive = value
        for offset in (1, 2, 4, 8, 16):
            addend = cute.arch.shuffle_sync_up(inclusive, offset)
            if thread >= offset:
                inclusive = inclusive + addend
        bucket_base[thread] = inclusive - value
    cute.arch.sync_threads()

    index = block * _BLOCK_SIZE + thread
    if index < count:
        key = keys[index]
        digit = cutlass.Int32(
            (key >> shift) & cutlass.Uint64(_TILE_RADIX_SIZE - 1)
        )
        target = (
            bucket_base[digit]
            + block_prefix[digit, block]
            + local_ranks[index]
        )
        output_keys[target] = key
        output_values[target] = values[index]


@cute.jit
def _launch_tile_radix_scatter(
    stream: cuda.CUstream,
    keys: cute.Tensor,
    values: cute.Tensor,
    block_prefix: cute.Tensor,
    bucket_totals: cute.Tensor,
    local_ranks: cute.Tensor,
    output_keys: cute.Tensor,
    output_values: cute.Tensor,
    *,
    count: cutlass.Constexpr[int],
    block_count: cutlass.Constexpr[int],
    shift: cutlass.Constexpr[int],
):
    _tile_radix_scatter_kernel(
        keys,
        values,
        block_prefix,
        bucket_totals,
        local_ranks,
        output_keys,
        output_values,
        count,
        shift,
    ).launch(
        grid=[block_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.jit
def _sort_depth_segment(
    keys: cute.Tensor,
    values: cute.Tensor,
    output_gaussians: cute.Tensor,
    output_tiles: cute.Tensor,
    start,
    segment_count,
    tile,
    segment_capacity: cutlass.Constexpr[int],
):
    thread, _, _ = cute.arch.thread_idx()
    warp = thread >> 5
    lane = thread & 31
    smem = cutlass.utils.SmemAllocator()
    keys_a = smem.allocate_tensor(cutlass.Uint32, segment_capacity)
    keys_b = smem.allocate_tensor(cutlass.Uint32, segment_capacity)
    values_a = smem.allocate_tensor(cutlass.Int32, segment_capacity)
    values_b = smem.allocate_tensor(cutlass.Int32, segment_capacity)
    histogram = smem.allocate_tensor(cutlass.Int32, _RADIX_SIZE)
    bucket_base = smem.allocate_tensor(cutlass.Int32, _RADIX_SIZE)
    warp_histogram = smem.allocate_tensor(
        cutlass.Uint8,
        cute.make_layout(
            (_WARPS_PER_BLOCK, _RADIX_SIZE),
            stride=(_RADIX_SIZE, 1),
        ),
    )
    warp_totals = smem.allocate_tensor(cutlass.Int32, _WARPS_PER_BLOCK)

    # Tile bits are constant inside a segment; only the 32-bit depth key sorts.
    for local_index in cutlass.range(
        cutlass.Int32(thread), segment_count, _BLOCK_SIZE
    ):
        keys_a[local_index] = cutlass.Uint32(keys[start + local_index])
        values_a[local_index] = values[start + local_index]
    cute.arch.sync_threads()

    for pass_index in cutlass.range_constexpr(4):
        shift = pass_index * _RADIX_BITS
        source_keys = keys_a if pass_index % 2 == 0 else keys_b
        source_values = values_a if pass_index % 2 == 0 else values_b
        target_keys = keys_b if pass_index % 2 == 0 else keys_a
        target_values = values_b if pass_index % 2 == 0 else values_a

        histogram[thread] = 0
        cute.arch.sync_threads()
        for local_index in cutlass.range(
            cutlass.Int32(thread), segment_count, _BLOCK_SIZE
        ):
            digit = cutlass.Int32(
                (source_keys[local_index] >> shift)
                & cutlass.Uint32(_RADIX_SIZE - 1)
            )
            cute.arch.atomic_add(
                histogram.iterator + digit,
                cutlass.Int32(1),
                sem="relaxed",
                scope="cta",
            )
        cute.arch.sync_threads()

        value = histogram[thread]
        inclusive = value
        for offset in (1, 2, 4, 8, 16):
            addend = cute.arch.shuffle_sync_up(inclusive, offset)
            if lane >= offset:
                inclusive = inclusive + addend
        if lane == 31:
            warp_totals[warp] = inclusive
        cute.arch.sync_threads()
        if warp == 0:
            warp_value = cutlass.Int32(0)
            if lane < _WARPS_PER_BLOCK:
                warp_value = warp_totals[lane]
            for offset in (1, 2, 4):
                addend = cute.arch.shuffle_sync_up(warp_value, offset)
                if lane >= offset:
                    warp_value = warp_value + addend
            if lane < _WARPS_PER_BLOCK:
                warp_totals[lane] = warp_value
        cute.arch.sync_threads()
        preceding_warps = cutlass.Int32(0)
        if warp > 0:
            preceding_warps = warp_totals[warp - 1]
        bucket_base[thread] = preceding_warps + inclusive - value
        cute.arch.sync_threads()

        for chunk_start in cutlass.range(
            cutlass.Int32(0), segment_count, _BLOCK_SIZE
        ):
            for current_warp in cutlass.range_constexpr(_WARPS_PER_BLOCK):
                warp_histogram[current_warp, thread] = cutlass.Uint8(0)
            cute.arch.sync_threads()

            local_index = chunk_start + thread
            active = local_index < segment_count
            digit = cutlass.Int32(0)
            key = cutlass.Uint32((1 << 32) - 1)
            item_value = cutlass.Int32(-1)
            if active:
                key = source_keys[local_index]
                item_value = source_values[local_index]
                digit = cutlass.Int32(
                    (key >> shift) & cutlass.Uint32(_RADIX_SIZE - 1)
                )
            active_mask = cutlass.Uint32(cute.arch.vote_ballot_sync(active))
            matches = cutlass.Uint32(0)
            if active_mask != 0:
                matches = cute.arch.match_sync(active_mask, digit)
            lower_lanes = (cutlass.Uint32(1) << lane) - cutlass.Uint32(1)
            rank = cutlass.Int32(cute.arch.popc(matches & lower_lanes))
            if active:
                if rank == 0:
                    warp_histogram[warp, digit] = cutlass.Uint8(
                        cute.arch.popc(matches)
                    )
            cute.arch.sync_threads()
            if active:
                for previous_warp in cutlass.range_constexpr(_WARPS_PER_BLOCK):
                    if previous_warp < warp:
                        rank = rank + cutlass.Int32(
                            warp_histogram[previous_warp, digit]
                        )
                target = bucket_base[digit] + rank
                target_keys[target] = key
                target_values[target] = item_value
            cute.arch.sync_threads()
            chunk_count = cutlass.Int32(0)
            for current_warp in cutlass.range_constexpr(_WARPS_PER_BLOCK):
                chunk_count = chunk_count + cutlass.Int32(
                    warp_histogram[current_warp, thread]
                )
            bucket_base[thread] = bucket_base[thread] + chunk_count
            cute.arch.sync_threads()

    for local_index in cutlass.range(
        cutlass.Int32(thread), segment_count, _BLOCK_SIZE
    ):
        output_gaussians[start + local_index] = values_a[local_index]
        output_tiles[start + local_index] = tile


@cute.kernel
def _segmented_depth_radix_kernel(
    keys: cute.Tensor,
    values: cute.Tensor,
    offsets: cute.Tensor,
    effective_valid_count: cute.Tensor,
    output_gaussians: cute.Tensor,
    output_tiles: cute.Tensor,
    capacity: int,
    tile_count: int,
    segment_capacity: cutlass.Constexpr[int],
):
    tile, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    count = cutlass.min(
        cutlass.max(effective_valid_count[0], cutlass.Int32(0)), capacity
    )
    start = cutlass.min(cutlass.max(offsets[tile], 0), count)
    end = count
    if tile + 1 < tile_count:
        end = cutlass.min(cutlass.max(offsets[tile + 1], start), count)
    _sort_depth_segment(
        keys,
        values,
        output_gaussians,
        output_tiles,
        start,
        end - start,
        tile,
        segment_capacity,
    )

    tail_index = count + tile * _BLOCK_SIZE + thread
    tail_stride = tile_count * _BLOCK_SIZE
    while tail_index < capacity:
        output_gaussians[tail_index] = -1
        output_tiles[tail_index] = -1
        tail_index = tail_index + tail_stride


@cute.jit
def _launch_segmented_depth_radix(
    stream: cuda.CUstream,
    keys: cute.Tensor,
    values: cute.Tensor,
    offsets: cute.Tensor,
    effective_valid_count: cute.Tensor,
    output_gaussians: cute.Tensor,
    output_tiles: cute.Tensor,
    *,
    capacity: cutlass.Constexpr[int],
    tile_count: cutlass.Constexpr[int],
    segment_capacity: cutlass.Constexpr[int],
):
    _segmented_depth_radix_kernel(
        keys,
        values,
        offsets,
        effective_valid_count,
        output_gaussians,
        output_tiles,
        capacity,
        tile_count,
        segment_capacity,
    ).launch(
        grid=[tile_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.jit
def _bounded_effective_count(keys, declared_valid_count, capacity):
    low = cutlass.Int32(0)
    high = cutlass.min(
        cutlass.max(declared_valid_count[0], cutlass.Int32(0)), capacity
    )
    while low < high:
        middle = low + (high - low) // 2
        if keys[middle] < cutlass.Uint64(_UINT64_MAX):
            low = middle + 1
        else:
            high = middle
    return low


@cute.jit
def _tile_lower_bound(keys, count, tile):
    low = cutlass.Int32(0)
    high = count
    while low < high:
        middle = low + (high - low) // 2
        if cutlass.Int32(keys[middle] >> 32) < tile:
            low = middle + 1
        else:
            high = middle
    return low


@cute.kernel
def _tail_mega_kernel(
    keys: cute.Tensor,
    values: cute.Tensor,
    declared_valid_count: cute.Tensor,
    means: cute.Tensor,
    conics: cute.Tensor,
    colors: cute.Tensor,
    opacities: cute.Tensor,
    output_gaussians: cute.Tensor,
    output_tiles: cute.Tensor,
    offsets: cute.Tensor,
    effective_valid_count: cute.Tensor,
    max_segment_count: cute.Tensor,
    foreground: cute.Tensor,
    alpha: cute.Tensor,
    accepted_final_transmittance: cute.Tensor,
    last_ids: cute.Tensor,
    tile_overflow: cute.Tensor,
    capacity: int,
    gaussian_count: int,
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_count: int,
    per_tile_bound: int,
    channels: cutlass.Constexpr[int],
    alpha_threshold: float,
    transmittance_threshold: float,
    segment_capacity: cutlass.Constexpr[int],
):
    tile, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    if tile < tile_count:
        bounds = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Int32, 4
        )
        if thread == 0:
            count = _bounded_effective_count(
                keys, declared_valid_count, capacity
            )
            start = _tile_lower_bound(keys, count, tile)
            end = _tile_lower_bound(keys, count, tile + 1)
            bounds[0] = count
            bounds[1] = start
            bounds[2] = end
            offsets[tile] = start
            if tile == 0:
                effective_valid_count[0] = count
                bounds[3] = 0
        cute.arch.sync_threads()
        count = bounds[0]
        start = bounds[1]
        end = bounds[2]
        segment_count = end - start
        # CTA 0 publishes the fallback predicate without a JAX reduction launch.
        if tile == 0:
            for check_tile in cutlass.range(
                cutlass.Int32(thread), tile_count, _BLOCK_SIZE
            ):
                check_start = _tile_lower_bound(keys, count, check_tile)
                check_end = _tile_lower_bound(keys, count, check_tile + 1)
                cute.arch.atomic_max(
                    bounds.iterator + 3,
                    check_end - check_start,
                    sem="relaxed",
                    scope="cta",
                )
            cute.arch.sync_threads()
            if thread == 0:
                max_segment_count[0] = bounds[3]
        cute.arch.sync_threads()

        if segment_count <= segment_capacity:
            _sort_depth_segment(
                keys,
                values,
                output_gaussians,
                output_tiles,
                start,
                segment_count,
                tile,
                segment_capacity,
            )
            cute.arch.sync_threads()

            tail_index = count + tile * _BLOCK_SIZE + thread
            tail_stride = tile_count * _BLOCK_SIZE
            while tail_index < capacity:
                output_gaussians[tail_index] = -1
                output_tiles[tail_index] = -1
                tail_index = tail_index + tail_stride
            cute.arch.sync_threads()

            _composite_tile_forward(
                means,
                conics,
                colors,
                opacities,
                output_gaussians,
                foreground,
                alpha,
                accepted_final_transmittance,
                last_ids,
                tile_overflow,
                tile,
                thread,
                start,
                end,
                gaussian_count,
                image_width,
                image_height,
                tile_width,
                per_tile_bound,
                channels,
                _MEGA_COMPOSITOR_BATCH_CAPACITY,
                alpha_threshold,
                transmittance_threshold,
            )


@cute.jit
def _launch_tail_mega(
    stream: cuda.CUstream,
    keys: cute.Tensor,
    values: cute.Tensor,
    declared_valid_count: cute.Tensor,
    means: cute.Tensor,
    conics: cute.Tensor,
    colors: cute.Tensor,
    opacities: cute.Tensor,
    output_gaussians: cute.Tensor,
    output_tiles: cute.Tensor,
    offsets: cute.Tensor,
    effective_valid_count: cute.Tensor,
    max_segment_count: cute.Tensor,
    foreground: cute.Tensor,
    alpha: cute.Tensor,
    accepted_final_transmittance: cute.Tensor,
    last_ids: cute.Tensor,
    tile_overflow: cute.Tensor,
    *,
    capacity: cutlass.Constexpr[int],
    gaussian_count: cutlass.Constexpr[int],
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_count: cutlass.Constexpr[int],
    per_tile_bound: int,
    channels: cutlass.Constexpr[int],
    alpha_threshold: float,
    transmittance_threshold: float,
    segment_capacity: cutlass.Constexpr[int],
):
    _tail_mega_kernel(
        keys,
        values,
        declared_valid_count,
        means,
        conics,
        colors,
        opacities,
        output_gaussians,
        output_tiles,
        offsets,
        effective_valid_count,
        max_segment_count,
        foreground,
        alpha,
        accepted_final_transmittance,
        last_ids,
        tile_overflow,
        capacity,
        gaussian_count,
        image_width,
        image_height,
        tile_width,
        tile_count,
        per_tile_bound,
        channels,
        alpha_threshold,
        transmittance_threshold,
        segment_capacity,
    ).launch(
        grid=[tile_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def _find_effective_count_kernel(
    keys: cute.Tensor,
    declared_valid_count: cute.Tensor,
    effective_valid_count: cute.Tensor,
    max_segment_count: cute.Tensor,
    capacity: int,
):
    if cute.arch.block_idx()[0] == 0 and cute.arch.thread_idx()[0] == 0:
        low = cutlass.Int32(0)
        high = cutlass.min(
            cutlass.max(declared_valid_count[0], cutlass.Int32(0)), capacity
        )
        while low < high:
            middle = low + (high - low) // 2
            if keys[middle] < cutlass.Uint64(_UINT64_MAX):
                low = middle + 1
            else:
                high = middle
        effective_valid_count[0] = low
        max_segment_count[0] = 0


@cute.kernel
def _finalize_sorted_outputs_kernel(
    keys: cute.Tensor,
    values: cute.Tensor,
    effective_valid_count: cute.Tensor,
    gaussian_ids: cute.Tensor,
    tile_ids: cute.Tensor,
    offsets: cute.Tensor,
    max_segment_count: cute.Tensor,
    capacity: int,
    tile_count: int,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * _BLOCK_SIZE + thread
    count = cutlass.min(
        cutlass.max(effective_valid_count[0], cutlass.Int32(0)), capacity
    )
    if index < capacity:
        key = keys[index]
        if index < count and key != cutlass.Uint64(_UINT64_MAX):
            gaussian_ids[index] = values[index]
            tile_ids[index] = cutlass.Int32(key >> 32)
        else:
            gaussian_ids[index] = -1
            tile_ids[index] = -1
    if count <= 0:
        if index < tile_count:
            offsets[index] = 0
    elif index < count:
        tile_current = cutlass.min(cutlass.Int32(keys[index] >> 32), tile_count)
        if index == 0:
            for tile in cutlass.range(cutlass.Int32(0), tile_current + 1):
                if tile < tile_count:
                    offsets[tile] = 0
        if index == count - 1:
            for tile in cutlass.range(tile_current + 1, tile_count):
                offsets[tile] = count
        if index > 0:
            tile_previous = cutlass.min(
                cutlass.Int32(keys[index - 1] >> 32), tile_count
            )
            if tile_previous != tile_current:
                for tile in cutlass.range(tile_previous + 1, tile_current + 1):
                    if tile < tile_count:
                        offsets[tile] = index
        is_segment_end = index == count - 1
        if not is_segment_end:
            is_segment_end = cutlass.Int32(keys[index + 1] >> 32) != tile_current
        if is_segment_end:
            low = cutlass.Int32(0)
            high = index + 1
            while low < high:
                middle = low + (high - low) // 2
                if cutlass.Int32(keys[middle] >> 32) < tile_current:
                    low = middle + 1
                else:
                    high = middle
            cute.arch.atomic_max(
                max_segment_count.iterator,
                index - low + 1,
                sem="relaxed",
                scope="gpu",
            )


@cute.jit
def _launch_finalize_sorted_outputs(
    stream: cuda.CUstream,
    keys: cute.Tensor,
    values: cute.Tensor,
    declared_valid_count: cute.Tensor,
    gaussian_ids: cute.Tensor,
    tile_ids: cute.Tensor,
    offsets: cute.Tensor,
    effective_valid_count: cute.Tensor,
    max_segment_count: cute.Tensor,
    *,
    capacity: cutlass.Constexpr[int],
    tile_count: int,
    block_count: cutlass.Constexpr[int],
):
    _find_effective_count_kernel(
        keys,
        declared_valid_count,
        effective_valid_count,
        max_segment_count,
        capacity,
    ).launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)
    final_blocks = cutlass.max(block_count, (tile_count + 255) // 256)
    _finalize_sorted_outputs_kernel(
        keys,
        values,
        effective_valid_count,
        gaussian_ids,
        tile_ids,
        offsets,
        max_segment_count,
        capacity,
        tile_count,
    ).launch(
        grid=[final_blocks, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


def _run_tail_mega_forward(
    grouped_keys: jax.Array,
    grouped_values: jax.Array,
    declared_valid_count: jax.Array,
    means2d: jax.Array,
    conics: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    *,
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    capacity = grouped_keys.shape[0]
    gaussian_count = means2d.shape[0]
    tile_count = tile_width * tile_height
    channels = colors.shape[-1]
    call = cjax.cutlass_call(
        _launch_tail_mega,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct((tile_count,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct(
                (image_height, image_width, channels), jnp.float32
            ),
            jax.ShapeDtypeStruct((image_height, image_width), jnp.float32),
            jax.ShapeDtypeStruct((image_height, image_width), jnp.float32),
            jax.ShapeDtypeStruct((image_height, image_width), jnp.int32),
            jax.ShapeDtypeStruct((tile_count,), jnp.uint8),
        ),
        use_static_tensors=True,
        capacity=capacity,
        gaussian_count=gaussian_count,
        image_width=image_width,
        image_height=image_height,
        tile_width=tile_width,
        tile_count=tile_count,
        per_tile_bound=per_tile_bound,
        channels=channels,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
        segment_capacity=per_tile_bound,
    )
    (
        gaussian_ids,
        tile_ids,
        offsets,
        effective_count,
        max_segment_count,
        foreground,
        alpha,
        accepted,
        last_ids,
        tile_overflow,
    ) = call(
        grouped_keys,
        grouped_values,
        declared_valid_count.reshape((1,)),
        means2d,
        conics,
        colors,
        opacities,
    )
    return (
        gaussian_ids,
        tile_ids,
        offsets,
        effective_count[0],
        max_segment_count[0],
        foreground,
        alpha,
        accepted,
        last_ids,
        tile_overflow.astype(jnp.bool_),
    )


def _prepare_sort_keys(
    gaussian_ids: jax.Array,
    tile_ids: jax.Array,
    depths: jax.Array,
    valid_count: jax.Array,
    *,
    tile_count: int,
) -> tuple[jax.Array, jax.Array]:
    capacity = gaussian_ids.shape[0]
    block_count = (capacity + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    call = cjax.cutlass_call(
        _launch_prepare_sort_keys,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((capacity,), jnp.uint64),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
        ),
        use_static_tensors=True,
        capacity=capacity,
        gaussian_count=depths.shape[0],
        tile_count=tile_count,
        block_count=block_count,
    )
    return call(
        gaussian_ids, tile_ids, depths, valid_count.reshape((1,))
    )


def _prepare_sort_keys_tile_histogram(
    gaussian_ids: jax.Array,
    tile_ids: jax.Array,
    depths: jax.Array,
    valid_count: jax.Array,
    *,
    tile_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    capacity = gaussian_ids.shape[0]
    block_count = (capacity + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    call = cjax.cutlass_call(
        _launch_prepare_sort_keys_tile_histogram,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((capacity,), jnp.uint64),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct(
                (_TILE_RADIX_SIZE, block_count), jnp.int32
            ),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
        ),
        use_static_tensors=True,
        capacity=capacity,
        gaussian_count=depths.shape[0],
        tile_count=tile_count,
        block_count=block_count,
    )
    return call(
        gaussian_ids, tile_ids, depths, valid_count.reshape((1,))
    )


def _tile_radix_scatter_from_histogram(
    keys: jax.Array,
    values: jax.Array,
    histogram: jax.Array,
    local_ranks: jax.Array,
    *,
    shift: int,
) -> tuple[jax.Array, jax.Array]:
    block_count = histogram.shape[1]
    scan_call = cjax.cutlass_call(
        _launch_scan_tile_radix_histogram,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(histogram.shape, jnp.int32),
            jax.ShapeDtypeStruct((_TILE_RADIX_SIZE,), jnp.int32),
        ),
        use_static_tensors=True,
        block_count=block_count,
    )
    block_prefix, bucket_totals = scan_call(histogram)
    scatter_call = cjax.cutlass_call(
        _launch_tile_radix_scatter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(keys.shape, jnp.uint64),
            jax.ShapeDtypeStruct(values.shape, jnp.int32),
        ),
        use_static_tensors=True,
        count=keys.shape[0],
        block_count=block_count,
        shift=shift,
    )
    return scatter_call(
        keys, values, block_prefix, bucket_totals, local_ranks
    )


def _radix_pass(
    keys: jax.Array, values: jax.Array, *, shift: int
) -> tuple[jax.Array, jax.Array]:
    count = keys.shape[0]
    block_count = (count + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    histogram_call = cjax.cutlass_call(
        _launch_radix_histogram,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((_RADIX_SIZE, block_count), jnp.int32),
            jax.ShapeDtypeStruct((count,), jnp.int32),
        ),
        use_static_tensors=True,
        count=count,
        block_count=block_count,
        shift=shift,
    )
    histogram, local_ranks = histogram_call(keys)
    scan_call = cjax.cutlass_call(
        _launch_scan_radix_histogram,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(histogram.shape, jnp.int32),
            jax.ShapeDtypeStruct((_RADIX_SIZE,), jnp.int32),
            jax.ShapeDtypeStruct((_RADIX_SIZE,), jnp.int32),
        ),
        use_static_tensors=True,
        block_count=block_count,
    )
    block_prefix, _, bucket_base = scan_call(histogram)
    scatter_call = cjax.cutlass_call(
        _launch_radix_scatter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(keys.shape, jnp.uint64),
            jax.ShapeDtypeStruct(values.shape, jnp.int32),
        ),
        use_static_tensors=True,
        count=count,
        block_count=block_count,
        shift=shift,
    )
    return scatter_call(
        keys, values, block_prefix, bucket_base, local_ranks
    )


def _tile_radix_pass(
    keys: jax.Array, values: jax.Array, *, shift: int
) -> tuple[jax.Array, jax.Array]:
    count = keys.shape[0]
    block_count = (count + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    histogram_call = cjax.cutlass_call(
        _launch_tile_radix_histogram,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(
                (_TILE_RADIX_SIZE, block_count), jnp.int32
            ),
            jax.ShapeDtypeStruct((count,), jnp.int32),
        ),
        use_static_tensors=True,
        count=count,
        block_count=block_count,
        shift=shift,
    )
    histogram, local_ranks = histogram_call(keys)
    scan_call = cjax.cutlass_call(
        _launch_scan_tile_radix_histogram,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(histogram.shape, jnp.int32),
            jax.ShapeDtypeStruct((_TILE_RADIX_SIZE,), jnp.int32),
        ),
        use_static_tensors=True,
        block_count=block_count,
    )
    block_prefix, bucket_totals = scan_call(histogram)
    scatter_call = cjax.cutlass_call(
        _launch_tile_radix_scatter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(keys.shape, jnp.uint64),
            jax.ShapeDtypeStruct(values.shape, jnp.int32),
        ),
        use_static_tensors=True,
        count=count,
        block_count=block_count,
        shift=shift,
    )
    return scatter_call(
        keys, values, block_prefix, bucket_totals, local_ranks
    )


def _radix_sort(
    keys: jax.Array, values: jax.Array, *, end_bit: int
) -> tuple[jax.Array, jax.Array]:
    for shift in range(0, end_bit, _RADIX_BITS):
        keys, values = _radix_pass(keys, values, shift=shift)
    return keys, values


def _tile_radix_sort(
    keys: jax.Array, values: jax.Array, *, tile_count: int
) -> tuple[jax.Array, jax.Array]:
    tile_bits = tile_count.bit_length()
    if tile_bits <= _TILE_RADIX_BITS:
        return _tile_radix_pass(keys, values, shift=32)
    if tile_bits <= _RADIX_BITS:
        return _radix_pass(keys, values, shift=32)
    if tile_bits <= 2 * _TILE_RADIX_BITS:
        for shift in range(32, 32 + tile_bits, _TILE_RADIX_BITS):
            keys, values = _tile_radix_pass(keys, values, shift=shift)
        return keys, values
    for shift in range(32, 32 + tile_bits, _RADIX_BITS):
        keys, values = _radix_pass(keys, values, shift=shift)
    return keys, values


def _prepare_grouped_sort_keys(
    gaussian_ids: jax.Array,
    tile_ids: jax.Array,
    depths: jax.Array,
    valid_count: jax.Array,
    *,
    tile_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    fuse_first_tile_histogram = (
        _RADIX_BITS < tile_count.bit_length() <= 2 * _TILE_RADIX_BITS
    )
    if fuse_first_tile_histogram:
        keys, values, first_histogram, first_local_ranks = (
            _prepare_sort_keys_tile_histogram(
                gaussian_ids,
                tile_ids,
                depths,
                valid_count,
                tile_count=tile_count,
            )
        )
        grouped_keys, grouped_values = _tile_radix_scatter_from_histogram(
            keys,
            values,
            first_histogram,
            first_local_ranks,
            shift=32,
        )
        grouped_keys, grouped_values = _tile_radix_pass(
            grouped_keys, grouped_values, shift=32 + _TILE_RADIX_BITS
        )
    else:
        keys, values = _prepare_sort_keys(
            gaussian_ids,
            tile_ids,
            depths,
            valid_count,
            tile_count=tile_count,
        )
        grouped_keys, grouped_values = _tile_radix_sort(
            keys, values, tile_count=tile_count
        )
    return keys, values, grouped_keys, grouped_values


def intersection_sort_offsets_cute(
    gaussian_ids: jax.Array,
    tile_ids: jax.Array,
    depths: jax.Array,
    valid_count: jax.Array,
    *,
    tile_count: int,
    segment_capacity: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Stable radix-sort pairs and construct tile-list offsets."""

    tile_count = _static_int("tile_count", tile_count, minimum=1)
    if segment_capacity is not None:
        segment_capacity = _static_int(
            "segment_capacity", segment_capacity, minimum=1
        )
        if segment_capacity > 2048:
            segment_capacity = None
    gaussian_ids = jnp.asarray(gaussian_ids)
    tile_ids = jnp.asarray(tile_ids)
    depths = jnp.asarray(depths)
    valid_count = jnp.asarray(valid_count)
    if gaussian_ids.ndim != 1 or gaussian_ids.dtype != jnp.int32:
        raise ValueError("gaussian_ids must be a rank-one int32 array")
    if tile_ids.shape != gaussian_ids.shape or tile_ids.dtype != jnp.int32:
        raise ValueError("tile_ids must match gaussian_ids and use int32")
    if gaussian_ids.shape[0] == 0:
        raise ValueError("the fixed intersection capacity must be non-zero")
    if depths.ndim != 1 or depths.dtype != jnp.float32 or depths.shape[0] == 0:
        raise ValueError("depths must be a non-empty rank-one float32 array")
    if valid_count.shape != () or valid_count.dtype != jnp.int32:
        raise ValueError("valid_count must be an int32 scalar")
    _cute_device()
    capacity = gaussian_ids.shape[0]
    block_count = (capacity + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    def finalize(sort_keys, sort_values):
        call = cjax.cutlass_call(
            _launch_finalize_sorted_outputs,
            output_shape_dtype=(
                jax.ShapeDtypeStruct(gaussian_ids.shape, jnp.int32),
                jax.ShapeDtypeStruct(tile_ids.shape, jnp.int32),
                jax.ShapeDtypeStruct((tile_count,), jnp.int32),
                jax.ShapeDtypeStruct((1,), jnp.int32),
                jax.ShapeDtypeStruct((1,), jnp.int32),
            ),
            use_static_tensors=True,
            capacity=capacity,
            tile_count=tile_count,
            block_count=block_count,
        )
        return call(sort_keys, sort_values, valid_count.reshape((1,)))

    if segment_capacity is None:
        keys, values = _prepare_sort_keys(
            gaussian_ids,
            tile_ids,
            depths,
            valid_count,
            tile_count=tile_count,
        )
        keys, values = _radix_sort(keys, values, end_bit=32)
        keys, values = _tile_radix_sort(
            keys, values, tile_count=tile_count
        )
        sorted_gaussians, sorted_tiles, offsets, effective_count, _ = (
            finalize(keys, values)
        )
        return sorted_gaussians, sorted_tiles, offsets, effective_count[0]

    keys, values, grouped_keys, grouped_values = _prepare_grouped_sort_keys(
        gaussian_ids,
        tile_ids,
        depths,
        valid_count,
        tile_count=tile_count,
    )
    _, _, grouped_offsets, grouped_count, max_segment_count = finalize(
        grouped_keys, grouped_values
    )
    segmented_call = cjax.cutlass_call(
        _launch_segmented_depth_radix,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(gaussian_ids.shape, jnp.int32),
            jax.ShapeDtypeStruct(tile_ids.shape, jnp.int32),
        ),
        use_static_tensors=True,
        capacity=capacity,
        tile_count=tile_count,
        segment_capacity=segment_capacity,
    )

    def segmented(_):
        sorted_gaussians, sorted_tiles = segmented_call(
            grouped_keys, grouped_values, grouped_offsets, grouped_count
        )
        return (
            sorted_gaussians,
            sorted_tiles,
            grouped_offsets,
            grouped_count[0],
        )

    def global_radix(_):
        sorted_keys, sorted_values = _radix_sort(keys, values, end_bit=32)
        sorted_keys, sorted_values = _tile_radix_sort(
            sorted_keys, sorted_values, tile_count=tile_count
        )
        sorted_gaussians, sorted_tiles, offsets, effective_count, _ = (
            finalize(sorted_keys, sorted_values)
        )
        return sorted_gaussians, sorted_tiles, offsets, effective_count[0]

    return jax.lax.cond(
        max_segment_count[0] <= segment_capacity,
        segmented,
        global_radix,
        operand=None,
    )


@cute.jit
def _rounded_multiply(left, right):
    return cute.math.mul(
        left,
        right,
        rounding=cute.math.RoundingMode.NEAREST_EVEN,
    )


@cute.jit
def _ellipse_line(
    b,
    disc,
    t,
    p_u,
    p_v,
    coefficient,
    coordinate,
):
    h = coordinate - p_u
    radicand = cutlass.max(disc * h * h + t * coefficient, 0.0)
    root = cutlass.Float32(0.0)
    if radicand > 0.0:
        root = radicand * cute.math.rsqrt(radicand, approx=True)
    return (
        cute.math.div(-b * h - root, coefficient, approx=True) + p_v,
        cute.math.div(-b * h + root, coefficient, approx=True) + p_v,
    )


@cute.kernel
def _prepare_accutile_base_extent_kernel(
    means: cute.Tensor,
    radii: cute.Tensor,
    depths: cute.Tensor,
    conics: cute.Tensor,
    opacities: cute.Tensor,
    valid: cute.Tensor,
    geometry: cute.Tensor,
    geometry_valid: cute.Tensor,
    gaussian_count: int,
    alpha_threshold: float,
    inverse_alpha_threshold: float,
):
    thread, _, _ = cute.arch.thread_idx()
    block_id, _, _ = cute.arch.block_idx()
    index = block_id * _BLOCK_SIZE + thread
    if index < gaussian_count:
        px = means[index, 0]
        py = means[index, 1]
        a = conics[index, 0]
        b = conics[index, 1]
        c = conics[index, 2]
        opacity = opacities[index]
        base_valid = (valid[index] != 0) & cute.math.isfinite(depths[index])
        base_valid = base_valid & cute.math.isfinite(px)
        base_valid = base_valid & cute.math.isfinite(py)
        base_valid = base_valid & cute.math.isfinite(a)
        base_valid = base_valid & cute.math.isfinite(b)
        base_valid = base_valid & cute.math.isfinite(c)
        base_valid = base_valid & cute.math.isfinite(opacity)
        base_valid = base_valid & (radii[index, 0] > 0)
        base_valid = base_valid & (radii[index, 1] > 0)
        base_valid = base_valid & (opacity > alpha_threshold)
        base_valid = base_valid & (a > 0.0)
        base_valid = base_valid & (c > 0.0)
        if not base_valid:
            px = 0.0
            py = 0.0
            a = 1.0
            b = 0.0
            c = 1.0
            opacity = alpha_threshold

        disc = _rounded_multiply(b, b) - _rounded_multiply(a, c)
        log_ratio = cute.math.log(
            _rounded_multiply(opacity, inverse_alpha_threshold)
        )
        t = cutlass.min(
            _rounded_multiply(3.33, 3.33),
            cute.math.add(
                log_ratio,
                log_ratio,
                rounding=cute.math.RoundingMode.NEAREST_EVEN,
            ),
        )
        ellipse_valid = base_valid & cute.math.isfinite(disc)
        ellipse_valid = ellipse_valid & cute.math.isfinite(t)
        ellipse_valid = ellipse_valid & (disc < 0.0)
        ellipse_valid = ellipse_valid & (t > 0.0)
        if not ellipse_valid:
            disc = -1.0
            t = 1.0
        scale = cute.math.div(-t, disc, full=True)
        x_extent = cute.math.sqrt(
            cutlass.max(_rounded_multiply(scale, c), 0.0),
            approx=True,
        )
        y_extent = cute.math.sqrt(
            cutlass.max(_rounded_multiply(scale, a), 0.0),
            approx=True,
        )
        geometry[index, 0] = b
        geometry[index, 1] = disc
        geometry[index, 2] = t
        geometry[index, 3] = px
        geometry[index, 4] = py
        geometry[index, 5] = a
        geometry[index, 6] = c
        geometry[index, 7] = x_extent
        geometry[index, 8] = y_extent
        extent_valid = ellipse_valid & cute.math.isfinite(scale)
        extent_valid = extent_valid & cute.math.isfinite(x_extent)
        extent_valid = extent_valid & cute.math.isfinite(y_extent)
        geometry_valid[index] = cutlass.Uint8(1 if extent_valid else 0)


@cute.kernel
def _finalize_accutile_state_kernel(
    geometry: cute.Tensor,
    geometry_valid: cute.Tensor,
    state_bounds: cute.Tensor,
    counts: cute.Tensor,
    gaussian_count: int,
    tile_width: int,
    tile_height: int,
):
    thread, _, _ = cute.arch.thread_idx()
    block_id, _, _ = cute.arch.block_idx()
    index = block_id * _BLOCK_SIZE + thread
    if index < gaussian_count:
        b = geometry[index, 0]
        disc = geometry[index, 1]
        t = geometry[index, 2]
        px = geometry[index, 3]
        py = geometry[index, 4]
        a = geometry[index, 5]
        c = geometry[index, 6]
        x_extent = geometry[index, 7]
        y_extent = geometry[index, 8]
        bbox_min_x = px - x_extent
        bbox_min_y = py - y_extent
        bbox_max_x = px + x_extent
        bbox_max_y = py + y_extent
        x_cross = cute.math.div(
            _rounded_multiply(b, x_extent), c, full=True
        )
        y_cross = cute.math.div(
            _rounded_multiply(b, y_extent), a, full=True
        )
        argmin_x = py + x_cross
        argmin_y = px + y_cross
        argmax_x = py - x_cross
        argmax_y = px - y_cross
        derived_valid = (geometry_valid[index] != 0) & cute.math.isfinite(
            bbox_min_x
        )
        derived_valid = derived_valid & cute.math.isfinite(bbox_min_y)
        derived_valid = derived_valid & cute.math.isfinite(bbox_max_x)
        derived_valid = derived_valid & cute.math.isfinite(bbox_max_y)
        derived_valid = derived_valid & cute.math.isfinite(argmin_x)
        derived_valid = derived_valid & cute.math.isfinite(argmin_y)
        derived_valid = derived_valid & cute.math.isfinite(argmax_x)
        derived_valid = derived_valid & cute.math.isfinite(argmax_y)
        if not derived_valid:
            bbox_min_x = 0.0
            bbox_min_y = 0.0
            bbox_max_x = 0.0
            bbox_max_y = 0.0
            argmin_x = 0.0
            argmin_y = 0.0
            argmax_x = 0.0
            argmax_y = 0.0

        rect_min_x = cutlass.min(
            cutlass.max(
                cutlass.Int32(_rounded_multiply(bbox_min_x, 1.0 / 16.0)),
                cutlass.Int32(0),
            ),
            tile_width,
        )
        rect_min_y = cutlass.min(
            cutlass.max(
                cutlass.Int32(_rounded_multiply(bbox_min_y, 1.0 / 16.0)),
                cutlass.Int32(0),
            ),
            tile_height,
        )
        rect_max_x = cutlass.min(
            cutlass.max(
                cutlass.Int32(
                    cute.math.add(
                        _rounded_multiply(bbox_max_x, 1.0 / 16.0),
                        1.0,
                        rounding=cute.math.RoundingMode.NEAREST_EVEN,
                    )
                ),
                cutlass.Int32(0),
            ),
            tile_width,
        )
        rect_max_y = cutlass.min(
            cutlass.max(
                cutlass.Int32(
                    cute.math.add(
                        _rounded_multiply(bbox_max_y, 1.0 / 16.0),
                        1.0,
                        rounding=cute.math.RoundingMode.NEAREST_EVEN,
                    )
                ),
                cutlass.Int32(0),
            ),
            tile_height,
        )
        span_x = rect_max_x - rect_min_x
        span_y = rect_max_y - rect_min_y
        state_valid = derived_valid & (span_x > 0) & (span_y > 0)
        is_y = span_y < span_x

        p_u = px
        p_v = py
        coefficient = c
        outer_min = rect_min_x
        outer_max = rect_max_x
        cross_min = rect_min_y
        cross_max = rect_max_y
        outer_bbox_min = bbox_min_x
        outer_bbox_max = bbox_max_x
        cross_bbox_min = bbox_min_y
        cross_bbox_max = bbox_max_y
        argmin_outer = argmin_y
        argmax_outer = argmax_y
        if is_y:
            p_u = py
            p_v = px
            coefficient = a
            outer_min = rect_min_y
            outer_max = rect_max_y
            cross_min = rect_min_x
            cross_max = rect_max_x
            outer_bbox_min = bbox_min_y
            outer_bbox_max = bbox_max_y
            cross_bbox_min = bbox_min_x
            cross_bbox_max = bbox_max_x
            argmin_outer = argmin_x
            argmax_outer = argmax_x

        geometry_valid[index] = cutlass.Uint8(0)
        for field in cutlass.range_constexpr(4):
            state_bounds[index, field] = 0
        if state_valid:
            geometry[index, 0] = b
            geometry[index, 1] = disc
            geometry[index, 2] = t
            geometry[index, 3] = p_u
            geometry[index, 4] = p_v
            geometry[index, 5] = coefficient
            geometry[index, 6] = outer_bbox_min
            geometry[index, 7] = outer_bbox_max
            geometry[index, 8] = cross_bbox_min
            geometry[index, 9] = cross_bbox_max
            geometry[index, 10] = argmin_outer
            geometry[index, 11] = argmax_outer
            state_bounds[index, 0] = outer_min
            state_bounds[index, 1] = outer_max
            state_bounds[index, 2] = cross_min
            state_bounds[index, 3] = cross_max
            geometry_valid[index] = cutlass.Uint8(1 if is_y else 0)
        else:
            for field in cutlass.range_constexpr(12):
                geometry[index, field] = 0.0

        total = cutlass.Int32(0)
        if state_valid:
            line_min, line_max = _ellipse_line(
                b,
                disc,
                t,
                p_u,
                p_v,
                coefficient,
                cutlass.Float32(outer_min * 16),
            )
            intersects = outer_bbox_min <= outer_min * 16
            previous_min = line_min if intersects else cross_bbox_max
            previous_max = line_max if intersects else cross_bbox_min
            for outer in cutlass.range(outer_min, outer_max):
                min_line = cutlass.Float32(outer * 16)
                max_line = min_line + 16.0
                line_min, line_max = _ellipse_line(
                    b,
                    disc,
                    t,
                    p_u,
                    p_v,
                    coefficient,
                    max_line,
                )
                current_intersects = max_line <= outer_bbox_max
                current_min = line_min if current_intersects else previous_min
                current_max = line_max if current_intersects else previous_max
                ellipse_min = cutlass.min(previous_min, current_min)
                if min_line <= argmin_outer and argmin_outer < max_line:
                    ellipse_min = cross_bbox_min
                ellipse_max = cutlass.max(previous_max, current_max)
                if min_line <= argmax_outer and argmax_outer < max_line:
                    ellipse_max = cross_bbox_max
                min_v = cutlass.max(
                    cross_min,
                    cutlass.min(
                        cross_max,
                        cutlass.Int32(
                            cute.math.div(ellipse_min, 16.0, approx=True)
                        ),
                    ),
                )
                max_v = cutlass.min(
                    cross_max,
                    cutlass.max(
                        cross_min,
                        cutlass.Int32(
                            cute.math.div(ellipse_max, 16.0, approx=True)
                            + 1.0
                        ),
                    ),
                )
                total = total + max_v - min_v
                previous_min = current_min
                previous_max = current_max
        counts[index] = cutlass.max(total, cutlass.Int32(0))


@cute.jit
def _launch_prepare_accutile_counts(
    stream: cuda.CUstream,
    means: cute.Tensor,
    radii: cute.Tensor,
    depths: cute.Tensor,
    conics: cute.Tensor,
    opacities: cute.Tensor,
    valid: cute.Tensor,
    state_floats: cute.Tensor,
    state_bounds: cute.Tensor,
    state_is_y: cute.Tensor,
    counts: cute.Tensor,
    *,
    gaussian_count: cutlass.Constexpr[int],
    tile_width: int,
    tile_height: int,
    alpha_threshold: float,
    inverse_alpha_threshold: float,
    block_count: cutlass.Constexpr[int],
):
    _prepare_accutile_base_extent_kernel(
        means,
        radii,
        depths,
        conics,
        opacities,
        valid,
        state_floats,
        state_is_y,
        gaussian_count,
        alpha_threshold,
        inverse_alpha_threshold,
    ).launch(
        grid=[block_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )
    _finalize_accutile_state_kernel(
        state_floats,
        state_is_y,
        state_bounds,
        counts,
        gaussian_count,
        tile_width,
        tile_height,
    ).launch(
        grid=[block_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def _emit_accutile_intersections_kernel(
    state_floats: cute.Tensor,
    state_bounds: cute.Tensor,
    state_is_y: cute.Tensor,
    cumulative: cute.Tensor,
    valid_count: cute.Tensor,
    gaussian_ids: cute.Tensor,
    tile_ids: cute.Tensor,
    gaussian_count: int,
    capacity: int,
    tile_width: int,
):
    thread, _, _ = cute.arch.thread_idx()
    block_id, _, _ = cute.arch.block_idx()
    rank = block_id * _BLOCK_SIZE + thread
    if rank < capacity:
        count = cutlass.min(
            cutlass.max(valid_count[0], cutlass.Int32(0)), capacity
        )
        gaussian_ids[rank] = -1
        tile_ids[rank] = -1
        if rank < count:
            low = cutlass.Int32(0)
            high = cutlass.Int32(gaussian_count)
            while low < high:
                middle = low + (high - low) // 2
                if cumulative[middle] <= rank:
                    low = middle + 1
                else:
                    high = middle
            owner = cutlass.min(low, gaussian_count - 1)
            start = cutlass.Int32(0)
            if owner > 0:
                start = cumulative[owner - 1]
            local_rank = rank - start

            b = state_floats[owner, 0]
            disc = state_floats[owner, 1]
            t = state_floats[owner, 2]
            p_u = state_floats[owner, 3]
            p_v = state_floats[owner, 4]
            coefficient = state_floats[owner, 5]
            outer_bbox_min = state_floats[owner, 6]
            outer_bbox_max = state_floats[owner, 7]
            cross_bbox_min = state_floats[owner, 8]
            cross_bbox_max = state_floats[owner, 9]
            argmin_outer = state_floats[owner, 10]
            argmax_outer = state_floats[owner, 11]
            outer_min = state_bounds[owner, 0]
            outer_max = state_bounds[owner, 1]
            cross_min = state_bounds[owner, 2]
            cross_max = state_bounds[owner, 3]

            line_min, line_max = _ellipse_line(
                b,
                disc,
                t,
                p_u,
                p_v,
                coefficient,
                cutlass.Float32(outer_min * 16),
            )
            intersects = outer_bbox_min <= outer_min * 16
            previous_min = line_min if intersects else cross_bbox_max
            previous_max = line_max if intersects else cross_bbox_min
            emitted = cutlass.Int32(0)
            selected_cross = cutlass.Int32(0)
            selected_outer = outer_min
            found = False
            for outer in cutlass.range(outer_min, outer_max):
                min_line = cutlass.Float32(outer * 16)
                max_line = min_line + 16.0
                line_min, line_max = _ellipse_line(
                    b,
                    disc,
                    t,
                    p_u,
                    p_v,
                    coefficient,
                    max_line,
                )
                current_intersects = max_line <= outer_bbox_max
                current_min = line_min if current_intersects else previous_min
                current_max = line_max if current_intersects else previous_max
                ellipse_min = cutlass.min(previous_min, current_min)
                if min_line <= argmin_outer and argmin_outer < max_line:
                    ellipse_min = cross_bbox_min
                ellipse_max = cutlass.max(previous_max, current_max)
                if min_line <= argmax_outer and argmax_outer < max_line:
                    ellipse_max = cross_bbox_max
                min_v = cutlass.max(
                    cross_min,
                    cutlass.min(
                        cross_max,
                        cutlass.Int32(
                            cute.math.div(ellipse_min, 16.0, approx=True)
                        ),
                    ),
                )
                max_v = cutlass.min(
                    cross_max,
                    cutlass.max(
                        cross_min,
                        cutlass.Int32(
                            cute.math.div(ellipse_max, 16.0, approx=True)
                            + 1.0
                        ),
                    ),
                )
                span = max_v - min_v
                if (
                    not found
                    and local_rank >= emitted
                    and local_rank < emitted + span
                ):
                    selected_cross = min_v + local_rank - emitted
                    selected_outer = outer
                    found = True
                emitted = emitted + span
                previous_min = current_min
                previous_max = current_max
            if found:
                tile_id = selected_cross * tile_width + selected_outer
                if state_is_y[owner] != 0:
                    tile_id = selected_outer * tile_width + selected_cross
                gaussian_ids[rank] = owner
                tile_ids[rank] = tile_id


@cute.jit
def _launch_emit_accutile_intersections(
    stream: cuda.CUstream,
    state_floats: cute.Tensor,
    state_bounds: cute.Tensor,
    state_is_y: cute.Tensor,
    cumulative: cute.Tensor,
    valid_count: cute.Tensor,
    gaussian_ids: cute.Tensor,
    tile_ids: cute.Tensor,
    *,
    gaussian_count: cutlass.Constexpr[int],
    capacity: cutlass.Constexpr[int],
    tile_width: int,
    block_count: cutlass.Constexpr[int],
):
    _emit_accutile_intersections_kernel(
        state_floats,
        state_bounds,
        state_is_y,
        cumulative,
        valid_count,
        gaussian_ids,
        tile_ids,
        gaussian_count,
        capacity,
        tile_width,
    ).launch(
        grid=[block_count, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


def _prepare_accutile_counts_cute(
    means2d: jax.Array,
    radii: jax.Array,
    depths: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    valid: jax.Array,
    *,
    tile_width: int,
    tile_height: int,
    alpha_threshold: float,
):
    gaussian_count = means2d.shape[0]
    block_count = (gaussian_count + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    threshold = float(np.float32(alpha_threshold))
    inverse_threshold = float(np.float32(1.0) / np.float32(threshold))
    call = cjax.cutlass_call(
        _launch_prepare_accutile_counts,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((gaussian_count, 12), jnp.float32),
            jax.ShapeDtypeStruct((gaussian_count, 4), jnp.int32),
            jax.ShapeDtypeStruct((gaussian_count,), jnp.uint8),
            jax.ShapeDtypeStruct((gaussian_count,), jnp.int32),
        ),
        use_static_tensors=True,
        gaussian_count=gaussian_count,
        tile_width=tile_width,
        tile_height=tile_height,
        alpha_threshold=threshold,
        inverse_alpha_threshold=inverse_threshold,
        block_count=block_count,
    )
    return call(
        means2d,
        radii,
        depths,
        conics,
        opacities,
        valid.astype(jnp.uint8),
    )


def _emit_accutile_intersections_cute(
    state_floats: jax.Array,
    state_bounds: jax.Array,
    state_is_y: jax.Array,
    cumulative: jax.Array,
    valid_count: jax.Array,
    *,
    capacity: int,
    tile_width: int,
):
    gaussian_count = state_floats.shape[0]
    block_count = (capacity + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    call = cjax.cutlass_call(
        _launch_emit_accutile_intersections,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
            jax.ShapeDtypeStruct((capacity,), jnp.int32),
        ),
        use_static_tensors=True,
        gaussian_count=gaussian_count,
        capacity=capacity,
        tile_width=tile_width,
        block_count=block_count,
    )
    return call(
        state_floats,
        state_bounds,
        state_is_y,
        cumulative,
        valid_count.reshape((1,)),
    )


def _run_raw_intersection_compositor_forward(
    radii,
    depths,
    means2d,
    conics,
    colors,
    opacities,
    valid,
    *,
    capacity: int,
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    state_floats, state_bounds, state_is_y, counts = (
        _prepare_accutile_counts_cute(
            means2d,
            radii,
            depths,
            conics,
            opacities,
            valid,
            tile_width=tile_width,
            tile_height=tile_height,
            alpha_threshold=alpha_threshold,
        )
    )
    cumulative, count, overflow, required = intersection_prefix_cute(
        counts, capacity=capacity
    )
    gaussian_ids, tile_ids = _emit_accutile_intersections_cute(
        state_floats,
        state_bounds,
        state_is_y,
        cumulative,
        count,
        capacity=capacity,
        tile_width=tile_width,
    )
    tile_count = tile_width * tile_height

    def staged_sort_composite(_):
        (
            sorted_gaussian_ids,
            sorted_tile_ids,
            sorted_offsets,
            sorted_count,
        ) = intersection_sort_offsets_cute(
            gaussian_ids,
            tile_ids,
            depths,
            count,
            tile_count=tile_count,
            segment_capacity=None,
        )
        (
            staged_foreground,
            staged_alpha,
            staged_accepted,
            staged_last_ids,
            staged_tile_overflow,
        ) = _run_forward(
            means2d,
            conics,
            colors,
            opacities,
            sorted_offsets.reshape((tile_height, tile_width)),
            sorted_gaussian_ids,
            sorted_count,
            image_width=image_width,
            image_height=image_height,
            per_tile_bound=per_tile_bound,
            alpha_threshold=alpha_threshold,
            transmittance_threshold=transmittance_threshold,
        )
        return (
            sorted_gaussian_ids,
            sorted_tile_ids,
            sorted_offsets,
            sorted_count,
            staged_foreground,
            staged_alpha,
            staged_accepted,
            staged_last_ids,
            staged_tile_overflow,
        )

    if per_tile_bound <= 2048:
        _, _, grouped_keys, grouped_values = _prepare_grouped_sort_keys(
            gaussian_ids,
            tile_ids,
            depths,
            count,
            tile_count=tile_count,
        )
        mega_outputs = _run_tail_mega_forward(
            grouped_keys,
            grouped_values,
            count,
            means2d,
            conics,
            colors,
            opacities,
            tile_width=tile_width,
            tile_height=tile_height,
            image_width=image_width,
            image_height=image_height,
            per_tile_bound=per_tile_bound,
            alpha_threshold=alpha_threshold,
            transmittance_threshold=transmittance_threshold,
        )
        (
            mega_gaussian_ids,
            mega_tile_ids,
            mega_offsets,
            mega_count,
            max_segment_count,
            mega_foreground,
            mega_alpha,
            mega_accepted,
            mega_last_ids,
            mega_tile_overflow,
        ) = mega_outputs

        def use_mega(_):
            return (
                mega_gaussian_ids,
                mega_tile_ids,
                mega_offsets,
                mega_count,
                mega_foreground,
                mega_alpha,
                mega_accepted,
                mega_last_ids,
                mega_tile_overflow,
            )

        outputs = jax.lax.cond(
            max_segment_count <= per_tile_bound,
            use_mega,
            staged_sort_composite,
            operand=None,
        )
    else:
        outputs = staged_sort_composite(None)

    (
        gaussian_ids,
        tile_ids,
        offsets,
        count,
        foreground,
        alpha,
        accepted,
        last_ids,
        tile_overflow,
    ) = outputs
    return (
        gaussian_ids,
        tile_ids,
        offsets,
        count,
        overflow,
        required,
        foreground,
        alpha,
        accepted,
        last_ids,
        tile_overflow,
        cumulative,
    )


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(7, 16)))
def _raw_fused_accutile_composite(
    radii,
    depths,
    means2d,
    conics,
    colors,
    opacities,
    valid,
    capacity: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    per_tile_bound: int,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    del tile_size
    result = _run_raw_intersection_compositor_forward(
        radii,
        depths,
        means2d,
        conics,
        colors,
        opacities,
        valid,
        capacity=capacity,
        tile_width=tile_width,
        tile_height=tile_height,
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    (
        gaussian_ids,
        tile_ids,
        offsets,
        count,
        overflow,
        required,
        foreground,
        alpha,
        _,
        _,
        tile_overflow,
        _,
    ) = result
    return (
        foreground,
        alpha[..., None],
        tile_overflow,
        gaussian_ids,
        tile_ids,
        offsets,
        count,
        overflow,
        required,
    )


def _raw_fused_accutile_composite_fwd(
    radii,
    depths,
    means2d,
    conics,
    colors,
    opacities,
    valid,
    capacity,
    tile_size,
    tile_width,
    tile_height,
    image_width,
    image_height,
    per_tile_bound,
    alpha_threshold,
    transmittance_threshold,
):
    del tile_size
    result = _run_raw_intersection_compositor_forward(
        radii,
        depths,
        means2d,
        conics,
        colors,
        opacities,
        valid,
        capacity=capacity,
        tile_width=tile_width,
        tile_height=tile_height,
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    (
        gaussian_ids,
        tile_ids,
        offsets,
        count,
        overflow,
        required,
        foreground,
        alpha,
        accepted,
        last_ids,
        tile_overflow,
        _,
    ) = result
    output = (
        foreground,
        alpha[..., None],
        tile_overflow,
        gaussian_ids,
        tile_ids,
        offsets,
        count,
        overflow,
        required,
    )
    residuals = (
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        gaussian_ids,
        count,
        accepted,
        last_ids,
    )
    return output, residuals


def _raw_fused_accutile_composite_bwd(
    capacity,
    tile_size,
    tile_width,
    tile_height,
    image_width,
    image_height,
    per_tile_bound,
    alpha_threshold,
    transmittance_threshold,
    residuals,
    cotangents,
):
    del capacity, tile_size
    (
        means2d,
        conics,
        colors,
        opacities,
        offsets,
        gaussian_ids,
        count,
        accepted,
        last_ids,
    ) = residuals
    rendered_cotangent, alpha_cotangent, *_ = cotangents
    gradients = _run_backward(
        means2d,
        conics,
        colors,
        opacities,
        offsets.reshape((tile_height, tile_width)),
        gaussian_ids,
        count,
        accepted,
        last_ids,
        rendered_cotangent,
        alpha_cotangent[..., 0],
        image_width=image_width,
        image_height=image_height,
        per_tile_bound=per_tile_bound,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=transmittance_threshold,
    )
    return None, None, *gradients, None


getattr(_raw_fused_accutile_composite, "defvjp")(
    _raw_fused_accutile_composite_fwd,
    _raw_fused_accutile_composite_bwd,
)


def rasterize_accutile_cute_raw_fused(
    radii,
    depths,
    means2d,
    conics,
    colors,
    opacities,
    valid,
    *,
    capacity: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    background,
    max_gaussians_per_tile: int,
    max_candidates_per_tile: int | None,
    alpha_threshold: float,
    transmittance_threshold: float,
):
    capacity = _static_int("capacity", capacity, minimum=1)
    tile_size = _static_int("tile_size", tile_size, minimum=1)
    tile_width = _static_int("tile_width", tile_width, minimum=1)
    tile_height = _static_int("tile_height", tile_height, minimum=1)
    image_width = _static_int("image_width", image_width, minimum=1)
    image_height = _static_int("image_height", image_height, minimum=1)
    max_gaussians_per_tile = _static_int(
        "max_gaussians_per_tile", max_gaussians_per_tile, minimum=1
    )
    if tile_size != 16:
        raise ValueError("the CuTe AccuTile compositor requires tile_size=16")
    if (
        tile_width * tile_size < image_width
        or tile_height * tile_size < image_height
        or tile_width * tile_size >= image_width + tile_size
        or tile_height * tile_size >= image_height + tile_size
    ):
        raise ValueError("the CuTe tile grid must cover the image exactly")

    radii = jnp.asarray(radii)
    depths = jnp.asarray(depths)
    means2d = jnp.asarray(means2d)
    conics = jnp.asarray(conics)
    colors = jnp.asarray(colors)
    opacities = jnp.asarray(opacities)
    valid = jnp.asarray(valid)
    gaussian_count = means2d.shape[0]
    channels = colors.shape[-1]
    if gaussian_count == 0:
        raise ValueError("CuTe AccuTile inputs must be non-empty")
    if radii.shape != (gaussian_count, 2) or radii.dtype != jnp.int32:
        raise ValueError("radii must have shape [N, 2] and use int32")
    if depths.shape != (gaussian_count,) or depths.dtype != jnp.float32:
        raise ValueError("depths must have shape [N] and use float32")
    if means2d.shape != (gaussian_count, 2) or means2d.dtype != jnp.float32:
        raise ValueError("means2d must have shape [N, 2] and use float32")
    if conics.shape != (gaussian_count, 3) or conics.dtype != jnp.float32:
        raise ValueError("conics must have shape [N, 3] and use float32")
    if (
        colors.shape != (gaussian_count, channels)
        or colors.dtype != jnp.float32
        or channels not in _SUPPORTED_CHANNELS
    ):
        raise ValueError("unsupported CuTe compositor color shape")
    if opacities.shape != (gaussian_count,) or opacities.dtype != jnp.float32:
        raise ValueError("opacities must have shape [N] and use float32")
    if valid.shape != (gaussian_count,) or valid.dtype != jnp.bool_:
        raise ValueError("valid must have shape [N] and use bool")

    per_tile_bound = min(gaussian_count, capacity)
    if max_candidates_per_tile is not None:
        per_tile_bound = min(
            per_tile_bound,
            _static_int(
                "max_candidates_per_tile",
                max_candidates_per_tile,
                minimum=1,
            ),
        )
    per_tile_bound = (
        math.ceil(per_tile_bound / max_gaussians_per_tile)
        * max_gaussians_per_tile
    )
    alpha_threshold = _static_float("alpha_threshold", alpha_threshold)
    transmittance_threshold = _static_float(
        "transmittance_threshold", transmittance_threshold
    )
    _cute_device()
    result = _raw_fused_accutile_composite(
        radii,
        depths,
        means2d,
        conics,
        colors,
        opacities,
        valid,
        capacity,
        tile_size,
        tile_width,
        tile_height,
        image_width,
        image_height,
        per_tile_bound,
        alpha_threshold,
        transmittance_threshold,
    )
    (
        foreground,
        alpha,
        tile_overflow,
        gaussian_ids,
        tile_ids,
        offsets,
        count,
        overflow,
        required,
    ) = result
    background = jnp.asarray(background)
    if background.shape != (channels,) or background.dtype != jnp.float32:
        raise ValueError("background must match the float32 color channels")
    rendered = foreground + background[None, None, :] * (1.0 - alpha)
    return rendered, alpha, {
        "tile_overflow": tile_overflow.reshape((tile_height, tile_width)),
        "gaussian_ids": gaussian_ids,
        "tile_ids": tile_ids,
        "offsets": offsets.reshape((tile_height, tile_width)),
        "valid_count": count,
        "overflow": overflow,
        "required_count": required,
    }


__all__ = [
    "intersection_prefix_cute",
    "intersection_sort_offsets_cute",
    "rasterize_accutile_cute_raw_fused",
]
