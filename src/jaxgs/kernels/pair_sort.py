"""Stable radix sort of the emitted (tile, Gaussian) pairs by tile.

The pair arena has a fixed capacity, but only its first pair_count entries
hold pairs. XLA's radix sort always sorts the whole arena; these kernels sort
only the pairs, so the cost follows the scene instead of the capacity. Each
pass is a stable counting sort by one digit of the tile key, least
significant first:

1. per-block digit histograms;
2. for every digit, an exclusive scan of its counts over the blocks;
3. per block, a stable rank of every pair among the pairs with its digit
   (warps rank their consecutive pairs with match.any), staged in shared
   memory in sorted order so that each digit's run is written contiguously.

Entries past pair_count are neither read nor written.
"""

import chex
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import jax
import jax.numpy as jnp
from cutlass.memory import SmemAllocator

_THREADS = 256
_WARPS = _THREADS // 32
_ITEMS = 8  # pairs per thread
_BLOCK_PAIRS = _THREADS * _ITEMS
_WARP_ROUNDS = _ITEMS  # each warp ranks _WARP_ROUNDS consecutive groups of 32 pairs


def digit_plan(num_tiles: int) -> tuple[int, int]:
    """(passes, digit bits) for keys below num_tiles; digits have at most 8 bits."""
    bits = max(1, (num_tiles - 1).bit_length())
    passes = -(-bits // 8)
    return passes, -(-bits // passes)


@cute.jit
def _pair_count(pair_count, max_pairs):
    return cute.min(cute.max(pair_count[0], 0), max_pairs)


@cute.kernel
def _histogram(
    keys: cute.Tensor,
    pair_count: cute.Tensor,
    histograms: cute.Tensor,
    max_pairs: int,
    max_blocks: int,
    shift: cutlass.Constexpr,
    digit_bits: cutlass.Constexpr,
):
    """histograms[digit * max_blocks + block]: the block's pairs with that digit."""
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    radix = 1 << digit_bits
    counters = SmemAllocator().allocate_tensor(cutlass.Int32, cute.make_layout(radix))
    count = _pair_count(pair_count, max_pairs)
    first = block * _BLOCK_PAIRS
    if first < count:
        if tid < radix:
            counters[tid] = cutlass.Int32(0)
        cute.arch.sync_threads()
        for k in cutlass.range_constexpr(_ITEMS):
            i = first + k * _THREADS + tid
            if i < count:
                digit = (cutlass.Int32(cutlass.Uint32(keys[i])) >> shift) & (radix - 1)
                cute.arch.atomic_add(counters.iterator + digit, cutlass.Int32(1))
        cute.arch.sync_threads()
        if tid < radix:
            histograms[tid * max_blocks + block] = counters[tid]


@cute.jit
def _exclusive_scan(value, shared):
    """Block-wide exclusive scan of one value per thread; returns (prefix, total)."""
    tid, _, _ = cute.arch.thread_idx()
    lane, warp = tid % 32, tid // 32
    inclusive = value
    for level in cutlass.range_constexpr(5):
        neighbor = cute.arch.shuffle_sync_up(inclusive, 1 << level)
        if lane >= (1 << level):
            inclusive += neighbor
    if lane == 31:
        shared[warp] = inclusive
    cute.arch.sync_threads()
    if warp == 0:
        warp_total = cutlass.Int32(0)
        if lane < _WARPS:
            warp_total = shared[lane]
        warp_inclusive = warp_total
        for level in cutlass.range_constexpr(5):
            neighbor = cute.arch.shuffle_sync_up(warp_inclusive, 1 << level)
            if lane >= (1 << level):
                warp_inclusive += neighbor
        if lane < _WARPS:
            shared[lane] = warp_inclusive - warp_total
        if lane == _WARPS - 1:
            shared[_WARPS] = warp_inclusive
    cute.arch.sync_threads()
    prefix = shared[warp] + inclusive - value
    total = shared[_WARPS]
    cute.arch.sync_threads()
    return prefix, total


@cute.kernel
def _scan_blocks(
    histograms: cute.Tensor,
    pair_count: cute.Tensor,
    digit_totals: cute.Tensor,
    max_pairs: int,
    max_blocks: int,
):
    """Per digit (one block each): exclusive prefix over the blocks, in place."""
    tid, _, _ = cute.arch.thread_idx()
    digit, _, _ = cute.arch.block_idx()
    shared = SmemAllocator().allocate_tensor(cutlass.Int32, cute.make_layout(_WARPS + 1))
    count = _pair_count(pair_count, max_pairs)
    blocks = (count + _BLOCK_PAIRS - 1) // _BLOCK_PAIRS
    carry = cutlass.Int32(0)
    start = cutlass.Int32(0)
    while start < blocks:
        b = start + tid
        value = cutlass.Int32(0)
        if b < blocks:
            value = histograms[digit * max_blocks + b]
        prefix, total = _exclusive_scan(value, shared)
        if b < blocks:
            histograms[digit * max_blocks + b] = carry + prefix
        carry += total
        start += _THREADS
    if tid == 0:
        digit_totals[digit] = carry


@cute.kernel
def _scatter(
    keys: cute.Tensor,
    values: cute.Tensor,
    sorted_keys: cute.Tensor,
    sorted_values: cute.Tensor,
    pair_count: cute.Tensor,
    histograms: cute.Tensor,
    digit_totals: cute.Tensor,
    max_pairs: int,
    max_blocks: int,
    shift: cutlass.Constexpr,
    digit_bits: cutlass.Constexpr,
):
    """Write the block's pairs to their stable positions for this digit."""
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane, warp = tid % 32, tid // 32
    radix = 1 << digit_bits
    smem = SmemAllocator()
    shared = smem.allocate_tensor(cutlass.Int32, cute.make_layout(_WARPS + 1))
    # Per warp and digit: pairs ranked so far, later the warp's offset.
    warp_counts = smem.allocate_tensor(
        cutlass.Int32, cute.make_layout((_WARPS, radix), stride=(radix, 1))
    )
    # Per digit: offset of its run inside the block, and in the output.
    block_offsets = smem.allocate_tensor(cutlass.Int32, cute.make_layout(radix))
    output_offsets = smem.allocate_tensor(cutlass.Int32, cute.make_layout(radix))
    staged_keys = smem.allocate_tensor(
        keys.element_type, cute.make_layout(_BLOCK_PAIRS), byte_alignment=16
    )
    staged_values = smem.allocate_tensor(
        cutlass.Int32, cute.make_layout(_BLOCK_PAIRS), byte_alignment=16
    )
    count = _pair_count(pair_count, max_pairs)
    first = block * _BLOCK_PAIRS
    if first < count:
        for d in cutlass.range_constexpr((radix + _THREADS - 1) // _THREADS):
            if d * _THREADS + tid < radix:
                for w in cutlass.range_constexpr(_WARPS):
                    warp_counts[w, d * _THREADS + tid] = cutlass.Int32(0)
        cute.arch.sync_threads()
        # Each warp ranks its consecutive pairs in order, 32 at a time.
        own_keys = cute.make_rmem_tensor(_WARP_ROUNDS, keys.element_type)
        own_values = cute.make_rmem_tensor(_WARP_ROUNDS, cutlass.Int32)
        digits = cute.make_rmem_tensor(_WARP_ROUNDS, cutlass.Int32)
        ranks = cute.make_rmem_tensor(_WARP_ROUNDS, cutlass.Int32)
        below = cute.arch.lanemask_lt()
        for r in cutlass.range_constexpr(_WARP_ROUNDS):
            i = first + (warp * _WARP_ROUNDS + r) * 32 + lane
            digit = cutlass.Int32(-1)
            if i < count:
                own_keys[r] = keys[i]
                own_values[r] = values[i]
                digit = (cutlass.Int32(cutlass.Uint32(own_keys[r])) >> shift) & (radix - 1)
            digits[r] = digit
            peers = cute.arch.match_sync(0xFFFFFFFF, digit, "any")
            rank = cutlass.Int32(0)
            if digit >= 0:
                rank = warp_counts[warp, digit] + cutlass.Int32(cute.arch.popc(peers & below))
            ranks[r] = rank
            cute.arch.sync_warp()
            if digit >= 0 and (peers & below) == 0:
                warp_counts[warp, digit] = warp_counts[warp, digit] + cutlass.Int32(
                    cute.arch.popc(peers)
                )
            cute.arch.sync_warp()
        cute.arch.sync_threads()
        # Warp offsets inside each digit's run, then the runs' block offsets.
        for d in cutlass.range_constexpr((radix + _THREADS - 1) // _THREADS):
            digit = d * _THREADS + tid
            total = cutlass.Int32(0)
            if digit < radix:
                for w in cutlass.range_constexpr(_WARPS):
                    pairs = warp_counts[w, digit]
                    warp_counts[w, digit] = total
                    total += pairs
            prefix, _ = _exclusive_scan(total, shared)
            if digit < radix:
                block_offsets[digit] = prefix
        # Output offsets: earlier digits' totals, then this digit's earlier blocks.
        for d in cutlass.range_constexpr((radix + _THREADS - 1) // _THREADS):
            digit = d * _THREADS + tid
            total = cutlass.Int32(0)
            if digit < radix:
                total = digit_totals[digit]
            prefix, _ = _exclusive_scan(total, shared)
            if digit < radix:
                output_offsets[digit] = prefix + histograms[digit * max_blocks + block]
        cute.arch.sync_threads()
        for r in cutlass.range_constexpr(_WARP_ROUNDS):
            digit = digits[r]
            if digit >= 0:
                slot = block_offsets[digit] + warp_counts[warp, digit] + ranks[r]
                staged_keys[slot] = own_keys[r]
                staged_values[slot] = own_values[r]
        cute.arch.sync_threads()
        # Consecutive threads write consecutive pairs of each digit's run.
        size = cute.min(count - first, _BLOCK_PAIRS)
        for k in cutlass.range_constexpr(_ITEMS):
            slot = k * _THREADS + tid
            if slot < size:
                key = staged_keys[slot]
                digit = (cutlass.Int32(cutlass.Uint32(key)) >> shift) & (radix - 1)
                position = output_offsets[digit] + slot - block_offsets[digit]
                sorted_keys[position] = key
                sorted_values[position] = staged_values[slot]


@cute.jit
def _sort_pass(
    stream,
    keys,
    values,
    sorted_keys,
    sorted_values,
    pair_count,
    histograms,
    digit_totals,
    max_pairs,
    max_blocks,
    shift: cutlass.Constexpr,
    digit_bits: cutlass.Constexpr,
):
    _histogram(keys, pair_count, histograms, max_pairs, max_blocks, shift, digit_bits).launch(
        grid=[max_blocks, 1, 1], block=[_THREADS, 1, 1], stream=stream
    )
    _scan_blocks(histograms, pair_count, digit_totals, max_pairs, max_blocks).launch(
        grid=[1 << digit_bits, 1, 1], block=[_THREADS, 1, 1], stream=stream
    )
    _scatter(
        keys,
        values,
        sorted_keys,
        sorted_values,
        pair_count,
        histograms,
        digit_totals,
        max_pairs,
        max_blocks,
        shift,
        digit_bits,
    ).launch(grid=[max_blocks, 1, 1], block=[_THREADS, 1, 1], stream=stream)


@cute.jit
def launch_sort_pairs(
    stream: cuda.CUstream,
    keys: cute.Tensor,
    values: cute.Tensor,
    pair_count: cute.Tensor,
    sorted_keys: cute.Tensor,
    sorted_values: cute.Tensor,
    scratch_keys: cute.Tensor,
    scratch_values: cute.Tensor,
    histograms: cute.Tensor,
    digit_totals: cute.Tensor,
    *,
    max_pairs: int,
    passes: cutlass.Constexpr,
    digit_bits: cutlass.Constexpr,
):
    """Stably sort the first pair_count (key, value) pairs by key."""
    max_blocks = (max_pairs + _BLOCK_PAIRS - 1) // _BLOCK_PAIRS
    # Alternate between the scratch and output buffers so the last pass
    # writes the output.
    source = (keys, values)
    for p in cutlass.range_constexpr(passes):
        target = (scratch_keys, scratch_values)
        if cutlass.const_expr((passes - 1 - p) % 2 == 0):
            target = (sorted_keys, sorted_values)
        _sort_pass(
            stream,
            *source,
            *target,
            pair_count,
            histograms,
            digit_totals,
            max_pairs,
            max_blocks,
            p * digit_bits,
            digit_bits,
        )
        source = target


def sort_pairs_by_tile(
    tile_ids: chex.Array, gaussian_ids: chex.Array, pair_count: chex.Array, num_tiles: int
) -> tuple[chex.Array, chex.Array]:
    """Stably sort the first pair_count pairs (clamped to the arena) by tile id.

    Tile ids must lie below num_tiles. Entries past the pairs are undefined.
    """
    from cutlass.jax import cutlass_call

    max_pairs = tile_ids.shape[0]
    passes, digit_bits = digit_plan(num_tiles)
    max_blocks = -(-max_pairs // _BLOCK_PAIRS)
    call = cutlass_call(
        launch_sort_pairs,
        compile_key=launch_sort_pairs,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(tile_ids.shape, tile_ids.dtype),
            jax.ShapeDtypeStruct(gaussian_ids.shape, gaussian_ids.dtype),
            jax.ShapeDtypeStruct(tile_ids.shape if passes > 1 else (1,), tile_ids.dtype),
            jax.ShapeDtypeStruct(gaussian_ids.shape if passes > 1 else (1,), gaussian_ids.dtype),
            jax.ShapeDtypeStruct(((1 << digit_bits) * max_blocks,), jnp.int32),
            jax.ShapeDtypeStruct((1 << digit_bits,), jnp.int32),
        ),
        use_static_tensors=True,
        max_pairs=max_pairs,
        passes=passes,
        digit_bits=digit_bits,
    )
    sorted_tiles, sorted_ids, *_ = call(
        tile_ids, gaussian_ids, jnp.asarray(pair_count, jnp.int32).reshape(1)
    )
    return sorted_tiles, sorted_ids
