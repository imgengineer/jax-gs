"""Stable split/clone partition with block counts instead of a full-array scan."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.memory import SmemAllocator


@cute.jit
def _split_prefix(selected_split, tid, warp_counts):
    lane, warp = tid % 32, tid // 32
    ballot = cutlass.Uint32(cute.arch.vote_ballot_sync(selected_split))
    if lane == 0:
        warp_counts[warp] = cute.arch.popc(ballot)
    cute.arch.sync_threads()
    prefix, total = cutlass.Int32(0), cutlass.Int32(0)
    for index in cutlass.range_constexpr(8):
        count = warp_counts[index]
        total += count
        if index < warp:
            prefix += count
    prefix += cutlass.Int32(cute.arch.popc(ballot & (cutlass.Uint32(0xFFFFFFFF) >> (31 - lane))))
    return prefix, total


@cute.kernel
def _count_splits(split, candidate_count, counts, capacity: int):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * 256 + tid
    selected_split = False
    if index < capacity:
        selected_split = index < candidate_count[0] and split[index] != 0
    warp_counts = SmemAllocator().allocate_tensor(cutlass.Int32, cute.make_layout(8))
    _, total = _split_prefix(selected_split, tid, warp_counts)
    if tid == 0:
        counts[block] = total


@cute.kernel
def _write_order(split, candidate_count, block_prefix, order, capacity: int):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * 256 + tid
    selected_split = False
    if index < capacity:
        selected_split = index < candidate_count[0] and split[index] != 0
    warp_counts = SmemAllocator().allocate_tensor(cutlass.Int32, cute.make_layout(8))
    prefix, _ = _split_prefix(selected_split, tid, warp_counts)
    if index < capacity:
        if block > 0:
            prefix += block_prefix[block - 1]
        destination = index
        if index < candidate_count[0]:
            if selected_split:
                destination = prefix - 1
            else:
                destination = block_prefix[(capacity + 255) // 256 - 1] + index - prefix
        order[destination] = index


@cute.jit
def launch_count_splits(
    stream: cuda.CUstream,
    split: cute.Tensor,
    candidate_count: cute.Tensor,
    counts: cute.Tensor,
    *,
    capacity: int,
):
    _count_splits(split, candidate_count, counts, capacity).launch(
        grid=[(capacity + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
    )


@cute.jit
def launch_partition_order(
    stream: cuda.CUstream,
    split: cute.Tensor,
    candidate_count: cute.Tensor,
    block_prefix: cute.Tensor,
    order: cute.Tensor,
    *,
    capacity: int,
):
    _write_order(split, candidate_count, block_prefix, order, capacity).launch(
        grid=[(capacity + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
    )
