import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


@cute.kernel
def _clear_slots(slots: cute.Tensor, valid: cute.Tensor, max_new: int):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 128 + tidx
    if i < max_new:
        slots[i] = cutlass.Int32(0)
        valid[i] = cutlass.Int8(0)


@cute.kernel
def _allocate_free(
    free_mask: cute.Tensor,
    rank: cute.Tensor,
    slots: cute.Tensor,
    valid: cute.Tensor,
    capacity: int,
    max_new: int,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 128 + tidx
    if i < capacity:
        if free_mask[i] != 0:
            slot = rank[i]
            if slot < max_new:
                slots[slot] = i
                valid[slot] = cutlass.Int8(1)


@cute.jit
def launch_allocate_free(
    stream: cuda.CUstream,
    free_mask: cute.Tensor,
    rank: cute.Tensor,
    slots: cute.Tensor,
    valid: cute.Tensor,
    *,
    capacity: int,
    max_new: int,
):
    _clear_slots(slots, valid, max_new).launch(
        grid=[(max_new + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
    )
    _allocate_free(free_mask, rank, slots, valid, capacity, max_new).launch(
        grid=[(capacity + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
    )
