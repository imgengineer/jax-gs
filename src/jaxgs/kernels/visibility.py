import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


@cute.kernel
def _clear_count(count: cute.Tensor):
    count[0] = cutlass.Int32(0)


@cute.kernel
def _cluster_bounds(
    mean: cute.Tensor,
    radius: cute.Tensor,
    visible: cute.Tensor,
    bounds: cute.Tensor,
    ids: cute.Tensor,
    count: cute.Tensor,
    capacity: int,
    cluster_size: int,
    num_clusters: int,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    cluster = bidx * bdx + tidx
    if cluster < num_clusters:
        min_x = cute.Float32(1.0e30)
        min_y = cute.Float32(1.0e30)
        max_x = cute.Float32(-1.0e30)
        max_y = cute.Float32(-1.0e30)
        for offset in range(cluster_size):
            gid = cluster * cluster_size + offset
            if gid < capacity:
                if visible[gid] != 0:
                    x = mean[gid * 2]
                    y = mean[gid * 2 + 1]
                    r = radius[gid]
                    min_x = cute.min(min_x, x - r)
                    min_y = cute.min(min_y, y - r)
                    max_x = cute.max(max_x, x + r)
                    max_y = cute.max(max_y, y + r)
        bounds[cluster * 4] = min_x
        bounds[cluster * 4 + 1] = min_y
        bounds[cluster * 4 + 2] = max_x
        bounds[cluster * 4 + 3] = max_y
        if min_x < max_x:
            slot = cute.arch.atomic_add(count.iterator, cutlass.Int32(1))
            ids[slot] = cluster


@cute.kernel
def _tile_table(
    mean: cute.Tensor,
    depth: cute.Tensor,
    radius: cute.Tensor,
    visible: cute.Tensor,
    bounds: cute.Tensor,
    cluster_ids: cute.Tensor,
    cluster_count: cute.Tensor,
    out_ids: cute.Tensor,
    out_depths: cute.Tensor,
    out_valid: cute.Tensor,
    out_count: cute.Tensor,
    out_overflow: cute.Tensor,
    capacity: int,
    cluster_size: int,
    k_max: int,
    tile_size: int,
    tiles_x: int,
    num_tiles: int,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    tile = bidx * bdx + tidx
    if tile < num_tiles:
        tile_x = (tile % tiles_x) * tile_size
        tile_y = (tile // tiles_x) * tile_size
        base = tile * k_max
        for k in range(k_max):
            out_ids[base + k] = cutlass.Int32(0)
            out_depths[base + k] = cute.Float32(1.0e30)
            out_valid[base + k] = cutlass.Int8(0)
        total = cutlass.Int32(0)
        for compact in range(cluster_count[0]):
            cluster = cluster_ids[compact]
            if (
                bounds[cluster * 4] < tile_x + tile_size
                and bounds[cluster * 4 + 2] >= tile_x
                and bounds[cluster * 4 + 1] < tile_y + tile_size
                and bounds[cluster * 4 + 3] >= tile_y
            ):
                for offset in range(cluster_size):
                    gid = cluster * cluster_size + offset
                    if gid < capacity:
                        if visible[gid] != 0:
                            x = mean[gid * 2]
                            y = mean[gid * 2 + 1]
                            r = radius[gid]
                            if (
                                x + r >= tile_x
                                and x - r < tile_x + tile_size
                                and y + r >= tile_y
                                and y - r < tile_y + tile_size
                            ):
                                total += 1
                                value = depth[gid]
                                insert = total <= k_max
                                if total > k_max:
                                    last = out_depths[base + k_max - 1]
                                    last_id = out_ids[base + k_max - 1]
                                    insert = value < last or (value == last and gid < last_id)
                                if insert:
                                    pos = cute.min(total - 1, k_max - 1)
                                    shifting = True
                                    while pos > 0 and shifting:
                                        previous = out_depths[base + pos - 1]
                                        previous_id = out_ids[base + pos - 1]
                                        if previous > value or (
                                            previous == value and previous_id > gid
                                        ):
                                            out_depths[base + pos] = previous
                                            out_ids[base + pos] = previous_id
                                            out_valid[base + pos] = cutlass.Int8(1)
                                            pos -= 1
                                        else:
                                            shifting = False
                                    out_depths[base + pos] = value
                                    out_ids[base + pos] = gid
                                    out_valid[base + pos] = cutlass.Int8(1)
        out_count[tile] = cute.min(total, k_max)
        out_overflow[tile] = cutlass.Int8(1) if total > k_max else cutlass.Int8(0)


@cute.jit
def launch_cluster_compact(
    stream: cuda.CUstream,
    mean: cute.Tensor,
    radius: cute.Tensor,
    visible: cute.Tensor,
    bounds: cute.Tensor,
    ids: cute.Tensor,
    count: cute.Tensor,
    *,
    capacity: int,
    cluster_size: int,
    num_clusters: int,
):
    block = 128
    _clear_count(count).launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)
    _cluster_bounds(
        mean, radius, visible, bounds, ids, count, capacity, cluster_size, num_clusters
    ).launch(grid=[(num_clusters + block - 1) // block, 1, 1], block=[block, 1, 1], stream=stream)


@cute.jit
def launch_visibility_table(
    stream: cuda.CUstream,
    mean: cute.Tensor,
    depth: cute.Tensor,
    radius: cute.Tensor,
    visible: cute.Tensor,
    bounds: cute.Tensor,
    cluster_ids: cute.Tensor,
    cluster_count: cute.Tensor,
    out_ids: cute.Tensor,
    out_depths: cute.Tensor,
    out_valid: cute.Tensor,
    out_count: cute.Tensor,
    out_overflow: cute.Tensor,
    *,
    capacity: int,
    cluster_size: int,
    k_max: int,
    tile_size: int,
    tiles_x: int,
    num_tiles: int,
):
    block = 128
    _tile_table(
        mean,
        depth,
        radius,
        visible,
        bounds,
        cluster_ids,
        cluster_count,
        out_ids,
        out_depths,
        out_valid,
        out_count,
        out_overflow,
        capacity,
        cluster_size,
        k_max,
        tile_size,
        tiles_x,
        num_tiles,
    ).launch(grid=[(num_tiles + block - 1) // block, 1, 1], block=[block, 1, 1], stream=stream)
