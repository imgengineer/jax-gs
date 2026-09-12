"""Pure-JAX point-cloud initialization utilities from gsplat current main."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def multi_frame_depth_unprojection(
    images: jax.Array,
    depths: jax.Array,
    masks: jax.Array,
    poses: jax.Array,
    intrinsics: jax.Array,
    max_points: int | None = None,
    *,
    key: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Unproject masked depth maps into one world-space point cloud.

    This is an eager initialization helper because the number of valid points
    depends on the input masks. ``key`` is a pure-JAX extension to the upstream
    API; a fixed key is used when it is omitted so subsampling is reproducible.
    """

    images = jnp.asarray(images)
    depths = jnp.asarray(depths)
    masks = jnp.asarray(masks)
    poses = jnp.asarray(poses)
    intrinsics = jnp.asarray(intrinsics)
    frame_count = images.shape[0]
    for name, value in (
        ("depths", depths),
        ("masks", masks),
        ("poses", poses),
        ("intrinsics", intrinsics),
    ):
        if value.shape[0] != frame_count:
            raise ValueError(
                "multi_frame_depth_unprojection: leading dim mismatch - "
                f"images has {frame_count} frames but {name} has "
                f"{value.shape[0]}"
            )

    height, width = images.shape[1:3]
    images_f = (
        images.astype(jnp.float32) / 255.0
        if images.dtype == jnp.uint8
        else images.astype(jnp.float32)
    )
    depths_f = depths.astype(jnp.float32)
    v_coords, u_coords = jnp.meshgrid(
        jnp.arange(height, dtype=jnp.float32),
        jnp.arange(width, dtype=jnp.float32),
        indexing="ij",
    )
    xyz_chunks: list[jax.Array] = []
    rgb_chunks: list[jax.Array] = []

    for frame_index in range(frame_count):
        valid = (masks[frame_index] != 0) & (depths_f[frame_index] > 0)
        if not bool(jnp.any(valid)):
            continue
        depth = depths_f[frame_index][valid]
        intrinsic = intrinsics[frame_index].astype(jnp.float32)
        x_camera = (u_coords[valid] - intrinsic[0, 2]) * depth / intrinsic[0, 0]
        y_camera = (v_coords[valid] - intrinsic[1, 2]) * depth / intrinsic[1, 1]
        camera_homogeneous = jnp.stack(
            (x_camera, y_camera, depth, jnp.ones_like(depth)), axis=-1
        )
        world_homogeneous = (
            camera_homogeneous @ poses[frame_index].astype(jnp.float32).T
        )
        xyz_chunks.append(world_homogeneous[:, :3])
        rgb_chunks.append(images_f[frame_index][valid])

    if not xyz_chunks:
        empty = jnp.zeros((0, 3), dtype=jnp.float32)
        return empty, empty.copy()

    xyz = jnp.concatenate(xyz_chunks, axis=0)
    rgb = jnp.concatenate(rgb_chunks, axis=0)
    if max_points is not None and xyz.shape[0] > max_points:
        if key is None:
            key = jax.random.key(0)
        indices = jax.random.permutation(key, xyz.shape[0])[:max_points]
        xyz, rgb = xyz[indices], rgb[indices]
    return xyz, rgb


def knn_scale_init(
    xyz: jax.Array,
    k: int = 3,
    eps: float = 1.0e-7,
    chunk_size: int = 1024,
) -> jax.Array:
    """Return log RMS distance to each point's ``k`` nearest neighbours.

    Query rows are processed in blocks, bounding the largest pairwise-distance
    buffer to ``O(chunk_size * N)`` rather than ``O(N**2)``.
    """

    xyz = jnp.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    point_count = xyz.shape[0]
    if point_count <= k:
        raise ValueError(
            f"knn_scale_init: need at least k+1={k + 1} points, got {point_count}"
        )

    chunk = max(1, min(int(chunk_size), point_count))
    rms_chunks: list[jax.Array] = []
    for start in range(0, point_count, chunk):
        end = min(start + chunk, point_count)
        differences = xyz[start:end, None, :] - xyz[None, :, :]
        squared_distances = jnp.sum(differences * differences, axis=-1)
        local_indices = jnp.arange(end - start)
        squared_distances = squared_distances.at[
            local_indices, start + local_indices
        ].set(jnp.inf)
        nearest_squared = -jax.lax.top_k(-squared_distances, k)[0]
        rms_chunks.append(jnp.sqrt(jnp.mean(nearest_squared, axis=-1)))
    rms = jnp.concatenate(rms_chunks, axis=0)
    return jnp.log(jnp.maximum(rms, jnp.asarray(eps, dtype=rms.dtype)))


__all__ = ["knn_scale_init", "multi_frame_depth_unprojection"]
