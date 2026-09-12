# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic host-side NumPy preprocessing for scene coordinates.

These helpers intentionally run outside the differentiable JAX runtime.  They
normalize dataset cameras and points once before training.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def _as_similarity_matrix(matrix: FloatArray) -> np.ndarray:
    """Validate and copy a homogeneous, orientation-preserving similarity."""

    raw = np.asarray(matrix)
    if raw.shape != (4, 4):
        raise ValueError(f"scene transform must have shape (4, 4), got {raw.shape}")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise ValueError("scene transform must be a real numeric matrix")
    result = raw.astype(np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("scene transform must be finite")
    if not np.allclose(
        result[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError("scene transform must be homogeneous")

    linear = result[:3, :3]
    scale = float(np.linalg.norm(linear[0]))
    if scale == 0.0:
        raise ValueError("scene transform must be a nonzero similarity")
    rotation = linear / scale
    if (
        not np.allclose(
            rotation @ rotation.T,
            np.eye(3),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        or np.linalg.det(rotation) <= 0.0
    ):
        raise ValueError("scene transform must be an orientation-preserving similarity")
    return result


def similarity_from_cameras(
    c2w: FloatArray,
    strict_scaling: bool = False,
    center_method: str = "focus",
) -> FloatArray:
    """Return the current-main similarity transform for OpenCV cameras."""
    t = c2w[:, :3, 3]
    rotation = c2w[:, :3, :3]

    ups = np.sum(rotation * np.array([0.0, -1.0, 0.0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)
    up_camspace = np.array([0.0, -1.0, 0.0])
    cosine = np.sum(up_camspace * world_up)
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if cosine > -1:
        align_rotation = np.eye(3) + skew + (skew @ skew) / (1 + cosine)
    else:
        align_rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    rotation = align_rotation @ rotation
    forwards = np.sum(rotation * np.array([0.0, 0.0, 1.0]), axis=-1)
    t = (align_rotation @ t[..., None])[..., 0]

    if center_method == "focus":
        nearest = t + np.sum(forwards * -t, axis=-1)[:, None] * forwards
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = align_rotation
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale
    return transform


def align_principal_axes(point_cloud: FloatArray) -> FloatArray:
    """Center a point cloud and align axes by descending PCA variance."""
    centroid = np.median(point_cloud, axis=0)
    translated_point_cloud = point_cloud - centroid
    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]

    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    rotation_matrix = eigenvectors.T

    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid
    return transform


def transform_points(matrix: FloatArray, points: FloatArray) -> FloatArray:
    """Transform ``Nx3`` points with a 4x4 similarity matrix."""
    assert matrix.shape == (4, 4)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix: FloatArray, camtoworlds: FloatArray) -> FloatArray:
    """Transform ``Nx4x4`` camera-to-world matrices and remove scale from R."""
    assert matrix.shape == (4, 4)
    assert len(camtoworlds.shape) == 3 and camtoworlds.shape[1:] == (4, 4)
    transformed = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(transformed[:, 0, :3], axis=1)
    transformed[:, :3, :3] /= scaling[:, None, None]
    return transformed


def normalize_scene(
    camtoworlds: FloatArray,
    points: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Normalize cameras and points using current-main's COLMAP sequence."""
    transform_1 = similarity_from_cameras(camtoworlds)
    normalized_cameras = transform_cameras(transform_1, camtoworlds)
    normalized_points = transform_points(transform_1, points)

    transform_2 = align_principal_axes(normalized_points)
    normalized_cameras = transform_cameras(transform_2, normalized_cameras)
    normalized_points = transform_points(transform_2, normalized_points)
    transform = transform_2 @ transform_1

    if np.median(normalized_points[:, 2]) > np.mean(normalized_points[:, 2]):
        transform_3 = np.diag([1.0, -1.0, -1.0, 1.0])
        normalized_cameras = transform_cameras(transform_3, normalized_cameras)
        normalized_points = transform_points(transform_3, normalized_points)
        transform = transform_3 @ transform

    return normalized_cameras, normalized_points, transform
