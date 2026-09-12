from __future__ import annotations

import operator

import jax
import jax.numpy as jnp

MAX_ALPHA = 0.999
DEFAULT_ALPHA_THRESHOLD = 1.0 / 255.0
DEFAULT_TRANSMITTANCE_THRESHOLD = 1.0e-4


def _static_positive_int(name: str, value: int) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a static integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def composite_sorted_tile(
    gaussian_ids: jax.Array,
    candidate_count: jax.Array | int,
    means2d: jax.Array,
    conics: jax.Array,
    opacities: jax.Array,
    features: jax.Array,
    pixel_coords: jax.Array,
    pixel_valid: jax.Array,
    *,
    chunk_size: int = 32,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    transmittance_threshold: float = DEFAULT_TRANSMITTANCE_THRESHOLD,
) -> tuple[jax.Array, jax.Array]:
    """Composite one tile's depth-sorted candidates in bounded chunks.

    Args:
        gaussian_ids: Padded, depth-sorted Gaussian ids with shape ``[K]``.
        candidate_count: Number of valid prefix slots in ``gaussian_ids``.
        means2d: Gaussian centers with shape ``[N, 2]``.
        conics: Inverse covariance upper triangles ``[N, 3]``.
        opacities: Gaussian opacities ``[N]``.
        features: Gaussian features ``[N, D]``.
        pixel_coords: Pixel-center coordinates ``[P, 2]``.
        pixel_valid: Valid image pixels ``[P]``.

    Returns:
        Accumulated features ``[P, D]`` and accumulated alpha ``[P, 1]``.

    The scan has a static number of chunks, but later chunks take an identity
    branch once no candidates remain or every valid pixel's transmittance is at
    or below the termination threshold. Peak Gaussian/pixel workspace is
    ``O(chunk_size * P)`` rather than ``O(K * P)``.
    """

    chunk_size = _static_positive_int("chunk_size", chunk_size)
    gaussian_ids = jnp.asarray(gaussian_ids, dtype=jnp.int32)
    means2d = jnp.asarray(means2d)
    conics = jnp.asarray(conics)
    opacities = jnp.asarray(opacities)
    features = jnp.asarray(features)
    pixel_coords = jnp.asarray(pixel_coords, dtype=means2d.dtype)
    pixel_valid = jnp.asarray(pixel_valid, dtype=jnp.bool_)
    if gaussian_ids.ndim != 1:
        raise ValueError("gaussian_ids must have shape [K]")
    gaussian_count = means2d.shape[0]
    if means2d.shape != (gaussian_count, 2):
        raise ValueError("means2d must have shape [N, 2]")
    if conics.shape != (gaussian_count, 3):
        raise ValueError("conics must have shape [N, 3]")
    if opacities.shape != (gaussian_count,):
        raise ValueError("opacities must have shape [N]")
    if features.ndim != 2 or features.shape[0] != gaussian_count:
        raise ValueError("features must have shape [N, D]")
    pixel_count = pixel_coords.shape[0]
    if pixel_coords.shape != (pixel_count, 2):
        raise ValueError("pixel_coords must have shape [P, 2]")
    if pixel_valid.shape != (pixel_count,):
        raise ValueError("pixel_valid must have shape [P]")

    feature_count = features.shape[-1]
    rendered = jnp.zeros((pixel_count, feature_count), dtype=features.dtype)
    accumulated_alpha = jnp.zeros((pixel_count,), dtype=opacities.dtype)
    capacity = gaussian_ids.shape[0]
    if capacity == 0 or gaussian_count == 0 or pixel_count == 0:
        return rendered, accumulated_alpha[:, None]

    candidate_count = jnp.clip(
        jnp.asarray(candidate_count, dtype=jnp.int32), 0, capacity
    )
    transmittance = jnp.ones((pixel_count,), dtype=opacities.dtype)
    alive = pixel_valid
    done = (candidate_count == 0) | jnp.all(~pixel_valid)
    local_slots = jnp.arange(chunk_size, dtype=jnp.int32)
    chunk_starts = (
        jnp.arange((capacity + chunk_size - 1) // chunk_size, dtype=jnp.int32)
        * chunk_size
    )
    alpha_threshold = jnp.asarray(alpha_threshold, dtype=opacities.dtype)
    transmittance_threshold = jnp.asarray(
        transmittance_threshold, dtype=opacities.dtype
    )

    def scan_chunk(carry, chunk_start):
        _, _, _, _, done = carry
        should_process = (~done) & (chunk_start < candidate_count)

        def process_chunk(state):
            rendered, accumulated_alpha, transmittance, alive, _ = state
            slots = chunk_start + local_slots
            safe_slots = jnp.clip(slots, 0, capacity - 1)
            ids = gaussian_ids[safe_slots]
            slot_valid = (
                (slots < candidate_count)
                & (slots < capacity)
                & (ids >= 0)
                & (ids < gaussian_count)
            )
            safe_ids = jnp.clip(ids, 0, gaussian_count - 1)
            selected_means = means2d[safe_ids]
            selected_conics = conics[safe_ids]
            selected_opacities = opacities[safe_ids]
            selected_features = features[safe_ids]

            delta_x = pixel_coords[None, :, 0] - selected_means[:, None, 0]
            delta_y = pixel_coords[None, :, 1] - selected_means[:, None, 1]
            sigma = (
                0.5
                * (
                    selected_conics[:, None, 0] * delta_x**2
                    + selected_conics[:, None, 2] * delta_y**2
                )
                + selected_conics[:, None, 1] * delta_x * delta_y
            )
            alpha = jnp.minimum(
                selected_opacities[:, None] * jnp.exp(-jnp.maximum(sigma, 0.0)),
                MAX_ALPHA,
            )
            alpha_valid = (
                slot_valid[:, None]
                & alive[None, :]
                & jnp.isfinite(sigma)
                & (sigma >= 0.0)
                & (alpha >= alpha_threshold)
            )
            alpha = jnp.where(alpha_valid, alpha, 0.0)
            exclusive_local_transmittance = jnp.concatenate(
                (
                    jnp.ones((1, pixel_count), dtype=alpha.dtype),
                    jnp.cumprod(1.0 - alpha, axis=0)[:-1],
                ),
                axis=0,
            )
            sample_transmittance = (
                transmittance[None, :] * exclusive_local_transmittance
            )
            next_transmittance = sample_transmittance * (1.0 - alpha)
            accepted = alpha_valid & (next_transmittance > transmittance_threshold)
            weights = jnp.where(accepted, alpha * sample_transmittance, 0.0)
            rendered = rendered + jnp.einsum("kp,kd->pd", weights, selected_features)
            accumulated_alpha = accumulated_alpha + jnp.sum(weights, axis=0)
            transmittance = transmittance * jnp.prod(
                jnp.where(accepted, 1.0 - alpha, 1.0), axis=0
            )
            terminated = jnp.any(alpha_valid & ~accepted, axis=0)
            alive = alive & ~terminated
            done = (chunk_start + chunk_size >= candidate_count) | jnp.all(~alive)
            return rendered, accumulated_alpha, transmittance, alive, done

        carry = jax.lax.cond(
            should_process,
            process_chunk,
            lambda state: (
                state[0],
                state[1],
                state[2],
                state[3],
                jnp.asarray(True),
            ),
            carry,
        )
        return carry, None

    (rendered, accumulated_alpha, _, _, _), _ = jax.lax.scan(
        scan_chunk,
        (rendered, accumulated_alpha, transmittance, alive, done),
        chunk_starts,
    )
    return rendered, accumulated_alpha[:, None]


__all__ = [
    "DEFAULT_ALPHA_THRESHOLD",
    "DEFAULT_TRANSMITTANCE_THRESHOLD",
    "MAX_ALPHA",
    "composite_sorted_tile",
]
