"""Differentiable image, regularization, depth, and normal losses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from typing import Literal, Optional

import jax
import jax.numpy as jnp
from jax import Array

from .math import safe_normalize


Reduction = Literal["mean", "sum", "none"]


ENFORCE_CONTRACTS = (
    os.environ.get("GSPLAT_ENFORCE_CONTRACTS") == "1"
    or os.environ.get("PYTHONOPTIMIZE") == "0"
)


def _reduce(values: Array, reduction: Reduction, mask: Optional[Array] = None) -> Array:
    values = jnp.asarray(values)
    if mask is not None:
        mask_array = jnp.asarray(mask, dtype=values.dtype)
        while mask_array.ndim > values.ndim and mask_array.shape[-1] == 1:
            mask_array = jnp.squeeze(mask_array, axis=-1)
        while mask_array.ndim < values.ndim:
            mask_array = mask_array[..., None]
        mask_array = jnp.broadcast_to(mask_array, values.shape)
        values = values * mask_array
    else:
        mask_array = None
    if reduction == "none":
        return values
    if reduction == "sum":
        return jnp.sum(values)
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction: {reduction}")
    if mask_array is None:
        return jnp.mean(values)
    denominator = jnp.maximum(jnp.sum(mask_array), 1.0)
    return jnp.sum(values) / denominator


def l1_loss(
    pred: Array,
    target: Array,
) -> Array:
    """Return unreduced element-wise absolute error."""

    return jnp.abs(jnp.asarray(pred) - target)


def mse_loss(
    pred: Array,
    target: Array,
) -> Array:
    """Return unreduced element-wise squared error."""

    error = jnp.asarray(pred) - target
    return error * error


def psnr(
    prediction: Array,
    target: Array,
    data_range: float = 1.0,
    mask: Optional[Array] = None,
    eps: float = 1e-8,
) -> Array:
    """Peak signal-to-noise ratio in decibels."""

    mse = _reduce(mse_loss(prediction, target), "mean", mask)
    mse = jnp.maximum(mse, jnp.asarray(eps, mse.dtype))
    data_range_array = jnp.asarray(data_range, mse.dtype)
    return 10.0 * jnp.log10(data_range_array * data_range_array / mse)


def _gaussian_window(size: int, sigma: float, dtype: jnp.dtype) -> Array:
    coordinates = jnp.arange(size, dtype=dtype) - (size - 1) * 0.5
    weights = jnp.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    weights = weights / jnp.sum(weights)
    return weights[:, None] * weights[None, :]


def _channelwise_filter(images: Array, window: Array) -> Array:
    channels = images.shape[-1]
    kernel = jnp.broadcast_to(window[..., None, None], window.shape + (1, channels))
    return jax.lax.conv_general_dilated(
        images,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=channels,
    )


def ssim_map(
    prediction: Array,
    target: Array,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Array:
    """Return a per-pixel SSIM map for channel-last images."""

    prediction = jnp.asarray(prediction)
    target = jnp.asarray(target)
    if prediction.ndim < 3:
        raise ValueError("SSIM expects [..., H, W, C] inputs")
    original_shape = prediction.shape
    images1 = prediction.reshape((-1,) + original_shape[-3:])
    images2 = target.reshape((-1,) + original_shape[-3:])
    window = _gaussian_window(window_size, sigma, prediction.dtype)
    mu1 = _channelwise_filter(images1, window)
    mu2 = _channelwise_filter(images2, window)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = _channelwise_filter(images1 * images1, window) - mu1_sq
    sigma2_sq = _channelwise_filter(images2 * images2, window) - mu2_sq
    sigma12 = _channelwise_filter(images1 * images2, window) - mu12
    c1 = jnp.asarray((k1 * data_range) ** 2, prediction.dtype)
    c2 = jnp.asarray((k2 * data_range) ** 2, prediction.dtype)
    numerator = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    result = numerator / jnp.maximum(denominator, jnp.finfo(prediction.dtype).eps)
    return result.reshape(original_shape)


def ssim(
    prediction: Array,
    target: Array,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    reduction: Reduction = "mean",
) -> Array:
    """Structural similarity index for channel-last images."""

    values = ssim_map(prediction, target, data_range, window_size, sigma, k1, k2)
    return _reduce(values, reduction)


def _gaussian_kernel_1d(
    window_size: int,
    sigma: float,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    coordinates = jnp.arange(window_size, dtype=dtype)
    weights = jnp.exp(
        -jnp.square(coordinates - window_size // 2) / (2.0 * sigma**2)
    )
    return weights / jnp.sum(weights)


def create_ssim_window(
    window_size: int,
    channel: int,
    device: jax.Device | None = None,
) -> Array:
    """Create the current-main channel-first SSIM convolution window."""

    weights = _gaussian_kernel_1d(window_size, 1.5)
    window_2d = weights[:, None] * weights[None, :]
    window = jnp.broadcast_to(
        window_2d[None, None, ...],
        (channel, 1, window_size, window_size),
    )
    return jax.device_put(window, device) if device is not None else window


def torch_ssim_loss(
    img1: Array,
    img2: Array,
    window: Array,
    window_size: int = 11,
    channel: int = 3,
) -> Array:
    """Return the unreduced NCHW SSIM map used by current-main gsplat."""

    padding = window_size // 2

    def convolve(image: Array) -> Array:
        return jax.lax.conv_general_dilated(
            image,
            window.astype(image.dtype),
            window_strides=(1, 1),
            padding=((padding, padding), (padding, padding)),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=channel,
        )

    mu1 = convolve(img1)
    mu2 = convolve(img2)
    mu1_sq = jnp.square(mu1)
    mu2_sq = jnp.square(mu2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = convolve(img1 * img1) - mu1_sq
    sigma2_sq = convolve(img2 * img2) - mu2_sq
    sigma12 = convolve(img1 * img2) - mu1_mu2
    c1 = jnp.asarray(0.01**2, dtype=img1.dtype)
    c2 = jnp.asarray(0.03**2, dtype=img1.dtype)
    return ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )


def ssim_loss(
    img1: Array,
    img2: Array,
    window_size: int = 11,
) -> Array:
    """Return scalar ``1 - SSIM`` for NCHW images in the range ``[0, 1]``."""

    if ENFORCE_CONTRACTS:
        if not bool(jnp.all((img1 >= 0.0) & (img1 <= 1.0))):
            raise AssertionError("img1 must be in [0, 1]")
        if not bool(jnp.all((img2 >= 0.0) & (img2 <= 1.0))):
            raise AssertionError("img2 must be in [0, 1]")
    channel = img1.shape[1]
    window = create_ssim_window(window_size, channel).astype(img1.dtype)
    return 1.0 - jnp.mean(
        torch_ssim_loss(img1, img2, window, window_size, channel)
    )


def total_variation_loss(
    x: Array,
    axes: Optional[Sequence[int]] = None,
    squared: bool = True,
) -> Array:
    """Normalized total variation regularizer.

    By default this matches gsplat's bilateral-grid helper: the first two axes
    are batch/channel and variation is accumulated over every later axis.
    """

    x = jnp.asarray(x)
    if axes is None:
        axes = tuple(range(2, x.ndim))
    batch_size = max(x.shape[0], 1)
    total = jnp.asarray(0.0, dtype=x.dtype)
    for axis in axes:
        axis = axis % x.ndim
        if x.shape[axis] <= 1:
            continue
        upper = [slice(None)] * x.ndim
        lower = [slice(None)] * x.ndim
        upper[axis] = slice(1, None)
        lower[axis] = slice(None, -1)
        difference = x[tuple(upper)] - x[tuple(lower)]
        penalty = difference * difference if squared else jnp.abs(difference)
        count = max(difference.size // batch_size, 1)
        total = total + jnp.sum(penalty) / count
    return total / batch_size


tv_loss = total_variation_loss


def depth_loss(
    prediction: Array,
    target: Array,
    mask: Optional[Array] = None,
    inverse_depth: bool = False,
    scale: float = 1.0,
) -> Array:
    """L1 depth (or disparity) supervision helper."""

    prediction = jnp.asarray(prediction)
    target = jnp.asarray(target)
    if inverse_depth:
        prediction = jnp.where(prediction > 0.0, 1.0 / jnp.maximum(prediction, 1e-8), 0.0)
        target = jnp.where(target > 0.0, 1.0 / jnp.maximum(target, 1e-8), 0.0)
    loss = _reduce(l1_loss(prediction, target), "mean", mask)
    return jnp.asarray(scale, prediction.dtype) * loss


def depth_l1_loss(
    pred_depth: Array,
    gt_depth: Array,
    scene_scale: float = 1.0,
) -> Array:
    """Mean L1 loss in inverse-depth space."""

    pred_valid = pred_depth > 0.0
    target_valid = gt_depth > 0.0
    pred_safe = jnp.where(pred_valid, pred_depth, 1.0)
    target_safe = jnp.where(target_valid, gt_depth, 1.0)
    disparity = jnp.where(pred_valid, 1.0 / pred_safe, 0.0)
    target_disparity = jnp.where(target_valid, 1.0 / target_safe, 0.0)
    return jnp.mean(jnp.abs(disparity - target_disparity)) * scene_scale


def masked_l1(pred: Array, gt: Array, mask: Array) -> Array:
    """Mean absolute error over the nonzero broadcast mask region."""

    if pred.shape != gt.shape:
        raise ValueError(
            f"masked_l1: pred shape {pred.shape} != gt shape {gt.shape}. "
            "Shapes must match."
        )
    absolute_error = jnp.abs(pred - gt)
    selected = jnp.broadcast_to(mask != 0, absolute_error.shape)
    weights = selected.astype(absolute_error.dtype)
    return jnp.sum(absolute_error * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def binocular_disparity_l1(
    pred_depth: Array,
    gt_depth: Array,
    mask: Array | None = None,
    eps: float = 1.0e-7,
) -> Array:
    """Masked L1 in disparity space, excluding pairs with invalid depth."""

    if pred_depth.shape != gt_depth.shape:
        raise ValueError(
            "binocular_disparity_l1: pred_depth shape "
            f"{pred_depth.shape} != gt_depth shape {gt_depth.shape}. "
            "Shapes must match."
        )
    pred_valid = jnp.abs(pred_depth) > eps
    target_valid = jnp.abs(gt_depth) > eps
    pred_safe = jnp.where(pred_valid, pred_depth, 1.0)
    target_safe = jnp.where(target_valid, gt_depth, 1.0)
    pair_mask = pred_valid & target_valid
    if mask is not None:
        pair_mask = pair_mask & (mask != 0)
    return masked_l1(1.0 / pred_safe, 1.0 / target_safe, pair_mask)


def pearson_depth_loss(
    pred_depth: Array,
    gt_depth: Array,
    mask: Array | None = None,
) -> Array:
    """Return ``1 - Pearson r`` over flattened, optionally masked depths."""

    if pred_depth.shape != gt_depth.shape:
        raise ValueError(
            f"pearson_depth_loss: pred_depth shape {pred_depth.shape} != "
            f"gt_depth shape {gt_depth.shape}. Shapes must match."
        )
    pred = pred_depth.reshape(-1)
    target = gt_depth.reshape(-1)
    if mask is None:
        weights = jnp.ones_like(pred)
    else:
        weights = (mask != 0).reshape(-1).astype(pred.dtype)
    count = jnp.sum(weights)
    denominator_count = jnp.maximum(count, 1.0)
    pred_centered = pred - jnp.sum(pred * weights) / denominator_count
    target_centered = target - jnp.sum(target * weights) / denominator_count
    numerator = jnp.sum(pred_centered * target_centered * weights)
    variance_product = jnp.sum(jnp.square(pred_centered) * weights) * jnp.sum(
        jnp.square(target_centered) * weights
    )
    correlation = numerator / jnp.sqrt(jnp.maximum(variance_product, 1.0e-12))
    loss = 1.0 - correlation
    return jnp.where(count < 2, jnp.sum(pred_depth) * 0.0, loss)


def masked_ssim(pred: Array, gt: Array, mask: Array) -> Array:
    """Apply current-main NCHW SSIM after zeroing both images by ``mask``."""

    if pred.shape != gt.shape:
        raise ValueError(
            f"masked_ssim: pred shape {pred.shape} != gt shape {gt.shape}. "
            "Shapes must match."
        )
    return ssim_loss(pred * mask, gt * mask)


def depth_to_points(
    depths: Array,
    camtoworlds: Array,
    Ks: Array,
    z_depth: bool = True,
) -> Array:
    """Convert ``[..., H, W, 1]`` depth maps to world-space points."""

    depths = jnp.asarray(depths)
    camtoworlds = jnp.asarray(camtoworlds)
    Ks = jnp.asarray(Ks)
    height, width = depths.shape[-3:-1]
    yy, xx = jnp.meshgrid(
        jnp.arange(height, dtype=depths.dtype),
        jnp.arange(width, dtype=depths.dtype),
        indexing="ij",
    )
    fx = Ks[..., 0, 0]
    fy = Ks[..., 1, 1]
    cx = Ks[..., 0, 2]
    cy = Ks[..., 1, 2]
    camera_dirs = jnp.stack(
        (
            (xx - cx[..., None, None] + 0.5) / fx[..., None, None],
            (yy - cy[..., None, None] + 0.5) / fy[..., None, None],
            jnp.ones(depths.shape[:-1], dtype=depths.dtype),
        ),
        axis=-1,
    )
    directions = jnp.einsum(
        "...ij,...hwj->...hwi", camtoworlds[..., :3, :3], camera_dirs
    )
    if not z_depth:
        directions = safe_normalize(directions, axis=-1)
    origins = camtoworlds[..., :3, 3]
    return origins[..., None, None, :] + depths * directions


def depth_to_normals(
    depths: Array,
    camtoworlds: Array,
    Ks: Array,
    z_depth: bool = True,
) -> Array:
    """Estimate world-space normals using gsplat's centered differences."""

    points = depth_to_points(depths, camtoworlds, Ks, z_depth=z_depth)
    dx = points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]
    dy = points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]
    normals = safe_normalize(jnp.cross(dx, dy, axis=-1), axis=-1)
    padding = [(0, 0)] * normals.ndim
    padding[-3] = (1, 1)
    padding[-2] = (1, 1)
    return jnp.pad(normals, padding)


depth_to_normal = depth_to_normals


def normal_loss(
    prediction: Array,
    target: Array,
    mask: Optional[Array] = None,
) -> Array:
    """Cosine normal-consistency loss ``mean(1 - dot(n1, n2))``."""

    prediction = safe_normalize(jnp.asarray(prediction), axis=-1)
    target = safe_normalize(jnp.asarray(target), axis=-1)
    error = 1.0 - jnp.sum(prediction * target, axis=-1)
    return _reduce(error, "mean", mask)


normal_consistency_loss = normal_loss


def depth_normal_loss(
    depths: Array,
    rendered_normals: Array,
    camtoworlds: Array,
    Ks: Array,
    alpha: Optional[Array] = None,
    z_depth: bool = True,
) -> Array:
    """Auxiliary consistency loss between rendered and depth-derived normals."""

    normals_from_depth = depth_to_normals(depths, camtoworlds, Ks, z_depth=z_depth)
    rendered_normals = safe_normalize(jnp.asarray(rendered_normals), axis=-1)
    if alpha is not None:
        normals_from_depth = normals_from_depth * jax.lax.stop_gradient(alpha)
    error = 1.0 - jnp.sum(rendered_normals * normals_from_depth, axis=-1)
    return jnp.mean(error)


depth_normal_consistency_loss = depth_normal_loss


# ---------------------------------------------------------------------------
# Current-main element-wise losses and reductions
# ---------------------------------------------------------------------------


def opacity_reg_loss(
    opacities: Array, mask: Optional[Array] = None
) -> Array:
    """Mean sigmoid-activated raw opacity logits over selected Gaussians."""

    return _reduce(jax.nn.sigmoid(opacities), "mean", mask)


def scale_reg_loss(
    log_scales: Array, mask: Optional[Array] = None
) -> Array:
    """Mean exponentiated raw log-scales over selected Gaussians."""

    return _reduce(jnp.exp(log_scales), "mean", mask)


def huber_loss(pred: Array, target: Array, delta: float = 1.0) -> Array:
    """Unreduced Huber loss."""

    absolute_error = jnp.abs(pred - target)
    return jnp.where(
        absolute_error <= delta,
        0.5 * jnp.square(absolute_error),
        delta * (absolute_error - 0.5 * delta),
    )


def smooth_l1_loss(pred: Array, target: Array, beta: float = 1.0) -> Array:
    """Unreduced Smooth-L1 loss."""

    absolute_error = jnp.abs(pred - target)
    if beta == 0:
        return absolute_error
    return jnp.where(
        absolute_error < beta,
        0.5 * jnp.square(absolute_error) / beta,
        absolute_error - 0.5 * beta,
    )


def bce_loss(pred: Array, target: Array) -> Array:
    """Unreduced binary cross entropy on probabilities."""

    log_pred = jnp.maximum(jnp.log(pred), -100.0)
    log_one_minus_pred = jnp.maximum(jnp.log1p(-pred), -100.0)
    return -(target * log_pred + (1.0 - target) * log_one_minus_pred)


def bce_with_logits_loss(pred: Array, target: Array) -> Array:
    """Unreduced numerically stable binary cross entropy on logits."""

    return jax.nn.softplus(pred) - target * pred


def cross_entropy_loss(pred: Array, target: Array) -> Array:
    """Unreduced class-index cross entropy for ``(N, C)`` logits."""

    log_probabilities = jax.nn.log_softmax(pred, axis=-1)
    return -jnp.take_along_axis(
        log_probabilities,
        target[..., None],
        axis=-1,
    )[..., 0]


def bce_clipped(input: Array, target: Array, eps: float = 0.001) -> Array:
    """Float32 binary cross entropy after clipping probabilities."""

    clipped = jnp.clip(input.astype(jnp.float32), eps, 1.0 - eps)
    return bce_loss(clipped, target.astype(jnp.float32))


def depth_inverse_mse(
    pred: Array,
    target: Array,
    eps: float = 1.0e-6,
) -> Array:
    """Unreduced squared error in reciprocal-depth space."""

    return jnp.square(
        1.0 / jnp.maximum(pred, eps) - 1.0 / jnp.maximum(target, eps)
    )


def log_l1(pred: Array, target: Array) -> Array:
    """Unreduced ``log(1 + abs(error))`` loss."""

    return jnp.log1p(jnp.abs(pred - target))


def normal_cosine_loss(pred_normal: Array, gt_normal: Array) -> Array:
    """Unreduced cosine distance for already normalized 3D normals."""

    assert pred_normal.shape == gt_normal.shape
    assert pred_normal.shape[-1] == 3
    if ENFORCE_CONTRACTS:
        pred_norms = jnp.linalg.norm(pred_normal, axis=-1)
        target_norms = jnp.linalg.norm(gt_normal, axis=-1)
        if not bool(jnp.allclose(pred_norms, 1.0, atol=1.0e-3)):
            raise AssertionError("pred_normal must be unit-normalized")
        if not bool(jnp.allclose(target_norms, 1.0, atol=1.0e-3)):
            raise AssertionError("gt_normal must be unit-normalized")
    return 1.0 - jnp.sum(pred_normal * gt_normal, axis=-1)


def relu_sum(value: Array, eps: float) -> Array:
    """Sum values above a ReLU threshold."""

    return jnp.sum(jax.nn.relu(value - eps))


def weights_reg(weights_list: list[Array], dim: int = 1) -> Array:
    """Mean of concatenated squared weight norms."""

    values = [jnp.sum(jnp.square(weight), axis=dim) for weight in weights_list]
    return jnp.mean(jnp.concatenate(values))


def identity_distance(
    grid: Array,
    num_rows: int = 3,
    num_cols: int = 4,
) -> Array:
    """Frobenius distance between affine-grid channels and identity."""

    reshaped = grid.reshape(
        (grid.shape[0], num_rows, num_cols) + tuple(grid.shape[2:])
    )
    identity = jnp.eye(num_rows, num_cols, dtype=grid.dtype).reshape(
        (1, num_rows, num_cols) + (1,) * (grid.ndim - 2)
    )
    return jnp.linalg.norm(reshaped - identity, axis=(1, 2))


def total_variation_temporal(x: Array, loss_mask: Array) -> Array:
    """Squared total variation along the leading temporal dimension."""

    if x.shape[0] <= 1:
        return jnp.zeros((1,), dtype=x.dtype)
    temporal = jnp.mean(jnp.square(jnp.diff(x, axis=0)), axis=(1, 2, 3, 4))
    return temporal * loss_mask


class LinearLambdaScheduler:
    """Piecewise-step linear interpolation for a loss weight."""

    def __init__(
        self,
        start: int,
        end: int,
        lambda_init: float,
        lambda_end: float = 0.0,
        update_interval: str = "step",
        update_frequency: int = 1,
    ) -> None:
        if update_frequency <= 0:
            raise ValueError(
                f"update_frequency must be > 0, got {update_frequency}"
            )
        if end <= start:
            raise ValueError(
                f"end must be > start, got start={start}, end={end}"
            )
        total_stages = (end - start) // update_frequency
        if total_stages <= 0:
            raise ValueError(
                "total_stages must be > 0 "
                f"(start={start}, end={end}, update_frequency={update_frequency} "
                f"yields total_stages={total_stages})"
            )
        self.start = start
        self.end = end
        self.lambda_init = lambda_init
        self.lambda_end = lambda_end
        self.update_interval = update_interval
        self.update_frequency = update_frequency
        self.total_stages = total_stages

    def __call__(self, epoch: int, global_step: int) -> float:
        counter = global_step if self.update_interval == "step" else epoch
        current_stage = (counter - self.start) // self.update_frequency
        ratio = min(1.0, max(0.0, current_stage / self.total_stages))
        return (1.0 - ratio) * self.lambda_init + ratio * self.lambda_end


def reduce_mean(value: Array, mask: Array | None = None) -> Array:
    """Mean reduction, optionally over a boolean or integer mask."""

    if mask is None:
        return jnp.mean(value)
    assert not jnp.issubdtype(mask.dtype, jnp.floating), (
        f"mask must be bool or integer dtype, got {mask.dtype}. "
        "Convert to bool first (e.g. mask > 0)."
    )
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), 1)


def reduce_quantile(value: Array, quantile: float) -> Array:
    """Mean of the bottom ``quantile`` fraction of flattened values."""

    assert 0 < quantile <= 1, "quantile must be in (0, 1]"
    flattened = value.reshape(-1)
    count = int(flattened.size * quantile)
    if count == 0:
        return jnp.zeros((), dtype=value.dtype)
    return jnp.mean(jnp.sort(flattened)[:count])


def reduce_sum(value: Array) -> Array:
    """Scalar sum reduction."""

    return jnp.sum(value)


def gaussian_scale_reg(
    scales: Array,
    visibility: Array | None = None,
) -> Array:
    """Unreduced absolute scale regularization."""

    if ENFORCE_CONTRACTS and not bool(jnp.all(scales >= 0.0)):
        raise AssertionError("scales must be post-activation (>= 0)")
    loss = jnp.abs(scales)
    if visibility is not None:
        loss = loss * visibility.reshape(
            (visibility.shape[0],) + (1,) * (loss.ndim - 1)
        )
    return loss


def gaussian_density_reg(
    densities: Array,
    visibility: Array | None = None,
) -> Array:
    """Unreduced absolute density regularization."""

    if ENFORCE_CONTRACTS and not bool(jnp.all(densities >= 0.0)):
        raise AssertionError("densities must be post-activation (>= 0)")
    loss = jnp.abs(densities)
    if visibility is not None:
        loss = loss * visibility.reshape(
            (visibility.shape[0],) + (1,) * (loss.ndim - 1)
        )
    return loss


def gaussian_z_scale_reg(z_scales: Array, threshold: float) -> Array:
    """Unreduced ReLU penalty above a z-scale threshold."""

    if ENFORCE_CONTRACTS and not bool(jnp.all(z_scales >= 0.0)):
        raise AssertionError("z_scales must be post-activation (>= 0)")
    return jax.nn.relu(z_scales - threshold)


def out_of_bound_loss(positions: Array, cuboid_dims: Array) -> Array:
    """Unreduced distance outside centered per-Gaussian cuboids."""

    if ENFORCE_CONTRACTS and not bool(jnp.all(cuboid_dims > 0.0)):
        raise AssertionError("cuboid_dims must be positive")
    return jax.nn.relu(jnp.abs(positions) - cuboid_dims / 2.0)


def bilateral_grid_drift_loss(
    grids: list[Array],
    num_rows: int = 3,
    num_cols: int = 4,
) -> Array:
    """Concatenate affine-grid distances from identity."""

    if not grids:
        return jnp.empty((0,), dtype=jnp.float32)
    return jnp.concatenate(
        [
            identity_distance(grid, num_rows, num_cols).reshape(-1)
            for grid in grids
        ]
    )


# ---------------------------------------------------------------------------
# Current-main LiDAR losses
# ---------------------------------------------------------------------------


_LOSS_FN_REGISTRY: dict[str, Callable[[Array, Array], Array]] = {
    "l1": l1_loss,
    "mse": mse_loss,
    "huber": huber_loss,
    "smooth_l1": smooth_l1_loss,
    "bce": bce_loss,
    "bce_with_logits": bce_with_logits_loss,
    "bce_clipped": bce_clipped,
}


def _resolve_loss_fn(
    loss_fn: str | Callable[[Array, Array], Array],
) -> Callable[[Array, Array], Array]:
    if callable(loss_fn):
        return loss_fn
    if loss_fn not in _LOSS_FN_REGISTRY:
        raise ValueError(
            f"Unknown loss_fn {loss_fn!r}. "
            f"Available: {sorted(_LOSS_FN_REGISTRY)}. "
            "Or pass a callable (pred, target) -> Array."
        )
    return _LOSS_FN_REGISTRY[loss_fn]


def _apply_loss_fn(
    function: Callable[[Array, Array], Array],
    pred: Array,
    target: Array,
) -> Array:
    output = function(pred, target)
    if output.shape != pred.shape:
        raise ValueError(
            "loss_fn must return a per-element array with the same shape as "
            f"pred ({tuple(pred.shape)}), got shape {tuple(output.shape)}."
        )
    return output


def _validate_lidar_shapes(
    pred: Array,
    gt: Array,
    mask: Array | None,
    name: str,
) -> None:
    if pred.shape != gt.shape:
        raise ValueError(
            f"{name}: pred shape {pred.shape} != gt shape {gt.shape}. "
            "Shapes must match before flattening."
        )
    if mask is not None and mask.shape != pred.shape:
        raise ValueError(
            f"{name}: mask shape {mask.shape} != pred shape {pred.shape}. "
            "Mask must have the same shape as pred/gt."
        )


def _masked_loss_mean(
    pred: Array,
    target: Array,
    mask: Array | None,
    function: Callable[[Array, Array], Array],
) -> Array:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.size == 0:
        return jnp.sum(pred)
    if mask is None:
        return jnp.mean(_apply_loss_fn(function, pred, target))
    selected = mask.reshape(-1) != 0
    safe_pred = jnp.where(selected, pred, 0.0)
    safe_target = jnp.where(selected, target, 0.0)
    values = _apply_loss_fn(function, safe_pred, safe_target)
    weights = selected.astype(values.dtype)
    return jnp.sum(values * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def lidar_distance_loss(
    pred_distance: Array,
    gt_distance: Array,
    valid_mask: Array | None = None,
    loss_fn: str | Callable[[Array, Array], Array] = "l1",
) -> Array:
    _validate_lidar_shapes(
        pred_distance,
        gt_distance,
        valid_mask,
        "lidar_distance_loss",
    )
    return _masked_loss_mean(
        pred_distance,
        gt_distance,
        valid_mask,
        _resolve_loss_fn(loss_fn),
    )


def lidar_intensity_loss(
    pred_intensity: Array,
    gt_intensity: Array,
    valid_mask: Array | None = None,
    loss_fn: str | Callable[[Array, Array], Array] = "l1",
) -> Array:
    _validate_lidar_shapes(
        pred_intensity,
        gt_intensity,
        valid_mask,
        "lidar_intensity_loss",
    )
    return _masked_loss_mean(
        pred_intensity,
        gt_intensity,
        valid_mask,
        _resolve_loss_fn(loss_fn),
    )


def lidar_raydrop_loss(
    pred_raydrop: Array,
    gt_raydrop: Array,
    valid_mask: Array | None = None,
    loss_fn: str | Callable[[Array, Array], Array] = "bce_with_logits",
) -> Array:
    _validate_lidar_shapes(
        pred_raydrop,
        gt_raydrop,
        valid_mask,
        "lidar_raydrop_loss",
    )
    return _masked_loss_mean(
        pred_raydrop,
        gt_raydrop.astype(jnp.float32),
        valid_mask,
        _resolve_loss_fn(loss_fn),
    )


def lidar_background_loss(
    pred_opacity: Array,
    background_mask: Array,
    valid_mask: Array | None = None,
    loss_fn: str | Callable[[Array, Array], Array] = "bce",
) -> Array:
    if pred_opacity.shape != background_mask.shape:
        raise ValueError(
            "lidar_background_loss: pred_opacity shape "
            f"{pred_opacity.shape} != background_mask shape "
            f"{background_mask.shape}."
        )
    if valid_mask is not None and valid_mask.shape != pred_opacity.shape:
        raise ValueError(
            "lidar_background_loss: valid_mask shape "
            f"{valid_mask.shape} != pred_opacity shape {pred_opacity.shape}."
        )
    target = (~background_mask.astype(jnp.bool_)).astype(jnp.float32)
    return _masked_loss_mean(
        jnp.clip(pred_opacity, 0.0, 1.0),
        target,
        valid_mask,
        _resolve_loss_fn(loss_fn),
    )
