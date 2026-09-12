import math

import jax
import jax.numpy as jnp
import pytest

import jax_gs
from jax_gs import losses
from jax_gs.losses import (
    LinearLambdaScheduler,
    bce_clipped,
    bce_loss,
    bce_with_logits_loss,
    bilateral_grid_drift_loss,
    binocular_disparity_l1,
    create_ssim_window,
    cross_entropy_loss,
    depth_inverse_mse,
    depth_l1_loss,
    gaussian_density_reg,
    gaussian_scale_reg,
    gaussian_z_scale_reg,
    huber_loss,
    identity_distance,
    lidar_background_loss,
    lidar_distance_loss,
    lidar_intensity_loss,
    lidar_raydrop_loss,
    log_l1,
    masked_l1,
    masked_ssim,
    normal_cosine_loss,
    opacity_reg_loss,
    out_of_bound_loss,
    pearson_depth_loss,
    reduce_mean,
    reduce_quantile,
    reduce_sum,
    relu_sum,
    scale_reg_loss,
    smooth_l1_loss,
    ssim_loss,
    torch_ssim_loss,
    total_variation_temporal,
    weights_reg,
)


def test_current_main_loss_root_exports_and_capability():
    assert jax_gs.has_losses()
    for name in (
        "l1_loss",
        "mse_loss",
        "ssim_loss",
        "depth_l1_loss",
    ):
        assert name not in jax_gs.__all__
        assert hasattr(jax_gs, name)
    for name in (
        "gaussian_scale_reg",
        "FusedGaussianLosses",
    ):
        assert name in jax_gs.__all__
        assert hasattr(jax_gs, name)


def test_ssim_reference_window_map_loss_and_gradient():
    window = create_ssim_window(11, 3)
    assert window.shape == (3, 1, 11, 11)
    assert jnp.allclose(jnp.sum(window, axis=(1, 2, 3)), 1.0)
    image = jnp.linspace(0.0, 1.0, 2 * 3 * 8 * 9).reshape(2, 3, 8, 9)
    ssim_map = torch_ssim_loss(image, image, window)
    assert ssim_map.shape == image.shape
    assert jnp.allclose(ssim_loss(image, image), 0.0, atol=1.0e-6)
    gradient = jax.jit(jax.grad(lambda value: ssim_loss(value, image * 0.9)))(image)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_depth_l1_and_binocular_depth_contracts():
    pred = jnp.asarray([1.0, 2.0, 0.0, 4.0])
    target = jnp.asarray([2.0, 2.0, 3.0, 0.0])
    assert jnp.allclose(depth_l1_loss(pred, target), (0.5 + 0.0 + 1 / 3 + 0.25) / 4)
    pair_loss = binocular_disparity_l1(pred, target)
    assert jnp.allclose(pair_loss, 0.25)
    masked = binocular_disparity_l1(
        pred,
        target,
        jnp.asarray([False, True, True, True]),
    )
    assert jnp.allclose(masked, 0.0)
    with pytest.raises(ValueError, match="Shapes must match"):
        binocular_disparity_l1(pred, target[:2])


def test_pearson_depth_loss_correlation_mask_and_degenerate_inputs():
    pred = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    assert jnp.allclose(pearson_depth_loss(pred, 2.0 * pred), 0.0)
    assert jnp.allclose(pearson_depth_loss(pred, -pred), 2.0)
    assert jnp.allclose(
        pearson_depth_loss(pred, pred, jnp.zeros_like(pred, dtype=jnp.bool_)),
        0.0,
    )
    assert jnp.isfinite(pearson_depth_loss(jnp.ones(4), jnp.ones(4)))


def test_masked_photometric_losses_handle_broadcast_and_empty_masks():
    pred = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 2, 2) / 12.0
    target = jnp.zeros_like(pred)
    mask = jnp.asarray([[[[True, False], [False, False]]]])
    expected = jnp.mean(pred[:, :, :1, :1])
    assert jnp.allclose(masked_l1(pred, target, mask), expected)
    assert jnp.allclose(masked_l1(pred, target, jnp.zeros_like(mask)), 0.0)
    assert jnp.allclose(masked_ssim(pred, pred, mask), 0.0, atol=1.0e-6)


@pytest.mark.parametrize("function", [lidar_distance_loss, lidar_intensity_loss])
def test_lidar_regression_losses_mask_dispatch_and_gradient(function):
    pred = jnp.asarray([1.0, 2.0, 4.0])
    target = jnp.asarray([2.0, 2.0, 1.0])
    mask = jnp.asarray([True, False, True])
    assert jnp.allclose(function(pred, target, mask), 2.0)
    assert jnp.allclose(function(pred, target, mask, loss_fn="mse"), 5.0)
    gradient = jax.grad(lambda value: function(value, target, mask))(pred)
    assert jnp.array_equal(gradient[1], 0.0)


def test_lidar_classification_and_background_losses():
    logits = jnp.asarray([5.0, -5.0])
    labels = jnp.asarray([1.0, 0.0])
    assert float(lidar_raydrop_loss(logits, labels)) < 0.01
    opacity = jnp.asarray([0.0, 1.0])
    background = jnp.asarray([True, False])
    assert jnp.allclose(
        lidar_background_loss(opacity, background, loss_fn="bce_clipped"),
        -math.log(0.999),
        rtol=1.0e-4,
    )
    wrong_boundary = lidar_background_loss(
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([True, False]),
        loss_fn="bce",
    )
    assert jnp.isfinite(wrong_boundary)
    assert wrong_boundary > 50.0


def test_lidar_loss_validation_and_custom_callable_contract():
    values = jnp.ones((3,))
    with pytest.raises(ValueError, match="Unknown loss_fn"):
        lidar_distance_loss(values, values, loss_fn="unknown")
    with pytest.raises(ValueError, match="Shapes must match"):
        lidar_distance_loss(values, values[:2])
    with pytest.raises(ValueError, match="per-element"):
        lidar_distance_loss(values, values, loss_fn=lambda x, y: jnp.mean(x - y))
    assert jnp.allclose(
        lidar_distance_loss(values, values + 1, loss_fn=lambda x, y: jnp.abs(x - y)),
        1.0,
    )


@pytest.mark.parametrize(
    ("function", "pred", "target", "expected"),
    [
        (huber_loss, [1.5, 3.0], [1.0, 1.0], [0.125, 1.5]),
        (smooth_l1_loss, [1.5, 3.0], [1.0, 1.0], [0.125, 1.5]),
        (bce_loss, [0.5], [1.0], [math.log(2.0)]),
        (bce_with_logits_loss, [0.0], [1.0], [math.log(2.0)]),
        (log_l1, [1.0], [0.0], [math.log(2.0)]),
    ],
)
def test_standard_elementwise_losses_known_values(function, pred, target, expected):
    actual = function(jnp.asarray(pred), jnp.asarray(target))
    assert jnp.allclose(actual, jnp.asarray(expected), atol=1.0e-6)
    gradient = jax.grad(lambda value: jnp.sum(function(value, jnp.asarray(target))))(
        jnp.asarray(pred)
    )
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_cross_entropy_clipped_bce_and_inverse_depth_values():
    logits = jnp.zeros((2, 3))
    assert jnp.allclose(
        cross_entropy_loss(logits, jnp.asarray([0, 2])),
        math.log(3.0),
    )
    assert jnp.allclose(
        bce_clipped(jnp.asarray([0.0]), jnp.asarray([0.0]), eps=0.1),
        -math.log(0.9),
    )
    assert jnp.allclose(
        depth_inverse_mse(jnp.asarray([2.0]), jnp.asarray([4.0])),
        0.0625,
    )


def test_normal_relu_and_weight_regularizers():
    x = jnp.asarray([[1.0, 0.0, 0.0]])
    y = jnp.asarray([[0.0, 1.0, 0.0]])
    assert jnp.allclose(normal_cosine_loss(x, y), 1.0)
    assert jnp.allclose(relu_sum(jnp.asarray([1.0, 2.0, 3.0]), 1.5), 2.0)
    weights = [jnp.asarray([[1.0, 2.0], [3.0, 4.0]])]
    assert jnp.allclose(weights_reg(weights), (5.0 + 25.0) / 2.0)


def test_identity_grid_and_temporal_variation_contracts():
    identity = jnp.eye(3, 4).reshape(1, 12)
    zero = jnp.zeros((1, 12))
    assert jnp.allclose(identity_distance(identity), 0.0)
    assert jnp.allclose(identity_distance(zero), math.sqrt(3.0))
    frames = jnp.stack((jnp.zeros((1, 1, 1, 1)), jnp.ones((1, 1, 1, 1))))
    assert jnp.allclose(total_variation_temporal(frames, jnp.ones(1)), 1.0)
    assert total_variation_temporal(frames[:1], jnp.ones(0)).shape == (1,)


def test_lambda_scheduler_and_reductions():
    scheduler = LinearLambdaScheduler(10, 20, 1.0, 0.0)
    assert scheduler(epoch=0, global_step=0) == 1.0
    assert scheduler(epoch=0, global_step=15) == 0.5
    assert scheduler(epoch=0, global_step=30) == 0.0
    with pytest.raises(ValueError, match="update_frequency"):
        LinearLambdaScheduler(0, 2, 1.0, update_frequency=0)

    values = jnp.asarray([1.0, 2.0, 3.0, 100.0])
    mask = jnp.asarray([True, False, True, False])
    assert jnp.allclose(reduce_mean(values, mask), 2.0)
    assert jnp.allclose(reduce_quantile(values, 0.75), 2.0)
    assert jnp.allclose(reduce_sum(values), 106.0)
    assert jnp.allclose(reduce_quantile(values[:1], 0.5), 0.0)
    with pytest.raises(AssertionError, match="bool or integer"):
        reduce_mean(values, mask.astype(jnp.float32))


def test_gaussian_regularizers_values_visibility_and_gradients():
    scales = jnp.asarray([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    visibility = jnp.asarray([1.0, 0.0])
    assert jnp.array_equal(
        gaussian_scale_reg(scales, visibility),
        jnp.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
    )
    assert jnp.array_equal(gaussian_density_reg(jnp.ones(2), visibility), visibility)
    assert jnp.allclose(
        gaussian_z_scale_reg(jnp.asarray([0.3, 0.7]), 0.5),
        jnp.asarray([0.0, 0.2]),
    )
    assert jnp.allclose(
        out_of_bound_loss(
            jnp.asarray([[2.0, -1.5, 0.5]]),
            jnp.asarray([[2.0, 2.0, 2.0]]),
        ),
        jnp.asarray([[1.0, 0.5, 0.0]]),
    )
    gradient = jax.grad(lambda value: jnp.sum(gaussian_scale_reg(value)))(scales)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_scalar_and_bilateral_regularizers():
    assert jnp.allclose(opacity_reg_loss(jnp.zeros(3)), 0.5)
    assert jnp.allclose(scale_reg_loss(jnp.zeros((2, 3))), 1.0)
    identity = jnp.eye(3, 4).reshape(1, 12)
    zero = jnp.zeros((1, 12))
    assert jnp.allclose(
        bilateral_grid_drift_loss([identity, zero]),
        jnp.asarray([0.0, math.sqrt(3.0)]),
    )
    assert bilateral_grid_drift_loss([]).shape == (0,)


def test_scalar_regularizers_average_only_active_gaussians():
    active_mask = jnp.asarray([True, True, False])
    opacity_logits = jnp.asarray([0.0, jnp.log(3.0), 100.0])
    log_scales = jnp.log(
        jnp.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [100.0, 100.0, 100.0],
            ]
        )
    )

    assert jnp.allclose(opacity_reg_loss(opacity_logits, mask=active_mask), 0.625)
    assert jnp.allclose(scale_reg_loss(log_scales, mask=active_mask), 3.5)


def test_optional_loss_contract_checks(monkeypatch):
    monkeypatch.setattr(losses, "ENFORCE_CONTRACTS", True)
    with pytest.raises(AssertionError, match="post-activation"):
        gaussian_scale_reg(jnp.asarray([[-1.0, 1.0, 1.0]]))
    with pytest.raises(AssertionError, match="unit-normalized"):
        normal_cosine_loss(
            jnp.asarray([[2.0, 0.0, 0.0]]), jnp.asarray([[1.0, 0.0, 0.0]])
        )
