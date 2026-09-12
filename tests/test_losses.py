import jax
import jax.numpy as jnp
import numpy as np

from jax_gs.losses import (
    depth_normal_loss,
    depth_to_normals,
    depth_to_points,
    l1_loss,
    mse_loss,
    normal_loss,
    psnr,
    ssim,
    total_variation_loss,
)


def test_pixel_losses_known_values() -> None:
    prediction = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    target = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    np.testing.assert_allclose(
        l1_loss(prediction, target),
        [0.0, 1.0, 2.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        mse_loss(prediction, target),
        [0.0, 1.0, 4.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(psnr(jnp.zeros(3), jnp.ones(3)), 0.0, atol=1e-6)


def test_ssim_identical_images_and_finite_gradient() -> None:
    image = jnp.linspace(0.0, 1.0, 8 * 9 * 3, dtype=jnp.float32).reshape(8, 9, 3)
    np.testing.assert_allclose(ssim(image, image), 1.0, rtol=1e-5, atol=1e-5)
    gradient = jax.jit(jax.grad(lambda value: ssim(value, image * 0.9)))(image)
    assert gradient.shape == image.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_total_variation_matches_gsplat_normalization() -> None:
    grid = jnp.array([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=jnp.float32)
    np.testing.assert_allclose(total_variation_loss(grid), 5.0, atol=1e-6)
    assert jnp.all(jnp.isfinite(jax.grad(total_variation_loss)(grid)))


def test_depth_points_and_flat_depth_normals_known_values() -> None:
    depths = jnp.ones((3, 3, 1), dtype=jnp.float32) * 2.0
    camtoworld = jnp.eye(4, dtype=jnp.float32)
    intrinsics = jnp.eye(3, dtype=jnp.float32)
    points = depth_to_points(depths, camtoworld, intrinsics)
    np.testing.assert_allclose(points[0, 0], [1.0, 1.0, 2.0], atol=1e-6)
    normals = depth_to_normals(depths, camtoworld, intrinsics)
    np.testing.assert_allclose(normals[1, 1], [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_array_equal(normals[0], 0.0)


def test_normal_and_depth_normal_losses_have_finite_gradients() -> None:
    normal = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
    np.testing.assert_allclose(normal_loss(normal, normal), 0.0, atol=1e-6)
    np.testing.assert_allclose(normal_loss(normal, -normal), 2.0, atol=1e-6)

    depths = jnp.ones((4, 4, 1), dtype=jnp.float32) * 2.0
    rendered = jnp.broadcast_to(normal, (4, 4, 3))
    camtoworld = jnp.eye(4, dtype=jnp.float32)
    intrinsics = jnp.eye(3, dtype=jnp.float32)
    objective = lambda value: depth_normal_loss(value, rendered, camtoworld, intrinsics)
    gradient = jax.jit(jax.grad(objective))(depths)
    assert jnp.all(jnp.isfinite(gradient))
