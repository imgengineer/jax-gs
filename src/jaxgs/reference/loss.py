import chex
import jax
import jax.numpy as jnp


def l1_loss(prediction: chex.Array, target: chex.Array) -> chex.Array:
    return jnp.mean(jnp.abs(prediction - target))


def ssim(prediction: chex.Array, target: chex.Array) -> chex.Array:
    """11x11 Gaussian-window SSIM with SAME padding, matching LiteGS loss."""
    x = jnp.arange(11, dtype=jnp.float32) - 5
    gaussian = jnp.exp(-0.5 * (x / 1.5) ** 2)
    gaussian = gaussian / jnp.sum(gaussian)
    kernel = (gaussian[:, None] * gaussian[None, :])[:, :, None, None]
    kernel = jnp.broadcast_to(kernel, (11, 11, 1, 3))

    def blur(image):
        return jax.lax.conv_general_dilated(
            image[None],
            kernel,
            (1, 1),
            "SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=3,
        )[0]

    mu_x, mu_y = blur(prediction), blur(target)
    sigma_x = blur(prediction**2) - mu_x**2
    sigma_y = blur(target**2) - mu_y**2
    sigma_xy = blur(prediction * target) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return jnp.clip(jnp.mean(numerator / denominator), -1.0, 1.0)


def photometric_loss(
    prediction: chex.Array, target: chex.Array, ssim_weight: float = 0.2
) -> chex.Array:
    return (1 - ssim_weight) * l1_loss(prediction, target) + ssim_weight * (
        1 - ssim(prediction, target)
    )
