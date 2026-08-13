"""Building the optimizers, and validating the configuration they come from."""

from __future__ import annotations

import sys

# Tests drive the trainer by patching seams on the package, for example
# monkeypatch.setattr(jax_gs.training, "rasterization", fake). Calls resolve
# through the package namespace at run time so those seams keep working now
# that the implementation lives in submodules.
_training = sys.modules[__package__]

import math
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from ..config import TrainConfig
from ..model import GaussianModel
from ..optimizers import (
    create_optimizer,
)
from .pose import CameraOptModule


def _validate_2dgs_mode(config: TrainConfig) -> None:
    if config.model_type != "2dgs":
        if config.normal_loss or config.dist_loss:
            raise ValueError(
                "normal_loss and dist_loss are available only for 2DGS"
            )
        return
    if config.camera_model != "pinhole":
        raise ValueError("2DGS supports only camera_model='pinhole'")
    if config.with_ut:
        raise ValueError("2DGS does not support with_ut=True")
    if config.with_eval3d:
        raise ValueError("2DGS does not support with_eval3d=True")
    if config.rasterizer.rasterize_mode != "classic":
        raise ValueError("2DGS supports only rasterize_mode='classic'")
    if config.strategy.kind != "default":
        raise ValueError("2DGS supports only the Default strategy")
    if config.packed and config.rasterizer.backend == "reference":
        raise ValueError(
            "2DGS packed training requires projection metadata from the "
            "intersection renderer"
        )


def _create_training_optimizer(
    model: GaussianModel,
    config: TrainConfig,
    *,
    world_size: int = 1,
    scene_scale: float = 1.0,
) -> nnx.Optimizer:
    optimizer_kwargs = {
        "batch_size": config.data.batch_size,
        "world_size": world_size,
        "scene_scale": scene_scale,
    }
    if config.sparse_grad:
        return _training.create_row_selective_optimizer(
            model, config.optimizer, **optimizer_kwargs
        )
    if config.visible_adam:
        return _training.create_visible_adam_optimizer(
            model, config.optimizer, **optimizer_kwargs
        )
    return create_optimizer(model, config.optimizer, **optimizer_kwargs)


def _pose_learning_rate(
    config: TrainConfig, step: int | jax.Array
) -> jax.Array:
    """Current-main pose LR: batch-scaled and exponentially decayed to 1%."""

    step = jnp.asarray(step, dtype=jnp.float32)
    initial = config.pose_opt_lr * math.sqrt(config.data.batch_size)
    progress = step / float(max(config.steps, 1))
    return jnp.asarray(initial, dtype=jnp.float32) * jnp.power(0.01, progress)


def _create_pose_optimizer(
    pose_adjust: CameraOptModule, config: TrainConfig
) -> nnx.Optimizer:
    """Build PyTorch-Adam-equivalent pose optimization with coupled L2 decay."""

    transform = optax.chain(
        optax.add_decayed_weights(config.pose_opt_reg),
        optax.adam(
            lambda count: _pose_learning_rate(config, count),
            eps=1.0e-8,
        ),
    )
    optimizer = nnx.Optimizer(pose_adjust, transform, wrt=nnx.Param)
    optimizer._jax_gs_pose_contract = (
        config.data.batch_size,
        config.steps,
        float(config.pose_opt_lr),
        float(config.pose_opt_reg),
    )
    return optimizer


def _validate_camera_module_resume_config(
    config: TrainConfig, checkpoint_path: str | Path
) -> None:
    """Reject resume changes that alter restored optimizer meaning."""

    saved = _training.load_checkpoint_config(checkpoint_path)
    for field in ("normalize_world_space", "global_scale"):
        saved_value = getattr(saved, field)
        current_value = getattr(config, field)
        if saved_value != current_value:
            raise ValueError(
                f"resume requires {field} to match the checkpoint config "
                f"({saved_value} saved, {current_value} requested)"
            )
    if saved.data.batch_size != config.data.batch_size:
        raise ValueError(
            "resume requires data.batch_size to match the checkpoint because "
            "it scales Gaussian Adam "
            f"({saved.data.batch_size} saved, "
            f"{config.data.batch_size} requested)"
        )
    if (
        saved.strategy.target_primitives
        != config.strategy.target_primitives
    ):
        raise ValueError(
            "resume requires target_primitives to match the checkpoint config "
            f"({saved.strategy.target_primitives} saved, "
            f"{config.strategy.target_primitives} requested)"
        )
    if saved.pose_opt != config.pose_opt:
        raise ValueError(
            "resume requires pose_opt to match the checkpoint config "
            f"({saved.pose_opt} saved, {config.pose_opt} requested)"
        )
    if saved.pose_noise != config.pose_noise:
        raise ValueError(
            "resume requires pose_noise to match the checkpoint config "
            f"({saved.pose_noise} saved, {config.pose_noise} requested)"
        )
    if config.pose_opt and saved.steps != config.steps:
        raise ValueError(
            "resume with pose_opt requires steps to match the checkpoint LR "
            f"horizon ({saved.steps} saved, {config.steps} requested)"
        )
    if config.pose_noise > 0.0 and saved.seed != config.seed:
        raise ValueError(
            "resume with pose_noise requires seed to match the checkpoint "
            f"config ({saved.seed} saved, {config.seed} requested)"
        )
    if saved.app_opt != config.app_opt:
        raise ValueError(
            "resume requires app_opt to match the checkpoint config "
            f"({saved.app_opt} saved, {config.app_opt} requested)"
        )
    if config.app_opt:
        appearance_fields = (
            "app_embed_dim",
            "app_opt_lr",
            "app_opt_reg",
        )
        for field in appearance_fields:
            saved_value = getattr(saved, field)
            current_value = getattr(config, field)
            if saved_value != current_value:
                raise ValueError(
                    f"resume requires {field} to match the checkpoint config "
                    f"({saved_value} saved, {current_value} requested)"
                )
        if saved.model.sh_degree != config.model.sh_degree:
            raise ValueError(
                "resume with app_opt requires model.sh_degree to match the "
                f"checkpoint ({saved.model.sh_degree} saved, "
                f"{config.model.sh_degree} requested)"
            )


def _invert_rigid_transforms(transforms: jax.Array) -> jax.Array:
    """Invert row-major homogeneous rigid transforms without a generic solve."""

    transforms = jnp.asarray(transforms)
    if transforms.ndim < 2 or transforms.shape[-2:] != (4, 4):
        raise ValueError(
            "transforms must have shape (..., 4, 4), "
            f"got {transforms.shape}"
        )
    rotation = transforms[..., :3, :3]
    translation = transforms[..., :3, 3]
    inverse_rotation = jnp.swapaxes(rotation, -1, -2)
    inverse_translation = -jnp.einsum(
        "...ij,...j->...i", inverse_rotation, translation
    )
    result = jnp.broadcast_to(
        jnp.eye(4, dtype=transforms.dtype), transforms.shape
    )
    result = result.at[..., :3, :3].set(inverse_rotation)
    return result.at[..., :3, 3].set(inverse_translation)


def _apply_camera_pose_modules(
    camtoworlds: jax.Array,
    image_ids: jax.Array,
    *,
    pose_adjust: CameraOptModule | None,
    pose_perturb: CameraOptModule | None,
) -> jax.Array:
    """Apply fixed noise followed by trainable local camera-pose deltas."""

    adjusted = jnp.asarray(camtoworlds)
    if pose_perturb is not None:
        adjusted = jax.lax.stop_gradient(pose_perturb(adjusted, image_ids))
    if pose_adjust is not None:
        adjusted = pose_adjust(adjusted, image_ids)
    return adjusted
