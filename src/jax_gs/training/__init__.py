"""Single-process trainer.

The implementation lives in the private submodules imported at the bottom;
this module is their public surface and every name they previously exported
still resolves here.

The imports below are not only for this module's own use. The submodules call
back through this namespace for the names a caller may substitute, so that
``monkeypatch.setattr(jax_gs.training, "rasterization", fake)`` still reaches
the code inside ``_step`` and ``_loop``. Removing an import that looks unused
here therefore breaks the substitution rather than merely tidying the file;
``_step`` and ``_loop`` name the specific seams they resolve this way.
"""

from __future__ import annotations

# Imported for its own sake: a test substitutes jax.process_count through
# this module, and the submodules import the same module object.
import jax  # noqa: F401

from ..capacity import compact_training_state, resize_training_state
from ..checkpoints import (
    load_checkpoint_active_prefix,
    load_checkpoint_config,
    load_checkpoint_intersection_capacity,
    load_checkpoint_scene_transform,
    load_checkpoint_storage_capacity,
    restore_checkpoint,
    save_checkpoint,
)
from ..config import RasterizationConfig, TrainConfig
from ..data import ColmapScene, create_grain_dataset, load_colmap_scene
from ..data.normalize import (
    _as_similarity_matrix,
    normalize_scene,
    transform_cameras,
    transform_points,
)
from ..losses import l1_loss, opacity_reg_loss, psnr, scale_reg_loss, ssim
from ..model import GaussianModel
from ..optimizers import (
    create_optimizer,
    create_row_selective_optimizer,
    create_visible_adam_optimizer,
    mask_inactive_gradients,
)
from ..rasterization import _automatic_intersection_capacity, rasterization
from ..strategy import (
    DefaultStrategy,
    DensificationStats,
    MCMCStrategy,
    StrategyState,
    build_densification_stats,
    reset_opacities,
)
from ..strategy.ops import mcmc_position_perturbation
from ..two_dgs import rasterization_2dgs
from ._data import (
    _grain_iter_dataset,
    _infinite_batches,
    _sample_patches,
)
from ._loop import (
    TrainingResult,
    _save_render,
    make_render_step,
    train,
)
from ._memory import (
    _check_bucket_transition_memory_budget,
    _check_distributed_bucket_transition_memory_budget,
    _check_evaluation_memory_budget,
    _check_memory_budget,
    _device_memory_usage,
    _intersection_bucket_capacity,
    _mcmc_required_capacity,
    _parameter_bytes,
    _pending_overflow_suffix,
    _raise_training_overflow,
    _training_config_with_intersection_capacity,
    _training_intersection_limit,
    _training_overflow_status,
    _training_state_bytes,
    estimate_bucket_transition_memory_bytes,
    estimate_rasterization_memory_bytes,
    estimate_training_memory_bytes,
)
from ._scene import (
    SceneTransform,
    _legacy_scene_transform,
    _legacy_training_scene_scale,
    _scene_training_render_size,
    _training_render_size,
    _training_scene_scale,
    compute_scene_transform,
)
from ._setup import (
    _apply_camera_pose_modules,
    _create_pose_optimizer,
    _create_training_optimizer,
    _invert_rigid_transforms,
    _pose_learning_rate,
    _validate_2dgs_mode,
    _validate_camera_module_resume_config,
)
from ._state import (
    DistributedCapacityDecision,
    _block_nnx_state,
    _grow_training_state,
    _initial_storage_capacity,
    _save_compacted_training_checkpoint,
    synchronize_distributed_capacity,
)
from ._step import (
    TrainingSafetyState,
    _make_train_step,
    _PendingTrainStep,
    _two_dgs_regularization_losses,
    _unpack_training_projection_metadata,
    make_distributed_train_step,
    make_train_step,
)
from .appearance import (
    APPEARANCE_FEATURE_DIM,
    AppearanceOptModule,
    create_appearance_optimizer,
)
from .pose import CameraOptModule
from .schedulers import TwoStageScheduler

__all__ = [
    "APPEARANCE_FEATURE_DIM",
    "AppearanceOptModule",
    "CameraOptModule",
    "ColmapScene",
    "DefaultStrategy",
    "DensificationStats",
    "DistributedCapacityDecision",
    "GaussianModel",
    "MCMCStrategy",
    "RasterizationConfig",
    "SceneTransform",
    "StrategyState",
    "TrainConfig",
    "TrainingResult",
    "TrainingSafetyState",
    "TwoStageScheduler",
    "_PendingTrainStep",
    "_apply_camera_pose_modules",
    "_as_similarity_matrix",
    "_automatic_intersection_capacity",
    "_block_nnx_state",
    "_check_bucket_transition_memory_budget",
    "_check_distributed_bucket_transition_memory_budget",
    "_check_evaluation_memory_budget",
    "_check_memory_budget",
    "_create_pose_optimizer",
    "_create_training_optimizer",
    "_device_memory_usage",
    "_grain_iter_dataset",
    "_grow_training_state",
    "_infinite_batches",
    "_initial_storage_capacity",
    "_intersection_bucket_capacity",
    "_invert_rigid_transforms",
    "_legacy_scene_transform",
    "_legacy_training_scene_scale",
    "_make_train_step",
    "_mcmc_required_capacity",
    "_parameter_bytes",
    "_pending_overflow_suffix",
    "_pose_learning_rate",
    "_raise_training_overflow",
    "_sample_patches",
    "_save_compacted_training_checkpoint",
    "_save_render",
    "_scene_training_render_size",
    "_training_config_with_intersection_capacity",
    "_training_intersection_limit",
    "_training_overflow_status",
    "_training_render_size",
    "_training_scene_scale",
    "_training_state_bytes",
    "_two_dgs_regularization_losses",
    "_unpack_training_projection_metadata",
    "_validate_2dgs_mode",
    "_validate_camera_module_resume_config",
    "build_densification_stats",
    "compact_training_state",
    "compute_scene_transform",
    "create_appearance_optimizer",
    "create_grain_dataset",
    "create_optimizer",
    "create_row_selective_optimizer",
    "create_visible_adam_optimizer",
    "estimate_bucket_transition_memory_bytes",
    "estimate_rasterization_memory_bytes",
    "estimate_training_memory_bytes",
    "l1_loss",
    "load_checkpoint_active_prefix",
    "load_checkpoint_config",
    "load_checkpoint_intersection_capacity",
    "load_checkpoint_scene_transform",
    "load_checkpoint_storage_capacity",
    "load_colmap_scene",
    "make_distributed_train_step",
    "make_render_step",
    "make_train_step",
    "mask_inactive_gradients",
    "mcmc_position_perturbation",
    "normalize_scene",
    "opacity_reg_loss",
    "psnr",
    "rasterization",
    "rasterization_2dgs",
    "reset_opacities",
    "resize_training_state",
    "restore_checkpoint",
    "save_checkpoint",
    "scale_reg_loss",
    "ssim",
    "synchronize_distributed_capacity",
    "train",
    "transform_cameras",
    "transform_points",
]
