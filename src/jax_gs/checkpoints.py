from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from .config import TrainConfig
from .model import GaussianModel
from .strategy import StrategyState


_CHECKPOINT_METADATA = "jax_gs_checkpoint.json"


def _pure_state(node: Any) -> Any:
    return nnx.as_pure(nnx.state(node))


def _encode_empty_arrays(state: Any) -> Any:
    """Encode typed keys and empty leaves for TensorStore serialization."""

    def encode(value: Any) -> Any:
        if isinstance(value, jax.Array) and jax.dtypes.issubdtype(
            value.dtype, jax.dtypes.prng_key
        ):
            return jax.random.key_data(value)
        if isinstance(value, jax.Array) and value.size == 0:
            return jnp.zeros((1,), dtype=value.dtype)
        return value

    return jax.tree.map(encode, state)


def _restore_empty_arrays(restored: Any, target: Any) -> Any:
    def decode(restored_value: Any, target_value: Any) -> Any:
        if isinstance(target_value, jax.Array) and jax.dtypes.issubdtype(
            target_value.dtype, jax.dtypes.prng_key
        ):
            return jax.random.wrap_key_data(
                restored_value, impl=jax.random.key_impl(target_value)
            )
        if isinstance(target_value, jax.Array) and target_value.size == 0:
            return target_value
        return restored_value

    return jax.tree.map(decode, restored, target)


def _validate_pose_arguments(
    pose_module: Any | None,
    pose_optimizer: nnx.Optimizer | None,
    pose_image_names: Sequence[str] | None,
) -> tuple[tuple[str, ...], int] | None:
    has_module = pose_module is not None
    has_optimizer = pose_optimizer is not None
    if has_module != has_optimizer:
        raise ValueError(
            "pose_module and pose_optimizer must be provided together"
        )
    if has_module and pose_image_names is None:
        raise ValueError(
            "pose_module, pose_optimizer, and pose_image_names must be "
            "provided together"
        )
    if pose_image_names is None:
        return None

    if isinstance(pose_image_names, str):
        raise TypeError("pose_image_names must be a sequence of image names")
    names = tuple(pose_image_names)
    if not all(isinstance(name, str) for name in names):
        raise TypeError("pose_image_names must contain only strings")
    camera_count = len(names)
    if has_module and camera_count != int(
        pose_module.embeds.embedding.shape[0]
    ):
        raise ValueError(
            f"pose_image_names has {len(names)} entries, but pose_module has "
            f"{pose_module.embeds.embedding.shape[0]} cameras"
        )
    return names, camera_count


def _validate_appearance_arguments(
    appearance_module: Any | None,
    appearance_optimizer: nnx.Optimizer | None,
    appearance_image_names: Sequence[str] | None,
) -> tuple[tuple[str, ...], int] | None:
    provided = (
        appearance_module is not None,
        appearance_optimizer is not None,
        appearance_image_names is not None,
    )
    if any(provided) and not all(provided):
        raise ValueError(
            "appearance_module, appearance_optimizer, and "
            "appearance_image_names must be provided together"
        )
    if not any(provided):
        return None
    if isinstance(appearance_image_names, str):
        raise TypeError(
            "appearance_image_names must be a sequence of image names"
        )
    assert appearance_image_names is not None
    names = tuple(appearance_image_names)
    if not all(isinstance(name, str) for name in names):
        raise TypeError("appearance_image_names must contain only strings")
    camera_count = int(appearance_module.embeds.embedding.shape[0])
    if len(names) != camera_count:
        raise ValueError(
            f"appearance_image_names has {len(names)} entries, but "
            f"appearance_module has {camera_count} cameras"
        )
    return names, camera_count


def _load_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    metadata_path = Path(checkpoint_path) / _CHECKPOINT_METADATA
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def save_checkpoint(
    directory: str | Path,
    model: GaussianModel,
    *,
    step: int,
    optimizer: nnx.Optimizer | None = None,
    strategy_state: StrategyState | None = None,
    config: TrainConfig | None = None,
    intersection_capacity: int | None = None,
    pose_module: Any | None = None,
    pose_optimizer: nnx.Optimizer | None = None,
    pose_image_names: Sequence[str] | None = None,
    appearance_module: Any | None = None,
    appearance_optimizer: nnx.Optimizer | None = None,
    appearance_image_names: Sequence[str] | None = None,
    force: bool = True,
) -> Path:
    """Save model and optional training state with Orbax."""

    if intersection_capacity is not None:
        intersection_capacity = int(intersection_capacity)
        if intersection_capacity <= 0:
            raise ValueError("intersection_capacity must be positive")
    pose = _validate_pose_arguments(
        pose_module, pose_optimizer, pose_image_names
    )
    appearance = _validate_appearance_arguments(
        appearance_module,
        appearance_optimizer,
        appearance_image_names,
    )
    directory = Path(directory).absolute()
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": _encode_empty_arrays(_pure_state(model)),
        "step": jnp.asarray(step, dtype=jnp.int32),
    }
    if optimizer is not None:
        payload["optimizer"] = _encode_empty_arrays(_pure_state(optimizer))
    if strategy_state is not None:
        payload["strategy"] = _encode_empty_arrays(_pure_state(strategy_state))
    if pose_module is not None:
        payload["pose"] = {
            "module": _encode_empty_arrays(_pure_state(pose_module)),
            "optimizer": _encode_empty_arrays(_pure_state(pose_optimizer)),
        }
    if appearance_module is not None:
        payload["appearance"] = {
            "module": _encode_empty_arrays(_pure_state(appearance_module)),
            "optimizer": _encode_empty_arrays(
                _pure_state(appearance_optimizer)
            ),
        }
    checkpoint_path = directory / f"step_{step:08d}"
    checkpointer = ocp.StandardCheckpointer()
    try:
        checkpointer.save(checkpoint_path, payload, force=force)
        if hasattr(checkpointer, "wait_until_finished"):
            checkpointer.wait_until_finished()
    finally:
        checkpointer.close()
    if config is not None:
        (checkpoint_path / "jax_gs_config.json").write_text(
            json.dumps(config.to_dict(), indent=2), encoding="utf-8"
        )
    active_count = int(jax.device_get(model.active_count))
    active_prefix = bool(
        jax.device_get(
            jnp.all(
                model.active_mask[...]
                == (jnp.arange(model.capacity) < active_count)
            )
        )
    )
    components = ["model"]
    if optimizer is not None:
        components.append("optimizer")
    if strategy_state is not None:
        components.append("strategy")
    if pose_module is not None:
        components.append("pose")
    if appearance_module is not None:
        components.append("appearance")
    metadata: dict[str, Any] = {
        "format_version": 5,
        "components": components,
        "model_color_mode": (
            "appearance" if model.has_appearance else "sh"
        ),
        "storage_capacity": model.capacity,
        "max_capacity": model.max_capacity,
        "active_count": active_count,
        "active_prefix": active_prefix,
        "intersection_capacity": intersection_capacity,
    }
    if pose is not None:
        names, camera_count = pose
        metadata["pose_camera_count"] = camera_count
        metadata["pose_image_names"] = list(names)
    if model.has_appearance:
        metadata["appearance_feature_dim"] = int(model.features.shape[1])
    if appearance is not None:
        names, camera_count = appearance
        metadata["appearance_camera_count"] = camera_count
        metadata["appearance_image_names"] = list(names)
    (checkpoint_path / _CHECKPOINT_METADATA).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return checkpoint_path


def restore_checkpoint(
    checkpoint_path: str | Path,
    model: GaussianModel,
    *,
    optimizer: nnx.Optimizer | None = None,
    strategy_state: StrategyState | None = None,
    pose_module: Any | None = None,
    pose_optimizer: nnx.Optimizer | None = None,
    pose_image_names: Sequence[str] | None = None,
    appearance_module: Any | None = None,
    appearance_optimizer: nnx.Optimizer | None = None,
    appearance_image_names: Sequence[str] | None = None,
) -> int:
    """Restore into NNX objects with matching physical storage capacity."""

    pose = _validate_pose_arguments(
        pose_module, pose_optimizer, pose_image_names
    )
    appearance = _validate_appearance_arguments(
        appearance_module,
        appearance_optimizer,
        appearance_image_names,
    )
    metadata = _load_metadata(checkpoint_path)
    format_version = int(metadata.get("format_version", 1))
    components = metadata.get("components", ())
    has_pose_state = format_version >= 4 and "pose" in components
    wants_pose_state = pose_module is not None
    has_appearance_state = (
        format_version >= 5 and "appearance" in components
    )
    wants_appearance_state = appearance_module is not None
    if wants_pose_state and not has_pose_state:
        raise ValueError("checkpoint does not contain pose state")
    if wants_appearance_state and not has_appearance_state:
        raise ValueError("checkpoint does not contain appearance state")
    saved_color_mode = metadata.get("model_color_mode", "sh")
    target_color_mode = "appearance" if model.has_appearance else "sh"
    if saved_color_mode != target_color_mode:
        raise ValueError(
            f"checkpoint model color mode is {saved_color_mode!r}, target is "
            f"{target_color_mode!r}"
        )
    if pose is not None:
        names, camera_count = pose
        saved_count = metadata.get("pose_camera_count")
        saved_names = metadata.get("pose_image_names")
        if saved_count is None or saved_names is None:
            raise ValueError("checkpoint pose metadata is incomplete")
        if int(saved_count) != camera_count:
            raise ValueError(
                f"checkpoint pose camera count is {saved_count}, target has "
                f"{camera_count}"
            )
        if tuple(saved_names) != names:
            raise ValueError("checkpoint pose image names do not match target")
    if appearance is not None:
        names, camera_count = appearance
        saved_count = metadata.get("appearance_camera_count")
        saved_names = metadata.get("appearance_image_names")
        if saved_count is None or saved_names is None:
            raise ValueError("checkpoint appearance metadata is incomplete")
        if int(saved_count) != camera_count:
            raise ValueError(
                f"checkpoint appearance camera count is {saved_count}, target "
                f"has {camera_count}"
            )
        if tuple(saved_names) != names:
            raise ValueError(
                "checkpoint appearance image names do not match target"
            )
        saved_feature_dim = metadata.get("appearance_feature_dim")
        if saved_feature_dim is None:
            raise ValueError("checkpoint appearance feature metadata is incomplete")
        if int(saved_feature_dim) != int(appearance_module.feature_dim):
            raise ValueError(
                f"checkpoint appearance feature dimension is "
                f"{saved_feature_dim}, target has {appearance_module.feature_dim}"
            )

    saved_capacity = load_checkpoint_storage_capacity(checkpoint_path)
    if model.capacity != saved_capacity:
        raise ValueError(
            f"checkpoint storage capacity is {saved_capacity}, target model has "
            f"capacity {model.capacity}"
        )

    model_target = _pure_state(model)
    target: dict[str, Any] = {
        "model": _encode_empty_arrays(model_target),
        "step": jnp.asarray(0, dtype=jnp.int32),
    }
    optimizer_target = None
    strategy_target = None
    pose_module_target = None
    pose_optimizer_target = None
    appearance_module_target = None
    appearance_optimizer_target = None
    if optimizer is not None:
        optimizer_target = _pure_state(optimizer)
        target["optimizer"] = _encode_empty_arrays(optimizer_target)
    if strategy_state is not None:
        strategy_target = _pure_state(strategy_state)
        target["strategy"] = _encode_empty_arrays(strategy_target)
    if wants_pose_state:
        pose_module_target = _pure_state(pose_module)
        pose_optimizer_target = _pure_state(pose_optimizer)
        target["pose"] = {
            "module": _encode_empty_arrays(pose_module_target),
            "optimizer": _encode_empty_arrays(pose_optimizer_target),
        }
    if wants_appearance_state:
        appearance_module_target = _pure_state(appearance_module)
        appearance_optimizer_target = _pure_state(appearance_optimizer)
        target["appearance"] = {
            "module": _encode_empty_arrays(appearance_module_target),
            "optimizer": _encode_empty_arrays(appearance_optimizer_target),
        }
    partial_restore = (
        optimizer is None
        or strategy_state is None
        or (has_pose_state and not wants_pose_state)
        or (has_appearance_state and not wants_appearance_state)
    )
    checkpointer = (
        ocp.PyTreeCheckpointer() if partial_restore else ocp.StandardCheckpointer()
    )
    try:
        if partial_restore:
            restored = checkpointer.restore(
                Path(checkpoint_path).absolute(),
                item=target,
                partial_restore=True,
            )
        else:
            restored = checkpointer.restore(
                Path(checkpoint_path).absolute(), target=target
            )
        if hasattr(checkpointer, "wait_until_finished"):
            checkpointer.wait_until_finished()
    finally:
        checkpointer.close()
    nnx.update(
        model, _restore_empty_arrays(restored["model"], model_target)
    )
    if optimizer is not None:
        nnx.update(
            optimizer,
            _restore_empty_arrays(restored["optimizer"], optimizer_target),
        )
    if strategy_state is not None:
        nnx.update(
            strategy_state,
            _restore_empty_arrays(restored["strategy"], strategy_target),
        )
    if wants_pose_state:
        nnx.update(
            pose_module,
            _restore_empty_arrays(
                restored["pose"]["module"], pose_module_target
            ),
        )
        nnx.update(
            pose_optimizer,
            _restore_empty_arrays(
                restored["pose"]["optimizer"], pose_optimizer_target
            ),
        )
    if wants_appearance_state:
        nnx.update(
            appearance_module,
            _restore_empty_arrays(
                restored["appearance"]["module"], appearance_module_target
            ),
        )
        nnx.update(
            appearance_optimizer,
            _restore_empty_arrays(
                restored["appearance"]["optimizer"],
                appearance_optimizer_target,
            ),
        )
    return int(restored["step"])


def load_checkpoint_config(checkpoint_path: str | Path) -> TrainConfig:
    path = Path(checkpoint_path) / "jax_gs_config.json"
    return TrainConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_checkpoint_storage_capacity(checkpoint_path: str | Path) -> int:
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path / _CHECKPOINT_METADATA
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        capacity = int(metadata["storage_capacity"])
    else:
        # Version-1 checkpoints used ModelConfig.capacity as the physical shape.
        capacity = load_checkpoint_config(checkpoint_path).model.capacity
    if capacity <= 0:
        raise ValueError("checkpoint storage capacity must be positive")
    return capacity


def load_checkpoint_appearance_image_names(
    checkpoint_path: str | Path,
) -> tuple[str, ...] | None:
    """Return the ordered training-image manifest for appearance state."""

    metadata = _load_metadata(checkpoint_path)
    if "appearance" not in metadata.get("components", ()):
        return None
    names = metadata.get("appearance_image_names")
    count = metadata.get("appearance_camera_count")
    if names is None or count is None:
        raise ValueError("checkpoint appearance metadata is incomplete")
    if not isinstance(names, list) or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError("checkpoint appearance image names are invalid")
    if len(names) != int(count):
        raise ValueError(
            "checkpoint appearance camera count does not match image names"
        )
    return tuple(names)


def load_checkpoint_intersection_capacity(
    checkpoint_path: str | Path,
) -> int | None:
    """Return the learned training intersection high-water mark, if present."""

    metadata_path = Path(checkpoint_path) / _CHECKPOINT_METADATA
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    value = metadata.get("intersection_capacity")
    if value is None:
        return None
    capacity = int(value)
    if capacity <= 0:
        raise ValueError("checkpoint intersection capacity must be positive")
    return capacity


def load_checkpoint_active_prefix(checkpoint_path: str | Path) -> bool:
    """Return whether checkpoint metadata guarantees a compact active prefix."""

    metadata_path = Path(checkpoint_path) / _CHECKPOINT_METADATA
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return bool(metadata.get("active_prefix", False))
