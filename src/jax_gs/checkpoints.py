from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from .capacity import (
    _distributed_local_capacity,
    _distributed_world_size,
    _stack_graphs,
    _unstack_graph,
    reshard_distributed_training_state,
)
from .config import OptimizerConfig, TrainConfig
from .data.normalize import _as_similarity_matrix
from .model import GaussianModel
from .strategy import StrategyState


_CHECKPOINT_METADATA = "jax_gs_checkpoint.json"
_DISTRIBUTED_KIND = "distributed"


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


def _validate_replicated_state(
    node: Any, world_size: int, label: str
) -> None:
    for value in jax.tree.leaves(_pure_state(node)):
        if not isinstance(value, jax.Array) or value.shape[0] != world_size:
            raise ValueError(
                f"distributed {label} must have a leading world axis"
            )
        comparable = (
            jax.random.key_data(value)
            if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
            else value
        )
        if not bool(
            jax.device_get(jnp.all(comparable == comparable[0]))
        ):
            raise ValueError(
                f"distributed {label} replicas disagree across ranks"
            )


def _validate_distributed_pose_arguments(
    pose_module: Any | None,
    pose_optimizer: nnx.Optimizer | None,
    pose_image_names: Sequence[str] | None,
    *,
    world_size: int,
    require_equal_replicas: bool,
) -> tuple[tuple[str, ...], int] | None:
    has_module = pose_module is not None
    if has_module != (pose_optimizer is not None):
        raise ValueError(
            "pose_module and pose_optimizer must be provided together"
        )
    if not has_module:
        return _validate_pose_arguments(None, None, pose_image_names)

    pose_world_size = _distributed_world_size(
        pose_module, pose_optimizer
    )
    if pose_world_size != world_size:
        raise ValueError(
            "distributed pose state must match the Gaussian world size "
            f"{world_size}; got {pose_world_size}"
        )
    pose = _validate_pose_arguments(
        _unstack_graph(pose_module, 0),
        _unstack_graph(pose_optimizer, 0),
        pose_image_names,
    )
    if require_equal_replicas:
        _validate_replicated_state(pose_module, world_size, "pose module")
        _validate_replicated_state(
            pose_optimizer, world_size, "pose optimizer"
        )
    return pose


def _validate_distributed_appearance_arguments(
    appearance_module: Any | None,
    appearance_optimizer: nnx.Optimizer | None,
    appearance_image_names: Sequence[str] | None,
    *,
    world_size: int,
    require_equal_replicas: bool,
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

    appearance_world_size = _distributed_world_size(
        appearance_module, appearance_optimizer
    )
    if appearance_world_size != world_size:
        raise ValueError(
            "distributed appearance state must match the Gaussian world "
            f"size {world_size}; got {appearance_world_size}"
        )
    appearance = _validate_appearance_arguments(
        _unstack_graph(appearance_module, 0),
        _unstack_graph(appearance_optimizer, 0),
        appearance_image_names,
    )
    if require_equal_replicas:
        _validate_replicated_state(
            appearance_module, world_size, "appearance module"
        )
        _validate_replicated_state(
            appearance_optimizer, world_size, "appearance optimizer"
        )
    return appearance


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


def _validate_scene_values(
    scene_transform: Any, scene_scale: Any
) -> tuple[np.ndarray, float]:
    try:
        matrix = _as_similarity_matrix(jax.device_get(scene_transform))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scene_transform is invalid: {exc}") from exc

    try:
        scale_array = np.asarray(jax.device_get(scene_scale))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "scene_scale must be a finite non-negative scalar"
        ) from exc
    if scale_array.shape != () or not np.issubdtype(
        scale_array.dtype, np.number
    ) or np.issubdtype(scale_array.dtype, np.complexfloating):
        raise ValueError("scene_scale must be a finite non-negative scalar")
    scale = float(scale_array)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scene_scale must be a finite non-negative scalar")
    return matrix, scale


def _config_fingerprint(config: TrainConfig) -> str:
    payload = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _optimizer_kind_for_config(config: TrainConfig) -> str:
    if config.sparse_grad:
        return "row_selective_adam"
    if config.visible_adam:
        return "visible_adam"
    return "adam"


def _distributed_optimizer_contract(
    optimizer: nnx.Optimizer,
) -> dict[str, Any]:
    batch_size = getattr(optimizer, "_jax_gs_batch_size", None)
    world_size = getattr(optimizer, "_jax_gs_world_size", None)
    scene_scale = getattr(optimizer, "_jax_gs_scene_scale", None)
    optimizer_config = getattr(
        optimizer, "_jax_gs_optimizer_config", None
    )
    kind = getattr(optimizer, "_jax_gs_optimizer_kind", None)
    if (
        not isinstance(batch_size, int)
        or batch_size <= 0
        or not isinstance(world_size, int)
        or world_size <= 0
        or not isinstance(scene_scale, (int, float))
        or not np.isfinite(scene_scale)
        or scene_scale < 0.0
        or not isinstance(optimizer_config, OptimizerConfig)
        or kind not in {"adam", "row_selective_adam", "visible_adam"}
    ):
        raise ValueError(
            "distributed optimizer does not record a valid static contract"
        )
    return {
        "batch_size": batch_size,
        "world_size": world_size,
        "scene_scale": float(scene_scale),
        "kind": kind,
        "config": asdict(optimizer_config),
    }


def _distributed_pose_optimizer_contract(
    optimizer: nnx.Optimizer,
) -> dict[str, Any]:
    contract = getattr(optimizer, "_jax_gs_pose_contract", None)
    if (
        not isinstance(contract, tuple)
        or len(contract) != 4
        or not isinstance(contract[0], int)
        or contract[0] <= 0
        or not isinstance(contract[1], int)
        or contract[1] < 0
        or not all(isinstance(value, float) for value in contract[2:])
        or not all(np.isfinite(value) for value in contract[2:])
        or any(value < 0.0 for value in contract[2:])
    ):
        raise ValueError(
            "distributed pose optimizer does not record a valid static "
            "contract"
        )
    return {
        "batch_size": contract[0],
        "steps": contract[1],
        "learning_rate": contract[2],
        "regularization": contract[3],
    }


def _distributed_appearance_optimizer_contract(
    optimizer: nnx.Optimizer,
) -> dict[str, Any]:
    contract = getattr(optimizer, "_jax_gs_appearance_contract", None)
    if (
        not isinstance(contract, tuple)
        or len(contract) != 4
        or contract[0] != "appearance_multi_adam_v1"
        or not isinstance(contract[1], int)
        or contract[1] <= 0
        or not all(isinstance(value, float) for value in contract[2:])
        or not all(np.isfinite(value) for value in contract[2:])
        or any(value < 0.0 for value in contract[2:])
    ):
        raise ValueError(
            "distributed appearance optimizer does not record a valid "
            "static contract"
        )
    return {
        "kind": contract[0],
        "batch_size": contract[1],
        "learning_rate": contract[2],
        "regularization": contract[3],
    }


def save_distributed_checkpoint(
    directory: str | Path,
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    strategy_state: StrategyState,
    safety_state: Any,
    *,
    step: int,
    config: TrainConfig | None = None,
    intersection_capacity: int | None = None,
    candidate_bound: int | None = None,
    scene_transform: Any | None = None,
    scene_scale: Any | None = None,
    pose_module: Any | None = None,
    pose_optimizer: nnx.Optimizer | None = None,
    pose_image_names: Sequence[str] | None = None,
    appearance_module: Any | None = None,
    appearance_optimizer: nnx.Optimizer | None = None,
    appearance_image_names: Sequence[str] | None = None,
    force: bool = True,
) -> Path:
    """Save one indivisible set of Gaussian-sharded training shards.

    All four nodes must be the stacked ``[world, ...]`` objects a bound
    ``nnx.pmap`` maps over. They are written together because a shard's
    parameters, Adam moments, densification statistics, and sticky overflow
    state are only consistent as a set. Replicated pose and appearance states
    are checked for exact rank agreement and stored once in canonical form.
    The manifest records the world size, per-shard and global capacity,
    per-shard slot layout, optimizer static contracts, optional scene/runtime
    capacities,
    and a configuration fingerprint so a resume cannot silently change the
    sharded contract.
    """

    if intersection_capacity is not None:
        intersection_capacity = int(intersection_capacity)
        if intersection_capacity <= 0:
            raise ValueError("intersection_capacity must be positive")
    if candidate_bound is not None:
        candidate_bound = int(candidate_bound)
        if candidate_bound <= 0:
            raise ValueError("candidate_bound must be positive")
    if (scene_transform is None) != (scene_scale is None):
        raise ValueError(
            "scene_transform and scene_scale must be provided together"
        )
    scene_metadata = None
    if scene_transform is not None:
        scene_matrix, saved_scene_scale = _validate_scene_values(
            scene_transform, scene_scale
        )
        scene_metadata = {
            "matrix": scene_matrix.tolist(),
            "scene_scale": saved_scene_scale,
        }

    world_size = _distributed_world_size(
        model, optimizer, strategy_state, safety_state
    )
    pose = _validate_distributed_pose_arguments(
        pose_module,
        pose_optimizer,
        pose_image_names,
        world_size=world_size,
        require_equal_replicas=True,
    )
    appearance = _validate_distributed_appearance_arguments(
        appearance_module,
        appearance_optimizer,
        appearance_image_names,
        world_size=world_size,
        require_equal_replicas=True,
    )
    if config is not None:
        if config.pose_opt != (pose_module is not None):
            raise ValueError(
                "distributed pose state does not match TrainConfig.pose_opt"
            )
        if (config.pose_opt or config.pose_noise > 0.0) and pose is None:
            raise ValueError(
                "distributed camera-pose training requires pose_image_names"
            )
    pose_optimizer_contract = None
    if pose_optimizer is not None:
        pose_optimizer_contract = _distributed_pose_optimizer_contract(
            pose_optimizer
        )
        if config is None:
            raise ValueError(
                "distributed pose checkpoints require TrainConfig"
            )
        expected_pose_contract = {
            "batch_size": config.data.batch_size,
            "steps": config.steps,
            "learning_rate": float(config.pose_opt_lr),
            "regularization": float(config.pose_opt_reg),
        }
        if pose_optimizer_contract != expected_pose_contract:
            raise ValueError(
                "distributed pose optimizer does not match TrainConfig"
            )
    appearance_optimizer_contract = None
    if config is not None:
        if model.has_appearance != config.app_opt:
            raise ValueError(
                "distributed model color representation does not match "
                "TrainConfig.app_opt"
            )
        if config.app_opt != (appearance_module is not None):
            raise ValueError(
                "distributed appearance state does not match "
                "TrainConfig.app_opt"
            )
    if appearance_optimizer is not None:
        if config is None:
            raise ValueError(
                "distributed appearance checkpoints require TrainConfig"
            )
        if not model.has_appearance:
            raise ValueError(
                "distributed appearance state requires an appearance model"
            )
        feature_dim = int(model.features.shape[-1])
        if (
            appearance_module.feature_dim != feature_dim
            or appearance_module.embed_dim != config.app_embed_dim
            or appearance_module.sh_degree != config.model.sh_degree
        ):
            raise ValueError(
                "distributed appearance module does not match the model or "
                "TrainConfig"
            )
        appearance_optimizer_contract = (
            _distributed_appearance_optimizer_contract(
                appearance_optimizer
            )
        )
        expected_appearance_contract = {
            "kind": "appearance_multi_adam_v1",
            "batch_size": config.data.batch_size,
            "learning_rate": float(config.app_opt_lr),
            "regularization": float(config.app_opt_reg),
        }
        if appearance_optimizer_contract != expected_appearance_contract:
            raise ValueError(
                "distributed appearance optimizer does not match TrainConfig"
            )
    optimizer_contract = _distributed_optimizer_contract(optimizer)
    if optimizer_contract["world_size"] != world_size:
        raise ValueError(
            "distributed optimizer must be created with "
            f"world_size={world_size}; got "
            f"world_size={optimizer_contract['world_size']}"
        )
    if config is not None and (
        optimizer_contract["batch_size"] != config.data.batch_size
        or optimizer_contract["kind"] != _optimizer_kind_for_config(config)
        or optimizer_contract["config"] != asdict(config.optimizer)
    ):
        raise ValueError(
            "distributed optimizer does not match the checkpoint TrainConfig"
        )
    if scene_metadata is not None and not np.isclose(
        optimizer_contract["scene_scale"],
        scene_metadata["scene_scale"],
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "scene_scale must match the distributed optimizer contract"
        )
    local_capacity = _distributed_local_capacity(model, world_size)
    optimizer_steps = np.asarray(jax.device_get(optimizer.step[...]))
    if optimizer_steps.shape != (world_size,):
        raise ValueError(
            "distributed optimizer must hold one step counter per shard"
        )
    if not np.all(optimizer_steps == optimizer_steps[0]):
        raise ValueError(
            "distributed shards disagree on the optimizer step: "
            f"{optimizer_steps.tolist()}"
        )
    if int(optimizer_steps[0]) != step:
        raise ValueError(
            f"checkpoint step argument {step} does not match distributed "
            f"optimizer step {int(optimizer_steps[0])}"
        )
    if pose_optimizer is not None:
        pose_optimizer_steps = np.asarray(
            jax.device_get(pose_optimizer.step[...])
        )
        if pose_optimizer_steps.shape != (world_size,):
            raise ValueError(
                "distributed pose optimizer must hold one step per rank"
            )
        if not np.all(pose_optimizer_steps == step):
            raise ValueError(
                "distributed pose optimizer step must match checkpoint step; "
                f"got {pose_optimizer_steps.tolist()} and step {step}"
            )
    if appearance_optimizer is not None:
        appearance_optimizer_steps = np.asarray(
            jax.device_get(appearance_optimizer.step[...])
        )
        if appearance_optimizer_steps.shape != (world_size,):
            raise ValueError(
                "distributed appearance optimizer must hold one step per rank"
            )
        if not np.all(appearance_optimizer_steps == step):
            raise ValueError(
                "distributed appearance optimizer step must match checkpoint "
                f"step; got {appearance_optimizer_steps.tolist()} and step "
                f"{step}"
            )
    active_counts = np.asarray(
        jax.device_get(jnp.count_nonzero(model.active_mask[...], axis=1))
    )
    active_prefix = np.asarray(
        jax.device_get(
            jnp.all(
                model.active_mask[...]
                == (
                    jnp.arange(local_capacity)[None, :]
                    < active_counts[:, None]
                ),
                axis=1,
            )
        )
    )
    directory = Path(directory).absolute()
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": _encode_empty_arrays(_pure_state(model)),
        "optimizer": _encode_empty_arrays(_pure_state(optimizer)),
        "strategy": _encode_empty_arrays(_pure_state(strategy_state)),
        "safety": _encode_empty_arrays(_pure_state(safety_state)),
        "step": jnp.asarray(step, dtype=jnp.int32),
    }
    if pose_module is not None:
        payload["pose"] = {
            "module": _encode_empty_arrays(
                _pure_state(_unstack_graph(pose_module, 0))
            ),
            "optimizer": _encode_empty_arrays(
                _pure_state(_unstack_graph(pose_optimizer, 0))
            ),
        }
    if appearance_module is not None:
        payload["appearance"] = {
            "module": _encode_empty_arrays(
                _pure_state(_unstack_graph(appearance_module, 0))
            ),
            "optimizer": _encode_empty_arrays(
                _pure_state(_unstack_graph(appearance_optimizer, 0))
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
    components = ["model", "optimizer", "strategy", "safety"]
    if pose_module is not None:
        components.append("pose")
    if appearance_module is not None:
        components.append("appearance")
    if scene_metadata is not None:
        components.append("scene")
    metadata = {
        "format_version": 6,
        "kind": _DISTRIBUTED_KIND,
        "components": components,
        "model_color_mode": "appearance" if model.has_appearance else "sh",
        "world_size": world_size,
        "local_capacity": local_capacity,
        "global_capacity": world_size * local_capacity,
        "max_capacity": model.max_capacity,
        "active_counts": [int(count) for count in active_counts],
        "active_prefix": [bool(value) for value in active_prefix],
        "optimizer_contract": optimizer_contract,
        "config_fingerprint": (
            None if config is None else _config_fingerprint(config)
        ),
    }
    if pose_optimizer_contract is not None:
        metadata["pose_optimizer_contract"] = pose_optimizer_contract
    if appearance_optimizer_contract is not None:
        metadata["appearance_optimizer_contract"] = (
            appearance_optimizer_contract
        )
    if intersection_capacity is not None:
        metadata["intersection_capacity"] = intersection_capacity
    if candidate_bound is not None:
        metadata["candidate_bound"] = candidate_bound
    if pose is not None:
        names, camera_count = pose
        metadata["pose_camera_count"] = camera_count
        metadata["pose_image_names"] = list(names)
    if model.has_appearance:
        metadata["appearance_feature_dim"] = int(
            model.features.shape[-1]
        )
    if appearance is not None:
        names, camera_count = appearance
        metadata["appearance_camera_count"] = camera_count
        metadata["appearance_image_names"] = list(names)
    if scene_metadata is not None:
        metadata["scene"] = scene_metadata
    (checkpoint_path / _CHECKPOINT_METADATA).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return checkpoint_path


def _world_shaped_like(
    node: Any, target_capacity: int, world_size: int, capacity: int
) -> Any:
    """Build a stacked node of the saved shape to restore a shard set into.

    Only the shapes and dtypes matter: every value is overwritten by the
    restore. Capacity-leading arrays are re-made at the saved per-shard
    capacity and the shard is replicated to the saved world size, so a host
    can read a checkpoint whose width it does not currently run at without
    having to construct one by hand.
    """

    shard = _unstack_graph(node, 0)
    state = _pure_state(shard)

    def reshape(value: Any) -> Any:
        if (
            isinstance(value, jax.Array)
            and value.ndim > 0
            and value.shape[0] == target_capacity
        ):
            return jnp.zeros(
                (capacity, *value.shape[1:]), dtype=value.dtype
            )
        return value

    nnx.update(shard, jax.tree.map(reshape, state))
    replicated = _stack_graphs([shard])
    return _stack_graphs(
        [_unstack_graph(replicated, 0) for _ in range(world_size)]
    )


def restore_distributed_checkpoint(
    checkpoint_path: str | Path,
    model: GaussianModel,
    optimizer: nnx.Optimizer,
    strategy_state: StrategyState,
    safety_state: Any,
    *,
    config: TrainConfig | None = None,
    model_config: Any | None = None,
    optimizer_config: Any | None = None,
    allow_reshard: bool = False,
    pose_module: Any | None = None,
    pose_optimizer: nnx.Optimizer | None = None,
    pose_image_names: Sequence[str] | None = None,
    appearance_module: Any | None = None,
    appearance_optimizer: nnx.Optimizer | None = None,
    appearance_image_names: Sequence[str] | None = None,
) -> int:
    """Restore an indivisible shard set into the given stacked nodes.

    By default only an exact same-world-size, same-per-shard-capacity resume
    is accepted, because a width that does not match the checkpoint is far
    more often a misconfigured run than a deliberate move, and that is worth
    refusing loudly.

    ``allow_reshard=True`` asks for the move instead: the shard set is
    restored at its saved shape and then redistributed onto the width and
    capacity of the nodes passed in, which then hold the result. That needs
    ``model_config`` and ``optimizer_config`` to rebuild the shards, and the
    target capacity has to fit the busiest new shard. See
    :func:`jax_gs.capacity.reshard_distributed_training_state` for what moves
    with each Gaussian and how the rows are dealt out. Optimizer transforms
    and static contract fields are not checkpoint Variables, so the target
    optimizer must already be constructed for the target world, batch,
    configuration, scene scale, and dense/row-selective kind. Legacy
    distributed manifests without that optimizer contract require ``config``
    as an explicit caller assertion; their original config and scene scale may
    not be verifiable. Optional replicated pose state is restored from one
    canonical copy and broadcast to the target world, including when Gaussian
    state is resharded. Replicated appearance state follows the same canonical
    restore rule, while its Gaussian features and color logits remain part of
    the owner-sharded model.
    """

    metadata = _load_metadata(checkpoint_path)
    if metadata.get("kind") != _DISTRIBUTED_KIND:
        raise ValueError(
            "checkpoint is not a distributed shard set; use restore_checkpoint"
        )
    world_size = _distributed_world_size(
        model, optimizer, strategy_state, safety_state
    )
    pose = _validate_distributed_pose_arguments(
        pose_module,
        pose_optimizer,
        pose_image_names,
        world_size=world_size,
        require_equal_replicas=False,
    )
    appearance = _validate_distributed_appearance_arguments(
        appearance_module,
        appearance_optimizer,
        appearance_image_names,
        world_size=world_size,
        require_equal_replicas=False,
    )
    local_capacity = _distributed_local_capacity(model, world_size)
    saved_world_size = int(metadata["world_size"])
    saved_local_capacity = int(metadata["local_capacity"])
    reshaping = (
        saved_world_size != world_size or saved_local_capacity != local_capacity
    )
    if reshaping and not allow_reshard:
        raise ValueError(
            f"checkpoint holds {saved_world_size} shards of capacity "
            f"{saved_local_capacity}, target world has {world_size} shards of "
            f"capacity {local_capacity}; pass allow_reshard=True with "
            "model_config and optimizer_config to redistribute it"
        )
    if reshaping and (model_config is None or optimizer_config is None):
        raise ValueError(
            "allow_reshard=True requires model_config and optimizer_config "
            "to rebuild the shards"
        )
    target_optimizer_contract = _distributed_optimizer_contract(optimizer)
    if target_optimizer_contract["world_size"] != world_size:
        raise ValueError(
            "target optimizer must be created with "
            f"world_size={world_size}; got "
            f"world_size={target_optimizer_contract['world_size']!r}"
        )
    if config is not None:
        expected_optimizer_kind = _optimizer_kind_for_config(config)
        if (
            target_optimizer_contract["batch_size"]
            != config.data.batch_size
            or target_optimizer_contract["config"]
            != asdict(config.optimizer)
            or target_optimizer_contract["kind"]
            != expected_optimizer_kind
        ):
            raise ValueError(
                "target optimizer must be created with "
                f"batch_size={config.data.batch_size}, "
                "the checkpoint TrainConfig's OptimizerConfig, and "
                f"kind={expected_optimizer_kind!r}; got "
                f"batch_size={target_optimizer_contract['batch_size']!r} "
                f"and kind={target_optimizer_contract['kind']!r}"
            )
    saved_optimizer_contract = metadata.get("optimizer_contract")
    if saved_optimizer_contract is None:
        if config is None:
            raise ValueError(
                "legacy distributed checkpoint without an optimizer "
                "contract requires config to validate the target optimizer"
            )
    else:
        required_contract_fields = {
            "batch_size",
            "world_size",
            "scene_scale",
            "kind",
            "config",
        }
        if not isinstance(saved_optimizer_contract, dict) or not (
            required_contract_fields <= saved_optimizer_contract.keys()
        ):
            raise ValueError(
                "checkpoint optimizer contract is missing required fields"
            )
        if saved_optimizer_contract["world_size"] != saved_world_size:
            raise ValueError(
                "checkpoint optimizer world_size disagrees with its shard "
                "manifest"
            )
        for field, label in (
            ("batch_size", "batch_size"),
            ("scene_scale", "scene_scale"),
            ("kind", "kind"),
            ("config", "OptimizerConfig"),
        ):
            if (
                saved_optimizer_contract[field]
                != target_optimizer_contract[field]
            ):
                raise ValueError(
                    f"checkpoint optimizer {label} does not match the "
                    "target optimizer"
                )
    components = metadata.get("components", ())
    has_pose_state = "pose" in components
    wants_pose_state = pose_module is not None
    has_appearance_state = "appearance" in components
    wants_appearance_state = appearance_module is not None
    if wants_pose_state and not has_pose_state:
        raise ValueError("checkpoint does not contain pose state")
    if wants_pose_state:
        saved_pose_optimizer_contract = metadata.get(
            "pose_optimizer_contract"
        )
        target_pose_optimizer_contract = (
            _distributed_pose_optimizer_contract(pose_optimizer)
        )
        if saved_pose_optimizer_contract != target_pose_optimizer_contract:
            raise ValueError(
                "checkpoint pose optimizer contract does not match target"
            )
        if config is None:
            raise ValueError(
                "restoring distributed pose state requires TrainConfig"
            )
        expected_pose_optimizer_contract = {
            "batch_size": config.data.batch_size,
            "steps": config.steps,
            "learning_rate": float(config.pose_opt_lr),
            "regularization": float(config.pose_opt_reg),
        }
        if target_pose_optimizer_contract != expected_pose_optimizer_contract:
            raise ValueError(
                "target pose optimizer does not match TrainConfig"
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
            raise ValueError(
                "checkpoint pose image names do not match target"
            )
    if wants_appearance_state and not has_appearance_state:
        raise ValueError("checkpoint does not contain appearance state")
    if wants_appearance_state:
        saved_appearance_optimizer_contract = metadata.get(
            "appearance_optimizer_contract"
        )
        target_appearance_optimizer_contract = (
            _distributed_appearance_optimizer_contract(
                appearance_optimizer
            )
        )
        if (
            saved_appearance_optimizer_contract
            != target_appearance_optimizer_contract
        ):
            raise ValueError(
                "checkpoint appearance optimizer contract does not match "
                "target"
            )
        if config is None:
            raise ValueError(
                "restoring distributed appearance state requires TrainConfig"
            )
        expected_appearance_optimizer_contract = {
            "kind": "appearance_multi_adam_v1",
            "batch_size": config.data.batch_size,
            "learning_rate": float(config.app_opt_lr),
            "regularization": float(config.app_opt_reg),
        }
        if (
            target_appearance_optimizer_contract
            != expected_appearance_optimizer_contract
        ):
            raise ValueError(
                "target appearance optimizer does not match TrainConfig"
            )
        if not model.has_appearance:
            raise ValueError(
                "target appearance state requires an appearance model"
            )
        target_feature_dim = int(model.features.shape[-1])
        if (
            appearance_module.feature_dim != target_feature_dim
            or appearance_module.embed_dim != config.app_embed_dim
            or appearance_module.sh_degree != config.model.sh_degree
        ):
            raise ValueError(
                "target appearance module does not match the model or "
                "TrainConfig"
            )
        saved_feature_dim = metadata.get("appearance_feature_dim")
        if (
            saved_feature_dim is None
            or int(saved_feature_dim) != target_feature_dim
        ):
            raise ValueError(
                "checkpoint appearance feature dimension does not match target"
            )
    if appearance is not None:
        names, camera_count = appearance
        saved_count = metadata.get("appearance_camera_count")
        saved_names = metadata.get("appearance_image_names")
        if saved_count is None or saved_names is None:
            raise ValueError("checkpoint appearance metadata is incomplete")
        if int(saved_count) != camera_count:
            raise ValueError(
                f"checkpoint appearance camera count is {saved_count}, "
                f"target has {camera_count}"
            )
        if tuple(saved_names) != names:
            raise ValueError(
                "checkpoint appearance image names do not match target"
            )
    saved_color_mode = metadata.get("model_color_mode", "sh")
    target_color_mode = "appearance" if model.has_appearance else "sh"
    if saved_color_mode != target_color_mode:
        raise ValueError(
            f"checkpoint model color mode is {saved_color_mode!r}, target is "
            f"{target_color_mode!r}"
        )
    if config is not None:
        saved_fingerprint = metadata.get("config_fingerprint")
        if (
            saved_fingerprint is None
            and saved_optimizer_contract is not None
        ):
            raise ValueError("checkpoint does not record a config fingerprint")
        if (
            saved_fingerprint is not None
            and saved_fingerprint != _config_fingerprint(config)
        ):
            raise ValueError(
                "checkpoint was written with a different training config"
            )

    # When resharding, restore into nodes of the saved shape first; the
    # caller's nodes receive the redistributed world below.
    if reshaping:
        holders = tuple(
            _world_shaped_like(
                node, local_capacity, saved_world_size, saved_local_capacity
            )
            for node in (model, optimizer, strategy_state, safety_state)
        )
    else:
        holders = (model, optimizer, strategy_state, safety_state)
    targets = {
        "model": _pure_state(holders[0]),
        "optimizer": _pure_state(holders[1]),
        "strategy": _pure_state(holders[2]),
        "safety": _pure_state(holders[3]),
    }
    target = {
        name: _encode_empty_arrays(state) for name, state in targets.items()
    }
    target["step"] = jnp.asarray(0, dtype=jnp.int32)
    pose_module_holder = None
    pose_optimizer_holder = None
    pose_module_target = None
    pose_optimizer_target = None
    if wants_pose_state:
        pose_module_holder = _unstack_graph(pose_module, 0)
        pose_optimizer_holder = _unstack_graph(pose_optimizer, 0)
        pose_module_target = _pure_state(pose_module_holder)
        pose_optimizer_target = _pure_state(pose_optimizer_holder)
        target["pose"] = {
            "module": _encode_empty_arrays(pose_module_target),
            "optimizer": _encode_empty_arrays(pose_optimizer_target),
        }
    appearance_module_holder = None
    appearance_optimizer_holder = None
    appearance_module_target = None
    appearance_optimizer_target = None
    if wants_appearance_state:
        appearance_module_holder = _unstack_graph(appearance_module, 0)
        appearance_optimizer_holder = _unstack_graph(
            appearance_optimizer, 0
        )
        appearance_module_target = _pure_state(appearance_module_holder)
        appearance_optimizer_target = _pure_state(
            appearance_optimizer_holder
        )
        target["appearance"] = {
            "module": _encode_empty_arrays(appearance_module_target),
            "optimizer": _encode_empty_arrays(
                appearance_optimizer_target
            ),
        }
    partial_restore = (has_pose_state and not wants_pose_state) or (
        has_appearance_state and not wants_appearance_state
    )
    checkpointer = (
        ocp.PyTreeCheckpointer()
        if partial_restore
        else ocp.StandardCheckpointer()
    )
    try:
        if partial_restore:
            restored = checkpointer.restore(
                Path(checkpoint_path).absolute(),
                item=target,
                restore_args=ocp.checkpoint_utils.construct_restore_args(
                    target
                ),
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
    restored_pose_module = None
    restored_pose_optimizer = None
    if wants_pose_state:
        nnx.update(
            pose_module_holder,
            _restore_empty_arrays(
                restored["pose"]["module"], pose_module_target
            ),
        )
        nnx.update(
            pose_optimizer_holder,
            _restore_empty_arrays(
                restored["pose"]["optimizer"], pose_optimizer_target
            ),
        )
        restored_pose_step = int(pose_optimizer_holder.step[...])
        restored_step = int(restored["step"])
        if restored_pose_step != restored_step:
            raise ValueError(
                "checkpoint pose optimizer step does not match its host step"
            )
        restored_pose_module = _stack_graphs(
            [pose_module_holder for _ in range(world_size)]
        )
        restored_pose_optimizer = _stack_graphs(
            [pose_optimizer_holder for _ in range(world_size)]
        )
    restored_appearance_module = None
    restored_appearance_optimizer = None
    if wants_appearance_state:
        nnx.update(
            appearance_module_holder,
            _restore_empty_arrays(
                restored["appearance"]["module"], appearance_module_target
            ),
        )
        nnx.update(
            appearance_optimizer_holder,
            _restore_empty_arrays(
                restored["appearance"]["optimizer"],
                appearance_optimizer_target,
            ),
        )
        restored_appearance_step = int(
            appearance_optimizer_holder.step[...]
        )
        restored_step = int(restored["step"])
        if restored_appearance_step != restored_step:
            raise ValueError(
                "checkpoint appearance optimizer step does not match its "
                "host step"
            )
        restored_appearance_module = _stack_graphs(
            [appearance_module_holder for _ in range(world_size)]
        )
        restored_appearance_optimizer = _stack_graphs(
            [appearance_optimizer_holder for _ in range(world_size)]
        )

    for node, name in (
        (holders[0], "model"),
        (holders[1], "optimizer"),
        (holders[2], "strategy"),
        (holders[3], "safety"),
    ):
        nnx.update(
            node, _restore_empty_arrays(restored[name], targets[name])
        )
    if reshaping:
        resharded = reshard_distributed_training_state(
            *holders,
            world_size,
            model_config,
            optimizer_config,
            local_capacity=local_capacity,
        )
        for node, source in zip(
            (model, optimizer, strategy_state, safety_state),
            resharded,
            strict=True,
        ):
            nnx.update(node, _pure_state(source))
    if wants_pose_state:
        nnx.update(pose_module, _pure_state(restored_pose_module))
        nnx.update(pose_optimizer, _pure_state(restored_pose_optimizer))
    if wants_appearance_state:
        nnx.update(
            appearance_module, _pure_state(restored_appearance_module)
        )
        nnx.update(
            appearance_optimizer,
            _pure_state(restored_appearance_optimizer),
        )
    return int(restored["step"])


def load_distributed_checkpoint_manifest(
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Return the shard manifest needed to rebuild matching target nodes.

    Hosts need ``world_size`` and ``local_capacity`` before they can allocate
    the stacked objects that :func:`restore_distributed_checkpoint` fills.
    """

    metadata = _load_metadata(checkpoint_path)
    if metadata.get("kind") != _DISTRIBUTED_KIND:
        raise ValueError("checkpoint is not a distributed shard set")
    return dict(metadata)


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
    scene_transform: Any | None = None,
    scene_scale: Any | None = None,
    force: bool = True,
) -> Path:
    """Save model and optional training state with Orbax."""

    if intersection_capacity is not None:
        intersection_capacity = int(intersection_capacity)
        if intersection_capacity <= 0:
            raise ValueError("intersection_capacity must be positive")
    has_scene_transform = scene_transform is not None
    has_scene_scale = scene_scale is not None
    if has_scene_transform != has_scene_scale:
        raise ValueError(
            "scene_transform and scene_scale must be provided together"
        )
    scene_metadata = None
    if has_scene_transform:
        matrix, validated_scene_scale = _validate_scene_values(
            scene_transform, scene_scale
        )
        scene_metadata = {
            "matrix": matrix.tolist(),
            "scene_scale": validated_scene_scale,
        }
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
    if scene_metadata is not None:
        components.append("scene")
    metadata: dict[str, Any] = {
        "format_version": 6,
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
    if scene_metadata is not None:
        metadata["scene"] = scene_metadata
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
    if metadata.get("kind") == _DISTRIBUTED_KIND:
        raise ValueError(
            "checkpoint holds distributed shards; use "
            "restore_distributed_checkpoint"
        )
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


def load_checkpoint_scene_transform(
    checkpoint_path: str | Path,
) -> tuple[np.ndarray, float] | None:
    """Load the persisted world-to-training transform and scene scale."""

    metadata = _load_metadata(checkpoint_path)
    components = metadata.get("components")
    if components is None:
        return None
    if not isinstance(components, list) or not all(
        isinstance(component, str) for component in components
    ):
        raise ValueError("checkpoint components metadata is invalid")
    if "scene" not in components:
        return None

    try:
        format_version = int(metadata.get("format_version", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint format_version is invalid") from exc
    if format_version < 6:
        raise ValueError("scene metadata requires checkpoint format version 6")
    scene = metadata.get("scene")
    if not isinstance(scene, dict):
        raise ValueError("checkpoint scene metadata is incomplete")
    if not {"matrix", "scene_scale"}.issubset(scene):
        raise ValueError("checkpoint scene metadata is incomplete")
    return _validate_scene_values(scene["matrix"], scene["scene_scale"])


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
