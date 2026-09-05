"""Bounded evaluation with persistent capacity high-water marks."""

from collections.abc import Callable, Sequence
import math
from typing import Any

import jax
import numpy as np

from ..config import TrainConfig
from ..rasterization import _automatic_intersection_capacity
from ._memory import (
    _candidate_bound_for_occupancy,
    _check_evaluation_memory_budget,
    _intersection_bucket_capacity,
    _training_config_with_candidate_bound,
    _training_config_with_intersection_capacity,
)


class _EvaluationRenderer:
    def __init__(
        self,
        config: TrainConfig,
        width: int,
        height: int,
        make_step: Callable[[TrainConfig], Any],
        *,
        devices: Sequence[jax.Device] | None = None,
    ) -> None:
        self.config = config
        self.width = width
        self.height = height
        self.make_step = make_step
        self.devices = devices
        self.intersection_capacity = 1
        self.candidate_bound: int | None = None
        self.runtime_config: TrainConfig | None = None
        self.render_step: Any = None

    def __call__(
        self,
        *args: Any,
        physical_capacity: int,
        intersection_capacity: int,
        candidate_bound: int | None,
        **kwargs: Any,
    ):
        tile_size = self.config.rasterizer.tile_size
        tile_count = math.ceil(self.width / tile_size) * math.ceil(self.height / tile_size)
        limit = _automatic_intersection_capacity(
            physical_capacity, tile_count, self.config.rasterizer,
        )
        self.intersection_capacity = min(
            limit, max(self.intersection_capacity, intersection_capacity)
        )
        self.candidate_bound = max(
            self.candidate_bound or 0, candidate_bound or 0
        ) or None
        while True:
            runtime = _training_config_with_intersection_capacity(
                self.config, self.intersection_capacity
            )
            if self.candidate_bound is not None:
                runtime = _training_config_with_candidate_bound(
                    runtime, self.candidate_bound
                )
            _check_evaluation_memory_budget(
                runtime, physical_capacity=physical_capacity,
                width=self.width, height=self.height, devices=self.devices,
            )
            if runtime != self.runtime_config:
                self.render_step = self.make_step(runtime)
                self.runtime_config = runtime
            output = self.render_step(*args, **kwargs)
            tiles, intersection, required, busiest = jax.device_get(output[2:])
            if not np.any(tiles) and not np.any(intersection):
                return output[:4]
            del output
            if np.any(intersection):
                next_capacity = _intersection_bucket_capacity(
                    int(np.max(required)),
                    minimum=self.config.intersection_bucket_min_capacity,
                    maximum=limit,
                )
                if next_capacity <= self.intersection_capacity:
                    raise RuntimeError("evaluation intersection overflow requested no growth")
                print(
                    f"evaluation_intersection_growth={self.intersection_capacity}"
                    f"->{next_capacity}", flush=True,
                )
                self.intersection_capacity = next_capacity
            else:
                next_bound = _candidate_bound_for_occupancy(
                    int(np.max(busiest)), self.config.rasterizer.max_gaussians_per_tile
                )
                if self.candidate_bound is None or next_bound <= self.candidate_bound:
                    raise RuntimeError("evaluation tile overflow requested no growth")
                print(
                    f"evaluation_candidate_growth={self.candidate_bound}->{next_bound}",
                    flush=True,
                )
                self.candidate_bound = next_bound
