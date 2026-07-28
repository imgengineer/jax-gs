"""Small deterministic training schedulers from gsplat current main."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TwoStageScheduleStep:
    """Resolved coarse/fine stage for one global training step."""

    stage: str
    frame_index: int
    shuffle: bool


class TwoStageScheduler:
    """Map global steps to G-SHARP's coarse then fine training stages."""

    def __init__(
        self,
        coarse_steps: int,
        fine_steps: int,
        coarse_frame_index: int = 0,
    ) -> None:
        if coarse_steps < 0 or fine_steps < 0:
            raise ValueError("step counts must be non-negative")
        self.coarse_steps = coarse_steps
        self.fine_steps = fine_steps
        self.coarse_frame_index = coarse_frame_index

    def step(self, global_step: int, num_frames: int) -> TwoStageScheduleStep:
        if global_step < 0:
            raise ValueError(f"global_step must be non-negative, got {global_step}")
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if not 0 <= self.coarse_frame_index < num_frames:
            raise ValueError(
                f"coarse_frame_index={self.coarse_frame_index} out of "
                f"[0, num_frames={num_frames})"
            )
        if global_step < self.coarse_steps:
            return TwoStageScheduleStep(
                stage="coarse",
                frame_index=self.coarse_frame_index,
                shuffle=False,
            )
        return TwoStageScheduleStep(
            stage="fine",
            frame_index=(global_step - self.coarse_steps) % num_frames,
            shuffle=True,
        )


__all__ = ["TwoStageScheduleStep", "TwoStageScheduler"]
