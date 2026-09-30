"""Fixed training buffers managed by NNX."""

from flax import nnx

from ..render.types import FragmentStatistics
from ..scene.point import GaussianModel
from .optimizer import AdamState


class TrainingState(nnx.Module):
    """Model, Optax state and statistics bound once to the training step."""

    def __init__(
        self, model: GaussianModel, adam: AdamState, fragments: FragmentStatistics
    ) -> None:
        # Keep Adam/statistics before the model in pytree order, matching the
        # implicit update outputs so XLA reuses each donated buffer correctly.
        self.adam = nnx.OptState(adam)
        self.fragments = nnx.Variable(fragments)
        self.model = model
