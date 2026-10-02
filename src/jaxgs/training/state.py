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
        # NNX flattens attributes in name order, not assignment order. These
        # names keep the optimizer state and statistics before the model,
        # matching the implicit update outputs so that XLA reuses each donated
        # buffer correctly; renaming an attribute can reorder the pytree.
        self.adam = nnx.OptState(adam)
        self.fragments = nnx.Variable(fragments)
        self.model = model
