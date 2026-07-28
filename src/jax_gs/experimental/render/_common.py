"""Shared inference-render helpers."""


def check_inference_grad_mode() -> None:
    """Document the JAX inference boundary.

    JAX has no process-global equivalent of ``torch.no_grad``. The packed-scene
    kernels enforce the same observable boundary with ``stop_gradient``.
    """


__all__ = ["check_inference_grad_mode"]
