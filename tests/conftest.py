"""Shared test markers for optional CUDA and CUTLASS coverage."""

import importlib.util

import jax
import pytest


def pytest_collection_modifyitems(items):
    """Skip optional backend tests once, after pytest has collected them."""
    has_cute = importlib.util.find_spec("cutlass") is not None
    has_gpu = has_cute and jax.default_backend() == "gpu"
    for item in items:
        if item.get_closest_marker("gpu") and not has_gpu:
            item.add_marker(pytest.mark.skip(reason="requires JAX CUDA and CuTe"))
        elif item.get_closest_marker("cute") and not has_cute:
            item.add_marker(pytest.mark.skip(reason="requires NVIDIA CUTLASS DSL"))
