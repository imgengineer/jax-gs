"""JAX profiler ranges matching gsplat current main's tracing surface."""

from __future__ import annotations

from contextlib import ContextDecorator
import threading
from typing import Any, Callable, ContextManager, TypeVar

import jax


_F = TypeVar("_F", bound=Callable)
_THREAD_STATE = threading.local()


def _annotation(name: str, kwargs: dict[str, Any]):
    return jax.profiler.TraceAnnotation(name, **kwargs)


class _Trace(ContextDecorator):
    def __init__(self, name: str, **kwargs: Any) -> None:
        self._name = name
        self._kwargs = kwargs
        self._active: list[Any] = []

    def __enter__(self) -> None:
        annotation = _annotation(self._name, self._kwargs)
        self._active.append(annotation)
        annotation.__enter__()
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        annotation = self._active.pop()
        return bool(annotation.__exit__(exc_type, exc_value, traceback))


def _push_stack() -> list[Any]:
    stack = getattr(_THREAD_STATE, "annotations", None)
    if stack is None:
        stack = []
        _THREAD_STATE.annotations = stack
    return stack


def trace_push(name: str, **kwargs: Any) -> None:
    annotation = _annotation(name, kwargs)
    annotation.__enter__()
    _push_stack().append(annotation)


def trace_pop() -> None:
    stack = _push_stack()
    if not stack:
        raise RuntimeError("trace_pop() called without a matching trace_push()")
    stack.pop().__exit__(None, None, None)


def trace_range(name: str, **kwargs: Any) -> ContextManager[None]:
    return _Trace(name, **kwargs)


def trace_function(name: str, **kwargs: Any) -> Callable[[_F], _F]:
    return _Trace(name, **kwargs)


__all__ = ["trace_function", "trace_pop", "trace_push", "trace_range"]
