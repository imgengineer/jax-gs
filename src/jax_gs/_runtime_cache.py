from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings


_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _loaded_jax():
    return sys.modules.get("jax")


def _sync_loaded_jax(path: str) -> bool:
    jax_module = _loaded_jax()
    if jax_module is None:
        return True
    try:
        jax_module.config.update("jax_compilation_cache_dir", path)
    except Exception as error:  # pragma: no cover - backend initialization edge
        warnings.warn(
            f"could not enable the JAX compilation cache at {path!r}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def _secure_directory(path: Path) -> None:
    if path.is_symlink():
        raise OSError(f"refusing symlinked cache directory: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise OSError(f"cache path is not a private directory: {path}")
    if hasattr(os, "getuid") and path.stat().st_uid != os.getuid():
        raise OSError(f"cache directory is not owned by the current user: {path}")
    path.chmod(0o700)


def configure_persistent_compilation_cache() -> str | None:
    """Set a private default cache without overriding explicit JAX settings."""

    jax_module = _loaded_jax()
    if jax_module is not None:
        configured = jax_module.config.jax_compilation_cache_dir
        if configured is not None:
            return configured
        if not jax_module.config.jax_enable_compilation_cache:
            return None

    explicit_path = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if "JAX_COMPILATION_CACHE_DIR" in os.environ:
        if explicit_path and not _sync_loaded_jax(explicit_path):
            return None
        return explicit_path

    enabled = os.environ.get("JAX_ENABLE_COMPILATION_CACHE", "true").lower()
    if enabled in _FALSE_VALUES:
        return None

    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    ).expanduser()
    project_root = cache_root / "jax-gs"
    cache_path = project_root / "jax-compilation-cache-v1"
    try:
        _secure_directory(project_root)
        _secure_directory(cache_path)
    except OSError as error:
        warnings.warn(
            f"could not create the private JAX compilation cache: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    resolved = str(cache_path.absolute())
    os.environ["JAX_COMPILATION_CACHE_DIR"] = resolved
    if not _sync_loaded_jax(resolved):
        os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
        return None
    return resolved


__all__ = ["configure_persistent_compilation_cache"]
