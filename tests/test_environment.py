import json
import os
import subprocess
import sys


_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _imported_thread_environment(environment):
    code = (
        "import json, os; import jax_gs; "
        f"print(json.dumps({{name: os.environ.get(name) for name in {_THREAD_VARIABLES!r}}}))"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
    )
    return json.loads(output)


def _imported_cache_environment(environment, *, import_jax_first=False):
    prefix = "import jax; " if import_jax_first else ""
    code = (
        "import json, os, stat; "
        + prefix
        + "import jax_gs; import jax; "
        "path = jax.config.jax_compilation_cache_dir; "
        "print(json.dumps({'path': path, 'env': os.environ.get("
        "'JAX_COMPILATION_CACHE_DIR'), 'mode': None if not path or '://' in path "
        "or not os.path.isdir(path) "
        "else stat.S_IMODE(os.stat(path).st_mode)}))"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
    )
    return json.loads(output)


def _initialized_default_xla_environment(environment):
    code = (
        "import json, os; import jax_gs; import jax; "
        "print(json.dumps({'flags': os.environ.get('XLA_FLAGS'), "
        "'platform': jax.devices()[0].platform}))"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
    )
    return json.loads(output)


def test_import_does_not_set_host_thread_defaults():
    environment = os.environ.copy()
    for name in _THREAD_VARIABLES:
        environment.pop(name, None)
    assert _imported_thread_environment(environment) == {
        name: None for name in _THREAD_VARIABLES
    }


def test_explicit_host_thread_configuration_is_not_clamped():
    environment = os.environ.copy()
    environment.update({name: "5" for name in _THREAD_VARIABLES})
    assert _imported_thread_environment(environment) == {
        name: "5" for name in _THREAD_VARIABLES
    }


def test_default_xla_flags_initialize_on_supported_jax():
    environment = os.environ.copy()
    environment.pop("XLA_FLAGS", None)
    environment["JAX_PLATFORMS"] = "cpu"

    assert _initialized_default_xla_environment(environment) == {
        "flags": "--xla_gpu_force_compilation_parallelism=1",
        "platform": "cpu",
    }


def test_default_compilation_cache_is_private_and_supports_late_jax_import(
    tmp_path,
):
    environment = os.environ.copy()
    environment.pop("JAX_COMPILATION_CACHE_DIR", None)
    environment.pop("JAX_ENABLE_COMPILATION_CACHE", None)
    environment["XDG_CACHE_HOME"] = str(tmp_path)

    result = _imported_cache_environment(environment, import_jax_first=True)

    expected = tmp_path / "jax-gs" / "jax-compilation-cache-v1"
    assert result == {"path": str(expected), "env": str(expected), "mode": 0o700}
    assert (tmp_path / "jax-gs").stat().st_mode & 0o777 == 0o700


def test_explicit_compilation_cache_path_is_preserved(tmp_path):
    environment = os.environ.copy()
    custom = tmp_path / "custom-cache"
    environment["JAX_COMPILATION_CACHE_DIR"] = str(custom)

    result = _imported_cache_environment(environment)

    assert result["path"] == str(custom)
    assert result["env"] == str(custom)


def test_compilation_cache_can_be_disabled(tmp_path):
    environment = os.environ.copy()
    environment.pop("JAX_COMPILATION_CACHE_DIR", None)
    environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    environment["XDG_CACHE_HOME"] = str(tmp_path)

    result = _imported_cache_environment(environment)

    assert result["path"] is None
    assert result["env"] is None
    assert not (tmp_path / "jax-gs").exists()
