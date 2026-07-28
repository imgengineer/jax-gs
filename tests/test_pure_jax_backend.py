from pathlib import Path
import subprocess
import sys


def test_project_has_no_cuda_tile_dependency_or_modules():
    project_root = Path(__file__).parents[1]
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    lockfile = (project_root / "uv.lock").read_text(encoding="utf-8")
    assert "cuda-tile" not in pyproject
    assert "cuda-tile" not in lockfile
    assert not list((project_root / "src" / "jax_gs").glob("cutile_*.py"))

    script = """
import importlib
import importlib.abc
import pkgutil

class BlockCudaTile(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "cuda.tile" or fullname.startswith("cuda.tile."):
            raise ImportError("cuda.tile imports are forbidden")
        return None

import sys
sys.meta_path.insert(0, BlockCudaTile())
import jax_gs
for module in pkgutil.walk_packages(jax_gs.__path__, prefix="jax_gs."):
    importlib.import_module(module.name)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=True,
    )
