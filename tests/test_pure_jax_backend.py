from pathlib import Path
import subprocess
import sys


def test_default_import_does_not_require_cuda_tile():
    project_root = Path(__file__).parents[1]

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
