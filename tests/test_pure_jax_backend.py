import subprocess
import sys
from pathlib import Path


def test_default_import_loads_required_cuda_tile_bridge():
    project_root = Path(__file__).parents[1]

    script = """
import sys
import jax_gs
assert "cuda.tile" in sys.modules
assert "cuda.tile.jax" in sys.modules
assert callable(jax_gs.fully_fused_projection_cutile)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=True,
    )
