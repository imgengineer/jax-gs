import os


# Must be set before test modules import JAX. This keeps the test runner from
# reserving most GPU memory and is especially important when a desktop session
# or another CUDA process is active.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "4")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_force_compilation_parallelism=1",
)
