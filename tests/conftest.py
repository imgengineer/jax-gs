import os

# Must be set before test modules import JAX. This keeps the test runner from
# reserving most GPU memory and is especially important when a desktop session
# or another CUDA process is active.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
