"""Project-owned CuTe DSL kernels and their JAX bindings.

Bindings use their module-level launch function as a CuTe compile key to skip
IR generation for cache lookup. Kernel code and module constants are immutable
within a process; specialization options are explicit keyword arguments, which
CuTe includes in the key along with tensor specifications and compiler options.
"""
