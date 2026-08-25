import math

import pytest

import jax
import jax.numpy as jnp

import cutlass.cute as cute
import cutlass.jax as cjax
import cuda.bindings.driver as cuda


_SHAPE = (1, 64, 2)


@cute.kernel
def _add_kernel(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    a_fragment = cute.make_rmem_tensor(cute.size(a, mode=[0]), a.element_type)
    b_fragment = cute.make_rmem_tensor(cute.size(b, mode=[0]), b.element_type)
    out_fragment = cute.make_rmem_tensor(
        cute.size(out, mode=[0]), out.element_type
    )
    cute.autovec_copy(a[None, thread, block], a_fragment)
    cute.autovec_copy(b[None, thread, block], b_fragment)
    out_fragment.store(a_fragment.load() + b_fragment.load())
    cute.autovec_copy(out_fragment, out[None, thread, block])


@cute.jit
def _launch_add(
    stream: cuda.CUstream,
    a: cute.Tensor,
    b: cute.Tensor,
    out: cute.Tensor,
):
    _add_kernel(a, b, out).launch(
        grid=[a.shape[-1], 1, 1],
        block=[a.shape[-2], 1, 1],
        stream=stream,
    )


_add = cjax.cutlass_call(
    _launch_add,
    output_shape_dtype=jax.ShapeDtypeStruct(_SHAPE, jnp.float32),
    use_static_tensors=True,
)


@jax.custom_vjp
def _double(value):
    return _add(value, value)


def _double_fwd(value):
    output = _add(value, value)
    return output, None


def _double_bwd(_residuals, cotangent):
    return (cotangent * 2.0,)


_double.defvjp(_double_fwd, _double_bwd)


@pytest.mark.gpu
def test_cutlass_call_runs_under_jit_and_custom_vjp():
    if jax.devices()[0].platform != "gpu":
        pytest.skip("CuTe DSL bridge requires a GPU")

    value = jnp.arange(math.prod(_SHAPE), dtype=jnp.float32).reshape(_SHAPE)
    total, gradient = jax.jit(
        jax.value_and_grad(lambda x: jnp.sum(_double(x)))
    )(value)

    assert total == pytest.approx(float(jnp.sum(value * 2.0)))
    assert jnp.array_equal(gradient, jnp.full(_SHAPE, 2.0, jnp.float32))
