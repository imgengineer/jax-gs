"""PTX half2 operations used by LiteGS's packed rasterizer."""

import cutlass.cute as cute
from cutlass import Float32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def pack(x, y, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            Uint32.mlir_type,
            [Float32(x).ir_value(loc=loc, ip=ip), Float32(y).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 lo, hi; cvt.rn.f16.f32 lo, $1; cvt.rn.f16.f32 hi, $2; mov.b32 $0, {lo, hi}; }",
            "=r,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def get(x, high=False, *, loc=None, ip=None):
    half = "hi" if high else "lo"
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32.f16 $0, " + half + "; }",
            "=f,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def splat(x, high=False, *, loc=None, ip=None):
    """Duplicate one half without a float32 conversion and rounding trip."""
    selector = "0x3232" if high else "0x1010"
    return Uint32(
        llvm.inline_asm(
            Uint32.mlir_type,
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "prmt.b32 $0, $1, $1, " + selector + ";",
            "=r,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def binary(a, b, op, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            Uint32.mlir_type,
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            op + " $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def exp2(a, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            Uint32.mlir_type,
            [Uint32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.f16x2 $0, $1;",
            "=r,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def add(a, b):
    return binary(a, b, "add.rn.f16x2")


@cute.jit
def sub(a, b):
    return binary(a, b, "sub.rn.f16x2")


@cute.jit
def mul(a, b):
    return binary(a, b, "mul.rn.f16x2")


@cute.jit
def minimum(a, b):
    return binary(a, b, "min.f16x2")


@cute.jit
def ge_mask(a, b):
    return binary(a, b, "set.ge.u32.f16x2")


@cute.jit
def gt_mask(a, b):
    return binary(a, b, "set.gt.u32.f16x2")


@cute.jit
def reciprocal(a):
    return pack(cute.arch.rcp_approx(get(a)), cute.arch.rcp_approx(get(a, True)))


@cute.jit
def exp(a):
    return exp2(mul(a, pack(1.4426950409, 1.4426950409)))


@cute.jit
def sum_pair(a):
    return get(a) + get(a, True)


@dsl_user_op
def float_bits(a, *, loc=None, ip=None):
    return Uint32(
        llvm.bitcast(Uint32.mlir_type, Float32(a).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def bits_float(a, *, loc=None, ip=None):
    return Float32(
        llvm.bitcast(Float32.mlir_type, Uint32(a).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def fma(a, b, c, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            Uint32.mlir_type,
            [Uint32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c)],
            "fma.rn.f16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def warp_sum(a):
    a = add(a, cute.arch.shuffle_sync_down(a, 16))
    a = add(a, cute.arch.shuffle_sync_down(a, 8))
    a = add(a, cute.arch.shuffle_sync_down(a, 4))
    a = add(a, cute.arch.shuffle_sync_down(a, 2))
    a = add(a, cute.arch.shuffle_sync_down(a, 1))
    return a


@cute.jit
def warp_sum_scaled(values):
    """LiteGS's shared-exponent integer redux for float/float2/float3."""
    from cutlass import Int32, range_constexpr

    exponent = Uint32(0)
    for i in range_constexpr(len(values)):
        exponent = cute.max(exponent, (float_bits(values[i]) >> 23) & 255)
    exponent = Int32(cute.arch.warp_redux_sync(exponent, "max")) - 127
    shift = 23 - exponent
    valid = exponent > -127 and shift < 128
    factor = Float32(0)
    inverse = Float32(0)
    if valid:
        factor = bits_float(Uint32(shift + 127) << 23)
        inverse = bits_float(Uint32(127 - shift) << 23)
    result = ()
    for i in range_constexpr(len(values)):
        scaled = Int32(values[i] * factor)
        summed = cute.arch.warp_redux_sync(scaled, "add")
        result += (Float32(summed) * inverse,)
    return result
