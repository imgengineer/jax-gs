"""PTX half2 and explicitly rounded float32 operations used by the CuTe kernels."""

import cutlass.cute as cute
from cutlass import Constexpr, Float32, Uint32, const_expr
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
def pack_pair_sums(a, b, *, loc=None, ip=None):
    """Pack each input's horizontal half sum without float32 conversions."""
    return Uint32(
        llvm.inline_asm(
            Uint32.mlir_type,
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 other, sa, sb; "
            "prmt.b32 other, $1, $1, 0x1032; add.rn.f16x2 sa, $1, other; "
            "prmt.b32 other, $2, $2, 0x1032; add.rn.f16x2 sb, $2, other; "
            "prmt.b32 $0, sa, sb, 0x5410; }",
            "=r,r,r",
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


def _f32_asm(op, operands, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [Float32(x).ir_value(loc=loc, ip=ip) for x in operands],
            op + " $0, " + ", ".join(f"${i + 1}" for i in range(len(operands))) + ";",
            "=f," + ",".join("f" for _ in operands),
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ffma(a, b, c, *, loc=None, ip=None):
    """fma.rn.f32 whose rounding does not depend on the compiler's contraction choice."""
    return _f32_asm("fma.rn.f32", (a, b, c), loc=loc, ip=ip)


@dsl_user_op
def fmul(a, b, *, loc=None, ip=None):
    """mul.rn.f32, which is never fused into a neighboring add."""
    return _f32_asm("mul.rn.f32", (a, b), loc=loc, ip=ip)


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    """add.rn.f32, which is never fused with a neighboring multiply."""
    return _f32_asm("add.rn.f32", (a, b), loc=loc, ip=ip)


@dsl_user_op
def fsub(a, b, *, loc=None, ip=None):
    """sub.rn.f32, which is never fused with a neighboring multiply."""
    return _f32_asm("sub.rn.f32", (a, b), loc=loc, ip=ip)


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
def warp_sums(values, lane, half2: Constexpr = False):
    """Warp sums of 2**k values; lane l receives the sum of values[l >> (5 - k)].

    Each butterfly step halves the values a lane keeps (sending the rest to
    its partner), so 2**k sums take 2**k + 4 - k shuffles instead of 5 each.
    half2 values are added as packed halves.
    """
    from cutlass import range_constexpr

    level = list(values)
    steps = len(level).bit_length() - 1
    for step in range_constexpr(5):
        bit = 4 - step
        if const_expr(step < steps):
            half = len(level) // 2
            upper = ((lane >> bit) & 1) != 0
            merged = []
            for j in range_constexpr(half):
                keep, send = level[j], level[j + half]
                if upper:
                    keep, send = send, keep
                received = cute.arch.shuffle_sync_bfly(send, 1 << bit)
                merged.append(add(keep, received) if half2 else keep + received)
            level = merged
        else:
            received = cute.arch.shuffle_sync_bfly(level[0], 1 << bit)
            level = [add(level[0], received) if half2 else level[0] + received]
    return level[0]
