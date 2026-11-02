/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */
#pragma once

#include <cstddef>
#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/common/fp/fpsr.h"
#include "dynarmic/common/fp/info.h"
#include "dynarmic/common/fp/op/FPMulAdd.h"

namespace Dynarmic::Backend::X64 {

template<typename FPT>
struct alignas(16) FPMulFrame {
    VectorArray<FPT> result, a, b;
    u32 fpcr;
    FP::FPSR* fpsr;
};

template<typename FPT, size_t lanes>
void FPMulFallback(FPMulFrame<FPT>* frame) {
    const FP::FPCR fpcr{frame->fpcr};
    for (size_t i = 0; i < lanes; ++i) {
        // Adding a zero with the product's sign preserves multiply's signed
        // zero rules. The existing ARM implementation rounds before emitting
        // exceptions, suppressing IXC only for each flushed result.
        const FPT zero = (frame->a[i] ^ frame->b[i]) & FP::FPInfo<FPT>::sign_mask;
        frame->result[i] = FP::FPMulAdd(zero, frame->a[i], frame->b[i], fpcr, *frame->fpsr);
    }
}

// Keep the native multiply for ordinary values. Recompute the entire active
// vector when a lane can require ARM's before-rounding underflow/NaN/input
// denormal rules. Restoring the operation's saved MXCSR preserves earlier
// exceptions; the ARM helper ORs the correct flags from every active lane.
template<size_t fsize, bool vector>
void EmitFPMulAccurate(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    using namespace Xbyak::util;
    using FPT = mcl::unsigned_integer_of_size<fsize>;
    constexpr size_t lanes = vector ? 128 / fsize : 1;
    constexpr u32 active = fsize == 32 ? (vector ? 15U : 1U) : (vector ? 5U : 1U);
    constexpr FPT normal = FPT{1} << FP::FPInfo<FPT>::explicit_mantissa_width;
    constexpr FPT infinity = FP::FPInfo<FPT>::exponent_mask;
    constexpr FPT non_sign = FP::FPInfo<FPT>::sign_mask - 1;
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool controlled = !vector || args[2].GetImmediateU1();
    const auto fpcr = ctx.FPCR(controlled);
    const bool standard = fpcr != ctx.FPCR() &&
        !ctx.HasOptimization(OptimizationFlag::Unsafe_IgnoreStandardFPCRValue);
    const auto a = ctx.reg_alloc.UseXmm(args[0]);
    const auto b = ctx.reg_alloc.UseXmm(args[1]);
    const auto result = ctx.reg_alloc.ScratchXmm();
    const auto magnitude = ctx.reg_alloc.ScratchXmm();
    const auto mask = ctx.reg_alloc.ScratchXmm();
    const auto auxiliary = ctx.reg_alloc.ScratchXmm();
    const auto bits = ctx.reg_alloc.ScratchGpr();
    auto fallback = GenSharedLabel(), end = GenSharedLabel();
    if (standard) code.EnterStandardASIMD();
    code.sub(rsp, 16);
    code.stmxcsr(dword[rsp]);
    code.movaps(result, a);
    if constexpr (fsize == 32) {
        if constexpr (vector) code.mulps(result, b);
        else code.mulss(result, b);
    } else {
        if constexpr (vector) code.mulpd(result, b);
        else code.mulsd(result, b);
    }
    code.stmxcsr(dword[rsp + 4]);

    const auto zero_mask = [&] {
        code.movaps(auxiliary, magnitude);
        code.pcmpeqd(auxiliary, code.Const(xword, u64{0}));
        if constexpr (fsize == 64) {
            code.pshufd(xmm0, auxiliary, 0xb1);
            code.pand(auxiliary, xmm0);
        }
    };
    if (fpcr.FZ()) {
        for (auto operand : {a, b}) {
            code.movaps(magnitude, operand);
            code.pand(magnitude, code.BConst<fsize>(xword, non_sign));
            if constexpr (fsize == 32) {
                code.movaps(mask, code.BConst<fsize>(xword, normal));
                code.pcmpgtd(mask, magnitude);
            } else {
                code.movaps(mask, magnitude);
                code.psrlq(mask, 52);
                code.pcmpeqd(mask, code.Const(xword, u64{0}));
            }
            zero_mask();
            code.pandn(auxiliary, mask);
            code.movmskps(bits.cvt32(), auxiliary);
            code.test(bits.cvt32(), active);
            code.jnz(*fallback, code.T_NEAR);
        }
    }
    code.movaps(magnitude, result);
    code.pand(magnitude, code.BConst<fsize>(xword, non_sign));
    code.movaps(mask, magnitude);
    if constexpr (fsize == 32) {
        code.pcmpgtd(mask, code.BConst<fsize>(xword, normal));
        code.movaps(auxiliary, code.BConst<fsize>(xword, infinity));
        code.pcmpgtd(auxiliary, magnitude);
        code.pand(mask, auxiliary);
    } else {
        // Exponent one also includes the smallest normal boundary. Its rare
        // extra fallback is conservative and needs only SSE2 comparisons.
        code.psrlq(mask, 52);
        code.movaps(auxiliary, mask);
        code.pcmpgtd(mask, code.BConst<64>(xword, u64{1}));
        code.movaps(xmm0, code.BConst<64>(xword, u64{2047}));
        code.pcmpgtd(xmm0, auxiliary);
        code.pand(mask, xmm0);
    }
    Xbyak::Label underflow;
    code.test(dword[rsp + 4], 0x10);
    code.jnz(underflow);
    zero_mask();
    code.por(mask, auxiliary);
    code.L(underflow);
    code.movmskps(bits.cvt32(), mask);
    code.and_(bits.cvt32(), active);
    code.cmp(bits.cvt32(), active);
    code.jne(*fallback, code.T_NEAR);
    code.L(*end);
    code.add(rsp, 16);
    if (standard) code.LeaveStandardASIMD();

    ctx.deferred_emits.emplace_back([=, &code] {
        code.L(*fallback);
        // Remove only this speculative native operation's flags. Preexisting
        // IXC/UFC and flags from other vector lanes are never cleared globally.
        code.ldmxcsr(dword[rsp]);
        code.sub(rsp, 8);
        ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(result.getIdx()));
        constexpr u32 frame_size = sizeof(FPMulFrame<FPT>);
        static_assert(frame_size % 16 == 0);
        code.sub(rsp, frame_size + ABI_SHADOW_SPACE);
        code.movaps(xword[rsp + ABI_SHADOW_SPACE + offsetof(FPMulFrame<FPT>, result)], result);
        code.movaps(xword[rsp + ABI_SHADOW_SPACE + offsetof(FPMulFrame<FPT>, a)], a);
        code.movaps(xword[rsp + ABI_SHADOW_SPACE + offsetof(FPMulFrame<FPT>, b)], b);
        code.mov(dword[rsp + ABI_SHADOW_SPACE + offsetof(FPMulFrame<FPT>, fpcr)], fpcr.Value());
        code.lea(bits, ptr[r15 + code.GetJitStateInfo().offsetof_fpsr_exc]);
        code.mov(qword[rsp + ABI_SHADOW_SPACE + offsetof(FPMulFrame<FPT>, fpsr)], bits);
        code.lea(code.ABI_PARAM1, ptr[rsp + ABI_SHADOW_SPACE]);
        code.CallFunction(&FPMulFallback<FPT, lanes>);
        code.movaps(result, xword[rsp + ABI_SHADOW_SPACE + offsetof(FPMulFrame<FPT>, result)]);
        code.add(rsp, frame_size + ABI_SHADOW_SPACE);
        ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(result.getIdx()));
        code.add(rsp, 8);
        code.jmp(*end, code.T_NEAR);
    });
    ctx.reg_alloc.DefineValue(inst, result);
}

} // namespace Dynarmic::Backend::X64
