/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <atomic>
#include <array>

#include <mcl/stdint.hpp>

namespace Dynarmic::Backend::X64 {

class BlockOfCode;

#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4324)  // Structure was padded due to alignment specifier
#endif

struct A32JitState {
    using ProgramCounterType = u32;

    A32JitState() { ResetRSB(); }

    std::array<u32, 16> Reg{};  // Current register file.
    // TODO: Mode-specific register sets unimplemented.

    u32 upper_location_descriptor = 0;

    u32 cpsr_ge = 0;
    u32 cpsr_q = 0;
    u32 cpsr_nzcv = 0;
    u32 cpsr_jaifm = 0;
    u32 Cpsr() const;
    void SetCpsr(u32 cpsr);

    alignas(16) std::array<u32, 64> ExtReg{};  // Extension registers.

    // For internal use (See: BlockOfCode::RunCode)
    u32 guest_MXCSR = 0x00001f80;
    u32 asimd_MXCSR = 0x00009fc0;
    volatile u32 halt_reason = 0;

    // Exclusive state
    u32 exclusive_state = 0;

    static constexpr size_t RSBSize = 8;  // MUST be a power of 2.
    static constexpr size_t RSBPtrMask = RSBSize - 1;
    u32 rsb_ptr = 0;
    std::array<u64, RSBSize> rsb_location_descriptors;
    std::array<u64, RSBSize> rsb_codeptrs;
    u64 stable_link_hits = 0;
    u64 stable_link_misses = 0;
    u64 rsb_hits = 0;
    u64 rsb_misses = 0;
    void ResetRSB();

    u32 fpsr_exc = 0;
    u32 fpsr_qc = 0;
    u32 fpsr_nzcv = 0;
    u32 Fpscr() const;
    void SetFpscr(u32 FPSCR);

    // Runtime-owned indirection. Generated code loads the current callback
    // owner through this link instead of embedding an executor-specific
    // UserCallbacks address.
    const std::atomic<u64>* callbacks_link = nullptr;

    // Runtime-owned indirection for the dispatcher lookup callback's opaque
    // executor argument.
    const std::atomic<u64>* lookup_link = nullptr;

    // Runtime-owned indirection for the backend-owned UserConfig used by
    // generated helper calls.
    const std::atomic<u64>* runtime_config_link = nullptr;

    // Runtime-owned indirection for the mutable fast-dispatch table used by
    // terminal handlers.
    const std::atomic<u64>* fast_dispatch_table_link = nullptr;

    // Runtime-owned page-table bases loaded by the run prelude. These links
    // keep generated code independent of an executor's AddressSpace object.
    const std::atomic<u64>* page_table_link = nullptr;
    const std::atomic<u64>* read_page_table_link = nullptr;

    // Optional runtime owner for Coprocessor::Callback user arguments.
    const std::atomic<u64>* coprocessor_user_arg_link = nullptr;

    // Runtime-owned exclusive monitor bases used by fast exclusive-memory
    // sequences. Keeping these in link cells makes emitted code independent of
    // the executor's monitor allocation.
    const std::atomic<u64>* exclusive_monitor_lock_link = nullptr;
    const std::atomic<u64>* exclusive_monitor_addresses_link = nullptr;
    const std::atomic<u64>* exclusive_monitor_values_link = nullptr;

    u64 GetUniqueHash() const noexcept {
        return (static_cast<u64>(upper_location_descriptor) << 32) | (static_cast<u64>(Reg[15]));
    }

    void TransferJitState(const A32JitState& src, bool reset_rsb) {
        Reg = src.Reg;
        upper_location_descriptor = src.upper_location_descriptor;
        cpsr_ge = src.cpsr_ge;
        cpsr_q = src.cpsr_q;
        cpsr_nzcv = src.cpsr_nzcv;
        cpsr_jaifm = src.cpsr_jaifm;
        ExtReg = src.ExtReg;
        guest_MXCSR = src.guest_MXCSR;
        asimd_MXCSR = src.asimd_MXCSR;
        fpsr_exc = src.fpsr_exc;
        fpsr_qc = src.fpsr_qc;
        fpsr_nzcv = src.fpsr_nzcv;

        exclusive_state = 0;

        if (reset_rsb) {
            ResetRSB();
        } else {
            rsb_ptr = src.rsb_ptr;
            rsb_location_descriptors = src.rsb_location_descriptors;
            rsb_codeptrs = src.rsb_codeptrs;
        }
    }
};

#ifdef _MSC_VER
#    pragma warning(pop)
#endif

using CodePtr = const void*;

}  // namespace Dynarmic::Backend::X64
