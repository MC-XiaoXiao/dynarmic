/* This file is part of the dynarmic project.
 * Copyright (c) 2021 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <chrono>
#include <memory>
#include <mutex>

#include <boost/icl/interval_set.hpp>
#include <mcl/assert.hpp>
#include <mcl/scope_exit.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/backend/arm64/a32_address_space.h"
#include "dynarmic/backend/arm64/a32_core.h"
#include "dynarmic/backend/arm64/a32_jitstate.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/interface/A32/a32.h"

namespace Dynarmic::A32 {

using namespace Backend::Arm64;

struct Jit::Impl final {
    Impl(Jit* jit_interface, A32::UserConfig conf)
            : jit_interface(jit_interface)
            , conf(conf)
            , current_address_space(conf)
            , core(this->conf) {
        BindExecutionContext();
    }

    HaltReason Run() {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&halt_reason)));

        jit_interface->is_executing = true;
        SCOPE_EXIT {
            jit_interface->is_executing = false;
        };

        HaltReason hr = core.Run(current_address_space, current_state, &halt_reason);

        PerformRequestedCacheInvalidation(hr);

        return hr;
    }

    HaltReason Step() {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&halt_reason)));

        jit_interface->is_executing = true;
        SCOPE_EXIT {
            jit_interface->is_executing = false;
        };

        HaltReason hr = core.Step(current_address_space, current_state, &halt_reason);

        PerformRequestedCacheInvalidation(hr);

        return hr;
    }

    bool Precompile(u64 descriptor) {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&halt_reason)));
        const IR::LocationDescriptor location{descriptor};
        if (current_address_space.Get(location)) {
            return false;
        }
        current_address_space.GetOrEmit(location);
        return current_address_space.Get(location) != nullptr;
    }

    void GeneratePortableIR(u64 descriptor) {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&halt_reason)));
        const auto started = std::chrono::steady_clock::now();
        auto block = current_address_space.TranslateIR(IR::LocationDescriptor{descriptor});
        const auto elapsed = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
        conf.callbacks->PortableIRGenerated(descriptor, elapsed, block);
    }

    PortableIREmitOutcome PrecompileWithResult(IR::Block block) {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&halt_reason)));
        return current_address_space.Precompile(block);
    }

    void SetPortableIRDemandProvider(PortableIRDemandProvider provider, void* user_arg) {
        current_address_space.SetPortableIRDemandProvider(provider, user_arg);
    }

    void SetPortableIREmitCompletion(PortableIREmitCompletion completion, void* user_arg) {
        current_address_space.SetPortableIREmitCompletion(completion, user_arg);
    }

    CodeCacheLookup LookupCodeCache(u64 descriptor) {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&halt_reason)));
        return {current_address_space.GetCodeCacheGeneration(),
                current_address_space.Get(IR::LocationDescriptor{descriptor}) != nullptr};
    }

    void PrepareRun() {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&halt_reason)));
        current_address_space.GetOrEmit(current_state.GetLocationDescriptor());
    }

    u64 CodeCacheGeneration() const {
        return current_address_space.GetCodeCacheGeneration();
    }

    void SetHostExecutionBlockBudget(std::uint32_t) noexcept {}

    HostExecutionBudgetResult GetHostExecutionBudgetResult() const noexcept {
        return HostExecutionBudgetResult{};
    }

    void ClearCache() {
        std::unique_lock lock{invalidation_mutex};
        invalidate_entire_cache = true;
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void InvalidateCacheRange(std::uint32_t start_address, std::size_t length) {
        if (length == 0) {
            return;
        }
        std::unique_lock lock{invalidation_mutex};
        invalid_cache_ranges.add(boost::icl::discrete_interval<u32>::closed(start_address, static_cast<u32>(start_address + length - 1)));
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void Reset() {
        current_state = {};
        BindExecutionContext();
    }

    void HaltExecution(HaltReason hr) {
        Atomic::Or(&halt_reason, static_cast<u32>(hr));
        Atomic::Barrier();
    }

    void ClearHalt(HaltReason hr) {
        Atomic::And(&halt_reason, ~static_cast<u32>(hr));
        Atomic::Barrier();
    }

    std::array<std::uint32_t, 16>& Regs() {
        return current_state.regs;
    }

    const std::array<std::uint32_t, 16>& Regs() const {
        return current_state.regs;
    }

    std::array<std::uint32_t, 64>& ExtRegs() {
        return current_state.ext_regs;
    }

    const std::array<std::uint32_t, 64>& ExtRegs() const {
        return current_state.ext_regs;
    }

    std::uint32_t Cpsr() const {
        return current_state.Cpsr();
    }

    void SetCpsr(std::uint32_t value) {
        current_state.SetCpsr(value);
    }

    std::uint32_t Fpscr() const {
        return current_state.Fpscr();
    }

    void SetFpscr(std::uint32_t value) {
        current_state.SetFpscr(value);
    }

    void ClearExclusiveState() {
        current_state.exclusive_state = false;
    }

    void DumpDisassembly() const {
        ASSERT_FALSE("Unimplemented");
    }

    size_t CodeCacheUsed() const {
        return current_address_space.GetCodeCacheUsed();
    }

private:
    void BindExecutionContext() {
        current_state.callbacks_link = conf.callbacks_link;
        current_state.lookup_link = conf.lookup_link;
        current_state.runtime_config_link = conf.runtime_config_link;
        current_state.fast_dispatch_table_link = conf.fast_dispatch_table_link;
        current_state.page_table_link = conf.page_table_link;
        current_state.read_page_table_link = conf.read_page_table_link;
        current_state.coprocessor_user_arg_link = conf.coprocessor_user_arg_link;
        if (conf.lookup_link) {
            conf.lookup_link->store(reinterpret_cast<u64>(&current_address_space), std::memory_order_release);
        }
        if (conf.runtime_config_link) {
            conf.runtime_config_link->store(reinterpret_cast<u64>(&conf), std::memory_order_release);
        }
        if (conf.fast_dispatch_table_link) {
            conf.fast_dispatch_table_link->store(reinterpret_cast<u64>(current_address_space.FastDispatchTableStorage()), std::memory_order_release);
        }
    }

    void PerformRequestedCacheInvalidation(HaltReason hr) {
        if (Has(hr, HaltReason::CacheInvalidation)) {
            std::unique_lock lock{invalidation_mutex};

            ClearHalt(HaltReason::CacheInvalidation);

            if (invalidate_entire_cache) {
                current_address_space.ClearCache();

                invalidate_entire_cache = false;
                invalid_cache_ranges.clear();
                return;
            }

            if (!invalid_cache_ranges.empty()) {
                current_address_space.InvalidateCacheRanges(invalid_cache_ranges);

                invalid_cache_ranges.clear();
                return;
            }
        }
    }

    Jit* jit_interface;
    A32::UserConfig conf;
    A32JitState current_state{};
    A32AddressSpace current_address_space;
    A32Core core;

    volatile u32 halt_reason = 0;

    std::mutex invalidation_mutex;
    boost::icl::interval_set<u32> invalid_cache_ranges;
    bool invalidate_entire_cache = false;
};

Jit::Jit(UserConfig conf)
        : impl(std::make_unique<Impl>(this, conf)) {}

Jit::~Jit() = default;

void Jit::PrepareRun() {
    impl->PrepareRun();
}

HaltReason Jit::Run() {
    return impl->Run();
}

HaltReason Jit::Step() {
    return impl->Step();
}

void Jit::SetHostExecutionBlockBudget(std::uint32_t block_budget) {
    impl->SetHostExecutionBlockBudget(block_budget);
}

Jit::HostExecutionBudgetResult
Jit::GetHostExecutionBudgetResult() const {
    return impl->GetHostExecutionBudgetResult();
}

bool Jit::Precompile(std::uint64_t descriptor) {
    return impl->Precompile(descriptor);
}

void Jit::GeneratePortableIR(std::uint64_t descriptor) {
    impl->GeneratePortableIR(descriptor);
}

Jit::PortableIREmitOutcome Jit::PrecompileWithResult(IR::Block block) {
    return impl->PrecompileWithResult(std::move(block));
}

bool Jit::Precompile(IR::Block block) {
    return PrecompileWithResult(std::move(block)) == PortableIREmitOutcome::NativeEmitted;
}

void Jit::SetPortableIRDemandProvider(PortableIRDemandProvider provider, void* user_arg) {
    impl->SetPortableIRDemandProvider(provider, user_arg);
}

void Jit::SetPortableIREmitCompletion(PortableIREmitCompletion completion, void* user_arg) {
    impl->SetPortableIREmitCompletion(completion, user_arg);
}

Jit::CodeCacheLookup Jit::LookupCodeCache(std::uint64_t descriptor) {
    return impl->LookupCodeCache(descriptor);
}

std::uint64_t Jit::CodeCacheGeneration() const {
    return impl->CodeCacheGeneration();
}

void Jit::ClearCache() {
    impl->ClearCache();
}

void Jit::InvalidateCacheRange(std::uint32_t start_address, std::size_t length) {
    impl->InvalidateCacheRange(start_address, length);
}

void Jit::Reset() {
    impl->Reset();
}

void Jit::HaltExecution(HaltReason hr) {
    impl->HaltExecution(hr);
}

void Jit::ClearHalt(HaltReason hr) {
    impl->ClearHalt(hr);
}

std::array<std::uint32_t, 16>& Jit::Regs() {
    return impl->Regs();
}

const std::array<std::uint32_t, 16>& Jit::Regs() const {
    return impl->Regs();
}

std::array<std::uint32_t, 64>& Jit::ExtRegs() {
    return impl->ExtRegs();
}

const std::array<std::uint32_t, 64>& Jit::ExtRegs() const {
    return impl->ExtRegs();
}

std::uint32_t Jit::Cpsr() const {
    return impl->Cpsr();
}

void Jit::SetCpsr(std::uint32_t value) {
    impl->SetCpsr(value);
}

std::uint32_t Jit::Fpscr() const {
    return impl->Fpscr();
}

void Jit::SetFpscr(std::uint32_t value) {
    impl->SetFpscr(value);
}

void Jit::ClearExclusiveState() {
    impl->ClearExclusiveState();
}

void Jit::DumpDisassembly() const {
    impl->DumpDisassembly();
}

std::size_t Jit::CodeCacheUsed() const {
    return impl->CodeCacheUsed();
}

}  // namespace Dynarmic::A32
