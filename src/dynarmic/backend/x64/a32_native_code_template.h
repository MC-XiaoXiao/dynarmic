/* This file is part of the dynarmic project. SPDX-License-Identifier: 0BSD */
#pragma once

#include <algorithm>
#include <array>
#include <tuple>

#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/native_code_relocation.h"
#include "dynarmic/frontend/A32/a32_location_descriptor.h"
#include "dynarmic/interface/A32/config.h"
#include "dynarmic/interface/A32/native_code_template.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::Backend::X64 {
inline auto NativeTemplateConfiguration(const A32::UserConfig& conf) {
    return std::tuple{conf.arch_version,
                      conf.optimizations,
                      conf.fast_compilation,
                      conf.unsafe_optimizations,
                      conf.define_unpredictable_behaviour,
                      conf.hook_hint_instructions,
                      conf.hook_isb,
                      conf.check_halt_on_memory_access,
                      conf.enable_cycle_counting,
                      conf.enable_host_execution_block_budget,
                      conf.always_little_endian,
                      conf.wall_clock_cntpct,
                      bool(conf.page_table),
                      bool(conf.read_page_table),
                      bool(conf.instruction_page_table),
                      conf.read_page_table == conf.page_table,
                      conf.absolute_offset_page_table,
                      conf.page_table_pointer_mask_bits,
                      conf.detect_misaligned_access_via_page_table,
                      conf.only_detect_misalignment_via_page_table_on_page_boundary,
                      bool(conf.global_monitor),
                      conf.fastmem_exclusive_access};
}

template<auto... Functions>
inline auto NativeTemplateCallbackBindings(A32::UserCallbacks* callbacks) {
    return std::array{Devirtualize<Functions>(callbacks).Binding(reinterpret_cast<u64>(callbacks))...};
}
inline auto NativeTemplateCallbacks(A32::UserCallbacks* callbacks) {
    using C = A32::UserCallbacks;
    return NativeTemplateCallbackBindings<
        &C::MemoryRead8, &C::MemoryRead16, &C::MemoryRead32, &C::MemoryRead64, &C::MemoryWrite8, &C::MemoryWrite16,
        &C::MemoryWrite32, &C::MemoryWrite64, &C::MemorySwap8, &C::MemorySwap32, &C::MemoryWriteExclusive8,
        &C::MemoryWriteExclusive16, &C::MemoryWriteExclusive32, &C::MemoryWriteExclusive64, &C::MemoryReadExclusive,
        &C::MemoryWriteExclusiveBegin, &C::MemoryExecutionSuspend, &C::MemoryExecutionResume, &C::InstructionFetch,
        &C::CallSVC, &C::ExceptionRaised, &C::InstructionSynchronizationBarrierRaised, &C::InterpreterFallback,
        &C::AddTicks, &C::GetTicksRemaining>(callbacks);
}
inline bool NativeTemplateRuntimeIsLinked(const A32::UserConfig& conf) {
    return conf.HasOptimization(OptimizationFlag::FastDispatch) && conf.callbacks_link && conf.lookup_link
        && conf.runtime_config_link && conf.fast_dispatch_table_link && !conf.fastmem_pointer
        && !conf.fastmem_exclusive_access && !conf.very_verbose_debugging_output
        && (!conf.page_table || conf.page_table_link) && (!conf.read_page_table || conf.read_page_table_link)
        && (!conf.global_monitor
            || (conf.exclusive_monitor_lock_link && conf.exclusive_monitor_addresses_link
                && conf.exclusive_monitor_values_link));
}

struct A32NativeTemplateEnvironment {
    A32NativeTemplateEnvironment(const A32::UserConfig& conf, BlockOfCode& code)
            : configuration{NativeTemplateConfiguration(conf)}
            , callbacks{NativeTemplateCallbacks(conf.callbacks)}
            , host_features{code.HostFeatures()}
            , prelude{CaptureNativePrelude(code)} {}
    const decltype(NativeTemplateConfiguration(std::declval<A32::UserConfig>())) configuration;
    const decltype(NativeTemplateCallbacks(nullptr)) callbacks;
    const HostFeature host_features;
    const std::shared_ptr<const NativePreludeShape> prelude;
    bool Compatible(const A32::UserConfig& conf, const A32NativeTemplateEnvironment& local) const {
        return prelude && prelude == local.prelude && host_features == local.host_features
            && NativeTemplateRuntimeIsLinked(conf) && configuration == local.configuration
            && callbacks == local.callbacks && configuration == NativeTemplateConfiguration(conf)
            && callbacks == NativeTemplateCallbacks(conf.callbacks);
    }
};

class A32NativeCodeTemplate final : public A32::NativeCodeTemplate {
public:
    A32NativeCodeTemplate(std::shared_ptr<const A32NativeTemplateEnvironment> environment,
                          const IR::Block& block,
                          std::vector<u8> bytes,
                          NativeCodeRelocations relocations,
                          std::vector<NativeBlockLink> links)
            : environment{std::move(environment)}
            , location{block.Location()}
            , end_location{block.EndLocation()}
            , bytes{std::move(bytes)}
            , relocations{std::move(relocations)}
            , links{std::move(links)} {}
    u64 LocationDescriptor() const noexcept override { return location.Value(); }
    size_t CodeSize() const noexcept override { return bytes.size(); }
    size_t MemoryFootprint() const noexcept override {
        // The environment and interned prelude are shared slab metadata;
        // charge each block's owned bytes/relocations rather than repeating
        // the entire callback table in every cache entry's budget.
        return sizeof(*this) + bytes.capacity() + relocations.MemoryFootprint()
             + links.capacity() * sizeof(NativeBlockLink);
    }
    static bool Eligible(const IR::Block& block) {
        return A32::LocationDescriptor{block.EndLocation()}.PC() > A32::LocationDescriptor{block.Location()}.PC()
            && std::none_of(block.begin(), block.end(), [](const IR::Inst& inst) {
                   return inst.IsCoprocessorInstruction() || inst.GetOpcode() == IR::Opcode::CallHostFunction;
               });
    }
    const std::shared_ptr<const A32NativeTemplateEnvironment> environment;
    const IR::LocationDescriptor location, end_location;
    const std::vector<u8> bytes;
    const NativeCodeRelocations relocations;
    const std::vector<NativeBlockLink> links;
};
}  // namespace Dynarmic::Backend::X64
