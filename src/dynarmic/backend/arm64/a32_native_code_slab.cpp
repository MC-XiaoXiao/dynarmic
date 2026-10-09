/* This file is part of the dynarmic project. SPDX-License-Identifier: 0BSD */
#include <array>
#include <memory>
#include <typeinfo>
#include <utility>

#include "dynarmic/backend/native_code_slab_lifetime.h"
#include "dynarmic/backend/arm64/a32_address_space.h"
#include "dynarmic/backend/arm64/a32_jitstate.h"
#include "dynarmic/interface/A32/native_code_template.h"
#include "dynarmic/backend/arm64/devirtualize.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/interface/A32/coprocessor.h"

namespace Dynarmic::A32 {
using namespace Backend::Arm64;

// Keep the devirtualized targets and this adjustments as the compatibility
// key, rather than retaining a possibly destroyed first executor's object.
template<auto... Functions>
static auto CallbackBindings(UserCallbacks* callbacks) {
    return std::array{std::pair{Devirtualize<Functions>(callbacks).fn_ptr,
                              Devirtualize<Functions>(callbacks).this_ptr - reinterpret_cast<u64>(callbacks)}...};
}
static auto RuntimeCallbackBindings(UserCallbacks* callbacks) {
    return CallbackBindings<&UserCallbacks::MemoryRead8, &UserCallbacks::MemoryRead16,
        &UserCallbacks::MemoryRead32, &UserCallbacks::MemoryRead64,
        &UserCallbacks::MemoryWrite8, &UserCallbacks::MemoryWrite16,
        &UserCallbacks::MemoryWrite32, &UserCallbacks::MemoryWrite64,
        &UserCallbacks::MemorySwap8, &UserCallbacks::MemorySwap32,
        &UserCallbacks::MemoryWriteExclusive8, &UserCallbacks::MemoryWriteExclusive16,
        &UserCallbacks::MemoryWriteExclusive32, &UserCallbacks::MemoryWriteExclusive64,
        &UserCallbacks::MemoryReadExclusive, &UserCallbacks::CallSVC,
        &UserCallbacks::ExceptionRaised, &UserCallbacks::InstructionSynchronizationBarrierRaised,
        &UserCallbacks::AddTicks, &UserCallbacks::GetTicksRemaining>(callbacks);
}

struct NativeCodeSlab::Impl : Backend::NativeCodeSlabLifetime<Impl, A32JitState, FastDispatchCache::Entry> {
    void initialize(UserConfig config, Jit*, void*, const void* (*lookup)(void*), void* lookup_arg, bool shared) {
        std::lock_guard lock{mutex};
        if (!shared || !config.callbacks_link || !config.lookup_link ||
            !config.runtime_config_link || !config.fast_dispatch_table_link ||
            !config.coprocessor_user_arg_link || config.fastmem_pointer ||
            (config.page_table && !config.page_table_link) ||
            (config.read_page_table && !config.read_page_table_link)) {
            throw std::invalid_argument{"shared ARM64 code requires linked runtime state"};
        }
        const auto bindings = RuntimeCallbackBindings(config.callbacks);
        if (initialized) {
            const auto& old = *conf;
            if (config.code_cache_size != old.code_cache_size || config.arch_version != old.arch_version ||
                config.fast_compilation != old.fast_compilation ||
                config.optimizations != old.optimizations || config.unsafe_optimizations != old.unsafe_optimizations ||
                config.define_unpredictable_behaviour != old.define_unpredictable_behaviour ||
                config.hook_hint_instructions != old.hook_hint_instructions || config.hook_isb != old.hook_isb ||
                config.check_halt_on_memory_access != old.check_halt_on_memory_access ||
                config.enable_cycle_counting != old.enable_cycle_counting ||
                config.always_little_endian != old.always_little_endian ||
                config.wall_clock_cntpct != old.wall_clock_cntpct ||
                bool(config.page_table) != bool(old.page_table) ||
                bool(config.read_page_table) != bool(old.read_page_table) ||
                config.absolute_offset_page_table != old.absolute_offset_page_table ||
                config.page_table_pointer_mask_bits != old.page_table_pointer_mask_bits ||
                config.detect_misaligned_access_via_page_table != old.detect_misaligned_access_via_page_table ||
                config.only_detect_misalignment_via_page_table_on_page_boundary != old.only_detect_misalignment_via_page_table_on_page_boundary ||
                bindings != callback_bindings) {
                throw std::invalid_argument{"ARM64 native code slab configuration mismatch"};
            }
            for (size_t i = 0; i < config.coprocessors.size(); ++i) {
                const auto* candidate = config.coprocessors[i].get();
                const auto* original = old.coprocessors[i].get();
                if (bool(candidate) != bool(original) ||
                    (candidate && typeid(*candidate) != typeid(*original)))
                    throw std::invalid_argument{"ARM64 native coprocessor configuration mismatch"};
            }
            return;
        }
        config.native_code_slab = nullptr;
        emitter = std::make_unique<A32AddressSpace>(config, lookup, lookup_arg);
        callback_bindings = bindings;
        conf = std::move(config);
        initialized = true;
    }

    bool find_block(u64 descriptor, u64 expected, BlockDescriptor& result) const {
        std::lock_guard lock{mutex};
        if (!initialized || clear_pending || !pending_ranges.empty() || expected != current_generation)
            return false;
        if (auto ptr = emitter->Get(IR::LocationDescriptor{descriptor})) {
            result = {ptr, 0, current_generation};
            return true;
        }
        return false;
    }

    BlockDescriptor emit(IR::Block& block, u64 expected, const UserConfig* source = nullptr) {
        std::lock_guard lock{mutex};
        if (clear_pending || !pending_ranges.empty() || expected != current_generation)
            return {};
        if (auto ptr = emitter->Get(block.Location()))
            return {ptr, 0, current_generation, false};
        if (emitter->SpaceRemaining() < 1024U * 1024U) {
            request_generation_transition(GenerationTransitionKind::RecycleSegment);
            return {};
        }
        const auto& emitted = emitter->EmitShared(block, source ? *source : *conf);
        // EmitShared synchronizes the new target before any incoming branch
        // can expose it to another active executor. Keep address loads deferred.
        if (active_executions != 0)
            emitter->PublishBranchLinks(block.Location());
        pending_direct_link_targets.insert(block.Location());
        finish_pending_direct_link_publication();
        return {emitted.entrypoint, emitted.size, current_generation, true, emitted.native_template};
    }

    BlockDescriptor import_template(const NativeCodeTemplate& native_template, u64 expected, const UserConfig& source) {
        std::lock_guard lock{mutex};
        if (!initialized || clear_pending || !pending_ranges.empty() || expected != current_generation)
            return {};
        const IR::LocationDescriptor location{native_template.LocationDescriptor()};
        if (auto ptr = emitter->Get(location))
            return {ptr, 0, current_generation, false};
        if (emitter->SpaceRemaining() < 1024U * 1024U) {
            request_generation_transition(GenerationTransitionKind::RecycleSegment);
            return {};
        }
        const auto emitted = emitter->ImportTemplate(native_template, source);
        if (!emitted.entrypoint)
            return {};
        if (active_executions != 0)
            emitter->PublishBranchLinks(location);
        pending_direct_link_targets.insert(location);
        finish_pending_direct_link_publication();
        return {emitted.entrypoint, emitted.size, current_generation, true};
    }

    size_t space_remaining() const {
        std::lock_guard lock{mutex};
        return emitter ? emitter->SpaceRemaining() : 0;
    }
    // Oaknut reserves the executable mapping at construction and commits it
    // through normal page faults as emission writes it.
    void ensure_memory_committed(size_t) {}

    void halt_executor(A32JitState& state) {
        Atomic::Or(state.halt_reason, static_cast<u32>(HaltReason::CacheInvalidation));
    }
    void clear_executor_tables() {
        for (const auto& executor : executors)
            if (executor.table)
                std::fill_n(executor.table, FastDispatchCache::entry_count, FastDispatchCache::Entry{});
    }
    void finish_pending_direct_link_publication() {
        if (active_executions != 0 || pending_direct_link_targets.empty())
            return;
        for (auto descriptor : pending_direct_link_targets)
            if (emitter->Get(descriptor))
                emitter->PublishLinks(descriptor);
        pending_direct_link_targets.clear();
    }
    void finish_pending_invalidation() {
        if (active_executions != 0)
            return;
        if (clear_pending) {
            const auto stats = emitter->SharedCacheStats();
            const auto used = emitter->GetCodeCacheUsed();
            if (pending_transition == GenerationTransitionKind::RecycleSegment) {
                ++segment_recycles;
                recycled_descriptors += stats.descriptor_count;
                recycled_code_bytes += used;
            } else {
                ++full_generation_clears;
            }
            clear_executor_tables();
            emitter->ClearCache();
            pending_direct_link_targets.clear();
            current_generation = pending_generation;
            published_generation.store(current_generation, std::memory_order_release);
            pending_ranges.clear();
            clear_pending = false;
            generation_changed.notify_all();
        } else if (!pending_ranges.empty()) {
            const auto locations = emitter->InvalidateCacheRanges(pending_ranges);
            for (const auto& executor : executors) {
                if (executor.table) {
                    for (const auto descriptor : locations)
                        FastDispatchCache::Invalidate(executor.table, descriptor);
                }
            }
            pending_ranges.clear();
            generation_changed.notify_all();
        }
    }
    HaltReason run_code(void* state, const void* entry) const {
        return emitter->RunCode(reinterpret_cast<CodePtr>(const_cast<void*>(entry)), state,
                                static_cast<A32JitState*>(state)->halt_reason, false);
    }
    HaltReason step_code(void* state, const void* entry) const {
        return emitter->RunCode(reinterpret_cast<CodePtr>(const_cast<void*>(entry)), state,
                                static_cast<A32JitState*>(state)->halt_reason, true);
    }
    const void* return_from_run_code() const { return emitter->ReturnFromRunCode(); }
    size_t code_cache_used() const {
        std::lock_guard lock{mutex};
        return emitter ? emitter->GetCodeCacheUsed() : 0;
    }
    CacheStats GetCacheStats() const {
        std::lock_guard lock{mutex};
        auto stats = emitter ? emitter->SharedCacheStats() : CacheStats{};
        stats.segment_recycles = segment_recycles;
        stats.recycled_descriptors = recycled_descriptors;
        stats.recycled_code_bytes = recycled_code_bytes;
        stats.full_generation_clears = full_generation_clears;
        return stats;
    }
    void dump_disassembly() const {
        std::lock_guard lock{mutex};
        if (emitter) emitter->DumpDisassembly();
    }
    std::vector<std::string> disassemble() const { return {}; }
    bool has_host_feature_sha() const { return false; }

    std::optional<UserConfig> conf;
    std::unique_ptr<A32AddressSpace> emitter;
    decltype(RuntimeCallbackBindings(nullptr)) callback_bindings{};
    tsl::robin_set<IR::LocationDescriptor> pending_direct_link_targets;
    u64 segment_recycles{}, recycled_descriptors{}, recycled_code_bytes{}, full_generation_clears{};
};

NativeCodeSlab::BlockDescriptor NativeCodeSlab::import_template(
        const NativeCodeTemplate& native_template, u64 generation, const UserConfig& source) {
    return impl->import_template(native_template, generation, source);
}

#include "dynarmic/backend/native_code_slab_interface.inc"
} // namespace Dynarmic::A32
