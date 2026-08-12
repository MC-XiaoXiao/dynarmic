/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include <boost/icl/interval_set.hpp>
#include <fmt/format.h>
#include <mcl/assert.hpp>
#include <mcl/bit_cast.hpp>
#include <mcl/scope_exit.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/backend/x64/a32_emit_x64.h"
#include "dynarmic/backend/x64/a32_jitstate.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/callback.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/jitstate_info.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/common/x64_disassemble.h"
#include "dynarmic/frontend/A32/translate/a32_translate.h"
#include "dynarmic/interface/A32/a32.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/location_descriptor.h"
#include "dynarmic/ir/opt/passes.h"

namespace Dynarmic::A32 {

using namespace Backend::X64;

[[nodiscard]] static std::uint32_t InclusiveRangeEnd(
        std::uint32_t start_address, std::size_t length) noexcept {
    ASSERT(length != 0);
    const auto offset = static_cast<std::uint64_t>(length - 1U);
    const auto maximum = static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max());
    if (offset > maximum - start_address) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    return static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(start_address) + offset);
}

template<auto callback>
static std::unique_ptr<Callback> GenRuntimeCallback(
        A32::UserCallbacks* cb, const A32::UserConfig& conf) {
    auto direct = Devirtualize<callback>(cb);
    if (conf.callbacks_link) {
        return std::make_unique<ArgCallbackFromLink>(
                std::move(direct), offsetof(A32JitState, callbacks_link));
    }
    return std::make_unique<ArgCallback>(std::move(direct));
}

static RunCodeCallbacks GenRunCodeCallbacks(A32::UserCallbacks* cb, CodePtr (*LookupBlock)(void* lookup_block_arg), void* arg, const A32::UserConfig& conf) {
    std::unique_ptr<Callback> lookup = std::make_unique<ArgCallback>(
            LookupBlock, reinterpret_cast<u64>(arg));
    if (conf.lookup_link) {
        lookup = std::make_unique<ArgCallbackFromLink>(
                ArgCallback{LookupBlock, 0},
                offsetof(A32JitState, lookup_link));
    }
    return RunCodeCallbacks{
        std::move(lookup),
        GenRuntimeCallback<&A32::UserCallbacks::AddTicks>(cb, conf),
        GenRuntimeCallback<&A32::UserCallbacks::GetTicksRemaining>(cb, conf),
        conf.enable_cycle_counting,
    };
}

static std::function<void(BlockOfCode&)> GenRCP(const A32::UserConfig& conf) {
    return [conf](BlockOfCode& code) {
        if (conf.page_table) {
            if (conf.page_table_link) {
                code.mov(code.r14,
                         code.qword[code.r15 +
                                    offsetof(A32JitState, page_table_link)]);
                code.mov(code.r14, code.qword[code.r14]);
            } else {
                code.mov(code.r14, mcl::bit_cast<u64>(conf.page_table));
            }
        }
        if (conf.read_page_table &&
            conf.read_page_table != conf.page_table &&
            !conf.fastmem_pointer) {
            if (conf.read_page_table_link) {
                code.mov(
                    code.r13,
                    code.qword[code.r15 +
                               offsetof(A32JitState, read_page_table_link)]);
                code.mov(code.r13, code.qword[code.r13]);
            } else {
                code.mov(code.r13, mcl::bit_cast<u64>(conf.read_page_table));
            }
        } else if (conf.fastmem_pointer) {
            code.mov(code.r13, *conf.fastmem_pointer);
        }
    };
}

static Optimization::PolyfillOptions GenPolyfillOptions(bool has_sha) {
    return Optimization::PolyfillOptions{
        .sha256 = !has_sha,
        .vector_multiply_widen = true,
    };
}

static A32::UserConfig WithFastDispatchTable(A32::UserConfig conf, void* storage) {
    conf.fast_dispatch_table_storage = storage;
    return conf;
}

struct NativeCodeSlab::Impl {
    using BlockDescriptor = NativeCodeSlab::BlockDescriptor;

    void initialize(A32::UserConfig config, A32::Jit* jit_interface,
                    void* jit_state, const void* (*lookup)(void*),
                    void* lookup_arg, bool shared) {
        std::lock_guard lock{mutex};
        if (initialized) {
            if (shared_mode != shared || config.code_cache_size != code_cache_size ||
                config.arch_version != conf->arch_version ||
                config.optimizations != conf->optimizations ||
                config.unsafe_optimizations != conf->unsafe_optimizations ||
                config.define_unpredictable_behaviour !=
                    conf->define_unpredictable_behaviour ||
                config.hook_hint_instructions != conf->hook_hint_instructions ||
                config.check_halt_on_memory_access !=
                    conf->check_halt_on_memory_access ||
                config.enable_cycle_counting != conf->enable_cycle_counting ||
                config.always_little_endian != conf->always_little_endian) {
                throw std::invalid_argument{
                    "native code slab configuration mismatch"};
            }
            return;
        }

        if (shared) {
            if (config.callbacks_link == nullptr ||
                config.lookup_link == nullptr ||
                config.runtime_config_link == nullptr ||
                config.fast_dispatch_table_link == nullptr ||
                config.coprocessor_user_arg_link == nullptr ||
                config.exclusive_monitor_lock_link == nullptr ||
                config.exclusive_monitor_addresses_link == nullptr ||
                config.exclusive_monitor_values_link == nullptr ||
                config.fastmem_pointer.has_value() ||
                (config.page_table != nullptr &&
                 config.page_table_link == nullptr) ||
                (config.read_page_table != nullptr &&
                 config.read_page_table_link == nullptr)) {
                throw std::invalid_argument{
                    "shared native code slab requires linked runtime state"};
            }
            config.fast_dispatch_table_storage = nullptr;
        }

        config.native_code_slab = nullptr;
        if (!config.fast_dispatch_table_storage) {
            owned_fast_dispatch_table =
                std::make_unique<A32EmitX64::FastDispatchEntry[]>(
                    A32EmitX64::fast_dispatch_table_size);
            config.fast_dispatch_table_storage =
                owned_fast_dispatch_table.get();
        }

        block_of_code = std::make_unique<BlockOfCode>(
            GenRunCodeCallbacks(config.callbacks, lookup, lookup_arg, config),
            JitStateInfo{*static_cast<A32JitState*>(jit_state)},
            config.code_cache_size, GenRCP(config));
        emitter = std::make_unique<A32EmitX64>(
            *block_of_code, WithFastDispatchTable(
                                config, config.fast_dispatch_table_storage),
            jit_interface);
        polyfill_options = GenPolyfillOptions(
            block_of_code->HasHostFeature(HostFeature::SHA));
        conf = std::move(config);
        code_cache_size = conf->code_cache_size;
        shared_mode = shared;
        initialized = true;
    }

    [[nodiscard]] std::uint64_t generation() const {
        std::unique_lock lock{mutex};
        generation_changed.wait(lock, [this] {
            return !clear_pending && pending_ranges.empty();
        });
        return current_generation;
    }

    [[nodiscard]] bool find_block(std::uint64_t location_descriptor,
                                  std::uint64_t expected_generation,
                                  BlockDescriptor& result) const {
        std::lock_guard lock{mutex};
        if (!initialized) {
            return false;
        }
        if (clear_pending) {
            return false;
        }
        if (!pending_ranges.empty()) {
            return false;
        }
        if (expected_generation != current_generation) {
            return false;
        }
        const auto block = emitter->GetBasicBlock(
            IR::LocationDescriptor{location_descriptor});
        if (!block) {
            return false;
        }
        result = BlockDescriptor{
            block->entrypoint, block->size, current_generation};
        return true;
    }

    [[nodiscard]] BlockDescriptor emit(
            IR::Block& block, std::uint64_t expected_generation) {
        std::lock_guard lock{mutex};
        if (clear_pending || !pending_ranges.empty() ||
            expected_generation != current_generation) {
            return {};
        }
        if (const auto existing = emitter->GetBasicBlock(block.Location())) {
            return BlockDescriptor{
                existing->entrypoint, existing->size, current_generation,
                false};
        }
        const auto result = emitter->Emit(block);
        return BlockDescriptor{
            result.entrypoint, result.size, current_generation,
            result.entrypoint != nullptr};
    }

    [[nodiscard]] std::size_t space_remaining() const {
        std::lock_guard lock{mutex};
        return block_of_code->SpaceRemaining();
    }

    void ensure_memory_committed(std::size_t codesize) {
        std::lock_guard lock{mutex};
        block_of_code->EnsureMemoryCommitted(codesize);
    }

    void register_executor(void* storage, void* jit_state) {
        if (storage == nullptr || jit_state == nullptr) return;
        std::lock_guard lock{mutex};
        auto* const table = static_cast<A32EmitX64::FastDispatchEntry*>(
            storage);
        auto* const state = static_cast<A32JitState*>(jit_state);
        const auto existing = std::find_if(
            executors.begin(), executors.end(),
            [table, state](const Executor& executor) {
                return executor.table == table || executor.state == state;
            });
        if (existing == executors.end()) {
            executors.push_back(Executor{table, state});
        }
    }

    void unregister_executor(void* storage, void* jit_state) {
        if (storage == nullptr || jit_state == nullptr) return;
        std::lock_guard lock{mutex};
        auto* const table = static_cast<A32EmitX64::FastDispatchEntry*>(
            storage);
        auto* const state = static_cast<A32JitState*>(jit_state);
        const auto existing = std::find_if(
            executors.begin(), executors.end(),
            [table, state](const Executor& executor) {
                return executor.table == table && executor.state == state;
            });
        if (existing == executors.end()) return;
        if (existing->active) {
            throw std::logic_error{
                "cannot unregister an active native code slab executor"};
        }
        executors.erase(existing);
    }

    void clear_executor_fast_dispatch_tables() {
        for (const auto& executor : executors) {
            std::fill_n(executor.table, A32EmitX64::fast_dispatch_table_size,
                        A32EmitX64::FastDispatchEntry{});
        }
    }

    void clear_executor_range_state(
            const tsl::robin_set<IR::LocationDescriptor>& locations) {
        for (const auto& executor : executors) {
            for (const auto& location : locations) {
                const auto value = location.Value();
                auto& entry = executor.table[
                    A32EmitX64::fast_dispatch_table_index(value) /
                    sizeof(A32EmitX64::FastDispatchEntry)];
                if (entry.location_descriptor == value) {
                    entry = {};
                }
            }

            bool reset_rsb = false;
            for (const auto value : executor.state->rsb_location_descriptors) {
                if (std::any_of(
                        locations.begin(), locations.end(),
                        [value](const auto& location) {
                            return location.Value() == value;
                        })) {
                    reset_rsb = true;
                    break;
                }
            }
            if (reset_rsb) {
                executor.state->ResetRSB();
            }
        }
    }

    void finish_pending_invalidation() {
        if (active_executions != 0) return;
        if (clear_pending) {
            clear_executor_fast_dispatch_tables();
            emitter->ClearCache();
            block_of_code->ClearCache();
            pending_ranges.clear();
            current_generation = pending_generation;
            clear_pending = false;
            generation_changed.notify_all();
            return;
        }
        if (pending_ranges.empty()) return;

        const auto locations = emitter->InvalidateCacheRanges(pending_ranges);
        clear_executor_range_state(locations);
        pending_ranges.clear();
        generation_changed.notify_all();
    }

    void request_generation_transition(bool finish = true) {
        if (!clear_pending) {
            clear_pending = true;
            pending_generation = current_generation + 1;
            pending_ranges.clear();
        }
        for (const auto& executor : executors) {
            if (executor.active) {
                Atomic::Or(&executor.state->halt_reason,
                           static_cast<u32>(HaltReason::CacheInvalidation));
            }
        }
        if (finish) finish_pending_invalidation();
    }

    void clear_cache() {
        std::lock_guard lock{mutex};
        if (!initialized) return;
        request_generation_transition();
    }

    void request_range_transition(std::uint32_t start_address,
                                  std::size_t length, bool finish = true) {
        if (length == 0 || clear_pending) return;
        const auto last_address = InclusiveRangeEnd(start_address, length);
        pending_ranges.add(boost::icl::discrete_interval<u32>::closed(
            start_address, last_address));
        for (const auto& executor : executors) {
            if (executor.active) {
                Atomic::Or(&executor.state->halt_reason,
                           static_cast<u32>(HaltReason::CacheInvalidation));
            }
        }
        if (finish) finish_pending_invalidation();
    }

    void invalidate_cache_range(std::uint32_t start_address,
                                std::size_t length) {
        std::lock_guard lock{mutex};
        if (!initialized) return;
        request_range_transition(start_address, length);
    }

    void request_cache_clear() {
        std::lock_guard lock{mutex};
        if (!initialized) return;
        request_generation_transition(false);
    }

    void request_cache_range(std::uint32_t start_address,
                             std::size_t length) {
        std::lock_guard lock{mutex};
        if (!initialized) return;
        request_range_transition(start_address, length, false);
    }

    void service_pending_invalidation() {
        std::lock_guard lock{mutex};
        if (!initialized) return;
        finish_pending_invalidation();
    }

    [[nodiscard]] HaltReason run_code(void* jit_state,
                                      const void* code_ptr) const {
        return block_of_code->RunCode(jit_state, code_ptr);
    }

    [[nodiscard]] HaltReason step_code(void* jit_state,
                                       const void* code_ptr) const {
        return block_of_code->StepCode(jit_state, code_ptr);
    }

    [[nodiscard]] const void* return_from_run_code() const {
        return block_of_code->GetReturnFromRunCodeAddress();
    }

    [[nodiscard]] std::size_t code_cache_used() const {
        std::lock_guard lock{mutex};
        return reinterpret_cast<const char*>(block_of_code->getCurr()) -
               reinterpret_cast<const char*>(block_of_code->GetCodeBegin());
    }

    void dump_disassembly() const {
        std::lock_guard lock{mutex};
        const auto size = reinterpret_cast<const char*>(block_of_code->getCurr()) -
                          reinterpret_cast<const char*>(block_of_code->GetCodeBegin());
        Common::DumpDisassembledX64(block_of_code->GetCodeBegin(), size);
    }

    [[nodiscard]] std::vector<std::string> disassemble() const {
        std::lock_guard lock{mutex};
        const auto size = reinterpret_cast<const char*>(block_of_code->getCurr()) -
                          reinterpret_cast<const char*>(block_of_code->GetCodeBegin());
        return Common::DisassembleX64(block_of_code->GetCodeBegin(), size);
    }

    [[nodiscard]] bool has_host_feature_sha() const {
        std::lock_guard lock{mutex};
        return block_of_code->HasHostFeature(HostFeature::SHA);
    }

    [[nodiscard]] std::uint64_t enter_execution(void* jit_state) {
        std::unique_lock lock{mutex};
        generation_changed.wait(lock, [this] {
            return !clear_pending && pending_ranges.empty();
        });
        auto* const state = static_cast<A32JitState*>(jit_state);
        const auto executor = std::find_if(
            executors.begin(), executors.end(),
            [state](const Executor& candidate) {
                return candidate.state == state;
            });
        if (executor == executors.end() || executor->active) {
            throw std::logic_error{
                "native code slab executor registration mismatch"};
        }
        executor->active = true;
        executor->generation = current_generation;
        ++active_executions;
        return current_generation;
    }

    void leave_execution(void* jit_state, std::uint64_t generation) {
        std::lock_guard lock{mutex};
        auto* const state = static_cast<A32JitState*>(jit_state);
        const auto executor = std::find_if(
            executors.begin(), executors.end(),
            [state](const Executor& candidate) {
                return candidate.state == state;
            });
        if (executor == executors.end() || !executor->active ||
            executor->generation != generation ||
            generation != current_generation || active_executions == 0) {
            throw std::logic_error{"native code slab execution underflow"};
        }
        executor->active = false;
        --active_executions;
        finish_pending_invalidation();
    }

    struct Executor {
        A32EmitX64::FastDispatchEntry* table{};
        A32JitState* state{};
        std::uint64_t generation{};
        bool active{};
    };

    mutable std::recursive_mutex mutex;
    std::optional<A32::UserConfig> conf;
    std::unique_ptr<A32EmitX64::FastDispatchEntry[]>
        owned_fast_dispatch_table;
    std::unique_ptr<BlockOfCode> block_of_code;
    std::unique_ptr<A32EmitX64> emitter;
    std::vector<Executor> executors;
    Optimization::PolyfillOptions polyfill_options{};
    std::size_t code_cache_size{};
    std::size_t active_executions{};
    std::uint64_t current_generation{1};
    std::uint64_t pending_generation{1};
    boost::icl::interval_set<u32> pending_ranges;
    bool initialized{};
    bool shared_mode{};
    bool clear_pending{};
    mutable std::condition_variable_any generation_changed;
};

NativeCodeSlab::NativeCodeSlab() : impl{std::make_unique<Impl>()} {}

NativeCodeSlab::~NativeCodeSlab() = default;

void NativeCodeSlab::initialize(
    UserConfig conf, Jit* jit_interface, void* jit_state,
    const void* (*lookup)(void*), void* lookup_arg, bool shared_mode) {
    impl->initialize(std::move(conf), jit_interface, jit_state, lookup,
                     lookup_arg, shared_mode);
}

std::uint64_t NativeCodeSlab::generation() const {
    return impl->generation();
}

bool NativeCodeSlab::find_block(
    std::uint64_t location_descriptor, std::uint64_t expected_generation,
    BlockDescriptor& result) const {
    return impl->find_block(
        location_descriptor, expected_generation, result);
}

NativeCodeSlab::BlockDescriptor NativeCodeSlab::emit(
        IR::Block& block, std::uint64_t expected_generation) {
    return impl->emit(block, expected_generation);
}

std::size_t NativeCodeSlab::space_remaining() const {
    return impl->space_remaining();
}

void NativeCodeSlab::ensure_memory_committed(std::size_t codesize) {
    impl->ensure_memory_committed(codesize);
}

void NativeCodeSlab::register_executor(void* storage, void* jit_state) {
    impl->register_executor(storage, jit_state);
}

void NativeCodeSlab::unregister_executor(void* storage, void* jit_state) {
    impl->unregister_executor(storage, jit_state);
}

void NativeCodeSlab::clear_cache() {
    impl->clear_cache();
}

void NativeCodeSlab::invalidate_cache_range(
    std::uint32_t start_address, std::size_t length) {
    impl->invalidate_cache_range(start_address, length);
}

void NativeCodeSlab::request_cache_clear() {
    impl->request_cache_clear();
}

void NativeCodeSlab::request_cache_range(
    std::uint32_t start_address, std::size_t length) {
    impl->request_cache_range(start_address, length);
}

void NativeCodeSlab::service_pending_invalidation() {
    impl->service_pending_invalidation();
}

HaltReason NativeCodeSlab::run_code(void* jit_state,
                                    const void* code_ptr) const {
    return impl->run_code(jit_state, code_ptr);
}

HaltReason NativeCodeSlab::step_code(void* jit_state,
                                     const void* code_ptr) const {
    return impl->step_code(jit_state, code_ptr);
}

const void* NativeCodeSlab::return_from_run_code() const {
    return impl->return_from_run_code();
}

std::size_t NativeCodeSlab::code_cache_used() const {
    return impl->code_cache_used();
}

void NativeCodeSlab::dump_disassembly() const {
    impl->dump_disassembly();
}

std::vector<std::string> NativeCodeSlab::disassemble() const {
    return impl->disassemble();
}

bool NativeCodeSlab::has_host_feature_sha() const {
    return impl->has_host_feature_sha();
}

std::uint64_t NativeCodeSlab::enter_execution(void* jit_state) {
    return impl->enter_execution(jit_state);
}

void NativeCodeSlab::leave_execution(
        void* jit_state, std::uint64_t generation) {
    impl->leave_execution(jit_state, generation);
}

struct Jit::Impl {
    Impl(Jit* jit, A32::UserConfig conf)
            : fast_dispatch_table_storage(std::make_unique<A32EmitX64::FastDispatchEntry[]>(A32EmitX64::fast_dispatch_table_size))
            , conf(std::move(conf))
            , jit_interface(jit) {
        std::fill_n(
            fast_dispatch_table_storage.get(),
            A32EmitX64::fast_dispatch_table_size,
            A32EmitX64::FastDispatchEntry{});
        this->conf.fast_dispatch_table_storage =
            fast_dispatch_table_storage.get();
        if (this->conf.native_code_slab == nullptr) {
            owned_native_code_slab = std::make_unique<NativeCodeSlab>();
            native_code_slab = owned_native_code_slab.get();
            native_code_slab_shared = false;
        } else {
            native_code_slab = this->conf.native_code_slab;
            native_code_slab_shared = true;
        }
        native_code_slab->initialize(
            this->conf, jit, &jit_state, &GetCurrentBlockThunk, this,
            native_code_slab_shared);
        polyfill_options = GenPolyfillOptions(
            native_code_slab->has_host_feature_sha());
        if (this->conf.lookup_link) {
            this->conf.lookup_link->store(
                    reinterpret_cast<u64>(this),
                    std::memory_order_release);
        }
        if (this->conf.runtime_config_link) {
            this->conf.runtime_config_link->store(
                    reinterpret_cast<u64>(&this->conf),
                    std::memory_order_release);
        }
        if (this->conf.fast_dispatch_table_link) {
            this->conf.fast_dispatch_table_link->store(
                reinterpret_cast<u64>(fast_dispatch_table_storage.get()),
                std::memory_order_release);
        }
        BindExecutionContext();
        native_code_slab->register_executor(
            fast_dispatch_table_storage.get(), &jit_state);
    }

    ~Impl() {
        if (native_code_slab != nullptr) {
            native_code_slab->unregister_executor(
                fast_dispatch_table_storage.get(), &jit_state);
        }
    }

    HaltReason Run() {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));

        const auto execution_generation =
            native_code_slab->enter_execution(&jit_state);
        if (observed_generation != execution_generation) {
            jit_state.ResetRSB();
            observed_generation = execution_generation;
        }
        active_generation = execution_generation;
        jit_interface->is_executing = true;
        SCOPE_EXIT {
            jit_interface->is_executing = false;
            active_generation = 0;
            native_code_slab->leave_execution(
                &jit_state, execution_generation);
        };

        const CodePtr current_codeptr = [this, execution_generation] {
            // RSB optimization
            const u32 new_rsb_ptr = (jit_state.rsb_ptr - 1) & A32JitState::RSBPtrMask;
            if (jit_state.GetUniqueHash() == jit_state.rsb_location_descriptors[new_rsb_ptr] &&
                jit_state.rsb_codeptrs[new_rsb_ptr] != 0) {
                ++jit_state.rsb_hits;
                jit_state.rsb_ptr = new_rsb_ptr;
                return reinterpret_cast<CodePtr>(jit_state.rsb_codeptrs[new_rsb_ptr]);
            }

            ++jit_state.rsb_misses;
            return GetCurrentBlock(execution_generation);
        }();

        const HaltReason hr = native_code_slab->run_code(
            &jit_state, current_codeptr);

        PerformRequestedCacheInvalidation(hr);

        return hr;
    }

    HaltReason Step() {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));

        const auto execution_generation =
            native_code_slab->enter_execution(&jit_state);
        if (observed_generation != execution_generation) {
            jit_state.ResetRSB();
            observed_generation = execution_generation;
        }
        active_generation = execution_generation;
        jit_interface->is_executing = true;
        SCOPE_EXIT {
            jit_interface->is_executing = false;
            active_generation = 0;
            native_code_slab->leave_execution(
                &jit_state, execution_generation);
        };

        const HaltReason hr = native_code_slab->step_code(
            &jit_state, GetCurrentSingleStep(execution_generation));

        PerformRequestedCacheInvalidation(hr);

        return hr;
    }

    bool Precompile(u64 location_descriptor) {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        const auto generation = native_code_slab->generation();
        return GetBasicBlock(
            IR::LocationDescriptor{location_descriptor}, generation)
            .newly_emitted;
    }

    void GeneratePortableIR(u64 location_descriptor) {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        const IR::LocationDescriptor descriptor{location_descriptor};
        const auto translation_started = std::chrono::steady_clock::now();
        auto ir_block = TranslateBlock(descriptor);
        const auto translation_nanoseconds = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - translation_started)
                        .count());
        conf.callbacks->PortableIRGenerated(
                descriptor.Value(), translation_nanoseconds, ir_block);
    }

    bool Precompile(IR::Block block) {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        NativeCodeSlab::BlockDescriptor existing;
        const auto generation = native_code_slab->generation();
        if (native_code_slab->find_block(
                block.Location().Value(), generation, existing)) {
            return false;
        }

        constexpr size_t MINIMUM_REMAINING_CODESIZE = 1 * 1024 * 1024;
        if (native_code_slab->space_remaining() <
            MINIMUM_REMAINING_CODESIZE) {
            invalidate_entire_cache = true;
            PerformRequestedCacheInvalidation(HaltReason::CacheInvalidation);
        }
        native_code_slab->ensure_memory_committed(
            MINIMUM_REMAINING_CODESIZE);
        return native_code_slab->emit(block, generation).newly_emitted;
    }

    void ClearCache() {
        std::unique_lock lock{invalidation_mutex};
        invalidate_entire_cache = true;
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void InvalidateCacheRange(std::uint32_t start_address, std::size_t length) {
        std::unique_lock lock{invalidation_mutex};
        if (length == 0) return;
        invalid_cache_ranges.add(boost::icl::discrete_interval<u32>::closed(
                start_address, InclusiveRangeEnd(start_address, length)));
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void Reset() {
        ASSERT(!jit_interface->is_executing);
        jit_state = {};
        BindExecutionContext();
    }

    void HaltExecution(HaltReason hr) {
        Atomic::Or(&jit_state.halt_reason, static_cast<u32>(hr));
    }

    void ClearHalt(HaltReason hr) {
        Atomic::And(&jit_state.halt_reason, ~static_cast<u32>(hr));
    }

    void ClearExclusiveState() {
        jit_state.exclusive_state = 0;
    }

    std::array<u32, 16>& Regs() {
        return jit_state.Reg;
    }

    const std::array<u32, 16>& Regs() const {
        return jit_state.Reg;
    }

    std::array<u32, 64>& ExtRegs() {
        return jit_state.ExtReg;
    }

    const std::array<u32, 64>& ExtRegs() const {
        return jit_state.ExtReg;
    }

    u32 Cpsr() const {
        return jit_state.Cpsr();
    }

    void SetCpsr(u32 value) {
        return jit_state.SetCpsr(value);
    }

    u32 Fpscr() const {
        return jit_state.Fpscr();
    }

    void SetFpscr(u32 value) {
        return jit_state.SetFpscr(value);
    }

    void DumpDisassembly() const {
        native_code_slab->dump_disassembly();
    }

    size_t CodeCacheUsed() const {
        return native_code_slab->code_cache_used();
    }

    DispatchCounters GetDispatchCounters() const {
        return DispatchCounters{
            jit_state.stable_link_hits,
            jit_state.stable_link_misses,
            jit_state.rsb_hits,
            jit_state.rsb_misses,
        };
    }

    std::vector<std::string> Disassemble() const {
        return native_code_slab->disassemble();
    }

private:
    void BindExecutionContext() {
        jit_state.callbacks_link = conf.callbacks_link;
        jit_state.lookup_link = conf.lookup_link;
        jit_state.runtime_config_link = conf.runtime_config_link;
        jit_state.fast_dispatch_table_link = conf.fast_dispatch_table_link;
        jit_state.page_table_link = conf.page_table_link;
        jit_state.read_page_table_link = conf.read_page_table_link;
        jit_state.coprocessor_user_arg_link =
                conf.coprocessor_user_arg_link;
        jit_state.exclusive_monitor_lock_link =
                conf.exclusive_monitor_lock_link;
        jit_state.exclusive_monitor_addresses_link =
                conf.exclusive_monitor_addresses_link;
        jit_state.exclusive_monitor_values_link =
                conf.exclusive_monitor_values_link;
    }

    static CodePtr GetCurrentBlockThunk(void* this_voidptr) {
        Jit::Impl& this_ = *static_cast<Jit::Impl*>(this_voidptr);
        return this_.GetCurrentBlock(this_.active_generation);
    }

    IR::LocationDescriptor GetCurrentLocation() const {
        return IR::LocationDescriptor{jit_state.GetUniqueHash()};
    }

    CodePtr GetCurrentBlock(std::uint64_t generation) {
        return GetBasicBlock(GetCurrentLocation(), generation).entrypoint;
    }

    CodePtr GetCurrentSingleStep(std::uint64_t generation) {
        return GetBasicBlock(
            A32::LocationDescriptor{GetCurrentLocation()}
                .SetSingleStepping(true),
            generation).entrypoint;
    }

    NativeCodeSlab::BlockDescriptor GetBasicBlock(
        IR::LocationDescriptor descriptor, std::uint64_t generation) {
        NativeCodeSlab::BlockDescriptor block;
        if (native_code_slab->find_block(
                descriptor.Value(), generation, block)) {
            return block;
        }

        const auto translation_started = std::chrono::steady_clock::now();

        constexpr size_t MINIMUM_REMAINING_CODESIZE = 1 * 1024 * 1024;
        if (native_code_slab->space_remaining() <
            MINIMUM_REMAINING_CODESIZE) {
            invalidate_entire_cache = true;
            if (jit_interface->is_executing) {
                // A shared slab cannot overwrite its code region while any
                // executor is still running from it. Return through the
                // dispatcher and let the host perform the pending clear at a
                // safe execution boundary.
                HaltExecution(HaltReason::CacheInvalidation);
                return NativeCodeSlab::BlockDescriptor{
                    native_code_slab->return_from_run_code(), 0};
            }
            PerformRequestedCacheInvalidation(HaltReason::CacheInvalidation);
            generation = native_code_slab->generation();
        }
        native_code_slab->ensure_memory_committed(
            MINIMUM_REMAINING_CODESIZE);

        IR::Block ir_block = TranslateBlock(descriptor);
        auto emitted = native_code_slab->emit(ir_block, generation);
        if (emitted.entrypoint == nullptr) {
            return NativeCodeSlab::BlockDescriptor{
                native_code_slab->return_from_run_code(), 0, generation};
        }
        const auto translation_nanoseconds = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - translation_started)
                        .count());
        if (emitted.newly_emitted) {
            conf.callbacks->CodeTranslationCompleted(
                    descriptor.Value(), translation_nanoseconds, ir_block);
        }
        return emitted;
    }

    IR::Block TranslateBlock(IR::LocationDescriptor descriptor) {
        IR::Block ir_block = A32::Translate(A32::LocationDescriptor{descriptor}, conf.callbacks, {conf.arch_version, conf.define_unpredictable_behaviour, conf.hook_hint_instructions});
        Optimization::PolyfillPass(ir_block, polyfill_options);
        Optimization::NamingPass(ir_block);
        if (conf.HasOptimization(OptimizationFlag::GetSetElimination) && !conf.check_halt_on_memory_access) {
            Optimization::A32GetSetElimination(ir_block, {.convert_nz_to_nzc = true});
            Optimization::DeadCodeElimination(ir_block);
        }
        if (conf.HasOptimization(OptimizationFlag::ConstProp)) {
            Optimization::A32ConstantMemoryReads(ir_block, conf.callbacks);
            Optimization::ConstantPropagation(ir_block);
            Optimization::DeadCodeElimination(ir_block);
        }
        Optimization::IdentityRemovalPass(ir_block);
        Optimization::VerificationPass(ir_block);
        return ir_block;
    }

    void PerformRequestedCacheInvalidation(HaltReason hr) {
        if (Has(hr, HaltReason::CacheInvalidation)) {
            std::unique_lock lock{invalidation_mutex};

            ClearHalt(HaltReason::CacheInvalidation);

            if (!invalidate_entire_cache && invalid_cache_ranges.empty()) {
                return;
            }

            jit_state.ResetRSB();
            if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
                // Fast-dispatch entries are executor-local even when the
                // emitted native blocks live in a shared slab.  Clear the
                // table before the slab drops its block descriptors so no
                // executor can jump through a stale native address after a
                // mapping invalidation.
                std::fill_n(
                    fast_dispatch_table_storage.get(),
                    A32EmitX64::fast_dispatch_table_size,
                    A32EmitX64::FastDispatchEntry{});
            }
            if (invalidate_entire_cache) {
                native_code_slab->clear_cache();
            } else {
                for (const auto& range : invalid_cache_ranges) {
                    const auto lower = range.lower();
                    const auto upper = range.upper();
                    native_code_slab->invalidate_cache_range(
                        lower, static_cast<std::size_t>(upper - lower) + 1U);
                }
            }
            invalid_cache_ranges.clear();
            invalidate_entire_cache = false;
            // A transition requested after RunCode exchanged halt_reason can
            // set the requesting executor's bit again. It is already at the
            // host boundary, so do not carry that zero-progress halt into the
            // next scheduler slice.
            ClearHalt(HaltReason::CacheInvalidation);
        }
    }

    A32JitState jit_state;
    std::unique_ptr<A32EmitX64::FastDispatchEntry[]> fast_dispatch_table_storage;
    std::unique_ptr<NativeCodeSlab> owned_native_code_slab;
    NativeCodeSlab* native_code_slab{};
    bool native_code_slab_shared{};
    std::uint64_t observed_generation{};
    std::uint64_t active_generation{};
    Optimization::PolyfillOptions polyfill_options;

    A32::UserConfig conf;

    Jit* jit_interface;

    // Requests made during execution to invalidate the cache are queued up here.
    bool invalidate_entire_cache = false;
    boost::icl::interval_set<u32> invalid_cache_ranges;
    std::mutex invalidation_mutex;
};

Jit::Jit(UserConfig conf)
        : impl(std::make_unique<Impl>(this, std::move(conf))) {}

Jit::~Jit() = default;

HaltReason Jit::Run() {
    return impl->Run();
}

HaltReason Jit::Step() {
    return impl->Step();
}

bool Jit::Precompile(std::uint64_t location_descriptor) {
    return impl->Precompile(location_descriptor);
}

void Jit::GeneratePortableIR(std::uint64_t location_descriptor) {
    impl->GeneratePortableIR(location_descriptor);
}

bool Jit::Precompile(IR::Block block) {
    return impl->Precompile(std::move(block));
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

DispatchCounters Jit::GetDispatchCounters() const {
    return impl->GetDispatchCounters();
}

}  // namespace Dynarmic::A32
