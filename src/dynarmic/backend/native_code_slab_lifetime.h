/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <boost/icl/interval_set.hpp>
#include <mcl/assert.hpp>
#include <mcl/stdint.hpp>

namespace Dynarmic::Backend {

[[nodiscard]] inline std::uint32_t InclusiveRangeEnd(
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

// Shared by host backends. Code retirement and multi-instruction patching
// require every executor to leave the published generation. Backend hooks
// may publish independent instruction replacements permitted by the host
// architecture earlier, and own cache maintenance and the block index.
template<class Implementation, class State, class Entry>
class NativeCodeSlabLifetime {
public:
    struct Executor {
        Entry* table{};
        State* state{};
        std::uint64_t generation{};
        bool active{};
    };

    enum class GenerationTransitionKind { RecycleSegment, ClearAll };

    mutable std::recursive_mutex mutex;
    mutable std::condition_variable_any generation_changed;
    std::vector<Executor> executors;
    std::size_t active_executions{};
    std::uint64_t current_generation{1};
    std::atomic<std::uint64_t> published_generation{1};
    std::uint64_t pending_generation{1};
    boost::icl::interval_set<u32> pending_ranges;
    bool initialized{};
    bool clear_pending{};
    GenerationTransitionKind pending_transition{GenerationTransitionKind::ClearAll};
    [[nodiscard]] std::uint64_t generation() const {
        std::unique_lock lock{mutex};
        generation_changed.wait(lock, [this] {
            return !clear_pending && pending_ranges.empty();
        });
        return current_generation;
    }

    [[nodiscard]] std::uint64_t generation_snapshot() const noexcept {
        return published_generation.load(std::memory_order_acquire);
    }

    void register_executor(void* storage, void* jit_state) {
        if (jit_state == nullptr)
            return;
        std::lock_guard lock{mutex};
        auto* const table = static_cast<Entry*>(
            storage);
        auto* const state = static_cast<State*>(jit_state);
        const auto existing = std::find_if(
            executors.begin(), executors.end(),
            [table, state](const Executor& executor) {
                return (table != nullptr && executor.table == table) || executor.state == state;
            });
        if (existing == executors.end()) {
            executors.push_back(Executor{table, state});
        }
    }

    void unregister_executor(void* storage, void* jit_state) {
        if (jit_state == nullptr)
            return;
        std::lock_guard lock{mutex};
        auto* const table = static_cast<Entry*>(
            storage);
        auto* const state = static_cast<State*>(jit_state);
        const auto existing = std::find_if(
            executors.begin(), executors.end(),
            [table, state](const Executor& executor) {
                return executor.table == table && executor.state == state;
            });
        if (existing == executors.end())
            return;
        if (existing->active) {
            throw std::logic_error{
                "cannot unregister an active native code slab executor"};
        }
        executors.erase(existing);
    }

    void request_generation_transition(GenerationTransitionKind kind,
                                       bool finish = true) {
        if (!clear_pending) {
            clear_pending = true;
            pending_generation = current_generation + 1;
            pending_transition = kind;
            if (kind == GenerationTransitionKind::ClearAll) {
                pending_ranges.clear();
            }
        } else if (kind == GenerationTransitionKind::ClearAll) {
            pending_transition = kind;
            pending_ranges.clear();
        }
        for (const auto& executor : executors) {
            if (executor.active) {
                self().halt_executor(*executor.state);
            }
        }
        if (finish)
            self().finish_pending_invalidation();
    }

    void clear_cache() {
        std::lock_guard lock{mutex};
        if (!initialized)
            return;
        request_generation_transition(GenerationTransitionKind::ClearAll);
    }

    void recycle_cache() {
        std::lock_guard lock{mutex};
        if (!initialized)
            return;
        request_generation_transition(
            GenerationTransitionKind::RecycleSegment);
    }

    void request_range_transition(std::uint32_t start_address,
                                  std::size_t length,
                                  bool finish = true) {
        if (length == 0 || (clear_pending && pending_transition == GenerationTransitionKind::ClearAll)) {
            return;
        }
        const auto last_address = InclusiveRangeEnd(start_address, length);
        pending_ranges.add(boost::icl::discrete_interval<u32>::closed(
            start_address, last_address));
        for (const auto& executor : executors) {
            if (executor.active) {
                self().halt_executor(*executor.state);
            }
        }
        if (finish)
            self().finish_pending_invalidation();
    }

    void invalidate_cache_range(std::uint32_t start_address,
                                std::size_t length) {
        std::lock_guard lock{mutex};
        if (!initialized)
            return;
        request_range_transition(start_address, length);
    }

    void request_cache_clear() {
        std::lock_guard lock{mutex};
        if (!initialized)
            return;
        request_generation_transition(
            GenerationTransitionKind::ClearAll, false);
    }

    void request_cache_range(std::uint32_t start_address,
                             std::size_t length) {
        std::lock_guard lock{mutex};
        if (!initialized)
            return;
        request_range_transition(start_address, length, false);
    }

    void service_pending_invalidation() {
        std::lock_guard lock{mutex};
        if (!initialized)
            return;
        self().finish_pending_invalidation();
        self().finish_pending_direct_link_publication();
    }

    [[nodiscard]] std::uint64_t enter_execution(void* jit_state) {
        std::unique_lock lock{mutex};
        generation_changed.wait(lock, [this] {
            return !clear_pending && pending_ranges.empty();
        });
        self().finish_pending_direct_link_publication();
        auto* const state = static_cast<State*>(jit_state);
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
        auto* const state = static_cast<State*>(jit_state);
        const auto executor = std::find_if(
            executors.begin(), executors.end(),
            [state](const Executor& candidate) {
                return candidate.state == state;
            });
        if (executor == executors.end() || !executor->active || executor->generation != generation || generation != current_generation || active_executions == 0) {
            throw std::logic_error{"native code slab execution underflow"};
        }
        executor->active = false;
        --active_executions;
        self().finish_pending_invalidation();
        self().finish_pending_direct_link_publication();
    }


private:
    Implementation& self() { return static_cast<Implementation&>(*this); }
};

} // namespace Dynarmic::Backend
