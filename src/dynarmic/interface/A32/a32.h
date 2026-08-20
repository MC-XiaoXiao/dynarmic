/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dynarmic/interface/A32/config.h"
#include "dynarmic/interface/halt_reason.h"

namespace Dynarmic {
namespace IR {
class Block;
}  // namespace IR

namespace A32 {

class Jit;

struct DispatchCounters {
    // Deprecated aggregate counters retained for source/ABI compatibility.
    std::uint64_t stable_link_hits{};
    std::uint64_t stable_link_misses{};
    std::uint64_t rsb_hits{};
    std::uint64_t rsb_misses{};
    // Independent counters for the shared-table fast path.
    std::uint64_t fast_link_hits{};
    std::uint64_t fast_link_misses{};
    std::uint64_t stable_table_probes{};
    std::uint64_t stable_table_collisions{};
};

class NativeCodeSlab final {
public:
    struct CacheStats {
        std::size_t range_count{};
        std::size_t descriptor_count{};
        std::uint64_t invalidated_descriptors{};
        std::uint64_t retired_code_bytes{};
    };

    struct BlockDescriptor {
        const void* entrypoint{};
        std::size_t size{};
        std::uint64_t generation{};
        bool newly_emitted{};
    };

    NativeCodeSlab();
    ~NativeCodeSlab();

    NativeCodeSlab(const NativeCodeSlab&) = delete;
    NativeCodeSlab& operator=(const NativeCodeSlab&) = delete;

    // Internal integration surface used by Jit. A shared slab publishes
    // immutable native blocks; each Jit supplies its own link cells and
    // fast-dispatch table.
    void initialize(UserConfig conf, Jit* jit_interface, void* jit_state,
                    const void* (*lookup)(void*), void* lookup_arg,
                    bool shared_mode);
    [[nodiscard]] std::uint64_t generation() const;
    // Lock-free observation for executor-side cache probes. It may lag a
    // requested invalidation until that transition is safe to publish, but it
    // never waits for an active executor.
    [[nodiscard]] std::uint64_t generation_snapshot() const noexcept;
    [[nodiscard]] bool find_block(std::uint64_t location_descriptor,
                                  std::uint64_t expected_generation,
                                  BlockDescriptor& result) const;
    [[nodiscard]] BlockDescriptor emit(IR::Block& block,
                                       std::uint64_t expected_generation);
    [[nodiscard]] std::size_t space_remaining() const;
    void ensure_memory_committed(std::size_t codesize);
    void register_executor(void* storage, void* jit_state);
    void unregister_executor(void* storage, void* jit_state);
    void clear_cache();
    void invalidate_cache_range(std::uint32_t start_address,
                                std::size_t length);
    void request_cache_clear();
    void request_cache_range(std::uint32_t start_address,
                             std::size_t length);
    void service_pending_invalidation();
    [[nodiscard]] HaltReason run_code(void* jit_state,
                                      const void* code_ptr) const;
    [[nodiscard]] HaltReason step_code(void* jit_state,
                                       const void* code_ptr) const;
    [[nodiscard]] const void* return_from_run_code() const;
    [[nodiscard]] std::size_t code_cache_used() const;
    [[nodiscard]] CacheStats GetCacheStats() const;
    void dump_disassembly() const;
    [[nodiscard]] std::vector<std::string> disassemble() const;
    [[nodiscard]] bool has_host_feature_sha() const;
    [[nodiscard]] std::uint64_t enter_execution(void* jit_state);
    void leave_execution(void* jit_state, std::uint64_t generation);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    friend class Jit;
};

class Jit final {
public:
    enum class PortableIREmitOutcome : std::uint8_t {
        NativeEmitted,
        AlreadyPresent,
        EmitFailed,
    };

    // Optional host-side source for a previously validated portable IR block.
    // Dynarmic calls this only after a NativeCodeSlab miss and consumes the
    // returned block while emitting native code. The provider must not do
    // I/O, acquire locks, allocate, or scan; it may return nullptr.
    using PortableIRDemandProvider = IR::Block* (*)(
            void* user_arg, std::uint64_t location_descriptor,
            std::uint64_t slab_generation) noexcept;
    using PortableIREmitCompletion = void (*)(
            void* user_arg, std::uint64_t location_descriptor,
            std::uint64_t slab_generation,
            PortableIREmitOutcome outcome) noexcept;

    explicit Jit(UserConfig conf);
    ~Jit();

    /**
     * Runs the emulated CPU.
     * Cannot be recursively called.
     */
    HaltReason Run();

    /**
     * Steps the emulated CPU.
     * Cannot be recursively called.
     */
    HaltReason Step();

    /**
     * Compiles a previously observed A32 location descriptor without
     * executing it. The caller must guarantee that execution is stopped.
     */
    bool Precompile(std::uint64_t location_descriptor);

    /**
     * Translates and optimizes a previously observed A32 location without
     * emitting host code. The optimized block is delivered through
     * UserCallbacks::PortableIRGenerated. The caller must guarantee that
     * execution is stopped.
     */
    void GeneratePortableIR(std::uint64_t location_descriptor);
    // iLEMU portable-IR patch state: GeneratePortableIR-v1.

    /**
     * Emits a previously optimized IR block without translating guest code.
     * The caller must guarantee that execution is stopped. The return value
     * is false when the location was already compiled.
     */
    bool Precompile(IR::Block block);

    /**
     * Emits a previously optimized IR block and reports whether this call
     * emitted native code, found an existing block, or failed to emit.
     */
    PortableIREmitOutcome PrecompileWithResult(IR::Block block);

    /**
     * Installs an optional provider consumed at the true NativeCodeSlab miss.
     * The provider and user argument must remain valid until the Jit is
     * destroyed or another provider is installed.
     */
    void SetPortableIRDemandProvider(PortableIRDemandProvider provider,
                                     void* user_arg);

    /**
     * Installs an optional completion callback for provider-supplied IR.
     * The callback runs after NativeCodeSlab::emit has produced its result.
     */
    void SetPortableIREmitCompletion(PortableIREmitCompletion completion,
                                     void* user_arg);

    /**
     * Clears the code cache of all compiled code.
     * Can be called at any time. Halts execution if called within a callback.
     */
    void ClearCache();

    /**
     * Invalidate the code cache at a range of addresses.
     * @param start_address The starting address of the range to invalidate.
     * @param length The length (in bytes) of the range to invalidate.
     */
    void InvalidateCacheRange(std::uint32_t start_address, std::size_t length);

    /**
     * Reset CPU state to state at startup. Does not clear code cache.
     * Cannot be called from a callback.
     */
    void Reset();

    /**
     * Stops execution in Jit::Run.
     */
    void HaltExecution(HaltReason hr = HaltReason::UserDefined1);

    /**
     * Clears a halt reason from flags.
     * Warning: Only use this if you're sure this won't introduce races.
     */
    void ClearHalt(HaltReason hr = HaltReason::UserDefined1);

    /// View and modify registers.
    std::array<std::uint32_t, 16>& Regs();
    const std::array<std::uint32_t, 16>& Regs() const;
    std::array<std::uint32_t, 64>& ExtRegs();
    const std::array<std::uint32_t, 64>& ExtRegs() const;

    /// View and modify CPSR.
    std::uint32_t Cpsr() const;
    void SetCpsr(std::uint32_t value);

    /// View and modify FPSCR.
    std::uint32_t Fpscr() const;
    void SetFpscr(std::uint32_t value);

    /// Clears exclusive state for this core.
    void ClearExclusiveState();

    /**
     * Returns true if Jit::Run was called but hasn't returned yet.
     * i.e.: We're in a callback.
     */
    bool IsExecuting() const {
        return is_executing;
    }

    /// Debugging: Dump a disassembly all compiled code to the console.
    void DumpDisassembly() const;

    /// Returns the number of bytes currently occupied in the code cache.
    std::size_t CodeCacheUsed() const;

    /// Returns executor-local stable-link and return-stack-buffer counters.
    DispatchCounters GetDispatchCounters() const;

    /**
     * Disassemble the instructions following the current pc and return
     * the resulting instructions as a vector of their string representations.
     */
    std::vector<std::string> Disassemble() const;

private:
    bool is_executing = false;

    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace A32
}  // namespace Dynarmic
