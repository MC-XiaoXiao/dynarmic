/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include <mcl/hash/xmrx.hpp>
#include <mcl/stdint.hpp>
#include <oaknut/code_block.hpp>
#include <oaknut/oaknut.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "dynarmic/backend/arm64/cold_code_layout.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/fast_dispatch.h"
#include "dynarmic/backend/arm64/fastmem.h"
#include "dynarmic/backend/native_code_segments.h"
#include "dynarmic/interface/halt_reason.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend::Arm64 {

struct StackLayout;

class AddressSpace : protected NativeCodeSegments<std::byte> {
public:
    explicit AddressSpace(size_t code_cache_size);
    virtual ~AddressSpace();

    virtual IR::Block GenerateIR(IR::LocationDescriptor) const = 0;

    CodePtr Get(IR::LocationDescriptor descriptor) const;

    // Returns "most likely" LocationDescriptor assocated with the emitted code at that location
    std::optional<IR::LocationDescriptor> ReverseGetLocation(CodePtr host_pc);

    // Returns "most likely" entry_point associated with the emitted code at that location
    CodePtr ReverseGetEntryPoint(CodePtr host_pc);

    virtual CodePtr GetOrEmit(IR::LocationDescriptor descriptor);
    CodePtr GetOrEmit(IR::LocationDescriptor descriptor, StackLayout& stack);

    void InvalidateBasicBlocks(const tsl::robin_set<IR::LocationDescriptor>& descriptors);

    virtual void ClearCache();

    void DumpDisassembly() const;

    size_t GetCodeCacheUsed() const;
    u64 GetCodeCacheGeneration() const { return cache_generation; }

protected:
    virtual EmitConfig GetEmitConfig() = 0;
    virtual void CodeTranslationCompleted(const IR::Block&, u64) const noexcept {}
    virtual void CaptureEmittedBlock(const IR::Block&, EmittedBlockInfo&) {}
    virtual void RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo& block_info) = 0;
    virtual CodePtr PublishEntry(IR::LocationDescriptor, CodePtr code) { return code; }
    virtual void FinishEntryPublication(IR::LocationDescriptor, CodePtr) {}

    void ProtectCodeMemory() {
#if defined(DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT) || defined(__APPLE__) || defined(__OpenBSD__)
        mem.protect();
#endif
    }

    void UnprotectCodeMemory() {
#if defined(DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT) || defined(__APPLE__) || defined(__OpenBSD__)
        mem.unprotect();
#endif
    }

    size_t GetRemainingSize();
    void EnsureEmissionSpace();
    const EmittedBlockInfo& Emit(IR::Block& ir_block);
    // Fixed relocations are consumed synchronously, never retained by the cache.
    const EmittedBlockInfo& PublishBlock(IR::LocationDescriptor location, EmittedBlockInfo block_info,
                                       std::span<const Relocation> fixed_relocations = {});
    void Link(const EmittedBlockInfo& block, std::span<const Relocation> fixed_relocations);
    void RecycleEmissionRelocations(std::vector<Relocation>& relocations);
    template<bool Synchronize = false>
    void LinkBlockLinks(const CodePtr entry_point, const CodePtr target_ptr, std::span<const BlockRelocation> block_relocations_list);
    void RelinkForDescriptor(IR::LocationDescriptor target_descriptor, CodePtr target_ptr);
    void RelinkBranchesForDescriptor(IR::LocationDescriptor target_descriptor, CodePtr target_ptr);
    void PublishBranch(CodePtr patch, CodePtr target);
    void EnableSegmentedCache();
    void ResetColdCodeLayout();
    void PlaceBlock(EmittedBlockInfo& info, const void* canonical_code);
    void RetireCodeIndex(std::size_t segment,
                         const tsl::robin_set<IR::LocationDescriptor>& locations);

    FakeCall FastmemCallback(u64 host_pc);

    bool defer_block_linking = false;
    const size_t code_cache_size;
    u64 cache_generation{};
    std::unique_ptr<FastDispatchCache> fast_dispatch_cache;
    oaknut::CodeBlock mem;
    oaknut::CodeGenerator code;
    std::unique_ptr<u32[]> emission_buffer;
    std::vector<Relocation> emission_relocations;
    ColdCodeLayout cold_code;

    // A IR::LocationDescriptor will have one current CodePtr.
    // However, there can be multiple other CodePtrs which are older, previously invalidated blocks.
    tsl::robin_map<IR::LocationDescriptor, CodePtr> block_entries;
    struct CodeIndex {
        boost::container::flat_map<CodePtr, IR::LocationDescriptor> reverse_block_entries;
        boost::container::flat_map<CodePtr, EmittedBlockInfo> block_infos;
        struct ColdOwner {
            u32 offset;
            u32 block_index;
        };
        // Each chunk's cold addresses descend in publication order.
        std::vector<std::vector<ColdOwner>> cold_owners;
    };
    // Each physical segment appends independently after the code ring wraps.
    // Retirement releases only that segment's packed metadata.
    std::vector<CodeIndex> code_indexes{1};
    const CodeIndex* FindCodeIndex(CodePtr address) const;
    static constexpr u32 invalid_block_reference = ~u32{};
    struct BlockReference {
        CodePtr entry_point;
        std::span<const BlockRelocation> relocations;
        u32 next;
    };
    struct DescriptorHash {
        size_t operator()(IR::LocationDescriptor descriptor) const noexcept {
            return mcl::hash::xmrx(descriptor.Value());
        }
    };
    // Relocation buffers are immutable and owned by block_infos (or its shared
    // templates) until ClearCache, including sources invalidated in between.
    // Each reference covers at least one instruction in the code cache, so
    // the 128 MiB maximum leaves room for all slots in a 32-bit index. Keep
    // the hash buckets compact and reuse slots released by segment retirement.
    tsl::robin_map<IR::LocationDescriptor, u32, DescriptorHash> block_reference_heads;
    std::vector<BlockReference> block_references;
    u32 free_block_reference = invalid_block_reference;

    ExceptionHandler exception_handler;
    FastmemManager fastmem_manager;

    struct PreludeInfo {
        std::ptrdiff_t end_of_prelude;

        using RunCodeFuncType = HaltReason (*)(CodePtr entry_point, void* jit_state, volatile u32* halt_reason);
        RunCodeFuncType run_code;
        RunCodeFuncType step_code;
        void* return_to_dispatcher;
        void* return_from_run_code;

        void* read_memory_8;
        void* read_memory_16;
        void* read_memory_32;
        void* read_memory_64;
        void* read_memory_128;
        void* wrapped_read_memory_8;
        void* wrapped_read_memory_16;
        void* wrapped_read_memory_32;
        void* wrapped_read_memory_64;
        void* wrapped_read_memory_128;
        void* exclusive_read_memory_8;
        void* exclusive_read_memory_16;
        void* exclusive_read_memory_32;
        void* exclusive_read_memory_64;
        void* exclusive_read_memory_128;
        void* write_memory_8;
        void* write_memory_16;
        void* write_memory_32;
        void* write_memory_64;
        void* write_memory_128;
        void* swap_memory_8;
        void* swap_memory_32;
        void* wrapped_write_memory_8;
        void* wrapped_write_memory_16;
        void* wrapped_write_memory_32;
        void* wrapped_write_memory_64;
        void* wrapped_write_memory_128;
        void* exclusive_write_memory_8;
        void* exclusive_write_memory_16;
        void* exclusive_write_memory_32;
        void* exclusive_write_memory_64;
        void* exclusive_write_memory_128;

        void* call_svc;
        void* exception_raised;
        void* dc_raised;
        void* ic_raised;
        void* isb_raised;

        void* get_cntpct;
        void* add_ticks;
        void* get_ticks_remaining;
    } prelude_info;
};

}  // namespace Dynarmic::Backend::Arm64
