/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>

#include <mcl/bit_cast.hpp>

#include "dynarmic/backend/arm64/a64_address_space.h"
#include "dynarmic/backend/arm64/a64_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/devirtualize.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/stack_layout.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/llvm_disassemble.h"
#include "dynarmic/interface/exclusive_monitor.h"

namespace Dynarmic::Backend::Arm64 {

AddressSpace::AddressSpace(size_t code_cache_size)
        : code_cache_size(code_cache_size)
        , mem(code_cache_size)
        , code(mem.ptr(), mem.ptr())
        , fastmem_manager(exception_handler) {
    // Mixed descriptor bits keep this large metadata index compact without
    // clustering aligned guest addresses in the power-of-two bucket table.
    block_references.max_load_factor(0.8f);
    ASSERT_MSG(code_cache_size <= 128 * 1024 * 1024, "code_cache_size > 128 MiB not currently supported");

    exception_handler.Register(mem, code_cache_size);
    exception_handler.SetFastmemCallback([this](u64 host_pc) {
        return FastmemCallback(host_pc);
    });
}

AddressSpace::~AddressSpace() = default;

CodePtr AddressSpace::Get(IR::LocationDescriptor descriptor) const {
    if (const auto iter = block_entries.find(descriptor); iter != block_entries.end()) {
        return iter->second;
    }
    return nullptr;
}

const AddressSpace::CodeIndex* AddressSpace::FindCodeIndex(CodePtr address) const {
    if (code_segments.empty())
        return &code_indexes.front();
    const auto segment = code_segment_index(address);
    return segment ? &code_indexes[*segment] : nullptr;
}

std::optional<IR::LocationDescriptor> AddressSpace::ReverseGetLocation(CodePtr host_pc) {
    const auto entry = ReverseGetEntryPoint(host_pc);
    const auto* index = FindCodeIndex(entry);
    if (index) {
        const auto it = index->reverse_block_entries.find(entry);
        if (it != index->reverse_block_entries.end())
            return it->second;
    }
    return std::nullopt;
}

CodePtr AddressSpace::ReverseGetEntryPoint(CodePtr host_pc) {
    const auto* index = FindCodeIndex(host_pc);
    if (index) {
        if (!code_segments.empty()) {
            const auto segment = *code_segment_index(host_pc);
            const auto offset = static_cast<size_t>(host_pc - code_segments[segment].begin);
            const auto chunk = offset / ColdCodeLayout::chunk_bytes;
            if (chunk < index->cold_owners.size()) {
                const auto& owners = index->cold_owners[chunk];
                const auto within = static_cast<u32>(offset % ColdCodeLayout::chunk_bytes);
                const auto found = std::lower_bound(owners.begin(), owners.end(), within,
                    [](const auto& owner, u32 value) { return owner.offset > value; });
                if (found != owners.end()) {
                    const auto block = index->block_infos.begin() + found->block_index;
                    if (within - found->offset < block->second.size - block->second.hot_size)
                        return block->first;
                }
            }
        }
        auto it = index->reverse_block_entries.upper_bound(host_pc);
        if (it != index->reverse_block_entries.begin())
            return std::prev(it)->first;
    }
    return nullptr;
}

CodePtr AddressSpace::GetOrEmit(IR::LocationDescriptor descriptor) {
    if (CodePtr block_entry = Get(descriptor)) {
        return block_entry;
    }

    EnsureEmissionSpace();
    const auto translation_started = std::chrono::steady_clock::now();
    IR::Block ir_block = GenerateIR(descriptor);
    const EmittedBlockInfo& block_info = Emit(ir_block);
    const auto translation_nanoseconds = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - translation_started).count());
    // Notify only once native code is linked, synchronized and indexed. The
    // optimized IR remains alive for clients of the existing portable codec.
    CodeTranslationCompleted(ir_block, translation_nanoseconds);
    return block_info.entry_point;
}

CodePtr AddressSpace::GetOrEmit(IR::LocationDescriptor descriptor, StackLayout& stack) {
    const auto generation = cache_generation;
    const auto entry_point = GetOrEmit(descriptor);
    if (generation != cache_generation) {
        // A demand miss can recycle code while Run is still on the stack.
        // Return predictions contain raw code pointers into that recycled
        // mapping; send them through dispatch before using the new code.
        for (auto& entry : stack.rsb) {
            entry.code_ptr = mcl::bit_cast<u64>(prelude_info.return_to_dispatcher);
        }
    }
    return entry_point;
}

void AddressSpace::InvalidateBasicBlocks(const tsl::robin_set<IR::LocationDescriptor>& descriptors) {
    // A range request can cover data or code that was never translated.
    // Keep dispatch predictions and executable permissions in that case.
    if (descriptors.empty()) {
        return;
    }
    if (fast_dispatch_cache) {
        fast_dispatch_cache->Clear();
    }
    UnprotectCodeMemory();

    for (const auto& descriptor : descriptors) {
        const auto iter = block_entries.find(descriptor);
        if (iter == block_entries.end()) {
            continue;
        }

        // Unlink before removal because InvalidateBasicBlocks can be called within a fastmem callback,
        // and the currently executing block may have references to itself which need to be unlinked.
        RelinkForDescriptor(descriptor, nullptr);

        block_entries.erase(iter);
    }

    ProtectCodeMemory();
}

void AddressSpace::ClearCache() {
    ++cache_generation;
    if (fast_dispatch_cache) {
        fast_dispatch_cache->Clear();
    }
    block_entries.clear();
    for (auto& index : code_indexes)
        index = {};
    reset_code_segments();
    block_references.clear();
    code.set_offset(prelude_info.end_of_prelude);
    if (!code_segments.empty())
        ResetColdCodeLayout();
}

void AddressSpace::EnableSegmentedCache() {
    code.align(16);
    prelude_info.end_of_prelude = code.offset();
    initialize_code_segments(code.xptr<CodePtr>(), GetRemainingSize());
    code_indexes.resize(code_segments.size());
    emission_buffer = std::make_unique_for_overwrite<u32[]>(code_cache_size / sizeof(u32));
    ResetColdCodeLayout();
}

void AddressSpace::ResetColdCodeLayout() {
    const auto& segment = code_segments[current_segment];
    cold_code.Reset(segment.begin, segment.end);
}

void AddressSpace::PlaceBlock(EmittedBlockInfo& info, const void* canonical_code) {
    if (code_segments.empty()) {
        info.entry_point = code.xptr<CodePtr>();
        std::memcpy(info.entry_point, canonical_code, info.size);
        code.set_offset(code.offset() + static_cast<std::ptrdiff_t>(info.size));
        return;
    }
    // Keep large canonical images contiguous; splitting uses imm19 branches.
    const auto hot_size = info.size + sizeof(u32) > ColdCodeLayout::chunk_bytes
        ? info.size : (info.hot_size ? info.hot_size : info.size);
    const auto cold_size = info.size - hot_size;
    const auto placement = cold_code.Allocate(hot_size, cold_size);
    info.entry_point = placement.hot;
    info.hot_size = static_cast<u32>(hot_size);
    info.cold_offset = cold_size ? static_cast<u32>(placement.cold - placement.hot) : 0;
    const auto* bytes = static_cast<const std::byte*>(canonical_code);
    std::memcpy(placement.hot, bytes, hot_size);
    if (cold_size)
        std::memcpy(placement.cold, bytes + hot_size, cold_size);
    code.set_xptr(reinterpret_cast<u32*>(placement.hot + hot_size));
}

void AddressSpace::RetireCodeIndex(
        std::size_t segment, const tsl::robin_set<IR::LocationDescriptor>& locations) {
    ++cache_generation;
    if (fast_dispatch_cache)
        fast_dispatch_cache->Clear();
    const auto& retired = code_segments[segment];
    // Remove sources before destroying their immutable relocation buffers.
    // Incoming references from surviving segments remain available for relinking.
    for (auto it = block_references.begin(); it != block_references.end();) {
        auto& references = it.value();
        references.erase(std::remove_if(references.begin(), references.end(),
            [&](const BlockReference& source) {
                return source.entry_point >= retired.begin && source.entry_point < retired.end;
            }), references.end());
        if (references.empty())
            it = block_references.erase(it);
        else
            ++it;
    }
    for (auto location : locations) {
        RelinkForDescriptor(location, nullptr);
        block_entries.erase(location);
    }
    code_indexes[segment] = {};
}

void AddressSpace::DumpDisassembly() const {
    const auto dump = [](u32* begin, u32* end) {
        for (auto* ptr = begin; ptr < end; ++ptr)
            std::printf("%s", Common::DisassembleAArch64(*ptr, mcl::bit_cast<u64>(ptr)).c_str());
    };
    if (code_segments.empty()) {
        dump(mem.ptr(), code.xptr<u32*>());
    } else {
        dump(mem.ptr(), mem.ptr() + prelude_info.end_of_prelude / sizeof(u32));
        for (const auto& segment : code_segments)
            dump(reinterpret_cast<u32*>(segment.begin),
                 reinterpret_cast<u32*>(segment.begin + segment.used_bytes));
    }
}

size_t AddressSpace::GetCodeCacheUsed() const {
    return code_segments.empty() ? static_cast<size_t>(code.offset())
                                : prelude_info.end_of_prelude + live_allocated_code_bytes;
}

size_t AddressSpace::GetRemainingSize() {
    return code_segments.empty() ? code_cache_size - static_cast<size_t>(code.offset())
                                : cold_code.Remaining();
}

void AddressSpace::EnsureEmissionSpace() {
    if (GetRemainingSize() < 1024 * 1024) {
        ClearCache();
    }
}

const EmittedBlockInfo& AddressSpace::Emit(IR::Block& block) {
    EnsureEmissionSpace();

    UnprotectCodeMemory();

    oaknut::CodeGenerator scratch{emission_buffer.get(), emission_buffer.get()};
    auto& emitter = code_segments.empty() ? code : scratch;
    EmittedBlockInfo block_info = EmitArm64(emitter, block, GetEmitConfig(), fastmem_manager);
    CaptureEmittedBlock(block, block_info);
    if (!code_segments.empty())
        PlaceBlock(block_info, block_info.entry_point);
    const auto& cached_info = PublishBlock(block.Location(), std::move(block_info));
    RegisterNewBasicBlock(block, cached_info);
    return cached_info;
}

const EmittedBlockInfo& AddressSpace::PublishBlock(IR::LocationDescriptor location, EmittedBlockInfo block_info,
                                                  std::span<const Relocation> fixed_relocations) {
    block_entries.insert_or_assign(location, PublishEntry(location, block_info.entry_point));
    auto& index = code_indexes[current_segment];
    auto& reverse_block_entries = index.reverse_block_entries;
    auto& block_infos = index.block_infos;
    ASSERT(reverse_block_entries.empty() ||
           std::less<CodePtr>{}(reverse_block_entries.rbegin()->first, block_info.entry_point));
    reverse_block_entries.emplace_hint(reverse_block_entries.end(),
                                      block_info.entry_point, location);
    Link(block_info, fixed_relocations.empty() ? std::span<const Relocation>{block_info.relocations} : fixed_relocations);
    // Prelude callback targets are fixed after initial linking. Retain only
    // block links and fastmem sites, which are needed for later patching.
    std::vector<Relocation>{}.swap(block_info.relocations);
    // The cache owns relocation metadata. Move it once instead of copying
    // its per-block containers into the index and back to the dispatcher.
    // Native addresses increase here, as in the reverse index above.
    const auto block_iter = block_infos.try_emplace(block_infos.end(), block_info.entry_point, std::move(block_info));
    const EmittedBlockInfo& cached_info = block_iter->second;
    if (cached_info.cold_offset) {
        const auto offset = static_cast<size_t>(cached_info.CodeAddress(cached_info.hot_size) -
                                              code_segments[current_segment].begin);
        const auto chunk = offset / ColdCodeLayout::chunk_bytes;
        index.cold_owners.resize(std::max(index.cold_owners.size(), chunk + 1));
        index.cold_owners[chunk].push_back({static_cast<u32>(offset % ColdCodeLayout::chunk_bytes),
            static_cast<u32>(block_infos.size() - 1)});
    }

    if (!defer_block_linking)
        RelinkForDescriptor(location, Get(location));

    const auto hot_size = cached_info.cold_offset ? cached_info.hot_size : cached_info.size;
    mem.invalidate(reinterpret_cast<u32*>(cached_info.entry_point), hot_size);
    if (cached_info.cold_offset)
        mem.invalidate(reinterpret_cast<u32*>(cached_info.CodeAddress(cached_info.hot_size)),
                       cached_info.size - cached_info.hot_size);
    FinishEntryPublication(location, cached_info.entry_point);
    if (!code_segments.empty()) {
        cold_code.CommitHot(code.xptr<CodePtr>());
        update_active_segment_usage(cold_code.UsedEnd());
        touch_code_segment(cached_info.entry_point);
    }
    ProtectCodeMemory();

    return cached_info;
}

void AddressSpace::Link(const EmittedBlockInfo& block_info, std::span<const Relocation> fixed_relocations) {
    using namespace oaknut;
    using namespace oaknut::util;

    for (auto [ptr_offset, target, block_offset] : fixed_relocations) {
        CodeGenerator c{mem.ptr(), mem.ptr()};
        c.set_xptr(reinterpret_cast<u32*>(block_info.CodeAddress(ptr_offset)));

        switch (target) {
        case LinkTarget::BlockConditionalBranch: {
            const auto displacement = block_info.CodeAddress(block_offset) - block_info.CodeAddress(ptr_offset);
            const auto word = *c.wptr<u32*>();
            c.dw((word & 0xff00001fU) | ((static_cast<u32>(displacement / 4) & 0x7ffffU) << 5));
            break;
        }
        case LinkTarget::BlockBranch:
            c.B(static_cast<const void*>(block_info.CodeAddress(block_offset)));
            break;
        case LinkTarget::ReturnToDispatcher:
            c.B(prelude_info.return_to_dispatcher);
            break;
        case LinkTarget::ReturnFromRunCode:
            c.B(prelude_info.return_from_run_code);
            break;
        case LinkTarget::ReadMemory8:
            c.BL(prelude_info.read_memory_8);
            break;
        case LinkTarget::ReadMemory16:
            c.BL(prelude_info.read_memory_16);
            break;
        case LinkTarget::ReadMemory32:
            c.BL(prelude_info.read_memory_32);
            break;
        case LinkTarget::ReadMemory64:
            c.BL(prelude_info.read_memory_64);
            break;
        case LinkTarget::ReadMemory128:
            c.BL(prelude_info.read_memory_128);
            break;
        case LinkTarget::WrappedReadMemory8:
            c.BL(prelude_info.wrapped_read_memory_8);
            break;
        case LinkTarget::WrappedReadMemory16:
            c.BL(prelude_info.wrapped_read_memory_16);
            break;
        case LinkTarget::WrappedReadMemory32:
            c.BL(prelude_info.wrapped_read_memory_32);
            break;
        case LinkTarget::WrappedReadMemory64:
            c.BL(prelude_info.wrapped_read_memory_64);
            break;
        case LinkTarget::WrappedReadMemory128:
            c.BL(prelude_info.wrapped_read_memory_128);
            break;
        case LinkTarget::ExclusiveReadMemory8:
            c.BL(prelude_info.exclusive_read_memory_8);
            break;
        case LinkTarget::ExclusiveReadMemory16:
            c.BL(prelude_info.exclusive_read_memory_16);
            break;
        case LinkTarget::ExclusiveReadMemory32:
            c.BL(prelude_info.exclusive_read_memory_32);
            break;
        case LinkTarget::ExclusiveReadMemory64:
            c.BL(prelude_info.exclusive_read_memory_64);
            break;
        case LinkTarget::ExclusiveReadMemory128:
            c.BL(prelude_info.exclusive_read_memory_128);
            break;
        case LinkTarget::WriteMemory8:
            c.BL(prelude_info.write_memory_8);
            break;
        case LinkTarget::WriteMemory16:
            c.BL(prelude_info.write_memory_16);
            break;
        case LinkTarget::WriteMemory32:
            c.BL(prelude_info.write_memory_32);
            break;
        case LinkTarget::WriteMemory64:
            c.BL(prelude_info.write_memory_64);
            break;
        case LinkTarget::WriteMemory128:
            c.BL(prelude_info.write_memory_128);
            break;
        case LinkTarget::SwapMemory8:
            c.BL(prelude_info.swap_memory_8);
            break;
        case LinkTarget::SwapMemory32:
            c.BL(prelude_info.swap_memory_32);
            break;
        case LinkTarget::WrappedWriteMemory8:
            c.BL(prelude_info.wrapped_write_memory_8);
            break;
        case LinkTarget::WrappedWriteMemory16:
            c.BL(prelude_info.wrapped_write_memory_16);
            break;
        case LinkTarget::WrappedWriteMemory32:
            c.BL(prelude_info.wrapped_write_memory_32);
            break;
        case LinkTarget::WrappedWriteMemory64:
            c.BL(prelude_info.wrapped_write_memory_64);
            break;
        case LinkTarget::WrappedWriteMemory128:
            c.BL(prelude_info.wrapped_write_memory_128);
            break;
        case LinkTarget::ExclusiveWriteMemory8:
            c.BL(prelude_info.exclusive_write_memory_8);
            break;
        case LinkTarget::ExclusiveWriteMemory16:
            c.BL(prelude_info.exclusive_write_memory_16);
            break;
        case LinkTarget::ExclusiveWriteMemory32:
            c.BL(prelude_info.exclusive_write_memory_32);
            break;
        case LinkTarget::ExclusiveWriteMemory64:
            c.BL(prelude_info.exclusive_write_memory_64);
            break;
        case LinkTarget::ExclusiveWriteMemory128:
            c.BL(prelude_info.exclusive_write_memory_128);
            break;
        case LinkTarget::CallSVC:
            c.BL(prelude_info.call_svc);
            break;
        case LinkTarget::ExceptionRaised:
            c.BL(prelude_info.exception_raised);
            break;
        case LinkTarget::InstructionSynchronizationBarrierRaised:
            c.BL(prelude_info.isb_raised);
            break;
        case LinkTarget::InstructionCacheOperationRaised:
            c.BL(prelude_info.ic_raised);
            break;
        case LinkTarget::DataCacheOperationRaised:
            c.BL(prelude_info.dc_raised);
            break;
        case LinkTarget::GetCNTPCT:
            c.BL(prelude_info.get_cntpct);
            break;
        case LinkTarget::AddTicks:
            c.BL(prelude_info.add_ticks);
            break;
        case LinkTarget::GetTicksRemaining:
            c.BL(prelude_info.get_ticks_remaining);
            break;
        default:
            ASSERT_FALSE("Invalid relocation target");
        }
    }

    for (const auto& [target_descriptor, list] : block_info.BlockRelocations()) {
        block_references[target_descriptor].push_back({block_info.entry_point, list});
        LinkBlockLinks(block_info.entry_point, Get(target_descriptor), list);
    }
}

template<bool Synchronize>
void AddressSpace::LinkBlockLinks(const CodePtr entry_point, const CodePtr target_ptr, std::span<const BlockRelocation> block_relocations_list) {
    using namespace oaknut;
    using namespace oaknut::util;

    CodePtr modified_begin = block_relocations_list.empty() ? entry_point : entry_point + block_relocations_list.front().code_offset;
    CodePtr modified_end = modified_begin;
    for (auto [ptr_offset, type] : block_relocations_list) {
        CodeGenerator c{mem.ptr(), mem.ptr()};
        c.set_xptr(reinterpret_cast<u32*>(entry_point + ptr_offset));

        switch (type) {
        case BlockRelocationType::Branch:
            if (target_ptr) {
                c.B((void*)target_ptr);
            } else {
                c.NOP();
            }
            break;
        case BlockRelocationType::MoveToScratch1:
            if (target_ptr) {
                c.ADRL(Xscratch1, (void*)target_ptr);
            } else {
                c.ADRL(Xscratch1, prelude_info.return_to_dispatcher);
            }
            break;
        default:
            ASSERT_FALSE("Invalid BlockRelocationType");
        }
        if constexpr (Synchronize) {
            const auto begin = entry_point + ptr_offset;
            const auto end = c.xptr<CodePtr>();
            modified_begin = std::min(modified_begin, begin);
            modified_end = std::max(modified_end, end);
        }
    }
    if constexpr (Synchronize) {
        if (block_relocations_list.empty())
            return;
        // Relinking changes only these instructions. Keep Oaknut's complete
        // cache-maintenance sequence, including multi-instruction ADRL sites.
        mem.invalidate(reinterpret_cast<u32*>(modified_begin), static_cast<size_t>(modified_end - modified_begin));
    }
}

void AddressSpace::RelinkBranchesForDescriptor(IR::LocationDescriptor target_descriptor, CodePtr target_ptr) {
#if !defined(DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT) && !defined(__APPLE__) && !defined(__OpenBSD__)
    // The target's cache maintenance has completed before this publication.
    // AArch64 permits concurrent NOP/B replacement: an executor may take
    // either the correct dispatcher fallback or the synchronized new target.
    // ADRL is a multi-instruction update and still requires quiescence.
    if (!target_ptr)
        return;
    const auto references = block_references.find(target_descriptor);
    if (references == block_references.end())
        return;
    for (const auto& [source, relocations] : references->second) {
        for (const auto [offset, type] : relocations) {
            if (type != BlockRelocationType::Branch)
                continue;
            PublishBranch(source + offset, target_ptr);
        }
    }
#else
    // Mappings that change executable permissions retain deferred linking.
    (void)target_descriptor;
    (void)target_ptr;
#endif
}

void AddressSpace::PublishBranch(CodePtr patch, CodePtr target) {
    u32 instruction;
    auto* word = reinterpret_cast<u32*>(patch);
    oaknut::CodeGenerator branch{&instruction, word};
    branch.B(static_cast<const void*>(target));
    static_assert(std::atomic_ref<u32>::is_always_lock_free);
    static_assert(std::atomic_ref<u32>::required_alignment <= alignof(u32));
    // Reuse AArch64's permitted concurrent B replacement, as for block links.
    std::atomic_ref<u32>{*word}.store(instruction, std::memory_order_release);
    mem.invalidate(word, sizeof(u32));
}

void AddressSpace::RelinkForDescriptor(IR::LocationDescriptor target_descriptor, CodePtr target_ptr) {
    const auto references = block_references.find(target_descriptor);
    if (references == block_references.end())
        return;
    for (const auto& [source, relocations] : references->second) {
        LinkBlockLinks<true>(source, target_ptr, relocations);
    }
}

FakeCall AddressSpace::FastmemCallback(u64 host_pc) {
    {
        const auto host_ptr = mcl::bit_cast<CodePtr>(host_pc);

        const auto entry_point = ReverseGetEntryPoint(host_ptr);
        if (!entry_point) {
            goto fail;
        }

        const auto& block_infos = FindCodeIndex(entry_point)->block_infos;
        const auto block_info = block_infos.find(entry_point);
        if (block_info == block_infos.end()) {
            goto fail;
        }

        const auto patch_entry = block_info->second.fastmem_patch_info.find(host_ptr - entry_point);
        if (patch_entry == block_info->second.fastmem_patch_info.end()) {
            goto fail;
        }

        const auto fc = patch_entry->second.fc;

        if (patch_entry->second.recompile) {
            const auto marker = patch_entry->second.marker;
            fastmem_manager.MarkDoNotFastmem(marker);
            InvalidateBasicBlocks({std::get<0>(marker)});
        }

        return fc;
    }

fail:
    fmt::print("dynarmic: Segfault happened within JITted code at host_pc = {:016x}\n", host_pc);
    fmt::print("Segfault wasn't at a fastmem patch location!\n");
    ASSERT_FALSE("segfault");
}

}  // namespace Dynarmic::Backend::Arm64
