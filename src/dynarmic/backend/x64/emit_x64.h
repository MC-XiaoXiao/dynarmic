/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <algorithm>
#include <array>
#include <deque>
#include <functional>
#include <memory>
#include <memory_resource>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <mcl/bitsizeof.hpp>
#include <mcl/container/ihmap.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>

#include "dynarmic/backend/exception_handler.h"
#include "dynarmic/backend/x64/reg_alloc.h"
#include "dynarmic/backend/x64/native_code_relocation.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/ir/location_descriptor.h"
#include "dynarmic/ir/terminal.h"

namespace Dynarmic::IR {
class Block;
class Inst;
}  // namespace Dynarmic::IR

namespace Dynarmic {
enum class OptimizationFlag : u32;
}  // namespace Dynarmic

namespace Dynarmic::Backend::X64 {

class BlockOfCode;

using A64FullVectorWidth = std::integral_constant<size_t, 128>;

// Array alias that always sizes itself according to the given type T
// relative to the size of a vector register. e.g. T = u32 would result
// in a std::array<u32, 4>.
template<typename T>
using VectorArray = std::array<T, A64FullVectorWidth::value / mcl::bitsizeof<T>>;

template<typename T>
using HalfVectorArray = std::array<T, A64FullVectorWidth::value / mcl::bitsizeof<T> / 2>;

struct EmitContext {
    EmitContext(RegAlloc& reg_alloc, IR::Block& block, std::deque<Xbyak::Label>& labels);
    EmitContext(const EmitContext&) = delete;
    EmitContext& operator=(const EmitContext&) = delete;
    virtual ~EmitContext();

    void EraseInstruction(IR::Inst* inst);
    Xbyak::Label* NewLabel();

    virtual FP::FPCR FPCR(bool fpcr_controlled = true) const = 0;

    virtual bool HasOptimization(OptimizationFlag flag) const = 0;

    RegAlloc& reg_alloc;
    IR::Block& block;

    std::vector<std::function<void()>> deferred_emits;

private:
    std::deque<Xbyak::Label>& labels;
    const size_t first_label;
};

using SharedLabel = std::shared_ptr<Xbyak::Label>;

inline SharedLabel GenSharedLabel() {
    return std::make_shared<Xbyak::Label>();
}

class EmitX64 {
public:
    struct BlockDescriptor {
        CodePtr entrypoint;  // Entrypoint of emitted code
        size_t size;         // Length in bytes of emitted code
    };

    explicit EmitX64(BlockOfCode& code);
    virtual ~EmitX64();

    /// Looks up an emitted host block in the cache.
    std::optional<BlockDescriptor> GetBasicBlock(IR::LocationDescriptor descriptor) const;

    /// Empties the entire cache.
    virtual void ClearCache();

    /// Invalidates a selection of basic blocks.
    void InvalidateBasicBlocks(const tsl::robin_set<IR::LocationDescriptor>& locations);

protected:
    RegAlloc::Storage register_allocator_storage;
    std::deque<Xbyak::Label> label_storage;

    // Microinstruction emitters
#define OPCODE(name, type, ...) void Emit##name(EmitContext& ctx, IR::Inst* inst);
#define A32OPC(...)
#define A64OPC(...)
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC

    // Helpers
    virtual std::string LocationDescriptorToFriendlyName(const IR::LocationDescriptor&) const = 0;
    virtual bool ShouldPatchExistingBlocks() const;
    void EmitAddCycles(size_t cycles);
    Xbyak::Label EmitCond(
        IR::Cond cond,
        Xbyak::CodeGenerator::LabelType label_type = Xbyak::CodeGenerator::T_AUTO);
    BlockDescriptor RegisterBlock(const IR::LocationDescriptor& location_descriptor, CodePtr entrypoint, size_t size);
    virtual void PushRSBHelper(Xbyak::Reg64 loc_desc_reg, Xbyak::Reg64 index_reg, IR::LocationDescriptor target);

    void EmitVerboseDebuggingOutput(RegAlloc& reg_alloc);

    // Terminal instruction emitters
    void EmitTerminal(IR::Terminal terminal, IR::LocationDescriptor initial_location, bool is_single_step);
    virtual void EmitTerminalImpl(IR::Term::Interpret terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::ReturnToDispatch terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::LinkBlock terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::LinkBlockFast terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::PopRSBHint terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::FastDispatchHint terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::If terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::CheckBit terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::CheckHalt terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;

    // Patching
    struct PatchInformation {
        struct Site {
            CodePtr location;
            NativeBlockLink::Kind kind;
        };
        using Sites = boost::container::small_vector<Site, 1,
            std::pmr::polymorphic_allocator<Site>>;

        explicit PatchInformation(std::pmr::memory_resource* storage)
                : sites(typename Sites::allocator_type{
                      std::pmr::polymorphic_allocator<Site>{storage}}) {}

        void Add(NativeBlockLink::Kind kind, CodePtr location) {
            sites.push_back({location, kind});
        }
        bool ForgetSources(CodePtr first, CodePtr last) {
            sites.erase(std::remove_if(sites.begin(), sites.end(),
                [first, last](const Site& site) {
                    return site.location >= first && site.location < last;
                }), sites.end());
            return sites.empty();
        }

        Sites sites;
    };
    void Patch(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr);
    virtual void Unpatch(const IR::LocationDescriptor& target_desc);
    virtual void EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) = 0;
    virtual void EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) = 0;
    virtual void EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) = 0;
    virtual void EmitPatchMovRcx(CodePtr target_code_ptr = nullptr) = 0;

    // Source registration only; repatching existing targets uses Patch().
    virtual PatchInformation& PatchesFor(const IR::LocationDescriptor& target) {
        return patch_information.try_emplace(target, &metadata_storage).first->second;
    }

    // The slab mutex serializes emission and invalidation. Pool small metadata
    // allocations per emitter instead of contending with guest/host allocations
    // for every patch node and short vector during generation retirement.
    std::pmr::unsynchronized_pool_resource metadata_storage{{0, 512}};

    // State
    BlockOfCode& code;
    ExceptionHandler exception_handler;
    mcl::ihmap<IR::LocationDescriptor, BlockDescriptor> block_descriptors;
    // Single-site targets and hash slots stay inline. PatchesFor references
    // are consumed before another insertion; larger site lists use the pool.
    mcl::ihmap<IR::LocationDescriptor, PatchInformation> patch_information;
};

}  // namespace Dynarmic::Backend::X64
