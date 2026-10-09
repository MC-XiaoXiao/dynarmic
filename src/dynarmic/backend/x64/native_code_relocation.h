/* This file is part of the dynarmic project. SPDX-License-Identifier: 0BSD */
#pragma once

#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <mcl/stdint.hpp>
#include <xbyak/xbyak.h>

namespace Dynarmic::Backend::X64 {
class BlockOfCode;

// Collect assembler provenance only while emitting a reusable guest block.
// Allocation failure disables optional capture without interrupting emission.
class NativeReferenceRecorder {
public:
    explicit NativeReferenceRecorder(Xbyak::CodeGenerator& code) : code{code} {}
    ~NativeReferenceRecorder();
    NativeReferenceRecorder(const NativeReferenceRecorder&) = delete;
    void Begin(bool enabled);
    void Finish();
    void ReleaseStorage() { std::vector<Xbyak::CodeGenerator::EncodedReference>().swap(references); }
    bool Valid() const { return valid; }
    std::span<const Xbyak::CodeGenerator::EncodedReference> References() const { return references; }

private:
    static void Observe(void* context, const Xbyak::CodeGenerator::EncodedReference& reference) noexcept;
    Xbyak::CodeGenerator& code;
    bool valid = true;
    std::vector<Xbyak::CodeGenerator::EncodedReference> references;
};

struct NativeBlockLink {
    enum class Kind : u8 {
        Jg,
        Jz,
        Jmp,
        MovRcx
    };
    Kind kind;
    u32 offset;
    u64 descriptor;
    size_t Size() const { return kind == Kind::MovRcx ? 10 : kind == Kind::Jmp ? 13 : 14; }
};

// Prelude shapes are interned by exact normalized bytes, not a hash alone.
// Compatible mappings may have different bases and constant insertion order.
using NativePreludeShape = std::vector<u8>;
std::shared_ptr<const NativePreludeShape> CaptureNativePrelude(BlockOfCode& code);

class NativeCodeRelocations {
public:
    static std::optional<NativeCodeRelocations> Capture(const BlockOfCode& code,
                                                        std::span<const u8> bytes,
                                                        std::span<const NativeBlockLink> links,
                                                        const NativeReferenceRecorder& references);
    bool Apply(BlockOfCode& code, std::span<u8> bytes, const void* destination) const;
    size_t MemoryFootprint() const;

private:
    enum class TargetKind : u8 {
        Prelude,
        Block,
        Constant,
        Host
    };
    struct Relocation {
        u32 offset, instruction_end;
        u8 width;
        bool relative;
        TargetKind kind;
        u8 constant_offset{};
        u64 target{}, lower{}, upper{};
    };
    bool Add(const BlockOfCode& code,
             std::span<const u8> bytes,
             u64 target,
             u32 field,
             u32 end,
             u8 width,
             bool relative,
             bool constants);
    std::vector<Relocation> relocations;
    friend std::shared_ptr<const NativePreludeShape> CaptureNativePrelude(BlockOfCode&);
};
}  // namespace Dynarmic::Backend::X64
