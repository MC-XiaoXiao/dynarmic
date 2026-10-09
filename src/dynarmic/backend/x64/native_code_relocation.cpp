/* This file is part of the dynarmic project. SPDX-License-Identifier: 0BSD */
#include "dynarmic/backend/x64/native_code_relocation.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>

#include "dynarmic/backend/x64/block_of_code.h"

namespace Dynarmic::Backend::X64 {

bool NativeCodeRelocations::Add(const BlockOfCode& code,
                                std::span<const u8> bytes,
                                u64 target,
                                u32 field,
                                u32 end,
                                u8 width,
                                bool relative,
                                bool constants) {
    const u64 begin = reinterpret_cast<u64>(bytes.data());
    const u64 allocation = reinterpret_cast<u64>(code.getCode());
    const u64 code_begin = reinterpret_cast<u64>(code.GetCodeBegin());
    const auto pool = code.Constants().Used();
    const u64 pool_begin = reinterpret_cast<u64>(pool.data());
    const u64 pool_end = pool_begin + pool.size_bytes();
    if (width != 1 && width != 2 && width != 4 && width != 8)
        return false;
    Relocation relocation{field, end, width, relative, TargetKind::Host};
    if (relocation.offset + width > end)
        return false;
    if (target >= begin && target < begin + bytes.size()) {
        if (relative)
            return true;  // Internal relative offsets survive copying.
        relocation.kind = TargetKind::Block;
        relocation.target = target - begin;
    } else if (target >= pool_begin && target < pool_end) {
        if (!constants)
            return false;
        const auto index = (target - pool_begin) / sizeof(pool.front());
        relocation.kind = TargetKind::Constant;
        relocation.constant_offset = static_cast<u8>((target - pool_begin) % sizeof(pool.front()));
        relocation.lower = pool[index].first;
        relocation.upper = pool[index].second;
    } else if (target >= allocation && target < code_begin) {
        // The reserved, unused constant pool is not executable prelude.
        const auto storage = code.Constants().Storage();
        if (target >= reinterpret_cast<u64>(storage.data())
            && target < reinterpret_cast<u64>(storage.data()) + storage.size_bytes())
            return false;
        relocation.kind = TargetKind::Prelude;
        relocation.target = target - allocation;
    } else if (target >= allocation && target < allocation + code.GetTotalCodeSize()) {
        // A guest link must have explicit descriptor patch metadata.
        return false;
    } else {
        relocation.target = target;
    }
    relocations.push_back(relocation);
    return true;
}

void NativeReferenceRecorder::Begin(bool enabled) {
    references.clear();
    valid = true;
    if (enabled)
        code.setReferenceObserver(Observe, this);
}
NativeReferenceRecorder::~NativeReferenceRecorder() {
    Finish();
}
void NativeReferenceRecorder::Finish() {
    code.setReferenceObserver(nullptr);
}
void NativeReferenceRecorder::Observe(void* context, const Xbyak::CodeGenerator::EncodedReference& reference) noexcept {
    auto& self = *static_cast<NativeReferenceRecorder*>(context);
    try {
        self.references.push_back(reference);
    } catch (...) {
        self.valid = false;
        self.Finish();
    }
}

std::optional<NativeCodeRelocations> NativeCodeRelocations::Capture(const BlockOfCode& code,
                                                                    std::span<const u8> bytes,
                                                                    std::span<const NativeBlockLink> links,
                                                                    const NativeReferenceRecorder& references) {
    if (!references.Valid() || bytes.size() > std::numeric_limits<u32>::max())
        return std::nullopt;
    NativeCodeRelocations result;
    const auto allocation = reinterpret_cast<u64>(code.getCode());
    const auto begin = reinterpret_cast<u64>(bytes.data());
    const auto base_offset = begin - allocation;
    size_t next_link = 0;
    for (const auto& reference : references.References()) {
        if (reference.offset < base_offset || reference.instructionEnd < reference.offset
            || reference.instructionEnd - base_offset > bytes.size())
            return std::nullopt;
        const auto offset = reference.offset - base_offset;
        const auto end = reference.instructionEnd - base_offset;
        if ((reference.width != 1 && reference.width != 2 && reference.width != 4 && reference.width != 8)
            || reference.width > end - offset)
            return std::nullopt;
        while (next_link < links.size() && offset >= links[next_link].offset + links[next_link].Size())
            ++next_link;
        if (next_link < links.size() && offset >= links[next_link].offset) {
            if (end > links[next_link].offset + links[next_link].Size())
                return std::nullopt;
            continue;
        }
        u64 value{};
        std::memcpy(&value, bytes.data() + offset, reference.width);
        using Reference = Xbyak::CodeGenerator::EncodedReference;
        if (reference.kind == Reference::Absolute) {
            // Numeric immediates may coincide with host addresses. Only guest
            // link metadata proves pointer identity; those sites were skipped.
            if (value >= allocation && value < allocation + code.GetTotalCodeSize())
                return std::nullopt;
            continue;
        }
        const auto shift = 64 - reference.width * 8;
        const auto displacement = static_cast<s64>(value << shift) >> shift;
        if (!result.Add(code, bytes, begin + end + displacement, static_cast<u32>(offset), static_cast<u32>(end),
                        reference.width, true, reference.kind == Reference::RipMemory))
            return std::nullopt;
    }
    return result;
}

bool NativeCodeRelocations::Apply(BlockOfCode& code, std::span<u8> bytes, const void* destination) const {
    const auto begin = reinterpret_cast<u64>(destination);
    for (const auto& relocation : relocations) {
        u64 target = relocation.target;
        switch (relocation.kind) {
        case TargetKind::Prelude:
            target += reinterpret_cast<u64>(code.getCode());
            break;
        case TargetKind::Block:
            target += begin;
            break;
        case TargetKind::Constant: {
            const auto constant = code.InternConstant(relocation.lower, relocation.upper);
            if (!constant)
                return false;
            target = reinterpret_cast<u64>(*constant) + relocation.constant_offset;
            break;
        }
        case TargetKind::Host:
            break;
        }
        if (relocation.relative) {
            target -= begin + relocation.instruction_end;
            if (relocation.width < 8) {
                const u64 limit = u64{1} << (relocation.width * 8 - 1);
                if (target >= limit && target < u64{0} - limit)
                    return false;
            }
        }
        if (relocation.offset > bytes.size() || relocation.width > bytes.size() - relocation.offset)
            return false;
        std::memcpy(bytes.data() + relocation.offset, &target, relocation.width);
    }
    return true;
}

size_t NativeCodeRelocations::MemoryFootprint() const {
    return relocations.capacity() * sizeof(Relocation);
}

std::shared_ptr<const NativePreludeShape> CaptureNativePrelude(BlockOfCode& code) {
    const auto pool = code.Constants().Storage();
    const auto* const begin = reinterpret_cast<const u8*>(pool.data()) + pool.size_bytes();
    const auto* const end = static_cast<const u8*>(code.GetCodeBegin());
    if (end < begin)
        return {};
    auto captured
        = NativeCodeRelocations::Capture(code, {begin, static_cast<size_t>(end - begin)}, {}, code.PreludeReferences());
    if (!captured)
        return {};
    const auto& relocations = *captured;
    auto shape = std::make_shared<NativePreludeShape>(begin, end);
    const auto append = [&](u64 value) {
        for (unsigned shift = 0; shift != 64; shift += 8)
            shape->push_back(static_cast<u8>(value >> shift));
    };
    for (const auto& relocation : relocations.relocations) {
        std::fill_n(shape->begin() + relocation.offset, relocation.width, u8{});
        append(relocation.offset);
        append(relocation.instruction_end);
        append(relocation.width | (u64{relocation.relative} << 8) | (static_cast<u64>(relocation.kind) << 16)
               | (u64{relocation.constant_offset} << 24));
        append(relocation.target);
        append(relocation.lower);
        append(relocation.upper);
    }
    append(reinterpret_cast<u64>(begin) - reinterpret_cast<u64>(code.getCode()));
    // Exact equality is checked once per slab; every block import compares
    // shared ownership identities instead of rescanning the whole prelude.
    static std::mutex mutex;
    static std::vector<std::weak_ptr<const NativePreludeShape>> shapes;
    std::lock_guard lock{mutex};
    for (auto it = shapes.begin(); it != shapes.end();) {
        if (auto existing = it->lock()) {
            if (*existing == *shape)
                return existing;
            ++it;
        } else {
            it = shapes.erase(it);
        }
    }
    shapes.emplace_back(shape);
    return shape;
}
}  // namespace Dynarmic::Backend::X64
