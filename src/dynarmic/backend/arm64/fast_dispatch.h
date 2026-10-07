/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <cstddef>

#include <mcl/stdint.hpp>

#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend::Arm64 {

// A bounded, executor-local front cache for the authoritative block index.
// Full descriptor tags distinguish ARM/Thumb and floating-point modes.
class FastDispatchCache {
public:
    struct alignas(16) Entry {
        u64 descriptor{};
        std::byte* code{};
    };
    static_assert(sizeof(Entry) == 16);
    static constexpr size_t entry_count = 1024;
    static constexpr size_t index_mask = entry_count - 1;

    Entry* Data() { return entries.data(); }
    void Clear() { entries.fill({}); }
    void Publish(IR::LocationDescriptor descriptor, std::byte* code) {
        const u64 value = descriptor.Value();
        entries[Index(value)] = {value, code};
    }
    // A colliding descriptor may already occupy the same slot. Retire only
    // the exact tag that the authoritative code index invalidated.
    static void Invalidate(Entry* table, IR::LocationDescriptor descriptor) {
        const u64 value = descriptor.Value();
        auto& entry = table[Index(value)];
        if (entry.descriptor == value)
            entry = {};
    }

private:
    static constexpr size_t Index(u64 descriptor) {
        return ((descriptor >> 2) ^ (descriptor >> 32)) & index_mask;
    }

    std::array<Entry, entry_count> entries{};
};

}  // namespace Dynarmic::Backend::Arm64
