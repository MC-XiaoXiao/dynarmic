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
        entries[((value >> 2) ^ (value >> 32)) & index_mask] = {value, code};
    }

private:
    std::array<Entry, entry_count> entries{};
};

}  // namespace Dynarmic::Backend::Arm64
