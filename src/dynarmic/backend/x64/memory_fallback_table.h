/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <stdexcept>
#include <tuple>

namespace Dynarmic::Backend::X64 {

template<std::size_t MaxBitsize>
class MemoryFallbackTable {
    static_assert(MaxBitsize >= 8 && MaxBitsize <= 128 && std::has_single_bit(MaxBitsize));

public:
    using Callback = void (*)();
    using Key = std::tuple<bool, std::size_t, int, int>;

    Callback& operator[](const Key& key) {
        const auto [ordered, bitsize, vaddr, value] = key;
        if (bitsize < 8 || bitsize > MaxBitsize || !std::has_single_bit(bitsize) ||
            vaddr < 0 || vaddr >= 16 || value < 0 || value >= 16) {
            throw std::out_of_range("invalid memory fallback key");
        }
        const auto width = static_cast<std::size_t>(std::countr_zero(bitsize) - 3);
        return callbacks[((width * 2 + ordered) * 16 + static_cast<std::size_t>(vaddr)) * 16 +
                         static_cast<std::size_t>(value)];
    }

private:
    // Order, access width and physical register indices have a fixed domain.
    // Unused combinations stay null, matching a missing map entry.
    std::array<Callback, (std::bit_width(MaxBitsize) - 3) * 2 * 16 * 16> callbacks{};
};

}  // namespace Dynarmic::Backend::X64
