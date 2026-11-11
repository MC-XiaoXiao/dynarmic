/* This file is part of the dynarmic project. SPDX-License-Identifier: 0BSD */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace Dynarmic::Backend::Arm64 {

// Conditional branches reach +/-1 MiB. Keep both ends of a split block inside
// one such chunk, with the ordinary code growing up and deferred code down.
class ColdCodeLayout {
public:
    static constexpr std::size_t chunk_bytes = 1U << 20;
    struct Placement {
        std::byte* hot;
        std::byte* cold;
    };

    void Reset(std::byte* begin, std::byte* end) {
        hot = cold = chunk_end = begin;
        segment_end = end;
    }

    std::size_t Remaining() const {
        return std::max(static_cast<std::size_t>(cold - hot),
                        static_cast<std::size_t>(segment_end - chunk_end));
    }

    Placement Allocate(std::size_t hot_bytes, std::size_t cold_bytes) {
        // Literal pools in the canonical emission image are 16-byte aligned.
        const auto padding = (16U - (reinterpret_cast<std::uintptr_t>(hot) & 15U)) & 15U;
        // A tiered entry may append one branch after the hot body.
        if (padding + hot_bytes + cold_bytes + sizeof(std::uint32_t) >
            static_cast<std::size_t>(cold - hot)) {
            hot = chunk_end;
            chunk_end += ((hot_bytes + cold_bytes + sizeof(std::uint32_t) + chunk_bytes - 1U) /
                          chunk_bytes) * chunk_bytes;
            cold = chunk_end;
        } else {
            hot += padding;
        }
        cold -= cold_bytes;
        Placement result{hot, cold};
        hot += hot_bytes;
        return result;
    }

    void CommitHot(std::byte* cursor) { hot = cursor; }
    std::byte* UsedEnd() const { return chunk_end; }

private:
    std::byte* hot{};
    std::byte* cold{};
    std::byte* chunk_end{};
    std::byte* segment_end{};
};

} // namespace Dynarmic::Backend::Arm64
