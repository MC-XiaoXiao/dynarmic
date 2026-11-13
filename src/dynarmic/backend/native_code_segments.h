/* This file is part of the dynarmic project. SPDX-License-Identifier: 0BSD */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace Dynarmic::Backend {

// Geometry, occupancy and recency shared by native emitters. Publication and
// retirement remain backend operations at the slab's quiescent boundary.
template<class Byte>
class NativeCodeSegments {
protected:
    struct CodeSegment {
        Byte* begin{};
        Byte* end{};
        std::uint64_t last_touch{};
        std::size_t used_bytes{};
        bool initialized{};
    };

    void initialize_code_segments(Byte* begin, std::size_t usable_bytes) {
        constexpr std::size_t minimum_segment_bytes = 16U * 1024U * 1024U;
        constexpr std::size_t target_segment_bytes = 32U * 1024U * 1024U;
        constexpr std::size_t maximum_segment_count = 48U;
        constexpr std::size_t alignment = 4096U;
        const auto target_count = std::max<std::size_t>(1U,
            (usable_bytes + target_segment_bytes - 1U) / target_segment_bytes);
        const auto maximum_by_minimum = std::max<std::size_t>(
            1U, usable_bytes / minimum_segment_bytes);
        const auto segment_count = std::min(
            {maximum_segment_count, target_count, maximum_by_minimum});
        auto segment_bytes = usable_bytes / segment_count;
        segment_bytes &= ~(alignment - 1U);
        if (segment_bytes == 0U)
            segment_bytes = usable_bytes;
        code_segments.clear();
        code_segments.reserve(segment_count);
        code_segments_begin = begin;
        regular_segment_bytes = segment_bytes;
        for (std::size_t index = 0; index < segment_count; ++index) {
            auto* const segment_begin = begin + segment_bytes * index;
            auto* const segment_end = index + 1U == segment_count
                                        ? begin + usable_bytes
                                        : segment_begin + segment_bytes;
            code_segments.push_back(
                CodeSegment{segment_begin, segment_end, 0U, 0U, false});
        }
        reset_code_segments();
    }

    void reset_code_segments() {
        for (auto& segment : code_segments) {
            segment.last_touch = 0U;
            segment.used_bytes = 0U;
            segment.initialized = false;
        }
        current_segment = 0U;
        segment_touch_clock = 1U;
        live_allocated_code_bytes = 0U;
        if (!code_segments.empty()) {
            code_segments.front().initialized = true;
            code_segments.front().last_touch = segment_touch_clock;
        }
    }

    std::optional<std::size_t> code_segment_index(const void* entrypoint) const {
        const auto* const address = static_cast<const Byte*>(entrypoint);
        if (address == nullptr || code_segments.empty() ||
            regular_segment_bytes == 0U || address < code_segments_begin ||
            address >= code_segments.back().end)
            return std::nullopt;
        return std::min(static_cast<std::size_t>(address - code_segments_begin) /
                            regular_segment_bytes,
                        code_segments.size() - 1U);
    }

    void touch_code_segment(const void* entrypoint) const {
        if (const auto index = code_segment_index(entrypoint))
            code_segments[*index].last_touch = ++segment_touch_clock;
    }

    void update_active_segment_usage(const Byte* cursor) {
        auto& segment = code_segments[current_segment];
        const auto occupied = cursor <= segment.begin ? 0U
                            : cursor >= segment.end
                                ? static_cast<std::size_t>(segment.end - segment.begin)
                                : static_cast<std::size_t>(cursor - segment.begin);
        if (occupied > segment.used_bytes) {
            live_allocated_code_bytes += occupied - segment.used_bytes;
            segment.used_bytes = occupied;
        }
    }

    std::size_t active_segment_space_remaining(const Byte* cursor) const {
        const auto* const end = code_segments[current_segment].end;
        return cursor >= end ? 0U : static_cast<std::size_t>(end - cursor);
    }

    std::size_t select_recycle_segment() const {
        if (code_segments.size() <= 1U)
            return 0U;
        for (std::size_t offset = 1U; offset <= code_segments.size(); ++offset) {
            const auto candidate = (current_segment + offset) % code_segments.size();
            if (!code_segments[candidate].initialized)
                return candidate;
        }
        std::size_t victim = current_segment == 0U ? 1U : 0U;
        for (std::size_t index = 0; index < code_segments.size(); ++index) {
            if (index != current_segment &&
                code_segments[index].last_touch < code_segments[victim].last_touch)
                victim = index;
        }
        return victim;
    }

    void activate_code_segment(std::size_t index) {
        auto& segment = code_segments[index];
        live_allocated_code_bytes -= segment.used_bytes;
        segment.initialized = true;
        segment.last_touch = ++segment_touch_clock;
        segment.used_bytes = 0U;
        current_segment = index;
    }

    mutable std::vector<CodeSegment> code_segments;
    Byte* code_segments_begin{};
    std::size_t regular_segment_bytes{};
    std::size_t current_segment{};
    std::size_t live_allocated_code_bytes{};
    mutable std::uint64_t segment_touch_clock{};
};

} // namespace Dynarmic::Backend
