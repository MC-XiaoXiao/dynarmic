/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
class BlockRangeInformation {
public:
    struct Stats {
        std::size_t range_count{};
        std::size_t descriptor_count{};
        std::uint64_t invalidated_descriptors{};
    };

    void AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location);
    void ClearCache();
    tsl::robin_set<IR::LocationDescriptor> InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges);
    void InvalidateLocations(
        const tsl::robin_set<IR::LocationDescriptor>& locations);
    [[nodiscard]] Stats GetStats() const noexcept;

private:
    using Interval = boost::icl::discrete_interval<ProgramCounterType>;
    using IntervalSet = boost::icl::interval_set<ProgramCounterType>;

    // A compiled descriptor normally covers one interval. Allocate the full
    // ICL set only when it is registered again; ICL retains responsibility for
    // joining adjacent/overlapping ranges and handling interval boundaries.
    class DescriptorRanges {
    public:
        explicit DescriptorRanges(Interval range) : first(range) {}

        void Add(Interval range) {
            if (!multiple) {
                multiple = std::make_unique<IntervalSet>();
                multiple->add(first);
            }
            multiple->add(range);
        }

        std::size_t Size() const {
            return multiple ? multiple->iterative_size() : !boost::icl::is_empty(first);
        }

        template<typename Function>
        void ForEach(Function&& function) const {
            if (multiple) {
                for (const auto& range : *multiple) {
                    function(range);
                }
            } else if (!boost::icl::is_empty(first)) {
                function(first);
            }
        }

    private:
        Interval first;
        std::unique_ptr<IntervalSet> multiple;
    };

    // Most intervals name one block. Keep that descriptor in the interval
    // node; overlapping blocks still grow through the existing set algebra.
    using DescriptorSet = boost::container::flat_set<IR::LocationDescriptor,
        std::less<IR::LocationDescriptor>,
        boost::container::small_vector<IR::LocationDescriptor, 1>>;
    using RangeMap = boost::icl::interval_map<ProgramCounterType, DescriptorSet>;
    RangeMap block_ranges;
    // A hint belongs to this map. Discard it on either side of a move,
    // as well as before operations that can erase its node.
    struct InsertionHint {
        std::optional<typename RangeMap::iterator> position;

        InsertionHint() = default;
        InsertionHint(InsertionHint&& other) noexcept { other.position.reset(); }
        InsertionHint& operator=(InsertionHint&& other) noexcept {
            position.reset();
            other.position.reset();
            return *this;
        }
    };
    InsertionHint insertion_hint;
    tsl::robin_map<IR::LocationDescriptor, DescriptorRanges> ranges_by_descriptor;
    std::size_t range_count{};
    std::uint64_t invalidated_descriptors{};
};

}  // namespace Dynarmic::Backend
