/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>

#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
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
    // Most intervals name one block. Keep that descriptor in the interval
    // node; overlapping blocks still grow through the existing set algebra.
    using DescriptorSet = boost::container::flat_set<IR::LocationDescriptor,
        std::less<IR::LocationDescriptor>,
        boost::container::small_vector<IR::LocationDescriptor, 1>>;
    boost::icl::interval_map<ProgramCounterType, DescriptorSet> block_ranges;
    std::map<IR::LocationDescriptor, boost::icl::interval_set<ProgramCounterType>>
        ranges_by_descriptor;
    std::size_t range_count{};
    std::uint64_t invalidated_descriptors{};
};

}  // namespace Dynarmic::Backend
