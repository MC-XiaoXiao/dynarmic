/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>

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
    [[nodiscard]] Stats GetStats() const noexcept;

private:
    boost::icl::interval_map<ProgramCounterType, std::set<IR::LocationDescriptor>> block_ranges;
    std::map<IR::LocationDescriptor, boost::icl::interval_set<ProgramCounterType>>
        ranges_by_descriptor;
    std::size_t range_count{};
    std::uint64_t invalidated_descriptors{};
};

}  // namespace Dynarmic::Backend
