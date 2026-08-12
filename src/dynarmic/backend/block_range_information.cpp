/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/block_range_information.h"

#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include <mcl/stdint.hpp>
#include <tsl/robin_set.h>

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location) {
    block_ranges.add(std::make_pair(range, std::set<IR::LocationDescriptor>{location}));

    auto& descriptor_ranges = ranges_by_descriptor[location];
    const auto previous_range_count = descriptor_ranges.iterative_size();
    descriptor_ranges.add(range);
    const auto current_range_count = descriptor_ranges.iterative_size();
    if (current_range_count >= previous_range_count) {
        range_count += current_range_count - previous_range_count;
    } else {
        range_count -= previous_range_count - current_range_count;
    }
}

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::ClearCache() {
    block_ranges.clear();
    ranges_by_descriptor.clear();
    range_count = 0;
}

template<typename ProgramCounterType>
tsl::robin_set<IR::LocationDescriptor> BlockRangeInformation<ProgramCounterType>::InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges) {
    tsl::robin_set<IR::LocationDescriptor> erase_locations;
    for (auto invalidate_interval : ranges) {
        auto pair = block_ranges.equal_range(invalidate_interval);
        for (auto it = pair.first; it != pair.second; ++it) {
            for (const auto& descriptor : it->second) {
                erase_locations.insert(descriptor);
            }
        }
    }

    // A descriptor is invalidated as a unit. Removing only the intersection
    // would leave its other old ranges discoverable by a later invalidation.
    // The reverse index keeps this cleanup proportional to the invalidated
    // descriptors and their recorded ranges instead of the whole map.
    invalidated_descriptors += erase_locations.size();
    for (const auto& descriptor : erase_locations) {
        const auto descriptor_it = ranges_by_descriptor.find(descriptor);
        if (descriptor_it == ranges_by_descriptor.end()) {
            continue;
        }

        for (const auto& descriptor_range : descriptor_it->second) {
            block_ranges.subtract(std::make_pair(
                descriptor_range,
                std::set<IR::LocationDescriptor>{descriptor}));
        }
        range_count -= descriptor_it->second.iterative_size();
        ranges_by_descriptor.erase(descriptor_it);
    }
    return erase_locations;
}

template<typename ProgramCounterType>
typename BlockRangeInformation<ProgramCounterType>::Stats
BlockRangeInformation<ProgramCounterType>::GetStats() const noexcept {
    return Stats{range_count, ranges_by_descriptor.size(),
                 invalidated_descriptors};
}

template class BlockRangeInformation<u32>;
template class BlockRangeInformation<u64>;

}  // namespace Dynarmic::Backend
