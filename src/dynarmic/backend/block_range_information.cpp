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
    block_ranges.add(std::make_pair(range, DescriptorSet{location}));

    auto [descriptor_it, inserted] = ranges_by_descriptor.try_emplace(location, range);
    auto& descriptor_ranges = descriptor_it.value();
    if (inserted) {
        range_count += descriptor_ranges.Size();
        return;
    }
    const auto previous_range_count = descriptor_ranges.Size();
    descriptor_ranges.Add(range);
    const auto current_range_count = descriptor_ranges.Size();
    if (current_range_count >= previous_range_count) {
        range_count += current_range_count - previous_range_count;
    } else {
        range_count -= previous_range_count - current_range_count;
    }
}

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::ClearCache() {
    block_ranges.clear();
    // A full cache release must also return the hash table's bucket storage.
    decltype(ranges_by_descriptor){}.swap(ranges_by_descriptor);
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

    InvalidateLocations(erase_locations);
    return erase_locations;
}

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::InvalidateLocations(
        const tsl::robin_set<IR::LocationDescriptor>& locations) {
    // A descriptor is invalidated as a unit. Removing only one interval would
    // leave its other old ranges discoverable by a later invalidation. This
    // exact-location path is also used when a host-code segment is recycled.
    invalidated_descriptors += locations.size();
    for (const auto& descriptor : locations) {
        const auto descriptor_it = ranges_by_descriptor.find(descriptor);
        if (descriptor_it == ranges_by_descriptor.end()) {
            continue;
        }

        descriptor_it->second.ForEach([&](const auto& descriptor_range) {
            block_ranges.subtract(std::make_pair(
                descriptor_range,
                DescriptorSet{descriptor}));
        });
        range_count -= descriptor_it->second.Size();
        ranges_by_descriptor.erase(descriptor_it);
    }
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
