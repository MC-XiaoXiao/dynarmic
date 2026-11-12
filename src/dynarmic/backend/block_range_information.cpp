/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/block_range_information.h"

#include <bit>
#include <iterator>
#include <vector>

#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include <mcl/stdint.hpp>
#include <tsl/robin_set.h>

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location) {
    // Compilation tends to visit neighboring guest blocks even when their
    // image is in the middle of the map. ICL still handles arbitrary order,
    // overlapping intervals and codomain merging through its normal add path.
    const auto hint = insertion_hint.position ? std::next(*insertion_hint.position) : block_ranges.end();
    const auto inserted_range = block_ranges.add(hint, std::make_pair(range, DescriptorSet{location}));
    if (inserted_range == block_ranges.end()) {
        insertion_hint.position.reset();
    } else {
        insertion_hint.position = inserted_range;
    }

    auto [descriptor_it, inserted] = ranges_by_descriptor.try_emplace(location, range);
    auto& descriptor_ranges = descriptor_it->second;
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
    insertion_hint.position.reset();
    block_ranges.clear();
    // A full cache release must also return the hash table's bucket storage.
    ranges_by_descriptor.clear();
    ranges_by_descriptor = {};
    range_count = 0;
}

template<typename ProgramCounterType>
std::vector<IR::LocationDescriptor> BlockRangeInformation<ProgramCounterType>::InvalidateAll() {
    std::vector<IR::LocationDescriptor> locations;
    locations.reserve(ranges_by_descriptor.size());
    for (const auto& [location, ranges] : ranges_by_descriptor)
        locations.push_back(location);
    invalidated_descriptors += locations.size();
    insertion_hint.position.reset();
    block_ranges.clear();
    // A recycled segment will be filled again. Reuse its bounded bucket
    // storage; a full ClearCache still releases that storage.
    ranges_by_descriptor.clear();
    range_count = 0;
    return locations;
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
    if (!locations.empty()) {
        insertion_hint.position.reset();
    }
    invalidated_descriptors += locations.size();
    // Small guest writes use the reverse index. A segment retirement can
    // remove a substantial fraction of the map; visit its interval nodes once
    // instead of repeatedly searching the tree for each retired descriptor.
    const bool bulk = locations.size() * std::bit_width(ranges_by_descriptor.size()) >
                      block_ranges.iterative_size();
    if (bulk) {
        std::vector<std::pair<Interval, DescriptorSet>> partial;
        boost::icl::erase_if([&](const auto& entry) {
            DescriptorSet retired;
            for (const auto& descriptor : entry.second) {
                if (locations.contains(descriptor))
                    retired.insert(descriptor);
            }
            if (retired.size() == entry.second.size())
                return true;
            if (!retired.empty())
                partial.emplace_back(entry.first, std::move(retired));
            return false;
        }, block_ranges);
        // ICL owns splitting and joining intervals with overlapping live
        // descriptors. Apply these subtractions after the traversal ends.
        for (const auto& entry : partial)
            block_ranges.subtract(entry);
    }
    for (const auto& descriptor : locations) {
        const auto descriptor_it = ranges_by_descriptor.find(descriptor);
        if (descriptor_it == ranges_by_descriptor.end()) {
            continue;
        }

        if (!bulk) {
            descriptor_it->second.ForEach([&](const auto& descriptor_range) {
                block_ranges.subtract(std::make_pair(
                    descriptor_range,
                    DescriptorSet{descriptor}));
            });
        }
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
