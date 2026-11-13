/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/block_range_information.h"

#include <bit>
#include <vector>

#include <boost/icl/interval_set.hpp>
#include <mcl/stdint.hpp>
#include <tsl/robin_set.h>

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location) {
    auto [descriptor_it, inserted] = ranges_by_descriptor.try_emplace(location, range);
    auto& descriptor_ranges = descriptor_it->second;
    if (inserted) {
        try {
            block_ranges.Add(range, location);
        } catch (...) {
            ranges_by_descriptor.erase(descriptor_it);
            throw;
        }
        range_count += descriptor_ranges.Size();
        return;
    }
    // Prepare the complete replacement before changing live nodes. Allocation
    // failure must leave every previous interval queryable.
    DescriptorRanges replacement{range};
    descriptor_ranges.ForEach([&](const auto& previous) { replacement.Add(previous); });
    std::vector<typename RangeIndex::PreparedRange> prepared;
    prepared.reserve(replacement.Size());
    replacement.ForEach([&](const auto& current) {
        prepared.push_back(RangeIndex::Prepare(current, location));
    });
    const auto previous_range_count = descriptor_ranges.Size();
    descriptor_ranges.ForEach([&](const auto& previous) { block_ranges.Remove(previous, location); });
    for (auto& entry : prepared) block_ranges.AddPrepared(std::move(entry));
    descriptor_ranges.Swap(replacement);
    const auto current_range_count = descriptor_ranges.Size();
    if (current_range_count >= previous_range_count) {
        range_count += current_range_count - previous_range_count;
    } else {
        range_count -= previous_range_count - current_range_count;
    }
}

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::ClearCache() {
    block_ranges.Clear();
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
    block_ranges.Clear();
    // A recycled segment will be filled again. Reuse its bounded bucket
    // storage; a full ClearCache still releases that storage.
    ranges_by_descriptor.clear();
    range_count = 0;
    return locations;
}

template<typename ProgramCounterType>
tsl::robin_set<IR::LocationDescriptor> BlockRangeInformation<ProgramCounterType>::InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges) {
    tsl::robin_set<IR::LocationDescriptor> erase_locations;
    for (const auto& interval : ranges)
        block_ranges.Find(interval, erase_locations);

    InvalidateLocations(erase_locations);
    return erase_locations;
}

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::InvalidateLocations(
        const tsl::robin_set<IR::LocationDescriptor>& locations) {
    // Remove every interval of a retired descriptor, including disjoint
    // registrations. Large retirements filter the augmented tree once.
    invalidated_descriptors += locations.size();
    const bool bulk = locations.size() * std::bit_width(ranges_by_descriptor.size()) > range_count;
    if (bulk) block_ranges.RemoveLocations(locations);
    for (const auto& descriptor : locations) {
        const auto descriptor_it = ranges_by_descriptor.find(descriptor);
        if (descriptor_it == ranges_by_descriptor.end()) {
            continue;
        }

        if (!bulk) {
            descriptor_it->second.ForEach([&](const auto& descriptor_range) {
                block_ranges.Remove(descriptor_range, descriptor);
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
