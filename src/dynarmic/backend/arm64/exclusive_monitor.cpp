/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/interface/exclusive_monitor.h"

#include <algorithm>

#include <mcl/assert.hpp>

namespace Dynarmic {

ExclusiveMonitor::ExclusiveMonitor(size_t processor_count)
        : exclusive_addresses(processor_count, INVALID_EXCLUSIVE_ADDRESS),
          exclusive_values(processor_count),
          reservation_previous(processor_count, INVALID_PROCESSOR_ID),
          reservation_next(processor_count, INVALID_PROCESSOR_ID) {}

size_t ExclusiveMonitor::GetProcessorCount() const {
    return exclusive_addresses.size();
}

void ExclusiveMonitor::Lock() {
    lock.Lock();
}

void ExclusiveMonitor::Unlock() {
    lock.Unlock();
}

bool ExclusiveMonitor::CheckAndClear(size_t processor_id, VAddr address) {
    const VAddr masked_address = address & RESERVATION_GRANULE_MASK;

    Lock();
    if (exclusive_addresses[processor_id] != masked_address) {
        Unlock();
        return false;
    }

    // The resolver-backed path disables generated inline reservation updates,
    // so its index is complete. Without a resolver, generated fast-memory
    // paths may update exclusive_addresses directly; retain the historical
    // scan in that mode for mixed/fallback execution.
    const auto it = HasAddressResolver() ? reservation_heads.find(masked_address)
                                         : reservation_heads.end();
    if (it == reservation_heads.end()) {
        // Fast exclusive-memory paths can write the exposed reservation array
        // directly. Keep the legacy scan as a compatibility fallback when
        // such a path has not populated the indexed reservation list.
        for (VAddr& other_address : exclusive_addresses) {
            if (other_address == masked_address) {
                other_address = INVALID_EXCLUSIVE_ADDRESS;
            }
        }
        return true;
    }

    auto current = it->second;
    while (current != INVALID_PROCESSOR_ID) {
        const auto next = reservation_next[current];
        exclusive_addresses[current] = INVALID_EXCLUSIVE_ADDRESS;
        reservation_previous[current] = INVALID_PROCESSOR_ID;
        reservation_next[current] = INVALID_PROCESSOR_ID;
        current = next;
    }
    reservation_heads.erase(it);
    return true;
}

void ExclusiveMonitor::Clear() {
    Lock();
    std::fill(exclusive_addresses.begin(), exclusive_addresses.end(), INVALID_EXCLUSIVE_ADDRESS);
    std::fill(reservation_previous.begin(), reservation_previous.end(), INVALID_PROCESSOR_ID);
    std::fill(reservation_next.begin(), reservation_next.end(), INVALID_PROCESSOR_ID);
    reservation_heads.clear();
    Unlock();
}

void ExclusiveMonitor::ClearProcessor(size_t processor_id) {
    Lock();
    RemoveReservation(processor_id);
    exclusive_addresses[processor_id] = INVALID_EXCLUSIVE_ADDRESS;
    Unlock();
}

}  // namespace Dynarmic
