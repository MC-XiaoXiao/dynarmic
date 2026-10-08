/* This file is part of the dynarmic project.
 * Copyright (c) 2026 Dynarmic Contributors
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace Dynarmic::Decoder {

/// An ordered matcher table with a rejection test derived from all its entries.
template<typename MatcherT>
class MatcherTable {
public:
    using opcode_type = typename MatcherT::opcode_type;

    explicit MatcherTable(std::vector<MatcherT> entries)
            : entries{std::move(entries)} {
        if (this->entries.empty()) {
            return;
        }
        common_mask = this->entries.front().GetMask();
        common_expected = this->entries.front().GetExpected();
        for (const auto& matcher : this->entries) {
            // Only retain bits fixed to the same value by every matcher.
            common_mask &= matcher.GetMask() & ~(common_expected ^ matcher.GetExpected());
        }
        common_expected &= common_mask;
    }

    std::optional<std::reference_wrapper<const MatcherT>> Find(opcode_type instruction) const {
        if ((instruction & common_mask) != common_expected) {
            return std::nullopt;
        }
        const auto iter = std::find_if(entries.begin(), entries.end(), [instruction](const auto& matcher) {
            return matcher.Matches(instruction);
        });
        return iter != entries.end() ? std::optional<std::reference_wrapper<const MatcherT>>{*iter} : std::nullopt;
    }

private:
    std::vector<MatcherT> entries;
    opcode_type common_mask = 0;
    opcode_type common_expected = 0;
};

}  // namespace Dynarmic::Decoder
