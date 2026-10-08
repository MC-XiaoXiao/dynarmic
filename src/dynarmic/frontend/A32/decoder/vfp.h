/* This file is part of the dynarmic project.
 * Copyright (c) 2032 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <vector>

#include <mcl/stdint.hpp>

#include "dynarmic/frontend/decoder/decoder_detail.h"
#include "dynarmic/frontend/decoder/matcher.h"
#include "dynarmic/frontend/decoder/matcher_table.h"

namespace Dynarmic::A32 {

template<typename Visitor>
using VFPMatcher = Decoder::Matcher<Visitor, u32>;

template<typename V>
std::optional<std::reference_wrapper<const VFPMatcher<V>>> DecodeVFP(u32 instruction) {
    using List = std::vector<VFPMatcher<V>>;
    using Table = Decoder::MatcherTable<VFPMatcher<V>>;

    static const struct Tables {
        Table unconditional;
        Table conditional;
    } tables = [] {
        List list = {

#define INST(fn, name, bitstring) DYNARMIC_DECODER_GET_MATCHER(VFPMatcher, fn, name, Decoder::detail::StringToArray<32>(bitstring)),
#include "./vfp.inc"
#undef INST

        };

        const auto division = std::stable_partition(list.begin(), list.end(), [&](const auto& matcher) {
            return (matcher.GetMask() & 0xF0000000) == 0xF0000000;
        });

        return Tables{
            Table{List{list.begin(), division}},
            Table{List{division, list.end()}},
        };
    }();

    const bool is_unconditional = (instruction & 0xF0000000) == 0xF0000000;
    const Table& table = is_unconditional ? tables.unconditional : tables.conditional;

    return table.Find(instruction);
}

}  // namespace Dynarmic::A32
