/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include <mcl/stdint.hpp>

#include "dynarmic/frontend/decoder/decoder_detail.h"
#include "dynarmic/frontend/decoder/matcher.h"

namespace Dynarmic::A32 {

template<typename Visitor>
using Thumb32Matcher = Decoder::Matcher<Visitor, u32>;

template<typename V>
std::optional<std::reference_wrapper<const Thumb32Matcher<V>>> DecodeThumb32(u32 instruction) {
    static const std::vector<Thumb32Matcher<V>> matchers = {

#define INST(fn, name, bitstring) DYNARMIC_DECODER_GET_MATCHER(Thumb32Matcher, fn, name, Decoder::detail::StringToArray<32>(bitstring)),
#include "./thumb32.inc"
#undef INST

    };

    // Partition on the first halfword's opcode, retaining the original match
    // order and full-mask check. References keep each handler stored once.
    static const auto table = [] {
        std::array<std::vector<const Thumb32Matcher<V>*>, 0x100> buckets;
        for (size_t i = 0; i < buckets.size(); ++i) {
            for (const auto& matcher : matchers) {
                const auto mask = (matcher.GetMask() >> 20) & 0xFF;
                const auto expected = (matcher.GetExpected() >> 20) & 0xFF;
                if ((i & mask) == expected) {
                    buckets[i].push_back(&matcher);
                }
            }
        }
        return buckets;
    }();

    const auto matches_instruction = [instruction](const auto* matcher) { return matcher->Matches(instruction); };

    const auto& subtable = table[(instruction >> 20) & 0xFF];
    auto iter = std::find_if(subtable.begin(), subtable.end(), matches_instruction);
    return iter != subtable.end() ? std::optional<std::reference_wrapper<const Thumb32Matcher<V>>>(**iter) : std::nullopt;
}

}  // namespace Dynarmic::A32
