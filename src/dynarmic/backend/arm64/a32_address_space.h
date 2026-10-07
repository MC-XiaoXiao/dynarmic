/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <optional>
#include <utility>

#include "dynarmic/backend/arm64/address_space.h"
#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/interface/A32/a32.h"

namespace Dynarmic::Backend::Arm64 {

struct EmittedBlockInfo;

class A32AddressSpace final : public AddressSpace {
public:
    explicit A32AddressSpace(const A32::UserConfig& conf);

    IR::Block GenerateIR(IR::LocationDescriptor) const override;
    IR::Block TranslateIR(IR::LocationDescriptor) const;
    A32::Jit::PortableIREmitOutcome Precompile(IR::Block& block);
    void SetPortableIRDemandProvider(A32::Jit::PortableIRDemandProvider provider, void* user_arg);
    void SetPortableIREmitCompletion(A32::Jit::PortableIREmitCompletion completion, void* user_arg);

    CodePtr GetOrEmit(IR::LocationDescriptor descriptor) override;
    CodePtr GetOrEmit(IR::LocationDescriptor descriptor, StackLayout& stack);

    void InvalidateCacheRanges(const boost::icl::interval_set<u32>& ranges);
    void ClearCache() override;

protected:
    friend class A32Core;

    void EmitPrelude();
    EmitConfig GetEmitConfig() override;
    void CodeTranslationCompleted(const IR::Block& block, u64 translation_nanoseconds) const noexcept override;
    void RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo& block_info) override;

    const A32::UserConfig conf;
    BlockRangeInformation<u32> block_ranges;

private:
    void CompletePortableEmit(A32::Jit::PortableIREmitOutcome outcome) const noexcept;
    A32::Jit::PortableIRDemandProvider portable_provider{};
    void* portable_provider_arg{};
    A32::Jit::PortableIREmitCompletion portable_completion{};
    void* portable_completion_arg{};
    // Set only after a provider block has been handed off. Ordinary
    // translation exceptions must retain their normal error semantics.
    mutable std::optional<std::pair<u64, u64>> portable_emit;
};

}  // namespace Dynarmic::Backend::Arm64
