/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <mcl/scope_exit.hpp>

#include "dynarmic/backend/arm64/a32_address_space.h"
#include "dynarmic/backend/arm64/a32_jitstate.h"

namespace Dynarmic::Backend::Arm64 {

class A32Core final {
public:
    explicit A32Core(const A32::UserConfig& conf) : callbacks{conf.callbacks} {}

    HaltReason Run(A32AddressSpace& process, A32JitState& thread_ctx, volatile u32* halt_reason) {
        const auto location_descriptor = thread_ctx.GetLocationDescriptor();
        const auto entry_point = process.GetOrEmit(location_descriptor);
        callbacks->MemoryExecutionResume();
        SCOPE_EXIT { callbacks->MemoryExecutionSuspend(); };
        return process.prelude_info.run_code(entry_point, &thread_ctx, halt_reason);
    }

    HaltReason Step(A32AddressSpace& process, A32JitState& thread_ctx, volatile u32* halt_reason) {
        const auto location_descriptor = A32::LocationDescriptor{thread_ctx.GetLocationDescriptor()}.SetSingleStepping(true);
        const auto entry_point = process.GetOrEmit(location_descriptor);
        callbacks->MemoryExecutionResume();
        SCOPE_EXIT { callbacks->MemoryExecutionSuspend(); };
        return process.prelude_info.step_code(entry_point, &thread_ctx, halt_reason);
    }

private:
    A32::UserCallbacks* callbacks;
};

}  // namespace Dynarmic::Backend::Arm64
