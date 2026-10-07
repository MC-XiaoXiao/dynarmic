/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include "dynarmic/interface/A32/config.h"

namespace Dynarmic::Backend::Arm64 {

// Compilation and reservation-monitor operations may acquire checked-memory
// locks. Release this lane's native lease before acquiring those locks, and
// resume only after the host operation has released them.
class A32MemoryExecutionPause final {
public:
    explicit A32MemoryExecutionPause(const A32::UserConfig& conf)
            : callbacks{conf.memory_execution_scope_active && *conf.memory_execution_scope_active ? conf.callbacks : nullptr} {
        if (callbacks)
            callbacks->MemoryExecutionSuspend();
    }
    ~A32MemoryExecutionPause() {
        if (callbacks)
            callbacks->MemoryExecutionResume();
    }
    A32MemoryExecutionPause(const A32MemoryExecutionPause&) = delete;
    A32MemoryExecutionPause& operator=(const A32MemoryExecutionPause&) = delete;

private:
    A32::UserCallbacks* callbacks;
};

}  // namespace Dynarmic::Backend::Arm64
