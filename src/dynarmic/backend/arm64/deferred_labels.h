/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <cstddef>
#include <list>
#include <memory_resource>
#include <optional>

#include <oaknut/oaknut.hpp>

namespace Dynarmic::Backend::Arm64 {

// Labels are borrowed by deferred emitters and owned by their block context.
// A list keeps their addresses stable when a large block exhausts the buffer.
class DeferredLabels {
public:
    oaknut::Label* Create() {
        if (!storage_) {
            storage_.emplace();
        }
        storage_->labels.emplace_back();
        return &storage_->labels.back();
    }

private:
    struct Storage {
        Storage() : memory(buffer.data(), buffer.size()), labels(&memory) {}

        // Leave the buffer uninitialized; construction does not read it.
        std::array<std::byte, 1024> buffer;
        std::pmr::monotonic_buffer_resource memory;
        std::pmr::list<oaknut::Label> labels;
    };

    // Arithmetic-only blocks do not construct the resource or the list.
    std::optional<Storage> storage_;
};

}  // namespace Dynarmic::Backend::Arm64
