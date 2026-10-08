/* This file is part of the dynarmic project. SPDX-License-Identifier: 0BSD */
#pragma once

#include <cstddef>
#include <cstdint>

namespace Dynarmic::A32 {

// Opaque, immutable, process-local compilation result. It owns its bytes and
// relocations independently of the producing Jit. It must never be persisted
// or transferred between host processes or Dynarmic library instances.
// Guest code identity, translation callback semantics and constant
// dependencies remain the client's contract:
// importing a template does not read or validate guest memory.
class NativeCodeTemplate {
public:
    virtual ~NativeCodeTemplate() = default;
    [[nodiscard]] virtual std::uint64_t LocationDescriptor() const noexcept = 0;
    [[nodiscard]] virtual std::size_t CodeSize() const noexcept = 0;
    [[nodiscard]] virtual std::size_t MemoryFootprint() const noexcept = 0;

protected:
    NativeCodeTemplate() = default;
};

}  // namespace Dynarmic::A32
