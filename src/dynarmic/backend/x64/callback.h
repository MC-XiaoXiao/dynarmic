/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

#include <mcl/stdint.hpp>
#include <xbyak/xbyak.h>

namespace Dynarmic::Backend::X64 {

using RegList = std::vector<Xbyak::Reg64>;

class BlockOfCode;

class Callback {
public:
    virtual ~Callback();

    void EmitCall(BlockOfCode& code) const {
        EmitCall(code, [](RegList) {});
    }

    virtual void EmitCall(BlockOfCode& code, std::function<void(RegList)> fn) const = 0;
    virtual void EmitCallWithReturnPointer(BlockOfCode& code, std::function<void(Xbyak::Reg64, RegList)> fn) const = 0;
};

class SimpleCallback final : public Callback {
public:
    template<typename Function>
    SimpleCallback(Function fn)
            : fn(reinterpret_cast<void (*)()>(fn)) {}

    using Callback::EmitCall;

    void EmitCall(BlockOfCode& code, std::function<void(RegList)> fn) const override;
    void EmitCallWithReturnPointer(BlockOfCode& code, std::function<void(Xbyak::Reg64, RegList)> fn) const override;

private:
    void (*fn)();
};

class ArgCallback final : public Callback {
public:
    template<typename Function>
    ArgCallback(Function fn, u64 arg)
            : fn(reinterpret_cast<void (*)()>(fn)), arg(arg) {}

    static ArgCallback FromLink(ArgCallback callback, std::size_t link_offset) {
        callback.link_offset = link_offset;
        return callback;
    }

    using Callback::EmitCall;

    void EmitCall(BlockOfCode& code, std::function<void(RegList)> fn) const override;
    void EmitCallWithReturnPointer(BlockOfCode& code, std::function<void(Xbyak::Reg64, RegList)> fn) const override;

private:
    friend class ArgCallbackFromLink;
    void LoadArg(BlockOfCode& code, const Xbyak::Reg64& destination) const;

    static constexpr std::size_t direct_link_offset =
            std::numeric_limits<std::size_t>::max();
    void (*fn)();
    u64 arg;
    std::size_t link_offset = direct_link_offset;
};

// Callback whose first argument is loaded through a pointer stored in the
// active JitState. The emitted load sequence is context-independent: only the
// mutable JitState chooses the executor-owned link cell.
class ArgCallbackFromLink final : public Callback {
public:
    ArgCallbackFromLink(ArgCallback callback, std::size_t link_offset);

    using Callback::EmitCall;

    void EmitCall(BlockOfCode& code, std::function<void(RegList)> fn) const override;
    void EmitCallWithReturnPointer(BlockOfCode& code, std::function<void(Xbyak::Reg64, RegList)> fn) const override;

private:
    void LoadArg(BlockOfCode& code, const Xbyak::Reg64& destination) const;

    ArgCallback callback;
    std::size_t link_offset;
};

}  // namespace Dynarmic::Backend::X64
