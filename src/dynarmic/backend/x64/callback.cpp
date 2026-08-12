/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/x64/callback.h"

#include <utility>

#include "dynarmic/backend/x64/block_of_code.h"

namespace Dynarmic::Backend::X64 {

Callback::~Callback() = default;

void SimpleCallback::EmitCall(BlockOfCode& code, std::function<void(RegList)> l) const {
    l({code.ABI_PARAM1, code.ABI_PARAM2, code.ABI_PARAM3, code.ABI_PARAM4});
    code.CallFunction(fn);
}

void SimpleCallback::EmitCallWithReturnPointer(BlockOfCode& code, std::function<void(Xbyak::Reg64, RegList)> l) const {
    l(code.ABI_PARAM1, {code.ABI_PARAM2, code.ABI_PARAM3, code.ABI_PARAM4});
    code.CallFunction(fn);
}

void ArgCallback::EmitCall(BlockOfCode& code, std::function<void(RegList)> l) const {
    l({code.ABI_PARAM2, code.ABI_PARAM3, code.ABI_PARAM4});
    LoadArg(code, code.ABI_PARAM1);
    code.CallFunction(fn);
}

void ArgCallback::EmitCallWithReturnPointer(BlockOfCode& code, std::function<void(Xbyak::Reg64, RegList)> l) const {
#if defined(WIN32) && !defined(__MINGW64__)
    l(code.ABI_PARAM2, {code.ABI_PARAM3, code.ABI_PARAM4});
    LoadArg(code, code.ABI_PARAM1);
#else
    l(code.ABI_PARAM1, {code.ABI_PARAM3, code.ABI_PARAM4});
    LoadArg(code, code.ABI_PARAM2);
#endif
    code.CallFunction(fn);
}

void ArgCallback::LoadArg(BlockOfCode& code, const Xbyak::Reg64& destination) const {
    if (link_offset == direct_link_offset) {
        code.mov(destination, arg);
        return;
    }
    code.mov(destination, code.qword[code.r15 + link_offset]);
    code.mov(destination, code.qword[destination]);
}

ArgCallbackFromLink::ArgCallbackFromLink(ArgCallback callback_, std::size_t link_offset_)
        : callback(std::move(callback_))
        , link_offset(link_offset_) {}

void ArgCallbackFromLink::LoadArg(BlockOfCode& code, const Xbyak::Reg64& destination) const {
    code.mov(destination, code.qword[code.r15 + link_offset]);
    code.mov(destination, code.qword[destination]);
}

void ArgCallbackFromLink::EmitCall(BlockOfCode& code, std::function<void(RegList)> l) const {
    l({code.ABI_PARAM2, code.ABI_PARAM3, code.ABI_PARAM4});
    LoadArg(code, code.ABI_PARAM1);
    code.CallFunction(callback.fn);
}

void ArgCallbackFromLink::EmitCallWithReturnPointer(BlockOfCode& code, std::function<void(Xbyak::Reg64, RegList)> l) const {
#if defined(WIN32) && !defined(__MINGW64__)
    l(code.ABI_PARAM2, {code.ABI_PARAM3, code.ABI_PARAM4});
    LoadArg(code, code.ABI_PARAM1);
#else
    l(code.ABI_PARAM1, {code.ABI_PARAM3, code.ABI_PARAM4});
    LoadArg(code, code.ABI_PARAM2);
#endif
    code.CallFunction(callback.fn);
}

}  // namespace Dynarmic::Backend::X64
