/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/ir/opcodes.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <stdexcept>

#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/type.h"

namespace Dynarmic::IR {

// Opcode information

namespace OpcodeInfo {

struct Meta {
    constexpr Meta(const char* name, Type type, std::initializer_list<Type> args)
            : name(name), type(type), num_args(static_cast<unsigned>(args.size())) {
        if (args.size() > arg_types.size()) {
            throw std::out_of_range("IR opcode has too many arguments");
        }
        std::copy(args.begin(), args.end(), arg_types.begin());
    }

    const char* name;
    Type type;
    unsigned num_args;
    std::array<Type, max_arg_count> arg_types{};
};

constexpr Type Void = Type::Void;
constexpr Type A32Reg = Type::A32Reg;
constexpr Type A32ExtReg = Type::A32ExtReg;
constexpr Type A64Reg = Type::A64Reg;
constexpr Type A64Vec = Type::A64Vec;
constexpr Type Opaque = Type::Opaque;
constexpr Type U1 = Type::U1;
constexpr Type U8 = Type::U8;
constexpr Type U16 = Type::U16;
constexpr Type U32 = Type::U32;
constexpr Type U64 = Type::U64;
constexpr Type U128 = Type::U128;
constexpr Type CoprocInfo = Type::CoprocInfo;
constexpr Type NZCV = Type::NZCVFlags;
constexpr Type Cond = Type::Cond;
constexpr Type Table = Type::Table;
constexpr Type AccType = Type::AccType;

static constexpr std::array opcode_info{
#define OPCODE(name, type, ...) Meta{#name, type, {__VA_ARGS__}},
#define A32OPC(name, type, ...) Meta{#name, type, {__VA_ARGS__}},
#define A64OPC(name, type, ...) Meta{#name, type, {__VA_ARGS__}},
#include "./opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC
};

}  // namespace OpcodeInfo

namespace {

[[noreturn]] void ThrowArgumentOutOfRange() {
    throw std::out_of_range("IR opcode argument index out of range");
}

}  // namespace

Type GetTypeOf(Opcode op) {
    return OpcodeInfo::opcode_info.at(static_cast<size_t>(op)).type;
}

size_t GetNumArgsOf(Opcode op) {
    return OpcodeInfo::opcode_info.at(static_cast<size_t>(op)).num_args;
}

Type GetArgTypeOf(Opcode op, size_t arg_index) {
    const auto& info = OpcodeInfo::opcode_info.at(static_cast<size_t>(op));
    if (arg_index >= info.num_args) {
        ThrowArgumentOutOfRange();
    }
    return info.arg_types[arg_index];
}

std::string GetNameOf(Opcode op) {
    return OpcodeInfo::opcode_info.at(static_cast<size_t>(op)).name;
}

}  // namespace Dynarmic::IR
