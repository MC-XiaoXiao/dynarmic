/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/interface/A32/config.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/opt/passes.h"

namespace Dynarmic::Optimization {

void A32ConstantMemoryReads(IR::Block& block, A32::UserCallbacks* cb) {
    for (auto& inst : block) {
        switch (inst.GetOpcode()) {
        case IR::Opcode::A32ReadMemory8: {
            if (!inst.AreAllArgsImmediates()) {
                break;
            }

            const u32 vaddr = inst.GetArg(1).GetU32();
            if (cb->IsReadOnlyMemory(vaddr)) {
                const u8 value_from_memory = static_cast<u8>(cb->MemoryReadConstant(vaddr, 1U));
                inst.ReplaceUsesWith(IR::Value{value_from_memory});
            }
            break;
        }
        case IR::Opcode::A32ReadMemory16: {
            if (!inst.AreAllArgsImmediates()) {
                break;
            }

            const u32 vaddr = inst.GetArg(1).GetU32();
            if (cb->IsReadOnlyMemory(vaddr)) {
                const u16 value_from_memory = static_cast<u16>(cb->MemoryReadConstant(vaddr, 2U));
                inst.ReplaceUsesWith(IR::Value{value_from_memory});
            }
            break;
        }
        case IR::Opcode::A32ReadMemory32: {
            if (!inst.AreAllArgsImmediates()) {
                break;
            }

            const u32 vaddr = inst.GetArg(1).GetU32();
            if (cb->IsReadOnlyMemory(vaddr)) {
                const u32 value_from_memory = static_cast<u32>(cb->MemoryReadConstant(vaddr, 4U));
                inst.ReplaceUsesWith(IR::Value{value_from_memory});
            }
            break;
        }
        case IR::Opcode::A32ReadMemory64: {
            if (!inst.AreAllArgsImmediates()) {
                break;
            }

            const u32 vaddr = inst.GetArg(1).GetU32();
            if (cb->IsReadOnlyMemory(vaddr)) {
                const u64 value_from_memory = static_cast<u64>(cb->MemoryReadConstant(vaddr, 8U));
                inst.ReplaceUsesWith(IR::Value{value_from_memory});
            }
            break;
        }
        default:
            break;
        }
    }
}

}  // namespace Dynarmic::Optimization
