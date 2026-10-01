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
    for (auto iter = block.begin(); iter != block.end(); ++iter) {
        auto& inst = *iter;
        const auto propagate_value = [&](IR::Value value) {
            // Read-only bytes are constant, but reading them can still fault.
            // Keep the original access at the same position and predicate;
            // propagate only its value through the following instructions.
            block.PrependNewInst(iter, inst.GetOpcode(),
                                 {inst.GetArg(0), inst.GetArg(1), inst.GetArg(2)});
            inst.ReplaceUsesWith(value);
        };
        switch (inst.GetOpcode()) {
        case IR::Opcode::A32ReadMemory8: {
            if (!inst.AreAllArgsImmediates()) {
                break;
            }

            const u32 vaddr = inst.GetArg(1).GetU32();
            if (cb->IsReadOnlyMemory(vaddr)) {
                const u8 value_from_memory = static_cast<u8>(cb->MemoryReadConstant(vaddr, 1U));
                propagate_value(IR::Value{value_from_memory});
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
                propagate_value(IR::Value{value_from_memory});
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
                propagate_value(IR::Value{value_from_memory});
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
                propagate_value(IR::Value{value_from_memory});
            }
            break;
        }
        default:
            break;
        }
    }
}

}  // namespace Dynarmic::Optimization
