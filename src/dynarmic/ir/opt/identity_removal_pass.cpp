/* This file is part of the dynarmic project.
 * Copyright (c) 2020 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/opt/passes.h"

namespace Dynarmic::Optimization {

void IdentityRemovalPass(IR::Block& block) {
    // Resolve every use before invalidating identities. Keeping the nodes in
    // the block until then avoids a per-block removal buffer and preserves
    // the lifetime of values referenced by later instructions.
    for (IR::Inst& inst : block) {
        const size_t num_args = inst.NumArgs();
        for (size_t i = 0; i < num_args; i++) {
            while (true) {
                IR::Value arg = inst.GetArg(i);
                if (!arg.IsIdentity())
                    break;
                inst.SetArg(i, arg.GetInst()->GetArg(0));
            }
        }
    }

    auto iter = block.begin();
    while (iter != block.end()) {
        IR::Inst& inst = *iter;
        if (inst.GetOpcode() == IR::Opcode::Identity || inst.GetOpcode() == IR::Opcode::Void) {
            iter = block.Instructions().erase(inst);
            inst.Invalidate();
        } else {
            ++iter;
        }
    }

}

}  // namespace Dynarmic::Optimization
