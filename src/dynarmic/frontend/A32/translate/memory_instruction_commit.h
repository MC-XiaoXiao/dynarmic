/* SPDX-License-Identifier: 0BSD */
#pragma once

#include <array>
#include <iterator>
#include <vector>
#include "dynarmic/frontend/A32/a32_types.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::A32 {

// Keep the base and GPR load destinations restartable until the instruction's
// final memory access succeeds. Extended-register load prefixes may remain
// visible on abort, but cannot change the addresses used by a retry.
class MemoryInstructionCommit {
public:
    explicit MemoryInstructionCommit(IR::Block& block)
        : block_(block), previous_(block.empty() ? nullptr : &block.back()) {}

    ~MemoryInstructionCommit() {
        auto first = previous_ ? std::next(IR::Block::iterator{previous_}) : block_.begin();
        auto last_memory = block_.end();
        for (auto it = first; it != block_.end(); ++it)
            if (it->IsMemoryReadOrWrite())
                last_memory = it;
        if (last_memory == block_.end())
            return;

        std::array<IR::Value, 16> values;
        std::vector<IR::Inst*> stores;
        bool before_last_memory = true;
        for (auto it = first; it != block_.end(); ++it) {
            if (it == last_memory)
                before_last_memory = false;
            if (it->GetOpcode() == IR::Opcode::A32GetRegister) {
                const auto reg = static_cast<std::size_t>(it->GetArg(0).GetA32RegRef());
                if (!values[reg].IsEmpty())
                    it->ReplaceUsesWith(values[reg]);
            } else if (it->GetOpcode() == IR::Opcode::A32SetRegister) {
                const auto reg = static_cast<std::size_t>(it->GetArg(0).GetA32RegRef());
                values[reg] = it->GetArg(1);
                if (before_last_memory)
                    stores.push_back(&*it);
            }
        }
        // Forwarded gets preserve within-instruction overlap behavior when an
        // address helper publishes writeback before obtaining a store operand.
        const auto commit = std::next(last_memory);
        for (auto* store : stores) {
            block_.Instructions().erase(store);
            block_.Instructions().insert(commit, store);
        }
    }

private:
    IR::Block& block_;
    IR::Inst* previous_;
};
}
