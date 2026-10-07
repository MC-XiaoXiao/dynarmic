/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <cstdio>
#include <map>

#include <boost/container/small_vector.hpp>

#include <mcl/assert.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/opt/passes.h"
#include "dynarmic/ir/type.h"

namespace Dynarmic::Optimization {
namespace {

// NamingPass supplies consecutive names for ordinary A32 blocks. Check both
// the names and referenced pointers before using them as array indices: other
// callers can supply unnamed instructions or references to a different block.
bool VerifyDenseUses(const IR::Block& block) {
    if (block.empty()) {
        return true;
    }
    if (block.front().GetName() != 1) {
        return false;
    }
    struct UseCount {
        const IR::Inst* inst;
        size_t count = 0;
    };
    boost::container::small_vector<UseCount, 32> uses;
    uses.reserve(block.size());
    for (const auto& inst : block) {
        if (inst.GetName() != uses.size() + 1) {
            return false;
        }
        uses.push_back({&inst, 0});
    }

    for (const auto& inst : block) {
        for (size_t i = 0; i < inst.NumArgs(); ++i) {
            const auto arg = inst.GetArg(i);
            if (arg.IsImmediate()) {
                continue;
            }
            const auto* referenced = arg.GetInst();
            const size_t name = referenced->GetName();
            if (name == 0 || name > uses.size() || uses[name - 1].inst != referenced) {
                return false;
            }
            ++uses[name - 1].count;
        }
    }

    for (const auto& use : uses) {
        if (use.count != 0) {
            ASSERT(use.inst->UseCount() == use.count);
        }
    }
    return true;
}

}  // namespace

void VerificationPass(const IR::Block& block) {
    for (const auto& inst : block) {
        for (size_t i = 0; i < inst.NumArgs(); i++) {
            const IR::Type t1 = inst.GetArg(i).GetType();
            const IR::Type t2 = IR::GetArgTypeOf(inst.GetOpcode(), i);
            if (!IR::AreTypesCompatible(t1, t2)) {
                std::puts(IR::DumpBlock(block).c_str());
                ASSERT_FALSE("above block failed validation");
            }
        }
    }

    if (VerifyDenseUses(block)) {
        return;
    }

    std::map<IR::Inst*, size_t> actual_uses;
    for (const auto& inst : block) {
        for (size_t i = 0; i < inst.NumArgs(); i++) {
            const auto arg = inst.GetArg(i);
            if (!arg.IsImmediate()) {
                actual_uses[arg.GetInst()]++;
            }
        }
    }

    for (const auto& pair : actual_uses) {
        ASSERT(pair.first->UseCount() == pair.second);
    }
}

}  // namespace Dynarmic::Optimization
