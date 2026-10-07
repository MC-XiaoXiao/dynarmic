/* This file is part of the dynarmic project. SPDX-License-Identifier: 0BSD */
#include "dynarmic/backend/arm64/a32_ir_translator.h"
#include "dynarmic/backend/arm64/a32_memory_execution_scope.h"
#include "dynarmic/frontend/A32/a32_location_descriptor.h"
#include "dynarmic/frontend/A32/translate/a32_translate.h"
#include "dynarmic/ir/opt/passes.h"
namespace Dynarmic::Backend::Arm64 {
IR::Block TranslateA32IR(const A32::UserConfig& conf, IR::LocationDescriptor descriptor) {
    auto* const callbacks = GetA32RuntimeCallbacks(conf);
    IR::Block ir_block = A32::Translate(A32::LocationDescriptor{descriptor}, callbacks, {conf.arch_version, conf.define_unpredictable_behaviour, conf.hook_hint_instructions});

    Optimization::PolyfillPass(ir_block, {});
    Optimization::NamingPass(ir_block);
    if (conf.HasOptimization(OptimizationFlag::GetSetElimination)) {
        Optimization::A32GetSetElimination(ir_block,
            {.convert_nzc_to_nz = true,
                .preserve_state_at_memory_access =
                    conf.check_halt_on_memory_access});
        Optimization::DeadCodeElimination(ir_block);
    }
    if (conf.HasOptimization(OptimizationFlag::ConstProp)) {
        Optimization::A32ConstantMemoryReads(ir_block, callbacks);
        Optimization::ConstantPropagation(ir_block);
        Optimization::DeadCodeElimination(ir_block);
    }
    Optimization::IdentityRemovalPass(ir_block);
    // Get/set elimination can insert new values after the first naming pass.
    Optimization::NamingPass(ir_block);
    Optimization::VerificationPass(ir_block);

    return ir_block;
}

}
