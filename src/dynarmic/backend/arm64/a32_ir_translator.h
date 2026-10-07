/* This file is part of the dynarmic project. SPDX-License-Identifier: 0BSD */
#pragma once
#include "dynarmic/interface/A32/config.h"
#include "dynarmic/ir/basic_block.h"
namespace Dynarmic::Backend::Arm64 {
IR::Block TranslateA32IR(const A32::UserConfig& conf, IR::LocationDescriptor descriptor);
}
