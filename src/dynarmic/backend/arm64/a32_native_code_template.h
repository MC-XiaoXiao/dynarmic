/* This file is part of the dynarmic project. SPDX-License-Identifier: 0BSD */
#pragma once

#include <algorithm>
#include <memory>
#include <utility>
#include <tuple>
#include <vector>

#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/frontend/A32/a32_location_descriptor.h"
#include "dynarmic/interface/A32/config.h"
#include "dynarmic/interface/A32/native_code_template.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::Backend::Arm64 {

inline auto NativeTemplateConfiguration(const A32::UserConfig& conf) {
    const auto* read_table = conf.read_page_table ? conf.read_page_table : conf.page_table;
    const auto* read_link = conf.read_page_table_link ? conf.read_page_table_link :
        (conf.read_page_table == nullptr ? conf.page_table_link : nullptr);
    return std::tuple{conf.arch_version, conf.optimizations, conf.unsafe_optimizations,
        conf.define_unpredictable_behaviour, conf.hook_hint_instructions,
        conf.hook_isb, conf.check_halt_on_memory_access, conf.enable_cycle_counting,
        conf.always_little_endian, conf.wall_clock_cntpct, bool(conf.page_table),
        bool(conf.read_page_table),
        bool(conf.page_table_link), bool(read_link),
        read_table == conf.page_table && read_link == conf.page_table_link,
        conf.absolute_offset_page_table, conf.page_table_pointer_mask_bits,
        conf.detect_misaligned_access_via_page_table,
        conf.only_detect_misalignment_via_page_table_on_page_boundary,
        bool(conf.global_monitor)};
}

inline bool NativeTemplateRuntimeIsLinked(const A32::UserConfig& conf) {
    return conf.native_code_slab && conf.callbacks_link && conf.lookup_link &&
        conf.runtime_config_link && conf.fast_dispatch_table_link &&
        !conf.fastmem_pointer && !conf.very_verbose_debugging_output &&
        (!conf.page_table || conf.page_table_link) &&
        (!conf.read_page_table || conf.read_page_table_link) &&
        (!conf.global_monitor || (conf.exclusive_monitor_lock_link &&
            conf.exclusive_monitor_addresses_link && conf.exclusive_monitor_values_link));
}

class A32NativeCodeTemplate final : public A32::NativeCodeTemplate {
public:
    A32NativeCodeTemplate(const A32::UserConfig& conf, const IR::Block& block,
                         EmittedBlockInfo& emitted)
        : configuration{NativeTemplateConfiguration(conf)}, location{block.Location()},
          end_location{block.EndLocation()}, info{ShareRelocations(emitted)},
          words{reinterpret_cast<const u32*>(emitted.entry_point),
                reinterpret_cast<const u32*>(emitted.entry_point + emitted.size)} {
        // Every external branch/address is still an unlinked placeholder.
        // The template must not retain a pointer into the producer's mapping.
        info.entry_point = nullptr;
    }

    u64 LocationDescriptor() const noexcept override { return location.Value(); }
    size_t CodeSize() const noexcept override { return info.size; }
    size_t MemoryFootprint() const noexcept override {
        size_t bytes = sizeof(*this) + words.capacity() * sizeof(u32) +
            info.relocations.capacity() * sizeof(Relocation) +
            RelocationMapBytes(info.BlockRelocations()) +
            (info.shared_block_relocations ? sizeof(EmittedBlockInfo::BlockRelocationMap) : 0);
        for (const auto& [target, relocations] : info.BlockRelocations())
            bytes += relocations.capacity() * sizeof(BlockRelocation);
        return bytes;
    }
    bool Compatible(const A32::UserConfig& conf) const {
        return NativeTemplateRuntimeIsLinked(conf) &&
            configuration == NativeTemplateConfiguration(conf);
    }
    static bool Eligible(const A32::UserConfig& conf, const IR::Block& block,
                         const EmittedBlockInfo& emitted) {
        if (A32::LocationDescriptor{block.EndLocation()}.PC() <= A32::LocationDescriptor{block.Location()}.PC())
            return false;
        if (!conf.enable_native_code_templates || !NativeTemplateRuntimeIsLinked(conf) ||
            !emitted.fastmem_patch_info.empty() || emitted.size == 0 || emitted.size % sizeof(u32))
            return false;
        // Coprocessors and host-call IR can embed caller-specific function or
        // data pointers outside the existing relocation stream.
        return std::none_of(block.begin(), block.end(), [](const IR::Inst& inst) {
            return inst.IsCoprocessorInstruction() || inst.GetOpcode() == IR::Opcode::CallHostFunction;
        });
    }

private:
    static const EmittedBlockInfo& ShareRelocations(EmittedBlockInfo& emitted) {
        // Emission is complete. Share the immutable offsets with the producer,
        // templates and imports; each cache still patches its own native code.
        if (!emitted.block_relocations.empty())
            emitted.shared_block_relocations = std::make_shared<const EmittedBlockInfo::BlockRelocationMap>(std::move(emitted.block_relocations));
        return emitted;
    }

    template<class Map>
    static size_t RelocationMapBytes(const Map& map) noexcept {
        if constexpr (requires { map.capacity(); })
            return map.capacity() * sizeof(typename Map::value_type);
        else
            return map.bucket_count() * (sizeof(typename Map::value_type) + 2 * sizeof(size_t));
    }

public:
    const decltype(NativeTemplateConfiguration(std::declval<A32::UserConfig>())) configuration;
    const IR::LocationDescriptor location;
    const IR::LocationDescriptor end_location;
    EmittedBlockInfo info;
    const std::vector<u32> words;
};

}  // namespace Dynarmic::Backend::Arm64
