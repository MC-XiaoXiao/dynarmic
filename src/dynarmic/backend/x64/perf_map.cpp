/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/x64/perf_map.h"

#include <cstddef>
#include <string>

#ifdef __linux__

#    include <atomic>
#    include <cstdio>
#    include <cstdlib>
#    include <mutex>

#    include <fmt/format.h>
#    include <mcl/stdint.hpp>
#    include <sys/types.h>
#    include <unistd.h>

namespace Dynarmic::Backend::X64 {

namespace {
std::mutex mutex;
std::FILE* file = nullptr;
std::atomic<bool> file_open = false;

void OpenFile() {
    const char* perf_dir = std::getenv("PERF_BUILDID_DIR");
    if (!perf_dir) {
        file = nullptr;
        return;
    }

    const pid_t pid = getpid();
    const std::string filename = fmt::format("{:s}/perf-{:d}.map", perf_dir, pid);

    file = std::fopen(filename.c_str(), "w");
    if (!file) {
        return;
    }

    std::setvbuf(file, nullptr, _IONBF, 0);
    file_open.store(true, std::memory_order_relaxed);
}
}  // anonymous namespace

bool PerfMapEnabled() {
    // An already opened map remains active until Clear, even if the
    // environment changes. The file itself is only accessed under mutex.
    return file_open.load(std::memory_order_relaxed) || std::getenv("PERF_BUILDID_DIR") != nullptr;
}

namespace detail {
void PerfMapRegister(const void* start, const void* end, std::string_view friendly_name) {
    if (start == end) {
        // Nothing to register
        return;
    }

    std::lock_guard guard{mutex};

    if (!file) {
        OpenFile();
        if (!file) {
            return;
        }
    }

    const std::string line = fmt::format("{:016x} {:016x} {:s}\n", reinterpret_cast<u64>(start), reinterpret_cast<u64>(end) - reinterpret_cast<u64>(start), friendly_name);
    std::fwrite(line.data(), sizeof *line.data(), line.size(), file);
}
}  // namespace detail

void PerfMapClear() {
    std::lock_guard guard{mutex};

    if (!file) {
        return;
    }

    std::fclose(file);
    file = nullptr;
    file_open.store(false, std::memory_order_relaxed);
    OpenFile();
}

}  // namespace Dynarmic::Backend::X64

#else

namespace Dynarmic::Backend::X64 {

bool PerfMapEnabled() {
    return false;
}

namespace detail {
void PerfMapRegister(const void*, const void*, std::string_view) {}
}  // namespace detail

void PerfMapClear() {}

}  // namespace Dynarmic::Backend::X64

#endif
