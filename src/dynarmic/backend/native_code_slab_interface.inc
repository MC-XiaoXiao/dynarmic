/* This file is part of the dynarmic project. SPDX-License-Identifier: 0BSD */
NativeCodeSlab::NativeCodeSlab() : impl{std::make_unique<Impl>()} {}

NativeCodeSlab::~NativeCodeSlab() = default;

void NativeCodeSlab::initialize(
    UserConfig conf, Jit* jit_interface, void* jit_state, const void* (*lookup)(void*), void* lookup_arg, bool shared_mode) {
    impl->initialize(std::move(conf), jit_interface, jit_state, lookup,
                     lookup_arg, shared_mode);
}

std::uint64_t NativeCodeSlab::generation() const {
    return impl->generation();
}

std::uint64_t NativeCodeSlab::generation_snapshot() const noexcept {
    return impl->generation_snapshot();
}

bool NativeCodeSlab::find_block(
    std::uint64_t location_descriptor, std::uint64_t expected_generation, BlockDescriptor& result) const {
    return impl->find_block(
        location_descriptor, expected_generation, result);
}

NativeCodeSlab::BlockDescriptor NativeCodeSlab::emit(
    IR::Block& block, std::uint64_t expected_generation) {
    return impl->emit(block, expected_generation);
}

NativeCodeSlab::BlockDescriptor NativeCodeSlab::emit(
    IR::Block& block, std::uint64_t expected_generation, const UserConfig& runtime_config) {
    return impl->emit(block, expected_generation, &runtime_config);
}

std::size_t NativeCodeSlab::space_remaining() const {
    return impl->space_remaining();
}

void NativeCodeSlab::ensure_memory_committed(std::size_t codesize) {
    impl->ensure_memory_committed(codesize);
}

void NativeCodeSlab::register_executor(void* storage, void* jit_state) {
    impl->register_executor(storage, jit_state);
}

void NativeCodeSlab::unregister_executor(void* storage, void* jit_state) {
    impl->unregister_executor(storage, jit_state);
}

void NativeCodeSlab::clear_cache() {
    impl->clear_cache();
}

void NativeCodeSlab::recycle_cache() {
    impl->recycle_cache();
}

void NativeCodeSlab::invalidate_cache_range(
    std::uint32_t start_address, std::size_t length) {
    impl->invalidate_cache_range(start_address, length);
}

void NativeCodeSlab::request_cache_clear() {
    impl->request_cache_clear();
}

void NativeCodeSlab::request_cache_range(
    std::uint32_t start_address, std::size_t length) {
    impl->request_cache_range(start_address, length);
}

void NativeCodeSlab::service_pending_invalidation() {
    impl->service_pending_invalidation();
}

HaltReason NativeCodeSlab::run_code(void* jit_state,
                                    const void* code_ptr) const {
    return impl->run_code(jit_state, code_ptr);
}

HaltReason NativeCodeSlab::step_code(void* jit_state,
                                     const void* code_ptr) const {
    return impl->step_code(jit_state, code_ptr);
}

const void* NativeCodeSlab::return_from_run_code() const {
    return impl->return_from_run_code();
}

std::size_t NativeCodeSlab::code_cache_used() const {
    return impl->code_cache_used();
}

NativeCodeSlab::CacheStats NativeCodeSlab::GetCacheStats() const {
    return impl->GetCacheStats();
}

void NativeCodeSlab::dump_disassembly() const {
    impl->dump_disassembly();
}

std::vector<std::string> NativeCodeSlab::disassemble() const {
    return impl->disassemble();
}

bool NativeCodeSlab::has_host_feature_sha() const {
    return impl->has_host_feature_sha();
}

std::uint64_t NativeCodeSlab::enter_execution(void* jit_state) {
    return impl->enter_execution(jit_state);
}

void NativeCodeSlab::leave_execution(
    void* jit_state, std::uint64_t generation) {
    impl->leave_execution(jit_state, generation);
}
