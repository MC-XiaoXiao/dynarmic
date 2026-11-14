// SPDX-License-Identifier: BSD-3-Clause
// Standalone label storage regression; optional robin tables are selected by CPPFLAGS.
#include <xbyak/xbyak.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <vector>
static thread_local bool count_allocations;
static thread_local std::size_t allocation_calls,fail_at;
void* operator new(std::size_t n) {
    if(count_allocations && ++allocation_calls==fail_at) throw std::bad_alloc{};
    if(void* p=std::malloc(n?n:1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void operator delete[](void* p,std::size_t) noexcept { std::free(p); }
static void need(bool v,const char* why) { if(!v) { std::fprintf(stderr,"FAIL %s\n",why); std::abort(); } }
using namespace Xbyak;
static void append(std::ofstream& out,CodeGenerator& code) {
    code.ready();std::uint64_t size=code.getSize();out.write(reinterpret_cast<char*>(&size),sizeof(size));out.write(reinterpret_cast<const char*>(code.getCode()),size);
}
static void mixed(CodeGenerator& code) {
    std::array<Label,128> labels;
    for(unsigned i=0;i<labels.size();++i) {
        code.jz(labels[i],CodeGenerator::T_NEAR);code.jmp(labels[i],CodeGenerator::T_NEAR);code.call(labels[i]);code.lea(code.rax,code.ptr[code.rip+labels[i]]);
    }
    for(unsigned i=0;i<labels.size();i+=2) { code.L(labels[i]);code.nop(); }
    // Reuse freed slots while half the original labels remain unresolved.
    for(unsigned i=0;i<256;++i) { Label local;code.jmp(local,CodeGenerator::T_SHORT);code.nop();code.L(local);code.jmp(local,CodeGenerator::T_SHORT);Label copy=local;need(copy.getAddress()==local.getAddress(),"copied defined label address"); }
    for(unsigned i=1;i<labels.size();i+=2) { code.L(labels[i]);code.nop(); }
    need(!code.hasUndefinedLabel(),"all interleaved references patched");
}
int main(int argc,char** argv) {
    need(argc==2,"output byte corpus path");std::ofstream out(argv[1],std::ios::binary);
    {
        CodeGenerator code(1<<20);mixed(code);append(out,code);
        code.reset();mixed(code);append(out,code);
    }
    {
        CodeGenerator code(128,AutoGrow);Label end;
        for(unsigned i=0;i<4096;++i) { code.jnz(end,CodeGenerator::T_NEAR);code.lea(code.rax,code.ptr[code.rip+end]); }
        code.L(end);code.ret();append(out,code);
    }
    {
        CodeGenerator code(1<<20);Label anchor;code.L(anchor);
        for(unsigned i=0;i<1000;++i) {code.inLocalLabel();code.jmp(".done",CodeGenerator::T_NEAR);code.nop();code.L(".done");code.outLocalLabel();code.jmp(anchor,CodeGenerator::T_NEAR);}
        code.jmp("@f");code.nop();code.L("@@");code.jmp("@b");append(out,code);
    }
    {
        CodeGenerator code;Label first;code.jmp(first,CodeGenerator::T_NEAR);Label alias=first;code.jmp(alias,CodeGenerator::T_NEAR);code.L(alias);code.ret();append(out,code);
        code.reset();Label destination,source;code.jmp(destination,CodeGenerator::T_NEAR);code.nop();code.L(source);code.nop();code.assignL(destination,source);code.ret();append(out,code);
    }
    for(bool grow:{false,true}) {
        CodeGenerator code(grow?128:1<<20,grow?AutoGrow:0);Label end;std::vector<std::size_t> positions;
        for(unsigned i=0;i<2048;++i) {positions.push_back(code.getSize());code.putL(end);code.nop();}
        const auto target=code.getSize();code.L(end);code.ready();std::vector<std::uint8_t> bytes(code.getCode(),code.getCode()+code.getSize());
        for(auto position:positions) {std::size_t absolute;std::memcpy(&absolute,bytes.data()+position,sizeof(absolute));need(absolute==std::size_t(code.getCode())+target,"absolute forward relocations survive growth");absolute-=std::size_t(code.getCode());std::memcpy(bytes.data()+position,&absolute,sizeof(absolute));}
        std::uint64_t size=bytes.size();out.write(reinterpret_cast<char*>(&size),sizeof(size));out.write(reinterpret_cast<char*>(bytes.data()),size);
    }
    {
        CodeGenerator code;Label done;code.mov(code.eax,7);code.jmp(done);code.mov(code.eax,3);code.L(done);code.ret();code.ready();need(code.getCode<int(*)()>()()==7,"generated machine code executes");
        bool far=false;try {code.reset();Label target;code.jmp(target,CodeGenerator::T_SHORT);for(unsigned i=0;i<256;++i)code.nop();code.L(target);} catch(const Xbyak::Error& e) {far=true;std::printf("short-error=%d\n",int(e));}need(far,"out-of-range short reference still errors");
        code.reset();Label missing;code.jmp(missing,CodeGenerator::T_NEAR);need(code.hasUndefinedLabel(),"undefined reference retained");code.reset();need(!code.hasUndefinedLabel(),"reset clears undefined references");
    }
    unsigned failures=0;
    for(unsigned reuse=0;reuse<2;++reuse) for(unsigned injection=1;injection<=96;++injection) {
        CodeGenerator code(1<<20);Label keep;std::array<Label,128> labels;
        code.jmp(keep,CodeGenerator::T_NEAR);
        if(reuse) {Label warm;for(unsigned i=0;i<256;++i)code.jmp(warm,CodeGenerator::T_NEAR);code.L(warm);}
        allocation_calls=0;fail_at=injection;count_allocations=true;
        try {for(unsigned i=0;i<labels.size();++i) {code.jmp(labels[i],CodeGenerator::T_NEAR);code.call(labels[i]);}} catch(const std::bad_alloc&) {++failures;}
        count_allocations=false;fail_at=0;
        const auto target=code.getSize();code.L(keep);std::int32_t displacement;std::memcpy(&displacement,code.getCode()+1,4);need(displacement==std::int32_t(target-5),"failed insertion retains earlier pending patch");
        for(auto& label:labels)if(label.getId())code.L(label);
        need(!code.hasUndefinedLabel(),"all retained pending references resolve after failure");
        code.reset();code.mov(code.eax,11);code.ret();code.ready();need(code.getCode<int(*)()>()()==11,"post-failure reset remains executable");
    }
    std::printf("allocation-failure-cases=%u\n",failures);
    for(unsigned run=0;run<5;++run) {
        CodeGenerator code(1<<20);allocation_calls=0;fail_at=0;count_allocations=true;auto begin=std::chrono::steady_clock::now();
        for(unsigned i=0;i<20000;++i) {Label label;code.jnz(label,CodeGenerator::T_NEAR);code.jmp(label,CodeGenerator::T_NEAR);code.call(label);code.lea(code.rax,code.ptr[code.rip+label]);code.nop();code.L(label);}
        const auto end=std::chrono::steady_clock::now();count_allocations=false;need(!code.hasUndefinedLabel(),"benchmark leaves no pending sites");
        std::printf("benchmark run=%u labels=20000 allocations=%zu ns=%lld\n",run,allocation_calls,(long long)std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
        if(run==0)append(out,code);
    }
    out.close();std::ifstream corpus(argv[1],std::ios::binary);std::uint64_t fingerprint=14695981039346656037ULL;char byte;
    while(corpus.get(byte))fingerprint=(fingerprint^static_cast<unsigned char>(byte))*1099511628211ULL;
    need(fingerprint==0x1577348e3fd43bdbULL,"golden encoded byte corpus (absolute addresses normalized)");
    std::puts("PASS forward references, shared targets, interleaved reuse, RIP-relative, AutoGrow, copies, strings, short errors, reset and allocation failures");
}
