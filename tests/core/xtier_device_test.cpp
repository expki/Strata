// The AMD tier on a real card: memory, host memory mapped by both makes and flags across them (exit 77 without a library or a card).
#include "strata/core/xtier.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "xtier_skip.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <sys/mman.h>
#endif

using namespace strata::core::xtier;
using strata::kernels::raise_flag;
using strata::kernels::wait_handoff;

static int g_fail = 0;
#define CHECK(c)                                                                                  \
    do {                                                                                          \
        const bool ok_ = (c);                                                                     \
        std::printf("  %-96s %s\n", #c, ok_ ? "ok" : "FAIL");                                     \
        if (!ok_) ++g_fail;                                                                       \
    } while (0)

static void* page_alloc(size_t bytes) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, 4096);
#else
    return std::aligned_alloc(4096, bytes);
#endif
}

static void page_free(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// Zeroed pages the kernel hands out by mmap, the way the engine's expert arenas come.
static void* map_pages(size_t bytes) {
#if defined(_WIN32)
    void* p = _aligned_malloc(bytes, 4096);
    if (p != nullptr) std::memset(p, 0, bytes);
    return p;
#else
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

static void unmap_pages(void* p, size_t bytes) {
#if defined(_WIN32)
    (void) bytes;
    _aligned_free(p);
#else
    munmap(p, bytes);
#endif
}

static uint32_t read_word(const void* base, size_t off) {
    return *(const volatile uint32_t*) ((const char*) base + off);
}

// One page of flags, a cache line apart: in, out, the library's error word, the engine's, then the same for a flag that is never raised.
enum : size_t { kIn = 0, kOut = 64, kHerr = 128, kCerr = 192, kIn2 = 256, kOut2 = 320, kHerr2 = 384 };

static void spin_us(int us) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(us);
    while (std::chrono::steady_clock::now() < end) {}
}

static void echo_across_makes(Device& a) {
    std::string e;
    const size_t bytes = 4096;
    char* page = (char*) map_pages(bytes);
    CHECK(page != nullptr);
    if (page == nullptr) return;

    CHECK(cudaHostRegister(page, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess);
    void* cdev = nullptr;
    CHECK(cudaHostGetDevicePointer(&cdev, page, 0) == cudaSuccess && cdev != nullptr);
    void* hdev = nullptr;
    CHECK(a.register_host(page, bytes, &hdev, e) && hdev != nullptr);
    cudaStream_t cs = nullptr;
    CHECK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) == cudaSuccess);
    if (cdev == nullptr || hdev == nullptr || cs == nullptr) return;

    uint32_t* const c = (uint32_t*) cdev;
    uint32_t* const h = (uint32_t*) hdev;
    const auto at = [](uint32_t* base, size_t off) { return (uint32_t*) ((char*) base + off); };

    // 1000 round trips: this card raises in, the other make's kernel echoes it to out, this card waits for it
    bool queued = true;
    for (uint32_t i = 0; i < 1000 && queued; ++i) {
        queued = a.flag_echo(at(h, kIn), at(h, kOut), at(h, kHerr), i, 2000000000ull, e);
        if (i % 100 == 99) spin_us(200);   // the AMD wait kernel is already spinning when the flag comes
        raise_flag(at(c, kIn), i, cs);
        wait_handoff(at(c, kOut), i, at(c, kCerr), 2000000000ull, cs);
        queued = queued && cudaStreamSynchronize(cs) == cudaSuccess && a.sync(e);
    }
    CHECK(queued);
    CHECK(read_word(page, kOut) == 999);
    CHECK(read_word(page, kHerr) == 0);
    CHECK(read_word(page, kCerr) == 0);

    // a flag nobody raises: the library's wait gives up at its timeout and says so in its error word
    CHECK(a.flag_echo(at(h, kIn2), at(h, kOut2), at(h, kHerr2), 1, 100000000ull, e));
    CHECK(read_word(page, kHerr2) == 0);   // still waiting, the timeout has not elapsed
    const auto t0 = std::chrono::steady_clock::now();
    while (read_word(page, kHerr2) == 0 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(1)) {}
    CHECK(read_word(page, kHerr2) == 1);
    CHECK(a.sync(e));
    CHECK(read_word(page, kOut2) == 1);   // the queued work still finishes, so nothing is left hanging

    a.unregister_host(page);
    CHECK(cudaStreamDestroy(cs) == cudaSuccess);
    CHECK(cudaHostUnregister(page) == cudaSuccess);
    unmap_pages(page, bytes);
}

int main() {
    std::string e;
    if (load(e) == nullptr) {
        std::printf("%s: %s\n", xtier_is_skip(e) ? "SKIP" : "FAIL", e.c_str());
        return xtier_is_skip(e) ? 77 : 1;
    }

    Device a;
    if (!a.open(0, e)) {
        std::printf("%s: open(0): %s\n", xtier_is_skip(e) ? "SKIP" : "FAIL", e.c_str());
        return xtier_is_skip(e) ? 77 : 1;
    }
    CHECK(a.valid());

    uint64_t f = 0, t = 0;
    CHECK(a.mem_info(f, t, e));
    std::printf("  AMD card 0: free %.2f GiB of %.2f GiB\n", (double) f / (1ull << 30), (double) t / (1ull << 30));
    CHECK(f > 0 && f <= t);

    {
        Device b;
        CHECK(b.open(0, e) && b.valid());
        uint64_t f2 = 0, t2 = 0;
        CHECK(b.mem_info(f2, t2, e) && t2 == t);
        b.close();
        CHECK(!b.valid());
    }
    CHECK(a.valid() && a.mem_info(f, t, e) && f > 0 && f <= t);

    const size_t bytes = 1u << 20;
    void* host = page_alloc(bytes);
    CHECK(host != nullptr);
    if (host != nullptr) {
        void* dev = nullptr;
        CHECK(a.register_host(host, bytes, &dev, e) && dev != nullptr);
        a.unregister_host(host);
        dev = nullptr;
        CHECK(a.register_host(host, bytes, &dev, e) && dev != nullptr);   // it can be mapped again once released
        a.unregister_host(host);
        CHECK(!a.register_host(nullptr, bytes, &dev, e) && !e.empty());
        page_free(host);
    }

    std::printf("flags across the makes\n");
    echo_across_makes(a);

    a.close();
    CHECK(!a.valid());
    std::printf("%s\n", g_fail == 0 ? "xtier_device_test: ok" : "xtier_device_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
