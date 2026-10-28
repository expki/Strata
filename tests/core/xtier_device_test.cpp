// The AMD tier on a real card: memory, host memory mapped by both makes and flags across them (exit 77 without a library or a card).
#include "strata/core/xtier.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "xtier_skip.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

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

// A staging ring pinned in pieces: both makes reach every piece through one base address, and the card's own calls never wait for its pinning
static void quiet_ring(Device& a) {
    std::string e;
    const size_t bytes = 96u << 20, piece = 32u << 20;
    SharedHost q;
    CHECK(q.alloc_quiet(a, bytes, piece, true, e) && q.variant() == 'Q');
    if (q.host() == nullptr) return;
    CHECK(q.cuda_dev() != nullptr && q.hip_dev() != nullptr);
    // the card writes the whole range through the first piece's address, the host reads every piece
    uint64_t wrong = 1;
    float ms = 0;
    CHECK(a.probe(0, q.host(), bytes, 7, wrong, ms, e));
    bool seen = true;
    for (size_t i = 0; i < bytes / 4; i += 4099) seen = seen && read_word(q.host(), i * 4) == STRATA_XTIER_PATTERN(7, i);
    CHECK(seen);
    // the card copies a whole piece out, as it copies a staged expert out of its slot
    CHECK(a.probe(3, (char*) q.host() + piece, piece, 0, wrong, ms, e) && ms > 0);
    // this make reads the last piece through the first piece's address too
    void* d = nullptr;
    uint32_t back[1024] = {};
    const size_t off = bytes - sizeof back;
    CHECK(cudaMalloc(&d, sizeof back) == cudaSuccess && cudaMemcpy(d, (char*) q.cuda_dev() + off, sizeof back, cudaMemcpyDefault) == cudaSuccess &&
          cudaMemcpy(back, d, sizeof back, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(std::memcmp(back, (char*) q.host() + off, sizeof back) == 0);
    if (d != nullptr) cudaFree(d);
    q.release();
    CHECK(q.host() == nullptr && q.variant() == 0);

    // the AMD card's alone, as the stager's ring: no CUDA address, and the card still reaches every piece through the first one's
    CHECK(q.alloc_quiet(a, bytes, piece, false, e) && q.variant() == 'Q');
    if (q.host() == nullptr) return;
    CHECK(q.cuda_dev() == nullptr && q.hip_dev() != nullptr);
    CHECK(a.probe(0, q.host(), bytes, 9, wrong, ms, e));
    seen = true;
    for (size_t i = 0; i < bytes / 4; i += 4099) seen = seen && read_word(q.host(), i * 4) == STRATA_XTIER_PATTERN(9, i);
    CHECK(seen);
    CHECK(a.probe(3, (char*) q.host() + 2 * piece, piece, 0, wrong, ms, e) && ms > 0);
    q.release();
    CHECK(q.host() == nullptr && q.variant() == 0);

    {   // a card that stopped answering is asked to map no piece
        Device b;
        CHECK(b.open(0, e));
        b.mark_hung();
        e.clear();
        CHECK(!q.alloc_quiet(b, bytes, piece, false, e) && e.find("stopped answering") != std::string::npos);
        CHECK(q.host() == nullptr);
    }

    // one large mapping on a thread of its own: the card's other calls (here mem_info) do not wait for it
    const size_t big = 1024u << 20;
    char* host = (char*) map_pages(big);
    CHECK(host != nullptr);
    if (host == nullptr) return;
    std::memset(host, 0, big);
    std::atomic<bool> done{false};
    double reg_ms = 0;
    std::thread th([&] {
        const auto t0 = std::chrono::steady_clock::now();
        void* dev = nullptr;
        std::string e2;
        (void) a.register_host(host, big, &dev, e2);
        reg_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        done = true;
    });
    double worst = 0;
    int calls = 0;
    while (!done) {
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t f = 0, t = 0;
        (void) a.mem_info(f, t, e);
        worst = std::max(worst, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        ++calls;
    }
    th.join();
    std::printf("  a 1 GiB mapping took %.0f ms, %d mem_info calls beside it, the slowest %.1f ms\n", reg_ms, calls, worst);
    CHECK(worst < 20.0);
    a.unregister_host(host);
    unmap_pages(host, big);
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

    std::printf("prompt records\n");
    {
        std::vector<strata_xtier_prefill_record> recs(2);
        CHECK(a.prefill_stats(recs, e) && recs.empty());   // no prompt layer queued yet
        const Api* api = load(e);
        char why[256] = {};
        void* h = api != nullptr ? api->open(0, why, sizeof why) : nullptr;
        CHECK(h != nullptr);
        if (h != nullptr) {   // the library itself refuses a place it cannot write
            strata_xtier_prefill_record r[2] = {};
            int64_t n = -1;
            CHECK(api->prefill_stats(h, nullptr, 2, &n) != 0 && n == -1 && std::strlen(api->last_error(h)) > 0);
            CHECK(api->prefill_stats(h, r, 2, nullptr) != 0);
            CHECK(api->prefill_stats(h, r, 0, &n) != 0 && n == -1);
            CHECK(api->prefill_stats(h, r, 2, &n) == 0 && n == 0);
            api->close(h);
        }
    }

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

    std::printf("a staging ring pinned in pieces\n");
    quiet_ring(a);

    a.close();
    CHECK(!a.valid());
    std::printf("%s\n", g_fail == 0 ? "xtier_device_test: ok" : "xtier_device_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
