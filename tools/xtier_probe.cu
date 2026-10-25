// Phase 0 of the AMD expert tier: a CUDA card and a HIP library in one process, pinned memory both map, and flags between them.
#include "../src/kernels/test_blobs.hpp"
#include "strata/core/xtier.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace strata::core::xtier;

namespace {

constexpr size_t kBytes = 64u << 20;   // one matrix cell moves 64 MiB
constexpr int kEchoes = 10000;
constexpr uint64_t kTimeoutNs = 2000000000ull;
constexpr int kSpinUs = 100;           // long enough for the library's wait kernel to be up and spinning

// The flag slots of the one page the latency sections use, a cache line apart.
enum : size_t { IN = 0, OUT = 64, HERR = 128, CERR = 192 };

__global__ void write_kernel(volatile uint32_t* p, uint64_t words, uint32_t seed) {
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (uint64_t) gridDim.x * blockDim.x)
        p[i] = STRATA_XTIER_PATTERN(seed, i);
}

__global__ void check_kernel(const volatile uint32_t* p, uint64_t words, uint32_t seed, unsigned long long* bad) {
    unsigned long long mine = 0;
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (uint64_t) gridDim.x * blockDim.x)
        mine += p[i] != STRATA_XTIER_PATTERN(seed, i);
    if (mine != 0) atomicAdd(bad, mine);
}

cudaStream_t g_cs = nullptr;
void* g_dbuf = nullptr;                    // device memory the CUDA card copies from
unsigned long long* g_bad = nullptr;       // the CUDA check's count of wrong words

std::string cuda_text(const char* what, cudaError_t e) { return std::string(what) + ": " + cudaGetErrorString(e); }

// One pinned region and what each card sees of it.
struct Region {
    char kind = 0;
    void* host = nullptr;
    void* cdev = nullptr;
    void* hdev = nullptr;
    size_t bytes = 0;
    bool cuda_reg = false;
    bool hip_reg = false;
};

const char* kind_text(char k) {
    return k == 'A' ? "cudaHostAlloc, then register_host"
         : k == 'B' ? "mmap, then cudaHostRegister and register_host"
                    : "host_alloc (hipHostMalloc), then cudaHostRegister";
}

bool region_make(char kind, Device& a, size_t bytes, Region& r, std::string& why) {
    r = Region{};
    r.kind = kind;
    r.bytes = bytes;
    std::string e;
    cudaError_t c = cudaSuccess;
    if (kind == 'A') {
        if ((c = cudaHostAlloc(&r.host, bytes, cudaHostAllocPortable | cudaHostAllocMapped)) != cudaSuccess) {
            r.host = nullptr;
            why = cuda_text("cudaHostAlloc", c);
            return false;
        }
    } else if (kind == 'B') {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            why = "mmap failed";
            return false;
        }
        r.host = p;
    } else {
        r.host = a.host_alloc(bytes, e);
        if (r.host == nullptr) {
            why = "host_alloc: " + e;
            return false;
        }
    }
    if (kind != 'A') {
        if ((c = cudaHostRegister(r.host, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped)) != cudaSuccess) {
            why = cuda_text("cudaHostRegister", c);
            return false;
        }
        r.cuda_reg = true;
    }
    if ((c = cudaHostGetDevicePointer(&r.cdev, r.host, 0)) != cudaSuccess) {
        why = cuda_text("cudaHostGetDevicePointer", c);
        return false;
    }
    if (kind == 'C') {
        r.hdev = r.host;   // the HIP runtime made it, so the card's pointer is the host pointer
    } else {
        if (!a.register_host(r.host, bytes, &r.hdev, e)) {
            why = "register_host: " + e;
            return false;
        }
        r.hip_reg = true;
    }
    return true;
}

void region_free(Region& r, Device& a) {
    if (r.hip_reg) a.unregister_host(r.host);
    if (r.cuda_reg) (void) cudaHostUnregister(r.host);
    if (r.host != nullptr) {
        if (r.kind == 'A') (void) cudaFreeHost(r.host);
        else if (r.kind == 'B') munmap(r.host, r.bytes);
        else a.host_free(r.host);
    }
    r = Region{};
}

// The CUDA card writes the seed's pattern into a region, or counts the words of it that differ from the pattern.
bool cuda_pattern(bool write, const Region& r, uint32_t seed, uint64_t& wrong, std::string& why) {
    const uint64_t words = r.bytes / 4;
    const unsigned blocks = (unsigned) std::min<uint64_t>((words + 255) / 256, 2048);
    cudaError_t c = cudaSuccess;
    if (write) {
        write_kernel<<<blocks, 256, 0, g_cs>>>((volatile uint32_t*) r.cdev, words, seed);
    } else {
        if ((c = cudaMemsetAsync(g_bad, 0, sizeof(*g_bad), g_cs)) != cudaSuccess) {
            why = cuda_text("cudaMemsetAsync", c);
            return false;
        }
        check_kernel<<<blocks, 256, 0, g_cs>>>((const volatile uint32_t*) r.cdev, words, seed, g_bad);
    }
    if ((c = cudaGetLastError()) != cudaSuccess) {
        why = cuda_text("kernel", c);
        return false;
    }
    unsigned long long bad = 0;
    if (!write && (c = cudaMemcpyAsync(&bad, g_bad, sizeof(bad), cudaMemcpyDeviceToHost, g_cs)) != cudaSuccess) {
        why = cuda_text("cudaMemcpyAsync", c);
        return false;
    }
    if ((c = cudaStreamSynchronize(g_cs)) != cudaSuccess) {
        why = cuda_text("cudaStreamSynchronize", c);
        return false;
    }
    wrong = bad;
    return true;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const size_t k = std::min(v.size() - 1, (size_t) std::ceil(p * (double) v.size()) - 1);
    return v[k];
}

// The time in ms of a copy from device memory into the region on the CUDA card: the median of 7 after a warm up.
bool cuda_d2h_ms(const Region& r, double& ms, std::string& why) {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    (void) cudaEventCreate(&e0);
    (void) cudaEventCreate(&e1);
    std::vector<double> t;
    cudaError_t c = cudaSuccess;
    for (int i = 0; i < 8 && c == cudaSuccess; ++i) {
        float f = 0.f;
        (void) cudaEventRecord(e0, g_cs);
        c = cudaMemcpyAsync(r.host, g_dbuf, r.bytes, cudaMemcpyDeviceToHost, g_cs);
        (void) cudaEventRecord(e1, g_cs);
        if (c == cudaSuccess) c = cudaEventSynchronize(e1);
        if (c == cudaSuccess) c = cudaEventElapsedTime(&f, e0, e1);
        if (i > 0) t.push_back(f);
    }
    (void) cudaEventDestroy(e0);
    (void) cudaEventDestroy(e1);
    if (c != cudaSuccess) {
        why = cuda_text("copy", c);
        return false;
    }
    ms = median(t);
    return true;
}

// The same on the AMD card, through the library.
bool hip_d2h_ms(Device& a, const Region& r, double& ms, std::string& why) {
    std::vector<double> t;
    for (int i = 0; i < 8; ++i) {
        uint64_t unused = 0;
        float f = 0.f;
        if (!a.probe(2, r.host, r.bytes, 0, unused, f, why)) return false;
        if (i > 0) t.push_back(f);
    }
    ms = median(t);
    return true;
}

std::string gbs(double ms) {
    char b[48];
    std::snprintf(b, sizeof(b), "%6.2f ms %5.1f GB/s", ms, (double) kBytes / (ms * 1e-3) / 1e9);
    return b;
}

// One row of the registration matrix, true when the variant works both ways.
bool matrix_row(char kind, Device& a) {
    Region r;
    std::string why, note;
    std::string reg = "ok", c2h = "-", h2c = "-", dc = "-", dh = "-";
    bool works = false;
    if (!region_make(kind, a, kBytes, r, why)) {
        reg = "FAIL";
        note = why;
    } else {
        uint64_t wrong = 0;
        float unused = 0.f;
        double ms = 0;
        std::string e;
        // each direction has its own seed, so a write that never arrived cannot match the old pattern
        const uint32_t s1 = 0x1111u + (uint32_t) kind, s2 = 0x2222u + (uint32_t) kind;
        c2h = "FAIL";
        if (cuda_pattern(true, r, s1, wrong, e) && a.probe(1, r.host, r.bytes, s1, wrong, unused, e)) {
            c2h = wrong == 0 ? "ok" : "FAIL";
            if (wrong != 0) note += "cuda->hip: " + std::to_string(wrong) + " words differ. ";
        } else {
            note += "cuda->hip: " + e + ". ";
        }
        h2c = "FAIL";
        if (a.probe(0, r.host, r.bytes, s2, wrong, unused, e) && cuda_pattern(false, r, s2, wrong, e)) {
            h2c = wrong == 0 ? "ok" : "FAIL";
            if (wrong != 0) note += "hip->cuda: " + std::to_string(wrong) + " words differ. ";
        } else {
            note += "hip->cuda: " + e + ". ";
        }
        dc = "FAIL";
        if (cuda_d2h_ms(r, ms, e)) dc = gbs(ms);
        else note += "cuda copy: " + e + ". ";
        dh = "FAIL";
        if (hip_d2h_ms(a, r, ms, e)) dh = gbs(ms);
        else note += "hip copy: " + e + ". ";
        works = c2h == "ok" && h2c == "ok";
    }
    std::printf("  %c  %-4s %-9s %-9s %-22s %-22s %s\n", kind, reg.c_str(), c2h.c_str(), h2c.c_str(), dc.c_str(), dh.c_str(),
                kind_text(kind));
    if (!note.empty()) std::printf("     %s\n", note.c_str());
    region_free(r, a);
    return works;
}

uint32_t* at_h(const Region& r, size_t off) { return (uint32_t*) ((char*) r.hdev + off); }
uint32_t* at_c(const Region& r, size_t off) { return (uint32_t*) ((char*) r.cdev + off); }
volatile uint32_t* at_w(const Region& r, size_t off) { return (volatile uint32_t*) ((char*) r.host + off); }

void spin_us(int us) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(us);
    while (std::chrono::steady_clock::now() < end) {}
}

void print_row(const char* name, const std::vector<double>& us, bool ok) {
    std::printf("  %-46s median %7.1f us   p99 %7.1f us   min %7.1f us   max %8.1f us   %s\n", name, median(us),
                percentile(us, 0.99), *std::min_element(us.begin(), us.end()), *std::max_element(us.begin(), us.end()),
                ok ? "ok" : "FAIL");
}

// 10000 echoes with the GPU waiting on both sides: this card raises in, the AMD kernel echoes it to out, this card waits for it.
bool flag_round_trip(Device& a, const Region& r) {
    std::vector<double> us;
    std::string e;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    (void) cudaEventCreate(&e0);
    (void) cudaEventCreate(&e1);
    bool ok = true;
    for (int i = 0; i < kEchoes && ok; ++i) {
        const uint32_t seq = (uint32_t) i + 1;
        float f = 0.f;
        ok = a.flag_echo(at_h(r, IN), at_h(r, OUT), at_h(r, HERR), seq, kTimeoutNs, e);
        spin_us(kSpinUs);
        (void) cudaEventRecord(e0, g_cs);
        strata::kernels::raise_flag(at_c(r, IN), seq, g_cs);
        strata::kernels::wait_handoff(at_c(r, OUT), seq, at_c(r, CERR), kTimeoutNs, g_cs);
        (void) cudaEventRecord(e1, g_cs);
        ok = ok && cudaStreamSynchronize(g_cs) == cudaSuccess && a.sync(e) && cudaEventElapsedTime(&f, e0, e1) == cudaSuccess;
        ok = ok && *at_w(r, HERR) == 0 && *at_w(r, CERR) == 0 && *at_w(r, OUT) == seq;
        us.push_back((double) f * 1000.0);
    }
    (void) cudaEventDestroy(e0);
    (void) cudaEventDestroy(e1);
    if (!e.empty()) std::printf("  %s\n", e.c_str());
    print_row("flag round trip (GPU events, NVIDIA side)", us, ok);
    return ok;
}

// 10000 cycles of the decode path's style: the host writes a flag the AMD kernel waits for, syncs the library, reads the echo.
bool host_round_trip(Device& a, const Region& r, bool spinning) {
    std::vector<double> us;
    std::string e;
    *at_w(r, IN) = 0;
    *at_w(r, OUT) = 0;
    *at_w(r, HERR) = 0;
    bool ok = true;
    for (int i = 0; i < kEchoes && ok; ++i) {
        const uint32_t seq = (uint32_t) i + 1;
        auto t0 = std::chrono::steady_clock::now();
        ok = a.flag_echo(at_h(r, IN), at_h(r, OUT), at_h(r, HERR), seq, kTimeoutNs, e);
        if (spinning) {
            spin_us(kSpinUs);
            t0 = std::chrono::steady_clock::now();
        }
        __atomic_store_n(at_w(r, IN), seq, __ATOMIC_RELEASE);
        ok = ok && a.sync(e);
        const uint32_t got = __atomic_load_n(at_w(r, OUT), __ATOMIC_ACQUIRE);
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        ok = ok && got == seq && *at_w(r, HERR) == 0;
    }
    if (!e.empty()) std::printf("  %s\n", e.c_str());
    print_row(spinning ? "host round trip, wait kernel already spinning" : "host round trip (enqueue, flag, sync, read)", us, ok);
    return ok;
}

// The decode path end to end on one IQ4_XS layer, host to host: decode, then decode_wait, for plans of 10, 40 and 80 entries.
// Each runs back to back, then 0.4 ms apart with the card idle, then 0.4 ms apart with a wait kernel spinning on the card (variant use's flags).
bool decode_timing(Device& a, char use) {
    constexpr int kSlots = 256, kBlobs = 16, kPlans = 64, kCalls = 1000, kWarm = 50, kTop = 10, kGapUs = 400;
    constexpr int64_t kEmbd = 2560, kFf = 640;
    const auto L = strata::kernels::native_expert_layout(23, 20, kEmbd, kFf);   // IQ4_XS gate/up, IQ4_NL down
    const strata_xtier_fmt f = {23, 20, kEmbd, kFf, L.up_off, L.down_off, L.bytes};
    std::mt19937 rng(2711);
    std::vector<std::vector<uint8_t>> blobs;
    for (int i = 0; i < kBlobs; ++i) {
        auto blob = random_rows(23, 2 * kFf, kEmbd, rng);
        const auto down = random_rows(20, kEmbd, kFf, rng);
        blob.insert(blob.end(), down.begin(), down.end());
        blobs.push_back(std::move(blob));
    }
    // 256 experts so the plans do not stay in the card's cache, 16 different blobs repeated over them
    std::string e;
    std::vector<uint64_t> sizes((size_t) kSlots, L.bytes), slot_addr;
    if (!a.arena(sizes, slot_addr, e)) {
        std::printf("  the arena failed: %s\n", e.c_str());
        return false;
    }
    for (int i = 0; i < kSlots; ++i) {
        const auto& b = blobs[(size_t) (i % kBlobs)];
        if (!a.fill(i, b.data(), b.size(), e)) {
            std::printf("  fill failed: %s\n", e.c_str());
            return false;
        }
    }
    if (!a.refill_record(e) || a.refill_done(true, &e) != 1) {
        std::printf("  the fills failed: %s\n", e.c_str());
        return false;
    }
    const std::vector<unsigned long long> addr(slot_addr.begin(), slot_addr.end());
    std::normal_distribution<float> nd(0.f, 1.f);
    Region r;
    std::string why;
    const bool flags = use != 0 && region_make(use, a, 4096, r, why);
    uint32_t seq = 0;
    if (flags) *at_w(r, IN) = *at_w(r, OUT) = *at_w(r, HERR) = 0;
    if (use != 0 && !flags) std::printf("  the flag page failed, so no spinning rows: %s\n", why.c_str());
    std::printf("\ndecode, one IQ4_XS layer (gate/up IQ4_XS, down IQ4_NL, 2560 x 640), host to host, %d calls each\n", kCalls);
    std::printf("  top-10 routing over %d experts, %d routings in turn, the card's cache is not warm for them\n", kSlots, kPlans);
    const char* kw = std::getenv("STRATA_XTIER_KEEPWARM");
    std::printf("  the library %s the card busy between calls (STRATA_XTIER_KEEPWARM=%s)\n", kw == nullptr || std::atoi(kw) != 0 ? "keeps" : "does not keep",
                kw != nullptr ? kw : "unset");
    bool all = true;
    for (const int n : {10, 40, 80}) {
        const int n_tok = n / kTop;
        std::vector<std::vector<float>> xs((size_t) kPlans, std::vector<float>((size_t) n_tok * kEmbd));
        std::vector<strata_xtier_meta> plans;
        double groups = 0;
        for (int p = 0; p < kPlans; ++p) {
            for (auto& v : xs[(size_t) p]) v = nd(rng);
            std::vector<int32_t> ids, pool((size_t) kSlots);
            for (int t = 0; t < n_tok; ++t) {
                std::iota(pool.begin(), pool.end(), 0);
                std::shuffle(pool.begin(), pool.end(), rng);
                ids.insert(ids.end(), pool.begin(), pool.begin() + kTop);
            }
            plans.push_back(xtier_plan(ids.data(), n_tok, kTop, addr));
            groups += plans.back().count[0];
        }
        for (const int mode : {0, 1, 2}) {   // back to back, then 0.4 ms apart idle, then 0.4 ms apart with a wait kernel spinning
            if (mode == 2 && !flags) continue;
            std::vector<double> us;
            bool ok = true;
            double card[3] = {};
            for (int i = 0; i < kWarm + kCalls && ok; ++i) {
                const size_t p = (size_t) (i % kPlans);
                const float* rows = nullptr;
                if (i == kWarm) (void) a.timing(card);
                if (mode == 2) ok = a.flag_echo(at_h(r, IN), at_h(r, OUT), at_h(r, HERR), ++seq, kTimeoutNs, e);
                if (mode > 0) spin_us(kGapUs);
                const auto t0 = std::chrono::steady_clock::now();
                ok = ok && a.decode(f, xs[p].data(), n_tok, plans[p], e);
                if (mode == 2) __atomic_store_n(at_w(r, IN), seq, __ATOMIC_RELEASE);   // the decode was queued behind the wait kernel
                ok = ok && a.decode_wait(&rows, e);
                const double d = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                if (i >= kWarm) us.push_back(d);
            }
            if (mode == 2) ok = ok && a.sync(e) && *at_w(r, HERR) == 0;
            if (!ok) std::printf("  %s\n", e.c_str());
            char name[96];
            std::snprintf(name, sizeof(name), "decode %d entries (n_tok %d, %.0f groups)%s", n, n_tok, groups / kPlans,
                          mode == 1 ? ", idle between" : mode == 2 ? ", card spinning between" : "");
            if (!us.empty()) print_row(name, us, ok);
            const int timed = a.timing(card);   // STRATA_XTIER_TIMING=1 only
            if (timed > 0)
                std::printf("    the card per decode: copies in %.1f us, kernels %.1f us, copy out %.1f us\n", card[0] * 1000.0 / timed,
                            card[1] * 1000.0 / timed, card[2] * 1000.0 / timed);
            all = all && ok;
        }
    }
    if (flags) region_free(r, a);
    return all;
}

}  // namespace

int main() {
    int n = 0;
    cudaError_t c = cudaGetDeviceCount(&n);
    if (c != cudaSuccess || n == 0 || cudaFree(nullptr) != cudaSuccess) {
        std::printf("xtier_probe: no CUDA card (%s)\n", cudaGetErrorString(c));
        return 1;
    }
    int cur = 0, rt = 0, drv = 0;
    cudaGetDevice(&cur);
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, cur);
    size_t cfree = 0, ctotal = 0;
    cudaMemGetInfo(&cfree, &ctotal);
    std::printf("xtier_probe: CUDA card %d %s (sm_%d%d, runtime %d, driver %d, free %.2f of %.2f GiB)\n", cur, prop.name, prop.major,
                prop.minor, rt, drv, (double) cfree / (1 << 30), (double) ctotal / (1 << 30));

    std::string e;
    if (load(e) == nullptr) {
        std::printf("xtier_probe: %s\n", e.c_str());
        return 1;
    }
    Device a;
    uint64_t f = 0, t = 0;
    if (!a.open(0, e) || !a.mem_info(f, t, e)) {
        std::printf("xtier_probe: AMD card 0: %s\n", e.c_str());
        return 1;
    }
    std::printf("xtier_probe: AMD card 0 through %s (free %.2f of %.2f GiB)\n\n", library_path().c_str(), (double) f / (1 << 30),
                (double) t / (1 << 30));

    if (cudaStreamCreateWithFlags(&g_cs, cudaStreamNonBlocking) != cudaSuccess || cudaMalloc(&g_dbuf, kBytes) != cudaSuccess ||
        cudaMalloc((void**) &g_bad, sizeof(*g_bad)) != cudaSuccess) {
        std::printf("xtier_probe: the CUDA card has no room for its buffers\n");
        return 1;
    }

    std::printf("registration matrix, %zu MiB of pinned host memory, copy times are the median of 7\n", kBytes >> 20);
    std::printf("  %-1s  %-4s %-9s %-9s %-22s %-22s %s\n", "v", "reg", "cuda->hip", "hip->cuda", "copy NVIDIA to host", "copy AMD to host",
                "how it is made");
    bool works[3] = {};
    for (int k = 0; k < 3; ++k) works[k] = matrix_row((char) ('A' + k), a);

    char use = 0;
    for (const char k : {'B', 'A', 'C'}) {
        if (works[k - 'A']) {
            use = k;
            break;
        }
    }
    std::printf("\nlatency, %d echoes each, one page of mapped flags made by variant %c\n", kEchoes, use ? use : '-');
    if (use == 0) {
        std::printf("  no registration variant works both ways, so the flags are not measured\n");
    } else {
        Region r;
        std::string why;
        if (!region_make(use, a, 4096, r, why)) {
            std::printf("  the flag page failed: %s\n", why.c_str());
        } else {
            flag_round_trip(a, r);
            host_round_trip(a, r, false);
            host_round_trip(a, r, true);
            region_free(r, a);
        }
    }

    decode_timing(a, use);

    a.close();
    std::printf("\nxtier_probe: done, leaving with both runtimes loaded\n");
    return 0;
}
