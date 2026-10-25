// The AMD tier's decode rows against the CUDA kernels on one IQ4_XS layer (exit 77 without a library or an AMD card).
#include "strata/core/xtier.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "test_blobs.hpp"
#include "xtier_skip.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
using namespace strata::core::xtier;

namespace {

constexpr int kGu = 23, kDown = 20;       // IQ4_XS gate/up and IQ4_NL down, the Swift IQ4_XS layout
constexpr int64_t kEmbd = 2560, kFf = 640;
constexpr int kExperts = 16, kTop = 10;
constexpr double kBound = 1e-3;

int g_fail = 0;
#define CHECK(c)                                                                                  \
    do {                                                                                          \
        const bool ok_ = (c);                                                                     \
        std::printf("  %-96s %s\n", #c, ok_ ? "ok" : "FAIL");                                     \
        if (!ok_) ++g_fail;                                                                       \
    } while (0)

void set_env(const char* name, const char* v) {
#if defined(_WIN32)
    _putenv_s(name, v);
#else
    setenv(name, v, 1);
#endif
}

void cuda_ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// The CUDA card's side: the experts in one arena and the buffers PeerExperts::launch uses.
struct Cuda {
    k::NativeExpertLayout L;
    size_t slot = 0;
    uint8_t* arena = nullptr;
    float *d_x = nullptr, *d_out = nullptr;
    void *d_q8 = nullptr, *d_scratch = nullptr;
    strata_xtier_meta* d_meta = nullptr;
    cudaStream_t s = nullptr;

    explicit Cuda(const k::NativeExpertLayout& layout, const std::vector<std::vector<uint8_t>>& blobs) : L(layout) {
        slot = (L.bytes + 255) & ~(size_t) 255;
        cuda_ck(cudaStreamCreate(&s), "stream");
        cuda_ck(cudaMalloc((void**) &arena, slot * blobs.size()), "arena");
        for (size_t i = 0; i < blobs.size(); ++i)
            cuda_ck(cudaMemcpy(arena + slot * i, blobs[i].data(), blobs[i].size(), cudaMemcpyHostToDevice), "blob");
        cuda_ck(cudaMalloc((void**) &d_x, (size_t) STRATA_XTIER_CAP * kEmbd * sizeof(float)), "x");
        cuda_ck(cudaMalloc((void**) &d_out, (size_t) STRATA_XTIER_CAP * kEmbd * sizeof(float)), "out");
        cuda_ck(cudaMalloc(&d_q8, (size_t) STRATA_XTIER_CAP * (kEmbd / 32) * 36), "q8");
        cuda_ck(cudaMalloc(&d_scratch, k::native_expert_scratch_bytes(STRATA_XTIER_CAP, kFf)), "scratch");
        cuda_ck(cudaMalloc((void**) &d_meta, sizeof(strata_xtier_meta)), "meta");
    }
    ~Cuda() {
        cudaFree(arena);
        cudaFree(d_x);
        cudaFree(d_out);
        cudaFree(d_q8);
        cudaFree(d_scratch);
        cudaFree(d_meta);
        cudaStreamDestroy(s);
    }
    Cuda(const Cuda&) = delete;
    Cuda& operator=(const Cuda&) = delete;

    std::vector<unsigned long long> addr() const {
        std::vector<unsigned long long> a;
        for (int i = 0; i < kExperts; ++i) a.push_back((unsigned long long) (arena + slot * (size_t) i));
        return a;
    }

    // The rows of a plan in its compact order, the way PeerExperts::launch computes them.
    std::vector<float> rows(const std::vector<float>& x, int64_t n_tok, const strata_xtier_meta& m) {
        cuda_ck(cudaMemcpyAsync(d_x, x.data(), (size_t) n_tok * kEmbd * sizeof(float), cudaMemcpyHostToDevice, s), "x");
        cuda_ck(cudaMemcpyAsync(d_meta, &m, sizeof(m), cudaMemcpyHostToDevice, s), "meta");
        k::quantize_q8_1_rows(d_x, n_tok, kEmbd, d_q8, s);
        k::native_expert_grouped(L, d_meta->ptr, d_meta->start, d_meta->count, d_meta->dst, d_meta->tok, m.count[0], m.count[1], d_q8,
                                 d_scratch, d_out, s);
        std::vector<float> out((size_t) m.count[1] * kEmbd);
        cuda_ck(cudaMemcpyAsync(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost, s), "rows");
        cuda_ck(cudaStreamSynchronize(s), "sync");
        return out;
    }
};

// The relative L2 error of a against b, or a negative number when a value is not finite or b is all zero.
double ratio(const float* a, const float* b, size_t n) {
    double diff = 0, ref = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return -1.0;
        const double d = (double) a[i] - (double) b[i];
        diff += d * d;
        ref += (double) b[i] * (double) b[i];
    }
    return ref > 0 ? std::sqrt(diff) / std::sqrt(ref) : -1.0;
}

// Top-10 routing of n_tok tokens: ten different random experts for each.
std::vector<int32_t> route(int64_t n_tok, std::mt19937& rng) {
    std::vector<int32_t> ids, all(kExperts);
    for (int64_t t = 0; t < n_tok; ++t) {
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        ids.insert(ids.end(), all.begin(), all.begin() + kTop);
    }
    return ids;
}

}  // namespace

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
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("FAIL: no CUDA card\n");
        return 1;
    }

    const auto L = k::native_expert_layout(kGu, kDown, kEmbd, kFf);
    CHECK(k::native_expert_supported(kGu, kDown, kEmbd, kFf));
    const strata_xtier_fmt f = {kGu, kDown, kEmbd, kFf, L.up_off, L.down_off, L.bytes};

    std::mt19937 rng(2710);
    std::vector<std::vector<uint8_t>> blobs;
    for (int i = 0; i < kExperts; ++i) {
        auto blob = random_rows(kGu, 2 * kFf, kEmbd, rng);   // gate rows, then up rows
        const auto down = random_rows(kDown, kEmbd, kFf, rng);
        blob.insert(blob.end(), down.begin(), down.end());
        blobs.push_back(std::move(blob));
    }
    CHECK(blobs[0].size() == L.bytes);

    std::printf("the experts on the AMD card\n");
    std::vector<uint64_t> sizes(kExperts, L.bytes), slot_addr;
    CHECK(!a.arena({1ull << 46}, slot_addr, e) && !e.empty());   // no room, and HIP keeps that error until a later call asks
    CHECK(a.arena(sizes, slot_addr, e) && slot_addr.size() == (size_t) kExperts);
    CHECK(!a.arena(sizes, slot_addr, e) && !e.empty());   // one arena per tier
    bool aligned = slot_addr.size() == (size_t) kExperts;
    for (size_t i = 0; aligned && i < slot_addr.size(); ++i)
        aligned = slot_addr[i] % 256 == 0 && (i == 0 || slot_addr[i] >= slot_addr[i - 1] + L.bytes);
    CHECK(aligned);
    bool filled = aligned;
    for (int i = 0; filled && i < kExperts; ++i) filled = a.fill(i, blobs[(size_t) i].data(), blobs[(size_t) i].size(), e);
    CHECK(filled && a.refill_record(e) && a.refill_done(true) == 1 && a.refill_done(false) == 1);
    bool same = filled;
    std::vector<uint8_t> back(L.bytes);
    for (int i = 0; same && i < kExperts; ++i)
        same = a.read(i, back.data(), back.size(), e) && std::memcmp(back.data(), blobs[(size_t) i].data(), back.size()) == 0;
    CHECK(same);
    CHECK(!a.fill(kExperts, blobs[0].data(), blobs[0].size(), e) && !e.empty());   // no such slot
    CHECK(!a.fill(0, blobs[0].data(), L.bytes + 1, e) && !e.empty());              // more than the slot holds
    if (!aligned || !filled || !same) {
        std::printf("xtier_decode_parity: FAILED\n");
        return 1;
    }

    Cuda cuda(L, blobs);
    const auto cuda_addr = cuda.addr();
    const std::vector<unsigned long long> amd_addr(slot_addr.begin(), slot_addr.end());

    std::printf("decode against the CUDA kernels, relative L2 error must be at most %.0e\n", kBound);
    std::normal_distribution<float> nd(0.f, 1.f);
    double worst = 0;
    std::vector<float> x1;
    std::vector<float> ref1;
    strata_xtier_meta m1;
    std::memset(&m1, 0, sizeof(m1));
    for (const int64_t n_tok : {1, 4, 8}) {
        const auto ids = route(n_tok, rng);
        std::vector<float> x((size_t) n_tok * kEmbd);
        for (auto& v : x) v = nd(rng);
        const auto mc = xtier_plan(ids.data(), n_tok, kTop, cuda_addr);
        const auto ma = xtier_plan(ids.data(), n_tok, kTop, amd_addr);
        const auto ref = cuda.rows(x, n_tok, mc);
        const float* rows = nullptr;
        const bool got = a.decode(f, x.data(), n_tok, ma, e) && a.decode_wait(&rows, e);
        if (!got) std::printf("  decode: %s\n", e.c_str());
        const double r = got ? ratio(rows, ref.data(), ref.size()) : -1.0;
        const bool pass = r >= 0 && r <= kBound;
        double norm = 0;
        for (const float v : ref) norm += (double) v * (double) v;
        std::printf("  n_tok %lld: %d groups, %d entries, reference norm %.4g, ratio %.3e  %s\n", (long long) n_tok, ma.count[0], ma.count[1],
                    std::sqrt(norm), r, pass ? "ok" : "FAIL");
        if (!pass) ++g_fail;
        worst = std::max(worst, r);
        if (n_tok == 4 && got) {   // the check can fail: the reference for another input is far from these rows
            std::vector<float> x2(x.size());
            for (auto& v : x2) v = nd(rng);
            const auto other = cuda.rows(x2, n_tok, mc);
            CHECK(ratio(rows, other.data(), other.size()) > 0.1);
        }
        if (n_tok == 1) {
            x1 = x;
            ref1 = ref;
            m1 = ma;
        }
    }
    std::printf("  maximum ratio %.3e (bound %.0e)\n", worst, kBound);

    std::printf("what decode refuses\n");
    {
        const float* rows = nullptr;
        auto bad = m1;
        CHECK(!a.decode(f, x1.data(), 0, m1, e) && !e.empty());                        // no tokens
        CHECK(!a.decode(f, x1.data(), STRATA_XTIER_CAP + 1, m1, e) && !e.empty());     // too many tokens
        bad.ptr[0] = 4096;
        CHECK(!a.decode(f, x1.data(), 1, bad, e) && e.find("arena") != std::string::npos);   // a blob outside the arena
        bad = m1;
        bad.tok[0] = 1;
        CHECK(!a.decode(f, x1.data(), 1, bad, e) && !e.empty());                       // a token the call does not have
        bad = m1;
        bad.count[1] = STRATA_XTIER_CAP + 1;
        CHECK(!a.decode(f, x1.data(), 1, bad, e) && !e.empty());                       // more entries than the library holds
        auto g = f;
        g.gu_type = 40;
        CHECK(!a.decode(g, x1.data(), 1, m1, e) && !e.empty());                        // NVFP4 keeps its refusal
        g = f;
        g.blob_bytes += 16;
        CHECK(!a.decode(g, x1.data(), 1, m1, e) && e.find("layout") != std::string::npos);   // the two sides disagree on the blob
        CHECK(!a.decode_wait(&rows, e) && !e.empty());                                 // nothing was queued by the refused calls
        CHECK(a.decode(f, x1.data(), 1, m1, e) && !a.decode(f, x1.data(), 1, m1, e) && !e.empty());   // the last one was not waited for
        CHECK(a.decode_wait(&rows, e) && ratio(rows, ref1.data(), ref1.size()) <= kBound);
        auto none = m1;
        none.count[0] = none.count[1] = 0;
        CHECK(a.decode(f, x1.data(), 1, none, e) && a.decode_wait(&rows, e) && rows != nullptr);   // an empty plan is no work
    }

    std::printf("the wait gives up\n");
    {
        void* page = a.host_alloc(4096, e, true);   // a flags page, coherent as the engine's is
        CHECK(page != nullptr);
        if (page != nullptr) {
            std::memset(page, 0, 4096);
            uint32_t* flags = (uint32_t*) page;
            uint32_t* in = flags;
            uint32_t* out = flags + 16;
            uint32_t* herr = flags + 32;
            const float* rows = nullptr;
            set_env("STRATA_XTIER_WAIT_MS", "100");
            // a wait nobody satisfies holds the decode stream for 500 ms, and the decode queues behind it
            CHECK(a.flag_echo(in, out, herr, 1, 500000000ull, e));
            CHECK(a.decode(f, x1.data(), 1, m1, e));
            const auto t0 = std::chrono::steady_clock::now();
            const bool timed_out = !a.decode_wait(&rows, e);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            CHECK(timed_out);
            CHECK(e == "the AMD tier did not answer within STRATA_XTIER_WAIT_MS (100 ms)");
            CHECK(ms >= 90 && ms < 400);
            CHECK(a.hung());   // so a later close leaves the card and its memory alone
            // the decode is still pending and finishes once the stream is free
            set_env("STRATA_XTIER_WAIT_MS", "30000");
            CHECK(a.decode_wait(&rows, e) && ratio(rows, ref1.data(), ref1.size()) <= kBound);
            CHECK(*(volatile uint32_t*) herr == 1 && *(volatile uint32_t*) out == 1);
            CHECK(a.sync(e));
            a.host_free(page);
        }
    }

    a.close();
    std::printf("%s\n", g_fail == 0 ? "xtier_decode_parity: ok" : "xtier_decode_parity: FAILED");
    return g_fail == 0 ? 0 : 1;
}
