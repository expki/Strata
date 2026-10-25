// The AMD tier's prompt rows against the CUDA MMQ products on one synthetic layer (exit 77 without a library or an AMD card).
#include "strata/core/xtier.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "../kernels/test_blobs.hpp"
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
#include <thread>
#include <vector>

namespace k = strata::kernels;
namespace mmq = strata::prefill::mmq;
using namespace strata::core::xtier;

namespace {

constexpr int kGu = 23, kDown = 20;       // IQ4_XS gate/up and IQ4_NL down, the Swift IQ4_XS layout
constexpr int64_t kN = 2560, kFf = 640, kTop = 10, kTmax = 4096;
constexpr int kExperts = 64;               // the experts on the tier
constexpr int kGroup = mmq::kGroup;
constexpr size_t kTail = mmq::kTail;
constexpr double kBound = 1e-3;
// FP16 inputs often make q8_1 ties that CUDA's fast math division and HIP's exact one round apart (9.5e-4 measured, 8.6e-5 off the grid).
constexpr double kBound16 = 2e-3;
// The flag words of the one page, a cache line apart.
enum : size_t { kReady = 0, kDone = 16, kHipErr = 32, kCudaErr = 48 };

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

uint32_t word(const SharedHost& flags, size_t i) { return ((volatile uint32_t*) flags.host())[i]; }

// One layer's routing over the tier's experts and the tables the engine's compact SUMS route builds from it (prefill.cpp).
struct Plan {
    int64_t T = 0, rows = 0;
    std::vector<int32_t> ids, cnt, off, order, pair, src, bounds, adds, rows_e, ntok;
    std::vector<float> w;
    std::vector<uint64_t> addr;
    std::vector<int64_t> first, last, adds_at;
};

Plan make_plan(int64_t T, std::mt19937& rng, const std::vector<uint64_t>& amd_addr) {
    Plan p;
    p.T = T;
    p.rows = T * kTop;
    std::vector<int32_t> all(kExperts);
    std::uniform_real_distribution<float> ud(0.05f, 1.f);
    for (int64_t t = 0; t < T; ++t) {   // ten different experts per token, weights that add up to 1
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        float s = 0;
        for (int64_t k = 0; k < kTop; ++k) {
            p.ids.push_back(all[(size_t) k]);
            p.w.push_back(ud(rng));
            s += p.w.back();
        }
        for (int64_t k = 0; k < kTop; ++k) p.w[(size_t) (t * kTop + k)] /= s;
    }
    p.cnt.assign(kExperts, 0);
    for (const int32_t e : p.ids) ++p.cnt[(size_t) e];
    p.off.assign(kExperts, 0);
    int32_t r = 0;
    for (int32_t e = 0; e < kExperts; ++e) {
        if (p.cnt[(size_t) e] == 0) continue;
        p.order.push_back(e);
        p.off[(size_t) e] = r;
        r += p.cnt[(size_t) e];
    }
    p.pair.assign((size_t) p.rows, 0);
    p.src.assign((size_t) p.rows, 0);
    std::vector<int32_t> cur = p.off;
    for (int64_t i = 0; i < p.rows; ++i) {
        const int32_t row = cur[(size_t) p.ids[(size_t) i]]++;
        p.pair[(size_t) row] = (int32_t) i;
        p.src[(size_t) row] = (int32_t) (i / kTop);
    }
    // groups of up to kGroup experts and at most T rows, their bounds, and each group's tokens and row lists
    const size_t n = p.order.size();
    size_t g0 = 0;
    int64_t gr = 0;
    for (size_t j = 0; j < n; ++j) {
        const int64_t c = p.cnt[(size_t) p.order[j]];
        if (j > g0 && (j - g0 == (size_t) kGroup || gr + c > T)) {
            p.first.push_back((int64_t) g0);
            p.last.push_back((int64_t) j);
            g0 = j;
            gr = 0;
        }
        gr += c;
    }
    if (n > g0) {
        p.first.push_back((int64_t) g0);
        p.last.push_back((int64_t) n);
    }
    const size_t ng = p.first.size();
    p.bounds.assign(n + 1 + ng * (kGroup + 1), 0);
    for (size_t j = 0; j < n; ++j) p.bounds[j] = p.off[(size_t) p.order[j]];
    p.bounds[n] = (int32_t) p.rows;
    for (size_t g = 0; g < ng; ++g)
        for (size_t i = 0; i <= (size_t) kGroup; ++i)
            p.bounds[n + 1 + g * (kGroup + 1) + i] =
                p.bounds[std::min((size_t) p.last[g], (size_t) p.first[g] + i)] - p.bounds[(size_t) p.first[g]];
    std::vector<int32_t> tcur((size_t) T, 0), touched;
    for (size_t g = 0; g < ng; ++g) {
        const int32_t r0 = p.bounds[(size_t) p.first[g]], r1 = p.bounds[(size_t) p.last[g]];
        touched.clear();
        for (int32_t q = r0; q < r1; ++q) {
            const int32_t t = p.pair[(size_t) q] / (int32_t) kTop;
            if (tcur[(size_t) t]++ == 0) touched.push_back(t);
        }
        const size_t nt = touched.size(), at = p.adds.size();
        p.adds.resize(at + 2 * nt + 1 + (size_t) (r1 - r0));
        int32_t* a = p.adds.data() + at;
        a[nt] = 0;
        for (size_t i = 0; i < nt; ++i) {
            const int32_t t = touched[i];
            a[i] = t;
            a[nt + 1 + i] = a[nt + i] + tcur[(size_t) t];
            tcur[(size_t) t] = a[nt + i];
        }
        int32_t* list = a + 2 * nt + 1;
        for (int32_t q = r0; q < r1; ++q) list[tcur[(size_t) (p.pair[(size_t) q] / (int32_t) kTop)]++] = q;
        for (const int32_t t : touched) tcur[(size_t) t] = 0;
        p.adds_at.push_back((int64_t) at);
        p.ntok.push_back((int32_t) nt);
    }
    for (const int32_t e : p.order) {
        p.addr.push_back(amd_addr[(size_t) e]);
        p.rows_e.push_back(p.cnt[(size_t) e]);
    }
    return p;
}

// The CUDA card's side: the MMQ products the engine runs, one expert at a time, in little memory (the card serves beside the test).
struct Cuda {
    static constexpr int64_t kRows = 1024;   // the most rows one expert has here: 4096 tokens x 10 of 64 is 640 on average
    size_t gub = 0, db = 0;
    uint8_t *blob = nullptr, *grp_gu = nullptr, *grp_d = nullptr;
    float *x = nullptr, *gu = nullptr, *h = nullptr, *dm = nullptr;
    void *xq = nullptr, *hq = nullptr;
    int32_t *ident = nullptr, *bnd = nullptr;
    cudaStream_t s = nullptr;
    mmq::Context* ctx = nullptr;

    explicit Cuda(const k::NativeExpertLayout& L) {
        gub = mmq::matrix_bytes(kGu, 2 * kFf, kN);
        db = mmq::matrix_bytes(kDown, kN, kFf);
        cuda_ck(cudaStreamCreate(&s), "stream");
        cuda_ck(cudaMalloc((void**) &blob, L.bytes), "blob");
        cuda_ck(cudaMalloc((void**) &grp_gu, gub + kTail), "grp_gu");
        cuda_ck(cudaMalloc((void**) &grp_d, db + kTail), "grp_d");
        cuda_ck(cudaMalloc((void**) &x, (size_t) kRows * kN * 4), "x");
        cuda_ck(cudaMalloc((void**) &gu, (size_t) kRows * 2 * kFf * 4), "gu");
        cuda_ck(cudaMalloc((void**) &h, (size_t) kRows * kFf * 4), "h");
        cuda_ck(cudaMalloc((void**) &dm, (size_t) kRows * kN * 4), "dm");
        cuda_ck(cudaMalloc(&xq, mmq::q8_bytes(kRows, kN)), "xq");
        cuda_ck(cudaMalloc(&hq, mmq::q8_bytes(kRows, kFf)), "hq");
        cuda_ck(cudaMalloc((void**) &ident, (size_t) kRows * 4), "ident");
        cuda_ck(cudaMalloc((void**) &bnd, 2 * 4), "bounds");
        mmq::iota(ident, kRows, s);
        ctx = new mmq::Context();
    }
    ~Cuda() {
        delete ctx;
        for (void* p : {(void*) blob, (void*) grp_gu, (void*) grp_d, (void*) x, (void*) gu, (void*) h, (void*) dm, xq, hq,
                        (void*) ident, (void*) bnd})
            cudaFree(p);
        cudaStreamDestroy(s);
    }
    Cuda(const Cuda&) = delete;
    Cuda& operator=(const Cuda&) = delete;

    // The weighted sums per token in double, from the plan's weights and from weights shuffled within each token.
    bool sums(const Plan& p, const std::vector<float>& x_ref, const k::NativeExpertLayout& L,
              const std::vector<std::vector<uint8_t>>& blobs, std::vector<double>& ref, std::vector<double>& alt) {
        const int64_t T = p.T;
        ref.assign((size_t) (T * kN), 0.0);
        alt.assign((size_t) (T * kN), 0.0);
        std::vector<float> rows, xr;
        for (const int32_t e : p.order) {
            const int32_t c = p.cnt[(size_t) e], r0 = p.off[(size_t) e];
            if (c > kRows) return false;
            xr.resize((size_t) c * kN);   // the expert's rows of x, gathered on the host
            for (int32_t q = 0; q < c; ++q)
                std::memcpy(xr.data() + (size_t) q * kN, x_ref.data() + (size_t) p.src[(size_t) (r0 + q)] * kN, kN * 4);
            const int32_t hb[2] = {0, c};   // CUDA copies pageable memory before the call returns
            cuda_ck(cudaMemcpyAsync(blob, blobs[(size_t) e].data(), L.bytes, cudaMemcpyHostToDevice, s), "blob");
            cuda_ck(cudaMemcpyAsync(x, xr.data(), xr.size() * 4, cudaMemcpyHostToDevice, s), "x");
            cuda_ck(cudaMemcpyAsync(bnd, hb, sizeof(hb), cudaMemcpyHostToDevice, s), "bounds");
            mmq::gather_native(blob, blob + L.up_off, gub / 2, blob + L.down_off, db, grp_gu, grp_d, s, nullptr, nullptr, kTail);
            mmq::quantize(x, nullptr, xq, kGu, kN, kN, c, s);
            mmq::Product g;
            g.w = grp_gu; g.type = kGu; g.w_rows = 2 * kFf; g.w_cols = kN; g.expert_bytes = gub;
            g.n = 1; g.xq = xq; g.bounds = bnd; g.ids = ident; g.total_rows = c; g.max_rows = c; g.dst = gu; g.ld_dst = 2 * kFf;
            ctx->run(g, s);
            mmq::swiglu(gu, h, c, kFf, false, s);
            mmq::quantize(h, nullptr, hq, kDown, kFf, kFf, c, s);
            mmq::Product d;
            d.w = grp_d; d.type = kDown; d.w_rows = kN; d.w_cols = kFf; d.expert_bytes = db;
            d.n = 1; d.xq = hq; d.bounds = bnd; d.ids = ident; d.total_rows = c; d.max_rows = c; d.dst = dm; d.ld_dst = kN;
            ctx->run(d, s);
            rows.resize((size_t) c * kN);
            cuda_ck(cudaMemcpyAsync(rows.data(), dm, rows.size() * 4, cudaMemcpyDeviceToHost, s), "rows");
            cuda_ck(cudaStreamSynchronize(s), "sync");
            for (int32_t q = 0; q < c; ++q) {
                const int32_t i = p.pair[(size_t) (r0 + q)], t = i / (int32_t) kTop;
                const int32_t i2 = t * (int32_t) kTop + (i % (int32_t) kTop + 3) % (int32_t) kTop;   // another weight of the token
                const float* row = rows.data() + (size_t) q * kN;
                double* o = ref.data() + (size_t) t * kN;
                double* o2 = alt.data() + (size_t) t * kN;
                for (int64_t d2 = 0; d2 < kN; ++d2) {
                    o[d2] += (double) p.w[(size_t) i] * row[d2];
                    o2[d2] += (double) p.w[(size_t) i2] * row[d2];
                }
            }
        }
        return true;
    }
};

// The relative L2 error of a against b, or a negative number when a value is not finite or b is all zero.
double ratio(const std::vector<float>& a, const std::vector<double>& b) {
    double diff = 0, ref = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return -1.0;
        const double d = (double) a[i] - b[i];
        diff += d * d;
        ref += b[i] * b[i];
    }
    return ref > 0 ? std::sqrt(diff) / std::sqrt(ref) : -1.0;
}

// The call the engine makes for a plan, with its shared blocks.
strata_xtier_prefill_args args_for(const Plan& p, uint32_t seq, bool f16, const SharedHost& flags, const SharedHost& x,
                                   const SharedHost& w, const SharedHost& sum, const strata_xtier_fmt& f, uint64_t timeout_ns) {
    strata_xtier_prefill_args a;
    std::memset(&a, 0, sizeof(a));
    uint32_t* fh = (uint32_t*) flags.hip_dev();
    a.seq = seq;
    a.x_ready = fh + kReady;
    a.done = fh + kDone;
    a.wait_err = fh + kHipErr;
    a.timeout_ns = timeout_ns;
    a.T = p.T;
    a.N = kN;
    a.K = kTop;
    a.f16 = f16 ? 1 : 0;
    a.x_hip = x.hip_dev();
    a.w_hip = (const float*) w.hip_dev();
    a.sum_host = sum.host();
    a.pair = p.pair.data();
    a.src = p.src.data();
    a.bounds = p.bounds.data();
    a.n_bounds = (int64_t) p.bounds.size();
    a.adds = p.adds.data();
    a.n_adds = (int64_t) p.adds.size();
    a.rows_peer = p.rows;
    a.mmq_gu_type = kGu;
    a.mmq_d_type = kDown;
    a.gu_bytes = mmq::matrix_bytes(kGu, 2 * kFf, kN);
    a.d_bytes = mmq::matrix_bytes(kDown, kN, kFf);
    a.fmt = f;
    a.n_experts = (int64_t) p.order.size();
    a.expert_addr = p.addr.data();
    a.expert_rows = p.rows_e.data();
    a.n_groups = (int64_t) p.first.size();
    a.group_first = p.first.data();
    a.group_last = p.last.data();
    a.group_adds_at = p.adds_at.data();
    a.group_ntok = p.ntok.data();
    return a;
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

    const auto L = k::native_expert_layout(kGu, kDown, kN, kFf);
    const strata_xtier_fmt f = {kGu, kDown, kN, kFf, L.up_off, L.down_off, L.bytes};
    std::mt19937 rng(2711);
    std::vector<std::vector<uint8_t>> blobs;
    for (int i = 0; i < kExperts; ++i) {
        auto blob = random_rows(kGu, 2 * kFf, kN, rng);   // gate rows, then up rows
        const auto down = random_rows(kDown, kN, kFf, rng);
        blob.insert(blob.end(), down.begin(), down.end());
        blobs.push_back(std::move(blob));
    }
    CHECK(blobs[0].size() == L.bytes);

    std::printf("the experts and the prompt buffers on the AMD card\n");
    std::vector<uint64_t> slot_addr;
    CHECK(a.arena(std::vector<uint64_t>(kExperts, L.bytes), slot_addr, e));
    bool filled = slot_addr.size() == (size_t) kExperts;
    for (int i = 0; filled && i < kExperts; ++i) filled = a.fill(i, blobs[(size_t) i].data(), blobs[(size_t) i].size(), e);
    CHECK(filled && a.refill_record(e) && a.refill_done(true) == 1);
    const uint64_t gub = mmq::matrix_bytes(kGu, 2 * kFf, kN), db = mmq::matrix_bytes(kDown, kN, kFf);
    CHECK(!a.prefill_open(kTmax, 1024, gub, db, e) && !e.empty());   // the prompt kernels are 2560 wide
    CHECK(a.prefill_open(kTmax, kN, gub, db, e));
    CHECK(a.prefill_open(64, kN, gub, db, e));                        // smaller asks keep the buffers
    SharedHost flags, xs, ws, sums;
    const bool shared = flags.alloc(a, 4096, e, true) && xs.alloc(a, (size_t) kTmax * kN * 4, e) &&
                        ws.alloc(a, (size_t) kTmax * kTop * 4, e) && sums.alloc(a, (size_t) kTmax * kN * 4, e);
    if (!shared) std::printf("  shared host: %s\n", e.c_str());
    CHECK(shared);
    if (!filled || !shared) {
        std::printf("xtier_prefill_parity: FAILED\n");
        return 1;
    }
    std::printf("  shared host memory: variant %c\n", flags.variant());
    std::memset(flags.host(), 0, 4096);

    Cuda cuda(L);
    uint32_t* fc = (uint32_t*) flags.cuda_dev();
    uint32_t seq = 0;
    constexpr uint64_t kWait = 10000000000ull;

    // one layer through the library, the way the engine hands it over: queue, raise x_ready on CUDA, wait for done on CUDA
    auto run = [&](const Plan& p, bool f16, std::vector<float>& got, double& ms) -> bool {
        const strata_xtier_prefill_args args = args_for(p, ++seq, f16, flags, xs, ws, sums, f, kWait);
        const auto t0 = std::chrono::steady_clock::now();
        if (!a.prefill_layer(args, e)) {
            std::printf("  prefill_layer: %s\n", e.c_str());
            return false;
        }
        k::raise_flag(fc + kReady, seq, cuda.s);
        k::wait_handoff(fc + kDone, seq, fc + kCudaErr, kWait, cuda.s);
        cuda_ck(cudaStreamSynchronize(cuda.s), "wait for done");
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        got.resize((size_t) (p.T * kN));
        if (f16) {
            const uint16_t* s16 = (const uint16_t*) sums.host();
            for (size_t i = 0; i < got.size(); ++i) got[i] = k::f32_from_f16(s16[i]);
        } else {
            std::memcpy(got.data(), sums.host(), got.size() * 4);
        }
        return true;
    };

    std::printf("prompt rows against the CUDA products, relative L2 error must be at most %.0e (FP16 transfers %.0e)\n", kBound, kBound16);
    std::normal_distribution<float> nd(0.f, 0.02f);   // sums of about 100, as a layer's are, inside the FP16 range
    Plan p64;
    for (const auto& [T, f16] : std::vector<std::pair<int64_t, bool>>{{64, false}, {64, true}, {512, false}, {512, true}, {4096, false}, {4096, true}}) {
        const Plan p = make_plan(T, rng, slot_addr);
        std::vector<float> x_ref((size_t) (T * kN));
        for (size_t i = 0; i < x_ref.size(); ++i) {
            const float v = nd(rng);
            if (f16) {   // the engine sends the FP16 copy of its activations, so the reference reads the same values
                const uint16_t hv = k::f16_from_f32(v);
                ((uint16_t*) xs.host())[i] = hv;
                x_ref[i] = k::f32_from_f16(hv);
            } else {
                ((float*) xs.host())[i] = v;
                x_ref[i] = v;
            }
        }
        std::memcpy(ws.host(), p.w.data(), p.w.size() * 4);
        std::vector<double> ref, alt;
        CHECK(cuda.sums(p, x_ref, L, blobs, ref, alt));
        std::vector<float> got;
        double ms = 0;
        const bool ok = run(p, f16, got, ms);
        const double r = ok ? ratio(got, ref) : -1.0;
        const bool pass = r >= 0 && r <= (f16 ? kBound16 : kBound) && word(flags, kHipErr) == 0 && word(flags, kCudaErr) == 0;
        double rms = 0;
        for (const double v : ref) rms += v * v;
        std::printf("  T %4lld %s: %zu experts in %zu groups, reference RMS %.3g, ratio %.3e, %.1f ms host to host  %s\n",
                    (long long) T, f16 ? "fp16" : "fp32", p.order.size(), p.first.size(), std::sqrt(rms / (double) ref.size()), r, ms,
                    pass ? "ok" : "FAIL");
        if (!pass) ++g_fail;
        if (T == 64 && f16 && ok) {   // the check can fail: other weights for the same rows give other sums
            CHECK(ratio(got, alt) > 0.1);
            p64 = p;
        }
    }
    CHECK(word(flags, kHipErr) == 0 && word(flags, kCudaErr) == 0);

    std::printf("what prefill_layer refuses\n");
    {
        strata_xtier_prefill_args bad = args_for(p64, seq + 1, true, flags, xs, ws, sums, f, kWait);
        std::vector<uint64_t> addr = p64.addr;
        addr[0] = 4096;
        bad.expert_addr = addr.data();
        CHECK(!a.prefill_layer(bad, e) && e.find("arena") != std::string::npos);   // a blob outside the arena
        bad = args_for(p64, seq + 1, true, flags, xs, ws, sums, f, kWait);
        bad.T = kTmax + 1;
        CHECK(!a.prefill_layer(bad, e) && !e.empty());                               // more tokens than the buffers hold
        bad = args_for(p64, seq + 1, true, flags, xs, ws, sums, f, kWait);
        bad.n_bounds -= 1;
        CHECK(!a.prefill_layer(bad, e) && !e.empty());                               // bounds that do not match the groups
        bad = args_for(p64, seq + 1, true, flags, xs, ws, sums, f, kWait);
        std::vector<int32_t> src = p64.src;
        src[0] = (int32_t) p64.T;
        bad.src = src.data();
        CHECK(!a.prefill_layer(bad, e) && !e.empty());                               // a row of a token the call does not have
        bad = args_for(p64, seq + 1, true, flags, xs, ws, sums, f, kWait);
        bad.fmt.blob_bytes += 16;
        CHECK(!a.prefill_layer(bad, e) && e.find("layout") != std::string::npos);    // the two sides disagree on the blob
        CHECK(a.sync(e) && word(flags, kDone) == seq);                               // nothing was queued by the refused calls
    }

    std::printf("a prompt layer right after a decode (keep-warm armed)\n");
    {
        const Plan p = make_plan(512, rng, slot_addr);
        std::memcpy(ws.host(), p.w.data(), p.w.size() * 4);
        std::vector<float> x1((size_t) kN), got;
        for (auto& v : x1) v = nd(rng);
        const std::vector<unsigned long long> amd(slot_addr.begin(), slot_addr.end());
        const strata_xtier_meta m = xtier_plan(p.ids.data(), 1, kTop, amd);
        double plain = 0, after = 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));   // no spin left from earlier calls
        CHECK(run(p, true, got, plain));
        const float* rows = nullptr;
        CHECK(a.decode(f, x1.data(), 1, m, e) && a.decode_wait(&rows, e));
        CHECK(run(p, true, got, after));
        std::printf("  info: %.1f ms alone, %.1f ms right after a decode (a 20 ms keep-warm spin ahead of it would show)\n", plain, after);
    }

    std::printf("a done that never comes fails the wait, it does not hang\n");
    {
        set_env("STRATA_TEST_XTIER_DROP", "1");   // read when a tier opens: the first prompt layer of tier b raises no done
        Device b;
        const bool opened = b.open(0, e);
        set_env("STRATA_TEST_XTIER_DROP", "0");
        CHECK(opened);
        SharedHost bf, bx, bw, bs;
        const bool ok = opened && b.prefill_open(64, kN, gub, db, e) && bf.alloc(b, 4096, e, true) && bx.alloc(b, 64 * kN * 2, e) &&
                        bw.alloc(b, 64 * kTop * 4, e) && bs.alloc(b, 64 * kN * 2, e);
        CHECK(ok);
        CHECK(b.prefill_open(128, kN, gub, db, e));                                   // a grow
        uint64_t vfree = 0, vtotal = 0;
        const bool sized = opened && b.mem_info(vfree, vtotal, e);
        CHECK(sized);
        if (sized && vtotal < (64ull << 30))   // 16 group slots of 4 GiB do not fit
            CHECK(!b.prefill_open(128, kN, 1ull << 32, db, e) && !e.empty());          // a grow that does not fit keeps the set in use
        else if (sized)
            std::printf("  skip: a grow of 64 GiB could fit this card's %.0f GiB\n", (double) vtotal / (double) (1ull << 30));
        if (ok) {
            std::memset(bf.host(), 0, 4096);
            std::memset(bx.host(), 0, 64 * kN * 2);
            std::memset(bw.host(), 0, 64 * kTop * 4);
            Plan none;
            none.T = 64;
            none.bounds = {0};
            const strata_xtier_prefill_args args = args_for(none, 1, true, bf, bx, bw, bs, f, kWait);
            const bool queued = b.prefill_layer(args, e);
            if (!queued) std::printf("  prefill_layer: %s\n", e.c_str());
            CHECK(queued);
            uint32_t* bc = (uint32_t*) bf.cuda_dev();
            const auto t0 = std::chrono::steady_clock::now();
            k::raise_flag(bc + kReady, 1, cuda.s);
            k::wait_handoff(bc + kDone, 1, bc + kCudaErr, 200000000ull, cuda.s);
            cuda_ck(cudaStreamSynchronize(cuda.s), "the dropped wait");
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  the CUDA wait gave up after %.0f ms\n", ms);
            CHECK(word(bf, kCudaErr) == 1 && word(bf, kHipErr) == 0 && word(bf, kDone) == 0);
            CHECK(ms >= 150 && ms < 1000);
            CHECK(b.sync(e));
        }
    }

    std::printf("%s\n", g_fail == 0 ? "xtier_prefill_parity: ok" : "xtier_prefill_parity: FAILED");
    return g_fail == 0 ? 0 : 1;
}
