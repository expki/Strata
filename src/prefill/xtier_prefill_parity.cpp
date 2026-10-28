// The AMD tier's prompt rows against the CUDA MMQ products on one synthetic layer of each Swift format (exit 77 without a library or an AMD card).
#include "strata/core/xtier.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/tier_prompt.hpp"
#include "../kernels/test_blobs.hpp"
#include "xtier_skip.hpp"

#include <cuda_runtime.h>

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace k = strata::kernels;
namespace mmq = strata::prefill::mmq;
using namespace strata::core::xtier;

namespace {

struct Format {
    int gu, down;
    const char* name;
};
// The expert types of the swift-iq4_xs pack and of the swift-q6_k pack (their native_experts.txt)
constexpr Format kFormats[] = {{23, 20, "IQ4_XS gate/up, IQ4_NL down"}, {14, 8, "Q6_K gate/up, Q8_0 down"}};
constexpr int64_t kN = 2560, kFf = 640, kTop = 10, kTmax = 4096;
constexpr int kExperts = 64;               // the experts on the tier
constexpr int kGroup = mmq::kGroup;
constexpr size_t kTail = mmq::kTail;
constexpr double kBound = 1e-3;
// FP16 inputs often make q8_1 ties that CUDA's fast math division and HIP's exact one round apart (9.5e-4 measured, 8.6e-5 off the grid).
constexpr double kBound16 = 2e-3;
// The flag words of the one page, a cache line apart, the word that says a layer's staged experts landed, and a wait that holds the decode stream.
enum : size_t { kReady = 0, kDone = 16, kHipErr = 32, kCudaErr = 48, kStaged = 64, kHold = 80, kHoldOut = 96, kHoldErr = 112 };
// The stream ring's slots on the card: fewer than a layer streams here, so its slots are reused within a call.
constexpr int64_t kRingSlots = 8;

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
        gub = mmq::matrix_bytes(L.gu_type, 2 * kFf, kN);
        db = mmq::matrix_bytes(L.d_type, kN, kFf);
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
            mmq::quantize(x, nullptr, xq, L.gu_type, kN, kN, c, s);
            mmq::Product g;
            g.w = grp_gu; g.type = L.gu_type; g.w_rows = 2 * kFf; g.w_cols = kN; g.expert_bytes = gub;
            g.n = 1; g.xq = xq; g.bounds = bnd; g.ids = ident; g.total_rows = c; g.max_rows = c; g.dst = gu; g.ld_dst = 2 * kFf;
            ctx->run(g, s);
            mmq::swiglu(gu, h, c, kFf, false, s);
            mmq::quantize(h, nullptr, hq, L.d_type, kFf, kFf, c, s);
            mmq::Product d;
            d.w = grp_d; d.type = L.d_type; d.w_rows = kN; d.w_cols = kFf; d.expert_bytes = db;
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

// A call's streamed experts, per expert of its plan: the HIP address of a host copy (0: from the arena), its ring slot and its tag.
struct Streamed {
    std::vector<uint64_t> host, tag;
    std::vector<int64_t> slot;
    explicit Streamed(size_t n) : host(n, 0), tag(n, 0), slot(n, 0) {}
};

// The call the engine makes for a plan, with its shared blocks, and the host copies of the experts st streams.
strata_xtier_prefill_args args_for(const Plan& p, uint32_t seq, bool f16, const SharedHost& flags, const SharedHost& x,
                                   const SharedHost& w, const SharedHost& sum, const strata_xtier_fmt& f, uint64_t timeout_ns,
                                   int32_t caller = 0, const Streamed* st = nullptr, bool ready = false) {
    strata_xtier_prefill_args a;
    std::memset(&a, 0, sizeof(a));
    uint32_t* fh = (uint32_t*) flags.hip_dev();
    a.seq = seq;
    a.caller = caller;
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
    a.mmq_gu_type = f.gu_type;
    a.mmq_d_type = f.d_type;
    a.gu_bytes = mmq::matrix_bytes(f.gu_type, 2 * kFf, kN);
    a.d_bytes = mmq::matrix_bytes(f.d_type, kN, kFf);
    a.fmt = f;
    a.n_experts = (int64_t) p.order.size();
    a.expert_addr = p.addr.data();
    a.expert_rows = p.rows_e.data();
    a.n_groups = (int64_t) p.first.size();
    a.group_first = p.first.data();
    a.group_last = p.last.data();
    a.group_adds_at = p.adds_at.data();
    a.group_ntok = p.ntok.data();
    if (st != nullptr) {
        a.expert_host = st->host.data();
        a.expert_slot = st->slot.data();
        a.expert_tag = st->tag.data();
    }
    if (ready) {   // the layer's staged experts wait for this word to reach the call's seq
        a.expert_ready = fh + kStaged;
        a.ready_seq = seq;
    }
    return a;
}

// Ordinary memory locked for both cards, as the engine's pinned RAM copy is: mmap, cudaHostRegister, then the library's register.
struct RamCopy {
    uint8_t* host = nullptr;
    void* hip = nullptr;
    size_t bytes = 0;
    bool cuda = false;
    Device* d = nullptr;
    bool make(Device& dev, size_t n, std::string& e) {
#if defined(_WIN32)
        (void) dev; (void) n;
        e = "not on Windows";
        return false;
#else
        void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { e = "mmap failed"; return false; }
        host = (uint8_t*) p;
        bytes = n;
        cuda = cudaHostRegister(host, n, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess;
        if (!cuda) { (void) cudaGetLastError(); e = "cudaHostRegister failed"; return false; }
        if (!dev.register_host(host, n, &hip, e)) return false;
        d = &dev;
        return true;
#endif
    }
    ~RamCopy() {
        if (d != nullptr) d->unregister_host(host);
        if (cuda) (void) cudaHostUnregister(host);
#if !defined(_WIN32)
        if (host != nullptr) munmap(host, bytes);
#endif
    }
};

// One format on the AMD card against the CUDA products, and only the last one also runs the checks after the sums
int run_format(const Format& fm, bool last) {
    std::string e;
    Device a;
    if (!a.open(0, e)) {
        std::printf("%s: open(0): %s\n", xtier_is_skip(e) ? "SKIP" : "FAIL", e.c_str());
        return xtier_is_skip(e) ? 77 : 1;
    }
    std::printf("%s\n", fm.name);

    const auto L = k::native_expert_layout(fm.gu, fm.down, kN, kFf);
    const strata_xtier_fmt f = {fm.gu, fm.down, kN, kFf, L.up_off, L.down_off, L.bytes};
    std::mt19937 rng(2711);
    std::vector<std::vector<uint8_t>> blobs;
    for (int i = 0; i < kExperts; ++i) {
        auto blob = random_rows(fm.gu, 2 * kFf, kN, rng);   // gate rows, then up rows
        const auto down = random_rows(fm.down, kN, kFf, rng);
        blob.insert(blob.end(), down.begin(), down.end());
        blobs.push_back(std::move(blob));
    }
    CHECK(blobs[0].size() == L.bytes);

    std::printf("the prompt buffers and the ring, then the experts, on the AMD card (the engine's order at load)\n");
    const uint64_t gub = mmq::matrix_bytes(fm.gu, 2 * kFf, kN), db = mmq::matrix_bytes(fm.down, kN, kFf);
    CHECK(!a.prefill_open(kTmax, 1024, gub, db, 0, 0, e) && !e.empty());          // the prompt kernels are 2560 wide
    CHECK(a.prompt_tokens() == 0 && a.stream_slots() == 0);
    CHECK(a.prefill_open(kTmax, kN, gub, db, kRingSlots, L.bytes, e));
    CHECK(a.prompt_tokens() == kTmax && a.stream_slots() == kRingSlots);
    std::vector<uint64_t> slot_addr;
    CHECK(a.arena(std::vector<uint64_t>(kExperts, L.bytes), slot_addr, e));
    bool filled = slot_addr.size() == (size_t) kExperts;
    for (int i = 0; filled && i < kExperts; ++i) filled = a.fill(i, blobs[(size_t) i].data(), blobs[(size_t) i].size(), e);
    CHECK(filled && a.refill_record(e) && a.refill_done(true) == 1);
    CHECK(a.prefill_open(kTmax / 2, kN, gub, db, 0, 0, e));                        // the prompt path's own ask after the fill
    CHECK(a.prefill_open(64, kN, gub, db, 2, L.bytes, e));                         // smaller asks keep the buffers and the ring
    CHECK(a.prompt_tokens() == kTmax && a.stream_slots() == kRingSlots);
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
    uint64_t tags = 0;   // each host copy streamed gets a tag of its own
    constexpr uint64_t kWait = 10000000000ull;

    // one layer through the library as the engine hands it over: queue, raise x_ready on CUDA, the staged word a little later, wait for done
    auto run = [&](const Plan& p, bool f16, std::vector<float>& got, double& ms, int32_t caller = 0, const Streamed* host = nullptr) -> bool {
        const strata_xtier_prefill_args args = args_for(p, ++seq, f16, flags, xs, ws, sums, f, kWait, caller, host, host != nullptr);
        const auto t0 = std::chrono::steady_clock::now();
        if (!a.prefill_layer(args, e)) {
            std::printf("  prefill_layer: %s\n", e.c_str());
            return false;
        }
        k::raise_flag(fc + kReady, seq, cuda.s);
        if (host != nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            __atomic_store_n((uint32_t*) flags.host() + kStaged, seq, __ATOMIC_RELEASE);
        }
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
    struct Ran { int64_t T, rows, experts; bool f16; uint32_t seq; uint64_t streamed; };
    std::vector<Ran> ran;   // the layers that ran, in order, each with its index as the caller
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
        const bool ok = run(p, f16, got, ms, (int32_t) ran.size());
        if (ok) ran.push_back({T, p.rows, (int64_t) p.order.size(), f16, seq, 0});
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

    std::printf("experts streamed from host copies give the arena's sums\n");
    {
        // a third of the experts staged in shared host memory and a third in a pinned RAM copy, mixed through the groups
        SharedHost staged;
        RamCopy ram;
        std::string re;
        const bool have_ram = ram.make(a, (size_t) kExperts * L.bytes, re);
        if (!have_ram) std::printf("  info: no pinned RAM copy the AMD card can read (%s), its experts go through the staging\n", re.c_str());
        CHECK(staged.alloc(a, (size_t) kExperts * L.bytes, e, true));   // coherent, as the engine's staging ring: the host writes it
        for (const auto& [T, f16] : std::vector<std::pair<int64_t, bool>>{{512, true}, {4096, true}, {4096, false}}) {
            const Plan p = make_plan(T, rng, slot_addr);
            for (size_t i = 0; i < (size_t) (T * kN); ++i) {
                const float v = nd(rng);
                if (f16) ((uint16_t*) xs.host())[i] = k::f16_from_f32(v);
                else ((float*) xs.host())[i] = v;
            }
            std::memcpy(ws.host(), p.w.data(), p.w.size() * 4);
            std::vector<float> arena, streamed;
            double ms_a = 0, ms_s = 0;
            const bool ok_a = run(p, f16, arena, ms_a, (int32_t) ran.size());
            if (ok_a) ran.push_back({T, p.rows, (int64_t) p.order.size(), f16, seq, 0});
            Streamed host(p.order.size());
            Plan ps = p;
            uint64_t bytes = 0;
            int64_t n_staged = 0, n_ram = 0;
            for (size_t j = 0; j < p.order.size(); ++j) {
                const int32_t ex = p.order[j];
                const size_t at = (size_t) ex * L.bytes;
                if (j % 3 == 1 || (j % 3 == 2 && !have_ram)) {
                    std::memcpy((uint8_t*) staged.host() + at, blobs[(size_t) ex].data(), L.bytes);
                    host.host[j] = (uint64_t) (uintptr_t) ((uint8_t*) staged.hip_dev() + at);
                    ++n_staged;
                } else if (j % 3 == 2) {
                    std::memcpy(ram.host + at, blobs[(size_t) ex].data(), L.bytes);
                    host.host[j] = (uint64_t) (uintptr_t) ((uint8_t*) ram.hip + at);
                    ++n_ram;
                } else {
                    continue;
                }
                host.slot[j] = (int64_t) ((n_staged + n_ram) % kRingSlots);   // more than the ring holds: a slot taken twice in the call
                host.tag[j] = ++tags;
                ps.addr[j] = 0;   // a streamed expert has no arena slot
                bytes += L.bytes;
            }
            const bool ok_s = run(ps, f16, streamed, ms_s, (int32_t) ran.size(), &host);
            if (ok_s) ran.push_back({T, p.rows, (int64_t) p.order.size(), f16, seq, bytes});
            const bool same = ok_a && ok_s && arena.size() == streamed.size() &&
                              std::memcmp(arena.data(), streamed.data(), arena.size() * 4) == 0;
            std::printf("  T %4lld %s: %lld staged and %lld from the RAM copy of %zu experts, %.1f ms against %.1f from the arena  %s\n",
                        (long long) T, f16 ? "fp16" : "fp32", (long long) n_staged, (long long) n_ram, p.order.size(), ms_s, ms_a,
                        same ? "ok" : "FAIL");
            if (!same) ++g_fail;
        }
        // a stage's second call of a layer reads the card's copies its first one made, and a new tag copies again
        {
            const Plan p = make_plan(512, rng, slot_addr);
            for (size_t i = 0; i < (size_t) (512 * kN); ++i) ((uint16_t*) xs.host())[i] = k::f16_from_f32(nd(rng));
            std::memcpy(ws.host(), p.w.data(), p.w.size() * 4);
            std::vector<float> arena, first, again, fresh;
            double ms = 0;
            const bool ok_a = run(p, true, arena, ms, (int32_t) ran.size());
            if (ok_a) ran.push_back({512, p.rows, (int64_t) p.order.size(), true, seq, 0});
            Streamed host(p.order.size());
            Plan ps = p;
            int64_t n = 0;
            for (size_t j = 0; j < p.order.size() && n < kRingSlots; j += 2, ++n) {   // as many as the ring holds, a slot each
                const int32_t ex = p.order[j];
                const size_t at = (size_t) ex * L.bytes;
                std::memcpy((uint8_t*) staged.host() + at, blobs[(size_t) ex].data(), L.bytes);
                host.host[j] = (uint64_t) (uintptr_t) ((uint8_t*) staged.hip_dev() + at);
                host.slot[j] = kRingSlots - 1 - n;
                host.tag[j] = ++tags;
                ps.addr[j] = 0;
            }
            bool ok = run(ps, true, first, ms, (int32_t) ran.size(), &host);
            if (ok) ran.push_back({512, p.rows, (int64_t) p.order.size(), true, seq, (uint64_t) n * L.bytes});
            ok = ok && run(ps, true, again, ms, (int32_t) ran.size(), &host);
            if (ok) ran.push_back({512, p.rows, (int64_t) p.order.size(), true, seq, 0});   // its record says it copied nothing
            uint64_t half = 0;
            for (size_t j = 0, i = 0; j < host.tag.size(); ++j)
                if (host.tag[j] != 0 && i++ % 2 == 0) {
                    host.tag[j] = ++tags;
                    half += L.bytes;
                }
            ok = ok && run(ps, true, fresh, ms, (int32_t) ran.size(), &host);
            if (ok) ran.push_back({512, p.rows, (int64_t) p.order.size(), true, seq, half});
            const auto same = [&](const std::vector<float>& v) {
                return v.size() == arena.size() && std::memcmp(v.data(), arena.data(), v.size() * 4) == 0;
            };
            const bool all = ok_a && ok && same(first) && same(again) && same(fresh);
            std::printf("  %lld experts streamed, read again from the card's copies, then half of them copied anew: the arena's sums  %s\n",
                        (long long) n, all ? "ok" : "FAIL");
            if (!all) ++g_fail;
        }
        CHECK(word(flags, kHipErr) == 0 && word(flags, kCudaErr) == 0);
        CHECK(a.sync(e));
    }

    std::printf("one record per layer that ran, taken once\n");
    {
        std::vector<strata_xtier_prefill_record> recs;
        CHECK(a.sync(e) && a.prefill_stats(recs, e) && recs.size() == ran.size());
        bool each = recs.size() == ran.size();
        for (size_t i = 0; each && i < recs.size(); ++i) {
            const Ran& w = ran[i];
            each = recs[i].caller == (int32_t) i && recs[i].seq == w.seq && recs[i].rows == w.rows && recs[i].experts == w.experts &&
                   recs[i].bytes_in >= (uint64_t) (w.T * kN * (w.f16 ? 2 : 4)) && recs[i].bytes_streamed == w.streamed &&
                   recs[i].kernel_ms > 0 && recs[i].copy_ms > 0 && recs[i].start_ms > 0;
            std::printf("  record %zu: caller %d, seq %u, %lld rows of %lld experts, %llu bytes in, %llu streamed, kernels %.3f ms, "
                        "copies %.3f ms, from queued to x seen %.3f ms\n", i, recs[i].caller, recs[i].seq, (long long) recs[i].rows,
                        (long long) recs[i].experts, (unsigned long long) recs[i].bytes_in, (unsigned long long) recs[i].bytes_streamed,
                        recs[i].kernel_ms, recs[i].copy_ms, recs[i].start_ms);
        }
        CHECK(each);
        CHECK(a.prefill_stats(recs, e) && recs.empty());
    }
    std::printf("a call reads its x before its streamed experts are staged, so its held experts never wait for their copies\n");
    if (p64.T > 0) {   // in every format, so the first tier's streams map onto the card's queues as the engine's one tier does
        SharedHost staged;
        CHECK(staged.alloc(a, L.bytes, e, true));
        std::memcpy(staged.host(), blobs[(size_t) p64.order.back()].data(), L.bytes);
        Streamed host(p64.order.size());
        Plan ps = p64;
        host.host.back() = (uint64_t) (uintptr_t) staged.hip_dev();
        host.tag.back() = ++tags;
        ps.addr.back() = 0;
        std::vector<strata_xtier_prefill_record> recs;
        CHECK(a.sync(e) && a.prefill_stats(recs, e));
        std::memset((uint32_t*) flags.host() + kStaged, 0, 4);
        const strata_xtier_prefill_args args = args_for(ps, ++seq, true, flags, xs, ws, sums, f, kWait, 7, &host, true);
        CHECK(a.prefill_layer(args, e));
        k::raise_flag(fc + kReady, seq, cuda.s);
        cuda_ck(cudaStreamSynchronize(cuda.s), "raise x_ready");
        std::this_thread::sleep_for(std::chrono::milliseconds(300));   // the stage stages the expert late
        __atomic_store_n((uint32_t*) flags.host() + kStaged, seq, __ATOMIC_RELEASE);
        k::wait_handoff(fc + kDone, seq, fc + kCudaErr, kWait, cuda.s);
        cuda_ck(cudaStreamSynchronize(cuda.s), "wait for done");
        CHECK(a.sync(e) && a.prefill_stats(recs, e) && recs.size() == 1 && recs[0].caller == 7);
        const double x_ms = recs.empty() ? -1.0 : recs[0].start_ms;
        std::printf("  x seen %.1f ms after the call was queued, its streamed expert staged 300 ms after\n", x_ms);
        CHECK(x_ms >= 0 && x_ms < 100);
        CHECK(word(flags, kHipErr) == 0 && word(flags, kCudaErr) == 0 && word(flags, kDone) == seq);
    }
    std::printf("a call's start counts from when it was queued, also when the card runs the decode stream's queued mark late\n");
    if (p64.T > 0) {
        uint32_t* fh = (uint32_t*) flags.hip_dev();
        std::vector<strata_xtier_prefill_record> recs;
        CHECK(a.sync(e) && a.prefill_stats(recs, e));
        std::memset((uint32_t*) flags.host() + kHold, 0, 4);
        CHECK(a.flag_echo(fh + kHold, fh + kHoldOut, fh + kHoldErr, 1, kWait, e));   // the decode stream waits until the call is done
        const strata_xtier_prefill_args args = args_for(p64, ++seq, true, flags, xs, ws, sums, f, kWait, 8);
        CHECK(a.prefill_layer(args, e));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));   // x comes 10 ms after the call was queued
        k::raise_flag(fc + kReady, seq, cuda.s);
        k::wait_handoff(fc + kDone, seq, fc + kCudaErr, kWait, cuda.s);
        cuda_ck(cudaStreamSynchronize(cuda.s), "wait for done");
        __atomic_store_n((uint32_t*) flags.host() + kHold, 1u, __ATOMIC_RELEASE);   // the decode stream's mark runs only now
        CHECK(a.sync(e) && a.prefill_stats(recs, e) && recs.size() == 1 && recs[0].caller == 8);
        const double x_ms = recs.empty() ? -1.0 : recs[0].start_ms;
        std::printf("  x seen %.2f ms after the call was queued, raised 10 ms after\n", x_ms);
        CHECK(x_ms >= 5 && x_ms < 100);
        CHECK(word(flags, kHoldOut) == 1 && word(flags, kHoldErr) == 0 && word(flags, kHipErr) == 0 && word(flags, kDone) == seq);
    }

    if (!last) return 0;
    if (p64.T == 0) {   // the refusals below start from the 64 token layer
        std::printf("xtier_prefill_parity: FAILED\n");
        return 1;
    }

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

    std::printf("prefill_stats does not wait for a call that waits inside the library\n");
    {
        // eight calls behind an x_ready not raised yet take the eight staging slots, so a ninth waits in prefill_layer holding the prompt lock
        constexpr int kHeld = 9;
        std::vector<strata_xtier_prefill_record> recs;
        CHECK(a.sync(e) && a.prefill_stats(recs, e));
        Plan none;
        none.T = 64;
        none.bounds = {0};
        const uint32_t base = seq;
        bool queued = true;
        for (int i = 0; i < kHeld - 1; ++i)
            queued = queued && a.prefill_layer(args_for(none, base + 1 + (uint32_t) i, true, flags, xs, ws, sums, f, kWait, 100 + i), e);
        CHECK(queued);
        std::atomic<bool> started{false}, done{false};
        std::string e5;
        std::thread last_call([&] {
            started = true;
            done = a.prefill_layer(args_for(none, base + kHeld, true, flags, xs, ws, sums, f, kWait, 100 + kHeld - 1), e5);
        });
        while (!started) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const auto t0 = std::chrono::steady_clock::now();
        const bool got = a.prefill_stats(recs, e);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  prefill_stats took %.2f ms while the ninth call waited\n", ms);
        CHECK(got && recs.empty() && ms < 50 && !done);
        k::raise_flag(fc + kReady, base + kHeld, cuda.s);
        cuda_ck(cudaStreamSynchronize(cuda.s), "raise x_ready");
        last_call.join();
        CHECK(done);
        seq = base + kHeld;
        CHECK(a.sync(e) && word(flags, kDone) == seq && word(flags, kHipErr) == 0);
        CHECK(a.prefill_stats(recs, e) && recs.size() == (size_t) kHeld);
        bool order = recs.size() == (size_t) kHeld;
        for (size_t i = 0; order && i < recs.size(); ++i) order = recs[i].caller == 100 + (int32_t) i && recs[i].rows == 0;
        CHECK(order);   // none was lost while the lock was held, and they come in the order they were queued
        CHECK(a.prefill_stats(recs, e) && recs.empty());
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

    std::printf("a staged word that never comes fails its wait, it does not hang\n");
    {
        SharedHost staged;
        CHECK(staged.alloc(a, L.bytes, e));
        Streamed host(p64.order.size());
        host.host.back() = (uint64_t) (uintptr_t) staged.hip_dev();
        host.tag.back() = ++tags;
        std::memset((uint32_t*) flags.host() + kStaged, 0, 4);
        const strata_xtier_prefill_args args = args_for(p64, ++seq, true, flags, xs, ws, sums, f, 200000000ull, 0, &host, true);
        const bool queued = a.prefill_layer(args, e);
        if (!queued) std::printf("  prefill_layer: %s\n", e.c_str());
        CHECK(queued);
        const auto t0 = std::chrono::steady_clock::now();
        k::raise_flag(fc + kReady, seq, cuda.s);
        k::wait_handoff(fc + kDone, seq, fc + kCudaErr, 5000000000ull, cuda.s);
        cuda_ck(cudaStreamSynchronize(cuda.s), "the staged wait");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  the layer was done after %.0f ms, its wait for the staged word gave up\n", ms);
        CHECK(word(flags, kHipErr) == 1 && word(flags, kCudaErr) == 0 && word(flags, kDone) == seq);
        CHECK(ms >= 150 && ms < 2000);
        CHECK(a.sync(e));
        std::memset((uint32_t*) flags.host() + kHipErr, 0, 4);
    }

    std::printf("a done that never comes fails the wait, it does not hang\n");
    {
        set_env("STRATA_TEST_XTIER_DROP", "1");   // read when a tier opens: no prompt layer of tier b raises its done
        Device b;
        const bool opened = b.open(0, e);
        set_env("STRATA_TEST_XTIER_DROP", "0");
        CHECK(opened);
        SharedHost bf, bx, bw, bs;
        const bool ok = opened && b.prefill_open(64, kN, gub, db, 0, 0, e) && bf.alloc(b, 4096, e, true) && bx.alloc(b, 64 * kN * 2, e) &&
                        bw.alloc(b, 64 * kTop * 4, e) && bs.alloc(b, 64 * kN * 2, e);
        CHECK(ok);
        CHECK(b.prefill_open(128, kN, gub, db, 0, 0, e));                             // a grow
        uint64_t vfree = 0, vtotal = 0;
        const bool sized = opened && b.mem_info(vfree, vtotal, e);
        CHECK(sized);
        if (sized && vtotal < (64ull << 30))   // 16 group slots of 4 GiB do not fit
            CHECK(!b.prefill_open(128, kN, 1ull << 32, db, 0, 0, e) && !e.empty());    // a grow that does not fit keeps the set in use
        if (sized && vtotal < (64ull << 30))
            CHECK(!b.prefill_open(128, kN, gub, db, 64, 1ull << 30, e) && !e.empty());   // a ring that does not fit leaves the prompt buffers
        else if (sized)
            std::printf("  skip: a grow of 64 GiB could fit this card's %.0f GiB\n", (double) vtotal / (double) (1ull << 30));
        if (ok) {
            std::memset(bf.host(), 0, 4096);
            std::memset(bx.host(), 0, 64 * kN * 2);
            std::memset(bw.host(), 0, 64 * kTop * 4);
            Plan none;
            none.T = 64;
            none.bounds = {0};
            if (p64.T > 0) {   // tier b has no stream ring, so it refuses a streamed expert
                Streamed host(p64.order.size());
                host.host[0] = (uint64_t) (uintptr_t) bx.hip_dev();
                host.tag[0] = ++tags;
                CHECK(!b.prefill_layer(args_for(p64, 1, true, bf, bx, bw, bs, f, kWait, 0, &host), e) && e.find("ring") != std::string::npos);
            }
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
            // a chunk in halves on a card that stopped: the second call raises no done either, and its wait ends with the first's
            std::memset((uint32_t*) bf.host() + kCudaErr, 0, 4);
            const bool two = b.prefill_layer(args_for(none, 2, true, bf, bx, bw, bs, f, kWait), e) &&
                             b.prefill_layer(args_for(none, 3, true, bf, bx, bw, bs, f, kWait), e);
            if (!two) std::printf("  prefill_layer: %s\n", e.c_str());
            CHECK(two);
            const auto t1 = std::chrono::steady_clock::now();
            k::raise_flag(bc + kReady, 3, cuda.s);
            k::wait_handoff_or_err(bc + kDone, 2, bc + kCudaErr, 200000000ull, cuda.s);
            k::wait_handoff_or_err(bc + kDone, 3, bc + kCudaErr, 200000000ull, cuda.s);
            cuda_ck(cudaStreamSynchronize(cuda.s), "the halves' dropped waits");
            const double ms2 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
            std::printf("  the two halves' CUDA waits gave up after %.0f ms\n", ms2);
            CHECK(word(bf, kCudaErr) == 1 && word(bf, kHipErr) == 0 && word(bf, kDone) == 0);
            CHECK(ms2 >= 150 && ms2 < 350);
            CHECK(b.sync(e));
        }
    }

    std::printf("the prompt buffers take what the start-up chunk plan counts for them\n");
    {
        Device c;
        uint64_t before = 0, after = 0, vtotal = 0;
        const bool measured = c.open(0, e) && c.mem_info(before, vtotal, e) && c.prefill_open(kTmax, kN, gub, db, 0, 0, e) &&
                              c.mem_info(after, vtotal, e);
        CHECK(measured);
        const uint64_t want = strata::prefill::tier_prompt_bytes(kTmax, gub, db), took = before > after ? before - after : 0;
        std::printf("  %llu MiB for %lld tokens, %llu MiB counted\n", (unsigned long long) (took >> 20), (long long) kTmax,
                    (unsigned long long) (want >> 20));
        CHECK(want > (100ull << 20) && took + (4ull << 20) >= want && took <= want + (32ull << 20));   // the card rounds each buffer up
    }

    return 0;
}

}  // namespace

int main() {
    std::string e;
    if (load(e) == nullptr) {
        std::printf("%s: %s\n", xtier_is_skip(e) ? "SKIP" : "FAIL", e.c_str());
        return xtier_is_skip(e) ? 77 : 1;
    }
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("FAIL: no CUDA card\n");
        return 1;
    }
    for (const Format& fm : kFormats)
        if (const int rc = run_format(fm, &fm == std::end(kFormats) - 1); rc != 0) return rc;
    std::printf("%s\n", g_fail == 0 ? "xtier_prefill_parity: ok" : "xtier_prefill_parity: FAILED");
    return g_fail == 0 ? 0 : 1;
}
