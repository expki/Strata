// The HIP library that holds an AMD card as an expert tier for the CUDA engine (include/strata/core/xtier_abi.hpp).
#include "strata/core/xtier_abi.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#define STRATA_XTIER_EXPORT extern "C" __declspec(dllexport)
#else
#define STRATA_XTIER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

namespace mmq = strata::prefill::mmq;

// A keep-warm spin lasts at most this long, so an idle card is back at idle this long after the last decode.
constexpr unsigned long long kWarmNs = 20000000ull;

// The prompt path's widths, the most experts a token takes and the bounds a call has.
constexpr int64_t kPfEmbd = 2560, kPfFf = 640, kPfMaxTop = 16, kPfMaxBounds = 1 << 15;
// Staged tables, one per call in flight: a layer split has one call in flight per stage, and a fifth stage's call would wait.
constexpr int kPfSlots = 4;

// The prompt buffers for chunks of up to T tokens (prefill_open), one set at a time.
struct PromptBufs {
    int64_t T = 0;
    uint64_t gu = 0, d = 0;
    std::vector<void*> dev;                   // every device buffer below
    float *x = nullptr, *w = nullptr, *sum = nullptr, *gu_rows = nullptr, *h = nullptr, *dm = nullptr;
    uint16_t* sum16 = nullptr;
    void *xq = nullptr, *hq = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    int32_t* tab = nullptr;                   // a call's tables: pair, src, bounds, adds
    int32_t* ident = nullptr;
    int32_t* stage[kPfSlots] = {};            // pinned: a call's tables until its upload has run
};

// The prompt rows' buffers on the card, sized for the largest chunk asked for and used on one stream behind pf_mu.
struct Prompt : PromptBufs {
    hipStream_t s = nullptr;
    hipEvent_t staged[kPfSlots] = {};
    hipEvent_t done = nullptr;                // the last layer's work, where its keep-warm spin starts
    uint64_t calls = 0;
    int fit_gu = -1, fit_d = -1;              // the type pair MMQ was last found to fit
    std::unique_ptr<strata::prefill::mmq::Context> ctx;
    // STRATA_XTIER_TIMING=1: events around each prompt layer, summed once they have run
    struct Tm {
        hipEvent_t e[5] = {};                 // start, x seen, inputs in, kernels done, sums out
        double host = 0, lock = 0, slot = 0;  // host ms in the call, waiting for pf_mu, waiting for a staging slot
        int64_t rows = 0, experts = 0;
    };
    std::vector<Tm> tq, tfree;                // in flight oldest first, and spare event sets
    Tm tlast;                                 // the last summed layer, kept for the card's idle gap before the next
    bool thave = false;
    double tms[5] = {}, thost = 0, tlock = 0, tslot = 0;
    int64_t trows = 0, texp = 0, tn = 0;
};

// The ints of a call's tables: pair and src, the bounds, and the adds of at most 3 per row plus one per group.
size_t tab_ints(int64_t T) { return (size_t) (5 * T * kPfMaxTop + 2 * kPfMaxBounds); }

struct Tier {
    int device = 0;
    hipStream_t decode = nullptr;
    hipStream_t refill = nullptr;
    hipEvent_t refill_ev = nullptr;
    hipEvent_t done_ev = nullptr;           // recorded after a decode's rows are copied out
    void* dbuf = nullptr;                   // the probe's device buffer for the copy timing
    uint64_t dbuf_bytes = 0;
    unsigned long long* dcount = nullptr;   // the probe's count of wrong words
    uint8_t* arena = nullptr;
    uint64_t arena_bytes = 0;
    std::vector<uint64_t> slot_off, slot_len;
    float* h_x = nullptr;                   // pinned: the input rows, the result rows and the plan
    float* h_out = nullptr;
    strata_xtier_meta* h_meta = nullptr;
    float* d_x = nullptr;
    float* d_out = nullptr;
    strata_xtier_meta* d_meta = nullptr;
    void* d_q8 = nullptr;
    void* d_scratch = nullptr;
    bool pending = false;                   // a decode was queued and its rows not taken yet
    bool queued = false;                    // ... and it has work on the stream
    bool timing = false;                    // STRATA_XTIER_TIMING=1: events around a decode's copies and kernels
    hipEvent_t tev[4] = {};
    double tms[3] = {};
    int tn = 0;
    bool keepwarm = false;                  // STRATA_XTIER_KEEPWARM (default on): a bounded spin on its own stream keeps the clock up between decodes
    hipStream_t warm = nullptr;
    uint32_t* wflag = nullptr;              // mapped: [0] ends the spin, [1] its timeout mark (unread)
    uint32_t* wflag_dev = nullptr;
    uint32_t wseq = 0;
    std::chrono::steady_clock::time_point warm_at{};
    std::string err;
    std::mutex mu;                          // guards err, pending, queued, the timing sums, wseq and warm_at: the engine decodes on one thread and fills on another
    std::mutex pf_mu;                       // guards pf: each layer split stage reads its prompts on its own thread
    Prompt pf;
    uint64_t pf_drop = 0;                   // STRATA_TEST_XTIER_DROP=N: the Nth prompt layer raises no done
};

// The engine calls from many threads, so every entry point selects the card first and holds the tier's lock.
struct Enter {
    std::unique_lock<std::mutex> lk;
    bool ok = false;
    explicit Enter(Tier* t) {
        if (t == nullptr) return;
        lk = std::unique_lock<std::mutex>(t->mu);
        (void) hipGetLastError();   // an earlier failure is already reported, and the kernels' launch checks must not see it
        const hipError_t e = hipSetDevice(t->device);
        if (e != hipSuccess) {
            t->err = std::string("hipSetDevice: ") + hipGetErrorString(e);
            return;
        }
        ok = true;
    }
};

int fail(Tier* t, const char* what, hipError_t e) {
    t->err = std::string(what) + ": " + hipGetErrorString(e);
    return (int) e;
}

void put(char* err, size_t len, const std::string& s) {
    if (err == nullptr || len == 0) return;
    std::snprintf(err, len, "%s", s.c_str());
}

// Frees a set of prompt buffers, with the card selected and the prompt stream idle.
void prompt_free(PromptBufs& b) {
    for (void* q : b.dev) (void) hipFree(q);
    for (int32_t* q : b.stage)
        if (q != nullptr) (void) hipHostFree(q);
    b = PromptBufs{};
}

// Frees everything the tier holds, with its card selected.
void release(Tier* t) {
    if (t->wflag != nullptr) __atomic_store_n(&t->wflag[0], t->wseq, __ATOMIC_RELEASE);   // the spin ends now, not at its bound
    if (t->warm != nullptr) (void) hipStreamSynchronize(t->warm);
    if (t->decode != nullptr) (void) hipStreamSynchronize(t->decode);
    if (t->refill != nullptr) (void) hipStreamSynchronize(t->refill);
    if (t->pf.s != nullptr) (void) hipStreamSynchronize(t->pf.s);
    prompt_free(t->pf);
    t->pf.ctx.reset();
    for (hipEvent_t& e : t->pf.staged)
        if (e != nullptr) (void) hipEventDestroy(e);
    if (t->pf.done != nullptr) (void) hipEventDestroy(t->pf.done);
    if (t->pf.s != nullptr) (void) hipStreamDestroy(t->pf.s);
    (void) hipFree(t->dbuf);
    (void) hipFree(t->dcount);
    (void) hipFree(t->arena);
    (void) hipFree(t->d_x);
    (void) hipFree(t->d_out);
    (void) hipFree(t->d_meta);
    (void) hipFree(t->d_q8);
    (void) hipFree(t->d_scratch);
    if (t->h_x != nullptr) (void) hipHostFree(t->h_x);
    if (t->h_out != nullptr) (void) hipHostFree(t->h_out);
    if (t->h_meta != nullptr) (void) hipHostFree(t->h_meta);
    if (t->wflag != nullptr) (void) hipHostFree(t->wflag);
    if (t->refill_ev != nullptr) (void) hipEventDestroy(t->refill_ev);
    if (t->done_ev != nullptr) (void) hipEventDestroy(t->done_ev);
    for (hipEvent_t& e : t->tev)
        if (e != nullptr) (void) hipEventDestroy(e);
    if (t->decode != nullptr) (void) hipStreamDestroy(t->decode);
    if (t->refill != nullptr) (void) hipStreamDestroy(t->refill);
    if (t->warm != nullptr) (void) hipStreamDestroy(t->warm);
}

// Ends the last spin and starts one for kWarmNs on a stream nothing waits for: after a decode is queued (at most every kWarmNs / 2), or when a prompt layer's work ends.
void keep_warm(Tier* t, hipEvent_t after = nullptr) {
    const auto now = std::chrono::steady_clock::now();
    if (after == nullptr && t->wseq != 0 && now - t->warm_at < std::chrono::nanoseconds(kWarmNs / 2)) return;
    __atomic_store_n(&t->wflag[0], t->wseq, __ATOMIC_RELEASE);
    ++t->wseq;
    if (after != nullptr) (void) hipStreamWaitEvent(t->warm, after, 0);
    strata::kernels::wait_handoff(t->wflag_dev, t->wseq, t->wflag_dev + 1, kWarmNs, t->warm);
    t->warm_at = now;
}

// Whether a decode call is something the kernels can run, with the reason in t->err when it is not. A bad address would fault the card.
bool decode_ok(Tier* t, const strata_xtier_fmt* f, const float* x, int64_t n_tok, const strata_xtier_meta* m) {
    const auto no = [t](const std::string& why) {
        t->err = "decode: " + why;
        return false;
    };
    if (f == nullptr || x == nullptr || m == nullptr) return no("no format, input or plan given");
    if (t->pending) return no("the last decode was not waited for");
    if (t->arena == nullptr) return no("there is no arena");
    if (n_tok < 1 || n_tok > STRATA_XTIER_CAP) return no("n_tok " + std::to_string(n_tok) + " is not 1 to " + std::to_string(STRATA_XTIER_CAP));
    if (f->n_embd < 1 || f->n_embd > STRATA_XTIER_MAX_EMBD || f->n_ff < 1 || f->n_ff > STRATA_XTIER_MAX_FF)
        return no("n_embd " + std::to_string(f->n_embd) + " and n_ff " + std::to_string(f->n_ff) + " are wider than the library takes");
    if (!strata::kernels::native_expert_supported(f->gu_type, f->d_type, f->n_embd, f->n_ff)) return no("the expert format is not supported");
    const auto L = strata::kernels::native_expert_layout(f->gu_type, f->d_type, f->n_embd, f->n_ff);
    if (L.tail_off != 0) return no("NVFP4 experts are not supported");
    if (L.up_off != f->up_off || L.down_off != f->down_off || L.bytes != f->blob_bytes)
        return no("the engine's blob layout is not the library's");
    const int32_t groups = m->count[0], rows = m->count[1];
    if (rows < 0 || rows > STRATA_XTIER_CAP || groups < 0 || groups > rows) return no("the plan's counts are out of range");
    if (rows == 0) return true;
    if (m->start[0] != 0 || m->start[groups] != rows) return no("the plan's groups do not cover its entries");
    for (int32_t g = 0; g < groups; ++g) {
        const unsigned long long p = m->ptr[g], lo = (unsigned long long) (uintptr_t) t->arena;
        if (m->start[g] > m->start[g + 1]) return no("the plan's groups are not in order");
        if (p < lo || p - lo > t->arena_bytes || t->arena_bytes - (p - lo) < L.bytes) return no("group " + std::to_string(g) + " is not inside the arena");
        if ((p - lo) % 256 != 0) return no("group " + std::to_string(g) + " is not 256 byte aligned in the arena");
    }
    for (int32_t i = 0; i < rows; ++i) {
        if (m->tok[i] < 0 || m->tok[i] >= n_tok) return no("entry " + std::to_string(i) + " names a token the call does not have");
        if (m->dst[i] < 0 || m->dst[i] >= rows) return no("entry " + std::to_string(i) + " names a row outside the result");
    }
    return true;
}

__global__ void probe_write_kernel(volatile uint32_t* p, uint64_t words, uint32_t seed) {
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (uint64_t) gridDim.x * blockDim.x)
        p[i] = STRATA_XTIER_PATTERN(seed, i);
}

__global__ void probe_check_kernel(const volatile uint32_t* p, uint64_t words, uint32_t seed, unsigned long long* bad) {
    unsigned long long mine = 0;
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (uint64_t) gridDim.x * blockDim.x)
        mine += p[i] != STRATA_XTIER_PATTERN(seed, i);
    if (mine != 0) atomicAdd(bad, mine);
}

// op 0 and 1 of the probe: a kernel writes the seed's pattern into mapped host memory, or counts the words that differ from it
int probe_pattern(Tier* t, bool write, void* host, uint64_t bytes, uint32_t seed, uint64_t* wrong) {
    uint32_t* dev = nullptr;
    hipError_t e = hipHostGetDevicePointer((void**) &dev, host, 0);
    if (e != hipSuccess) return fail(t, "hipHostGetDevicePointer", e);
    const uint64_t words = bytes / 4;
    const unsigned blocks = (unsigned) (words / 256 < 2048 ? (words + 255) / 256 : 2048);
    if (write) {
        probe_write_kernel<<<blocks, 256, 0, t->decode>>>(dev, words, seed);
    } else {
        if (t->dcount == nullptr && (e = hipMalloc((void**) &t->dcount, sizeof(unsigned long long))) != hipSuccess)
            return fail(t, "hipMalloc", e);
        if ((e = hipMemsetAsync(t->dcount, 0, sizeof(unsigned long long), t->decode)) != hipSuccess) return fail(t, "hipMemsetAsync", e);
        probe_check_kernel<<<blocks, 256, 0, t->decode>>>(dev, words, seed, t->dcount);
    }
    if ((e = hipGetLastError()) != hipSuccess) return fail(t, "probe kernel", e);
    unsigned long long bad = 0;
    if (!write && (e = hipMemcpyAsync(&bad, t->dcount, sizeof(bad), hipMemcpyDeviceToHost, t->decode)) != hipSuccess)
        return fail(t, "hipMemcpyAsync", e);
    if ((e = hipStreamSynchronize(t->decode)) != hipSuccess) return fail(t, "hipStreamSynchronize", e);
    *wrong = bad;
    return 0;
}

// op 2 of the probe: the time of one copy from the library's device buffer into host memory, with events
int probe_copy(Tier* t, void* host, uint64_t bytes, float* ms) {
    hipError_t e = hipSuccess;
    if (t->dbuf_bytes < bytes) {
        if (t->dbuf != nullptr) (void) hipFree(t->dbuf);
        t->dbuf = nullptr;
        t->dbuf_bytes = 0;
        if ((e = hipMalloc(&t->dbuf, (size_t) bytes)) != hipSuccess) return fail(t, "hipMalloc", e);
        t->dbuf_bytes = bytes;
    }
    hipEvent_t a = nullptr, b = nullptr;
    if ((e = hipEventCreate(&a)) != hipSuccess) return fail(t, "hipEventCreate", e);
    if ((e = hipEventCreate(&b)) != hipSuccess) {
        (void) hipEventDestroy(a);
        return fail(t, "hipEventCreate", e);
    }
    (void) hipEventRecord(a, t->decode);
    e = hipMemcpyAsync(host, t->dbuf, (size_t) bytes, hipMemcpyDeviceToHost, t->decode);
    (void) hipEventRecord(b, t->decode);
    if (e == hipSuccess) e = hipEventSynchronize(b);
    if (e == hipSuccess) e = hipEventElapsedTime(ms, a, b);
    (void) hipEventDestroy(a);
    (void) hipEventDestroy(b);
    return e == hipSuccess ? 0 : fail(t, "probe copy", e);
}

// The slot's address in the arena, or null with the reason in t->err.
uint8_t* slot_ptr(Tier* t, const char* who, int64_t slot, uint64_t bytes) {
    if (t->arena == nullptr || slot < 0 || (size_t) slot >= t->slot_off.size()) {
        t->err = std::string(who) + ": slot " + std::to_string(slot) + " does not exist";
        return nullptr;
    }
    if (bytes > t->slot_len[(size_t) slot]) {
        t->err = std::string(who) + ": " + std::to_string(bytes) + " bytes do not fit slot " + std::to_string(slot) + " of " +
                 std::to_string(t->slot_len[(size_t) slot]);
        return nullptr;
    }
    return t->arena + t->slot_off[(size_t) slot];
}

// Sizes the prompt buffers for chunks of up to T_max tokens, or grows them, with pf_mu held.
int prompt_open(Tier* t, int64_t T_max, int64_t N, uint64_t gu_max, uint64_t d_max, std::string& why) {
    (void) hipGetLastError();
    hipError_t e = hipSetDevice(t->device);
    if (e != hipSuccess) {
        why = std::string("hipSetDevice: ") + hipGetErrorString(e);
        return (int) e;
    }
    if (N != kPfEmbd) {
        why = "prefill_open: the prompt kernels take rows of " + std::to_string(kPfEmbd) + " values, not " + std::to_string(N);
        return -1;
    }
    if (T_max < 1 || T_max > (1 << 20) || gu_max == 0 || d_max == 0 || gu_max > (1ull << 32) || d_max > (1ull << 32)) {
        why = "prefill_open: a chunk of " + std::to_string(T_max) + " tokens or matrices of " + std::to_string(gu_max) + " and " +
              std::to_string(d_max) + " bytes are out of range";
        return -1;
    }
    Prompt& p = t->pf;
    if (p.T >= T_max && p.gu >= gu_max && p.d >= d_max) return 0;
    const int64_t T = std::max(T_max, p.T);
    const uint64_t gu = std::max(gu_max, p.gu), d = std::max(d_max, p.d);
    const auto step = [&why](hipError_t r, const char* what) {
        if (r == hipSuccess) return true;
        why = std::string("prefill_open: ") + what + ": " + hipGetErrorString(r);
        (void) hipGetLastError();   // a failed allocation stays the last error, which a kernel's launch check would take as its own
        return false;
    };
    bool ok = p.s != nullptr || step(hipStreamCreateWithFlags(&p.s, hipStreamNonBlocking), "prompt stream");
    for (int i = 0; ok && i < kPfSlots; ++i)
        if (p.staged[i] == nullptr) ok = step(hipEventCreateWithFlags(&p.staged[i], hipEventDisableTiming), "staging event");
    if (ok && p.done == nullptr) ok = step(hipEventCreateWithFlags(&p.done, hipEventDisableTiming), "prompt event");
    if (ok && !p.ctx) p.ctx = std::make_unique<strata::prefill::mmq::Context>();
    if (!ok) return -1;
    // the new set first, so a grow that does not fit leaves the stages the set they already use
    PromptBufs n;
    uint64_t total = 0;
    const auto take = [&](size_t bytes) -> void* {
        void* q = nullptr;
        total += bytes;
        if (!ok || !(ok = step(hipMalloc(&q, bytes), "hipMalloc"))) return nullptr;
        n.dev.push_back(q);
        return q;
    };
    const size_t TN = (size_t) T * kPfEmbd;
    n.x = (float*) take(TN * 4);
    n.w = (float*) take((size_t) T * kPfMaxTop * 4);
    n.sum = (float*) take(TN * 4);
    n.sum16 = (uint16_t*) take(TN * 2);
    n.gu_rows = (float*) take((size_t) T * 2 * kPfFf * 4);
    n.h = (float*) take((size_t) T * kPfFf * 4);
    n.dm = (float*) take(TN * 4);
    n.xq = take(mmq::q8_bytes(T, kPfEmbd));
    n.hq = take(mmq::q8_bytes(T, kPfFf));
    n.grp_gu = (uint8_t*) take(mmq::kGroup * gu + mmq::kTail);
    n.grp_d = (uint8_t*) take(mmq::kGroup * d + mmq::kTail);
    n.tab = (int32_t*) take(tab_ints(T) * 4);
    n.ident = (int32_t*) take((size_t) T * 4);
    for (int i = 0; ok && i < kPfSlots; ++i) ok = step(hipHostMalloc((void**) &n.stage[i], tab_ints(T) * 4, hipHostMallocDefault), "staging");
    if (ok) {
        mmq::iota(n.ident, T, p.s);
        ok = step(hipStreamSynchronize(p.s), "prompt stream");   // the old set is idle after it too
    }
    if (!ok) {
        why += " (the prompt buffers for " + std::to_string(T) + " tokens, " + std::to_string(total >> 20) + " MiB on the card)";
        prompt_free(n);
        return -1;
    }
    n.T = T;
    n.gu = gu;
    n.d = d;
    prompt_free(p);
    static_cast<PromptBufs&>(p) = std::move(n);
    return 0;
}

// Whether a prompt call is something the kernels can run, with the reason in why when it is not. A bad address or row would fault the card.
bool prefill_ok(Tier* t, const strata_xtier_prefill_args* a, std::string& why) {
    const auto no = [&why](const std::string& s) {
        why = "prefill_layer: " + s;
        return false;
    };
    Prompt& p = t->pf;
    if (a == nullptr) return no("no call given");
    if (p.T == 0) return no("the prompt buffers are not open");
    const int64_t T = a->T, n = a->n_experts, ng = a->n_groups, rows = a->rows_peer;
    if (T < 1 || T > p.T) return no(std::to_string(T) + " tokens do not fit the buffers of " + std::to_string(p.T));
    if (a->N != kPfEmbd || a->K < 1 || a->K > kPfMaxTop) return no("N " + std::to_string(a->N) + " or K " + std::to_string(a->K) + " is not one the library takes");
    if (a->x_ready == nullptr || a->done == nullptr || a->wait_err == nullptr || a->x_hip == nullptr || a->w_hip == nullptr ||
        a->sum_host == nullptr)
        return no("a flag or a shared buffer is missing");
    const strata_xtier_fmt& f = a->fmt;
    if (f.n_embd != kPfEmbd || f.n_ff != kPfFf || f.gu_type != a->mmq_gu_type || f.d_type != a->mmq_d_type)
        return no("the expert format is not the prompt path's");
    if (!strata::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) return no("the expert format is not supported");
    const auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
    if (L.tail_off != 0) return no("NVFP4 experts are not supported");
    if (L.up_off != f.up_off || L.down_off != f.down_off || L.bytes != f.blob_bytes) return no("the engine's blob layout is not the library's");
    if (a->gu_bytes != mmq::matrix_bytes(f.gu_type, 2 * kPfFf, kPfEmbd) || a->d_bytes != mmq::matrix_bytes(f.d_type, kPfEmbd, kPfFf) ||
        a->gu_bytes > p.gu || a->d_bytes > p.d || f.up_off + a->gu_bytes / 2 > f.blob_bytes || f.down_off + a->d_bytes > f.blob_bytes)
        return no("the matrices are not the blob's or do not fit the group buffers");
    if (p.fit_gu != f.gu_type || p.fit_d != f.d_type) {   // MMQ has a tile for both products on this card
        if (!mmq::fits(f.gu_type, 2 * kPfFf) || !mmq::fits(f.d_type, kPfEmbd)) return no("MMQ has no tile for this expert format on the AMD card");
        p.fit_gu = f.gu_type;
        p.fit_d = f.d_type;
    }
    if (rows < 0 || rows > T * a->K || n < 0 || ng < 0 || ng > n || (n > 0 && ng == 0)) return no("the counts are out of range");
    if (a->n_bounds != n + 1 + ng * (mmq::kGroup + 1) || a->n_bounds > kPfMaxBounds || a->bounds == nullptr)
        return no("the bounds do not match the groups");
    if (a->n_adds < 0 || (size_t) (2 * rows + a->n_bounds + a->n_adds) > tab_ints(T)) return no("the tables are larger than the library holds");
    if (n > 0 && (a->expert_addr == nullptr || a->expert_rows == nullptr || a->group_first == nullptr || a->group_last == nullptr ||
                  a->group_adds_at == nullptr || a->group_ntok == nullptr || a->adds == nullptr))
        return no("a table is missing");
    if (rows > 0 && (a->pair == nullptr || a->src == nullptr)) return no("a table is missing");
    const int32_t* b = a->bounds;
    if (b[0] != 0 || b[n] != rows) return no("the bounds do not cover the rows");
    const uint64_t lo = (uint64_t) (uintptr_t) t->arena;
    for (int64_t j = 0; j < n; ++j) {
        const uint64_t q = a->expert_addr[j];
        if (q < lo || q - lo > t->arena_bytes || t->arena_bytes - (q - lo) < L.bytes) return no("expert " + std::to_string(j) + " is not inside the arena");
        if ((q - lo) % 256 != 0) return no("expert " + std::to_string(j) + " is not 256 byte aligned in the arena");
        if (a->expert_rows[j] < 1 || b[j + 1] - b[j] != a->expert_rows[j]) return no("expert " + std::to_string(j) + "'s rows are not its bounds'");
    }
    for (int64_t g = 0; g < ng; ++g) {
        const int64_t j0 = a->group_first[g], j1 = a->group_last[g];
        if (j0 != (g == 0 ? 0 : a->group_last[g - 1]) || j1 <= j0 || j1 - j0 > mmq::kGroup || j1 > n || (g + 1 == ng && j1 != n))
            return no("group " + std::to_string(g) + " does not follow the one before");
        const int32_t r0 = b[j0], r1 = b[j1];
        if (r1 - r0 > p.T) return no("group " + std::to_string(g) + " has more rows than the buffers hold");
        const int32_t* rel = b + n + 1 + g * (mmq::kGroup + 1);
        for (int64_t i = 0; i <= mmq::kGroup; ++i)
            if (rel[i] != b[std::min(j1, j0 + i)] - r0) return no("group " + std::to_string(g) + "'s bounds are not its experts'");
        const int64_t at = a->group_adds_at[g], nt = a->group_ntok[g];
        if (at < 0 || nt < 0 || nt > r1 - r0 || at + 2 * nt + 1 + (r1 - r0) > a->n_adds) return no("group " + std::to_string(g) + "'s sums table is out of range");
        const int32_t* ad = a->adds + at;
        if (ad[nt] != 0 || ad[2 * nt] != r1 - r0) return no("group " + std::to_string(g) + "'s sums table does not cover its rows");
        for (int64_t i = 0; i < nt; ++i)
            if (ad[i] < 0 || ad[i] >= T || ad[nt + 1 + i] < ad[nt + i]) return no("group " + std::to_string(g) + "'s sums table is out of range");
        for (int32_t i = 0; i < r1 - r0; ++i)
            if (ad[2 * nt + 1 + i] < r0 || ad[2 * nt + 1 + i] >= r1) return no("group " + std::to_string(g) + "'s sums table names another group's row");
    }
    for (int64_t r = 0; r < rows; ++r)
        if (a->pair[r] < 0 || a->pair[r] >= T * a->K || a->src[r] < 0 || a->src[r] >= T)
            return no("row " + std::to_string(r) + " names a token the call does not have");
    return true;
}

int pf_fail(std::string& why, const char* what, hipError_t e) {
    why = std::string("prefill_layer: ") + what + ": " + hipGetErrorString(e);
    return (int) e;
}

// STRATA_XTIER_TIMING=1: sums the prompt layers whose events have run, and prints the means every 45 layers.
void pf_harvest(Prompt& p) {
    float ms = 0;
    while (!p.tq.empty() && hipEventQuery(p.tq.front().e[4]) == hipSuccess) {
        Prompt::Tm m = p.tq.front();
        p.tq.erase(p.tq.begin());
        for (int i = 0; i < 4; ++i)
            if (hipEventElapsedTime(&ms, m.e[i], m.e[i + 1]) == hipSuccess) p.tms[i + 1] += ms;
        if (p.thave && hipEventElapsedTime(&ms, p.tlast.e[4], m.e[0]) == hipSuccess) p.tms[0] += ms;
        if (p.thave) p.tfree.push_back(p.tlast);
        p.tlast = m;
        p.thave = true;
        p.thost += m.host;
        p.tlock += m.lock;
        p.tslot += m.slot;
        p.trows += m.rows;
        p.texp += m.experts;
        if (++p.tn < 45) continue;
        const double n = (double) p.tn;
        std::fprintf(stderr, "strata xtier prompt timing: %lld layers, %.0f rows and %.1f experts each; ms per layer: card idle %.2f, "
                             "waits for x %.2f, in %.2f, kernels %.2f, out %.2f; host in the call %.2f (pf_mu %.2f, slot %.2f)\n",
                     (long long) p.tn, p.trows / n, p.texp / n, p.tms[0] / n, p.tms[1] / n, p.tms[2] / n, p.tms[3] / n,
                     p.tms[4] / n, p.thost / n, p.tlock / n, p.tslot / n);
        std::fill(p.tms, p.tms + 5, 0.0);
        p.thost = p.tlock = p.tslot = 0;
        p.trows = p.texp = p.tn = 0;
    }
}

// Queues one layer's prompt rows on the prompt stream, with pf_mu held: prefill.cpp's compact SUMS route, the peer's half.
int prefill_enqueue(Tier* t, const strata_xtier_prefill_args* a, std::string& why, Prompt::Tm* tm) {
    (void) hipGetLastError();
    hipError_t e = hipSetDevice(t->device);
    if (e != hipSuccess) return pf_fail(why, "hipSetDevice", e);
    if (!prefill_ok(t, a, why)) return -1;
    namespace k = strata::kernels;
    namespace pk = strata::prefill;
    Prompt& p = t->pf;
    const hipStream_t s = p.s;
    const int64_t T = a->T, N = a->N, n = a->n_experts, rows = a->rows_peer;
    // the tables go up from a pinned slot, free again once the upload of the call that used it last has run
    const int slot = (int) (p.calls % kPfSlots);
    const auto ts0 = std::chrono::steady_clock::now();
    if ((e = hipEventSynchronize(p.staged[slot])) != hipSuccess) return pf_fail(why, "staging", e);
    if (tm) tm->slot = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts0).count();
    if (tm) hipEventRecord(tm->e[0], s);
    int32_t* st = p.stage[slot];
    size_t at = 0;
    const std::pair<const int32_t*, int64_t> parts[] = {{a->pair, rows}, {a->src, rows}, {a->bounds, a->n_bounds}, {a->adds, a->n_adds}};
    for (const auto& [v, c] : parts) {
        if (c > 0) std::memcpy(st + at, v, (size_t) c * 4);
        at += (size_t) c;
    }
    const int32_t *d_pair = p.tab, *d_src = p.tab + rows, *d_bounds = d_src + rows, *d_adds = d_bounds + a->n_bounds;
    k::wait_handoff(a->x_ready, a->seq, a->wait_err, a->timeout_ns, s);
    if (tm) hipEventRecord(tm->e[1], s);
    if (a->f16) pk::f16_to_f32_wide(p.x, (const uint16_t*) a->x_hip, T * N, s);
    else pk::copy_f32_wide(p.x, (const float*) a->x_hip, T * N, s);
    pk::copy_f32_wide(p.w, a->w_hip, T * a->K, s);
    if ((e = hipMemcpyAsync(p.tab, st, at * 4, hipMemcpyHostToDevice, s)) != hipSuccess) return pf_fail(why, "tables", e);
    if ((e = hipEventRecord(p.staged[slot], s)) != hipSuccess) return pf_fail(why, "hipEventRecord", e);
    ++p.calls;
    if ((e = hipMemsetAsync(p.sum, 0, (size_t) (T * N) * 4, s)) != hipSuccess) return pf_fail(why, "hipMemsetAsync", e);
    if (tm) hipEventRecord(tm->e[2], s);
    const strata_xtier_fmt& f = a->fmt;
    const size_t gub = a->gu_bytes, db = a->d_bytes;
    for (int64_t g = 0; g < a->n_groups; ++g) {
        const int64_t j0 = a->group_first[g], j1 = a->group_last[g];
        const int ngx = (int) (j1 - j0);
        int64_t maxr = 0;
        for (int64_t j = j0; j < j1; ++j) {
            maxr = std::max<int64_t>(maxr, a->expert_rows[j]);
            const uint8_t* bd = (const uint8_t*) (uintptr_t) a->expert_addr[j];
            const size_t q = (size_t) (j - j0);
            mmq::gather_native(bd, bd + f.up_off, gub / 2, bd + f.down_off, db, p.grp_gu + q * gub, p.grp_d + q * db, s);
        }
        const int64_t r0 = a->bounds[j0], nr = a->bounds[j1] - r0;
        if (nr <= 0) continue;
        const int32_t* rel = d_bounds + n + 1 + g * (mmq::kGroup + 1);
        if ((e = hipMemsetAsync(p.grp_gu + (size_t) ngx * gub, 0, mmq::kTail, s)) != hipSuccess ||
            (e = hipMemsetAsync(p.grp_d + (size_t) ngx * db, 0, mmq::kTail, s)) != hipSuccess)
            return pf_fail(why, "hipMemsetAsync", e);
        mmq::quantize(p.x, d_src + r0, p.xq, f.gu_type, N, N, nr, s);
        mmq::Product gu;
        gu.w = p.grp_gu; gu.type = f.gu_type; gu.w_rows = 2 * kPfFf; gu.w_cols = N; gu.expert_bytes = gub;
        gu.n = ngx; gu.xq = p.xq; gu.bounds = rel; gu.ids = p.ident;
        gu.total_rows = nr; gu.max_rows = maxr; gu.dst = p.gu_rows; gu.ld_dst = 2 * kPfFf;
        p.ctx->run(gu, s);
        mmq::swiglu(p.gu_rows, p.h, nr, kPfFf, false, s);
        mmq::quantize(p.h, nullptr, p.hq, f.d_type, kPfFf, kPfFf, nr, s);
        mmq::Product dn;
        dn.w = p.grp_d; dn.type = f.d_type; dn.w_rows = N; dn.w_cols = kPfFf; dn.expert_bytes = db;
        dn.n = ngx; dn.xq = p.hq; dn.bounds = rel; dn.ids = p.ident;
        dn.total_rows = nr; dn.max_rows = maxr; dn.dst = p.dm; dn.ld_dst = N;
        p.ctx->run(dn, s);
        const int32_t* ad = d_adds + a->group_adds_at[g];
        const int64_t nt = a->group_ntok[g];
        pk::peer_gather_add(p.sum, p.dm, r0, p.w, d_pair, ad, ad + nt, ad + 2 * nt + 1, nt, s);
    }
    if (tm) hipEventRecord(tm->e[3], s);
    if (a->f16) {
        pk::sums_to_f16(p.sum, p.sum16, T * N, s);
        e = hipMemcpyAsync(a->sum_host, p.sum16, (size_t) (T * N) * 2, hipMemcpyDeviceToHost, s);
    } else {
        e = hipMemcpyAsync(a->sum_host, p.sum, (size_t) (T * N) * 4, hipMemcpyDeviceToHost, s);
    }
    if (e != hipSuccess) return pf_fail(why, "sums", e);
    if (t->pf_drop == 0 || p.calls != t->pf_drop) k::raise_flag(a->done, a->seq, s);
    if (tm) {
        hipEventRecord(tm->e[4], s);
        tm->rows = rows;
        tm->experts = n;
    }
    // the card idles between prompt layers: a 16K prompt without this ran 70 of its 122 busy samples at the 500 MHz idle clock
    if (t->keepwarm && hipEventRecord(p.done, s) == hipSuccess) {
        std::unique_lock<std::mutex> lk(t->mu, std::try_to_lock);   // the decode lock may be held across a card wait: then no spin this time
        if (lk.owns_lock()) keep_warm(t, p.done);
    }
    return 0;
}

}  // namespace

STRATA_XTIER_EXPORT int strata_xtier_abi(void) {
    return STRATA_XTIER_ABI_VERSION * 100000 + strata_xtier_structs_size();
}

STRATA_XTIER_EXPORT void* strata_xtier_open(int hip_device, char* err, size_t err_len) {
    (void) hipGetLastError();
    int n = 0;
    const hipError_t c = hipGetDeviceCount(&n);
    if (c == hipErrorNoDevice || (c == hipSuccess && n == 0)) {
        put(err, err_len, "no AMD GPU");
        return nullptr;
    }
    if (c != hipSuccess) {
        put(err, err_len, std::string("no AMD GPU (hipGetDeviceCount: ") + hipGetErrorString(c) + ")");
        return nullptr;
    }
    if (hip_device < 0 || hip_device >= n) {
        put(err, err_len, "AMD card " + std::to_string(hip_device) + " does not exist, the system has " + std::to_string(n));
        return nullptr;
    }
    const hipError_t s = hipSetDevice(hip_device);
    if (s != hipSuccess) {
        put(err, err_len, std::string("hipSetDevice: ") + hipGetErrorString(s));
        return nullptr;
    }
    // the first runtime call that needs the card creates its context here, not in the middle of a decode
    const hipError_t f = hipFree(nullptr);
    if (f != hipSuccess) {
        put(err, err_len, std::string("AMD card ") + std::to_string(hip_device) + " did not start: " + hipGetErrorString(f));
        return nullptr;
    }
    Tier* t = new Tier();
    t->device = hip_device;
    // everything a decode needs is allocated here, sized for the most entries a call takes, so a decode never allocates
    constexpr size_t kRows = (size_t) STRATA_XTIER_CAP * STRATA_XTIER_MAX_EMBD * sizeof(float);
    const unsigned host_flags = hipHostMallocMapped | hipHostMallocPortable;
    const auto step = [&](hipError_t e, const char* what) {
        if (e == hipSuccess) return true;
        put(err, err_len, "AMD card " + std::to_string(hip_device) + " did not start: " + what + ": " + hipGetErrorString(e));
        return false;
    };
    const char* tv = std::getenv("STRATA_XTIER_TIMING");
    t->timing = tv != nullptr && std::atoi(tv) != 0;
    bool ok =
        step(hipStreamCreateWithFlags(&t->decode, hipStreamNonBlocking), "decode stream") &&
        step(hipStreamCreateWithFlags(&t->refill, hipStreamNonBlocking), "refill stream") &&
        step(hipEventCreateWithFlags(&t->refill_ev, hipEventDisableTiming), "refill event") &&
        step(hipEventCreateWithFlags(&t->done_ev, hipEventDisableTiming), "decode event") &&
        step(hipHostMalloc((void**) &t->h_x, kRows, host_flags), "input staging") &&
        step(hipHostMalloc((void**) &t->h_out, kRows, host_flags), "result staging") &&
        step(hipHostMalloc((void**) &t->h_meta, sizeof(strata_xtier_meta), host_flags), "plan staging") &&
        step(hipMalloc((void**) &t->d_x, kRows), "input") &&
        step(hipMalloc((void**) &t->d_out, kRows), "result") &&
        step(hipMalloc((void**) &t->d_meta, sizeof(strata_xtier_meta)), "plan") &&
        step(hipMalloc(&t->d_q8, (size_t) STRATA_XTIER_CAP * (STRATA_XTIER_MAX_EMBD / 32) * 36), "activations") &&
        step(hipMalloc(&t->d_scratch, strata::kernels::native_expert_scratch_bytes(STRATA_XTIER_CAP, STRATA_XTIER_MAX_FF)), "scratch");
    for (int i = 0; ok && t->timing && i < 4; ++i) ok = step(hipEventCreate(&t->tev[i]), "timing event");
    // the card drops its clock when it idles between decodes, which made each layer's kernels about 3x as long
    const char* kv = std::getenv("STRATA_XTIER_KEEPWARM");
    t->keepwarm = kv == nullptr || std::atoi(kv) != 0;
    if (ok && t->keepwarm) {   // without its stream or flag the tier still opens, only without keep-warm
        const char* what = "warm stream";
        hipError_t w = hipStreamCreateWithFlags(&t->warm, hipStreamNonBlocking);
        if (w == hipSuccess) what = "warm flag";
        if (w == hipSuccess) w = hipHostMalloc((void**) &t->wflag, 64, host_flags | hipHostMallocCoherent);
        if (w == hipSuccess) w = hipHostGetDevicePointer((void**) &t->wflag_dev, t->wflag, 0);
        if (w == hipSuccess) {
            t->wflag[0] = t->wflag[1] = 0;
        } else {
            std::fprintf(stderr, "strata xtier: keep-warm is off, the %s did not start: %s\n", what, hipGetErrorString(w));
            (void) hipGetLastError();
            if (t->wflag != nullptr) (void) hipHostFree(t->wflag);
            if (t->warm != nullptr) (void) hipStreamDestroy(t->warm);
            t->wflag = t->wflag_dev = nullptr;
            t->warm = nullptr;
            t->keepwarm = false;
        }
    }
    const char* dv = std::getenv("STRATA_TEST_XTIER_DROP");
    t->pf_drop = dv != nullptr ? std::strtoull(dv, nullptr, 10) : 0;
    if (!ok) {
        release(t);
        delete t;
        return nullptr;
    }
    return t;
}

STRATA_XTIER_EXPORT void strata_xtier_close(void* h) {
    Tier* t = static_cast<Tier*>(h);
    if (t == nullptr) return;
    {
        const Enter in(t);
        if (in.ok) release(t);
    }
    delete t;
}

STRATA_XTIER_EXPORT int strata_xtier_mem_info(void* h, uint64_t* free_b, uint64_t* total_b) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return -1;
    size_t f = 0, tot = 0;
    const hipError_t e = hipMemGetInfo(&f, &tot);
    if (e != hipSuccess) return fail(t, "hipMemGetInfo", e);
    *free_b = f;
    *total_b = tot;
    return 0;
}

STRATA_XTIER_EXPORT int strata_xtier_register(void* h, void* host, uint64_t bytes, void** hip_dev) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return -1;
    if (host == nullptr || bytes == 0 || hip_dev == nullptr) {
        t->err = "register: no memory given";
        return -1;
    }
    hipError_t e = hipHostRegister(host, (size_t) bytes, hipHostRegisterMapped | hipHostRegisterPortable);
    if (e != hipSuccess) return fail(t, "hipHostRegister", e);
    e = hipHostGetDevicePointer(hip_dev, host, 0);
    if (e != hipSuccess) {
        (void) hipHostUnregister(host);
        return fail(t, "hipHostGetDevicePointer", e);
    }
    return 0;
}

STRATA_XTIER_EXPORT int strata_xtier_unregister(void* h, void* host) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return -1;
    const hipError_t e = hipHostUnregister(host);
    return e == hipSuccess ? 0 : fail(t, "hipHostUnregister", e);
}

STRATA_XTIER_EXPORT const char* strata_xtier_last_error(void* h) {
    if (h == nullptr) return "";
    Tier* t = static_cast<Tier*>(h);
    thread_local std::string copy;   // the caller reads it after the lock is released, when another thread may change err
    const std::lock_guard<std::mutex> lk(t->mu);
    copy = t->err;
    return copy.c_str();
}

STRATA_XTIER_EXPORT int strata_xtier_flag_echo(void* h, const uint32_t* in_hip, uint32_t* out_hip, uint32_t* err_hip, uint32_t seq,
                                                uint64_t timeout_ns) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return -1;
    if (in_hip == nullptr || out_hip == nullptr || err_hip == nullptr) {
        t->err = "flag_echo: no flag given";
        return -1;
    }
    strata::kernels::wait_handoff(in_hip, seq, err_hip, timeout_ns, t->decode);
    strata::kernels::raise_flag(out_hip, seq, t->decode);
    return 0;
}

STRATA_XTIER_EXPORT int strata_xtier_sync(void* h) {
    Tier* t = static_cast<Tier*>(h);
    hipStream_t pf = nullptr;
    if (t != nullptr) {   // taken alone, so the wait below never holds both locks
        const std::lock_guard<std::mutex> lk(t->pf_mu);
        pf = t->pf.s;
    }
    const Enter in(t);
    if (!in.ok) return -1;
    hipError_t e = hipStreamSynchronize(t->decode);
    if (e == hipSuccess && pf != nullptr) e = hipStreamSynchronize(pf);
    return e == hipSuccess ? 0 : fail(t, "hipStreamSynchronize", e);
}

STRATA_XTIER_EXPORT void* strata_xtier_host_alloc(void* h, uint64_t bytes, int coherent) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return nullptr;
    void* p = nullptr;
    const unsigned flags = hipHostMallocMapped | hipHostMallocPortable | (coherent != 0 ? hipHostMallocCoherent : 0u);
    const hipError_t e = hipHostMalloc(&p, (size_t) bytes, flags);
    if (e != hipSuccess) {
        (void) fail(t, "hipHostMalloc", e);
        return nullptr;
    }
    return p;
}

STRATA_XTIER_EXPORT void strata_xtier_host_free(void* h, void* host) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (in.ok && host != nullptr) (void) hipHostFree(host);
}

STRATA_XTIER_EXPORT int strata_xtier_probe(void* h, int op, void* host, uint64_t bytes, uint32_t seed, uint64_t* result, float* ms) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return -1;
    if (host == nullptr || bytes < 4 || result == nullptr || ms == nullptr) {
        t->err = "probe: no memory given";
        return -1;
    }
    *result = 0;
    *ms = 0.f;
    if (op == 0 || op == 1) return probe_pattern(t, op == 0, host, bytes, seed, result);
    if (op == 2) return probe_copy(t, host, bytes, ms);
    t->err = "probe: unknown op " + std::to_string(op);
    return -1;
}

STRATA_XTIER_EXPORT int strata_xtier_arena(void* h, const uint64_t* slot_bytes, int64_t n, uint64_t* slot_addr) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return -1;
    if (slot_bytes == nullptr || slot_addr == nullptr || n < 1) {
        t->err = "arena: no slots given";
        return -1;
    }
    if (t->arena != nullptr) {
        t->err = "arena: the tier already has one";
        return -1;
    }
    std::vector<uint64_t> off((size_t) n), len((size_t) n);
    uint64_t total = 0;
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t b = slot_bytes[i];
        const uint64_t padded = (b + 255) / 256 * 256;
        if (b == 0 || b > (1ull << 48) || total + padded > (1ull << 48)) {
            t->err = "arena: slot " + std::to_string(i) + " has a size of " + std::to_string(b) + " bytes";
            return -1;
        }
        off[(size_t) i] = total;
        len[(size_t) i] = b;
        total += padded;
    }
    void* p = nullptr;
    const hipError_t e = hipMalloc(&p, (size_t) total);
    if (e != hipSuccess) return fail(t, ("hipMalloc of the arena (" + std::to_string(total >> 20) + " MiB)").c_str(), e);
    t->arena = static_cast<uint8_t*>(p);
    t->arena_bytes = total;
    t->slot_off = std::move(off);
    t->slot_len = std::move(len);
    for (int64_t i = 0; i < n; ++i) slot_addr[i] = (uint64_t) (uintptr_t) (t->arena + t->slot_off[(size_t) i]);
    return 0;
}

STRATA_XTIER_EXPORT int strata_xtier_fill(void* h, int64_t slot, const void* host, uint64_t bytes) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return -1;
    uint8_t* dst = slot_ptr(t, "fill", slot, bytes);
    if (dst == nullptr) return -1;
    if (host == nullptr) {
        t->err = "fill: no memory given";
        return -1;
    }
    // HIP copies unpinned host memory before hipMemcpyAsync returns, which is what keeps the engine's short lived blob() buffers safe here.
    const hipError_t e = hipMemcpyAsync(dst, host, (size_t) bytes, hipMemcpyHostToDevice, t->refill);
    return e == hipSuccess ? 0 : fail(t, "hipMemcpyAsync", e);
}

STRATA_XTIER_EXPORT int strata_xtier_read(void* h, int64_t slot, void* host, uint64_t bytes) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return -1;
    const uint8_t* src = slot_ptr(t, "read", slot, bytes);
    if (src == nullptr) return -1;
    if (host == nullptr) {
        t->err = "read: no memory given";
        return -1;
    }
    hipError_t e = hipMemcpyAsync(host, src, (size_t) bytes, hipMemcpyDeviceToHost, t->refill);
    if (e == hipSuccess) e = hipStreamSynchronize(t->refill);
    return e == hipSuccess ? 0 : fail(t, "read", e);
}

STRATA_XTIER_EXPORT int strata_xtier_refill_record(void* h) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return -1;
    const hipError_t e = hipEventRecord(t->refill_ev, t->refill);
    return e == hipSuccess ? 0 : fail(t, "hipEventRecord", e);
}

STRATA_XTIER_EXPORT int strata_xtier_refill_done(void* h, int wait) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return -1;
    const hipError_t e = wait != 0 ? hipEventSynchronize(t->refill_ev) : hipEventQuery(t->refill_ev);
    if (e == hipSuccess) return 1;
    if (e == hipErrorNotReady) return 0;
    (void) fail(t, "refill", e);
    return -1;
}

STRATA_XTIER_EXPORT int strata_xtier_decode(void* h, const strata_xtier_fmt* f, const float* x, int64_t n_tok, const strata_xtier_meta* m) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return -1;
    if (!decode_ok(t, f, x, n_tok, m)) return -1;
    const int32_t groups = m->count[0], rows = m->count[1];
    if (rows == 0) {   // nothing to compute, and decode_wait has nothing to wait for
        t->pending = true;
        t->queued = false;
        return 0;
    }
    try {
        std::memcpy(t->h_x, x, (size_t) n_tok * (size_t) f->n_embd * sizeof(float));
        std::memcpy(t->h_meta, m, sizeof(*m));
        const hipStream_t s = t->decode;
        if (t->timing) (void) hipEventRecord(t->tev[0], s);
        hipError_t e = hipMemcpyAsync(t->d_x, t->h_x, (size_t) n_tok * (size_t) f->n_embd * sizeof(float), hipMemcpyHostToDevice, s);
        if (e != hipSuccess) return fail(t, "input", e);
        e = hipMemcpyAsync(t->d_meta, t->h_meta, sizeof(*m), hipMemcpyHostToDevice, s);
        if (e != hipSuccess) return fail(t, "plan", e);
        if (t->timing) (void) hipEventRecord(t->tev[1], s);
        strata_xtier_meta* dm = t->d_meta;
        namespace k = strata::kernels;
        k::quantize_q8_1_rows(t->d_x, n_tok, f->n_embd, t->d_q8, s);
        k::native_expert_grouped(k::native_expert_layout(f->gu_type, f->d_type, f->n_embd, f->n_ff), dm->ptr, dm->start, dm->count, dm->dst,
                                 dm->tok, groups, rows, t->d_q8, t->d_scratch, t->d_out, s);
        if (t->timing) (void) hipEventRecord(t->tev[2], s);
        e = hipMemcpyAsync(t->h_out, t->d_out, (size_t) rows * (size_t) f->n_embd * sizeof(float), hipMemcpyDeviceToHost, s);
        if (e != hipSuccess) return fail(t, "results", e);
        if (t->timing) (void) hipEventRecord(t->tev[3], s);
        e = hipEventRecord(t->done_ev, s);
        if (e != hipSuccess) return fail(t, "hipEventRecord", e);
        if (t->keepwarm) keep_warm(t);
    } catch (const std::exception& ex) {
        t->err = std::string("decode: ") + ex.what();
        return -1;
    }
    t->pending = true;
    t->queued = true;
    return 0;
}

STRATA_XTIER_EXPORT int strata_xtier_decode_wait(void* h, const float** rows, uint64_t timeout_ns) {
    Tier* t = static_cast<Tier*>(h);
    Enter in(t);
    if (!in.ok) return -1;
    if (rows == nullptr) {
        t->err = "decode_wait: no place for the rows given";
        return -1;
    }
    if (!t->pending) {
        t->err = "decode_wait: no decode is waiting";
        return -1;
    }
    if (t->queued) {
        // a spin, since a blocking wait would sleep this thread and wake it late, and without the lock so a fill is not held up
        in.lk.unlock();
        const auto t0 = std::chrono::steady_clock::now();
        hipError_t e = hipErrorNotReady;
        bool late = false;
        while ((e = hipEventQuery(t->done_ev)) == hipErrorNotReady) {
            const auto waited = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0);
            if (timeout_ns != 0 && (uint64_t) waited.count() >= timeout_ns) {
                late = true;
                break;
            }
        }
        in.lk.lock();
        if (late) {
            t->err = "decode_wait: the card did not finish within " + std::to_string(timeout_ns / 1000000) + " ms";
            return STRATA_XTIER_TIMEOUT;
        }
        if (e != hipSuccess) return fail(t, "decode", e);
        float a = 0.f, b = 0.f, c = 0.f;
        if (t->timing && hipEventElapsedTime(&a, t->tev[0], t->tev[1]) == hipSuccess &&
            hipEventElapsedTime(&b, t->tev[1], t->tev[2]) == hipSuccess && hipEventElapsedTime(&c, t->tev[2], t->tev[3]) == hipSuccess) {
            t->tms[0] += a;
            t->tms[1] += b;
            t->tms[2] += c;
            ++t->tn;
        }
    }
    t->pending = false;
    t->queued = false;
    *rows = t->h_out;
    return 0;
}

STRATA_XTIER_EXPORT int strata_xtier_timing(void* h, double* ms) {
    Tier* t = static_cast<Tier*>(h);
    const Enter in(t);
    if (!in.ok) return -1;
    if (ms == nullptr) {
        t->err = "timing: no place for the times given";
        return -1;
    }
    for (int i = 0; i < 3; ++i) {
        ms[i] = t->tms[i];
        t->tms[i] = 0;
    }
    const int n = t->tn;
    t->tn = 0;
    return n;
}

STRATA_XTIER_EXPORT int strata_xtier_prefill_open(void* h, int64_t T_max, int64_t N, uint64_t gu_max, uint64_t d_max) {
    Tier* t = static_cast<Tier*>(h);
    if (t == nullptr) return -1;
    std::string why;
    int r = -1;
    try {
        const std::lock_guard<std::mutex> lk(t->pf_mu);
        r = prompt_open(t, T_max, N, gu_max, d_max, why);
    } catch (const std::exception& ex) {
        why = std::string("prefill_open: ") + ex.what();
    }
    if (r != 0) {   // the decode lock only after the prompt lock is free
        const std::lock_guard<std::mutex> lk(t->mu);
        t->err = why;
    }
    return r;
}

STRATA_XTIER_EXPORT int strata_xtier_prefill_layer(void* h, const strata_xtier_prefill_args* a) {
    Tier* t = static_cast<Tier*>(h);
    if (t == nullptr) return -1;
    std::string why;
    int r = -1;
    try {
        const auto t0 = std::chrono::steady_clock::now();
        const std::lock_guard<std::mutex> lk(t->pf_mu);
        const auto t1 = std::chrono::steady_clock::now();
        Prompt& p = t->pf;
        Prompt::Tm tm;
        bool timed = false;
        if (t->timing) {   // STRATA_XTIER_TIMING=1: this layer's events, from a spare set or new ones
            if (!p.tfree.empty()) {
                tm = p.tfree.back();
                p.tfree.pop_back();
                timed = true;
            } else {
                timed = true;
                for (hipEvent_t& ev : tm.e) timed = timed && hipEventCreate(&ev) == hipSuccess;
            }
        }
        r = prefill_enqueue(t, a, why, timed ? &tm : nullptr);
        if (timed && r == 0) {
            using ms = std::chrono::duration<double, std::milli>;
            tm.lock = ms(t1 - t0).count();
            tm.host = ms(std::chrono::steady_clock::now() - t0).count();
            p.tq.push_back(tm);
            pf_harvest(p);
        }
    } catch (const std::exception& ex) {
        why = std::string("prefill_layer: ") + ex.what();
    }
    if (r != 0) {
        const std::lock_guard<std::mutex> lk(t->mu);
        t->err = why;
    }
    return r;
}
