// src/core/dflash2_window_test.cpp - the DFlash2 drafter's window save and restore (GPU, no model).
//
//     dflash2_window_test <pack-or-gguf>
//
// Seeded random taps, head rows and embedding stand in for the target (no torch, no case file).  Checks: a restored window drafts like the
// live ring (a), (b), a window of another position is refused (c), its size (d), a ring that wrapped holds the last 2048 rows (e), the
// time of 2047 context rows (f) and that clear_ring leaves no stale row (g).
#include "strata/core/dflash2.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using strata::core::Dflash2Drafter;

namespace {

constexpr int H = 2560, TAPN = 5 * H, VP = 20000, ANCHOR = 123456;

uint16_t to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int e = (int) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;   // the scales here are normal numbers, flush the rest
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    if ((m & 0x1fffu) > 0x1000u || ((m & 0x1fffu) == 0x1000u && (h & 1))) ++h;
    return (uint16_t) h;
}

std::vector<float> randn(size_t n, uint32_t seed, float scale) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = nd(g);
    return v;
}

// Q8_0 rows (type 8): per 32 values a half scale then 32 int8
std::vector<uint8_t> quant_q8_0(const std::vector<float>& w, int rows, int cols) {
    std::vector<uint8_t> out((size_t) rows * (cols / 32) * 34);
    uint8_t* o = out.data();
    for (size_t b = 0; b < (size_t) rows * (cols / 32); ++b, o += 34) {
        const float* x = w.data() + b * 32;
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
        const uint16_t d = to_half(amax / 127.0f);
        std::memcpy(o, &d, 2);
        const float inv = amax > 0 ? 127.0f / amax : 0.0f;
        for (int i = 0; i < 32; ++i) o[2 + i] = (uint8_t) (int8_t) std::lrintf(x[i] * inv);
    }
    return out;
}

struct Dev {
    float* p = nullptr;
    explicit Dev(const std::vector<float>& v) {
        cudaMalloc((void**) &p, v.size() * 4);
        cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
    }
    Dev(const Dev&) = delete;
    ~Dev() { cudaFree(p); }
};

struct Answer {
    int32_t tok[7];
    float prob[7];
};

int g_bad = 0;
void expect(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_bad;
}

bool same(const Answer& a, const Answer& b, const char* what) {
    bool tok = true;
    float dp = 0;
    for (int i = 0; i < 7; ++i) {
        tok = tok && a.tok[i] == b.tok[i];
        dp = std::max(dp, std::fabs(a.prob[i] - b.prob[i]));
    }
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s (tokens %s, max |dp| %.2e)", what, tok ? "equal" : "DIFFER", dp);
    expect(tok && dp < 2e-3f, buf);
    return tok && dp < 2e-3f;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: dflash2_window_test <pack-or-gguf>\n"); return 2; }
    const std::vector<float> emb = randn(2 * H, 7, 1.0f);
    std::vector<float> headf = randn((size_t) VP * H, 8, 0.05f);
    const std::vector<uint8_t> head = quant_q8_0(headf, VP, H);
    std::vector<int32_t> dvocab;
    {
        std::mt19937 g(9);
        std::vector<int32_t> all(248320);
        for (int i = 0; i < (int) all.size(); ++i) all[i] = i;
        std::shuffle(all.begin(), all.end(), g);
        dvocab.assign(all.begin(), all.begin() + VP);
        std::sort(dvocab.begin(), dvocab.end());
    }
    Dev d_emb(emb);
    uint8_t* d_head = nullptr;
    cudaMalloc((void**) &d_head, head.size());
    cudaMemcpy(d_head, head.data(), head.size(), cudaMemcpyHostToDevice);
    float* d_last = nullptr;   // the one taps buffer every draft() reads (its graphs are captured on it)
    cudaMalloc((void**) &d_last, (size_t) TAPN * 4);
    auto embed = [&](const int32_t*, int n, float* out, cudaStream_t cs) {
        cudaMemcpyAsync(out, d_emb.p, (size_t) H * 4, cudaMemcpyDeviceToDevice, cs);
        for (int i = 1; i < n; ++i) cudaMemcpyAsync(out + (size_t) i * H, d_emb.p + H, (size_t) H * 4, cudaMemcpyDeviceToDevice, cs);
    };
    std::string err;
    auto open = [&](Dflash2Drafter& d) {
        if (!d.load(argv[1], 0, err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return false; }
        if (!d.bind_raw(embed, d_head, 8, VP, dvocab, err)) { std::fprintf(stderr, "bind: %s\n", err.c_str()); return false; }
        return true;
    };
    auto context = [&](Dflash2Drafter& d, const Dev& t, int64_t row0, int64_t n, int64_t pos0) {
        if (!d.context(t.p + (size_t) row0 * TAPN, n, pos0, err)) { std::fprintf(stderr, "context: %s\n", err.c_str()); return false; }
        return true;
    };
    // one round: row `row` of `t` is the new context row at position `pos`
    auto draft = [&](Dflash2Drafter& d, const Dev& t, int64_t row, int64_t pos, Answer& a) {
        cudaMemcpy(d_last, t.p + (size_t) row * TAPN, (size_t) TAPN * 4, cudaMemcpyDeviceToDevice);
        if (!d.draft(d_last, 1, pos, ANCHOR, a.tok, a.prob, err)) { std::fprintf(stderr, "draft: %s\n", err.c_str()); return false; }
        return true;
    };

    const int n_ctx = 2300;
    const int64_t up = n_ctx - 1;
    Dev t1(randn((size_t) n_ctx * TAPN, 1, 1.0f)), t2(randn((size_t) 300 * TAPN, 2, 1.0f));
    Answer A{}, B{};
    std::vector<uint8_t> w;
    {   // (a) the live ring
        Dflash2Drafter d;
        if (!open(d) || !context(d, t1, 0, up, 0)) return 1;
        expect(d.save_window(up, w, err), "(a) save_window");
        if (!draft(d, t1, up, up, A)) return 1;
        for (int i = 0; i < 7; ++i) std::printf("  draft %d: %d (p %.3f)\n", i, A.tok[i], A.prob[i]);
    }
    {   // (b) a ring that moved on, then the restore; (c) the refusal; (d) the size
        Dflash2Drafter d;
        if (!open(d) || !context(d, t1, 0, up, 0) || !context(d, t2, 0, 300, up)) return 1;
        expect(!d.restore_window(w, n_ctx, err), "(c) restore_window at another upto is refused");
        std::printf("  refusal: %s\n", err.c_str());
        expect(d.restore_window(w, up, err), "(b) restore_window");
        if (!err.empty()) std::printf("  err: %s\n", err.c_str());
        if (!draft(d, t1, up, up, B)) return 1;
        same(A, B, "(b) restored drafts equal the live ring's");
        expect(w.size() == d.window_bytes(up), "(d) w.size() == window_bytes(n_ctx - 1)");
        std::printf("  window %zu bytes for upto %lld, %zu bytes for upto 100\n", w.size(), (long long) up, d.window_bytes(100));
    }
    {   // (g) clear_ring leaves no stale row
        Dflash2Drafter d;
        Answer G{};
        if (!open(d) || !context(d, t2, 0, 300, 0)) return 1;
        d.clear_ring(0);
        if (!context(d, t1, 0, up, 0) || !draft(d, t1, up, up, G)) return 1;
        same(A, G, "(g) clear_ring then the same context drafts like a fresh drafter");
    }
    {   // (e) the ring wraps twice
        const int n = 5000;
        const int64_t u = n - 1;
        Dev t3(randn((size_t) n * TAPN, 3, 1.0f));
        Answer C{}, D{}, E{};
        std::vector<uint8_t> w3;
        {
            Dflash2Drafter d;
            if (!open(d) || !context(d, t3, 0, u, 0)) return 1;
            expect(d.save_window(u, w3, err), "(e) save_window after the wrap");
            if (!draft(d, t3, u, u, C)) return 1;
        }
        {
            Dflash2Drafter d;   // saw only the last 2048 rows
            if (!open(d) || !context(d, t3, u - 2048, 2048, u - 2048) || !draft(d, t3, u, u, D)) return 1;
        }
        {
            Dflash2Drafter d;
            if (!open(d) || !context(d, t2, 0, 300, 0)) return 1;
            expect(d.restore_window(w3, u, err), "(e) restore_window after the wrap");
            if (!draft(d, t3, u, u, E)) return 1;
        }
        same(C, D, "(e) a wrapped ring drafts like a drafter that saw only the last 2048 rows");
        same(C, E, "(e) a restored wrapped window drafts like the live ring");
        expect(w3.size() == w.size(), "(e) the wrapped window has 2048 rows like the first");
        // (f) the time of 2047 context rows, cold then warm
        Dflash2Drafter d;
        if (!open(d)) return 1;
        for (int run = 0; run < 2; ++run) {
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            if (!context(d, t3, 0, 2047, 0)) return 1;
            std::printf("(f) context() of 2047 rows: %.2f ms (%s)\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), run ? "warm" : "cold");
        }
    }
    std::printf(g_bad ? "dflash2_window_test: FAIL\n" : "dflash2_window_test: PASS\n");
    return g_bad ? 1 : 0;
}
