// src/prefill/gr_vec_parity.cu: the float4 hyper-connection write+norm and mix against the scalar kernels, bit for bit (--bench adds the ms).
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N = 2560, HC = 4, D = N * HC;
int g_fail = 0;

bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    if (!ck(cudaMalloc(&p, n * sizeof(T) + 64), "cudaMalloc")) std::exit(2);
    cudaMemset(p, 0, n * sizeof(T) + 64);
    return p;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "copy down");
    return h;
}
template <typename T> bool same(const T* a, const T* b, size_t n) {
    const auto x = down(a, n), y = down(b, n);
    return std::memcmp(x.data(), y.data(), n * sizeof(T)) == 0;
}
void check(bool ok, const std::string& label) {
    std::printf("  %-78s %s\n", label.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// One token count and xn16 stride: the write+norm and then the mix, scalar against vector, every output compared.
void run(int64_t T, int64_t ldx, bool f16img, bool lo_images, int shift, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> R((size_t) T * D), bo((size_t) T * N), inj((size_t) T * HC), w(D), gated((size_t) T * D);
    for (auto& x : R) x = nd(g) * 4.0f;
    for (auto& x : bo) x = nd(g) * 2.0f;
    for (auto& x : inj) x = nd(g);
    for (auto& x : w) x = 1.0f + 0.25f * nd(g);
    for (auto& x : gated) x = nd(g) * 3.0f;
    if (T > 2) for (int d = 0; d < D; ++d) R[(size_t) 2 * D + d] *= 3000.0f;   // a token past FP16's range in its image
    const size_t xn_n = (size_t) (T - 1) * ldx + D;
    float *Ra = dalloc<float>((size_t) T * D + shift), *Rb = dalloc<float>((size_t) T * D + shift);
    float *bo_d = dalloc<float>((size_t) T * N), *inj_d = dalloc<float>((size_t) T * HC), *w_d = dalloc<float>(D);
    float *g_d = dalloc<float>((size_t) T * D), *rsa = dalloc<float>((size_t) T * HC), *rsb = dalloc<float>((size_t) T * HC);
    uint16_t *xa = dalloc<uint16_t>(xn_n), *xb = dalloc<uint16_t>(xn_n), *la = dalloc<uint16_t>(xn_n), *lb = dalloc<uint16_t>(xn_n);
    float *ma = dalloc<float>((size_t) T * N), *mb = dalloc<float>((size_t) T * N);
    uint16_t *m16a = dalloc<uint16_t>((size_t) T * N), *m16b = dalloc<uint16_t>((size_t) T * N);
    uint16_t *mha = dalloc<uint16_t>((size_t) T * N), *mhb = dalloc<uint16_t>((size_t) T * N);
    uint16_t *mla = dalloc<uint16_t>((size_t) T * N), *mlb = dalloc<uint16_t>((size_t) T * N);
    float *ra = Ra + shift, *rb = Rb + shift;   // shift 1: R not 16-byte aligned, so the vector path must step aside
    ck(cudaMemcpy(ra, R.data(), R.size() * 4, cudaMemcpyHostToDevice), "R");
    ck(cudaMemcpy(rb, R.data(), R.size() * 4, cudaMemcpyHostToDevice), "R");
    ck(cudaMemcpy(bo_d, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice), "bo");
    ck(cudaMemcpy(inj_d, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice), "inj");
    ck(cudaMemcpy(w_d, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "w");
    ck(cudaMemcpy(g_d, gated.data(), gated.size() * 4, cudaMemcpyHostToDevice), "gated");
    namespace p = strata::prefill;
    p::set_hc_vec(0);
    p::gr_write_norm_rs(ra, bo_d, inj_d, HC, w_d, 1e-6f, rsa, xa, T, nullptr, lo_images ? la : nullptr, ldx, f16img);
    p::gr_mix_r(ra, rsa, w_d, g_d, ma, m16a, T, nullptr, mha, lo_images ? mla : nullptr);
    p::set_hc_vec(1);
    p::gr_write_norm_rs(rb, bo_d, inj_d, HC, w_d, 1e-6f, rsb, xb, T, nullptr, lo_images ? lb : nullptr, ldx, f16img);
    p::gr_mix_r(rb, rsb, w_d, g_d, mb, m16b, T, nullptr, mhb, lo_images ? mlb : nullptr);
    p::set_hc_vec(-1);
    ck(cudaDeviceSynchronize(), "run");
    char head[160];
    std::snprintf(head, sizeof head, "T %lld, xn16 stride %lld, %s image%s%s", (long long) T, (long long) ldx,
                  f16img ? "FP16" : "BF16", lo_images ? ", lo images" : "", shift ? ", R unaligned" : "");
    bool ok = same(ra, rb, (size_t) T * D) && same(rsa, rsb, (size_t) T * HC) && same(xa, xb, xn_n);
    if (lo_images) ok = ok && same(la, lb, xn_n);
    check(ok, std::string(head) + ": write+norm R, rs, xn16");
    ok = same(ma, mb, (size_t) T * N) && same(m16a, m16b, (size_t) T * N) && same(mha, mhb, (size_t) T * N);
    if (lo_images) ok = ok && same(mla, mlb, (size_t) T * N);
    check(ok, std::string(head) + ": mix and its images");
    for (void* q : {(void*) Ra, (void*) Rb, (void*) bo_d, (void*) inj_d, (void*) w_d, (void*) g_d, (void*) rsa, (void*) rsb,
                    (void*) xa, (void*) xb, (void*) la, (void*) lb, (void*) ma, (void*) mb, (void*) m16a, (void*) m16b,
                    (void*) mha, (void*) mhb, (void*) mla, (void*) mlb})
        cudaFree(q);
}

// The ms of each kernel, scalar and vector, at T tokens (10 calls after 3 warm-ups).
void bench(int64_t T) {
    float *R = dalloc<float>((size_t) T * D), *bo = dalloc<float>((size_t) T * N), *inj = dalloc<float>((size_t) T * HC);
    float *w = dalloc<float>(D), *gd = dalloc<float>((size_t) T * D), *rs = dalloc<float>((size_t) T * HC), *m = dalloc<float>((size_t) T * N);
    uint16_t *x = dalloc<uint16_t>((size_t) T * D), *m16 = dalloc<uint16_t>((size_t) T * N);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    namespace p = strata::prefill;
    for (int mode = 0; mode < 2; ++mode) {
        p::set_hc_vec(mode);
        float ms[2] = {};
        for (int which = 0; which < 2; ++which) {
            for (int i = 0; i < 13; ++i) {
                if (i == 3) cudaEventRecord(a);
                if (which == 0) p::gr_write_norm_rs(R, bo, inj, HC, w, 1e-6f, rs, x, T, nullptr);
                else p::gr_mix_r(R, rs, w, gd, m, m16, T, nullptr);
            }
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            cudaEventElapsedTime(&ms[which], a, b);
        }
        const double wb = (double) T * (2.0 * D * 4 + N * 4 + D * 2), mb = (double) T * (2.0 * D * 4 + N * 4 + N * 2);
        std::printf("  T %lld %s: write+norm %.3f ms (%.0f GB/s), mix %.3f ms (%.0f GB/s)\n", (long long) T,
                    mode ? "vector" : "scalar", ms[0] / 10, wb / (ms[0] / 10 * 1e6), ms[1] / 10, mb / (ms[1] / 10 * 1e6));
    }
    p::set_hc_vec(-1);
    for (void* q : {(void*) R, (void*) bo, (void*) inj, (void*) w, (void*) gd, (void*) rs, (void*) m, (void*) x, (void*) m16}) cudaFree(q);
}

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    std::printf("hyper-connection write+norm and mix: float4 kernels against the scalar ones, bit for bit\n");
    uint32_t seed = 7;
    for (int64_t T : {1, 3, 7, 64, 513, 2048}) {
        run(T, D, false, false, 0, seed++);
        run(T, D, false, true, 0, seed++);
        run(T, D, true, true, 0, seed++);
        run(T, D + 64, false, true, 0, seed++);   // the token stride of an image wider than one token's rows
    }
    run(5, D, false, true, 1, seed++);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) bench(20000);
    if (g_fail == 0) std::printf("gr_vec_parity: PASS\n");
    else std::printf("gr_vec_parity: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
