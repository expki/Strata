// The router's top ten and the MoE combine over a chunk's two token halves give each token the whole chunk's bytes.
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace pf = strata::prefill;

constexpr int64_t kN = 2560, kK = 10, kExperts = 512;
int failures = 0, checks = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return (T*) p; }
};

template <typename T> void up(const Dev& d, const std::vector<T>& h) { ck(cudaMemcpy(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
template <typename T> std::vector<T> down(const Dev& d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d.p, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}

void check(bool ok, const char* what, int64_t T) {
    ++checks;
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s (T %lld)\n", what, (long long) T);
}

// One chunk of T tokens: its routing as the prompt path builds it, then each kernel over the chunk and over its halves
void chunk(int64_t T, std::mt19937& rng) {
    const int64_t pairs = T * kK, t0 = T / 2;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    // the router: logits in, ids and weights out
    std::vector<float> logits((size_t) (T * kExperts));
    for (float& x : logits) x = 3.f * nd(rng);
    Dev dl(logits.size() * 4), ids1(pairs * 4), w1(pairs * 4), ids2(pairs * 4), w2(pairs * 4);
    up(dl, logits);
    pf::route(dl.as<float>(), ids1.as<int32_t>(), w1.as<float>(), T, kExperts, nullptr);
    pf::route(dl.as<float>(), ids2.as<int32_t>(), w2.as<float>(), t0, kExperts, nullptr);
    pf::route(dl.as<float>() + t0 * kExperts, ids2.as<int32_t>() + t0 * kK, w2.as<float>() + t0 * kK, T - t0, kExperts, nullptr);
    ck(cudaDeviceSynchronize(), "route");
    const std::vector<int32_t> ids = down<int32_t>(ids1, (size_t) pairs);
    check(ids == down<int32_t>(ids2, (size_t) pairs), "route: the halves' ids are the chunk's", T);
    check(std::memcmp(down<float>(w1, (size_t) pairs).data(), down<float>(w2, (size_t) pairs).data(), (size_t) pairs * 4) == 0,
          "route: the halves' weights are the chunk's, bit for bit", T);
    // the chunk's rows in expert order, a third of the experts the AMD tier's (their rows last), and each pair's row
    std::vector<int32_t> cnt(kExperts, 0), off(kExperts, 0);
    std::vector<char> tier(kExperts, 0);
    for (int64_t e = 0; e < kExperts; ++e) tier[(size_t) e] = rng() % 3 == 0;
    for (int32_t e : ids) ++cnt[(size_t) e];
    int32_t r = 0;
    for (int kind = 0; kind < 2; ++kind)
        for (int64_t e = 0; e < kExperts; ++e)
            if (tier[(size_t) e] == kind) { off[(size_t) e] = r; r += cnt[(size_t) e]; }
    int64_t local = 0;
    for (int64_t e = 0; e < kExperts; ++e) local += tier[(size_t) e] ? 0 : cnt[(size_t) e];
    std::vector<int32_t> slot((size_t) pairs);
    for (int64_t i = 0; i < pairs; ++i) slot[(size_t) i] = off[(size_t) ids[(size_t) i]]++;
    std::vector<float> D((size_t) (pairs * kN)), shared((size_t) (T * kN)), sg((size_t) T), sums((size_t) (T * kN)), sd((size_t) pairs);
    for (float& x : D) x = nd(rng);
    for (float& x : shared) x = nd(rng);
    for (float& x : sg) x = nd(rng);
    for (float& x : sums) x = nd(rng);
    for (float& x : sd) x = 0.5f + ud(rng);
    std::vector<uint16_t> sums16((size_t) (T * kN));
    for (size_t i = 0; i < sums16.size(); ++i) {   // FP16 bits of a value in range
        const float v = sums[i];
        uint32_t u;
        std::memcpy(&u, &v, 4);
        const uint32_t sign = (u >> 16) & 0x8000u, ex = (u >> 23) & 0xff, man = u & 0x7fffffu;
        sums16[i] = (uint16_t) (ex < 113 ? sign : ex > 142 ? (sign | 0x7bffu) : (sign | ((ex - 112) << 10) | (man >> 13)));
    }
    Dev dD(D.size() * 4), dslot(slot.size() * 4), dsh(shared.size() * 4), dsg(sg.size() * 4), dsum(sums.size() * 4), dsum16(sums16.size() * 2),
        dsd(sd.size() * 4), bo1(shared.size() * 4), bo2(shared.size() * 4);
    up(dD, D); up(dslot, slot); up(dsh, shared); up(dsg, sg); up(dsum, sums); up(dsum16, sums16); up(dsd, sd);
    const float* W = w1.as<float>();
    const int32_t* S = dslot.as<int32_t>();
    auto same = [&](const char* what) {
        ck(cudaDeviceSynchronize(), what);
        check(std::memcmp(down<float>(bo1, shared.size()).data(), down<float>(bo2, shared.size()).data(), shared.size() * 4) == 0, what, T);
    };
    // each half reads its rows of the chunk's tables, weights, shared rows, gates and sums, and writes its rows of bo, as run_impl combines it
    auto halves = [&](auto&& one) {
        one((int64_t) 0, t0);
        one(t0, T - t0);
    };
    pf::moe_combine(dD.as<float>(), S, W, dsh.as<float>(), dsg.as<float>(), bo1.as<float>(), T, nullptr);
    halves([&](int64_t a, int64_t n) {
        pf::moe_combine_rows(dD.as<float>(), S, W, dsh.as<float>(), dsg.as<float>(), nullptr, false, 0, nullptr, bo2.as<float>(), a, n, nullptr);
    });
    same("combine: the halves give the chunk's bytes");
    pf::moe_combine(dD.as<float>(), S, W, dsh.as<float>(), dsg.as<float>(), bo1.as<float>(), T, nullptr, dsd.as<float>());
    halves([&](int64_t a, int64_t n) {
        pf::moe_combine_rows(dD.as<float>(), S, W, dsh.as<float>(), dsg.as<float>(), nullptr, false, 0, dsd.as<float>(), bo2.as<float>(), a, n, nullptr);
    });
    same("combine with row scales: the halves give the chunk's bytes");
    pf::moe_combine_peer(dD.as<float>(), S, W, dsh.as<float>(), dsg.as<float>(), dsum.as<float>(), local, bo1.as<float>(), T, nullptr);
    halves([&](int64_t a, int64_t n) {
        pf::moe_combine_rows(dD.as<float>(), S, W, dsh.as<float>(), dsg.as<float>(), dsum.as<float>(), false, local, nullptr, bo2.as<float>(), a, n, nullptr);
    });
    same("combine with the tier's sums: the halves give the chunk's bytes");
    pf::moe_combine_peer16(dD.as<float>(), S, W, dsh.as<float>(), dsg.as<float>(), dsum16.as<uint16_t>(), local, bo1.as<float>(), T, nullptr);
    halves([&](int64_t a, int64_t n) {
        pf::moe_combine_rows(dD.as<float>(), S, W, dsh.as<float>(), dsg.as<float>(), dsum16.as<uint16_t>(), true, local, nullptr, bo2.as<float>(), a,
                             n, nullptr);
    });
    same("combine with the tier's FP16 sums: the halves give the chunk's bytes");
    if (T < 2) return;
    pf::moe_combine(dD.as<float>(), S, W, dsh.as<float>(), dsg.as<float>(), bo1.as<float>(), T, nullptr);
    halves([&](int64_t a, int64_t n) {   // the second half reads the first half's weights
        pf::moe_combine(dD.as<float>(), S + a * kK, W, dsh.as<float>() + a * kN, dsg.as<float>() + a, bo2.as<float>() + a * kN, n, nullptr);
    });
    ck(cudaDeviceSynchronize(), "mutant");
    check(std::memcmp(down<float>(bo1, shared.size()).data(), down<float>(bo2, shared.size()).data(), shared.size() * 4) != 0,
          "a half that reads the wrong rows is caught", T);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        std::printf("prefill_combine_halves_test: SKIP (no GPU)\n");
        return 77;
    }
    try {
        std::mt19937 rng(12);
        for (const int64_t T : {1024, 515, 2}) chunk(T, rng);   // small enough for the free memory beside the service
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill_combine_halves_test: %s\n", e.what());
        return 1;
    }
    std::printf("prefill_combine_halves_test: %d of %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
