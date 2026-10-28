// PageSpread touches every page once and spreads an expert's pages over the whole fill (CPU only).
#include "strata/core/page_spread.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::core::PageSpread;

namespace {

int checks = 0;
void check(bool ok, const char* label, uint64_t n) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s (n = %llu)\n", label, (unsigned long long) n); std::exit(1); }
}

}  // namespace

int main() {
    for (uint64_t n : {1ull, 2ull, 3ull, 4ull, 7ull, 10ull, 1081ull, 4096ull, 65536ull, 524287ull, 524288ull, 600000ull}) {
        const PageSpread s(n);
        std::vector<uint64_t> when(n, n);   // the step each page is touched at
        for (uint64_t i = 0; i < n; ++i) {
            const uint64_t p = s.page(i);
            check(p < n, "a page inside the range", n);
            check(when[p] == n, "no page touched twice", n);
            when[p] = i;
        }
        for (uint64_t p = 0; p < n; ++p) check(when[p] < n, "every page touched", n);
        if (n < 65536) continue;
        // a Q6_K expert is 1,081 pages: wherever it starts, every sixth of the fill touches at least half its share of them
        constexpr uint64_t kExpert = 1081;
        for (uint64_t p0 = 0; p0 + kExpert <= n; p0 += n / 7) {
            uint64_t sixth[6] = {};
            for (uint64_t p = p0; p < p0 + kExpert; ++p) ++sixth[when[p] * 6 / n];
            for (uint64_t k = 0; k < 6; ++k) check(sixth[k] >= kExpert / 12, "an expert's pages spread over the fill", n);
        }
    }
    std::printf("page_spread_test: %d checks passed\n", checks);
    return 0;
}
