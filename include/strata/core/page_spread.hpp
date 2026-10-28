// include/strata/core/page_spread.hpp: the order the RAM copy's pages are first touched in, so each expert's pages spread.
#pragma once

#include <cstdint>

namespace strata::core {

/// Touches page p of n at step p * g mod n (g near n times the golden ratio), so an expert's pages get frames from far apart
struct PageSpread {
    uint64_t n = 0;
    uint64_t step = 0;   ///< g's inverse mod n: the page touched at step i is i * step mod n

    explicit PageSpread(uint64_t pages) : n(pages) {
        if (n < 2) return;
        uint64_t g = (uint64_t) ((double) n * 0.6180339887498949);
        if (g == 0) g = 1;
        while (gcd(g, n) != 1) ++g;
        step = inverse(g % n, n);
    }
    /// The page touched at step i (i < n).  n stays below 2^32, so the product cannot overflow.
    uint64_t page(uint64_t i) const { return n < 2 ? 0 : i * step % n; }

private:
    static uint64_t gcd(uint64_t a, uint64_t b) {
        while (b != 0) { const uint64_t t = a % b; a = b; b = t; }
        return a;
    }
    static uint64_t inverse(uint64_t a, uint64_t m) {   // a coprime to m (extended Euclid)
        int64_t t = 0, nt = 1, r = (int64_t) m, nr = (int64_t) a;
        while (nr != 0) {
            const int64_t q = r / nr, t2 = t - q * nt, r2 = r - q * nr;
            t = nt; nt = t2; r = nr; nr = r2;
        }
        return (uint64_t) (t < 0 ? t + (int64_t) m : t);
    }
};

}  // namespace strata::core
