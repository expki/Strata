// Synthetic expert blobs and decode plans for the GPU parity tests and the AMD tier probe. No model needed.
#pragma once

#include "strata/core/xtier_abi.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

inline int block_values(int t) { return t == 20 || t == 7 || t == 8 ? 32 : t == 42 ? 64 : 256; }

// `rows` rows of `n` values of format t: random bytes, then a finite fp16 scale in every block
inline std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = strata::kernels::iq_row_bytes(t, n), bs = strata::kernels::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        if (t == 29) {
            // IQ1_M: the fp16 scale is the top nibbles of its four scale words (bytes 48..55), the sign and the
            // exponent's top bits in the last: 0x1 / 0x2 there (0x9 / 0xA negative) keeps it in 2^-11 .. 2^-3
            const uint8_t nib = (uint8_t) ((sgn(rng) ? 0x8 : 0x0) | (1 + (byte(rng) & 1)));
            w[o + 55] = (uint8_t) ((w[o + 55] & 0x0F) | (nib << 4));
        } else {   // the other formats start their block with the fp16 scale: 2^-13 .. 2^-6
            // (Q4_K / Q5_K: 2^-14 .. 2^-11, their 6-bit sub-scales multiply it by up to 63)
            const int e = (t == 12 || t == 13) ? 1 + ex(rng) % 4 : ex(rng);
            const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (e << 10) | man(rng));
            std::memcpy(&w[o], &h, 2);
            if (t == 12 || t == 13 || t == 7) {   // Q4_K / Q5_K: dmin, Q5_1: m - the second fp16, as small
                const int em = (t == 7) ? ex(rng) : 1 + ex(rng) % 4;
                const uint16_t m = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (em << 10) | man(rng));
                std::memcpy(&w[o + 2], &m, 2);
            }
        }
    }
    return w;
}

// The plan of a decode call from its routing (n_tok rows of k expert ids), grouped as PeerExperts::launch does, addr[e] being expert e's address.
inline strata_xtier_meta xtier_plan(const int32_t* ids, int64_t n_tok, int64_t k, const std::vector<unsigned long long>& addr) {
    strata_xtier_meta m;
    std::memset(&m, 0, sizeof(m));
    const int64_t n = n_tok * k;
    int groups = 0, rows = 0;
    for (int64_t i = 0; i < n; ++i) {
        bool seen = false;
        for (int64_t j = 0; j < i && !seen; ++j) seen = ids[j] == ids[i];
        if (seen) continue;
        m.ptr[groups] = addr[(size_t) ids[i]];
        m.start[groups] = rows;
        for (int64_t j = i; j < n; ++j) {
            if (ids[j] != ids[i]) continue;
            m.dst[rows] = rows;
            m.tok[rows] = (int32_t) (j / k);
            m.out_row[rows] = (int32_t) j;
            ++rows;
        }
        ++groups;
    }
    m.start[groups] = rows;
    m.count[0] = groups;
    m.count[1] = rows;
    return m;
}
