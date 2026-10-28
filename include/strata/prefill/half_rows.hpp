// A chunk read as two token halves: when it is, the rows each half's call to the AMD tier takes, and the stage's rows of both halves. No GPU needed.
#pragma once

#include "strata/prefill/tier_planner.hpp"

#include <cstdint>
#include <vector>

namespace strata::prefill::halves {

/// A take row's values, as the plan gives them per expert: the stage computes it, the tier from its VRAM, the tier streaming it.
inline constexpr uint8_t kStage = planner::kTakeStage, kTier = planner::kTakeTier, kStream = planner::kTakeStream;

/// Whether a chunk that may run as two token halves and has room for them does: once its stage's plan acts.
constexpr bool run_halved(bool wanted, bool fits, bool plan_acts) { return wanted && fits && plan_acts; }

/// One half's call to the tier: its rows per expert (cnt) and their start (off), its tier experts held then streamed (order), each row's pair and token.
struct TierRows {
    std::vector<int32_t> cnt, off, order, pair, src;
    int64_t rows = 0;   // the call's rows, 0 for no call
};

/// A half's rows for the tier from its T tokens' ids (k a token) and take row, grouped as a chunk of those tokens would be, or false for a bad id.
inline bool tier_rows(const int32_t* ids, int64_t T, int64_t k, const uint8_t* take_row, int64_t n_expert, TierRows& h) {
    h.cnt.assign((size_t) n_expert, 0);
    for (int64_t i = 0; i < T * k; ++i) {
        if (ids[i] < 0 || ids[i] >= n_expert) return false;
        ++h.cnt[(size_t) ids[i]];
    }
    h.order.clear();
    h.off.assign((size_t) n_expert + 1, 0);
    int32_t r = 0;
    for (const uint8_t kind : {kTier, kStream})
        for (int64_t e = 0; e < n_expert; ++e)
            if (take_row[e] == kind && h.cnt[(size_t) e] > 0) {
                h.order.push_back((int32_t) e);
                h.off[(size_t) e] = r;
                r += h.cnt[(size_t) e];
            }
    h.rows = r;
    h.pair.resize((size_t) r);
    h.src.resize((size_t) r);
    std::vector<int32_t> fill(h.off.begin(), h.off.end() - 1);
    for (int64_t i = 0; i < T * k; ++i)
        if (take_row[ids[i]] != kStage) {
            const int32_t p = fill[(size_t) ids[i]]++;
            h.pair[(size_t) p] = (int32_t) i;
            h.src[(size_t) p] = (int32_t) (i / k);
        }
    return true;
}

/// The stage's rows of both halves (A the first ta tokens) by expert in pair order, then the pairs the tier takes, from checked ids. Returns the stage's rows.
inline int64_t stage_rows(const int32_t* ids, int64_t T, int64_t ta, int64_t k, const uint8_t* take_a, const uint8_t* take_b, int64_t n_expert,
                          std::vector<int32_t>& cnt, std::vector<int32_t>& off, std::vector<int32_t>& order, int32_t* slot, int32_t* src) {
    const int64_t pb = ta * k;   // the second half's first pair
    cnt.assign((size_t) n_expert, 0);
    for (int64_t i = 0; i < T * k; ++i)
        if ((i < pb ? take_a : take_b)[ids[i]] == kStage) ++cnt[(size_t) ids[i]];
    off.assign((size_t) n_expert + 1, 0);
    order.clear();
    int32_t r = 0;
    for (int64_t e = 0; e < n_expert; ++e) {
        off[(size_t) e] = r;
        r += cnt[(size_t) e];
        if (cnt[(size_t) e] > 0) order.push_back((int32_t) e);
    }
    off[(size_t) n_expert] = r;
    std::vector<int32_t> fill(off.begin(), off.end() - 1);
    int32_t t = r;
    for (int64_t i = 0; i < T * k; ++i) {
        const int32_t p = (i < pb ? take_a : take_b)[ids[i]] == kStage ? fill[(size_t) ids[i]]++ : t++;
        slot[i] = p;
        src[p] = (int32_t) (i / k);
    }
    return r;
}

}  // namespace strata::prefill::halves
