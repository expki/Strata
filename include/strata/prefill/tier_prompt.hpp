// The AMD tier's prompt buffers on its card: the tier library takes them, and the engine's start-up chunk plan counts them.
#pragma once

#include "strata/prefill/moe_mmq.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::prefill {

/// The prompt path's widths on the AMD tier, the most experts a token takes and the bounds a call has.
constexpr int64_t kTierEmbd = 2560, kTierFf = 640, kTierMaxTop = 16, kTierMaxBounds = 1 << 15;

/// A call's tables in ints: pair, src, bounds, adds.
constexpr size_t tier_tab_ints(int64_t T) { return (size_t) (5 * T * kTierMaxTop + 2 * kTierMaxBounds); }

/// The card's prompt buffers in bytes for chunks of up to T tokens and expert matrices of gu and d bytes.
struct TierPromptSizes {
    size_t x = 0, w = 0, sum = 0, sum16 = 0, gu_rows = 0, h = 0, dm = 0, xq = 0, hq = 0, grp_gu = 0, grp_d = 0, tab = 0, ident = 0;
    uint64_t total() const { return (uint64_t) x + w + sum + sum16 + gu_rows + h + dm + xq + hq + grp_gu + grp_d + tab + ident; }
};

inline TierPromptSizes tier_prompt_sizes(int64_t T, uint64_t gu, uint64_t d) {
    const size_t TN = (size_t) T * kTierEmbd;
    TierPromptSizes z;
    z.x = TN * 4;
    z.w = (size_t) T * kTierMaxTop * 4;
    z.sum = TN * 4;
    z.sum16 = TN * 2;
    z.gu_rows = (size_t) T * 2 * kTierFf * 4;
    z.h = (size_t) T * kTierFf * 4;
    z.dm = TN * 4;
    z.xq = mmq::q8_bytes(T, kTierEmbd);
    z.hq = mmq::q8_bytes(T, kTierFf);
    z.grp_gu = mmq::kGroup * gu + mmq::kTail;
    z.grp_d = mmq::kGroup * d + mmq::kTail;
    z.tab = tier_tab_ints(T) * 4;
    z.ident = (size_t) T * 4;
    return z;
}

/// What the card gives the prompt buffers for chunks of up to T tokens.
inline uint64_t tier_prompt_bytes(int64_t T, uint64_t gu, uint64_t d) { return tier_prompt_sizes(T, gu, d).total(); }

}  // namespace strata::prefill
