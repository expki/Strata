// The prompt path's streamed ring sized by the chunk, for one NVIDIA card beside the AMD tier with every expert it streams page-locked.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::prefill {

/// Measured on the test box (RTX 3090 with a W6800, Swift 1.5 Q6_K, the whole copy): the best ring was 96 slots at 2K tokens, 192 at 8K, 384 to 512 at 20K.
inline constexpr int64_t kChunkRingTokensPerSlot = 40;
inline constexpr int kChunkRingMin = 96;

/// The ring for a chunk of `tokens`: one slot per kChunkRingTokensPerSlot tokens, from kChunkRingMin up to `cap`.
inline int chunk_ring_slots(int64_t tokens, int cap) {
    const int64_t want = tokens / kChunkRingTokensPerSlot;
    return (int) std::clamp<int64_t>(want, kChunkRingMin, std::max(cap, kChunkRingMin));
}

}  // namespace strata::prefill
