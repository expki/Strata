// The AMD card's prompt room: the prompt buffers and the ring of copies it opens before its experts fill, so they take only what is left.
#pragma once

#include "strata/prefill/tier_stream.hpp"   // kTierCardMin

#include <algorithm>
#include <cstdint>

namespace strata::core {

/// The prompt room an AMD tier opens before its experts fill: buffers for chunks of up to `tokens` and a ring of up to `ring` copies.
struct PeerRoom {
    int64_t tokens = 0;                 ///< the chunk the prompt buffers take, 0 for no room
    int64_t ring = 0;                   ///< the copies a full ring holds
    bool want_ring = true;              ///< the box plan streams RAM experts to the card
    uint64_t gu = 0, d = 0, blob = 0;   ///< the prompt kernels' largest matrices and the largest expert
    uint64_t prompt_bytes = 0;          ///< what the buffers take on the card
    uint64_t ring_bytes = 0;            ///< what one copy of the ring takes on the card
};

/// The ring to open before the fill when it is wanted and pairs stay in RAM: what fits past the spare and the buffers, or 0 under kTierCardMin.
inline int64_t peer_room_ring(const PeerRoom& r, uint64_t rest_bytes, uint64_t free_bytes, uint64_t keep_bytes) {
    if (r.tokens <= 0 || !r.want_ring || r.ring <= 0 || r.ring_bytes == 0) return 0;
    const uint64_t taken = keep_bytes + r.prompt_bytes;
    const uint64_t left = free_bytes > taken ? free_bytes - taken : 0;
    if (rest_bytes <= left) return 0;   // every pair the card could hold fits, so none streams
    const int64_t n = std::min<int64_t>(r.ring, (int64_t) (left / r.ring_bytes));
    return n >= prefill::kTierCardMin ? n : 0;
}

}  // namespace strata::core
