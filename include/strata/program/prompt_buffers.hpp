#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

// Where a layer split stage keeps a chunk's prompt buffers (#340): lent from its expert cache's tail or allocated beside it, whichever it can.
namespace strata::program::prompt_buffers {

// a loan leaves at least this many slots in the cache
constexpr int64_t kLendFloor = 128;
// own buffers leave this much VRAM for what is made after them: the verify windows, the draft head, BLAS workspaces
constexpr uint64_t kOwnHeadroom = 3ull << 29;

enum class Source { lend, own, none };

struct Stage {
    int64_t slots = 0;   ///< the stage's expert cache slots
    int64_t lend = 0;    ///< the slots a loan of the chunk's buffers takes
    uint64_t need = 0;   ///< the chunk's buffer bytes
    uint64_t free = 0;   ///< the card's free VRAM with its cache in place (0 when unknown)
};

struct Rules {
    bool pct_cap = false;   ///< the auto scan's rule: a loan takes at most `pct` % of the cache
    int64_t pct = 0;
    bool own = false;       ///< a stage may hold its own buffers (a layer split, STRATA_SPLIT_OWN_BUFFERS not 0)
};

inline bool can_lend(const Stage& s, const Rules& r) {
    if (s.lend <= 0 || s.lend + kLendFloor > s.slots) return false;
    return !(r.pct_cap && s.lend * 100 > r.pct * s.slots);
}

inline bool can_own(const Stage& s) { return s.free >= s.need + kOwnHeadroom; }

// a stage that can lend lends, else it holds its own buffers when the rules allow and its card has the room
inline Source source(const Stage& s, const Rules& r) {
    if (can_lend(s, r)) return Source::lend;
    return r.own && can_own(s) ? Source::own : Source::none;
}

inline bool chunk_ok(const std::vector<Stage>& stages, const Rules& r) {
    for (const Stage& s : stages)
        if (source(s, r) == Source::none) return false;
    return true;
}

// the largest chunk one NVIDIA card with the AMD tier reads, under the 20,972 tokens where chunk x 10 x 2560 floats pass 2^31 bytes
constexpr int64_t kOneCardTierChunkMax = 20480;
// the free VRAM the card keeps beside the chunk: the engine's own LOW line
constexpr uint64_t kOneCardVramFloor = 256ull << 20;

// The largest chunk on the 256 grid up to the context, the operator's ceiling and the cap whose loan keeps the slot floor and whose other bytes keep the VRAM floor, or 0.
inline int64_t one_card_tier_chunk(int64_t max_context, int64_t ceiling, int64_t slots, const std::function<int64_t(int64_t)>& lend,
                                   uint64_t vram_free, const std::function<uint64_t(int64_t)>& prompt_bytes) {
    const int64_t context = max_context > 0 ? (max_context + 255) / 256 * 256 : kOneCardTierChunkMax;
    for (int64_t c = std::min({kOneCardTierChunkMax, context, ceiling / 256 * 256}); c >= 256; c -= 256) {
        const uint64_t beside = prompt_bytes(c);
        if (can_lend(Stage{slots, lend(c), 0, 0}, Rules{}) && vram_free >= beside && vram_free - beside >= kOneCardVramFloor) return c;
    }
    return 0;
}

}  // namespace strata::program::prompt_buffers
