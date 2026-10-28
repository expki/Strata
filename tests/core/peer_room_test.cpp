// The AMD card's prompt room before its experts fill: when it opens a ring of copies and how many. CPU only.
#include "strata/core/peer_room.hpp"

#include <cstdint>
#include <cstdio>

using strata::core::PeerRoom;
using strata::core::peer_room_ring;
using strata::prefill::kTierCardMin;
using strata::prefill::kTierCardSpare;

namespace {

int failed = 0, checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failed;
    }
}

constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kGiB = 1ull << 30;

// the box's room at chunk 4096: 255 MiB of buffers and a ring of 128 Q6_K copies
PeerRoom box_room() {
    PeerRoom r;
    r.tokens = 4096;
    r.ring = 128;
    r.prompt_bytes = 255 * kMiB;
    r.ring_bytes = 4428800;   // 4.2236 MiB, a blob rounded to 256 bytes
    return r;
}

// a room with no buffers and copies of `slot` bytes, to check the edges of the smallest ring
PeerRoom bare_room(uint64_t prompt, uint64_t slot) {
    PeerRoom r;
    r.tokens = 1024;
    r.ring = 128;
    r.prompt_bytes = prompt;
    r.ring_bytes = slot;
    return r;
}

}  // namespace

int main() {
    const PeerRoom box = box_room();
    const uint64_t keep = 320 * kMiB;
    // the pairs no NVIDIA card holds are 88 GiB, far past the card's 30 GiB: some stay in RAM and stream, so the full ring opens
    check(peer_room_ring(box, 88 * kGiB, 30 * kGiB, keep) == 128, "rest past the room: a full ring");
    // they fit beside the buffers and the spare: none streams, no ring
    check(peer_room_ring(box, 20 * kGiB, 30 * kGiB, keep) == 0, "rest that fits: no ring");
    check(peer_room_ring(box, 30 * kGiB - keep - box.prompt_bytes, 30 * kGiB, keep) == 0, "rest that fits to the byte: no ring");
    check(peer_room_ring(box, 30 * kGiB - keep - box.prompt_bytes + 1, 30 * kGiB, keep) == 128, "one byte past: a ring");
    PeerRoom none = box;
    none.tokens = 0;
    check(peer_room_ring(none, 88 * kGiB, 30 * kGiB, keep) == 0, "no prompt room: no ring");
    PeerRoom unwanted = box;
    unwanted.want_ring = false;
    check(peer_room_ring(unwanted, 88 * kGiB, 30 * kGiB, keep) == 0, "a box plan that does not stream: no ring");
    // a card with room for fewer than a full ring opens what fits, and none under the smallest ring
    const uint64_t room40 = keep + box.prompt_bytes + 40 * box.ring_bytes;
    check(peer_room_ring(box, 88 * kGiB, room40, keep) == 40, "room for 40 copies: 40");
    const uint64_t room15 = keep + box.prompt_bytes + (kTierCardMin - 1) * box.ring_bytes;
    check(peer_room_ring(box, 88 * kGiB, room15, keep) == 0, "room under the smallest ring: none");

    // the old start-up check's edges, in these terms: the spare kept, the buffers off the free VRAM first, the smallest ring
    const uint64_t slot = 2 * kMiB;
    const uint64_t rest = 100 * kGiB;
    check(peer_room_ring(bare_room(0, slot), rest, 252 * kMiB, kTierCardSpare) == 0, "252 MiB free holds no ring past the spare");
    check(peer_room_ring(bare_room(0, slot), rest, kTierCardSpare + kTierCardMin * slot, kTierCardSpare) == kTierCardMin,
          "the spare and the smallest ring are enough");
    check(peer_room_ring(bare_room(0, slot), rest, kTierCardSpare + kTierCardMin * slot - 1, kTierCardSpare) == 0,
          "one byte short is not");
    check(peer_room_ring(bare_room(300 * kMiB, slot), rest, 580 * kMiB, kTierCardSpare) == 0 &&
              peer_room_ring(bare_room(300 * kMiB, slot), rest, 700 * kMiB, kTierCardSpare) == 72,
          "the prompt buffers come off the free VRAM first");
    check(peer_room_ring(bare_room(300 * kMiB, slot), rest, 100 * kMiB, kTierCardSpare) == 0 &&
              peer_room_ring(bare_room(0, 0), rest, 600 * kMiB, kTierCardSpare) == 0,
          "buffers past the free VRAM, or no copy size");

    if (failed) {
        std::fprintf(stderr, "peer_room_test: %d of %d checks failed\n", failed, checks);
        return 1;
    }
    std::printf("peer_room_test: %d checks passed\n", checks);
    return 0;
}
