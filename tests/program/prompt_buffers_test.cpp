// Where a layer split stage keeps the prompt path's buffers for a chunk: lent from its cache, its own, or neither.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "strata/program/prompt_buffers.hpp"

using namespace strata::program::prompt_buffers;

static int failed = 0;
static int checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failed;
    }
}

static constexpr uint64_t kMiB = 1ull << 20;

// a stage at a chunk of `c` tokens: the buffers grow with the chunk, a slot is 2 MiB
static Stage at(int64_t slots, uint64_t free_mib, int64_t c) {
    const uint64_t need = (200 + (uint64_t) c * 420 / 1024) * kMiB;
    return Stage{slots, (int64_t) (need / (2 * kMiB)) + 1, need, free_mib * kMiB};
}

// the explicit --prefill scan: the operator's chunk, halved until every stage can take it
static int64_t scan(int64_t want, const std::vector<std::pair<int64_t, uint64_t>>& cards, bool own_ok) {
    const Rules r{false, 0, own_ok};
    for (int64_t c = want; c >= 256; c /= 2) {
        std::vector<Stage> v;
        for (const auto& k : cards) v.push_back(at(k.first, k.second, c));
        if (chunk_ok(v, r)) return c;
    }
    return 0;
}

int main() {
    const Rules own{false, 0, true}, lend_only{false, 0, false};
    // a cache that can lend lends, whatever the card has free
    check(source(Stage{1295, 767, 1900 * kMiB, 0}, own) == Source::lend, "a cache with room lends");
    check(source(Stage{1295, 767, 1900 * kMiB, 20000 * kMiB}, own) == Source::lend, "lend wins over own room");
    // the loan leaves 128 slots: 1295 - 1167 is the edge
    check(source(Stage{1295, 1167, 0, 0}, lend_only) == Source::lend, "a loan that leaves 128 slots fits");
    check(source(Stage{1295, 1168, 0, 0}, lend_only) == Source::none, "a loan that leaves 127 slots does not");
    check(source(Stage{100, 0, 0, 0}, lend_only) == Source::none, "a loan of no slots is no loan");
    // the auto scan's percentage cap
    check(source(Stage{1000, 500, 0, 0}, Rules{true, 50, false}) == Source::lend, "half the cache under a 50% cap");
    check(source(Stage{1000, 501, 0, 0}, Rules{true, 50, false}) == Source::none, "over the 50% cap");
    check(source(Stage{1000, 1, 0, 0}, Rules{true, 0, false}) == Source::none, "a 0% cap lends nothing, as today");
    // a capped cache too small to lend, on a card with room for the buffers and the 1.5 GiB headroom
    const uint64_t need = 1900 * kMiB;
    check(source(Stage{568, 1100, need, need + kOwnHeadroom}, own) == Source::own, "no loan but room: own");
    check(source(Stage{568, 1100, need, need + kOwnHeadroom - 1}, own) == Source::none, "one byte short: none");
    check(source(Stage{568, 1100, need, 20000 * kMiB}, lend_only) == Source::none, "STRATA_SPLIT_OWN_BUFFERS=0: none");
    // every stage must lend or own
    const Stage lend{1295, 767, need, 0}, mine{568, 1100, need, 20000 * kMiB}, neither{568, 1100, need, 2520 * kMiB};
    check(chunk_ok({lend, mine, lend}, own), "lend, own, lend keeps the chunk");
    check(!chunk_ok({lend, mine, neither}, own), "one stage that can do neither cuts it");
    check(!chunk_ok({lend, mine, lend}, lend_only), "without own buffers every stage must lend");
    check(chunk_ok({mine, mine}, own), "every stage on its own buffers");
    check(chunk_ok({}, own), "no stage, nothing to cut");
    // the scan: three caches that all lend keep 4096 either way (no cap: today's result)
    const std::vector<std::pair<int64_t, uint64_t>> lending{{1295, 448}, {2547, 545}, {6002, 327}};
    check(scan(4096, lending, true) == 4096 && scan(4096, lending, false) == 4096, "lending stages keep 4096");
    // a small cache on a card full to its reserve cuts the chunk with or without the rule
    const std::vector<std::pair<int64_t, uint64_t>> full{{568, 700}, {2547, 545}, {6002, 327}};
    check(scan(4096, full, true) == scan(4096, full, false), "no room on the small card: today's chunk");
    check(scan(4096, full, false) == 1024, "that chunk is 1024");
    // a capped cache on a card with VRAM to spare no longer cuts the chunk
    const std::vector<std::pair<int64_t, uint64_t>> capped{{1310, 1500}, {600, 4900}, {2530, 10580}};
    check(scan(4096, capped, false) == 1024, "today the capped stage cuts 4096 to 1024");
    check(scan(4096, capped, true) == 4096, "with own buffers the chunk stays 4096");
    // the capped card is also short of room: the chunk is cut to what it can do
    const std::vector<std::pair<int64_t, uint64_t>> tight{{568, 2520}, {1661, 3410}, {2530, 10580}};
    check(scan(4096, tight, true) == 1024, "a capped 3050 with 2.5 GiB free still cuts to 1024");

    // one NVIDIA card with the AMD tier: the box's loan fit (the 3090 lent 506, 871, 1,600 and 2,329 slots at 4096 to 24576)
    const auto loan = [](int64_t c) { return (int64_t) (142.0 + 0.089 * (double) c); };
    const auto beside = [](int64_t) -> uint64_t { return 0; };
    const uint64_t ample = 20000 * kMiB;
    check(loan(4096) == 506 && loan(8192) == 871 && loan(16384) == 1600 && loan(24576) == 2329, "the fit gives the measured loans");
    check(kOneCardTierChunkMax == 20480 && one_card_tier_chunk(262144, kOneCardTierChunkMax, 3103, loan, ample, beside) == 20480,
          "the box's 3,103 slots: the cap, which still reads a 20,053-token prompt whole");
    check(one_card_tier_chunk(262144, kOneCardTierChunkMax, 2000, loan, ample, beside) == 19200, "2,000 slots: the largest grid chunk above the 128-slot floor");
    check(one_card_tier_chunk(8192, kOneCardTierChunkMax, 3103, loan, ample, beside) == 8192, "a context of 8192: no chunk past it");
    check(one_card_tier_chunk(8000, kOneCardTierChunkMax, 3103, loan, ample, beside) == 8192, "a context off the grid: the grid chunk that holds it");
    check(one_card_tier_chunk(262144, kOneCardTierChunkMax, 200, loan, ample, beside) == 0, "200 slots lend no chunk: 0, today's scan");
    // the operator's ceiling (--prefill auto:8192 or STRATA_PREFILL_AUTO_MAX) holds the chunk, and one past the cap does not lift it
    check(one_card_tier_chunk(262144, 8192, 3103, loan, ample, beside) == 8192, "auto:8192: at most 8192");
    check(one_card_tier_chunk(262144, 8192, 600, loan, ample, beside) == 3584, "auto:8192 on 600 slots: what the slots lend");
    check(one_card_tier_chunk(262144, 32768, 3103, loan, ample, beside) == 20480, "auto:32768: still the cap");
    // the VRAM floor: the chunk's bytes beside its loan leave the card 256 MiB
    const auto pb = [](int64_t c) { return (uint64_t) c * 40 * 1024; };
    check(one_card_tier_chunk(262144, kOneCardTierChunkMax, 3103, loan, 900 * kMiB, pb) == 16384, "900 MiB free at 40 KiB a token: 16384 leaves 256 MiB");
    check(one_card_tier_chunk(262144, kOneCardTierChunkMax, 3103, loan, 1100 * kMiB, pb) == 20480, "1,100 MiB free: the floor does not bind under the cap");
    check(one_card_tier_chunk(262144, kOneCardTierChunkMax, 3103, loan, kOneCardVramFloor + pb(256), pb) == 256, "room for one grid step: 256");
    check(one_card_tier_chunk(262144, kOneCardTierChunkMax, 3103, loan, kOneCardVramFloor + pb(256) - 1, pb) == 0, "one byte short: 0");
    check(one_card_tier_chunk(262144, kOneCardTierChunkMax, 3103, loan, 100 * kMiB, beside) == 0, "a card under the floor already: 0");
    if (failed) {
        std::fprintf(stderr, "prompt_buffers_test: %d of %d checks failed\n", failed, checks);
        return 1;
    }
    std::printf("prompt_buffers_test: %d checks passed\n", checks);
    return 0;
}
