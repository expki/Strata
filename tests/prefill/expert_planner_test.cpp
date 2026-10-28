// expert_planner: where each routed expert of one layer runs, and the measured rates that price it. CPU only.
#include "strata/prefill/expert_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::prefill::planner;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

bool near(double a, double b, double rel) { return std::fabs(a - b) <= rel * std::fabs(b); }

const Rates kR{0.001, 0.003, 1e-7, 4e-8, 1e-7, 4e-8, 0.5, 0.2, 0.01, 1e-7};

Item mk(int expert, int rows, int can, uint64_t bytes = 0, bool pinned = false) {
    return Item{expert, rows, (uint8_t) can, bytes, pinned};
}

// the cost of one item where it runs, written out again so the planner's own sums are checked against it
double cost(const Item& it, Where w, const Rates& r) {
    const double rows = it.rows;
    switch (w) {
        case Where::StageVram:   return rows * r.stage_row_ms;
        case Where::TierVram:    return rows * r.tier_row_ms + r.tier_expert_ms;
        case Where::StageStream: return rows * r.stage_row_ms + (double) it.bytes * (it.pinned ? r.stage_byte_ms_pinned : r.stage_byte_ms_paged);
        case Where::TierStream:  return rows * r.tier_row_ms + r.tier_expert_ms + (double) it.bytes * (it.pinned ? r.tier_byte_ms_pinned : r.tier_byte_ms_paged);
    }
    return 0;
}

bool on_tier(Where w) { return w == Where::TierVram || w == Where::TierStream; }

int bit_of(Where w) { return 1 << (int) w; }   // Where's order is the k* bit order

int count(const Choice& c, Where w) { return (int) std::count(c.where.begin(), c.where.end(), w); }

// every choice is allowed, the slot cap holds and the reported finish times are the sums of the placed costs
void audit(const std::vector<Item>& items, const Choice& c, const Rates& r, const Load& start, int slots, const char* label) {
    check(c.where.size() == items.size(), label);
    double stage = 0, tier = 0;
    bool stage_used = false, tier_used = false;
    for (size_t i = 0; i < items.size(); ++i) {
        check((items[i].can & (kStageVram | kTierVram | kStageStream)) != 0, "audit: every item allows a home");
        check((items[i].can & bit_of(c.where[i])) != 0, label);
        if (items[i].rows <= 0) continue;
        if (on_tier(c.where[i])) { tier += cost(items[i], c.where[i], r); tier_used = true; }
        else { stage += cost(items[i], c.where[i], r); stage_used = true; }
    }
    const double want_stage = start.stage_ms + (stage_used ? r.stage_fixed_ms + stage : 0);
    check(near(c.stage_ms, want_stage, 1e-9) || std::fabs(c.stage_ms - want_stage) < 1e-12, label);
    const double want_tier = start.tier_ms + (tier_used ? r.tier_fixed_ms + tier : 0);
    check(near(c.tier_ms, want_tier, 1e-9) || std::fabs(c.tier_ms - want_tier) < 1e-12, label);
    check(count(c, Where::TierStream) <= slots, label);
}

void empty() {
    const Load start{1.5, 2.5};
    const Choice c = plan_layer({}, kR, start, 4);
    check(c.where.empty(), "empty: no choices");
    check(c.stage_ms == start.stage_ms && c.tier_ms == start.tier_ms, "empty: finish times are the start");
}

void homes_only() {
    const std::vector<Item> items = {mk(0, 100, kStageVram), mk(1, 100, kTierVram), mk(2, 50, kStageVram), mk(3, 50, kTierVram)};
    for (const Load start : {Load{0, 0}, Load{0, 1000}, Load{1000, 0}}) {
        const Choice c = plan_layer(items, kR, start, 4);
        audit(items, c, kR, start, 4, "homes_only: audit");
        check(c.where[0] == Where::StageVram && c.where[2] == Where::StageVram, "homes_only: stage items stay");
        check(c.where[1] == Where::TierVram && c.where[3] == Where::TierVram, "homes_only: tier items stay");
    }
    const Choice c = plan_layer(items, kR, Load{}, 4);
    check(near(c.stage_ms, 0.2 + 0.15, 1e-9), "homes_only: stage finish with its fixed cost");
    check(near(c.tier_ms, 0.5 + 0.45 + 2 * kR.tier_expert_ms, 1e-9), "homes_only: tier finish with the call's and each expert's fixed costs");
}

// 20 tier-held experts of 400 down to 20 rows, each 8 MB in a paged RAM copy if the stage takes it
std::vector<Item> tier_held() {
    std::vector<Item> items;
    for (int i = 0; i < 20; ++i) items.push_back(mk(i, 400 - 20 * i, kTierVram | kStageStream, 8'000'000, false));
    return items;
}

void tier_slow_moves_busiest() {
    const std::vector<Item> items = tier_held();
    const Load start{};
    const Choice c = plan_layer(items, kR, start, 0);
    audit(items, c, kR, start, 0, "tier_slow: audit");
    const int moved = count(c, Where::StageStream);
    check(moved > 0 && moved < 20, "tier_slow: some move and some stay");
    int min_moved_rows = 1 << 30, max_kept_rows = 0;
    double largest_moved_cost = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        if (c.where[i] == Where::StageStream) {
            min_moved_rows = std::min(min_moved_rows, items[i].rows);
            largest_moved_cost = std::max(largest_moved_cost, cost(items[i], Where::StageStream, kR));
        } else {
            check(c.where[i] == Where::TierVram, "tier_slow: the rest stay in the tier");
            max_kept_rows = std::max(max_kept_rows, items[i].rows);
        }
    }
    check(min_moved_rows >= max_kept_rows, "tier_slow: the moved ones have the most rows");
    check(std::fabs(c.stage_ms - c.tier_ms) <= largest_moved_cost, "tier_slow: balanced within the largest moved item");
    const double all_tier = kR.tier_fixed_ms + 4200 * kR.tier_row_ms + 20 * kR.tier_expert_ms;
    check(std::max(c.stage_ms, c.tier_ms) < all_tier, "tier_slow: finishes before leaving everything in the tier");
}

void tier_busy_elsewhere() {
    const std::vector<Item> items = tier_held();
    const Load start{0, 1000};
    const Choice c = plan_layer(items, kR, start, 0);
    audit(items, c, kR, start, 0, "tier_busy: audit");
    check(count(c, Where::StageStream) == 20, "tier_busy: every movable item goes to the stage");
    check(c.tier_ms == 1000, "tier_busy: no call to the tier, no fixed cost");
}

// 30 experts in RAM (pinned copies), 100 to 300 rows
std::vector<Item> ram_items() {
    std::vector<Item> items;
    for (int i = 0; i < 30; ++i) items.push_back(mk(i, 100 + 7 * i, kStageStream | kTierStream, 12'000'000, true));
    return items;
}

void ram_to_tier() {
    const std::vector<Item> items = ram_items();
    const Load start{};
    const Choice base = plan_layer(items, kR, start, 0);
    check(count(base, Where::StageStream) == 30, "ram_to_tier: with no slots the stage takes all");
    const Choice c = plan_layer(items, kR, start, 8);
    audit(items, c, kR, start, 8, "ram_to_tier: audit");
    const int n = count(c, Where::TierStream);
    check(n > 0 && n <= 8, "ram_to_tier: some stream to the tier, within the slots");
    check(count(c, Where::TierStream) + count(c, Where::StageStream) == 30, "ram_to_tier: nothing else");
    check(std::max(c.stage_ms, c.tier_ms) < std::max(base.stage_ms, base.tier_ms), "ram_to_tier: finishes sooner than all on the stage");
    int min_moved_rows = 1 << 30, max_kept_rows = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        if (c.where[i] == Where::TierStream) min_moved_rows = std::min(min_moved_rows, items[i].rows);
        else max_kept_rows = std::max(max_kept_rows, items[i].rows);
    }
    check(min_moved_rows >= max_kept_rows, "ram_to_tier: the busiest go first");
    const Load busy{0, 1000};
    const Choice b = plan_layer(items, kR, busy, 8);
    check(count(b, Where::TierStream) == 0, "ram_to_tier: a tier busy elsewhere gets nothing");
    const Load slow_stage{1000, 0};
    const Choice s = plan_layer(items, kR, slow_stage, 8);
    audit(items, s, kR, slow_stage, 8, "ram_to_tier: slow stage audit");
    check(count(s, Where::TierStream) == 8, "ram_to_tier: a stage busy elsewhere fills the slots");
}

void no_tier_slots() {
    const std::vector<Item> items = ram_items();
    for (int slots : {0, -3}) {
        const Choice c = plan_layer(items, kR, Load{}, slots);
        audit(items, c, kR, Load{}, 0, "no_tier_slots: audit");
        check(count(c, Where::TierStream) == 0, "no_tier_slots: no TierStream");
    }
}

void zero_rows() {
    std::vector<Item> items = tier_held();
    items.push_back(mk(100, 0, kTierVram | kStageStream, 8'000'000, false));
    items.push_back(mk(101, 0, kStageStream | kTierStream, 8'000'000, true));
    items.push_back(mk(102, 0, kStageVram | kTierVram, 8'000'000, true));
    const Load start{0, 1000};
    const Choice c = plan_layer(items, kR, start, 4);
    audit(items, c, kR, start, 4, "zero_rows: audit");
    check(c.where[20] == Where::TierVram, "zero_rows: a tier-held item stays");
    check(c.where[21] == Where::StageStream, "zero_rows: a RAM item stays");
    check(c.where[22] == Where::StageVram, "zero_rows: a stage-held item stays");
    std::vector<Item> idle;
    for (const Item& it : items) idle.push_back(mk(it.expert, 0, it.can, it.bytes, it.pinned));
    const Choice z = plan_layer(idle, kR, Load{1.5, 2.5}, 4);
    check(z.stage_ms == 1.5 && z.tier_ms == 2.5, "zero_rows: no rows cost nothing, no call to the tier");
}

void small_items_stay() {
    // a chat turn: 2 rows per expert, the stage would pay 0.8 ms of paged bytes for each one it takes
    std::vector<Item> items;
    for (int i = 0; i < 10; ++i) items.push_back(mk(i, 2, kTierVram | kStageStream, 8'000'000, false));
    const Choice c = plan_layer(items, kR, Load{}, 4);
    audit(items, c, kR, Load{}, 4, "small_items: audit");
    check(count(c, Where::TierVram) == 10, "small_items: a short chunk keeps the tier's experts");
}

void hot_expert_blocks_the_rest() {
    // the pass stops at the first move that does not help, so one hot expert blocks the small ones (moving five of them would end at 17.6 ms, not 18.61)
    std::vector<Item> items;
    for (int i = 0; i < 10; ++i) items.push_back(mk(i, 100, kTierVram | kStageStream, 8'000'000, false));
    items.push_back(mk(10, 5000, kTierVram | kStageStream, 8'000'000, false));
    const Load start{13, 0};
    const Choice c = plan_layer(items, kR, start, 0);
    audit(items, c, kR, start, 0, "hot_expert: audit");
    check(count(c, Where::TierVram) == 11, "hot_expert: nothing moves");
    check(near(c.stage_ms, 13, 1e-9) && near(c.tier_ms, 18.61, 1e-9), "hot_expert: it finishes at 18.61 ms");
}

// Where an item stands before any move: the stage's VRAM, else the tier's VRAM, else streamed to the stage.
Where home(const Item& it) {
    if (it.can & kStageVram) return Where::StageVram;
    if (it.can & kTierVram) return Where::TierVram;
    return Where::StageStream;
}

void random_layers() {
    std::mt19937 rng(41);
    const int cans[] = {kStageVram, kTierVram | kStageStream, kStageStream | kTierStream, kStageVram | kStageStream, kTierVram | kStageStream | kTierStream};
    for (int round = 0; round < 300; ++round) {
        std::vector<Item> items;
        const int n = (int) (rng() % 40);
        for (int i = 0; i < n; ++i)
            items.push_back(mk(i, (int) (rng() % 500), cans[rng() % 5], (1 + rng() % 30) * 1'000'000ull, rng() % 2 == 0));
        const Load start{(double) (rng() % 20) * 0.5, (double) (rng() % 20) * 0.5};
        const int slots = (int) (rng() % 8);
        const Choice c = plan_layer(items, kR, start, slots);
        audit(items, c, kR, start, slots, "random: audit");
        Load home_end = start;
        bool stage_used = false, tier_used = false;
        for (const Item& it : items) {
            if (it.rows <= 0) continue;
            const Where w = home(it);
            (on_tier(w) ? home_end.tier_ms : home_end.stage_ms) += cost(it, w, kR);
            (on_tier(w) ? tier_used : stage_used) = true;
        }
        if (stage_used) home_end.stage_ms += kR.stage_fixed_ms;
        if (tier_used) home_end.tier_ms += kR.tier_fixed_ms;
        check(std::max(c.stage_ms, c.tier_ms) <= std::max(home_end.stage_ms, home_end.tier_ms) + 1e-9, "random: never later than leaving every item home");
    }
}

// the stage's draws: rows and paged bytes vary on their own
struct Draw { int64_t rows; uint64_t bytes; };
Draw draw(std::mt19937& rng) { return Draw{50 + (int64_t) (rng() % 2000), (1 + (uint64_t) (rng() % 40)) * 4'000'000ull}; }

void ratebook_recovers() {
    RateBook book(kR);
    std::mt19937 rng(7);
    for (int i = 0; i < 200; ++i) {
        const Draw d = draw(rng);
        book.observe_stage(d.rows, d.bytes, 0, 0, 0.002 * (double) d.rows + 3e-8 * (double) d.bytes);
    }
    const Rates got = book.rates();
    check(near(got.stage_row_ms, 0.002, 0.05), "ratebook_recovers: stage_row_ms");
    check(near(got.stage_byte_ms_paged, 3e-8, 0.05), "ratebook_recovers: stage_byte_ms_paged");
    check(got.stage_byte_ms_pinned == kR.stage_byte_ms_pinned, "ratebook_recovers: a variable that never varies keeps its prior");
    check(got.tier_row_ms == kR.tier_row_ms && got.tier_fixed_ms == kR.tier_fixed_ms, "ratebook_recovers: the tier is untouched");
    check(got.stage_fixed_ms == kR.stage_fixed_ms, "ratebook_recovers: no calls, so the fixed cost keeps its prior");
}

void ratebook_recovers_tier() {
    RateBook book(kR);
    std::mt19937 rng(11);
    const double row = 0.004, paged = 1.5e-7, fixed = 0.8;
    for (int i = 0; i < 200; ++i) {
        const Draw d = draw(rng);
        const int64_t calls = 1 + (int64_t) (rng() % 4);
        book.observe_tier(d.rows, 0, 0, d.bytes, 0, calls, row * (double) d.rows + paged * (double) d.bytes + fixed * (double) calls);
    }
    const Rates got = book.rates();
    check(near(got.tier_row_ms, row, 0.05), "ratebook_recovers_tier: tier_row_ms");
    check(near(got.tier_byte_ms_paged, paged, 0.05), "ratebook_recovers_tier: tier_byte_ms_paged");
    check(near(got.tier_fixed_ms, fixed, 0.05), "ratebook_recovers_tier: tier_fixed_ms");
    check(got.tier_byte_ms_pinned == kR.tier_byte_ms_pinned, "ratebook_recovers_tier: pinned bytes never varied");
    check(got.tier_expert_ms == kR.tier_expert_ms && got.tier_input_byte_ms == kR.tier_input_byte_ms,
          "ratebook_recovers_tier: no experts or input counted, so their rates keep the priors");
    check(got.stage_row_ms == kR.stage_row_ms, "ratebook_recovers_tier: the stage is untouched");
}

void ratebook_noisy() {
    RateBook book(kR);
    std::mt19937 rng(23);
    for (int i = 0; i < 200; ++i) {
        const Draw d = draw(rng);
        const uint64_t pinned = (uint64_t) (rng() % 40) * 3'000'000ull;
        const double noise = 1 + 0.03 * ((double) (rng() % 2001) / 1000.0 - 1);
        book.observe_stage(d.rows, d.bytes, pinned, 0, noise * (0.002 * (double) d.rows + 3e-8 * (double) d.bytes + 9e-8 * (double) pinned));
    }
    const Rates got = book.rates();
    check(near(got.stage_row_ms, 0.002, 0.05), "ratebook_noisy: stage_row_ms");
    check(near(got.stage_byte_ms_paged, 3e-8, 0.05), "ratebook_noisy: stage_byte_ms_paged");
    check(near(got.stage_byte_ms_pinned, 9e-8, 0.05), "ratebook_noisy: stage_byte_ms_pinned");
}

void ratebook_follows_changes() {
    RateBook book(kR);
    std::mt19937 rng(5);
    for (int i = 0; i < 300; ++i) {
        const Draw d = draw(rng);
        book.observe_stage(d.rows, d.bytes, 0, 0, 0.002 * (double) d.rows + 3e-8 * (double) d.bytes);
    }
    for (int i = 0; i < 400; ++i) {
        const Draw d = draw(rng);
        book.observe_stage(d.rows, d.bytes, 0, 0, 0.001 * (double) d.rows + 6e-8 * (double) d.bytes);
    }
    const Rates got = book.rates();
    check(near(got.stage_row_ms, 0.001, 0.05), "ratebook_follows: stage_row_ms moved with the change");
    check(near(got.stage_byte_ms_paged, 6e-8, 0.05), "ratebook_follows: stage_byte_ms_paged moved with the change");
}

// the true stage time of a draw
double truth(const Draw& d) { return 0.002 * (double) d.rows + 3e-8 * (double) d.bytes; }

bool same(const Rates& a, const Rates& b) { return std::memcmp(&a, &b, sizeof(Rates)) == 0; }

void feed_good(RateBook& book, std::mt19937& rng, int n) {
    for (int i = 0; i < n; ++i) {
        const Draw d = draw(rng);
        book.observe_stage(d.rows, d.bytes, 0, 0, truth(d));
    }
}

void check_recovered(const Rates& got, const char* label) {
    check(near(got.stage_row_ms, 0.002, 0.05), label);
    check(near(got.stage_byte_ms_paged, 3e-8, 0.05), label);
}

void ratebook_waits() {
    RateBook book(kR);
    std::mt19937 rng(13);
    feed_good(book, rng, RateBook::kMinObs - 1);
    check(same(book.rates(), kR), "waits: too few observations leave the priors");
    feed_good(book, rng, 30);
    const Rates got = book.rates();
    check(std::fabs(got.stage_row_ms - 0.002) < std::fabs(kR.stage_row_ms - 0.002), "waits: enough observations move the row rate toward the truth");
    check(std::fabs(got.stage_byte_ms_paged - 3e-8) < std::fabs(kR.stage_byte_ms_paged - 3e-8), "waits: and the byte rate");
    check(got.tier_row_ms == kR.tier_row_ms && got.tier_fixed_ms == kR.tier_fixed_ms, "waits: the other device waits for its own observations");
}

void ratebook_few_wild() {
    // stalls of 100 times the true time, at the very start and among the first observations
    std::mt19937 rng(17);
    RateBook early(kR);
    for (int i = 0; i < 3; ++i) { const Draw d = draw(rng); early.observe_stage(d.rows, d.bytes, 0, 0, 100 * truth(d)); }
    check(same(early.rates(), kR), "few_wild: three stalls leave every coefficient at its prior");
    feed_good(early, rng, 200);
    check_recovered(early.rates(), "few_wild: 200 good ones after three stalls recover");

    RateBook among(kR);
    for (int i = 0; i < 212; ++i) {
        const Draw d = draw(rng);
        among.observe_stage(d.rows, d.bytes, 0, 0, (i == 1 || i == 5 || i == 9) ? 100 * truth(d) : truth(d));
    }
    check_recovered(among.rates(), "few_wild: stalls among the first ten recover");

    RateBook late(kR);
    feed_good(late, rng, 40);
    const Rates before = late.rates();
    for (int i = 0; i < 3; ++i) { const Draw d = draw(rng); late.observe_stage(d.rows, d.bytes, 0, 0, 1000 * truth(d)); }
    const Rates after = late.rates();
    check(near(after.stage_row_ms, before.stage_row_ms, 0.3) && near(after.stage_byte_ms_paged, before.stage_byte_ms_paged, 0.3),
          "few_wild: three stalls of 1000 times move the rates by under 30%");
}

void ratebook_guards() {
    RateBook book(kR);
    std::mt19937 rng(19);
    feed_good(book, rng, 30);
    for (int i = 0; i < 30; ++i) {
        const Draw d = draw(rng);
        book.observe_tier(d.rows, 0, 0, d.bytes, 0, 1, 0.003 * (double) d.rows + 1e-7 * (double) d.bytes + 0.5);
    }
    const Rates base = book.rates();
    const double nan = std::nan(""), inf = HUGE_VAL;
    check(!book.observe_stage(100, 4'000'000, 0, 0, nan), "guards: NaN ms");
    check(!book.observe_stage(100, 4'000'000, 0, 0, inf), "guards: infinite ms");
    check(!book.observe_stage(100, 4'000'000, 0, 0, -1), "guards: negative ms");
    check(!book.observe_stage(100, 4'000'000, 0, 0, 4e6), "guards: more than an hour");
    check(!book.observe_stage(-1, 4'000'000, 0, 0, 1), "guards: negative rows");
    check(!book.observe_stage(100, 4'000'000, 0, -1, 1), "guards: negative calls");
    check(!book.observe_stage(0, 0, 0, 0, 1), "guards: nothing to fit");
    check(!book.observe_tier(100, 0, 0, 4'000'000, 0, 1, nan), "guards: tier NaN ms");
    check(!book.observe_tier(-5, 0, 0, 4'000'000, 0, 1, 1), "guards: tier negative rows");
    check(!book.observe_tier(100, 0, 0, 4'000'000, 0, -1, 1), "guards: tier negative calls");
    check(!book.observe_tier(100, -1, 0, 4'000'000, 0, 1, 1), "guards: tier negative experts");
    check(!book.observe_tier(0, 0, 0, 0, 0, 0, 1), "guards: tier nothing to fit");
    check(same(book.rates(), base), "guards: dropped observations leave the rates as they were");
    check(book.observe_stage(100, 4'000'000, 0, 0, 1) && book.observe_tier(100, 0, 0, 4'000'000, 0, 1, 1), "guards: a good observation is used");
    feed_good(book, rng, 200);
    check_recovered(book.rates(), "guards: the book still recovers after the dropped ones");
}

void ratebook_priors() {
    RateBook book(kR);
    const Rates a = book.rates();
    check(same(a, kR), "priors: with no observations the rates are the priors");
    check(!book.observe_stage(0, 0, 0, 0, 5.0), "priors: a degenerate stage observation is dropped");
    check(!book.observe_tier(0, 0, 0, 0, 0, 0, 5.0), "priors: a degenerate tier observation is dropped");
    check(same(book.rates(), kR), "priors: and changes nothing");
}

void ratebook_zero_priors() {
    RateBook book(Rates{});
    const Rates a = book.rates();
    const double all[] = {a.stage_row_ms, a.tier_row_ms, a.stage_byte_ms_paged, a.stage_byte_ms_pinned,
                          a.tier_byte_ms_paged, a.tier_byte_ms_pinned, a.tier_fixed_ms, a.stage_fixed_ms, a.tier_expert_ms,
                          a.tier_input_byte_ms};
    for (double v : all) check(std::isfinite(v) && v > 0, "zero_priors: a prior of zero still gives a rate above zero");
    std::mt19937 rng(29);
    feed_good(book, rng, 100);
    const Rates g = book.rates();
    check(g.stage_row_ms > 0 && g.stage_byte_ms_paged > 0 && g.stage_byte_ms_pinned > 0, "zero_priors: and after observations");
}

void ratebook_stays_positive() {
    RateBook book(kR);
    std::mt19937 rng(3);
    for (int i = 0; i < 100; ++i) {
        const Draw d = draw(rng);
        check(book.observe_stage(d.rows, d.bytes, d.bytes, 1, 0.0), "positive: an instant stage observation is a measurement");   // the fit wants zero or less
        check(book.observe_tier(d.rows, 1, d.bytes, d.bytes, d.bytes, 1, 0.0), "positive: an instant tier observation is a measurement");
    }
    const Rates g = book.rates();
    const double got[] = {g.stage_row_ms, g.stage_byte_ms_paged, g.stage_byte_ms_pinned, g.tier_row_ms, g.tier_byte_ms_paged,
                          g.tier_byte_ms_pinned, g.tier_fixed_ms, g.stage_fixed_ms, g.tier_expert_ms, g.tier_input_byte_ms};
    const double prior[] = {kR.stage_row_ms, kR.stage_byte_ms_paged, kR.stage_byte_ms_pinned, kR.tier_row_ms, kR.tier_byte_ms_paged,
                            kR.tier_byte_ms_pinned, kR.tier_fixed_ms, kR.stage_fixed_ms, kR.tier_expert_ms, kR.tier_input_byte_ms};
    for (int i = 0; i < 10; ++i)
        check(std::isfinite(got[i]) && got[i] >= RateBook::kFloorFrac * prior[i] && got[i] > 0, "positive: no coefficient goes below its floor");
}

void stage_fixed_with_rows() {
    // the stage pays its fixed cost once it has rows, and the planner counts it before it moves the first item there
    const Rates r{0.001, 0.003, 1e-7, 4e-8, 1e-7, 4e-8, 0.5, 2.0, 0, 1e-7};
    const std::vector<Item> tier_only = {mk(0, 300, kTierVram | kStageStream, 1'000'000, true)};
    const Choice c = plan_layer(tier_only, r, Load{}, 0);   // tier 0.5 + 0.9 ms, on the stage it would be 2.0 + 0.3 + 0.04 ms
    audit(tier_only, c, r, Load{}, 0, "stage_fixed: audit");
    check(c.where[0] == Where::TierVram && c.stage_ms == 0, "stage_fixed: an empty stage pays nothing, and the item stays");
    Rates cheap = r;
    cheap.stage_fixed_ms = 0.2;
    const Choice m = plan_layer(tier_only, cheap, Load{}, 0);
    check(m.where[0] == Where::StageStream && near(m.stage_ms, 0.54, 1e-9) && m.tier_ms == 0, "stage_fixed: a small one lets it move");
    std::vector<Item> both = tier_only;
    both.push_back(mk(1, 100, kStageVram));
    const Choice b = plan_layer(both, r, Load{}, 0);
    audit(both, b, r, Load{}, 0, "stage_fixed: audit with stage rows");
    check(b.where[0] == Where::TierVram && near(b.stage_ms, 2.1, 1e-9), "stage_fixed: the stage's own rows carry its fixed cost");
}

void placed() {
    // placements given from outside: a plan's own price as the plan did, any others as the audit sums them, an expert with no rows costs nothing
    const std::vector<Item> items = tier_held();
    const Load start{0.5, 1.0};
    const Choice c = plan_layer(items, kR, start, 4);
    const Choice p = place_layer(items, c.where, kR, start);
    check(p.where == c.where && p.stage_ms == c.stage_ms && p.tier_ms == c.tier_ms, "placed: a plan's placements price as the plan did");
    std::vector<Where> half(items.size(), Where::TierVram);
    for (size_t i = 0; i < half.size(); i += 2) half[i] = Where::StageStream;
    audit(items, place_layer(items, half, kR, start), kR, start, 0, "placed: audit of placements the plan did not make");
    const std::vector<Item> idle = {mk(0, 0, kTierVram | kStageStream, 8'000'000), mk(1, 0, kStageStream | kTierStream, 8'000'000)};
    const Choice z = place_layer(idle, {Where::TierVram, Where::TierStream}, kR, start);
    check(z.stage_ms == 0.5 && z.tier_ms == 1.0, "placed: an expert with no rows costs neither device");
}

void queue_ahead() {
    check(queued_ahead({}, 5) == 0, "queue: nothing queued");
    check(near(queued_ahead({{0, 10}}, 4), 6, 1e-12), "queue: the rest of one call");
    check(near(queued_ahead({{2, 5}, {0, 10}}, 4), 11, 1e-12), "queue: a call waits for the one queued before it");
    check(queued_ahead({{0, 2}, {1, 1}}, 6) == 0, "queue: calls already done");
    check(near(queued_ahead({{0, 2}, {5, 3}}, 6), 2, 1e-12), "queue: the tier idles between calls");
}

void starting() {
    const Rates a = starting_rates(25), b = starting_rates(0), c = starting_rates(-1);
    check(near(a.stage_byte_ms_pinned, 4e-8, 1e-12), "starting: the probed link sets the stage's byte rate");
    check(near(b.stage_byte_ms_pinned, 1e-6 / 12, 1e-12) && same(b, c), "starting: a link that was not probed counts as 12 GB/s");
    check(a.tier_row_ms < a.stage_row_ms, "starting: the tier counts as the faster until it is measured");
    const double all[] = {a.stage_row_ms, a.tier_row_ms, a.stage_byte_ms_paged, a.stage_byte_ms_pinned,
                          a.tier_byte_ms_paged, a.tier_byte_ms_pinned, a.tier_fixed_ms, a.stage_fixed_ms, a.tier_expert_ms,
                          a.tier_input_byte_ms};
    for (double v : all) check(std::isfinite(v) && v > 0, "starting: every rate is above zero");
}

// a prompt's chunks: rows from 2 to 4096 (log uniform), paged bytes that vary on their own
Draw wide(std::mt19937& rng) {
    const double u = (double) (rng() % 10001) / 10000.0;
    return Draw{(int64_t) std::lround(2 * std::pow(2048.0, u)), (1 + (uint64_t) (rng() % 40)) * 4'000'000ull};
}

void ratebook_stage_intercept() {
    // a fixed 0.3 ms a layer: with calls the fit finds it, without them the row and byte rates take it in
    const auto t = [](const Draw& d) { return 0.002 * (double) d.rows + 3e-8 * (double) d.bytes + 0.3; };
    RateBook with(kR), without(kR);
    std::mt19937 rng(31);
    for (int i = 0; i < 300; ++i) {
        const Draw d = wide(rng);
        with.observe_stage(d.rows, d.bytes, 0, 1, t(d));
        without.observe_stage(d.rows, d.bytes, 0, 0, t(d));
    }
    const Rates w = with.rates(), o = without.rates();
    check(near(w.stage_row_ms, 0.002, 0.05) && near(w.stage_byte_ms_paged, 3e-8, 0.05), "intercept: the rates are found");
    check(near(w.stage_fixed_ms, 0.3, 0.1), "intercept: and the fixed cost");
    check(!near(o.stage_row_ms, 0.002, 0.1) || !near(o.stage_byte_ms_paged, 3e-8, 0.1), "intercept: without it the rates are more than 10% off");
}

void ratebook_measured() {
    RateBook book(kR);
    std::mt19937 rng(37);
    check(!book.stage_measured() && !book.tier_measured(), "measured: not before any observation");
    feed_good(book, rng, RateBook::kMinObs - 1);
    check(!book.stage_measured(), "measured: not one observation short");
    check(!book.observe_stage(0, 0, 0, 0, 1) && !book.stage_measured(), "measured: a dropped observation does not count");
    feed_good(book, rng, 1);
    check(book.stage_measured() && !book.tier_measured(), "measured: the stage at kMinObs, the tier still waits for its own");
    for (int i = 0; i < RateBook::kMinObs; ++i) book.observe_tier(100 + i, 0, 0, 0, 0, 1, 0.5 + 0.003 * (100 + i));
    check(book.tier_measured(), "measured: the tier after its own");
}

void ratebook_collinear() {
    // a streamed walk copies the same bytes every layer (Task 10's 3050), so a rate pushed below its floor leaves its share to the others
    const uint64_t bytes = 754'000'000ull;
    for (const double ms : {52.8, 20.0, 300.0}) {
        RateBook book(starting_rates(13.0));
        for (int i = 0; i < 60; ++i) book.observe_stage(i % 5 == 4 ? 36000 : 40340, bytes, 0, 1, ms);
        const Rates g = book.rates();
        for (const int64_t rows : {40340, 36000}) {
            const double fit = g.stage_fixed_ms + (double) rows * g.stage_row_ms + (double) bytes * g.stage_byte_ms_paged;
            check(near(fit, ms, 0.03), "collinear: the fit predicts what was measured, within 3%");
        }
        const Rates p = starting_rates(13.0);
        check(g.stage_row_ms >= RateBook::kFloorFrac * p.stage_row_ms && g.stage_fixed_ms >= RateBook::kFloorFrac * p.stage_fixed_ms &&
                  g.stage_byte_ms_paged >= RateBook::kFloorFrac * p.stage_byte_ms_paged,
              "collinear: no rate below its floor");
    }
}

void ratebook_floor_one_at_a_time() {
    // rows and paged bytes fall while pinned bytes rise: with the fixed cost at its floor the row and pinned rates both fall below theirs, and only the pinned one belongs there
    const Rates p = starting_rates(13.0);
    const double row = 5e-4, paged = 6e-8, fixed = 1.0;
    std::vector<std::pair<std::vector<double>, double>> obs;
    RateBook book(p);
    for (int i = 0; i < 120; ++i) {
        const int j = (i * 7) % 120;   // the layers in a scrambled order
        const double u = j / 119.0, a = (j * 37 % 11) / 10.0, b = (j * 53 % 13) / 12.0, c = (j * 29 % 7) / 6.0;
        const int64_t rows = std::llround(20000 + 20000 * (-0.9 * u + 0.05 * a));
        const uint64_t pg = (uint64_t) std::llround(2.6e6 * (200 + 150 * (-0.5 * u + 0.05 * b)));
        const uint64_t pn = (uint64_t) std::llround(2.6e6 * (100 + 90 * (0.9 * u + 0.05 * c)));
        const double ms = row * (double) rows + paged * (double) pg + fixed;
        book.observe_stage(rows, pg, pn, 1, ms);
        obs.push_back({{(double) rows, (double) pg, (double) pn}, ms});
    }
    const Rates g = book.rates();
    double worst = 0;
    for (const auto& [x, ms] : obs) {
        const double fit = g.stage_row_ms * x[0] + g.stage_byte_ms_paged * x[1] + g.stage_byte_ms_pinned * x[2] + g.stage_fixed_ms;
        worst = std::max(worst, std::fabs(fit / ms - 1));
    }
    std::printf("  floor: worst error %.1f%%, row rate %.2f of its prior\n", 100 * worst, g.stage_row_ms / p.stage_row_ms);
    check(worst <= 0.05, "floor: the fit predicts every layer within 5%");
    check(g.stage_row_ms > 2 * RateBook::kFloorFrac * p.stage_row_ms, "floor: the row rate is not held at its floor");
    check(g.stage_byte_ms_pinned == RateBook::kFloorFrac * p.stage_byte_ms_pinned, "floor: the pinned rate is");
    check(g.stage_fixed_ms >= RateBook::kFloorFrac * p.stage_fixed_ms, "floor: no rate below its floor");
}

void ratebook_tier_experts() {
    // the tier's calls: rows, experts, input and streamed bytes each vary on their own, as layers and chunks do
    RateBook book(kR);
    std::mt19937 rng(53);
    const double row = 0.0006, expert = 0.06, input = 1.4e-7, paged = 9e-8, fixed = 3.0;
    for (int i = 0; i < 300; ++i) {
        const int64_t experts = 20 + (int64_t) (rng() % 140), rows = experts * (10 + (int64_t) (rng() % 90));
        const uint64_t in = (256 + (uint64_t) (rng() % 3841)) * 5'200ull, streamed = (uint64_t) (rng() % 60) * 2'600'000ull;
        const double ms = row * (double) rows + expert * (double) experts + input * (double) in + paged * (double) streamed + fixed;
        check(book.observe_tier(rows, experts, in, streamed, 0, 1, ms), "tier_experts: a call is an observation");
    }
    const Rates got = book.rates();
    std::printf("  tier_experts: %.4f ms a row x1000, %.4f an expert, %.3g ms a byte in, %.3g streamed, %.3f a call\n", 1000 * got.tier_row_ms,
                got.tier_expert_ms, got.tier_input_byte_ms, got.tier_byte_ms_paged, got.tier_fixed_ms);
    check(near(got.tier_expert_ms, expert, 0.05), "tier_experts: the cost per expert");
    check(near(got.tier_row_ms, row, 0.1), "tier_experts: the cost per row, which moves with the experts");
    check(near(got.tier_input_byte_ms, input, 0.1), "tier_experts: the input rate");
    check(near(got.tier_byte_ms_paged, paged, 0.05), "tier_experts: the streamed rate");
    check(near(got.tier_fixed_ms, fixed, 0.1), "tier_experts: the cost per call");
    check(got.tier_byte_ms_pinned == kR.tier_byte_ms_pinned, "tier_experts: pinned bytes never varied");
}

void expert_cost_moves_fewer() {
    // forty small experts in RAM and a stage busy with other work: each expert the tier computes also costs it a fixed part
    std::vector<Item> items;
    for (int i = 0; i < 40; ++i) items.push_back(mk(i, 20 + i, kStageStream | kTierStream, 3'000'000, false));
    const Load start{10, 0};
    Rates cheap = kR, dear = kR;
    cheap.tier_expert_ms = 0.01;
    dear.tier_expert_ms = 0.2;
    const Choice c = plan_layer(items, cheap, start, 40), d = plan_layer(items, dear, start, 40);
    audit(items, c, cheap, start, 40, "expert_cost: audit");
    audit(items, d, dear, start, 40, "expert_cost: audit, dear");
    const int nc = count(c, Where::TierStream), nd = count(d, Where::TierStream);
    std::printf("  expert_cost: %d of 40 to the tier at 0.01 ms an expert, %d at 0.2\n", nc, nd);
    check(nc > 0 && nd > 0 && nd < nc, "expert_cost: a dearer expert on the tier moves fewer of them");
    Rates held = kR;
    held.tier_expert_ms = 0.2;
    std::vector<Item> tier = tier_held();
    const Choice h0 = plan_layer(tier, kR, Load{}, 0), h1 = plan_layer(tier, held, Load{}, 0);
    check(count(h1, Where::StageStream) >= count(h0, Where::StageStream), "expert_cost: and leaves the stage at least as many of its own");
}

}  // namespace

int main() {
    empty();
    homes_only();
    tier_slow_moves_busiest();
    tier_busy_elsewhere();
    ram_to_tier();
    no_tier_slots();
    zero_rows();
    small_items_stay();
    hot_expert_blocks_the_rest();
    random_layers();
    ratebook_recovers();
    ratebook_recovers_tier();
    ratebook_noisy();
    ratebook_follows_changes();
    ratebook_waits();
    ratebook_few_wild();
    ratebook_guards();
    ratebook_priors();
    ratebook_zero_priors();
    ratebook_stays_positive();
    stage_fixed_with_rows();
    placed();
    queue_ahead();
    starting();
    ratebook_stage_intercept();
    ratebook_measured();
    ratebook_collinear();
    ratebook_floor_one_at_a_time();
    ratebook_tier_experts();
    expert_cost_moves_fewer();
    std::printf("expert_planner_test: %d checks passed\n", checks);
    return 0;
}
