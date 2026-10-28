// tier_planner: a stage's rows predicted from its routing, a layer's items, and the take a plan gives. CPU only.
#include "strata/prefill/tier_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

using namespace strata::prefill::planner;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

bool near(double a, double b, double rel) { return std::fabs(a - b) <= rel * std::fabs(b); }

constexpr int64_t kLayers = 3, kExperts = 6, kTop = 2;
constexpr uint64_t kBlob = 3'000'000;

Held at(Holder h, bool pinned = false) { return Held{h, pinned}; }

void holders() {
    check(holder_of(true, true) == Holder::Stage, "holder: the stage's own copy wins over the tier's");
    check(holder_of(true, false) == Holder::Stage, "holder: the stage's own copy");
    check(holder_of(false, true) == Holder::Tier, "holder: the tier's copy");
    check(holder_of(false, false) == Holder::Ram, "holder: nobody holds it, it streams from RAM");
}

void rows_before_routing() {
    const RouteHistory h(kLayers, kExperts);
    for (int64_t e = 0; e < kExperts; ++e) check(h.rows(1, e, 300, kTop) == 100, "rows: an even share before any routing");
    check(h.rows(0, 0, 7, kTop) == 2, "rows: the even share rounds to the nearest row");
}

void rows_from_history() {
    RouteHistory h(kLayers, kExperts);
    const int32_t cnt[kExperts] = {120, 0, 40, 30, 10, 0};   // 200 rows of a 100 token chunk
    h.add(1, cnt, 100);
    check(h.rows(1, 0, 100, kTop) == 120 && h.rows(1, 1, 100, kTop) == 0 && h.rows(1, 4, 100, kTop) == 10,
          "rows: a chunk of the same size gets the last chunk's rows");
    check(h.rows(1, 0, 250, kTop) == 300 && h.rows(1, 2, 250, kTop) == 100, "rows: the history scaled to the chunk's tokens");
    check(h.rows(1, 3, 50, kTop) == 15, "rows: scaled down too");
    check(h.rows(0, 0, 300, kTop) == 100 && h.rows(2, 5, 300, kTop) == 100, "rows: other layers keep the even share");
}

void rows_decay() {
    RouteHistory h(kLayers, kExperts);
    const int32_t a[kExperts] = {100, 100, 0, 0, 0, 0}, b[kExperts] = {0, 40, 160, 0, 0, 0};
    h.add(0, a, 100);
    h.add(0, b, 100);
    // h = 0.5 a + b over 0.5 * 100 + 100 tokens: the older chunk counts half
    check(h.rows(0, 0, 150, kTop) == 50 && h.rows(0, 1, 150, kTop) == 90 && h.rows(0, 2, 150, kTop) == 160,
          "decay: the older chunk counts half as much as the newer");
    check(h.rows(0, 2, 300, kTop) == 320, "decay: and the sum scales to the chunk");
    for (int i = 0; i < 40; ++i) h.add(0, b, 100);
    check(h.rows(0, 0, 100, kTop) == 0 && h.rows(0, 2, 100, kTop) == 160, "decay: an old chunk fades away");
}

void items_from_holdings() {
    RouteHistory h(kLayers, kExperts);
    const int32_t cnt[kExperts] = {10, 20, 30, 40, 0, 50};
    h.add(2, cnt, 75);
    const std::vector<Held> held = {at(Holder::Stage), at(Holder::Tier), at(Holder::Ram), at(Holder::Tier, true),
                                    at(Holder::Ram, true), at(Holder::Stage)};
    const std::vector<Item> it = layer_items(h, 2, 75, kTop, held, kBlob, false);
    check(it.size() == (size_t) kExperts, "items: one per expert");
    for (int64_t e = 0; e < kExperts; ++e) {
        check(it[(size_t) e].expert == (int32_t) e, "items: in expert order");
        check(it[(size_t) e].rows == cnt[e], "items: the predicted rows");
    }
    check(it[0].can == kStageVram && it[0].bytes == 0 && !it[0].pinned, "items: the stage's own expert stays in its VRAM");
    check(it[1].can == (kTierVram | kStageStream) && it[1].bytes == kBlob && !it[1].pinned,
          "items: a tier-held expert may stay on the tier or stream to the stage");
    check(it[3].can == (kTierVram | kStageStream) && it[3].pinned, "items: a tier-held expert's RAM copy may be pinned");
    check(it[2].can == kStageStream && it[2].bytes == kBlob && !it[2].pinned, "items: a RAM expert streams to the stage");
    check(it[4].can == kStageStream && it[4].pinned, "items: a pinned RAM expert streams too");
    for (const Item& x : it)
        check(((x.can & kTierVram) != 0) == (held[(size_t) x.expert].holder == Holder::Tier), "items: only tier-held experts can stay on the tier");
    for (const Item& x : it) check((x.can & kTierStream) == 0, "items: without the tier's staging nothing streams to it");
    const std::vector<Item> st = layer_items(h, 2, 75, kTop, held, kBlob, true);
    check(st[2].can == (kStageStream | kTierStream) && st[4].can == (kStageStream | kTierStream), "items: with it a RAM expert may stream to the tier");
    check(st[0].can == it[0].can && st[1].can == it[1].can && st[3].can == it[3].can && st[5].can == it[5].can,
          "items: the stage's and the tier's own experts are as before");
}

void idle_copies() {
    const Rates r{0.001, 0.0005, 2e-8, 1e-8, 1e-7, 5e-8, 0.5, 0.2, 0, 5e-8};
    RouteHistory h(kLayers, kExperts);
    const int32_t cnt[kExperts] = {0, 0, 0, 10, 0, 5};
    h.add(0, cnt, 10);
    const std::vector<Held> held = {at(Holder::Ram), at(Holder::Ram, true), at(Holder::Tier), at(Holder::Ram), at(Holder::Stage),
                                    at(Holder::Tier)};
    const std::vector<Item> it = layer_items(h, 0, 10, kTop, held, kBlob, false);
    // experts 0 and 1 are in RAM with no rows: the streamed walk copies them anyway, the staged walk does not
    check(near(idle_copies_ms(it, held, true, r), (double) kBlob * (2e-8 + 1e-8), 1e-12), "idle: the streamed walk copies RAM experts no row needs");
    check(idle_copies_ms(it, held, false, r) == 0, "idle: the staged walk copies only what is routed");
}

void take_from_choice() {
    const std::vector<Held> held = {at(Holder::Tier), at(Holder::Stage), at(Holder::Tier), at(Holder::Ram), at(Holder::Tier),
                                    at(Holder::Tier)};
    Choice c;
    c.where = {Where::TierVram, Where::StageVram, Where::StageStream, Where::StageStream, Where::TierVram, Where::StageStream};
    c.stage_ms = c.tier_ms = 0;
    std::vector<uint8_t> take(kExperts, 9);
    const Counts n = fill_take(held, c, take.data());
    const std::vector<uint8_t> want = {1, 0, 0, 0, 1, 0};
    check(take == want, "take: 1 where a tier-held expert stays on the tier, 0 everywhere else");
    check(n.held == 4 && n.moved == 2, "take: four tier-held experts, two of them moved to the stage");
    check(n.streamed == 0, "take: nothing streamed to the tier");
    std::vector<Held> none(kExperts, at(Holder::Ram));
    const Counts z = fill_take(none, c, take.data());
    check(z.held == 0 && z.moved == 0 && take == std::vector<uint8_t>(kExperts, 0), "take: nothing tier-held, nothing taken");
    Choice s;
    s.where = {Where::TierVram, Where::StageVram, Where::StageStream, Where::TierStream, Where::TierVram, Where::StageStream};
    s.stage_ms = s.tier_ms = 0;
    const Counts m = fill_take(held, s, take.data());
    const std::vector<uint8_t> streamed = {kTakeTier, kTakeStage, kTakeStage, kTakeStream, kTakeTier, kTakeStage};
    check(take == streamed, "take: 2 where a RAM expert streams to the tier");
    check(m.held == 4 && m.moved == 2 && m.streamed == 1, "take: one expert streamed to the tier, the counts of the held ones as before");
}

void routed_counts() {
    const std::vector<Held> held = {at(Holder::Tier), at(Holder::Stage), at(Holder::Tier), at(Holder::Ram), at(Holder::Tier),
                                    at(Holder::Tier)};
    const int32_t cnt[kExperts] = {7, 100, 11, 200, 0, 13};
    const uint8_t take[kExperts] = {1, 0, 0, 0, 0, 1};
    const Counts n = routed_rows(cnt, held, take);
    check(n.held == 31, "routed: the rows of tier-held experts, routed or not");
    check(n.moved == 11, "routed: of them, the rows the stage computed");
    check(n.streamed == 0, "routed: none streamed to the tier");
    const uint8_t streamed[kExperts] = {1, 0, 0, 2, 0, 1};
    const Counts s = routed_rows(cnt, held, streamed);
    check(s.held == 31 && s.moved == 11 && s.streamed == 200, "routed: the rows of the RAM experts streamed to the tier");
}

void plans_a_layer() {
    // the tier is slower per row: the busiest tier-held experts move to the stage while the later finish drops
    const Rates r{0.001, 0.002, 1e-9, 1e-9, 1e-7, 5e-8, 0.5, 0.2, 0, 5e-8};
    RouteHistory h(kLayers, kExperts);
    const int32_t cnt[kExperts] = {300, 200, 100, 50, 40, 10};
    h.add(1, cnt, 350);
    const std::vector<Held> held = {at(Holder::Tier), at(Holder::Tier), at(Holder::Tier), at(Holder::Tier), at(Holder::Stage),
                                    at(Holder::Ram)};
    std::vector<uint8_t> take(kExperts, 9);
    const LayerPlan p = plan_stage_layer(h, 1, 350, kTop, held, kBlob, false, r, 0.0, 0, take.data());
    // home: the tier 0.5 + 650 x 0.002 = 1.8 ms, then moving 300 and 200 rows gives 0.76 and 0.8, a third would end the stage at 0.86
    check(p.held == 4 && p.moved == 2, "plan: the two busiest of four tier-held experts move");
    check(take[0] == 0 && take[1] == 0 && take[2] == 1 && take[3] == 1, "plan: the busiest move first and the quietest stay");
    check(near(p.stage_ms, 0.759, 1e-9) && near(p.tier_ms, 0.8, 1e-9), "plan: the finish times of the plan");
    check(take[4] == 0 && take[5] == 0, "plan: experts the tier does not hold are never taken");
    int64_t moved = 0;
    for (int e = 0; e < 4; ++e) moved += take[(size_t) e] == 0;
    check(moved == p.moved, "plan: the count of moved experts is the take's");
    check(p.stage_ms > 0 && p.tier_ms > 0, "plan: both devices end with work");

    // the same layer with the tier far ahead of the stage keeps every expert there
    const Rates fast{0.1, 0.0001, 1e-6, 1e-6, 1e-7, 5e-8, 0.1, 0.2, 0, 5e-8};
    const LayerPlan s = plan_stage_layer(h, 1, 350, kTop, held, kBlob, false, fast, 0.0, 0, take.data());
    check(s.moved == 0 && take[0] == 1 && take[1] == 1 && take[2] == 1 && take[3] == 1, "plan: a fast tier keeps all its experts");

    // queued work on the tier moves more to the stage
    const LayerPlan q = plan_stage_layer(h, 1, 350, kTop, held, kBlob, false, r, 5.0, 0, take.data());
    check(q.moved >= p.moved, "plan: a busy tier gives the stage at least as much");
}

void plan_counts_idle_copies() {
    // the streamed walk's copies of RAM experts no token routes to count before the plan, so less moves to the stage
    const Rates r{0.001, 0.004, 2e-7, 2e-7, 1e-7, 5e-8, 0.5, 0.2, 0, 5e-8};
    RouteHistory h(kLayers, kExperts);
    const int32_t cnt[kExperts] = {400, 300, 0, 0, 0, 0};
    h.add(0, cnt, 350);
    const std::vector<Held> held = {at(Holder::Tier), at(Holder::Tier), at(Holder::Ram), at(Holder::Ram), at(Holder::Ram),
                                    at(Holder::Ram)};
    std::vector<uint8_t> take(kExperts);
    const LayerPlan staged = plan_stage_layer(h, 0, 350, kTop, held, kBlob, false, r, 0.0, 0, take.data());
    // staged: the stage takes the 400 row expert (0.2 + 0.4 + 0.6 ms) against the tier's 0.5 + 1.2
    check(staged.moved == 1 && take[0] == 0 && near(staged.stage_ms, 1.2, 1e-9), "idle plan: the staged walk moves the busiest");
    const LayerPlan streamed = plan_stage_layer(h, 0, 350, kTop, held, kBlob, true, r, 0.0, 0, take.data());
    // streamed: four idle copies of 0.6 ms each start the stage at 2.4 ms, and 3.6 would be later than the tier's 3.3
    check(streamed.moved == 0 && near(streamed.stage_ms, 2.4, 1e-9) && near(streamed.tier_ms, 3.3, 1e-9),
          "idle plan: the streamed walk's idle copies keep both on the tier");
}

void plans_streams_to_tier() {
    // a stage that streams from RAM at 5 GB/s and an idle tier that streams at 25: the busiest RAM experts go to the tier, within its slots
    const Rates r{0.001, 0.001, 2e-7, 2e-7, 4e-8, 4e-8, 0.5, 0.2, 0.05, 5e-8};
    RouteHistory h(kLayers, kExperts);
    const int32_t cnt[kExperts] = {300, 250, 200, 150, 100, 50};
    h.add(0, cnt, 525);
    const std::vector<Held> held = {at(Holder::Ram), at(Holder::Ram), at(Holder::Ram), at(Holder::Ram), at(Holder::Stage),
                                    at(Holder::Tier)};
    std::vector<uint8_t> take(kExperts, 9);
    const LayerPlan p = plan_stage_layer(h, 0, 525, kTop, held, kBlob, true, r, 0.0, 2, take.data());
    check(p.streamed == 2 && take[0] == kTakeStream && take[1] == kTakeStream, "stream plan: the two busiest RAM experts fill the two slots");
    check(take[2] == kTakeStage && take[3] == kTakeStage && take[4] == kTakeStage && take[5] == kTakeTier,
          "stream plan: the others stay where they were");
    const LayerPlan w = plan_stage_layer(h, 0, 525, kTop, held, kBlob, true, r, 0.0, 6, take.data());
    check(w.streamed >= 2 && w.streamed <= 4, "stream plan: more slots, more go, never the stage's own");
    check(std::max(w.stage_ms, w.tier_ms) <= std::max(p.stage_ms, p.tier_ms), "stream plan: and the layer ends no later");
    const LayerPlan none = plan_stage_layer(h, 0, 525, kTop, held, kBlob, true, r, 0.0, 0, take.data());
    check(none.streamed == 0 && std::count(take.begin(), take.end(), kTakeStream) == 0, "stream plan: no slots, nothing streams to the tier");
    const LayerPlan staged = plan_stage_layer(h, 0, 525, kTop, held, kBlob, false, r, 0.0, 6, take.data());
    check(staged.streamed == 0 && std::count(take.begin(), take.end(), kTakeStream) == 0,
          "stream plan: the staged walk streams nothing to the tier");
    const LayerPlan busy = plan_stage_layer(h, 0, 525, kTop, held, kBlob, true, r, 1000.0, 6, take.data());
    check(busy.streamed == 0, "stream plan: a tier busy with the other stages gets none");
}

void tier_start() {
    // the time from a call queued to the tier's x seen, measured on each call: a decayed mean, and the plan's tier load from it
    TierStart t;
    check(t.ms() == 0 && t.ahead(1.5) == 1.5, "start: before any call the plan takes the predicted queue");
    t.observe(8.0);
    check(t.ms() == 8.0, "start: the first call sets it");
    t.observe(4.0);
    check(near(t.ms(), 7.0, 1e-12), "start: then each call counts a quarter");
    t.observe(std::nan(""));
    t.observe(-1.0);
    t.observe(HUGE_VAL);
    check(near(t.ms(), 7.0, 1e-12), "start: a value that is not a time is dropped");
    check(t.ahead(1.5) == t.ms() && t.ahead(9.0) == 9.0, "start: the plan takes the larger of the predicted queue and the measured start");

    // a tier that starts a layer 7 ms after the call is given fewer RAM experts than one the plan thought starts at once
    const Rates r{0.001, 0.001, 2e-7, 2e-7, 4e-8, 4e-8, 0.5, 0.2, 0.05, 5e-8};
    RouteHistory h(kLayers, kExperts);
    const int32_t cnt[kExperts] = {300, 250, 200, 150, 100, 50};
    h.add(0, cnt, 525);
    const std::vector<Held> held(kExperts, at(Holder::Ram));
    std::vector<uint8_t> take(kExperts);
    const LayerPlan now = plan_stage_layer(h, 0, 525, kTop, held, kBlob, true, r, TierStart().ahead(0.0), 6, take.data());
    const LayerPlan late = plan_stage_layer(h, 0, 525, kTop, held, kBlob, true, r, t.ahead(0.0), 6, take.data());
    std::printf("  start: %lld RAM experts to a tier that starts at once, %lld to one that starts %.1f ms late\n", (long long) now.streamed,
                (long long) late.streamed, t.ms());
    check(now.streamed > 0 && late.streamed < now.streamed, "start: a later start moves fewer experts to the tier");
}

void x_counted_once() {
    // the stage's copy of x to RAM is part of the measured start, so a call with one counts only the tier's read of x
    const Rates r{0.001, 0.001, 2e-7, 3e-7, 4e-8, 4e-8, 0.5, 0.2, 0.05, 5e-8};
    const uint64_t x = 1'000'000;
    const Rates both = tier_call_rates(r, x, false);
    check(near(both.tier_fixed_ms, 0.5 + 1e6 * (5e-8 + 3e-7), 1e-12), "x once: before a measured start a call counts x out and in");
    const Rates in = tier_call_rates(r, x, true);
    check(near(in.tier_fixed_ms, 0.5 + 1e6 * 5e-8, 1e-12), "x once: with one it counts only the tier's read");
    check(in.stage_byte_ms_pinned == r.stage_byte_ms_pinned && in.tier_row_ms == r.tier_row_ms && in.stage_fixed_ms == r.stage_fixed_ms,
          "x once: the other rates stay");
    TierStart t;
    check(!t.measured(), "x once: no start before a call");
    t.observe(6.0);
    check(t.measured(), "x once: a call measured one");

    // a layer the tier takes whole: its planned end is the measured start, then its call, with x out not counted a second time
    RouteHistory h(kLayers, kExperts);
    const int32_t cnt[kExperts] = {300, 250, 200, 150, 100, 50};
    h.add(0, cnt, 525);
    const std::vector<Held> held(kExperts, at(Holder::Tier));
    std::vector<uint8_t> take(kExperts);
    const Rates slow_stage{1.0, 0.001, 2e-7, 3e-7, 4e-8, 4e-8, 0.5, 0.2, 0.05, 5e-8};
    const LayerPlan once = plan_stage_layer(h, 0, 525, kTop, held, kBlob, true, tier_call_rates(slow_stage, x, t.measured()), t.ahead(0.0), 0,
                                            take.data());
    const double call = 0.5 + 1e6 * 5e-8 + 1050 * 0.001 + kExperts * 0.05;
    check(once.moved == 0 && near(once.tier_ms, 6.0 + call, 1e-9), "x once: the tier's end is its measured start and the call");
}

// a chunk's layer as two token halves of 350 tokens each, from a history of a 700 token chunk: each half gets half of these rows
const int32_t kHalfCnt[kExperts] = {600, 400, 200, 100, 80, 20};

void stage_calls() {
    // each stage keeps its two latest calls on the tier: another stage's queue counts both until their done words pass, a stage's own never
    StageCalls calls;
    calls.stages(3);
    uint32_t done[3] = {0, 0, 0};
    const auto ahead = [](const std::vector<Queued>& q) {
        double ms = 0;
        for (const Queued& c : q) ms += c.ms;
        return ms;
    };
    check(calls.queue(0, StageCalls::Call{&done[0], 1, 0.0, 2.0}).empty(), "calls: an idle tier has nothing ahead");
    check(calls.queue(0, StageCalls::Call{&done[0], 2, 1.0, 3.0}).empty(), "calls: a stage's own first half is not ahead of its second");
    const std::vector<Queued> q1 = calls.queue(1, StageCalls::Call{&done[1], 1, 2.0, 4.0});
    check(q1.size() == 2 && ahead(q1) == 5.0, "calls: both of another stage's calls are ahead");
    check(calls.queue(0, StageCalls::Call{&done[0], 3, 3.0, 1.0}).size() == 1, "calls: the other stage's call is ahead of a third");
    done[0] = 2;   // the tier is through stage 0's first two calls
    const std::vector<Queued> q2 = calls.queue(2, StageCalls::Call{&done[2], 1, 4.0, 1.0});
    check(q2.size() == 2 && ahead(q2) == 5.0, "calls: a stage's third call takes its oldest entry, and a done call is not ahead");
    done[1] = 1;
    check(ahead(calls.queue(2, StageCalls::Call{&done[2], 2, 5.0, 1.0})) == 1.0, "calls: only stage 0's third call is ahead now");
    (void) calls.queue(0, StageCalls::Call{&done[0], 4, 5.5, 1.0});   // both of stage 0's entries hold a call the tier has not done
    calls.clear(0);
    check(calls.queue(1, StageCalls::Call{&done[1], 2, 6.0, 1.0}).size() == 2, "calls: a stage that cleared has none ahead, the other stage's two are");
    done[2] = 2;
    check(calls.queue(1, StageCalls::Call{&done[1], 3, 7.0, 1.0}).empty(), "calls: none ahead once the other stages' calls are through");
    (void) calls.queue(0, StageCalls::Call{&done[0], 0xffffffffu, 8.0, 1.0});
    done[0] = 0;   // seqs wrap: a done word just past zero is through a call at the top of the range
    check(calls.queue(1, StageCalls::Call{&done[1], 4, 9.0, 1.0}).empty(), "calls: a wrapped done word passes the seq");
}

void ram_counts() {
    // a stage's weight for its part of the card's ring: the experts of its layers that neither it nor the tier holds
    const auto who = [](int64_t l, int64_t e) { return e < l ? Holder::Stage : e < 2 * l + 1 ? Holder::Tier : Holder::Ram; };
    // the stage, tier and RAM experts: layer 0 has 0, 1 and 5, layer 1 has 1, 2 and 3, layer 2 has 2, 3 and 1
    check(ram_experts(0, 3, kExperts, who) == 9 && ram_experts(1, 3, kExperts, who) == 4 && ram_experts(2, 2, kExperts, who) == 0,
          "ram counts: a stage's experts held neither by it nor by the tier");
    check(ram_experts(0, 3, kExperts, [](int64_t, int64_t) { return Holder::Stage; }) == 0, "ram counts: a stage that holds every expert has none");
    // the most RAM experts one layer holds, which sizes the stage's staging ring
    check(ram_widest(0, 3, kExperts, who) == 5 && ram_widest(1, 3, kExperts, who) == 3 && ram_widest(2, 3, kExperts, who) == 1,
          "ram counts: the widest layer's RAM experts");
    check(ram_widest(2, 2, kExperts, who) == 0 && ram_widest(0, 3, kExperts, [](int64_t, int64_t) { return Holder::Tier; }) == 0,
          "ram counts: no layers or no RAM experts give none");
}

void halves_pay_two_calls() {
    // every expert on the tier and a stage too slow to take any: each half's call pays the tier's fixed and per expert cost
    const Rates r{1.0, 0.001, 2e-7, 2e-7, 4e-8, 4e-8, 0.5, 0.2, 0.05, 5e-8};
    RouteHistory h(kLayers, kExperts);
    h.add(0, kHalfCnt, 700);
    const std::vector<Held> held(kExperts, at(Holder::Tier));
    std::vector<uint8_t> ta(kExperts, 9), tb(kExperts, 9);
    const HalvesPlan p = plan_stage_halves(h, 0, 350, 350, kTop, held, kBlob, true, r, HalfStarts{0.0, 0.0, 0.0}, 0, ta.data(), tb.data());
    // a call: 0.5 + 700 rows x 0.001 + 6 experts x 0.05 = 1.5 ms, and B's starts after A's
    check(near(p.a.tier_ms, 1.5, 1e-9) && near(p.b.tier_ms, 3.0, 1e-9), "halves: each call pays the tier's fixed and per expert cost");
    check(p.a.stage_ms == 0 && p.b.stage_ms == 0 && p.a.moved == 0 && p.b.moved == 0, "halves: a stage too slow takes nothing");
    check(std::count(ta.begin(), ta.end(), kTakeTier) == kExperts && std::count(tb.begin(), tb.end(), kTakeTier) == kExperts,
          "halves: both halves leave every expert on the tier");
    const HalvesPlan q = plan_stage_halves(h, 0, 350, 350, kTop, held, kBlob, true, r, HalfStarts{2.0, 0.0, 2.0}, 0, ta.data(), tb.data());
    check(near(q.a.tier_ms, 3.5, 1e-9) && near(q.b.tier_ms, 5.0, 1e-9), "halves: work queued ahead delays both calls");
    const HalvesPlan l = plan_stage_halves(h, 0, 350, 350, kTop, held, kBlob, true, r, HalfStarts{0.0, 5.0, 0.0}, 0, ta.data(), tb.data());
    check(near(l.a.tier_ms, 1.5, 1e-9) && near(l.b.tier_ms, 6.5, 1e-9), "halves: B's call goes out once the stage reaches the layer's experts");
}

void halves_stream_once() {
    // A streams the two busiest RAM experts to the tier, and B reads the tier's copies of them without their bytes
    const Rates r{0.001, 0.001, 2e-7, 2e-7, 4e-8, 4e-8, 0.5, 0.2, 0.05, 5e-8};
    RouteHistory h(kLayers, kExperts);
    h.add(0, kHalfCnt, 700);
    const std::vector<Held> held = {at(Holder::Ram), at(Holder::Ram), at(Holder::Ram), at(Holder::Ram), at(Holder::Stage), at(Holder::Tier)};
    std::vector<uint8_t> ta(kExperts, 9), tb(kExperts, 9);
    const HalvesPlan p = plan_stage_halves(h, 0, 350, 350, kTop, held, kBlob, true, r, HalfStarts{0.0, 0.0, 0.0}, 2, ta.data(), tb.data());
    // A: stage 3.29 and tier 0.56, then the 300 and 200 row experts to the tier at 0.05 + 0.12 each: stage 1.59, tier 1.4
    check(ta[0] == kTakeStream && ta[1] == kTakeStream && ta[2] == kTakeStage && ta[3] == kTakeStage && ta[4] == kTakeStage && ta[5] == kTakeTier,
          "halves stream: A streams the two busiest RAM experts");
    check(p.a.streamed == 2 && near(p.a.stage_ms, 1.59, 1e-9) && near(p.a.tier_ms, 1.4, 1e-9), "halves stream: A's ends");
    // B: the stage adds 0.1 + 0.05 + 0.04 with no fixed cost and no bytes, the tier 0.5 + 0.35 + 0.25 + 0.06 with no bytes
    check(tb[0] == kTakeStream && tb[1] == kTakeStream && tb[2] == kTakeStage && tb[3] == kTakeStage && tb[4] == kTakeStage && tb[5] == kTakeTier,
          "halves stream: B reads the copies A streamed and streams nothing more");
    check(p.b.streamed == 2 && near(p.b.stage_ms, 1.78, 1e-9) && near(p.b.tier_ms, 2.56, 1e-9),
          "halves stream: B's call pays no bytes for A's copies, and the stage none for its own");
    const HalvesPlan none = plan_stage_halves(h, 0, 350, 350, kTop, held, kBlob, true, r, HalfStarts{0.0, 0.0, 0.0}, 0, ta.data(), tb.data());
    check(none.a.streamed == 0 && none.b.streamed == 0 && std::count(tb.begin(), tb.end(), kTakeStream) == 0,
          "halves stream: without slots neither half streams, B no more than A");
}

void halves_lead() {
    // A's call reaches the tier half a millisecond before the stage starts the layer's experts, so A keeps more on the tier
    const Rates r{0.001, 0.002, 1e-9, 1e-9, 1e-7, 5e-8, 0.5, 0.2, 0, 5e-8};
    RouteHistory h(kLayers, kExperts);
    h.add(1, kHalfCnt, 700);
    const std::vector<Held> held = {at(Holder::Tier), at(Holder::Tier), at(Holder::Tier), at(Holder::Tier), at(Holder::Stage), at(Holder::Ram)};
    std::vector<uint8_t> ta(kExperts), tb(kExperts);
    const HalvesPlan now = plan_stage_halves(h, 1, 350, 350, kTop, held, kBlob, false, r, HalfStarts{0.0, 0.0, 0.0}, 0, ta.data(), tb.data());
    check(now.a.moved == 2 && near(now.a.stage_ms, 0.759, 1e-9) && near(now.a.tier_ms, 0.8, 1e-9), "halves lead: without a lead A moves as a chunk would");
    const HalvesPlan lead = plan_stage_halves(h, 1, 350, 350, kTop, held, kBlob, false, r, HalfStarts{0.0, 0.5, 0.0}, 0, ta.data(), tb.data());
    // the stage starts at 0.5: moving the 300 row expert ends it at 1.056 against the tier's 1.2, the next one would end it at 1.259
    check(lead.a.moved == 1 && ta[0] == kTakeStage && ta[1] == kTakeTier && near(lead.a.stage_ms, 1.056, 1e-9),
          "halves lead: the tier's head start keeps more of A on it");
}

void halves_stage_fixed_once() {
    // the stage computes both halves' experts at once: its fixed cost is paid by the first half that gives it rows
    const Rates r{0.001, 0.0005, 1e-9, 1e-9, 1e-7, 5e-8, 0.5, 1.0, 0, 5e-8};
    RouteHistory h(kLayers, kExperts);
    h.add(2, kHalfCnt, 700);
    const std::vector<Held> own(kExperts, at(Holder::Stage));
    std::vector<uint8_t> ta(kExperts), tb(kExperts);
    const HalvesPlan s = plan_stage_halves(h, 2, 350, 350, kTop, own, kBlob, false, r, HalfStarts{0.0, 0.0, 0.0}, 0, ta.data(), tb.data());
    check(near(s.a.stage_ms, 1.7, 1e-9) && near(s.b.stage_ms, 2.4, 1e-9) && s.b.tier_ms == 0, "halves fixed: the stage's own experts pay it once");
    // all on the tier: A keeps them there, and B, whose call waits for A's, gives the stage two, paying the fixed cost A did not
    const std::vector<Held> tier(kExperts, at(Holder::Tier));
    const HalvesPlan t = plan_stage_halves(h, 2, 350, 350, kTop, tier, kBlob, false, r, HalfStarts{0.0, 0.0, 0.0}, 0, ta.data(), tb.data());
    check(t.a.moved == 0 && near(t.a.tier_ms, 0.85, 1e-9) && t.a.stage_ms == 0, "halves fixed: A keeps every expert on the tier");
    check(t.b.moved == 2 && tb[0] == kTakeStage && tb[1] == kTakeStage && tb[2] == kTakeTier, "halves fixed: B's later call moves the busiest");
    check(near(t.b.stage_ms, 1.506, 1e-9) && near(t.b.tier_ms, 1.45, 1e-9), "halves fixed: B pays the stage's fixed cost once");
    // A of one token gives the stage's own expert no rows: B, whose rows it has, pays the fixed cost (A's tier 0.5 + 2 rows, B's from there)
    std::vector<Held> late = tier;
    late[4] = at(Holder::Stage);
    const HalvesPlan u = plan_stage_halves(h, 2, 1, 699, kTop, late, kBlob, false, r, HalfStarts{0.0, 0.0, 0.0}, 0, ta.data(), tb.data());
    check(u.a.stage_ms == 0 && near(u.a.tier_ms, 0.501, 1e-9) && near(u.b.stage_ms, 1.08, 1e-9) && near(u.b.tier_ms, 1.66, 1e-9) && u.b.moved == 0,
          "halves fixed: an expert of the stage with no rows in A leaves the fixed cost to B");
}

void halves_b_start() {
    // B's call measured to start b_ahead_ms after it is queued with the stage's experts: the tier takes it then, or once A is through if later
    const Rates r{1.0, 0.001, 2e-7, 2e-7, 4e-8, 4e-8, 0.5, 0.2, 0.05, 5e-8};
    RouteHistory h(kLayers, kExperts);
    h.add(0, kHalfCnt, 700);
    const std::vector<Held> held(kExperts, at(Holder::Tier));
    std::vector<uint8_t> ta(kExperts), tb(kExperts);
    // each call 1.5 ms: A ends at 1.5, B queued with the stage's experts at 5 starts 4 ms later, at 9
    const HalvesPlan late = plan_stage_halves(h, 0, 350, 350, kTop, held, kBlob, true, r, HalfStarts{0.0, 5.0, 4.0}, 0, ta.data(), tb.data());
    check(near(late.a.tier_ms, 1.5, 1e-9) && near(late.b.tier_ms, 10.5, 1e-9), "halves b start: B starts its own measured start after the stage's experts");
    const HalvesPlan behind = plan_stage_halves(h, 0, 350, 350, kTop, held, kBlob, true, r, HalfStarts{3.0, 0.5, 1.0}, 0, ta.data(), tb.data());
    check(near(behind.a.tier_ms, 4.5, 1e-9) && near(behind.b.tier_ms, 6.0, 1e-9), "halves b start: A's end binds when it is later");
    check(near(planned_times(late, HalfStarts{0.0, 5.0, 4.0}).b_start, 4.0, 1e-12), "halves b start: the planned times show B's own start");
    // a stage that can take experts: the later B starts, the more of B it computes
    const Rates s{0.001, 0.0005, 1e-9, 1e-9, 1e-7, 5e-8, 0.5, 1.0, 0, 5e-8};
    RouteHistory h2(kLayers, kExperts);
    h2.add(2, kHalfCnt, 700);
    const HalvesPlan at_a = plan_stage_halves(h2, 2, 350, 350, kTop, held, kBlob, false, s, HalfStarts{0.0, 0.0, 0.0}, 0, ta.data(), tb.data());
    check(at_a.b.moved == 2, "halves b start: B right after A moves the two busiest");
    // B starts at 1.2, after A's 0.85: the five busiest move, the stage 1.0 + 0.705 and the tier 1.2 + 0.5 + 10 x 0.0005
    const HalvesPlan later = plan_stage_halves(h2, 2, 350, 350, kTop, held, kBlob, false, s, HalfStarts{0.0, 0.0, 1.2}, 0, ta.data(), tb.data());
    check(later.b.moved == 5 && tb[4] == kTakeStage && tb[5] == kTakeTier && near(later.b.stage_ms, 1.705, 1e-9) && near(later.b.tier_ms, 1.705, 1e-9),
          "halves b start: B's later start moves more of B to the stage");
    check(later.a.moved == 0 && near(later.a.tier_ms, 0.85, 1e-9), "halves b start: A's plan is A's own");
}

void halves_times() {
    // the plan's times of a layer in halves from the stage's own experts' start: A's call queued at its routing, the stage's experts lead_ms later
    const Rates r{0.001, 0.0005, 1e-9, 1e-9, 1e-7, 5e-8, 0.5, 1.0, 0, 5e-8};
    RouteHistory h(kLayers, kExperts);
    h.add(2, kHalfCnt, 700);
    const std::vector<Held> tier(kExperts, at(Holder::Tier));
    std::vector<uint8_t> ta(kExperts), tb(kExperts);
    const HalvesPlan p = plan_stage_halves(h, 2, 350, 350, kTop, tier, kBlob, false, r, HalfStarts{2.0, 5.0, 2.0}, 0, ta.data(), tb.data());
    const HalfTimes t = planned_times(p, HalfStarts{2.0, 5.0, 2.0});
    // A starts 2 ms after its routing and ends at its call's end, B starts once A is done or at the lead plus the start, whichever is later
    check(near(t.a_start, -3.0, 1e-12) && near(t.a_end, p.a.tier_ms - 5.0, 1e-12), "times: A's call from the stage's experts' start");
    check(near(t.b_start, std::max(p.a.tier_ms, 7.0) - 5.0, 1e-12) && near(t.b_end, p.b.tier_ms - 5.0, 1e-12), "times: B's call after A's and its own");
    check(near(t.stage_end, p.b.stage_ms - 5.0, 1e-12), "times: the stage's end");
    HalfTimes sum;
    sum += t;
    sum += t;
    check(near(sum.b_end, 2 * t.b_end, 1e-12) && near(sum.a_start, 2 * t.a_start, 1e-12), "times: layers add up for a chunk's mean");
}

void slack_whole() {
    // a stage with slack takes tier work, the most tier ms per stage ms first, each move that ends it by the plan's end plus the slack
    const Rates r{0.001, 0.002, 1e-9, 1e-9, 1e-7, 5e-8, 0.5, 0.2, 0, 5e-8};
    RouteHistory h(kLayers, kExperts);
    const int32_t cnt[kExperts] = {300, 200, 100, 50, 40, 10};
    h.add(1, cnt, 350);
    const std::vector<Held> held = {at(Holder::Tier), at(Holder::Tier), at(Holder::Tier), at(Holder::Tier), at(Holder::Stage),
                                    at(Holder::Ram)};
    std::vector<uint8_t> take(kExperts, 9);
    const LayerPlan none = plan_stage_layer(h, 1, 350, kTop, held, kBlob, false, r, 0.0, 0, take.data(), 0.0);
    check(none.moved == 2 && take[2] == kTakeTier && take[3] == kTakeTier && near(none.stage_ms, 0.759, 1e-9) && near(none.tier_ms, 0.8, 1e-9),
          "slack whole: none gives today's plan");
    // the layer ends at 0.8 on the tier: 0.07 more lets the stage take the 100 row expert (0.862), the 50 row one would end it at 0.915
    const LayerPlan part = plan_stage_layer(h, 1, 350, kTop, held, kBlob, false, r, 0.0, 0, take.data(), 0.07);
    check(part.moved == 3 && take[2] == kTakeStage && take[3] == kTakeTier && near(part.stage_ms, 0.862, 1e-9) && near(part.tier_ms, 0.6, 1e-9),
          "slack whole: a part of it takes the expert that saves the tier the most for its stage ms");
    // with 0.06 the 100 row expert would end the stage at 0.862, past 0.86: it stays and the 50 row one moves
    const LayerPlan skip = plan_stage_layer(h, 1, 350, kTop, held, kBlob, false, r, 0.0, 0, take.data(), 0.06);
    check(skip.moved == 3 && take[2] == kTakeTier && take[3] == kTakeStage && near(skip.stage_ms, 0.812, 1e-9) && near(skip.tier_ms, 0.7, 1e-9),
          "slack whole: an expert past the end is skipped and a smaller one still moves");
    const LayerPlan all = plan_stage_layer(h, 1, 350, kTop, held, kBlob, false, r, 0.0, 0, take.data(), 1.0);
    check(all.moved == 4 && std::count(take.begin(), take.end(), kTakeTier) == 0 && near(all.stage_ms, 0.915, 1e-9) && all.tier_ms == 0,
          "slack whole: enough of it moves every tier-held expert to the stage and gives the tier no call");
}

void slack_streams() {
    // the stream plan's tier: the two busiest RAM experts streamed to it and a tier-held one, given back to a stage with slack
    const Rates r{0.001, 0.001, 2e-7, 2e-7, 4e-8, 4e-8, 0.5, 0.2, 0.05, 5e-8};
    RouteHistory h(kLayers, kExperts);
    const int32_t cnt[kExperts] = {300, 250, 200, 150, 100, 50};
    h.add(0, cnt, 525);
    const std::vector<Held> held = {at(Holder::Ram), at(Holder::Ram), at(Holder::Ram), at(Holder::Ram), at(Holder::Stage),
                                    at(Holder::Tier)};
    std::vector<uint8_t> take(kExperts, 9);
    const LayerPlan none = plan_stage_layer(h, 0, 525, kTop, held, kBlob, true, r, 0.0, 2, take.data(), 0.0);
    // the stage 3.6 less 0.9 and 0.85, the tier 0.6 and 0.47 and 0.42 for the streamed ones
    check(none.streamed == 2 && take[0] == kTakeStream && take[1] == kTakeStream && take[5] == kTakeTier && near(none.stage_ms, 1.85, 1e-9) &&
              near(none.tier_ms, 1.49, 1e-9),
          "slack streams: none gives today's plan");
    // ends by 2.85: the 300 row expert comes back (0.47 of the tier for 0.9 of the stage), then neither the 250 row one nor the tier-held one fits
    const LayerPlan part = plan_stage_layer(h, 0, 525, kTop, held, kBlob, true, r, 0.0, 2, take.data(), 1.0);
    check(part.streamed == 1 && take[0] == kTakeStage && take[1] == kTakeStream && take[5] == kTakeTier && near(part.stage_ms, 2.75, 1e-9) &&
              near(part.tier_ms, 1.02, 1e-9),
          "slack streams: a streamed expert comes back to the stage first");
    // ends by 2.55: neither streamed one fits, the tier-held one does (2.5)
    const LayerPlan held_only = plan_stage_layer(h, 0, 525, kTop, held, kBlob, true, r, 0.0, 2, take.data(), 0.7);
    check(held_only.streamed == 2 && held_only.moved == 1 && take[5] == kTakeStage && near(held_only.stage_ms, 2.5, 1e-9) &&
              near(held_only.tier_ms, 1.39, 1e-9),
          "slack streams: a smaller tier-held expert moves where the streamed ones do not fit");
    const LayerPlan all = plan_stage_layer(h, 0, 525, kTop, held, kBlob, true, r, 0.0, 2, take.data(), 3.0);
    check(all.streamed == 0 && all.moved == 1 && std::count(take.begin(), take.end(), kTakeStage) == kExperts && near(all.stage_ms, 4.25, 1e-9) &&
              all.tier_ms == 0,
          "slack streams: enough of it streams nothing and moves every tier-held expert");
}

void slack_halves() {
    // halves_stream_once's layer: A streams the two busiest RAM experts and B reads their copies, then a stage with slack takes them back
    const Rates r{0.001, 0.001, 2e-7, 2e-7, 4e-8, 4e-8, 0.5, 0.2, 0.05, 5e-8};
    RouteHistory h(kLayers, kExperts);
    h.add(0, kHalfCnt, 700);
    const std::vector<Held> held = {at(Holder::Ram), at(Holder::Ram), at(Holder::Ram), at(Holder::Ram), at(Holder::Stage), at(Holder::Tier)};
    std::vector<uint8_t> ta(kExperts, 9), tb(kExperts, 9);
    const HalfStarts at0{0.0, 0.0, 0.0};
    const HalvesPlan none = plan_stage_halves(h, 0, 350, 350, kTop, held, kBlob, true, r, at0, 2, ta.data(), tb.data(), 0.0);
    check(ta[0] == kTakeStream && ta[1] == kTakeStream && tb[0] == kTakeStream && tb[1] == kTakeStream && tb[5] == kTakeTier &&
              near(none.a.stage_ms, 1.59, 1e-9) && near(none.a.tier_ms, 1.4, 1e-9) && near(none.b.stage_ms, 1.78, 1e-9) &&
              near(none.b.tier_ms, 2.56, 1e-9),
          "slack halves: none gives today's plan");
    // the layer ends at 2.56 on the tier: by 3.06 the 300 row expert comes back in both halves (the stage 2.98), the next would end it later
    const HalvesPlan part = plan_stage_halves(h, 0, 350, 350, kTop, held, kBlob, true, r, at0, 2, ta.data(), tb.data(), 0.5);
    check(ta[0] == kTakeStage && ta[1] == kTakeStream && tb[0] == kTakeStage && tb[1] == kTakeStream && tb[5] == kTakeTier,
          "slack halves: an expert comes back to the stage in both halves");
    check(near(part.a.stage_ms, 2.49, 1e-9) && near(part.a.tier_ms, 0.93, 1e-9) && near(part.b.stage_ms, 2.98, 1e-9) &&
              near(part.b.tier_ms, 1.74, 1e-9) && part.a.streamed == 1 && part.b.streamed == 1,
          "slack halves: B's call starts after A's shorter one, and the stage pays the bytes once");
    // by 2.86 the 300 row expert does not fit (2.98), the 200 row one does (2.78)
    const HalvesPlan skip = plan_stage_halves(h, 0, 350, 350, kTop, held, kBlob, true, r, at0, 2, ta.data(), tb.data(), 0.3);
    check(ta[0] == kTakeStream && ta[1] == kTakeStage && tb[0] == kTakeStream && tb[1] == kTakeStage && near(skip.a.stage_ms, 2.39, 1e-9) &&
              near(skip.a.tier_ms, 1.03, 1e-9) && near(skip.b.stage_ms, 2.78, 1e-9) && near(skip.b.tier_ms, 1.94, 1e-9),
          "slack halves: an expert past the end is skipped and the next still moves");
    const HalvesPlan all = plan_stage_halves(h, 0, 350, 350, kTop, held, kBlob, true, r, at0, 2, ta.data(), tb.data(), 3.0);
    check(std::count(ta.begin(), ta.end(), kTakeStage) == kExperts && std::count(tb.begin(), tb.end(), kTakeStage) == kExperts &&
              all.a.streamed == 0 && all.b.streamed == 0 && all.a.moved == 1 && all.b.moved == 1,
          "slack halves: enough of it streams nothing and moves every tier-held expert in both halves");
    check(near(all.a.stage_ms, 3.9, 1e-9) && all.a.tier_ms == 0 && near(all.b.stage_ms, 4.6, 1e-9) && all.b.tier_ms == 0,
          "slack halves: the tier has no call left");
}

void slack_halves_b_none() {
    // A of 699 tokens streams the two busiest RAM experts, B of one token is predicted no rows: taken back, they are the stage's in B as well
    const Rates r{0.001, 0.001, 2e-7, 2e-7, 4e-8, 4e-8, 0.5, 0.2, 0.05, 5e-8};
    RouteHistory h(kLayers, kExperts);
    const int32_t cnt[kExperts] = {300, 250, 200, 150, 100, 50};
    h.add(0, cnt, 700);
    const std::vector<Held> held = {at(Holder::Ram), at(Holder::Ram), at(Holder::Ram), at(Holder::Ram), at(Holder::Stage), at(Holder::Tier)};
    std::vector<uint8_t> ta(kExperts, 9), tb(kExperts, 9);
    (void) plan_stage_halves(h, 0, 699, 1, kTop, held, kBlob, true, r, HalfStarts{0.0, 0.0, 0.0}, 2, ta.data(), tb.data(), 0.0);
    check(ta[0] == kTakeStream && ta[1] == kTakeStream && tb[0] == kTakeStream && tb[1] == kTakeStream, "halves b none: B reads the copies A streams");
    (void) plan_stage_halves(h, 0, 699, 1, kTop, held, kBlob, true, r, HalfStarts{0.0, 0.0, 0.0}, 2, ta.data(), tb.data(), 5.0);
    check(ta[0] == kTakeStage && ta[1] == kTakeStage && tb[0] == kTakeStage && tb[1] == kTakeStage,
          "halves b none: an expert A no longer streams is the stage's in B too, with no B rows predicted");
}

// Today's shed before Task 15a, which reprices the whole layer for each move: the reference the shed by delta must match
namespace full {

constexpr bool on_tier(Where w) { return w == Where::TierVram || w == Where::TierStream; }

struct Ends { double stage = 0, tier = 0; };

template <class F>
void shed(const std::vector<Item>& ia, const std::vector<int32_t>& rows_b, std::vector<Where>& wa, std::vector<Where>& wb, double cap, F&& ends) {
    const auto move = [&](size_t e, std::vector<Where>& a, std::vector<Where>& b) {
        const bool streamed = a[e] == Where::TierStream;
        bool any = false;
        if (ia[e].rows > 0 && on_tier(a[e])) a[e] = Where::StageStream, any = true;
        if (!b.empty() && (rows_b[e] > 0 || streamed) && on_tier(b[e])) b[e] = Where::StageStream, any = true;
        return any;
    };
    const Ends now = ends(wa, wb);
    std::vector<std::pair<size_t, double>> order;
    for (size_t e = 0; e < ia.size(); ++e) {
        std::vector<Where> a = wa, b = wb;
        if (!move(e, a, b)) continue;
        const Ends m = ends(a, b);
        const double added = m.stage - now.stage;
        order.emplace_back(e, added > 0 ? (now.tier - m.tier) / added : HUGE_VAL);
    }
    std::stable_sort(order.begin(), order.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
    for (const auto& [e, per] : order) {
        std::vector<Where> a = wa, b = wb;
        move(e, a, b);
        if (ends(a, b).stage > cap) continue;
        wa = std::move(a);
        wb = std::move(b);
    }
}

void b_items(const std::vector<Item>& ib0, const std::vector<Item>& ia, const std::vector<Where>& wa, const std::vector<Held>& held, const Rates& r,
             std::vector<Item>& ib, Rates& rb) {
    ib = ib0;
    bool stage_paid = false;
    for (size_t e = 0; e < ib.size(); ++e) {
        const Where w = wa[e];
        stage_paid = stage_paid || (ia[e].rows > 0 && (w == Where::StageVram || w == Where::StageStream));
        if (w == Where::StageStream) {
            ib[e].bytes = 0;
            if (held[e].holder == Holder::Ram) ib[e].can = kStageStream;
        } else if (w == Where::TierStream) {
            ib[e].can = kTierVram | kStageStream;
        }
    }
    rb = r;
    if (stage_paid) rb.stage_fixed_ms = 0;
}

LayerPlan plan_stage_layer(const RouteHistory& h, int64_t l, int64_t T, int64_t k, const std::vector<Held>& held, uint64_t blob,
                           bool streamed, const Rates& r, double tier_ahead_ms, int tier_slots, uint8_t* take_row, double slack_ms) {
    const int slots = streamed ? tier_slots : 0;
    const std::vector<Item> items = layer_items(h, l, T, k, held, blob, slots > 0);
    const Load load{idle_copies_ms(items, held, streamed, r), tier_ahead_ms};
    Choice c = plan_layer(items, r, load, slots);
    if (slack_ms > 0) {
        std::vector<Where> w = std::move(c.where), none;
        shed(items, {}, w, none, std::max(c.stage_ms, c.tier_ms) + slack_ms, [&](const std::vector<Where>& p, const std::vector<Where>&) {
            const Choice e = place_layer(items, p, r, load);
            return Ends{e.stage_ms, e.tier_ms - load.tier_ms};
        });
        c = place_layer(items, w, r, load);
    }
    const Counts n = fill_take(held, c, take_row);
    return LayerPlan{c.stage_ms, c.tier_ms, n.held, n.moved, n.streamed};
}

HalvesPlan plan_stage_halves(const RouteHistory& h, int64_t l, int64_t ta, int64_t tb, int64_t k, const std::vector<Held>& held, uint64_t blob,
                             bool streamed, const Rates& r, const HalfStarts& at, int tier_slots, uint8_t* take_a, uint8_t* take_b,
                             double slack_ms) {
    const int slots = streamed ? tier_slots : 0;
    const std::vector<Item> ia = layer_items(h, l, ta, k, held, blob, slots > 0);
    const Load la{idle_copies_ms(ia, held, streamed, r) + at.lead_ms, at.ahead_ms};
    const std::vector<Item> ib0 = layer_items(h, l, tb, k, held, blob, false);
    std::vector<Item> ib;
    Rates rb;
    Choice ca = plan_layer(ia, r, la, slots);
    b_items(ib0, ia, ca.where, held, r, ib, rb);
    Choice cb = plan_layer(ib, rb, Load{ca.stage_ms, std::max(ca.tier_ms, at.lead_ms + at.b_ahead_ms)}, 0);
    if (slack_ms > 0) {
        std::vector<int32_t> rows_b(ib0.size());
        for (size_t e = 0; e < ib0.size(); ++e) rows_b[e] = ib0[e].rows;
        std::vector<Where> wa = std::move(ca.where), wb = std::move(cb.where);
        const auto ends = [&](const std::vector<Where>& pa, const std::vector<Where>& pb) {
            const Choice a = place_layer(ia, pa, r, la);
            b_items(ib0, ia, pa, held, r, ib, rb);
            const double b0 = std::max(a.tier_ms, at.lead_ms + at.b_ahead_ms);
            const Choice b = place_layer(ib, pb, rb, Load{a.stage_ms, b0});
            return Ends{b.stage_ms, (a.tier_ms - la.tier_ms) + (b.tier_ms - b0)};
        };
        shed(ia, rows_b, wa, wb, std::max(cb.stage_ms, cb.tier_ms) + slack_ms, ends);
        ca = place_layer(ia, wa, r, la);
        b_items(ib0, ia, wa, held, r, ib, rb);
        cb = place_layer(ib, wb, rb, Load{ca.stage_ms, std::max(ca.tier_ms, at.lead_ms + at.b_ahead_ms)});
    }
    const Counts na = fill_take(held, ca, take_a);
    Counts nb;
    for (size_t e = 0; e < held.size(); ++e) {
        const bool tier = held[e].holder == Holder::Tier, on_tier = cb.where[e] == Where::TierVram;
        take_b[e] = !on_tier ? kTakeStage : ca.where[e] == Where::TierStream ? kTakeStream : kTakeTier;
        nb.held += tier;
        nb.moved += tier && !on_tier;
        nb.streamed += take_b[e] == kTakeStream;
    }
    return HalvesPlan{LayerPlan{ca.stage_ms, ca.tier_ms, na.held, na.moved, na.streamed}, LayerPlan{cb.stage_ms, cb.tier_ms, nb.held, nb.moved, nb.streamed}};
}

}  // namespace full

bool plans_equal(const LayerPlan& x, const LayerPlan& y) {
    return x.stage_ms == y.stage_ms && x.tier_ms == y.tier_ms && x.held == y.held && x.moved == y.moved && x.streamed == y.streamed;
}

bool plans_near(const LayerPlan& x, const LayerPlan& y) {
    return near(x.stage_ms, y.stage_ms, 1e-12) && near(x.tier_ms, y.tier_ms, 1e-12) && x.held == y.held && x.moved == y.moved &&
           x.streamed == y.streamed;
}

void shed_by_delta() {
    // random layers, whole chunks and halves, 16 to 512 experts, slack from none to the whole gap: the shed by delta against today's
    std::mt19937 rng(59);
    const auto uni = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); };
    int64_t exact = 0, twin_layers = 0, shed_acts = 0;
    for (int trial = 0; trial < 4000; ++trial) {
        // half the layers give each routed expert its own rows, half are routed as the engine sees it, where experts of equal rows are common
        const bool twins = trial % 4 >= 2, halves = trial % 2 == 1, streamed = rng() % 3 != 0;
        const int64_t ne = trial < 8 ? 512 : 16 + (int64_t) (rng() % 497), T = 1024 + (int64_t) (rng() % 3073), k = 10;
        RouteHistory h(1, ne);
        std::vector<int32_t> cnt((size_t) ne);
        for (int64_t e = 0; e < ne; ++e) cnt[(size_t) e] = (int32_t) (3 * e);   // 3 apart, so each half's rows differ too
        std::shuffle(cnt.begin(), cnt.end(), rng);
        for (int32_t& c : cnt)
            if (twins) c = rng() % 5 == 0 ? 0 : (int32_t) std::exp(uni(0, 6.5));
        h.add(0, cnt.data(), T);
        std::vector<Held> held((size_t) ne);
        for (Held& x : held) {
            const uint32_t u = rng() % 10;
            x = Held{u < 2 ? Holder::Stage : u < 6 ? Holder::Tier : Holder::Ram, rng() % 4 == 0};
        }
        const uint64_t blob = 1'000'000 + rng() % 4'000'000;
        const double paged = uni(5e-8, 3e-7), tpaged = uni(3e-8, 2e-7);
        const Rates r{uni(3e-4, 2e-3), uni(2e-4, 1e-3), paged, paged * uni(0.4, 1.0), tpaged, tpaged * uni(0.5, 1.0), uni(0.1, 1.0), uni(0.05, 0.5),
                      uni(0.002, 0.05), uni(1e-8, 1e-7)};
        const int slots = (int) (rng() % (uint32_t) (ne / 4 + 1));
        // the whole gap: what the stage would add to take every routed expert, so the slack runs from none to all of it
        double gap = r.stage_fixed_ms;
        for (int64_t e = 0; e < ne; ++e) gap += (double) cnt[(size_t) e] * r.stage_row_ms + (double) blob * paged;
        const double u = uni(0, 1), slack = trial % 10 == 0 ? 0.0 : u * u * gap;
        std::vector<uint8_t> ta((size_t) ne, 9), tb((size_t) ne, 9), fa((size_t) ne, 9), fb((size_t) ne, 9), za((size_t) ne, 9), zb((size_t) ne, 9);
        bool same_plan, near_plan, as_slack0;
        if (!halves) {
            const double ahead = uni(0, 5);
            const LayerPlan p = plan_stage_layer(h, 0, T, k, held, blob, streamed, r, ahead, slots, ta.data(), slack);
            const LayerPlan f = full::plan_stage_layer(h, 0, T, k, held, blob, streamed, r, ahead, slots, fa.data(), slack);
            const LayerPlan z = plan_stage_layer(h, 0, T, k, held, blob, streamed, r, ahead, slots, za.data(), 0.0);
            same_plan = plans_equal(p, f);
            near_plan = plans_near(p, f);
            as_slack0 = plans_equal(p, z);
        } else {
            const HalfStarts at{uni(0, 5), uni(0, 20), uni(0, 5)};
            const HalvesPlan p = plan_stage_halves(h, 0, T / 2, T - T / 2, k, held, blob, streamed, r, at, slots, ta.data(), tb.data(), slack);
            const HalvesPlan f = full::plan_stage_halves(h, 0, T / 2, T - T / 2, k, held, blob, streamed, r, at, slots, fa.data(), fb.data(), slack);
            const HalvesPlan z = plan_stage_halves(h, 0, T / 2, T - T / 2, k, held, blob, streamed, r, at, slots, za.data(), zb.data(), 0.0);
            same_plan = plans_equal(p.a, f.a) && plans_equal(p.b, f.b);
            near_plan = plans_near(p.a, f.a) && plans_near(p.b, f.b);
            as_slack0 = plans_equal(p.a, z.a) && plans_equal(p.b, z.b);
        }
        shed_acts += !(as_slack0 && ta == za && tb == zb);
        if (!twins) {
            check(same_plan && ta == fa && tb == fb, "delta: a layer of distinct rows plans, takes and ends as today's full repricing");
            ++exact;
            continue;
        }
        // twins, experts of the same rows in each half, holder and pinning, may trade places: today's code picked by its sums' rounding
        std::map<std::tuple<int32_t, int32_t, int, bool, uint8_t, uint8_t>, int> n;
        for (int64_t e = 0; e < ne; ++e) {
            const int32_t ra = h.rows(0, e, halves ? T / 2 : T, k), rb = halves ? h.rows(0, e, T - T / 2, k) : 0;
            const auto key = [&](uint8_t a, uint8_t b) { return std::make_tuple(ra, rb, (int) held[(size_t) e].holder, held[(size_t) e].pinned, a, b); };
            ++n[key(ta[(size_t) e], tb[(size_t) e])];
            --n[key(fa[(size_t) e], fb[(size_t) e])];
        }
        const bool traded = std::all_of(n.begin(), n.end(), [](const auto& x) { return x.second == 0; });
        check(near_plan && traded, "delta: a layer with twins plans as today's, its twins may trade places");
        twin_layers += !(same_plan && ta == fa && tb == fb);
    }
    check(exact == 2000, "delta: two thousand layers of distinct rows");
    std::printf("shed_by_delta: the shed changed %lld of 4000 layers, %lld of the 2000 with twins differ from today's\n", (long long) shed_acts,
                (long long) twin_layers);
    check(shed_acts >= 3000, "delta: the shed moves experts in most layers, so the two sides do not match by doing nothing");
    check(twin_layers > 0 && twin_layers < 200, "delta: some layers with twins differ from today's, and only a few");
}

void stage_times() {
    // each stage's chunk time per token as a decayed mean, and the ms a layer a faster stage may add before it reaches the slowest
    StageTimes t;
    t.stages(3);
    check(t.slack(1, 8, 4096, 1, 3) == 0, "times: no slack before any chunk");
    for (int i = 0; i < 2; ++i) {
        t.observe(0, 0, 400.0, 4096, 4096);
        t.observe(1, 3, 600.0, 4096, 4096);
    }
    check(t.slack(1, 8, 4096, 1, 3) == 0 && t.slack(0, 3, 4096, 2, 2) == 0 && t.chunk_ms(2, 4096) == 0, "times: no slack while a stage is not measured");
    t.observe(2, 11, 1900.0, 4096, 4096);
    check(t.slack(1, 8, 4096, 1, 3) == 0, "times: a stage's first full chunk alone gives no slack");
    t.observe(2, 11, 1900.0, 4096, 4096);
    check(near(t.slack(1, 8, 4096, 1, 3), 650.0 / 8, 1e-12), "times: half the slowest stage's chunk less its own, over its layers");
    check(t.slack(2, 37, 4096, 1, 3) == 0, "times: the slowest stage has none");
    check(near(t.slack(1, 8, 2048, 1, 3), 325.0 / 8, 1e-12) && near(t.chunk_ms(2, 2048), 950.0, 1e-12), "times: per token, for the chunk's tokens");
    check(t.slack(1, 8, 4096, 0, 4) == 0, "times: none on the run's first chunk for the stage just before the slowest");
    check(t.slack(0, 3, 4096, 1, 3) == 0 && near(t.slack(0, 3, 4096, 2, 2), 250.0, 1e-12),
          "times: none on the first two for the stage two before it");
    t.observe(2, 11, 3000.0, 2048, 4096);
    check(near(t.slack(1, 8, 4096, 1, 3), 650.0 / 8, 1e-12) && near(t.chunk_ms(2, 4096), 1900.0, 1e-12),
          "times: a chunk shorter than its run's leaves the slack as it was");
    t.observe(2, 11, 1500.0, 4096, 4096);
    check(near(t.chunk_ms(2, 4096), 1800.0, 1e-12), "times: each later chunk counts a quarter");
    t.observe(2, 11, std::nan(""), 4096, 4096);
    t.observe(2, 11, -1.0, 4096, 4096);
    t.observe(2, 11, 100.0, 0, 4096);
    check(near(t.chunk_ms(2, 4096), 1800.0, 1e-12), "times: a value that is not a time is dropped");
    // the stages' order is their layers', not the order they joined in: here stage 0 runs the last layers and is the slowest
    StageTimes u;
    u.stages(3);
    for (int i = 0; i < 2; ++i) {
        u.observe(0, 20, 1000.0, 1000, 1000);
        u.observe(1, 0, 300.0, 1000, 1000);
        u.observe(2, 10, 500.0, 1000, 1000);
    }
    check(u.slack(2, 10, 1000, 0, 5) == 0 && near(u.slack(2, 10, 1000, 1, 5), 25.0, 1e-12) && u.slack(1, 5, 1000, 1, 5) == 0 &&
              near(u.slack(1, 5, 1000, 2, 5), 70.0, 1e-12),
          "times: a stage's place before the slowest goes by its first layer");
    u.observe(0, 0, 1000.0, 1000, 1000);   // the slowest runs the first layers now
    u.observe(1, 20, 300.0, 1000, 1000);
    check(near(u.slack(1, 5, 1000, 0, 2), 70.0, 1e-12) && u.slack(1, 5, 1000, 0, 1) == 0 && u.slack(2, 10, 1000, 4, 0) == 0 &&
              near(u.slack(2, 10, 1000, 4, 1), 25.0, 1e-12),
          "times: none on the run's last chunks for a stage after the slowest");
}

void stage_gap() {
    // the gap is taken low: the slowest stage's time is the lower of its mean and its last chunk, a stage's own the higher of its two
    StageTimes t;
    t.stages(2);
    t.observe(0, 0, 600.0, 4096, 4096);
    t.observe(0, 0, 600.0, 4096, 4096);
    t.observe(1, 3, 5000.0, 4096, 4096);   // a cold first chunk after a start
    t.observe(1, 3, 1900.0, 4096, 4096);
    const StageTimes::Gap g = t.gap(0, 4096);
    check(g.slow == 1 && near(g.slow_mean, 4225.0, 1e-12) && near(g.slow_last, 1900.0, 1e-12) && near(g.own_mean, 600.0, 1e-12) &&
              near(g.own_last, 600.0, 1e-12),
          "gap: the slowest stage's mean and last chunk, and the stage's own");
    check(near(t.slack(0, 8, 4096, 1, 1), 0.5 * (1900.0 - 600.0) / 8, 1e-12), "gap: the slowest's side from its last chunk, not its decayed mean");
    t.observe(0, 0, 900.0, 4096, 4096);
    check(near(t.slack(0, 8, 4096, 1, 1), 0.5 * (1900.0 - 900.0) / 8, 1e-12), "gap: a stage's own side from its last chunk when that is higher");
    t.observe(0, 0, 600.0, 4096, 4096);
    check(near(t.slack(0, 8, 4096, 1, 1), 0.5 * (1900.0 - 656.25) / 8, 1e-12), "gap: and from its mean when that is higher");
    t.observe(0, 0, 4000.0, 4096, 4096);
    check(t.slack(0, 8, 4096, 1, 1) == 0, "gap: no slack, never less, when the stage's own side passes the slowest's");
    StageTimes v;
    v.stages(2);
    for (const double ms : {1900.0, 1900.0, 3000.0}) v.observe(1, 3, ms, 4096, 4096);
    v.observe(0, 0, 600.0, 4096, 4096);
    v.observe(0, 0, 600.0, 4096, 4096);
    check(near(v.gap(0, 4096).slow_last, 3000.0, 1e-12) && near(v.slack(0, 8, 4096, 1, 1), 0.5 * (2175.0 - 600.0) / 8, 1e-12),
          "gap: the slowest's side from its mean when its last chunk is higher");
    // the slowest is the stage whose lower time is highest: a mean of 2,500 that its last chunk of 1,000 belies is not it
    StageTimes u;
    u.stages(3);
    u.observe(0, 0, 500.0, 1000, 1000);
    u.observe(0, 0, 500.0, 1000, 1000);
    u.observe(1, 10, 1500.0, 1000, 1000);
    u.observe(1, 10, 1500.0, 1000, 1000);
    u.observe(2, 20, 3000.0, 1000, 1000);
    u.observe(2, 20, 1000.0, 1000, 1000);
    check(u.gap(0, 1000).slow == 1 && near(u.slack(0, 10, 1000, 5, 5), 0.5 * (1500.0 - 500.0) / 10, 1e-12) && u.slack(1, 10, 1000, 5, 5) == 0,
          "gap: the slowest stage is the one with the highest lower time");
}

void slack_settles() {
    // two stages share the tier, a slow one of 30 layers after a fast one of 10, 64 tier-held experts a layer, each chunk's time fed back
    constexpr int64_t kNe = 64, T = 2800, kSlow = 30, kFast = 10;
    constexpr double kSlowAttn = 3.0, kFastAttn = 14.0;
    constexpr int kChunks = 30;
    const Rates r{0.001, 0.0005, 1e-9, 1e-9, 1e-7, 5e-8, 0.5, 0.2, 0.01, 5e-8};
    RouteHistory h(1, kNe);
    std::vector<int32_t> cnt(kNe);
    for (int64_t e = 0; e < kNe; ++e) cnt[(size_t) e] = (int32_t) (2800 * (kNe - e) / (kNe * (kNe + 1)));
    h.add(0, cnt.data(), 700);
    const std::vector<Held> tier(kNe, at(Holder::Tier));
    std::vector<uint8_t> take(kNe);
    // gain 0 plans as today, 1 with the stages' slack, 2 with twice it (the whole gap), and cold reads the slow stage's first chunk at 2.5 times
    const auto run = [&](double gain, bool cold, std::vector<double>& slow_ms, std::vector<double>& fast_ms) {
        StageTimes t;
        t.stages(2);
        for (int c = 0; c < kChunks; ++c) {
            const double s_fast = gain * t.slack(1, kFast, T, c, kChunks - 1 - c), s_slow = gain * t.slack(0, kSlow, T, c, kChunks - 1 - c);
            const LayerPlan f = plan_stage_layer(h, 0, T, kTop, tier, kBlob, false, r, 0.0, 0, take.data(), s_fast);
            // the slow stage's calls wait for the fast one's tier ms a chunk, spread over its layers
            const LayerPlan s = plan_stage_layer(h, 0, T, kTop, tier, kBlob, false, r, kFast * f.tier_ms / kSlow, 0, take.data(), s_slow);
            fast_ms.push_back(kFast * (kFastAttn + std::max(f.stage_ms, f.tier_ms)));
            slow_ms.push_back(kSlow * (kSlowAttn + std::max(s.stage_ms, s.tier_ms)) * (cold && c == 0 ? 2.5 : 1.0));
            t.observe(0, 10, slow_ms.back(), T, T);
            t.observe(1, 0, fast_ms.back(), T, T);
        }
    };
    const auto flips = [](const std::vector<double>& slow, const std::vector<double>& fast) {
        int n = 0;
        for (size_t c = 0; c < slow.size(); ++c) n += fast[c] >= slow[c];
        return n;
    };
    std::vector<double> slow0, fast0, slow, fast, slow2, fast2, slowc, fastc;
    run(0, false, slow0, fast0);
    run(1, false, slow, fast);
    run(2, false, slow2, fast2);
    run(1, true, slowc, fastc);
    const bool first = slow[0] == slow0[0] && fast[0] == fast0[0], none = flips(slow, fast) == 0, whole = flips(slow2, fast2) > 0,
               cold = flips(slowc, fastc) == 0, gains = slow.back() < slow0.back() && fast.back() > fast0.back(),
               settled = slow[kChunks - 1] == slow[kChunks - 2] && fast[kChunks - 1] == fast[kChunks - 2];
    if (!(first && none && whole && cold && gains && settled))
        std::printf("  settles: without slack the slow stage %.2f and the fast %.2f ms a chunk, with it %.2f and %.2f after the first, %.2f and "
                    "%.2f at the end; the whole gap ends the fast one after the slow one on %d chunks, a cold first chunk on %d\n",
                    slow0.back(), fast0.back(), slow[1], fast[1], slow.back(), fast.back(), flips(slow2, fast2), flips(slowc, fastc));
    check(first, "settles: the first chunk plans as today");
    check(none, "settles: the slow stage bounds every chunk, the fast one never takes its place");
    check(whole, "settles: with the whole gap the fast stage would end after the slow one");
    check(cold, "settles: a cold first chunk of the slow stage gives the fast one no more than the real gap");
    check(gains, "settles: the fast stage takes tier work and the slow one gains");
    check(settled, "settles: both settle");
}

// the stage time of one layer in each walk: the staged walk waits for every copy at 12.5 GB/s, the streamed walk copied half ahead
double walk_ms(bool streamed, int64_t rows, uint64_t bytes) {
    return streamed ? 0.0005 * (double) rows + 4e-8 * (double) bytes + 0.5 : 0.002 * (double) rows + 8e-8 * (double) bytes + 0.3;
}

void books_per_walk() {
    const Rates priors = starting_rates(13.0);
    StageBooks books(priors);
    check(!books.measured(false) && !books.measured(true), "walks: nothing measured at first");
    std::mt19937 rng(41);
    for (int i = 0; i < RateBook::kMinObs; ++i) {
        const int64_t rows = 30000 + (int64_t) (rng() % 15000);
        const uint64_t bytes = (100 + (uint64_t) (rng() % 300)) * 2'600'000ull;
        check(books.observe(true, rows, bytes, 0, walk_ms(true, rows, bytes)), "walks: a streamed layer is an observation");
    }
    check(books.measured(true) && !books.measured(false), "walks: each walk counts its own observations");
    const Rates st = books.rates(false);
    check(st.stage_row_ms == priors.stage_row_ms && st.stage_byte_ms_paged == priors.stage_byte_ms_paged &&
              st.stage_fixed_ms == priors.stage_fixed_ms,
          "walks: the staged walk keeps its priors");
    check(!books.observe(false, -1, 0, 0, 1.0), "walks: a bad observation is dropped as the book drops it");
}

void walks_fit_apart() {
    // short prompts (the staged walk) mixed with long ones (the streamed walk): one book misses both, a book per walk meets each
    RateBook one(starting_rates(13.0));
    StageBooks two(starting_rates(13.0));
    std::mt19937 rng(43);
    std::vector<std::pair<int64_t, uint64_t>> staged, streamed;
    for (int i = 0; i < 400; ++i) {
        const bool s = i % 3 == 0;
        const int64_t rows = s ? 20000 + (int64_t) (rng() % 25000) : 500 + (int64_t) (rng() % 9500);
        const uint64_t bytes = (20 + (uint64_t) (rng() % (s ? 280 : 200))) * 2'600'000ull;
        const double ms = walk_ms(s, rows, bytes);
        one.observe_stage(rows, bytes, 0, 1, ms);
        two.observe(s, rows, bytes, 0, ms);
        (s ? streamed : staged).push_back({rows, bytes});
    }
    const auto err = [](const Rates& r, bool s, const std::vector<std::pair<int64_t, uint64_t>>& xs) {
        double worst = 0;
        for (const auto& [rows, bytes] : xs) {
            const double fit = r.stage_fixed_ms + (double) rows * r.stage_row_ms + (double) bytes * r.stage_byte_ms_paged;
            worst = std::max(worst, std::fabs(fit / walk_ms(s, rows, bytes) - 1));
        }
        return worst;
    };
    std::printf("  walks: worst error of one book, staged %.0f%% streamed %.0f%%; of a book per walk %.1f%% and %.1f%%\n",
                100 * err(one.rates(), false, staged), 100 * err(one.rates(), true, streamed), 100 * err(two.rates(false), false, staged),
                100 * err(two.rates(true), true, streamed));
    check(err(one.rates(), true, streamed) > 0.3, "walks: one book misses the streamed walk by more than 30%");
    check(err(two.rates(false), false, staged) < 0.05 && err(two.rates(true), true, streamed) < 0.05, "walks: a book per walk meets each within 5%");
}

}  // namespace

int main() {
    holders();
    rows_before_routing();
    rows_from_history();
    rows_decay();
    items_from_holdings();
    idle_copies();
    take_from_choice();
    routed_counts();
    plans_a_layer();
    plan_counts_idle_copies();
    plans_streams_to_tier();
    tier_start();
    x_counted_once();
    stage_calls();
    ram_counts();
    halves_pay_two_calls();
    halves_stream_once();
    halves_lead();
    halves_stage_fixed_once();
    halves_b_start();
    halves_times();
    slack_whole();
    slack_streams();
    slack_halves();
    slack_halves_b_none();
    shed_by_delta();
    stage_times();
    stage_gap();
    slack_settles();
    books_per_walk();
    walks_fit_apart();
    std::printf("tier_planner_test: %d checks passed\n", checks);
    return 0;
}
