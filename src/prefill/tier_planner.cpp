// The pure part of a stage's prompt plan for the AMD tier (see tier_planner.hpp).
#include "strata/prefill/tier_planner.hpp"

#include <algorithm>
#include <cmath>

namespace strata::prefill::planner {

RouteHistory::RouteHistory(int64_t n_layers, int64_t n_expert)
    : ne_(n_expert), h_((size_t) (n_layers * n_expert), 0.0f), t_((size_t) n_layers, 0.0f) {}

void RouteHistory::add(int64_t l, const int32_t* cnt, int64_t T) {
    float* h = h_.data() + (size_t) (l * ne_);
    for (int64_t e = 0; e < ne_; ++e) h[e] = 0.5f * h[e] + (float) cnt[e];
    t_[(size_t) l] = 0.5f * t_[(size_t) l] + (float) T;
}

int32_t RouteHistory::rows(int64_t l, int64_t e, int64_t T, int64_t k) const {
    const double scale = t_[(size_t) l] > 0 ? (double) T / t_[(size_t) l] : 0.0;
    return (int32_t) std::lround(scale > 0 ? h_[(size_t) (l * ne_ + e)] * scale : (double) (T * k) / (double) ne_);
}

std::vector<Item> layer_items(const RouteHistory& h, int64_t l, int64_t T, int64_t k, const std::vector<Held>& held, uint64_t blob,
                              bool tier_stream) {
    const uint8_t ram = (uint8_t) (kStageStream | (tier_stream ? kTierStream : 0));
    std::vector<Item> items;
    items.reserve(held.size());
    for (size_t e = 0; e < held.size(); ++e) {
        const int32_t rows = h.rows(l, (int64_t) e, T, k);
        switch (held[e].holder) {
            case Holder::Stage: items.push_back({(int32_t) e, rows, kStageVram, 0, false}); break;
            case Holder::Tier: items.push_back({(int32_t) e, rows, (uint8_t) (kTierVram | kStageStream), blob, held[e].pinned}); break;
            case Holder::Ram: items.push_back({(int32_t) e, rows, ram, blob, held[e].pinned}); break;
        }
    }
    return items;
}

double idle_copies_ms(const std::vector<Item>& items, const std::vector<Held>& held, bool streamed, const Rates& r) {
    if (!streamed) return 0;
    double ms = 0;
    for (size_t e = 0; e < items.size(); ++e)
        if (held[e].holder == Holder::Ram && items[e].rows <= 0)
            ms += (double) items[e].bytes * (items[e].pinned ? r.stage_byte_ms_pinned : r.stage_byte_ms_paged);
    return ms;
}

Counts fill_take(const std::vector<Held>& held, const Choice& c, uint8_t* take_row) {
    Counts n;
    for (size_t e = 0; e < held.size(); ++e) {
        const bool tier = held[e].holder == Holder::Tier, stream = c.where[e] == Where::TierStream;
        take_row[e] = tier && c.where[e] == Where::TierVram ? kTakeTier : stream ? kTakeStream : kTakeStage;
        n.held += tier;
        n.moved += tier && c.where[e] != Where::TierVram;
        n.streamed += stream;
    }
    return n;
}

Counts routed_rows(const int32_t* cnt, const std::vector<Held>& held, const uint8_t* take_row) {
    Counts n;
    for (size_t e = 0; e < held.size(); ++e) {
        if (take_row[e] == kTakeStream) n.streamed += cnt[e];
        if (held[e].holder != Holder::Tier) continue;
        n.held += cnt[e];
        if (take_row[e] == kTakeStage) n.moved += cnt[e];
    }
    return n;
}

void TierStart::observe(double ms) {
    if (!std::isfinite(ms) || ms < 0) return;
    ms_ = ms_ > 0 ? 0.75 * ms_ + 0.25 * ms : ms;
}

void StageCalls::stages(size_t n) {
    calls_.resize(2 * n);
    n_.resize(n, 0);
}

std::vector<Queued> StageCalls::queue(size_t s, const Call& c) {
    std::vector<Queued> ahead;
    for (size_t i = 0; i < calls_.size(); ++i) {
        const Call& o = calls_[i];
        if (i / 2 == s || o.done == nullptr || (int32_t) (*o.done - o.seq) >= 0) continue;
        ahead.push_back({o.at_ms, o.ms});
    }
    calls_[2 * s + (n_[s]++ & 1)] = c;   // the stage's other call may still be ahead of the others' next ones
    return ahead;
}

void StageCalls::clear(size_t s) { calls_[2 * s] = calls_[2 * s + 1] = Call{}; }

Rates tier_call_rates(const Rates& walk, uint64_t x_bytes, bool start_measured) {
    Rates r = walk;
    r.tier_fixed_ms += (double) x_bytes * (r.tier_input_byte_ms + (start_measured ? 0.0 : r.stage_byte_ms_pinned));
    return r;
}

namespace {

constexpr bool on_tier(Where w) { return w == Where::TierVram || w == Where::TierStream; }

// When a layer's placements end the stage, and the ms of work their calls give the tier
struct Ends { double stage = 0, tier = 0; };

// B's item of expert e after A placed it at wa: A's copy costs no bytes again, and nothing new streams to the tier
Item b_item(const Item& b0, Where wa, Holder holder) {
    Item it = b0;
    if (wa == Where::StageStream) {
        it.bytes = 0;
        if (holder == Holder::Ram) it.can = kStageStream;
    } else if (wa == Where::TierStream) {
        it.can = kTierVram | kStageStream;
    }
    return it;
}

// B's items and rates after A's placements: each from b_item, and the stage's fixed cost paid once
void b_items(const std::vector<Item>& ib0, const std::vector<Item>& ia, const std::vector<Where>& wa, const std::vector<Held>& held, const Rates& r,
             std::vector<Item>& ib, Rates& rb) {
    ib.resize(ib0.size());
    bool stage_paid = false;
    for (size_t e = 0; e < ib.size(); ++e) {
        stage_paid = stage_paid || (ia[e].rows > 0 && !on_tier(wa[e]));
        ib[e] = b_item(ib0[e], wa[e], held[e].holder);
    }
    rb = r;
    if (stage_paid) rb.stage_fixed_ms = 0;
}

// A layer's placements as shed moves them and the tallies they give: A's after la, and with halves B's after A's stage and its own start
class ShedLayer {
public:
    // A whole chunk: its items placed at wa after la
    ShedLayer(const std::vector<Item>& ia, std::vector<Where>& wa, const Load& la, const Rates& r) : ia_(ia), wa_(wa), la_(la), r_(r), a_(la), b_(Load{}) {
        for (size_t e = 0; e < ia.size(); ++e)
            if (ia[e].rows > 0) a_.put(ia[e], wa[e], r, +1);
    }
    // Halves: B's items as b_item makes them from ib0 and A's places, placed at wb, its call no earlier than b_start
    ShedLayer(const std::vector<Item>& ia, std::vector<Where>& wa, const Load& la, const Rates& r, const std::vector<Item>& ib0,
              std::vector<Where>& wb, const std::vector<Held>& held, double b_start)
        : ShedLayer(ia, wa, la, r) {
        ib0_ = &ib0;
        wb_ = &wb;
        held_ = &held;
        b_start_ = b_start;
        for (size_t e = 0; e < ib0.size(); ++e)
            if (ib0[e].rows > 0) b_.put(b_item(ib0[e], wa[e], held[e].holder), wb[e], r, +1);
    }

    size_t size() const { return ia_.size(); }

    // Expert e taken to the stage in both halves: the tallies and places it gives, and whether anything moved
    struct Move { Tally a, b; Where wa, wb; bool any = false; };
    Move move(size_t e) const {
        Move m{a_, b_, wa_[e], wb_ ? (*wb_)[e] : Where::StageVram};
        const bool streamed = m.wa == Where::TierStream;   // B reads the copy A's call makes, so B's part goes with it, rows or not
        if (ia_[e].rows > 0 && on_tier(m.wa)) {
            m.a.put(ia_[e], m.wa, r_, -1);
            m.a.put(ia_[e], Where::StageStream, r_, +1);
            m.wa = Where::StageStream;
            m.any = true;
        }
        if (wb_ == nullptr) return m;
        const Item& b0 = (*ib0_)[e];
        if ((b0.rows > 0 || streamed) && on_tier(m.wb)) m.wb = Where::StageStream, m.any = true;
        if (b0.rows > 0 && m.any) {   // B's item follows A's place, and its tally its own
            const Holder holder = (*held_)[e].holder;
            m.b.put(b_item(b0, wa_[e], holder), (*wb_)[e], r_, -1);
            m.b.put(b_item(b0, m.wa, holder), m.wb, r_, +1);
        }
        return m;
    }
    void take(size_t e, const Move& m) {
        a_ = m.a;
        b_ = m.b;
        wa_[e] = m.wa;
        if (wb_ != nullptr) (*wb_)[e] = m.wb;
    }

    // When the stage ends and the tier's work, as place_layer would sum them for A's places and then B's
    Ends ends() const { return ends(a_, b_); }
    Ends ends(const Move& m) const { return ends(m.a, m.b); }

private:
    Ends ends(const Tally& a, const Tally& b) const {
        const double a_stage = a.stage_ms(r_), a_tier = a.tier_ms(r_), b0 = std::max(a_tier, b_start_);
        Tally tb = b;
        tb.stage_base = a_stage;
        tb.tier_base = b0;
        Rates rb = r_;
        if (a.stage_items > 0) rb.stage_fixed_ms = 0;   // as b_items: the stage's fixed cost is paid once
        return Ends{tb.stage_ms(rb), (a_tier - la_.tier_ms) + (tb.tier_ms(rb) - b0)};
    }

    const std::vector<Item>& ia_;
    std::vector<Where>& wa_;
    const std::vector<Item>* ib0_ = nullptr;
    std::vector<Where>* wb_ = nullptr;
    const std::vector<Held>* held_ = nullptr;
    Load la_;
    const Rates& r_;
    double b_start_ = 0;
    Tally a_, b_;
};

// Moves tier work to the stage an expert at a time in both halves, the most tier ms saved per stage ms first, keeping each that ends the stage by cap
void shed(ShedLayer& s, double cap) {
    const Ends now = s.ends();
    std::vector<std::pair<size_t, double>> order;   // an expert, and the tier ms it saves per stage ms it adds
    for (size_t e = 0; e < s.size(); ++e) {
        const ShedLayer::Move m = s.move(e);
        if (!m.any) continue;
        const Ends to = s.ends(m);
        const double added = to.stage - now.stage;
        order.emplace_back(e, added > 0 ? (now.tier - to.tier) / added : HUGE_VAL);
    }
    std::stable_sort(order.begin(), order.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
    for (const auto& [e, per] : order) {
        const ShedLayer::Move m = s.move(e);
        if (s.ends(m).stage > cap) continue;
        s.take(e, m);
    }
}

}  // namespace

LayerPlan plan_stage_layer(const RouteHistory& h, int64_t l, int64_t T, int64_t k, const std::vector<Held>& held, uint64_t blob,
                           bool streamed, const Rates& r, double tier_ahead_ms, int tier_slots, uint8_t* take_row, double slack_ms) {
    const int slots = streamed ? tier_slots : 0;
    const std::vector<Item> items = layer_items(h, l, T, k, held, blob, slots > 0);
    const Load load{idle_copies_ms(items, held, streamed, r), tier_ahead_ms};
    Choice c = plan_layer(items, r, load, slots);
    if (slack_ms > 0) {
        std::vector<Where> w = std::move(c.where);
        ShedLayer s(items, w, load, r);
        shed(s, std::max(c.stage_ms, c.tier_ms) + slack_ms);
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
        std::vector<Where> wa = std::move(ca.where), wb = std::move(cb.where);
        ShedLayer s(ia, wa, la, r, ib0, wb, held, at.lead_ms + at.b_ahead_ms);
        shed(s, std::max(cb.stage_ms, cb.tier_ms) + slack_ms);
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

void StageTimes::observe(size_t s, int64_t lb, double ms, int64_t T, int64_t C) {
    if (!std::isfinite(ms) || ms < 0 || T <= 0 || T < C) return;
    if (s >= st_.size()) st_.resize(s + 1);
    Stage& t = st_[s];
    const double v = ms / (double) T;
    t.per_token = t.per_token > 0 ? 0.75 * t.per_token + 0.25 * v : v;
    t.last = v;
    t.lb = lb;
    ++t.n;
}

double StageTimes::chunk_ms(size_t s, int64_t T) const { return s < st_.size() && T > 0 ? st_[s].per_token * (double) T : 0; }

StageTimes::Gap StageTimes::gap(size_t s, int64_t T) const {
    if (s >= st_.size()) return Gap{};
    const auto low = [&](size_t i) { return std::min(st_[i].per_token, st_[i].last); };
    size_t slow = s;
    for (size_t i = 0; i < st_.size(); ++i)
        if (low(i) > low(slow)) slow = i;
    const double t = (double) T;
    return Gap{slow, st_[slow].per_token * t, st_[slow].last * t, st_[s].per_token * t, st_[s].last * t};
}

double StageTimes::slack(size_t s, int64_t layers, int64_t T, int64_t before, int64_t after) const {
    if (s >= st_.size() || layers <= 0 || T <= 0) return 0;
    for (const Stage& t : st_)
        if (t.n < 2) return 0;
    const Gap g = gap(s, T);
    // the stages from this one to the slowest by their first layers: the slowest waits for this one's first (or last) that many chunks
    const int64_t lo = std::min(st_[s].lb, st_[g.slow].lb), hi = std::max(st_[s].lb, st_[g.slow].lb);
    int64_t d = 0;
    for (const Stage& t : st_) d += t.lb > lo && t.lb <= hi;
    if ((st_[s].lb < st_[g.slow].lb ? before : after) < d) return 0;
    // the gap taken low: the slowest stage's time no more than its last chunk, this stage's own no less than its last
    const double gap = std::min(g.slow_mean, g.slow_last) - std::max(g.own_mean, g.own_last);
    return gap > 0 ? 0.5 * gap / (double) layers : 0;
}

HalfTimes planned_times(const HalvesPlan& p, const HalfStarts& at) {
    const double lead = at.lead_ms;
    return HalfTimes{at.ahead_ms - lead, p.a.tier_ms - lead, std::max(p.a.tier_ms, lead + at.b_ahead_ms) - lead, p.b.tier_ms - lead, p.b.stage_ms - lead};
}

}  // namespace strata::prefill::planner
