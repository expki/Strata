// The pure part of a stage's prompt plan for the AMD tier: rows predicted from the stage's routing, a layer's items and the take they give. No GPU needed.
#pragma once

#include "strata/prefill/expert_planner.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::prefill::planner {

/// Where a stage finds an expert: in its own VRAM, in the AMD tier's VRAM, or only in RAM.
enum class Holder : uint8_t { Stage, Tier, Ram };

/// The one rule for who holds an expert: the stage's own copy comes first, then the tier's.
constexpr Holder holder_of(bool on_stage, bool on_tier) { return on_stage ? Holder::Stage : on_tier ? Holder::Tier : Holder::Ram; }

/// A stage's RAM experts over layers [lb, le), those holder(l, e) says neither it nor the tier holds: its weight for its part of the card's ring.
template <class Who>
int64_t ram_experts(int64_t lb, int64_t le, int64_t n_expert, Who&& holder) {
    int64_t n = 0;
    for (int64_t l = lb; l < le; ++l)
        for (int64_t e = 0; e < n_expert; ++e) n += holder(l, e) == Holder::Ram;
    return n;
}

/// The most RAM experts one of a stage's layers [lb, le) holds: the most one call of the stage can stream to the tier.
template <class Who>
int64_t ram_widest(int64_t lb, int64_t le, int64_t n_expert, Who&& holder) {
    int64_t most = 0;
    for (int64_t l = lb; l < le; ++l) {
        int64_t n = 0;
        for (int64_t e = 0; e < n_expert; ++e) n += holder(l, e) == Holder::Ram;
        most = std::max(most, n);
    }
    return most;
}

/// Each layer's routed rows per expert over a stage's recent chunks, each chunk counting half as much as the one after it.
class RouteHistory {
public:
    RouteHistory(int64_t n_layers, int64_t n_expert);
    /// Layer l's routing in a chunk of T tokens: cnt[e] rows went to expert e.
    void add(int64_t l, const int32_t* cnt, int64_t T);
    /// The rows expert e of layer l is predicted to get in a chunk of T tokens of k experts each: the history scaled to T, or an even share before any.
    int32_t rows(int64_t l, int64_t e, int64_t T, int64_t k) const;

private:
    int64_t ne_;
    std::vector<float> h_, t_;   // per layer and expert the decayed rows, per layer the decayed tokens
};

/// One expert of a layer as the plan sees it: who holds it, and whether its RAM copy is pinned.
struct Held { Holder holder; bool pinned; };

/// A take row's values: the stage computes the expert, the tier computes it from its VRAM, the tier streams it from RAM.
enum : uint8_t { kTakeStage = 0, kTakeTier = 1, kTakeStream = 2 };

/// Layer l's items in expert order: a tier-held expert stays or streams to the stage, a RAM expert streams to the stage or with tier_stream to the tier.
std::vector<Item> layer_items(const RouteHistory& h, int64_t l, int64_t T, int64_t k, const std::vector<Held>& held, uint64_t blob,
                              bool tier_stream);

/// What the stage spends before the plan's items: the streamed walk copies every RAM expert, those no row is predicted for too.
double idle_copies_ms(const std::vector<Item>& items, const std::vector<Held>& held, bool streamed, const Rates& r);

/// Tier-held experts or their rows, how many of them the stage computes, and the RAM experts or rows streamed to the tier.
struct Counts { int64_t held = 0, moved = 0, streamed = 0; };

/// A layer's row of take from a plan: kTakeTier where a tier-held expert stays on the tier, kTakeStream where a RAM expert streams to it.
Counts fill_take(const std::vector<Held>& held, const Choice& c, uint8_t* take_row);

/// A layer's routed rows on tier-held experts, of them the rows the stage computed, and the rows streamed to the tier.
Counts routed_rows(const int32_t* cnt, const std::vector<Held>& held, const uint8_t* take_row);

/// A stage's own rates fitted per walk on that walk's layers: the streamed walk copies every expert ahead, the staged walk copies the routed ones as it goes.
class StageBooks {
public:
    explicit StageBooks(const Rates& priors) : book_{RateBook(priors), RateBook(priors)} {}
    /// One layer's expert phase in the given walk, or false when the book drops it.
    bool observe(bool streamed, int64_t rows, uint64_t paged, uint64_t pinned, double ms) {
        return book_[streamed].observe_stage(rows, paged, pinned, 1, ms);
    }
    bool measured(bool streamed) const { return book_[streamed].stage_measured(); }
    /// The walk's stage rates, and the priors for the tier.
    Rates rates(bool streamed) const { return book_[streamed].rates(); }

private:
    RateBook book_[2];   // the staged walk, then the streamed walk
};

/// When the tier starts a stage's layer after the stage queued it (the calls ahead and the handoff of x), measured on each call.
class TierStart {
public:
    /// One call's ms from queued to x seen: the first sets the mean, each later one counts a quarter. A value that is not a time is dropped.
    void observe(double ms);
    /// The mean so far, 0 before any call.
    double ms() const { return ms_; }
    /// Whether a call measured it yet.
    bool measured() const { return ms_ > 0; }
    /// The plan's tier load for a layer: the larger of the queue the other stages' predicted calls leave and the measured start.
    double ahead(double predicted_queue_ms) const { return predicted_queue_ms > ms_ ? predicted_queue_ms : ms_; }

private:
    double ms_ = 0;
};

/// The calls the stages queued on the tier: each stage's two latest, as a chunk in halves calls twice a layer.
class StageCalls {
public:
    /// A call: its predicted ms count as queued on the tier from at_ms until its done word reaches seq.
    struct Call { const volatile uint32_t* done = nullptr; uint32_t seq = 0; double at_ms = 0, ms = 0; };
    /// Room for n stages, each with no call.
    void stages(size_t n);
    /// Stage s queues c: the other stages' calls still on the tier, then c in place of s's older call.
    std::vector<Queued> queue(size_t s, const Call& c);
    /// Stage s has no call the others should wait for.
    void clear(size_t s);

private:
    std::vector<Call> calls_;   // two per stage, at 2 s and 2 s + 1
    std::vector<uint64_t> n_;   // per stage, its calls so far
};

/// Each stage's chunk time on its own GPU per token, a decayed mean as TierStart's, and the slack a faster stage has against the slowest.
class StageTimes {
public:
    /// Room for n stages, none measured.
    void stages(size_t n) { st_.resize(n); }
    /// Stage s, whose layers start at lb, read T tokens of its run's chunks of C in ms: a shorter chunk or a value not a time is dropped, the first sets the mean, each later counts a quarter.
    void observe(size_t s, int64_t lb, double ms, int64_t T, int64_t C);
    /// Stage s's mean for a chunk of T tokens, 0 before any chunk.
    double chunk_ms(size_t s, int64_t T) const;
    /// For T tokens: the slowest stage, the one whose lower of mean and last chunk is highest, with its mean and last, and stage s's.
    struct Gap { size_t slow = 0; double slow_mean = 0, slow_last = 0, own_mean = 0, own_last = 0; };
    Gap gap(size_t s, int64_t T) const;
    /// Half the slowest's lower time less stage s's higher, in ms a layer of its for T tokens, 0 until all read two chunks and on a chunk the slowest waits for.
    double slack(size_t s, int64_t layers, int64_t T, int64_t before, int64_t after) const;

private:
    struct Stage { double per_token = 0, last = 0; int64_t lb = 0, n = 0; };
    std::vector<Stage> st_;
};

/// A call's rates in a walk: the tier's fixed cost gains x_bytes in over its link, and out over the stage's unless a measured start holds that.
Rates tier_call_rates(const Rates& walk, uint64_t x_bytes, bool start_measured);

/// One layer planned: when each device ends, the tier-held experts and how many moved to the stage, and the RAM experts streamed to the tier.
struct LayerPlan { double stage_ms = 0, tier_ms = 0; int64_t held = 0, moved = 0, streamed = 0; };

/// Plans layer l into its take row after tier_ahead_ms of tier load, the tier streaming at most tier_slots (streamed walk), the stage ending by slack_ms past the balance.
LayerPlan plan_stage_layer(const RouteHistory& h, int64_t l, int64_t T, int64_t k, const std::vector<Held>& held, uint64_t blob,
                           bool streamed, const Rates& r, double tier_ahead_ms, int tier_slots, uint8_t* take_row, double slack_ms = 0);

/// A layer of a chunk read as two token halves: A's plan, then B's, whose ends are the layer's.
struct HalvesPlan { LayerPlan a, b; };

/// When a layer's calls in halves reach the tier from A's routing: A's ahead_ms after it, B's b_ahead_ms after the stage's experts start at lead_ms.
struct HalfStarts { double ahead_ms = 0, lead_ms = 0, b_ahead_ms = 0; };

/// Plans layer l's halves of ta and tb tokens into take_a and take_b: a tier call per half, B's after A's and its own start, one stage MoE for both.
HalvesPlan plan_stage_halves(const RouteHistory& h, int64_t l, int64_t ta, int64_t tb, int64_t k, const std::vector<Held>& held, uint64_t blob,
                             bool streamed, const Rates& r, const HalfStarts& at, int tier_slots, uint8_t* take_a, uint8_t* take_b,
                             double slack_ms = 0);

/// A layer in halves in ms from the stage's own experts' start: when each half's call starts and ends on the tier, and when the stage ends.
struct HalfTimes {
    double a_start = 0, a_end = 0, b_start = 0, b_end = 0, stage_end = 0;
    HalfTimes& operator+=(const HalfTimes& o) {
        a_start += o.a_start;
        a_end += o.a_end;
        b_start += o.b_start;
        b_end += o.b_end;
        stage_end += o.stage_end;
        return *this;
    }
};

/// A layer's planned times, its plan made with these starts.
HalfTimes planned_times(const HalvesPlan& p, const HalfStarts& at);

}  // namespace strata::prefill::planner
