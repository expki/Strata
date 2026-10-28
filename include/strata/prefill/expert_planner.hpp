// Where each routed expert of one layer runs in a prompt chunk, and the measured rates that price it. No GPU needed.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::prefill::planner {

enum class Where : uint8_t { StageVram, TierVram, StageStream, TierStream };
enum : uint8_t { kStageVram = 1, kTierVram = 2, kStageStream = 4, kTierStream = 8 };
static_assert(kStageVram == 1 << (int) Where::StageVram && kTierVram == 1 << (int) Where::TierVram &&
              kStageStream == 1 << (int) Where::StageStream && kTierStream == 1 << (int) Where::TierStream,
              "Where's order is the k* bit order");

/// can: the k* bits the expert may run at, and it must include kStageVram, kTierVram or kStageStream. bytes: its size when streamed. pinned: its RAM copy is pinned, not paged.
struct Item { int32_t expert; int32_t rows; uint8_t can; uint64_t bytes; bool pinned; };

struct Rates {
    double stage_row_ms, tier_row_ms;                  ///< compute per routed row
    double stage_byte_ms_paged, stage_byte_ms_pinned;  ///< RAM to the stage GPU per byte, by source kind
    double tier_byte_ms_paged, tier_byte_ms_pinned;    ///< RAM to the W6800 per byte of the experts it streams, by source kind
    double tier_fixed_ms;                              ///< the tier's per call cost (the handoff and the sums back)
    double stage_fixed_ms;                             ///< the stage's per layer cost (what its expert phase takes at any size)
    double tier_expert_ms;                             ///< the tier's cost for each expert it computes, whatever its rows
    double tier_input_byte_ms;                         ///< the tier's input per byte: a call's x, routing weights and tables
};

/// Starting rates before anything is measured, from the stage's probed link (gbps <= 0: not probed). The tier counts as the faster device.
Rates starting_rates(double stage_link_gbps);

/// Work already queued before this layer. For the tier it is what the other stages sent.
struct Load { double stage_ms = 0, tier_ms = 0; };

/// Both finish times include the load. Each device's fixed cost counts only when it has rows.
struct Choice { std::vector<Where> where; double stage_ms, tier_ms; };

/// The two devices' finish times for the items put so far, as place_layer sums them. Each device's fixed cost is paid once, if it has any rows.
struct Tally {
    double stage_base, stage_work = 0, tier_base, tier_work = 0;
    int stage_items = 0, tier_items = 0;
    explicit Tally(const Load& start) : stage_base(start.stage_ms), tier_base(start.tier_ms) {}
    double stage_ms(const Rates& r) const { return stage_base + (stage_items > 0 ? r.stage_fixed_ms + stage_work : 0); }
    double tier_ms(const Rates& r) const { return tier_base + (tier_items > 0 ? r.tier_fixed_ms + tier_work : 0); }
    double finish(const Rates& r) const { return std::max(stage_ms(r), tier_ms(r)); }
    /// Puts the item at w (sign 1) or takes it from there (sign -1).
    void put(const Item& it, Where w, const Rates& r, int sign);
};

/// From each item's home (StageVram, else TierVram, else StageStream) moves the busiest while the later finish drops. Slots cap TierStream.
Choice plan_layer(const std::vector<Item>& items, const Rates& r, const Load& start, int tier_stream_slots);

/// The finish times of the items placed at where, each at a place it can take.
Choice place_layer(const std::vector<Item>& items, const std::vector<Where>& where, const Rates& r, const Load& start);

/// One call on the tier's queue: when it was queued and the ms it is predicted to take.
struct Queued { double at_ms, ms; };

/// How long after now_ms the tier is busy with these calls, taking them in the order they were queued and idle only when it has none.
double queued_ahead(std::vector<Queued> calls, double now_ms);

/// Measured rates fitted over recent observations: ms = a * rows + b * bytes per device and source kind + c * calls, and the tier's experts and input bytes too.
class RateBook {
public:
    /// Observations a device needs before its rates leave the priors.
    static constexpr int kMinObs = 16;
    /// A fitted rate never goes below this fraction of its prior.
    static constexpr double kFloorFrac = 0.01;
    /// An observation's weight halves every half_life_obs observations of its device. A prior at or below zero is raised to a tiny value.
    explicit RateBook(const Rates& priors, double half_life_obs = 64);
    /// Returns whether the observation was used. It is dropped if it has no rows, bytes or calls, a negative count, or an ms that is negative, not a number or above an hour.
    bool observe_stage(int64_t rows, uint64_t bytes_paged, uint64_t bytes_pinned, int64_t calls, double ms);
    /// bytes_in: x, the routing weights and the tables. bytes_paged and bytes_pinned: the experts it streamed from RAM, by source kind.
    bool observe_tier(int64_t rows, int64_t experts, uint64_t bytes_in, uint64_t bytes_paged, uint64_t bytes_pinned, int64_t calls, double ms);
    /// The priors until enough observations exist, and never a negative or zero coefficient. Not thread safe.
    Rates rates() const;
    /// Whether a device has had enough observations to leave its priors.
    bool stage_measured() const { return stage_.taken >= kMinObs; }
    bool tier_measured() const { return tier_.taken >= kMinObs; }

private:
    static constexpr int kMaxVars = 6;
    struct Fit {
        int n = 0;                           ///< variables in use
        double prior[kMaxVars] = {};
        double a[kMaxVars][kMaxVars] = {};   ///< decayed sums of x x^T
        double b[kMaxVars] = {};             ///< decayed sums of x y
        double sumsq[kMaxVars] = {};         ///< undecayed sums of x^2, the scale of each variable
        double seen = 0;                     ///< undecayed sum of the observations' weights
        int taken = 0;                       ///< observations used, counted up to kMinObs
        void add(const double* x, double y, double keep);
        void solve(double* out) const;
    };
    double keep_;   ///< the weight an older observation keeps when a newer one arrives
    Fit stage_, tier_;
};

}  // namespace strata::prefill::planner
