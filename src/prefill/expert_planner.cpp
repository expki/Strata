// Where each routed expert of one layer runs in a prompt chunk, and the measured rates that price it (see expert_planner.hpp).
#include "strata/prefill/expert_planner.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace strata::prefill::planner {
namespace {

constexpr bool on_tier(Where w) { return w == Where::TierVram || w == Where::TierStream; }
constexpr int bit_of(Where w) { return 1 << (int) w; }   // Where's order is the k* bit order

double cost(const Item& it, Where w, const Rates& r) {
    const double rows = (double) it.rows, bytes = (double) it.bytes;
    switch (w) {
        case Where::StageVram:   return rows * r.stage_row_ms;
        case Where::TierVram:    return rows * r.tier_row_ms + r.tier_expert_ms;
        case Where::StageStream: return rows * r.stage_row_ms + bytes * (it.pinned ? r.stage_byte_ms_pinned : r.stage_byte_ms_paged);
        case Where::TierStream:  return rows * r.tier_row_ms + r.tier_expert_ms + bytes * (it.pinned ? r.tier_byte_ms_pinned : r.tier_byte_ms_paged);
    }
    return 0;
}

Where home_of(const Item& it) {
    if (it.can & kStageVram) return Where::StageVram;
    if (it.can & kTierVram) return Where::TierVram;
    return Where::StageStream;
}

class Layer {
public:
    Layer(const std::vector<Item>& items, const Rates& r, const Load& start)
        : items_(items), r_(r), start_(start), where_(homes(items)), tally_(sum()) {}

    // Moves the items at `from` that may go `to`, most rows first, while the later finish drops. Stops at the first that does not.
    void drain(Where from, Where to, int limit) {
        std::vector<size_t> order;
        for (size_t i = 0; i < items_.size(); ++i)
            if (where_[i] == from && items_[i].rows > 0 && (items_[i].can & bit_of(to))) order.push_back(i);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return items_[a].rows > items_[b].rows; });
        int moved = 0;
        for (size_t i : order) {
            if (moved >= limit) break;
            Tally next = tally_;
            next.put(items_[i], from, r_, -1);
            next.put(items_[i], to, r_, +1);
            if (!(next.finish(r_) < tally_.finish(r_))) break;
            tally_ = next;
            where_[i] = to;
            ++moved;
        }
    }

    // The finish times are summed again from the placements, so no rounding of the moves carries into them.
    Choice take() {
        const Tally t = sum();
        return Choice{std::move(where_), t.stage_ms(r_), t.tier_ms(r_)};
    }

private:
    static std::vector<Where> homes(const std::vector<Item>& items) {
        std::vector<Where> w;
        w.reserve(items.size());
        for (const Item& it : items) w.push_back(home_of(it));
        return w;
    }
    Tally sum() const {
        Tally t(start_);
        for (size_t i = 0; i < items_.size(); ++i)
            if (items_[i].rows > 0) t.put(items_[i], where_[i], r_, +1);
        return t;
    }
    const std::vector<Item>& items_;
    const Rates& r_;
    Load start_;
    std::vector<Where> where_;
    Tally tally_;
};

}  // namespace

void Tally::put(const Item& it, Where w, const Rates& r, int sign) {
    if (on_tier(w)) { tier_work += sign * cost(it, w, r); tier_items += sign; }
    else { stage_work += sign * cost(it, w, r); stage_items += sign; }
}

Choice plan_layer(const std::vector<Item>& items, const Rates& r, const Load& start, int tier_stream_slots) {
    Layer layer(items, r, start);
    layer.drain(Where::TierVram, Where::StageStream, (int) items.size());
    if (tier_stream_slots > 0) layer.drain(Where::StageStream, Where::TierStream, tier_stream_slots);
    return layer.take();
}

Choice place_layer(const std::vector<Item>& items, const std::vector<Where>& where, const Rates& r, const Load& start) {
    Tally t(start);
    for (size_t i = 0; i < items.size(); ++i)
        if (items[i].rows > 0) t.put(items[i], where[i], r, +1);
    return Choice{where, t.stage_ms(r), t.tier_ms(r)};
}

Rates starting_rates(double stage_link_gbps) {
    constexpr double kLinkGbps = 12;   // a link that was not probed, as the engine's other plans take it
    const double stage_pinned = 1e-6 / (stage_link_gbps > 0 ? stage_link_gbps : kLinkGbps), tier_pinned = 1e-6 / kLinkGbps;
    Rates r{};
    r.stage_row_ms = 1e-3;                     // a microsecond a routed row
    r.tier_row_ms = 0.5e-3;                    // the tier taken as the faster, so nothing leaves it before it is measured
    r.stage_byte_ms_pinned = stage_pinned;
    r.stage_byte_ms_paged = 2 * stage_pinned;   // a paged blob is copied twice: by a stager thread, then over the link
    r.tier_byte_ms_pinned = tier_pinned;
    r.tier_byte_ms_paged = 2 * tier_pinned;
    r.tier_fixed_ms = 0.5;
    r.stage_fixed_ms = 0.2;
    r.tier_expert_ms = 0.01;                    // a gather of one expert's weights on the card
    r.tier_input_byte_ms = tier_pinned;
    return r;
}

double queued_ahead(std::vector<Queued> calls, double now_ms) {
    std::stable_sort(calls.begin(), calls.end(), [](const Queued& a, const Queued& b) { return a.at_ms < b.at_ms; });
    double end = -HUGE_VAL;   // when the tier is through the calls taken so far
    for (const Queued& c : calls) end = std::max(end, c.at_ms) + c.ms;
    return std::max(0.0, end - now_ms);
}

namespace {

// A quarter of a typical observation keeps 200 observations within 5% of a truth a prior 3x off.
constexpr double kPriorWeight = 0.25;
// An observation off its prediction by more than this fraction counts for less, in proportion to the excess.
constexpr double kTrustRel = 0.5;
// An observation longer than this is not a measurement (an hour).
constexpr double kMaxObsMs = 3.6e6;
// A prior at or below zero, or not a number, is raised to this so no rate is ever zero.
constexpr double kMinPrior = 1e-12;

double positive(double v) { return v > kMinPrior ? v : kMinPrior; }

bool all_zero(const double* x, int n) {
    for (int i = 0; i < n; ++i) if (x[i] != 0) return false;
    return true;
}

}  // namespace

RateBook::RateBook(const Rates& priors, double half_life_obs) : keep_(half_life_obs > 0 ? std::exp2(-1.0 / half_life_obs) : 1.0) {
    stage_.n = 4;
    stage_.prior[0] = positive(priors.stage_row_ms);
    stage_.prior[1] = positive(priors.stage_byte_ms_paged);
    stage_.prior[2] = positive(priors.stage_byte_ms_pinned);
    stage_.prior[3] = positive(priors.stage_fixed_ms);
    tier_.n = 6;
    tier_.prior[0] = positive(priors.tier_row_ms);
    tier_.prior[1] = positive(priors.tier_byte_ms_paged);
    tier_.prior[2] = positive(priors.tier_byte_ms_pinned);
    tier_.prior[3] = positive(priors.tier_fixed_ms);
    tier_.prior[4] = positive(priors.tier_expert_ms);
    tier_.prior[5] = positive(priors.tier_input_byte_ms);
}

bool RateBook::observe_stage(int64_t rows, uint64_t bytes_paged, uint64_t bytes_pinned, int64_t calls, double ms) {
    if (rows < 0 || calls < 0 || !(ms >= 0 && ms <= kMaxObsMs)) return false;
    const double x[4] = {(double) rows, (double) bytes_paged, (double) bytes_pinned, (double) calls};
    if (all_zero(x, 4)) return false;
    stage_.add(x, ms, keep_);
    return true;
}

bool RateBook::observe_tier(int64_t rows, int64_t experts, uint64_t bytes_in, uint64_t bytes_paged, uint64_t bytes_pinned, int64_t calls,
                            double ms) {
    if (rows < 0 || experts < 0 || calls < 0 || !(ms >= 0 && ms <= kMaxObsMs)) return false;
    const double x[6] = {(double) rows, (double) bytes_paged, (double) bytes_pinned, (double) calls, (double) experts, (double) bytes_in};
    if (all_zero(x, 6)) return false;
    tier_.add(x, ms, keep_);
    return true;
}

Rates RateBook::rates() const {
    double s[kMaxVars], t[kMaxVars];
    stage_.solve(s);
    tier_.solve(t);
    Rates r{};
    r.stage_row_ms = s[0];
    r.stage_byte_ms_paged = s[1];
    r.stage_byte_ms_pinned = s[2];
    r.tier_row_ms = t[0];
    r.tier_byte_ms_paged = t[1];
    r.tier_byte_ms_pinned = t[2];
    r.tier_fixed_ms = t[3];
    r.stage_fixed_ms = s[3];
    r.tier_expert_ms = t[4];
    r.tier_input_byte_ms = t[5];
    return r;
}

// Each observation is scaled by its predicted time and weighed down when far off it, so one stall cannot set the rates.
void RateBook::Fit::add(const double* x, double y, double keep) {
    double now[kMaxVars];
    solve(now);
    double p = 0;
    for (int i = 0; i < n; ++i) p += now[i] * x[i];
    const double off = std::fabs(y / p - 1);
    const double w = off > kTrustRel ? kTrustRel / off : 1.0;
    for (int i = 0; i < n; ++i) {
        const double xi = x[i] / p;
        for (int j = 0; j < n; ++j) a[i][j] = keep * a[i][j] + w * xi * (x[j] / p);
        b[i] = keep * b[i] + w * xi * (y / p);
        sumsq[i] += w * xi * xi;
    }
    seen += w;
    if (taken < RateBook::kMinObs) ++taken;
}

// Ridge least squares toward the priors. Each variable is scaled by its root mean square so the prior weighs the same everywhere.
void RateBook::Fit::solve(double* out) const {
    for (int i = 0; i < n; ++i) out[i] = prior[i];
    if (taken < RateBook::kMinObs) return;
    double scale[kMaxVars] = {};
    bool seen_var[kMaxVars] = {}, floored[kMaxVars] = {};
    for (int i = 0; i < n; ++i) {
        if (!(a[i][i] > 0)) continue;   // never seen above zero: it keeps its prior
        scale[i] = std::sqrt(sumsq[i] / seen);
        seen_var[i] = true;
    }
    // the rate furthest below its floor is held there and the others are solved again, so together they still meet the data (a round per rate)
    for (int round = 0; round <= n; ++round) {
        int idx[kMaxVars], m = 0;
        for (int i = 0; i < n; ++i)
            if (seen_var[i] && !floored[i]) idx[m++] = i;
        double l[kMaxVars][kMaxVars] = {}, z[kMaxVars] = {};
        for (int p = 0; p < m; ++p) {
            const int i = idx[p];
            for (int q = 0; q <= p; ++q) {
                const int j = idx[q];
                double v = a[i][j] / (scale[i] * scale[j]) + (p == q ? kPriorWeight : 0);
                for (int k = 0; k < q; ++k) v -= l[p][k] * l[q][k];
                if (p == q) {
                    if (!(v > 0)) return;
                    l[p][p] = std::sqrt(v);
                } else {
                    l[p][q] = v / l[q][q];
                }
            }
            double v = b[i] / scale[i] + kPriorWeight * prior[i] * scale[i];
            for (int j = 0; j < n; ++j)   // what the floored rates already explain
                if (floored[j]) v -= a[i][j] * prior[j] * RateBook::kFloorFrac / scale[i];
            for (int k = 0; k < p; ++k) v -= l[p][k] * z[k];
            z[p] = v / l[p][p];
        }
        double g[kMaxVars] = {};
        for (int p = m - 1; p >= 0; --p) {
            double v = z[p];
            for (int k = p + 1; k < m; ++k) v -= l[k][p] * g[k];
            g[p] = v / l[p][p];
        }
        int worst = -1;   // one at a time: a rate pushed under only by another one comes back once that one is held
        double worst_of_floor = 1;
        for (int p = 0; p < m; ++p) {
            const int i = idx[p];
            const double v = g[p] / scale[i] / (prior[i] * RateBook::kFloorFrac);
            if (std::isfinite(v) && v < worst_of_floor) {
                worst = i;
                worst_of_floor = v;
            }
        }
        if (worst >= 0) {
            floored[worst] = true;
            continue;
        }
        for (int p = 0; p < m; ++p) {
            const double v = g[p] / scale[idx[p]];
            if (std::isfinite(v)) out[idx[p]] = v;
        }
        break;
    }
    for (int i = 0; i < n; ++i)
        if (floored[i]) out[i] = prior[i] * RateBook::kFloorFrac;
}

}  // namespace strata::prefill::planner
