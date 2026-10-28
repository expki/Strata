// Picks how many CPU pool workers to park while the engine decodes, from measured window times (pure, no clocks).
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace strata::kernels::cpu {

struct DecodeSparesTuning {
    int block = 16;           ///< windows in a block, the counts alternate block by block inside a request
    int warmup = 4;           ///< windows ignored at the start of a request
    int requests = 8;         ///< requests a probe compares two counts in
    int recheck = 4096;       ///< windows between looks once settled
    double min_gain = 0.02;   ///< the least share of the cost a neighbour must save
    double z = 2.0;           ///< the standard errors of the saving it must beat
    double clip = 2.0;        ///< a window counts for at most this many times the recent cost (0: as measured)
};

class DecodeSpares {
public:
    /// What a finished probe found, for the log line
    struct Look {
        int from = 0, to = 0;     ///< the best count and the neighbour it was compared with
        bool moved = false;       ///< the neighbour became the best count
        double gain = 0;          ///< the neighbour's saving as a share of the best's cost
        double margin = 0;        ///< the saving it had to beat
        double cost_from = 0, cost_to = 0;   ///< mean costs in ms per unit of work
        long windows = 0;         ///< windows observed so far
    };

    /// At most `max_spares` of `workers` are parked, 0 never parks
    DecodeSpares(int workers, int max_spares, DecodeSparesTuning t = {})
        : t_(t), max_(std::max(0, std::min(max_spares, workers - 1))) {
        t_.block = std::max(1, t_.block);
        t_.requests = std::max(3, t_.requests);
        since_ = t_.recheck;   // the first look at the first request
        want_ = true;
    }

    /// One decode window took `ms` for `work` units (rows verified), returns the spare count for the next window
    int observe(double ms, double work = 1.0) {
        if (max_ == 0) return applied_;
        if (skip_ > 0) { --skip_; return applied_; }
        if (!(ms > 0.0) || !(work > 0.0)) return applied_;
        ++seen_;
        const double cost = clipped(ms / work);
        if (!probing_) {
            if (++since_ >= t_.recheck) want_ = true;   // the look starts with the next request
            return applied_;
        }
        acc_ms_ += cost * work;
        acc_work_ += work;
        if (++in_block_ < t_.block) return applied_;
        (on_b(block_) ? b_ms_ : a_ms_) += acc_ms_;
        (on_b(block_) ? b_work_ : a_work_) += acc_work_;
        acc_ms_ = acc_work_ = 0.0;
        in_block_ = 0;
        ++block_;
        applied_ = on_b(block_) ? cand_ : best_;
        return applied_;
    }

    /// The count to apply now
    int spares() const { return applied_; }
    int best() const { return best_; }
    int max_spares() const { return max_; }

    /// A request starts: its first windows are skipped, the block being filled is dropped, and a look begins here
    void begin_request() {
        skip_ = t_.warmup;
        acc_ms_ = acc_work_ = 0.0;
        in_block_ = 0;
        if (probing_) close_request();
        if (!probing_ && want_ && max_ > 0) start_sequence();
        if (probing_) {
            first_b_ = !first_b_;
            block_ = 0;
            applied_ = on_b(0) ? cand_ : best_;
        } else {
            applied_ = best_;
        }
    }

    /// The last finished probe, once
    bool take_look(Look& out) {
        if (!looked_) return false;
        looked_ = false;
        out = look_;
        return true;
    }

private:
    // The side of a block: a request starts on A and on B in turn, so a drift inside requests falls on both alike
    bool on_b(int block) const { return (block + (first_b_ ? 1 : 0)) % 2 == 1; }

    // A stall (a page fault, a disk read) counts for at most `clip` times the recent cost
    double clipped(double cost) {
        if (!(t_.clip > 0.0)) return cost;
        if (ref_ <= 0.0) ref_ = cost;
        const double c = std::min(cost, t_.clip * ref_);
        ref_ += 0.02 * (c - ref_);
        return c;
    }

    // Up from the best count first (down at the cap)
    void start_sequence() {
        moved_ = false;
        start_probe(best_ < max_ ? +1 : -1);
    }

    void start_probe(int dir) {
        const int cand = best_ + dir;
        if (cand < 0 || cand > max_) { settle(); return; }
        dir_ = dir;
        cand_ = cand;
        probing_ = true;
        want_ = false;
        a_ms_ = a_work_ = b_ms_ = b_work_ = 0.0;
        gains_.clear();
        cost_a_.clear();
        cost_b_.clear();
        seen_requests_ = 0;
    }

    void settle() {
        probing_ = false;
        want_ = false;
        since_ = 0;
        applied_ = best_;
    }

    // A request ended: what it measured on each side is one paired comparison, a request with one side adds none
    void close_request() {
        ++seen_requests_;
        if (a_work_ > 0.0 && b_work_ > 0.0) {
            const double a = a_ms_ / a_work_, b = b_ms_ / b_work_;
            gains_.push_back((a - b) / a);
            cost_a_.push_back(a);
            cost_b_.push_back(b);
        }
        a_ms_ = a_work_ = b_ms_ = b_work_ = 0.0;
        if ((int) gains_.size() >= t_.requests) finish_probe();
        else if (seen_requests_ >= 4 * t_.requests) settle();   // the requests are too short to tell
    }

    static double mean(const std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x;
        return s / (double) v.size();
    }

    void finish_probe() {
        const double n = (double) gains_.size();
        const double gain = mean(gains_);
        double var = 0.0;
        for (double g : gains_) var += (g - gain) * (g - gain);
        const double se = std::sqrt(var / (n - 1.0) / n);
        const double margin = std::max(t_.min_gain, t_.z * se);
        // Parking has to pay for itself: more workers parked win only by the margin, fewer win unless they lose by it
        const bool win = dir_ > 0 ? gain > margin : gain > -margin;
        look_ = Look{best_, cand_, win, gain, margin, mean(cost_a_), mean(cost_b_), seen_};
        looked_ = true;
        if (win) {
            best_ = cand_;
            moved_ = true;
            start_probe(dir_);   // keep going the same direction
            return;
        }
        if (dir_ > 0 && !moved_ && best_ > 0) { start_probe(-1); return; }   // nothing above, try below
        settle();
    }

    DecodeSparesTuning t_;
    int max_ = 0;
    int best_ = 0;
    int applied_ = 0;
    int cand_ = 0;
    int dir_ = 1;
    bool probing_ = false;
    bool want_ = false;
    bool moved_ = false;
    bool first_b_ = false;
    int block_ = 0;
    int in_block_ = 0;
    int skip_ = 0;
    int seen_requests_ = 0;
    long since_ = 0;
    long seen_ = 0;
    double acc_ms_ = 0.0, acc_work_ = 0.0;
    double a_ms_ = 0.0, a_work_ = 0.0, b_ms_ = 0.0, b_work_ = 0.0;   // this request's blocks on each side
    double ref_ = 0.0;
    std::vector<double> gains_, cost_a_, cost_b_;   // one entry per request of the probe
    bool looked_ = false;
    Look look_;
};

}  // namespace strata::kernels::cpu
