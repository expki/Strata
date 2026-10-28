// The decode spare-worker search on synthetic window times.
#include "strata/kernels/cpu/decode_spares.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

using strata::kernels::cpu::DecodeSpares;
using strata::kernels::cpu::DecodeSparesTuning;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("decode_spares_test: FAILED: %s\n", what);
        ++g_failures;
    }
}

using Cost = std::function<double(int spares, long window)>;

// Feeds windows to the search the way the engine does: a request of `request_len` windows at a time
struct Run {
    DecodeSpares d;
    std::mt19937 rng;
    double noise;
    long t = 0;
    int highest = 0;   // the highest count ever applied
    Run(int workers, int max_spares, double noise_, unsigned seed, DecodeSparesTuning tun = {})
        : d(workers, max_spares, tun), rng(seed), noise(noise_) {}
    void step(const Cost& cost, int request_len) {
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        if (t % request_len == 0) d.begin_request();
        const double ms = cost(d.spares(), t) * (1.0 + noise * u(rng));
        highest = std::max(highest, d.spares());
        d.observe(ms);
        ++t;
    }
    void go(long windows, const Cost& cost, int request_len = 100) {
        for (long i = 0; i < windows; ++i) step(cost, request_len);
    }
    // runs until the best count is `want` or `limit` windows pass: the windows it took, or -1
    long until_best(int want, const Cost& cost, long limit, int request_len = 100) {
        for (long i = 0; i < limit; ++i) {
            step(cost, request_len);
            if (d.best() == want) return i + 1;
        }
        return -1;
    }
};

Cost table(std::vector<double> by_spares) {
    return [by_spares](int k, long) { return by_spares[(size_t) std::min<int>(k, (int) by_spares.size() - 1)]; };
}

}  // namespace

int main() {
    // settles_on_best: 30 ms at 0, 28 at 1, 26.5 at 2, 26.8 at 3, 5% noise on every window
    {
        Run r(16, 4, 0.05, 1);
        const long got = r.until_best(2, table({30, 28, 26.5, 26.8, 28}), 3000);
        check(got > 0 && got < 2500, "settles_on_best: reaches 2 within 2,500 windows (two probes of 8 requests)");
        r.go(2000, table({30, 28, 26.5, 26.8, 28}));
        check(r.d.best() == 2 && r.d.spares() == 2, "settles_on_best: and stays at 2 (3 was tried and lost)");
        check(r.highest <= 3, "settles_on_best: never above the one probe of 3");
        DecodeSpares::Look c;
        bool moved_1_to_2 = false;
        Run q(16, 4, 0.05, 1);
        for (long i = 0; i < 4000; ++i) {
            q.step(table({30, 28, 26.5, 26.8, 28}), 100);
            if (q.d.take_look(c) && c.moved && c.from == 1 && c.to == 2 && c.gain > 0.02 && c.cost_to < c.cost_from) moved_1_to_2 = true;
        }
        check(moved_1_to_2, "settles_on_best: the change 1 to 2 is reported");
        check(!q.d.take_look(c) || c.windows > 0, "settles_on_best: a look is reported once");
    }
    // the work varies (rows 1 to 4) and the time follows it: the cost per row decides
    {
        Run r(16, 4, 0.0, 5);
        std::mt19937 w(9);
        const std::vector<double> by = {30, 28, 26.5, 26.8, 28};
        for (long i = 0; i < 4000; ++i) {
            if (i % 100 == 0) r.d.begin_request();
            const double rows = 1.0 + (double) (w() % 4);
            r.d.observe(by[(size_t) r.d.spares()] * (0.4 + 0.2 * rows) * (1.0 + 0.04 * ((double) (w() % 200) / 100.0 - 1.0)), rows);
        }
        check(r.d.best() == 2, "work_normalised: the best is 2 with windows of 1 to 4 rows");
    }
    // no_gain_stays: a flat cost under AR(1) noise (s 0.10, rho 0.9, levels 30% apart) rarely parks and never sits above 1
    {
        struct Share { double ge1 = 0, ge2 = 0; int highest = 0; };
        auto flat_run = [](double z, unsigned seeds, long windows) {
            Share out;
            for (unsigned seed = 1; seed <= seeds; ++seed) {
                DecodeSparesTuning tun;
                tun.z = z;
                DecodeSpares d(15, 3, tun);
                std::mt19937 rng(seed);
                std::normal_distribution<double> n01(0.0, 1.0);
                double e = 0.0, level = 1.0;
                long left = 0, a1 = 0, a2 = 0;
                for (long w = 0; w < windows; ++w) {
                    if (left == 0) {
                        left = 40 + (long) (rng() % 91);
                        level = std::exp(0.3 * n01(rng));
                        e = 0.0;
                        d.begin_request();
                    }
                    --left;
                    e = 0.9 * e + std::sqrt(1.0 - 0.81) * 0.10 * n01(rng);
                    d.observe(30.0 * level * std::exp(e));
                    a1 += d.best() >= 1;
                    a2 += d.best() >= 2;
                    out.highest = std::max(out.highest, d.best());
                }
                out.ge1 += (double) a1 / (double) windows / seeds;
                out.ge2 += (double) a2 / (double) windows / seeds;
            }
            return out;
        };
        const Share two = flat_run(2.0, 20, 40000);
        std::printf("  no_gain_stays at z 2: %.1f%% of the windows at 1 or more, %.1f%% at 2 or more, highest %d\n", 100 * two.ge1, 100 * two.ge2, two.highest);
        check(two.ge1 < 0.10, "no_gain_stays: under 10% of the windows at 1 or more");
        check(two.ge2 < 0.03, "no_gain_stays: under 3% of the windows at 2 or more");
        const Share one = flat_run(1.0, 20, 40000);
        std::printf("  no_gain_stays at z 1: %.1f%% of the windows at 1 or more, %.1f%% at 2 or more, highest %d\n", 100 * one.ge1, 100 * one.ge2, one.highest);
        check(one.ge1 > 0.10, "no_gain_stays: the test sees a one-sigma rule park on noise");
    }
    // worse_returns: a cost that rises with the count goes back to 0 after one probe
    {
        Run r(16, 4, 0.05, 2);
        const Cost worse = table({30, 31.5, 33, 34.5, 36});
        r.go(900, worse);
        check(r.d.best() == 0 && r.d.spares() == 0, "worse_returns: back at 0 after one probe");
        r.go(6000, worse);
        check(r.d.best() == 0 && r.highest == 1, "worse_returns: only 1 was ever tried");
    }
    // drift_safe: a cost that grows 0.1% a window (a growing context), the best at 1
    {
        Run r(16, 4, 0.03, 3);
        const std::vector<double> base = {30, 28, 30, 31};
        r.go(3000, [&](int k, long t) { return base[(size_t) std::min(k, 3)] * (1.0 + 0.001 * (double) t); });
        check(r.d.best() == 1, "drift_safe: settles on 1 under a rising cost");
        Run flat(16, 4, 0.03, 4);
        flat.go(3000, [](int, long t) { return 30.0 * (1.0 - 0.0003 * (double) t); });
        check(flat.d.best() == 0, "drift_safe: a falling cost with no gain from parking stays at 0");
        Run rising(16, 4, 0.03, 6);
        rising.go(3000, [](int, long t) { return 30.0 * (1.0 + 0.001 * (double) t); });
        check(rising.d.best() == 0, "drift_safe: a rising cost with no gain from parking stays at 0");
    }
    // cap: never above max_spares, and 0 means always 0
    {
        Run r(16, 2, 0.02, 7);
        r.go(6000, table({30, 28, 26, 24, 22}));
        check(r.d.best() == 2 && r.highest <= 2, "cap: stops at max_spares 2");
        Run z(16, 0, 0.02, 8);
        z.go(1000, table({30, 20}));
        check(z.d.best() == 0 && z.highest == 0, "cap: max_spares 0 never parks");
        Run one(1, 4, 0.02, 9);
        one.go(1000, table({30, 20}));
        check(one.highest == 0, "cap: a pool of 1 worker never parks");
        Run two(2, 5, 0.02, 10);
        two.go(6000, table({30, 20, 10}));
        check(two.highest == 1, "cap: a pool of 2 parks at most 1");
    }
    // recheck: the best moves from 2 to 0 and back, and the search follows both
    {
        Run r(16, 4, 0.03, 11);
        const Cost best2 = table({30, 28, 26.5, 28, 29}), best0 = table({30, 32, 34, 36, 38});
        check(r.until_best(2, best2, 3000) > 0, "recheck: starts at 2");
        check(r.until_best(0, best0, 4096 + 4500) > 0, "recheck: follows a cost whose best is now 0");
        check(r.until_best(2, best2, 4096 + 4500) > 0, "recheck: follows it back to 2");
    }
    // begin_request: the first windows of a request are not used
    {
        Run r(16, 4, 0.02, 13);
        const std::vector<double> by = {30, 28, 26.5, 26.8, 28};
        for (long i = 0; i < 5000; ++i) {
            if (i % 100 == 0) r.d.begin_request();
            const bool cold = i % 100 < 4;   // the 4 ignored windows are wildly slow
            r.d.observe((cold ? 1000.0 : 1.0) * by[(size_t) r.d.spares()]);
        }
        check(r.d.best() == 2, "begin_request: slow first windows do not change the answer");
    }
    // clip: a stall in a window does not decide a probe
    {
        Run r(16, 4, 0.02, 14);
        const std::vector<double> by = {30, 28, 26.5, 26.8, 28};
        for (long i = 0; i < 5000; ++i) {
            if (i % 100 == 0) r.d.begin_request();
            const bool stall = i % 50 == 17 && r.d.spares() == 1;   // the 1 blocks meet a 1 s stall
            r.d.observe((stall ? 1000.0 : 1.0) * by[(size_t) r.d.spares()]);
        }
        check(r.d.best() >= 1, "clip: stalls that only hit the neighbour do not hide its gain");
    }
    if (g_failures != 0) {
        std::printf("decode_spares_test: %d FAILED\n", g_failures);
        return 1;
    }
    std::printf("decode_spares_test OK\n");
    return 0;
}
