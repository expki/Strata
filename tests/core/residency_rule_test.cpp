// ResidencyRule: when the decode samples whether every expert is in RAM, and when its look-ahead and io prefetch run. CPU only.
#include "strata/core/residency_rule.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::core::ResidencyRule;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

}  // namespace

int main() {
    {   // samples on requests 0, 64 and 128 only
        const ResidencyRule r;
        check(r.due(0) && r.due(64) && r.due(128), "due on 0, 64 and 128");
        check(!r.due(1) && !r.due(63) && !r.due(65) && !r.due(127), "not due between samples");
    }
    {   // begin runs the sample only when due, and the helpers stay off while the last sample held
        ResidencyRule r;
        std::vector<int64_t> sampled;
        int64_t off = 0;
        for (int64_t q = 0; q <= 130; ++q)
            off += r.begin(q, [&] { sampled.push_back(q); return true; }) ? 1 : 0;
        check(sampled == std::vector<int64_t>({0, 64, 128}), "begin samples on 0, 64 and 128");
        check(off == 131, "every request after a held sample runs without the helpers");
    }
    {   // a failed sample turns the helpers back on until a later sample holds
        ResidencyRule r;
        check(r.begin(0, [] { return true; }), "request 0 held");
        check(r.begin(1, [] { return true; }), "request 1 keeps the last sample");
        check(!r.begin(64, [] { return false; }), "a failed sample on 64 turns the helpers on");
        check(!r.begin(65, [] { return true; }), "65 is not sampled, the helpers stay on");
        check(!r.begin(127, [] { return true; }), "127 is not sampled either");
        check(r.begin(128, [] { return true; }), "a held sample on 128 turns them off again");
    }
    {   // a request that read over the limit from the drive turns them on until the next sample
        ResidencyRule r;
        check(r.begin(0, [] { return true; }), "held at 0");
        r.end(r.drive_limit);
        check(r.begin(1, [] { return false; }), "a read of exactly the limit keeps the fast path");
        r.end(r.drive_limit + 1);
        check(!r.begin(2, [] { return true; }), "a read over the limit turns the helpers on");
        check(!r.begin(63, [] { return true; }), "still on before the next sample");
        check(r.begin(64, [] { return true; }), "the next held sample turns them off");
    }
    {   // never sampled: the helpers run as before
        ResidencyRule r;
        r.every = 0;
        int calls = 0;
        for (int64_t q = 0; q < 200; ++q) check(!r.begin(q, [&] { ++calls; return true; }), "every 0: helpers on");
        check(calls == 0, "every 0: no sample");
    }
    std::printf("residency_rule_test: %d checks passed\n", checks);
    return 0;
}
