// --expert-gib: the per-card expert memory cap parses as asked and refuses what it cannot read. CPU only.
#include "strata/program/expert_gib.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

using strata::program::ExpertGib;
using strata::program::expert_cap_bytes;
using strata::program::parse_expert_gib;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

}  // namespace

int main() {
    ExpertGib g;
    std::string err;
    check(parse_expert_gib("cuda0=1,cuda2=5.5,hip0=6.5", g, err), "a good list parses");
    check(expert_cap_bytes(g, false, 0) == 1ull << 30, "cuda0 is 1 GiB");
    check(expert_cap_bytes(g, false, 2) == (uint64_t) (5.5 * 1073741824.0), "cuda2 is 5.5 GiB");
    check(expert_cap_bytes(g, true, 0) == (uint64_t) (6.5 * 1073741824.0), "hip0 is 6.5 GiB");
    check(expert_cap_bytes(g, false, 1) == UINT64_MAX, "a CUDA card not listed has no cap");
    check(expert_cap_bytes(g, true, 1) == UINT64_MAX, "an AMD card not listed has no cap");
    check(expert_cap_bytes(g, false, -1) == UINT64_MAX, "a negative device has no cap");

    // the same index names two different cards on the two makes
    check(parse_expert_gib("cuda1=3,hip1=2", g, err), "the same index on both makes parses");
    check(expert_cap_bytes(g, false, 1) == 3ull << 30 && expert_cap_bytes(g, true, 1) == 2ull << 30, "the makes stay apart");
    check(expert_cap_bytes(g, false, 0) == UINT64_MAX, "a second parse replaces the first");

    // one key alone, and a cap below one GiB
    check(parse_expert_gib("hip0=0.25", g, err) && expert_cap_bytes(g, true, 0) == 1ull << 28, "a quarter GiB");

    for (const char* bad : {"", "cuda0", "cuda0=", "cuda=1", "gpu0=1", "cuda0=-1", "cuda0=1,cuda0=2", "cuda0=x",
                            "cuda0=0", "hip0=0", "cuda0=nan", "cuda0=inf", "cuda0=1e999", "cuda-1=1", "cuda0=1,",
                            ",cuda0=1", "cuda0=1,,cuda1=2", "cuda0=1,hip0=2,hip0=3", "cuda0=1gib", "=1", "cuda0=1=2"}) {
        err.clear();
        const ExpertGib before = g;
        check(!parse_expert_gib(bad, g, err) && !err.empty(), bad);
        check(g.cuda == before.cuda && g.hip == before.hip, "a refused list leaves the result as it was");
    }
    std::printf("expert_gib_test: %d checks passed\n", checks);
    return 0;
}
