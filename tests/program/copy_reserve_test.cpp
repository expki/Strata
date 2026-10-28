// The NVIDIA card's extra cache reserve beside the pinned RAM copy: its registration and the verify captures, sized to a request floor.
#include "strata/program/copy_reserve.hpp"

#include <cstdint>
#include <cstdio>

using namespace strata::program;

static int failed = 0, checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failed;
    }
}

static constexpr uint64_t kMiB = 1ull << 20;
static constexpr uint64_t kGiB = 1ull << 30;

int main() {
    // the registration: 2 MiB a GiB, the box's 98 GiB copy took 198 MiB of the 3090 (431 MiB free at load without it, 233 with it)
    check(copy_register_bytes(0) == 0, "no copy registers nothing");
    check(copy_register_bytes(100 * kGiB) == 200 * kMiB, "100 GiB registers 200 MiB");
    const uint64_t r98 = copy_register_bytes((uint64_t) (97.86 * (double) kGiB));
    check(r98 >= 190 * kMiB && r98 <= 205 * kMiB, "the box's copy: within the measured 198 MiB");
    // the box: at most 101.4 GiB of experts and 140 MiB of captures ask 350 + 140 + 203 - 431 = about 262 MiB more
    const uint64_t box = copy_vram_reserve((uint64_t) (101.4 * (double) kGiB), kVerifyCaptureBytes);
    check(box >= 255 * kMiB && box <= 270 * kMiB, "the box's reserve: about 262 MiB");
    // what it buys: the box's lowest request start (93 MiB at the old reserve) lands at the floor or above
    check(93 * kMiB + box >= kRequestFloorBytes, "the lowest request start reaches the floor");
    // it grows with the copy and the captures, by what each takes
    const uint64_t c50 = copy_vram_reserve(50 * kGiB, kVerifyCaptureBytes), c100 = copy_vram_reserve(100 * kGiB, kVerifyCaptureBytes);
    check(c100 - c50 == 100 * kMiB, "50 GiB more copy: 100 MiB more reserve");
    check(copy_vram_reserve(100 * kGiB, 240 * kMiB) - c100 == 100 * kMiB, "100 MiB more captures: 100 MiB more reserve");
    // a copy and captures the load's spare already holds ask nothing more
    check(copy_vram_reserve(0, 0) == 0, "nothing to hold: no extra");
    check(copy_vram_reserve(0, kLoadSpareBytes - kRequestFloorBytes) == 0, "captures the spare holds to the byte: no extra");
    check(copy_vram_reserve(0, kLoadSpareBytes - kRequestFloorBytes + kMiB) == kMiB, "a MiB past it: a MiB");
    if (failed) {
        std::fprintf(stderr, "copy_reserve_test: %d of %d checks failed\n", failed, checks);
        return 1;
    }
    std::printf("copy_reserve_test: %d checks passed\n", checks);
    return 0;
}
