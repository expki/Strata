// chunk_taps_test: which rows of a prompt chunk the DFlash2 taps cover (include/strata/prefill/chunk_taps.hpp). Pure.
#include "strata/prefill/chunk_taps.hpp"

#include <cstdio>

using strata::prefill::chunk_tap_first;

static int g_fail = 0;
static void expect(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

int main() {
    expect(chunk_tap_first(20480, 0, -1, 2047) == 20480 - 2047, "per-chunk mode: the last 2047 rows");
    expect(chunk_tap_first(1000, 4096, -1, 2047) == 0, "per-chunk mode: a short chunk taps all rows");
    expect(chunk_tap_first(2047, 0, -1, 2047) == 0, "per-chunk mode: a chunk of exactly 2047 rows taps all rows");
    expect(chunk_tap_first(2048, 0, -1, 2047) == 1, "per-chunk mode: a chunk of 2048 rows skips the first");
    expect(chunk_tap_first(4096, 8192, 99999, 2047) == 4096 - 2047, "per-chunk mode ignores first_pos");
    expect(chunk_tap_first(4096, 0, 18000, 0) == 4096, "range mode: a chunk before first_pos taps nothing");
    expect(chunk_tap_first(4096, 16384, 18000, 0) == 18000 - 16384, "range mode: first_pos inside the chunk");
    expect(chunk_tap_first(4096, 20480, 18000, 0) == 0, "range mode: a chunk after first_pos taps all rows");
    expect(chunk_tap_first(4096, 16384, 20480, 0) == 4096, "range mode: first_pos at the chunk end taps nothing");
    if (g_fail == 0) std::printf("chunk_taps_test: ok\n");
    return g_fail == 0 ? 0 : 1;
}
