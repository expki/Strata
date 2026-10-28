// A stand-in for the AMD tier library at this interface version, with every export (XTIER_STUB_NO_STATS: all but prefill_stats).
#include "strata/core/xtier_abi.hpp"

#if defined(_WIN32)
#define XTIER_STUB_EXPORT extern "C" __declspec(dllexport)
#else
#define XTIER_STUB_EXPORT extern "C" __attribute__((visibility("default")))
#endif

XTIER_STUB_EXPORT int strata_xtier_abi(void) { return STRATA_XTIER_ABI_VERSION * 100000 + strata_xtier_structs_size(); }
XTIER_STUB_EXPORT void* strata_xtier_open(int, char*, size_t) { return nullptr; }
XTIER_STUB_EXPORT void strata_xtier_close(void*) {}
XTIER_STUB_EXPORT int strata_xtier_mem_info(void*, uint64_t*, uint64_t*) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_register(void*, void*, uint64_t, void**) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_unregister(void*, void*) { return -1; }
XTIER_STUB_EXPORT const char* strata_xtier_last_error(void*) { return "stub"; }
XTIER_STUB_EXPORT int strata_xtier_flag_echo(void*, const uint32_t*, uint32_t*, uint32_t*, uint32_t, uint64_t) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_sync(void*) { return -1; }
XTIER_STUB_EXPORT void* strata_xtier_host_alloc(void*, uint64_t, int) { return nullptr; }
XTIER_STUB_EXPORT void strata_xtier_host_free(void*, void*) {}
XTIER_STUB_EXPORT int strata_xtier_probe(void*, int, void*, uint64_t, uint32_t, uint64_t*, float*) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_arena(void*, const uint64_t*, int64_t, uint64_t*) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_fill(void*, int64_t, const void*, uint64_t) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_read(void*, int64_t, void*, uint64_t) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_refill_record(void*) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_refill_done(void*, int) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_decode(void*, const strata_xtier_fmt*, const float*, int64_t, const strata_xtier_meta*) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_decode_out(void*, const strata_xtier_fmt*, const float*, int64_t, const strata_xtier_meta*, void*) {
    return -1;
}
XTIER_STUB_EXPORT int strata_xtier_decode_counts(void*, uint64_t*) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_decode_wait(void*, const float**, uint64_t) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_timing(void*, double*) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_prefill_open(void*, int64_t, int64_t, uint64_t, uint64_t, int64_t, uint64_t) { return -1; }
XTIER_STUB_EXPORT int strata_xtier_prefill_layer(void*, const strata_xtier_prefill_args*) { return -1; }
#if !defined(XTIER_STUB_NO_STATS)
// two fixed records, so the loader test can see the call go through
XTIER_STUB_EXPORT int strata_xtier_prefill_stats(void*, strata_xtier_prefill_record* out, int64_t cap, int64_t* n) {
    const strata_xtier_prefill_record r[2] = {{2, 7, 1234, 30, 5678, 9000, 1.5, 0.25, 3.0}, {0, 8, 10, 1, 20, 0, 0.5, 0.125, 0.0}};
    *n = cap < 2 ? cap : 2;
    for (int64_t i = 0; i < *n; ++i) out[i] = r[i];
    return 0;
}
#endif
