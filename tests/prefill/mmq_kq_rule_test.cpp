// Which K-quant types take the MMQ prompt kernels (kq::wanted): STRATA_MMQ_KQ unset, 0, 1 and anything else. No GPU, no model.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "strata/prefill/mmq_kq_abi.hpp"

using strata::prefill::mmq::kq::wanted;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static void set_env(const char* v) {
#if defined(_WIN32)
    _putenv_s("STRATA_MMQ_KQ", v != nullptr ? v : "");
#else
    if (v != nullptr) setenv("STRATA_MMQ_KQ", v, 1);
    else unsetenv("STRATA_MMQ_KQ");
#endif
}

// the rule reads the variable once, so each process checks one setting: unset | 0 | 1 | <anything else>
int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "unset";
    set_env(std::strcmp(mode, "unset") == 0 ? nullptr : mode);
    constexpr int kQ4K = 12, kQ5K = 13, kQ5_1 = 7, kQ6K = 14;
    const bool off = std::strcmp(mode, "0") == 0, all = std::strcmp(mode, "1") == 0;
    CHECK(wanted(kQ6K) == !off);                // Q6_K is on by itself, STRATA_MMQ_KQ=0 turns it off
    CHECK(wanted(kQ4K) == all);                 // the others need STRATA_MMQ_KQ=1
    CHECK(wanted(kQ5K) == all);
    CHECK(wanted(kQ5_1) == all);
    std::printf("mmq_kq_rule_test %s: %s\n", mode, failed == 0 ? "ok" : "FAILED");
    return failed == 0 ? 0 : 1;
}
