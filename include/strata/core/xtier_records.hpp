// The AMD tier library's finished prompt layers until strata_xtier_prefill_stats takes them. No GPU needed.
#pragma once

#include "strata/core/xtier_abi.hpp"

#include <algorithm>
#include <cstdint>

namespace strata::core::xtier {

// The prompt layers whose events have run, oldest first: a ring that drops the oldest when full.
struct PromptRecords {
    strata_xtier_prefill_record r[STRATA_XTIER_PF_RECORDS];
    int64_t first = 0, n = 0;
    void push(const strata_xtier_prefill_record& v) {
        if (n == STRATA_XTIER_PF_RECORDS) {
            first = (first + 1) % STRATA_XTIER_PF_RECORDS;
            --n;
        }
        r[(first + n) % STRATA_XTIER_PF_RECORDS] = v;
        ++n;
    }
    // At most cap of the oldest into out, and their count. The rest wait for the next call.
    int64_t take(strata_xtier_prefill_record* out, int64_t cap) {
        const int64_t m = std::min(cap, n);
        for (int64_t i = 0; i < m; ++i) out[i] = r[(first + i) % STRATA_XTIER_PF_RECORDS];
        first = (first + m) % STRATA_XTIER_PF_RECORDS;
        n -= m;
        return m;
    }
};

}  // namespace strata::core::xtier
