// include/strata/core/residency_rule.hpp: when the decode's helpers for a short RAM (router look-ahead, io prefetch) run.
#pragma once

#include <cstdint>

namespace strata::core {

/// The look-ahead and io prefetch stay off while the last sample found every expert in RAM (docs/DETAILS.md).
struct ResidencyRule {
    int64_t every = 64;
    uint64_t drive_limit = 100000000;   ///< 100 MB
    bool held = false;                  ///< the last sample found every expert in RAM, and no request since read past the limit

    bool due(int64_t request) const { return every > 0 && request % every == 0; }
    /// Request `request` starts: `sample()` runs when due.  True when the helpers stay off for this request.
    template <class Sample>
    bool begin(int64_t request, Sample&& sample) {
        if (due(request)) held = sample();
        return held;
    }
    /// A request read `drive_bytes` of experts that were not in RAM.
    void end(uint64_t drive_bytes) {
        if (drive_bytes > drive_limit) held = false;
    }
};

}  // namespace strata::core
