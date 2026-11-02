// include/strata/prefill/chunk_taps.hpp - which rows of a prompt chunk the DFlash2 taps cover.  Pure, so it is unit
// tested without a GPU (chunk_taps_test).
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::prefill {

/// First tapped row of a chunk of `T` rows at position `p0`, relative to the chunk (T means no row).
/// `per_chunk` > 0: the last `per_chunk` rows of every chunk.  Otherwise the rows at positions >= `first_pos`.
inline int64_t chunk_tap_first(int64_t T, int64_t p0, int64_t first_pos, int64_t per_chunk) {
    if (per_chunk > 0) return std::max<int64_t>(0, T - per_chunk);
    return std::min<int64_t>(T, std::max<int64_t>(0, first_pos - p0));
}

}  // namespace strata::prefill
