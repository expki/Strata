// Which experts a peer tier swaps in at one adapt round. It needs no CUDA or HIP, so it is tested on the CPU.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::core {

struct PeerSwap { float gain; int32_t layer, in, out; };   // out = -1: a free slot

/// One adapt round's swaps, free slots first. `other` is the stages' residency (host_res) and `mine` the tier's.
std::vector<PeerSwap> peer_adapt_choose(const float* usage, const int32_t* other, const int32_t* mine,
                                        const std::vector<std::vector<int32_t>>& free_slots_by_layer,
                                        int64_t n_layers, int64_t n_expert, int max_swaps);

}  // namespace strata::core
