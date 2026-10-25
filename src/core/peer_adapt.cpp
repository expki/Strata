// Which experts a peer tier swaps in at one adapt round (see include/strata/core/peer_adapt.hpp).
#include "strata/core/peer_adapt.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace strata::core {

std::vector<PeerSwap> peer_adapt_choose(const float* usage, const int32_t* other, const int32_t* mine,
                                        const std::vector<std::vector<int32_t>>& free_slots_by_layer,
                                        int64_t n_layers, int64_t n_expert, int max_swaps) {
    std::vector<PeerSwap> swaps;
    if (max_swaps <= 0) return swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage + l * n_expert;
        const int32_t* r0 = other + l * n_expert;
        const int32_t* r1 = mine + l * n_expert;
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
            if (r1[e] >= 0) vict.emplace_back(u[e], e);
            else if (r0[e] < 0 && u[e] >= 2.0f) cand.emplace_back(u[e], e);
        }
        const size_t nf = (size_t) l < free_slots_by_layer.size() ? free_slots_by_layer[(size_t) l].size() : 0;
        if (cand.empty() || (vict.empty() && nf == 0)) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t f = std::min(nf, cand.size());   // the hottest take the free slots
        for (size_t i = 0; i < f; ++i) swaps.push_back({cand[i].first, (int32_t) l, cand[i].second, -1});
        const size_t nc = std::min(cand.size() - f, vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[f + i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[f + i].first - vict[i].first, (int32_t) l, cand[f + i].second, vict[i].second});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](const PeerSwap& a, const PeerSwap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    return swaps;
}

}  // namespace strata::core
