// peer_adapt_choose: the AMD tier's swaps for one adapt round, and the CUDA peer's choices pinned to today's. CPU only.
#include "strata/core/peer_adapt.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

using strata::core::PeerSwap;
using strata::core::peer_adapt_choose;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

struct Swap { float gain; int32_t layer, in, out; };

// PeerExperts::adapt's selection as it was at ec479015, kept unchanged to pin the CUDA peer's choices
std::vector<Swap> todays_choice(const float* usage, const int32_t* res0, const int32_t* res1, int64_t n_layers_,
                                int64_t n_expert_, int max_swaps) {
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage + l * n_expert_;
        const int32_t* r0 = res0 + l * n_expert_;
        const int32_t* r1 = res1 + l * n_expert_;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            if (r1[e] >= 0) vict.emplace_back(u[e], e);
            else if (r0[e] < 0 && u[e] >= 2.0f) cand.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    if (swaps.empty()) return swaps;
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    return swaps;
}

// the (layer, expert) pairs swapped in, or out, in sorted order: the same on any standard library
template <class S>
std::vector<std::pair<int32_t, int32_t>> pairs(const std::vector<S>& s, bool in) {
    std::vector<std::pair<int32_t, int32_t>> p;
    for (const S& w : s) p.emplace_back(w.layer, in ? w.in : w.out);
    std::sort(p.begin(), p.end());
    return p;
}

template <class S>
std::vector<float> gains(const std::vector<S>& s) {
    std::vector<float> g;
    for (const S& w : s) g.push_back(w.gain);
    return g;
}

template <class A, class B>
bool same(const std::vector<A>& a, const std::vector<B>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].gain != b[i].gain || a[i].layer != b[i].layer || a[i].in != b[i].in || a[i].out != b[i].out) return false;
    return true;
}

const std::vector<std::vector<int32_t>> kNoFree;

}  // namespace

int main() {
    // today's order on a hand-made table: equal usage among candidates, victims and gains, and the cap cuts a tie
    {
        const float u[16] = {5, 0, 3, 5, 0.5f, 2, 0, 4,   4, 4, 0, 0, 2, 9, 1, 0};
        const int32_t other[16] = {0, -1, -1, -1, -1, -1, -1, -1,   -1, -1, -1, -1, -1, 7, -1, -1};
        const int32_t mine[16] = {-1, 0, -1, -1, 1, -1, 2, -1,   -1, -1, 3, 4, -1, -1, 5, -1};
        const auto t3 = todays_choice(u, other, mine, 2, 8, 3), t8 = todays_choice(u, other, mine, 2, 8, 8);
        check(same(peer_adapt_choose(u, other, mine, kNoFree, 2, 8, 3), t3), "today's choice, ties too, cap 3");
        check(same(peer_adapt_choose(u, other, mine, kNoFree, 2, 8, 8), t8), "today's choice, ties too, cap 8");
        check(gains(t3) == std::vector<float>{5, 4, 4} && t3[0].layer == 0 && t3[0].in == 3, "today's choice, cap 3");
        check(gains(t8) == std::vector<float>{5, 4, 4, 4, 2.5f}, "today's gains, cap 8");
        const std::vector<std::pair<int32_t, int32_t>> in8 = {{0, 2}, {0, 3}, {0, 7}, {1, 0}, {1, 1}};
        const std::vector<std::pair<int32_t, int32_t>> out8 = {{0, 1}, {0, 4}, {0, 6}, {1, 2}, {1, 3}};
        check(pairs(t8, true) == in8 && pairs(t8, false) == out8, "today's experts in and out, cap 8");
    }
    // a candidate a stage holds is never chosen
    {
        const float u[4] = {0, 5, 0, 0};
        const int32_t other[4] = {-1, 3, -1, -1};
        const int32_t mine[4] = {0, -1, -1, -1};
        check(peer_adapt_choose(u, other, mine, kNoFree, 1, 4, 8).empty(), "no swap of an expert a stage holds");
    }
    // a free slot is taken before a victim, and the hottest candidates take the free slots
    {
        const float u[4] = {0, 0, 4, 0};
        const int32_t other[4] = {-1, -1, -1, -1};
        const int32_t mine[4] = {0, -1, -1, -1};
        check(same(peer_adapt_choose(u, other, mine, {{7}}, 1, 4, 8), std::vector<PeerSwap>{{4, 0, 2, -1}}),
              "a free slot before a victim");
        const int32_t none[4] = {-1, -1, -1, -1};
        check(same(peer_adapt_choose(u, other, none, {{7}}, 1, 4, 8), std::vector<PeerSwap>{{4, 0, 2, -1}}),
              "a free slot in a layer with no victim");
        const float u6[6] = {0, 1, 0, 6, 5, 4};
        const int32_t other6[6] = {-1, -1, -1, -1, -1, -1};
        const int32_t mine6[6] = {0, 1, -1, -1, -1, -1};
        const std::vector<PeerSwap> want6 = {{6, 0, 3, -1}, {5, 0, 4, 0}, {3, 0, 5, 1}};
        check(same(peer_adapt_choose(u6, other6, mine6, {{9}}, 1, 6, 8), want6), "the hottest takes the free slot, the next ones the victims");
    }
    // the gain threshold
    {
        float u[4] = {2.0f, 0, 3.0f, 0};
        const int32_t other[4] = {-1, -1, -1, -1};
        const int32_t mine[4] = {0, -1, -1, -1};
        check(peer_adapt_choose(u, other, mine, kNoFree, 1, 4, 8).empty(), "gain 1.0 is below 1.5: no swap");
        u[2] = 3.6f;
        check(same(peer_adapt_choose(u, other, mine, kNoFree, 1, 4, 8), std::vector<PeerSwap>{{3.6f - 2.0f, 0, 2, 0}}),
              "gain 1.6: one swap");
    }
    // the cap: 10 eligible swaps, max_swaps 4 gives the 4 largest gains
    {
        float u[20] = {};
        int32_t other[20], mine[20];
        for (int e = 0; e < 20; ++e) {
            other[e] = -1;
            mine[e] = e < 10 ? e : -1;
            if (e >= 10) u[e] = (float) (e - 8);   // 2 .. 11
        }
        const std::vector<PeerSwap> s = peer_adapt_choose(u, other, mine, kNoFree, 1, 20, 4);
        check(s.size() == 4, "the cap keeps 4");
        check(s.size() == 4 && s[0].gain == 11 && s[1].gain == 10 && s[2].gain == 9 && s[3].gain == 8, "the largest gains");
        check(s.size() == 4 && s[0].in == 19 && s[1].in == 18 && s[2].in == 17 && s[3].in == 16, "the hottest experts");
        check(peer_adapt_choose(u, other, mine, kNoFree, 1, 20, 0).empty(), "max_swaps 0 gives none");
    }
    // random tables: no free slots gives today's choice, and no result swaps in an expert either side holds
    {
        std::mt19937 rng(7);
        const int64_t L = 3, E = 256;
        std::vector<float> u((size_t) (L * E));
        std::vector<int32_t> other((size_t) (L * E)), mine((size_t) (L * E));
        for (int round = 0; round < 300; ++round) {
            for (size_t i = 0; i < u.size(); ++i) {
                u[i] = (float) (rng() % 13) * 0.5f;   // halves from 0 to 6: many ties
                const unsigned r = rng() % 10;
                other[i] = r < 3 ? (int32_t) i : -1;
                mine[i] = r >= 3 && r < 6 ? (int32_t) i : -1;
            }
            const int cap = 1 + (int) (rng() % 64);
            const std::vector<std::vector<int32_t>> empty_layers((size_t) L);
            const auto t = todays_choice(u.data(), other.data(), mine.data(), L, E, cap);
            check(same(peer_adapt_choose(u.data(), other.data(), mine.data(), kNoFree, L, E, cap), t),
                  "no free slots: today's choice");
            check(same(peer_adapt_choose(u.data(), other.data(), mine.data(), empty_layers, L, E, cap), t),
                  "empty free lists: today's choice");
            std::vector<std::vector<int32_t>> fr((size_t) L);
            for (int64_t l = 0; l < L; ++l)
                for (unsigned k = rng() % 6; k > 0; --k) fr[(size_t) l].push_back((int32_t) (1000 + k));
            const auto s = peer_adapt_choose(u.data(), other.data(), mine.data(), fr, L, E, cap);
            check((int) s.size() <= cap, "at most max_swaps");
            std::vector<int> ins((size_t) (L * E), 0), outs((size_t) (L * E), 0), frees((size_t) L, 0);
            for (const PeerSwap& w : s) {
                const size_t in = (size_t) (w.layer * E + w.in);
                check(other[in] < 0 && mine[in] < 0, "no swap brings in an expert a stage or the tier holds");
                check(u[in] >= 2.0f, "a candidate is routed at least twice");
                check(++ins[in] == 1, "an expert comes in once");
                if (w.out < 0) {
                    check(++frees[(size_t) w.layer] <= (int) fr[(size_t) w.layer].size(), "no more free swaps than free slots");
                } else {
                    const size_t out = (size_t) (w.layer * E + w.out);
                    check(mine[out] >= 0, "a victim is the tier's");
                    check(++outs[out] == 1, "a victim goes once");
                    check(w.gain >= 1.5f, "a victim swap gains at least 1.5");
                }
            }
            for (size_t i = 1; i < s.size(); ++i) check(s[i - 1].gain >= s[i].gain, "the largest gains first");
        }
    }
    std::printf("peer_adapt_test: %d checks ok\n", checks);
    return 0;
}
