// half_rows: a chunk read as two token halves, when it is, each half's rows for its call to the AMD tier and the stage's rows of both halves. CPU only.
#include "strata/prefill/half_rows.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace strata::prefill;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

constexpr int64_t kK = 10, kExperts = 64;

// A routing as the router gives it: k distinct experts per token, skewed toward the low ids
std::vector<int32_t> routing(int64_t T, std::mt19937& rng) {
    std::vector<int32_t> ids((size_t) (T * kK));
    std::geometric_distribution<int> skew(0.08);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t j = 0; j < kK; ++j) {
            int32_t e;
            do e = skew(rng) % kExperts; while (std::find(ids.begin() + t * kK, ids.begin() + t * kK + j, e) != ids.begin() + t * kK + j);
            ids[(size_t) (t * kK + j)] = e;
        }
    return ids;
}

std::vector<uint8_t> takes(std::mt19937& rng) {
    std::vector<uint8_t> t((size_t) kExperts);
    for (uint8_t& x : t) x = (uint8_t) (rng() % 3);   // the stage, the tier from its VRAM, the tier streaming it
    return t;
}

// A half's call: every pair the tier takes is one row, each tier expert's rows together in pair order, the held experts before the streamed
void one_half(const std::vector<int32_t>& ids, int64_t T, const std::vector<uint8_t>& take, const char* what) {
    halves::TierRows h;
    check(halves::tier_rows(ids.data(), T, kK, take.data(), kExperts, h), what);
    std::vector<int32_t> cnt((size_t) kExperts, 0);
    int64_t tier = 0;
    for (const int32_t e : ids) {
        ++cnt[(size_t) e];
        tier += take[(size_t) e] != halves::kStage;
    }
    check(h.cnt == cnt, "tier rows: each expert's rows of the half");
    check(h.rows == tier && (int64_t) h.pair.size() == tier && (int64_t) h.src.size() == tier, "tier rows: one row per pair the tier takes");
    int32_t r = 0;
    bool streamed = false, order = true, groups = true;
    for (const int32_t e : h.order) {
        const bool s = take[(size_t) e] == halves::kStream;
        order = order && take[(size_t) e] != halves::kStage && cnt[(size_t) e] > 0 && (!streamed || s);
        streamed = streamed || s;
        groups = groups && h.off[(size_t) e] == r;
        for (int32_t q = 0; q < cnt[(size_t) e]; ++q, ++r) {
            const int32_t p = h.pair[(size_t) r];
            groups = groups && ids[(size_t) p] == e && h.src[(size_t) r] == p / kK && (q == 0 || h.pair[(size_t) r - 1] < p);
        }
    }
    check(order, "tier rows: the held experts, then the streamed ones, each routed and taken by the tier");
    check(groups && r == tier, "tier rows: an expert's rows together in pair order, each with its pair's token");
    for (size_t i = 1; i < h.order.size(); ++i)
        check(take[(size_t) h.order[i - 1]] != take[(size_t) h.order[i]] || h.order[i - 1] < h.order[i], "tier rows: in id order within a kind");
}

// The stage's rows of a chunk: its pairs first, by expert and in pair order, then the tier's, each pair a row of its own
void the_stage(const std::vector<int32_t>& ids, int64_t T, int64_t ta, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    std::vector<int32_t> cnt, off, order, slot((size_t) (T * kK), -1), src((size_t) (T * kK), -1);
    const int64_t local = halves::stage_rows(ids.data(), T, ta, kK, a.data(), b.data(), kExperts, cnt, off, order, slot.data(), src.data());
    std::vector<int32_t> want((size_t) kExperts, 0);
    int64_t stage = 0;
    for (int64_t i = 0; i < T * kK; ++i)
        if ((i < ta * kK ? a : b)[(size_t) ids[(size_t) i]] == halves::kStage) {
            ++want[(size_t) ids[(size_t) i]];
            ++stage;
        }
    check(cnt == want && local == stage, "stage rows: each expert's pairs the stage takes in either half");
    check((int64_t) off.size() == kExperts + 1 && off[(size_t) kExperts] == local, "stage rows: the starts end at the stage's rows");
    std::vector<char> seen((size_t) (T * kK), 0);
    bool rows = true;
    for (int64_t i = 0; i < T * kK; ++i) {
        const int32_t e = ids[(size_t) i], r = slot[(size_t) i];
        const bool mine = (i < ta * kK ? a : b)[(size_t) e] == halves::kStage;
        rows = rows && r >= 0 && r < T * kK && !seen[(size_t) r] && src[(size_t) r] == i / kK;
        rows = rows && (mine ? r >= off[(size_t) e] && r < off[(size_t) e] + cnt[(size_t) e] : r >= local);
        if (r >= 0 && r < T * kK) seen[(size_t) r] = 1;
    }
    check(rows, "stage rows: every pair a row of its own with its token, the stage's in their expert's range, the tier's after them");
    std::vector<int32_t> last((size_t) kExperts, -1);
    bool in_order = true;
    for (int64_t i = 0; i < T * kK; ++i) {
        const int32_t e = ids[(size_t) i], r = slot[(size_t) i];
        if (r >= local) continue;
        in_order = in_order && r > last[(size_t) e];
        last[(size_t) e] = r;
    }
    check(in_order, "stage rows: an expert's rows in pair order, the first half's before the second's");
    std::vector<int32_t> routed;
    for (int64_t e = 0; e < kExperts; ++e)
        if (cnt[(size_t) e] > 0) routed.push_back((int32_t) e);
    check(order == routed, "stage rows: the experts with stage rows, in id order");
}

void halves_as_a_chunk() {
    // both halves give the stage everything: the rows a whole chunk's grouping gives, pair for pair
    std::mt19937 rng(5);
    const int64_t T = 300;
    const std::vector<int32_t> ids = routing(T, rng);
    const std::vector<uint8_t> all((size_t) kExperts, halves::kStage);
    std::vector<int32_t> cnt, off, order, slot((size_t) (T * kK)), src((size_t) (T * kK));
    check(halves::stage_rows(ids.data(), T, 150, kK, all.data(), all.data(), kExperts, cnt, off, order, slot.data(), src.data()) == T * kK,
          "as a chunk: every pair is the stage's");
    std::vector<int32_t> fill((size_t) kExperts, 0), start((size_t) kExperts, 0);
    for (const int32_t e : ids) ++fill[(size_t) e];
    for (int64_t e = 1; e < kExperts; ++e) start[(size_t) e] = start[(size_t) e - 1] + fill[(size_t) e - 1];
    bool same = true;
    for (int64_t i = 0; i < T * kK; ++i) same = same && slot[(size_t) i] == start[(size_t) ids[(size_t) i]]++;
    check(same, "as a chunk: the whole chunk's slots");
    halves::TierRows h;
    check(halves::tier_rows(ids.data(), 150, kK, all.data(), kExperts, h) && h.rows == 0 && h.order.empty(), "as a chunk: no call to the tier");
}

void refuses_bad_ids() {
    std::vector<int32_t> ids((size_t) (4 * kK), 3);
    ids[17] = (int32_t) kExperts;
    const std::vector<uint8_t> take((size_t) kExperts, halves::kTier);
    halves::TierRows h;
    check(!halves::tier_rows(ids.data(), 4, kK, take.data(), kExperts, h), "bad ids: an expert past the last is refused");
    ids[17] = -1;
    check(!halves::tier_rows(ids.data(), 4, kK, take.data(), kExperts, h), "bad ids: a negative one too");
}

void whole_until_planned() {
    // a chunk its stage has not planned runs whole, as before halves, whatever the room
    check(!halves::run_halved(true, true, false), "when: an unplanned chunk runs whole");
    check(halves::run_halved(true, true, true), "when: a planned chunk runs in halves");
    // a chunk that may not run in halves or has no room for them stays whole, planned or not
    for (const bool plan : {false, true}) {
        check(!halves::run_halved(false, true, plan), "when: a chunk that may not run in halves stays whole");
        check(!halves::run_halved(true, false, plan), "when: a chunk without room for halves stays whole");
    }
}

}  // namespace

int main() {
    std::mt19937 rng(11);
    for (const int64_t T : {2, 9, 64, 515, 2048}) {
        for (int round = 0; round < 4; ++round) {
            const std::vector<int32_t> ids = routing(T, rng);
            const std::vector<uint8_t> a = takes(rng), b = round == 0 ? a : takes(rng);
            const int64_t ta = T / 2;
            one_half(std::vector<int32_t>(ids.begin(), ids.begin() + ta * kK), ta, a, "tier rows: the first half's");
            one_half(std::vector<int32_t>(ids.begin() + ta * kK, ids.end()), T - ta, b, "tier rows: the second half's");
            the_stage(ids, T, ta, a, b);
        }
    }
    halves_as_a_chunk();
    refuses_bad_ids();
    whole_until_planned();
    std::printf("half_rows_test: %d checks passed\n", checks);
    return 0;
}
