// xtier_records: the AMD tier library's ring of finished prompt layers, which drops the oldest when full. CPU only.
#include "strata/core/xtier_records.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

using strata::core::xtier::PromptRecords;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

strata_xtier_prefill_record rec(uint32_t seq) {
    strata_xtier_prefill_record r{};
    r.seq = seq;
    r.rows = 10 * (int64_t) seq;
    return r;
}

// the records' seqs are first, first + 1, ... in order
bool in_order(const strata_xtier_prefill_record* r, int64_t n, uint32_t first) {
    for (int64_t i = 0; i < n; ++i)
        if (r[i].seq != first + (uint32_t) i || r[i].rows != 10 * (int64_t) (first + (uint32_t) i)) return false;
    return true;
}

void partial_takes() {
    auto ring = std::make_unique<PromptRecords>();
    strata_xtier_prefill_record out[4];
    check(ring->take(out, 4) == 0, "partial: an empty ring gives nothing");
    for (uint32_t s = 1; s <= 5; ++s) ring->push(rec(s));
    check(ring->take(out, 2) == 2 && in_order(out, 2, 1), "partial: cap 2 takes the two oldest");
    check(ring->take(out, 2) == 2 && in_order(out, 2, 3), "partial: the next call takes the next two");
    check(ring->take(out, 2) == 1 && in_order(out, 1, 5), "partial: and the last one");
    check(ring->take(out, 2) == 0, "partial: then nothing");
    ring->push(rec(6));
    check(ring->take(out, 4) == 1 && in_order(out, 1, 6), "partial: a record pushed after the ring emptied comes next");
}

void overflow() {
    constexpr int64_t kCap = STRATA_XTIER_PF_RECORDS, kPushed = kCap + 6;
    auto ring = std::make_unique<PromptRecords>();
    for (uint32_t s = 0; s < (uint32_t) kPushed; ++s) ring->push(rec(s));
    check(ring->n == kCap, "overflow: the ring holds its capacity");
    auto out = std::make_unique<strata_xtier_prefill_record[]>((size_t) kPushed);
    check(ring->take(out.get(), 1000) == 1000 && in_order(out.get(), 1000, 6), "overflow: the six oldest were dropped, the rest come in order");
    check(ring->take(out.get(), kPushed) == kCap - 1000 && in_order(out.get(), kCap - 1000, 1006), "overflow: then the rest");
    check(ring->take(out.get(), kPushed) == 0, "overflow: and nothing after");
    for (uint32_t s = 0; s < (uint32_t) (2 * kCap + 3); ++s) ring->push(rec(s));
    check(ring->take(out.get(), kPushed) == kCap && in_order(out.get(), kCap, (uint32_t) kCap + 3), "overflow: twice around, the newest kept");
}

}  // namespace

int main() {
    partial_takes();
    overflow();
    std::printf("xtier_records_test: %d checks passed\n", checks);
    return 0;
}
