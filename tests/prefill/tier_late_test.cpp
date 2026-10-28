// tier_late: the clause a stage logs when a wait on the AMD tier gave up, naming the wait from the error words and the flags. CPU only.
#include "strata/prefill/tier_late.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace strata::prefill;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

void names_the_wait() {
    TierLate w;
    w.x_ready = w.x_seq = 40;
    w.done = 38;
    w.called = 40;
    check(tier_late_why(w).empty(), "late: no error word set, nothing to say");
    w.cuda_err = 1;
    const std::string done = tier_late_why(w);
    check(has(done, "this GPU's wait for the AMD tier's done flag") && has(done, "seq 40") && has(done, "done to 38"),
          "late: the stage's done wait, with its last call and how far the card got");
    check(!has(done, "card's wait"), "late: the card's wait is not named when its word is clear");
    w = TierLate{};
    w.hip_err = 1;
    w.x_ready = 39;
    w.x_seq = w.called = 40;
    const std::string x = tier_late_why(w);
    check(has(x, "the AMD card's wait for x") && has(x, "x at 39 of 40") && !has(x, "this GPU's"), "late: the card's wait for x when x is not all up");
    w.x_ready = 40;
    w.staged_layer = 17;
    w.staged = 3;
    w.staged_gen = 4;
    const std::string staged = tier_late_why(w);
    check(has(staged, "layer 17's staged experts") && has(staged, "at 3 of 4"), "late: the card's wait for a layer's staged experts");
    w.staged_layer = -1;
    check(has(tier_late_why(w), "for x or for its staged experts"), "late: the card's wait, unnamed when the flags say neither");
    w.cuda_err = 1;
    const std::string both = tier_late_why(w);
    check(has(both, "the AMD card's wait") && has(both, "this GPU's wait"), "late: both words set, both waits named");
}

void names_the_queued_call() {
    // in halves both of layer l's x go out before the done wait for layer l-1 gives up, so the call queued last is not the x raised last
    TierLate w;
    w.cuda_err = 1;
    w.x_ready = w.x_seq = 42;
    w.called = 40;
    w.done = 39;
    const std::string line = tier_late_line(w, 30000, 2, 17);
    check(has(line, "its last queued call seq 40") && !has(line, "seq 42"), "line: the done wait names the call queued last, not the x raised last");
    check(has(line, "seen at layer 17 by CUDA device 2"), "line: the layer is where the stage saw the error");
    check(has(line, "STRATA_XTIER_WAIT_MS (30000 ms): this GPU's wait"), "line: the wait's limit, then the wait that gave up");
    check(has(tier_late_line(w, 30000, 2, -1), "seen after its chunk by CUDA device 2"), "line: an error seen after the chunk names no layer");
}

}  // namespace

int main() {
    names_the_wait();
    names_the_queued_call();
    std::printf("tier_late_test: %d checks passed\n", checks);
    return 0;
}
