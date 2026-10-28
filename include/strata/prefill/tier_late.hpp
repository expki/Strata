// What a stage says when one of its waits on the AMD tier gave up: which error word was set and, from the flags, which wait that was. No GPU needed.
#pragma once

#include <cstdint>
#include <string>

namespace strata::prefill {

/// The words a stage reads once a wait on the AMD tier gave up.
struct TierLate {
    uint32_t hip_err = 0, cuda_err = 0;   ///< the card's wait (for x or for staged experts) gave up, this GPU's wait for done gave up
    uint32_t x_ready = 0, x_seq = 0;      ///< the x word and the last value the stage raised it to
    uint32_t done = 0, called = 0;        ///< the done word and the flag value of the stage's last queued call
    int64_t staged_layer = -1;            ///< a layer of the chunk whose staged experts have not all landed, or -1
    uint32_t staged = 0, staged_gen = 0;  ///< that layer's word and the value it waits for
};

/// The waits that gave up and how far each side got, for the log line, or empty when no error word is set.
inline std::string tier_late_why(const TierLate& w) {
    std::string s;
    if (w.hip_err != 0) {
        if (w.x_ready != w.x_seq)
            s = "the AMD card's wait for x gave up (x at " + std::to_string(w.x_ready) + " of " + std::to_string(w.x_seq) + ")";
        else if (w.staged_layer >= 0)
            s = "the AMD card's wait for layer " + std::to_string(w.staged_layer) + "'s staged experts gave up (their word at " +
                std::to_string(w.staged) + " of " + std::to_string(w.staged_gen) + ")";
        else
            s = "the AMD card's wait for x or for its staged experts gave up";
    }
    if (w.cuda_err != 0)
        s += std::string(s.empty() ? "" : ", and ") + "this GPU's wait for the AMD tier's done flag gave up (its last queued call seq " +
             std::to_string(w.called) + ", the card done to " + std::to_string(w.done) + ")";
    return s;
}

/// The log line once a wait of wait_ms gave up, seen by CUDA device dev at layer l (-1 after its chunk).
inline std::string tier_late_line(const TierLate& w, uint64_t wait_ms, int dev, int64_t l) {
    return "the AMD tier did not answer within STRATA_XTIER_WAIT_MS (" + std::to_string(wait_ms) + " ms): " + tier_late_why(w) + ", seen " +
           (l >= 0 ? "at layer " + std::to_string(l) : std::string("after its chunk")) + " by CUDA device " + std::to_string(dev);
}

}  // namespace strata::prefill
