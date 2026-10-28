// A stage's planner for the AMD tier on its GPU: the rates it measures as prompts run, its chunk times, and each chunk's take (see tier_planner.hpp for the pure part).
#pragma once

#include "strata/core/xtier_abi.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/prefill/tier_planner.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace strata::prefill {

// Who holds expert e of layer l for a stage whose cache slots are host_res (or none): fill_tier_take and TierPlanner ask only this
planner::Holder holder(const core::PeerExperts* peer, const int32_t* host_res, int64_t l, int64_t e, int64_t n_expert);

// --peer-device hip:N: which of the AMD tier's experts this stage computes in each chunk, from rates measured as prompts run
class TierPlanner {
public:
    using Clock = std::chrono::steady_clock;

    TierPlanner(std::shared_ptr<TierShare> tier, const planner::Rates& priors, int device, int64_t n_layers, int64_t n_expert);
    ~TierPlanner();
    TierPlanner(const TierPlanner&) = delete;
    TierPlanner& operator=(const TierPlanner&) = delete;

    // The stage's stream to the tier as a plan sees it: the RAM experts a layer may stream now and once the rings open, and whether a plan with the latter would stream some
    struct Slots { int now = 0, open = 0; bool wants = false; };

    // Reads the tier's records once a chunk and says whether its plan acts, which plan is then given
    bool acts(core::PeerExperts& peer, bool stream_all);

    // Fills take (with ta > 0 the first half's, take_b the second's) for T tokens over layers [lb, le) and sets slots.wants, or keeps today's split unless on
    bool plan(bool on, std::vector<uint8_t>& take, std::vector<uint8_t>& take_b, int64_t ta, core::PeerExperts& peer, const int32_t* host_res,
              core::ExpertSource& src, int64_t lb, int64_t le, int64_t T, int64_t chunks_before, int64_t chunks_after, bool stream_all,
              uint64_t x_bytes, Slots& slots);

    // A layer's routing for the history and the request's tier-held rows, cnt and take the first half's and cnt_b and take_b the second's in halves
    void routed(int64_t l, const std::vector<int32_t>& cnt, int64_t T, const uint8_t* take, const core::PeerExperts& peer,
                const int32_t* host_res, const std::vector<int32_t>* cnt_b = nullptr, const uint8_t* take_b = nullptr);

    // The caller a tier call of this stage names, a second half's apart, so its record comes back to it
    int32_t caller(bool half_b = false) const { return (int32_t) (2 * slot_ + half_b); }

    // The stage queued a layer on the tier: the other stages' calls still ahead of it, and this one in its place
    void queued(const volatile uint32_t* done, uint32_t seq, int64_t rows, int64_t experts, uint64_t x_bytes, uint64_t streamed, bool stream_all);

    // The stage's chunk on its compute stream from its rows landed to its end, its time read once the end ran (no wait for it)
    void chunk_start(cudaStream_t s);
    void chunk_end(cudaStream_t s, int64_t lb, int64_t T, int64_t C);
    // This stage's slack in the chunk it last planned, ms a layer
    double slack() const { return slack_; }

    // A chunk in halves: the first half's routing of layer l is done here on the compute stream, so its call reaches the tier from here on
    void lead_mark(int64_t l, cudaStream_t s);

    // STRATA_PREFILL_TIMING, a chunk in halves: mark k of layer l on the stage's stream (0 the sync after B's router, then before and after each done wait)
    void trace_mark(int64_t l, int k, cudaStream_t s);
    // ... the host is back from the sync after mark 0, and half i's call to the tier with flag value seq is made now
    void trace_synced(int64_t l);
    void trace_call(int64_t l, int i, uint32_t seq, Clock::time_point at);

    // Around the stage's own expert work of layer l in the walk the chunk takes, on its compute stream (a failed event skips the layer)
    void start_layer(int64_t l, cudaStream_t s);
    void end_layer(int64_t l, cudaStream_t s, bool stream_all, int64_t rows, uint64_t paged, uint64_t pinned);

    // Right after a sync of the compute stream, so every recorded pair has run: each one becomes an observation
    void harvest();

    // The run is over or failed: this stage has no call the others should wait for
    void idle();

    // What one request did, and the rates it ended with per walk (staged, streamed), for its log line
    struct Request {
        int64_t chunks = 0, planned = 0, held_rows = 0, moved_rows = 0, streamed_rows = 0, stage_n = 0, tier_layers = 0, queue_n = 0;
        int64_t tokens = 0, first_planned = -1;   // the stage's tokens so far, and those before its first planned chunk
        int64_t walk_n[2] = {};                   // the stage's observations per walk
        double stage_ms = 0, stage_fit = 0, tier_ms = 0, tier_fit = 0, queue_sum = 0, start_ms = 0, start_b_ms = 0;
        planner::Rates rates[2] = {};
        bool stage_measured[2] = {}, tier_measured = false;
    };
    Request take_request();

private:
    static double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }

    // The last chunk's time once its end event ran, for every stage's slack
    void chunk_read();

    // Who holds each expert of layer l for this stage, into held_ (without src no expert counts as pinned)
    void layer_held(const core::PeerExperts& peer, const int32_t* host_res, const core::ExpertSource* src, int64_t l);

    struct Obs {
        cudaEvent_t a = nullptr, b = nullptr, c = nullptr;   // the expert phase's start and end, and a chunk in halves' first call
        cudaEvent_t t[5] = {};                               // STRATA_PREFILL_TIMING's marks (trace_mark)
        bool started = false, live = false, streamed = false, lead = false, traced = false;
        int64_t rows = 0;
        uint64_t paged = 0, pinned = 0;
    };

    // The tier's layers finished since any stage asked: one observation each, counted for the stage that queued it, then the rates
    void tier_stats(core::PeerExperts& peer);

    // A plan acts once this GPU and the tier both measured the walk
    bool acting(bool stream_all) const { return measured_[stream_all]; }

    // This GPU's rates per walk from its own books and the tier's from the shared one, with the tier's mutex held
    void refresh();

    // The events of a layer, all or none: a set made in part is not kept
    static bool make_events(Obs& o);

    // STRATA_PREFILL_TIMING: a chunk in halves' layer as planned and as measured, in ms from the stage's own experts' start
    struct Trace {
        planner::HalfTimes plan, got;
        double plan_lead = 0, got_lead = 0, wait[2] = {};
        double sync_at = 0;                        // the stage's sync after B's router, from its experts' start
        Clock::time_point synced{}, called[2]{};   // the host back from that sync, and each half's call made
        uint32_t seq[2] = {};
        bool planned = false, timed = false;
        int recs = 0;
    };
    static bool tracing();
    static bool trace_events(Obs& o);
    // The stage's side of a traced layer once its events ran: its end, its lead, its sync and its waits for each half's done
    void trace_stage(Obs& o, Trace& tr, double lead_ms);
    // A record of one of this stage's calls: its half's start on the tier from the host's clock, anchored at the sync, and its end
    void trace_record(const TierShare::Seen& s);
    // One line for the layers of the last chunk in halves that have every reading, planned against measured, then they are done
    void trace_report();

    std::shared_ptr<TierShare> tier_;
    size_t slot_ = 0;
    planner::StageBooks books_;         // this GPU's rates, fitted per walk
    planner::Rates rates_[2] = {};      // per walk (staged, streamed): both devices' rates as the last chunk read them
    bool measured_[2] = {};             // per walk: this GPU and the tier both measured
    int dev_ = -1;
    int64_t nl_ = 0, ne_ = 0;
    planner::RouteHistory hist_;        // the routing of the stage's recent chunks
    std::vector<planner::Held> held_;   // one layer's experts, filled where it is needed
    std::vector<uint8_t> probe_;        // one layer's take in a plan that only asks whether the tier would stream
    std::vector<Obs> obs_;              // per layer
    std::vector<double> lead_;          // per layer, a chunk in halves' first call to the stage's experts on its timeline (ms, a decayed mean)
    std::vector<Trace> trace_;          // per layer, STRATA_PREFILL_TIMING's reading of a chunk in halves
    std::vector<int32_t> sum_;          // a layer's routed rows of both halves
    cudaEvent_t chunk_ev_[2] = {};      // the chunk's start and end on the compute stream (chunk_start, chunk_end)
    bool chunk_started_ = false, chunk_live_ = false;
    int64_t chunk_lb_ = 0, chunk_T_ = 0, chunk_C_ = 0;   // the chunk's first layer and tokens, and its run's chunk size
    double slack_ = 0;                  // the ms a layer this stage's last plan could add to its own experts
    std::vector<strata_xtier_prefill_record> recs_;   // the tier's records of one harvest
    double queue_sum_ = 0, queue_ms_ = 0;   // the tier's queue ahead of this stage's calls: this chunk's sum, the last chunk's mean
    planner::TierStart start_;              // this stage's calls queued to the tier's x seen, as tier_stats last read it
    planner::TierStart start_b_;            // ... of its second halves' calls alone
    int64_t queue_n_ = 0;
    std::mutex req_mu_;                 // guards req_, which the engine takes from another thread once a request
    Request req_;
};

}  // namespace strata::prefill
