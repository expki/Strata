// A stage's planner for the AMD tier on its GPU (see tier_planner_engine.hpp).
#include "tier_planner_engine.hpp"

#include "strata/core/on_device.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/prefill/kernels.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

namespace strata::prefill {
namespace {

constexpr int64_t K = kCombineTop;   // the experts a token routes to, the prompt path's K

}  // namespace

planner::Holder holder(const core::PeerExperts* peer, const int32_t* host_res, int64_t l, int64_t e, int64_t n_expert) {
    return planner::holder_of(host_res != nullptr && host_res[(size_t) (l * n_expert + e)] >= 0, peer != nullptr && peer->has(l, e));
}

TierPlanner::TierPlanner(std::shared_ptr<TierShare> tier, const planner::Rates& priors, int device, int64_t n_layers, int64_t n_expert)
    : tier_(std::move(tier)), books_(priors), dev_(device), nl_(n_layers), ne_(n_expert), hist_(n_layers, n_expert),
      held_((size_t) n_expert), probe_((size_t) n_expert), obs_((size_t) n_layers), lead_((size_t) n_layers, 0.0), trace_((size_t) n_layers) {
    const std::lock_guard<std::mutex> lk(tier_->mu);
    slot_ = tier_->took.size();
    tier_->calls.stages(slot_ + 1);
    tier_->took.emplace_back();
    tier_->start.emplace_back();
    tier_->start_b.emplace_back();
    tier_->seen.emplace_back();
    tier_->times.stages(slot_ + 1);
    refresh();
}

TierPlanner::~TierPlanner() {
    idle();
    const core::OnDevice on(dev_);
    for (cudaEvent_t e : chunk_ev_)
        if (e) (void) cudaEventDestroy(e);
    for (Obs& o : obs_) {
        if (o.a) (void) cudaEventDestroy(o.a);
        if (o.b) (void) cudaEventDestroy(o.b);
        if (o.c) (void) cudaEventDestroy(o.c);
        for (cudaEvent_t e : o.t)
            if (e) (void) cudaEventDestroy(e);
    }
}

bool TierPlanner::acts(core::PeerExperts& peer, bool stream_all) {
    tier_stats(peer);
    if (tracing()) trace_report();
    return acting(stream_all);
}

bool TierPlanner::plan(bool on, std::vector<uint8_t>& take, std::vector<uint8_t>& take_b, int64_t ta, core::PeerExperts& peer, const int32_t* host_res,
                       core::ExpertSource& src, int64_t lb, int64_t le, int64_t T, int64_t chunks_before, int64_t chunks_after, bool stream_all,
                       uint64_t x_bytes, Slots& slots) {
    if (queue_n_ > 0) queue_ms_ = queue_sum_ / (double) queue_n_;
    const double ahead = start_.ahead(queue_ms_);   // the tier starts a layer once the calls ahead are done and its x landed
    const double ahead_b = start_b_.measured() ? start_b_.ahead(queue_ms_) : ahead;   // a second half's call, as its own calls measured it
    queue_sum_ = 0;
    queue_n_ = 0;
    {
        const std::lock_guard<std::mutex> lk(req_mu_);
        ++req_.chunks;
        req_.planned += on;
        if (on && req_.first_planned < 0) req_.first_planned = req_.tokens;
        req_.tokens += T;
    }
    slack_ = 0;
    if (!on) return false;
    planner::StageTimes::Gap gap;
    {
        const std::lock_guard<std::mutex> lk(tier_->mu);
        slack_ = tier_->times.slack(slot_, le - lb, T, chunks_before, chunks_after);
        gap = tier_->times.gap(slot_, T);
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    const planner::Rates r = planner::tier_call_rates(rates_[stream_all], ta > 0 ? x_bytes / 2 : x_bytes, start_.measured());
    take.assign((size_t) (nl_ * ne_), 0);
    if (ta > 0) take_b.assign((size_t) (nl_ * ne_), 0);
    int64_t held = 0, moved = 0, moved_b = 0, streamed = 0;
    double stage_sum = 0, tier_sum = 0, lead_sum = 0;
    for (int64_t l = lb; l < le; ++l) {
        layer_held(peer, host_res, &src, l);
        const uint64_t blob = (uint64_t) lay.blob_bytes(l);
        uint8_t* const row = take.data() + (size_t) (l * ne_);
        planner::LayerPlan p;
        Trace& tr = trace_[(size_t) l];
        tr = Trace{};
        if (ta > 0) {   // the second half's call follows the first's, which the tier starts while the stage reads that second half
            const planner::HalfStarts at{ahead, lead_[(size_t) l], ahead_b};
            const planner::HalvesPlan h = planner::plan_stage_halves(hist_, l, ta, T - ta, K, held_, blob, stream_all, r, at, slots.now, row,
                                                                     take_b.data() + (size_t) (l * ne_), slack_);
            p = planner::LayerPlan{h.b.stage_ms, h.b.tier_ms, h.a.held, h.a.moved, h.a.streamed};   // the layer's tier-held experts, A's moves
            moved_b += h.b.moved;
            lead_sum += lead_[(size_t) l];
            tr.plan = planner::planned_times(h, at);
            tr.plan_lead = lead_[(size_t) l];
            tr.planned = true;
        } else {
            p = planner::plan_stage_layer(hist_, l, T, K, held_, blob, stream_all, r, ahead, slots.now, row, slack_);
        }
        held += p.held;
        moved += p.moved;
        streamed += p.streamed;
        stage_sum += p.stage_ms;
        tier_sum += p.tier_ms;
        if (!slots.wants && slots.now == 0 && slots.open > 0)   // the whole-chunk plan for halves too, as the halves plan opened rings that slowed the slowest stage
            slots.wants = planner::plan_stage_layer(hist_, l, T, K, held_, (uint64_t) lay.blob_bytes(l), stream_all, r, ahead,
                                                    slots.open, probe_.data()).streamed > 0;
    }
    if (tracing() && le > lb) {
        char moves[64];
        std::snprintf(moves, sizeof(moves), ta > 0 ? "%lld and %lld" : "%lld", (long long) moved, (long long) moved_b);
        std::fprintf(stderr, "strata prefill planner: layers %lld..%lld, %lld tokens%s, the %s walk: %s of %lld tier-held experts to this "
                             "GPU%s, %lld RAM experts to the AMD tier; a layer ends at %.1f ms here and %.1f ms on the AMD tier (it starts "
                             "%.1f ms after the call and a second half's %.1f, %.1f predicted queued ahead, the first half's call %.1f ms "
                             "before the stage's experts); its slack %.1f ms a layer, the slowest stage's chunk %.0f ms (mean %.0f, last %.0f) against "
                             "its own %.0f (mean %.0f, last %.0f), chunk %lld of the run, %lld after\n",
                     (long long) lb, (long long) le - 1, (long long) T, ta > 0 ? " in two halves" : "", stream_all ? "streamed" : "staged",
                     moves, (long long) held, ta > 0 ? " in each half" : "", (long long) streamed, stage_sum / (double) (le - lb),
                     tier_sum / (double) (le - lb), start_.ms(), ahead_b, queue_ms_, lead_sum / (double) (le - lb), slack_,
                     std::min(gap.slow_mean, gap.slow_last), gap.slow_mean, gap.slow_last, std::max(gap.own_mean, gap.own_last), gap.own_mean,
                     gap.own_last, (long long) chunks_before + 1, (long long) chunks_after);
    }
    return true;
}

void TierPlanner::routed(int64_t l, const std::vector<int32_t>& cnt, int64_t T, const uint8_t* take, const core::PeerExperts& peer,
                         const int32_t* host_res, const std::vector<int32_t>* cnt_b, const uint8_t* take_b) {
    if (cnt_b != nullptr) {
        sum_.resize(cnt.size());
        for (size_t e = 0; e < cnt.size(); ++e) sum_[e] = cnt[e] + (*cnt_b)[e];
    }
    hist_.add(l, (cnt_b != nullptr ? sum_ : cnt).data(), T);
    layer_held(peer, host_res, nullptr, l);
    planner::Counts n = planner::routed_rows(cnt.data(), held_, take);
    if (cnt_b != nullptr) {
        const planner::Counts nb = planner::routed_rows(cnt_b->data(), held_, take_b);
        n.held += nb.held;
        n.moved += nb.moved;
        n.streamed += nb.streamed;
    }
    const std::lock_guard<std::mutex> lk(req_mu_);
    req_.held_rows += n.held;
    req_.moved_rows += n.moved;
    req_.streamed_rows += n.streamed;
}

void TierPlanner::queued(const volatile uint32_t* done, uint32_t seq, int64_t rows, int64_t experts, uint64_t x_bytes, uint64_t streamed, bool stream_all) {
    const double now = now_ms();
    const planner::Rates r = planner::tier_call_rates(rates_[stream_all], x_bytes, false);   // its ms from queued to done holds its handoff
    const double ms = r.tier_fixed_ms + (double) rows * r.tier_row_ms + (double) experts * r.tier_expert_ms +
                      (double) streamed * r.tier_byte_ms_paged;
    std::vector<planner::Queued> ahead;
    {
        const std::lock_guard<std::mutex> lk(tier_->mu);
        ahead = tier_->calls.queue(slot_, planner::StageCalls::Call{done, seq, now, ms});
    }
    const double q = planner::queued_ahead(std::move(ahead), now);
    queue_sum_ += q;
    ++queue_n_;
    const std::lock_guard<std::mutex> lk(req_mu_);
    req_.queue_sum += q;
    ++req_.queue_n;
}

void TierPlanner::chunk_start(cudaStream_t s) {
    chunk_read();
    chunk_live_ = chunk_started_ = false;
    for (cudaEvent_t& e : chunk_ev_)
        if (e == nullptr && cudaEventCreate(&e) != cudaSuccess) {
            e = nullptr;
            (void) cudaGetLastError();
            return;
        }
    chunk_started_ = cudaEventRecord(chunk_ev_[0], s) == cudaSuccess;
    if (!chunk_started_) (void) cudaGetLastError();
}

void TierPlanner::chunk_end(cudaStream_t s, int64_t lb, int64_t T, int64_t C) {
    if (!std::exchange(chunk_started_, false)) return;
    if (cudaEventRecord(chunk_ev_[1], s) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    chunk_live_ = true;
    chunk_lb_ = lb;
    chunk_T_ = T;
    chunk_C_ = C;
}

void TierPlanner::lead_mark(int64_t l, cudaStream_t s) {
    Obs& o = obs_[(size_t) l];
    o.lead = false;
    if (o.a == nullptr && !make_events(o)) return;
    o.lead = cudaEventRecord(o.c, s) == cudaSuccess;
    if (!o.lead) (void) cudaGetLastError();
}

void TierPlanner::trace_mark(int64_t l, int k, cudaStream_t s) {
    Obs& o = obs_[(size_t) l];
    if (!tracing() || !trace_[(size_t) l].planned || !trace_events(o)) return;
    if (cudaEventRecord(o.t[k], s) != cudaSuccess) {
        (void) cudaGetLastError();
        o.traced = false;
    } else if (k == 0) {
        o.traced = true;
    }
}

void TierPlanner::trace_synced(int64_t l) {
    if (tracing()) trace_[(size_t) l].synced = Clock::now();
}

void TierPlanner::trace_call(int64_t l, int i, uint32_t seq, Clock::time_point at) {
    if (!tracing()) return;
    Trace& tr = trace_[(size_t) l];
    tr.called[i] = at;
    tr.seq[i] = seq;
}

void TierPlanner::start_layer(int64_t l, cudaStream_t s) {
    Obs& o = obs_[(size_t) l];
    o.live = o.started = false;
    if (o.a == nullptr && !make_events(o)) return;
    o.started = cudaEventRecord(o.a, s) == cudaSuccess;
    if (!o.started) (void) cudaGetLastError();
}

void TierPlanner::end_layer(int64_t l, cudaStream_t s, bool stream_all, int64_t rows, uint64_t paged, uint64_t pinned) {
    Obs& o = obs_[(size_t) l];
    if (!o.started) return;
    if (cudaEventRecord(o.b, s) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    o.streamed = stream_all;
    o.rows = rows;
    o.paged = paged;
    o.pinned = pinned;
    o.live = true;
}

void TierPlanner::harvest() {
    chunk_read();
    for (size_t l = 0; l < obs_.size(); ++l) {
        Obs& o = obs_[l];
        if (!o.live) continue;
        o.live = false;
        float ms = 0;
        if (o.lead) {   // the first half's call to the stage's experts
            const bool lead_ok = cudaEventElapsedTime(&ms, o.c, o.a) == cudaSuccess && ms >= 0;
            if (lead_ok) lead_[l] = lead_[l] > 0 ? 0.75 * lead_[l] + 0.25 * ms : ms;
            else (void) cudaGetLastError();
            o.lead = false;
            if (o.traced) trace_stage(o, trace_[l], lead_ok ? ms : -1.0);
        }
        if (cudaEventElapsedTime(&ms, o.a, o.b) != cudaSuccess) {
            (void) cudaGetLastError();
            continue;
        }
        const planner::Rates& r = rates_[o.streamed];
        const double fit = r.stage_fixed_ms + (double) o.rows * r.stage_row_ms + (double) o.paged * r.stage_byte_ms_paged +
                           (double) o.pinned * r.stage_byte_ms_pinned;
        if (!books_.observe(o.streamed, o.rows, o.paged, o.pinned, ms)) continue;
        const std::lock_guard<std::mutex> lk(req_mu_);
        req_.stage_ms += ms;
        req_.stage_fit += fit;
        ++req_.stage_n;
        ++req_.walk_n[o.streamed];
    }
}

void TierPlanner::idle() {
    const std::lock_guard<std::mutex> lk(tier_->mu);
    tier_->calls.clear(slot_);
}

TierPlanner::Request TierPlanner::take_request() {
    TierShare::Took k;
    {
        const std::lock_guard<std::mutex> lk(tier_->mu);
        k = std::exchange(tier_->took[slot_], TierShare::Took{});
    }
    const std::lock_guard<std::mutex> lk(req_mu_);
    Request r = std::exchange(req_, Request{});
    r.tier_layers = k.layers;
    r.tier_ms = k.ms;
    r.tier_fit = k.fit;
    for (int w = 0; w < 2; ++w) {
        req_.rates[w] = r.rates[w];
        req_.stage_measured[w] = r.stage_measured[w];
    }
    req_.tier_measured = r.tier_measured;
    req_.start_ms = r.start_ms;
    req_.start_b_ms = r.start_b_ms;
    return r;
}

void TierPlanner::chunk_read() {
    if (!chunk_live_) return;
    const cudaError_t q = cudaEventQuery(chunk_ev_[1]);
    if (q == cudaErrorNotReady) return;
    chunk_live_ = false;
    float ms = 0;
    if (q != cudaSuccess || cudaEventElapsedTime(&ms, chunk_ev_[0], chunk_ev_[1]) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    const std::lock_guard<std::mutex> lk(tier_->mu);
    tier_->times.observe(slot_, chunk_lb_, ms, chunk_T_, chunk_C_);
}

void TierPlanner::layer_held(const core::PeerExperts& peer, const int32_t* host_res, const core::ExpertSource* src, int64_t l) {
    for (int64_t e = 0; e < ne_; ++e) {
        const planner::Holder h = holder(&peer, host_res, l, e, ne_);
        held_[(size_t) e] = {h, src != nullptr && h != planner::Holder::Stage && src->pinned(l, e)};
    }
}

void TierPlanner::tier_stats(core::PeerExperts& peer) {
    std::string e;
    if (peer.xdev() == nullptr || !peer.xdev()->prefill_stats(recs_, e)) recs_.clear();
    const std::lock_guard<std::mutex> lk(tier_->mu);
    for (const strata_xtier_prefill_record& rc : recs_) {
        const double ms = rc.kernel_ms + rc.copy_ms;
        const size_t stage = (size_t) (rc.caller >> 1);   // the caller names the stage and whether the call was a second half's
        const bool ours = rc.caller >= 0 && stage < tier_->took.size();
        const planner::Rates& t = rates_[1];   // both walks share the tier's half
        const double fit = t.tier_fixed_ms + (double) rc.rows * t.tier_row_ms + (double) rc.experts * t.tier_expert_ms +
                           (double) rc.bytes_in * t.tier_input_byte_ms + (double) rc.bytes_streamed * t.tier_byte_ms_paged;
        if (ours && tracing()) tier_->seen[stage].push_back({rc.seq, rc.start_ms, ms});
        if (!tier_->book.observe_tier(rc.rows, rc.experts, rc.bytes_in, rc.bytes_streamed, 0, 1, ms) || !ours) continue;
        tier_->start[stage].observe(rc.start_ms);   // measured each call, so a busier tier or a slower handoff shows at once
        if (rc.caller & 1) tier_->start_b[stage].observe(rc.start_ms);
        TierShare::Took& k = tier_->took[stage];
        ++k.layers;
        k.ms += ms;
        k.fit += fit;
    }
    for (const TierShare::Seen& s : tier_->seen[slot_]) trace_record(s);   // this stage's calls, whichever stage read their records
    tier_->seen[slot_].clear();
    refresh();
}

void TierPlanner::refresh() {
    start_ = tier_->start[slot_];
    start_b_ = tier_->start_b[slot_];
    const planner::Rates t = tier_->book.rates();
    const bool tier_ok = tier_->book.tier_measured();
    for (int w = 0; w < 2; ++w) {
        rates_[w] = books_.rates(w == 1);
        rates_[w].tier_row_ms = t.tier_row_ms;
        rates_[w].tier_byte_ms_paged = t.tier_byte_ms_paged;
        rates_[w].tier_byte_ms_pinned = t.tier_byte_ms_paged;   // every expert the tier streams is staged, a pinned one too
        rates_[w].tier_fixed_ms = t.tier_fixed_ms;
        rates_[w].tier_expert_ms = t.tier_expert_ms;
        rates_[w].tier_input_byte_ms = t.tier_input_byte_ms;
        measured_[w] = books_.measured(w == 1) && tier_ok;
    }
    const std::lock_guard<std::mutex> lk(req_mu_);
    for (int w = 0; w < 2; ++w) {
        req_.rates[w] = rates_[w];
        req_.stage_measured[w] = books_.measured(w == 1);
    }
    req_.tier_measured = tier_ok;
    req_.start_ms = start_.ms();
    req_.start_b_ms = start_b_.ms();
}

bool TierPlanner::make_events(Obs& o) {
    if (cudaEventCreate(&o.a) != cudaSuccess) {
        o.a = nullptr;
        (void) cudaGetLastError();
        return false;
    }
    if (cudaEventCreate(&o.b) != cudaSuccess || cudaEventCreate(&o.c) != cudaSuccess) {
        (void) cudaEventDestroy(o.a);
        if (o.b) (void) cudaEventDestroy(o.b);
        o.a = o.b = o.c = nullptr;
        (void) cudaGetLastError();
        return false;
    }
    return true;
}

bool TierPlanner::tracing() {
    static const bool on = std::getenv("STRATA_PREFILL_TIMING") != nullptr;
    return on;
}

bool TierPlanner::trace_events(Obs& o) {
    for (cudaEvent_t& e : o.t)
        if (e == nullptr && cudaEventCreate(&e) != cudaSuccess) {
            e = nullptr;
            (void) cudaGetLastError();
            return false;
        }
    return true;
}

void TierPlanner::trace_stage(Obs& o, Trace& tr, double lead_ms) {
    o.traced = false;
    float end = 0, sync = 0, wa = 0, wb = 0;
    const bool ok = lead_ms >= 0 && cudaEventElapsedTime(&end, o.a, o.b) == cudaSuccess && cudaEventElapsedTime(&sync, o.t[0], o.a) == cudaSuccess &&
                    cudaEventElapsedTime(&wa, o.t[1], o.t[2]) == cudaSuccess && cudaEventElapsedTime(&wb, o.t[3], o.t[4]) == cudaSuccess;
    if (!ok) {
        (void) cudaGetLastError();
        return;
    }
    tr.got.stage_end = end;
    tr.got_lead = lead_ms;
    tr.sync_at = -sync;
    tr.wait[0] = wa;
    tr.wait[1] = wb;
    tr.timed = true;
}

void TierPlanner::trace_record(const TierShare::Seen& s) {
    for (Trace& tr : trace_)
        for (int i = 0; i < 2; ++i) {
            if (!tr.planned || tr.seq[i] != s.seq || s.start_ms < 0 || tr.synced == Clock::time_point{}) continue;
            const double start = tr.sync_at + std::chrono::duration<double, std::milli>(tr.called[i] - tr.synced).count() + s.start_ms;
            (i == 0 ? tr.got.a_start : tr.got.b_start) = start;
            (i == 0 ? tr.got.a_end : tr.got.b_end) = start + s.ms;
            tr.seq[i] = 0;
            ++tr.recs;
            return;
        }
}

void TierPlanner::trace_report() {
    planner::HalfTimes plan, got;
    double lead[2] = {}, wait[2] = {};
    int64_t n = 0, lo = -1, hi = -1;
    for (size_t l = 0; l < trace_.size(); ++l) {
        Trace& tr = trace_[l];
        if (!tr.planned || !tr.timed || tr.recs < 2) continue;
        plan += tr.plan;
        got += tr.got;
        lead[0] += tr.plan_lead;
        lead[1] += tr.got_lead;
        wait[0] += tr.wait[0];
        wait[1] += tr.wait[1];
        if (lo < 0) lo = (int64_t) l;
        hi = (int64_t) l;
        ++n;
        tr = Trace{};
    }
    if (n == 0) return;
    const double d = (double) n;
    std::fprintf(stderr, "strata prefill halves: CUDA device %d, layers %lld..%lld (%lld with every reading), ms from the stage's own experts' "
                         "start, planned against measured: A on the tier %.1f to %.1f against %.1f to %.1f, B %.1f to %.1f against %.1f to %.1f, "
                         "the stage ends %.1f against %.1f; A's routing %.1f against %.1f before; the stage waited %.1f for A and %.1f for B\n",
                 dev_, (long long) lo, (long long) hi, (long long) n, plan.a_start / d, plan.a_end / d, got.a_start / d, got.a_end / d,
                 plan.b_start / d, plan.b_end / d, got.b_start / d, got.b_end / d, plan.stage_end / d, got.stage_end / d, lead[0] / d,
                 lead[1] / d, wait[0] / d, wait[1] / d);
}

}  // namespace strata::prefill
