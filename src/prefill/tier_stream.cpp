// A stage's streaming of RAM experts to the AMD tier in prompts: its rings, and each chunk's copies (see tier_stream.hpp).
#include "strata/prefill/tier_stream.hpp"

#include "strata/core/conversation_memory.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/prefill/tier_planner.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <numeric>
#include <optional>
#include <string>

namespace strata::prefill {

TierStream::TierStream(core::xtier::Device& d, const Setup& s) : d_(d), s_(s), place_(d.stream_join(s.weight)) {
    static std::atomic<uint64_t> streams{0};
    tag_base_ = tier_tag_base(streams.fetch_add(1));
    land_.left = std::make_unique<std::atomic<int32_t>[]>((size_t) s_.n_layers);
    land_.words = s_.words;
    staged_.assign((size_t) s_.n_layers, 0);
}

TierStream::~TierStream() {
    if (opener_.joinable()) opener_.join();
}

void TierStream::open() {
    int closed = kClosed;
    if (!state_.compare_exchange_strong(closed, kOpening, std::memory_order_acq_rel)) return;
    opener_ = std::thread([this] {
        (void) cudaSetDevice(s_.device);
        state_.store(open_rings() ? kOpen : kFailed, std::memory_order_release);
    });
}

bool TierStream::open_rings() {
    static std::mutex card;   // the stages' openers: the first opens the card's ring, the others find it
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t blob = s_.blob, slot = tier_slot_bytes(blob);
    const std::optional<uint64_t> avail = core::available_host_bytes();
    const int64_t want = tier_ring_slots(s_.widest, s_.n_expert, avail ? *avail : 0, slot);
    const char* fail = std::getenv("STRATA_TEST_XTIER_RING_FAIL");
    const std::vector<int64_t> weights = d_.stream_weights();   // every stage joined at start, so every opener reads the same
    const int stages = (int) weights.size();
    const int64_t layer = tier_card_layer(s_.n_expert);   // every stage's parts of the ring from the same layer, whoever opens first
    std::string e;
    int64_t cslots = 0;   // the card's ring this opener would open, when no stage opened one yet
    TierCardShare share;
    // this stage's slots of the card's ring, by its place, from the ring or from the room for one
    const auto share_of = [&](int64_t ring) {
        share = tier_card_share(ring, weights, place_, layer);
        if (e.empty() && share.slots < 1)
            e = "the card's ring of " + std::to_string(ring) + " copies has no slots for it beside the other " + std::to_string(stages - 1) + " stages";
    };
    if (fail != nullptr && std::atoi(fail) != 0) e = "its staging ring could not be had (STRATA_TEST_XTIER_RING_FAIL)";
    else if (want < 2) e = s_.widest > 0 ? "too little RAM is available for its staging ring" : "it holds no RAM experts";
    if (e.empty()) {   // the card's room and this stage's share before any pinning, so a stage that gets no slots pins nothing
        const std::lock_guard<std::mutex> lk(card);
        if (d_.stream_slots() == 0) {
            uint64_t vfree = 0, vtotal = 0;
            // room for every stage to keep a layer's copies there, as far as the card's free VRAM goes
            cslots = d_.mem_info(vfree, vtotal, e) ? tier_card_ring((int64_t) ((vfree > kTierCardSpare ? vfree - kTierCardSpare : 0) / slot), weights, layer) : 0;
            if (e.empty() && cslots == 0) e = "the AMD card has " + std::to_string(vfree >> 20) + " MiB of VRAM free, too little for a ring of its copies";
            share_of(cslots);
        } else {
            share_of(d_.stream_slots());
        }
    }
    constexpr uint64_t kPiece = 32ull << 20;   // on the test box a copy beside it waited at most 3 ms a piece, and up to 1.6 s for a ring pinned whole
    // the stager's threads write it and the AMD card reads it, so CUDA never maps it
    if (e.empty() && !mem_.alloc_quiet(d_, (size_t) (want * slot), (size_t) tier_piece_bytes(slot, kPiece), false, e))
        e = "its staging ring of " + std::to_string((want * slot) >> 20) + " MiB: " + e;
    if (e.empty()) {   // the card's ring after the staging ring, so the card keeps none for a stage that cannot stream
        const std::lock_guard<std::mutex> lk(card);
        if (d_.stream_slots() == 0 && cslots >= kTierCardMin && !d_.prefill_open(s_.T_max, s_.n_embd, s_.gu_max, s_.d_max, cslots, blob, e))
            e = "a ring of its copies on the card: " + e;
        if (e.empty() && (d_.stream_slots() < 1 || d_.stream_blob() < blob)) e = "the card's ring of copies is too small";
        if (e.empty()) share_of(d_.stream_slots());   // the ring as opened, which every stage's share is of
        cpart_ = e.empty() ? share : TierCardShare{};
    }
    if (!e.empty()) {
        mem_.release();
        (void) cudaGetLastError();
        std::fprintf(stderr, "strata prefill: CUDA device %d streams no experts from RAM to the AMD tier: %s\n", s_.device, e.c_str());
        return false;
    }
    ring_ = TierRing(want);
    slot_bytes_ = slot;
    stager_ = std::make_unique<TierStager>(land_, s_.threads, s_.batch, s_.place);
    std::fprintf(stderr, "strata prefill: CUDA device %d may stream experts from RAM to the AMD tier: %lld of them a layer through a staging "
                         "ring of %lld MiB, %lld at a time on the card, its slots %lld to %lld of the %lld its ring holds for %d stages, by its "
                         "%lld of their %lld RAM experts (opened in %.0f ms)\n", s_.device, (long long) ring_.cap(), (long long) ((want * slot) >> 20),
                 (long long) cpart_.slots, (long long) cpart_.base, (long long) (cpart_.base + cpart_.slots - 1), (long long) d_.stream_slots(), stages,
                 (long long) s_.weight, (long long) std::accumulate(weights.begin(), weights.end(), int64_t{0}),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

void TierStream::stage(core::ExpertSource& src, std::vector<uint8_t>& take, int64_t lb, int64_t le, bool pool_runs) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint64_t hip = (uint64_t) (uintptr_t) mem_.hip_dev();
    uint8_t* const host = (uint8_t*) mem_.host();
    std::vector<TierCopy> copies;
    first_.assign((size_t) s_.n_layers + 1, ring_.entries());
    std::fill(staged_.begin(), staged_.end(), 0);
    card_.assign(take.size(), 0);
    cslot_.assign(take.size(), 0);
    ctag_.assign(take.size(), 0);
    land_.gen += 1;
    for (int64_t l = lb; l < le; ++l) {
        first_[(size_t) l] = ring_.entries();
        ring_.start_layer();
        const size_t bytes = (size_t) lay.blob_bytes(l);
        int32_t staged = 0;
        for (int64_t e = 0; e < s_.n_expert; ++e) {
            const size_t ie = (size_t) (l * s_.n_expert + e);
            if (take[ie] != planner::kTakeStream) continue;
            const bool whole = !src.transient(l, e);   // a pointer to it lives as long as the chunk
            const uint8_t* b = whole ? src.blob_stable(l, e) : nullptr;
            TierRing::Entry en{};
            const int64_t k = ring_.entries();
            if ((whole && b == nullptr) || !ring_.take(en)) {   // past the layer's cap the stage streams it
                take[ie] = planner::kTakeStage;
                continue;
            }
            const uint64_t at = (uint64_t) en.slot * slot_bytes_;
            copies.push_back(TierCopy{b, bytes, whole ? nullptr : &src, (int32_t) l, (int32_t) e, host + at, en.free_at});
            const TierCardCopy c = tier_card_copy(cpart_, tag_base_, k);
            card_[ie] = hip + at;
            cslot_[ie] = c.slot;
            ctag_[ie] = c.tag;
            ++staged;
        }
        land_.left[(size_t) l].store(staged, std::memory_order_relaxed);
        staged_[(size_t) l] = staged > 0;
    }
    stager_->start(std::move(copies), pool_runs);
}

void TierStream::drain() { stager_->drain(); }

}  // namespace strata::prefill
