// The AMD tier's RAM experts a stage streams to it in prompts: the staging ring, the threads that copy into it, and the word each layer raises once its copies landed.
#pragma once

#include "strata/core/xtier.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::core { class ExpertSource; }

namespace strata::prefill {

/// Whether a layer that stages n entries of a ring of slots can land them all while the tier still reads the layer before: at most half the ring.
constexpr bool tier_layer_fits(int64_t n, int64_t slots) { return n >= 0 && 2 * n <= slots; }

/// The nice levels the copy threads run below their starter, so a prompt thread on their core gets it first: the mildest setting measured on more than one start, beside +2 and SCHED_BATCH.
constexpr int kTierCopyNice = 5;

/// A staging ring's slot for a blob, in whole pages, so pieces of whole slots pin whole pages.
constexpr uint64_t tier_slot_bytes(uint64_t blob) { return (blob + 4095) / 4096 * 4096; }

/// The pieces a staging ring is pinned in: as many whole slots as `want` bytes hold, at least one, so the card never copies a slot across two pieces.
constexpr uint64_t tier_piece_bytes(uint64_t slot, uint64_t want) { return slot * (want / slot > 0 ? want / slot : 1); }

/// A stage's staging ring in slots: two of its fullest layers' RAM experts, capped at half the experts and a 96th of free RAM, even, or 0 below 32 slots.
inline int64_t tier_ring_slots(int64_t widest, int64_t n_expert, uint64_t avail, uint64_t slot) {
    if (widest <= 0 || slot == 0) return 0;
    const int64_t cap = std::min<int64_t>(n_expert / 2, (int64_t) (avail / 96 / slot)) / 2 * 2;
    return cap < 32 ? 0 : std::min<int64_t>(cap, 2 * widest);
}

/// The most RAM experts a layer of any stage streams to the tier, the same for every stage whatever RAM it finds free.
constexpr int64_t tier_card_layer(int64_t n_expert) { return n_expert / 4; }

/// The AMD card's VRAM a ring of copies leaves free, and the smallest ring it opens.
constexpr uint64_t kTierCardSpare = 256ull << 20;
constexpr int64_t kTierCardMin = 16;

/// The card's ring of copies: a layer's for each stage with RAM experts and one for each without, as far as its room goes, or 0 under kTierCardMin.
inline int64_t tier_card_ring(int64_t room, const std::vector<int64_t>& weights, int64_t layer) {
    int64_t want = 0;
    for (const int64_t w : weights) want += w > 0 ? layer : 1;
    const int64_t n = layer > 0 ? std::min<int64_t>(want, room) : 0;
    return n >= kTierCardMin ? n : 0;
}

/// A stage's slots of the card's ring: the first and how many.
struct TierCardShare { int64_t base = 0, slots = 0; };

/// The stage at place index gets a part of the ring by its weight (its RAM experts) among the stages', at most a layer's copies and at least one.
inline TierCardShare tier_card_share(int64_t ring, const std::vector<int64_t>& weights, int index, int64_t layer) {
    const size_t n = weights.size();
    if (n < 1 || index < 0 || (size_t) index >= n || layer < 1 || ring < (int64_t) n) return {};
    std::vector<int64_t> w(weights), part(n, 0), rem(n, -1);
    std::vector<char> full(n, 0);
    int64_t left = ring, sum = 0;
    for (size_t i = 0; i < n; ++i)   // a stage with nothing to stream keeps one slot and no share
        if (w[i] <= 0) {
            full[i] = 1;
            part[i] = 1;
            --left;
        }
    for (bool again = true; again;) {   // a part its weight would take past a layer's copies stops there, and the others share the rest
        again = false;
        sum = 0;
        for (size_t i = 0; i < n; ++i) sum += full[i] ? 0 : w[i];
        for (size_t i = 0; i < n && !again; ++i)
            if (!full[i] && left * w[i] >= layer * sum) {
                full[i] = 1;
                part[i] = layer;
                left -= layer;
                again = true;
            }
    }
    int64_t given = 0;
    for (size_t i = 0; i < n; ++i)
        if (!full[i]) {
            part[i] = left * w[i] / sum;
            rem[i] = left * w[i] % sum;
            given += part[i];
        }
    for (int64_t k = sum > 0 ? left - given : 0; k > 0; --k) {   // the slots the floors left, each to the largest remainder
        const size_t best = (size_t) (std::max_element(rem.begin(), rem.end()) - rem.begin());
        if (rem[best] < 0) break;
        ++part[best];
        rem[best] = -1;
    }
    for (size_t i = 0; i < n; ++i)   // every stage at least one slot, taken from the largest part
        if (part[i] == 0) {
            --*std::max_element(part.begin(), part.end());
            part[i] = 1;
        }
    int64_t base = 0;
    for (int i = 0; i < index; ++i) base += part[(size_t) i];
    return {base, part[(size_t) index]};
}

/// A streamed copy's slot of the card's ring and its tag.
struct TierCardCopy { int64_t slot; uint64_t tag; };

/// A stream's tags, apart from every other stream's while each takes fewer than 2^40 entries.
constexpr uint64_t tier_tag_base(uint64_t stream) { return (stream + 1) << 40; }

/// Entry k of a stream with this part of the card's ring: the part's slots in turn, so a layer of no more entries keeps each apart, and k's own tag.
constexpr TierCardCopy tier_card_copy(TierCardShare part, uint64_t tag_base, int64_t k) {
    return {part.base + k % part.slots, tag_base + (uint64_t) k + 1};
}

/// A staging ring's entries in order: entry k takes slot k % slots once the tier read entry k - slots, and a layer takes at most cap() of them.
class TierRing {
public:
    TierRing() = default;
    /// A ring of slots slots, an even count of at least 2.
    explicit TierRing(int64_t slots) : slots_(slots) {
        if (slots_ < 2 || slots_ % 2 != 0 || !tier_layer_fits(cap(), slots_)) {
            std::fprintf(stderr, "prefill: a staging ring of %lld slots for the AMD tier\n", (long long) slots_);
            std::abort();
        }
    }
    int64_t slots() const { return slots_; }
    /// The most entries a layer stages: the most tier_layer_fits allows.
    int64_t cap() const { return slots_ / 2; }
    /// The entries taken so far.
    int64_t entries() const { return next_; }
    /// One entry: its slot, and how many entries the tier must have read before its copy may land there.
    struct Entry { int64_t slot, free_at; };
    /// Starts the next layer's entries.
    void start_layer() { layer_ = 0; }
    /// The layer's next entry into e, or false and nothing taken once the layer holds cap() entries.
    bool take(Entry& e) {
        if (!tier_layer_fits(layer_ + 1, slots_)) return false;
        e = Entry{next_ % slots_, next_ - slots_ + 1};
        ++next_;
        ++layer_;
        return true;
    }

private:
    int64_t slots_ = 0, next_ = 0, layer_ = 0;
};

/// A stage's staging ring as its copies fill it: a copy once its slot is free, and a layer's last copy raising its word.
struct TierLanding {
    std::mutex mu;
    std::condition_variable cv;
    int64_t freed = 0;                              ///< staged entries the tier read already (guarded by mu)
    bool stop = false;                              ///< the chunk is over: no copy waits for a slot any more (guarded by mu)
    std::unique_ptr<std::atomic<int32_t>[]> left;   ///< per layer, its copies still to land
    uint32_t* words = nullptr;                      ///< per layer, the word the tier waits for (host side of the flags page)
    uint32_t gen = 0;                               ///< the value this chunk's words reach
    void free_to(int64_t n) {
        {
            const std::lock_guard<std::mutex> lk(mu);
            freed = std::max(freed, n);
        }
        cv.notify_all();
    }
    void wait_free(int64_t need) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return freed >= need || stop; });
    }
    void set_stop(bool v) {
        {
            const std::lock_guard<std::mutex> lk(mu);
            stop = v;
        }
        cv.notify_all();
    }
    void landed(int32_t l) {
        if (left[(size_t) l].fetch_sub(1, std::memory_order_acq_rel) == 1) __atomic_store_n(&words[l], gen, __ATOMIC_RELEASE);
    }
};

/// One expert into the ring: bytes from src, or copied by the source itself (from set), to dst once free_at entries are read.
struct TierCopy {
    const uint8_t* src = nullptr;
    size_t bytes = 0;
    core::ExpertSource* from = nullptr;
    int32_t l = 0, e = 0;
    uint8_t* dst = nullptr;
    int64_t free_at = 0;
};

/// The threads that copy a chunk's experts into the ring: each takes a run of one layer's copies and lands them once their slots are free.
class TierStager {
public:
    static constexpr int kMaxBatch = 32;
    /// place: where a thread goes as the pool can run or not (null: it stays on the prompt's CPUs).
    TierStager(TierLanding& land, int threads, int batch, void (*place)(bool pool_runs));
    ~TierStager();
    TierStager(const TierStager&) = delete;
    TierStager& operator=(const TierStager&) = delete;
    /// A chunk's copies in entry order, once the last chunk's are drained. pool_runs: the pool can run meanwhile.
    void start(std::vector<TierCopy>&& copies, bool pool_runs);
    /// Every copy of the chunk landed, none waiting for a slot any more, and no thread at work.
    void drain();

private:
    void work();
    int claim(uint32_t g, int& k);
    void finish();

    TierLanding& land_;
    const int batch_;
    void (*const place_)(bool);
    std::vector<TierCopy> copies_;
    std::atomic<size_t> done_{0};
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for, so a thread late from the last chunk takes none of this one's
    std::atomic<uint64_t> head_{0};
    std::atomic<int> active_{0};
    uint32_t gen_ = 0;
    bool quit_ = false;
    bool pool_runs_ = false;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::thread> threads_;
};

/// A stage's streaming to the AMD tier: its staging ring and the card's, opened once a plan first streams, its copy threads, and each chunk's copies.
class TierStream {
public:
    /// What the rings are sized from and the copy threads run as.
    struct Setup {
        int device = 0;                   ///< the stage's CUDA device
        int64_t n_layers = 0, n_expert = 0, T_max = 0, n_embd = 0;
        int64_t weight = 1;               ///< the stage's RAM experts at load, held neither by it nor the tier: its weight for its part of the card's ring
        int64_t widest = 0;               ///< the most of them one of its layers holds, which sizes its staging ring
        uint64_t gu_max = 0, d_max = 0;   ///< the prompt kernels' largest matrices
        uint64_t blob = 0;                ///< the largest expert
        int threads = 1, batch = 1;       ///< as the stage's own stager
        void (*place)(bool pool_runs) = nullptr;
        uint32_t* words = nullptr;        ///< per layer, the word the tier waits for
    };
    TierStream(core::xtier::Device& d, const Setup& s);
    ~TierStream();
    TierStream(const TierStream&) = delete;
    TierStream& operator=(const TierStream&) = delete;
    /// Starts opening the rings on a thread of its own, once: the chunks after that may stream.
    void open();
    /// RAM experts a layer may stream to the tier: half the ring once it is open, else 0.
    int slots() const { return state_.load(std::memory_order_acquire) == kOpen ? (int) ring_.cap() : 0; }
    /// The card's ring slots this stage's copies take once open: a layer that streams no more keeps its copies there until its next call.
    int card_slots() const { return state_.load(std::memory_order_acquire) == kOpen ? (int) cpart_.slots : 0; }
    /// The most a layer could stream to the tier once open, while no open was tried, else 0.
    int could() const { return state_.load(std::memory_order_acquire) == kClosed ? (int) tier_card_layer(s_.n_expert) : 0; }
    /// A chunk's streamed experts of layers [lb, le): an entry and a copy each, the copies started, and past a layer's cap or without a blob the stage's.
    void stage(core::ExpertSource& src, std::vector<uint8_t>& take, int64_t lb, int64_t le, bool pool_runs);
    /// The card's address of expert e of layer l's copy in this chunk, or 0.
    uint64_t card_addr(int64_t l, int64_t e) const { return card_[(size_t) (l * s_.n_expert + e)]; }
    /// The slot of the card's ring that copy goes to, and the tag no other copy of this stream has.
    int64_t card_slot(int64_t l, int64_t e) const { return cslot_[(size_t) (l * s_.n_expert + e)]; }
    uint64_t card_tag(int64_t l, int64_t e) const { return ctag_[(size_t) (l * s_.n_expert + e)]; }
    /// Whether layer l staged any copies in this chunk, and the value its word then reaches.
    bool staged(int64_t l) const { return staged_[(size_t) l] != 0; }
    uint32_t gen() const { return land_.gen; }
    /// Layer l's routing: the tier read every layer before it.
    void free_before(int64_t l) { land_.free_to(first_[(size_t) l]); }
    /// Every copy of the chunk landed and none waits: the tier may be waiting for any of them.
    void drain();

private:
    enum : int { kClosed, kOpening, kOpen, kFailed };
    // The staging ring (two layers of its widest layer's RAM experts, at most a 96th of the RAM available), the card's ring if no stage opened it, and the copy threads, or one log line and none
    bool open_rings();

    core::xtier::Device& d_;
    Setup s_;
    std::atomic<int> state_{kClosed};
    std::thread opener_;
    core::xtier::SharedHost mem_;          // the staging ring the stager's threads write and the AMD card reads
    TierRing ring_;
    uint64_t slot_bytes_ = 0;
    int place_ = 0;                        // this stage's place among the stages that may stream to the card
    TierCardShare cpart_;                  // this stage's slots of the card's ring, which a copy's entry picks from
    uint64_t tag_base_ = 0;                // this stream's tags, apart from every other stream's
    TierLanding land_;
    std::unique_ptr<TierStager> stager_;   // after mem_ and land_, so it stops before they go
    std::vector<int64_t> first_;           // per layer of the chunk, its first entry
    std::vector<uint64_t> card_;           // per layer and expert of the chunk, the card's address of a streamed expert's copy
    std::vector<int64_t> cslot_;           // ... the card's ring slot it goes to
    std::vector<uint64_t> ctag_;           // ... and its tag
    std::vector<char> staged_;             // per layer of the chunk, whether it staged any
};

}  // namespace strata::prefill
