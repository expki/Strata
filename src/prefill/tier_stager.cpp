// The threads that copy a stage's streamed experts into the AMD tier's staging ring (see tier_stream.hpp). No GPU needed.
#include "strata/prefill/tier_stream.hpp"

#include "strata/core/expert_source.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/platform/thread_name.hpp"

#include <algorithm>
#include <cstring>

#if defined(__linux__)
#include <cerrno>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::prefill {

namespace {

// Linux keeps a nice value per thread, so this lowers the calling thread only. Elsewhere it changes nothing.
void lower_this_thread(int steps) {
#if defined(__linux__)
    const id_t tid = (id_t) syscall(SYS_gettid);
    errno = 0;
    const int now = getpriority(PRIO_PROCESS, tid);
    if (errno == 0) (void) setpriority(PRIO_PROCESS, tid, std::min(19, now + steps));
#else
    (void) steps;
#endif
}

}  // namespace

TierStager::TierStager(TierLanding& land, int threads, int batch, void (*place)(bool pool_runs))
    : land_(land), batch_(std::clamp(batch, 1, kMaxBatch)), place_(place) {
    for (int t = 0; t < threads; ++t)
        threads_.emplace_back([this] {
            // prompt threads: work() moves them off the workers' cores while the pool can run
            strata::platform::name_this_thread("strata-stage");
            strata::aux_cpus::place_prompt_thread();
            lower_this_thread(kTierCopyNice);
            work();
        });
}

TierStager::~TierStager() {
    land_.set_stop(true);
    finish();
    {
        const std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    for (std::thread& t : threads_) t.join();
}

void TierStager::work() {
    uint32_t seen = 0;
    bool placed_off = false;
    for (;;) {
        bool off;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return quit_ || gen_ != seen; });
            if (quit_) return;
            seen = gen_;
            off = pool_runs_;
        }
        if (off != placed_off && place_ != nullptr) {
            place_(off);
            placed_off = off;
        }
        for (;;) {
            active_.fetch_add(1, std::memory_order_acq_rel);
            int k = 1;
            const int j = claim(seen, k);
            if (j < 0) {
                active_.fetch_sub(1, std::memory_order_acq_rel);
                break;
            }
            for (int i = j; i < j + k; ++i) land_.wait_free(copies_[(size_t) i].free_at);   // its slot, once the tier read what it held
            int32_t rl[kMaxBatch], re[kMaxBatch];
            uint8_t* rd[kMaxBatch];
            size_t nr = 0;
            core::ExpertSource* from = nullptr;
            for (int i = j; i < j + k; ++i) {
                const TierCopy& c = copies_[(size_t) i];
                if (c.from == nullptr) {
                    std::memcpy(c.dst, c.src, c.bytes);
                    continue;
                }
                from = c.from;
                rl[nr] = c.l;
                re[nr] = c.e;
                rd[nr] = c.dst;
                ++nr;
            }
            if (nr > 0 && !from->copy_blobs(rl, re, rd, nr)) {
                std::fprintf(stderr, "prefill: the expert source could not copy experts %d.. of layer %d\n", re[0], rl[0]);
                std::abort();
            }
            for (int i = j; i < j + k; ++i) land_.landed(copies_[(size_t) i].l);
            done_.fetch_add((size_t) k, std::memory_order_acq_rel);
            active_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
}

int TierStager::claim(uint32_t g, int& k) {
    uint64_t cur = head_.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (cur >> 32) != g) return -1;
        const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
        if (j >= n) return -1;
        k = std::min(batch_, n - j);
        while (k > 1 && copies_[(size_t) (j + k - 1)].l != copies_[(size_t) j].l) --k;   // a layer's copies apart from the next's
        if (head_.compare_exchange_weak(cur, cur + (uint64_t) k, std::memory_order_acq_rel, std::memory_order_acquire)) return j;
    }
}

void TierStager::start(std::vector<TierCopy>&& copies, bool pool_runs) {
    if (copies.empty()) return;
    land_.set_stop(false);
    const std::lock_guard<std::mutex> lk(mu_);
    copies_ = std::move(copies);
    pool_runs_ = pool_runs;
    done_.store(0, std::memory_order_relaxed);
    ++gen_;
    head_.store((uint64_t) gen_ << 32 | (uint64_t) copies_.size() << 16, std::memory_order_release);
    cv_.notify_all();
}

void TierStager::drain() {
    land_.set_stop(true);
    while (done_.load(std::memory_order_acquire) < copies_.size()) std::this_thread::yield();
    finish();
}

void TierStager::finish() {
    head_.store((uint64_t) gen_ << 32, std::memory_order_release);   // n = 0: nothing more to claim
    while (active_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
}

}  // namespace strata::prefill
