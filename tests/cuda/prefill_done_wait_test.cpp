// The AMD tier's done wait returns at once while its error word is set, and waits as before while the word is 0.
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
namespace k = strata::kernels;
using Clock = std::chrono::steady_clock;

int failures = 0, checks = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

void check(bool ok, const char* what, double ms) {
    ++checks;
    if (!ok) ++failures;
    std::printf("%s: %s (%.1f ms)\n", ok ? "ok" : "FAIL", what, ms);
}

// The flag and the error word in mapped host memory, as the engine's flags page holds them
struct Words {
    uint32_t* host = nullptr;
    uint32_t* dev = nullptr;
    Words() {
        ck(cudaHostAlloc((void**) &host, 4096, cudaHostAllocMapped), "cudaHostAlloc");
        ck(cudaHostGetDevicePointer((void**) &dev, host, 0), "cudaHostGetDevicePointer");
    }
    ~Words() { cudaFreeHost(host); }
    Words(const Words&) = delete;
    Words& operator=(const Words&) = delete;
    void set(size_t at, uint32_t v) {
        __atomic_store_n(&host[at], v, __ATOMIC_RELEASE);
    }
    uint32_t get(size_t at) const { return __atomic_load_n(&host[at], __ATOMIC_ACQUIRE); }
};

constexpr size_t kFlag = 0, kErr = 16;
constexpr unsigned long long kLong = 10000000000ull;   // 10 s: a wait that ends sooner ended for another reason

using Wait = void (*)(const uint32_t*, uint32_t, uint32_t*, unsigned long long, void*);

// Queues n waits for value on one stream, sets word at to v after_ms on the host when at is given, and gives the ms until the stream is through
double waits(Words& w, Wait f, int n, uint32_t value, unsigned long long timeout_ns, cudaStream_t s, size_t at = SIZE_MAX, uint32_t v = 0,
             int after_ms = 0) {
    const auto t0 = Clock::now();
    for (int i = 0; i < n; ++i) f(w.dev + kFlag, value, w.dev + kErr, timeout_ns, s);
    std::thread later;
    if (at != SIZE_MAX)
        later = std::thread([&w, at, v, after_ms] {
            std::this_thread::sleep_for(std::chrono::milliseconds(after_ms));
            w.set(at, v);
        });
    ck(cudaStreamSynchronize(s), "the waits");
    if (later.joinable()) later.join();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void run(cudaStream_t s) {
    Words w;
    // the error word set and the flag short: the done wait returns at once and leaves the word set
    w.set(kFlag, 0);
    w.set(kErr, 1);
    double ms = waits(w, k::wait_handoff_or_err, 1, 5, kLong, s);
    check(ms < 1000 && w.get(kErr) == 1, "the error word set: the done wait returns at once", ms);

    // the error word 0 and the flag reached: both waits return at once and leave the word 0
    w.set(kFlag, 5);
    w.set(kErr, 0);
    ms = waits(w, k::wait_handoff_or_err, 1, 5, kLong, s);
    check(ms < 1000 && w.get(kErr) == 0, "the flag reached: the done wait returns at once", ms);
    ms = waits(w, k::wait_handoff, 1, 5, kLong, s);
    check(ms < 1000 && w.get(kErr) == 0, "the flag reached: the hand-off wait returns at once", ms);

    // the error word 0 and the flag raised later: the done wait returns after the raise, the word still 0
    w.set(kFlag, 0);
    ms = waits(w, k::wait_handoff_or_err, 1, 5, kLong, s, kFlag, 5, 200);
    check(ms >= 150 && ms < 5000 && w.get(kErr) == 0, "the flag raised later: the done wait returns after it", ms);

    // the error word 0 and the flag never raised: the done wait gives up after its timeout and sets the word, as the hand-off wait does
    w.set(kFlag, 0);
    w.set(kErr, 0);
    ms = waits(w, k::wait_handoff_or_err, 1, 5, 300000000ull, s);
    check(ms >= 250 && ms < 5000 && w.get(kErr) == 1, "the flag never raised: the done wait times out and sets the word", ms);
    w.set(kErr, 0);
    ms = waits(w, k::wait_handoff, 1, 5, 300000000ull, s);
    check(ms >= 250 && ms < 5000 && w.get(kErr) == 1, "the flag never raised: the hand-off wait times out and sets the word", ms);

    // the error word set while the done wait spins: it returns then
    w.set(kErr, 0);
    ms = waits(w, k::wait_handoff_or_err, 1, 5, kLong, s, kErr, 1, 200);
    check(ms >= 150 && ms < 5000, "the error word set while it spins: the done wait returns", ms);

    // a chunk in halves: two done waits back to back and no flag. The second ends with the first's timeout, not one of its own
    w.set(kErr, 0);
    ms = waits(w, k::wait_handoff_or_err, 2, 5, 400000000ull, s);
    check(ms >= 350 && ms < 700 && w.get(kErr) == 1, "two done waits and no flag: one timeout", ms);
    w.set(kErr, 0);
    ms = waits(w, k::wait_handoff, 2, 5, 400000000ull, s);
    check(ms >= 750 && w.get(kErr) == 1, "two hand-off waits and no flag: two timeouts, as before", ms);

    // the layer split's hand-off wait reads no error word: set, it still waits for the flag
    w.set(kFlag, 0);
    w.set(kErr, 1);
    ms = waits(w, k::wait_handoff, 1, 5, kLong, s, kFlag, 5, 200);
    check(ms >= 150 && ms < 5000 && w.get(kErr) == 1, "the hand-off wait ignores a set error word", ms);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        std::printf("prefill_done_wait_test: SKIP (no GPU)\n");
        return 77;
    }
    try {
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
        run(s);
        ck(cudaStreamDestroy(s), "cudaStreamDestroy");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill_done_wait_test: %s\n", e.what());
        return 1;
    }
    std::printf("prefill_done_wait_test: %d of %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
