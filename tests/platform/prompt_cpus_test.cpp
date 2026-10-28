// prompt_cpus_test: where prompt-only threads run, and that a sweep, a restore and a pool hold keep them placed
#include "strata/platform/aux_cpus.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <thread>

using namespace strata::aux_cpus;

namespace {
int failures = 0;
#define CHECK(c)                                                                                                  \
    do {                                                                                                          \
        if (!(c)) {                                                                                               \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #c);                                   \
            ++failures;                                                                                           \
        }                                                                                                         \
    } while (0)

using V = std::vector<int>;

void test_prompt_cpus() {
    CHECK(prompt_cpus({0, 1, 2, 3}, 0) == std::vector<int>({1, 2, 3}));
    CHECK(prompt_cpus({0, 1, 2, 3}, 2) == std::vector<int>({0, 1, 3}));
    CHECK(prompt_cpus({5}, 5) == std::vector<int>({5}));
    CHECK(prompt_cpus({1, 2}, -1) == std::vector<int>({1, 2}));
    CHECK(prompt_cpus({1, 2}, 7) == (V{1, 2}));   // a host outside the allowed set takes nothing
    CHECK(prompt_cpus({}, 0).empty());             // configure never ran: nothing to place on
}

void test_off_pool_cpus() {
    // while the pool can run: the aux set when it is on, else off the host's and the workers' CPUs, else off the host's
    CHECK(off_pool_cpus({0, 1, 2, 3}, 0, {1, 2}, {}) == (V{3}));
    CHECK(off_pool_cpus({0, 1, 2, 3}, 0, {1, 2}, {5}) == (V{5}));
    CHECK(off_pool_cpus({0, 1, 2, 3}, 0, {1, 2, 3}, {}) == (V{1, 2, 3}));   // nothing spare: never the host's CPU
    CHECK(off_pool_cpus({0, 1}, -1, {1}, {}) == (V{0}));                     // the host's CPU unknown
    CHECK(off_pool_cpus({4}, 4, {}, {}) == (V{4}));                          // the host's CPU is all there is
    CHECK(off_pool_cpus({}, 0, {1}, {}).empty());
    V all, low;
    for (int c = 0; c < 16; ++c) all.push_back(c);
    for (int c = 1; c < 8; ++c) low.push_back(c);
    // the test box (16 cores, the host on 0, 15 workers) and 8 cores x 2 threads with the default pool (the siblings)
    CHECK(off_pool_cpus(all, 0, V(all.begin() + 1, all.end()), {}) == V(all.begin() + 1, all.end()));
    CHECK(off_pool_cpus(all, 0, low, {}) == V(all.begin() + 8, all.end()));
}

#if defined(__linux__)
int tid_of_self() { return (int) syscall(SYS_gettid); }

std::vector<int> mask_of(int tid) {
    cpu_set_t m;
    CPU_ZERO(&m);
    std::vector<int> out;
    if (sched_getaffinity(tid, sizeof m, &m) != 0) return out;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &m)) out.push_back(c);
    return out;
}

// A thread that runs `step` each time the test bumps `go`, until `stop`: the test reads its mask between steps.
struct Helper {
    std::atomic<int> tid{0}, go{0}, done{0};
    std::atomic<bool> stop{false};
    const std::function<void(int)> step;
    std::thread th;
    explicit Helper(std::function<void(int)> f) : step(std::move(f)) {
        th = std::thread([this] {
            tid = tid_of_self();
            int seen = 0;
            while (!stop.load()) {
                if (go.load() != seen) {
                    seen = go.load();
                    step(seen);
                    done = seen;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        for (int i = 0; i < 1000 && !tid; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    void run(int k) {
        go = k;
        for (int i = 0; i < 2000 && done.load() != k; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ~Helper() {
        stop = true;
        th.join();
    }
};

void test_live() {
    const V allowed = mask_of(0);
    if (allowed.size() < 4) {
        std::printf("prompt_cpus_test: live placement skipped (needs 4 allowed CPUs, have %zu)\n", allowed.size());
        return;
    }
    const int host = allowed[0], worker = allowed[1], aux = allowed[2];
    const V off_host(allowed.begin() + 1, allowed.end()), off_both(allowed.begin() + 2, allowed.end());

    // no --aux-cpus: a prompt thread takes every CPU but the host's, and the sweep (off) leaves it alone
    std::string err;
    CHECK(configure("", err));
    (void) set_topology(host, V{worker});
    CHECK(!enabled());
    Helper prompt([](int k) { if (k == 1) place_prompt_thread(); });
    prompt.run(1);
    CHECK(mask_of(prompt.tid) == off_host);

    // a thread the sweep moved before it placed itself: placed, and a restore does not put the old mask back
    Helper late([](int k) { if (k == 1) place_prompt_thread(); });
    CHECK(configure(std::to_string(aux), err));
    (void) set_topology(host, V{worker});
    CHECK(enabled());
    CHECK(mask_of(late.tid) == V{aux});                   // the start-up sweep took it
    CHECK(mask_of(prompt.tid) == off_host);               // owned: the sweep left it
    late.run(1);
    CHECK(mask_of(late.tid) == off_host);
    CHECK(sweep() == 0);
    CHECK(mask_of(late.tid) == off_host);
    set_enabled(false);
    CHECK(mask_of(late.tid) == off_host);
    CHECK(mask_of(prompt.tid) == off_host);

    // while the pool can run a prompt thread leaves the workers' cores (the aux set when it is on), then comes back
    Helper chunk([](int k) {
        if (k % 2 == 1) place_prompt_thread();
        else (void) place_off_pool();
    });
    chunk.run(1);
    CHECK(mask_of(chunk.tid) == off_host);
    chunk.run(2);
    CHECK(mask_of(chunk.tid) == off_both);                // aux off: every CPU but the host's and the worker's
    chunk.run(3);
    CHECK(mask_of(chunk.tid) == off_host);
    set_enabled(true);
    CHECK(mask_of(chunk.tid) == off_host);                // owned: the sweep left it
    chunk.run(4);
    CHECK(mask_of(chunk.tid) == V{aux});
    set_enabled(false);
    CHECK(mask_of(chunk.tid) == V{aux});                  // the sweep never moved it, so the restore leaves it

    // a list with no spare CPU: no plan, so the aux set is not used either
    CHECK(configure(std::to_string(worker), err));
    (void) set_topology(host, V{worker});
    CHECK(plan().empty());
    chunk.run(5);
    CHECK(mask_of(chunk.tid) == off_host);
    chunk.run(6);
    CHECK(mask_of(chunk.tid) == off_both);

    // every CPU but the host's is a worker's: the workers' cores, never the host's (it pumps the decode windows)
    CHECK(configure("", err));
    (void) set_topology(host, off_host);
    chunk.run(7);
    CHECK(mask_of(chunk.tid) == off_host);
    chunk.run(8);
    CHECK(mask_of(chunk.tid) == off_host);
    CHECK(mask_of(tid_of_self()) == allowed);             // the host (configure registered it) never moves
}
#else
void test_live() { std::printf("prompt_cpus_test: live placement is Linux only\n"); }
#endif

}  // namespace

int main() {
    test_prompt_cpus();
    test_off_pool_cpus();
    test_live();
    if (failures) {
        std::fprintf(stderr, "prompt_cpus_test: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("prompt_cpus_test: ok\n");
    return 0;
}
