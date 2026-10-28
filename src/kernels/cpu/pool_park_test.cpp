// ExpertPool::set_active: parked workers sleep, take no work and change no result.
#include "ggml.h"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#endif

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int kWorkers = 6;
constexpr int kJobs = 7;

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("pool_park_test: FAILED: %s\n", what);
        ++g_failures;
    }
}

bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool all_finite(const std::vector<float>& a) {
    return std::all_of(a.begin(), a.end(), [](float v) { return std::isfinite(v); });
}

// One kind of batch: run() it on a pool, then result() holds the outputs it wrote (NaN where it wrote nothing)
struct Batch {
    const char* name;
    std::function<void(cpu::ExpertPool&)> run;
    std::function<std::vector<float>()> result;
};

// A native Q4_0 pack's experts: run_split_multi_native, the decode path of a native pack
struct Native {
    cpu::NativeFmt f;
    std::vector<uint8_t> blob;
    std::vector<std::vector<uint8_t>> act;
    std::vector<cpu::ActQ> qact;
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<float> out;
    bool ok = false;
    Native() : act(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes)), qact(cpu::MAXT), jobs(kJobs),
               out((size_t) kJobs * cpu::MAXT * cpu::H) {
        std::string err;
        if (!cpu::native_fmt(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, cpu::H, cpu::FF, f, err)) return;
        blob.assign(f.bytes, 0);
        std::vector<float> weights((size_t) cpu::H * cpu::FF);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.02f * std::sin((float) i * 0.17f);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data(), 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + f.up_off, 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + f.down_off, 0, cpu::H, cpu::FF, nullptr);
        std::vector<float> x(cpu::H);
        for (int t = 0; t < cpu::MAXT; ++t) {
            for (int i = 0; i < cpu::H; ++i) x[(size_t) i] = std::sin((float) (i + 13 * t) * 0.031f);
            cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
            cpu::act_quant_any(x.data(), cpu::H, qact[(size_t) t]);
        }
        for (int e = 0; e < kJobs; ++e) {
            jobs[(size_t) e].blob = blob.data();
            jobs[(size_t) e].nt = e % 3 == 0 ? 1 : e % 3 == 1 ? 4 : cpu::MAXT;
            for (int t = 0; t < jobs[(size_t) e].nt; ++t) {
                jobs[(size_t) e].nact[t] = act[(size_t) t].data();
                jobs[(size_t) e].act[t] = &qact[(size_t) t];
                jobs[(size_t) e].out[t] = out.data() + ((size_t) e * cpu::MAXT + (size_t) t) * cpu::H;
            }
        }
        ok = true;
    }
    void run(cpu::ExpertPool& pool) {
        std::fill(out.begin(), out.end(), 0.0f);   // the token slots a job does not use stay zero
        for (const auto& job : jobs)
            for (int t = 0; t < job.nt; ++t)
                std::fill(job.out[t], job.out[t] + cpu::H, std::numeric_limits<float>::quiet_NaN());
        pool.run_split_multi_native(f, jobs.data(), kJobs);
    }
};

// The Strata-form kernels (they need AVX-512 VBMI): run(), run_split() and run_split_multi()
struct Strata {
    std::vector<std::vector<uint8_t>> blobs;
    std::vector<cpu::ActQ> act;
    std::vector<float> out;
    std::vector<cpu::ExpertJob> jobs;
    std::vector<cpu::ExpertJobMulti> multi;
    Strata() : blobs(kJobs, std::vector<uint8_t>(cpu::BLOB, 0)), act(2), out((size_t) kJobs * cpu::H * 3), jobs(kJobs), multi(kJobs) {
        std::mt19937 rng(7);
        for (auto& b : blobs) {
            for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
            for (size_t i = cpu::O_GU_SCALES; i + 1 < cpu::BLOB; i += 2) {
                const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
                std::memcpy(b.data() + i, &h, 2);
            }
        }
        for (int t = 0; t < 2; ++t) {
            std::vector<float> x(cpu::H);
            for (int i = 0; i < cpu::H; ++i) x[(size_t) i] = std::sin((float) (i + 13 * t) * 0.031f);
            cpu::act_quant_q8_1(x.data(), cpu::H, act[(size_t) t]);
        }
        for (int e = 0; e < kJobs; ++e) {
            jobs[(size_t) e].blob = blobs[(size_t) e].data();
            jobs[(size_t) e].act = &act[0];
            jobs[(size_t) e].out = out.data() + (size_t) e * cpu::H;
            jobs[(size_t) e].slot = e;
            multi[(size_t) e].blob = blobs[(size_t) e].data();
            multi[(size_t) e].nt = 2;
            for (int t = 0; t < 2; ++t) {
                multi[(size_t) e].act[t] = &act[(size_t) t];
                multi[(size_t) e].out[t] = out.data() + ((size_t) kJobs + (size_t) e * 2 + (size_t) t) * cpu::H;
            }
        }
    }
    void run(cpu::ExpertPool& pool, int which) {
        std::fill(out.begin(), out.end(), std::numeric_limits<float>::quiet_NaN());
        if (which == 0) pool.run(jobs.data(), kJobs);
        else if (which == 1) pool.run_split(jobs.data(), kJobs);
        else pool.run_split_multi(multi.data(), kJobs);
    }
    std::vector<float> result(int which) const {
        const size_t from = which == 2 ? (size_t) kJobs * cpu::H : 0;
        const size_t n = (size_t) cpu::H * (which == 2 ? 2 * kJobs : kJobs);
        return std::vector<float>(out.begin() + (long) from, out.begin() + (long) (from + n));
    }
};

#if defined(__linux__)
// utime + stime ticks of "strata-pool-<id>" for each id, from /proc/self/task
std::vector<long> worker_ticks(int n) {
    std::vector<long> ticks((size_t) n, -1);
    if (DIR* d = opendir("/proc/self/task")) {
        while (const dirent* e = readdir(d)) {
            if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
            char path[320];
            std::snprintf(path, sizeof path, "/proc/self/task/%s/stat", e->d_name);
            std::FILE* f = std::fopen(path, "r");
            if (!f) continue;
            char buf[1024];
            const size_t got = std::fread(buf, 1, sizeof buf - 1, f);
            std::fclose(f);
            buf[got] = '\0';
            const char* open = std::strchr(buf, '(');
            const char* close = std::strrchr(buf, ')');
            if (!open || !close || close < open) continue;
            const std::string comm(open + 1, close);
            int id = -1;
            if (std::sscanf(comm.c_str(), "strata-pool-%d", &id) != 1 || id < 0 || id >= n) continue;
            long ut = 0, st = 0;
            std::sscanf(close + 2, "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %ld %ld", &ut, &st);
            ticks[(size_t) id] = ut + st;
        }
        closedir(d);
    }
    return ticks;
}
#endif

double ms_of(const std::function<void()>& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main() {
    if (!cpu::cpu_avx1_ok()) {
        std::printf("pool_park_test: no AVX on this CPU, skipped\n");
        return 0;
    }
    Native nat;
    if (!nat.ok) {
        std::printf("pool_park_test: no native Q4_0 expert layout, skipped\n");
        return 0;
    }
    std::unique_ptr<Strata> s2;
    if (cpu::cpu_features().usable()) s2.reset(new Strata());
    std::vector<Batch> batches;
    batches.push_back({"native", [&](cpu::ExpertPool& p) { nat.run(p); }, [&] { return nat.out; }});
    if (s2)
        for (int which = 0; which < 3; ++which)
            batches.push_back({"strata", [&s2, which](cpu::ExpertPool& p) { s2->run(p, which); }, [&s2, which] { return s2->result(which); }});

    cpu::ExpertPool pool(kWorkers, /*pin=*/false, /*host_works=*/true);
    if (pool.workers() != kWorkers) {
        std::printf("pool_park_test: SKIPPED, the pool has %d workers\n", pool.workers());
        return 0;
    }

    // 4. the count: clamped, and workers() wakes everyone
    check(pool.active_workers() == kWorkers, "a new pool has every worker active");
    check(pool.set_active(0) && pool.active_workers() == 1, "0 is clamped to 1");
    check(pool.set_active(99) && pool.active_workers() == kWorkers, "99 is clamped to workers()");
    check(pool.set_active(3) && pool.active_workers() == 3, "3 active");

    // 1. the same bytes at every active count
    std::vector<std::vector<float>> want;
    pool.set_active(kWorkers);
    for (Batch& b : batches) {
        b.run(pool);
        want.push_back(b.result());
        check(all_finite(want.back()), "every output is written");
    }
    for (int k : {1, 2, 3, kWorkers - 1, kWorkers})
        for (size_t i = 0; i < batches.size(); ++i) {
            pool.set_active(k);
            for (int rep = 0; rep < 3; ++rep) {
                batches[i].run(pool);
                check(same(batches[i].result(), want[i]), "the outputs do not depend on the active count");
            }
        }

    // 2. parked workers neither spin nor claim: a steady load for 400 ms
#if defined(__linux__)
    {
        auto load_for_ms = [&](int ms) {
            const auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(ms)) batches[0].run(pool);
        };
        pool.set_active(3);
        load_for_ms(50);   // the parked workers settle
        const std::vector<long> a = worker_ticks(kWorkers);
        load_for_ms(400);
        const std::vector<long> b = worker_ticks(kWorkers);
        for (int i = 0; i < kWorkers; ++i) {
            if (a[(size_t) i] < 0 || b[(size_t) i] < 0) {
                std::printf("pool_park_test: worker %d has no thread\n", i);
                ++g_failures;
                continue;
            }
            const long used = b[(size_t) i] - a[(size_t) i];   // 10 ms ticks
            std::printf("  worker %d (%s): %ld ms of CPU in 400 ms\n", i, i < 3 ? "active" : "parked", used * 10);
            if (i < 3) check(used >= 8, "an active worker works or spins under a steady load");
            else check(used <= 3, "a parked worker uses no CPU");
        }
        pool.set_active(kWorkers);   // and they wake again
        load_for_ms(50);
        const std::vector<long> c = worker_ticks(kWorkers);
        load_for_ms(400);
        const std::vector<long> d = worker_ticks(kWorkers);
        for (int i = 3; i < kWorkers; ++i) check(d[(size_t) i] - c[(size_t) i] >= 8, "a woken worker works again");
    }
#endif

    // 3. parking and waking between batches, with gaps that put the workers to sleep, every batch checked
    {
        std::mt19937 rng(11);
        long count = 0;
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(4)) {
            const uint32_t r = rng();
            if (r % 40 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(21 + (int) (rng() % 8)));
            else if (r % 5 == 0) std::this_thread::sleep_for(std::chrono::microseconds(rng() % 200));
            if (r % 3 == 0) pool.set_active(1 + (int) (rng() % kWorkers));
            const size_t i = (size_t) (rng() % batches.size());
            batches[i].run(pool);
            if (!same(batches[i].result(), want[i])) {
                check(false, "a batch while parking and waking");
                break;
            }
            ++count;
        }
        std::printf("  %ld batches while parking and waking\n", count);
        pool.set_active(kWorkers);
    }

    // 5. the first batch after waking parked workers against the first batch after the whole pool slept
    {
        std::vector<double> slept, woken;
        for (int trial = 0; trial < 40; ++trial) {
            pool.set_active(kWorkers);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));   // every worker sleeps
            slept.push_back(ms_of([&] { batches[0].run(pool); }));
            pool.set_active(3);
            batches[0].run(pool);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            pool.set_active(kWorkers);   // the wake is asked for here, the batch follows at once
            woken.push_back(ms_of([&] { batches[0].run(pool); }));
        }
        std::sort(slept.begin(), slept.end());
        std::sort(woken.begin(), woken.end());
        std::printf("  first batch after a sleep: median %.3f ms, after waking parked workers: median %.3f ms\n",
                    slept[slept.size() / 2], woken[woken.size() / 2]);
    }

    if (g_failures != 0) {
        std::printf("pool_park_test: %d FAILED\n", g_failures);
        return 1;
    }
    std::printf("pool_park_test OK\n");
    return 0;
}
