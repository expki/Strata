# Mixed-tier optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faster prompts and decode for NVIDIA + AMD serving, decided from run-time measurements, so it holds up when most experts live in RAM (Swift 1.5 Q6_K, 157 GB, on the same four GPUs).

**Architecture:** Stage 0 builds the measurement (permanent diagnostics, a per-card expert cap, a RAM-heavy setup, a prompt-path KL check, Q6_K support). Stage 1 fixes CPU placement in the engine. Stage 2 adds one expert planner that decides per prompt chunk who computes each expert, including the W6800 streaming experts from RAM (library ABI v3). Stage 3 overlaps the W6800 with the stage GPU's attention by splitting each chunk's layer work into two token halves. Stage 4 trims the W6800's decode round trip.

**Tech Stack:** C++17, CUDA 13.2 (sm_86), HIP (ROCm 7.2.4, gfx1030), CMake + Ninja, Python 3.12 tools. Builds and runs happen on the test box only.

**Spec:** `docs/superpowers/specs/2026-10-25-mixed-tier-optimizations-design.md`

## Global Constraints

- Hard fork: breaking changes are fine. A better behavior becomes the default, and switches it makes redundant are removed.
- Linux with CUDA and HIP only. Build both trees for every change. Windows, Mac (Metal) and SYCL need not keep building.
- Every split of work between devices is decided from what the engine measures or reads at run time, never from constants fitted to this box.
- A change stays only if it is faster beyond noise in both setups (NORMAL and RAMHEAVY), or faster in one and neutral in the other.
- Correctness: a change that only moves data or threads keeps greedy answers identical (`ab_engine.py --identity`) and the parity tests passing. A change that splits the math differently keeps the parity tests passing and passes the prompt KL gate: an exact match (KL 0) where an exact reference path exists, and against `kl-ref-normal` on the all set of 30 prompts a mean KL at or below 0.03 (and the long set's mean too), a median at or below 0.005 and top-1 agreement at or above 90% (gate restated after Task 8: every valid rounding change measured 0.010 to 0.020 mean KL, medians 0.0013 to 0.0032 and top-1 93 to 100% on the all set).
- Code comments: one line, plain, no em dashes, no semicolons, no hyphens joining clauses.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push the branch `mixed-tier-perf` to origin after each task, never force-push, keep `main` clean.
- Never edit files inside the box checkout `~/source/strata` (the live service's tree). Build in `~/source/strata-base` (new) and `~/source/strata-prev` (the previous accepted commit).
- Box runs: stop the service only through a stint script that backs up and restores the live unit, caps the 3090 and 3080 Ti at 250 W, and brings the service back. About 5 minutes per arm.
- Every gain claimed is a measurement on the box, written down with the setup and the numbers.

## Review Focus

- Runs without the AMD tier (an NVIDIA-only layer split, one NVIDIA card, the HIP-only engine) must keep working and not get slower. Pinned in Tasks 6, 10 and 13 (NVIDIA-only arms) and Tasks 6 and 13 (the HIP-only smoke run).
- Short prompts (a chat turn of tens of tokens, a prompt below 1,024 tokens that takes the per-layer path) must stay correct and not slow down. Pinned in Task 9 (`zero_rows`, small items), Task 12 (a 700-word prompt) and every A/B (the chat prompts).
- The first prompt after the engine starts (no measured rates, no previous routing) must not be slower than today. Pinned in Task 10.
- A W6800 that stops answering during a prompt with streamed experts must end the engine cleanly after the 30 s wait, as today. Pinned in Task 11.
- RAM too short for the W6800's staging ring, or a pinned RAM copy that cannot be registered with HIP, must fall back to the previous behavior with one log line, not fail. Pinned in Task 11.

## Box protocol (every task that builds or measures uses this)

- **Access:** `ssh vdh@192.168.10.5`. Bench files live in `~/bench-mixed/`. The live service is `strata.service` (port 5999, config `~/source/strata/strata-swift-iq4_xs-mix.json`).
- **Build NEW:** push the branch, then on the box: `cd ~/source/strata-base && git fetch -q origin mixed-tier-perf && git checkout -q --detach FETCH_HEAD && nice -n 10 cmake --build build-cuda -j10 && nice -n 10 cmake --build build-hip --target strata_xtier_hip strata -j10`. Run unit tests with `ctest --test-dir build-cuda -R <name> --output-on-failure`. GPU tests use the 3080 Ti while the service runs: `CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0`.
- **Build PREV:** the same in `~/source/strata-prev` at the commit before the task (`git checkout -q --detach <sha>`).
- **Stints:** copy `~/bench-mixed/perf5-stint.sh` and `perf5-restore.sh` to `taskN-stint.sh` and `taskN-restore.sh`, rename the `perf5` prefix, and replace the A/B lines with the task's arms. They check that the service is idle first (no `answering` or `reading the prompt` in the last 2 minutes), back up the live unit, set the caps, run, restore and wait for `/health` to say `"loaded": true`.
- **Arms:** `NAME:config=<cfg>,exe=<tree>/build-cuda/strata,STRATA_XTIER_LIB=<tree>/build-hip/libstrata_xtier_hip.so,xargs=<a;b>,<ENV=V>`.
- **Setups** (Task 4 creates the configs):
  - NORMAL: `~/bench-mixed/cfg-normal.json`, Swift 1.5 IQ4_XS, all four GPUs, `--prefill 4096`.
  - RAMHEAVY: `~/bench-mixed/cfg-ramheavy.json`, Swift 1.5 IQ3_S, all four GPUs, `--prefill 4096`, `--expert-gib cuda0=2.5,cuda1=1.5,cuda2=6.5,hip0=5` (about a third of 46.8 GiB on GPUs). Changed in Task 4b: with `cuda0=1` the RTX 3050 could neither lend nor hold a 4096-token chunk's buffers, so every prompt ran in 1024-token chunks.
  - NVIDIA-only: `~/bench-mixed/cfg-nvonly.json`, Swift 1.5 IQ3_S on the three NVIDIA cards, NORMAL's other arguments, no `--peer-device`, no library env, no `--expert-gib`. Changed after Task 4b: IQ4_XS without the tier read 1.3 to 8.6 GB from the drives per request on this box.
  - HIP-only smoke: `~/source/strata/strata-swift-iq4_xs-amd.json` with `exe` pointed at `<tree>/build-hip/strata`, one chat and one 4K-word prompt.
  - Per setup: `--rounds 1 --warm 1 --measure 2 --max-new 128 --long-words 19000`.
  - Identity is its own pair of arms: `xargs=--adapt-swaps;0;--pcie-frac;0;--prompt-cache;0` on both, `--identity --warm 0 --measure 1 --full-text`.
  - KL: `quant_quality.py prompt --config cfg-normal.json --out RUN --keep-caches`, then `pcompare ~/bench-mixed/kl-ref-normal RUN` (30 prompts, the long set crosses a 4096-token chunk). Read the long set's mean for prompt-path changes.
  - Never run an IQ4_XS arm right after an IQ3_S arm without a warm pass. Read `file tier I/O` before comparing arms.
  - The last stint of each stage (Tasks 6, 11 and 13) adds one NORMAL arm with `xargs=--prefill;auto`, so the gain is also checked at the chunk the start-up plan picks.
- **Results:** append each task's numbers to `~/bench-mixed/plan-results.md`: date and time, commits, setup, the arms' prompt and decode tok/s per pass, the KL where measured.

---

## Stage 0: measurement

### Task 1: Keep the prompt diagnostics, drop the test switches

**Files:**
- Modify: `src/xtier/xtier_lib.cu`, `src/prefill/prefill.cpp`, `src/program/generate.cpp` (from commit `f70980e5` on branch `perf-diag`)
- Modify: `docs/MULTI_GPU.md` (the `STRATA_XTIER_TIMING` row), `docs/DETAILS.md` (the `STRATA_PREFILL_TIMING` paragraph)

**Interfaces:**
- Produces: the log lines `strata xtier prompt timing: ...` (every 45 prompt layers under `STRATA_XTIER_TIMING=1`) and `strata prefill timing: host per MoE layer (...)` (under `STRATA_PREFILL_TIMING=1`). Later tasks read them.

- [ ] **Step 1:** `git cherry-pick f70980e5` onto `mixed-tier-perf`.
- [ ] **Step 2:** Remove `STRATA_XTIER_PF_STAGES` (`generate.cpp`) and `STRATA_XTIER_PF_SHARE` (`prefill.cpp`) and their code. Keep the timing.
- [ ] **Step 3:** Document both lines in the two docs, one sentence each on what every field means.
- [ ] **Step 4:** Build NEW. Configure the tests once: `cmake -S . -B build-cuda -DSTRATA_BUILD_TESTS=ON` in `~/source/strata-base`. Run `ctest --test-dir build-cuda -R "xtier|peer_adapt" --output-on-failure`. Expected: all pass or exit 77 (no card).
- [ ] **Step 5:** Commit `diag: the W6800's per-layer prompt timing and the host's per-layer prompt steps, kept`, push.

### Task 2: A per-card expert cap, `--expert-gib`

**Files:**
- Create: `include/strata/program/expert_gib.hpp`, `tests/program/expert_gib_test.cpp`
- Modify: `src/program/generate.cpp` (Options at ~419, the parse loop at ~2418, `usage()` ~808-1080, the primary's auto size ~6056-6146, `stage_room` ~4575), `src/core/peer_experts.cpp` (`open_foreign` budget ~280), `include/strata/core/peer_experts.hpp`, `CMakeLists.txt` (the test), `docs/MULTI_GPU.md`

**Interfaces:**
- Produces:
```cpp
namespace strata::program {
// "cuda0=1,cuda1=3,hip0=6.5": an expert-memory cap in GiB per CUDA device index and per AMD tier card
struct ExpertGib { std::map<int, double> cuda; std::map<int, double> hip; };
bool parse_expert_gib(const std::string& text, ExpertGib& out, std::string& err);
// the cap in bytes for a device, or UINT64_MAX when none is set
uint64_t expert_cap_bytes(const ExpertGib& g, bool hip, int device);
}
```
- `PeerExperts::open_foreign` gains a `uint64_t cap_bytes` parameter (UINT64_MAX = no cap).

- [ ] **Step 1: Write the failing test** `tests/program/expert_gib_test.cpp`:
```cpp
ExpertGib g; std::string err;
CHECK(parse_expert_gib("cuda0=1,cuda2=5.5,hip0=6.5", g, err));
CHECK(expert_cap_bytes(g, false, 0) == 1ull << 30);
CHECK(expert_cap_bytes(g, false, 2) == (uint64_t) (5.5 * 1073741824.0));
CHECK(expert_cap_bytes(g, true, 0) == (uint64_t) (6.5 * 1073741824.0));
CHECK(expert_cap_bytes(g, false, 1) == UINT64_MAX);
for (const char* bad : {"", "cuda0", "cuda0=", "cuda=1", "gpu0=1", "cuda0=-1", "cuda0=1,cuda0=2", "cuda0=x"})
    CHECK(!parse_expert_gib(bad, g, err) && !err.empty());
```
- [ ] **Step 2:** Register it in `CMakeLists.txt` beside `peer_adapt_test` and run it on the box. Expected: FAIL (no header).
- [ ] **Step 3:** Implement the header (inline functions). Wire `--expert-gib TEXT` into Options and the parse loop (a bad value exits 2 with the parser's message). Apply the cap as `min(computed, cap)` in three places: the primary's auto slots, `stage_room` (so `--layer-split auto` sees it), and `open_foreign`'s budget. An explicit `--expert-cache N` still wins on the primary. Add the help line and a `docs/MULTI_GPU.md` paragraph.
- [ ] **Step 4:** Run the test. Expected: PASS. Build both trees.
- [ ] **Step 5:** Box check, no stint needed beyond one start: start `cfg-ramheavy.json`'s engine args with `--expert-gib cuda0=1,cuda1=3,cuda2=5,hip0=6.5` in a stint, read the start-up lines. Expected: each card's expert cache at or below its cap, and the AMD tier line at or below 6.5 GiB. (If Task 4 is not done yet, do this check inside Task 4's first stint.)
- [ ] **Step 6:** Commit `engine: --expert-gib caps each card's expert memory`, push.

### Task 3: The start-up chunk plan and the RAM copy count the AMD tier

**Files:**
- Modify: `src/program/generate.cpp` (`host_plan_chunk` call ~8640-8645 where `gc_bytes`/`gc_slots` are summed over `pf_parts`, and `stage_pairs` ~8168-8174)

**Interfaces:**
- Consumes: `PeerExperts::gib()` / `resident()` (`include/strata/core/peer_experts.hpp:93,100`), `peer.foreign()`, `peer.has(l, e)`.

- [ ] **Step 1:** Add the AMD tier's expert bytes to `gc_bytes` when `peer.foreign()` (the tier's experts are not read from the drives during a prompt). Keep `gc_slots` for the NVIDIA caches only if `host_plan_chunk` uses it per slot. Read how `kept_bytes` uses it before deciding.
- [ ] **Step 2:** In `stage_pairs`, leave out pairs the AMD tier holds (`peer.has(l, e)`), so a pinned RAM copy (`--resident-budget-gib`) spends its budget on experts no GPU holds.
- [ ] **Step 3:** Build both trees. Box check inside the next stint (Task 4's baseline stint): the `prefill plan inputs:` line counts about 31 GiB more in the GPU caches for the IQ4_XS mix config, and with `--prefill auto` the plan's chunk is 4096 or larger on that config. Record the line.
- [ ] **Step 4:** Commit `prefill plan: the AMD tier's experts count as held, the RAM copy skips them`, push.

### Task 4: The measurement kit and the baseline

**Files:**
- Modify: `tools/quant_quality.py` (`engine_args` ~82-98, `cmd_prompt` ~176): add `--keep-caches`
- Modify: `tools/ab_engine.py` only if an arm needs something it cannot express
- Create on the box: `~/bench-mixed/cfg-normal.json`, `~/bench-mixed/cfg-ramheavy.json`, `~/bench-mixed/plan-results.md`, `~/source/strata-prev` (a second worktree with `build-cuda` and `build-hip` configured like `strata-base`'s)

**Interfaces:**
- Produces: `python tools/quant_quality.py prompt --config CFG --out RUN --keep-caches [--env K=V ...]` (the config's cache and tier arguments kept as they are), and `pcompare REF RUN` as today.

- [ ] **Step 1:** Add `--keep-caches` to `quant_quality.py prompt`: `engine_args` then leaves `--expert-cache` and every other cache argument as the config has them. Add `tools/test_quant_quality.py` (none exists yet) with a case that checks `engine_args` keeps `--expert-cache auto` with the flag and replaces it without.
- [ ] **Step 2:** Create the three configs (NORMAL, RAMHEAVY, NVIDIA-only) from `~/bench-mixed/cfg-perf.json` (NORMAL: `--prefill 4096`) and from the IQ3_S config (`~/source/strata/strata-swift-iq3_s.json`, pack `swift-iq3_s`, plus `--peer-device hip:0`, the mix config's env and lib dirs, `--prefill 4096`, `--expert-gib cuda0=1,cuda1=3,cuda2=5,hip0=6.5`). Both point at `~/source/strata-base` builds. Arms override `exe` and `STRATA_XTIER_LIB` for PREV.
- [ ] **Step 3:** Create `~/source/strata-prev` with `git worktree add --detach`, configure both trees with the same options as `strata-base` (copy the cache values; HIP with `-DSTRATA_XTIER=ON`), build at the current head.
- [ ] **Step 4:** Baseline stint (about 10 minutes): one arm per setup at the current head, the RAMHEAVY page cache checked (the `prefill plan inputs:` line's page cache against the file set: the RAM side must fit), and a KL reference: `quant_quality.py prompt --keep-caches` on NORMAL, written to `~/bench-mixed/kl-ref-normal`. Expected: RAMHEAVY's `file tier I/O` lines read under 1 GB from storage per measured request after the warm pass. If they read more, lower the caps until the RAM side fits and write the new caps into the configs and this plan.
- [ ] **Step 5:** Write the baseline into `plan-results.md`. Commit the tool change `tools: quant_quality prompt keeps the config's caches with --keep-caches`, push.

### Task 4b: The measurement kit, fixed (added after Task 4's baseline)

Done (commits `0e0ffcc2`, `9997e38e`). Task 4's baseline showed RAMHEAVY never reaching 4096-token chunks, a prompt KL set that never reached the 4096-token path, and an NVIDIA-only arm confounded by drive reads.

- The chunk scan keeps a chunk when a stage that cannot lend its buffers from its cache can hold them as its own (`include/strata/program/prompt_buffers.hpp`, test `tests/program/prompt_buffers_test.cpp`). With no `--expert-gib` every start that lends today lends exactly as today.
- `quant_quality.py prompt --keep-caches` runs the engine as the config starts it, and the prompt set gains long prompts (4,500 to 6,000 tokens). `kl-ref-normal` was regenerated (30 prompts).
- RAMHEAVY's caps and the NVIDIA-only setup changed (see the box protocol).

### Task 5: Swift 1.5 Q6_K runs with the AMD tier

**Files:**
- Modify: `src/prefill/moe_mmq.cu` (~45, `kq_ok()`), `src/kernels/mmq_kq_loader.cpp` (~110) or wherever the K-quant MMQ is gated, `src/xtier/xtier_lib.cu` (~407 `mmq::fits`)
- Modify: `src/kernels/xtier_decode_parity.cpp`, `src/prefill/xtier_prefill_parity.cpp`, `src/kernels/test_blobs.hpp` (a case with the Q6_K pack's gate/up and down types)
- Create on the box: `~/source/Strata-data/packs/swift-q6_k` (via `tools/iq_pack.py`), `~/bench-mixed/cfg-q6k.json`

**Interfaces:**
- Produces: K-quant prompt MMQ is on by itself for Q6_K experts (no `STRATA_MMQ_KQ=1`). Q4_K, Q5_K and Q5_1 stay opt-in with `STRATA_MMQ_KQ=1` (ruling: their mean KL was worse with MMQ on the R9700). `STRATA_MMQ_KQ=0` turns all off. Swift Q6_K's own KL (MMQ on against off) is checked in Task 15.

- [ ] **Step 1:** Build the pack: `python tools/iq_pack.py --gguf /opt/models/ukisai_Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00001-of-00005.gguf --base ~/source/Strata-data/packs/swift-iq4_xs --out ~/source/Strata-data/packs/swift-q6_k` on the box (if `--base` refuses the different quant, build standalone). Read `native_experts.txt`'s gate/up and down types.
- [ ] **Step 2:** Add those types to both parity tests (the decode and the prompt one). Run them. Expected: FAIL only if the K-quant MMQ is off by default (the prompt test), PASS for decode.
- [ ] **Step 3:** Make K-quant MMQ default on where the pack needs it, on CUDA and HIP. Run both parity tests. Expected: PASS.
- [ ] **Step 4:** Smoke stint: `cfg-q6k.json` (the mix config with the Q6_K pack and shards, `--prefill 2048`), one chat request and one 4K-word prompt. Expected: answers that make sense, the DONE line's `amd_tier=` above 0, no error lines. The drives carry most reads today, so no speed is recorded.
- [ ] **Step 5:** Commit `Q6_K: K-quant prompt kernels on by themselves, parity cases for the Q6_K pack's types`, push.

## Stage 1: CPU placement

### Task 6: Prompt-only threads use every CPU but the host's

**Files:**
- Modify: `include/strata/platform/aux_cpus.hpp` (beside `place_off_host` ~318 and `set_topology` ~378)
- Modify: `src/prefill/prefill.cpp` (the Stager threads ~589, `strata-hand` ~391, `strata-issue` ~3261, `next_run_` ~5152, the PLE gather `std::async` ~3045 and ~3074)
- Create: `tests/platform/prompt_cpus_test.cpp`, register in `CMakeLists.txt`
- Modify: `docs/DETAILS.md` (where `--pool-workers` and `--aux-cpus` are explained)

**Interfaces:**
- Produces:
```cpp
namespace strata::aux_cpus {
// The CPUs a prompt-only thread may use: every allowed CPU but the host's, or all allowed when that leaves none
std::vector<int> prompt_cpus(const std::vector<int>& allowed, int host_cpu);
// Moves the calling thread there and marks it owned so a sweep leaves it alone (works with or without an --aux-cpus plan)
void place_prompt_thread();
}
```

- [ ] **Step 1: Write the failing test:**
```cpp
CHECK(prompt_cpus({0,1,2,3}, 0) == std::vector<int>({1,2,3}));
CHECK(prompt_cpus({0,1,2,3}, 2) == std::vector<int>({0,1,3}));
CHECK(prompt_cpus({5}, 5) == std::vector<int>({5}));
CHECK(prompt_cpus({1,2}, -1) == std::vector<int>({1,2}));
```
- [ ] **Step 2:** Run it on the box. Expected: FAIL (no function).
- [ ] **Step 3:** Implement both. `set_topology` (or `configure` when no pool exists) stores the host CPU. Call `place_prompt_thread()` first thing in the Stager threads, `strata-hand`, `strata-issue` and `next_run_`. The PLE gather calls it only when its chunk does not hold the pool (`pool_hold.held` false), else `pin_current_thread()` as today. Threads that work during decode keep their placement (Task 7 measures them).
- [ ] **Step 4:** Run the test. Expected: PASS. Build both trees.
- [ ] **Step 5:** Stint, about 15 minutes, arms PREV and NEW on NORMAL (with `--identity`), on RAMHEAVY, and on an NVIDIA-only NORMAL (`xargs` without the tier: a config copy with `--peer-device` removed). Add the HIP-only smoke run and the `--prefill auto` arm. Expected: NORMAL's 20K prompt at least 1,790 tok/s with 15 pool workers (the `--pool-workers 7` + `STRATA_AUX_CPUS=auto` result), decode not lower, RAMHEAVY and NVIDIA-only not slower, `identical`, the HIP-only engine answers.
- [ ] **Step 6:** Write the results. Commit `cpu placement: prompt-only threads use every CPU but the host's`, push.

### Task 7: Decode-time helper threads, decided by measurement

**Files:**
- Modify (only if a winner): `src/core/expert_source.cpp` (`strata-io` ~1161, `strata-ram` ~4388, `strata-fill` ~6179), `src/program/generate.cpp` (`strata-adapt` ~14555/15226/16780, `JobThread` ~1673)

- [ ] **Step 1:** Stint on RAMHEAVY, three arms of the Task 6 build: as is; `STRATA_AUX_CPUS=auto` with `--pool-workers 13` (two spare cores for helpers); and `--pool-workers 15` with the decode helpers temporarily calling `place_prompt_thread()` (a throwaway local build, not committed). Measure decode on the chat prompts and the long prompt.
- [ ] **Step 2:** If an arm is faster beyond noise in RAMHEAVY and not slower in NORMAL, implement its rule generically (for example: spare cores for helpers sized from the pool's measured idle share, not a fixed count) and re-measure. If none wins, write that down and change nothing.
- [ ] **Step 3:** Write the results. Commit only if code changed, push.

### Task 7b: Decode parks the CPU workers that cost more than they give (added after Task 7)

Done (commits `e8d63b8a..052dde9f`). Task 7 found that a pool with a spinning worker on every core loses to the box's
other threads while it decodes, and that the RAMHEAVY prompt runs at one of two speeds an engine start picks (so Task
7's 4.5% prompt cost was withdrawn, and every later A/B ran two starts per arm, A B B A). Task 7b lets
`ExpertPool::set_active` park workers in the `--serve` decode loop, and `DecodeSpares` finds the count by measurement:
inside each request it alternates the best count and one more parked worker in blocks of 16 decode windows, and after
8 requests it moves when the mean saving is above 2% and above two standard errors (one at first, two after the
review). `STRATA_DECODE_SPARES=0` keeps every worker. Measured on the box at one standard error, one or two starts a
setup: RAMHEAVY decode 49.5 tok/s against 45.0 (+10.1%), with six noise threads +13.7% and +10.5%, NORMAL +2.6% and
+1.2%, greedy answers identical. The two-error rule was replayed on recorded windows (RAMHEAVY at 0.891 of never
parking). Task 15b measured it again at two starts per arm.

## Stage 2: the expert planner

### Task 8: A tier-held expert can be computed by the stage

**Files:**
- Modify: `src/prefill/prefill.cpp` (the stream plan at ~3165-3211, the grouping at ~4105-4140, the streamed walk at ~4926-4948, the short-chunk walk at ~4631-4647 and ~4898-4901)

**Interfaces:**
- Produces, in `Prefill::Impl`: `std::vector<uint8_t> tier_take` (n_layers x n_expert, 1 = the tier computes this expert in this chunk, 0 = the stage does even if the tier holds it). Filled per chunk before the stream plan. Until Task 10 it is 1 for every tier-held expert. Test hook: `STRATA_TEST_TIER_MOVE=N` sets every Nth tier-held expert of each layer to 0.

- [ ] **Step 1: Reproduce.** NEW with `STRATA_TEST_TIER_MOVE=3` on NORMAL, one 4K-word prompt. Expected today: the illegal memory access first reported in `gather_native_group`.
- [ ] **Step 2: Root cause.** The likeliest cause (read, not run): the moved expert is missing from the per-chunk stream plan `seq`, so the streamed walk's else branch takes `peer->slot_ptr(l, e)`, an AMD address. Confirm with `compute-sanitizer --tool memcheck` on a short run or with a log of the branch taken, and write the confirmed cause in the commit message.
- [ ] **Step 3:** Make `tier_take` the one source of truth: the stream plan skips only experts with `tier_take == 1`, the grouping's `on_peer` uses it, and the streamed walk's else branch never takes a foreign tier's slot as a stage blob (an error, not a wrong pointer). The same in the short-chunk walk.
- [ ] **Step 4:** Stint: NORMAL with `STRATA_TEST_TIER_MOVE=3` (no error, answers make sense) and the KL against `kl-ref-normal` with `quant_quality.py prompt --keep-caches --env STRATA_TEST_TIER_MOVE=3`. Expected: the prompt KL gate (Global Constraints). Also NORMAL without the hook with `--identity` against PREV: `identical`.
- [ ] **Step 5:** Commit `prefill: one per-chunk tier_take decides who computes a tier-held expert`, push.

### Task 9: The planner and its rates, as pure code

**Files:**
- Create: `include/strata/prefill/expert_planner.hpp`, `src/prefill/expert_planner.cpp`, `tests/prefill/expert_planner_test.cpp`
- Modify: `CMakeLists.txt` (the source in the prefill library of both trees, the test)

**Interfaces:**
- Produces:
```cpp
namespace strata::prefill::planner {
enum class Where : uint8_t { StageVram, TierVram, StageStream, TierStream };
enum : uint8_t { kStageVram = 1, kTierVram = 2, kStageStream = 4, kTierStream = 8 };
struct Item { int32_t expert; int32_t rows; uint8_t can; uint64_t bytes; bool pinned; };   // can: the k* bits allowed
struct Rates {
    double stage_row_ms, tier_row_ms;                 // compute per routed row
    double stage_byte_ms_paged, stage_byte_ms_pinned; // RAM to the stage GPU per byte, by source kind
    double tier_byte_ms_paged, tier_byte_ms_pinned;   // RAM to the W6800 per byte, by source kind
    double tier_fixed_ms;                             // the tier's per-call cost (handoff, copies of x and the sums)
};
struct Load { double stage_ms = 0, tier_ms = 0; };    // work already queued before this layer (the tier: other stages)
struct Choice { std::vector<Where> where; double stage_ms, tier_ms; };
// Picks where each item runs so that max(stage finish, tier finish) is smallest. tier_stream_slots caps TierStream items.
Choice plan_layer(const std::vector<Item>& items, const Rates& r, const Load& start, int tier_stream_slots);

// Measured rates: ms = a * rows + b * bytes per device and source kind, least squares over recent observations
class RateBook {
public:
    explicit RateBook(const Rates& priors, double half_life_obs = 64);
    void observe_stage(int64_t rows, uint64_t bytes_paged, uint64_t bytes_pinned, double ms);
    void observe_tier(int64_t rows, uint64_t bytes_paged, uint64_t bytes_pinned, int64_t calls, double ms);
    Rates rates() const;   // the priors until enough observations exist, never a negative or zero coefficient
};
}
```
- `plan_layer` starts from each item's home (StageVram if allowed, else TierVram if allowed, else StageStream) and moves items while the larger finish time drops: tier-held items to StageStream (most rows first), StageStream items to TierStream (most rows first, within `tier_stream_slots`). Items with `rows == 0` stay home and cost nothing.

- [ ] **Step 1: Write the failing tests** (`expert_planner_test.cpp`), with `Rates r{0.001, 0.003, 1e-7, 4e-8, 1e-7, 4e-8, 0.5}`:
  - `empty`: no items gives `where` empty and both finish times equal to `start`.
  - `homes_only`: items allowed only at StageVram or TierVram stay home.
  - `tier_slow_moves_busiest`: 20 tier-held items (`can = kTierVram | kStageStream`, rows 400 down to 20) and an idle stage: the moved items are the ones with the most rows, and after planning `|stage_ms - tier_ms|` is at most the cost of the largest moved item.
  - `tier_busy_elsewhere`: the same with `start.tier_ms = 1000`: every movable item goes to StageStream.
  - `ram_to_tier`: 30 RAM items (`can = kStageStream | kTierStream`, pinned) and an idle tier: some go TierStream, at most `tier_stream_slots`, and the result's max finish is below the all-StageStream max finish.
  - `no_tier_slots`: `tier_stream_slots = 0` gives no TierStream.
  - `zero_rows`: an item with 0 rows stays home.
  - `ratebook_recovers`: 200 synthetic stage observations from `ms = 0.002 * rows + 3e-8 * bytes` (rows and bytes varied) give `rates().stage_row_ms` within 5% of 0.002 and `stage_byte_ms_paged` within 5% of 3e-8.
  - `ratebook_priors`: with no observations `rates()` equals the priors, and one degenerate observation (rows 0, bytes 0) changes nothing.
- [ ] **Step 2:** Run on the box. Expected: FAIL (no header).
- [ ] **Step 3:** Implement. The search is greedy as described. `RateBook` keeps exponentially weighted sums for two-variable least squares per device and source kind, with the priors as weighted pseudo-observations.
- [ ] **Step 4:** Run. Expected: PASS. Build both trees.
- [ ] **Step 5:** Commit `prefill: the expert planner and its measured rates`, push.

### Task 10: The planner decides the W6800's and the stage's share

**Files:**
- Modify: `src/prefill/prefill.cpp` (per chunk before the stream plan: build Items per layer, call `plan_layer`, fill `tier_take`), `include/strata/prefill/prefill.hpp`
- Modify: `include/strata/core/xtier_abi.hpp`, `src/xtier/xtier_lib.cu`, `include/strata/core/xtier.hpp`, `src/core/xtier.cpp`: `strata_xtier_prefill_stats` (ABI version 3)
- Modify: `src/program/generate.cpp`: remove `STRATA_XTIER_PREFILL` and the tier's use of `--peer-prefill-rows`. One `RateBook` shared by the stages for the tier, one per stage for its GPU.
- Modify: `docs/MULTI_GPU.md`

**Interfaces:**
- Consumes: Task 8's `tier_take`, Task 9's `plan_layer`, `RateBook`.
- Produces, in the ABI:
```c
#define STRATA_XTIER_ABI_VERSION 3
/* The prompt layers finished since the last call: their rows, bytes copied in, kernel ms, copy ms and count. Always on. */
int strata_xtier_prefill_stats(void* t, int64_t* rows, uint64_t* bytes_in, double* kernel_ms, double* copy_ms, int64_t* layers);
```
- Predicted rows per expert: the stage's previous chunk's routing counts for the same layer (`std::vector<int32_t> last_cnt`, n_layers x n_expert, kept across requests), scaled to this chunk's T. Before any routing exists: `T * K / n_expert` for every expert.
- Observations per chunk: the stage's own expert work timed with one event pair around it per layer (always on, read at the chunk's end), the tier's from `strata_xtier_prefill_stats`.
- The tier's `Load.tier_ms`: work other stages have queued, kept as an atomic estimate the stages update when they queue and when stats arrive.

- [ ] **Step 1:** ABI v3 with `prefill_stats` (cheap events around each layer's kernels and copies, summed in the library). Extend `xtier_loader_test` to check the version and the export. Run. Expected: PASS after the change, FAIL before.
- [ ] **Step 2:** Wire the planner per chunk and fill `tier_take` from its choice. Remove `STRATA_XTIER_PREFILL` and the tier branch of `--peer-prefill-rows`. Starting priors: measure once at start-up with the existing probes where they exist (PCIe probe per card), else use conservative values that keep today's split (tier-held stays on the tier), so a cold engine plans as today.
- [ ] **Step 3:** Build both trees. Run the parity tests and `expert_planner_test`. Expected: PASS.
- [ ] **Step 4:** Stint, about 20 minutes: PREV and NEW on NORMAL and RAMHEAVY. Also: the first measured long prompt after start (the warm pass) of NEW not slower than PREV's; NEW on an NVIDIA-only NORMAL (no tier: the planner does nothing, `--identity` against PREV); the KL on NORMAL against `kl-ref-normal` (the prompt KL gate). Expected: faster than PREV in at least one setup and not slower in the other.
- [ ] **Step 5:** Write the results. Commit `prefill: the planner splits each layer's experts between the stage GPU and the W6800`, push.

### Task 10b: The planner learns within the first long prompt (added after Task 10)

Done (commits `aeada6eb..f8aab11c`, ABI 4). Task 10's planner acted only after about two long prompts. Task 10b moves
the planner's pure core into its own tested file, makes the W6800 return one record per layer call (its rows, bytes and
time, naming the stage that queued it), fits each stage's rates in one rate book per walk from the first measurements,
and plans once a stage and the W6800 both have 16 of them. On NORMAL the
3090 plans from token 4,096 of the first long prompt after a start and the 3080 Ti from 8,192. The first measured long
prompt after a start read at 2,204 tok/s against 1,787 (+23.4%, two starts each), the second at the same speed
(+1.0%), RAMHEAVY at the same speed. The stage's fitted time came within 2.5% of the measured one (the 3050: 57
against 57 ms, before 92 against 53). KL all 0.0024, long 0.0073, the same top token on all 30.

### Task 11: The W6800 streams experts from RAM

**Files:**
- Modify: `include/strata/core/xtier_abi.hpp`, `src/xtier/xtier_lib.cu` (prompt path: a copy stream and streamed experts in place of arena reads), `include/strata/core/xtier.hpp`, `src/core/xtier.cpp`
- Modify: `src/prefill/prefill.cpp` (a per-chunk tier stream list beside `seq`, like `pseq` for the CUDA peer; a tier staging ring in host memory both runtimes map; stager jobs for paged experts; direct addresses for a registered pinned copy), `src/program/generate.cpp` (register the pinned RAM copy with `strata_xtier_register` when there is one)
- Modify: `src/prefill/xtier_prefill_parity.cpp` (streamed experts)
- Modify: `docs/MULTI_GPU.md`

**Interfaces:**
- Produces, in `strata_xtier_prefill_args` (still ABI 3, the struct size check catches a mismatch):
```c
const uint64_t* expert_host;   /* per expert in order_peer order: 0 = in the tier's arena (expert_addr), else the HIP address of its staged or pinned host copy */
const uint32_t* expert_ready;  /* HIP address of the flag word the engine raises when the staged copies of this layer landed, or null */
uint32_t ready_seq;
```
- The planner's `can` gains `kTierStream` for RAM experts when the tier staging ring exists. `tier_stream_slots` is the ring's free slots for the layer.

- [ ] **Step 1: Write the failing parity case:** `xtier_prefill_parity` with some experts given as host copies (`expert_host` set) gives the same sums as the arena path (the existing tolerance). Run. Expected: FAIL (fields missing).
- [ ] **Step 2:** Library: copy streamed experts host to device on a copy stream into the group buffers, waiting on `expert_ready`. Run the parity case. Expected: PASS.
- [ ] **Step 3:** Engine: the tier staging ring (allocated like `SharedHost`, size from the measured free RAM: if it cannot be had, log one line and plan without `kTierStream`), the per-chunk tier stream list fed by the stagers, the flag raised after the copies, and registration of a pinned RAM copy (on failure, one log line and the stager path).
- [ ] **Step 4: Failure paths.** With `STRATA_TEST_XTIER_DROP=N` during a prompt that streams to the tier: the engine ends with the existing 30 s error, no hang. With a forced ring allocation failure (test hook `STRATA_TEST_XTIER_RING_FAIL=1`): one log line, the prompt answers, KL as without streaming.
- [ ] **Step 5:** Teach the start-up chunk plan the tier's link: `host_plan_chunk`'s H2D bandwidth adds the W6800's probed link when `kTierStream` is possible.
- [ ] **Step 6:** Stint, about 20 minutes: PREV (Task 10) and NEW on RAMHEAVY and NORMAL, plus a RAMHEAVY arm with `--resident-budget-gib 20` (pinned copy) if RAM allows, and the `--prefill auto` arm. KL on NORMAL. Expected: RAMHEAVY faster, NORMAL not slower.
- [ ] **Step 7:** Write the results. Commit `xtier: the W6800 streams experts from RAM during prompts (ABI 3)`, push.

### Task 11c: The tier's streaming in its own file, the ring tested, rings opened late (added after Task 11)

Done (commits `da70a1ab..166d0449`, ABI 5). Review findings of Task 11 fixed before Task 12 split the same code: the
tier's streaming moved out of `prefill.cpp` into its own file, the staging ring's invariant is asserted and tested
(`TierLanding`, 47 checks under ASan, UBSan and TSan), a stage opens its rings at its first chunk that would stream
rather than at start, the x handoff is no longer counted twice in the plan, a record's -1 means not measured, and the
link speed is scaled. No A/B of its own: Task 12's stint, with PREV at `da70a1ab`, covered it. Box ctest 16/16 and a
RAMHEAVY smoke in which the rings opened at the first streamed chunk.

## Stage 3: the overlap

### Task 12: A chunk's layer work runs as two token halves (no overlap yet)

**Files:**
- Modify: `src/prefill/prefill.cpp` (`run_impl` ~2785-5317: the attention half on a token range, per-half MoE buffers, per-half AMD handoff blocks), `include/strata/prefill/prefill.hpp`

**Interfaces:**
- Produces, inside `Prefill::Impl`:
  - a second MoE buffer set (`w`, `ids`, `slot_dev`, `src_dev`, `Xs`, `GU`, `Hh`, `Dm`, `shared`, `sg`, `Xq`, `H`, `Hq`, `row_sd`, the mapped `grp_host`, `bounds_host`) outside the attention region, sized for T/2;
  - a second `PeerPrefill` handoff set (`xx`, `xw`, `xsum`, flags) for the foreign tier;
  - the attention half of a layer on tokens `[t0, t1)`: `p0 + t0` for RoPE and the K/V append, the step records offset by `t0`, PLE rows from `ple_emb + t0 * N`, the HC `normed` flag per half, GDN state updated by half A before half B on the same layer.
- Rules:
  - Halves only when the stage hands MoE rows to another device and `T / 2 >= stream_all_min()`. Otherwise the chunk runs whole, as today.
  - A layer's streamed experts are fetched once for both halves.
  - The KV prefetch's "prefix fixed for the chunk" assumption stays valid: half A's keys are appended before half B's attention reads the prefix.

- [ ] **Step 1:** Implement the halves run one after the other (A's attention, A's MoE, B's attention, B's MoE per layer). Nothing overlaps yet.
- [ ] **Step 2:** Build both trees. Parity tests pass.
- [ ] **Step 3:** Stint: NEW against PREV on NORMAL. Measure the KL against `kl-ref-normal`. Expected: the prompt KL gate, prompts not much slower (this step only reorders). Also the chat prompts, and a 700-word prompt (one chunk below 1,024 tokens), answer as before.
- [ ] **Step 4:** Commit `prefill: a chunk's layers run as two token halves where a stage shares its MoE rows`, push.

### Task 13: The W6800 works on half A while the stage GPU runs half B's attention

**Files:**
- Modify: `src/prefill/prefill.cpp`

**Interfaces:**
- Consumes: Task 12's halves and buffer sets.
- Per layer, in order: A's attention and router, A's grouping and handoff (the tier starts A), B's attention and router on the stage GPU, B's grouping and handoff, the stage's own experts for A and B, A's combine after A's done flag, then B's combine. The next layer starts A's attention after A's combine. The done-flag wait for A is enqueued after B's attention, so it never stalls B. Builder's call: one stream with that order or a second stream with events.

- [ ] **Step 1:** Implement the order. The planner plans each half (its `Load.tier_ms` includes half A's queued work when it plans half B).
- [ ] **Step 2:** Build both trees, parity tests pass.
- [ ] **Step 3:** Cancel check: a request cancelled during a long prompt (the server's stop) leaves the engine answering the next request correctly.
- [ ] **Step 4:** Stint, about 20 minutes: PREV (Task 12 or the last accepted commit) and NEW on NORMAL and RAMHEAVY, the NVIDIA-only NORMAL (unchanged: no halves, `--identity`), the `--prefill auto` arm, the HIP-only smoke run, and the KL. Expected: faster in both setups.
- [ ] **Step 5:** Write the results, including the 3090's `combine` and the W6800's `card idle` per chunk from the timing lines. Commit `prefill: the W6800 computes half A while the stage GPU reads half B`, push.

### Task 13b: The halves' plan sees when each call really starts (added after Task 13)

Done (commits `d37fecb3..286174ef`). In RAMHEAVY the 3090 still waited 383 ms of each chunk for the W6800, and the
W6800 waited 4.9 ms a call for its input: its prompt stream shared a hardware queue with the stream that copied
streamed experts. The library now copies those experts on the refill stream and puts its timing marks on the decode
stream, so its four streams have the HIP runtime's four hardware queues. The plan starts B's call from B's own
measured start, the 30 s timeout line names the wait that gave up, `STRATA_PREFILL_TIMING=1` adds a per-half planned
against measured line, and the W6800's room for copies is split between the stages by their layers. Against
`d7de134f`, two starts each: RAMHEAVY 1,766 and 1,705 tok/s against 1,545 and 1,579 (+11.1%), the 3090's wait 161
against 383 ms a chunk, the W6800's input wait 0.71 against 4.92 ms. NORMAL read at the same speed once the room went
by layers. KL all 0.0039, long 0.0116, the same top token on all 30.

### Task 13c: The W6800's time goes first to the stage that bounds the prompt (added after Task 13b)

Done (commits `6d1b7176..f2526116`). Each stage measures its chunk time per token, from full chunks only. A faster stage
gets half its gap to the slowest as slack, taken low (the slowest's lower of mean and last chunk, its own higher, and
only from a second full chunk), and none on the chunks the slowest waits for. With slack its plan takes W6800 work
back, the experts that save the W6800 the most for the stage's time first. Against `286174ef`, two starts each, A B B
A: RAMHEAVY 1,813 and 1,838 tok/s against 1,750 and 1,774 (+3.6%), NORMAL 2,285 and 2,334 against 2,213 and 2,234
(+3.9%), every start faster than every start of the build before. The 3090 waited 20 against 72 ms of each chunk
(RAMHEAVY) and 21 against 78 (NORMAL). KL all 0.0035, long 0.0103, the same top token on all 30.

## Stage 4: decode trims

### Task 14: The W6800's decode round trip

Measured and dropped (commits `238473b8..c43860ef` keep only the parity test's round trip print behind
`STRATA_TEST_XTIER_ROUND_TRIP=1` and a comment). The host's time from a decode call to its rows, median of 500 calls,
10 and 20 entries with IQ4_XS: 111 and 148 us before. Candidate 1 (one copy of the plan and the rows) 106 and 143 us,
but decode was not faster beyond the spread between engine starts in either setup, so it is dropped (kept as
`~/bench-mixed/t14-c1.patch` on the box). Candidate 2 (the kernels on mapped host memory) 299 to 713 us. Candidate 3
(the quantize fused into the gate/up kernel) 155 and 236 us, and the quantize itself costs only 3 to 4.5 us.

**Files:**
- Modify: `src/xtier/xtier_lib.cu` (`strata_xtier_decode` ~874, `strata_xtier_decode_wait` ~916), `src/core/peer_experts.cpp` (launch/finish), and the decode kernels only for the fusion candidate

Candidates, each built, measured and kept or dropped on its own, in this order:
1. One host-to-device copy of the input rows and the plan (contiguous pinned and device buffers).
2. The quantize kernel reads the input from mapped host memory and the grouped kernel writes the rows to mapped host memory, with no copies.
3. The quantize step fused into the grouped kernel, only if its rounding stays the same.

- [ ] **Step 1:** For each candidate: implement, build both trees, run `xtier_decode_parity` (passes within its tolerance), then a stint against PREV on NORMAL and RAMHEAVY with `--identity` on NORMAL. Keep it only if decode is faster beyond noise in one setup and not slower in the other, and the answers are identical.
- [ ] **Step 2:** One commit per kept candidate, `xtier decode: <what>`, with the numbers. Push. Write dropped candidates and their numbers into `plan-results.md`.

## Wrap-up

### Task 15: Final measurement, docs and the service

Split into Task 15a (the deferred code findings) and Task 15b (the final measurement, the docs and the service
proposal), below.

**Files:**
- Modify: `docs/MULTI_GPU.md` (the NVIDIA and AMD section: the planner, the overlap, CPU placement, `--expert-gib`, Q6_K), `docs/DETAILS.md` (removed switches), `docs/AMD_HIP.md` if the library build changed

- [ ] **Step 1:** Final stint: the start commit (`70695236`'s tree, built in `strata-prev`) against the final head on NORMAL, RAMHEAVY and NVIDIA-only, plus `--prefill auto` arms (the chunk auto picks) and the Q6_K smoke run. KL on NORMAL.
- [ ] **Step 2:** Docs: say what each change does, measured on what, with the numbers. Remove every removed switch from the docs.
- [ ] **Step 3:** Propose the service config for the final head (the IQ4_XS mix config, no hand tuning) to the user with the numbers. Do not change the service without their yes.
- [ ] **Step 4:** Commit `docs: the mixed tier after the optimizations, measured`, push.

### Task 15a: The deferred code findings (split from Task 15)

Done (commits `f7677fe3..5a50e6fe`). The 30 s timeout line names the call the card had queued last. The tier planner's
engine class moved to its own file (`prefill.cpp` from 6,513 to 5,986 lines). The shed prices each move by its change to
the layer's tallies instead of placing the whole layer again: 0.031 against 1.0 to 1.1 ms a layer with halves at 512
experts (a dev machine), and the same plans as before on 3,000 random layers of distinct rows. Measured and not kept: a
cap for prompts of one or two minimum chunks (not slower without it) and halves for chunks the planner does not plan yet
(no first-prompt gain at three starts a build). The test hooks `STRATA_TEST_TIER_MOVE`, `STRATA_TEST_TIER_STREAM` and
`STRATA_TEST_NO_HALVES` are gone. Against `f2526116` it read at the same speed in NORMAL and RAMHEAVY. KL all 0.0040,
long 0.0119, the same top token on all 30.

### Task 15b: The final measurement, the parked checks, the docs and the service proposal (split from Task 15)

Done (commits `1459fe6f`, `a9c145fd` and the docs commit). The prompt parity test's start once read -0.010 ms where it
reads about 1.1 ms: the library timed it from a mark on the decode stream, which the card can run late, so it now counts
from the earlier of that mark and the prompt stream's own (a test holding the decode stream read -4.1 to -5.9 ms before,
10.4 to 10.5 after, 20 runs in a row passed, neutral in NORMAL and RAMHEAVY). The planner test gives whole chunks the
engine-like routing too and asserts that the shed acts. Against the plan's start (`70695236`; `9997e38e` for RAMHEAVY,
the first build with `--expert-gib`), two starts each, a 20,053-token prompt: NORMAL 2,296 against 1,477 tok/s (+55%),
RAMHEAVY 1,768 against 749 (2.36 times), NVIDIA-only 1,590 against 827 (+92%), `--prefill auto` 2,190 against 1,461
(+50%, one start each), decode not slower in any. RAMHEAVY's chats were slower: prompt 51.5 and 52.1 against 69.9 and
58.0 tok/s a start, with more drive reads (see Task 15d). KL all 0.0046, long 0.0138. Also measured: RAMHEAVY's chats no
slower than `d7de134f` at like drive reads; decode spares at two starts (RAMHEAVY chats +14.6%, NORMAL parked nothing);
Q6_K's own KL with its K-quant prompt kernels (0.0134 on 20 short prompts), Q6_K drive-bound at 133 to 198 tok/s on an
8,450-token prompt and not slower than `286174ef`, so the slack's take stays unbounded. The W6800's streaming made
NORMAL 2.9% slower and RAMHEAVY 18.6% faster (Task 15c). The service proposal is the live config with the new build's
paths, not applied.

### Task 15c: NORMAL's loss with the W6800's streaming (added during Task 15b)

Measured, no code. Task 15b's Step 3b found NORMAL's 20,053-token prompt 2.9% slower with the W6800 streaming RAM experts
and RAMHEAVY's 18.6% faster. Two rounds of throwaway builds logged every W6800 call. The planner's price is right in the
median (the 3090's calls within 0.2 ms of their fit), and the loss is not the W6800's clock (held high, still 2.7%). Most
of the tail is the 3090's prompt thread being preempted while it queues a call's kernels (79 to 83% of its time in those
calls, involuntary switches in most), with and without streaming alike, so it does not explain the gap between them.
The gap lines up with streaming's extra host time in the 3090's calls (737 and 809 ms over two prompts against 608) and a
few late stagings. Also found: the 3090's first call of each chunk runs about 9 ms over its fit either way, not
explained. Streaming stays, and the three are open paths in `docs/MULTI_GPU.md`.

---

## Phase 2: Swift 1.5 Q6_K on one RTX 3090 + W6800 box (Task 16 onward)

**Goal:** The most prompt and decode speed for Swift 1.5 Q6_K on the box as it is now (RTX 3090 at 1500 MHz and 350 W,
W6800, 188 GiB RAM), as engine defaults under the user's flags, with no loss of quality. First reach RE24 (the copy plus
one chunk, measured 20K 2,528 tok/s) by default with a fast start, then go past it.

**Architecture:** Measurement first (a bench kit that checks start facts, KL gate v2, loud lines and size checks). Then
the prompt levers as one box plan at load (the chunk, the W6800's room), then one pinned RAM copy filled from the page cache
and made the default, then the planner. Decode follows, ordered by the bound with the copy (the CPU pool): diagnostics, a
decode KL with the lanes that cut CPU work, old-box helpers off, CPU prefetch, config sweeps, then the gated swap work and
the W6800's round trip. The final A/B, the docs and a service proposal close it.

**Spec:** `docs/superpowers/specs/2026-10-27-q6k-one-box-design.md` (it wins over Phase 1's spec for Task 16 onward).

Number marks used below: "Task 16 (250 W)" and "Task 16 (350 W)" are Part 1 and Part 2 of `task-16-report.md`. Part 2
holds runs A and B, probes 7 to 10 (run C) and run D (RE24, RE24AS). Most are one start, so they are leads. "Read from
the code" and "estimate" are not measurements.

### Phase 2 Global Constraints (they supersede Phase 1's where they differ)

- Hard fork: breaking changes are fine. A better behavior becomes the default, and switches it makes redundant go.
- Tuning to this box is allowed. Prefer a run-time read (free VRAM, MemAvailable, RLIMIT_MEMLOCK, page residency, probed
  links, measured costs) when it is as fast. Other hardware, models and platforms need not stay fast. CUDA and HIP must
  build for every change.
- **Keep rule:** a change stays only if Q6_K's prompt or decode speed on this box is faster beyond the spread between engine
  starts, and the other is not slower. The verdict is `tools/box_bench.py verdict` (from Task 17): KEEP when min NEW is
  above max PREV on the target and the mean gap is at least the larger arm's spread. Not slower: the other metric's NEW
  mean is at least PREV's mean minus the smaller of PREV's spread and 5% of PREV's mean. A target whose expected gain is
  under the protocol spread runs at least three starts per arm. The controller may accept such a task on deterministic
  timing counters, with the A/B as the not-slower check.
- **Defaults:** a task that makes a default is KEEP against `cfg-q6k2` (the user's flags) on PREV, and not slower than
  `cfg-q6k2-re24` (RE24 as flags) on PREV.
- **Correctness:** a change that only moves data or threads keeps greedy answers identical (identity kinds below). A change
  to the math passes KL gate v2 on two starts whenever the W6800 is in prompts. A decode-math change passes the decode KL
  (Task 27).
- The user's calls stay: the 1500 MHz lock and 350 W limit, `--max-context 262144`, `--kv int8`, the user's MTP flags and
  `--pool-workers 15`, the W6800's power state, the service (stopped, unit untouched, a config proposed at the end).
- Prompts of about 2K, 8K and 20K tokens only. No 64K prompts.
- Code comments: one line, plain, no em dashes, no semicolons, no hyphens joining clauses.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push `mixed-tier-perf` after each task, never
  force-push, never commit `.superpowers/`. Never edit `~/source/strata` on the box.
- Every gain claimed is a measurement on the box, written with the setup, the commit and every start's numbers.

### Phase 2 Review Focus

- The NVIDIA-only engine and the HIP-only engine must build and start. The multi-stage tier path must build and pass its
  unit tests (`tier_planner_test`, `split_lend_complement_test`, the halves tests) in both trees. This box has one NVIDIA
  card, so it cannot start the multi-stage path. Checked in Task 35.
- Short prompts (a chat turn, a prompt under 1,024 tokens) stay correct and not slower. Every A/B has the chats and the 2K.
- The first prompt after a start must not be slower. Every A/B's first measured pass shows it.
- A W6800 that stops answering ends the engine cleanly after the 30 s wait, as today.
- A copy that cannot be pinned or registered, or a W6800 room that cannot be opened, falls back to today's behavior with
  one log line.
- The 3090 keeps at least 256 MiB of VRAM free after its first one-chunk prompt (the engine's own `LOW` line).

### Phase 2 box protocol (every task that builds or measures uses this)

- **Access:** `ssh vdh@192.168.10.5`. Bench files in `~/bench-mixed/`. Write scripts locally and copy them with `scp`
  (ssh heredocs strip single quotes). Launch long jobs detached and poll with bounded checks.
- **Prefixes:** stints are named with a separator that cannot collide: `t21-s1`, `t21-s2` for Task 21, `t21b-s1` for 21b,
  `t23-s*`, `t23b-s*`. Run `ls ~/bench-mixed/tNN-* ~/bench-mixed/tNNb-*` before each run.
- **Lock, power, clock:** every build, ctest and measuring script runs under `flock ~/bench-mixed/.box.lock` for its whole
  length. Every run script runs and checks `sudo nvidia-smi -i 0 -pl 350` and `sudo nvidia-smi -i 0 -lgc 0,1500` before
  it starts, and logs both. The 3090 never runs unlocked. `tools/box_ab.sh` is sourced inside a script that is itself run
  under the lock.
- **Build NEW:** push, then
  `flock ~/bench-mixed/.box.lock bash -c 'cd ~/source/strata-base && git fetch -q origin mixed-tier-perf && git checkout -q --detach FETCH_HEAD && nice -n 10 cmake --build build-cuda -j10 && nice -n 10 cmake --build build-hip --target strata_xtier_hip strata -j10'`.
- **Build PREV:** the same command in `~/source/strata-prev` at the previous accepted commit, only once Task 16b no longer
  runs from it. Until then PREV is `ad9157b4`, which it holds.
- **ctest set:**
  `flock ~/bench-mixed/.box.lock bash -c 'cd ~/source/strata-base && CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 STRATA_XTIER_LIB=$PWD/build-hip/libstrata_xtier_hip.so LD_LIBRARY_PATH=/opt/rocm/lib ctest --test-dir build-cuda -R "prefill|xtier|peer|prompt|planner|tier|half" --output-on-failure'`,
  plus each task's own tests named explicitly.
- **Configs:** `cfg-q6k2.json` (the user's flags, chunk 4096, `--conversation-cache-mib 0`), `cfg-q6k2-p24576.json`
  (K24/K24R's chunk), `cfg-q6k2-re24.json` (RE24 as flags: `--resident-experts --pcie-frac 0 --prefill 24576
  --peer-reserve-mib 1450`), `cfg-q6k3.json` (Task 21: `--prefill auto`). Each has a `-prev` twin whose exe and
  `STRATA_XTIER_LIB` point at strata-prev (the server applies the config's env after the process env, and the library ABI
  can differ). Task 17 creates the `-prev` twins. Every PREV arm uses one.
- **Arms:** `tools/box_bench.py` (Task 17) with `--expect` start facts. Until Task 17 lands, `~/bench-mixed/t16-bench.py`.
  Passes: `--pass warm:chat0,p8k --pass m:chat0,chat1,chat2,p2k,p8k,p20k --pass m:chat0,chat1,chat2,p2k,p8k,p20k`.
  Keep arms run with timing lines off. Decode arms run at the default spares unless Task 26 rules otherwise.
- **Starts:** Task 16b Step 1's protocol. Until then A B B A, two starts per arm, three when the expected gain is under
  the spread.
- **Copy arms:** every arm with a copy, and every arm from Task 23 on, runs under `prlimit --memlock=unlimited` (bench
  shells have RLIMIT_MEMLOCK 23.6 GiB) and passes `--expect pinned_gib=95..100`. After a copy arm, rebuild the page cache
  (`cat` the five shards, or 16b's prep) before an arm without a copy. Until Task 22 lands, each copy start costs about 2
  minutes and 120 to 140 GB of drive reads, so keep multi-start copy measurements to the ones a keep decision needs.
- **Drive reads:** compare `the OS read N MB from storage` on measured passes only, against PREV's same request (the warm
  8K reads 188 to 743 MB in any arm, cause not found).
- **Identity kinds:**
  - prompt: `quant_quality.py prompt` dumps with `STRATA_PF_PEER_HOST=0` in both arms, KL exactly 0 on every prompt;
  - decode: both arms `xargs=--adapt-swaps;0;--pcie-frac;0;--prompt-cache;0` and `STRATA_PF_PEER_HOST=0`, `--identity
    --warm 0 --measure 1 --full-text`;
  - adaptation on: per Task 17's G-ID result (PREV against PREV repeats, or else `STRATA_ADAPT_CHECKRES=1` and
    `STRATA_XTIER_CHECK=1` with zero mismatches plus decode identity).
- **KL:** gate v2 from Task 18 (`quant_quality.py gate`), two starts whenever the W6800 is in prompts. Until the ruling,
  `kl-ref-q6k` and the v1 set as well.
- **Results:** append to `~/bench-mixed/plan-results.md` under "Phase 2, Task N": date and time, commits, configs, power
  and clock, every start's numbers, the verdict, the KL.
- **Box end state:** service stopped, the 3090 at 350 W and locked at 1500 MHz, no engine running.

---

### Task 16: Q6_K on the new box: bring-up, baseline and where the time goes (done)

Done 2026-10-27 (impl-task16, no engine code, no commits). Report: `task-16-report.md`. Box files `t16-*` and `t16p-*`,
configs `cfg-q6k2*.json`, KL directories `kl-ref-q6k`, `kl-t16-ref2`, `kl-t16-head`, `kl-t16-c16g`.

- Built `ad9157b4` in strata-base and strata-prev. The ctest set passed (the three xtier tests need `STRATA_XTIER_LIB`).
- Baseline at 350 W, chunk 4096 (B3, B4): 2K 360 to 387, 8K 719 to 754, 20K 746 to 785 tok/s, chats 24.2 to 32.2.
- Prompt bound: each chunk streams every RAM expert once through host staging (RAM bandwidth). The W6800 had no room for
  its ring. Decode bound without the copy: the CPU pool's experts and the adaptive join. With the copy: the CPU pool.
- Levers as configs: the ring room (G305, +37% on 20K, two starts, 250 W), one chunk for the 20K prompt (K24 1,715 and
  K24R 1,776, one start each), the pinned copy (RE 2K 619, 8K 1,200, 20K 1,221, chats 42.0, 32.7, 30.8, one start, ready
  in 111 s), RE24 (2K 590, 8K 1,766, 20K 2,528, chats 44.5, 43.1, 36.0, ready in 130 s, one start), `--adapt-swaps 16`
  (chats +26 to +29% without the copy, two starts, 250 W; no better with the copy, RE24AS, one start).
- Probes 7 to 10 (run C): the tail skip, look-ahead off and verify width as one-start leads.
- KL reference `kl-ref-q6k` repeats exactly. The head passes with little room (long mean 0.0222, long top-1 90.0%).

### Task 16b: the measurement protocol, the chunk's KL, the bound at RE24 (running)

Brief: `task-16b-brief.md`, as revised by the controller after Task 16 Part 2: RE24 on `cfg-q6k2.json` is the base config.
Runs from `~/source/strata-prev` (`ad9157b4`) under the box lock. No code, no commits. Its first run `t16b-r1` is four RE24
starts.

- Step 1: the spread of four RE24 starts, and starts per arm for a 5% and a 10% gain. **Gates Task 17.**
- Step 2: KL of RE24 on two starts, RE at 4096, and as factor arms RE24 with `STRATA_PF_PEER_HOST=0` (the tier out of
  prompts) and with `STRATA_MMQ_KQ=0` (the chunk alone; at reserve 1450 no ring opens, so a ring-fail arm would equal RE24),
  C16G again if time allows; the factor; a tool for decode-side quality. **Gates Tasks 18, 21 and 27.**
- Step 3: the bound at RE24 (the 3090's per-layer timeline, the 96-slot VRAM ring's lead, the W6800's busy share) and one
  RE24 start with `--peer-reserve-mib 2200`. **Gates Tasks 23 (the ring), 24, 25.**
- Step 4: RAM and start at RE24, one `--resident-budget-gib 60` start at 24576. That budget pins the W6800's experts first
  (read from the code, `expert_source.cpp:3546-3556`), so it is a RAM-short data point. **Read in Tasks 22, 23.**
- Step 5: on RE24, the tail skip and look-ahead off (two starts each), `--kv-resident 32768`, `STRATA_MMQ_DIRECT=1`, the
  131072 slot count. **Gates Tasks 27, 28, 30.**

---

### Phase 2 execution order (controller, 2026-10-27, revised for speed)

Measurement found the levers. The engine work that makes them defaults now starts at once, in three parallel streams,
while Tasks 17, 18 and 16b finish their box runs. Each stream has its own local branch and its own box tree
(`~/source/strata-p`, `strata-c`, `strata-d`, configured like `strata-base`). It rebases on `origin/mixed-tier-perf`
before each push. Every box build and run takes `flock ~/bench-mixed/.box.lock`. PREV is `ad9157b4` in
`~/source/strata-prev` until a stream's own base moves.

- **Stream P (prompts):** Task 20, then Task 21. Task 19 is folded in, keeping only what the one-pass chunk needs: the
  size audit with T = 24576 tests, the halved-chunk line, both cards' VRAM lines after the first prompt and the first
  full chunk. The refill line goes to Task 23b and the FP16 counter to Task 21b, if they run. Task 21b follows if the KL
  needs it.
- **Stream C (the copy):** Task 22, then Task 23. Until Stream P lands, Task 23 measures with the chunk as flags
  (`--prefill 24576 --peer-reserve-mib 1450`), not `cfg-q6k3`.
- **Stream D (decode):** Task 28 first. Its code starts now, and its keep waits for 16b Step 5's look-ahead arm. Then
  Task 26, then Task 27 once Task 18's gate and the decode KL tool exist.
- After the streams: Tasks 24, 25 and 25b by their gates, then the new speed tasks 25c and 29b, then the rest of decode,
  then Tasks 35 and 36. Task 25's proposals for the 3090's own prompt work become numbered tasks when it reports.

## Stage 0: measurement

Tasks 17 and 18 change only `tools/` and box configs, never the engine, so they may run at the same time (separate
worktrees, `git pull --rebase` before each push). Their box runs share the lock and the `ad9157b4` head. Task 19 follows both.

### Task 17: The bench kit: Task 16's box scripts in the repo, start facts checked, a keep verdict, -prev configs, and identity with adaptation on

**Why (measured, Task 16):** the 20K prompt read 497 to 673 tok/s over eleven starts of one build. Arms silently measured
another config than their name: C16 lost the tier (`reads its prompts without the AMD tier`, 20K at 794), and `--prefill auto`
picked 4,352 with a ring that never opened. RE24 left the 3090 at 231 MiB free (`LOW`) and later at 93 MiB, which no arm
checked. Identity with adaptation on was never measured (every identity arm used `--adapt-swaps 0`), so no swap-path change
can yet be called data only. No `-prev` config exists for `cfg-q6k2*` (only `cfg-q6k-prev.json`).

**Files:**
- Create: `tools/box_bench.py` (from `~/bench-mixed/t16-bench.py`), `tools/engine_log.py` (from `t16-parse.py` and
  `t16-klper.py`), `tools/box_ab.sh` (from `t16p-common.sh`), `tools/test_engine_log.py`, `tools/test_box_bench.py`,
  `tools/fixtures/engine-log-{k16,c16,g305,rb60,pauto,k24r,re,re24}.txt` (short excerpts of `t16p-a-K16`, `t16-s4-P16384`,
  `t16-s7-G305`, `t16-s7-RB60`, `t16-s4-Pauto`, `t16p-a-K24R`, `t16p-b-RE` and `t16p-d-RE24` engine logs, copied read only)
- Modify: `tools/ab_engine.py` (identity presets only), `docs/MULTI_GPU.md` (a short "how this box is measured" paragraph)
- Box only: `cfg-q6k2-re24.json`, and `-prev` twins of `cfg-q6k2`, `cfg-q6k2-p24576` and `cfg-q6k2-re24`

**Interfaces:**
```text
python tools/engine_log.py facts ENGINE_LOG        # JSON of one start's facts, from the last "engine started"
python tools/engine_log.py rows ENGINE_LOG         # the requests as rows
python tools/engine_log.py decode ENGINE_LOG       # the decode timing rows
python tools/box_bench.py ROOT CONFIG --log OUT.jsonl --pass NAME:req,req [--pass ...] [--sample] [--keep-logs PREFIX]
       [--expect chunk=24576,tier_in_prompts=1,ring_slots=128,w6800_experts=7100..7300,pinned_gib=95..100] ARM...
python tools/box_bench.py verdict OUT.jsonl --a NAME --b NAME --target p20k.prompt [--other chat1.decode]
source tools/box_ab.sh   # power, at TREE SHA, prep keep|rebuild, abba ARM_A ARM_B N (sourced in a script run under the lock)
```
Facts: `chunk`, `loan_slots`, `w6800_experts`, `ring_slots` (0 on "too little for a ring"), `tier_in_prompts`,
`pinned_gib`, `pinned_s`, `vram_free_mib` (the 3090's `VRAM free with everything loaded`), `vram_low` (the per-request
`only N MiB` warnings), `cuda_slots`, `cached_share`. Per request: drive MB, Cached, MemAvailable. Per start: the sampler's
clock and power summary. `--expect` mismatch exits 4. `verdict` implements the keep rule exactly and exits 0 on KEEP.

- [ ] **Step 1:** `ls ~/bench-mixed/t17-*`. Read the four scripts with `ssh cat` (read only) and copy them into `tools/` with
  box paths turned into arguments.
- [ ] **Step 2: Write the failing tests** in `tools/test_engine_log.py`. The fixtures must give: K16 chunk 16384, loan 1600,
  W6800 7472, ring 0. C16 at the default reserve: `tier_in_prompts` false. G305: ring 128. RB60: pinned 60.00 GiB in 56 s.
  Pauto: chunk 4352. K24R: loan 2329, W6800 7188, ring 128. RE: pinned 90.66 GiB in 92 s. RE24: pinned 98.17 GiB in 109 s,
  `vram_free_mib` 231, a `vram_low` of 93. Run `python -m pytest tools/test_engine_log.py`. Expected: FAIL (no module).
- [ ] **Step 3:** Implement `facts`, `rows` and `decode`. Expected: PASS.
- [ ] **Step 4:** Port `box_bench.py`, add `--expect`, the covariates and `verdict`. Test them in `tools/test_box_bench.py`
  with a fake server in the style of `tools/test_ab_engine.py`. The verdict tests: PREV 100 and 110 against NEW 111 and 112
  gives NOT RESOLVED (min above max, gap 6.5 under the spread 10); a clear gap gives KEEP; an other metric 6% lower with a
  2% PREV spread gives not KEEP. Expected: PASS, and `tools/test_ab_engine.py` still passes.
- [ ] **Step 5:** Write `box_ab.sh`: power and clock set and checked, the tree check, prep, A B B A and A B A B B A orders,
  and a check that it runs under the lock (`flock -n` on the lock fd fails).
- [ ] **Step 6 (gate 16b Step 1):** set the defaults from 16b's report: prep mode, warm passes, starts per arm. If 16b has no
  protocol under 10%, the default is 3 starts per arm. Write the `-prev` configs and `cfg-q6k2-re24.json` on the box.
- [ ] **Step 7 (box, under the lock):** two protocol starts of the head at `cfg-q6k2-re24` with `--expect chunk=24576,
  pinned_gib=95..100`. One start with a wrong `--expect chunk=8192`: exit 4 expected. Then G-ID: the adaptation-on
  identity, PREV against PREV twice (both `ad9157b4`, `cfg-q6k2-prev`), with `STRATA_PF_PEER_HOST=0`,
  `xargs=--pcie-frac;0;--prompt-cache;0`, adaptation at its default, `--identity --warm 0 --measure 1 --full-text`. Write
  G-ID: repeats or not.
- [ ] **Step 8:** Append to `plan-results.md`. Commit and push.

**Box stint:** about 30 min. **Expected:** the RE24 starts inside 16b's spread. G-ID unknown.

**Keep rule:** tooling, no speed claim. Accepted when every fixture fact matches, the verdict tests pass, the head's starts
land inside 16b's spread, and G-ID is written down.

**Commit:** `tools: the box bench kit, start facts checked before an arm counts, A B B A with a keep verdict`

### Task 18: KL gate v2: the served tier state, 30 long prompts, seven xl prompts, per-prompt entropy, two starts

**Why:** C16G failed on two long prompts (Task 16 per prompt, KL(ref||head) / KL(ref||C16G)): prompt 0, 0.0201 / 0.5746,
and 4500, 0.0489 / 0.1293. Computed from the dumps: prompt 0 ends inside the hash `4e1b9c2f7a0d` (reference entropy 3.18
nats). The head's 90% long top-1 is that digit. Long 0 always runs unplanned and long 1 opens the ring, so only 8 of 10
long prompts run the served state. The long prompts are 4,600 to 4,960 tokens, rotations of one 5,000-token text, each
scored at one position (`quant_quality.py:198-211`). No KL prompt reaches the one-chunk 20K geometry Task 21 makes the
default, nor the 20,972 to 24,576 zone where a product of T x 10 x 2560 x 4 bytes passes 2^31. No KL ran on RE24.

**Files:**
- Modify: `tools/quant_quality.py` (`cmd_ids` :63-84 with the segments read from the set directory or a `--segments` flag,
  and whole texts kept when `--seg-tokens 0`; `LONG_LENS` and `prompt_set` :194-211, `cmd_prompt` :240-283, `prompt_keys`
  :288-291, `cmd_pcompare` :294-324, argparse :404-440, a new `cmd_gate`)
- Modify: `tools/test_quant_quality.py`
- Create: `data/quant_quality/xl/docs.txt` (a frozen copy of `docs/DETAILS.md` at `ad9157b4`), `data/quant_quality/xl/code.txt`
  (a frozen copy of `src/prefill/prefill.cpp` at `ad9157b4`), `data/quant_quality/xl/set.ids.json` (made on the box)
- Modify: `docs/DETAILS.md` (one sentence on gate v2 beside the quant_quality paragraph)

**Interfaces:**
```text
quant_quality.py ids --set DIR [--segments a,b] [--seg-tokens N|0]
quant_quality.py prompt --config CFG --out DIR --keep-caches [--set v1|v2] [--no-warm] [--env K=V]
quant_quality.py pcompare REF RUN [--per-prompt]
quant_quality.py gate REF RUN1 [RUN2] [--xl-ref REFXL]   # exit 0 only if every run passes
```
- `--set v2`: 20 short prompts as v1. 30 long: prompt j of 30 starts at `j*len(joined)//30` with `4600 + 40*(j//3) +
  13*(j%3)` tokens, so every third one is exactly one of v1's 10 (same start and length) and old directories still compare.
  Seven xl prompts: three per xl text at 20,000, 20,160 and 20,320 tokens at spread starts, and one of 24,300 tokens.
- With v2, one unscored warm request (4,800 tokens of the xl docs text, `"set": "warm"` in `index.json`) runs first.
  `--no-warm` turns it off. `pcompare` skips set `warm`.
- `pcompare --per-prompt`: set, start, tokens, KL, top-1 match, reference entropy (nats), reference top-1 probability.
  It also prints a line for the v1 keys alone.
- `gate`: per run, all (short and long) mean at or below 0.03, median at or below 0.005, top-1 at or above 90%; long mean at
  or below 0.03, top-1 at or above 90%; xl mean at or below 0.03, no top-1 miss where the reference's top-1 probability is
  0.5 or more, and no non-finite logit. No position is dropped.

- [ ] **Step 1 (gate 16b Step 2):** read its result. Record the factor it names, RE24's two-start KL, and whether long 0 failed
  again.
- [ ] **Step 2: Write the failing tests:** `ids` on a directory with only `docs.txt` and `code.txt` and `--seg-tokens 0` keeps
  both whole. `prompt_set` v2 gives 20 short, 30 long and 7 xl prompts with distinct starts, and v1's 10 (start, tokens)
  pairs are among the long ones. `pcompare` on two synthetic directories (3 short, 3 long, 1 xl, 1 warm float32 logit files,
  hand-computed KL and entropy) prints the expected rows and skips warm. `gate` fails a run with one set past a line. Run
  `python tools/test_quant_quality.py`. Expected: FAIL.
- [ ] **Step 3:** Implement. Expected: PASS.
- [ ] **Step 4 (box):** check that `quant_quality.py ids` with the pack's tokenizer reproduces v1's `set.ids.json`, then make
  `xl/set.ids.json` (both texts must hold at least 24,300 tokens). Copy it back and commit it with the two texts.
- [ ] **Step 5 (box, lock, power and clock):** the reference `kl-ref-q6k-v2` on `cfg-q6k2.json` with `--keep-caches --env
  STRATA_MMQ_KQ=0` (chunk 4096, the W6800 out of prompts), twice (the second to `kl-t18-ref2`). Expected: KL 0 and top-1
  100% between them.
- [ ] **Step 6:** the head (`cfg-q6k2.json`) on two starts (`kl-t18-head-a`, `-b`), C16G once (`kl-t18-c16g`), K24R as a
  config once (`cfg-q6k2-p24576.json` plus `--peer-reserve-mib 2200`, `kl-t18-k24r`), RE24 once (`cfg-q6k2-re24.json`,
  `kl-t18-re24`, under `prlimit --memlock=unlimited`). `pcompare --per-prompt` each against v2, head-a against head-b (the
  tier's own start-to-start KL spread), and the v1 keys against `kl-ref-q6k`.
- [ ] **Step 7:** send the controller a ruling request: v2 as the gate, with the head's two-start results, RE24's table, the
  per-prompt rows whose reference entropy is above 2.5 nats, whether the head-a against head-b spread leaves room for Task
  21, and a proposal for the long top-1 line relative to the head (for example the candidate's long misses at most the
  head's plus one), since the long prompts share one text.
- [ ] **Step 8:** Append to `plan-results.md`. Commit and push.

**Box stint:** about 60 to 75 min (seven runs, one a copy start). **Expected:** the reference repeats with KL 0. The head's
long top-1 unknown (the hash position is 1 of 30 now, but other hard positions may exist). xl unknown.

**Keep rule:** tooling and a ruling. Accepted when the v2 reference repeats exactly and the controller has ruled on v2.

**Commit:** `quant_quality: gate v2 with a warm request, 30 long prompts, seven xl prompts up to 24300 tokens, per-prompt entropy and a two-start gate`

### Task 19: Lines and limits the one-pass chunk leans on

**Why:** some fallbacks are silent or unmeasured.
- A fixed chunk the 3090 cannot lend is halved with no line (`generate.cpp:8728-8731`, read from the code).
- The refill of lent slots counts in prompt time and is printed only under `STRATA_TRACE`. No box log has the line. K24R's
  prompts took 0.48, 1.07 and 2.15 s more than the prompt path's wall at 2K, 8K and 20K (measured), about 1.1 to 1.5 ms per
  lent slot.
- The W6800 takes 25 to 32 MiB more than `tier_prompt_bytes` counts (measured: 602 MiB minus 255 predicts 347 free at 4096,
  and 322 was measured). The link probe's kept 64 MiB buffer is one candidate (read from the code, `xtier_lib.cu:297-305`).
- The 3090 fell to 231 MiB free with everything loaded at RE24 and to 93 MiB at a later request's start (measured). The
  start line exists (`generate.cpp:11244-11262`, `LOW` under 256 MiB), but nothing ties it to the chunk.
- Nothing counts FP16 sums past 65504 (`set_f16_range` is HIP-engine only, read from the code).
- Chunks above 16384 ran with 20K prompts only (20,053 tokens). No product of T was audited at T = 24576. An example: FP32
  rows at 24576 x 10 x 2560 x 4 B are 2.5e9 bytes, past 2^31 from T = 20,972.

**Files:**
- Modify: `src/program/generate.cpp` (the fixed scan :8728-8731, the refill :13504-13567, start lines near :6838 and
  :9081, the 3090's free VRAM after the first one-chunk prompt), `src/xtier/xtier_lib.cu` (`prompt_grow` :341-389,
  `sums_to_f16` :712), `src/prefill/moe_mmq.cu`, `src/prefill/kernels.cu`, `src/prefill/prefill.cpp` (`bytes_needed_impl`
  :2746-2799; widenings only where needed)
- Modify: `include/strata/program/prompt_buffers.hpp`, `tests/program/prompt_buffers_test.cpp`, the
  `prefill_mmq_kquant_test` source, `src/prefill/xtier_prefill_parity.cpp`, `docs/MULTI_GPU.md` (the new lines)

**Interfaces:**
```cpp
// include/strata/program/prompt_buffers.hpp
/// The line a fixed chunk prints when the card cannot lend it and the scan takes a smaller one.
std::string halved_chunk_note(int64_t asked, int64_t need_slots, int64_t have_slots, int64_t taken);
```
- Lines: `strata serve: --prefill N needs M CUDA0 slots of S, so prompts read in chunks of K`; under `STRATA_PREFILL_TIMING=1`
  `strata prefill timing: refilled N lent slots in X ms (Y GB/s)`; at start `the AMD card has A MiB of VRAM free after its
  experts, B after its prompt buffers, C after its ring, P after its link probe`; after the first prompt that used the tier
  `the AMD card has D MiB of VRAM free after its first prompt`; after the first prompt of a full chunk `CUDA0 has E MiB of
  VRAM free after its first N-token chunk`.
- Library env `STRATA_XTIER_F16_CHECK=1`: a device counter of sums past 65504, printed with `strata xtier prompt timing`.
  Under `STRATA_XTIER_TIMING=1`, `hipMemGetInfo` around each of `prompt_grow`'s allocations. No ABI change.

- [ ] **Step 1: Write the failing tests:** `prompt_buffers_test` checks the halving note. `prefill_mmq_kquant_test` gets a
  Q6_K/Q8_0 case at T = 24576 (10 routes) against the float reference with the existing tolerance and no non-finite value,
  and the same at 32768 as a headroom check. `xtier_prefill_parity` gets a T = 24576 case. Run
  `ctest -R 'prompt_buffers|mmq_kquant|xtier_prefill'`. Expected: FAIL, or a crash where a size overflows.
- [ ] **Step 2:** The audit: list each product of T, top-k and a width (2560, 1280, 640) in elements and bytes, with
  file:line, in MoE MMQ, the gather, scatter and combine kernels, the AMD tier's prompt kernels, `tier_prompt.hpp` and
  `bytes_needed_impl`. Widen to int64 only where a product can pass 2^31 at 24576. Record the 32768 headroom in the report.
  Expected: PASS in both trees.
- [ ] **Step 3:** Add the lines and the F16 counter.
- [ ] **Step 4:** Build both trees under the lock. Run the ctest set and the three tests by name.
- [ ] **Step 5 (box, `t19-s1`):** decode identity pair, NEW against PREV at 4096. Prompt identity at chunk 24576 (K24R as
  config, `cfg-q6k2-p24576` and its `-prev` twin with `--peer-reserve-mib 2200`, `STRATA_PF_PEER_HOST=0`, KL 0). One
  diagnostic start each of NEW and PREV with `STRATA_PREFILL_TIMING=1 STRATA_XTIER_TIMING=1` (NEW also
  `STRATA_XTIER_F16_CHECK=1`) at K24R: 2K, 8K, 20K and one 24,300-token request (no non-finite value, the F16 count). Compare
  the kernel groups' ms per chunk (GEMMs, gather, combine) NEW against PREV.
- [ ] **Step 6:** Append to `plan-results.md` (the refill ms, the saturation count, both cards' VRAM after their first
  prompt, where the 25 to 32 MiB go). Commit and push.

**Box stint:** about 35 min. **Expected:** refill after the 20K at 24576 about 2.15 s (K24R). Saturation count unknown,
hoped 0. No error at 24,300 tokens.

**Keep rule:** lines and limits. Kept when both identities hold, the ctest set passes in both trees, and no kernel group's ms
per chunk rises beyond its own start-to-start spread. A widening that slows a kernel is limited to T over 16384.

**Commit:** `prefill, xtier: loud chunk fallbacks, the refill time, both cards' VRAM after each step, an FP16 sum check, and sizes checked to 24576 tokens`

---

## Stage 1: prompts, one box plan at load

Tasks 20 and 21 are two reviewable units and one box plan. Task 20 lands with start-log checks only. Task 21 carries the
combined A/B and the KL for both. If Task 21 waits on its KL or on Task 21b, Task 22 may run next (it depends on the chunk
only through the lend region's size).

### Task 20: The W6800 opens its prompt buffers, and its ring when the box plan asks, before its experts fill

**Why (measured):** the ring room is the lever G305 measured (+37% on the 20K at chunk 4096, two starts, 250 W). Today's
fill takes free VRAM minus `--peer-reserve-mib 600` before any prompt buffer exists, so the ring never opens and a chunk of
16384 drops the tier (Task 16). Read from the code: `open_foreign` fills at `peer_experts.cpp:276-294` and
`generate.cpp:6838-6840`, the link probe follows at `:6866` and keeps a 64 MiB card buffer, the buffers come later in
`Prefill::set_peer` (`prefill.cpp:2416-2419`), the card ring later still in `TierStream::open_rings`
(`tier_stream.cpp:62-80`). The library's `prompt_open` only grows (`xtier_lib.cu:410-416`), and `open_rings` takes a card
ring someone else opened when `stream_slots() > 0`.

**Files:**
- Modify: `src/core/peer_experts.cpp` (:244-294), `include/strata/core/peer_experts.hpp` (`open_foreign`, `PeerRoom`,
  `peer_room_ring`)
- Modify: `src/core/xtier.cpp` (:426-438), `include/strata/core/xtier.hpp` (:113-117): remember the opened tokens
- Modify: `include/strata/prefill/prefill.hpp`, `src/prefill/prefill.cpp` (`tier_prompt_ok` and `tier_prompt_mats` beside
  `tier_prompt_bytes` :2842, `set_peer` :2416-2445)
- Modify: `src/prefill/tier_stream.cpp` (:29-37, 62-81), `include/strata/prefill/tier_stream.hpp` (`tier_card_has_room`
  :57-60 goes)
- Modify: `tests/prefill/tier_landing_test.cpp` (:317-322, the `tier_card_has_room` checks move to `peer_room_test`)
- Modify: `src/program/generate.cpp` (:580 default, :6830-6840 the call, :6860-6870 the link probe moved before the fill,
  :8654-8681 `tier_room`, :9081-9087, the rings opened at serve start)
- Create: `tests/core/peer_room_test.cpp`. Modify: `CMakeLists.txt`, `src/prefill/xtier_prefill_parity.cpp` (:832-842),
  `docs/MULTI_GPU.md` (:620-622, :808-812)

**Interfaces:**
```cpp
// include/strata/core/peer_experts.hpp
struct PeerRoom { int64_t tokens = 0; int64_t ring = 0; bool want_ring = true; uint64_t gu = 0, d = 0, blob = 0, prompt_bytes = 0, ring_bytes = 0; };
/// The ring of copies to open before the fill: the room's when it is wanted and the pairs left in RAM need it, else 0.
int64_t peer_room_ring(const PeerRoom& r, uint64_t rest_bytes, uint64_t free_bytes, uint64_t keep_bytes);
bool open_foreign(const xtier::HipCard& card, const std::vector<std::pair<int32_t, int32_t>>& ranked,
                  const std::function<bool(int32_t, int32_t)>& held, ExpertSource& src, int64_t n_layers,
                  int64_t n_expert, int reserve_mib, int64_t max_slots, uint64_t cap_bytes, const PeerRoom& room,
                  std::string& err);
// include/strata/core/xtier.hpp (Device)
int64_t prompt_tokens() const;   // the tokens prefill_open opened, 0 when none
// include/strata/prefill/prefill.hpp
static bool tier_prompt_ok();
static void tier_prompt_mats(uint64_t& gu, uint64_t& d);
```
- `--peer-reserve-mib` on the AMD path now means the VRAM left after experts, buffers, ring and the link probe's buffer.
  Default 320 MiB when `peer_hip >= 0` and the flag is not given. The NVIDIA peer keeps 600.
- Start line: `strata generate: the AMD card keeps prompt buffers for N tokens (X MiB) and a ring of R copies (Y MiB)
  before its experts, Z MiB stay free`.

Mechanism (read from the code where it cites lines):
1. After `xdev_.open`: run the link probe (it keeps 64 MiB on the card). Then sum the 256-byte rounded blobs of the ranked
   pairs no NVIDIA card holds (`rest`). `ring = peer_room_ring(room, rest, free, keep)`. Call `xdev_.prefill_open(room.tokens,
   kTierEmbd, gu, d, ring, blob)`. Then read `mem_info` and fill as today, so the card's own rounding and the probe's
   buffer are in the reading.
2. `room.tokens` is `o.prefill_chunk` (under bare auto `prefill_auto_max` until Task 21 sets the chunk). `room.ring` is
   `tier_card_layer(n_expert)` (128). `room.want_ring` is true here. Task 23 sets it from the copy. `gu` and `d` from
   `tier_prompt_mats()`, `blob` from the layout's `max_blob`.
3. The room is wanted only when `o.serve`, `tokens >= Prefill::stream_all_min_tokens()` and `tier_prompt_ok()` (every layer on
   MMQ, no `wmma_only`, `fused_ring` off: the tests at `prefill.cpp:2379, 2398-2401`; `STRATA_PF_PEER_HOST=0` and
   `STRATA_MMQ_KQ=0` make it false). The ring is skipped under `STRATA_TEST_XTIER_RING_FAIL`.
4. In the auto scan, `tier_room(c)` becomes `c <= xdev()->prompt_tokens()`. `tier_card_has_room` is deleted.
5. Both rings (card and host staging) open when serving starts, after `set_peer`, not at the first planned chunk.
6. If `prefill_open` fails at load: one line, retry without the ring, then without buffers, then fill as today.

Sketch:
```cpp
// the AMD card's prompt buffers and ring of copies first, so its experts take only what they leave
const int64_t ring = peer_room_ring(room, rest, free0, keep);
```

- [ ] **Step 1: Write the failing test** `tests/core/peer_room_test.cpp`: `rest` larger than free minus keep minus buffers
  gives 128. `rest` that fits gives 0. `tokens` 0 gives 0. `want_ring` false gives 0. A room under `kTierCardMin` copies
  gives 0. Move the five `tier_card_has_room` checks from `tier_landing_test.cpp:317-322` here in `peer_room_ring` terms.
  Register it and run. Expected: FAIL (no function).
- [ ] **Step 2:** Add an `xtier_prefill_parity` case: `prefill_open(T=4096, ring 16)` before the arena, then a later
  `prefill_open(T=2048, ring 0)` is a no-op, `stream_slots()` stays 16 and the sums match.
- [ ] **Step 3:** Implement mechanism items 1 to 6.
- [ ] **Step 4:** Build both trees under the lock. Run the ctest set, `peer_room_test`, `prompt_buffers_test` and
  `tier_landing_test`. Expected: all pass. The NVIDIA-only and HIP-only paths are untouched (`peer_hip < 0`).
- [ ] **Step 5 (box, `t20-s1`, start logs only, one start each):** fixed `--prefill 4096`, 16384 and 24576. Expected (read
  from the code, 4.2236 MiB a blob): W6800 experts about 7,444, 7,311 and 7,222, less about 15 for the probe's 64 MiB if
  today's reading did not hold it; `may stream ... 128 at a time on the card`, no `reads its prompts without the AMD tier`,
  about 320 MiB free after the ring and the probe, and Task 19's line after one 20K prompt at 24576 at 256 MiB or more.
  `STRATA_TEST_XTIER_RING_FAIL=1` and `STRATA_MMQ_KQ=0` starts open no ring and fill as before.
- [ ] **Step 6:** Update `docs/MULTI_GPU.md` (the reserve's meaning, the ring opened at load). Append the start facts.
  Commit and push.

**Box stint:** about 15 min (starts only). The speed and KL A/B is Task 21's. If Task 21 is not kept, Task 20 gets its own
A B B A at chunk 4096 against PREV, with KL v2 on two starts, before it is kept or reverted.

**Keep rule:** decided with Task 21.

**Commit:** `AMD tier: the card opens its prompt buffers and its ring of copies before its experts fill, and --peer-reserve-mib is what stays free after them`

### Task 21: `--prefill auto` on one NVIDIA card with the AMD tier reads prompts of up to 24,576 tokens in one chunk

**Why (measured):** each chunk of 1,024 tokens or more streams every RAM expert once, so a staged prompt costs about 5 s a
chunk (Task 16 350 W fit). K24 and K24R (one start each, timing on) read the 20K at 1,715 and 1,776 tok/s, the 8K at 1,064
to 1,105, against B3 and B4's 746 to 785 and 719 to 754 at 4096. `--prefill auto` picks 4,352 today (20K 567 to 572 at 250
W). Read from the code: a request lends only `min(chunk, its tokens rounded up to 256)` (`generate.cpp:7685-7717,
13576-13631`), so a large configured chunk costs a short prompt nothing on the 3090. The 3090 lends 506, 871, 1,600 and
2,329 slots at 4096, 8192, 16384 and 24576 (measured). At RE24 the 3090 had 231 MiB free with everything loaded and 93 at
a later request (measured), so the rule needs a VRAM floor, not only a slot floor. The bench's 20K prompt is 20,053
tokens, so a chunk of 20,480 still reads it whole.

**Gate:** Task 18's ruling is in. Task 19's T = 24576 tests pass. 16b Step 2 is read: if it names the W6800's FP16
transport or the chunk shape, Task 21b's matching fix lands first.

**Files:**
- Modify: `src/program/generate.cpp` (:6820-6845, hoist :7661-7682 and :7788-7796 above the W6800 fill, delete :8655-8674
  for this topology, :8727-8749)
- Modify: `include/strata/program/prompt_buffers.hpp` (the rule), `tests/program/prompt_buffers_test.cpp`
- Modify: `docs/MULTI_GPU.md` (the auto chunk paragraph that describes `c150b99a`)
- Create on the box: `~/bench-mixed/cfg-q6k3.json` (`cfg-q6k2.json` with `--prefill auto`) and `cfg-q6k3-prev.json`

**Interfaces:**
```cpp
// include/strata/program/prompt_buffers.hpp
constexpr int64_t kOneCardTierChunkMax = 24576;   // the largest chunk run on the 3090 and W6800 box
constexpr uint64_t kOneCardVramFloor = 256ull << 20;   // the engine's own LOW line
/// The chunk one NVIDIA card with the AMD tier reads prompts in: the largest on the 256 grid it lends above its floors, or 0.
int64_t one_card_tier_chunk(int64_t max_context, int64_t slots, const std::function<int64_t(int64_t)>& lend,
                            uint64_t vram_free, const std::function<uint64_t(int64_t)>& prompt_bytes);
```
- Start line: `strata generate: prompt chunk auto: N tokens (one card with the AMD tier: the largest the 3090 lends above
  its 128-slot floor with 256 MiB of VRAM left, at most 24576)`.

Mechanism:
1. Hoist `slots_from_bytes` and `lend_slots` (:7661-7682) and `biggest_chunk` (:7788-7796) above the W6800 fill at :6830.
   The pinned-share block (:7646-7658) reads `srcp->pinned`, which is about 0 at that point because `--resident-experts`
   pins later (`:7761`, `:8244` on). Leave it where it is. Task 23 prices the share from the copy's plan.
2. When `o.prefill_auto`, there is no layer split and `peer_hip >= 0`: set `o.prefill_chunk` to the rule's chunk and
   `o.prefill_auto = false` before `open_foreign`, so Task 20's room is sized for it. The fixed scan then takes it unchanged.
3. The VRAM floor is a run-time read: the 3090's free VRAM after its cache and dense allocations, minus the chunk's
   prompt bytes (`bytes_needed_impl`) and the verify captures' measured cost, must stay at or above 256 MiB.
4. For this topology, delete `c150b99a`'s `tier_streams`, `tier_vfree` and `tier_room`, and skip `kAutoLendPct` and the
   bare-auto 8192 ceiling (:7727-7781). Other topologies keep their rules (the code path is unchanged when the condition is
   false).
5. If the rule returns 0, today's scan runs. Check every later reader of `o.prefill_auto`.

Sketch:
```cpp
// the chunk one card with the AMD tier reads prompts in, the largest it lends above its floors
const int64_t box_chunk = pb::one_card_tier_chunk(o.max_context, xcache.slots(), lend_slots, vfree, prompt_bytes);
```

- [ ] **Step 1: Write the failing test** in `prompt_buffers_test` with `lend(c) = 142 + 0.089 c` and ample VRAM: slots 3,103
  gives 24576, slots 2,000 gives 19200, `max_context` 8192 gives 8192, slots 200 gives 0. The measured loans 506, 871,
  1,600 and 2,329 all fit 3,103 slots. With `prompt_bytes(c) = c * 40 KiB` and 1,100 MiB free, the floor picks the largest
  grid chunk that leaves 256 MiB. Run. Expected: FAIL.
- [ ] **Step 2:** Implement mechanism items 1 to 5.
- [ ] **Step 3:** Build both trees. Run the ctest set, `prompt_buffers_test`, `split_rules_test` and `prefill_plan_test`.
- [ ] **Step 4 (box, `t21-s1`, start log on `cfg-q6k3`):** expect `prompt chunk auto:` at most 24576, W6800 about 7,222, a
  128-copy ring, no `without the AMD tier`, a 20K request that `borrows` about 1,942 slots, and one 24,300-token request with
  Task 19's CUDA0 line at 256 MiB or more and no non-finite value. Write the `--expect` set for the A/B.
- [ ] **Step 5 (the floor's tie-break, one start each):** if the floor bound the chunk below 20,480, compare the rule's chunk
  against 24576 with `--vram-reserve-mib` raised by the missing room (fewer slots). Keep the faster one as the rule's
  behavior and say which.
- [ ] **Step 6 (KL):** KL v2 on two starts of NEW `cfg-q6k3`. If it fails, stop and run Task 21b before any speed A/B.
- [ ] **Step 7 (speed A/B, `t21-s2`):** PREV `cfg-q6k2-prev` (`ad9157b4`, chunk 4096) against NEW `cfg-q6k3`, protocol starts.
  Add one start of PREV with `cfg-q6k2-p24576-prev` plus `--peer-reserve-mib 2200` (K24R as a config, `--expect
  chunk=24576,w6800_experts=7100..7300,ring_slots=128`) and one attribution start of NEW at `cfg-q6k2` (Task 20's room at
  4096).
- [ ] **Step 8:** Docs. Append. Commit and push.

**Box stint:** about 60 to 75 min. **Expected (one-start stand-ins, timing on):** 20K 1,700 to 1,800 or more with timing off,
8K 1,060 to 1,170, 2K and chats within the spread (K arms read the 2K at 337 to 345, so watch it).

**Keep rule:** 20K and 8K KEEP against PREV `cfg-q6k2-prev`, 2K and chats not slower, KL v2 passes on both starts, no tier
drop, the 3090 at 256 MiB or more after every request. On a KL failure Task 21b runs. If no fix passes, set
`kOneCardTierChunkMax` to the largest chunk that passes (16384, then 8192), re-run the KL and the A/B, and record it.

**Commit:** `prompt chunk: --prefill auto on one NVIDIA card with the AMD tier reads up to 24576 tokens at once, sized by the card's loan and free VRAM before the AMD card fills`

### Task 21b: (Conditional) The precision fix for one-pass prompts

**Gate:** Task 21's KL fails, or 16b Step 2 names a numeric factor first.

**Why (read from the code and computed from dumps):** C16G's long 0 ran unplanned and unstreamed at a 3.18-nat position, and
long 4500 read 0.1293 against the head's 0.0489. Known losses that grow with the W6800's share of rows: x goes FP32 to FP16
to FP32 before q8_1 (`kernels.cu:2434`, `xtier_lib.cu:655, 694`), and the W6800's per-token sums come back through FP16
with saturation (`xtier_lib.cu:712-714`). Losses that follow the chunk: NVIDIA MMQ stream-K splits K at tile-count
boundaries (`moe_mmq.cu:524-551`), and cuBLAS picks its algorithm per M. The largest shared loss is MMQ's q8_1 activations
on both cards.

**Files (by factor):**
- O4 (FP32 sums back, x kept FP16): `include/strata/core/xtier_abi.hpp` (ABI bump), `src/xtier/xtier_lib.cu` (:655-656,
  :709-719), `src/prefill/prefill.cpp` (:2180-2184, :2409-2418, :3973, :5576-5584), `src/prefill/kernels.cu` (:2822-2853),
  `src/core/xtier.cpp` (loader)
- O5 (x as the 3090's q8_1 rows): the same plus a HIP gather by src
- MMQ order (no stream-K): `src/prefill/moe_mmq.cu` (whole tiles :522-551)
- O6 (a second q8_1 pass on h's residual): `src/prefill/prefill.cpp` (:5385-5405), `src/xtier/xtier_lib.cu` (:700-706),
  `src/prefill/moe_mmq.cu`
- Tests: `xtier_prefill_parity`, `prefill_mmq_kquant_test`

**Interfaces:**
```c
/* O4: strata_xtier_prefill_args gains these in place of one f16 flag, ABI version + 1 */
uint32_t x_f16, sum_f16;
```
- MMQ order: env `STRATA_MMQ_STREAMK=1` keeps the old path until Task 35.

- [ ] **Step 1 (attribution, no code, two starts each, per-prompt tables with entropy, `t21b-s1`):** (a) `STRATA_PF_PEER_F16=0`
  (FP32 x and sums), (b) `STRATA_PF_PEER_HOST=0` at the box chunk (the W6800 out of prompts), (c) `STRATA_MMQ_KQ=0` at the
  box chunk (the chunk's shape alone), (d) NEW with `--no-warm`. Name the factor.
- [ ] **Step 2: Write the failing test** for the chosen fix: O4 or O5, `xtier_prefill_parity` bitwise against the FP32 path
  (O4) or an `xq` bitwise equal to the 3090's for the same rows (O5); MMQ order, `prefill_mmq_kquant_test` gives the same
  output for one row at T 4096 and T 24576. Expected: FAIL.
- [ ] **Step 3:** Implement. Build both trees. Run the ctest set and the two tests by name.
- [ ] **Step 4:** KL v2 on two starts. Then A B B A of the fix against Task 21's NEW without it.
- [ ] **Step 5:** If unsure positions alone carry it (entropy above about 2.5 nats, and starts differ as much as from the
  reference), send it back to the controller's ruling. Append. Commit and push.

**Expected (estimates):** O4 costs T x 2560 x 2 bytes more a layer on the 3090's link, about 126 MB a layer at 24576. O6
costs about 2.5% of the 3090's timeline at 4096 and some VRAM. KL effects unknown.

**Keep rule:** KL passes on two starts, and the fix's speed cost is inside the spread or smaller than Task 21's gain.

**Commit:** `prefill: the precision fix the one-pass chunk needs (the factor named by its KL arms)`

---

## Stage 2: one pinned RAM copy

### Task 22: The pinned copy fills from the page cache layer by layer, registers each layer, and frees the file pages it copied

**Why (measured):** RE (Task 16 350 W, chunk 4096, one start each) read 2K 619, 8K 1,180 to 1,200, 20K 1,221, but was ready
in 111 to 115 s and read 109 to 132 GB from the drives. RE24 and RE24AS were ready in 130 and 123 s, with 119 and 140 GB
read ("pinned 98.17 GiB in 109 s"). The copy is re-read at 1.0 to 1.2 GB/s, below the 1.7 GB/s a `cat` reaches. Read from
the code: the #1250 guard drops the page cache of all five shards when MemFree is short (`expert_source.cpp:3766-3778,
3702-3720`), then one `cudaHostAlloc` of the whole copy (`:3785-3787`), then 6 threads read it back from NVMe (`:4069`).
Nothing drops the copied ranges' file pages afterwards (`:4039-4058`). Making it the default first would put a 2-minute
start and a 120 GB drive read into every later A/B. The RAM is likely not channel interleaved (SMBIOS, Task 16 memory
probe): pinned ring buffers landed up to 121 of 128 pages in one range, and the pool reads this copy in decode.

**Gate:** 16b Step 4 is read (drive reads, Cached, MemAvailable at RE24).

**Files:**
- Modify: `src/core/expert_source.cpp` (:3700-3990 allocation, guard and pin; :3984-4070 the fill worker; `spans()`
  :979-993; `read_direct` :2694 on; `pinned` and `device_alias` :4601-4622), `include/strata/core/expert_source.hpp`
- Modify: `tests/core/file_expert_source_test.cpp`, `tests/core/exchange_storage_test.cpp`
- Box only (not committed): `~/bench-mixed/t22-pinprobe.cu`

**Interfaces:**
```cpp
// include/strata/core/expert_source.hpp
/// The fill order of a layer-order copy and, per expert, 1 when its spans are cached or 0 when read with O_DIRECT.
struct CopyFillPlan { std::vector<int32_t> layer_order; std::vector<uint8_t> cached; };
CopyFillPlan copy_fill_plan(const std::vector<uint8_t>& cached, int64_t n_layers, int64_t n_expert);
```
- The new fill applies to the layer-order layout (no budget). A budget keeps today's rank-order path.
- Start line: `strata generate: RAM copy: N experts (X GiB) and the lend region (M slots), filled in A s (B s registering),
  C GB from the drives, the copied file pages dropped, pages by range r0..r5`.
- Test hook `STRATA_VERIFY_COPY=1`: after each request, compare every copy blob with its GGUF spans and print mismatches.
- `STRATA_PIN_ALLOC=host` keeps today's single `cudaHostAlloc` path for the A/B, until Task 35.

Mechanism:
1. One anonymous range in layer order, `MADV_HUGEPAGE` only if the probe says both H2D and the CPU read rate are not slower.
2. Copy layer by layer on 12 to 15 threads: memcpy an expert whose three spans `mincore` shows cached, else `read_direct`
   (O_DIRECT, 32 in flight).
3. After each layer: `madvise(MADV_DONTNEED)` on the mapping's range, then `posix_fadvise(POSIX_FADV_DONTNEED)` on each
   copied expert's three spans. RAM in use stays flat, so the whole-file drop is not needed. `pin_step_fits` stays as a
   per-layer reserve check for O_DIRECT layers.
4. One registrar thread runs `cudaHostRegister(Portable | Mapped)` on each completed layer in order and checks that
   `cudaHostGetDevicePointer` equals the host pointer. On a failure it stops at the registered prefix with one line.
   `pinned()` already handles a partial pin.
5. At the end, count the copy's pages per physical range (as `t16-pinmap` did) and print it on the start line.

- [ ] **Step 1 (probe, box, under the lock, about 20 min, `t22-s1`):** time `cudaHostAlloc(64 GiB)`; an anonymous 64 GiB
  `mmap` with and without `MADV_HUGEPAGE`, a 15-thread touch and one `cudaHostRegister`; 48 per-layer registrations from 1
  and 4 threads; check device pointer equals host pointer; read AnonHugePages; for both backings, the 3090's H2D GB/s for
  random 4.43 MB blobs across 64 GiB into a 256 MiB buffer (its IOMMU group is DMA-FQ) and a 15-thread CPU read of random
  4.43 MB blobs; the pages per physical range; registration under the bench shell's 23.6 GiB memlock and under
  `prlimit --memlock=unlimited`. Choose huge pages and the registration pattern from these numbers.
- [ ] **Step 2: Write the failing tests:** `file_expert_source_test`: the layer-order fill from a cached fixture GGUF gives
  blobs bitwise equal to `blob()`; a failed registration at layer k leaves layers before k pinned and the rest resident but
  pageable; the dropped spans are no longer resident (`mincore`). `exchange_storage_test`: after a 3090 exchange the copy holds
  `out` at `in`'s old place. Expected: FAIL.
- [ ] **Step 3:** Implement items 1 to 5 and the hook.
- [ ] **Step 4:** Build both trees. Run the ctest set plus `file_expert_source_test`, `exchange_storage_test` and
  `expert_cache_*`.
- [ ] **Step 5 (identity, `cfg-q6k2-re24` and its `-prev` twin, PREV with the old fill):** prompt dumps with
  `STRATA_PF_PEER_HOST=0` (KL 0), decode identity, and one 20-minute chat session with adaptation on and
  `STRATA_VERIFY_COPY=1`: 0 mismatches.
- [ ] **Step 6 (A/B, `t22-s2`):** the same configs, PREV against NEW, protocol starts, page cache rebuilt before every start.
  Record ready time after a `cat` of the shards and on a restart right after a copy run, drive GB at start and per measured
  request, Cached, MemAvailable and the pages per range.
- [ ] **Step 7:** Append. Commit and push.

**Box stint:** about 60 min plus the probe. **Expected (estimates):** after a `cat`, ready in 15 to 30 s (98 GiB of memcpy at
the measured 23.4 GB/s is about 4.5 s, plus registration, unknown). A restart after a copy run reads the dropped ranges with
O_DIRECT: about 85 to 105 s at RE's measured 1.0 to 1.2 GB/s, or about 60 s at the `cat` rate of 1.7 GB/s, against RE24's
123 to 130 s. Measured requests not above PREV's drive MB. Prompt and decode speed as RE24's.

**Keep rule:** an enabling change. Kept when identity holds, 0 copy mismatches, the start is shorter than PREV's beyond the
spread in both start kinds, measured requests read no more from the drive than PREV's, and prompts and decode are not
slower.

**Commit:** `pinned RAM copy: filled from the page cache layer by layer, registered as each layer completes, and the copied file pages freed`

### Task 23: The pinned copy by default beside the AMD tier, and the W6800's ring re-decided with it

**Why (measured, Task 16 350 W, one start each):** RE24 read 2K 590, 8K 1,700 to 1,766, 20K 2,528, chats 44.5, 43.1, 36.0,
against B3 and B4's 360 to 387, 719 to 754, 746 to 785 and 24.2 to 32.2. RE (chunk 4096) read +55 to +72% per start on the
three prompts. With the copy, the ring added nothing: RER 2K 552, 20K 1,209 and RERF 618, 1,203 against RE 619, 1,221, and
RE24 ran with no ring. Read from the code: `--resident-experts` on a lone card with the tier pins every expert the 3090 does
not hold plus the lend region (`generate.cpp:8243-8275`, `expert_source.cpp:3600-3614`). Its blobs are DMA'd directly in
prompts (`prefill.cpp:4046-4064`), refills and 3090 swaps read the copy, and the CPU pool reads it with no mincore. With every
blob pinned, the decode PCIe share would fly at 0.52 to 0.55 (spec correction 3). The stager profile is chosen from a cache
sample the drop can empty (`prefill.cpp:1592-1625`). The pinned share that sizes the 3090's streamed ring and lend cap is
read before the copy pins (`generate.cpp:7646-7658`, `:7761`), so it is about 0 today. The copy changes the planner's rates
and the row split (K24R's 3090 computed 44 to 46% of the W6800's rows, RE24's 0 to 15%), so the math changes.

**Gate:** Task 22 kept. 16b Steps 3 and 4 read.

**Files:**
- Modify: `src/program/generate.cpp` (:2799-2808 parse and default, new `--no-resident-experts`, the `--pcie-frac` default
  near :3605-3646, the pinned share from the copy's plan at :7646-7658, the box plan's ring input, :8243-8275 `tier_pairs`)
- Modify: `src/prefill/prefill.cpp` (:1592-1625 stager profile)
- Modify: `src/core/expert_source.cpp` (`pin_cache_complement`, `make_cache_complement_plan` :163-215),
  `include/strata/core/expert_source.hpp`
- Create: `include/strata/program/resident_default.hpp`, `tests/program/resident_default_test.cpp`. Modify: `CMakeLists.txt`,
  `tests/core/split_lend_complement_test.cpp`, `docs/MULTI_GPU.md` (the RAM copy paragraph), `docs/DETAILS.md` (settings)

**Interfaces:**
```cpp
// include/strata/program/resident_default.hpp
/// Whether a start keeps every expert its NVIDIA card does not hold in pinned RAM without being asked.
bool resident_by_default(bool linux_build, int nvidia_cards, bool amd_tier, bool resident_flag_given, bool budget_given,
                         bool off_flag, uint64_t copy_bytes, uint64_t mem_available, uint64_t keep_free, uint64_t memlock_limit);
// include/strata/core/expert_source.hpp
bool pin_cache_complement(..., const std::vector<std::pair<int32_t, int32_t>>* tier_pairs = nullptr);   // left out of a budget's rank, lend kept
```
- New flag `--no-resident-experts`. Default changes beside `--peer-device hip:N`: the copy (when MemAvailable minus the copy
  keeps at least 16 GiB plus the dense pages, and RLIMIT_MEMLOCK covers the copy, both read at run time and logged; a
  short limit gives one line and no copy), `--pcie-frac 0` unless given.
- The pinned share and the loan are priced from the copy's plan before the loan is priced.
- When RAM is short, the budget path runs with the W6800's experts in `tier_pairs`, so its rank covers RAM experts first and
  the lend region stays (the lend tests at `expert_source.cpp:3573, 3600` keep checking only stage and helper pairs).
- The box plan sets `PeerRoom::want_ring` false when the copy holds every expert the 3090 streams, unless Step 6 shows the
  ring paying.
- The stager takes the RAM profile when every blob the 3090 streams is in the copy.

- [ ] **Step 1: Write the failing tests:** `resident_default_test`: a lone card with the tier and no flags gives true; flags
  given give as given; `--no-resident-experts` gives false; a layer split gives false; a copy that leaves under 16 GiB gives
  false; a memlock limit under the copy gives false. `split_lend_complement_test`: a lone-card plan with `tier_pairs` under
  a budget keeps the lend region and ranks RAM experts first. Expected: FAIL.
- [ ] **Step 2:** Implement the default, the flag, `tier_pairs`, the share from the plan, the `--pcie-frac` default, the stager
  profile and the ring input.
- [ ] **Step 3:** Build both trees. Run the ctest set plus `resident_default_test`, `file_expert_source_test`,
  `exchange_storage_test` and `split_lend_complement_test`.
- [ ] **Step 4 (identity):** PREV (Task 22 head with `--resident-experts --pcie-frac 0`) against NEW on `cfg-q6k3`, both with
  the W6800 out of prompts: prompt dumps KL 0 and decode identity.
- [ ] **Step 5 (KL):** KL v2 on two starts of NEW `cfg-q6k3` (the served config). Both must pass.
- [ ] **Step 6 (the ring, `t23-s1`):** NEW against NEW with the ring forced on (`STRATA_TEST_XTIER_RING=1` or the room's
  input), two starts each, read beside 16b Step 3's RE24 with reserve 2200. If the ring is not faster beyond the spread,
  it stays off with the copy. If it wins, KL v2 on two starts of that config too.
- [ ] **Step 7 (A/B, `t23-s2`, every arm under `prlimit --memlock=unlimited`):** PREV `cfg-q6k3-prev` against NEW `cfg-q6k3`
  (KEEP), and PREV `cfg-q6k2-re24-prev` against NEW `cfg-q6k3` (not slower), protocol starts, page cache rebuilt before
  every PREV start without a copy. 2K, 8K, 20K, chats, long answers. `--expect pinned_gib=95..100`. Re-measure the spread
  on NEW's starts and tell the controller if it is wider than 16b's.
- [ ] **Step 8 (the 3090's prompt ring with the copy, `t23-s3`):** measured in 16b (one start each, RE24): the ring is 96
  slots, because `g_pinned_share` is read before the copy pins (`prefill.cpp:268-353` picks 384 at a pinned share of 0.9
  or more, else 96). With `STRATA_PREFILL_RING=384` the 20K read 2,917 and the 8K 2,030 (+14%), and the 3090's wait for
  copies fell from 1,156 to 32 ms. The prompt is then bound by the 3090's own work (non-expert 0.226 ms a token, expert
  GEMMs 0.073). With the share priced from the copy's plan (Step 2), check which ring the rule resolves to on this pack
  (the byte budget `ring_budget_slots` gives about 119 Q6_K slots above `g_ring_small_max`). Arms at 192, 384 and 512
  slots, two starts each, with Task 19's CUDA0 VRAM line after the first one-chunk prompt. Set the rule for one NVIDIA
  card with the AMD tier and a whole copy to the fastest ring that keeps the 256 MiB floor (a run-time read), then
  `STRATA_PREFILL_TIMING=1 STRATA_XTIER_TIMING=1` on one start (wait copy, the 3090's GB/s, Task 19's refill and VRAM
  lines).
- [ ] **Step 9:** Docs. Append. Commit and push.

**Box stint:** about 120 min (copy starts, cache rebuilds, the ring arms). **Expected (RE24, one start, timing on):** 2K about
590 to 605, 8K about 1,700 to 1,790 (about 2,030 with the 384 ring), 20K about 2,500 to 2,570 (about 2,917 with the 384
ring), chats about +40 to +70% over `cfg-q6k2`. The 20K's refill falls from about
2.15 s to under 0.5 s (estimate).

**Keep rule:** at least two of 2K, 8K and 20K KEEP against `cfg-q6k3-prev`, the third not slower, chats and long answers not
slower, not slower than `cfg-q6k2-re24-prev`, identity holds, KL v2 passes on two starts of the served config, MemAvailable
at least 16 GiB after start, warm start under 60 s after a `cat`. The ring rule follows its own arms and KL.

**Commit:** `pinned RAM copy: one NVIDIA card with the AMD tier keeps every expert the card does not hold, and its lend region, in page-locked RAM by default`

### Task 23b: (Conditional) The path without a copy: stager threads at 350 W, cheaper staging, refills through pinned buffers

**Gate:** Task 23 not kept.

**Why (measured):** at 350 W, 2, 4 and 8 stager threads read the 20K at 348, 568 and 713 tok/s (Task 16 probe 2, one start
each, timing on), and the W6800's copy threads use the same count (`prefill.cpp:2441`, read from the code). The refill of lent
slots runs serially with pageable copies on the request thread (`generate.cpp:13504-13567`, `expert_cache.cpp:728-743`), and
`STRATA_REFILL_PARTS` does nothing under io prefetch (`expert_source.cpp:4546`). C7 (non-temporal stores and no mincore in
the stager copies) survives its skeptic and only helps this path.

**Files:** `src/prefill/prefill.cpp` (:1592-1625 thread count, the stager copy loop for C7), `src/program/generate.cpp`
(:13502-13567), a new `Prefill::refill` entry that reuses the idle Stager, `src/core/expert_cache.cpp` (:728-766), a new
refill test.

**Interfaces:**
```cpp
/// Refills the lent slots through the stager's pinned buffers on several threads, one wait at the end.
bool Prefill::refill(ExpertCache& c, const std::vector<std::pair<int64_t, int64_t>>& lent, ExpertSource& src, std::string& err);
```

- [ ] **Step 1:** Two starts each of 4 and 8 threads (`STRATA_STAGER_THREADS`) on `cfg-q6k3`. Set the RAM profile's count
  from the winner.
- [ ] **Step 2: Write the failing test:** a host test with a fake source and cache checks every lent slot holds its blob after
  `refill`, and no copy reads a stage buffer still being filled. Expected: FAIL.
- [ ] **Step 3:** Implement the refill, and C7 as its own commit with its own one-start timing check. Remove
  `STRATA_REFILL_PARTS`. Build both trees, ctest with the new test by name.
- [ ] **Step 4:** Prompt identity, then the A/B with Task 19's refill line on one start each.

**Expected (estimates):** the 20K's refill from about 2.15 s to about 0.5 s: 20K +15 to +18%, 8K +11 to +13%, 2K about +6%.

**Keep rule:** prompts KEEP, chats not slower, identity holds.

**Commit:** `prompt loan: the lent slots refill through the stager's pinned buffers on several threads, and the RAM profile uses the measured thread count`

### Task 24: (Gated) The planner prices the W6800 from its probed link, scales the halves' lead by tokens, and counts streamed experts with no rows

**Why (measured, one start each):** in K24R's 20K prompt the planner fitted the W6800's layer at 48.1 ms and it took 23.5,
the W6800 streamed about 42 experts a layer of the 128 it could, and the 3090 computed 44 to 46% of the W6800's routed rows.
In K16R's first planned chunk the W6800's layer end was planned at 100.7 ms and measured at 34.5. In RE24 the W6800's
experts took 17 to 19 ms a layer against the 3090's 57 to 59. Read from the code: the W6800's streamed-byte prior is 6 GB/s
(`starting_rates(0)`, `generate.cpp:9074`) until a streaming call is observed (probe: 20.1 to 25.4 GB/s, pinned 27.8). The
halves' lead `TierPlanner::lead_` is a per-layer decayed mean in ms that does not scale with tokens
(`tier_planner_engine.cpp:28, 251-263`, used as `HalfStarts` at `:97`). The drain moves W6800-held experts to the 3090 while
it lowers the later finish (`expert_planner.cpp:37-53, 88-93`), and the 3090 must then stream them. With one-chunk prompts
every request is a chunk of a new size.

**Gate:** one start of Task 23's head with `STRATA_PREFILL_TIMING=1 STRATA_XTIER_TIMING=1` on 8K and 20K. Run only if the
halves lines show the W6800's layers ending 10 ms or more before the 3090's, or the 3090 computing more than 15% of the
W6800's routed rows, or 16b Step 3 names the 3090 as the bound with the W6800 under-used.

**Files:**
- Modify: `src/prefill/expert_planner.cpp` (:14-23, 102-117, 221, 276-277), `include/strata/prefill/expert_planner.hpp`
  (:29-30, 67-69), `src/program/generate.cpp` (:9074, 7910-7912), `src/prefill/tier_planner_engine.cpp` (:58-75, :97,
  :245-263, 327-368), `include/strata/prefill/tier_planner.hpp` (the `TierPlanner` lead), `src/prefill/prefill.cpp`
  (the counter in :4035-4066)
- For the ring and the cap (Step 6 only): `include/strata/prefill/tier_stream.hpp` (:36-55), `src/prefill/tier_stream.cpp`
  (:44, 48, 65)
- Tests: `tests/prefill/expert_planner_test.cpp`, `tests/prefill/tier_planner_test.cpp`

**Interfaces:**
```cpp
// include/strata/prefill/expert_planner.hpp
Rates starting_rates(double stage_link_gbps, double tier_link_gbps = 0.0);   // 0 keeps today's tier priors
// src/prefill/tier_planner_engine.cpp: lead_ holds ms per token of the first half, and HalfStarts takes lead_[l] * ta
```
- `TierStart` (the tier's start lag) is unchanged.
- The planner line gains `N of M streamed RAM experts got no rows`.

- [ ] **Step 1:** The gate start. Stop and record if the gate fails.
- [ ] **Step 2: Write the failing tests:** `expert_planner_test`: priors from a 25 GB/s tier link give the probed streamed-byte
  rate, and a fit never prices streamed bytes faster than the probe's floor. `tier_planner_test`: a lead learned on a
  16,384-token half applied to a 3,662-token half scales by 3,662/16,384. Expected: FAIL.
- [ ] **Step 3:** Implement the priors, the per-token lead, the floor and the counter. Build both trees, ctest with both tests
  by name.
- [ ] **Step 4:** One timing start: planned against measured layer ends within about 10% on the first planned chunk.
- [ ] **Step 5:** KL v2 on two starts, then the A/B with prompts in the order 2K, 8K, 20K, 8K (sizes change between requests).
- [ ] **Step 6:** With the planner fixed, re-check the ring with the copy (two starts each, ring on against off). Only if the
  per-chunk lines then show the W6800 streaming 128 in most layers and still ending first: raise the stream cap to what
  the room affords, staging ring at twice it, as its own A/B. Keep the halves' min(staging cap, card slots) at
  `prefill.cpp:3977`.
- [ ] **Step 7:** Append (with the no-row counter at 2K and 8K). Commit and push.

**Expected (inferred, not measured):** 0 to +10% on prompts, mostly the first prompts after a start and after size changes.

**Keep rule:** prompts KEEP (three starts per arm, the gain is under the staged spread), chats not slower, KL v2 passes on two
starts.

**Commit:** `prompt planner: the AMD tier is priced from its probed link, the halves' lead scales with the half's tokens, and streamed experts with no rows are counted`

### Task 25: Where the 3090's own prompt time goes once copies stop binding (measure only)

**Why (measured):** K24R's 20K prompt: of the 9,062 ms GPU timeline, wait copy was 3,199 ms, hyper-connections 1,093, GDN
1,471, QSA 1,174, combine 447, router and shared 154, PLE 124, expert GEMMs 1,015, dequant 252, gather 47. RE24's fit is
0.25 ms a token plus 2.7 to 2.8 s a chunk, so at 20K about 5 s of 7.9 s is per-token work (estimate from the fit).

**Gate:** after Task 23 (and Task 24 if it ran). 16b Step 3's split read.

**Files:** none in the repo. Box scripts `t25-*` only.

- [ ] **Step 1:** One start with `STRATA_PREFILL_TIMING=1` on 2K, 8K, 20K on the current head. Record 16b Step 5's
  `STRATA_MMQ_DIRECT` result beside it.
- [ ] **Step 2:** One nsys capture of a single 20K request (under the lock, GPU otherwise idle, delay and duration set to that
  request only).
- [ ] **Step 3:** A ranked table: ms, share and bound (memory or compute, from bytes and FLOPs) per kernel group, and each
  candidate's KL risk (C15 dense int8 MMQ, the hyper-connection read and write fused, the GDN recurrence, QSA attention).
  Add the 2K prompt's split: stream wait against compute, the W6800's busy share and its streamed GB/s.
- [ ] **Step 4:** Send the controller proposed follow-up tasks with numbers.

**Expected:** non-expert work at 50 to 65% of the 20K prompt (estimate). **Keep rule:** no code, done when the table and the
proposals are in the report. **Commit:** none (measurement only).

### Task 25b: (Conditional, probe only) Can the W6800 DMA from the copy without Task 11's stall

**Gate:** Task 25's 2K split shows the 3090's stream wait at 25% or more of the 2K prompt with the copy, and the W6800 busy
under half of it.

**Why:** the 2K prompt is one full stream over the 3090's link (552 to 619 tok/s in every copy arm, Task 16). With the copy
the W6800 streams through its own stager memcpy, which competes for RAM (read from the code). Its direct DMA from a
HIP-registered copy (C19) is the only route left, and Task 11 stalled on HIP registration with no known cause.

**Files:** none in the repo. Box probe `t25b-hipreg.cpp` only.

- [ ] **Step 1:** With no engine running, under the lock: register 1, 4 and 48 layers of a 98 GiB CUDA-registered anonymous
  range with `hipHostRegister` (read only use), time each, and run the W6800's H2D of random 4.43 MB blobs from it while the
  3090 DMAs from the same range. Record any stall, its length and the dmesg lines.
- [ ] **Step 2:** Report the W6800's GB/s and the 3090's GB/s alone and together. Propose a C19 task only if both rates hold
  and no stall shows.

**Keep rule:** probe only. **Commit:** none.

---

## Stage 3: decode

With the copy on, the CPU pool is the decode bound (RE24: CPU experts 26.5 and 30.1 ms of 54.0 and 56.4 ms chat windows,
the join 5.2 and 4.5 ms). The tasks that cut CPU work come first. The swap tasks (31 to 33) are expected to stop at their
gates.

### Task 25c: (Gated on Task 25b) The W6800 DMAs its streamed experts straight from the copy

**Gate:** Task 25b shows the W6800's H2D from a HIP-registered copy at 20 GB/s or more, the 3090's rate held, and no stall.

**Why (measured):** with the copy and one chunk, a 2K prompt reads at about 603 tok/s (16b, four RE24 starts). That is one
full stream of the RAM experts over the 3090's link alone. The W6800's link (27.8 GB/s pinned) sits idle, because its
streaming goes through the stager's memcpy (16b: ring room at RE24 gave nothing). Sharing the stream between both links
could take most of a 2K prompt's stream time (estimate: up to +50 to +80% on 2K, and less on 8K and 20K, which the
3090's own compute bounds at about 0.3 ms a token).

**Files:** `src/core/xtier.cpp` and `src/xtier/xtier_lib.cu` (registration of the copy with HIP, Task 25b's pattern),
`src/prefill/tier_stream.cpp` and `prefill.cpp` (direct addresses for copy-held experts in place of stager jobs, as
`expert_host` already allows), `src/program/generate.cpp` (registration after the copy pins), tests in
`xtier_prefill_parity` and `tier_landing_test`.

- [ ] **Step 1: Write the failing test:** `xtier_prefill_parity` with experts given as addresses inside a HIP-registered
  range gives the same sums as the stager path. Expected: FAIL.
- [ ] **Step 2:** Implement registration, with one log line and today's path on failure. Copy-held streamed experts go
  to the W6800 as direct addresses. The planner prices the W6800's streamed bytes at its probed link.
- [ ] **Step 3:** Both builds, the ctest set. Prompt identity is not expected (the row split moves), so KL v2 on two starts.
- [ ] **Step 4:** A/B on protocol starts: 2K, 8K, 20K, chats. A 20-minute chat session with adaptation on: no late decode,
  no timeout (Task 11's failure).

**Keep rule:** the 2K prompt KEEP, 8K and 20K not slower, chats not slower, KL v2 passes on two starts, no stall.

**Commit:** `AMD tier: the card DMAs its streamed experts straight from the pinned copy`

### Task 26: Decode diagnostics: both cards' copies per round, a pool timing line, and a pool bench on real experts

**Why:** Task 16's decode line counts only the 3090's swaps (`adapt_nswaps`, `generate.cpp:10570`) and averages the join over
all windows although a round runs every 4th (:15424, :15485, read from the code). So "1.8 ms per swap" and "a fixed 7 ms a
round" were misreadings (spec corrections 1 and 2). The pool's phase timers exist but serve never prints them
(`generate.cpp:17038-17050`). The CPU leg's split is a cross-request fit (0.13 to 0.145 ms per expert plus 0.08 to 0.27 ms
fixed per layer-window, inferred), and the pool's rate was worked out from the timing line (about 25 GB/s while it computes,
probe 7). Whether DecodeSpares moves the worker count is not known (spec correction 13). Every later decode task is gated on
per-call numbers.

**Files:**
- Modify: `src/program/generate.cpp` (:10570, :15473-15486, the W6800 fills from `PeerExperts::swaps()`),
  `src/kernels/cpu/pool.cpp`, `include/strata/kernels/cpu/pool.hpp`
- Create: `src/kernels/cpu/pool_native_bench.cpp`, `tests/core/pool_stats_test.cpp`. Modify: `CMakeLists.txt`

**Interfaces:**
```cpp
// include/strata/kernels/cpu/pool.hpp
struct PoolCallStats { int64_t calls = 0; int64_t e_hist[5] = {}; double ms_gu = 0, ms_q = 0, ms_down = 0, ms_sync = 0;
                       uint64_t bytes = 0; double ms_shard[8] = {}; int64_t calls_shard[8] = {}; int64_t worker_moves = 0; void reset(); };
const PoolCallStats& ExpertPool::call_stats() const;
int e_bucket(int64_t experts);   // 1, 2, 3-4, 5-8, 9+
```
```text
pool_native_bench --pack DIR --gguf FIRST_SHARD --layer L --e 1,2,3,4,6,8,12 --nt 1,2,3,5 --reps 5 [--from-copy]
```
- The decode timing line: `3090 swaps/window`, `W6800 fills/window` (the delta of `PeerExperts::swaps()`, which already
  counts each fill at `peer_experts.cpp:501`), `rounds`, `join/round ms`, and copy ms per copy per card.
- Under `STRATA_DECODE_TIMING`: `strata decode pool timing: calls N (E: 1 a%, 2 b%, 3-4 c%, 5-8 d%, 9+ e%), gate/up X ms,
  quantize Y, down Z, park/drain/re-park P, GB/s G, ms per call by shard 3/4/5, worker moves M`.

- [ ] **Step 1: Write the failing test** `pool_stats_test`: `e_bucket`'s edges, and `reset()` clears every field.
  Expected: FAIL (no header).
- [ ] **Step 2:** Implement the counters, the lines and the bench.
- [ ] **Step 3:** Build both trees. Run the ctest set plus `pool_stats_test` by name.
- [ ] **Step 4:** Decode identity, PREV against NEW (timing off).
- [ ] **Step 5:** One diagnostic start on the current head with `STRATA_DECODE_TIMING=1`: the pool line's per-window sums
  match the `CPU` field within 2%, join per round against the fit (1.11 x swaps + 0.87 x fills - 1.4 ms, without the copy)
  and RE24's 0.2 ms a swap, and the worker moves.
- [ ] **Step 6:** The bench under the lock with no engine running: E x nt, five repeats, layers from shards 3 and 4, from the
  copy and from the mapping.
- [ ] **Step 7:** Report the pool's fixed and per-expert cost per call, the per-shard rate, W6800 fills per round, join per
  round, copy ms per card, and whether spares move workers (if yes, decode arms add `STRATA_DECODE_SPARES=0` in both arms
  until Task 28). Append. Commit and push.

**Box stint:** about 30 min. **Expected:** no speed change. The numbers gate Tasks 28, 29, 31, 32, 33, 34.

**Keep rule:** diagnostics. Kept when identity holds and the new sums match the old fields within 2%.

**Commit:** `decode timing: both cards' copies per round and a pool timing line, and a pool bench on real GGUF experts`

### Task 27: A decode KL, and the route tail skip and PCIe share beside the tier decided by it

**Why:** the prompt gate cannot see decode (`quant_quality.py prompt --keep-caches` turns off the tail skip, the PCIe share and
adaptation). Tail skip: probe 8 (TS1, one start, no copy) read chats 34.0, 29.3, 28.0 against DR1 and DR2's 31.1 to 32.2,
25.1 to 25.7, 23.3 to 23.5, and halved the CPU's experts per layer-window. Elsewhere it measured +13 to +16% decode and
teacher-forced KL 0.0032 to 0.0058 with top-1 97.5 to 98.2% (`docs/DETAILS.md:1876-1926`, other cards). PCIe share: with a
pinned copy it flies at 0.52 to 0.55, so the 3090 takes CPU misses. RB60 (half the RAM experts pinned, share flying) cut
the CPU share from 12 to 13% to 4.4 to 6.4% (Task 16 log reading), with chats "not slower, maybe faster" over two starts.
Task 23 holds the share at 0 until this task. With the copy the CPU pool is the bound, so these lanes come first.

**Gate:** 16b Step 5's tail skip result read (two starts on RE24). Task 23 kept, for the PCIe share.

**Files:** `tools/quant_quality.py` (`cmd_run` :124-190, `FIXED`, `FIXED_ENV`, `KEEP_DROP_VALUE` and `KEEP_FIXED` :47-58,
argparse), `tools/test_quant_quality.py`, `src/program/generate.cpp` (:3605-3646, :6861, :10209-10292),
`src/core/expert_source.cpp` (:4858-4880), `docs/DETAILS.md` (the tail skip section).

**Interfaces:**
```text
quant_quality.py run --config CFG --out RUN.tsv --keep-caches [--allow tail-skip|pcie|adapt] [--env K=V]
```
- `--keep-caches` runs the config's engine as the server starts it (tier, caches, copy, context, int8 K/V), adaptation off.
  `--allow NAME` leaves that switch as the config and `--env` set it (`adapt` keeps the config's adaptation, for Task 32).
  Teacher-forced over `data/quant_quality/v1`'s 5,000 positions.
- Floor F = KL(head || head with `--expert-gib hip0=27`), a fixed placement change of about 650 W6800 experts (read from the
  code). Proposed gate (the controller rules): mean KL at most max(2F, 0.006), top-1 at least 95% against the switch off,
  two starts.
- Defaults beside `--peer-device hip:N`, as measured: the tail skip (then `STRATA_XTIER_TAIL_SKIP` goes) and `--pcie-frac`
  (0, a value, or auto with PcieAuto allowed beside the tier after reading that its model leaves out the W6800's entries).

- [ ] **Step 1: Write the failing test:** `run --keep-caches` keeps the config's tier, cache, copy and K/V args and drops only
  the fixed switches. `--allow tail-skip` leaves the skip as set. `--allow adapt` keeps `--adapt-every` and
  `--adapt-swaps`. Expected: FAIL.
- [ ] **Step 2:** Implement. Expected: PASS. Read what 16b Step 2 found for decode-side quality first.
- [ ] **Step 3 (box, `t27-s1`):** decode KL: the head twice (repeat check, KL 0 expected), the head with adaptation on twice
  (`--allow adapt`, repeat check for Task 32), hip0=27 (F), the tail skip on, `--pcie-frac 0.25` and 0.55.
- [ ] **Step 4:** Speed with decode timing on one start each: `--pcie-frac` 0, 0.25, 0.55, then the best against 0 on
  protocol starts. The tail skip on against off on protocol starts. Read the skip count, accepted tokens per window and CPU
  experts per layer-window.
- [ ] **Step 5:** Set the defaults the measurements earned. Build both trees, ctest. One confirming A/B against PREV.
- [ ] **Step 6:** Docs. Append. Commit and push.

**Expected (estimates):** tail skip chats +5 to +15%, long answers +0 to +4%. PCIe share chats +5 to +12%. They overlap.

**Keep rule:** per lane: decode KEEP, prompts not slower, the decode KL passes on two starts, acceptance falls 5% or less.

**Commit:** `decode: quant_quality run measures the decode KL with the config's tier, and the route tail skip and PCIe share beside the AMD tier are set from it`

### Task 28: Decode helpers built for a 62 GB box stay off when RAM holds every expert

**Why:** read from the code: the router look-ahead runs on an unpinned thread every layer with no spare CPU, warms nothing on
this box ("0 read ahead"), and predicts W6800-held experts too (`expert_source.cpp:3223`). Probe 9 (LA0, one start, no copy)
read chats 33.5, 27.3, 24.6 against DR1 and DR2's 31.1 to 32.2, 25.1 to 25.7, 23.3 to 23.5, inside the spread, and its thread
takes 38 to 43 ms of CPU a window (measured). io prefetch skips 5,976 to 9,746 cached experts a request. The exact mincore
per CPU miss (`resident_in`, `cached_bytes` with `exact=true`) costs 0.76 to 2.6 ms a window (read from the logs); with the
copy the pool already skips it for copy-held experts (`expert_source.cpp:2138-2141`). DecodeSpares: only if Task 26 saw it
move workers.

**Gate:** 16b Step 5's look-ahead arm (two starts on RE24) not slower in decode. The mincore part only if Task 26 shows `jobs`
at 1 ms or more a window. The spares part only if Task 26 saw worker moves.

**Files:** `src/program/generate.cpp` (:6895-6931, :7042-7060, :15441-15459), `src/core/expert_source.cpp` (:941-963
`resident_in` and `resident_sampled`, :1000 `cached_bytes` and its callers at :1288, :2108, :2141, :2155, :4475 that the
change touches; :3035-3056, :3153-3277, :3223 the look-ahead), `include/strata/core/expert_source.hpp`. Create
`include/strata/core/residency_rule.hpp`, `tests/core/residency_rule_test.cpp`. Modify `CMakeLists.txt`, `docs/DETAILS.md`.

**Interfaces:**
```cpp
// include/strata/core/residency_rule.hpp
struct ResidencyRule { int64_t every = 64; bool due(int64_t request) const; };
// include/strata/core/expert_source.hpp
bool FileExpertSource::all_resident(const std::function<bool(int32_t, int32_t)>& gpu_held, bool resample);
```
- Start line: `strata generate: every expert is in RAM, so the decode look-ahead and io prefetch stay off (checked again every
  64 requests)`. They start again when a sample fails or a measured request reads over 100 MB from the drive.
- If its gate passed: DecodeSpares off for one NVIDIA card with the AMD tier. `STRATA_DECODE_SPARES=auto` keeps the probe.
  `STRATA_LOOKAHEAD=1` forces the look-ahead on.

- [ ] **Step 1:** Read 16b Step 5 and Task 26's `jobs` and worker moves. Check the gates.
- [ ] **Step 2: Write the failing test** `residency_rule_test`: samples on requests 0, 64 and 128, a failed sample turns the
  helpers back on, the fast path only while the last sample held. Expected: FAIL.
- [ ] **Step 3:** Implement the parts that passed their gates. Build both trees. Run the ctest set plus the test by name.
- [ ] **Step 4:** Decode identity. Then the A/B (decode timing on one extra start: `jobs` ms and the look-ahead thread's CPU
  time from `/proc`). Append. Commit and push.

**Expected (estimate):** decode +0 to +8%, a core back, a smaller decode spread.

**Keep rule:** identical answers, decode KEEP (or by counter acceptance: CPU ms per window lower with the A/B not slower, the
controller rules), prompts not slower, measured requests' drive reads not above PREV's.

**Commit:** `decode: while every expert is in RAM, no router look-ahead or io prefetch beside the AMD tier`

### Task 29: (Conditional) The CPU expert row loops prefetch ahead (bit for bit)

**Gate:** Task 26's bench shows per-expert time at E = 1 to 4 above 1.2x the time to read 4.43 MB at the bench's own read rate.

**Why (read from the code and measured rates):** Q6_K gate and up rows call ggml's vec_dot per row (`native_expert.cpp:359-371`),
Q8_0 down rows the same (:415-423). The pool reads about 25 GB/s while it computes (probe 7) against the 52 to 68 GB/s the
RAM gives 8 to 16 threads (Task 16 memory probe). With the copy it is the decode bound.

**Files:** `src/kernels/cpu/native_expert.cpp` (:359-371, :415-423, the zero-copy path), `src/kernels/cpu/pool_native_bench.cpp`
(`--prefetch D`), the bench's argument test.

**Interfaces:** none outside the kernel. The bench gains `--prefetch D`.

Sketch:
```cpp
// the row D ahead, so its pages start loading early
_mm_prefetch((const char*) gr + D * f.gu_row + c, _MM_HINT_T0);
```

- [ ] **Step 1:** Check the gate.
- [ ] **Step 2: Write the failing tests:** the bench parses `--prefetch D` and D = 0 runs the old path; a
  `native_expert_parity_q6_K_q8_0` case runs with prefetch on and stays bitwise. Expected: FAIL.
- [ ] **Step 3:** Bench arms D = 0, 1, 2, 4 rows. Pick D.
- [ ] **Step 4:** Implement. `native_expert_parity_q6_K_q8_0` stays bitwise on the box build (the box links
  `third_party/llama.cpp` at 2e44008, not the CMake pin).
- [ ] **Step 5:** Both builds, ctest with the parity test by name, decode identity, the A/B comparing the first chat's CPU
  field (fixed routing: 8.94 experts, 12.55 entries a layer-window). Commit and push.

**Expected (estimate):** 0 to +20% on chats. **Keep rule:** decode KEEP, prompts not slower, identity holds.

**Commit:** `CPU experts: the Q6_K and Q8_0 row loops prefetch the rows ahead (outputs unchanged)`

### Task 29b: (Gated on Task 26's bench) A faster Q6_K and Q8_0 expert kernel for the CPU pool in decode

**Gate:** Task 26's bench shows the pool's per-expert time at E = 1 to 4 above 1.5x the time to read 4.43 MB at the
bench's own read rate, after Task 29.

**Why (measured):** with the copy, the CPU pool is the decode bound: 26.5 to 30.1 ms of a 54 to 56 ms chat window (RE24).
The pool reads about 25 GB/s while it computes, against 52 to 71 GB/s the RAM gives (Task 16 and 16b). Read from the code:
it calls ggml's per-row vec_dot (`native_expert.cpp:359-371, 415-423`), one activation row at a time, so a verify window of 2
to 5 positions reads each expert's rows once per position.

**Files:** `src/kernels/cpu/native_expert.cpp`, a new AVX-2 kernel file beside it, `pool_native_bench.cpp`, the
`native_expert_parity` tests.

- [ ] **Step 1:** In the bench: time per expert for 1 to 5 positions with today's kernel, against a kernel that reads each
  weight row once for all positions of the window (the same q8 dot products in the same order, so bit for bit).
- [ ] **Step 2: Write the failing test:** the new kernel's outputs are bitwise equal to today's for 1 to 5 positions on
  Q6_K and Q8_0 rows. Expected: FAIL.
- [ ] **Step 3:** Implement. If bitwise equality cannot hold, stop and report the cost and the rounding change.
- [ ] **Step 4:** Both builds, ctest, decode identity, the A/B on protocol starts (chats, long answers).

**Expected (estimate):** with 2 to 3 positions a window, the rows read once in place of 2 to 3 times: the pool's time down by a
third or more, chats +10 to +25%.

**Keep rule:** identical answers, decode KEEP, prompts not slower.

**Commit:** `CPU experts: one pass over each weight row serves every position of a verify window`

### Task 30: Config sweeps: verify width, the draft head's bytes, a frozen learned start profile, and resident KV cells

**Why:** chat0 runs at T 3.84 of the cap of 4 and keeps 3.37 tokens, chat1 and chat2 keep 2.0 to 2.2 (Task 16). Probe 10
(one start each, no copy): `--spec 5` chats 36.3, 27.5, 23.3; `--spec 3` 29.7, 28.1, 25.6; `--spec-min-p 0.7` 31.1, 26.8,
26.1, no clear winner. Read from the code: `--spec` defaults to 0 and `--spec-min-p` to 0 (`generate.cpp:661-663, 712`),
and with the suffix drafter on (`suffix_draft = 3`, `:784`) `--spec N` becomes an MTP cap of N and a verify cap of min(N+2,
8) (`:3255-3257`). So `--spec 5` and 6 verify up to 7 and 8 tokens. The draft takes 2.1 to 2.5 ms of a 54 to 56 ms window
with the copy (RE24), and C14's options (`--draft-vocab en`, `--mtp-q4`) exist. The first 4 windows after a start sent
38.6% of entries to the CPU and it took about 3,400 copies to reach 12% (Task 16 logs). `--kv-resident 32768` gave the
3090 717 more slots (KV32, one start, chats inside the spread, 8K 627) and keeps `--max-context 262144`. The user's config
sets `--spec 4 --spec-min-p 0.5` explicitly, so an engine default change would not reach it.

**Gate:** 16b Step 5 read. Arms it settled are skipped.

**Files:** configs in `~/bench-mixed`. No engine default changes for the user's own flags. Only if another default changes:
`src/program/generate.cpp` and its parse test.

- [ ] **Step 1:** Spec arms, one start each, chats with decode timing: `--spec 5`, `--spec 6`, `--spec 5 --mtp-max-t 4`,
  `--spec-min-p 0.3`, `STRATA_SPEC_COST=1`. The best against the head on protocol starts. The text can differ through
  placement: note it.
- [ ] **Step 2:** Draft head (C14), one start each with acceptance by depth: `--draft-vocab en`, `--mtp-q4`.
- [ ] **Step 3:** Profile: one warm run saving a profile (`expert_profile_save`) on chats outside the bench set, frozen, then
  the head against the head with `--expert-profile <frozen>` on protocol starts, requests in a fixed order. Read the CPU share
  per request in order.
- [ ] **Step 4:** `--kv-resident 32768` against none on protocol starts: chats, 8K and 20K prompts, long answers, and the 3090's
  free VRAM after the first one-chunk prompt (it buys back the room Task 21's floor may take).
- [ ] **Step 5:** Record every value with its numbers for Task 35's proposal. Only a default outside the user's flags (for
  example the profile behavior) changes, on KEEP: a failing parse test, the change, both builds, ctest, commit.

**Expected (estimates):** width 0 to 8% on chats, the draft head 0 to 3%, the profile faster on the first requests after a
start, KV +1 to +4%.

**Keep rule:** a default changes only on KEEP with the other side not slower. Values for the user's flags go into the
proposal as open choices with their numbers.

**Commit (only if a default changes):** `decode: defaults from this box's sweeps`

### Task 31: (Gated) Cheap adaptive copies: rotation by default and W6800 fills from a pinned ring

**Why:** with the copy, a 3090 swap is about 0.2 ms (RE24, join 5.2 and 4.5 ms for 22 and 23 swaps). Read from the code: the
copy-backs are already queued with one wait per stream (`generate.cpp:383-411`, spec correction 14). What is left per swap is
that wait before `stage_exchange` and `commit_exchanges`' memcpy of `out` into the copy on the host thread before the next
window (`expert_source.cpp:4291-4320`, the memcpy at :4311). `STRATA_EXCHANGE_ROTATE=1` skips that memcpy when the copy is
whole and pinned (`expert_source.cpp:4194`). The NUMA split reads the same env to refuse itself (`pool.cpp:633`). W6800
fills read the copy with no assembly, but HIP stages them because the copy is CUDA-pinned, not HIP-registered
(`xtier_lib.cu:999-1011`).

**Gate:** Task 26's numbers on the current head: the join at 5 ms or more a window in chats, or a W6800 fill at 0.3 ms or
more. RE24 measured 4.5 to 5.2 ms, so this may stop here. Otherwise stop and record.

**Files:**
- Modify: `src/core/expert_source.cpp` (:4174-4243 reserve and rotation, :4251-4323 stage and commit),
  `include/strata/core/expert_source.hpp`, `src/program/generate.cpp` (:355-421, :10393-10421, :10526-10700, near :10305
  the ring), `src/core/peer_experts.cpp` (:479-531), `include/strata/core/peer_experts.hpp`, `src/xtier/xtier_lib.cu` (:1009
  the comment)
- Modify: `src/kernels/cpu/pool.cpp` (:629-640, the NUMA split's refusal reads a rotation flag passed in from
  `FileExpertSource::rotation_eligible()`, not the env), `include/strata/kernels/cpu/numa_pool.hpp`,
  `src/kernels/cpu/numa_pool_test.cpp` (:120), `src/kernels/cpu/pool_tasks_test.cpp` (:220-231, :338-346, the NUMA case
  keeps refusing rotation), `tests/core/file_expert_source_test.cpp` (:733-735, :823-825, :912-914),
  `include/strata/core/nv_defaults.hpp` (:7 comment)
- Create: `include/strata/core/swap_ring.hpp`, `src/core/swap_ring.cpp`, `tests/core/swap_ring_test.cpp`
- Modify: `tests/core/exchange_storage_test.cpp`, `tests/core/xtier_device_test.cpp`, `CMakeLists.txt`
- Docs: `docs/EXCHANGE_ROTATION.md`, `docs/DETAILS.md`, `docs/MULTI_GPU.md` (the env's mentions)
- If Task 23 was not kept (A0): `src/core/expert_source.cpp` (`blob_parts` guard :4546), a library export
  `strata_xtier_fill_parts` (ABI bump), `src/core/xtier.cpp`, `include/strata/core/xtier_abi.hpp`

**Interfaces:**
```cpp
// include/strata/core/expert_source.hpp
bool FileExpertSource::rotation_eligible() const;   // a whole pinned copy rotates with no env
// include/strata/core/swap_ring.hpp
class SwapRing { public: bool open(core::xtier::Device& d, int64_t entries, uint64_t entry_bytes, std::string& err);
                 uint8_t* entry(int64_t i); int64_t entries() const; };
// include/strata/core/peer_experts.hpp
bool PeerExperts::fill_from(int64_t slot, const uint8_t* host, uint64_t bytes, std::string& err);
```
- The NUMA split learns that rotation is on from the source at pool start: `decide_real_numa_split` takes the flag.
- The ring is SharedHost kind C memory (`hipHostMalloc`, then `cudaHostRegister`, `xtier.cpp:501-545`), one entry per W6800
  swap of a round, written only in round r. `apply_pending(true)` waits on `refill_done` before the next window, so round r+1
  finds it free. Never use the source's stage buffers as async sources (#1237).
- `STRATA_EXCHANGE_ROTATE` goes.

- [ ] **Step 1:** Check the gate on Task 26's lines.
- [ ] **Step 2: Write the failing tests:** `exchange_storage_test`: a whole pinned complement turns rotation on with no env,
  a partial one keeps the copy path. `numa_pool_test`: the split refuses when the flag passed in is on (the message names
  rotation, not the env). `swap_ring_test`: every pick gets a distinct entry, none above capacity. `xtier_device_test`: a fill
  from `hipHostMalloc` memory lands (read back). Expected: FAIL.
- [ ] **Step 3:** Implement rotation by default (the commit memcpy goes), the refusal's flag, and the W6800's fills from the
  ring when Task 26 showed them at 0.3 ms or more. Update the tests that set the env.
- [ ] **Step 4:** Build both trees. Run the ctest set plus `exchange_storage_test`, `pool_tasks_test`, `numa_pool_test`,
  `swap_ring_test`, `xtier_device_test` and `file_expert_source_test`.
- [ ] **Step 5:** Identity with adaptation on (per G-ID), plus `STRATA_VERIFY_COPY=1` over a 20-minute adaptive chat session:
  0 mismatches.
- [ ] **Step 6:** A/B, protocol, decode timing on one extra start each (join per round, copy ms per card, `adapt wait`).
  Docs. Append. Commit and push.

**Expected (estimates):** join from about 5 ms a window to 2 to 3 ms. Chats +3 to +6% over the head.

**Keep rule:** decode KEEP (or counter acceptance on join ms per window), prompts not slower, identity or checks clean, 0 copy
mismatches.

**Commit:** `adaptive tier: a swap is a DMA from the pinned copy with its slots rotated, and the AMD card fills from a pinned ring`

### Task 32: (Gated) The adaptive swaps per round from measured costs, on both cards

**Why (measured):** without the copy, `--adapt-swaps 16` gave chats +26 to +29%, the 8K answer +22%, the 20K answer -7%
(Task 16 250 W, two starts), and `--adapt-swaps 0` dropped the 20K answer from 48 to 29 tok/s. With the copy, RE24AS was no
better than RE24 (one start). Read from the code: the 96 default dates from about 0.1 ms swaps, the W6800's gain is a
hard-coded 1.5 (`peer_adapt.cpp:35`, candidate floor 2.0 at `:24`), and `--peer-adapt-swaps` follows `--adapt-swaps`
(`generate.cpp:10681`). Defaults: `adapt_every = 4` (`:697`), `adapt_decay = 0.7` (`:702`).

**Gate:** after Task 31, the join is still 5 ms or more a window in chats, or an `--adapt-swaps 16` arm on that head beats the
default beyond the spread. Otherwise stop and record. Task 27's adaptation-on decode KL repeats (KL 0 head against head).

**Files:** `src/program/generate.cpp` (:10526-10552, :10674-10684, the decode sums near :14455 and :15473-15486),
`src/core/peer_adapt.cpp` (:10-41), `include/strata/core/peer_adapt.hpp`, `src/core/peer_experts.cpp` (:479-484). Create
`include/strata/program/adapt_budget.hpp`, `tests/program/adapt_budget_test.cpp`. Modify `tests/core/peer_adapt_test.cpp`,
`CMakeLists.txt`.

**Interfaces:**
```cpp
// include/strata/program/adapt_budget.hpp
struct AdaptCosts { double copy_ms[2] = {0, 0}; double cpu_ms = 0; int64_t seen = 0;
                    void add_copy(int card, double ms, int64_t n); void add_cpu(double ms, double experts); };
/// The smallest usage gain a swap on card c needs to pay for its copy over the horizon, or a negative value before any measurement.
double adapt_min_gain(const AdaptCosts& k, int card, double decay, int every, double horizon_windows);
// include/strata/core/peer_adapt.hpp
std::vector<PeerSwap> peer_adapt_choose(const float* usage, const int32_t* other, const int32_t* mine,
                                        const std::vector<std::vector<int32_t>>& free_slots_by_layer,
                                        int64_t n_layers, int64_t n_expert, int max_swaps, float min_gain);
```
- Rule: keep swap i, in gain order, while gain_i x (1 - adapt_decay) / adapt_every x c_cpu x H >= c_copy[card], with
  H = min(16, remaining max_new / tokens per window). So the break-even gain is c_copy x every / ((1 - decay) x c_cpu x H).
  `--adapt-swaps` stays as a cap (96), `--adapt-min-gain` as a floor. Before any measurement, today's rule runs. One line per
  request: copies kept per round per card, c_copy, c_cpu, the break-even gain.

- [ ] **Step 1:** Check the gate.
- [ ] **Step 2: Write the failing tests:** `adapt_budget_test`: copy 1.1 ms, CPU 0.1 ms, H 16, decay 0.7 and every 4 give
  1.1 x 4 / (0.3 x 0.1 x 16) = 9.17; copy 0.25 ms gives 2.08; no measurement gives a negative value. The arithmetic goes in
  the test's comment. `peer_adapt_test`: min_gain 4 drops a candidate 1.5 keeps. Expected: FAIL.
- [ ] **Step 3:** Implement. Build both trees. Run the ctest set plus both tests by name.
- [ ] **Step 4:** `STRATA_ADAPT_CHECKRES=1` and `STRATA_XTIER_CHECK=1` on one start: 0 mismatches. The decode KL with
  `--allow adapt` against PREV, two starts: it passes Task 27's gate.
- [ ] **Step 5:** Arms PREV, PREV with `--adapt-swaps 16`, NEW, protocol starts, decode timing on one extra start each.
  Append. Commit and push.

**Expected (estimate):** chats +0 to +10% over Task 31's head, the 20K answer not slower.

**Keep rule:** chats KEEP against both PREV and its `--adapt-swaps 16` arm, long answers and prompts not slower, the decode
KL passes.

**Commit:** `adaptive tier: the swaps per round come from the measured cost of a copy against the CPU's measured cost per expert, on both cards`

### Task 33: (Conditional) The adaptive round off the window's critical path

**Why (read from the code):** the round is joined after the draft every 4th window (`generate.cpp:15366-15367, 15420-15425`).
`--adapt-async` is refused with `--peer-device` (`:8213-8222`). Its arena flow states the invariant: at a tick nothing in flight
reads a slot (`:10749-10753`; the commit's part rests on a comment).

**Gate:** after Tasks 31 and 32, the join is above 10% of a chat window on two starts.

**Files:** `src/program/generate.cpp` (:8213-8222, :10706-11060, the JobThread :1634-1700), `src/core/peer_experts.cpp`,
`include/strata/core/peer_experts.hpp`, `tests/core/peer_adapt_test.cpp`.

**Interfaces:**
```cpp
// include/strata/core/peer_experts.hpp: adapt() stays for the blocking path
std::vector<PeerSwap> choose(const float* usage, const int32_t* other, float min_gain, int max_swaps);
void evict(const std::vector<PeerSwap>& s);
bool fill(const std::vector<PeerSwap>& s, std::string& err);
void admit(const std::vector<PeerSwap>& s);
```

- [ ] **Step 1:** Check the gate.
- [ ] **Step 2: Write the failing test:** `peer_adapt_test` drives choose, evict, fill and admit over two rounds, and `res_`
  and the free slots match the blocking `adapt` for the same usage.
- [ ] **Step 3:** Implement: at the boundary choose and mark evictions, post the byte work to the JobThread pinned off CPU 0,
  wait and admit at the next boundary. Lift the refusal for the peer with the copy. Build both trees, ctest with the test by
  name.
- [ ] **Step 4:** NEW against NEW gives the same text with adaptation on. `STRATA_ADAPT_CHECKRES=1` and
  `STRATA_XTIER_CHECK=1` with 0 mismatches over long chats. Decode identity with `--adapt-swaps 0`.
- [ ] **Step 5:** The A/B. Append. Commit and push.

**Expected (estimate):** the join near 0, chats +5 to +10% over the head.

**Keep rule:** deterministic, 0 mismatches, chats KEEP, the rest not slower.

**Commit:** `adaptive tier: with the AMD tier the round's copies run beside the next window and land at its boundary`

### Task 34: (Gated) The W6800's decode round trip: rows straight into mapped memory and one replayed launch

**Why (measured, Task 16 T1 with XTIER timing, no copy):** per layer-window, launch 0.041 to 0.063 ms (2.0 to 3.0 ms a
window), copy-out 0.018 to 0.039 ms (0.9 to 1.9 ms), card time 0.18 to 0.25 ms, finish wait 0.056, 0.085 and 0.098 ms in the
8K, 20K and 64K answers. In the 20K answer the CPU leg was 0.453 ms a layer against the W6800's launch-to-done of about 0.47
ms, so the round trip was the floor of long answers. Read from the code: two host memcpys, two H2D copies, a quantize, a
grouped kernel, a D2H and an event (`xtier_lib.cu:1048-1085`), launched on the host thread (`peer_experts.cpp:330-471`). The
export is `strata_xtier_decode(void* h, const strata_xtier_fmt* f, const float* x, int64_t n_tok, const strata_xtier_meta*
m)` (`xtier_lib.cu:1048`). There is no decode args struct. The ABI is at version 6 (`xtier_abi.hpp:8`), and its size check
(`:87`) sums `strata_xtier_meta` with the other structs.

**Gate:** after Tasks 27 to 33, Task 26's line with `STRATA_XTIER_TIMING=1` shows launch plus finish wait plus copy-out at 3 ms
or more a window in the 8K or 20K answer.

**Files:** `src/xtier/xtier_lib.cu` (:1048-1105), `include/strata/core/xtier_abi.hpp`, `src/core/xtier.cpp`,
`include/strata/core/xtier.hpp`, `src/core/peer_experts.cpp` (:330-471), `src/core/expert_source.cpp` (:4973, :5178-5198),
`src/kernels/xtier_decode_parity.cpp`.

**Interfaces:**
```c
/* a new export beside strata_xtier_decode, ABI version 7: the rows land at out_host, a buffer both cards map */
int strata_xtier_decode_out(void* h, const strata_xtier_fmt* f, const float* x, int64_t n_tok,
                            const strata_xtier_meta* m, void* out_host);
```
- The rows land in SharedHost kind C memory (`hipHostMalloc` then `cudaHostRegister`), so `PeerExperts::finish` copies no rows
  when the new export is used. The fixed sequence replays as one HIP graph per window shape. `STRATA_XTIER_DECODE_GRAPH=0`
  keeps the old path until Task 35. `strata_xtier_meta` is unchanged.

- [ ] **Step 1:** Check the gate.
- [ ] **Step 2: Write the failing test:** `xtier_decode_parity` with `strata_xtier_decode_out` gives rows bitwise equal to the
  copy-out path, and a graph replay gives the same rows as direct launches for windows of 1 to 6 tokens. Expected: FAIL.
- [ ] **Step 3:** Implement and bump the ABI. Build both trees. Run the ctest set plus `xtier_decode_parity` and
  `xtier_device_test`.
- [ ] **Step 4:** Decode identity. The A/B with XTIER timing on one extra start each. Append. Commit and push.

**Expected (estimates):** long answers +6 to +12%, chats +2 to +4%.

**Keep rule:** identical answers, long answers or chats KEEP, the rest not slower.

**Commit:** `AMD tier decode: rows land in host memory both cards map and the fixed launch sequence replays as a graph (ABI bump)`

---

## Wrap-up

### Task 35: The final measurement against `ad9157b4`, the switch cleanup, and a Q6_K service config proposed to the user

**Why:** each task was measured against its predecessor, and the levers interact. The user asked for a Q6_K config for this box,
proposed and not applied. `strata.service`'s config serves IQ4_XS on four cards, and its ExecStartPre fails on the missing
3080 Ti after it sets the 3090's lock.

**Files:** removals in `src/program/generate.cpp`, `src/prefill/*`, `src/xtier/xtier_lib.cu` (switches whose old path lost and
no test needs: for example `STRATA_PIN_ALLOC`, `STRATA_MMQ_STREAMK`, `STRATA_XTIER_DECODE_GRAPH`, `STRATA_XTIER_TAIL_SKIP`,
`STRATA_REFILL_PARTS`); `docs/DETAILS.md` (the switches removed). Box: `~/bench-mixed/strata-swift-q6k-proposed.json` and a
note.

- [ ] **Step 1:** List every switch Tasks 17 to 34 added or kept, with its winner.
- [ ] **Step 2:** Rebuild `~/source/strata-prev` at `ad9157b4` (both trees, under the lock, with the Build PREV command).
  Its configs are `cfg-q6k2-prev.json`, `cfg-q6k2-re24-prev.json` and `cfg-q6k3-prev.json` (`--prefill auto`).
- [ ] **Step 3:** The final A/B on protocol starts, every arm under `prlimit --memlock=unlimited`: `ad9157b4` at `cfg-q6k2`
  (chunk 4096), at `--prefill auto` and at RE24 as flags, against the final head at `cfg-q6k3`. 2K, 8K, 20K, chats, long
  answers, start time, RAM locked, drive reads. KL v2 on two starts and the decode KL if a decode lane landed.
- [ ] **Step 4:** Remove the losers' switches. Keep operator switches (`--no-resident-experts`, `--prefill N`,
  `--peer-reserve-mib`) and test hooks (`STRATA_VERIFY_COPY`, `STRATA_TEST_XTIER_RING_FAIL`, `STRATA_XTIER_F16_CHECK`). Both
  builds, the ctest set, the multi-stage unit tests (`tier_planner_test`, `split_lend_complement_test`, the halves tests),
  and smoke runs of the NVIDIA-only engine (one chat) and the HIP-only engine (one chat, one 4K-word prompt).
- [ ] **Step 5:** One start of the final head after the removals, against Step 3's spread.
- [ ] **Step 6:** Write the proposed config: the user's flags unchanged (`--max-context 262144`, `--kv int8`, `--mtp rt --spec
  4 --spec-min-p 0.5`, `--pool-workers 15`, `--mmap-experts`, `--expert-cache auto`, `--peer-device hip:0`), `--prefill
  auto`, the conversation cache at its service default, `expert_profile_save`, and only the kept settings that are not
  defaults. Start it once (one chat) to prove it starts. Never touch the unit or the service config.
- [ ] **Step 7:** The note for the user: measured numbers against the Phase 2 baseline and RE24, the start time, RAM locked,
  the unit's ExecStartPre for this box (`nvidia-smi -i 0 -pl 350`, `-lgc 0,1500`), LimitMEMLOCK (already infinity), the open
  choices with numbers (a measured `--spec` or `--spec-min-p`, `--kv-resident`, the draft head options, the tail skip if not
  a default, a shorter context), and what is not tested. Append. Commit and push.

**Box stint:** about 90 min. **Expected (RE24 plus the tasks' estimates):** 20K about 2,500 tok/s or more against 746 to 785,
8K about 1,750 or more against 719 to 754, 2K about 590 against 360 to 387, chats +40 to +70% or more, and a warm start
under 60 s against RE24's 130 s.

**Keep rule:** prompts and decode each KEEP against `ad9157b4` at `cfg-q6k2`, not slower than `ad9157b4` at RE24, KL passes on
two starts. A removal that moves a number beyond the spread is reverted.

**Commit:** `phase 2: the switches the old paths no longer need are removed after the final measurement`

### Task 36: Docs for this box: MULTI_GPU.md and DETAILS.md with Phase 2's measured numbers

**Why:** Phase 1's docs describe a four-card, 62 GB box that no longer exists. `docs/MULTI_GPU.md` also holds uncommitted
lines from Phase 1 (`git status`: `M docs/MULTI_GPU.md`). Phase 2 changed defaults and meanings on one NVIDIA card with the
AMD tier (`--peer-reserve-mib`, `--prefill auto`, the pinned copy and `--no-resident-experts`, the PCIe share, decode lanes,
the swap rule, rotation).

**Files:** `docs/MULTI_GPU.md`, `docs/DETAILS.md`, `docs/EXCHANGE_ROTATION.md`, `strata_cli/launch/params.py` or
`docs/TERMINAL_LAUNCH.md` if a changed flag is described there.

- [ ] **Step 1:** List every default and flag Phase 2 changed (git log of Tasks 17 to 35). Decide the uncommitted MULTI_GPU.md
  lines: fold in or drop, and say which.
- [ ] **Step 2:** Rewrite the sections in plain words and short sentences: how a prompt is read (one chunk, the W6800's room,
  the copy), how decode works (the CPU pool, copies, swaps, lanes). Every number names its box (RTX 3090 at 1500 MHz and 350
  W, W6800, six DDR4-2133 DIMMs, EPYC 7F52), commit, config and starts. Read-from-code claims are marked. Old-box numbers are
  labeled as such or removed.
- [ ] **Step 3:** Check no sentence describes a removed switch (`tier_room`, `STRATA_EXCHANGE_ROTATE`, and the ones Task 35
  removed). Run `python -m pytest tests/terminal` if `strata_cli` text changed.
- [ ] **Step 4:** Commit and push.

**Keep rule:** every number traces to a `plan-results.md` line and its log.

**Commit:** `docs: the AMD tier on one RTX 3090 with a W6800 and Q6_K, as Phase 2 left it, with this box's measured numbers`
