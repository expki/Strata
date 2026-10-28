# Faster NVIDIA + AMD serving that holds up when most experts live in RAM

Design, 2026-10-25. Follows `2026-10-24-mixed-make-expert-tier-design.md` (the AMD expert tier, branch
`amd-expert-tier`). Source of the plan for this project.

## Why

The AMD expert tier works: on the test box Swift 1.5 IQ4_XS decodes at about 90 tok/s with the W6800 against 34 tok/s on
the three NVIDIA cards alone, and reads a 16K prompt at about 1,480 tok/s against 526. Profiling on 2026-10-25 found
where the rest of the time goes (numbers below). The user wants every optimization path that profiling found built.

**The changes must be generic.** The user will add RAM and run Swift 1.5 Q6_K (157 GB) on the same four GPUs. Then
the GPUs hold about a third of the experts and RAM holds the rest, so the CPU pool and the PCIe links carry far more of
the work than today. A change tuned to today's box (for example fewer pool workers) can turn into a loss there.

**The fork.** expki/Strata is now a hard fork. Nothing goes upstream. Breaking changes are welcome where they make the
engine better or remove technical debt: a better behavior becomes the default, and switches it makes redundant are
removed rather than kept beside it. Supported platforms are Linux with CUDA and HIP. Windows, Mac (Metal) and SYCL stay
in the tree but this project does not keep them building.

## The box, the target and the measurements

Test box: `ssh vdh@192.168.10.5`, Ubuntu 24.04, EPYC 7F52 (16 cores, SMT off), 62 GB RAM, models on NVMe (LVM over a
980 PRO, a 970 EVO Plus and a 1 TB drive). RTX 3090 24 GB, RTX 3080 Ti 12 GB, RTX 3050 8 GB (CUDA, layer split
3050 layers 0-2, 3080 Ti 3-13, 3090 14-47) and Radeon PRO W6800 32 GB (`--peer-device hip:0`). The live service and
its config are described in the memory notes and `~/bench-mixed/` on the box.

Target: the same GPUs with more RAM (amount not fixed yet) and
`/opt/models/ukisai_Swift-1.5-Qwen3.8-Flash-Next-Q6_K-0000{1..5}-of-00005.gguf` (157 GB, downloaded). No pack exists
for it yet. With 62 GB of RAM today it runs from the drives, so it can show that things work, not how fast.

What profiling found (IQ4_XS, 20K-token prompt, 250 W caps, details in `~/bench-mixed/perf1` to `perf5`):

| Where | Finding |
|---|---|
| CPU placement | With `--pool-workers 15` on 16 cores, the host thread, every stage's prompt loop, the 96 stager threads and the adapt/fill threads all run on CPU 0. The stage loops' `cudaStreamSynchronize` returns 10 to 16 ms late (nsys), so the 3090 idles about 9 ms per layer. `--pool-workers 7` with `STRATA_AUX_CPUS=auto` took the 20K prompt from 1,491 to 1,792 tok/s, decode unchanged. |
| Prompt, after that | The 3090 waits about 640 ms of each 1.65 s chunk for the W6800's sums. The W6800 computes about 65% of the rows: about 17 ms of kernels and 3.4 ms of PCIe copies per layer, then idles 14 to 18 ms per layer, because a layer's rows reach it only after the 3090 finishes that layer's attention. |
| Prompt, the W6800's reach | The W6800 computes only experts it holds. Experts in RAM go to the NVIDIA cards over PCIe. |
| Decode at 20K context | 29.4 ms per window: NVIDIA compute about 19.7 ms, waiting for the W6800 4.7 ms (0.18 ms round trip per layer: launch 0.03, copies in 0.02, kernels 0.09, copy out 0.02), draft and commit 2.9 ms. |
| Measurement trap | `--prefill auto` picks the prompt chunk at engine start from the free page cache (1,024, 2,304 or 4,096 tokens seen on this box). Small chunks are much slower, so A/B runs must fix the chunk. |

## Goals

1. Prompts and decode faster on today's box and in a "most experts in RAM" setup, without per-box tuning by the user.
2. Swift 1.5 Q6_K runs with the AMD tier: a pack, decode and prompts on all four GPUs.
3. Every split of work between devices is decided from what the engine measures or reads at run time (core counts,
   free memory, link speeds, measured kernel times), never from constants fitted to this box.
4. Less technical debt: one mechanism where there are several overlapping switches today.

## Non-goals

- Windows, Mac, SYCL, the Tauri app, `strata launch` and the installer.
- Decode draft acceptance (users' text accepts 46 to 53% of drafts, the bench prompts about 70%). Not tier work.
- The user's clock locks and power caps. They stay as they are.
- A decode-time planner. Decode keeps today's split: NVIDIA hits, the W6800's held experts, the CPU pool for the rest.

## Anti-goals

- A gain on today's box that is a loss with most experts in RAM, or the reverse.
- Settings the user has to tune per box to get the speed.
- Wrong answers. The bar (builder's call, from the review):
  - A change that only moves data or threads (stage 1, stage 4): the parity tests keep passing, and greedy answers on the
    `ab_engine` prompts with the deterministic flags are identical to before (`--identity`).
  - A change that splits the math differently (stages 2 and 3): the parity tests keep passing, an exact match (KL 0)
    where an exact reference path exists, and against the unchanged path on a fixed text set (`quant_quality.py
    prompt`, 30 prompts, a long set crossing a 4096-token chunk) on the all set a mean KL of the next-token distributions
    at or below 0.03 (the long set's mean too), a median at or below 0.005 and top-1 agreement at or above 90%.
  - Amended 2026-10-26 after Task 8 (the first bar was a mean KL of 0.004, the repo's bar for the CPU prompt share):
    on this set every valid rounding change measured 0.010 to 0.020 mean KL (the tier's FP16 path against the stage's
    FP32 path, FP16 against FP32 transfers on one split), carried by a few sensitive prompts, with all set medians of
    0.0013 to 0.0032 and top-1 agreement of 93 to 100%. The bounds sit just outside that envelope.
    A wrong expert or a broken kernel moves the median and top-1, which rounding does not. The stage computing every
    tier-held expert matched the all-stage path exactly (KL 0).
- Long benchmark runs. About 5 minutes per arm, as before.

## Approaches considered

1. **One project, in stages, each measured (chosen).** Build the measurement first, then the changes in order of
   expected gain and risk. Each change stays only if it is faster beyond noise in both measurement setups (or faster in
   one and neutral in the other). Separate commits.
2. **The prompt overlap first**, because it restructures the prompt path the planner plugs into. Rejected: the riskiest
   work first and the safe gains last.
3. **One fixed switch per path** (a row share, a W6800 streaming switch). Rejected: every box would need hand tuning,
   which is the opposite of goal 3.

The user chose 1.

## Stage 0: measurement for both setups

- **Normal setup:** today's config with `--prefill 4096` fixed.
- **RAM-heavy setup (builder's call, from the review):** Swift 1.5 IQ3_S (`/opt/models/...GSQ-RCO-IQ3_S-*`, pack
  `swift-iq3_s` exists, 46.8 GiB of experts) with each GPU's expert cache capped so that about a third of the experts
  sit on GPUs, as with Q6_K on bigger RAM. That leaves about 31 GiB in RAM, which fits today's page cache with the
  box's other services running. IQ4_XS would not fit (about 41 GiB plus other file pages). If no existing flag caps
  each card's expert memory (each NVIDIA stage and the AMD tier), add one, in GiB per card. It is a real feature of the
  fork, not a test hack.
- **Where RAM experts come from (builder's call):** both sources the engine has today must work and be measured where
  they fit: the page cache (`--mmap-experts`, copied by the stager threads) and a pinned RAM copy
  (`--resident-budget-gib`, copied to a card directly). The RAM amount of the upgrade is not fixed, so neither is
  assumed.
- **The start-up chunk choice is in scope (builder's call):** `--prefill auto` must pick well without help, or the
  gains measured at a fixed chunk never reach the service. From reading the code (not run): the pick
  (`host_plan_chunk`, `src/program/generate.cpp`) counts only the NVIDIA caches as held, so the W6800's experts are
  counted as read from the drives, which pushes the chunk down. Fix that in stage 0, and teach it the planner's
  streaming in stage 2. A/B arms still fix the chunk, and each stage's result is also checked at the chunk auto picks.
- **Q6_K smoke run:** a pack built with `tools/iq_pack.py`, then one short request through the server with the tier on.
- **Diagnostics stay.** The timing added on branch `perf-diag` (commit `f70980e5`) becomes permanent: the W6800's
  per-layer prompt timing under `STRATA_XTIER_TIMING=1` and the host's per-layer prompt steps under
  `STRATA_PREFILL_TIMING=1`. Its two test switches (`STRATA_XTIER_PF_STAGES`, `STRATA_XTIER_PF_SHARE`) are dropped.
- **Tooling:** `tools/ab_engine.py` runs arms of either setup. Every arm fixes the prompt chunk.

## Stage 1: CPU placement in the engine

**Problem.** The host thread pins itself to the first core, every thread it creates inherits that one core, and
`aux_cpus::pin_current_thread()` does nothing when no CPU is spare. With a full-size pool nothing is spare, so all
helpers share the host's core.

**The rule (decided):**
- The host thread keeps its core.
- Pool workers own their cores while the pool can run: in decode, and in a prompt that uses the CPU prompt share.
- A thread that only works during a prompt without a CPU share may use any CPU except the host's. These are the
  stage prompt loops, the stagers, the PLE gather and anything else that runs only while a prompt is read. In the
  RAM-heavy setup the stagers copy most experts of every chunk, so they need many cores.
- Threads that work during decode (adapt, fill, io, lookahead) keep today's placement unless measurement shows a
  better generic rule. The builder measures this in the RAM-heavy setup, where decode keeps the pool busy.
- The user no longer needs `--pool-workers` or `--aux-cpus` for speed. Keep `--pool-workers` as a count. `--aux-cpus`
  may change meaning or go, whichever leaves the clearer code.

**Done when:** in the normal setup the 20K prompt is at least as fast as the measured `--pool-workers 7` +
`STRATA_AUX_CPUS=auto` result (about 1,790 tok/s) with today's 15 workers, decode does not drop, and the RAM-heavy
setup does not lose.

## Stage 2: one expert planner for prompts

**Problem.** Today each expert's rows go to whoever holds it: the stage GPU (VRAM hit), the W6800 (tier hit), or the
stage GPU by streaming from RAM. Nothing balances the devices, and the W6800 cannot help with experts in RAM.

**The planner (decided):** per stage, per MoE layer of a chunk, it decides who computes each routed expert's rows:

| Choice | Possible for |
|---|---|
| Stage GPU, from its VRAM | experts the stage holds |
| W6800, from its VRAM | experts the tier holds |
| Stage GPU, streamed from RAM | any expert not in the stage's VRAM, the tier's included |
| W6800, streamed from RAM | any expert not in the tier's VRAM |

- It minimizes the layer's finish time: the stage GPU's own work and the W6800's work run at the same time, and the
  combine waits for both.
- The W6800 is shared by all stages, so the planner accounts for work other stages have queued on it.
- Costs come from measured rates the engine keeps up to date while it runs (for example rows per ms per device and
  quant type, bytes per ms per link and per source kind, the handoff's fixed cost). Before enough measurements exist it
  uses safe starting values. No rate is a constant fitted to this box.
- It replaces today's fixed rules for the AMD tier's prompt rows (`STRATA_XTIER_PREFILL`, the tier's use of
  `--peer-prefill-rows`). The CUDA peer (`--peer-device N`) and its fixed stream share (`STRATA_PF_PEER_STREAM`) may
  move onto the planner too if that is cleaner; it must not get slower.
- Scope (builder's call, from the review): the planner's devices are the stage GPU and the W6800. The layer-split
  helper (`STRATA_PREFILL_HELP`) and the CPU prompt share stay as they are. They were not among the paths profiling
  found, and a CPU computes prompt experts far slower than a GPU streams them. Build the planner so another device can
  be added later.
- With a pinned RAM copy, the W6800 copies a streamed expert straight from it when the copy can be registered with HIP,
  and through the stagers otherwise.
- **Known blocker:** handing a tier-held expert to the stage GPU crashed with an illegal memory access first reported in
  `gather_native_group` (the `STRATA_XTIER_PF_SHARE` test, `perf2-SH45-engine.log`). Find the cause before building on
  this path.
- **W6800 streaming from RAM** needs new library support. The engine stages those experts into host memory that both
  runtimes map (as the existing `SharedHost` does), using the existing stager threads, and the library copies them in
  on its own copy stream ahead of the kernels. This changes the library's C ABI (version 3). Breaking the old version is
  fine.

**Done when:** faster than stage 1 in the RAM-heavy setup, and not slower in the normal setup. `xtier_prefill_parity`
covers the new streamed path.

## Stage 3: overlap the W6800 with the stage GPU's attention

**Problem.** Within a stage, a layer's MoE rows reach the W6800 only after the stage GPU has finished that layer's
attention and router, and the next layer's attention waits for the W6800's sums. So each device idles while the other
works.

**The change (decided):** split each chunk's tokens into two halves, A and B, and run them one step apart. While the
W6800 computes half A's experts of a layer, the stage GPU runs half B's attention for that layer. Then the W6800 takes
half B and the stage GPU combines half A and starts the next layer for it. Causal attention allows this: half B's
attention at a layer needs half A's keys and values (and the linear attention's state) for that layer, which exist by
then. Each half gets its own handoff buffers (ping-pong), and the planner plans each half.

- Expected gain: up to the point where the busier device is busy all the time. On today's box that bound is about 50%
  on prompts. Not measured yet.
- It applies wherever a stage's MoE work goes partly to another device. Without that it changes nothing.
- Answers may differ in the last bits from the unsplit path. The parity tests and an answer check must pass.

**Done when:** faster in both setups.

## Stage 4: decode trims for the W6800 round trip

Candidates, each kept only if measured faster (each is roughly 1 to 4% of a decode window today, less in the RAM-heavy
setup):
- one host-to-device copy for the input rows and the plan, not two;
- the kernels read their input from and write their output to mapped host memory, removing the copies;
- fewer kernel launches per layer (fusing the quantize step into the grouped kernel).

`xtier_decode_parity` keeps passing (it compares against the CUDA kernels within a relative L2 error of 1e-3), and
greedy answers stay identical to before (`--identity`). A fusion that changes the W6800's rounding is out.

## Q6_K support

Part of acceptance, checked from stage 0 on:
- a pack for the Q6_K files (`tools/iq_pack.py`; fix the tool if it needs it);
- the CPU pool, the NVIDIA stages and the W6800 (decode kernels and prompt MMQ) handle Q6_K experts. The HIP and CUDA
  trees are built with `STRATA_Q6K_EXPERTS=ON` and `STRATA_MMQ_KQUANTS=ON` today. Verify, add what is missing;
- a smoke run serves a request with the tier on.

## Constraints

- Linux, CUDA and HIP trees on the box. Build both trees for every change.
- Commit and push to origin (expki/Strata), never force-push. Keep `main` clean. Work on a new branch from
  `amd-expert-tier`.
- Code comments: one line, plain, no em dashes, no semicolons, no hyphens joining clauses.
- Never edit files inside the box checkout (`~/source/strata`). Build experiments in the spare worktree
  (`~/source/strata-base`) so the service's binaries stay untouched.
- Box runs: back up and restore the live unit around each stop, cap the 3090 and 3080 Ti at 250 W, about 5 minutes per
  arm, service back afterwards.
- Every gain is measured on the box with the setup and the numbers written down. No claims without a measurement.

## Left to the builder

- Which pinned-copy registration works across the two runtimes (the earlier design measured the variants for
  `SharedHost`).
- How the planner keeps and ages its rates, its starting values, and its exact search (a greedy pass is expected to be
  enough).
- How the per-card expert cap is spelled and where it lives.
- How the overlap is sequenced in code, and which buffers are split in two or doubled.
- Which decode trims survive measurement.
- Whether `--aux-cpus` stays, changes or goes, and the placement of decode-time helper threads, decided by
  measurement.
- The order inside a stage and how many commits it takes.
