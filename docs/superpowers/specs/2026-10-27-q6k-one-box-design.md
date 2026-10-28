# Phase 2: the most speed for Swift 1.5 Q6_K on one RTX 3090 + W6800 box

Design, 2026-10-27. Follows `2026-10-25-mixed-tier-optimizations-design.md` (Phase 1, Tasks 1 to 15d, branch
`mixed-tier-perf`, head `ad9157b4`). It is the source of the plan's Phase 2 section
(`docs/superpowers/plans/2026-10-25-mixed-tier-optimizations.md`). Where the two specs differ, this one wins for Task 16
onward.

How numbers are marked in this spec and the plan:

- **Task 16 (250 W)**: Part 1 of `task-16-report.md`. The 3090 was capped at 250 W, which never bound (it drew 70 to 119 W).
- **Task 16 (350 W)**: Part 2 of the same report: runs A and B (`t16p-a`, `t16p-b`), probes 7 to 10 (run C, `t16p-c`),
  run D (`t16p-d`), the power A/B and the memory probe. The user's rule since 2026-10-27. Most arms there are one start,
  so their numbers are leads, not results.
- **Computed from Task 16's dumps**: worked out by a code reader from Task 16's KL directories, no engine ran.
- **Read from the code**: static reading at `ad9157b4`, not measured.
- **Estimate**: worked out from measured rates. Not measured.

## Why

### The box and the target (the user, 2026-10-27)

- The box: RTX 3090 24 GB (CUDA 0, PCIe Gen4 x16, locked at 1500 MHz with a 350 W limit), Radeon PRO W6800 32 GB (hip:0,
  the AMD expert tier through its HIP library, PCIe Gen4 x16 on another root complex), 188 GiB RAM (eight 32 GB DDR4-2133 DIMMs
  are installed, but the firmware mapped out A0 and H0 after fatal DRAM errors at boot, so six run on channels B to G,
  each on its own address range per SMBIOS, likely no channel interleave; fresh buffers read 67.6 to 70.8 GB/s with 16
  threads, Task 16b), EPYC 7F52 16
  cores (SMT off, AVX-2, no AVX-512).
- The model: Swift 1.5 Qwen3.8-Flash-Next Q6_K, a 157 GB GGUF in five shards, pack `~/source/Strata-data/packs/swift-q6_k`.
  Experts: 48 layers x 512, top-10, 4,428,800 bytes each (Q6_K gate and up, Q8_0 down), 101.4 GiB in all.
- The serving flags the user keeps: `--max-context 262144`, `--kv int8`, MTP (`--mtp rt --spec 4 --spec-min-p 0.5`),
  `--pool-workers 15`, `--mmap-experts`, `--expert-cache auto`, `--peer-device hip:0`.
- What counts: prompts of about 2K, 8K and 20K tokens (the bench's 2,041, 8,013 and 20,053), the chats chat0 to chat2,
  and the long answers after the 8K and 20K prompts. No 64K prompts.

### Where the experts live today (Task 16, measured)

- The 3090 holds 3,103 experts (12.8 GiB) by profile rank. The W6800 holds 7,566 (31.2 GiB) at the default
  `--peer-reserve-mib 600`. The other 13,907 (57.4 GiB) are read in place from the page-cached GGUF.
- VRAM left: 431 to 433 MiB on the 3090 with everything loaded at chunk 4096, 602 MiB on the W6800 before its prompt
  buffers.

### What bounds a prompt

Measured (Task 16):

- Every chunk of 1,024 tokens or more streams every expert neither card holds, once, whatever its size. A fit over the
  four no-ring arms of run A (350 W, one start each): prompt ms = 0.284 ms a token + 5,103 ms a chunk. So a prompt
  costs about 5 s per chunk when staged, and a prompt read in one chunk is the fastest.
- The stream is staged: a memcpy from the page cache into a pinned ring, then DMA. A streamed byte costs three bytes of
  RAM traffic. At 250 W the 3090 waited for copies 63% of its timeline and its link carried 8.5 of 18.6 GB/s.
- The 350 W baseline at the user's flags (B3 and B4, chunk 4096, timing off, one start each): 2K 360 to 387, 8K 719 to
  754, 20K 746 to 785 tok/s. Chats 24.2 to 32.2 tok/s, long answers 42.6 to 51.8.
- The spread on the staged path is large. The 20K prompt read 497 to 673 tok/s over eleven starts of one build at 250 W,
  and 594 to 785 over Part 2's starts at 350 W. With the pinned copy at chunk 4096, three starts read 1,203 to 1,221
  (1.5%, the W6800's reserve differing).

Why the levers are not on by themselves (measured, start logs):

- The W6800's prompt buffers take 255 MiB at chunk 4096, 442 at 8192, 816 at 16384 and 1,191 at 24576 (read from the
  code, `tier_prompt_bytes`, matching the start logs). A full ring of 128 copies needs 541 MiB plus the 256 MiB spare
  (`kTierCardSpare`). The card opens a smaller ring from 16 copies (about 68 MiB plus 256), so the floor is about 324 MiB
  free (read from the code, `tier_stream.hpp:46-55`, `tier_stream.cpp:65-66`).
- At the default reserve the ring never opens (322 MiB free at 4096, 134 at 8192). At chunk 16384 the buffers do not fit
  and the tier silently leaves prompts (`stage 0 reads its prompts without the AMD tier`, 20K at 794 tok/s).
- `--prefill auto` (`c150b99a`) picks 4,352 tokens and the ring still does not open (304 MiB free).

The levers as configs (measured, Task 16):

| Lever (config) | Starts | 2K | 8K | 20K | Chats |
|---|---|---|---|---|---|
| Ring room, `--expert-gib hip0=30.5`, chunk 4096 (G305, 250 W) | 2 | | 719, 768 | 793, 835 (+37%) | not slower |
| One chunk for the 20K prompt: K24, `--prefill 24576 --peer-reserve-mib 1450` (350 W, timing on) | 1 | 337 | 1,073 to 1,105 | 1,715 | normal |
| The same with the ring: K24R, reserve 2200 (350 W, timing on) | 1 | 345 | 1,064 to 1,102 | 1,776 | chat1 26.9 |
| Pinned copy of everything the 3090 does not hold: RE, `--resident-experts --pcie-frac 0`, chunk 4096 (350 W) | 1 | 619 | 1,180 to 1,200 | 1,221 | 42.0, 32.7, 30.8 |
| RE with the ring (RER) and with the ring failed (RERF) | 1 each | 552, 618 | 1,126 to 1,148 | 1,209, 1,203 | chat1 33.1, 35.6 |
| **RE24**: the copy and one chunk, `--resident-experts --pcie-frac 0 --prefill 24576 --peer-reserve-mib 1450` (run D, 350 W, decode timing on) | 1 | 590 | 1,700, 1,766 | **2,528** | 44.5, 43.1, 36.0 |
| RE24AS: RE24 plus `--adapt-swaps 16` (run D) | 1 | 605 | 1,720, 1,791 | **2,566** | 43.3, 37.7, 33.6 |

- **RE24 is the target served state.** Against B3 and B4 it read the 20K 3.3x, the 8K 2.4x, the 2K 1.6x, chats +38 to
  +68%, long answers +21 to +38% (64.4 tok/s after the 20K). The controller named it the new baseline for Task 16b.
  Phase 2's Stages 1 and 2 make it the engine's default for the user's flags, fix its costs, and then go past it.
- RE24's costs (measured): ready in 130 s ("pinned 98.17 GiB in 109 s"), 119 GB read from the drives at start. The guard
  drops the whole GGUF's page cache, one `cudaHostAlloc` follows, and the copy is read back from NVMe at 1.0 to 1.2 GB/s.
  The 3090 had 231 MiB of VRAM free with everything loaded (the engine's `LOW` line), its verify captures fell to 167
  MiB, and later requests warned at 93 MiB. The W6800 had 236 MiB free, so no ring.
- RE (chunk 4096) per start against B3 and B4: +55 to +72% on the three prompts, chats +20 to +35%. RER's 2K (552) is
  +43 to +53%.
- The ring paid most when the 3090's staging was slow (250 W, +49% against the same cap with the ring failed). At 350 W it
  gave +1 to +13% (one start each), and nothing with the pinned copy (RER against RE and RERF).
- RE24's 20K prompt: one chunk in 7.9 s. The 3090's experts took 57 to 59 ms a layer, the W6800's 17 to 19 ms. Fit: 0.25
  ms a token plus 2.7 to 2.8 s a chunk. So with the copy the 3090 still bounds, and the W6800 waits.
- K24R's 20K prompt took 11,292 ms. The prompt path's own GPU timeline was 9,062 ms: wait copy 3,199 ms (35%),
  hyper-connections 1,093, GDN 1,471, QSA 1,174, combine 447, expert GEMMs 1,015, dequant 252. About 2.15 s lies after
  the prompt path's wall. Read from the code, that is mostly the serial pageable refill of the 1,942 lent slots.
- **The 2K ceiling.** A 2K prompt is one chunk, so it costs one full stream of the RAM experts over the 3090's link. With
  the copy it read 552 to 619 tok/s in every arm (RE, RER, RERF, RE24, RE24AS). Only the W6800 taking a larger share of
  the stream can lift it, and with the copy its streaming is a memcpy that competes for RAM (read from the code). No task
  promises a 2K gain beyond the copy. Task 25b probes the one route left (C19).

### What bounds decode

Measured (Task 16 at 250 W, decode timing on, no copy):

- A chat window is 85 to 115 ms: the CPU pool's Q6_K experts 29 to 46 ms, the adaptive tier's join 33 to 47 ms, the
  3090's GPU work 12 to 13 ms, draft 2 to 2.5 ms. The W6800 is off the critical path in chats (0.005 to 0.10 ms a
  layer-window).
- A fit over 57 requests on 8 starts (inferred from the logs, not a per-call measurement): join per round = 1.11 ms x
  3090 swaps + 0.87 ms x W6800 fills - 1.4 ms. A round runs every 4th window.
- `--adapt-swaps 16` (two starts against three references): chats +26 to +29%, the 8K answer +22%, the 20K answer -7%.
  `--adapt-swaps 0`: the 20K answer falls from 48 to 29 tok/s.

With the copy (Task 16 350 W, run D, RE24, one start, decode timing on):

- chat1 and chat2 windows are 54.0 and 56.4 ms (84 to 87 without the copy): the CPU's experts 26.5 and 30.1 ms, the
  3090's GPU work 12.1 and 12.0, the adaptive join 5.2 and 4.5 for 22 and 23 swaps (about 0.2 ms a swap against 1.6 to
  1.9 staged), the draft 2.1 and 2.5. The answers after the 2K, 8K and 20K prompts run 50.7, 38.7 and 37.5 ms a window.
- So with the copy the **CPU pool is the decode bound**, and the join is small. `--adapt-swaps 16` on top was no better
  (RE24AS, one start): fewer swaps left the CPU 5.5 to 5.8 experts a layer-window against 2.9 to 3.1.
- The pool reads about 25 GB/s while it computes (probe 7) against the 52 to 68 GB/s the RAM gives 8 to 16 threads.

Leads from probes 7 to 10 (Task 16 350 W, run C, chunk 4096, no copy, one start each, against DR1 and DR2's chats 31.1 to
32.2, 25.1 to 25.7 and 23.3 to 23.5):

- `STRATA_XTIER_TAIL_SKIP=1` (TS1): 34.0, 29.3, 28.0 (+6 to +20% per start, the report says +7 to +20). The CPU's experts
  per layer-window halve. It changes the answers.
- `STRATA_LOOKAHEAD=0 STRATA_IO_PREFETCH=0` (LA0): 33.5, 27.3, 24.6 (+4 to +8%, inside the spread). The look-ahead
  thread takes 38 to 43 ms of CPU a window.
- `--spec 5`: 36.3, 27.5, 23.3. `--spec 3`: 29.7, 28.1, 25.6. `--spec-min-p 0.7`: 31.1, 26.8, 26.1. No clear winner.

### What bounds correctness

- The KL reference `~/bench-mixed/kl-ref-q6k` (`STRATA_MMQ_KQ=0`, the W6800 out of prompts) repeats exactly (KL 0).
- The head passes with little room (Task 16): all mean 0.0205, median 0.0019, top-1 96.7%; long mean 0.0222 (74% of the
  budget), top-1 90.0% (the limit).
- C16G (chunk 16384, `hip0=29.5`) failed in its one run: long mean 0.0793. Two prompts carry it (Task 16 per-prompt
  table, KL(ref||head) / KL(ref||C16G)):
  - Long prompt 0: 0.0201 / 0.5746, top-1 differs for both. Computed from Task 16's dumps: it ends inside the git hash
    `4e1b9c2f7a0d` (`data/quant_quality/v1/agent.txt:64`), where the reference is nearly flat (entropy 3.18 nats, top token
    10.4%). It ran unplanned and streamed nothing in C16G's run (its planner lines).
  - Long prompt 4500: 0.0489 / 0.1293, 2.6 times as large. Computed from the dumps, KL(head||C16G) there is 0.0020. Whether
    the two share one gap is not confirmed until Task 18's per-prompt table.
  - Without prompt 0 the long mean is 0.0224 (head) and 0.0243 (C16G), top-1 9 of 9 for both (computed from the dumps).
- The 30 v1 long prompts are rotations of one 5,000-token text (five 1,000-token segments), each scored at the one
  position after it (`quant_quality.py:198-211`). They are correlated positions, not 30 texts.
- The gate's long prompts are 4,600 to 4,960 tokens. No KL prompt reaches the one-chunk 20K geometry Phase 2 will serve,
  and none reaches 20,972 tokens, past which a product of T x 10 x 2560 x 4 bytes passes 2^31 (arithmetic).
- No KL run covers RE24. The copy changes the planner's measured rates and so the 3090 and W6800 row split: in K24R the
  3090 computed 44 to 46% of the W6800's routed rows, in RE24 0 to 15% (planner lines). 16b's revised Step 2 runs RE24's
  KL on two starts.
- The prompt gate cannot see decode math: `quant_quality.py prompt --keep-caches` turns off the tail skip, the PCIe share
  and adaptation.

## Corrections to earlier readings

These were wrong or loose in Task 16's report, the code map or the plan drafts. The plan uses the corrected form.

1. **Swap cost.** "About 1.8 ms per swap" divided the join by the 3090's swaps only. The join also holds the W6800's fills.
   The fit gives about 1.11 ms per 3090 swap and 0.87 ms per W6800 fill (inferred). With the copy it is about 0.2 ms a
   3090 swap (RE24).
2. **"A fixed ~7 ms per round".** AS16's 7.4 ms is per window, averaged over all windows. A round runs every 4th window, so
   it is about 30 ms a round, which 16 swaps plus 16 W6800 fills explain. The "swaps/window" field counts only the 3090.
3. **`--pcie-frac` does nothing beside the tier** because no blob is pinned without a pinned RAM copy
   (`expert_source.cpp:4879, 4915`, read from the code), not because of the tier. With a pinned copy the share flies at
   0.52 to 0.55, because PcieAuto is off beside any peer (`generate.cpp:10255`).
4. **RB60 was not "a pinned copy of the RAM experts".** The budget ranks everything the 3090 does not hold and took the
   W6800's 7,566 experts first, so about half the RAM experts and 0 of 506 lent slots were pinned (read from the code,
   `expert_source.cpp:3546-3556`, and the log's "14546 of the 88.57 GiB"). Its decode PCIe share flew at 0.52 to 0.55, and
   its 3090 stager ran the SSD profile after the whole-file drop. Its numbers are not the copy's ceiling. Its chats were
   "not slower, maybe faster" over two starts (Task 16).
5. **Task 16b's budget arms** pin the W6800's experts first by the same rule (read from the code). The revised brief's one
   `--resident-budget-gib 60` start at 24576 is a RAM-short data point, not a partial copy of the RAM experts.
6. **"18% dequant" at chunk 4096** was mostly the 3090 waiting for the host memcpy, which lands on that mark (Task 16
   350 W, probe 2: 70% with 2 stager threads, 22% with 8).
7. **"Stager threads are not a lever"** holds for 8 and 16 against 4 within one start state at 250 W. At 350 W, 2, 4 and 8
   threads read the 20K at 348, 568 and 713 tok/s (one start each, timing on).
8. **The W6800 swap thresholds** are at `peer_adapt.cpp:24` (the 2.0 candidate floor) and `:35` (the 1.5 gain), not `:21`.
9. **`kTierMaxBounds` (1 << 15)** bounds the group-bounds table (`tier_tab_ints`), not the tokens. The library accepts
   `T_max` up to 1 << 20 (`xtier_lib.cu:403`).
10. **The code map's C3 verdict** ("seeding priors changes nothing") holds for the 3090's book only. The W6800's
    streamed-byte prior stays at 6 GB/s (`starting_rates(0)`, `generate.cpp:9074`) until it streams. **C8's cite** for the
    drain is `expert_planner.cpp:88-93`, not `:121-126`. **C19's verdict** ("Q6_K is a pack, so transient() is false") is
    wrong here: the experts are read in place from the GGUF, so they are transient. **C14** is not refuted: its verdict
    says it partly holds and is a config probe of existing options.
11. **KV size.** The 3.5 GiB estimate is replaced by a measurement: `--kv-resident 32768` put "the K/V in 3.09 GiB of
    pinned RAM" and gave the 3090 717 more slots (3,103 to 3,820, Task 16 350 W, KV32).
12. **The halves' lead** is `TierPlanner::lead_`, a per-layer decayed mean (`tier_planner_engine.cpp:28, 258-260`, used at
    `:97`). `TierStart` (`tier_planner.hpp:91-101`, `tier_planner.cpp:71-74`) is the tier's start lag, another number.
13. **DecodeSpares "never settled"** is not a measurement. The `decode pool` line prints only on a move or a kept look with
    timing on (Task 16 probe 7), so its absence can mean nothing moved. That it alternates 14 and 15 workers is read from
    the code only.
14. **The copy-backs of 3090 swaps are already batched** with one wait per stream (`generate.cpp:383-411`, read from the
    code). What is left per swap is the wait before staging and `commit_exchanges`' memcpy (`expert_source.cpp:4311`).
15. **The AMD link probe keeps a 64 MiB card buffer.** `probe_tier_h2d_gbps` runs after the fill (`generate.cpp:6866`) and
    the library's `probe_copy` keeps its `hipMalloc` (`xtier_lib.cu:297-305`, read from the code). Today's 602 MiB reading
    does not reconcile with this yet.

## Goals

1. The most prompt speed (2K, 8K, 20K) and decode speed (chats, long answers) for Swift 1.5 Q6_K on this box, as
   **engine defaults** for one NVIDIA card with the AMD tier under the user's flags, not hand-set flags. First reach RE24
   by default with a fast start, then go past it.
2. No loss of quality. A change that only moves data or threads keeps greedy answers identical. A change to the math
   passes the KL gate on two starts.
3. A measurement good enough for the keep rule: an arm that silently fell back fails before it measures, and a gain is
   only called when it beats the spread between starts.
4. Less technical debt: the switches the new defaults make redundant go.

## Non-goals (the user's calls, and out of scope)

- **The user's calls, reported and never changed:** the 3090's 1500 MHz lock and 350 W limit, `--max-context 262144`,
  `--kv int8`, the user's MTP flags and `--pool-workers 15`, the W6800's power or DPM state, the DIMMs, `strata.service`
  (stays stopped, unit untouched; a Q6_K config is proposed at the end, not applied). A measured better value for one of
  the user's flags is proposed with its numbers, never written into the proposal.
- Speed on other hardware, models or platforms. Windows, Mac and SYCL builds. CUDA and HIP must keep building, and the
  multi-stage and NVIDIA-only paths must build and pass their unit tests (this box has one NVIDIA card, so the multi-stage
  path cannot start here).
- 64K prompts and chunks above 24,576 tokens (never run on this box, see Left out).
- A 2K gain beyond the copy, unless Task 25b finds a route.
- Rounding-changing CPU kernels, layer-major reading (C26), dense int8 MMQ (C15) unless Task 25 names it.

## Architecture of the changes

### Measurement first (Tasks 17 to 19)

- **The bench kit in the repo.** Task 16's box scripts become `tools/box_bench.py`, `tools/engine_log.py` and
  `tools/box_ab.sh`. Before an arm measures, its start facts (chunk, the 3090's loan, the W6800's experts, ring slots,
  whether the tier is in prompts, the pinned copy's size, both cards' free VRAM) are read from the engine log and checked
  against `--expect`. A verdict command prints every start and KEEP, NOT RESOLVED or SLOWER. The `-prev` configs exist
  for every config an A/B uses.
- **KL gate v2.** One unscored warm prompt first, so every scored long prompt runs the tier state the config serves. 30
  long prompts that keep the old 10 as a subset. Seven xl prompts with their own reference: six of about 20K tokens and one
  of about 24,300, so the one-chunk geometry and the int32 zone are gated. Per-prompt reference entropy. A `gate` command
  over two starts. RE24 gets a per-prompt table.
- **Lines and limits.** Loud lines where the engine falls back or hides time (a halved chunk, the refill of lent slots,
  the W6800's VRAM after each step and after its first prompt, the 3090's VRAM after the first one-chunk prompt), an
  opt-in count of FP16 sums past 65504, and a size audit with tests at T = 24,576.

### Prompts: one box plan at load (Tasks 20, 21)

1. **The chunk.** For one NVIDIA card with the AMD tier and `--prefill auto`, the engine resolves the chunk at start: the
   largest on the 256 grid whose loan leaves the 3090's 128-slot floor and whose prompt buffers leave the 3090 at least
   256 MiB free (the engine's own `LOW` line, read at run time), at most 24,576 (the largest chunk run on the box). It is
   set before the W6800 fills. A request still lends only for its own tokens, so short prompts cost nothing more.
   c150b99a's `tier_room`, the 85% lend cap and the bare-auto 8192 ceiling go for this topology.
2. **The W6800's room.** The link probe runs before the room is read. Before its experts fill, the W6800 opens its prompt
   buffers for that chunk and, when the box plan says it streams, its 128-copy ring. Then it reads its free VRAM and
   fills to what is left minus a spare. `--peer-reserve-mib` on the AMD path means that spare (default 320 MiB). The
   card's real rounding (25 to 32 MiB beyond the counted buffers, measured) is inside the reading.
3. **Rings open at load**, so the first planned chunk streams. Whether the ring opens at all is an input of the box plan:
   on without the copy (G305 measured it), decided again with the copy (Task 23) and after the planner fix (Task 24).

### Prompts and decode: one pinned RAM copy (Tasks 22, 23)

1. **The copy's fill.** The copy holds every expert the 3090 does not hold plus its lend region (today's lone-card
   `--resident-experts` set, 98.17 GiB at chunk 24,576 in RE24). It is filled layer by layer from the page cache
   (O_DIRECT for layers not cached), each layer registered with CUDA as it completes, and each copied expert's file pages
   dropped. No whole-file drop. A probe decides huge pages and the registration pattern first, from H2D and CPU read rates.
2. **The default.** It becomes the default for one NVIDIA card with the AMD tier on Linux when MemAvailable and
   RLIMIT_MEMLOCK cover it with room to spare (read at run time). Then the 3090 DMAs straight from the copy, refills its
   lent slots by DMA, and its adaptive swaps exchange with the copy. The CPU pool reads the copy with no mincore. The decode
   PCIe share defaults to 0 beside the tier, so decode math stays as today until Task 27 decides. The pinned share and the
   loan are priced from the copy's plan, not from what is pinned before it.
3. **The ring with the copy.** RE24 ran with no ring, and RER against RE and RERF showed none paying at 4096. Unless 16b's
   revised Step 3 (RE24 with `--peer-reserve-mib 2200`) or Task 23's arms show a gain, the box plan does not open the
   ring when the copy holds every streamed expert, and the W6800 keeps that room for experts.
4. **A budget that skips the W6800.** When RAM is short, the ranked budget leaves the W6800's experts out of its rank and
   keeps the lend region (fixes correction 4).

### Prompts: the planner (Tasks 24, 25, 25b)

- Gated on the halves lines after the copy: the W6800 is priced from its probed link, the halves' lead (`lead_`) is kept
  per token, and a counter shows streamed RAM experts that got no rows. The ring and the stream cap are re-checked after.
- Task 25 profiles the 3090's own prompt work once copies stop binding (49% of K24R's 20K timeline was non-expert work)
  and proposes follow-up tasks with numbers.
- Task 25b (conditional, probe only) measures whether the W6800 can DMA from a HIP-registered copy without Task 11's stall.

### Decode (Tasks 26 to 34)

With the copy the CPU pool is the bound and the join is small (RE24). The order follows that.

1. **Diagnostics first** (26). The decode line counts both cards' copies per round and the join per round. A serve-side
   pool line splits the pool's time per call. A pool bench runs real GGUF experts.
2. **A decode KL, then the lanes that cut CPU work** (27): the route tail skip and the PCIe share from the copy, each
   behind a decode KL with a measured placement floor.
3. **Old-box helpers off** when RAM holds every expert (28): look-ahead, io prefetch, per-miss mincore, and spares only if
   Task 26 shows them moving.
4. **The CPU row loops prefetch ahead** (29), gated on the bench.
5. **Config sweeps** (30): verify width, the draft head's bytes (C14), a frozen learned start profile, `--kv-resident`.
   Values for the user's flags are proposed, not set.
6. **Gated, expected to stop at their gates with the copy:** cheap adaptive copies (31), a swap budget from measured costs
   (32), the adaptive round off the critical path (33), the W6800's decode round trip (34).

### Wrap-up (Tasks 35, 36)

The final A/B against `ad9157b4`, the switch cleanup, the docs for this box and a proposed Q6_K service config.

## Measurement protocol

- **Reference configs.** `cfg-q6k2.json` is the user's flags (chunk 4096). `cfg-q6k2-re24.json` is RE24 as flags (the
  baseline the controller named for 16b). A task that makes a default is KEEP against `cfg-q6k2` on PREV, and not slower
  than `cfg-q6k2-re24` on PREV. From Task 23 on, the head's defaults are RE24's, and later tasks compare head against head.
- **Starts per arm.** Task 16b Step 1 sets the protocol (four RE24 starts, starts needed for a 5% and a 10% gain). Until
  it reports: A B B A, two engine starts per arm, PREV and NEW in one session, every start's numbers in the report. Any
  target whose expected gain is under the protocol's spread gets at least three starts per arm (A B A B B A). If 16b finds
  no protocol under 10%, three starts per arm everywhere.
- **Passes per start** (`tools/box_bench.py`, from Task 17): `--pass warm:chat0,p8k --pass m:chat0,chat1,chat2,p2k,p8k,p20k
  --pass m:chat0,chat1,chat2,p2k,p8k,p20k`, about 5 minutes an arm (copy starts take longer until Task 22).
- **Keep arms** run with `--conversation-cache-mib 0` and timing lines off (timing lines cost: K4 read the 2K at 305
  against B3 and B4's 360 to 387). Diagnostic starts (one each) turn timing lines on.
- **Decode arms** run at the default spares. `STRATA_DECODE_SPARES=0` goes into both arms only if Task 26's pool line
  shows the worker count moving inside a request.
- **Start facts.** Every arm passes `--expect` (chunk, loan slots, W6800 experts, ring slots, tier in prompts, pinned GiB).
  An arm whose facts differ stops with exit 4. From Task 23 on, `pinned_gib` is a required fact.
- **Copy arms and memlock.** Bench shells have RLIMIT_MEMLOCK 23.6 GiB. Every arm with a copy, and every arm from Task 23
  on, runs under `prlimit --memlock=unlimited`.
- **Page cache.** After an arm with a pinned copy, rebuild the page cache (`cat` the five shards, or 16b's prep) before an
  arm without one. Compare drive MB (`the OS read N MB from storage`) on measured passes only, against PREV's same
  request: the warm pass's 8K reads 188 to 743 MB with or without the copy, cause not found.
- **The verdict.** KEEP when every NEW start beats every PREV start on the target (min NEW above max PREV) and the mean gap
  is at least the larger arm's spread (max minus min). The other metric is not slower when its NEW mean is at least PREV's
  mean minus the smaller of PREV's spread and 5% of PREV's mean. Otherwise NOT RESOLVED or SLOWER. With two starts per arm,
  min above max alone happens 1 time in 6 by chance, and 1 in 20 with three.
- **Counter acceptance.** When 16b's spread is larger than a task's expected gain, the controller may accept the task on
  deterministic counters from timing starts (ms per window part, CPU experts per layer-window, ms per layer), with the
  whole-start A/B used only as the not-slower check. The report says which.
- **Identity kinds.**
  - Prompt identity: `quant_quality.py prompt` dumps with `STRATA_PF_PEER_HOST=0` in both arms, KL exactly 0 on every
    prompt.
  - Decode identity (Task 14's ruling): both arms `xargs=--adapt-swaps;0;--pcie-frac;0;--prompt-cache;0`,
    `STRATA_PF_PEER_HOST=0`, `--identity --warm 0 --measure 1 --full-text`. Answers identical.
  - Identity with adaptation on: only if Task 17's G-ID check shows PREV against PREV repeats. Otherwise
    `STRATA_ADAPT_CHECKRES=1` and `STRATA_XTIER_CHECK=1` with zero mismatches, plus decode identity.
- **KL.** Gate v2 (Task 18) against `kl-ref-q6k-v2`, Phase 1's thresholds: the all set (short and long) and the long set's
  mean at or below 0.03, the all set's median at or below 0.005, top-1 at or above 90%. The xl set: mean at or below 0.03,
  no top-1 miss where the reference's top-1 probability is 0.5 or more. Two starts of the candidate whenever the W6800 is
  in prompts, both must pass. No position is ever dropped. The binding set, and whether the long top-1 line becomes
  relative to the head's own misses, are the controller's ruling after Task 18.
- **Decode KL** (Task 27): `quant_quality.py run --keep-caches`, teacher-forced. Floor F = KL(head || head with
  `--expert-gib hip0=27`), a fixed placement change of about 650 W6800 experts (read from the code). Proposed gate for a
  decode-math switch: mean KL at most max(2F, 0.006) and top-1 at least 95% against the switch off, on two starts. The
  controller rules it.
- **Box hygiene.** Every build, ctest and measuring script runs under `flock ~/bench-mixed/.box.lock` for its whole
  length, and sets and checks `sudo nvidia-smi -i 0 -pl 350` and `-lgc 0,1500`. Stint prefixes use a separator that cannot
  collide (`t21-s1`, `t21b-s1`). Results go to `~/bench-mixed/plan-results.md`.

## Gates on Task 16b

16b runs on RE24 since the controller's revision (its first run, `t16b-r1`, is four RE24 starts).

- **Step 1 (the spread of RE24):** sets the starts per arm and the prep in Task 17. If 16b names a spread cause the engine
  controls (for example where the pinned staging ring's pages land), the controller adds a task before Task 20.
- **Step 2 (KL: RE24 on two starts, RE at 4096, RE24 with the ring failed, C16G):** read in Task 18 (the gate ruling) and
  Task 21 (whether Task 21b's fix lands first). Its decode-quality finding feeds Task 27.
- **Step 3 (the bound at RE24, and RE24 with ring room):** read in Tasks 23 (the ring), 24 and 25.
- **Step 4 (RAM and start at RE24, one RB60 start at 24576):** read in Tasks 22 and 23.
- **Step 5 (on RE24: tail skip and look-ahead off on two starts, `--kv-resident 32768`, `STRATA_MMQ_DIRECT=1`, the 131072
  slot count):** the tail skip gates Task 27, the look-ahead Task 28, `--kv-resident` and the slot count Task 30,
  `STRATA_MMQ_DIRECT` is recorded for Task 25.

## Constraints

- Hard fork: breaking changes are fine. A better behavior becomes the default, and the switches it makes redundant go.
  `--peer-reserve-mib` changes meaning on the AMD path.
- Linux with CUDA and HIP only. Build both trees for every change. Library ABI bumps are allowed (Tasks 21b, 34 may
  bump it). Every PREV arm uses a PREV config whose exe and `STRATA_XTIER_LIB` point at `~/source/strata-prev`.
- Tuning to this box is allowed (the 24,576 chunk cap, the 320 MiB spare). Prefer a run-time read when it is as fast:
  free VRAM, MemAvailable, RLIMIT_MEMLOCK, page residency, probed links, measured costs.
- Code comments: one line, plain prose, no em dashes, no semicolons, no hyphens joining clauses.
- Tests first where code changes. Each task is one reviewable unit and one commit (or a few small ones).
- Commits end with the `Co-Authored-By:` line. Push `mixed-tier-perf` after each task, never force-push, never commit
  `.superpowers/`. Never edit `~/source/strata` on the box.
- Docs keep the repo's style: plain words, short sentences, numbers with what they were measured on, read-from-code
  claims marked.

## Left out, and why

- **D1** (an exact copy of only the RAM experts, with a new W6800 exchange): the default copy pins the W6800's experts too
  (about 30 GiB more on a 188 GiB box) and needs no new swap code. A wrong exchange gives silently wrong answers.
  Revisit only if RAM pressure shows (16b Step 4, Task 23's MemAvailable line).
- **C19 as code:** large and high risk (Task 11's stall). Task 25b probes it first, gated.
- **Chunks above 24,576:** never run here, and no measured prompt needs them. Task 19's audit records the headroom to
  32,768 so a later one-line raise can be measured.
- **Equal chunks** for prompts longer than the chunk: zero on 2K, 8K and 20K.
- **Short chunks moving only routed experts:** waits on Task 24's no-row counter. Added as a task only if at least 25% of
  streamed RAM experts get no rows at 2K on real text.
- **C7** (non-temporal stager stores): it only helps the staged path. Listed under Task 23b, the path without a copy.
- **C21** (Q6_K decode kernels on both GPUs): refuted. The GPU kernels finish before the CPU pool, so they do not shorten
  a window.
- **C22** (L3 prefetch of the next layer's CPU misses), **the pool's one-publish rewrite**, **a multi-token Q6_K CPU
  kernel**: large or small and risky. Revisit after Task 26's bench.
- **C13, C16, C17, C18, C20, C23, C24, C26, MMQ_DIRECT as its own task:** small, refuted by skeptics, or recorded through
  16b Step 5 and Task 25. C14 is a config arm in Task 30.
- **Moving the hottest experts to the W6800, a permanent 3090 lend region, W6800 streaming of decode misses:** zero or
  negative by the decode reading.
- **`--kv-grow` beside the peer:** refused, and the reason was not found in the code.
- **Deleting the multi-stage W6800 sharing:** no measured cost on this box (the halves were on time, host grouping 0.4%).

## Left to the builder

- The exact spelling of the start facts and of the new log lines.
- How the copy's fill splits threads between memcpy and O_DIRECT, and whether huge pages are used (the probe decides).
- Whether the ring decision with the copy is a rule (the copy holds every streamed expert) or a measured switch.
- When the 3090's VRAM floor binds, whether the rule shrinks the chunk or the expert cache (Task 21 measures both).
- The xl text's exact cuts, as long as they are frozen copies with committed ids.
- How each conditional task's gate number is read (which log line), as long as the gate is written in the report.
- The order inside a task and how many commits it takes.
