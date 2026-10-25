# NVIDIA and AMD cards on one model: an adaptive expert tier on the other make's card

Design, 2026-10-24. Branch `mixed-make-expert-tier`. Not part of the upstream pull request; it is the plan's source.

## Why

Strata's engine drives one make of card per run. The CUDA and HIP engines are the same sources compiled twice
(`include/strata/hip_compat/` maps the `cuda*` names to `hip*`), and `CMakeLists.txt:112` refuses to build both at
once. The app, `strata launch` and setup hide the other make's cards on purpose.

So on a PC with both makes, one model uses one make's VRAM and the other make's VRAM sits idle, while the experts
that do not fit are read from RAM or the drive. The test box shows the cost: Swift 1.5 IQ4_XS has 60.9 GiB of
experts. The three NVIDIA cards cache 24.4 GiB of them. The W6800's 32 GB is unused in that run.

**Intent.** One model served by both makes at the same time. Every card keeps an adaptive expert cache, and an
expert's rows are computed by the card that holds it, in decode and in prompts. NVIDIA stays the main engine and the
AMD card is the extra tier. The result must be faster than the NVIDIA run on the same box, and as fast as we can
make it.

**Who uses it.** For now: a build from source on Linux, set up by engine arguments in the model's `strata-*.json`.
Later: a pull request to Niko1221/Strata. Keep the change small for their review, but performance comes first
where the two conflict.

## The test box and today's numbers

`ssh vdh@192.168.10.5`, Ubuntu 24.04, kernel 6.8.

| | |
|---|---|
| NVIDIA | RTX 3090 24 GB (x16), RTX 3080 Ti 12 GB (x16), RTX 3050 8 GB (x8, 13 GB/s probed); all sm_86; driver 595.91, CUDA 13.2 |
| AMD | Radeon PRO W6800 32 GB (gfx1030, RDNA2, PCIe 4.0 x16); ROCm 7.2.4 in `/opt/rocm`. No WMMA, no hipBLASLt |
| CPU, RAM, drive | EPYC 7F52 16 cores, 62 GB RAM, a SATA-class SSD by the engine's guess |
| Builds | `build-cuda-sm86`, `build-hip-gfx1030`; installed engines in `~/source/Strata-data/engines/{cuda-sm86,hip-gfx1030}` (recipes in `~/source/Strata-data/LOCAL_INSTALL.md`) |
| Service | `strata.service` runs `serve.server` with `strata-swift-iq4_xs.json` (NVIDIA) or `strata-swift-iq4_xs-amd.json` (AMD): one pair of lines in the unit is commented out to pick the make. Port 5999 |
| Model | Swift 1.5 Qwen3.8-Flash-Next IQ4_XS (`/opt/models/ukisai_Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-0000{1,2,3}-of-00003.gguf`), 48 layers, 60.9 GiB of experts. Settings: 262,144-token context, int8 KV, `--mmap-experts`, MTP (`--spec 4`), vision on (CPU reader), 15 pool workers |

The NVIDIA run is a three-stage layer split: the 3050 runs layers 0-2 (3.21 GiB of experts), the 3080 Ti layers
3-13 (6.32 GiB), the 3090 layers 14-47 and the head (14.88 GiB).

**Same-day baselines** (measured 2026-10-24 with `tools/strata_bench.py --url`, page cache dropped before each start,
the service's settings; short cold-start runs, good for direction, not for a published claim):

| Same run number on each side | NVIDIA trio | W6800 alone |
|---|---|---|
| Chat decode, run 1 / run 2 (median of 3 questions, 100 tokens) | 32.1 / 41.3 tok/s | 15.4 / 21.1 tok/s |
| 4K prompt, run 1 / run 2 | 278 / 358 tok/s | 338 / 420 tok/s |
| 16K prompt, run 1 / run 2 | 512 / 523 tok/s | 674 / 683 tok/s |

NVIDIA's third run reached 48-54 tok/s chat decode and 464 tok/s on 4K. Raw output: `~/bench-mixed/run.log` and
`~/bench-mixed/run-amd.log` on the box.

The W6800 alone reads prompts faster; NVIDIA decodes about twice as fast. The prompt edge may come from memory, not
from the card: the NVIDIA engine's prefill plan reports a 44.8 GiB file set against a 49.3 GiB page cache and
shrinks its prompt chunk (6144 to 3840) to avoid re-reading from the drive, while the W6800's plan has a 42.3 GiB
file set against 50.5 GiB and keeps chunk 8192. This is read from the logs, not measured; phase 0 below checks it.

## Goals

1. On the test box, the mixed run beats the NVIDIA run on decode and on prompts, measured the same day (see
   Measurement). Then keep tuning while changes measure faster. Beating the W6800's own prompt speed is a stretch
   target for that tuning, not a pass condition.
2. Every card of both makes keeps an adaptive expert cache that follows the conversation.
3. One owner per expert: no expert sits on two cards at once.
4. All existing expert caching keeps working: the stage caches and their swaps, the PCIe share, the CPU pool, the
   file tier (mmap experts, page cache, prefetch, read-ahead) and the RAM copy.
5. Opt-in. Without the new argument the engine is byte-identical in output to the build it came from.

## Non-goals for the first version

- The app, `strata launch`, setup, installers, the update feed, Docker. Build from source only.
- Windows. Nothing in the design should block it (the engine already loads `strata_kq.dll` the same way), but it is
  not built or tested.
- Several conversations at once (`--batch`) and saved session files. The peer tier refuses session files today
  (`generate.cpp:12097`); every refusal the peer has today stays, with its message.
- Packs the peer tier does not serve today. It serves native packs only (`--native`, which Swift IQ4_XS is;
  `src/core/peer_experts.cpp:302`). The tier does the same, and any other pack is refused with a message that says
  why. (Decided in review, not by the user.)
- A HIP engine that uses NVIDIA cards as its tier, even if the W6800 alone keeps reading prompts faster. The user
  chose NVIDIA as the main engine. The tier's sources are make-neutral, so this stays a later build option.
- More than one card of the other make. The interface takes a list; v1 accepts one, because the box has one.
- Intel (SYCL) and Mac (Metal).
- Direct card-to-card copies between makes (no P2P exists) and splitting single layers across cards.
- Bit-identical answers between the mixed run and a single-make run (other kernels compute the AMD rows).

## Anti-goals

- A mixed mode that is not faster than the NVIDIA run on both decode and prompts. Then it is not recommended or
  claimed.
- Any change, in output or speed, to runs without the new argument.
- A refactor of `generate.cpp` or the build beyond what the tier needs.
- Tying a job to a make. What each card computes follows what it holds and what is measured, not whose card it is.

## Decisions

Made by the user:

- Build from source on Linux is enough for now; an upstream pull request comes later. Keep the change small, but
  performance comes first.
- Every GPU of both makes keeps an adaptive expert cache. A static expert set on the AMD card was rejected.
- All existing expert caching keeps working.
- Do not assume one make is better at a job: the W6800's prompt edge may come from its VRAM.
- NVIDIA is the main engine, the AMD card is the extra tier (approach 1). Beating the NVIDIA run on decode and
  prompts is the bar.
- Test on Swift 1.5 IQ4_XS, a model larger than VRAM.

Made while writing this design (the user can overrule them):

- Build the tier from the peer tier (`PeerExperts`), loaded as a library in the engine's process.
- One owner per expert; the ownership policy default is chosen by measurement.
- The `--peer-device hip:N` spelling.
- v1 takes one card of the other make, serves native packs only, and keeps the peer's other refusals.
- Phase 0 measurements come first and end with a stop to report to the user.
- Pass or fail is judged warm; cold numbers are reported beside them.

## Constraints

- One engine binary drives one make. Code for the other make has to be a second compile of the shared sources, with
  its symbols kept private so the two copies cannot clash (Strata's own and ggml's).
- The CUDA and HIP runtimes have different symbol names, so one process can load both.
- Data between makes goes only through pinned host memory. Buffers one make writes and the other reads must be
  registered with both runtimes. A CUDA event cannot be waited on by HIP or the other way round: sync goes through
  the host or through flags in mapped memory.
- From CONTRIBUTING.md: one problem per pull request; build every backend a file touches (here CUDA and HIP); change
  no default without measurements; mark findings from reading code as unconfirmed.
- RAM is tight: the page cache already competes with the expert file set. New pinned buffers cost page cache.

## Approaches

**1. An adaptive expert tier on the other make's card (chosen).** The NVIDIA layer split runs as today. The W6800
becomes an expert tier over all layers: it holds experts no NVIDIA card holds and computes their rows at each layer,
at the same time as the NVIDIA stage computes its own and the CPU computes the rest. Only the tier's slice of the
engine is built for HIP. It matches the request: both makes busy at once, experts cached on both, adaptive on both.
Its risk is one host round trip per layer; a 2080 Ti helper once lost more to that than it saved
(docs/MULTI_GPU.md), an RX 6600 helper beside an RX 6800 gained (docs/SECOND_GPU.md).

**2. The other make's card as a relay stage (rejected).** The W6800 would own a block of layers in the split. Stages
take turns, so the cards are not busy at the same time in decode, and each stage caches only its own layers. A
stage is the whole engine (weights, KV, checkpoints, prompt chunks, pictures, control vectors), so it needs either
the full engine compiled twice into one process or a second engine process that mirrors all of it. By far the
largest change.

**For approach 1: a library in the engine's process (chosen), not a helper process.** One process shares pinned
memory directly and needs no inter-process messages per layer. The engine already loads a GPU library at run time
with private symbols and a versioned C interface (`libstrata_kq.so`, `include/strata/prefill/mmq_kq_abi.hpp`,
`src/prefill/mmq_kq_loader.cpp`), so the pattern is known to the maintainers. A helper process would add a message
round trip per layer and a second copy of the model's host state.

## Design

### Shape

- The engine is the CUDA build. The layer split across the NVIDIA cards runs as today.
- The tier is built from the peer tier (`PeerExperts`, `src/core/peer_experts.cpp`). It already does most of what is
  asked: an adaptive cache that swaps in experts routed often (`adapt`), decode rows launched before the CPU's own
  work and written into the same mapped host rows (`launch` / `finish`), and prompt rows through mapped host
  memory when the cards cannot reach each other (the host route in `PeerPrefill`, `src/prefill/prefill.cpp:706`).
- Each MoE layer in decode and in verify windows: the pool's dispatch already sorts the routed entries into stage
  hits, peer rows and CPU rows (`src/core/expert_source.cpp`, the `kind == 2` entries around line 4970). The tier
  takes the place of `drive.d.peer`. All three NVIDIA stages share that dispatch, so the tier serves every stage.
- Prompts: each stage's prompt path gives the tier the rows of the experts it holds, through the host route, and
  gets back one weighted sum per token (the existing SUMS mode).

### The library

- Name: `libstrata_xtier_hip.so`, a target in the HIP build tree.
- Built from existing sources: the peer tier, the expert cache (`expert_cache.cpp`), the expert kernels it calls
  (`native_moe`, `s2_expert_grouped`, `quantize_act`, the Hadamard helpers) and the MMQ prompt products, which cover
  IQ4_XS (`src/prefill/moe_mmq.cu`). Position-independent, hidden visibility, loaded with `RTLD_LOCAL`, as
  `strata_kq` is.
- A C interface with a version number: open, which experts it holds, decode rows (launch, finish), one adapt step,
  prompt rows, close, counters.
- The tier's host logic (which experts it holds, adapt choices, the row plans) stays in the engine, in `PeerExperts`
  in a "foreign" mode. The library does the device work only. The engine reads each blob through its own
  `ExpertSource` and hands the bytes to the library, so the file tier, page cache and prefetch keep serving it.
- On the engine's side: a loader like `mmq_kq_loader.cpp`. (Planning replaced the callbacks and proxy first
  described here.)

### Adaptive caches and ownership

- Every card's cache adapts between windows: the NVIDIA stages as today (`--adapt-every`, `--adapt-swaps`), the tier
  with its own swaps (`--peer-adapt-swaps`).
- An expert has at most one owner. The peer's `adapt` receives `host_res`, which in a layer split already holds every
  stage's residency (built with `stage_of`), and the stages' own adapt already skips experts the peer holds.
- When a stage takes an expert the tier holds, the expert moves: the tier frees that slot. An expert a stage
  evicts becomes a candidate for the tier.
- The other policy is the one `STRATA_DISJOINT_ADAPT=1` uses for helpers: stages never take what the tier holds.
  Which of the two is faster is measured. The default is chosen by that measurement.

### Today's refusals and switches

- `--peer-device` with `--layer-split` is refused because the peer card would also carry a stage
  (`generate.cpp:3371`). A card of the other make cannot be a stage, so the refusal is lifted for the tier only.
- A peer turns off some decode defaults today: the route tail skip (`STRATA_ROUTE_TAIL_SKIP`, worth +10-20% on
  cards that miss experts; `generate.cpp:10008`), the automatic PCIe share (`generate.cpp:10031`), `--kv-grow`
  (`generate.cpp:5203`) and `STRATA_PIN_BY_NEED` (`generate.cpp:5836`). Each needs a decision for the tier, made by
  measurement. Turning one back on is in scope when it is safe and faster.

### Arguments

- `--peer-device hip:N` names the Nth AMD card, reached through the library (proposal; the exact spelling is the
  builder's). `--peer-device N` keeps its meaning.
- The library is found beside the engine by default; an argument or environment variable can point elsewhere.
- The existing `--peer-reserve-mib`, `--peer-slots`, `--peer-adapt-swaps` and `--peer-prefill-rows` apply to it.
- In the server config: `"args"` carries the argument, `"lib_dirs"` gains `/opt/rocm/lib` beside the CUDA one. The
  `"gpu"` list stays NVIDIA-only.

### Sync between the makes

- Decode: through the host, as the peer does today. The pool thread starts the tier's work and waits for it, and the
  rows come back through host memory.
- Prompts: flags in host memory that both makes map, with the split hand-off's bounded `wait_handoff` kernel and a
  new `raise_flag`. A host wait there would block the thread that queues the NVIDIA card's work. (Decided in
  planning.)

## Measurement

**Phase 0, before writing the tier.** These decide details above.

1. Round trip: a small program on the box with a CUDA context and a HIP library in one process. A pinned buffer is
   registered with both runtimes. One decode layer's tier rows (a realistic batch of IQ4_XS experts and tokens) go
   to the W6800 and back. Compare with the CPU pool's time for the same rows.
2. Memory: does `hipHostRegister` work on memory from `cudaHostAlloc`, and the other way round?
3. The 3050: the NVIDIA split with the 3050 Off against the trio (decode, 4K, 16K).
4. VRAM-equal check: the W6800 alone with its cache capped at the trio's 24.4 GiB. This answers whether its prompt
   edge is memory or the card.

Phase 0 ends with a stop: the numbers go to the user before the tier is built. If the round trip costs more per
layer than the CPU pool spends on the same rows, the tier cannot win decode, and the plan is revisited with the user.

**Acceptance.**

- Arms: the best NVIDIA set from phase 0, the W6800 alone, and the mixed run. Same day, at least 3 rounds with the
  arms' order alternating, medians. `tools/ab_engine.py` takes one config, so the three configs need a wrapper or
  an extension of it (the builder's choice).
- Two states, as in the user's own comparison table. **Warm steady state** decides pass or fail: each arm is
  started, warmed with the suite once, then measured without a restart. **Cold** is reported beside it: a restart
  with the page cache dropped (`echo 3 > /proc/sys/vm/drop_caches`) and the first requests after it.
- Keep runs short: about 5 minutes per arm (chat, 4K and 16K prompts, 2-3 runs each). The user waits on them.
- Measures: chat decode tok/s, 4K and 16K prompt tok/s, time to first token; one 32K prompt when a decision
  needs it.
- Pass: the mixed run's decode median and 16K prompt median are both above the NVIDIA arm's. The W6800 arm's 16K
  prompt median is reported beside them as the stretch target.
- Without the new argument: greedy output of a fixed prompt set identical to the build without the change, speed
  the same within noise.
- The tier's rows: a parity test that runs the same experts on the same input on the CUDA and HIP paths and checks
  the rows agree within a stated tolerance. The answers of a mixed run are not bit-identical; a short greedy prompt
  set is compared with the NVIDIA run and the differences reported.
- Both the CUDA and the HIP trees build.

## Risks

- The round trip per layer eats the gain. Phase 0 measures it before anything else is built.
- Two runtimes in one process: driver or runtime conflicts. Phase 0 starts both.
- Memory pinned by both drivers at once. Phase 0 checks it.
- RAM: new pinned buffers take page cache, which is already short. Moving about 30 GiB of experts onto the W6800
  should more than pay for that; to be checked in the prefill plan's log line.
- Prompt chunks overlap across stages, so two stages may ask the tier for work at once.
- The switches a peer turns off today (above) may cost more than the tier gains until they are back on.

## Left to the builder

- The exact C interface and names, and whether the engine side is a proxy class or a small interface.
- The sync mechanism beyond the first version.
- The ownership policy default (move, or stages never take tier-held experts), after measuring both.
- How the tier's prompt work is scheduled when stages overlap.
- Which buffers are registered with which runtime.
- The tolerance and method of the parity test.

## Suggested order

Phase 0 measurements; the library and loader with decode rows beside the layer split; ownership and adaptivity
across all stages; prompt rows; then tuning (the switches above, sync, the 3050's role).
