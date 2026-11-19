# gfx1030 (RDNA2) optimization design

Date: 2026-11-20. Branch: `w6800` (cut from upstream main, 0.2.16, engine 0.1.62).

## Intent

The fork will run on one AMD Radeon PRO W6800 (gfx1030, RDNA2, 32 GB) with the int8 KV cache (`--kv int8`, the default,
what the user calls kv=q8_0). The user wants the engine's code to use gfx1030 to its full capability. This is about the
code, not about configs or setups: no tuning of service flags, no model choice, no per-box launch recipes.

Upstream treats gfx1030 as "builds, unvalidated". Its fast paths go to NVIDIA tensor cores, RDNA3/RDNA4 WMMA, hipBLASLt or
Strix Halo tables. On gfx1030 those paths fall back to FP32 FMA, sequential kernels, extra launches or rocBLAS.

## Chosen approach: fill every gap

Three approaches were offered:

1. Profile first and rewrite only the kernels that cost the most.
2. **Fill every gap: give gfx1030 its own version of every path where it falls back today (chosen by the user).**
3. Retune only: rocBLAS tables and launch sizes, no kernel work.

The user chose 2. Every gap in the inventory below gets a dedicated gfx1030 path, and that path becomes the gfx1030
default.

**Working assumption (user, 2026-11-20).** Dedicated gfx1030 code is always better than a generic fallback. No change
is speed tested on its own, and no change is quality tested on its own. The reasons:

- testing every path slows development;
- it can miss compounding effects that only show when everything is together.

Speed and quality are compared once, at the very end, between the starting build and the finished build (see "Final
comparison").

## Goals

- Every gap in the inventory has a dedicated gfx1030 path that is on by default on gfx1030.
- gfx1030 gets its own defaults table in `src/core/arch_defaults.cpp`, the same way gfx12 and gfx1151 have theirs.
- New RDNA2 kernels use what the card has: wave32, 64 KiB LDS, `v_dot4_i32_i8` (sdot4, already mapped by `__dp4a`),
  `v_dot2_f32_f16` (unused anywhere in the engine today), DPP and permlanex16.
- The "unvalidated" markers for gfx1030 go away once the card has run the test suite and a model.

## Non-goals

- Other architectures. Their defaults and code paths stay as they are. A shared kernel may change only if its output on
  other archs stays bitwise the same (checked by the existing parity tests).
- Configs, service files, launch flags, model or quant choice for the box.
- Multi-GPU, PCIe share, the CPU expert pool, host scheduling, Windows, Metal, SYCL.
- hipBLASLt (it ships no gfx1030 kernels) and TF32 paths (no hardware for them).
- `strata-split` (`engine/src/split`). It needs 2 to 4 cards, so one W6800 cannot run it. The DeepSeek engine is out
  too, because upstream removed it in 0.2.13.

## Engines in scope

The user ruled that every engine is in scope:

- **The main engine** (`strata`, Qwen3.8-Flash-Next): tiers 1 to 3 below.
- **The Qwen3.6-35B-A3B engine** (`strata-q36`, `engine/src/qwen35`): the "Qwen3.6 engine" section.
- **The image engine:** the "Image engine" section.
- **Drafters (MTP, DFlash2):** inside the engines that run them.

## Anti-goals

- **No quality trade-offs.** No expert skipping, no fewer experts, and no precision drop chosen to buy speed. This
  holds even as an opt-in. `STRATA_ROUTE_TAIL_SKIP` must never be turned on for gfx1030.
- No speed claims in docs or commits before the final comparison has measured them (AGENTS.md rule).

## Correctness during development

Development checks only that a kernel is right, never how fast it is or how much it drifts. These checks are my call,
because a kernel bug found only at the final comparison is much harder to find.

- **A bitwise change**, for example a switch turned on or a straight port: the existing parity test of that path runs
  on the W6800 and must pass.
- **A new kernel with different arithmetic**, for example int8 q·k on sdot4 or FP16 products on `v_dot2`: it gets a
  parity test in `engine/tests` against the path it replaces. The test uses a tolerance that catches bugs, such as
  wrong indexing, NaNs or a missed tail, and not rounding.
- **The full engine test suite** (`ctest` on the HIP build) passes on the W6800 at the end of each engine's work.

Every change keeps an env off switch until the final comparison is done. A drift or a regression can then be traced by
turning switches off in groups.

## Final comparison

There is one comparative test, at the very end. It runs the starting build (`w6800` as cut) against the finished build,
on the W6800, with the int8 KV cache. Both builds use the same fixed workload in the same stint, on two or more engine
starts each, in swapped order.

**Speed:**

- Text engines (main and q36):
  - prompts of about 2K and 20K tokens;
  - answers at about 2K and about 20K context;
  - with no drafter, with MTP, and with DFlash2 on the main engine.
- Image: time per DiT step and VAE decode time, at the shipped default size and one size above 12,288 keys.

**Quality:**

- **Text.** KL of the finished build against the starting build's logits, over short and long prompt sets. For q36,
  the starting build's FP32 KV is the reference. The line is decode KL 0.03. Rounding between devices that passes
  that line is fine.
- **Image.** `terminal/tools/image/gate/compare.py` (PSNR, SSIM) on a fixed set of prompts and seeds, against the
  starting build. The line is how far the 3090's pictures of the same seeds are from the starting W6800 pictures, i.e.
  the device rounding already accepted. The 3090 is used only to make those reference pictures.

**If quality drifts past the line**, turn off-switch groups off to find the cause, then fix it. Fixing means a better
dedicated kernel, not falling back to the generic path. The user decides on any case that cannot be fixed.

**The KL tool (my call).** `terminal/tools/quant_quality.py` on this branch forces `--kv fp16`, keeps experts on the
CPU, and uses texts of at most 1,000 tokens, so it cannot see most of these kernels. Before the final comparison:

- port the short and long prompt sets and `--keep-caches` from `tools/quant_quality.py` on `origin/mixed-tier-perf`;
- let it run `--kv int8`, experts on the GPU, and prompts of at least 20K tokens.

The starting build's reference logits and pictures can be made at any time, since that build does not change.

**After the comparison:**

- Results go into `docs/superpowers/results/2026-11-20-gfx1030/`.
- The off switches are removed.
- docs/AMD_HIP.md gets the measured numbers.
- gfx1030 moves from "untested" to "tested" in `terminal/strata_control/drafting.py`, so that Auto treats DFlash2 on
  the W6800 like the other measured cards.

**Branch base (my call, from the user's branch cut).** `w6800` is a clean restart from upstream 0.2.16, which already
serves DFlash2. The 23 `mixed-tier-perf` commits are for the 3090 + W6800 tier and are not brought over. The one
exception is the bench tooling: the KL tool above and `tools/box_ab.sh`'s GPU pinning. The baseline is `w6800` as cut.

## The inventory (gaps found on 2026-11-20)

Paths are relative to `engine/`. The full search report, with file and line references for every item, is the basis.
Line numbers drift, so the builder re-finds each gate by name.

### Tier 1: existing exact switches that are off on gfx1030

These compile for gfx1030 and are bitwise elsewhere. They need a gfx1030 defaults-table entry, not new code.
Some are gated by an arch whitelist in code rather than the defaults table. The whitelist must be widened.

- **GDN prompt:** `STRATA_GDN_HEAD`, `STRATA_GDN_PP=2`, `STRATA_GDN_CONVL2`, `STRATA_GDN_NOY`, `STRATA_CVEC_FUSE`.
- **Decode attention:** `STRATA_ATTN_LANECELL`, `STRATA_ATTN_MERGE_V2`, `STRATA_QSA_ROWS`.
- **Matmul decode:** `STRATA_TSUM`, `STRATA_Q8_PACKED`, `STRATA_Q6_PACKED`, `STRATA_MMVF_ROWS`, `STRATA_SMALL_UNR`
  (the HIP return 0 must be lifted), `STRATA_B6_MMVQ_ROWS`, `q8_0_rows1`.
- **Experts decode:** `STRATA_EXP_MODE=8`, `STRATA_EXPERT_V2`, `STRATA_EXPERT_V2K`, `STRATA_EXPDEC`.
- **Fusions:** `STRATA_FUSE_HITS`, `STRATA_LFUSE`, `STRATA_QFUSE`, `STRATA_PLE_BATCH`, `STRATA_GDN_SPLIT`,
  `STRATA_FUSE_CONVAB`, `STRATA_FUSE_WAIT`, `STRATA_ONE_TOKEN_COMMIT`, `STRATA_SIDE_MIXER`.
- **Hyper-connection:** `STRATA_HEAD_MIX_MULTI`, `GR_NORM_FAST`, `HC_Q8`, `GR_Q32_G`, `GR_Q8_PFS`, `QFUSE_HC`, and
  `STRATA_HC_PERSIST`. HC_PERSIST was written for gfx1030 and measured at -2.8% to -4.2% per verify window on
  2x RX 6900 XT, but it is still opt-in.
- **Sampling:** multi-block argmax and `argmax_rows_wanted`. These are gfx12-only on HIP today.
- **Scorer:**
  - the exact thread scorer `STRATA_SCORES_THREAD`, which `scores_thread_arch()` refuses on anything but gfx1100 and gfx12;
  - the simt2 scorer's 28,672-token threshold, which was copied from an R9700;
  - `STRATA_TOPK_REG_MIN_BLOCKS`, tuned on gfx1201.
- **Drafting:**
  - `STRATA_MTP_HIP_PRIO`;
  - the DFlash2 `df2_q4_auto` variant choice, which was tuned on sm_120;
  - `STRATA_DF2_PRIO` and `STRATA_DF2_FORK`;
  - pipelined MTP windows and the drafter on stage 0, both gfx12-only.
- **Prompt chunk ceiling:** gfx1030 stays at 8,192 under `--prefill auto`. gfx12 and NVIDIA cc 8.0+ get 32K. This is a
  code default, not a config.
- **Retuned constants:** the simt2 scorer threshold, `STRATA_TOPK_REG_MIN_BLOCKS`, the DFlash2 variant choice, the
  q36 chunk size and the MTP head's grid cap. Without per-change runs, these are set from the card's own numbers: 60 CUs,
  64 KiB LDS, 512 GB/s, 32 GB. A kernel's own unit bench may be used to pick a constant while writing it.
- **Shared-expert stream:** `STRATA_SH_STREAM` stays off. It is a generic stream split, not gfx1030 code, and a gfx1030
  report (#816) found it slower.
- **rocBLAS table:** the gfx1030 rocBLAS solution table loads only when `STRATA_ROCBLAS_TUNING` is set. On gfx1030 it
  should load by itself when the installed rocBLAS version matches a shipped table.

### Tier 2: portable code that is compiled out or refused on HIP

- **A5. Batch slots of the gfx103x decode attention kernel.** `qsa_decode_attn_slots` returns false and
  `qsa_decode_attn_batch_rows` loops one launch per row when `pre75_attn()`. Add a slots/rows grid to the DPP kernel.
- **M2. Interleaved multi-column MMVQ** (`native_mmvq_il`, `IL_PF`, the 5-8 column wide path). It is under
  `#if !__HIPCC__`, and the HIP stubs fall back to plain `native_mmvq`. Port it to HIP (dp4a/sdot4 code, CUDA loads
  replaced) with its own gfx1030 row table. It serves the verify windows of MTP and DFlash2.
- **A3. Split decode attention** (`STRATA_QSA_SPLIT_ATTN`). HIP returns false. The split-K and log-sum-exp merge is
  arch-neutral.
- **qsa_pattn** (prompt attention without merge scratch, bitwise to the decode kernels). It is only reachable from the
  CUDA `STRATA_EXPERIMENTAL_SM60` build.
- **G1 part 1. The FMA chunked GDN recurrence** in `src/prefill/gdn_chunk.cu` (non-WMMA branch). It is only reachable
  from the CUDA SM60 build. Wire it into the HIP branch of `gdn_recurrence_impl`.
- **E6. Decode expert kernels:** `STRATA_GU_SWIGLU` (HIP returns false unless set), and `STRATA_TQ_STAGE` /
  `STRATA_TQ_SUB8` (always false on HIP).
- **S4.** The MTP shared-expert branch, which is CUDA only.
- **A2. Gated prompt attention.** The output-gate epilogue is fused only in the WMMA kernels. Fuse it into whatever
  gfx1030 prompt attention kernel results (tier 3, A1).

### Tier 3: new RDNA2 kernels

- **A1. Prompt attention for int8 KV.**
  - Today: gfx1030 loops `qsa_decode_attn_batch`, with FP32 dot products, partials through global scratch and a
    second merge pass.
  - New: an LDS-tiled online-softmax kernel. q is quantized to int8 once and q·k uses sdot4 on the raw int8 K codes
    with the per-64 group scales. p·v uses `v_dot2_f32_f16` or FP32. There is no scratch or merge pass, and the
    output gate is fused (A2).
- **A4. Decode attention q·k on sdot4.** The same int8 q·k idea in the gfx103x DPP decode kernel, in place of
  dequantizing K to FP32. No arch does this today.
- **M7. Dense prompt GEMMs on quantized weights.** Today HIP dequantizes to FP16 and calls rocBLAS for every dense
  prompt projection (`STRATA_PREFILL_MMQ` is never defined for HIP). New: an LDS-tiled sdot4 GEMM on q8_1 activations,
  modeled on `src/kernels/cuda/i8_expert.cuh`. It replaces dequantize plus rocBLAS for every quantized dense weight,
  in the main engine and in q36.
- **B5. Hyper-connection up GEMM with the mix epilogue** (`STRATA_HC_UPMIX`, today gfx11 WMMA or CUDA mma). The gfx1030
  version is an LDS-tiled FP16 GEMM on `v_dot2` that writes the mixed result directly. It saves the separate
  `gr_mix_r` pass (about 336 MB of traffic per 8K chunk).
- **G1 part 2. Chunked GDN recurrence for gfx1030.** A chunked WY recurrence with LDS tiles and `v_dot2`, built from
  the FMA chunked path ported in tier 2. It replaces the per-token `gdn_rec_cols_pipe_kernel`.
- **E2. int8 expert coverage.** Extend `i8_gu_type` / `i8_d_type` to the expert quant pairs it refuses today: Q3_K,
  Q4_0, IQ4_XS down, trellis via `tq_lut_i8.cuh`, and Hadamard-folded if possible. Set the 256-entry group floor,
  which was timed on gfx1031, from the new kernels' tile shape.
- **A7. Block scorer tile on `v_dot2`**, in place of the FP32 simt2 tile.

### Qwen3.6 engine (`strata-q36`)

The q36 engine has no gfx1030-specific code. Two facts shape the work:

- **No arch defaults.** `strata --model` jumps to q36 (`src/program/generate.cpp`, around `:3698`) before
  `apply_arch_defaults` runs (around `:4134`). So no arch defaults table reaches q36 on any card. q36 sets its own
  defaults in `q36_engine.cu` (around `:3946`). The gfx1030 table must be made to reach q36, either by applying it
  before the jump or by having q36 read it.
- **No int8 KV.** q36's KV cache is FP32 only, and an unknown `--kv` flag is logged and skipped. So the user's
  kv=q8_0 is silently ignored there today.

**Tier 1 (default and whitelist):**

- **Q2.** Add gfx103x to the `STRATA_EXPERT_V2K` default (bitwise; the Qwen3.6 shape with Q5_K down).
- **Q8.** Add gfx103x to `native_mmvq`'s `dev_class` (`q8_0_rows1`, bitwise).
- **Q3.** q36 never calls `gemm.set_f16_io(prompt_f16())`, so its prompt GEMMs miss the gfx103x FP16-out route and the
  rocBLAS table. Add the call. It covers the GEMMs that M7's sdot4 GEMM does not take (FP16 or BF16 weights).
- **Q4.** The FP16 weight cache is not used. M7's sdot4 GEMM reads quantized weights directly, so there is no
  dequantize step left to cache.
- **Retune for gfx1030:** the q36 auto chunk size and the MTP head's 1024-warp grid cap (both calibrated on gfx1201).
  See tier 1's "retuned constants".

**Tier 2 (port):**

- **Q1.** Port the tile-8 (128x128) int8 expert kernel to the K-quants. It is kept off them on every arch, and its
  54 to 62 KB of LDS fits gfx1030.

**Tier 3 (new):**

- **Q7. int8 KV cache for q36**, the same format as the main engine (`kv_q8.hpp`: int8 codes, FP16 scale per 64
  values), wired to `--kv int8` and made the default. This is the user's kv=q8_0. In the final comparison its KL is
  checked against q36's FP32 KV.
- **Q6. q36 decode and verify attention.** Today it is one thread per cell doing a serial 256-dim dot from global
  memory. The new kernel reads the int8 KV coalesced and does q·k on sdot4.
- **Q5. q36 prompt attention.** Today it is two batched rocBLAS GEMMs per KV head (FP16 in, FP32 out) plus a softmax
  pass and a 256 MB score buffer. The new kernel is a flash kernel for head dim 256 and GQA 16:2 on 64 KiB LDS tiles,
  reading the int8 KV. It may share code with A1.
- **q36 GDN.** It runs its own sequential kernels on every arch. Point it at the main engine's gfx1030 GDN kernels
  (G1, G2), and adapt them to the Qwen3.6 shapes where those differ.

### Image engine

**What ships.** The engine that ships is stable-diffusion.cpp's `sd-cli`, built by `terminal/setup.py` with Strata's
patches:

- `sdcpp_native`: the DiT kernels, opt-in `--image-native on`;
- `sdcpp_vae`: the VAE, opt-in `--image-vae on`;
- `sdcpp_dit`: the DiT GEMMs.

`strata-image` (`src/image`) is a hand-built test program that has never been compiled for HIP. The work targets the
shipped patches. `strata-image` and the patches share kernel sources, so changes go into both copies.

**Changes (my call):**

- **I1/I2. DiT attention.** A gfx1030 flash attention kernel (64-key LDS tiles, online softmax, `v_dot2_f32_f16`) for
  `qi21_kernels.cu` and the shipped `qi21-native.cu`. It replaces the exact FP32 `k_attn_causal` and ggml's
  `flash_attn_ext`. The same change removes the 48 KiB cap that fails a step above 12,288 keys on gfx1030 today.
- **I3. DiT GEMMs.** These dequantize every call and run rocBLAS FP16 to FP32. Replace them with an sdot4 GEMM that
  reads the Q8_0 blocks directly (shared with M7 where the shapes allow).
- **I4. VAE convolutions.** These gather to FP16 and then call rocBLAS. Replace them with an implicit-GEMM conv on
  `v_dot2` from LDS tiles. The mid-block attention uses I1's kernel.
- **I5. Text encoder.** Only `strata-image` uses it, not what ships. Skip it unless the shipped encoder starts using
  Strata's code.
- **Build.** Make `strata-image` build for HIP so it can serve as the harness for the image kernels' parity tests.

**Image correctness and speed** are part of the final comparison (see "Final comparison").

### Cleanup

- Drop gfx1030 from `_strata_hip_unvalidated` in `cmake/hip_backend.cmake`, and drop its build warning.
- Update `docs/AMD_HIP.md`: the RDNA2 section, the table rows, the settings table, and the stale `STRATA_DENSE_MMQ`
  line, with the final comparison's numbers.
- Remove the off switches once the final comparison is done.

## Box

- **Hardware.** The test box: one W6800, ROCm 7.2.4, EPYC 7F52. The 3090 stays in the box but is not used, except to
  make the image reference pictures. Builds are HIP-only for `gfx1030`.
- **Box rules:**
  - stop `strata.service` while using the card, and start it again after;
  - take `flock ~/bench-mixed/.box.lock`;
  - turn the W6800 GFXOFF off for the final comparison;
  - build with `-j$(nproc)`.
- **Workspace.** Work in a separate worktree of `w6800`, so the service's checkout on `mixed-tier-perf` is not
  touched.
- **Development.** During development the box builds, runs parity tests and smoke-runs each engine. There are no
  benchmark arms.

## Builder's calls

- The order of the work. Tiers are a way to group the inventory, not a required sequence. Engine order is open. My proposal: main engine, then q36, then image.
- Whether the q36 and main engine attention kernels share code.
- Which models and quants exercise each path. The builder picks the files on the box needed to cover the quant types
  each gap touches.
- Kernel tile sizes, launch shapes and LDS layouts.
- How to group changes under off switches, so that the final comparison can trace a drift.
- Commit granularity. Code comments follow the user's rule: one plain line each, no em dashes or semicolons.
