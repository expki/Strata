# gfx1030 Plan 1: foundation and the main engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the main engine (`strata`, Qwen3.8-Flash-Next) a dedicated gfx1030 path for every gap in the spec's
tiers 1 to 3, on by default on gfx1030. Each path has an off switch, and the kernels are proven correct by parity
tests on the W6800.

**Architecture:** One switch helper (`rdna2.hpp`) answers "is this a gfx103x device, and is this path on?" for every
gate. Paths that already exist get turned on by a gfx103x defaults table plus widened arch whitelists. Code that is
compiled out on HIP gets ported. Missing kernels are written new on sdot4 (`v_dot4_i32_i8`) and `v_dot2_f32_f16`.
Nothing is speed tested or KL tested per change. The one comparison happens in Plan 4.

**Tech Stack:** C++17, HIP (ROCm 7.2.4, clang), CMake/Ninja, ctest. Test box: `vdh@192.168.10.5`, W6800 = hip:0.

**Spec:** `docs/superpowers/specs/2026-11-20-gfx1030-optimization-design.md`

**The project's plans:**

1. This plan: foundation and the main engine. It also builds the shared kernels the later plans reuse: the int8 dense
   GEMM, the int8 expert kernels and the gfx1030 attention.
2. `2026-11-20-gfx1030-q36-engine.md`: the Qwen3.6 engine. Written when this plan finishes.
3. `2026-11-20-gfx1030-image-engine.md`: the image engine. Written when Plan 2 finishes.
4. `2026-11-20-gfx1030-final-comparison.md`: the KL tool, the one speed and quality comparison, and cleanup. Written
   when Plan 3 finishes.

## Global Constraints

- **Correctness:** dedicated gfx1030 code is assumed better than a generic fallback. No benchmark arms and no KL runs in
  this plan. Kernel parity tests run, and they catch bugs, not rounding.
- **Off switches:**
  - every change keeps an env off switch until Plan 4;
  - the master switch is `STRATA_GFX103X_DEFAULTS=0`;
  - a switch the user sets in the environment always wins.
- **Other archs:** their behavior does not change. A shared kernel may change only if its output on other archs is
  bitwise the same. The existing CUDA and gfx11/gfx12 parity tests in `engine/` must still compile.
- **Quality:** no expert skipping, no fewer experts. `STRATA_ROUTE_TAIL_SKIP` is never turned on for gfx1030.
- **Arch set:** "gfx103x" means gfx1030, gfx1031 and gfx1032 (one ISA). Only gfx1030 is run.
- **Out of scope:** multi-GPU, PCIe share, the CPU expert pool, host scheduling, Windows, Metal, SYCL, hipBLASLt, TF32,
  `strata-split`.
- **Comments:** one plain line each. No em dashes, no hyphens joining clauses, no semicolons (user's CLAUDE.md).
- **Branch:** `w6800`. Commit after every task. Do not push to GitHub. The box gets commits with
  `git push box w6800` (remote set up in Task 1).
- **Box rules:**
  - stop `strata.service` before using the card and start it after;
  - every build and test runs under `flock ~/bench-mixed/.box.lock`;
  - build with `-j$(nproc)`;
  - write scripts locally and copy them with scp, because ssh heredocs strip single quotes.
- **Line numbers** in this plan are from 0.2.16 and drift. Re-find each gate by the name given.

## Review Focus

1. **`STRATA_GFX103X_DEFAULTS=0` must truly return the starting behavior.** Plan 4 bisects drift with it. Task 2 pins
   it with a unit test, and Task 15 runs the suite with it set.
2. **Switches whose kernels are alternatives** (for example `STRATA_EXP_MODE=8` and `STRATA_EXPDEC`, `STRATA_EXPERT_V2`
   and `STRATA_EXPERT_V2K`) must not both claim one launch. Task 4's smoke run asserts that the decode output matches
   the starting build's tokens on the first 16 greedy tokens, as an identity smoke check for the bitwise switches.
3. **Tails and odd sizes in new kernels.** These are cell counts that are not a multiple of the tile, T not a multiple
   of 64, K not a multiple of 256, and a single query. Every new-kernel parity test in Tasks 9 to 14 includes those
   cases.
4. **An int8 KV page with all-zero scales** (fresh, unwritten cells) must not give NaN in the new attention kernels.
   Tasks 9 and 10 include a zero-scale page case.
5. **VRAM growth from new scratch buffers.** The new kernels must use the scratch they replace, or less. Tasks 9, 11
   and 12 check that the allocation sizes do not grow. Plan 4 checks VRAM at a 262,144 context.

---

### Task 1: Box worktree, baseline build and baseline test record

**Files:**
- Create: `tools/gfx1030/box_build.sh`: configure, build and ctest on the box for one worktree
- Create: `docs/superpowers/results/2026-11-20-gfx1030/baseline-ctest.txt`

**Interfaces:**
- Produces:
  - `tools/gfx1030/box_build.sh <worktree> <build_dir>`, which configures HIP gfx1030 with tests ON, builds every
    target and leaves `ctest` ready;
  - the baseline binaries in `~/gfx1030/baseline/` on the box;
  - the git remote `box`.

- [ ] **Step 1: Add the remote and the worktree**

  - Run locally: `git remote add box vdh@192.168.10.5:source/strata && git push box w6800`.
  - On the box: `cd ~/source/strata && git worktree add ~/source/strata-w6800 w6800`.
  - Expected: the worktree is on the same commit as local `w6800`.
- [ ] **Step 2: Write `tools/gfx1030/box_build.sh`**

  Use `~/source/Strata-data/LOCAL_INSTALL.md`'s HIP line, with these changes:
  - `-DSTRATA_BUILD_TESTS=ON`;
  - no `STRATA_PREFILL_MMQ` (gone in 0.2.16);
  - `-DCMAKE_HIP_ARCHITECTURES=gfx1030`;
  - `cmake --build <build_dir> -j$(nproc)` (all targets);
  - run under `flock ~/bench-mixed/.box.lock`.
- [ ] **Step 3: Build the baseline**

  - Stop `strata.service`.
  - Run `tools/gfx1030/box_build.sh ~/source/strata-w6800 build-hip`.
  - Copy `strata`, `strata-q36`, `strata-device` and the `.so` files to `~/gfx1030/baseline/`, with a `COMMIT` file
    holding `git rev-parse HEAD`.
  - Expected: the build succeeds.
- [ ] **Step 4: Record the baseline ctest**

  - Run `ctest --test-dir build-hip -j1 --output-on-failure 2>&1 | tee baseline-ctest.txt` with
    `HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=`.
  - Copy the file into the results folder.
  - Expected: a pass/skip/fail list. Skips (77) on gfx1030 are the gaps. Failures are noted, not fixed here.
- [ ] **Step 5: Commit**

  ```bash
  git add tools/gfx1030/box_build.sh docs/superpowers/results/2026-11-20-gfx1030/baseline-ctest.txt
  git commit -m "gfx1030: box build script and the baseline test record"
  ```

### Task 2: The rdna2 switch helper and the gfx103x defaults table

**Files:**
- Create: `engine/include/strata/kernels/rdna2.hpp`, `engine/src/kernels/rdna2.cpp`
- Modify: `engine/include/strata/kernels/gfx_arch.hpp` (add the predicate),
  `engine/src/core/arch_defaults.cpp` (the table)
- Test: `engine/src/core/arch_defaults_test.cpp`, new `engine/src/kernels/rdna2_test.cpp` (CPU test, registered in
  `engine/cmake/backends/cpu.cmake` next to `arch_defaults_test`)

**Interfaces:**
- Produces:
  - `inline bool gfx_arch_is_gfx103x(const char* gcn)` in `gfx_arch.hpp`, true for gfx1030, 1031 and 1032.
    `i8_expert_rule.hpp`'s `gfx_arch_is_gfx103x_i8` becomes a call to it.
  - `inline bool rdna2_switch_on(const char* gcn, const char* master, const char* own)` in `rdna2.hpp` (pure).
    It returns false unless `gfx_arch_is_gfx103x(gcn)`, because RDNA2 kernels are never forced onto another card.
    On gfx103x:
    - `own` (the path's own env value) starting with '1' gives true;
    - `own` starting with '0' gives false;
    - otherwise the answer is true unless `master` (the value of `STRATA_GFX103X_DEFAULTS`) starts with '0'.
  - `bool rdna2_on(const char* own_env_name)` in `rdna2.hpp`, defined in `rdna2.cpp`. It reads the current device's
    `gcnArchName` (cached per device id, as `scores_thread_arch()` does) and both env values, then calls
    `rdna2_switch_on`. It returns false on CUDA builds.
  - `std::vector<std::pair<std::string,std::string>> gfx103x_default_env(const char* gcn)` in `arch_defaults.cpp`.
    `arch_default_env` returns it for gfx103x (an empty table when `STRATA_GFX103X_DEFAULTS=0`).
- Every later task names its own switch `STRATA_RDNA2_<NAME>` and asks `rdna2_on("STRATA_RDNA2_<NAME>")`.

- [ ] **Step 1: Write the failing tests**

  `rdna2_test.cpp`:
  - `rdna2_switch_on("gfx1030", nullptr, nullptr)` is true;
  - `"gfx1030:xnack-"` is true;
  - `"gfx1100"` is false;
  - `("gfx1030", "0", nullptr)` is false;
  - `("gfx1030", nullptr, "0")` is false;
  - `("gfx1100", nullptr, "1")` is false;
  - `("gfx1030", "0", "1")` is true;
  - `("gfx1031", nullptr, nullptr)` is true;
  - `(nullptr, nullptr, nullptr)` is false.

  `arch_defaults_test.cpp`:
  - `arch_default_env("gfx1030", false, false)` holds every key of Step 3's table with its value;
  - with `STRATA_GFX103X_DEFAULTS=0` set it is empty;
  - `arch_default_env("gfx1100", false, false)` has none of the keys that are only in the gfx103x table.
- [ ] **Step 2: Run them to verify they fail**

  - Run locally: `cmake --build build-cpu --target rdna2_test arch_defaults_test && ctest --test-dir build-cpu -R "rdna2_test|arch_defaults_test"`.
  - Expected: FAIL. `rdna2_switch_on` is undefined and the keys are missing.
- [ ] **Step 3: Implement the helper and the table**

  The table holds the env-only switches from the spec's tier 1. These keys must each exist as a `getenv` in
  `engine/src`, so grep every name before adding it and drop any that does not exist (say so in the commit message):
  - `STRATA_GDN_HEAD=1`, `STRATA_GDN_PP=2`, `STRATA_GDN_CONVL2=1`, `STRATA_GDN_NOY=1`, `STRATA_CVEC_FUSE=1`
  - `STRATA_ATTN_LANECELL=1`, `STRATA_ATTN_MERGE_V2=1`, `STRATA_QSA_ROWS=1`
  - `STRATA_TSUM=1`, `STRATA_Q8_PACKED=1`, `STRATA_Q6_PACKED=1`, `STRATA_MMVF_ROWS=1`, `STRATA_SMALL_UNR=3`,
    `STRATA_B6_MMVQ_ROWS=1`
  - `STRATA_EXPERT_V2=1`, `STRATA_EXPERT_V2K=1`, `STRATA_EXPDEC=1`, `STRATA_EXP_MODE=8`
  - `STRATA_FUSE_HITS=1`, `STRATA_LFUSE=1`, `STRATA_QFUSE=1`, `STRATA_PLE_BATCH=1`, `STRATA_GDN_SPLIT=1`,
    `STRATA_FUSE_CONVAB=1`, `STRATA_FUSE_WAIT=1`, `STRATA_ONE_TOKEN_COMMIT=1`, `STRATA_SIDE_MIXER=1`
  - `STRATA_HEAD_MIX_MULTI=1`, `STRATA_GR_NORM_FAST=1`, `STRATA_HC_Q8=1`, `STRATA_GR_Q32_G=2`, `STRATA_GR_Q8_PFS=1`,
    `STRATA_QFUSE_HC=1`, `STRATA_HC_PERSIST=1`
  - `STRATA_SCORES_THREAD=1`, `STRATA_MTP_HIP_PRIO=1`, `STRATA_DF2_PRIO=1`, `STRATA_DF2_FORK=1`,
    `STRATA_GU_SWIGLU=1`

  When two keys pick alternative kernels for the same launch (read the code at each `getenv`), keep the one gfx1151's
  table uses. Note the dropped one in a one-line comment.

  `STRATA_SH_STREAM` and `STRATA_ROUTE_TAIL_SKIP` are never in the table.

  `apply_arch_defaults` prints one line, `strata: gfx103x (RDNA2): N switches on by default
  (STRATA_GFX103X_DEFAULTS=0 turns them off)`, the same way the gfx12 block does.
- [ ] **Step 4: Run the tests to verify they pass**

  Run the Step 2 command. Expected: PASS.
- [ ] **Step 5: Commit**

  `git commit -m "gfx1030: the rdna2 switch helper and the gfx103x defaults table"`

### Task 3: Widen the arch whitelists in attention, scoring and sampling

Each gate below refuses gfx1030 in code, not through the env table. Each gets `|| rdna2_on("<its switch>")` (or the
gcn form where the gate already holds a gcn string), so that it reads on by default on gfx103x.

**Files:**
- Modify `engine/src/kernels/cuda/qsa_select.cu`:
  - `scores_thread_arch()` returns 2 on gfx103x;
  - `sel_gfx12_device` is untouched (WMMA);
  - the simt2 start (28,672) becomes 16,384 on gfx103x: half the R9700's 64 CUs' worth, scaled to the W6800's 60 CUs
    and 64 KiB LDS, same tile;
  - the HIP `kRegMinBlocks` becomes 4,096 on gfx103x, as the gfx1151 table uses.
- Modify `engine/src/kernels/cuda/verify_kernels.cu`: `multi_block_head_ops`, `argmax_rows_wanted` and the
  `SMALL_UNR` HIP `return 0`.
- Modify `engine/cmake/strata_gpu_engine.cmake`: register `qsa_scores_gfx1100_parity`, `sampler_parity` and
  `qsa_topk_parity` for the HIP build if they are CUDA-only today.

**Interfaces:**
- Consumes: `rdna2_on(const char*)` (Task 2).
- Produces: none.

- [ ] **Step 1: Run the parity tests on the box to see them skip or refuse**

  Run `ctest -R "qsa_scores_gfx1100_parity|qsa_scores_parity|qsa_topk|sampler_parity"`.

  Expected: `qsa_scores_gfx1100_parity` skips (77) on gfx1030. Record the others.
- [ ] **Step 2: Change the gates as listed above**

  `qsa_scores_gfx1100_parity` must accept gfx103x as a device it tests. Add gfx103x to its device check.
- [ ] **Step 3: Run the tests on the box**

  Run the same command. Expected: all PASS, and `qsa_scores_gfx1100_parity` runs and passes its 35 cases
  bitwise.
- [ ] **Step 4: Commit**

  `git commit -m "gfx1030: exact scorer, register top-k and multi-block argmax on gfx103x"`

### Task 4: Widen the arch whitelists in matmul, experts, hyper-connection, drafting and prompt chunking

**Files** (each gate gains gfx103x through `rdna2_on` or `gfx_arch_is_gfx103x`):
- `engine/src/kernels/cuda/native_mmvq.cu`: `dev_class` / `q8_0_rows1` (switch `STRATA_RDNA2_Q8ROWS1`).
- `engine/src/kernels/cuda/iq_kernels.cu`:
  - the `STRATA_EXPERT_V2K` default (`g12 || strix`);
  - `STRATA_GU_SWIGLU`'s HIP `return false` (now on by default on gfx103x);
  - `STRATA_TQ_STAGE` and `STRATA_TQ_SUB8`, which are always false on HIP: build their kernels for HIP and make them
    on by default on gfx103x (switch `STRATA_RDNA2_TQ`).
- `engine/src/kernels/cuda/fused_gr.cu`: `GR_NORM_FAST`'s gfx12 default.
- `engine/src/core/native_dense.cpp`: the `HC_Q8` default.
- `engine/src/core/verify.cpp`:
  - `QFUSE_HC`, `FUSE_HITS`, `FUSE_WAIT`, `ONE_TOKEN_COMMIT`, `SIDE_MIXER`;
  - `HEAD_MIX_MULTI` (also `mtp.cpp`).
- `engine/src/core/mtp.cpp`:
  - `STRATA_MTP_HIP_PRIO`;
  - the shared-expert branch (around `:1255-1265`, CUDA only): build it for HIP, on for gfx103x (switch
    `STRATA_RDNA2_MTP_SHARED`).
- `engine/src/core/dflash2.cpp`:
  - `STRATA_DF2_PRIO` and `STRATA_DF2_FORK` default on for gfx103x;
  - `df2_q4_auto` in `engine/src/kernels/cuda/dflash2.cu` picks variant 26 on gfx103x, as gfx12 does (wave32 non-mma
    variant).
- `engine/src/program/generate.cpp`:
  - the pipelined MTP windows gate (gfx12 only);
  - `STRATA_HIP_QUEUE_PRIME`;
  - the `--prefill auto` ceiling, which becomes 32K on gfx103x (switch `STRATA_RDNA2_CHUNK32K`).

  The drafter-on-stage-0 and PCIe start share gates are multi-GPU and stay as they are.

**Interfaces:**
- Consumes: `rdna2_on`, `gfx_arch_is_gfx103x` (Task 2).

- [ ] **Step 1: Run the parity tests on the box before the change**

  Run `ctest -R "mmvq|native_dense|gr_parity|gr_multi|hc_persist|native_expert|native_grouped|iq_parity|s2_|shared_expert|route_window"`.

  Expected: a pass list. Record which ones skip on gfx1030.
- [ ] **Step 2: Change the gates as listed**
- [ ] **Step 3: Run the same tests plus `hc_persist_parity`**

  Expected: all PASS. None skip because of the arch.
- [ ] **Step 4: Run the identity smoke on the box**

  - Write `tools/gfx1030/smoke.sh <exe_dir>`. It starts the server from a fixed config (IQ4_XS files,
    `gpu [0]`, `--kv int8`, greedy), sends one 2K-token prompt from `terminal/data/quant_quality/xl`, prints the first
    16 token ids, and repeats with `--mtp` and with `--draft dflash2`.
  - Run it on `~/gfx1030/baseline` and on the new build.
  - Expected: the run completes without error in all three drafter modes. A token difference is printed and recorded,
    not failed. Bitwise switches should match, and the value-changing ones (`HC_Q8`) may not.
- [ ] **Step 5: Commit**

  `git commit -m "gfx1030: gfx103x defaults for matmul, expert, hyper-connection, drafting and chunk gates"`

### Task 5: rocBLAS solution table that loads by itself on gfx103x

**Files:**
- Modify: `engine/src/prefill/gemm.cu` (`create_rocblas_state`). With `STRATA_ROCBLAS_TUNING` unset on gfx103x, it uses
  an embedded table whose version string equals `rocblas_get_version_string`.
- Create: `engine/cmake/embed_rocblas_tables.cmake`, which generates `rocblas_tables_embedded.hpp` from
  `terminal/tools/hip/gfx1030-rocblas-*.txt`.
- Create: `terminal/tools/hip/gfx1030-rocblas-5.2.0.dabb6df2b9.txt`, made on the box with `terminal/tools/hip/tune_rocblas.cpp`
  for the box's exact rocBLAS version, only if no shipped file matches it byte for byte.
- Test: `engine/tests/prefill/rocblas_embedded_test.cpp` (CPU). It looks up a table by version string: an exact match
  is found, a "-dirty" variant of another hash is not, an unknown version gives nullptr.

**Interfaces:**
- Produces: `const char* rocblas_embedded_table(const char* version)` in `engine/src/prefill/rocblas_tuning.hpp`,
  nullptr when no table matches.
- Off switch: `STRATA_RDNA2_ROCBLAS=0`.

- [ ] **Step 1: Write the failing test** with the three cases above.
- [ ] **Step 2: Run it to verify it fails.** Expected: FAIL, undefined function.
- [ ] **Step 3: Make the table for the box's rocBLAS if needed, embed the tables, and wire `create_rocblas_state`**
- [ ] **Step 4: Run the test (PASS). On the box, run `hip_prefill_decode_golden`.** Expected: the engine start line
  says the table loaded.
- [ ] **Step 5: Commit**

  `git commit -m "gfx1030: the rocBLAS solution table loads by itself on gfx103x"`

### Task 6: Port the interleaved multi-column MMVQ to HIP

**Files:**
- Modify: `engine/src/kernels/cuda/native_mmvq.cu`. Move `native_mmvq_il` and its kernels (inside
  `#if !__HIPCC__` around `:1607-2337` and `:3588-4003`) out of the CUDA-only block. Replace the CUDA-specific loads
  (`ld.global.nc`, `__ldg` forms) with the `hip_compat` equivalents. Add a gfx1030 row in the rows table, `rows=4` for
  every type, the gfx1151 value if one exists. Delete the HIP stubs.
- Modify: `engine/cmake/strata_gpu_engine.cmake`, which registers `mmvq_il_parity` for HIP.

**Interfaces:**
- Produces: `native_mmvq_il_supported(type, ncols, n_out)` returns true on gfx103x for IQ4_XS, Q4_K, Q5_K, Q6_K and
  Q3_K at 2 to 8 columns.
- Off switch: `STRATA_RDNA2_MMVQ_IL=0` makes `native_mmvq_il_supported` false.

- [ ] **Step 1: Register `mmvq_il_parity` for HIP and run it on the box.** Expected: it fails or throws "CUDA only".
- [ ] **Step 2: Port the kernels as described**
- [ ] **Step 3: Run `mmvq_il_parity` and `mmvq_multi_parity` on the box.** Expected: PASS, bitwise to the plain
  `native_mmvq` for every type and column count the test covers.
- [ ] **Step 4: Commit**

  `git commit -m "gfx1030: interleaved multi-column MMVQ on HIP"`

### Task 7: Batch slots and rows for the gfx103x decode attention kernel

**Files:**
- Modify: `engine/src/kernels/cuda/qsa_decode_attn.cu`. `qsa_decode_attn_slots` and `qsa_decode_attn_batch_rows` launch
  `attn_chunk_kernel_pre75<M, 8, true>` with the slot index in `blockIdx.z` (pools from `pools.p[z]`), instead of
  returning false or looping when `pre75_attn()`.

**Interfaces:**
- Produces: `qsa_decode_attn_slots` returns true on gfx103x.
- Off switch: `STRATA_RDNA2_ATTN_SLOTS=0`.

- [ ] **Step 1: Run `qsa_slots_parity` and `batch_rows_parity` on the box.** Expected: the slots case reports it fell
  back.
- [ ] **Step 2: Add the z-indexed launch**
- [ ] **Step 3: Run both tests.** Expected: PASS, bitwise against the per-query calls, with 1, 3 and 8 slots.
- [ ] **Step 4: Commit**

  `git commit -m "gfx1030: one launch for all batch slots in the RDNA2 decode attention"`

### Task 8: Chunked GDN prompt recurrence on gfx1030

**Files:**
- Modify: `engine/src/prefill/gdn_chunk.cu`:
  - build its FMA (non-WMMA) branch for HIP;
  - `gdn_chunk_available()` returns true when `rdna2_on("STRATA_RDNA2_GDN_CHUNK")`;
  - in `gdn_chunk_fwd_kernel`, the q·k and k·v inner products of the 64-token chunk use `v_dot2_f32_f16`
    (`__builtin_amdgcn_fdot2`) on FP16 pairs of the LDS tiles, with FP32 accumulation, the solve and the state in FP32.
- Modify: `engine/src/prefill/kernels.cu`. In `gdn_recurrence_impl`'s `__HIPCC__` branch, call
  `gdn_chunk_recurrence` for `T - T % 64` tokens, then the existing kernel for the rest, the same way the CUDA SM60 path
  does.
- Modify: `engine/cmake/backends/hip.cmake` (add `gdn_chunk.cu` to the HIP sources) and
  `engine/cmake/strata_gpu_engine.cmake` (register `gdn_chunk_parity` for HIP).

**Interfaces:**
- Consumes: `gdn_chunk_recurrence(float* state, const float* h, const float* gate, const float* beta, float* oc, int64_t T, void* stream)`.
- Produces: none.

- [ ] **Step 1: Register `gdn_chunk_parity` for HIP. Add cases for T = 64, 1000 (a tail) and 8192. Run it on the box.**
  Expected: FAIL ("not built for HIP").
- [ ] **Step 2: Implement it as described**
- [ ] **Step 3: Run `gdn_chunk_parity` and `gdn_parity`.** Expected: PASS within `gdn_chunk_parity`'s existing
  tolerance, with no NaN in any case.
- [ ] **Step 4: Commit**

  `git commit -m "gfx1030: chunked GDN prompt recurrence with v_dot2 on gfx103x"`

### Task 9: RDNA2 prompt attention for int8 KV with the gate fused

**Files:**
- Create: `engine/src/kernels/cuda/qsa_prompt_attn_rdna2.cu`
- Modify: `engine/src/kernels/cuda/qsa_prompt_attn.cu`. On `__HIPCC__` and `rdna2_on("STRATA_RDNA2_PATTN")`,
  `qsa_prompt_attn_batch` and `qsa_prompt_attn_batch_gated` call the new kernel for int8 and FP16 pools.
- Modify: `engine/src/prefill/prefill.cpp`. The gated branch (around `:5300`) takes the gated call on gfx103x.
- Modify: `engine/cmake/strata_gpu_engine.cmake`. Register `qsa_prompt_attn_parity` for HIP, as `hip_prompt_attn_wmma`
  does.

**Interfaces:**
- Produces:
  - `bool qsa_prompt_attn_rdna2(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap, const QsaShapes& s, const float* qf, uint16_t* o16, int64_t ld16, float* attn, int64_t n_q, void* stream)`.
    With `qf != nullptr` it writes the gated FP16 output to `o16`. Otherwise it writes FP32 to `attn`. It returns false
    for geometry other than 24/2/256.
  - Plan 2 reuses the kernel's tile code for Qwen3.6's 16:2 GQA.
- Kernel shape (the algorithm, which the signature does not fix):
  - one block per (query, KV head), 12 query heads of that KV head together;
  - the query is quantized once to int8 with one FP32 scale per 64 dims;
  - cells are staged in LDS in chunks of 32;
  - q·k is `__builtin_amdgcn_sdot4` over raw int8 K codes, times the q scale and the cell's per-64 K scale;
  - online softmax in FP32;
  - p·v is `__builtin_amdgcn_fdot2` over FP16 p and FP16-widened V codes, times the V scale;
  - no global scratch and no merge pass.

- [ ] **Step 1: Run `qsa_prompt_attn_parity` on the box.** Expected: it skips or reports "no kernel".

  Add cases: cap not a multiple of 32, n_q = 1, and a page whose scales are all zero (output finite).
- [ ] **Step 2: Write the kernel and the hooks**
- [ ] **Step 3: Run `qsa_prompt_attn_parity`.** Expected: PASS within its existing bound against
  `qsa_decode_attn_batch`, with every output finite. Check that `prefill.cpp`'s attention scratch is not enlarged for
  this path.
- [ ] **Step 4: Commit**

  `git commit -m "gfx1030: RDNA2 prompt attention, int8 q.k on sdot4, fused gate"`

Spec note: the spec's tier 2 listed porting `qsa_pattn`, the FP32 prompt attention without merge scratch, to HIP.
Task 9's kernel replaces the same loop on gfx103x, so a port would be dead code there. It is not ported.

### Task 10: RDNA2 decode attention with int8 q·k and split cells

**Files:**
- Modify: `engine/src/kernels/cuda/qsa_decode_attn.cu`. The gfx103x path of `attn_chunk_kernel_pre75`, for int8 pools
  only (switch `STRATA_RDNA2_DATTN`), quantizes q once per head (int8, FP32 scale per 64) and does q·k with
  `__builtin_amdgcn_sdot4` on the raw K codes instead of `load8_q8` to FP32.
- Modify: `engine/src/kernels/cuda/qsa_prompt_attn.cu`. `qsa_decode_attn_split` runs on gfx103x (switch
  `STRATA_RDNA2_SPLIT`). It splits each query's cells across up to 60 blocks (one per CU), runs the same int8 chunk
  routine, and merges with the log-sum-exp rescale in `scratch`.
- Modify: `engine/src/core/verify.cpp` and `engine/src/core/mtp.cpp` so the callers take the split on gfx103x.

**Interfaces:**
- Produces: `qsa_decode_attn_split` returns true on gfx103x for int8 pools.

- [ ] **Step 1: Add a gfx103x case to `qsa_parity` and `qsa_merge_daily_parity`.** The case compares the new int8 path
  against the FP32 path with tolerance `max_abs_err <= 2e-3` on unit-scale inputs. Include zero-scale pages, cap = 1,
  and cap not a multiple of 8. Run it. Expected: FAIL (no int8 path).
- [ ] **Step 2: Implement both**
- [ ] **Step 3: Run `qsa_parity`, `qsa_merge_daily_parity`, `qsa_slots_parity` and `batch_rows_parity`.** Expected:
  PASS.
- [ ] **Step 4: Commit**

  `git commit -m "gfx1030: RDNA2 decode attention, int8 q.k on sdot4 and split cells"`

### Task 11: Int8 dense GEMM for quantized prompt weights

**Files:**
- Create: `engine/src/prefill/dense_i8.cu`, `engine/include/strata/prefill/dense_i8.hpp`
- Modify: `engine/src/prefill/gemm.cu`. On `rdna2_on("STRATA_RDNA2_DENSE_I8")`, `Gemm::native` calls
  `dense_i8_gemm` when `dense_i8_supported(type, K)`, and otherwise dequantizes as today.
- Test: `engine/src/kernels/dense_i8_parity.cpp`, registered for HIP.

**Interfaces:**
- Produces (Plans 2 and 3 call these):
  - `bool dense_i8_supported(int ggml_type, int64_t K)`: true for Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, IQ4_XS and IQ4_NL
    when K is a multiple of the block.
  - `void dense_i8_gemm(const uint16_t* X_f16, int64_t ldx, int ggml_type, const void* W, float* Y, int64_t ldy, int64_t T, int64_t N, int64_t K, float beta, void* stream)`.
  - `void dense_i8_gemm_q8(const void* X_q8_1, int ggml_type, const void* W, float* Y, int64_t ldy, int64_t T, int64_t N, int64_t K, float beta, void* stream)`.
- Kernel shape:
  - X is quantized to q8_1 rows once per call;
  - W blocks are decoded with `i8_expert.cuh`'s per-type decode routines into LDS int8 tiles;
  - the product uses the same sdot4 + scale structure as `native_expert_i8`, at tile 64x64, FP32 out, `beta` add.

- [ ] **Step 1: Write the failing test**

  `dense_i8_parity`: for each supported type, with T in {1, 37, 512}, N in {320, 2560}, K in {2560, 10240}, compare
  `dense_i8_gemm` against dequantize-to-FP32 plus a CPU FP32 GEMM. The pass bound is relative Frobenius error
  `<= 1e-2`, all finite. `beta = 1` must add.
- [ ] **Step 2: Run it on the box.** Expected: FAIL, undefined symbol.
- [ ] **Step 3: Implement the kernel and the `Gemm::native` hook**
- [ ] **Step 4: Run `dense_i8_parity` and `hip_prefill_decode_golden`.** Expected: PASS. The prompt scratch from
  `Gemm::init` is not grown for this path.
- [ ] **Step 5: Commit**

  `git commit -m "gfx1030: int8 dense GEMM on sdot4 for quantized prompt weights"`

### Task 12: Hyper-connection up projection with the mix epilogue on v_dot2

**Files:**
- Modify: `engine/src/prefill/kernels.cu`. In `gr_upmix`, a gfx103x kernel (switch `STRATA_RDNA2_UPMIX`) computes
  lo16 x w_up^T in LDS tiles with `__builtin_amdgcn_fdot2` on BF16 converted to FP16 pairs, with FP32 accumulation.
  The epilogue is `gr_mix_r`'s math, written straight to `mixed` and `mixed16`.
- Modify: `engine/src/prefill/prefill.cpp`. The caller takes `gr_upmix` on gfx103x.

**Interfaces:**
- Consumes: the `gr_upmix` signature in `include/strata/prefill/kernels.hpp:30`.

- [ ] **Step 1: Run `hip_gr_upmix_parity --max-t 8192` on the box.** Expected: SKIP (77) on gfx1030.
- [ ] **Step 2: Implement it, and make the test accept gfx103x**
- [ ] **Step 3: Run `hip_gr_upmix_parity`.** Expected: PASS within its bound, including T = 1 and T = 8191.
- [ ] **Step 4: Commit**

  `git commit -m "gfx1030: hyper-connection up projection with its mix on v_dot2"`

### Task 13: Int8 expert kernels, wider format coverage and tile 8 for K-quants

**Files:**
- Modify: `engine/src/kernels/cuda/i8_expert.cuh`:
  - `i8_gu_type` adds Q3_K (11) and Q4_0 (2);
  - `i8_d_type` adds IQ4_XS (23), Q4_K (12) and Q3_K (11);
  - `i8_tile8_type` allows 12, 13, 14 and 7 on gfx103x (tile 8 fits 64 KiB);
  - the per-type decode and `I8NR`/`I8NW` entries for the new types;
  - the trellis types go through `tq_lut_i8.cuh` if `i8_gu_type` can take them without a new decode loop, and
    otherwise are left out with a comment.
- Modify: `engine/include/strata/kernels/i8_expert_rule.hpp`. `kI8MinGroupEntries` is 128 on gfx103x (one tile-8 row
  block), and 256 elsewhere.
- Modify: `engine/src/prefill/prefill.cpp`. Hadamard-folded layers (`had_gu`/`had_d`) stay refused, with a one-line
  comment: the fold changes the input, which the int8 path cannot take.
- Test: `engine/src/kernels/native_i8_parity.cpp` gains cases for each new type pair and the K-quant tile-8 cases.

**Interfaces:**
- Produces: `native_expert_i8_supported(layout)` is true for the new pairs. Plan 2's Qwen3.6 tile-8 work (spec Q1) is
  done here.
- Off switch: `STRATA_RDNA2_I8_WIDE=0` restores the old type sets and the 256 floor.

- [ ] **Step 1: Add the cases to `native_i8_parity` and run it on the box.** Expected: the new cases report
  "unsupported".
- [ ] **Step 2: Implement it**
- [ ] **Step 3: Run `native_i8_parity`, `hip_prefill_expert_paths_bench --quick` and `--shape=q36`.** Expected: PASS
  within `native_i8_parity`'s existing bounds.
- [ ] **Step 4: Commit**

  `git commit -m "gfx1030: int8 experts cover Q3_K, Q4_0, IQ4_XS/Q4_K down, tile 8 for K-quants"`

### Task 14: Block scorer tile on v_dot2

**Files:**
- Modify: `engine/src/kernels/cuda/qsa_select.cu`. On gfx103x, the simt2 tile kernel (around `:2217-2260`) computes its
  q·k products as `__builtin_amdgcn_fdot2` on FP16 pairs with FP32 accumulation (switch `STRATA_RDNA2_SCORE_DOT2`).
  Everything else in simt2 stays.
- Test: `qsa_scores_parity`. Add a gfx103x case with tolerance `max_abs_err <= 1e-3` on the scores, and a top-k
  overlap of at least 99% of the selected cells.

- [ ] **Step 1: Add the case and run it.** Expected: FAIL (no dot2 path).
- [ ] **Step 2: Implement it**
- [ ] **Step 3: Run `qsa_scores_parity`, `qsa_select_long_parity` and `qsa_topk_parity`.** Expected: PASS.
- [ ] **Step 4: Commit**

  `git commit -m "gfx1030: block scorer tile on v_dot2"`

### Task 15: Whole-suite run and the main engine smoke run

**Files:**
- Create: `docs/superpowers/results/2026-11-20-gfx1030/plan1-ctest.txt` and `plan1-smoke.txt`

- [ ] **Step 1: Run the full suite on the box**

  Run `box_build.sh` and `ctest -j1`.

  Expected: every test that passed in `baseline-ctest.txt` still passes. Each gfx1030 skip from the baseline that a
  task in this plan targets now passes.
- [ ] **Step 2: Run the suite again with `STRATA_GFX103X_DEFAULTS=0`**

  Expected: the same pass list as the baseline.
- [ ] **Step 3: Run `tools/gfx1030/smoke.sh` on the new build in all three drafter modes, plus a 20K-token prompt**

  Expected:
  - no errors;
  - finite logits;
  - the startup line names the gfx103x switches and the rocBLAS table.
- [ ] **Step 4: Commit the two result files**

  `git commit -m "gfx1030: Plan 1 suite and smoke results"`
