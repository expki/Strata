# DFlash2 in the Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `strata.service` can start with the DFlash2 drafter instead of MTP, chosen by the run config, with every conversation-cache restore drafting from a correct DFlash2 window.

**Architecture:** A new `ServeDrafter` interface sits between the serve loop in `src/program/generate.cpp` and the drafters. `MtpServeDrafter` wraps today's `MtpDrafter` calls, and `Dflash2ServeDrafter` wraps `Dflash2Drafter`. MTP-only features (chain, E-9 draft K/V, batch slots, coupled sampling, draft history, snapshot K/V) stay reachable through `ServeDrafter::mtp()`. DFlash2 mode does not load MTP at all. It gets its context from taps on the last 2047 rows of every prompt chunk and on every verify window. Its 2048-row window is saved in half precision with every `ConversationCheckpoint` and restored on every restore.

**Tech Stack:** C++17, CUDA 13.2 (sm_86, RTX 3090) and HIP / ROCm 7.2.4 (gfx1030, W6800), CMake + Ninja, Python 3 (`serve/`, `strata_control/`), unittest.

**Spec:** `docs/superpowers/specs/2026-11-02-dflash2-serve-design.md`

## Global Constraints

- Branch `dflash2-serve` off `mixed-tier-perf`. Commit and push to origin (`expki/Strata`), never force-push, keep `main` untouched.
- Platforms: Linux, CUDA and HIP. Every C++ change builds in both the CUDA tree and the HIP tree before its task is done. Metal and SYCL builds need not keep building.
- MTP mode: greedy output identical to the `mixed-tier-perf` build, and decode and 20K-prompt speed within noise. This gates Task 4 and is re-checked in Task 7.
- No change to the model's answers in either mode. KL-passing rounding differences from a different verify width are fine. No skipped or approximated model work.
- Only the chosen drafter is loaded. DFlash2 mode never calls `MtpDrafter::load`.
- A DFlash2 request that cannot be met never silently runs MTP. An early failure (file, VRAM at load, refused mode) loads MTP and prints why. A failure after the expert cache is sized ends the start with an error that names the cause.
- Refused modes, which start MTP with a plain line: `--batch` > 0, a layer split (`multi_gpu`), `--pipeline-windows 2`, Metal, SYCL.
- Window storage: FP16, positions `[max(0, upto - 2048), upto)`, about 42 MB a checkpoint (5 layers x K and V x 2048 x 8 x 128 x 2 bytes). Charged through `ConversationCheckpoint::bytes()`.
- Code comments: a single short line each, plain prose, no em dashes, no semicolons (user's global CLAUDE.md).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Box (`ssh vdh@192.168.10.5`) rules:
  - Write scripts locally and `scp` them. Never use `ssh '...'` heredocs with quotes.
  - Take `flock ~/bench-mixed/.box.lock` for every build and run.
  - Back up `/etc/systemd/system/strata.service` before each stop. Stop `strata.service` for GPU stints and start it again after.
  - Keep each A/B arm to about 5 minutes. Use `tools/box_ab.sh` for clocks.
  - Dev builds live in a box worktree `~/source/strata-df2`. Until Task 7, don't touch `~/source/strata` or its builds: the service runs them.

## Review Focus

1. **A rewind into the middle of a long chat** (an agent edits an earlier turn). DFlash2 must draft from the window saved at that checkpoint, not from later rows still in the ring. Test: Task 6, step 6.
2. **A prompt shorter than the 2048 window, down to a one-token "hi" on a fresh start.** The ring is partly filled, the drafter drafts without out-of-range positions, and the answer equals MTP mode's. Test: Task 5, step 6.
3. **A request with an image.** The image rows pass through the prompt taps like text rows, and drafting carries on after them. Test: Task 5, step 6.
4. **A config whose `drafting_model` file is missing.** The service starts with MTP, and `/metrics` and the start line say why. Test: Task 7, step 2.
5. **Positions past the ring size, where `pos % 2064` wraps more than once.** A 20K-token prompt followed by a long answer keeps acceptance steady and matches MTP's answer. Test: Task 5, step 6.

---

### Task 1: DFlash2 drafter file from the incoai GGUF

**Files:**
- Create: `include/strata/core/dflash2_file.hpp`, `src/core/dflash2_file.cpp`
- Create: `src/core/dflash2_file_test.cpp` (CPU unit test), `src/core/dflash2_file_compare.cpp` (box tool, host only)
- Modify: `src/core/dflash2.cpp:122-226` (`load` reads through the new source), `CMakeLists.txt` (add the file to `strata_core` near line 769, the test near 1953, the tool near 1001 as `EXCLUDE_FROM_ALL`)

**Interfaces:**
- Produces:
  ```cpp
  namespace strata::core {
  struct Dflash2Tensor { std::string name, dtype; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
  class Dflash2Source {   // one drafter file, read on the host
  public:
      static std::unique_ptr<Dflash2Source> open(const std::string& path, std::string& err);   // DF2P pack or GGUF by magic
      virtual ~Dflash2Source() = default;
      virtual const std::map<std::string, std::string>& cfg() const = 0;   // the pack's cfg keys: hidden, layers, heads, ...
      virtual const std::vector<Dflash2Tensor>& tensors() const = 0;      // pack names, pack dtypes, offsets in a pack-layout blob
      virtual bool read(const Dflash2Tensor& t, uint8_t* dst, std::string& err) const = 0;   // t.bytes bytes in pack layout
  };
  std::string dflash2_pack_name(const std::string& gguf_name);           // "" when unknown
  void dflash2_q8_0_to_bf16(const uint8_t* blocks, int64_t n_values, uint16_t* out);
  }
  ```
  `Dflash2Drafter::load(path, device, err)` keeps its signature and accepts both formats.

- [ ] **Step 1: Set up the branch and the box worktree**

  Locally: `git switch -c dflash2-serve mixed-tier-perf && git push -u origin dflash2-serve`. On the box: `git -C ~/source/strata fetch origin && git -C ~/source/strata worktree add ~/source/strata-df2 dflash2-serve`. Configure `build-cuda` and `build-hip` in it with the same cache options as `~/source/strata/build-cuda` and `build-hip` (read their `CMakeCache.txt`, use `/usr/local/cuda/bin/nvcc`). Expected: both trees configure.

- [ ] **Step 2: Write the failing CPU test `dflash2_file_test`**

  ```cpp
  assert(dflash2_pack_name("blk.3.attn_q.weight") == "layers.3.self_attn.q_proj.weight");
  assert(dflash2_pack_name("blk.0.attn_output.weight") == "layers.0.self_attn.o_proj.weight");
  assert(dflash2_pack_name("blk.4.ffn_conv_base") == "layers.4.mlp_conv.base_kernel");
  assert(dflash2_pack_name("blk.1.attn_conv_proj.weight") == "layers.1.attention_conv.kernel_projection.weight");
  assert(dflash2_pack_name("selector_hidden.weight") == "candidate_selector.hidden_projection.weight");
  assert(dflash2_pack_name("selector_successor.weight") == "candidate_selector.successor_codebook");
  assert(dflash2_pack_name("token_embd.weight") == "");
  // one Q8_0 block, d = 0.5 (fp16 0x3800), q = -2, 0, 3, rest 0: values -1.0, 0.0, 1.5
  uint8_t b[34] = {0x00, 0x38, (uint8_t) -2, 0, 3};
  uint16_t o[32]; dflash2_q8_0_to_bf16(b, 32, o);
  assert(o[0] == 0xBF80 && o[1] == 0x0000 && o[2] == 0x3FC0);
  ```

- [ ] **Step 3: Run it and check it fails**

  Run: `cmake --build build-cuda --target dflash2_file_test` (local CPU build or the box CUDA tree). Expected: a compile error, because `dflash2_file.hpp` does not exist.

- [ ] **Step 4: Implement `dflash2_file.{hpp,cpp}` and move `Dflash2Drafter::load` onto `Dflash2Source`**

  The GGUF source uses `strata::artifact::GgufFile` (`include/strata/artifact/gguf_reader.hpp`).
  - **Name map:**

    | GGUF | Pack |
    |---|---|
    | `fc.weight` | `fc.weight` |
    | `enc.output_norm.weight` | `hidden_norm.weight` |
    | `output_norm.weight` | `norm.weight` |
    | `selector_hidden.weight` | `candidate_selector.hidden_projection.weight` |
    | `selector_{predecessor,successor}.weight` | `candidate_selector.{predecessor,successor}_codebook` |

    Per layer, `blk.N.` maps to `layers.N.`, with these name parts:

    | GGUF | Pack |
    |---|---|
    | `attn_q`, `attn_k`, `attn_v`, `attn_output` | `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` |
    | `ffn_gate`, `ffn_up`, `ffn_down` | `mlp.gate_proj`, `up_proj`, `down_proj` |
    | `attn_conv_proj`, `ffn_conv_proj` | `attention_conv.kernel_projection`, `mlp_conv.kernel_projection` |
    | `attn_norm`, `ffn_norm` | `input_layernorm`, `post_attention_layernorm` |
    | `attn_q_norm`, `attn_k_norm` | `self_attn.q_norm`, `self_attn.k_norm` |
    | `attn_conv_base`, `ffn_conv_base` | `attention_conv.base_kernel`, `mlp_conv.base_kernel` (no `.weight`) |

  - **Tensor data:** Q8_0 matrices copy byte for byte as dtype `q8_0`, with rows = `shape[1]` and cols = `shape[0]`. F32 tensors copy as `f32`. The two codebooks become `bf16` through `dflash2_q8_0_to_bf16`, with rows = 248320 and cols = 256.
  - **cfg from `dflash.*` metadata:**
    - `hidden` = `embedding_length`, `layers` = `block_count`, `heads` and `kv_heads` from the attention head counts, `head_dim` = `attention.key_length`
    - `inter` = `feed_forward_length`, `window` = `attention.sliding_window`, `rope_theta` = `rope.freq_base`, `eps` = `attention.layer_norm_rms_epsilon`
    - `block`, `conv_group` and `conv_kernel` from `block_size`, `conv_group_size` and `conv_kernel_size`, `rank` = `selector_rank`, `top_k` = `selector_top_k`
    - `taps` = `target_layers`, `mask_token` = `tokenizer.ggml.mask_token_id`, `vocab` = the codebook row count
  - Offsets are laid out 256-byte aligned as in `tools/dflash2_pack.py`.

- [ ] **Step 5: Run the CPU test**

  Run: `ctest --test-dir build-cuda -R dflash2_file_test --output-on-failure`. Expected: PASS.

- [ ] **Step 6: Write and run `dflash2_file_compare` on the box (host only, runs beside the service)**

  Usage: `dflash2_file_compare <pack> <gguf>`. It opens both sources and checks four things:
  - the same tensor set and every cfg key equal;
  - each `q8_0` and `f32` tensor byte-equal;
  - each codebook value within `|a - b| <= max(|a|) / 254 + bf16 ulp(|b|)` of its row's maximum;
  - prints `OK <n> tensors` or the first mismatch.

  Run it with `~/source/Strata-data/dflash2/qwen3.8-flash-next-dflash2.q8.pack` and `/opt/models/incoai_Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf`. Expected: `OK 81 tensors` (3 top, 15 per layer x 5, 3 selector). If the two norms or the taps differ, fix the map, not the check.

- [ ] **Step 7: Build both trees and commit**

  Build the `strata` and `dflash2_parity` targets in both `build-cuda` and `build-hip` (the HIP tree with its own cache). Then commit with the message `dflash2: the drafter loads from the incoai GGUF as well as the DF2P pack`.

---

### Task 2: Drafter window save and restore, batched context

**Files:**
- Modify: `include/strata/core/dflash2.hpp`, `src/core/dflash2.cpp` (`context` at 489, new methods), `src/kernels/cuda/dflash2.cu` and `include/strata/kernels/` for an f32 ↔ f16 ring copy if no existing kernel fits
- Create: `src/core/dflash2_window_test.cpp` (GPU, `EXCLUDE_FROM_ALL`, next to `dflash2_parity`)

**Interfaces:**
- Consumes: `Dflash2Drafter::load` from Task 1.
- Produces, on `Dflash2Drafter`:
  ```cpp
  static constexpr int kWindow = 2048;
  size_t window_bytes(int64_t upto) const;   // header + min(upto, 2048) rows x L x 2 x NKV x HD x 2 bytes
  bool save_window(int64_t upto, std::vector<uint8_t>& out, std::string& err);    // FP16 K and V of positions [max(0, upto - 2048), upto)
  bool restore_window(const std::vector<uint8_t>& in, int64_t upto, std::string& err);   // false if `in` was saved at another upto, ring untouched then
  void clear_ring(int64_t upto);   // zero the ring, then context starts at upto (no stale rows)
  ```
  `context()` keeps its signature but launches all 8-row groups before one stream sync.

- [ ] **Step 1: Write the failing GPU test `dflash2_window_test <pack-or-gguf> <case.bin>`**

  The case file comes from `tools/dflash2/mk_parity_case.py` and gives `n_ctx` tap rows. The drafter is bound with `bind_raw`, as in `dflash2_parity.cpp`. The test has six parts:
  - **(a)** `context(rows 0..n_ctx-2)`, then `save_window(n_ctx-1, w)`, then `draft(last row, 1, n_ctx-1, anchor)` gives drafts A.
  - **(b)** On a fresh drafter: the same context, plus 300 rows of different taps at positions `n_ctx-1 .. n_ctx+298`. Then `restore_window(w, n_ctx-1)` and the same draft give drafts B. Assert `A == B` (all 7 tokens) and `|pA - pB| < 2e-3`.
  - **(c)** `restore_window(w, n_ctx)` returns false.
  - **(d)** `w.size() == window_bytes(n_ctx-1)`.
  - **(e)** Repeat (a) with `n_ctx = 5000`, so positions wrap the ring twice, and assert the drafts match a drafter that only saw the last 2048 rows.
  - **(f)** Time `context()` over 2047 rows and print it.

- [ ] **Step 2: Run it and check it fails**

  Build the target. Expected: a compile error, because `save_window` is undeclared.

- [ ] **Step 3: Implement the four methods and the batched `context`**

  The header records `upto`, the row count and the layout. Ring slot for position `q` is `q % kRing`.

- [ ] **Step 4: Run the test on the box in a stint, on the 3090 (CUDA tree) and the W6800 (HIP tree)**

  Stop the service and generate a case with `mk_parity_case.py`. Run `dflash2_window_test` from both trees with the GGUF. Expected: `PASS` from both, and the `context` time printed. Then start the service again.

- [ ] **Step 5: Commit**

  Message: `dflash2: the drafter's window saves and restores in FP16, context rows run as one batch`.

---

### Task 3: Prompt taps on the last 2047 rows of every chunk

**Files:**
- Modify: `include/strata/prefill/prefill.hpp:229-235`, `src/prefill/prefill.cpp:3996-4040, 6257-6260`
- Create: `src/prefill/chunk_taps_test.cpp` (CPU), registered in `CMakeLists.txt`

**Interfaces:**
- Produces:
  ```cpp
  namespace strata::prefill {
  // rows of a chunk at p0 of T rows that are tapped: [t0, T), t0 relative to the chunk
  int64_t chunk_tap_first(int64_t T, int64_t p0, int64_t first_pos, int64_t per_chunk);
  }
  void Prefill::set_chunk_taps(const std::vector<int>& layers, int64_t per_chunk);   // per_chunk 2047: every chunk taps its last rows
  ```
  `set_taps` (the CLI's single range) keeps working. `on_taps(taps, rows, pos0, err)` is unchanged.

- [ ] **Step 1: Write the failing CPU test**

  ```cpp
  using strata::prefill::chunk_tap_first;
  assert(chunk_tap_first(20480, 0, -1, 2047) == 20480 - 2047);   // per-chunk mode
  assert(chunk_tap_first(1000, 4096, -1, 2047) == 0);             // short chunk: all rows
  assert(chunk_tap_first(4096, 0, 18000, 0) == 4096);            // CLI range mode: chunk before first_pos taps nothing
  assert(chunk_tap_first(4096, 16384, 18000, 0) == 18000 - 16384);
  ```

- [ ] **Step 2: Run it and check it fails**

  Expected: a compile error.

- [ ] **Step 3: Implement it**

  Replace the inline `tap_t0` at `prefill.cpp:4032` with `chunk_tap_first`. Size `tap_cap_` to `per_chunk` in per-chunk mode. `set_chunk_taps` and `set_taps` exclude each other.

- [ ] **Step 4: Run the CPU test and the existing prefill tests**

  Run: `ctest --test-dir build-cuda -R "chunk_taps_test|prefill_plan_test" --output-on-failure`. Expected: PASS. Build both trees.

- [ ] **Step 5: Commit**

  Message: `prefill: per-chunk DFlash2 taps (the last 2047 rows of every chunk)`.

---

### Task 4: `ServeDrafter` interface, with MTP behind it

**Files:**
- Create: `include/strata/program/serve_drafter.hpp`, `src/program/serve_drafter.cpp` (`MtpServeDrafter` here, `Dflash2ServeDrafter` in Task 5)
- Modify: `src/program/generate.cpp`, serve block 9564-17779:
  - request setup 14930-14935
  - draft round 17130-17140
  - `read_windows` 15091-15140
  - verifier init 10471
  - the 39 `use_mtp` sites
- Modify: `CMakeLists.txt` (add `serve_drafter.cpp` to the engine sources)

**Interfaces:**
- Consumes: `core::MtpDrafter`, `core::Verifier`.
- Produces:
  ```cpp
  namespace strata::program {
  struct DraftRound { int n = 0; int32_t tok[8] = {}; float prob[8] = {}; bool has_prob = false; };
  struct RoundInput { const core::Verifier& ver; int T; const int32_t* outv; int64_t p; int a; };   // p before the advance, a accepted
  class ServeDrafter {
  public:
      virtual ~ServeDrafter() = default;
      virtual const char* name() const = 0;                                   // "mtp" | "dflash2"
      virtual core::MtpDrafter* mtp() { return nullptr; }                      // MTP-only features
      virtual std::vector<int> tap_layers() const { return {}; }               // verifier and prompt taps
      virtual int max_window() const = 0;                                      // widest T this drafter fills
      virtual bool begin_request(int64_t resume, int64_t prompt_len, std::string& err) = 0;
      virtual bool prompt_rows(const float* taps_dev, int64_t n, int64_t pos0, std::string& err) { return true; }
      virtual bool window_rows(const core::Verifier& ver, int T, int64_t pos0, std::string& err) { return true; }
      virtual bool draft(const RoundInput& in, DraftRound& out, std::string& err) = 0;
      virtual bool save_window(int64_t upto, std::vector<uint8_t>& out, std::string& err) { out.clear(); return true; }
      virtual bool restore_window(const std::vector<uint8_t>& in, int64_t upto, const char* kind, std::string& err) { return true; }
      virtual void report(FILE* f) const = 0;                                  // rounds, ms a round, VRAM
  };
  std::unique_ptr<ServeDrafter> make_mtp_serve_drafter(core::MtpDrafter& mtp);
  }
  ```

- [ ] **Step 1: Record the MTP baseline on the box, from the current `~/source/strata` build, in a stint**

  Use the IQ4_XS config with the identity flags `--adapt-swaps 0 --pcie-frac 0` plus greedy requests. Send the fixed prompt set: chats, code, the 20K prompt, and a 3-turn conversation that hits the cache. Save each full text and the decode and 20K-prompt tok/s to `~/bench-mixed/df2-mtp-base/`. Write the driver locally and `scp` it.

- [ ] **Step 2: Add the interface and `MtpServeDrafter`, and route the serve loop through it**

  - `serve_drafter` is a `std::unique_ptr<ServeDrafter>` made after `ver.init` (10471).
  - Classify each `use_mtp`:
    - A model drafter exists: change it to `serve_drafter != nullptr`.
    - An MTP-only feature: change it to `serve_drafter && serve_drafter->mtp()`.
  - `begin_request` takes 14932-14935. `draft` takes 17138-17140. MTP's snapshot K/V calls keep using `mtp()->kv_state()`.
  - The CLI loop stays as it is.

- [ ] **Step 3: Build both trees and run the local tests**

  Run: `python -m pytest tests/terminal -q` and `ctest --test-dir build-cuda --output-on-failure` (box CUDA tree). Expected: all pass.

- [ ] **Step 4: Run the identity check and speed A/B on the box**

  Run the new build with the same driver and flags as step 1. Expected:
  - every text byte-identical to `df2-mtp-base`;
  - decode and 20K prompt within the start-to-start spread, with two starts per build in A B B A order.

  Record the numbers in `~/bench-mixed/plan-results.md` under a "DFlash2 serve" heading.

- [ ] **Step 5: Commit**

  Message: `serve: one drafter interface, MTP behind it, no change to its output or speed`.

---

### Task 5: DFlash2 drafts in the server

**Files:**
- Modify: `src/program/generate.cpp:6316-6400` (drafter choice at start), the VRAM reserve at 6745 and 6992, the verifier init at 10471, the prompt reads around 15378 and 15609, `read_windows` 15091-15140, and the round at 17138
- Modify: `src/program/serve_drafter.cpp` (add `Dflash2ServeDrafter`)

**Interfaces:**
- Consumes:
  - from Task 1: `Dflash2Drafter::load`, which takes the GGUF;
  - from Task 2: `save_window`, `restore_window`, `clear_ring` and the batched `context`;
  - from Task 3: `Prefill::set_chunk_taps(layers, 2047)`;
  - from Task 4: `ServeDrafter`.
- Produces: `std::unique_ptr<ServeDrafter> make_dflash2_serve_drafter(core::Dflash2Drafter& d, const core::Verifier& ver, int cap)`. Its `max_window()` is `min(o.spec, 1 + kDrafts, cap)`.
- Start line: `strata serve: DFlash2 drafter on (<file>, graphs captured in N ms), MTP not loaded`.
- Refusal line: `strata serve: --draft dflash2 is off: <reason>; the MTP drafts instead`, where `<reason>` is one of these:
  - `it is not built for --batch, a layer split or --pipeline-windows 2`
  - `it needs a native pack`
  - `<load error>`

- [ ] **Step 1: Pick the drafter at start**

  If `o.draft == "dflash2"` and no refused mode applies, `df2.load` runs before MTP.
  - **On success:** keep `draft_vocab = o.mtp_draft_vocab.empty() ? o.mtp + "/draft_vocab.bin" : o.mtp_draft_vocab`, then clear `o.mtp`. MTP is then never loaded, and every MTP path is off as with no `--mtp`.
  - **On failure:** print the refusal line and let MTP load as today.
  - **VRAM:** reserve the draft-head bytes (`6745` and `6992` already add them under `use_df2`) in the serve path too.

- [ ] **Step 2: Turn on taps and wire DFlash2 into the loop**

  - Before `ver.init`, call `ver.set_taps(df2.tap_layers())`.
  - After `ver.init`, run `df2.bind(wt, &native_head, draft_vocab)` and `df2.prepare(ver.tap_dev())`. A failure here exits with code 1 and the CLI's VRAM hint text.
  - Around each prompt read (`sp.run`), call `sp.set_chunk_taps(layers, 2047)` with `sp.on_taps` going to `prompt_rows` (which runs `df2.context`). Clear them after the read.
  - In `read_windows`, call `window_rows` after `ver.commit`.
  - `begin_request`: with `resume == 0`, call `clear_ring(0)`. Any other resume is handled by the restores in Task 6. Until then, `clear_ring(resume)`.
  - Round: `df2.draft(ver.tap_dev(), a + 1, p, outv[a], ...)`. Drafts are offered without probabilities (`has_prob = false`), as the suffix drafter does.
  - The window is cut by `--spec-min-p` on the product of the selector probabilities, as the CLI does at 18669.

- [ ] **Step 3: Build both trees**

  Run: build `strata` in `build-cuda` and `build-hip`. Expected: no errors or warnings in the touched files.

- [ ] **Step 4: Start it on the box in a stint**

  Use the IQ4_XS config with `--draft dflash2 --draft-model /opt/models/incoai_Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf`. Expected:
  - the DFlash2 start line, and no `strata mtp:` lines;
  - the server ready;
  - `/metrics` shows drafts offered after one chat.

- [ ] **Step 5: Run the refusal checks**

  Start with `--batch 2` added, and separately with a missing `--draft-model` path. Expected: the refusal line, and MTP serves.

- [ ] **Step 6: Run the behavior checks against MTP mode (Review Focus 2, 3 and 5)**

  Use the identity flags and greedy requests, comparing against `df2-mtp-base`:
  - **(a)** The fixed prompt set: each text identical to MTP mode, or the KL gate passes on the ones that differ. Use `tools/quant_quality.py prompt` against `kl-ref` built with the same flags.
  - **(b)** A one-token prompt "hi" on a fresh start: the answer matches MTP mode, no error.
  - **(c)** One vision request with an image: an answer, and drafts accepted after it.
  - **(d)** The 20K prompt plus a 1,500-token answer: the answer matches MTP mode, and the accept rate in the second half is within 10 points of the first half.
  - **(e)** One sampled chat at temperature 1.0.

  Record the accept rate, decode tok/s and 20K prompt tok/s for both modes.

- [ ] **Step 7: Commit**

  Message: `serve: --draft dflash2 drafts in the server (MTP not loaded), taps from every prompt chunk and window`.

---

### Task 6: DFlash2 window in every checkpoint and on every restore

**Files:**
- Modify: `include/strata/core/conversation_cache.hpp:24-57` (`ConversationCheckpoint::drafter`, counted in `bytes()`)
- Modify: `include/strata/core/conversation_file.hpp` and its `.cpp` (session files carry it, file version bumped)
- Modify: `src/program/generate.cpp`:
  - `checkpoint_at` 11113-11279 and the park save 10874-11003
  - restores at 14730-14757 (borrow), 14758-14817 (whole image), 14856-14907 (rewind and `pin=`), 13979-14045 (disk RESTORE)
- Test: the existing CPU tests `conversation_cache_test.cpp` and `conversation_file_test.cpp`

**Interfaces:**
- Consumes: `ServeDrafter::save_window` and `restore_window(in, upto, kind, err)` (Task 4), implemented by `Dflash2ServeDrafter` (Task 5).
- Produces:
  - `std::vector<uint8_t> ConversationCheckpoint::drafter`, where empty means no window;
  - the log line `strata serve: dflash2: window <restored|missing> at <pos> (<kind>)`, where kind is `rewind`, `image`, `borrow` or `disk`;
  - the counters `df2_windows_restored` and `df2_windows_missing` in the engine stats the server reads.

- [ ] **Step 1: Write the failing CPU tests**

  - **`conversation_cache_test`:** a checkpoint with `drafter.resize(42u << 20)` has `bytes()` larger by at least `42u << 20`. A `SavedConversation` whose `live.drafter` is set counts it in its `bytes()`.
  - **`conversation_file_test`:** write and read back a session file whose deepest checkpoint and `live` carry 1 MiB `drafter` vectors of a known pattern. Assert the vectors read back are equal. A file from the old version is refused with the existing version error.

- [ ] **Step 2: Run them and check they fail**

  Expected: a compile error on `drafter`.

- [ ] **Step 3: Add the field, count it, and carry it in session files**

- [ ] **Step 4: Save and restore in the serve loop**

  - **Save:** `checkpoint_at` and the park call `save_window` with the position the checkpoint or the image ends at.
  - **Restore:** each restore calls `restore_window` with the window that matches its resume position:
    - **rewind and `pin=`:** the mounted checkpoint;
    - **whole image:** `live`;
    - **borrow:** the donor checkpoint at `parked.tokens`;
    - **disk:** the file's checkpoint or live state.
  - **`STRATA_CKPT_REREAD`:** re-collects through the prompt taps and needs no call.
  - **Missing window:** an empty or mismatched window means `clear_ring(resume)`, plus the `missing` line and counter. It never drafts from rows past the resume point.

- [ ] **Step 5: Run the CPU tests and build both trees**

  Run: `ctest --test-dir build-cuda -R "conversation_" --output-on-failure`. Expected: PASS.

- [ ] **Step 6: Run the restore checks on the box in a stint (Review Focus 1)**

  DFlash2 config, conversation cache on, greedy. Write the driver locally and `scp` it.
  - **(a)** A 3-turn chat of about 6K tokens a turn. Expected: turn 2 and 3 logs show `window restored ... (image)`. Record each turn's accept rate next to the same text read fresh (a restart plus the whole conversation as one prompt). This is recorded, not a gate: the window stays FP16 whatever it shows (user, 2026-11-02).
  - **(b)** Then a request that keeps turns 1 and 2 but replaces turn 3. Expected: `window restored ... (rewind)` and an answer identical to MTP mode's for the same request.
  - **(c)** A second conversation that shares turn 1's system prompt. Expected: `(borrow)` or `(rewind)` restored.
  - **(d)** Disk: `SAVE` and then `RESTORE` through the server's session API. Expected: `(disk)` restored.
  - **(e)** The `missing` counter stays 0 across (a) to (d).

  Record checkpoint RAM with and without DFlash2 from the log's cache lines.

- [ ] **Step 7: Commit**

  Message: `conversation cache: the DFlash2 window is saved with every checkpoint and restored on every restore`.

---

### Task 7: Config switch, box configs, deploy

**Files:**
- Modify: `strata_control/drafting.py`:
  - `ENGINE_SERVES_DFLASH2 = True` (line 146)
  - Swift names in `_SUPPORTED_NAME` and `"swift"` in `SUPPORTED_FAMILIES`
  - a `"drafting_model"` config key that `decide`/`apply` use instead of `pack_path`
  - `fallback_note` matching the new refusal text
- Modify: `tests/terminal/test_drafting.py`:
  - flip the test at line 118 to assert the switch is on;
  - swift names supported;
  - `drafting_model` used when given, and a missing file gives MTP with `NOT_DOWNLOADED`;
  - `fallback_note` on the new lines.
- Box: `~/source/strata/strata-swift-iq4_xs-df2-mix.json` and `strata-swift-q6k-df2-mix.json`

**Interfaces:**
- Consumes: the engine's start and refusal lines (Task 5).
- Produces: config keys `"drafting": "dflash2"` and `"drafting_model": "<path>"`, and switch targets `iq4_xs-df2` and `q6k-df2` for `~/bench-mixed/strata-switch.sh`, which needs no change because it resolves `strata-swift-<name>-mix.json`.

- [ ] **Step 1: Write the failing Python tests**

  ```python
  def test_engine_serves_dflash2(self): self.assertTrue(drafting.ENGINE_SERVES_DFLASH2)
  def test_swift_is_supported(self):
      self.assertTrue(drafting.supported_model("swift-1.5-qwen3.8-flash-next-iq4_xs"))
      self.assertTrue(drafting.supported_model(family="swift"))
  def test_drafting_model_path_is_used(self):  # a 2 MiB temp file
      cfg = {"drafting": "dflash2", "drafting_model": str(f), "model_name": "swift-1.5-qwen3.8-flash-next-q6_k", "args": ["--mtp", "x/mtp/rt"]}
      d = drafting.apply(cfg, klass="ampere"); self.assertEqual(d.drafter, "dflash2"); self.assertIn(str(f), cfg["args"])
  def test_missing_drafting_model_gives_mtp(self):  # Review Focus 4
      d = drafting.apply({**cfg, "drafting_model": "/nope.gguf"}, klass="ampere"); self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_DOWNLOADED))
  def test_fallback_note_reads_new_refusal(self):
      self.assertEqual(drafting.fallback_note("--draft dflash2 is off: it is not built for --batch, a layer split or --pipeline-windows 2"), drafting.mtp_line(drafting.NOT_HERE))
  ```

  Also update the existing tests that assumed Swift unsupported or the switch off.

- [ ] **Step 2: Run them, implement, and run them again**

  Run: `python -m pytest tests/terminal/test_drafting.py -q`. Expected: the new tests fail first, then all pass. Then run `python -m pytest tests/terminal -q`, and expect all to pass.

- [ ] **Step 3: Commit and push**

  Message: `drafting: the server serves DFlash2, Swift is supported, the config can name the drafter file`.

- [ ] **Step 4: Run a whole-branch review**

  Run `superpowers:requesting-code-review` on `dflash2-serve` against `mixed-tier-perf`. Fix what it confirms.

- [ ] **Step 5: Deploy on the box in a stint**

  - Merge `dflash2-serve` into `mixed-tier-perf` with a fast-forward or merge commit, no force, and push.
  - On the box: `git -C ~/source/strata pull`. Rebuild `build-cuda` (`strata`) and `build-hip` (`libstrata_xtier_hip.so`, `strata`).
  - Write the two DFlash2 configs as copies of the IQ4_XS and Q6_K configs plus `"drafting": "dflash2"` and `"drafting_model": "/opt/models/incoai_Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf"`.
  - Run `strata-switch.sh iq4_xs-df2`. Expected: ready, a chat answers, and `/metrics` drafting shows DFlash2.
  - Run `strata-switch.sh iq4_xs`. Expected: ready on MTP, and one fixed prompt's text identical to `df2-mtp-base`.
  - Leave the service on `iq4_xs` (the user's current choice). Remove the box worktree `~/source/strata-df2`.

- [ ] **Step 6: Report**

  Write a short report to the user with these measured numbers, each with what it ran on (IQ4_XS, 3090 at -lgc 0,1590, W6800):
  - **Both modes:** decode tok/s, accept rate, 20K prompt tok/s, and the prompt cost of the taps.
  - **Checkpoints:** RAM per checkpoint.
  - **Restores:** the restore counters.

  Say what was not tested. Add a memory note on the new switch targets.
