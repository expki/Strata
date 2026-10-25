# Mixed-Make Expert Tier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CUDA engine serving a layer split across NVIDIA cards uses an AMD card as an adaptive expert tier (`--peer-device hip:N`) for decode and prompt rows, faster than the NVIDIA run alone.

**Architecture:** The tier is the existing peer tier (`PeerExperts`) in a new "foreign" mode. Its host logic (which experts it holds, adapt choices, the per-layer row plan) stays in the CUDA engine, compiled once. Its device work runs in `libstrata_xtier_hip.so`, a HIP library built in the HIP tree from the existing kernels, with private symbols and a versioned C interface, loaded like `libstrata_kq.so`. Decode syncs through the host, as the peer's pool-thread `launch`/`finish` does today. Prompt rows sync through flags in host memory that both makes map: the existing `wait_handoff` kernel and a new `raise_flag` kernel, so the engine's prompt thread never blocks.

**Tech Stack:** C++17, CUDA 13.2 (sm_86), HIP / ROCm 7.2.4 (gfx1030), CMake + Ninja, Python 3 for the bench tool.

**Spec:** `docs/superpowers/specs/2026-10-24-mixed-make-expert-tier-design.md`

## Global Constraints

- Opt-in. Only `--peer-device hip:N` reaches new engine code. Without it the greedy output of a fixed prompt set is identical to the base commit `7dd5cf18` built the same way, on both the CUDA and the HIP engine.
- The HIP library is built only with `-DSTRATA_XTIER=ON` in a HIP tree. A default HIP or CUDA configure builds exactly what it built before.
- Build both trees for every task that touches shared sources (CONTRIBUTING.md). Linux only for v1. Nothing may break the Windows build (copy the `strata_kq` loader's `#if defined(_WIN32)` shape).
- One AMD card. Native packs only (`lay.native`). NVFP4 and Hadamard-folded packs keep today's peer refusals and messages.
- No P2P between makes. Data crosses only through pinned host memory.
- The `held` rule: an expert is on at most one card at any time.
- Code comments: one line each, plain words, no em dashes, no semicolons, no hyphen joining two clauses (the user's global rule). Otherwise match the surrounding code.
- Docs: plain words, short sentences, every number with what it was measured on (AGENTS.md).
- Benchmarks: about 5 minutes per configuration. Say the expected time before starting a run.
- The test box is `ssh vdh@192.168.10.5`. Before stopping `strata.service`, back up `/etc/systemd/system/strata.service`. Afterwards restore the NVIDIA unit and check `/health` reports `"loaded": true`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- The design doc and this plan stay out of the upstream pull request.

## Review Focus

1. **The AMD library is missing, from another version, or ROCm's libraries are not on the path.** The engine must stop at start with one line naming the file and the fix, not crash or hang. Tests: Task 1 (loader) and Task 6 (engine start).
2. **The engine exits or restarts with both runtimes loaded.** The server restarts the engine after a failure, so a crash at exit must not happen and the next start must work. The library handle is never `dlclose`d. Tests: Task 2 (probe exit code) and Task 6 (stop and restart).
3. **The AMD side stalls or fails mid-request.** Every cross-make wait is bounded by `STRATA_XTIER_WAIT_MS` (default 30000). The request then fails with an error naming that setting, never a hang. Tests: Task 3 (decode) and Task 8 (prompt drop hook).
4. **A request is stopped mid-prompt and the next one starts at once.** Tier work still in flight must not corrupt the next answer. Test: Task 10 runs `tools/stop_resend_test.py` against the mixed config.
5. **The AMD card has little free VRAM** (another process on it, such as a HIP vision reader or a leftover engine). Slots are sized from the card's real free memory minus `--peer-reserve-mib`, and no room gives the existing clean refusal. Test: Task 6.

---

## How work moves between machines

The dev machine (`/home/dev/source/strata`, branch `mixed-make-expert-tier`) has CUDA but no ROCm. Everything that needs both makes runs on the box. Do this once:

```sh
git -C /home/dev/source/strata remote add box ssh://vdh@192.168.10.5/home/vdh/source/strata
git -C /home/dev/source/strata push box mixed-make-expert-tier:xtier-incoming
ssh vdh@192.168.10.5 'git -C ~/source/strata worktree add --detach ~/source/strata-xtier xtier-incoming'
```

After each commit: `git push -f box mixed-make-expert-tier:xtier-incoming`, then on the box `git -C ~/source/strata-xtier switch --detach xtier-incoming`. Never edit inside the box worktree. `~/source/strata` stays on main for `strata.service`.

Configure the two box trees once (Task 1, Step 1). Use `-j8` because the box serves other things:

```sh
cd ~/source/strata-xtier
GG=/home/vdh/source/strata/third_party/llama.cpp
cmake -G Ninja -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_PORTABLE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DSTRATA_GGML_DIR=$GG
cmake -G Ninja -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PREFILL_MMQ=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_PORTABLE=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm \
  "-DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/amdgcn/bitcode" \
  -DSTRATA_GGML_DIR=$GG -DSTRATA_XTIER=ON
```

Every GPU test in the CUDA tree that uses the library runs with this environment, called `XENV` below:
`LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/rocm/lib STRATA_XTIER_LIB=$HOME/source/strata-xtier/build-hip/libstrata_xtier_hip.so`

---

### Task 1: Interface header, library skeleton, engine-side loader

**Files:**
- Create: `include/strata/core/xtier_abi.hpp`, `src/xtier/xtier_lib.cu`, `src/xtier/xtier_exports.map`, `include/strata/core/xtier.hpp`, `src/core/xtier.cpp`, `tests/core/xtier_loader_test.cpp`, `tests/core/xtier_stub.cpp`, `tests/core/xtier_device_test.cpp`
- Modify: `CMakeLists.txt` (the `STRATA_XTIER` option and library target in the HIP branch; `src/core/xtier.cpp` in `strata_engine`'s source list; the three test targets under `STRATA_BUILD_TESTS` in the CUDA branch)

**Interfaces:**
- Produces, in `xtier_abi.hpp` (C linkage, `<stdint.h>` types only):
  - `#define STRATA_XTIER_ABI_VERSION 1`, `#define STRATA_XTIER_LIB_NAME "libstrata_xtier_hip.so"`, `#define STRATA_XTIER_CAP 80` (decode entries per layer: `MAXT` 8 x top-10, `static_assert`ed against `strata::kernels::cpu::MAXT * 10` in `xtier.cpp`).
  - `int strata_xtier_abi(void)` returns `STRATA_XTIER_ABI_VERSION * 100000 + strata_xtier_structs_size()`. `strata_xtier_structs_size()` is an inline header function summing `sizeof` of every struct in the header, modulo 100000. Later tasks add structs, so the check follows them.
  - `void* strata_xtier_open(int hip_device, char* err, size_t err_len)` (null on failure, `err` filled).
  - `void strata_xtier_close(void* t)`.
  - `int strata_xtier_mem_info(void* t, uint64_t* free_b, uint64_t* total_b)`.
  - `int strata_xtier_register(void* t, void* host, uint64_t bytes, void** hip_dev)` (`hipHostRegister` mapped + portable, then `hipHostGetDevicePointer`), and `int strata_xtier_unregister(void* t, void* host)`.
  - `const char* strata_xtier_last_error(void* t)`.
  - Every `int` return: 0 = ok, nonzero = error, text in `strata_xtier_last_error`. Every library entry point calls `hipSetDevice(device)` first, because the engine's threads call it.
- Produces, in `xtier.hpp` (namespace `strata::core::xtier`):
  - `struct PeerSpec { int cuda = -1; int hip = -1; };`
  - `bool parse_peer_device(const std::string& v, PeerSpec& out, std::string& err);` accepts `"N"` (N >= 1, the CUDA peer as today) and `"hip:N"` (N >= 0). Refuses `"hip:N"` in a HIP build (`STRATA_USE_HIP`) with `this is the AMD engine: --peer-device takes a card number`.
  - `std::string library_path();` is `STRATA_XTIER_LIB` if set, else `STRATA_XTIER_LIB_NAME` beside the engine (the `own_folder()` of `mmq_kq_loader.cpp`).
  - `struct Api` holds one function pointer per `strata_xtier_*` export.
  - `const Api* load_from(const std::string& path, std::string& err);` is uncached and never `dlclose`s.
  - `const Api* load(std::string& err);` is `load_from(library_path())`, cached for the process.
  - `class Device { public: bool open(int hip_device, std::string& err); void close(); bool valid() const; bool mem_info(uint64_t& free_b, uint64_t& total_b, std::string& err); bool register_host(void* host, uint64_t bytes, void** hip_dev, std::string& err); void unregister_host(void* host); ~Device(); };`
- Load errors, exact text:
  - Missing file: `"AMD tier library " + path + " not found (build it with -DSTRATA_XTIER=ON in the HIP tree, or set STRATA_XTIER_LIB)"`.
  - Wrong version: `"AMD tier library " + path + " is from another engine version"`.
  - `dlopen` failure on an existing file: `dlerror()` text plus `" (is /opt/rocm/lib on LD_LIBRARY_PATH?)"`.

- [ ] **Step 1: Set up the box worktree and both build trees** (commands above). Run `cmake --build build-cuda --target strata -j8` and `cmake --build build-hip --target strata -j8`. Expected: both succeed on the base commit. This proves the recipe before any change.

- [ ] **Step 2: Write the failing loader tests** in `tests/core/xtier_loader_test.cpp` (CPU only):

```cpp
// parse
PeerSpec s; std::string e;
CHECK(parse_peer_device("1", s, e) && s.cuda == 1 && s.hip == -1);
CHECK(parse_peer_device("hip:0", s, e) && s.hip == 0 && s.cuda == -1);
CHECK(!parse_peer_device("hip:-1", s, e));
CHECK(!parse_peer_device("hip:x", s, e));
CHECK(!parse_peer_device("0", s, e));   // the CUDA peer is N >= 1, as today
// load
CHECK(load_from("/nonexistent/libstrata_xtier_hip.so", e) == nullptr && e.find("not found") != std::string::npos
      && e.find("/nonexistent/libstrata_xtier_hip.so") != std::string::npos);
CHECK(load_from(STUB_PATH, e) == nullptr && e.find("another engine version") != std::string::npos);
```

`tests/core/xtier_stub.cpp` is a SHARED library that exports only `strata_xtier_abi()` returning `999 * 100000`. `STUB_PATH` comes in as a compile definition with the stub's build path.

- [ ] **Step 3: Run it to see it fail.** Run on the box: `cmake --build build-cuda --target xtier_loader_test && ctest --test-dir build-cuda -R xtier_loader --output-on-failure`. Expected: build error (no `xtier.hpp`).

- [ ] **Step 4: Implement** `xtier_abi.hpp`, `xtier.hpp` and `xtier.cpp`: the parser, `library_path`, `load_from` and `load`, and `Device`. Copy the dlopen/LoadLibrary shape of `src/prefill/mmq_kq_loader.cpp`.

- [ ] **Step 5: Implement the library skeleton** in `src/xtier/xtier_lib.cu`:
  - The entry points above. A handle struct holds `device` and the last error string.
  - `xtier_exports.map` exports `strata_xtier_*` only.
  - CMake, in the HIP branch under `if(STRATA_XTIER)`:
    - `add_library(strata_xtier_hip SHARED src/xtier/xtier_lib.cu)`.
    - Link `strata_hip_runtime` and `hip::host`, plus later tasks' libraries.
    - `target_link_options(... -Wl,--version-script=... -Wl,--exclude-libs,ALL -Wl,-Bsymbolic -Wl,--no-undefined)`.
    - `CXX_VISIBILITY_PRESET hidden`, `HIP_VISIBILITY_PRESET hidden`.
    - `set_target_properties(strata_kernels strata_mmq strata_prefill PROPERTIES POSITION_INDEPENDENT_CODE ON)` inside the same `if`.

- [ ] **Step 6: Run the loader test.** Expected: PASS.

- [ ] **Step 7: Write `tests/core/xtier_device_test.cpp`** (CUDA tree, GPU, ctest `SKIP_RETURN_CODE 77`):
  - Exit 77 when `load()` fails or `hipGetDeviceCount` would report 0. The library's `strata_xtier_open` fails with `no AMD GPU` in that case.
  - Check `open(0)` succeeds.
  - Check `mem_info` gives `0 < free <= total` and `total > 30ull << 30` on the box (W6800, 32 GB).
  - Check that a second `Device` on the same card opens and closes without affecting the first.

- [ ] **Step 8: Run on the box.** Build `strata_xtier_hip` (HIP tree) and `xtier_device_test` (CUDA tree). Run: `env $XENV ctest --test-dir build-cuda -R xtier --output-on-failure`. Expected: PASS. Also run with `STRATA_XTIER_LIB` unset and the library not beside the test. Expected: exit 77 (skipped).

- [ ] **Step 9: Commit**

```bash
git add include/strata/core/xtier_abi.hpp include/strata/core/xtier.hpp src/core/xtier.cpp src/xtier tests/core/xtier_*.cpp CMakeLists.txt
git commit -m "xtier: the AMD tier library's interface, skeleton and loader (opt-in, -DSTRATA_XTIER=ON)"
```

---

### Task 2: Phase 0 probe: two runtimes, shared memory, flags

**Files:**
- Modify: `include/strata/kernels/verify_kernels.hpp`, `src/kernels/cuda/verify_kernels.cu` (add `raise_flag`), `include/strata/core/xtier_abi.hpp`, `src/xtier/xtier_lib.cu`, `include/strata/core/xtier.hpp`, `src/core/xtier.cpp`, `CMakeLists.txt`
- Create: `tools/xtier_probe.cpp` (target `xtier_probe`, CUDA tree, `EXCLUDE_FROM_ALL`)
- Test: `tests/core/xtier_device_test.cpp`

**Interfaces:**
- Produces:
  - `void raise_flag(uint32_t* flag, uint32_t value, void* stream);` is a one-thread kernel: `__threadfence_system()`, then a volatile store of `value` to the mapped `flag`. It is ordered after earlier work on `stream`, and it is the same source on both makes.
  - ABI: `int strata_xtier_flag_echo(void* t, const uint32_t* in_hip, uint32_t* out_hip, uint32_t* err_hip, uint32_t seq, uint64_t timeout_ns)` enqueues `wait_handoff(in_hip, seq, err_hip, timeout_ns)` then `raise_flag(out_hip, seq)` on the library's decode stream and returns without waiting. `int strata_xtier_sync(void* t)` drains the library's streams.
  - ABI, for the probe's variant C: `void* strata_xtier_host_alloc(void* t, uint64_t bytes)` (`hipHostMalloc`, mapped and portable, null on failure) and `void strata_xtier_host_free(void* t, void* host)`.
  - `Device::flag_echo(...)` and `Device::sync(std::string&)` wrap them.

- [ ] **Step 1: Write the failing test** in `xtier_device_test.cpp`:
  - Allocate one page with `mmap`, `cudaHostRegister` it (portable | mapped), and `register_host` it with the library.
  - Echo 1000 sequence numbers: CUDA `raise_flag(in, i)`, the library's `flag_echo(in, out, err, i)`, CUDA `wait_handoff(out, i, cerr, 2e9)`. Assert `*out == 999` and both error words are 0.
  - Echo once with a flag that is never raised, `timeout_ns = 50e6`. Assert the library's error word becomes 1 within 1 s (the Review Focus 3 bound).

- [ ] **Step 2: Run it to see it fail.** On the box: `env $XENV ctest --test-dir build-cuda -R xtier_device`. Expected: FAIL (no `flag_echo`).

- [ ] **Step 3: Implement** `raise_flag`, `strata_xtier_flag_echo` and `strata_xtier_sync`. The library links the HIP tree's `strata_kernels` for `wait_handoff` and `raise_flag`.

- [ ] **Step 4: Run the test.** Expected: PASS on the box. Also rebuild `strata` in both trees (the shared `verify_kernels.cu` changed). Expected: both build.

- [ ] **Step 5: Write `tools/xtier_probe.cpp`.** It prints one table per section and exits 0. Its exit code is checked (Review Focus 2).
  - **Registration matrix.** Three variants:
    - A: `cudaHostAlloc` (portable | mapped), then the library's `register_host`.
    - B: `mmap`, then `cudaHostRegister` and `register_host`.
    - C: `strata_xtier_host_alloc`, then `cudaHostRegister`.
    - For each: register ok; a CUDA kernel writes a 64 MiB pattern that a HIP kernel reads back exactly; the same the other way. Also time a 64 MiB D2H copy into it from each make.
  - **Flag round trip:** the median and p99 of 10,000 echoes, GPU waits on both sides (as in the test).
  - **Host round trip:** the median of 10,000 cycles of host write flag → library sync → host read. This is the decode path's style.
  - Then the process exits normally with both runtimes loaded.

- [ ] **Step 6: Run on the box:** `env $XENV build-cuda/xtier_probe | tee ~/bench-mixed/phase0-probe.txt`. Expected: three matrix rows and two latency rows, exit 0. Record which variants work. Task 8's `SharedHost` uses the first working one in the order B, A, C.

- [ ] **Step 7: Commit**

```bash
git add include/strata/kernels/verify_kernels.hpp src/kernels/cuda/verify_kernels.cu include/strata/core/xtier*.hpp src/core/xtier.cpp src/xtier tools/xtier_probe.cpp tests/core/xtier_device_test.cpp CMakeLists.txt
git commit -m "xtier: raise_flag, the cross-make flag echo and the phase 0 probe"
```

---

### Task 3: Decode rows in the library, with a parity test

**Files:**
- Create: `src/kernels/test_blobs.hpp`, `src/kernels/xtier_decode_parity.cpp` (CUDA tree test target `xtier_decode_parity`, `SKIP_RETURN_CODE 77`)
- Modify: `src/kernels/native_grouped_parity.cpp` (use `test_blobs.hpp`, no behaviour change), `include/strata/core/xtier_abi.hpp`, `src/xtier/xtier_lib.cu`, `include/strata/core/xtier.hpp`, `src/core/xtier.cpp`, `tools/xtier_probe.cpp`, `CMakeLists.txt`

**Interfaces:**
- Produces, in the ABI:

```c
typedef struct strata_xtier_fmt { int32_t gu_type, d_type; int64_t n_embd, n_ff; uint64_t up_off, down_off, blob_bytes; } strata_xtier_fmt;
typedef struct strata_xtier_meta {   /* PeerExperts' Meta, moved here unchanged */
    unsigned long long ptr[STRATA_XTIER_CAP]; int32_t start[STRATA_XTIER_CAP + 1]; int32_t dst[STRATA_XTIER_CAP];
    int32_t tok[STRATA_XTIER_CAP]; int32_t count[4]; int32_t out_row[STRATA_XTIER_CAP];
} strata_xtier_meta;
int strata_xtier_arena(void* t, const uint64_t* slot_bytes, int64_t n, uint64_t* slot_addr);   /* one allocation, 256-byte aligned slots */
int strata_xtier_fill(void* t, int64_t slot, const void* host, uint64_t bytes);                /* async on the refill stream */
int strata_xtier_read(void* t, int64_t slot, void* host, uint64_t bytes);                      /* sync, for the fill check */
int strata_xtier_refill_record(void* t);
int strata_xtier_refill_done(void* t, int wait);                                               /* 1 done, 0 not yet */
int strata_xtier_decode(void* t, const strata_xtier_fmt* f, const float* x, int64_t n_tok, const strata_xtier_meta* m);
int strata_xtier_decode_wait(void* t, const float** rows, uint64_t timeout_ns);                /* spins; rows = m->count[1] x n_embd */
```

- Produces, in `Device`: `arena(const std::vector<uint64_t>&, std::vector<uint64_t>&, std::string&)`, `fill`, `read`, `refill_record`, `int refill_done(bool wait)`, `decode(const strata_xtier_fmt&, const float* x, int64_t n_tok, const strata_xtier_meta&, std::string&)`, `decode_wait(const float** rows, std::string&)`. `decode_wait` passes `STRATA_XTIER_WAIT_MS` (default 30000) as the bound. On timeout the error is `"the AMD tier did not answer within STRATA_XTIER_WAIT_MS (" + ms + " ms)"`.
- `strata_xtier_decode` does what `PeerExperts::launch` does on its device for a native pack:
  - Copy `x` into the library's pinned staging and send it up with the meta.
  - `quantize_q8_1_rows`, then `native_expert_grouped(native_expert_layout(f...), ...)`.
  - Copy the rows to the library's pinned output.
  - Scratch is `native_expert_scratch_bytes(STRATA_XTIER_CAP, n_ff)`, allocated at `open`.
- `src/kernels/test_blobs.hpp`: `std::vector<uint8_t> random_rows(int ggml_type, int64_t rows, int64_t n, std::mt19937& rng);` is moved unchanged out of `native_grouped_parity.cpp`.

- [ ] **Step 1: Write the failing parity test** `xtier_decode_parity.cpp`:
  - Format: gate/up IQ4_XS (23), down IQ4_NL (20), `n_embd` 2560, `n_ff` 640, the Swift IQ4_XS layout.
  - Blobs: 16 experts from `random_rows`.
  - Cases: for `n_tok` in {1, 4, 8}, top-10 routing to random experts.
  - Reference: CUDA `native_expert_grouped` in process.
  - Candidate: the library's `decode`, then `decode_wait`.
  - Assertions: for every case, `||lib - cuda||_2 / ||cuda||_2 <= 1e-3`, and every value is finite. Print the observed maximum ratio. Exit 77 when the library or an AMD card is missing.

- [ ] **Step 2: Run it to see it fail.** Run `env $XENV ctest --test-dir build-cuda -R xtier_decode_parity --output-on-failure` on the box. Expected: FAIL (no `decode`).

- [ ] **Step 3: Implement** the decode ABI in the library (link the HIP tree's `strata_kernels`). Also implement the `Device` wrappers and `test_blobs.hpp`. Point `native_grouped_parity.cpp` at the shared header.

- [ ] **Step 4: Run the tests on the box:**
  - `xtier_decode_parity`. Expected: PASS, maximum ratio printed.
  - `native_grouped_parity` in the CUDA tree. Expected: PASS, unchanged.

- [ ] **Step 5: Add the decode timing to `xtier_probe`.** One IQ4_XS layer, 10/40/80 entries. Time host-to-host (`decode` then `decode_wait`), median of 1,000. Run on the box and append to `~/bench-mixed/phase0-probe.txt`.

- [ ] **Step 6: Commit**

```bash
git add src/kernels/test_blobs.hpp src/kernels/xtier_decode_parity.cpp
git commit -am "xtier: decode rows on the AMD card, checked against the CUDA kernels"
```

---

### Task 4: The A/B tool takes per-arm configs, warm runs, cold starts and full texts

**Files:**
- Modify: `tools/ab_engine.py`
- Create: `tools/test_ab_engine.py` (plain `python tools/test_ab_engine.py`, like `tools/test_setup_*.py`)

**Interfaces:**
- Produces: `parse_arm(s: str) -> tuple[str, dict]` (factored out of `main`), with these arm keys:
  - `config=PATH` replaces the base config for that arm (the positional `config` stays the default).
  - `exe`, `xargs` and environment keys work as today.
- New flags:
  - `--warm N`: unmeasured passes of the request suite after ready. Default 0, today's behaviour.
  - `--measure N`: measured passes per visit. Default 1, today's behaviour.
  - `--drop-caches`: run `sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'` before each arm start. The first measured request of each visit is tagged `cold: true`.
  - `--long-words W`: repeatable. Default `[6000]`, today's single long prompt. Requests are tagged `long<W>`.
  - `--full-text`: log whole answers instead of 80 characters.
  - `--identity`: after the run, compare each request's text across arms and print `identical` or the first differing request and arm.

- [ ] **Step 1: Write the failing tests:**

```python
assert parse_arm("B:config=/c.json,exe=/e,STRATA_A=1") == ("B", {"config": "/c.json", "exe": "/e", "STRATA_A": "1"})
assert parse_arm("A:") == ("A", {})
assert first_difference({"A": {"chat0": "x"}, "B": {"chat0": "x"}}) is None
assert first_difference({"A": {"chat0": "x"}, "B": {"chat0": "y"}}) == ("chat0", "B")
```

- [ ] **Step 2: Run** `python tools/test_ab_engine.py`. Expected: FAIL (`parse_arm` not defined).
- [ ] **Step 3: Implement.** `first_difference(texts: dict[str, dict[str, str]]) -> tuple[str, str] | None` compares every arm with the first.
- [ ] **Step 4: Run.** Expected: PASS. Also run `python tools/ab_engine.py --help`. Expected: the new flags are listed.
- [ ] **Step 5: Commit**

```bash
git add tools/ab_engine.py tools/test_ab_engine.py
git commit -m "ab_engine: per-arm configs, warm passes, cold starts, several long prompts, the identity check"
```

---

### Task 5: Phase 0 measurements, then STOP for the user

No code. The deliverable is `~/bench-mixed/phase0.md` on the box and a summary to the user. It takes about 30 minutes in all; tell the user before starting.

- [ ] **Step 1: Write the configs on the box**, copies of the service's:
  - `cfg-nv3.json`: `strata-swift-iq4_xs.json` with port 8095.
  - `cfg-nv2.json`: the same with `"gpu": [0, 1]` (3090 + 3080 Ti, the 3050 Off).
  - `cfg-amd.json`: `strata-swift-iq4_xs-amd.json` with port 8095.
  - `cfg-amd-capped.json`: the AMD config with `--expert-cache 9844` instead of `auto`. That's the trio's 1295 + 2547 + 6002 slots.

  All use the installed 0.1.49.1 engines.
- [ ] **Step 2: Measure the CPU pool's time per decode window** on the trio. Run one `ab_engine` visit of `cfg-nv3.json` with `STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1` as arm keys, and read the per-request stage lines from the server log (docs/DETAILS.md, "Where a decode window's time goes").
- [ ] **Step 3: Stop `strata.service`** (back up the unit first) and run, from the box worktree:

```sh
python tools/ab_engine.py ~/bench-mixed/cfg-nv3.json NV3:config=$HOME/bench-mixed/cfg-nv3.json \
  NV2:config=$HOME/bench-mixed/cfg-nv2.json AMD:config=$HOME/bench-mixed/cfg-amd.json \
  AMDCAP:config=$HOME/bench-mixed/cfg-amd-capped.json --port 8095 --rounds 1 --warm 1 --measure 2 \
  --drop-caches --max-new 128 --long-words 3700 --long-words 15000 --log ~/bench-mixed/phase0-ab.jsonl
```

  Expected: about 4 minutes per arm.
- [ ] **Step 4: Restore and check the service.** Restore the unit, restart, and check `/health`.
- [ ] **Step 5: Write `~/bench-mixed/phase0.md`:**
  - The probe tables.
  - The pool's per-window time against the probe's decode round trip, at the entry counts the trio's windows actually send to the CPU.
  - NV3 against NV2.
  - AMD against AMDCAP (is the W6800's prompt edge memory or the card?).
  - Which `SharedHost` variant Task 8 uses.
- [ ] **Step 6: STOP.** Send the user the summary and these decisions to confirm:
  - **Go or no-go:** if the decode round trip is longer than the pool's time for the same rows, the tier cannot win decode. Revisit the plan with the user.
  - **The NVIDIA set** for the mixed runs: the trio, or the pair if NV2 is not slower.

  Do not start Task 6 without the user's answer.

---

### Task 6: The foreign peer for decode, beside the layer split

**Files:**
- Modify: `include/strata/core/peer_experts.hpp`, `src/core/peer_experts.cpp`, `src/program/generate.cpp`
- Test: on the box (engine runs). No new unit test: the parts that can fail without a GPU are in Tasks 1 and 7.

**Interfaces:**
- Consumes: `xtier::Device` (Tasks 1 and 3) and `xtier::parse_peer_device`.
- Produces, in `PeerExperts`:
  - `bool open_foreign(int hip_device, const std::vector<std::pair<int32_t, int32_t>>& ranked, const std::function<bool(int32_t, int32_t)>& held, ExpertSource& src, int64_t n_layers, int64_t n_expert, int reserve_mib, int64_t max_slots, std::string& err);`
  - `bool foreign() const;`
  - `valid()` is true for either kind.
  - `device()` is -1 for a foreign peer.
  - `p2p()` is false for a foreign peer.
  - `slot_ptr` returns the AMD slot address, meaningful only to the library.
  - `gib()` sums the foreign slot bytes.
- In foreign mode, `launch`, `finish`, `adapt`, `apply_pending` and `close` keep their signatures and host logic. Every CUDA call on the peer's device becomes the matching `Device` call:
  - Fills and swaps read the blob through `src.blob()` and call `Device::fill` (so the file tier, page cache and prefetch serve the tier).
  - `launch` never takes the direct (scatter into `out`) path.
  - `finish` copies rows from `decode_wait`'s pointer by `row_of_`.
  - A pack that is not native refuses in `open_foreign` with the existing text `peer experts: only native packs are supported`.
- `generate.cpp`:
  - `Options` gains `int peer_hip = -1;`. `--peer-device` goes through `parse_peer_device`.
  - After parsing: `const bool peer_any = o.peer_device >= 1 || o.peer_hip >= 0;`.
  - Every `o.peer_device >= 1` and `o.peer_device < 0` check becomes `peer_any` / `!peer_any`. The exceptions keep `o.peer_device` (CUDA peer only):
    - the parser;
    - `set_peer_portable`;
    - the CUDA peer's `open` block;
    - the `STRATA_PEER_HOT` reordering;
    - the layer-split refusal, which becomes `o.peer_device >= 1 && !o.layer_split.empty()`.
  - Both kinds still refuse `--expert-cache-device1..3`.
  - Grep `o.peer_device` to find every site (about 12).
- The foreign open, beside the CUDA one:
  - `held = [&](int32_t l, int32_t e) { const int st = multi_gpu ? stage_of(l) : 0; return (st > 0 ? stages[st - 1]->cache : xcache).slot_of(l, e) >= 0; }`.
  - Log line: `strata generate: AMD tier on HIP device N: E experts, G GiB (filled in S s); with the NVIDIA cards' C that is T of A on the GPUs`.
- The prompt path does not use the foreign peer yet (Task 8). Skip the `sp.set_peer` call with the log line `strata serve: the AMD tier takes no prompt rows yet`.
- Verify while implementing:
  - `drive.d.layers` is the absolute layer index when `drive_pool_multi` calls the pool on a later stage, since `kind == 2` uses it.
  - The tier's `adapt` gets `host_res`. It already holds every stage's residency (built with `stage_of` in `generate.cpp`), so no change is needed there. The spec's line about `res0` was a mistake.

- [ ] **Step 1: Write the box mixed config** `~/bench-mixed/cfg-mix.json`:
  - Based on the NVIDIA set chosen in Task 5.
  - `exe` = `~/source/strata-xtier/build-cuda/strata`, `"args"` + `"--peer-device", "hip:0"`.
  - `"lib_dirs"`: `["/usr/local/cuda/lib64", "/opt/rocm/lib"]`, `"env"`: `{"STRATA_XTIER_LIB": "<build-hip path>"}`, port 8095.
  - Also write `cfg-nvb.json`: the same NVIDIA config with the branch's `exe` and no peer argument.
- [ ] **Step 2: Run it to see it fail.** Start the server with `cfg-mix.json` (service stopped). Expected: the engine starts with no AMD tier line. `atoi("hip:0")` reads 0, so today's code leaves the peer off.
- [ ] **Step 3: Implement** the `PeerExperts` foreign mode and the `generate.cpp` changes above. Build `strata` in both trees.
- [ ] **Step 4: Run on the box:**
  - Start `cfg-mix.json`. Expected: the AMD tier log line, the split's three stage lines, and `/health` loaded.
  - Send one chat request. Expected: an answer, and the dispatch's `d.peer_entries` counter above 0. If no log line prints it yet, add it to the request's existing decode line for a foreign peer.
  - Restart once. Expected: the second start answers too (Review Focus 2).
  - Stop with SIGTERM. Expected: no "Segmentation fault" or core in the log.
  - Start with `STRATA_XTIER_LIB=/nonexistent`. Expected: the engine exits 1, and the log has one line with the path and "not found" (Review Focus 1).
  - Start with `--peer-reserve-mib 40000`. Expected: `peer experts: no room or no expert left for the peer` and exit 1 (Review Focus 5).
- [ ] **Step 5: Run the decode A/B**, one visit per arm (about 8 minutes):
  - `ab_engine` with `NVB:config=cfg-nvb.json MIX:config=cfg-mix.json --warm 1 --measure 2 --full-text --max-new 128 --long-words 3700`.
  - Record decode tok/s for both arms and whether the texts match (`--identity`). Differences are expected and are reported, not fixed.
- [ ] **Step 6: Restore the service. Commit.**

```bash
git commit -am "peer tier: an AMD card as the expert tier beside a CUDA layer split (--peer-device hip:N), decode rows"
```

---

### Task 7: Ownership: pure selection, the invariant, and the move policy

**Files:**
- Create: `include/strata/core/peer_adapt.hpp`, `src/core/peer_adapt.cpp` (no CUDA), `tests/core/peer_adapt_test.cpp`
- Modify: `src/core/peer_experts.cpp` (the `adapt` selection calls `peer_adapt_choose`, with the same result for the CUDA peer), `include/strata/core/peer_experts.hpp`, `src/program/generate.cpp`, `CMakeLists.txt`

**Interfaces:**
- Produces:

```cpp
namespace strata::core {
struct PeerSwap { float gain; int32_t layer, in, out; };   // out = -1: a free slot
/// The tier's swaps for one adapt round: today's rule (u >= 2, gain >= 1.5, the hottest first, at most max_swaps),
/// plus free slots taken first. `other` = every NVIDIA stage's residency (host_res), `mine` = the tier's.
std::vector<PeerSwap> peer_adapt_choose(const float* usage, const int32_t* other, const int32_t* mine,
                                        const std::vector<std::vector<int32_t>>& free_slots_by_layer,
                                        int64_t n_layers, int64_t n_expert, int max_swaps);
}
```

  - `PeerExperts::release(int64_t layer, int64_t expert)` marks the slot free for this layer. Used by the move policy.
  - `STRATA_XTIER_OWNERSHIP=disjoint|move`, default `disjoint`:
    - `disjoint` is today's rule. A stage's adapt skips experts the tier holds (the existing `peer.has` filter in `generate.cpp`'s `adapt`).
    - `move` lifts that filter for the foreign peer only. After the stage's swaps are chosen, it calls `peer.release(l, e)` for every expert a stage takes.
  - `STRATA_XTIER_CHECK=1`: after every adapt round, count pairs with `host_res >= 0 && peer.has`. Above 0, print `strata serve: N experts on two cards` and fail the request.

- [ ] **Step 1: Write the failing tests** `peer_adapt_test.cpp`:

```cpp
// a candidate a stage holds is never chosen
// usage {5 on (0,1)}, other[(0,1)] = slot 3, mine empty but one victim at u 0 -> no swap of expert 1
// a free slot is taken before a victim
// free_slots_by_layer[0] = {7}, candidate (0,2) u 4 -> PeerSwap{4, 0, 2, -1}
// the gain threshold
// candidate u 3.0 vs victim u 2.0 -> no swap (gain < 1.5); u 3.6 vs 2.0 -> one swap
// the cap
// 10 eligible swaps, max_swaps 4 -> 4, the largest gains
// no result swaps in an expert the tier already holds, or one a stage holds
```

- [ ] **Step 2: Run** `ctest -R peer_adapt`. Expected: FAIL (no header).
- [ ] **Step 3: Implement** `peer_adapt_choose`. Call it from `PeerExperts::adapt` for both kinds. The CUDA peer passes empty `free_slots_by_layer`, so its choices are unchanged. Add `release` and the two environment switches.
- [ ] **Step 4: Run** `ctest -R peer_adapt`. Expected: PASS. Build both trees.
- [ ] **Step 5: Run on the box.** With `STRATA_XTIER_CHECK=1`, run one `cfg-mix.json` visit under each policy. Expected: no "experts on two cards" line under either. Then run the short A/B `MIXD:...STRATA_XTIER_OWNERSHIP=disjoint` against `MIXM:...STRATA_XTIER_OWNERSHIP=move`, one visit each, `--warm 1 --measure 2`, about 8 minutes.
- [ ] **Step 6: Set the default** to the faster policy (keep `disjoint` if within noise). Put the numbers in the commit message.

```bash
git add include/strata/core/peer_adapt.hpp src/core/peer_adapt.cpp tests/core/peer_adapt_test.cpp
git commit -am "peer tier: one owner per expert across the NVIDIA stages and the AMD tier, the move policy measured"
```

---

### Task 7b: The AMD round trip on the decode critical path

Added after Task 6 measured it (ledger ruling R24). With the tier on, the pool waits about 19 ms of a 37 ms decode window for the AMD rows, about 0.4 ms per layer-window at about 14 entries. The Task 3 probe gives about 0.155 ms host to host for 14 entries. The goal is to close that gap or hide it behind the pool's own work.

**Files:**
- Modify: `src/core/peer_experts.cpp`, `src/xtier/xtier_lib.cu`, `include/strata/core/xtier_abi.hpp`, `include/strata/core/xtier.hpp`, `src/core/xtier.cpp`, `src/core/expert_source.cpp` (only the order of the peer's launch against the CPU rows), `tools/xtier_probe.cu`

- [ ] **Step 1: Measure where the time goes.** `STRATA_XTIER_TIMING=1` prints, per request, the mean per layer-window of: host time in `launch` (staging and enqueue), the AMD card's time (HIP events around the kernels), the time `finish` waits, and the copy out. Add the CPU rows' time and the NVIDIA stages' time from `STRATA_DECODE_TIMING=1`. One `cfg-mix.json` visit. Write the breakdown into the report.
- [ ] **Step 2: Fix what the breakdown points at**, one change per commit, each kept only when the A/B (ledger R21 settings) shows it faster beyond noise and `xtier_decode_parity` still passes. Candidates, in the order the breakdown makes likely:
  - The peer's `launch` goes before the CPU rows in the pool dispatch, so the AMD card works while the CPU does.
  - One host to device copy for `x` and the meta instead of two.
  - The rows land in mapped pinned memory written by the kernel, so there is no copy out.
  - `finish` spins on a mapped flag raised by `raise_flag` instead of a HIP event query (the probe's 13 us against 23 us).
  - A HIP graph for the per-layer kernels, if launch overhead dominates.
- [ ] **Step 3:** Record the before and after breakdown and the A/B numbers in `~/bench-mixed/phase7b.md` and in the commit messages.

---

### Task 8: Prompt rows on the AMD tier

**Files:**
- Modify: `include/strata/core/xtier_abi.hpp`, `src/xtier/xtier_lib.cu`, `include/strata/core/xtier.hpp`, `src/core/xtier.cpp`, `src/prefill/prefill.cpp`, `src/program/generate.cpp`, `CMakeLists.txt` (the library links the HIP tree's `strata_mmq` and `strata_prefill` too)
- Create: `src/prefill/xtier_prefill_parity.cpp` (CUDA tree test `xtier_prefill_parity`, `SKIP_RETURN_CODE 77`)

**Interfaces:**
- Produces, in the ABI:

```c
typedef struct strata_xtier_prefill_args {
    uint32_t seq;                        /* this call's flag value */
    const uint32_t* x_ready;             /* HIP addresses of shared host words: the engine raises x_ready when x and w landed */
    uint32_t* done;                      /* the library raises done when the sums landed */
    uint32_t* wait_err;                  /* 1 = the wait for x_ready timed out */
    uint64_t timeout_ns;
    int64_t T, N, K; int32_t f16;
    const void* x_hip; const float* w_hip;   /* shared host buffers, HIP addresses */
    void* sum_host;                          /* shared host buffer, host address (the D2H target) */
    const int32_t* pair; const int32_t* src; /* plain host arrays, rows_peer each, copied at the call */
    const int32_t* bounds; int64_t n_bounds;
    const int32_t* adds; int64_t n_adds;
    int64_t rows_peer;
    int32_t mmq_gu_type, mmq_d_type; uint64_t gu_bytes, d_bytes;
    strata_xtier_fmt fmt;
    int64_t n_experts; const uint64_t* expert_addr; const int32_t* expert_rows;   /* order_peer order */
    int64_t n_groups; const int64_t* group_first; const int64_t* group_last;
    const int64_t* group_adds_at; const int32_t* group_ntok;
} strata_xtier_prefill_args;
int strata_xtier_prefill_open(void* t, int64_t T_max, int64_t N, uint64_t gu_max, uint64_t d_max);
int strata_xtier_prefill_layer(void* t, const strata_xtier_prefill_args* a);   /* enqueues, never waits; thread safe */
```

- `strata_xtier_prefill_layer` runs, on one prefill stream behind a mutex, the device half of today's compact SUMS host route (`prefill.cpp`, the `peer_now` block):
  1. `wait_handoff(x_ready, seq, wait_err, timeout_ns)`.
  2. `f16_to_f32_wide` or `copy_f32_wide` of `x_hip`, then `copy_f32_wide` of `w_hip`.
  3. Upload `pair`, `src`, `bounds` and `adds` from staging.
  4. Zero `sum`.
  5. Per group: `gather_native` from `expert_addr`, `mmq::quantize`, the gate/up `mmq::Product`, `swiglu`, `quantize`, the down `Product`, `peer_gather_add`.
  6. `sums_to_f16` when `f16`, then D2H into `sum_host`, then `raise_flag(done, seq)`.

  Test hook: `STRATA_TEST_XTIER_DROP=N` skips the `raise_flag` of the Nth call (counting from 1).
- Produces, in `xtier.hpp`: `class SharedHost { public: bool alloc(Device& d, size_t bytes, std::string& err); void* host() const; void* cuda_dev() const; void* hip_dev() const; ~SharedHost(); };`. It uses the registration variant Task 5 recorded.
- `prefill.cpp`:
  - `PeerPrefill` gains `bool foreign`, `uint32_t seq` and the `SharedHost` blocks: flags in one page; `host_x16` or `host_x`; `host_w`; `host_sum16` or `host_sum`.
  - `set_peer` for a foreign peer:
    - Forces `p2p = false`, `compact = true`, `sums = true`, `gather = true`, `ps_frac = 0`.
    - `f16` follows `STRATA_PF_PEER_F16` as today.
    - Allocates no CUDA device buffers for the peer, only `ev_in`, `x_stream` and the shared blocks.
    - Calls `prefill_open`.
  - In the layer, a foreign branch:
    - Builds the host tables with the same code as today. Factor the host-only table building into `static void plan_peer_groups(PeerPrefill& P, ...)` shared by both branches, so the CUDA peer's tables are unchanged.
    - On `xs`: the D2H copies of x and w into the shared blocks, then `raise_flag(x_ready_cuda, ++seq)`.
    - `prefill_layer` (the HIP addresses).
  - Before the combine: `wait_handoff(done_cuda, seq, wait_err_cuda, timeout, m.cs)` replaces `cudaStreamWaitEvent(m.cs, ev_done)`. The combine reads `host_sum16`/`host_sum` through `cuda_dev()`.
  - After each chunk's sync: if either error word is set, fail the request with `"the AMD tier did not answer within STRATA_XTIER_WAIT_MS (" + ms + " ms)"`.
- `generate.cpp`: when `peer.foreign()`, call `set_peer` on `sp` and on every `st->sp`. `bind_stage_helper` then declines one-chunk stage help, because it refuses when `m.pp` is set. Task 9 measures what that costs.

- [ ] **Step 1: Write the failing parity test** `xtier_prefill_parity.cpp`:
  - One synthetic layer: IQ4_XS gate/up, IQ4_NL down, `N` 2560, `n_ff` 640, `K` 10.
  - T in {64, 512, 4096} tokens, routing over 64 experts on the "tier", random weights.
  - Reference: the same MMQ products in process on CUDA (`mmq::quantize`, `gather_native`, `Context::run`, `swiglu`), weighted on the host.
  - Candidate: `SharedHost` blocks, CUDA `raise_flag`, `prefill_layer`, CUDA `wait_handoff`, then read the sums.
  - Assertions: `||lib - ref||_2 / ||ref||_2 <= 1e-3` per T, all finite, and both error words 0.
  - Then with `STRATA_TEST_XTIER_DROP=1` and a 200 ms timeout: the CUDA error word becomes 1 within 1 s, and the test does not hang (Review Focus 3).

- [ ] **Step 2: Run it on the box to see it fail.** Expected: FAIL (no `prefill_layer`).
- [ ] **Step 3: Implement** the library side, then `SharedHost`, then the `prefill.cpp` and `generate.cpp` parts. Build both trees.
- [ ] **Step 4: Run** `env $XENV ctest --test-dir build-cuda -R "xtier|peer_adapt" --output-on-failure` on the box. Expected: all PASS.
- [ ] **Step 5: Run on the box with `cfg-mix.json`:**
  - The log shows no "takes no prompt rows" line.
  - A 4K and a 16K prompt answer.
  - The prompt path's peer counters are above 0.
  - Then run the A/B `NVB` against `MIX`, one visit each, `--warm 1 --measure 2 --long-words 3700 --long-words 15000`, about 8 minutes. Record the prompt tok/s.
- [ ] **Step 6: Commit**

```bash
git add src/prefill/xtier_prefill_parity.cpp
git commit -am "peer tier: the AMD tier computes its experts' prompt rows (host-mapped flags both ways)"
```

---

### Task 9: The switches a peer turns off, decided by measurement

**Files:**
- Modify: `src/program/generate.cpp`, `src/core/expert_source.cpp` (only where a switch is re-enabled for the foreign peer)

These switches are off whenever a peer exists today:

| Switch | Where |
|---|---|
| Route tail skip default | `generate.cpp` `tail_default_path`, and `expert_source.cpp` `tail_rank > 0 && d.peer == nullptr` |
| Automatic PCIe share | `generate.cpp` `pcie_auto` |
| `--kv-grow` | `generate.cpp`, `kv-grow` |
| `STRATA_PIN_BY_NEED` | `generate.cpp` |
| Split one-chunk stage help | `prefill.cpp` `bind_stage_helper` |

Also try a candidate the peer never had: hand the tier's experts back to the OS like the VRAM-held ones (`srcp->release`, as `STRATA_ARENA_RELEASE` does), so the page cache has room for the file tier.

- [ ] **Step 1:** For each switch, add an opt-in environment variable that turns it back on for the foreign peer only: `STRATA_XTIER_TAIL_SKIP=1`, `STRATA_XTIER_PCIE_AUTO=1`, `STRATA_XTIER_KV_GROW=1`, `STRATA_XTIER_PIN_BY_NEED=1`, `STRATA_XTIER_STAGE_HELP=1`, `STRATA_XTIER_RELEASE=1`. With none set, behaviour is Task 8's.
- [ ] **Step 2:** For each, run `STRATA_XTIER_CHECK=1` with one `cfg-mix.json` visit. Expected: answers, no "experts on two cards". A switch that breaks something stays off, with the reason recorded.
- [ ] **Step 3:** A/B each working switch against the Task 8 baseline, one visit each, about 8 minutes per switch.
- [ ] **Step 4:** Make each switch that is faster beyond noise on by default for the foreign peer, keeping its variable as the way to turn it off (`=0`). Record the numbers in `~/bench-mixed/phase9.md` and in the commit message.
- [ ] **Step 5: Commit**

```bash
git commit -am "peer tier: the AMD tier's defaults for the switches a peer turned off, by measurement"
```

---

### Task 10: Acceptance, the default path's identity, the stop test, docs

**Files:**
- Modify: `docs/MULTI_GPU.md` (a section "NVIDIA and AMD together (experimental)" after the peer tier, plus an exception to the "One make of card per engine" paragraph), `docs/AMD_HIP.md` (the "mixed AMD/NVIDIA execution" sentence points to it)

- [ ] **Step 1: Default path identity.**
  - Build `strata` at `7dd5cf18` in a second box worktree with the same CUDA and HIP recipes, without `-DSTRATA_XTIER`.
  - Run `ab_engine --identity --full-text`, one visit each, about 6 minutes per pair:
    - `BASE:config=cfg-nv-base.json` against `NVB:config=cfg-nvb.json`.
    - The same pair for the HIP engine (`cfg-amd-base.json`, `cfg-amdb.json`).
  - Expected: `identical` for both pairs. Any difference blocks the work until it is explained and fixed.
- [ ] **Step 2: Stop test** (Review Focus 4):
  - Against a running `cfg-mix.json` server: `python3 tools/stop_resend_test.py http://127.0.0.1:8095 40`. Expected: exit 0.
  - Then one greedy chat request. Expected: the same text as the first request of a fresh `cfg-mix.json` start.
- [ ] **Step 3: Acceptance run**, about 25 minutes:
  - Arms: the best NVIDIA set from Task 5 on the branch build (`NVB`), `AMD` (`cfg-amdb.json`) and `MIX`.
  - Flags: `--rounds 2` (ABC then CBA), `--warm 1 --measure 2 --drop-caches --max-new 128 --long-words 3700 --long-words 15000`.
  - Pass: `MIX`'s warm decode median and 16K prompt median both above `NVB`'s. Report `AMD`'s 16K prompt median beside them as the stretch target, and the cold first-request numbers separately.
  - Check in the `MIX` server log that the expert caching still runs: the prefill plan line, the file tier I/O lines, the adapt swaps on the NVIDIA stages and on the tier.
- [ ] **Step 4: Docs.**
  - **The section:** what it does, the exact config (`--peer-device hip:N`, `lib_dirs`, `STRATA_XTIER_LIB`), how to build the library, the measured numbers from Step 3 with the box's hardware, and its limits (one AMD card, native packs, Linux, the peer's refusals).
  - **No claims** beyond what Step 3 measured.
- [ ] **Step 5: Restore and check the service.** Restore `strata.service` and check `/health`.
- [ ] **Step 6: Commit**

```bash
git commit -am "docs: NVIDIA and AMD cards together, the AMD tier (experimental), measured on the test box"
```

- [ ] **Step 7: Report to the user.** Give the acceptance table, the identity result, the stop test result, and what is left for the upstream pull request: drop `docs/superpowers/`, and the PR title starts with `[Engine]`.
