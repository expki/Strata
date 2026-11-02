# DFlash2 in the server, switchable with MTP

Date: 2026-11-02. Branch: off `mixed-tier-perf`. Target: the test box (RTX 3090 + Radeon PRO W6800, `strata.service`,
Swift 1.5 Qwen3.8 Flash Next, IQ4_XS today, Q6_K as the other config).

## Intent

The user wants the DFlash2 block drafter in the server that `strata.service` runs, chosen at start instead of MTP, so the
two can be compared and switched by config. Done means "working and switchable". Making DFlash2 faster than MTP on this
box is follow-up tuning, not part of this design.

Today DFlash2 runs only in the command-line spec loop on one card. The server refuses it at start
(`src/program/generate.cpp:6356`, "it is for one card and the CLI run") and drafts with MTP.

## Decided with the user

- Drafter choice is a start-up setting. The run config's `"drafting": "mtp" | "dflash2"` picks it, the service restarts
  to switch. No switch while running.
- Only the chosen drafter is loaded. In DFlash2 mode the MTP drafter is not loaded (no VRAM spent on a fallback).
- "Working and switchable" is enough. Tuning DFlash2 comes after.
- Approach 3 (below): first put MTP and DFlash2 behind one drafter interface in the server, then plug DFlash2 in.

## Goals

1. `--serve --draft dflash2 --draft-model <file>` works with the box's configs: `--peer-device hip:0` (the W6800 tier),
   the conversation cache, vision, sampling (temperature 1.0, top_p 0.95, top_k 20), contexts up to 262144, `--kv int8`.
2. Every conversation-cache restore drafts with full DFlash2 context (the last 2047 tokens): a whole parked entry, a
   rewind to any checkpoint, a `pin=` sibling and a borrowed prefix (`conversation_snapshot_restore_prefix`). The user
   chose this on 2026-11-02: DFlash2's window is saved with every checkpoint (not only at an entry's end, and not
   rebuilt by reading tokens again), in half precision (about 42 MB a checkpoint), charged to `--conversation-cache-mib`.
3. MTP mode keeps today's speed and answers: greedy output identical to the current build, decode speed within noise.
4. The suffix/lookup drafter (`DraftPolicy`'s from_sfx rounds) keeps working in both modes.
5. `--draft-model` accepts the GGUF `/opt/models/incoai_Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf` directly, as well as the
   existing pack `~/source/Strata-data/dflash2/qwen3.8-flash-next-dflash2.q8.pack`.
6. A third switch target on the box: DFlash2 variants of the service config, reachable with `strata-switch.sh`.

## Non-goals

- DFlash2 with `--batch` slots, a layer split across GPUs, `--pipeline-windows 2`, Metal or SYCL. These keep refusing it
  with a clear line and start with MTP.
- Switching drafters while the server runs.
- Making DFlash2 beat MTP. Speed numbers are recorded, not required.
- App or `strata launch` work beyond what `serve/server.py` already does with `"drafting"`.

## Anti-goals

- Any change to the model's answers. Spec decoding verifies every draft, so a drafter may change speed only. KL-passing
  rounding differences from a wider verify window are acceptable. No skipped or approximated model work.
- MTP mode getting slower because of the refactor.
- A DFlash2 start that silently runs MTP. If DFlash2 cannot load, the log and `/metrics` drafting note say why.

## Constraints

- Hard fork (`expki/Strata`): clean design over smallest diff, no upstream rules. Linux, CUDA + HIP only.
- The 3090 is the decode device and holds the drafter. VRAM there is mostly expert cache; DFlash2 needs about 1.2 GiB
  (pack 0.76, buffers and K/V ring 0.25, draft head 0.2). MTP's own draft layer and head are not loaded in DFlash2 mode,
  which frees part of that.
- The DFlash2 drafter needs the taps (residual rows of layers 6, 16, 26, 36, 46) of every committed row, and of the last
  2047 prompt rows. `Prefill::set_taps` and `Verifier::set_taps` already produce them.
- Prompt taps turn off the W6800 half-split overlap for a tapped chunk (`prefill.cpp:4269`) and the layer-major walk
  (`prefill.cpp:6484`). Because periodic and tail checkpoints fall at chunk ends, every chunk taps its last 2047 rows
  (found while planning, 2026-11-02), so in DFlash2 mode every chunk loses the overlap, not only the last. Measure the
  prompt cost; reducing it is follow-up tuning. MTP mode is not affected.
- A DFlash2 failure found after the expert cache is sized (bind, prepare) ends the start with an error naming the cause,
  since MTP can no longer be loaded then. Failures at load time fall back to MTP.
- Swift 1.5 names and a `"drafting_model"` config key are added to `strata_control/drafting.py` (planning, 2026-11-02).
- Box rules: stop `strata.service` for stints, take `flock ~/bench-mixed/.box.lock`, short runs (about 5 minutes an arm),
  GPU clock settings as in the service's pin scripts, restart the service when done.

## Approaches considered

1. Add DFlash2 beside MTP in the server loop, and save the drafter's context in the conversation cache. The least work,
   leaves MTP untouched.
2. Same as 1 without saving the context. Restored chats would draft from only the new message. Rejected: agent sessions
   almost always hit the cache.
3. **Chosen by the user.** Put MTP and DFlash2 behind one drafter interface in the server loop, then plug DFlash2 in.
   It costs a refactor of the MTP paths in the 8,000-line serve block (lines ~9564 to ~17779) before DFlash2 runs, and
   puts MTP's speed at risk, which goal 3 guards. It gives the cleanest base for the tuning that follows. Context saving
   from approach 1 is kept (goal 2).

## The design

### One drafter interface for the server

A `Drafter` (name the builder's call) that the serve loop calls instead of `mtp.*` directly. What the loop needs from
it, from today's MTP uses in the serve block (about 150 call sites: `kv_state`, `prefill`, `draft`, the chain calls,
`prob`, `set_draft_sampling`, `set_prompt_len`, `kv_restore`, `draft_kv`, snapshots `draft0`):

- start: load, bind to the weights and head, prepare (graph capture), VRAM it holds;
- prompt context: what to collect while a prompt is read (MTP: its draft K/V from the final rows; DFlash2: the taps of
  the last 2047 rows) and the hook that takes it;
- one round: drafts and their probabilities for the next window, from the committed rows (MTP: the chain; DFlash2: the
  block from the committed rows' taps);
- after verify: what was accepted, so the drafter's own state follows the commit and a rejection;
- snapshots: save and restore the drafter's state with a conversation-cache entry (MTP: its draft K/V, today's
  `draft0`; DFlash2: its K/V ring at that checkpoint's position, in BF16 or FP16, about 42 MB, charged to
  `--conversation-cache-mib`). The ring is position-keyed (`pos % 2064`), so a checkpoint saves the window ending at
  its own position, and a prefix borrow restores the window that ends where the borrow ends. Where no saved window
  matches a restore point exactly, the builder reports which case it is and how often it happens; it must not draft
  from a ring holding rows past the restore point;
- sampling: draft sampling and the draft probabilities the verify step needs for temperature 1.0;
- stats for the log and `/metrics` (rounds, ms per round, accepted/offered).

MTP-only features (the async chain, `--pipeline-windows`, draft history, chain rules) may stay as MTP-specific
capabilities the interface exposes as optional, rather than forcing DFlash2 to fake them. The exact shape is the
builder's call; the rule is that the serve loop holds no `use_df2` / `use_mtp` branches beyond the ones a capability
check needs.

The command-line spec loop may move to the interface too if it is cheap; it is not required.

### Start-up selection

- `--draft dflash2 --draft-model F` with `--serve` loads only DFlash2. `--mtp` still names the MTP folder because the
  draft vocabulary (`draft_vocab.bin`) lives there; the MTP layer itself is not loaded.
- If DFlash2 cannot load (VRAM, a refused mode from the non-goals, a bad file), the engine loads MTP instead and says why
  in plain words, and `serve/server.py`'s `drafting.fallback_note` reports it. The "one card and the CLI run" refusal
  text goes; `fallback_note`'s matching is updated.
- `serve/server.py` turns `"drafting": "dflash2"` into `--draft dflash2 --draft-model <pack>` through
  `strata_control/drafting.py`, but only while `ENGINE_SERVES_DFLASH2` is True; it is False today, so the choice falls
  back to MTP. My call: set it True once the engine serves DFlash2, update `fallback_note` and
  `tests/terminal/test_drafting.py`. This is not a default change: Auto stays MTP because no `SUGGESTED` row has
  `faster_everywhere`. The config may name a drafter file (the GGUF) instead of the data-dir pack.
- The box service runs `serve.server --config <json>` (seen 2026-11-02), so the DFlash2 variants are the same json with
  `"drafting": "dflash2"` and the drafter path. The current configs use neither `--batch` nor `--pipeline-windows`, so
  the two modes compare like for like. The configs and `strata-switch.sh` live on the box, not in the repo.

### GGUF as a drafter file

The engine's DFlash2 loader accepts the incoai GGUF (architecture `dflash`): the Q8_0 matrices map byte for byte onto
the pack's q8_0 rows, F32 norms and conv bases map directly, and the two selector codebooks (Q8_0 in the GGUF, BF16 in
the pack) are converted at load. Names map as `blk.N.attn_q` -> `layers.N.self_attn.q_proj`,
`enc.output_norm` -> `hidden_norm`, `output_norm` -> `norm`, `selector_hidden` -> `candidate_selector.hidden_projection`,
and so on; the builder confirms each against the pack written from the safetensors. Config values come from the GGUF
metadata (`dflash.*`). The pack path keeps working.

### Box configs

Add DFlash2 variants of the IQ4_XS and Q6_K service configs and a `strata-switch.sh` target for each (names the
builder's call). MTP configs stay as they are.

## Validation

- Unit or tool tests where the code allows: the GGUF loader against the pack (same weights after conversion, the
  codebooks within Q8_0 rounding), the drafting Python tests (`tests/terminal/test_drafting.py`).
- On the box, IQ4_XS config, each arm about 5 minutes:
  - MTP mode, new build against the current build: greedy output identical on a fixed prompt set, decode tok/s and the
    20K prompt within noise. This is the refactor's gate.
  - DFlash2 mode: greedy output matches MTP mode (or passes the KL gate), a sampled chat works, a conversation-cache hit
    drafts with the restored context (acceptance on the second turn close to a fresh read of the same text), a rewind
    to an earlier checkpoint drafts with that checkpoint's window, vision
    request works, the 20K prompt cost of the taps recorded.
  - Switching: `strata-switch.sh` between the MTP and DFlash2 configs, both start and serve.
- Record accept rate, tok/s and prompt tok/s for both modes in the report. No speed bar for DFlash2.

## Left to the builder

- The interface's exact shape and file layout, and whether the CLI loop adopts it now.
- Whether the saved ring is BF16 or FP16 (check the drafter's acceptance does not drop against FP32 on the parity case).
- The DFlash2 window cap on this box (`--draft-cap`, `dflash2_auto_cap` gives "--spec as given" on sm_86 24 GB).
- Config names and the switch script's target names.
