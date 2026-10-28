# Strata on two or more GPUs (layer split)

This page is for people with two or more graphics cards of the same make in one PC. A layer split puts one model on
several cards. NVIDIA is described here; AMD limits and measurements are in [AMD_HIP.md](AMD_HIP.md).

> **On this page:** [Using it](#using-it) · [Main, Helper, Off](#main-helper-off) · [How the split works](#how-the-split-works) ·
> [Tuning options](#tuning-options) · [Pipeline windows](#pipeline-windows) · [Two models on two cards](#two-models-on-two-cards) ·
> [DeepSeek](#deepseek-v41-flash) · [Limits](#limits) · [NVIDIA and AMD together](#nvidia-and-amd-together-experimental) ·
> [History](#history--older-measurements)

## Using it

**Desktop app:** choose Main, Helper or Off for each card. **Terminal:** run `strata launch` and use the Cards step, or name
the cards:

```sh
strata launch --gpu 0,1
strata launch --gpu-roles 1=main,0=helper     # cards you do not name are Off
```

Choose a model that fits the combined memory. A split can add capacity without adding speed.

**Source checkout only:** `START-HERE.bat` and `./setup.sh` have their own card picker. Their flags are not `strata launch` flags:

| Source setup flag | Meaning |
| --- | --- |
| `--gpus 0,2` | These cards together, numbered as `nvidia-smi` numbers them. Strata puts the faster card last. Remembered. |
| `--gpus all` | Every card that can share the model. |
| `--gpu 0` | One card (at a start: for that start only). |
| `--layer-split auto` | The default. Or the first layer of each later card, for example `18` or `16,32`. |
| `--gpu-roles 1=main,0=helper` | Roles, saved. |

Or edit an existing run config (`strata-*.json`) and restart. Merge these fields into the config; this is not a complete config:

```json
{
  "gpu": [0, 2],
  "layer_split": "auto"
}
```

Docker: `GPU_ROLES=1=main,0=helper` is passed as `--gpu-roles` ([DOCKER.md](DOCKER.md)).

## Main, Helper, Off

Every card has one of three roles.

| Role | What it does |
| --- | --- |
| **Main** | Runs the last stage of the split: the output head, the draft layer and the check of its guesses. A prompt chunk waits on it. Exactly one card is Main. |
| **Helper** | Holds a share of the layers and of the expert cache, so more of the model sits on graphics cards instead of being read from the drive. The effect on speed depends on the cards and the workload. |
| **Off** | Strata does not touch the card (a game, the screen, a card that is too small). |

The Main card is the last card in the engine's order. The first card runs the first layers and, by default, the picture reader.

**Recommended roles** (the same rule in the app, `strata launch` and setup; `strata_control/gpu_roles.py`):

- The faster card is Main. Speed is multiprocessors x max clock. Cards within 5% count as the same card, and the one with more memory wins, then the lower number. Of two identical cards the highest number is Main. A card with clearly more memory beats a faster one that is at least half as fast.
- Every other card starts as Helper, except a card with 8 GB or less when a bigger card is present (Off; with only 8 GB cards the second one is a Helper), built-in graphics beside a card with memory of its own (Off), and a card the engine cannot run (Off).
- A role is one click (app), one key (`strata launch`: M, H, O; R restores the recommendation) or one line (above).

Only a few things are warned about, one short line each: built-in graphics as Main, a card the engine cannot run, and too
little memory for the chosen setup. All cards Off is allowed. The CUDA, HIP and SYCL engines cannot start without a card and
say so; the Mac engine says "Performance is degraded on CPU only." and goes on.

**In the run config** (what the app writes). Merge these fields into the existing config:

```json
{
  "gpu_roles": {"1": "main", "0": "helper", "2": "off"},
  "gpu": [0, 1],
  "gpu_order": "as_given"
}
```

`gpu` is the same choice for older engines and the Monitor: the Helpers (slowest first), then the Main card, Off cards left out.
The server reads `gpu_roles`. If `gpu` was edited by hand and no longer matches, `gpu` wins and the log says so. A file with only
`"gpu": [1, 0]` shows as Main = the last card, Helper = the first, the rest Off. The recommendation is not saved, so a card you
add later is recommended again; a card you add after choosing roles starts Off.

**Which cards count** (`strata_control/gpu_fit.py` holds the memory arithmetic): not an integrated chip beside a real card; at
least 4 GB of graphics memory to be listed; 8 GB or less starts Off beside a bigger card. Each extra card of a split costs an
estimated 2 GB of its own buffers.

**Not supported** (setup says so and names the cards that can be used):

- A card older than the RTX 20 series (compute capability below 7.5). Pascal and Volta cards are admitted when you name them
  with `--gpus` (or `--gpu`, `--cuda 12`) and then run the experimental CUDA 12 engine ([OLDER_GPUS.md](OLDER_GPUS.md)).
- A card with less than 8 GB of VRAM together with others, because each card holds a copy of the dense weights and its own
  prompt buffers. If you name it with `--gpus`, setup states the risk and asks (`--yes` goes ahead).
- Intel GPUs in a split, and a mix of NVIDIA and AMD cards. An engine drives one make of card. The app, `strata launch` and setup
  offer only the cards the engine can drive, and say so when the other make has the bigger card. AMD cards share a model among
  themselves: `./setup.sh --backend hip --gpus 1,0` ([AMD_HIP.md](AMD_HIP.md)). One exception, experimental and built
  from source on Linux only: the NVIDIA engine can use one AMD card as an extra expert cache. The app, `strata launch`
  and setup do not set it up. See [NVIDIA and AMD together](#nvidia-and-amd-together-experimental).
  A prompt read on one card and an answer made on another, with the session handed over in RAM, is a different opt-in: [SPLIT_SERVE.md](SPLIT_SERVE.md).

## How the split works

The layers are cut into contiguous ranges, one per card. The first card runs layers 0 to K-1, the next runs K onward, and the
last card also runs the output head and the draft (MTP) layer. Each card keeps an expert cache for its own layers only, so two
cards hold about twice the experts one card holds.

This is pipeline parallelism, not tensor parallelism. A token crosses from one card to the next once per verify window (a few
hundred KB through pinned RAM). No NVLink or peer-to-peer access is needed, cards on x4 or x1 slots work, and each card's PCIe
share is probed on its own link. A `--pcie-frac` you give applies to every card and skips those probes.

**What each card holds**

- Every card: a copy of the dense weights, its own session state (the KV cache of the full context), its verify window, its
  prompt-path buffers, and an expert cache for its layers filled from the expert profile.
- The last card: also the output head and the draft layer (about 0.8 GB).
- Host RAM: the expert arena once, shared by all cards. The CPU pool computes whatever no card holds.

Prompts are read in chunks that flow through the cards in turn: while a later card reads chunk c, the first card reads chunk c+1.
Conversation checkpoints save and restore every card's state.

**`"layer_split"`** is `"auto"` (placed by each card's free VRAM) or the first layer of each later card: one rising number per
card after the first, not a count per card. With four cards and a 48-layer model, `"24,36,42"` puts layers 0-23 on the first
card, 24-35 on the second, 36-41 on the third and 42-47 on the last. The server checks the value before the start and says what
is wrong.

**`auto`** tries every placement for two, three or four cards (proportional to free VRAM beyond that) and keeps the one whose
caches hold the most of the expert profile, hottest pairs weighted most. Ties go to the placement that leaves the fullest card
the most room. The startup log prints the choice:

```
strata generate: layer split auto: K=19 - the caches hold 11767 of 12288 profiled pairs (fullest device 100%)
strata serve: layer split: layers 0-18 (CUDA0), 19-47 (CUDA1), one hand-off per window
```

**Card order with `auto`:** the faster card goes last, because the last stage runs the head, the draft layer and the verify, and
a prompt chunk waits on it. Cards within 5% keep your order, and so does a manual split. `"gpu_order": "as_given"` keeps your
order under `auto` as well. The log line `layer split: card order ...` says when the order changed.

**Unequal cards under `auto`.** The search asks for at least 3 layers per card and, on two cards, first for a placement in which
each card owns at least its share of the layers by its room for experts. The log line says which rule chose the split. Prefill
that stays flat from 4K to 16K tokens is the sign of an idle card. On an unequal pair, sweep `"layer_split"` once, because a
4K-only benchmark cannot see it. On x16 plus x4 rigs the link-miss scaling can move `auto` off the balanced placement;
`STRATA_SPLIT_UNIFORM=1` (with `"gpu_order": "as_given"`) prices every link the same.

## Tuning options

These are engine flags and environment variables. Put flags in the config's `"args"` and variables in its `"env"` object, then
restart. Defaults are off unless the table says otherwise.

| Setting | What it does | Notes |
| --- | --- | --- |
| `"split_skip_if_fits": true` (flag `--split-skip-if-fits`) | Runs on the first card alone when it holds every profiled expert plus the context's KV, the draft layer and the reserve. Otherwise the split stays. | Opt-in. Needs `--layer-split auto`. |
| `STRATA_SPLIT_OWN=1` or `auto` | Gives every card its own prompt buffers. `auto` does it only where the buffers are at most 12% of each card's VRAM. | Output differs from the default's (stable and coherent). |
| `STRATA_PREFILL_HELP=1` | The idle card computes a share of the streamed experts of a one-chunk prompt. `STRATA_PREFILL_HELP_FRAC=f` fixes the share. | Helps prompts up to about 3.3K tokens on slow links only; not bit-identical; native packs on the MMQ prompt path only. |
| `STRATA_SPLIT_MTP_BATCH=1` | The last card fills the draft layer's prompt K/V in batches, as on one card. | Needs a `--native` pack and a paged or ring draft K/V; says why once if it declines. |
| `STRATA_STAGE_TRIM=1` | Each card loads only its own layers' dense weights (needs explicit `--layer-split K`, not `auto`). The freed VRAM goes to that card's expert cache. | Output can differ slightly. |
| `--vram-reserve-later-mib N` | A separate VRAM reserve for the later cards (default: `--vram-reserve-mib`). Use it when the display sits on the last card. | `--vram-reserve-mib 300 --vram-reserve-later-mib 1800` gives the first card that VRAM. |
| `--prefill-pipe R` (or `STRATA_PREFILL_PIPE=R`) | Reads each prompt in the chunk count with the least pipeline time for b/a = R tokens. `1` evens out today's chunk count. | Opt-in. `./setup.sh --calibrate` measures it per PC and writes it to the config. |
| `STRATA_SPLIT_RING=N` | Sets the prompt path's streamed ring to N slots (`0`: the pinned-share rule). A split uses 96 when at least 75% of the (layer, expert) pairs sit in some card's cache. | The startup line `lend sizing: ...` shows the result. |
| `STRATA_SPLIT_CHUNK16K=1` | Keeps the 16384-token auto chunk on AMD gfx12 splits (default 8192). | |
| `STRATA_SPLIT_OVERLAP=1` | Overlapped hand-off between stages on different cards. | Helps AMD; no gain on NVIDIA. See [below](#overlapped-hand-off). |
| `--split-device D1[,D2..]` | The later stages' devices (default: the next visible ones). `--layer-split K --split-device 0` runs both stages on one card; a bit-exact hand-off check, not a speed mode. | |

**The resident RAM mode works on a split.** `--resident-experts` (the low-RAM mode) leaves out of the RAM copy the experts any
card's cache holds. An adaptive swap copies the evicted expert back into RAM from the card that owns its layer. A RAM budget
(`--resident-budget-gib`) works on a split as well ([UNSLOTH_Q4.md](UNSLOTH_Q4.md)). With `--pcie-frac 0 --adapt-every 0` the
split's greedy output is the same in either mode.

**Under WDDM (Windows and WSL2)** only 8 GiB of the expert arena is pinned on a multi-GPU run; the rest streams through the
pinned staging ring. Linux pins the whole arena. `STRATA_ARENA_PIN_GIB=N` pins at most N GiB, `0` the whole arena, on any OS.

### Holding a card to fewer experts (`--expert-gib`)

`--expert-gib` caps how much of each card's memory the experts may take. The experts left over stay in RAM. Use it to
try a setup in which most of the experts live in RAM, or to keep memory free on a card for something else.

```
--expert-gib cuda0=1,cuda1=3,cuda2=5,hip0=6.5
```

- `cudaN` is the card the log calls CUDAN: a layer-split stage, or the second NVIDIA card of `--peer-device N`.
  `hipN` is the AMD card of `--peer-device hip:N`. The numbers are GiB, and they may have a decimal point.
- A card that is not in the list has no cap. A number that is zero, negative, repeated or not a number stops the start
  with a message.
- The cap is in bytes of expert slots, so it does not count the weights, the session or the prompt buffers. It
  lowers the size the start would pick on its own. It never raises it.
- With `--layer-split auto` the search sees the caps, so a small cap on one card gives its layers to the others.
- A capped split stage whose cache is too small to lend the prompt path its chunk's buffers holds them on the card
  instead, when the card has room for them plus 1.5 GiB (`CUDAN keeps its own prompt buffers` in the log). The chunk
  is cut only when a stage can do neither: `prompt chunk 4096 -> 1024 tokens so its buffers fit in every expert cache`.
- `--expert-cache N` on the first card is kept as asked, and the cap does not apply to it. The same goes for the
  helper caches of `--expert-cache-device1..3`, which this option does not cover.
- The start-up lines show the size after the cap: `expert cache auto: ... -> N slots`, `layer split: CUDA1 runs
  layers ..., expert cache N slots (X GiB)` and `AMD tier on HIP device 0: N experts, X GiB`.

### Overlapped hand-off

By default each verify window runs the stages one after another. `STRATA_SPLIT_OVERLAP=1` (stages on different cards only) gives
the hand-off a ready flag, so the next stage's graph is launched while the earlier stage still runs and waits on the flag on
its GPU. It takes one stream sync and one graph launch per window off the critical path. Batch windows keep the serial order, and
it is off beside `--pipeline-windows`. The wait is bounded by `STRATA_SPLIT_WAIT_MS` (default 30000, `0` = no bound); on timeout
the window fails with an error naming the setting and the server restarts the engine.

## Pipeline windows

`--pipeline-windows 2` lets the first card start the next verify window while the last card still verifies this one. The next
window is a guess: that this window is accepted whole and that its bonus token is the one the draft layer predicts. If the guess
holds, half of the next window is already done. If not, the first card restores its state and builds the window from the real
tokens. The tokens are the serial loop's, up to CPU rounding.

- **Default:** on for AMD RDNA4 (gfx1200 / gfx1201) with `--serve`, the draft layer, and a split of two or more stages when the
  drafter can sit on the first card, otherwise three or more. The log says "is on by default for a layer split of N stages".
  `--pipeline-windows 0` or `STRATA_PIPELINE_AUTO=0` turns it off. NVIDIA builds keep it opt-in (`--pipeline-windows 2`).
- **Draft layer placement:** with the pipeline on, the draft layer loads on the first card, whose window has finished when the
  chain starts. `STRATA_MTP_STAGE0=0` keeps the old place.
- **Off, with one log line saying why:** `--batch` slots, `--peer-device`, the helper caches (`--expert-cache-device1..3`, which
  `--remote-expert-opt` builds on), a split onto one GPU, no draft layer, and on three or more stages `--pipeline-windows 1` and
  `--adapt-async 1`. A request with repetition penalties or coupled draft sampling decodes serially.
- **Cost:** a second verify window on each card, 160 MiB less expert cache per card, plus two copies of the first card's
  recurrent state (about 3 MiB per GDN layer it runs).
- **Config:** `"args": [ ..., "--pipeline-windows", "2" ]` with `"layer_split": "20"`.

`--remote-expert-opt` does something only with a helper cache; on a plain layer split it is inert.
The `STRATA_PIPELINE_*` tuning and test variables are read only with `STRATA_PIPELINE_DEBUG=1`.

## Two models on two cards

The other way to use two cards is two models, one per card, each answering on its own. In the app, turn on **Run multiple
models** on the Models page. The card list then says which card a loaded model uses ("In use by ..."), starts that card as Off
for the next model, and explains a failed start on a card that is taken. Two models at once on two makes is not supported.

**Measured on:** a PC with two of four Radeon AI PRO R9700 cards, Linux, AMD build, Q2_0 on one card and IQ3_XXS on the other.
Both loaded (33 s and 59 s with the files cached, 114 s the first time) and answered chats at about 60 tokens/s each. Unloading
one left the other answering. The same app code drives NVIDIA cards, but two models on two NVIDIA cards were not run.

## DeepSeek V4.1-Flash

DeepSeek V4.1-Flash runs in the app and the Terminal and spreads its experts and its dense weights over several cards. It does not use
the Qwen layer split described above. On four Radeon AI PRO R9700 cards it reaches about 31 tokens/s. Hardware, sets and speeds:
[MODELS.md](MODELS.md#deepseek-v41-flash).

## Limits

- Every card needs compute capability 7.5 (RTX 20 or newer), unless you opt into the CUDA 12 engine for older cards.
- Works across cards: images (`--vision`), control vectors, KV streaming (`--kv-resident`) and mid-prompt checkpoints
  (`--prompt-cache-every`). Each card keeps its own part.
- The older helper-GPU caches (`--expert-cache-device1..3`, [SECOND_GPU.md](SECOND_GPU.md)) take the visible GPUs no stage runs on.
  A weak helper can make decoding slower, because its per-layer round trip costs more than the CPU pool needs for those experts.
- `--mmap-experts` needs a canonical pack (`experts.bin`); a native (IQ) pack says so at start.
- The prompt path has its own buffers on every card (1.5 GB each at the default 2048-token chunk; `--prefill 1024` halves that).
  An explicit `--expert-cache` on the first card is capped to leave room for them.
- More cards pay off when the model's routed experts do not fit the faster ones. An extra slow card costs its own round per
  window and can make a good pair slower.
- **The split pays most when the CPU pool is the bottleneck.** On two identical RTX 2080 Ti 22 GB over a dual-socket Xeon E5-2696 v4 (AVX2 only,
  88 logical CPUs), a plain split moved decode from 35.2 to 51.0 tok/s (+45%) and an 11.6K-token prompt from 672 to 908 tok/s (+35%) at 180 W,
  engine 0.1.49, six 300-token coding prompts. `STRATA_DECODE_TIMING=1` shows why: the window fell from 70.06 to 37.96 ms, almost entirely
  because the CPU pool fell from 9.52 to 0.79 ms per window. The 5080 + 3090 measurement below, where decode with the second card is on par
  with the 5080 alone, was taken over a fast CPU (Ryzen 9 9950X3D). When the CPU pool is what the window waits on, moving half the layers to a
  second identical card removes that work. This is one pair of cards and one CPU, so there is no threshold, and the window breakdown from the
  engine log is not the same statistic as the decode numbers beside it. Setup splits across two cards by itself; with two identical older cards
  on a slow workstation CPU, compare one card against the split and tell us in [#3930](https://github.com/Niko1221/Strata/pull/3930).
- For several conversations at once, `--batch N --batch-groups G --trim-stage-weights` pipelines them through the cards
  ([BATCHING.md](BATCHING.md)).

## NVIDIA and AMD together (experimental)

**What it does.** On a PC with NVIDIA cards and one AMD card, one model can use both makes. The NVIDIA engine runs the
model as usual, on one card or split across several. The AMD card holds more of the model's experts: the ones no
NVIDIA card has room for. It computes those experts itself, while it answers and while it reads a prompt. Without it,
the CPU computes every expert that is not on a card, and that is the slow part when a model does not fit in VRAM.

- The AMD card's experts change while the server runs, as the NVIDIA cards' do. An expert is on one card at a time.
- The two makes never talk to each other directly. The inputs and the results go through pinned RAM.
- It is off unless the engine gets `--peer-device hip:N`. Without that flag the answers are byte-identical to the build
  before this feature (checked below).
- It is for Linux and the NVIDIA (CUDA) engine, with one AMD card and a native pack. The limits are at the end of this
  section.

**Build it** (from source, on Linux with ROCm). The engine is the usual CUDA build. The AMD part is a library,
`libstrata_xtier_hip.so`. Build it in its own HIP build tree with `-DSTRATA_XTIER=ON`, which needs
`-DSTRATA_PREFILL_MMQ=ON`. Set the architecture to your card's ([AMD_HIP.md](AMD_HIP.md#build) has the other HIP
flags):

```sh
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_XTIER=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030
cmake --build build-hip --target strata_xtier_hip
```

- The engine looks for the library next to its own file, or where `STRATA_XTIER_LIB` points.
- The library needs ROCm's libraries when it runs, `libhipblas` among them. Put `/opt/rocm/lib` in the config's
  `"lib_dirs"` (or on `LD_LIBRARY_PATH`). Without it the start stops and asks "is /opt/rocm/lib on LD_LIBRARY_PATH?".
- Build the library and the engine from the same commit. The engine checks the library's interface number and the
  sizes of the structures the two share, and refuses the library if either differs.

**The config.** This is the test box's config, without its picture settings. Next to an NVIDIA config it has
`--peer-device hip:0` at the end of `"args"`, the ROCm folder in `"lib_dirs"` and `STRATA_XTIER_LIB` in `"env"`:

```json
{
 "exe": "/home/you/strata/build-cuda/strata",
 "args": ["--pack", "/home/you/Strata-data/packs/swift-iq4_xs",
          "--native", "/home/you/models/ukisai_Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf",
          "--ple-gguf", "/home/you/models/ukisai_Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf",
          "--expert-profile", "/home/you/strata/data/expert-profile.bin", "--mmap-experts",
          "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
          "--mtp", "/home/you/Strata-data/mtp/rt", "--max-context", "262144", "--kv", "int8", "--pool-workers", "15",
          "--peer-device", "hip:0"],
 "cwd": "/home/you/strata",
 "tokenizer": "/home/you/Strata-data/packs/swift-iq4_xs/tokenizer",
 "model_name": "swift-1.5-qwen3.8-flash-next-iq4_xs",
 "arch": "qwen4exp",
 "backend": "cuda",
 "gpu": [0, 1, 2],
 "layer_split": "auto",
 "lib_dirs": ["/usr/local/cuda/lib64", "/opt/rocm/lib"],
 "env": {"STRATA_XTIER_LIB": "/home/you/strata/build-hip/libstrata_xtier_hip.so"}
}
```

- `"gpu"` lists the NVIDIA cards only, here three as a layer split.
- `hip:0` is the first AMD card. N counts the AMD cards only, from 0. If you set `HIP_VISIBLE_DEVICES` yourself, N
  counts the cards it leaves visible.
- `"lib_dirs"` has both runtimes. `"env"` says where the library is, because it was built in another tree.
- `--peer-reserve-mib` and `--peer-slots` size the AMD card's share as they do for a second NVIDIA card
  ([SECOND_GPU.md](SECOND_GPU.md#peer-tier---peer-device)). When the server reads prompts in chunks of 1,024 tokens or
  more, the AMD card first opens its buffers for reading prompts, sized for the chunk, and its ring of copies (below)
  when the experts no card holds do not all fit on it. Its experts fill what is left. `--peer-reserve-mib` is the memory
  that stays free after all of them (320 MiB by default on an AMD card, 600 on a second NVIDIA card). The log says so.
  On the test box (RTX 3090 and W6800, Swift 1.5 Q6_K, chunks of 4,096 tokens):
  `strata generate: the AMD card keeps prompt buffers for 4096 tokens (255 MiB) and a ring of 128 copies (540 MiB)
  before its experts, 320 MiB stay free`. The W6800 then held 7,442 experts. With chunks of 20,480 tokens the buffers
  took 1,004 MiB and it held 7,264.
- With one NVIDIA card and `--prefill auto`, the server reads prompts of up to 20,480 tokens in one chunk. It takes the
  largest chunk on the 256-token grid that the NVIDIA card can lend from its expert cache while 128 slots stay, and
  picks it before the AMD card's experts fill, so the AMD card's prompt buffers are sized for it. `--prefill auto:8192`
  (or `STRATA_PREFILL_AUTO_MAX`) holds it to at most that many tokens. A request lends only
  what its own prompt needs, so a short prompt costs nothing more. The log says
  `strata generate: prompt chunk auto: 20480 tokens (one card with the AMD tier: ...)`. On the test box (the RTX 3090
  locked at 1,500 MHz, the W6800, Swift 1.5 Q6_K, two engine starts each, two passes a start) a 20,053-token prompt read
  at 1,676 to 1,877 tok/s, against 741 to 801 with chunks of 4,096 tokens. An 8,013-token prompt read at 1,170 to 1,251
  against 718 to 765. 2K prompts read at 375 tok/s on average in both. The three chats decoded at 27.67 tok/s on average
  in both, but the third one alone went from 25.2 to 24.5. Answers change slightly: the KL against the reference
  answers read in one chunk was 0.031 on all 50 prompts (the gate is 0.03), with 98% of the top tokens the same.
- The AMD card gets a prompt chunk's rows and sends its sums back in FP32 (`STRATA_PF_PEER_F16=1`: FP16, half the
  bytes). On the build above, FP32 sums brought the KL on all 50 prompts from 0.031 to 0.025 (one start each). With the
  pinned RAM copy and FP32 sums two starts read 0.024 and 0.025. FP32 costs prompt speed. With the copy and the 3090
  at full clocks (two starts each, two passes a start) a 20,053-token prompt read at 3,035 to 3,113 tok/s against 3,135
  to 3,211 in FP16 (3% slower), an 8,013-token prompt at 2,003 to 2,050 against 2,103 to 2,125 (4%), and a 2,041-token
  prompt 1% slower. Chats decoded as fast. Rows in FP16 with the sums in FP32 (half the bytes on the way to the card)
  were tried and removed: with the copy, the 3090 at 1,500 MHz and two starts each, a 20,053-token prompt read at 2,874
  and 2,917 tok/s with them, against 2,882 and 2,920 without.
  Copying each half's sums into the NVIDIA card's memory by DMA as the half lands, so the combine reads VRAM instead of
  mapped host memory, was tried and removed too. On the test box (b3-s1, Swift 1.5 Q6_K, the 3090 at 1,500 MHz, two
  starts each) the 20K-token prompt read at 2,310 and 2,499 tok/s with it, against 2,884 and 2,856 without.
  Two more were tried in the same stint and removed. Refilling the slots a prompt lends behind the first decode windows,
  instead of before the first token, never showed a resolved gain on the 20K-token answer: one A/B read 53.8 against
  51.7 tok/s, and the leave-one-out read 51.2 without it against 42.5 with it, single starts each. It was also the only
  thing that made decode differ from the reference. SwiGLU and the down product's q8_1 quantize as one prompt kernel
  read 2,404.1 tok/s on the 20K-token prompt against 2,412.6 without it, so it was not faster.
- `--expert-gib hip0=6.5` holds the AMD card to 6.5 GiB of experts
  ([above](#holding-a-card-to-fewer-experts---expert-gib)).
- **The pinned RAM copy is on by default with one NVIDIA card** (Linux, the experts read in place with
  `--mmap-experts` and a profile). It works as `--resident-experts`: every expert the NVIDIA card does not hold, the
  slots it lends to prompts included, sits in page-locked RAM. The prompt path copies them to the card by DMA, and no
  request reads them from the drive. It needs the RAM available at the start to hold every expert of the model with
  16 GiB and the pack's other files to spare. If it does not, the log says so and the start runs without it.
  `--no-resident-experts` turns it off. While the copy is on, the NVIDIA card reads 30% of the CPU's missed experts
  over PCIe from the copy. On the test box the 20K-token answer decoded at 62.3 tok/s with that share, and at
  53.5 tok/s with the link probe's 0.54. That was one stint on build `b4b6978d`, with the AMD card's direct reads of the
  copy still on (they were removed later). At today's build the decode KL at 0.30 is 0.0307 against share 0: the 3090
  and the CPU round those experts differently. The link probe still runs, because the prompt planner takes the 3090's link speed from it. `--pcie-frac N`
  sets another share, but it skips the probe, so the planner then starts from 12 GB/s for that link and learns the rest.
  `STRATA_COPY_PCIE=N` holds N (`0` keeps every miss on the CPU, also with
  `STRATA_PCIE_BALANCE=1`), and `STRATA_COPY_PCIE=probe` uses the probe's share. The AMD card reading the experts it
  streams straight from the registered copy was tried and removed. On the test box it was not faster: the 20K prompt
  read 2,867 tok/s against 2,891 without it, the chats were slower, the W6800 kept 43 MiB of VRAM free against 175, and
  the start took 3 to 6 s longer. In a probe (no engine) the AMD card read the registered copy at 22 GB/s and the 3090
  at 20.4, each alone, and 19.1 and 16.9 together. The AMD card opens no ring beside the copy: through the engine's
  staging threads the ring gave nothing. Because every expert the NVIDIA card streams is page-locked, its
  ring for prompts grows with the chunk: one expert per 40 tokens, from 96 to 512 (`STRATA_PREFILL_RING` overrides it).
  The sizes come from the test box (RTX 3090 locked at 1500 MHz and W6800, 188 GiB RAM, Swift 1.5 Q6_K,
  `--prefill 24576`), one start at each fixed ring: 192 read the 8K prompt fastest and 384 to 512 the 20K one. At 2K,
  192 read 614 tok/s and 384 read 580. A ring of 96 read 626 to 632, but on the build before and with
  `--peer-reserve-mib 1450`, so that comparison crosses builds. With the fixed ring of 384 (two starts each, one
  session) the 20K-token prompt read at 2,969 and 2,971 tok/s and the 8K at 2,004 and 2,069, against 2,243 and 2,321,
  and 1,388 and 1,464, with `--resident-experts` on the build before. The copy held 99.36 GiB, and the server was ready
  in 39 and 43 s.
- With the copy the NVIDIA card's expert cache leaves more VRAM free: the copy's registration takes about 2 MiB of it
  for each GiB of the copy, and the captured verify windows about 140 MiB more after the start. On the test box the
  3090 had 233 MiB free at the start and 91 to 95 MiB at later requests, under the 256 MiB the engine calls LOW. The
  cache now keeps those two back and aims at 350 MiB free when a request starts. On the test box the log says
  `expert cache auto: 13.70 GiB free, 700 MiB reserved (+218 MiB for the draft head, +261 MiB for the pinned copy and
  the verify captures) -> 3042 slots`, 61 fewer than the 3,103 before. The 3090 then had 487 to 489 MiB free at the
  start and 355 to 359 MiB at the start of every request (four starts), against 233 and 99 on the build before. A
  `--vram-reserve-mib` you give is kept as it is.

At the start the log says how much the AMD card took:

```
strata generate: AMD tier on HIP device 0: 12588 experts, 31.21 GiB (filled in 16.4 s); with the NVIDIA cards' 9844 that is 22432 of 24576 on the GPUs
strata generate: AMD tier ownership: disjoint (STRATA_XTIER_OWNERSHIP=disjoint|move)
```

After each request the server's line gives the AMD card's share of the experts the answer used. This one is from
the stop test below:

```
[strata] done: 128 tokens in 2 s (100.4 tok/s) (length, cancel=False), expert cache 98.9% hit (+47.6% of the routed experts on the AMD tier)
```

The engine's hit-rate line says the same and adds how many experts the AMD card took in since the start. This one is
from a run on the test box, in the setup measured below plus `STRATA_DECODE_TIMING=1`, on the build that added the
AMD card's part of this line:

```
strata serve: decode expert cache hit rate: 99.4% (50702 hits / 51020 lookups); the AMD tier computed 27700 entries (35.2% of all 78720 routed) and took in 49 experts since the start
```

**Measured** on the test box (Linux, Ubuntu 24.04): an RTX 3050 8 GB, an RTX 3080 Ti 12 GB and an RTX 3090 24 GB as a
layer split, a Radeon PRO W6800 32 GB (gfx1030), an EPYC 7F52 (16 cores) and 62 GB of RAM. CUDA 13.2, ROCm 7.2.4.
The model is Swift 1.5 IQ4_XS, with 60.9 GiB of experts, so most of them do not fit on the cards.

How it was measured, on 2026-10-25:
- One server start per setup. The page cache was emptied before each start.
- One unmeasured pass, then three measured passes. A pass is three short chats and two long prompts (3,930 and
  15,846 tokens), 128 new tokens each, greedy. The conversation cache was off (`--conversation-cache-mib 0`), so every
  prompt was read in full.
- The 3090 and the 3080 Ti were capped at 250 W, with their clocks locked (the 3090 at 1500 MHz, the 3080 Ti at
  1095 MHz). The 3050 (130 W) and the W6800 (213 W) ran at their default limits.
- The numbers are tokens per second, the mean of the three passes, with the lowest and the highest in brackets.
- The table and the identity check below ran on the same build. One later change adds the AMD card's part to the
  engine's hit-rate line above. The other later changes were checked with the tests and the answer comparison below,
  not timed again.

| Setup | Chat answers | 4K prompt | 16K prompt |
|---|---|---|---|
| The three NVIDIA cards alone | 34.4 (32.7 to 36.4) | 369.5 (323.0 to 410.2) | 526.0 (482.6 to 568.0) |
| The W6800 alone (the AMD engine) | 24.2 (22.3 to 25.5) | 425.7 (420.0 to 431.6) | 678.3 (671.9 to 690.7) |
| The three NVIDIA cards and the W6800 | 90.7 (87.1 to 93.6) | 1,019.3 (1,010.2 to 1,035.4) | 1,476.9 (1,460.6 to 1,494.7) |

- With the W6800 beside them, the NVIDIA cards answer chats 2.6 times as fast, and read a 16K prompt 2.8 times as
  fast. Together they read a 16K prompt 2.2 times as fast as the W6800 alone.
- The answers after the long prompts got faster too: 87.7 and 92.6 tok/s with both makes, 41.2 and 75.0 with the
  NVIDIA cards alone.
- The first request after a start, with the page cache empty, is slower for every setup. A short chat answered at
  35.4 tok/s with both makes, 15.7 with the NVIDIA cards alone and 10.2 with the W6800 alone.
- The highest power draw over 5 seconds was 391 W with both makes, 260 W with the NVIDIA cards alone and 256 W with
  the W6800 alone. That is the four cards together, as nvidia-smi and the W6800's sensor report it (each an average
  over about a second). The CPU and the rest of the PC are not in it.

Without `--peer-device hip:N`, this build gave the same text as the build before it, byte for byte. That was checked
on the NVIDIA engine (the three cards) and on the AMD engine (the W6800), with three chats and the 3,930-token prompt,
128 tokens each, greedy, and `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0 --conversation-cache-mib 0` (the
settings for byte-identical repeats, see [DETAILS.md](DETAILS.md)). Two starts of the older build matched each other
too. With the AMD card on, 40 rounds of stopping an answer part-way and sending the next request at once
(`tools/stop_resend_test.py`) gave real text every time, and the next answer was the same as the first one after the
start.

With `--peer-device hip:N` the answers are close to the NVIDIA cards' answers but not the same text: the AMD card
computes experts the CPU computed before, and its prompt rows came back in FP16 then, so the sums round differently. On
the latest build, on the test box with the same caps, chats, 3,930-token prompt and settings as the check above, 3
of the 4 answers differed from those of the three NVIDIA cards alone, after the first 11, 23 and 57 of 128 tokens.
All three stayed on topic in other words: the word list's summary, how a refrigerator works and the language tips.

**Switches.** The defaults are the measured best. The others are there to compare or to look for a problem.

| Variable | Default | What it does |
|---|---|---|
| `STRATA_XTIER_LIB` | next to the engine | Where the library is. |
| `STRATA_XTIER_WAIT_MS` | 30000 | How long the engine waits for the AMD card's answer, in milliseconds. |
| `STRATA_XTIER_KEEPWARM` | 1 | `0` turns off the short spin that keeps the AMD card's clock up while it works. |
| `STRATA_XTIER_OWNERSHIP` | `disjoint` | `move` lets an NVIDIA card take an expert away from the AMD card. Measured slower. |
| `STRATA_XTIER_DECODE_GRAPH` | 1 | While decoding, the AMD card writes its experts' rows straight into the host rows the NVIDIA card reads, and replays its fixed launch sequence as one HIP graph per window shape. A shape is the window's tokens, its experts rounded up to 4 and its rows rounded up to 8, so shapes repeat. The kernels still stop at the real counts. `0` copies the rows out and launches step by step. The rows are bitwise the same either way (`xtier_decode_parity`). Each request logs a `strata serve: AMD tier decode launches:` line with how many launches ran step by step, were captured or were replayed: 99.9 and 100.0% replayed in the final stint on the test box. Its speed against `0` alone is not measured. |
| `STRATA_XTIER_DECODE_GRAPHS` | 256 | How many decode graphs the AMD card keeps. Past it, the one used least recently goes. |
| `STRATA_COPY_PCIE` | 0.30 | Beside the pinned RAM copy, the NVIDIA card's share of the CPU's missed experts. `N` holds N (`0` keeps every miss on the CPU), `probe` takes the link probe's share. `--pcie-frac` wins over it. |
| `STRATA_XTIER_CHECK` | 0 | `1` checks after every adaptive round that no expert is on two cards. A failed check ends the engine. For finding bugs. |
| `STRATA_TEST_XTIER_RING_FAIL` | 0 | A test. `1` acts as if the staging ring for experts the AMD card streams from RAM could not be had: one log line when a plan first would stream, and the AMD card streams nothing. It also skips the start-up probe of the AMD card's link, so the start-up chunk plan does not count that link. |
| `STRATA_TEST_XTIER_DROP` | 0 | A test. `N` makes the AMD card stop answering from its Nth call on, as a card that stopped. |
| `STRATA_TEST_XTIER_ROUND_TRIP` | 0 | A test print. `1` makes `xtier_decode_parity` also time 500 decode calls per format and print the host's time from a call to its rows (median and 10th to 90th percentile). It never fails the test. |
| `STRATA_XTIER_TIMING` | 0 | `1` logs the AMD card's time per request: the launch, the wait, the copies and the kernels. While a prompt is read it also logs one `strata xtier prompt timing:` line every 45 layers (the fields are below the table). For finding slow spots. |

The timing lines cost some prompt speed: with `STRATA_PREFILL_TIMING=1` and `STRATA_XTIER_TIMING=1` a 20,053-token prompt
read at 2,250 and 2,254 tok/s, against 2,278 to 2,311 without them (Swift 1.5 IQ4_XS on the test box, two starts each).

The `strata xtier prompt timing:` line gives the mean over 45 layers. Its fields, in the order they print:

- `layers`, `rows and experts each (N streamed)`: how many layers the line sums up, the prompt rows and experts the AMD card got in
  one layer, and how many of those experts it streamed from RAM.
- `card idle`: the ms between the end of the card's last layer and the start of this one. It is time the card has no work.
  The first layer after a decode adds nothing here, so the time between two prompts stays out only when an answer was
  decoded between them.
- `waits for x`: the ms the card waits for the NVIDIA card to hand over the layer's input.
- `in`: the ms to copy the input, the router weights and the row tables onto the card.
- `kernels`: the ms the card spends on the experts' math.
- `out`: the ms to convert the sums and copy them back to RAM.
- `host in the call`: the ms the engine's host thread spends in the call that queues the layer.
  `pf_mu` is the part spent waiting for the prompt lock, and `slot` the part spent waiting for a free staging slot.

**Measured again after the prompt work (2026-10-26).** The table above is from the first build with the AMD card. Later
changes, each described below with its own numbers, made prompts much faster. This compares the build they started from
(commit `70695236`) with the build after them (`a9c145fd`; later commits change only the docs), on the test box (the
3090 and 3080 Ti capped at 250 W, chunks of 4,096 tokens unless said), two engine starts of each build in the order new,
old, old, new. Each start had one unmeasured pass, then two measured ones. A pass is three short chats and a
20,053-token prompt, 128 new tokens each, with the conversation cache off.

| Setup | 20K prompt, before | 20K prompt, after | Decode after it, before | after |
|---|---|---|---|---|
| Swift 1.5 IQ4_XS, the three NVIDIA cards and the W6800 | 1,440.6 to 1,503.7 | 2,174.4 to 2,347.1 | 87.6 to 93.0 | 87.3 to 91.3 |
| The same with `--prefill auto` (one start each) | 1,456.4 and 1,465.9 | 2,183.8 and 2,196.9 | 79.8 and 85.6 | 86.3 and 92.7 |
| Swift 1.5 IQ3_S, the three NVIDIA cards alone | 803.8 to 849.4 | 1,542.9 to 1,626.2 | 64.3 to 79.1 | 72.5 to 77.8 |
| Swift 1.5 IQ3_S with a third of its experts on the four cards | 731.3 to 775.7 | 1,684.5 to 1,809.0 | 38.4 to 47.2 | 44.2 to 50.4 |

- Swift 1.5 IQ4_XS: +55%. The first long prompt after a start read at 1,817 and 2,097 tok/s against 1,319 and 1,368.
  The chats answered at the same speed (92.0 and 90.6 against 92.9 and 91.5 tok/s).
- `--prefill auto` picked chunks of 6,144 tokens on both builds (one start each). On the new one its passes fall inside
  the passes of chunks of 4,096, so this run alone shows no difference. A later run gives the reason it can be slower
  (see the open paths below).
- The NVIDIA cards alone (no AMD card): +92%. Most of it came from the prompt threads' CPUs, which alone took this setup
  from 840 to 1,570 tok/s ([DETAILS.md](DETAILS.md#thread-placement-prompt-threads-linux-always-on)).
- The W6800 alone (the AMD engine, one start each): a 4,252-token prompt read at 488 tok/s against 283, both reading
  15 to 16 GB from the drive, and the chats answered at 21.1 against 17.9 tok/s. The answers made sense on both builds.
- Swift 1.5 IQ3_S with a third of its experts on the cards (`--expert-gib cuda0=2.5,cuda1=1.5,cuda2=6.5,hip0=5`): 2.36
  times as fast. The build the work started from has no `--expert-gib` (it came in `634422cc`), and until `0e0ffcc2` the
  caps cut every chunk to 2,048 tokens. So here the build before is `9997e38e`, the first that runs this setup as it is.
  The chats were slower with the new build. Over the six measured chats of each start they read their prompts at 51.5
  and 52.1 tok/s against 69.9 and 58.0, and decoded at 41.3 and 40.1 against 44.2 and 40.5. They read more from the
  drive: 9.5 and 13.6 GB over the nine chats after each start's first request, against 5.3 and 7.2. Only 1 and 2 of the
  new build's nine chats read under 100 MB, against 6 and 6, and those few read their prompts at 65.9 and 68.8 against
  58.0 and 69.9. The new build holds about 1.8 GB more RAM, most of it the AMD card's staging rings (below), and in this
  setup the box's page cache is close to full.
- Answers change slightly: against the reference answers of 30 prompts, mean KL 0.0046 (0.0138 on the 10 long ones),
  the same top token on 29 of 30.

**Who works on which expert in a prompt.** Each NVIDIA card reads its own layers of the prompt. In each layer, an
expert the AMD card holds can be worked on by the AMD card, or by the NVIDIA card, which then copies it from RAM. Both
cards work at the same time, and the layer is done when the slower one is done. The AMD card works for all three
NVIDIA cards, so it is often the slower one. So before each chunk of the prompt, each NVIDIA card plans its layers.
It guesses each expert's rows from the chunks it read before, and takes the AMD card's busiest experts while that makes
the layer end sooner. The plan uses speeds the engine measures while it reads prompts: each NVIDIA card's time for its
own experts in each layer, the AMD card's time for each layer it got, and what the other cards have queued on the AMD
card. An NVIDIA card reads its experts in one of two ways, and it measures each way on its own. A chunk of 1,024 tokens
or more (the default of `STRATA_PREFILL_STREAM_MIN`) copies every expert it does not hold ahead of time, needed or not. A
shorter chunk copies the experts it needs one by one as it goes.

Until a card and the AMD card both have 16 measurements, the AMD card keeps all its experts, as it did before the plan.
On the test box (3050, 3080 Ti and 3090 with the W6800, Swift 1.5 IQ4_XS, chunks of 4,096 tokens) the plan starts
within the first long prompt after the engine starts: the 3090 plans from token 4,096 and the 3080 Ti from token 8,192.
The 3050 has 3 layers, so it needs a second long prompt, but the plan gives it none of the AMD card's experts anyway:
its own layers already take longer than the AMD card's. A 20,053-token prompt, the second after a start (the first reads
its experts from the drive), read at 2,198 and 2,211 tok/s, against 1,769 and 1,804 with the plan of the build before,
which needed about two long prompts to start (two engine starts each, the 3090 and 3080 Ti capped at 250 W).

Once the plan acts, answers change slightly. An expert the NVIDIA card takes is rounded differently than on the AMD
card. A fresh engine plans from its second or third chunk on and a warm one from the first, so the same prompt can get
a slightly different answer after a restart. Measured on the test box against the reference answers of 30 prompts: mean
KL 0.0024 (0.0073 on the 10 long ones), the same top token on all 30. For answers that repeat from engine to engine, set
`STRATA_PF_PEER_HOST=0`: the NVIDIA cards then read every prompt without the AMD card, which still helps decode. On the
test box two builds gave the same answers that way, with `--adapt-swaps 0 --pcie-frac 0 --prompt-cache 0`. It costs
prompt speed: with Swift 1.5 IQ4_XS a 20,053-token prompt read at 1,368 and 1,371 tok/s, against 2,175 and 2,331 with
the AMD card on prompts (one start each, one after the other in the same session, the 3090 and 3080 Ti at 250 W).
Decode after that prompt stayed the same: 90.4 and 89.1 tok/s, against 88.9 and 90.0.

After each request the log has one line on it, `strata prefill planner:`. It says how many chunks were planned and at
which token each NVIDIA card started to plan, what share of the AMD card's rows the NVIDIA cards took (in total and per
card), how many rows of RAM experts each NVIDIA card gave the AMD card to stream, the time a layer's experts took on each
NVIDIA card and the AMD card's time on that card's layers, each against the time the measured speeds predict, how long
after each call the AMD card started it, and the
speeds themselves for the way most of the request's chunks were read. For the AMD card these are its time per 1,000
rows, per expert and per call, and the speed of its input and of the experts it streams. With
`STRATA_PREFILL_TIMING=1` each NVIDIA card also logs one line per chunk with its plan.

**The AMD card streams experts from RAM.** In a chunk of 1,024 tokens or more, the plan can also give the AMD card
experts that no card holds. Without it the NVIDIA card copies them from RAM over its own link. The AMD card copies them
over its own link instead, beside the work on the experts before them, and the NVIDIA card has fewer to copy. The plan
moves the busiest of them while that makes the layer end sooner. It uses the AMD card's measured time per expert and per
byte it streams, and the time from a call until the AMD card starts it. That last time is measured on every call: it
holds the other cards' calls ahead and the handoff of the layer's input, so the plan does not count that handoff a second
time. The experts are first copied by the engine's threads into a staging ring in RAM that both cards read, those of a
pinned RAM copy (`--resident-budget-gib`) too. Each NVIDIA card has its own ring, at most a 96th of the RAM available,
for two layers. A layer streams at most half of it. The AMD card keeps room in its VRAM for a layer's copies of every
NVIDIA card. The server opens that room at load, before the AMD card's experts fill, when the experts no card holds do
not all fit on the card and at least 16 copies fit beside its prompt buffers and `--peer-reserve-mib`. Then each NVIDIA
card takes its ring when serving starts, so the first chunk may stream. Without that room a card takes its ring only
when its plan first would give the AMD card an expert from RAM, so a card that never streams pins no RAM for it. That
chunk streams nothing to the AMD card, and the chunks after the rings are ready do. The card's room is then made from
what its experts left, if 256 MiB stay free after it, and at least 16 copies. A layer's copies are a quarter of the
model's experts in a layer (128 for Swift 1.5), the most any card's layer streams, so the room does not depend on how
much RAM the first card to open found free. Each NVIDIA card gets a part of that room by
its share of the RAM experts at load (the experts of its layers that neither it nor the AMD card holds), a layer's
copies at most and one copy at least, so the card with the most to stream gets the most. What a part capped at a
layer's copies leaves goes to the others. A card that holds all of its experts keeps one copy and no share, and the
room counts one copy for it, not a layer's. A card's second call of a layer finds the copies its first call made. The
part goes by the card's place in the layer split, not by which card streams first, so every start splits the room the
same way. A card's part is known before it pins its ring, so a card that gets no part pins nothing. The ring is pinned
in pieces of about 32 MiB, outside the AMD library's locks, so the prompt running beside it does not wait for it. On
the test box an open took 0.45 to 0.57 s and the NVIDIA cards' calls into the AMD library waited at most 0.32 ms
meanwhile. Pinned in one piece, it took 1.4 to 5.9 s and those calls waited 31 to 35 ms each. The log says what each
card got, in this form:

```text
strata prefill: CUDA device 2 may stream experts from RAM to the AMD tier: 98 of them a layer through a staging ring of 497 MiB, 128 at a time on the card, its slots 256 to 383 of the 384 its ring holds for 3 stages, by its 13435 of their 16343 RAM experts (opened in 342 ms)
```

When a card's ring or its part of the room in VRAM cannot be had, the log says for that card `streams no experts from
RAM to the AMD tier` and why, and its prompts run as before. On the test box with Swift 1.5 IQ4_XS and chunks of 4,096
tokens the W6800 kept room for 34 to 36 copies. When the first card to stream took them all, the 3090 streamed 242,000
to 272,000 rows of a prompt through them. With equal parts of 12 it streamed 196,000 to 243,000, its own experts took
0.4 to 1.7 ms more a layer, and the prompt read 2.3% and 3.9% slower (two starts each). By their RAM experts (88, 531
and 1,525 of 2,144) the 3050, 3080 Ti and 3090 get 1, 9 and 26, and the prompt read as fast as with the 3090 taking all
36: 2,239 and 2,301 tok/s against 2,233 and 2,259 (the second measured prompt), 2,272 and 2,209 against 2,249 and 2,269
(the first). The 3090 streamed 242,000 to 246,000 rows of a prompt, the 3050 47,000. The 3080 Ti streamed none in those
starts: its plan moved tier-held experts to itself instead and never opened its rings. By the share of the layers
(2, 8 and 26) it read as fast too. With IQ3_S and a third of the experts on the cards it kept 384, 128 for each card.
With `--prefill auto` the chunk was 6,144 tokens, the prompt buffers took that room, and the log said
`the AMD card has 268 MiB of VRAM free, too little for a ring of its copies`.

What it does on the test box (3050, 3080 Ti and 3090 with the W6800, chunks of 4,096 tokens, the 3090 and 3080 Ti capped at
250 W, two engine starts per build in the order A B B A, the build before against this one):
- Swift 1.5 IQ3_S with a third of its experts on the cards (the `--expert-gib` caps of the setup above): the NVIDIA cards
  gave the W6800 6.4 to 6.5 million rows of RAM experts per 20,053-token prompt. On the second measured prompt a layer's
  expert work on the 3050, 3080 Ti and 3090 fell from 70.1 to 73.5, 37.2 to 42.9 and 34.1 to 36.8 ms to 36.3 to 42.2,
  29.9 to 31.5 and 30.7 to 31.0 ms. The W6800 took 21.0 to 27.1 ms of each layer, and started it 4.7 to 7.0 ms after the
  call. The prompt read at 1,526 and 1,532 tok/s (the first and the second measured prompt after a start) against 1,490
  and 1,491: 2.4% and 2.7% faster, but the build before differed by 3.5% and 3.8% between its own two starts, so that is
  not a result yet. The 3090 still waits for the W6800 at the end of its layers. It waited 258 ms of each chunk (median
  of 12) before the plan counted when the W6800 starts a layer. With that counted, and with a pinned RAM copy, it waited
  118 ms (median of 8).
- Swift 1.5 IQ4_XS: the 3050's layers stream 376,000 and 474,000 rows a prompt to the W6800, and its expert work fell from
  59.4 and 56.6 ms a layer to 36.4 and 31.4 ms. The 3090 bounds the prompt, so it read at 2,192 and 2,216 tok/s against
  2,181 and 2,201 (one start each).
- Answers change slightly, as when the plan moves the AMD card's experts: against the reference answers of 30 prompts,
  mean KL 0.0049 (0.0148 on the 10 long ones), the same top token on all 30. This was measured before the plan counted
  when the W6800 starts a layer.

**Two token halves.** A chunk of 2,048 tokens or more that hands rows to the AMD card is read as two token halves, so
the AMD card works on the first half while the NVIDIA card reads the second. That starts once the plan acts for that
NVIDIA card. Until then, in the first chunks after a start, chunks are read whole as before. Each layer runs:
1. The first half's attention, its MoE input and its router. Its rows go to the AMD card at once.
2. The second half's attention, which reads the first half's keys, its MoE input and its router. Its rows go next.
3. The shared expert, and the NVIDIA card's own experts of both halves in one pass.
4. Each half's sum of its expert rows, once the AMD card's flag says that half's call is through.

The plan prices the two calls of a layer one after the other. Each call pays the AMD card's fixed cost and its cost per
expert, and the second call starts after the first. The plan knows how long before the NVIDIA card's own experts the
first call goes out: it measures that per layer (14 ms on the 3090's layers in the setup below). An expert from RAM is
copied to the AMD card once: the second call reads the copies the first one made. So a chunk in halves streams no more
experts a layer than the card keeps for that NVIDIA card.

Each half's input goes to the AMD card as soon as its router is done. The second half's call goes out once the NVIDIA
card's own work on the experts is queued, so that card does not wait while the engine talks to the AMD card's library.
The plan starts the second call once the first is through and once its own start has passed: the time from a second
half's call until the AMD card reads its input, measured on those calls alone. Other cards' calls are often queued ahead
of it. On the test box with IQ3_S it was 3.2 to 3.5 ms on the 3090's layers and 18 ms on the 3080 Ti's, against 2.0
and 8 ms for every call of those cards. The first call still starts about 2 ms later than planned on the 3090's layers
with IQ3_S. A plan that also counted the host's time from the first half's routing to its call (0.3 to 0.5 ms there)
read no faster (1,774 and 1,743 against 1,761 and 1,762 tok/s), so the plan leaves it out. The rest of the 2 ms is
likely the library's lock, which another card's call can hold (up to 2 ms a call), and the host reaching its wait
for the routing late.

The AMD card's library keeps four streams, one for each of the HIP runtime's four hardware queues: decode, refill, the
spin that keeps the clock up, and the prompt. A prompt call copies the experts it streams from RAM on the refill stream
(no refill runs while a prompt is read) and puts its timing marks on the decode stream. A call's start counts from the
earlier of that mark and the prompt stream's own mark, both queued with the call: the card can run the decode stream's
mark late. Once, in the prompt parity test, a start that reads about 1.1 ms read -0.010 ms. Before, the copies had a
stream of their own that shared the prompt stream's queue: a call's wait for its input then lasted as long as its copies
(5.2 to 6.1 ms a call with IQ3_S), and the experts it holds could not run beside the copies. A small test program on the
W6800 showed it: the wait matched the copies (2.67 ms against 2.66 for 64 MB) and fell to 0.08 ms with the copies on a
queue of their own.

With `STRATA_PREFILL_TIMING=1` each NVIDIA card also logs one `strata prefill halves:` line per chunk in halves. In ms
from the start of its own experts, means over the chunk's layers, it gives planned against measured: when each half's
call ran on the AMD card, when the card's own experts ended, how long before them the first half was routed, and how
long the card waited for each half's flag.

What it does on the test box (the setup above, chunks of 4,096 tokens, the same build with and without halves, two
starts each in the order A B B A):
- Swift 1.5 IQ4_XS: the 20,053-token prompt read at 2,241 and 2,266 tok/s (the second measured prompt after a start)
  against 2,188 and 2,129 without halves, 4.4% faster. In an earlier run of the build before the last two changes it
  read 2,218 and 2,264 against 2,169 and 2,212 (2.3%). The first start of each kind in the later run also logged its
  plan and its waits. With halves the NVIDIA cards computed 42% of the W6800's experts' rows, against 90% without
  (the earlier run's starts gave the same shares). The 3090's own expert work took 6.7 ms a layer, against 9.5. It
  still waited for the W6800 145 ms of each chunk, against 119.
- Swift 1.5 IQ3_S with a third of its experts on the cards: 1,584 tok/s against 1,563 (one start each), and 1,527 and
  1,573 against 1,516 and 1,581 in the earlier run: the same. In the later run's two starts the 3090's own expert work
  took 19.2 ms a layer with halves against 27.9 without, but its wait for the W6800 was 457 ms of each chunk against
  136. Without the RAM experts the W6800 streams (`STRATA_TEST_XTIER_RING_FAIL=1`) the earlier run read at 1,453 and
  1,444 tok/s, so streaming to the W6800 is 7.1% faster (against that run's starts with halves).
- Answers change slightly: against the reference answers of 30 prompts, mean KL 0.0067 (0.0200 on the 10 long ones), the
  same top token on 29 of 30. Each half is routed alone. On the 3050 the router's product over half the rows rounds
  differently from one over the chunk, at every chunk size measured. On the 3080 Ti it does so only at some sizes: at
  3,000 tokens 1,521,022 of 1,536,000 logits differ, at 4,096 and 2,048 none do. Chunks read whole give the same
  answers as before.

With the library's four queues, the second call's own start and the parts by layers, against the build before them
(the setup above, two starts each in the order A B B A):
- Swift 1.5 IQ3_S with a third of its experts on the cards: 1,766 and 1,705 tok/s against 1,545 and 1,579 on the second
  measured prompt (11.1% faster), 1,720 and 1,609 against 1,484 and 1,543 on the first (10.0%). Every start read faster
  than every start of the build before. The 3090 waited for the W6800 161 ms of each chunk, against 383, and the W6800
  waited 0.71 ms a call for its input, against 4.92. This ran before the parts by layers, which give every card 128
  copies there, as the equal parts did.
- Swift 1.5 IQ4_XS: 2,246 and 2,277 tok/s against 2,244 and 2,281 on the second measured prompt, 2,218 and 2,243 against
  2,206 and 2,267 on the first: the same.
- Answers change slightly: against the reference answers of 30 prompts, mean KL 0.0039 (0.0116 on the 10 long ones),
  the same top token on all 30. This was measured before the parts by layers.

**The slowest NVIDIA card goes first.** The cards of a layer split read a prompt as a pipeline, so the slowest card
sets the prompt's speed and the others wait for it. Each card's plan balances only its own layers: it gives the AMD
card work whenever that ends its layer sooner, also on a card that waits for the slowest one anyway. Those calls then
sit ahead of the slowest card's calls on the AMD card. Before this change, on the test box, the 3090 waited for the
W6800 69 ms of each 4,096-token chunk while the other cards worked and 17 ms on each prompt's last chunk, which it
reads after they are done (Swift 1.5 IQ4_XS, 73 and 19 ms with IQ3_S). So every card now measures its chunks of the
prompt's full chunk size: the time on its GPU from its rows' arrival to its end, per token, each chunk counting a
quarter. A prompt's shorter last chunk and a prompt of one chunk do not count, as a chunk's fixed costs would weigh
more per token. A faster card has half of what it is ahead of the slowest one as slack, spread over its layers. The
gap is taken low: the slowest card's time is the lower of its mean and its last chunk, and the card's own the higher
of its two. So the first chunks after a start, which read the drive and take two or three times as long, do not give
the faster cards slack for many chunks after, and a card counts as measured only from its second chunk. With slack its
plan gives the card back work from the AMD card, the experts that save the AMD card the most time for the time they
cost the card first, in both token halves at once, as long as its layer still ends within its balanced plan's end plus
the slack. Half and not all, as what a card takes off the AMD card makes the slowest card faster too: with all of it
the faster card ended after the slowest one in the planner's test. A card has no slack while any card is not measured
yet, and none on a chunk the slowest card waits for: a card d places before the slowest one gets none on a prompt's
first d chunks, a card d places after it none on the last d. The slowest card plans as before, and so does a split
with one NVIDIA card.

With `STRATA_PREFILL_TIMING=1` each card's plan line ends with its slack and the gap it comes from, and its line
`strata prefill timing (AMD tier)` gives the slack of its last chunk:

```text
its slack 95.7 ms a layer, the slowest stage's chunk 2313 ms (mean 3599, last 2313) against its own 782 (mean 782, last 702), chunk 4 of the run, 1 after
```

What it does on the test box (the setup above, two starts each in the order A B B A, against the build before):
- Swift 1.5 IQ3_S with a third of its experts on the cards: 1,813 and 1,838 tok/s against 1,750 and 1,774 on the
  second measured prompt (3.6% faster, every start faster than every start of the build before), 1,794 and 1,656
  against 1,565 and 1,713 on the first (5.3%, the starts mixed). In the chunks it had slack in, the 3080 Ti took back
  every expert its plan had given the W6800 and the 3050 streamed it none. The W6800's time on the 3080 Ti's calls
  fell from 221 to 114 ms a chunk (its first chunk of a prompt has no slack and still streams). While the other cards
  worked, the 3090 waited for the W6800 20 ms of each chunk against 72, and its chunk took 1,836 ms against 1,927. On
  the first prompt after a start, the 3090's first chunk took 4,042 ms and its next 2,326, so the 3080 Ti's slack came
  from 2,313 ms (96 ms a layer) and not from the mean of 3,599.
- Swift 1.5 IQ4_XS: 2,285 and 2,334 tok/s against 2,213 and 2,234 on the second measured prompt (3.9% faster, every
  start faster), 2,217 and 2,335 against 2,239 and 2,255 on the first (1.3%, the starts mixed). The 3080 Ti computed
  the tier-held experts its plan saw rows for itself, so the W6800's time on its calls fell from 173 to 59 ms a chunk,
  and the 3090 waited 21 ms of each chunk against 78. The 3080 Ti set up its ring of copies in all four starts of both
  builds, as the question whether to set it up asks the plan without slack.
- Answers change slightly, as the faster cards now compute more of the experts themselves: against the reference
  answers of 30 prompts, mean KL 0.0035 (0.0103 on the 10 long ones), the same top token on all 30. A run before the
  gap was taken low read 0.0086 (0.0258), the same top token on 28 of 30.

Leave the HIP runtime's `GPU_MAX_HW_QUEUES` at its default. With 8 the W6800's prompt calls took 619 ms each and the
engine stopped waiting after 30 s, on both starts tried. Two more queues for the library's streams, by giving them
other priorities, slowed prompts about threefold the same way (567 against 1,581 tok/s with IQ3_S, one start each).

**Swift 1.5 Q6_K (157 GB).** The Q6_K pack runs with the AMD card too. Its experts are Q6_K (gate and up) and Q8_0
(down). The prompt kernels for K-quant experts (MMQ) are on by themselves for Q6_K experts: `STRATA_MMQ_KQ=0` turns them
off, and Q4_K, Q5_K and Q5_1 experts still need `STRATA_MMQ_KQ=1`. Without them the AMD card takes no prompt rows: each
stage logs that it "reads its prompts without the AMD tier". With them, against `STRATA_MMQ_KQ=0`, on 20 prompts of 256
to 1,000 tokens: mean KL 0.0134, median 0.0007, the same top token on all 20. On the test box (62 GB of RAM, chunks of
2,048 tokens) most of the 157 GB stays on the drives: an 8,450-token prompt read 120 to 148 GB from the drive and ran at
133 to 198 tok/s (six starts of the new build), and the answers after it at 18 to 30 tok/s (18.3 to 27.5 against the
build before, 23.4 to 30.4 in the streaming pair). The drive sets the speed there: streaming RAM experts to the AMD card
made no difference (152.5 against 153.3 tok/s, two starts each). The faster cards' slack (above) acts here too. Against
the build before it, the 3080 Ti took back much of the W6800's work on its layers (the W6800's time there 4.4 to 9.0 ms
a layer against 11.8 to 13.9), and the drive read about as much per prompt (120 to 137 GB against 114 to 136). The speed
followed the drive reads: 193.5 and 198.3 tok/s in one start and 132.8 and 137.4 in the other, against 138.5 to 158.0
for the build before (two starts each).

**Swift 1.5 Q6_K on one RTX 3090 and the W6800.** This is how the test box runs today. It has one NVIDIA card, the AMD
card and enough RAM for a pinned copy of every expert.

- **Settings.** Use the defaults and add `--adapt-every 8`. The box's `"args"`:

  ```
  --pack <Strata-data>/packs/swift-q6_k --native <models>/ukisai_Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00001-of-00005.gguf
  --ple-gguf <models>/ukisai_Swift-1.5-Qwen3.8-Flash-Next-Q6_K-00002-of-00005.gguf --expert-profile <profile>
  --mmap-experts --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <Strata-data>/mtp/rt
  --max-context 262144 --kv int8 --pool-workers 15 --conversation-cache-mib 0 --peer-device hip:0 --adapt-every 8
  ```

- **What the start prints.** The pinned RAM copy is on by default. The prompt chunk is `auto: 20480 tokens`. The
  start says `PCIe share beside the pinned copy: 0.30`. The W6800 writes its decode rows straight into the rows the
  3090 reads, and replays a graph per window shape. The server was ready 45.5 and 49.3 s after it started.
- **Before and after.** Measured on the test box: an RTX 3090 at 350 W, locked at 1,500 MHz with its memory at 9,751
  MHz, and a Radeon PRO W6800 at its top clock with GFXOFF off. The CPU is an EPYC 7F52 with 188 GiB of RAM on 6 of its
  8 DIMM slots. Phase 2's starting point (`ad9157b4`, chunks of 4,096 tokens, no pinned copy) against these settings
  (`136774fa`), two starts each in the order A B B A, on 2026-10-27 (UTC):

  | | Before | After | |
  |---|---|---|---|
  | 20K-token prompt | 595.6 tok/s | 2,913.2 tok/s | 4.89x |
  | Answer after the 20K prompt | 30.4 tok/s | 48.9 tok/s | +61% |
  | Chats | 25.8 / 26.2 / 23.7 tok/s | 37.0 / 38.2 / 34.3 tok/s | |
  | Ready after the start | 33 to 35 s | 45 to 49 s | |
  | The 3090's lowest free VRAM | 295 MiB | 339 MiB | |

- **Accuracy.** Prompts against a reference run on the same box: mean KL 0.0276 and 0.0278 over all prompts (two
  starts), 0.0238 on the short set. Decode at the 0.30 share, against the same build at share 0: mean KL 0.0307, the
  same top token 94.65% of the time. That difference is rounding: at 0.30 some of the CPU's missed experts are computed
  on the 3090 instead, and the 3090 rounds them differently.
- **After the upstream 0.2.7 merge.** The merged build (`f42cbaab`) against `136774fa`, both with `--adapt-every 8`,
  two starts each in the order A B B A, on the same box on 2026-10-27 (UTC):
  - The 20K-token prompt read 2,962 tok/s against 2,954, the same within the spread between starts.
  - The answer after it read 53.1 tok/s against 46.5. That is not resolved: the starts before the merge spread by 10.4.
    The chats read 39.0 tok/s on average against 38.2 (one of the three slower, two faster).
  - Prompt KL: 0.0234 and 0.0232 over all prompts, 0.0149 on the short set, lower than before. One start read 0.0367 on
    the 7 longest prompts. Decode KL at the 0.30 share: 0.0308, the same top token 94.73% of the time.
  - The chats' answers are the same words as before the merge. The 20K answer takes a different word at a near tie
    about 50 tokens in, from upstream's rounding changes. The merged build repeats its own answers exactly.
  - Upstream's new default puts the 50.7 GiB n-gram table in RAM on this box (`STRATA_PLE_AUTO_RAM`). That takes about
    4 s at the start, so the server was ready after 54.0 and 53.8 s. The RAM left free while it answered fell from about
    76 GB to about 24 GB. `STRATA_PLE_AUTO_RAM=0`, or an explicit `--ple-io`, keeps the direct read.
  - The INT8 prompt attention (`STRATA_PROMPT_ATTN_IMMA`) is now on for the 3090. It prints no start line.
- **What would help next.** The RAM runs on 6 of its 8 DIMM slots (52 to 68 GB/s in a copy test). The CPU's experts
  are the largest part of each decode window (21.6 of 49.1 ms in a timing start of the 20K answer, an earlier build at
  share 0.54), so the two missing DIMMs are the next decode gain to measure.

**Why the AMD card's clock matters.** The AMD card's work comes in short bursts. Between the bursts the W6800 drops
to its lowest clock, 500 MHz of 2,555. So the library runs a short spin on the card while the engine decodes or reads
a prompt, which keeps the clock up. The spin stops at most 20 ms after the last bit of work. Measured on the same box
with no power caps and no clock locks, before the AMD card read prompts (3 passes each):
- Chat answers: 65.2 tok/s without the spin, 100.1 with it.
- Power during those requests: the W6800 drew 26.5 W on average without the spin and 35 W with it. Energy per token
  was not measured.
- A minute after the last request the W6800 was back at 0 MHz and 7.0 W, the same as before the start.

What the other defaults are worth, on the same box:
- The AMD card out of prompts (`STRATA_XTIER_PREFILL=0`, a switch that later builds no longer have): 4K prompt 500.0
  tok/s and 16K prompt 588.4, against 1,008.3 and 1,452.4 with it on. Same caps and clock locks as the table above, 3
  passes each.
- `STRATA_XTIER_OWNERSHIP=move`: chat answers 40.0 tok/s against 66.6 for `disjoint`. No power caps, before the spin
  was added, 3 passes each.
- The route tail skip beside the AMD card (a switch that later builds no longer have, as the engine never skips an
  expert): chat answers 93.1 tok/s (92.3 to 93.9) against 88.8 (82.8 to 92.1) without it. Same caps
  and clock locks as the table above, one server start each, 3 passes. The two ranges nearly touch (92.1 and 92.3).
  Without it, one chat ran at 65.6 tok/s in one pass and at 86.5 and 85.1 in the other two. With it, that chat
  accepted 64 to 65 of 136 to 138 drafts per pass, against 70 to 73 of 126 to 128 without it. Prompts: 4K 1,007.2
  against 1,009.6, 16K 1,463.9 against 1,469.4, the ranges overlapping. It changed the answers slightly.

**Power.** The AMD card adds its own power draw to the NVIDIA cards'. The test box's power supply switched off once,
with all four cards reading a prompt at the same time and no power caps. It needed a power cycle at the wall. Before
you run this, check that the power supply has room for every card at full load plus the CPU. A power cap lowers what
an NVIDIA card may draw: `sudo nvidia-smi -i N -pl W` sets one until the next reboot. The numbers above ran with the
3090 and the 3080 Ti capped at 250 W. Every run on the box since the trip had those caps, and none tripped it.

**Open paths.** These were measured or read, and are not done yet:
- **The first half's call starts late.** On the 3090's layers with IQ3_S, the first half's call reaches the AMD card
  about 2 ms after its plan. The host's own time to the call is 0.3 to 0.5 ms, and a plan that counted it read no faster
  (above). The rest is likely the library's lock, which another card's call can hold for up to 2 ms, and the host
  reaching its wait for the routing late. A clock on the host sees neither. To take it, measure the call's start on the
  card's timeline, and let the library put its lock wait in each call's record. That changes the record, so the
  library's interface would go to version 7.
- **The second half's start holds the first half.** The AMD card sees the second half's input only after the first
  half's call has left its prompt stream. So the second half's measured start carries the first half's time in the last
  chunks, and the plan counts the first half twice: as its end and inside that start. When a chunk's first half is
  lighter than the last ones, the second half is planned later than it starts and gets less of the AMD card. On the box
  the second half ended between 1.2 ms before and 2.4 ms after its plan (the 3090's chunks with Swift 1.5 IQ4_XS, two
  starts, the timing lines). Timing the second half's start from the first half's end would keep the two
  apart. This is read from the code.
- **The slack's take.** A faster card takes back the AMD card's work up to its whole slack, also past the point where
  the slowest card stops waiting for the AMD card. Past that point the take can only move the answers further (an expert
  a card takes back rounds differently) and add RAM traffic, with no speed to gain. This is read from the plan, not
  measured. A take bounded by the slowest card's measured waits for the AMD card would stop there. It was not needed so
  far: with the current slack the KL on the 10 long prompts stayed at 0.0103 to 0.0163 (the limit is 0.03), and with
  Q6_K the new build was not slower than the build before the slack.
- **`GPU_MAX_HW_QUEUES=8`.** With 8 hardware queues the W6800's prompt calls took 619 ms each, and the engine stopped
  after its 30 s wait, on both starts tried. Whether that is a missing dependency between the library's streams or the
  queues' contention was not found. The default of 4 runs clean.
- **The decode round trip.** The host's time from a decode call to its rows is 111 us for 10 entries and 148 us for 20
  (IQ4_XS, median of 500 calls; `STRATA_TEST_XTIER_ROUND_TRIP=1` prints it). One copy of the plan and the input rows
  made that 106 and 143 us, with the same answers. Decode was not faster beyond the spread between engine starts (0.7%
  and 1.7% for NORMAL's chats, 14% in RAMHEAVY, whose decode waits on the drive), so it was not kept. The change is
  `~/bench-mixed/t14-c1.patch` on the test box. The kernels on mapped host memory took 299 to 713 us, and the quantize
  fused into the gate/up kernel 155 and 236 us, so neither was kept. The quantize itself costs 3 to 4.5 us. Most of what
  is left is the four kernels, about 0.12 of 0.16 ms on the card.
- **Stalls in the AMD card's prompt calls.** Measured with Swift 1.5 IQ4_XS (the 3050, 3080 Ti and 3090 with the W6800,
  chunks of 4,096 tokens, a 20,053-token prompt, timing lines on) in two rounds of throwaway builds that log every call.
  The box has 16 cores without SMT: the host thread on CPU 0, 15 pool workers asleep while a prompt is read, and the
  prompt threads on CPUs 1 to 15, among them 32 copy threads per NVIDIA card for its own experts and, when it streams,
  32 more for the AMD card's. The builds' changes are `~/bench-mixed/t15b-d15c.patch`, `t15b-d15c2.patch` and
  `t15b-d15c3.patch` on the test box. Three things showed up:
  1. **The queuing thread is preempted.** With and without streaming, the 3090's prompt thread sometimes spends 5 to 53
     ms inside a call to the AMD card's library, against a median of 0.4 to 0.6 ms. 79 to 83% of that time goes to
     queuing the call's kernels, not its copies, and most of those calls saw involuntary context switches. The AMD card
     waits for the kernels meanwhile. Over two prompts those calls ran 172 to 239 ms over their predicted time (about
     1% to 1.4% of the prompt time), with streaming and without. With streaming, the AMD card's copy threads ran on
     the preempted thread's CPU during 13 of 29 such calls, against 1 of 290 calls under 1 ms. Without streaming it is
     not known which threads took the CPU (the NVIDIA cards' own copy threads were not logged). Since then the AMD
     card's copy threads run 5 nice levels below the thread that starts them (Linux only; see part 2 for what it
     changed). Not measured: queuing the AMD card's call from a thread of its own, off the card's prompt loop; whether
     threads that spin while they wait for events are what preempts it.
  2. **Streaming's own cost.** Streaming RAM experts to the AMD card made the prompt 1.8% to 2.9% slower: 2,250 and
     2,254 tok/s against 2,320 and 2,318 with `STRATA_TEST_XTIER_RING_FAIL=1` (two starts each), and 2,287 and 2,278
     against 2,325 in a second round. Every start without it was faster than every start with it. The plan prices a
     streamed call right on average: the 3090's calls ran within 0.2 ms of their predicted time in the median, so a
     price alone does not fix it. What streaming adds sits on the 3090's thread: 737 and 809 ms in the AMD card's calls
     over two prompts against 608, and in 4 to 6 of the slow calls of a start the staged experts landed 2.5 to 15 ms
     after the call (the copy threads never waited for ring slots there). Holding the AMD card's clock at its highest
     level did not change this (2.7% slower with streaming). With a third of IQ3_S on the cards streaming made the
     prompt 18.6% faster (1,753 and 1,775 against 1,452 and 1,522), and with Q6_K, where the drive sets the speed, it
     changed nothing.
     With the copy threads 5 nice levels down (Swift 1.5 IQ4_XS on the earlier box with three NVIDIA cards and the
     W6800, a test build, three starts each with two measured passes a start) the prompt read at 2,308 to 2,344 tok/s, 2,329 on average, against 2,206 to 2,340 and 2,292 for
     the build as it was, and 2,293 to 2,360 with streaming off (two starts in two rounds). Two of the build's six passes
     were slow (2,222 and 2,206), none with the lower priority, and the slow passes did not read more from the drive
     (713 and 339 MB, against 67 to 690). So streaming in this setup now costs no more than the spread between starts.
     With a third of IQ3_S on the cards the prompt read at 1,750 to 1,819 against 1,708 to 1,815 (two starts each), so
     its gain from streaming stays. Chats and decode did not change beyond the spread. Five levels is the mildest
     setting with more than one start behind it: two levels ran one start (2,298 and 2,335), and `SCHED_BATCH` one
     start (2,278 and 2,244). It is Linux only. On Windows and macOS the copy threads keep their priority.
  3. **The first call of each chunk.** The 3090's first call of a chunk (its first layer, the first token half) runs
     8.7 to 9.0 ms over its predicted time on average, with and without streaming, while its other calls run 0.1 to 0.2
     ms under. That is about 0.75% of the 3090's chunk (1,166 to 1,201 ms). Its host time is 1.9 ms in the median
     without streaming, so the host does not explain it there. It is not explained.
- **`--prefill auto` and the chunk size.** With Swift 1.5 IQ4_XS on the test box the start-up plan picks chunks of 6,144
  tokens: the largest every NVIDIA card can lend with a full ring, under the 8,192 ceiling, and its files fit the page
  cache. Their buffers leave the W6800 252 MiB of VRAM, too little for its ring of copies, so it streams nothing. That
  is not what makes it slower: without streaming on either side, chunks of 6,144 read a 20,053-token prompt at 2,079 to
  2,202 tok/s (two starts) against 2,348 and 2,360 for chunks of 4,096 (one start), and NORMAL reads faster without
  streaming anyway. The chunk size itself costs. With the timing lines, at 6,144 the 3090's own GPU time over the prompt
  was smaller (5,626 against 5,815 ms), but it sat idle 3.6 s of 9.25 s against 2.8 s of 8.60: the first chunk takes
  longer to reach it through the other cards (the 3080 Ti's chunk 658 to 885 ms against 446 to 639). The start-up plan
  picks once, before any stage is measured, so it cannot weigh that. `ring_room` (generate.cpp) prices every card by its
  lend budget, also a card that holds its own buffers; every card lends here, so it does not change this pick.

**Limits.**
- One AMD card: `hip:N` names one.
- Native packs only (`--native`). The start stops with a plain message for a canonical pack, an NVFP4 pack or a pack
  with folded Hadamard rotations (APR).
- Linux only. It has not been built or run on Windows.
- The NVIDIA (CUDA) engine only. The AMD engine and the Mac engine refuse `hip:N` at start.
- If the AMD card stops answering while the engine reads a prompt or writes an answer, the engine waits
  `STRATA_XTIER_WAIT_MS` (30 s by default), logs `the AMD tier did not answer within STRATA_XTIER_WAIT_MS` and ends.
  The line says which wait gave up (the AMD card's wait for a layer's input or for its staged experts, or the NVIDIA
  card's wait for the AMD card's flag), how far each side got, the NVIDIA card and the layer.
  As it ends, it does not wait for the AMD card again. Restart the server, without `--peer-device hip:N` if the card
  still does not answer. It was not tried with a hung card. With one lost answer from the card, in a prompt that
  streamed experts to it from RAM, the request ended with that line 46 s after the prompt started, and nothing hung.
  In a chunk read as two token halves the second half's wait ends with the first one's, so it adds no 30 s of its
  own. The test hook `STRATA_TEST_XTIER_DROP=N` now stops every answer from the Nth call on, as a card that stopped.
  With it in a prompt read in halves with experts streamed to the AMD card, the request ended 41.6 s after it started,
  with `this GPU's wait for the AMD tier's done flag gave up`, and the serve watchdog (60 s) did not fire.
- With the AMD card on, the automatic PCIe share is off,
  as with a second NVIDIA card. `--pipeline-windows`, `--kv-grow` and `STRATA_PIN_BY_NEED` stay off too.
- `STRATA_PREFILL_HELP=1` does not help a stage that gives its prompt rows to the AMD card. Every stage does, unless
  the log says the stage "reads its prompts without the AMD tier". `--peer-prefill-rows` is for a second NVIDIA card
  only. It does not change the AMD card's share.
- On an NVIDIA card of compute capability 12.0 with 80 or more SMs (an RTX 5080 or 5090, an RTX PRO 5000 or 6000),
  the fused prompt kernels are on by default for packs whose gate and up weights are IQ2_S, IQ3_XXS, IQ3_S or
  IQ4_XS. The AMD card then takes no prompt rows: the log says the stage "reads its prompts without the AMD tier".
  `STRATA_PF_FUSED=0` gives them back. This is read from the code. No such card was tried.
- A small `--peer-reserve-mib` might end the engine in the middle of a prompt, because the AMD card's prompt kernels
  take some of their memory only when they run. This is read from the code. It was not tested.
- The prompt chunk plan counts the AMD card's experts as experts read from the drive. On the test box that cut the
  chunk from 6,144 to 4,096 tokens. Even so, the chunk it picked was the fastest of the three tried (the setup of the
  table above, 3 passes each): a 16K prompt read at 1,467.8 tok/s with the plan's chunk (4 chunks of about 3,962
  tokens), 1,434.9 with `--prefill 6144`, and 1,465.2 with `--prefill 8192` (which fell back to 4,096).
- With `--prefill auto`, the start-up chunk plan adds the AMD card's own link to the NVIDIA cards' when it may stream RAM
  experts. The engine measures that link at the start: `strata generate: AMD tier PCIe probe: 25.7 GB/s host->card` on
  the test box. It adds only the part the AMD card can use: it streams at most a quarter of a layer's experts, so when
  the experts no card holds are many more than that, the NVIDIA card's copies of the rest set the time. The log's
  `prefill plan inputs` line says how much it added. The link counts even when the AMD card later has no room for its
  ring of copies, because the plan picks the chunk before the rings are set up. It changes the plan only when the
  experts read from RAM do not fit the page cache. There they fit, so the
  plan kept a chunk of 6,144 tokens, with no room on the W6800 to stream, and a 20,053-token prompt read at 2,054 and 2,062
  tok/s, against 2,178 to 2,203 with `--prefill 4096` on the same build.
- A pinned RAM copy (`--resident-budget-gib`) is not mapped into the AMD card. Mapped (hipHostRegister of a 20 GiB copy,
  Swift 1.5 IQ3_S on the test box), it made the W6800's answers slow (2.1 and 5.3 tok/s, the card 12.5 and 6.3 ms late
  each layer), and then one answer ended with `peer experts: the AMD tier did not answer within STRATA_XTIER_WAIT_MS
  (30000 ms) (layer 29)`. Without the mapping, and in the build before, the card was at most 0.064 ms late, and a 20,053
  token prompt read at 1,659 and 1,625 tok/s (one start each).
- Not measured: other AMD cards than the W6800, and model families other than Qwen3.8-Flash-Next (Swift 1.5 IQ3_S,
  IQ4_XS and Q6_K were measured).

## History / older measurements

Dated numbers behind the settings above. Each row is one machine; none is a claim about yours.

<a id="measured"></a>

**Layer split, Coder model.** Measured on: RTX 5080 + RTX 3090, Ryzen 9 9950X3D, Coder, 32K context, engine 0.1.20-era
(`bench/results/2026-09-29-layer-split/`).

| Setup | Prompt 16K / 28K tok/s | Decode story / code tok/s |
| --- | --- | --- |
| 5080 alone | 1,726-2,017 / 1,970 | 83-87 / 88-105 |
| 5080 + 3090, best split (K=26) | 2,039 / 2,357 | 84 / 110 |
| 5080 + 3090, auto (K=22) | 2,037 / 2,073 | 80 / 109 |

A third card (RTX 2080 Ti) made that pair slower: 68 / 90 tok/s decode.

**Option measurements**

| Setting | Measured on | Result |
| --- | --- | --- |
| Unequal pair, `auto` rule | 2x RTX A4000 made unequal, IQ3_XXS, 14.5K prompt, 5 interleaved pairs | prompt 945 -> 1,441 tok/s (+52%), decode 50.2 -> 65.9 (+31%) |
| Unequal pair, hand sweep of `"layer_split"` | 4060 Ti 16 GB + 3060 12 GB, IQ3_S | K=12/18/24/28/32: 582 / 693 / 762 / 802 / 803 prompt tok/s; the old rule chose K=3: 452 |
| x16 + x4 link scaling | 2x RTX 3090, x16 + x4 | K=31: 114-119 tok/s decode, K=27: 130-136 |
| `split_skip_if_fits` | R9700 32 GB + RX 9070 XT, Coder | 4K prompts 1,244 -> 1,776 tok/s, decode ~51 -> ~60; 16K prompts ~5% slower |
| `STRATA_SPLIT_OWN=1` | R9700 + RX 9070 XT | 2K prompts 1,450 and 16K 2,227 tok/s (default 1,265 and 1,950) |
| `STRATA_PREFILL_HELP=1` | 2x RTX 3090, UD-Q4_K_XL, no P2P | prompt tok/s without / with: 1.1K 496 / 723, 2K 878 / 1,128, 3K 1,260 / 1,440; 4K and 8K unchanged |
| `STRATA_PREFILL_HELP=1` | 2x R9700 (16 GB each), Q2_0, PCIe 4.0 x16 | 700 tokens 638 -> 672; 1.5K 839 -> 809; 3K 1,105 -> 929; slower from 1.5K, so it stays off |
| `STRATA_SPLIT_MTP_BATCH=1` | 2x RTX 3080 20 GB, Xeon E5-2696 v4, UD-Q4_K_XL, `--layer-split 23` | prompt read +6.7% at 25K, +4.3% at 51K, +3.3% at 104K; decode not shown unchanged (about 3% lower, within restart noise) |
| `STRATA_STAGE_TRIM=1` | 2x MI50 16 GB, Coder | experts in VRAM 8,819 -> 10,626; decode 39.2 -> 41.7 tok/s |
| Prompt chunk on a split, AMD gfx12 | 2x / 3x / 4x R9700 | 16K prompt +36% / +45% / +53%; 32K +18% / +21% / +29%; 64K +4% / +12% / +17% |
| `STRATA_SPLIT_OVERLAP=1` | 2x RX 7900 XTX, Swift 1.5 IQ3_XXS, 128K, median of 8 | code decode 53 -> 70.1 tok/s (one card ~70); prose 45 -> 55.9 (one card ~68) |
| `STRATA_SPLIT_OVERLAP=1` | 2x RTX A4000, IQ3_XXS | +0.3% story, +2.1% code, -0.3% facts, inside the noise |
| Two Tesla P40, layer split | community report, IQ2_XS (#1028), experimental | decode 19.6 tok/s on one card, 34.8 on both |

**Resident RAM mode on a split.** Measured on: RTX 4060 Ti (layers 0-19) + RTX 5080 (20-47), i9-14900KF, 32 GB RAM, Windows 11,
Swift 1.5 IQ3_XXS, 160K context, four greedy prompts at a time, decode tok/s.

| Setup | First four prompts | After three more rounds |
| --- | ---: | ---: |
| 5080 alone, `--resident-experts` | 25 | 29 |
| split, `--mmap-experts` | 32 | 64 |
| split, `--resident-experts` (22 GiB locked in RAM) | 71 | 69 |

**Prompt chunks for the pipeline (`--prefill-pipe`).** Prompt tok/s, fresh prompts, two interleaved rounds per row.

| Prompt tokens | `--prefill auto` | `--prefill 2048` | `--prefill-pipe 1024` (chunk) |
| ---: | ---: | ---: | ---: |
| 2,013 | 180.9 | 181.4 | 228.0 (1024) |
| 3,865 | 207.6 | 276.5 | 312.4 (1024) |
| 10,041 | 261.1 | 433.0 | 432.6 (2048) |
| 28,998 | 475.7 | 554.1 | 559.7 (3328) |

Measured on: four GP100 dies (PH402 SKU 200), `--layer-split 12,24,36`, Flash-Next IQ3_XXS, 64K context. On 2x RTX 3090 Ti
(Swift 1.5 IQ3_XXS, 1M context) calibration kept it off, and a fixed 1024 chunk read 19-24% slower there and 22-36% slower on
2x RTX 4090 (community report).

**Pipeline windows.** Decode tok/s.

| Measured on | Result |
| --- | --- |
| 2 cards, R9700, code, serial loop -> draft layer on the first card | 103 -> 125 (+21%); 3 cards +19%, 4 cards +22% |
| RTX 4060 Ti + RTX 5080, i9-14900KF, 32 GB, Swift 1.5 IQ3_XXS, 160K, greedy 500 tokens | mean 72.4 -> 84.0 (+16%): Python 90.4 -> 103.1, English prose 63.0 -> 71.1, copy-heavy edit 90.5 -> 117.9 |
| 4x RX 7900 XT, Flash-Next GSQ-RCO IQ3_S, 262K, `--layer-split 12,24,36` | 4K 59.4 -> 67.6; 32K 51.8 -> 65.6 |

**Several settings on one rig.** Measured on: 2x RTX 3090 (250 W, x16 + x4, no NVLink), Ryzen 7 9800X3D, 32 GB RAM, Flash-Next
GSQ-RCO IQ3_S, `--resident-experts`, int8 KV, `--spec 4 --mtp`; decode is the mean of 1,500-token coding replies, prefill one
cold 19.9K prompt; one run per arm (community report #1447, not repeated here).

| Arm | Decode tok/s | Prefill tok/s |
| --- | ---: | ---: |
| setup defaults | 137.0 | 1,798 |
| `--adapt-async 1` | 144.5 | 1,799 |
| `--pipeline-windows 2` | 136.7 | 2,097 |
| `STRATA_PF_FUSED=1` | 137.2 | 1,913 |
| pipeline-windows 2 + async + lag 2 + rotate + pf_fused | 142.0 | 2,302 |
