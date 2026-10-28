"""One engine start's log, read: the start's facts, the requests as rows, and the decode timing rows.

    python tools/engine_log.py facts ENGINE_LOG            # JSON of one start's facts
    python tools/engine_log.py rows ENGINE_LOG [--timing]  # the requests, one JSON object a line
    python tools/engine_log.py decode ENGINE_LOG           # the decode timing rows, one JSON object a line

Engine logs append across starts, so every command reads from the last "engine started" line (the whole file when it
has none).  The facts: `chunk` (the auto pick, else the start line's --prefill), `loan_slots` (the CUDA0 cache slots the
prompt path borrows), `w6800_experts` (the AMD tier's experts), `ring_slots` (the AMD card's ring of copies, 0 when the
prompt path streams it none), `tier_in_prompts` (false without an AMD tier or when a stage reads its prompts without it),
`pinned_gib` and `pinned_s` (the pinned RAM copy, 0 GiB without one), `vram_free_mib` (CUDA0's VRAM free with everything
loaded), `vram_low` (the lowest of the per-request `only N MiB of VRAM free` warnings), `cuda_slots` (CUDA0's expert
cache) and `cached_share` (the share of the GGUF experts' pages in the page cache when the stager chose its profile).
Any other fact the log does not show is null.  A row is a `strata serve: prompt` line with the lines that follow it
(the decode hit rate, the planner, the file tier's reads) and the low VRAM warning before it; `--timing` adds the timing
lines that came before it.  A decode row is a `strata decode timing` line, numbered by the request it comes before.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

START = "engine started:"
FACTS = ("chunk", "loan_slots", "w6800_experts", "ring_slots", "tier_in_prompts", "pinned_gib", "pinned_s",
         "vram_free_mib", "vram_low", "cuda_slots", "cached_share")
CHUNK_AUTO = re.compile(r"prompt chunk auto: (\d+) tokens")
CHUNK_ARG = re.compile(r"(?:^|\s)--prefill (\d+)(?=\s|$)")
LOAN = re.compile(r"the prompt path borrows (\d+) CUDA0 cache slots")
W6800 = re.compile(r"AMD tier on HIP device \d+: (\d+) experts")
RING = re.compile(r"may stream experts from RAM to the AMD tier: .* of the (\d+) its ring holds")
NO_RING = "streams no experts from RAM to the AMD tier"
TIER_IN = "the AMD tier computes its experts' rows of each prompt chunk"
TIER_OUT = "reads its prompts without the AMD tier"
PINNED = re.compile(r"cache complement ready: .*pinned ([\d.]+) GiB in (\d+) s")
VRAM_FREE = re.compile(r"^strata serve: (?:WARNING: )?(\d+) MiB of VRAM free with everything loaded")
VRAM_LOW = re.compile(r"only (\d+) MiB of VRAM free at the start of this request")
SLOTS = re.compile(r"expert cache (\d+) slots")
CACHED = re.compile(r"the GGUF experts are (\d+)% in the page cache")

SERVE = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                   r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)(?:, drafts accepted (\d+) of (\d+))?")
SERVE_KEYS = ("prompt", "reused", "read", "pms", "ptoks", "gen", "dms", "dtoks", "acc", "drafts")
HIT = re.compile(r"decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\); the AMD tier computed (\d+) "
                 r"entries \(([\d.]+)% of all (\d+) routed\)")
OWN = re.compile(r"a layer's own experts took ([\d.]+) ms on the stages \(fitted ([\d.]+)\)")
AMD_MS = re.compile(r"the AMD tier's ([\d.]+) ms on their layers")
STREAMED = re.compile(r"streamed it (\d+) rows")
PAGED = re.compile(r"read paged experts at ([\d.]+) GB/s")
FILE = re.compile(r"file tier I/O this request: the OS read ([\d.]+) MB from storage")
TIMING = ("strata prefill timing", "strata xtier", "strata prefill halves", "strata decode timing",
          "strata decode GPU stages", "strata decode pool")

N = r"([\d.]+)"
DECODE = re.compile(
    rf"strata decode timing: (\d+) windows, avg T {N}, {N} tokens/window, {N} ms/window = verify {N} \(GPU-reach wait {N} "
    rf"\+ per-layer host {N} \[plan {N} actq {N} jobs {N} CPU {N}\] \+ stage {N}\) \+ commit/emit {N} \+ draft {N}; "
    rf"per layer-window: CPU experts {N} \({N} entries\), VRAM hits {N}, PCIe {N}; adapt wait {N} join {N}, {N} swaps/window")
DECODE_KEYS = ("windows", "avg_t", "tok_window", "ms_window", "verify", "gpu_wait", "host", "plan", "actq", "jobs", "cpu",
               "stage", "commit", "draft", "cpu_experts", "cpu_entries", "vram_hits", "pcie", "adapt_wait", "join", "swaps")


def read(path) -> list[str]:
    """The log's lines from its last engine start (all of them when it has none)."""
    lines = Path(path).read_text(errors="replace").splitlines()
    starts = [i for i, x in enumerate(lines) if START in x]
    return lines[starts[-1]:] if starts else lines


def num(s: str | None):
    if s is None:
        return None
    return float(s) if "." in s else int(s)


def facts(lines: list[str]) -> dict:
    f = dict.fromkeys(FACTS)
    f["pinned_gib"] = 0.0
    auto = arg = None
    tier_in = tier_out = False
    for x in lines:
        if START in x:
            given = CHUNK_ARG.findall(x)
            arg = int(given[-1]) if given else None
        elif (m := CHUNK_AUTO.search(x)):
            auto = int(m.group(1))
        elif (m := LOAN.search(x)):
            f["loan_slots"] = int(m.group(1))
        elif (m := W6800.search(x)):
            f["w6800_experts"] = int(m.group(1))
        elif (m := RING.search(x)):
            f["ring_slots"] = int(m.group(1))
        elif NO_RING in x:
            f["ring_slots"] = 0
        elif TIER_IN in x:
            tier_in = True
        elif TIER_OUT in x:
            tier_out = True
        elif (m := PINNED.search(x)):
            f["pinned_gib"], f["pinned_s"] = float(m.group(1)), float(m.group(2))
        elif (m := VRAM_FREE.search(x)):
            f["vram_free_mib"] = int(m.group(1))
        elif (m := VRAM_LOW.search(x)):
            low = int(m.group(1))
            f["vram_low"] = low if f["vram_low"] is None else min(low, f["vram_low"])
        elif (m := SLOTS.search(x)):
            f["cuda_slots"] = int(m.group(1))
        elif (m := CACHED.search(x)):
            f["cached_share"] = int(m.group(1)) / 100
    f["chunk"] = auto if auto is not None else arg
    f["tier_in_prompts"] = tier_in and not tier_out
    return f


def rows(lines: list[str], timing: bool = False) -> list[dict]:
    reqs: list[dict] = []
    pending: list[str] = []
    low = None
    for x in lines:
        if (m := SERVE.search(x)):
            r = {k: num(v) for k, v in zip(SERVE_KEYS, m.groups())}
            r["vram_low"] = low
            if timing:
                r["timing"] = pending
            reqs.append(r)
            pending, low = [], None
        elif x.startswith(TIMING):
            pending.append(x)
        elif (m := VRAM_LOW.search(x)):
            low = int(m.group(1))
        elif reqs:
            r = reqs[-1]
            if (m := HIT.search(x)):
                r.update(hit=float(m.group(1)), hits=int(m.group(2)), lookups=int(m.group(3)), amd=int(m.group(4)),
                         amd_pct=float(m.group(5)), routed=int(m.group(6)))
            elif x.startswith("strata prefill planner") and (m := OWN.search(x)):
                r.update(own_ms=float(m.group(1)), own_fit=float(m.group(2)))
                for key, rx in (("amd_ms", AMD_MS), ("streamed_rows", STREAMED), ("paged_gbs", PAGED)):
                    if (m := rx.search(x)):
                        r[key] = num(m.group(1))
            elif (m := FILE.search(x)):
                r["file_mb"] = float(m.group(1))
    return reqs


def decode(lines: list[str]) -> list[dict]:
    out: list[dict] = []
    served = 0
    for x in lines:
        if (m := DECODE.search(x)):
            d = {"req": served, "prompt": None}
            d.update((k, num(v)) for k, v in zip(DECODE_KEYS, m.groups()))
            out.append(d)
        elif (m := SERVE.search(x)):
            for d in out:
                if d["req"] == served:
                    d["prompt"] = int(m.group(1))
            served += 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("facts", "rows", "decode"))
    ap.add_argument("log")
    ap.add_argument("--timing", action="store_true", help="rows: add the timing lines that came before each request")
    a = ap.parse_args(argv)
    lines = read(a.log)
    if a.cmd == "facts":
        print(json.dumps(facts(lines)))
    else:
        for r in rows(lines, a.timing) if a.cmd == "rows" else decode(lines):
            print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
