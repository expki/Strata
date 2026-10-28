#!/usr/bin/env python3
"""The quant quality yardstick (MODEL-UPDATE.md, Phase 0): teacher-forced top-1 agreement and KL of a quant against
Q8_0, on one fixed text set (data/quant_quality/v1: code, docs, English, other languages, an agent transcript).

    python tools/quant_quality.py ids --tokenizer DIR [--set data/quant_quality/v1] [--segments a,b] [--seg-tokens N|0]
        tokenizes the set's texts (every .txt, v1's names first, or --segments in that order; each cut to --seg-tokens,
        default 1000, 0 keeps it whole) - the ids file is committed with the set (data/quant_quality/v1/set.ids.json,
        data/quant_quality/xl/set.ids.json), so a box without the tokenizer scores the same positions
    python tools/quant_quality.py run --config CFG.json --out RUN.tsv [--ids set.ids.json] [--env K=V ...] [--gpu-experts]
                                      [--keep-caches [--allow pcie]]
        one `strata --serve` engine (the config's exe and args), every segment one request whose tokens are all read
        through the decode windows (--short-read), where STRATA_LOGPOS / STRATA_LOGPOS_TOPK=64 write, per position,
        the log-probability of the text's next token and the 64 most likely tokens.  Fixed: the experts on the CPU
        (one VRAM slot, --expert-cache 1: the verify windows need a profile-filled tier, so the config must have
        --expert-profile; --gpu-experts: 1500 slots), no expert adaptation, no PCIe share, no suffix drafts, no prompt
        cache, FP16 K/V, STRATA_PREFILL_CPU_SHARE=0.  RUN.tsv + RUN.tsv.seg.json (line ranges).
        --keep-caches reads them through the decode windows of the config's engine as the server starts it (its tier,
        caches, RAM copy, context and K/V), with only --keep-caches' fixed switches; --allow pcie keeps the config's
        --pcie-frac (and --env's) instead of 0.
    python tools/quant_quality.py compare REF.tsv RUN.tsv [...]   (REF = the Q8_0 run)
        per segment and overall: top-1 agreement (same argmax), mean / p99 KL(REF || RUN) over REF's 64 most likely
        tokens (both renormalized to that set; a token RUN does not list gets RUN's 64th log-probability), and the
        perplexity of the text under each.
    python tools/quant_quality.py prompt --config CFG --out DIR [--keep-caches] [--set v1|v2] [--no-warm] [--env K=V]
        the batched prompt path: each segment's first 256 / 512 / 768 / 1000 tokens as one prompt (short), and
        prompts of 4,600 to 4,986 tokens cut from the segments joined (long: a 4096-token chunk and a rest), the whole
        vocabulary's logits after each (DIR/first.N).  v1: 20 short, 10 long.  v2 (the default): one unscored warm
        request first (4,800 tokens of the xl docs text, so every scored prompt meets the tier state the config
        serves), the 20 short, 30 long that keep v1's 10, and 7 xl prompts of 20,000 to 24,300 tokens from
        data/quant_quality/xl (one chunk at 24576, and past 20,972 tokens where T x 10 x 2560 x 4 bytes > 2^31).
        --keep-caches starts the config's engine as the server does (serve.server's engine_args and child_env: its
        layer split, card order, context, K/V, chunk, caches and tier), with no adaptation, PCIe share, suffix drafts,
        prompt cache or CPU prompt share, and no learned profile written.
    python tools/quant_quality.py pcompare REF_DIR RUN_DIR [...] [--per-prompt]
        full-vocabulary KL(REF || RUN) and top-1 agreement of `prompt` directories: the short prompts, the long ones,
        the xl ones, all (short and long) and v1's 30 alone.  --per-prompt adds a row per prompt with the reference's
        entropy and top-1 probability (nats).
    python tools/quant_quality.py gate REF_DIR RUN_DIR [RUN_DIR2] [--xl-ref REFXL_DIR]
        gate v2 on each start of a candidate: all mean <= 0.03, median <= 0.005, top-1 >= 90%; long mean <= 0.03,
        top-1 >= 90%; xl mean <= 0.03 and no top-1 miss where the reference's top-1 probability is 0.5 or more; no
        non-finite logit and no prompt missing.  Exit 0 only if every run passes.

The RUN.tsv rows are the engine's STRATA_LOGPOS format: pos, target id, logp(target), argmax id, logp(argmax),
argmax==target, logp(<|im_end|>), logp(target without <|im_end|>), then 64 `id:logp` fields (natural log).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SET_DIR = HERE.parent / "data" / "quant_quality" / "v1"
XL_DIR = HERE.parent / "data" / "quant_quality" / "xl"
SEGMENTS = ["code", "docs", "english", "multilingual", "agent"]
TOPK = 64
FIXED = ["--expert-cache", "1", "--adapt-every", "100000", "--pcie-frac", "0", "--suffix-draft", "0",
         "--prompt-cache", "0", "--kv", "fp16"]
FIXED_ENV = {"STRATA_PREFILL_CPU_SHARE": "0", "STRATA_PCIE_FRAC_DEFAULT": "old"}
# config args the yardstick replaces (value-taking) or drops (flags)
DROP_VALUE = {"--expert-cache", "--adapt-every", "--pcie-frac", "--suffix-draft", "--prompt-cache", "--max-context",
              "--prefill", "--prompt-cache-every", "--short-read", "--conversation-cache-mib", "--eos-ids",
              "--kv", "--kv-resident"}   # (--resident-budget-gib stays: a PC whose RAM does not hold the experts)
DROP_FLAG = {"--serve", "--vision", "--resident-experts", "--resident-cpu-experts"}
# --keep-caches drops only these from the config's engine (and --serve), then adds KEEP_FIXED for a repeatable dump
KEEP_DROP_VALUE = {"--adapt-every", "--pcie-frac", "--suffix-draft", "--prompt-cache", "--short-read",
                   "--expert-profile-save", "--expert-profile-save-every"}
KEEP_FIXED = ["--adapt-every", "100000", "--pcie-frac", "0", "--suffix-draft", "0", "--prompt-cache", "0"]
# run --keep-caches --allow NAME: the config's own value of that switch stays
ALLOW = {"pcie": "--pcie-frac"}


# ------------------------------------------------------------------------------------------------ ids
def load_tokenizer(path: str):
    sys.path.insert(0, str(HERE))
    import strata_tokenizer as ST  # noqa: E402
    tp = pathlib.Path(path)
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((tp / "token_type.json").read_text()))


def set_segments(d: pathlib.Path) -> list[str]:
    """A set directory's texts: v1's names in v1's order, then any other `.txt` by name."""
    names = {p.stem for p in d.glob("*.txt")}
    return [n for n in SEGMENTS if n in names] + sorted(names - set(SEGMENTS))


def cmd_ids(a) -> int:
    tok = load_tokenizer(a.tokenizer)
    names = a.segments.split(",") if a.segments else set_segments(pathlib.Path(a.set))
    segs = []
    for name in names:
        text = (pathlib.Path(a.set) / f"{name}.txt").read_text(encoding="utf-8")
        ids = tok.encode(text, parse_special=(name == "agent"))
        if len(ids) < a.seg_tokens:
            raise SystemExit(f"{name}: only {len(ids)} tokens (< {a.seg_tokens})")
        keep = a.seg_tokens or len(ids)     # 0 keeps the whole text
        segs.append({"name": name, "ids": ids[:keep]})
        print(f"{name}: {len(ids)} tokens, kept {keep}")
    out = pathlib.Path(a.out or (pathlib.Path(a.set) / "set.ids.json"))
    out.write_text(json.dumps({"version": pathlib.Path(a.set).name, "seg_tokens": a.seg_tokens, "segments": segs}))
    print(f"wrote {out}")
    return 0


# ------------------------------------------------------------------------------------------------ run
def mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float("inf")


def strip_args(args: list[str], flags: set, values: set) -> list[str]:
    out, i = [], 0
    while i < len(args):
        if args[i] in flags:
            i += 1
        elif args[i] in values:
            i += 2
        else:
            out.append(args[i])
            i += 1
    return out


def engine_args(cfg: dict, gpu_experts, seg_tokens: int, keep_caches: bool = False, allow=(),
                short_read: int = 0) -> list[str]:
    if keep_caches:
        if gpu_experts:
            raise ValueError("gpu_experts and keep_caches exclude each other")
        kept = {ALLOW[n] for n in allow}
        fixed = [x for i in range(0, len(KEEP_FIXED), 2) if KEEP_FIXED[i] not in kept for x in KEEP_FIXED[i:i + 2]]
        read = ["--short-read", str(short_read)] if short_read > 0 else []
        return [cfg["exe"], "--serve", *strip_args(cfg["args"], {"--serve"}, KEEP_DROP_VALUE - kept), *fixed, *read]
    out = strip_args(cfg["args"], DROP_FLAG, DROP_VALUE)
    fixed = list(FIXED)
    if gpu_experts:
        fixed[1] = gpu_experts if isinstance(gpu_experts, str) else "1500"
    return [cfg["exe"], "--serve", *out, *fixed, "--max-context", str(seg_tokens + 512),
            "--short-read", str(seg_tokens + 64), "--prefill", "512"]


def config_env(cfg: dict, a, as_server: bool) -> dict:
    """The engine's environment: the server's for the config, or the config's env and libraries, then the fixed ones."""
    if as_server:
        sys.path.insert(0, str(HERE.parent))
        from serve.server import child_env
        env = child_env(cfg)               # the config as written: --layer-split in the args would keep the card order
    else:
        env = dict(os.environ)
        env.update(cfg.get("env", {}))
        if cfg.get("lib_dirs"):
            env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
    env.update(FIXED_ENV)
    if "pcie" in (getattr(a, "allow", None) or []):
        env.pop("STRATA_PCIE_FRAC_DEFAULT")   # the engine's own rule for a share the config leaves to it
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    return env


def server_config(cfg: dict) -> dict:
    """The config with the arguments the server starts its engine with."""
    sys.path.insert(0, str(HERE.parent))
    from serve.server import engine_args as server_args
    return dict(cfg, args=server_args(cfg))


def run_engine(cfg: dict, a, seg_tokens: int) -> tuple[list[str], dict]:
    """The run's engine arguments and environment, with --keep-caches as the server starts the config."""
    env = config_env(cfg, a, a.keep_caches)
    if a.keep_caches:
        return engine_args(server_config(cfg), False, 0, True, a.allow, seg_tokens + 64), env
    return engine_args(cfg, a.gpu_experts, seg_tokens), env


def cmd_run(a) -> int:
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    st = json.load(open(a.ids))
    out = pathlib.Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    args, env = run_engine(cfg, a, st["seg_tokens"])
    env["STRATA_LOGPOS"] = str(out)
    env["STRATA_LOGPOS_TOPK"] = str(TOPK)
    if mem_available_gb() < a.min_free_gb:
        raise SystemExit(f"only {mem_available_gb():.0f} GB available (< {a.min_free_gb}): not starting an engine")
    if a.wrap:
        args = a.wrap.split() + args
    errf = open(str(out) + ".stderr.txt", "w")
    print("engine:", " ".join(args), flush=True)
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                         cwd=cfg.get("cwd"), env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY (exit {p.wait()}); see {out}.stderr.txt")
    segs, info = [], []

    def lines() -> int:
        return sum(1 for _ in open(out)) if out.exists() else 0

    for s in st["segments"]:
        n0, t0 = lines(), time.time()
        p.stdin.write("GEN 1 " + ",".join(map(str, s["ids"])) + "\n")
        p.stdin.flush()
        done = None
        for line in p.stdout:
            if line.startswith(("DONE", "ERR")):
                done = line.strip()
                break
        if done is None or not done.startswith("DONE") or "nonfinite" in done:
            p.kill()
            raise SystemExit(f"{s['name']}: the engine answered {done!r}; see {out}.stderr.txt")
        n1 = lines()
        segs.append({"name": s["name"], "first_line": n0, "end_line": n1})
        info.append(f"{s['name']}: {n1 - n0} positions, {time.time() - t0:.1f} s")
        print(info[-1], flush=True)
        if n1 - n0 < len(s["ids"]) // 2:
            p.kill()
            raise SystemExit(f"{s['name']}: only {n1 - n0} positions were scored (the prompt took the batched path?)")
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    err = pathlib.Path(str(out) + ".stderr.txt").read_text(errors="replace")
    cache = [l for l in err.splitlines() if "expert cache" in l][:3]
    pathlib.Path(str(out) + ".seg.json").write_text(json.dumps({"segments": segs, "config": a.config,
                                                                 "args": args, "env": a.env or [],
                                                                 "expert_cache_lines": cache}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ prompt path
PROMPT_LENS = (256, 512, 768, 1000)
# the long prompts: start k at k tenths of the joined segments, 4600 + 40 k tokens (no token read twice in one prompt)
LONG_LENS = tuple(4600 + 40 * k for k in range(10))
# v2's: start j at j thirtieths, so every third one is v1's prompt j / 3 with its start and length
LONG_LENS_V2 = tuple(4600 + 40 * (j // 3) + 13 * (j % 3) for j in range(30))
# v2's xl prompts: three per xl text a third apart, and one past 20,972 tokens where T x 10 x 2560 x 4 B > 2^31
XL_LENS = (20000, 20160, 20320)
XL_LONGEST = 24300
WARM_TOKENS = 4800


def rotated(ids: list[int], start: int, n: int) -> list[int]:
    return (ids[start:] + ids[:start])[:n]


def prompt_set(st: dict, version: str = "v1", xl: dict | None = None, warm: bool = True) -> list[dict]:
    """A `prompt` run's requests in order: v2's unscored warm request, the short, the long, then v2's xl prompts."""
    reqs = [{"set": "short", "segment": s["name"], "tokens": n, "ids": s["ids"][:n]}
            for s in st["segments"] for n in PROMPT_LENS]
    lens = LONG_LENS if version == "v1" else LONG_LENS_V2
    joined = [t for s in st["segments"] for t in s["ids"]]
    if len(joined) < max(lens):
        raise SystemExit(f"the set holds {len(joined)} tokens, the long prompts need {max(lens)}")
    for j, n in enumerate(lens):
        start = j * len(joined) // len(lens)
        reqs.append({"set": "long", "segment": "joined", "start": start, "tokens": n, "ids": rotated(joined, start, n)})
    if version == "v1":
        return reqs
    if xl is None:
        raise SystemExit("set v2 needs the xl texts' ids (--xl-ids)")
    texts = {s["name"]: s["ids"] for s in xl["segments"]}
    if not {"code", "docs"} <= set(texts):
        raise SystemExit(f"the xl set holds {sorted(texts)}, it needs code and docs")
    for name, ids in texts.items():
        if len(ids) < XL_LONGEST:
            raise SystemExit(f"xl {name}: only {len(ids)} tokens, the xl prompts need {XL_LONGEST}")
    for name, ids in texts.items():
        for k, n in enumerate(XL_LENS):
            start = k * len(ids) // len(XL_LENS)
            reqs.append({"set": "xl", "segment": name, "start": start, "tokens": n, "ids": rotated(ids, start, n)})
    code = texts["code"]
    reqs.append({"set": "xl", "segment": "code", "start": len(code) // 6, "tokens": XL_LONGEST,
                 "ids": rotated(code, len(code) // 6, XL_LONGEST)})
    if not warm:
        return reqs
    docs = texts["docs"]
    start = len(docs) // 2              # no scored prompt starts here
    return [{"set": "warm", "segment": "docs", "start": start, "tokens": WARM_TOKENS,
             "ids": rotated(docs, start, WARM_TOKENS)}] + reqs


def prompt_engine(cfg: dict, a, longest: int) -> tuple[list[str], dict]:
    """The prompt run's engine arguments and environment, with --keep-caches as the server starts the config."""
    env = config_env(cfg, a, a.keep_caches)
    if a.keep_caches:                      # the config's context, K/V and chunk, as the A/B arms run it
        return engine_args(server_config(cfg), False, 0, True), env
    args = engine_args(cfg, a.gpu_experts, longest, False)
    k = args.index("--short-read")
    del args[k:k + 2]                      # the default (64): these prompts take the batched prompt path
    k = args.index("--prefill")
    args[k + 1] = "auto"
    return args, env


def cmd_prompt(a) -> int:
    """The batched prompt path's answer to every request of `prompt_set` (no prompt cache), the logits after each."""
    cfg = json.loads(pathlib.Path(a.config).read_text(encoding="utf-8-sig"))
    st = json.loads(pathlib.Path(a.ids).read_text())
    xl = json.loads(pathlib.Path(a.xl_ids).read_text()) if a.set == "v2" else None
    reqs = prompt_set(st, a.set, xl, warm=not a.no_warm)
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in [*out.glob("first.*"), out / "index.json"]:
        f.unlink(missing_ok=True)
    args, env = prompt_engine(cfg, a, max(r["tokens"] for r in reqs))
    env["STRATA_DUMP_FIRST_LOGITS"] = str(out / "first")
    if mem_available_gb() < a.min_free_gb:
        raise SystemExit(f"only {mem_available_gb():.0f} GB available (< {a.min_free_gb}): not starting an engine")
    if a.wrap:
        args = a.wrap.split() + args
    index = []

    def write_index(**extra) -> None:     # after each request, so a run the engine ends keeps the prompts before it
        (out / "index.json").write_text(json.dumps({"version": a.set, "requests": index, "args": args,
                                                    "env": a.env or [], **extra}, indent=1))

    print("engine:", " ".join(args), flush=True)
    with open(out / "stderr.txt", "w") as errf, \
         subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                          cwd=cfg.get("cwd"), env=env, bufsize=1) as p:
        try:
            for line in p.stdout:
                if line.startswith("READY"):
                    break
            else:
                raise SystemExit(f"the engine ended before READY (exit {p.wait()}); see {out}/stderr.txt")
            for r in reqs:
                t0 = time.time()
                p.stdin.write("GEN 1 " + ",".join(map(str, r["ids"])) + "\n")
                p.stdin.flush()
                done = None
                for line in p.stdout:
                    if line.startswith(("DONE", "ERR")):
                        done = line.strip()
                        break
                if done is None or not done.startswith("DONE") or "nonfinite" in done:
                    write_index(failed={**{k: v for k, v in r.items() if k != "ids"}, "answer": done})
                    raise SystemExit(f"{r['set']} {r['segment']} {r['tokens']}: the engine answered {done!r}")
                index.append({k: v for k, v in r.items() if k != "ids"})
                write_index()
                print(f"{r['set']} {r['segment']} {r.get('start', 0)} {r['tokens']}: {time.time() - t0:.1f} s",
                      flush=True)
            p.stdin.write("QUIT\n")
            p.stdin.flush()
            p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            pass
        finally:
            if p.poll() is None:
                p.kill()
    print(f"{len(index)} prompts")
    return 0


def prompt_keys(d: pathlib.Path) -> dict:
    """A `prompt` directory's scored requests as {(set, segment, tokens, start): file number}, old entries are short."""
    reqs = json.loads((d / "index.json").read_text())["requests"]
    return {(q.get("set", "short"), q["segment"], q["tokens"], q.get("start", 0)): i for i, q in enumerate(reqs)
            if q.get("set") != "warm"}


def is_v1(key: tuple) -> bool:
    """v1's 30 prompts: every short one, and the long ones of v1's lengths (v2's others are 13 or 26 tokens off)."""
    return key[0] == "short" or (key[0] == "long" and key[2] in LONG_LENS)


def logp(d: pathlib.Path, i: int):
    import numpy as np
    x = np.fromfile(d / f"first.{i}", dtype=np.float32).astype(np.float64)
    x -= x.max()
    return x - np.log(np.exp(x).sum())


def prompt_rows(ref: pathlib.Path, run: pathlib.Path, xl_ref: pathlib.Path | None = None) -> tuple[list[dict], int]:
    """Per reference prompt: full-vocabulary KL(ref || run), top-1 match, the reference's entropy and top-1 p."""
    import numpy as np
    keys = [(ref, k, i) for k, i in prompt_keys(ref).items() if xl_ref is None or k[0] != "xl"]
    if xl_ref is not None:
        keys += [(xl_ref, k, i) for k, i in prompt_keys(xl_ref).items() if k[0] == "xl"]
    ak = prompt_keys(run) if (run / "index.json").exists() else prompt_keys(ref)
    rows, missing = [], 0
    for d, key, i in keys:
        j = ak.get(key)
        if j is None or not (run / f"first.{j}").exists():
            missing += 1
            continue
        lr, la = logp(d, i), logp(run, j)
        pr = np.exp(lr)
        finite = bool(np.isfinite(la).all())
        rows.append({"set": key[0], "segment": key[1], "tokens": key[2], "start": key[3], "finite": finite,
                     "kl": float((pr * (lr - la)).sum()) if finite else float("inf"),
                     "top1": finite and bool(lr.argmax() == la.argmax()),
                     "entropy": float(-(pr * lr).sum()), "p1": float(pr.max())})
    return rows, missing


def summary(rows: list[dict]) -> dict:
    kls = sorted(r["kl"] for r in rows)
    n = len(kls)
    nan = float("nan")
    return {"n": n, "top1": sum(r["top1"] for r in rows) / n if n else nan, "mean": sum(kls) / n if n else nan,
            "median": kls[n // 2] if n else nan, "max": kls[-1] if n else nan}


def scored_sets(rows: list[dict]) -> dict:
    """The sets the gate reads: short, long, xl, all (short and long), and v1's keys alone."""
    sets = {name: [r for r in rows if r["set"] == name] for name in ("short", "long", "xl")}
    sets["all"] = sets["short"] + sets["long"]
    sets["v1"] = [r for r in rows if is_v1((r["set"], r["segment"], r["tokens"], r["start"]))]
    return sets


def cmd_pcompare(a) -> int:
    ref = pathlib.Path(a.ref)
    print(f"reference: {ref}")
    for run in a.runs:
        r = pathlib.Path(run)
        rows, missing = prompt_rows(ref, r)
        if a.per_prompt:
            print(f"  {'set':5s} {'segment':12s} {'start':>6s} {'tokens':>6s} {'KL':>9s} {'top-1':>5s} {'H(ref)':>7s} "
                  f"{'p1(ref)':>7s}")
            for q in rows:
                print(f"  {q['set']:5s} {q['segment']:12s} {q['start']:6d} {q['tokens']:6d} {q['kl']:9.5f} "
                      f"{'yes' if q['top1'] else 'no':>5s} {q['entropy']:7.3f} {q['p1']:7.3f}")
        for name, s in scored_sets(rows).items():
            if not s and name in ("xl", "v1"):
                continue
            m = summary(s)
            print(f"{r.name:40s} {name:5s} prompts {m['n']:3d}  top-1 {100 * m['top1'] if m['n'] else 0:6.2f}%  "
                  f"KL mean {m['mean']:.5f}  median {m['median']:.5f}  max {m['max']:.5f}"
                  + (f"  ({missing} of the reference's missing)" if name == "all" and missing else ""))
    return 0


# the gate's lines (Phase 1's, and for the xl set no top-1 miss where the reference was this sure)
GATE_MEAN, GATE_MEDIAN, GATE_TOP1, GATE_SURE = 0.03, 0.005, 0.90, 0.5


def gate_checks(rows: list[dict], missing: int) -> list[tuple[str, str, str, str, bool]]:
    """One run's checks as (set, what, value, line, passed).  No prompt is ever left out."""
    sets, out = scored_sets(rows), []
    for name in ("all", "long", "xl"):
        s = summary(sets[name])
        out.append((name, "prompts", str(s["n"]), ">= 1", s["n"] >= 1))
        if not s["n"]:
            continue
        out.append((name, "mean", f"{s['mean']:.5f}", f"<= {GATE_MEAN}", s["mean"] <= GATE_MEAN))
        if name == "all":
            out.append((name, "median", f"{s['median']:.5f}", f"<= {GATE_MEDIAN}", s["median"] <= GATE_MEDIAN))
        if name != "xl":
            out.append((name, "top-1", f"{100 * s['top1']:.2f}%", f">= {100 * GATE_TOP1:.0f}%", s["top1"] >= GATE_TOP1))
        else:
            miss = sum(1 for r in sets[name] if not r["top1"] and r["p1"] >= GATE_SURE)
            out.append((name, "misses", str(miss), f"0 at p1(ref) >= {GATE_SURE}", miss == 0))
    bad = sum(1 for r in rows if not r["finite"])
    out.append(("every", "non-finite", str(bad), "0", bad == 0))
    out.append(("every", "missing", str(missing), "0", missing == 0))
    return out


def cmd_gate(a) -> int:
    """Gate v2 on one or two starts: exit 0 only if every run passes every line."""
    ref = pathlib.Path(a.ref)
    xl_ref = pathlib.Path(a.xl_ref) if a.xl_ref else None
    print(f"reference: {ref}" + (f"  xl reference: {xl_ref}" if xl_ref else ""))
    passed = 0
    for run in a.runs:
        r = pathlib.Path(run)
        checks = gate_checks(*prompt_rows(ref, r, xl_ref))
        print(r.name)
        for name, what, val, line, ok in checks:
            print(f"  {name:5s} {what:10s} {val:>9s}  {line:22s} {'ok' if ok else 'FAIL'}")
        ok = all(c[-1] for c in checks)
        passed += ok
        print(f"{r.name}: {'PASS' if ok else 'FAIL'}")
    print(f"gate: {'PASS' if passed == len(a.runs) else 'FAIL'}, {passed} of {len(a.runs)} runs pass")
    return 0 if passed == len(a.runs) else 1


# ------------------------------------------------------------------------------------------------ compare
def read_run(path: str) -> dict:
    rows = open(path).read().splitlines()
    segs = json.load(open(path + ".seg.json"))["segments"]
    out = {}
    for s in segs:
        d = {}
        for line in rows[s["first_line"]:s["end_line"]]:
            f = line.split("\t")
            if len(f) < 8 + TOPK:
                continue
            d[int(f[0])] = (int(f[1]), float(f[2]), int(f[3]), [(int(x), float(y)) for x, y in
                                                                (z.split(":") for z in f[8:8 + TOPK])])
        out[s["name"]] = d
    return out


def compare(ref: dict, run: dict) -> dict:
    res, allk, tot = {}, [], {"n": 0, "agree": 0, "nll_r": 0.0, "nll_a": 0.0, "kl": 0.0}
    for name, rr in ref.items():
        ra = run.get(name, {})
        pos = sorted(set(rr) & set(ra))
        n = agree = 0
        kls, nll_r, nll_a = [], 0.0, 0.0
        for q in pos:
            tr, lr, topr, kr = rr[q]
            ta, la, topa, ka = ra[q]
            if tr != ta:
                raise SystemExit(f"{name} pos {q}: target {tr} != {ta} (not the same text)")
            n += 1
            agree += topr == topa
            nll_r -= lr
            nll_a -= la
            da = dict(ka)
            floor = min(b for _, b in ka)
            pr = [math.exp(b) for _, b in kr]
            pa = [math.exp(da.get(i, floor)) for i, _ in kr]
            sr, sa = sum(pr), sum(pa)
            kls.append(sum((x / sr) * math.log((x / sr) / (y / sa)) for x, y in zip(pr, pa) if x > 0))
        allk += kls
        if n:
            res[name] = {"positions": n, "top1": agree / n, "kl": sum(kls) / n,
                         "ppl_ref": math.exp(nll_r / n), "ppl": math.exp(nll_a / n)}
        tot["n"] += n
        tot["agree"] += agree
        tot["nll_r"] += nll_r
        tot["nll_a"] += nll_a
        tot["kl"] += sum(kls)
    n = max(tot["n"], 1)
    allk.sort()
    res["all"] = {"positions": tot["n"], "top1": tot["agree"] / n, "kl": tot["kl"] / n,
                  "kl_p99": allk[int(0.99 * (len(allk) - 1))] if allk else float("nan"),
                  "ppl_ref": math.exp(tot["nll_r"] / n), "ppl": math.exp(tot["nll_a"] / n)}
    return res


def cmd_compare(a) -> int:
    ref = read_run(a.ref)
    print(f"reference: {a.ref}")
    print(f"{'run':40s} {'segment':13s} {'pos':>5s} {'top1':>7s} {'KL':>8s} {'KL p99':>8s} {'PPL':>7s} {'PPL ref':>7s}")
    allres = {}
    for path in a.runs:
        r = compare(ref, read_run(path))
        allres[path] = r
        for seg, v in r.items():
            if seg != "all" and not a.segments:
                continue
            print(f"{pathlib.Path(path).name[:40]:40s} {seg:13s} {v['positions']:5d} {100 * v['top1']:6.2f}% "
                  f"{v['kl']:8.5f} {v.get('kl_p99', float('nan')):8.4f} {v['ppl']:7.3f} {v['ppl_ref']:7.3f}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(allres, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ids")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--set", default=str(SET_DIR))
    p.add_argument("--segments", help="the texts to read, in order (default: every .txt of the set, v1's names first)")
    p.add_argument("--seg-tokens", type=int, default=1000, help="tokens kept of each text, 0 keeps it whole")
    p.add_argument("--out")
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", default=str(SET_DIR / "set.ids.json"))
    p.add_argument("--env", action="append")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu-experts", nargs="?", const=True, default=False,
                   help="--expert-cache 1500 (or the N given, e.g. auto) instead of one slot (the experts on the GPU kernels)")
    g.add_argument("--keep-caches", action="store_true",
                   help="the config's engine as the server starts it (tier, caches, RAM copy, context, K/V), with no "
                        "adaptation, PCIe share, suffix drafts or prompt cache")
    p.add_argument("--allow", action="append", default=[], choices=sorted(ALLOW),
                   help="with --keep-caches: keep the config's value of this switch (pcie: --pcie-frac)")
    p.add_argument("--min-free-gb", type=float, default=16)
    p.add_argument("--wrap", help="a command prefix for the engine, e.g. 'systemd-run --user --scope -p MemoryMax=200G'")
    p = sub.add_parser("prompt", help="the batched prompt path's next-token logits after each prompt of a set")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--set", choices=("v1", "v2"), default="v2",
                   help="v1: 20 short and 10 long prompts; v2: a warm request, 20 short, 30 long and 7 xl prompts")
    p.add_argument("--no-warm", action="store_true", help="v2 without its unscored warm request")
    p.add_argument("--ids", default=str(SET_DIR / "set.ids.json"))
    p.add_argument("--xl-ids", default=str(XL_DIR / "set.ids.json"))
    p.add_argument("--env", action="append")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu-experts", action="store_true")
    g.add_argument("--keep-caches", action="store_true",
                   help="the config's engine as the server starts it (layer split, card order, context, K/V, chunk, "
                        "caches, tier), with no adaptation, PCIe share, suffix drafts, prompt cache, CPU "
                        "prompt share or profile save")
    p.add_argument("--min-free-gb", type=float, default=16)
    p.add_argument("--wrap")
    p = sub.add_parser("compare")
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    p.add_argument("--segments", action="store_true")
    p.add_argument("--json")
    p = sub.add_parser("pcompare", help="full-vocabulary KL and top-1 of `prompt` directories against a reference one")
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    p.add_argument("--per-prompt", action="store_true", help="a row per prompt with the reference's entropy")
    p = sub.add_parser("gate", help="gate v2 over one or two starts of a candidate, exit 0 only if each passes")
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    p.add_argument("--xl-ref", help="the xl prompts' own reference directory (default: REF)")
    a = ap.parse_args()
    if a.cmd == "run" and a.allow and not a.keep_caches:
        ap.error("--allow needs --keep-caches")
    return {"ids": cmd_ids, "run": cmd_run, "prompt": cmd_prompt, "compare": cmd_compare,
            "pcompare": cmd_pcompare, "gate": cmd_gate}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
