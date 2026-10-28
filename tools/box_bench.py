"""The box bench: one engine start per arm, passes of named requests, the start's facts checked before an arm measures,
and the keep verdict over the starts of two arms.

    python tools/box_bench.py ROOT CONFIG --log OUT.jsonl --pass warm:chat0,p8k --pass m:chat0,chat1,chat2,p2k,p8k,p20k \\
        [--sample] [--keep-logs PREFIX] [--expect chunk=24576,tier_in_prompts=1,pinned_gib=95..100] ARM...
    python tools/box_bench.py verdict OUT.jsonl --a PREV --b NEW --target p20k.prompt [--other chat1.decode ...]

ROOT is the tree whose serve/server.py runs the engine.  An arm is NAME:key=value,... as in tools/ab_engine.py (config,
exe, xargs ';'-separated, every other key goes to the engine's environment).  The requests: chat0 to chat2 and the long
prompts p2k, p8k and p20k (about 2K, 8K and 20K tokens, each with its own seed).  A pass of kind `warm` is not measured.
Before the first measured pass (after the last pass when none is measured) the start's facts are read from the engine
log (tools/engine_log.py) and checked against --expect: FACT=V or FACT=LO..HI (either end may be left open), a fact the
log does not show never matches.  A mismatch stops the bench with exit 4.  Every row of OUT.jsonl names its arm, its
start and its kind (`row`): `ready`, `req` (the server's timings, the drives' MB read, Cached and MemAvailable across the
request), `facts` (at the check and at the end of the start), `engine` (the engine's own row for each request: the OS's
drive reads, the hit rate, the planner) and `clock` (with --sample: the 3090's clock and power from 0.5 s samples, which
also go to OUT-samples.jsonl).

`verdict` reads the measured passes of two arms (--a is PREV, --b is NEW).  A start's value is the mean of its passes.
KEEP when every NEW start beats every PREV start on the target (min NEW above max PREV) and the mean gap is at least the
larger arm's spread (max minus min).  An --other metric is not slower when its NEW mean is at least PREV's mean minus the
smaller of PREV's spread and 5% of PREV's mean.  SLOWER when an --other metric is slower, or when the target loses the
same way KEEP wins.  Otherwise NOT RESOLVED.  Exit 0 on KEEP, 1 on NOT RESOLVED, 3 on SLOWER, 2 on an error (fewer than
two starts of an arm).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_log  # noqa: E402

PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List ten practical tips for learning a new language as an adult, one sentence each.",
)
LONGS = {"p2k": (1900, 2), "p8k": (7600, 8), "p20k": (19000, 7)}
EXIT_EXPECT = 4


def long_prompt(words: int, seed: int) -> str:
    rng = random.Random(seed)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return f"Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."


def request_text(tag: str) -> str:
    if tag.startswith("chat") and tag[4:].isdigit() and int(tag[4:]) < len(PROMPTS):
        return PROMPTS[int(tag[4:])]
    if tag not in LONGS:
        raise ValueError(f"unknown request {tag!r} (chat0 to chat{len(PROMPTS) - 1}, {', '.join(LONGS)})")
    w, s = LONGS[tag]
    return long_prompt(w, s)


def post(port: int, prompt: str, max_tokens: int) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                         "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    post(port, "Say hi.", 8)
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("server not ready")


def meminfo_mb(key: str) -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key):
            return int(line.split()[1]) / 1024
    return 0.0


def drive_read_mb() -> float:
    total = 0
    for line in Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if f[2].startswith("nvme") and "p" not in f[2][4:]:
            total += int(f[5])
    return total * 512 / 1e6


def mem_row() -> dict:
    return {"cached_mb": round(meminfo_mb("Cached:")), "avail_mb": round(meminfo_mb("MemAvailable:"))}


class Sampler(threading.Thread):
    """0.5 s samples of the cards and the CPU, tagged with the start and the request running."""

    def __init__(self, path: Path):
        super().__init__(daemon=True)
        self.out = open(path, "a")
        self.tag = "idle"
        self.start_id = None
        self.rows: list[dict] = []
        self.stop = threading.Event()
        cards = sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))
        self.amd = cards[0].parent if cards else None

    def run(self) -> None:
        prev = self.cpu()
        while not self.stop.wait(0.5):
            row = {"t": round(time.time(), 2), "start": self.start_id, "tag": self.tag}
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=clocks.gr,clocks.mem,power.draw,utilization.gpu,"
                                    "temperature.gpu,clocks_throttle_reasons.active", "--format=csv,noheader,nounits",
                                    "-i", "0"], capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
                row.update(nv_clk=float(q[0]), nv_mclk=float(q[1]), nv_w=float(q[2]), nv_use=float(q[3]),
                           nv_temp=float(q[4]), nv_thr=q[5])
            except Exception:
                pass
            if self.amd is not None:
                try:
                    row["amd_use"] = int((self.amd / "gpu_busy_percent").read_text())
                    sclk = [x for x in (self.amd / "pp_dpm_sclk").read_text().splitlines() if x.endswith("*")]
                    row["amd_sclk"] = sclk[0].split()[1] if sclk else None
                    hw = next((self.amd / "hwmon").iterdir())
                    row["amd_w"] = int((hw / "power1_average").read_text()) / 1e6
                except Exception:
                    pass
            cur = self.cpu()
            row["cpu"] = [round(100.0 * (1 - (c[1] - p[1]) / max(1, c[0] - p[0])), 1) for p, c in zip(prev, cur)]
            prev = cur
            self.rows.append(row)
            self.out.write(json.dumps(row) + "\n")
            self.out.flush()

    @staticmethod
    def cpu() -> list[tuple[int, int]]:
        rows = []
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("cpu") and line[3].isdigit():
                v = [int(x) for x in line.split()[1:]]
                rows.append((sum(v), v[3] + v[4]))
        return rows


def clock_summary(samples: list[dict]) -> dict:
    """The 3090's clock and power over one start's samples, and while it read the long prompts."""
    rows = [r for r in samples if "nv_clk" in r]
    out = {"samples": len(rows), "clk_max": max((r["nv_clk"] for r in rows), default=None)}
    longs = [r for r in rows if r["tag"].split(":")[-1].startswith("p")]
    if longs:
        busy = [r["nv_clk"] for r in longs if r.get("nv_use", 0) > 50]
        out.update(long_w_mean=round(statistics.mean(r["nv_w"] for r in longs), 1),
                   long_w_max=max(r["nv_w"] for r in longs), busy_clk_min=min(busy, default=None),
                   busy_clk_max=max(busy, default=None))
    out["clock_over"] = out["clk_max"] is not None and out["clk_max"] > int(os.environ.get("BOX_CLK_MAX", "1500")) + 15
    return out


def parse_expect(text: str) -> dict[str, tuple]:
    """FACT=V or FACT=LO..HI pairs, ',' separated: the bounds of each fact (None for an open end)."""
    def value(s: str):
        s = s.strip().lower()
        if s in ("true", "false"):
            return float(s == "true")
        return float(s)

    out = {}
    for item in filter(None, text.split(",")):
        fact, eq, spec = item.partition("=")
        if fact not in engine_log.FACTS or not eq or not spec:
            raise ValueError(f"--expect {item!r}: FACT=V or FACT=LO..HI, a fact of {', '.join(engine_log.FACTS)}")
        if ".." in spec:
            lo, hi = spec.split("..", 1)
            out[fact] = (value(lo) if lo else None, value(hi) if hi else None)
        else:
            out[fact] = (value(spec), value(spec))
    return out


def mismatches(facts: dict, expect: dict[str, tuple]) -> list[str]:
    bad = []
    for fact, (lo, hi) in expect.items():
        v = facts.get(fact)
        if v is None:
            bad.append(f"{fact} is not in the log")
        elif (lo is not None and v < lo) or (hi is not None and v > hi):
            want = f"{lo:g}" if lo == hi else f"{'' if lo is None else f'{lo:g}'}..{'' if hi is None else f'{hi:g}'}"
            bad.append(f"{fact} is {v}, not {want}")
    return bad


def engine_rows(lines: list[str], asked: list[dict]) -> list[dict]:
    """The engine's rows of the last len(asked) requests, labeled with the bench's, when their prompt sizes agree."""
    eng = engine_log.rows(lines)[-len(asked):] if asked else []
    pairs = list(zip(asked[len(asked) - len(eng):], eng))
    if any(b.get("prompt_n") is not None and e.get("read") != b["prompt_n"] for b, e in pairs):
        print("  the engine's requests do not line up with the bench's: engine rows left unlabeled", flush=True)
        return [{**e, "pass": None, "kind": None, "req": None} for e in eng]
    return [{**e, "pass": b["pass"], "kind": b["kind"], "req": b["req"]} for b, e in pairs]


def run_bench(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="box_bench.py")
    ap.add_argument("root")
    ap.add_argument("config")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--pass", dest="passes", action="append", required=True,
                    help="KIND:req,req (a KIND of warm is not measured)")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--log", required=True)
    ap.add_argument("--keep-logs", default="", help="copy each arm's engine log to this prefix + NAME-engine.log")
    ap.add_argument("--sample", action="store_true", help="0.5 s samples of the cards and the CPU, and a clock row")
    ap.add_argument("--expect", action="append", default=[], help="FACT=V or FACT=LO..HI, ',' separated")
    a = ap.parse_args(argv)
    passes = []
    try:
        for p in a.passes:
            kind, _, reqs = p.partition(":")
            passes.append((kind, reqs.split(",")))
            for r in reqs.split(","):
                request_text(r)
        expect = parse_expect(",".join(a.expect))
    except ValueError as e:
        ap.error(str(e))
    check_at = next((k for k, (kind, _) in enumerate(passes) if kind != "warm"), len(passes))
    sampler = None
    if a.sample:
        sampler = Sampler(Path(a.log).with_name(Path(a.log).stem + "-samples.jsonl"))
        sampler.start()
    root = Path(a.root)
    code = 0
    with open(a.log, "a") as out:
        def write(row: dict) -> None:
            out.write(json.dumps(row) + "\n")
            out.flush()

        for arm in a.arms:
            name, _, kv = arm.partition(":")
            env = dict(x.split("=", 1) for x in kv.split(",") if x)
            cfg = json.loads(Path(env.pop("config", a.config)).read_text())
            if "exe" in env:
                cfg["exe"] = env.pop("exe")
            if "xargs" in env:
                cfg["args"] = list(cfg["args"]) + env.pop("xargs").split(";")
            cfg["port"] = a.port
            elog = Path(tempfile.gettempdir()) / f"box_{name}.log"
            elog.unlink(missing_ok=True)
            cfg["log"] = str(elog)
            cpath = Path(tempfile.gettempdir()) / f"box_{name}.json"
            cpath.write_text(json.dumps(cfg, indent=1))
            start = uuid.uuid4().hex[:8]
            head = {"arm": name, "start": start}
            if sampler:
                sampler.start_id, sampler.tag = start, f"{name}:start"
            print(f"== arm {name} start {start}: env {env} xargs tail {cfg['args'][-6:]} {time.strftime('%T')}",
                  flush=True)
            d0, c0, t0 = drive_read_mb(), meminfo_mb("Cached:"), time.time()
            proc = subprocess.Popen([sys.executable, str(root / "serve" / "server.py"), "--engine", "strata",
                                     "--config", str(cpath), "--port", str(a.port)], cwd=str(root),
                                    env=dict(os.environ, **env), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            asked: list[dict] = []
            try:
                wait_ready(a.port, proc)
                ready = time.time() - t0
                row = dict(head, row="ready", time=time.strftime("%F %T"), ready_s=round(ready, 1),
                           drive_mb=round(drive_read_mb() - d0), cached0_mb=round(c0), **mem_row())
                write(row)
                print(f"  ready in {ready:.0f} s, drives read {row['drive_mb']} MB, Cached {row['cached_mb']} MB",
                      flush=True)
                # the empty pass at the end checks the facts when no pass is measured
                for k, (kind, reqs) in enumerate(passes + [("end", [])]):
                    if k == check_at:
                        f = engine_log.facts(engine_log.read(elog))
                        bad = mismatches(f, expect)
                        write(dict(head, row="facts", at="check", ok=not bad, mismatch=bad, **f))
                        print(f"  facts {json.dumps(f)}", flush=True)
                        if bad:
                            print(f"  EXPECT MISMATCH: {'; '.join(bad)}", flush=True)
                            code = EXIT_EXPECT
                            break
                    for tag in reqs:
                        if sampler:
                            sampler.tag = f"{name}:{k}:{tag}"
                        d0, w0 = drive_read_mb(), time.time()
                        r = post(a.port, request_text(tag), a.max_new)
                        wall = time.time() - w0
                        t = r.get("timings") or {}
                        row = {**head, "row": "req", "pass": k, "kind": kind, "req": tag, "wall_s": round(wall, 2),
                               "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
                               "prompt": t.get("prompt_per_second"), "prompt_ms": t.get("prompt_ms"),
                               "decode": t.get("predicted_per_second"), "n": t.get("predicted_n"),
                               "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                               "drive_mb": round(drive_read_mb() - d0), **mem_row(),
                               "text": r["choices"][0]["message"].get("content", "")[:80]}
                        write(row)
                        asked.append(row)
                        print(f"  {k}{kind[0]} {tag:6s} prompt {row['prompt']} tok/s ({row['prompt_n']}, cache "
                              f"{row['cache_n']})  decode {row['decode']} tok/s  drafts {row['draft_acc']}/"
                              f"{row['draft_n']}  drive {row['drive_mb']} MB  Cached {row['cached_mb']} MemAvailable "
                              f"{row['avail_mb']} MB  wall {wall:.1f} s", flush=True)
            finally:
                if sampler:
                    sampler.tag = f"{name}:stop"
                proc.terminate()
                try:
                    proc.wait(90)
                except subprocess.TimeoutExpired:
                    proc.kill()
                if elog.exists():
                    lines = engine_log.read(elog)
                    write(dict(head, row="facts", at="end", **engine_log.facts(lines)))
                    for e in engine_rows(lines, asked):
                        write(dict(head, row="engine", **e))
                    if a.keep_logs:
                        shutil.copy(elog, f"{a.keep_logs}{name}-engine.log")
                if sampler:
                    c = clock_summary([s for s in sampler.rows if s["start"] == start])
                    write(dict(head, row="clock", **c))
                    print(f"  clock {json.dumps(c)}{' CLOCK_OVER' if c['clock_over'] else ''}", flush=True)
                time.sleep(5)
            if code:
                break
    if sampler:
        sampler.stop.set()
    print(f"== bench {'stopped: the facts differ' if code else 'done'} {time.strftime('%T')}", flush=True)
    return code


def spread(v: list[float]) -> float:
    return max(v) - min(v)


def keep(prev: list[float], new: list[float]) -> str:
    """The keep rule on one target, higher is faster: KEEP, SLOWER (KEEP's mirror) or NOT RESOLVED."""
    gap, big = statistics.mean(new) - statistics.mean(prev), max(spread(prev), spread(new))
    if min(new) > max(prev) and gap >= big:
        return "KEEP"
    if max(new) < min(prev) and -gap >= big:
        return "SLOWER"
    return "NOT RESOLVED"


def not_slower(prev: list[float], new: list[float]) -> bool:
    m = statistics.mean(prev)
    return statistics.mean(new) >= m - min(spread(prev), 0.05 * m)


def start_values(rows: list[dict], arm: str, metric: str) -> dict[str, list[float]]:
    """Each start of the arm (in log order) with its measured passes' values of REQ.FIELD."""
    req, _, field = metric.rpartition(".")
    out: dict[str, list[float]] = {}
    for r in rows:
        if r.get("row") == "req" and r["arm"] == arm and r["kind"] != "warm" and r["req"] == req \
                and r.get(field) is not None:
            out.setdefault(r["start"], []).append(r[field])
    return out


def describe_start(rows: list[dict], start: str, req: str) -> str:
    """A start's facts, clock and the drive reads of its measured target requests."""
    mine = [r for r in rows if r.get("start") == start]
    ready = next((r for r in mine if r["row"] == "ready"), {})
    f = next((r for r in mine if r["row"] == "facts" and r.get("at") == "check"), {})
    c = next((r for r in mine if r["row"] == "clock"), None)
    eng = [r.get("file_mb") for r in mine if r["row"] == "engine" and r.get("req") == req and r.get("kind") != "warm"]
    reqs = [r for r in mine if r["row"] == "req" and r["req"] == req and r["kind"] != "warm"]
    out = (f"ready {ready.get('ready_s')} s at {ready.get('time')}; chunk {f.get('chunk')} w6800 {f.get('w6800_experts')} "
           f"ring {f.get('ring_slots')} tier {f.get('tier_in_prompts')} pinned {f.get('pinned_gib')} GiB vram_low "
           f"{f.get('vram_low')}; {req}: engine drive MB {eng}, drives MB {[r.get('drive_mb') for r in reqs]}, Cached MB "
           f"{[r.get('cached_mb') for r in reqs]}, MemAvailable MB {[r.get('avail_mb') for r in reqs]}")
    if c:
        out += f"; 3090 max {c.get('clk_max')} MHz, {c.get('long_w_mean')} W mean{' CLOCK_OVER' if c.get('clock_over') else ''}"
    return out


def run_verdict(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="box_bench.py verdict")
    ap.add_argument("log")
    ap.add_argument("--a", required=True, help="the PREV arm")
    ap.add_argument("--b", required=True, help="the NEW arm")
    ap.add_argument("--target", required=True, help="REQ.FIELD that must be faster, for example p20k.prompt")
    ap.add_argument("--other", action="append", default=[], help="REQ.FIELD that must not be slower, repeatable")
    a = ap.parse_args(argv)
    rows = [json.loads(x) for x in Path(a.log).read_text().splitlines() if x.strip()]
    vals = {m: {arm: start_values(rows, arm, m) for arm in (a.a, a.b)} for m in [a.target] + a.other}
    for arm in (a.a, a.b):
        print(f"{arm} ({'PREV' if arm == a.a else 'NEW'}):")
        for start, v in vals[a.target][arm].items():
            cells = "  ".join(f"{m} {statistics.mean(vals[m][arm][start]):.1f} {vals[m][arm][start]}"
                              for m in vals if start in vals[m][arm])
            print(f"  start {start}: {cells}")
            print(f"    {describe_start(rows, start, a.target.rpartition('.')[0])}")
    short = [f"{arm} has {len(vals[m][arm])} starts of {m}" for m in vals for arm in (a.a, a.b) if len(vals[m][arm]) < 2]
    if short:
        print("verdict: error, " + "; ".join(short) + " (the keep rule needs two or more)")
        return 2
    means = {m: {arm: [statistics.mean(v) for v in vals[m][arm].values()] for arm in (a.a, a.b)} for m in vals}
    prev, new = means[a.target][a.a], means[a.target][a.b]
    verdict = keep(prev, new)
    gap = statistics.mean(new) - statistics.mean(prev)
    print(f"target {a.target}: PREV mean {statistics.mean(prev):.1f} spread {spread(prev):.1f}, NEW mean "
          f"{statistics.mean(new):.1f} spread {spread(new):.1f}, gap {gap:+.1f} ({100 * gap / statistics.mean(prev):+.1f}%), "
          f"min NEW {min(new):.1f} against max PREV {max(prev):.1f}: {verdict}")
    for m in a.other:
        p, n = means[m][a.a], means[m][a.b]
        ok = not_slower(p, n)
        floor = statistics.mean(p) - min(spread(p), 0.05 * statistics.mean(p))
        print(f"other {m}: PREV mean {statistics.mean(p):.1f} spread {spread(p):.1f}, NEW mean {statistics.mean(n):.1f}, "
              f"floor {floor:.1f}: {'not slower' if ok else 'SLOWER'}")
        if not ok:
            verdict = "SLOWER"
    print(f"verdict: {verdict}")
    return {"KEEP": 0, "NOT RESOLVED": 1, "SLOWER": 3}[verdict]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return run_verdict(argv[1:]) if argv[:1] == ["verdict"] else run_bench(argv)


if __name__ == "__main__":
    sys.exit(main())
