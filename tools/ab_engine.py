"""A/B two engine builds, configs or environments: restart the server per arm, alternate arms over rounds.

    python tools/ab_engine.py strata-q2_0.json A:exe=/path/a B:exe=/path/b,STRATA_EXP_MODE=0 [--rounds 3]
    python tools/ab_engine.py base.json A:config=a.json B:config=b.json --warm 1 --measure 2 --drop-caches --identity

An arm is NAME:key=value,... and `config` replaces the positional config for that arm, `exe` replaces the config's
engine, `xargs` appends ';'-separated engine arguments, every other key is set in the engine's environment.  Each round
starts the server once per arm (the arms' order alternates) and waits for it to answer a short question.  `--warm N`
then sends N unmeasured passes of the requests and `--measure N` sends N measured passes (default 1).  The requests
are greedy with fixed prompts: a short chat set and one long prompt per `--long-words W` (default: one of 6000 words,
tagged `long`).  `--drop-caches` empties the page cache before every server start (it needs `sudo -n`), tags the first
request after the start `cold` (even in a warm pass, which is then logged too) and prints the cold requests apart from
the medians.  `--full-text` logs whole answers, not 80 characters.  `--identity` compares the answers of the first
measured pass of each arm, prints `identical` or the first request and arm that differ, and exits 1 on a difference.
An empty answer, or an arm with none, is an error with exit 1, not `identical`.  Arm names must differ.
Prints the server's own timings (llama.cpp's `timings`: decode and prompt tokens per second) per request and the
per-arm medians.  Decode numbers move run to run (the adaptive expert tier, the OS), so compare medians over rounds.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List ten practical tips for learning a new language as an adult, one sentence each.",
)


def long_prompt(words: int) -> str:
    rng = random.Random(7)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return f"Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."


def post(port: int, prompt: str, max_tokens: int) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                         "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
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
        time.sleep(3)
    raise TimeoutError("server not ready")


def parse_arm(s: str) -> tuple[str, dict]:
    name, _, kv = s.partition(":")
    return name, dict(x.split("=", 1) for x in kv.split(",") if x)


def first_difference(texts: dict[str, dict[str, str]]) -> tuple[str, str] | None:
    arms = list(texts)
    if not arms:
        return None
    first = texts[arms[0]]
    for tag in dict.fromkeys(t for arm in arms for t in texts[arm]):
        for arm in arms[1:]:
            if texts[arm].get(tag) != first.get(tag):
                return tag, arm
    return None


def identity_verdict(texts: dict[str, dict[str, str]]) -> tuple[int, str]:
    """The exit code and the identity line: no answers or an empty answer is an error, never `identical`."""
    if not texts:
        return 1, "identity: error, no arm has measured answers"
    for arm, t in texts.items():
        if not t:
            return 1, f"identity: error, arm {arm} has no measured answers"
        tag = next((tag for tag, x in t.items() if not x), None)
        if tag is not None:
            return 1, f"identity: error, arm {arm} gave an empty answer at {tag}"
    diff = first_difference(texts)
    if diff is None:
        return 0, "identity: identical"
    return 1, f"identity: first difference at {diff[0]} in arm {diff[1]} (against {next(iter(texts))})"


def drop_caches() -> None:
    cmd = ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
    os.sync()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        raise SystemExit(f"--drop-caches failed: {' '.join(cmd)}: {e}")
    if r.returncode != 0:
        raise SystemExit(f"--drop-caches failed ({r.returncode}): {' '.join(cmd)}: {r.stderr.strip()}")


def ask(a: argparse.Namespace, rnd: int, name: str, tag: str, prompt: str, ps, cold: bool) -> tuple[dict, str]:
    r = post(a.port, prompt, a.max_new)
    t = r.get("timings", {})
    text = r["choices"][0]["message"].get("content", "")
    row = {"round": rnd, "arm": name, "req": tag, "decode": t.get("predicted_per_second"),
           "prompt": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
           "n": r.get("usage", {}).get("completion_tokens"),
           "text": text if a.full_text else text[:80], "cold": cold, "pass": ps}
    return row, text


def describe(row: dict, w: int, many: bool) -> str:
    more = (f"  pass {row['pass']}" if many or row["pass"] == "warm" else "") + ("  cold" if row["cold"] else "")
    return (f"  {row['req']:{w}s} decode {row['decode']} tok/s  prompt {row['prompt']} tok/s "
            f"({row['prompt_n']})  n {row['n']}{more}")


def tok_s(x) -> str:
    return f"{x:.1f}" if x else "n/a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--long-words", type=int, action="append", metavar="W",
                    help="words of a long prompt, repeatable (default: 6000)")
    ap.add_argument("--warm", type=int, default=0, help="unmeasured passes after ready (default: 0)")
    ap.add_argument("--measure", type=int, default=1, help="measured passes per server start (default: 1)")
    ap.add_argument("--drop-caches", action="store_true",
                    help="empty the page cache before each server start (sudo -n), tag the first request cold")
    ap.add_argument("--full-text", action="store_true", help="log whole answers, not 80 characters")
    ap.add_argument("--identity", action="store_true",
                    help="compare the first measured pass across arms, print the first difference and exit 1 on one")
    ap.add_argument("--log", default="ab_engine.jsonl")
    a = ap.parse_args(argv)
    longs = [("long", 6000)] if a.long_words is None else [(f"long{x}", x) for x in a.long_words]
    if a.warm < 0 or a.measure < 1:
        ap.error("--warm needs 0 or more and --measure needs 1 or more")
    if a.identity and len(a.arms) < 2:
        ap.error("--identity needs at least two arms")
    arms = []
    for s in a.arms:
        name, env = parse_arm(s)
        cpath = env.pop("config", a.config)
        arms.append((name, env, json.loads(Path(cpath).read_text())))
    names = [n for n, _, _ in arms]
    if len(set(names)) < len(names):
        ap.error("arm names must differ: " + ", ".join(sorted({n for n in names if names.count(n) > 1})))
    reqs = [(f"chat{i}", p) for i, p in enumerate(PROMPTS)] + [(tag, long_prompt(words)) for tag, words in longs]
    w = max([6] + [len(tag) for tag, _ in reqs])
    res: dict = {n: {} for n in names}      # measured rows that are not cold, by arm and request
    colds: dict = {n: [] for n in names}    # the cold rows, by arm
    texts: dict = {n: {} for n in names}    # whole answers of the first measured pass, by arm and request
    with open(a.log, "a") as out:
        for rnd in range(a.rounds):
            order = arms if rnd % 2 == 0 else arms[::-1]
            for name, env, base in order:
                cfg = dict(base)
                e = dict(env)
                if "exe" in e:
                    cfg["exe"] = e.pop("exe")
                if "xargs" in e:                       # extra engine arguments, ';'-separated
                    cfg["args"] = list(base["args"]) + e.pop("xargs").split(";")
                cfg["port"] = a.port
                cfg["log"] = str(Path(tempfile.gettempdir()) / f"ab_{name}_r{rnd}.log")
                cpath = Path(tempfile.gettempdir()) / f"ab_{name}.json"
                cpath.write_text(json.dumps(cfg, indent=1))
                penv = dict(os.environ, **e)
                if a.drop_caches:
                    drop_caches()
                proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                                         "--config", str(cpath), "--port", str(a.port)], cwd=str(ROOT), env=penv,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    t0 = time.time()
                    wait_ready(a.port, proc)
                    print(f"round {rnd} {name}: ready in {time.time() - t0:.0f} s", flush=True)
                    cold = a.drop_caches              # the first request after the start
                    for k, ps in enumerate(["warm"] * a.warm + list(range(a.measure))):
                        for tag, p in reqs:
                            row, text = ask(a, rnd, name, tag, p, ps, cold)
                            cold = False
                            if ps == "warm" and not a.drop_caches:
                                continue                  # a warm pass is logged only with --drop-caches
                            out.write(json.dumps(row) + "\n")
                            out.flush()
                            if row["cold"]:
                                colds[name].append(row)
                            elif ps != "warm":
                                res[name].setdefault(tag, []).append(row)
                            if ps != "warm":
                                texts[name].setdefault(tag, text)
                            print(describe(row, w, a.measure > 1), flush=True)
                        if ps == "warm" and not a.drop_caches:
                            print(f"  warm pass {k + 1} of {a.warm} done", flush=True)
                finally:
                    proc.terminate()
                    try:
                        proc.wait(60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    time.sleep(5)
    print("\nmedians over rounds" + (" (measured passes, not cold)" if a.drop_caches else "") + ":")
    for tag, _ in reqs:
        cells = []
        for name in names:
            rows = res[name].get(tag, [])
            d = statistics.median([x["decode"] for x in rows if x["decode"]]) if rows else float("nan")
            pr = statistics.median([x["prompt"] for x in rows if x["prompt"]]) if rows else float("nan")
            cells.append(f"{name}: decode {d:6.1f}  prompt {pr:7.1f}")
        print(f"  {tag:{w}s} | " + " | ".join(cells))
    if a.drop_caches:
        print("\ncold requests (the first request after each start):")
        for name in names:
            cells = [f"round {x['round']} {x['req']} decode {tok_s(x['decode'])}  prompt {tok_s(x['prompt'])}"
                     for x in colds[name]]
            print(f"  {name}: " + " | ".join(cells))
    if a.identity:
        code, line = identity_verdict(texts)
        print("\n" + line)
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
