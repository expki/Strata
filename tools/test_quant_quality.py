"""Tests for tools/quant_quality.py's ids, engine arguments, prompt sets, pcompare and gate, with no GPU or engine."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_quality as qq  # noqa: E402

CFG = {"exe": "/e/strata", "args": ["--pack", "/p", "--expert-cache", "auto", "--prefill", "4096", "--kv", "int8",
                                    "--max-context", "262144", "--vision", "--resident-experts",
                                    "--resident-cpu-experts", "--prompt-cache", "4", "--conversation-cache-mib", "512",
                                    "--adapt-every", "4", "--short-read", "128", "--expert-profile-save", "/l.bin",
                                    "--peer-device", "hip:0", "--expert-gib", "cuda0=1,hip0=6.5"]}


def value(args: list[str], name: str) -> list[str]:
    return [args[i + 1] for i, x in enumerate(args) if x == name]


def the_set(seg_tokens: int = 1000) -> dict:
    return {"segments": [{"name": n, "ids": list(range(k * seg_tokens, (k + 1) * seg_tokens))}
                         for k, n in enumerate(qq.SEGMENTS)]}


def the_xl_set(docs: int = 30000, code: int = 40000) -> dict:
    return {"segments": [{"name": "code", "ids": list(range(10 ** 6, 10 ** 6 + code))},
                         {"name": "docs", "ids": list(range(2 * 10 ** 6, 2 * 10 ** 6 + docs))}]}


def the_v1_set() -> dict:
    return json.loads((qq.SET_DIR / "set.ids.json").read_text())


class FakeTokenizer:
    def encode(self, text: str, parse_special: bool = False) -> list[int]:
        return [ord(c) for c in text]


class Ids(unittest.TestCase):
    def run_ids(self, d: Path, seg_tokens: int, segments=None) -> dict:
        ns = argparse.Namespace(tokenizer="tok", set=str(d), segments=segments, seg_tokens=seg_tokens, out=None)
        with mock.patch.object(qq, "load_tokenizer", return_value=FakeTokenizer()), \
             contextlib.redirect_stdout(io.StringIO()):
            qq.cmd_ids(ns)
        return json.loads((d / "set.ids.json").read_text())

    def test_two_texts_and_seg_tokens_0_keep_both_whole(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "xl"
            d.mkdir()
            (d / "docs.txt").write_text("the docs text", encoding="utf-8")
            (d / "code.txt").write_text("int main() { return 0; }", encoding="utf-8")
            st = self.run_ids(d, 0)
        self.assertEqual(st["version"], "xl")
        self.assertEqual(st["seg_tokens"], 0)
        self.assertEqual([s["name"] for s in st["segments"]], ["code", "docs"])
        self.assertEqual(st["segments"][0]["ids"], [ord(c) for c in "int main() { return 0; }"])
        self.assertEqual(st["segments"][1]["ids"], [ord(c) for c in "the docs text"])

    def test_the_segments_flag_sets_the_order(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "xl"
            d.mkdir()
            (d / "docs.txt").write_text("ab", encoding="utf-8")
            (d / "code.txt").write_text("cde", encoding="utf-8")
            st = self.run_ids(d, 0, "docs,code")
        self.assertEqual([(s["name"], len(s["ids"])) for s in st["segments"]], [("docs", 2), ("code", 3)])

    def test_a_directory_keeps_v1_order_then_the_rest_by_name(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "v9"
            d.mkdir()
            for name in ["zeta", "agent", "alpha", *qq.SEGMENTS]:
                (d / f"{name}.txt").write_text("x" * 20, encoding="utf-8")
            st = self.run_ids(d, 10)
        self.assertEqual([s["name"] for s in st["segments"]], [*qq.SEGMENTS, "alpha", "zeta"])
        self.assertTrue(all(len(s["ids"]) == 10 for s in st["segments"]))

    def test_a_text_shorter_than_seg_tokens_stops(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "xl"
            d.mkdir()
            (d / "docs.txt").write_text("short", encoding="utf-8")
            with self.assertRaises(SystemExit):
                self.run_ids(d, 1000)


class EngineArgs(unittest.TestCase):
    def test_without_the_flag_the_yardstick_fixes_the_engine(self):
        args = qq.engine_args(CFG, False, 1000)
        self.assertEqual(value(args, "--expert-cache"), ["1"])
        self.assertNotIn("--resident-experts", args)
        self.assertNotIn("--resident-cpu-experts", args)
        self.assertEqual(value(args, "--kv"), ["fp16"])
        self.assertEqual(value(args, "--max-context"), ["1512"])
        self.assertEqual(value(args, "--prefill"), ["512"])
        self.assertNotIn("--vision", args)

    def test_with_the_flag_the_config_engine_stays(self):
        args = qq.engine_args(CFG, False, 1000, keep_caches=True)
        self.assertEqual(value(args, "--expert-cache"), ["auto"])
        self.assertIn("--resident-experts", args)
        self.assertIn("--resident-cpu-experts", args)
        self.assertEqual(value(args, "--max-context"), ["262144"])
        self.assertEqual(value(args, "--kv"), ["int8"])
        self.assertEqual(value(args, "--prefill"), ["4096"])
        self.assertIn("--vision", args)
        self.assertEqual(args[:2], ["/e/strata", "--serve"])

    def test_with_the_flag_only_what_the_dump_needs_is_fixed(self):
        args = qq.engine_args(CFG, False, 1000, keep_caches=True)
        self.assertEqual(value(args, "--prompt-cache"), ["0"])
        self.assertEqual(value(args, "--adapt-every"), ["100000"])
        self.assertEqual(value(args, "--pcie-frac"), ["0"])
        self.assertEqual(value(args, "--suffix-draft"), ["0"])
        self.assertEqual(value(args, "--short-read"), [])
        self.assertEqual(value(args, "--expert-profile-save"), [])

    def test_with_the_flag_and_no_expert_cache_in_the_config_the_engine_default_holds(self):
        cfg = {"exe": "/e", "args": ["--pack", "/p"]}
        self.assertEqual(value(qq.engine_args(cfg, False, 1000, keep_caches=True), "--expert-cache"), [])

    def test_the_tier_arguments_stay_either_way(self):
        for keep in (False, True):
            args = qq.engine_args(CFG, False, 1000, keep_caches=keep)
            self.assertEqual(value(args, "--peer-device"), ["hip:0"])
            self.assertEqual(value(args, "--expert-gib"), ["cuda0=1,hip0=6.5"])

    def test_gpu_experts_and_keep_caches_exclude_each_other(self):
        with self.assertRaises(ValueError):
            qq.engine_args(CFG, True, 1000, keep_caches=True)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()), \
             mock.patch.object(sys, "argv", ["qq", "prompt", "--config", "c", "--out", "o", "--keep-caches",
                                             "--gpu-experts"]):
            qq.main()


class PromptSet(unittest.TestCase):
    def test_the_short_prompts_come_first_as_before(self):
        reqs = qq.prompt_set(the_set())
        short = [r for r in reqs if r["set"] == "short"]
        self.assertEqual(reqs[:len(short)], short)
        self.assertEqual([(r["segment"], r["tokens"]) for r in short],
                         [(s, n) for s in qq.SEGMENTS for n in qq.PROMPT_LENS])
        self.assertTrue(all(len(r["ids"]) == r["tokens"] for r in short))

    def test_the_long_prompts_reach_past_one_4096_chunk(self):
        long = [r for r in qq.prompt_set(the_set()) if r["set"] == "long"]
        self.assertGreaterEqual(len(long), 8)
        for r in long:
            self.assertEqual(len(r["ids"]), r["tokens"])
            self.assertTrue(4500 <= len(r["ids"]) <= 6000, len(r["ids"]))
            self.assertGreaterEqual(len(r["ids"]) - 4096, 400)
            self.assertEqual(len(set(r["ids"])), len(r["ids"]))   # no token of the set read twice
        self.assertEqual(len({r["start"] for r in long}), len(long))

    def test_a_set_too_short_for_the_long_prompts_stops(self):
        with self.assertRaises(SystemExit):
            qq.prompt_set(the_set(500))


class PromptSetV2(unittest.TestCase):
    def v2(self, warm: bool = True) -> list[dict]:
        return qq.prompt_set(the_v1_set(), "v2", the_xl_set(), warm=warm)

    def test_a_warm_request_then_20_short_30_long_and_7_xl(self):
        reqs = self.v2()
        self.assertEqual([r["set"] for r in reqs], ["warm"] + ["short"] * 20 + ["long"] * 30 + ["xl"] * 7)
        warm = reqs[0]
        self.assertEqual((warm["segment"], warm["tokens"], len(warm["ids"])), ("docs", 4800, 4800))
        docs = the_xl_set()["segments"][1]["ids"]
        self.assertEqual(warm["ids"], docs[warm["start"]:warm["start"] + 4800])

    def test_no_warm_drops_only_the_warm_request(self):
        self.assertEqual(self.v2(warm=False), self.v2()[1:])

    def test_the_short_prompts_are_v1s(self):
        short = [r for r in self.v2() if r["set"] == "short"]
        self.assertEqual(short, [r for r in qq.prompt_set(the_v1_set()) if r["set"] == "short"])

    def test_the_long_prompts_keep_v1s_ten(self):
        long = [r for r in self.v2() if r["set"] == "long"]
        v1 = [r for r in qq.prompt_set(the_v1_set()) if r["set"] == "long"]
        joined = [t for s in the_v1_set()["segments"] for t in s["ids"]]
        self.assertEqual(len({r["start"] for r in long}), 30)
        self.assertEqual(long[::3], v1)
        for r in long:
            self.assertLessEqual(r["tokens"], len(joined))     # no token of the set read twice
            self.assertEqual(r["ids"], (joined[r["start"]:] + joined[:r["start"]])[:r["tokens"]])
            self.assertGreaterEqual(r["tokens"] - 4096, 400)
        self.assertEqual({(r["start"], r["tokens"]) for r in v1} & {(r["start"], r["tokens"]) for r in long},
                         {(r["start"], r["tokens"]) for r in v1})

    def test_seven_xl_prompts_three_per_text_and_one_of_24300(self):
        xl = [r for r in self.v2() if r["set"] == "xl"]
        self.assertEqual(len(xl), 7)
        self.assertEqual(sorted(r["tokens"] for r in xl), [20000, 20000, 20160, 20160, 20320, 20320, 24300])
        for name in ("code", "docs"):
            self.assertEqual(sorted(r["tokens"] for r in xl if r["segment"] == name and r["tokens"] < 24300),
                             [20000, 20160, 20320])
        self.assertEqual(len({(r["segment"], r["start"]) for r in xl}), 7)
        texts = {s["name"]: s["ids"] for s in the_xl_set()["segments"]}
        for r in xl:
            t = texts[r["segment"]]
            self.assertEqual(r["ids"], (t[r["start"]:] + t[:r["start"]])[:r["tokens"]])
            self.assertEqual(len(set(r["ids"])), r["tokens"])

    def test_the_warm_request_shares_no_start_with_a_scored_docs_prompt(self):
        reqs = self.v2()
        self.assertNotIn(reqs[0]["start"], {r["start"] for r in reqs if r["set"] == "xl" and r["segment"] == "docs"})

    def test_v2_needs_the_xl_set(self):
        with self.assertRaises(SystemExit):
            qq.prompt_set(the_v1_set(), "v2", None)

    def test_an_xl_text_shorter_than_24300_tokens_stops(self):
        with self.assertRaises(SystemExit):
            qq.prompt_set(the_v1_set(), "v2", the_xl_set(docs=24000))


class PromptEngine(unittest.TestCase):
    CFG3 = {"exe": "/e/strata", "args": ["--pack", "/p", "--expert-cache", "auto", "--peer-device", "hip:0",
                                         "--max-context", "262144", "--kv", "int8", "--prefill", "4096"],
            "gpu": [0, 1, 2], "layer_split": "auto", "env": {"STRATA_XTIER_LIB": "/x.so"}}

    def ns(self, **kw) -> argparse.Namespace:
        return argparse.Namespace(**{"gpu_experts": False, "keep_caches": False, "env": None, **kw})

    def test_keep_caches_starts_the_engine_as_the_server_does(self):
        with mock.patch("serve.server.gpu_speed_scores", return_value={0: 3.0, 1: 2.0, 2: 1.0}), \
             mock.patch("serve.server.gpu_vram_mib", return_value={0: 24576, 1: 12288, 2: 8192}), \
             mock.patch("serve.server.drafting_card_class", return_value="other"), \
             contextlib.redirect_stdout(io.StringIO()):
            args, env = qq.prompt_engine(self.CFG3, self.ns(keep_caches=True), 24300)
        self.assertEqual(value(args, "--layer-split"), ["auto"])
        self.assertEqual(value(args, "--expert-cache"), ["auto"])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,1,0")
        self.assertEqual(env["STRATA_XTIER_LIB"], "/x.so")
        self.assertEqual(value(args, "--prefill"), ["4096"])
        self.assertEqual(value(args, "--max-context"), ["262144"])
        self.assertEqual(value(args, "--kv"), ["int8"])
        self.assertEqual(value(args, "--short-read"), [])

    def test_without_keep_caches_the_context_holds_the_longest_prompt(self):
        args, env = qq.prompt_engine(self.CFG3, self.ns(env=["STRATA_X=1"]), 24300)
        self.assertEqual(value(args, "--layer-split"), [])
        self.assertEqual(value(args, "--expert-cache"), ["1"])
        self.assertEqual(value(args, "--max-context"), [str(24300 + 512)])
        self.assertEqual(value(args, "--prefill"), ["auto"])
        self.assertEqual(value(args, "--short-read"), [])
        self.assertEqual(env["STRATA_X"], "1")
        self.assertEqual(env["STRATA_PREFILL_CPU_SHARE"], "0")


class RunKeepCaches(unittest.TestCase):
    CFG = {"exe": "/e/strata", "args": ["--pack", "/p", "--expert-cache", "auto", "--peer-device", "hip:0",
                                        "--max-context", "262144", "--kv", "int8", "--prefill", "24576",
                                        "--resident-experts", "--pcie-frac", "0.55", "--adapt-every", "4"],
           "gpu": [0], "env": {"STRATA_XTIER_LIB": "/x.so"}}

    def ns(self, **kw) -> argparse.Namespace:
        return argparse.Namespace(**{"gpu_experts": False, "keep_caches": True, "allow": [], "env": None, **kw})

    def run_engine(self, **kw) -> tuple[list[str], dict]:
        with mock.patch("serve.server.gpu_speed_scores", return_value={0: 3.0}), \
             mock.patch("serve.server.gpu_vram_mib", return_value={0: 24576}), \
             mock.patch("serve.server.drafting_card_class", return_value="other"), \
             contextlib.redirect_stdout(io.StringIO()):
            return qq.run_engine(self.CFG, self.ns(**kw), 1000)

    def test_the_config_engine_reads_every_segment_through_the_decode_windows(self):
        args, env = self.run_engine()
        self.assertEqual(value(args, "--short-read"), ["1064"])
        self.assertIn("--resident-experts", args)
        self.assertEqual(value(args, "--peer-device"), ["hip:0"])
        self.assertEqual(value(args, "--kv"), ["int8"])
        self.assertEqual(value(args, "--prefill"), ["24576"])
        self.assertEqual(value(args, "--max-context"), ["262144"])
        self.assertEqual(value(args, "--expert-cache"), ["auto"])
        self.assertEqual(env["STRATA_XTIER_LIB"], "/x.so")

    def test_only_the_fixed_switches_are_dropped(self):
        args, env = self.run_engine()
        self.assertEqual(value(args, "--pcie-frac"), ["0"])
        self.assertEqual(value(args, "--adapt-every"), ["100000"])
        self.assertEqual(value(args, "--prompt-cache"), ["0"])
        self.assertNotIn("STRATA_ROUTE_TAIL_SKIP", env)   # the engine removed the switch

    def test_allow_pcie_keeps_the_config_share(self):
        args, _ = self.run_engine(allow=["pcie"])
        self.assertEqual(value(args, "--pcie-frac"), ["0.55"])
        self.assertEqual(value(args, "--adapt-every"), ["100000"])

    def test_allow_pcie_leaves_the_engine_its_default_rule(self):
        _, env = self.run_engine(allow=["pcie"])
        self.assertNotIn("STRATA_PCIE_FRAC_DEFAULT", env)
        _, env = self.run_engine()
        self.assertEqual(env["STRATA_PCIE_FRAC_DEFAULT"], "old")

    def test_allow_pcie_without_a_share_in_the_config_adds_none(self):
        args = qq.engine_args({"exe": "/e", "args": ["--pack", "/p"]}, False, 1000, keep_caches=True,
                              allow=["pcie"], short_read=1064)
        self.assertEqual(value(args, "--pcie-frac"), [])
        self.assertEqual(value(args, "--short-read"), ["1064"])

    def test_allow_needs_keep_caches(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()), \
             mock.patch.object(sys, "argv", ["qq", "run", "--config", "c", "--out", "o", "--allow", "pcie"]):
            qq.main()


def pcompare_text(ref: Path, run: Path, per_prompt: bool = False) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        qq.cmd_pcompare(argparse.Namespace(ref=str(ref), runs=[str(run)], per_prompt=per_prompt))
    return out.getvalue()


class Pcompare(unittest.TestCase):
    def write(self, d: Path, reqs: list[dict], logits: list[list[float]]) -> None:
        import numpy as np
        d.mkdir()
        (d / "index.json").write_text(json.dumps({"requests": reqs}))
        for i, x in enumerate(logits):
            np.asarray(x, dtype=np.float32).tofile(d / f"first.{i}")

    def run_pcompare(self, ref: Path, run: Path) -> dict:
        rows = {}
        for line in pcompare_text(ref, run).splitlines()[1:]:
            f = line.split()
            rows[f[1]] = (int(f[3]), float(f[f.index("mean") + 1]))
        return rows

    def test_short_and_long_apart_and_together(self):
        reqs = [{"set": "short", "segment": "code", "tokens": 256}, {"set": "long", "segment": "joined",
                                                                       "start": 0, "tokens": 4600}]
        with tempfile.TemporaryDirectory() as t:
            self.write(Path(t) / "ref", reqs, [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
            self.write(Path(t) / "run", reqs, [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
            rows = self.run_pcompare(Path(t) / "ref", Path(t) / "run")
        self.assertEqual(rows["short"], (1, 0.0))
        self.assertEqual(rows["long"][0], 1)
        self.assertGreater(rows["long"][1], 0.0)
        self.assertEqual(rows["all"][0], 2)
        self.assertAlmostEqual(rows["all"][1], rows["long"][1] / 2, places=5)

    def test_an_old_reference_is_matched_by_request_not_by_file_number(self):
        old = [{"segment": "code", "tokens": 256}, {"segment": "code", "tokens": 512}]
        new = [{"set": "short", "segment": "code", "tokens": 512}, {"set": "short", "segment": "code", "tokens": 256},
               {"set": "long", "segment": "joined", "start": 0, "tokens": 4600}]
        with tempfile.TemporaryDirectory() as t:
            self.write(Path(t) / "ref", old, [[0.0, 1.0], [1.0, 0.0]])
            self.write(Path(t) / "run", new, [[1.0, 0.0], [0.0, 1.0], [5.0, 0.0]])
            rows = self.run_pcompare(Path(t) / "ref", Path(t) / "run")
        self.assertEqual(rows["short"], (2, 0.0))
        self.assertEqual(rows["long"][0], 0)


L2, L8 = math.log(2), math.log(8)
A, B, C, D = [L2, 0.0, 0.0], [L8, 0.0, 0.0], [0.0, L2, 0.0], [0.0, 0.0, 0.0]
NAN = [float("nan"), 0.0, 0.0]
WARM = {"set": "warm", "segment": "docs", "start": 15000, "tokens": 4800}
S256 = {"set": "short", "segment": "code", "tokens": 256}
S512 = {"set": "short", "segment": "code", "tokens": 512}
SD256 = {"set": "short", "segment": "docs", "tokens": 256}
L0 = {"set": "long", "segment": "joined", "start": 0, "tokens": 4600}
L1 = {"set": "long", "segment": "joined", "start": 166, "tokens": 4613}
L2R = {"set": "long", "segment": "joined", "start": 333, "tokens": 4626}
X0 = {"set": "xl", "segment": "code", "start": 0, "tokens": 20000}


def write_dir(d: Path, reqs: list[dict], logits: list[list[float]]) -> None:
    import numpy as np
    d.mkdir()
    (d / "index.json").write_text(json.dumps({"version": "v2", "requests": reqs}))
    for i, x in enumerate(logits):
        np.asarray(x, dtype=np.float32).tofile(d / f"first.{i}")


class PcomparePerPrompt(unittest.TestCase):
    """KL(B || A) = .8 ln 1.6 + .2 ln .4, KL(A || C) = .25 ln 2, KL(B || C) = .8 ln 3.2 + .1 ln .2 + .1 ln .4."""
    KL_BA = .8 * math.log(1.6) + .2 * math.log(.4)
    KL_AC = .25 * L2
    KL_BC = .8 * math.log(3.2) + .1 * math.log(.2) + .1 * math.log(.4)
    H_A, H_B, H_D = 1.5 * L2, -(.8 * math.log(.8) + .2 * math.log(.1)), math.log(3)

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.ref, self.run = Path(self.t.name) / "ref", Path(self.t.name) / "run"
        write_dir(self.ref, [WARM, S256, S512, SD256, L0, L1, L2R, X0], [B, A, B, D, A, B, B, B])
        # the run lists its requests in another order, and its warm logits are not finite
        write_dir(self.run, [X0, L2R, L1, L0, SD256, S512, S256, WARM], [C, A, B, C, D, A, A, NAN])

    def tearDown(self):
        self.t.cleanup()

    def text(self) -> str:
        return pcompare_text(self.ref, self.run, per_prompt=True)

    def test_the_rows_are_hand_computed(self):
        rows = [ln.split() for ln in self.text().splitlines() if ln.startswith("  ") and ln.split()[0] != "set"]
        expect = [("short", "code", 0, 256, 0.0, "yes", self.H_A, .5),
                  ("short", "code", 0, 512, self.KL_BA, "yes", self.H_B, .8),
                  ("short", "docs", 0, 256, 0.0, "yes", self.H_D, 1 / 3),
                  ("long", "joined", 0, 4600, self.KL_AC, "no", self.H_A, .5),
                  ("long", "joined", 166, 4613, 0.0, "yes", self.H_B, .8),
                  ("long", "joined", 333, 4626, self.KL_BA, "yes", self.H_B, .8),
                  ("xl", "code", 0, 20000, self.KL_BC, "no", self.H_B, .8)]
        self.assertEqual(len(rows), len(expect))
        for got, (s, seg, start, n, kl, top1, h, p1) in zip(rows, expect):
            self.assertEqual(got[:4], [s, seg, str(start), str(n)])
            self.assertAlmostEqual(float(got[4]), kl, places=5)
            self.assertEqual(got[5], top1)
            self.assertAlmostEqual(float(got[6]), h, places=3)
            self.assertAlmostEqual(float(got[7]), p1, places=3)

    def test_the_sets_and_the_v1_keys_alone(self):
        sums = {}
        for ln in self.text().splitlines():
            f = ln.split()
            if "prompts" in f:
                sums[f[1]] = (int(f[3]), float(f[5].rstrip("%")), float(f[f.index("mean") + 1]),
                              float(f[f.index("median") + 1]))
        self.assertEqual(set(sums), {"short", "long", "xl", "all", "v1"})
        self.assertEqual(sums["short"][:2], (3, 100.0))
        self.assertAlmostEqual(sums["short"][2], self.KL_BA / 3, places=5)
        self.assertEqual(sums["long"][:2], (3, 66.67))
        self.assertAlmostEqual(sums["long"][2], (self.KL_AC + self.KL_BA) / 3, places=5)
        self.assertAlmostEqual(sums["long"][3], self.KL_AC, places=5)
        self.assertEqual(sums["xl"][:2], (1, 0.0))
        self.assertAlmostEqual(sums["xl"][2], self.KL_BC, places=5)
        self.assertEqual(sums["all"][:2], (6, 83.33))
        self.assertAlmostEqual(sums["all"][2], (2 * self.KL_BA + self.KL_AC) / 6, places=5)
        self.assertEqual(sums["v1"][:2], (4, 75.0))
        self.assertAlmostEqual(sums["v1"][2], (self.KL_BA + self.KL_AC) / 4, places=5)

    def test_the_warm_request_is_never_scored(self):
        text = self.text()
        self.assertNotIn("warm", text)
        self.assertNotIn("nan", text.lower())
        self.assertNotIn("missing", text)


class Gate(unittest.TestCase):
    REQS = [WARM, S256, S512, SD256, L0, L1, L2R, X0]
    REF = [B, A, B, D, A, B, B, B]

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)
        write_dir(self.root / "ref", self.REQS, self.REF)

    def tearDown(self):
        self.t.cleanup()

    def make(self, name: str, change: dict | None = None, reqs: list[dict] | None = None) -> str:
        logits = [list(x) for x in self.REF]
        for i, x in (change or {}).items():
            logits[i] = x
        reqs = reqs or self.REQS
        write_dir(self.root / name, reqs, [logits[self.REQS.index(q)] for q in reqs])
        return str(self.root / name)

    def gate(self, *runs: str, xl_ref: str | None = None, ref: str | None = None) -> tuple[int, list[tuple]]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = qq.cmd_gate(argparse.Namespace(ref=ref or str(self.root / "ref"), runs=list(runs), xl_ref=xl_ref))
        fails = [tuple(ln.split()[:2]) for ln in out.getvalue().splitlines() if ln.rstrip().endswith("FAIL")
                 and ln.startswith("  ")]
        return rc, fails

    def test_a_run_equal_to_the_reference_passes(self):
        self.assertEqual(self.gate(self.make("same")), (0, []))

    def test_two_runs_must_both_pass(self):
        good, bad = self.make("good"), self.make("bad", {6: C})
        self.assertEqual(self.gate(good, good)[0], 0)
        self.assertEqual(self.gate(good, bad)[0], 1)
        self.assertEqual(self.gate(bad, good)[0], 1)

    def test_the_long_mean_past_its_line_fails(self):
        rc, fails = self.gate(self.make("r", {5: A}))       # KL(B || A) = 0.193 on one long prompt, top-1 the same
        self.assertEqual(rc, 1)
        self.assertIn(("long", "mean"), fails)

    def test_the_long_top1_past_its_line_fails(self):
        flip = [math.log(.33), math.log(.34), math.log(.33)]   # KL(A || flip) = 0.061 on one of three long prompts
        rc, fails = self.gate(self.make("r", {4: flip}))
        self.assertEqual(rc, 1)
        self.assertIn(("long", "top-1"), fails)
        self.assertNotIn(("long", "mean"), fails)

    def test_the_all_median_past_its_line_fails(self):
        # the top logit 0.4 higher on every scored prompt: KL 0.012 to 0.020 each, no top-1 changes
        nudge = {i: [x + (0.4 if k == 0 else 0.0) for k, x in enumerate(self.REF[i])] for i in range(1, 8)}
        rc, fails = self.gate(self.make("r", nudge))
        self.assertEqual(rc, 1)
        self.assertEqual(fails, [("all", "median")])

    def test_the_xl_mean_past_its_line_fails(self):
        rc, fails = self.gate(self.make("r", {7: A}))       # KL(B || A), top-1 the same
        self.assertEqual(rc, 1)
        self.assertEqual(fails, [("xl", "mean")])

    def test_an_xl_top1_miss_where_the_reference_is_sure_fails(self):
        sure = [math.log(.52), math.log(.47), math.log(.01)]   # KL(sure || miss) = 0.005, under the mean's line
        miss = [math.log(.47), math.log(.52), math.log(.01)]
        write_dir(self.root / "ref2", self.REQS, self.REF[:7] + [sure])
        run = self.root / "r"
        write_dir(run, self.REQS, self.REF[:7] + [miss])
        rc, fails = self.gate(str(run), ref=str(self.root / "ref2"))
        self.assertEqual(rc, 1)
        self.assertEqual(fails, [("xl", "misses")])

    def test_a_non_finite_logit_fails(self):
        rc, fails = self.gate(self.make("r", {7: NAN}))
        self.assertEqual(rc, 1)
        self.assertIn(("every", "non-finite"), fails)

    def test_a_prompt_the_run_lacks_fails(self):
        rc, fails = self.gate(self.make("r", reqs=self.REQS[:-2] + self.REQS[-1:]))
        self.assertEqual(rc, 1)
        self.assertIn(("every", "missing"), fails)

    def test_a_reference_without_xl_prompts_fails(self):
        write_dir(self.root / "ref1", self.REQS[:-1], self.REF[:-1])
        rc, fails = self.gate(self.make("r"), ref=str(self.root / "ref1"))
        self.assertEqual(rc, 1)
        self.assertIn(("xl", "prompts"), fails)

    def test_the_xl_prompts_can_have_their_own_reference(self):
        write_dir(self.root / "ref1", self.REQS[:-1], self.REF[:-1])
        write_dir(self.root / "refxl", [X0], [C])
        run = self.make("r", {7: C})
        self.assertEqual(self.gate(run, ref=str(self.root / "ref1"), xl_ref=str(self.root / "refxl")), (0, []))
        self.assertEqual(self.gate(run, ref=str(self.root / "ref"))[0], 1)


class PromptRun(unittest.TestCase):
    """`prompt` against a fake engine that answers each GEN line with logits [prompt length, 0, 0]."""
    FAKE = (f"#!{sys.executable}\n"
            "import os, struct, sys\n"
            "n = 0\n"
            "print('READY', flush=True)\n"
            "for line in sys.stdin:\n"
            "    if line.startswith('QUIT'):\n"
            "        break\n"
            "    k = len(line.split()[2].split(','))\n"
            "    if k == int(os.environ.get('FAKE_NONFINITE_AT', 0)):\n"
            "        print('DONE 0 0 0 0 nonfinite 0', flush=True)\n"
            "        continue\n"
            "    open(os.environ['STRATA_DUMP_FIRST_LOGITS'] + '.' + str(n), 'wb').write(struct.pack('3f', k, 0, 0))\n"
            "    n += 1\n"
            "    print('DONE', flush=True)\n")

    def run_prompt(self, t: Path, *extra: str) -> dict:
        (t / "fake.py").write_text(self.FAKE)
        (t / "fake.py").chmod(0o755)
        (t / "cfg.json").write_text(json.dumps({"exe": str(t / "fake.py"), "args": []}))
        (t / "xl.ids.json").write_text(json.dumps(the_xl_set()))
        argv = ["qq", "prompt", "--config", str(t / "cfg.json"), "--out", str(t / "out"), "--xl-ids",
                str(t / "xl.ids.json"), "--min-free-gb", "0", *extra]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(qq.main(), 0)
        return json.loads((t / "out" / "index.json").read_text())

    def test_v2_is_the_default_and_the_warm_request_runs_first(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as t:
            index = self.run_prompt(Path(t))
            self.assertEqual(index["version"], "v2")
            reqs = index["requests"]
            self.assertEqual(len(reqs), 58)
            self.assertEqual(reqs[0]["set"], "warm")
            for i, q in enumerate(reqs):
                self.assertEqual(np.fromfile(Path(t) / "out" / f"first.{i}", dtype=np.float32)[0], q["tokens"])
            self.assertEqual(len(qq.prompt_keys(Path(t) / "out")), 57)

    def test_a_run_the_engine_ends_keeps_the_prompts_before_it(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.dict("os.environ", {"FAKE_NONFINITE_AT": "24300"}):
            with self.assertRaises(SystemExit):
                self.run_prompt(Path(t))
            index = json.loads((Path(t) / "out" / "index.json").read_text())
            self.assertEqual(len(index["requests"]), 57)
            self.assertEqual(index["failed"]["tokens"], 24300)
            self.assertIn("nonfinite", index["failed"]["answer"])
            self.assertEqual(len(qq.prompt_keys(Path(t) / "out")), 56)

    def test_no_warm_and_v1(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(len(self.run_prompt(Path(t), "--no-warm")["requests"]), 57)
        with tempfile.TemporaryDirectory() as t:
            index = self.run_prompt(Path(t), "--set", "v1")
            self.assertEqual((index["version"], len(index["requests"])), ("v1", 30))


if __name__ == "__main__":
    unittest.main()
