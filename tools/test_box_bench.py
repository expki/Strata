"""Tests for tools/box_bench.py (no GPU, no server): --expect, the keep verdict, the clock summary, and the flow of a
bench run with a fake server that writes Task 16's RE24 engine log (tools/fixtures/engine-log-re24.txt).

    python -m pytest tools/test_box_bench.py
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_bench  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"
PASSES = ["--pass", "warm:chat0,p8k", "--pass", "m:chat0,chat1,chat2,p2k,p8k,p20k"]
READ = {"chat0": 34, "chat1": 27, "chat2": 29, "p2k": 2041, "p8k": 8013, "p20k": 20053}   # as the RE24 log read them


class Expect(unittest.TestCase):
    def test_values_ranges_and_open_ends(self):
        self.assertEqual(box_bench.parse_expect("chunk=24576,pinned_gib=95..100,vram_low=256..,w6800_experts=..7300"),
                         {"chunk": (24576, 24576), "pinned_gib": (95, 100), "vram_low": (256, None),
                          "w6800_experts": (None, 7300)})

    def test_booleans(self):
        self.assertEqual(box_bench.parse_expect("tier_in_prompts=1")["tier_in_prompts"], (1, 1))
        self.assertEqual(box_bench.parse_expect("tier_in_prompts=false")["tier_in_prompts"], (0, 0))

    def test_an_unknown_fact_or_a_bad_pair_is_refused(self):
        for bad in ("chunks=4096", "chunk", "chunk=", "chunk=big"):
            with self.assertRaises(ValueError):
                box_bench.parse_expect(bad)

    def test_mismatches(self):
        f = {"chunk": 24576, "pinned_gib": 98.17, "tier_in_prompts": True, "cached_share": None}
        self.assertEqual(box_bench.mismatches(f, box_bench.parse_expect("chunk=24576,pinned_gib=95..100,"
                                                                        "tier_in_prompts=1")), [])
        self.assertEqual(box_bench.mismatches(f, box_bench.parse_expect("chunk=8192")), ["chunk is 24576, not 8192"])
        self.assertEqual(box_bench.mismatches(f, box_bench.parse_expect("pinned_gib=..90")),
                         ["pinned_gib is 98.17, not ..90"])
        self.assertEqual(box_bench.mismatches(f, box_bench.parse_expect("tier_in_prompts=0")),
                         ["tier_in_prompts is True, not 0"])
        self.assertEqual(box_bench.mismatches(f, box_bench.parse_expect("cached_share=0.9..1")),
                         ["cached_share is not in the log"])


class Keep(unittest.TestCase):
    def test_min_above_max_but_a_gap_under_the_spread_is_not_resolved(self):
        self.assertEqual(box_bench.keep([100, 110], [111, 112]), "NOT RESOLVED")   # gap 6.5, spread 10

    def test_a_clear_gap_keeps(self):
        self.assertEqual(box_bench.keep([100, 102], [110, 111]), "KEEP")

    def test_a_gap_equal_to_the_spread_keeps(self):
        self.assertEqual(box_bench.keep([100, 104], [106, 106]), "KEEP")           # gap 4, spread 4

    def test_overlapping_starts_are_not_resolved(self):
        self.assertEqual(box_bench.keep([100, 120], [110, 130]), "NOT RESOLVED")

    def test_the_mirror_of_keep_is_slower(self):
        self.assertEqual(box_bench.keep([110, 111], [100, 102]), "SLOWER")

    def test_not_slower_takes_the_smaller_of_the_spread_and_5_percent(self):
        self.assertFalse(box_bench.not_slower([100, 102], [94.9, 95.0]))           # 6% lower, PREV spread 2%
        self.assertTrue(box_bench.not_slower([100, 102], [99.0, 99.1]))            # floor 99
        self.assertTrue(box_bench.not_slower([80, 120], [95.1, 95.1]))             # floor 100 - 5


def req(arm, start, tag, kind="m", **kw):
    return {"row": "req", "arm": arm, "start": start, "kind": kind, "req": tag, "pass": 1, **kw}


class Verdict(unittest.TestCase):
    def run_verdict(self, rows, *flags):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = box_bench.main(["verdict", str(p), "--a", "PREV", "--b", "NEW", *flags])
        return code, out.getvalue()

    def rows(self, prev, new, other_prev=(30, 30), other_new=(30, 30)):
        rows = []
        for arm, vals, others in (("PREV", prev, other_prev), ("NEW", new, other_new)):
            for i, (v, o) in enumerate(zip(vals, others)):
                s = f"{arm}{i}"
                rows += [{"row": "ready", "arm": arm, "start": s, "ready_s": 120, "time": "t"},
                         req(arm, s, "p20k", kind="warm", prompt=1.0),
                         req(arm, s, "p20k", prompt=v, drive_mb=79, cached_mb=180000, avail_mb=78000),
                         req(arm, s, "chat1", decode=o)]
        return rows

    def test_not_resolved(self):
        code, out = self.run_verdict(self.rows([100, 110], [111, 112]), "--target", "p20k.prompt")
        self.assertEqual(code, 1)
        self.assertIn("verdict: NOT RESOLVED", out)
        self.assertIn("gap +6.5", out)

    def test_keep_prints_every_start(self):
        code, out = self.run_verdict(self.rows([100, 102], [110, 111]), "--target", "p20k.prompt",
                                     "--other", "chat1.decode")
        self.assertEqual(code, 0)
        self.assertIn("verdict: KEEP", out)
        for s in ("PREV0", "PREV1", "NEW0", "NEW1"):
            self.assertIn(f"start {s}:", out)
        self.assertEqual(out.count("drives MB [79], Cached MB [180000], MemAvailable MB [78000]"), 4)

    def test_the_warm_pass_does_not_count(self):
        _, out = self.run_verdict(self.rows([100, 102], [110, 111]), "--target", "p20k.prompt")
        self.assertIn("PREV mean 101.0", out)

    def test_an_other_metric_6_percent_lower_is_not_keep(self):
        code, out = self.run_verdict(self.rows([100, 102], [110, 111], other_prev=(100, 102), other_new=(95, 94.9)),
                                     "--target", "p20k.prompt", "--other", "chat1.decode")
        self.assertEqual(code, 3)
        self.assertIn("other chat1.decode", out)
        self.assertIn("verdict: SLOWER", out)

    def test_a_start_is_the_mean_of_its_passes(self):
        rows = self.rows([100, 102], [110, 111])
        rows.append(req("NEW", "NEW0", "p20k", prompt=130))                  # NEW0 becomes (110 + 130) / 2
        _, out = self.run_verdict(rows, "--target", "p20k.prompt")
        self.assertIn("p20k.prompt 120.0 [110, 130]", out)

    def test_one_start_an_arm_is_an_error(self):
        code, out = self.run_verdict(self.rows([100], [110]), "--target", "p20k.prompt")
        self.assertEqual(code, 2)
        self.assertIn("verdict: error", out)


class ClockSummary(unittest.TestCase):
    def test_the_lock_held_and_the_power_while_reading_long_prompts(self):
        s = [{"tag": "A:start", "nv_clk": 210, "nv_w": 30, "nv_use": 0},
             {"tag": "A:1:chat0", "nv_clk": 1500, "nv_w": 80, "nv_use": 90},
             {"tag": "A:1:p20k", "nv_clk": 1500, "nv_w": 130, "nv_use": 95},
             {"tag": "A:1:p20k", "nv_clk": 1500, "nv_w": 110, "nv_use": 20}]
        c = box_bench.clock_summary(s)
        self.assertEqual((c["samples"], c["clk_max"], c["long_w_mean"], c["long_w_max"]), (4, 1500, 120.0, 130))
        self.assertEqual((c["busy_clk_min"], c["busy_clk_max"], c["clock_over"]), (1500, 1500, False))

    def test_a_clock_above_the_lock_is_flagged(self):
        self.assertTrue(box_bench.clock_summary([{"tag": "A:0:chat0", "nv_clk": 1695, "nv_w": 90}])["clock_over"])


class FakeProc:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class FakeSampler:
    """Takes one 1500 MHz sample each time the bench sets its tag."""

    def __init__(self, path):
        self.rows: list[dict] = []
        self.start_id = None
        self._tag = "idle"
        self.stop = mock.Mock()

    def start(self):
        pass

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, v):
        self._tag = v
        self.rows.append({"start": self.start_id, "tag": v, "nv_clk": 1500, "nv_w": 100, "nv_use": 90})


class Flow(unittest.TestCase):
    """run_bench with a fake server: Popen writes the RE24 engine log, post answers with its prompt sizes."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.base = self.dir / "base.json"
        self.base.write_text(json.dumps({"exe": "/base", "args": ["--base"], "port": 1}))
        self.engine_log = FIX / "engine-log-re24.txt"
        self.started: list = []      # (cfg, env, proc) of every start
        self.posts: list = []        # the request tags in the order they were sent
        self.read = dict(READ)

    def popen(self, argv, cwd=None, env=None, **kw):
        cfg = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
        shutil.copy(self.engine_log, cfg["log"])
        proc = FakeProc()
        self.started.append((cfg, env, proc))
        return proc

    def post(self, port, prompt, max_tokens):
        tag = next(t for t in READ if box_bench.request_text(t) == prompt)
        self.posts.append(tag)
        return {"timings": {"prompt_n": self.read[tag], "cache_n": 0, "prompt_per_second": 1000.0,
                            "predicted_per_second": 40.0, "predicted_n": 128},
                "choices": [{"message": {"content": f"answer to {tag}"}}]}

    def run_main(self, *argv):
        log = self.dir / "out.jsonl"
        out = io.StringIO()
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(box_bench.subprocess, "Popen", self.popen))
            st.enter_context(mock.patch.object(box_bench, "wait_ready", lambda *a, **k: None))
            st.enter_context(mock.patch.object(box_bench, "post", self.post))
            st.enter_context(mock.patch.object(box_bench, "Sampler", FakeSampler))
            st.enter_context(mock.patch.object(box_bench, "drive_read_mb", lambda: 0.0))
            st.enter_context(mock.patch.object(box_bench, "meminfo_mb", lambda key: 1024.0))
            st.enter_context(mock.patch.object(box_bench.time, "sleep", lambda s: None))
            st.enter_context(mock.patch.object(box_bench.tempfile, "gettempdir", lambda: str(self.dir)))
            st.enter_context(contextlib.redirect_stdout(out))
            code = box_bench.main([str(self.dir), str(self.base), *argv, "--log", str(log)])
        rows = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
        return code, rows, out.getvalue()

    def test_the_protocol_passes_with_the_facts_expected(self):
        code, rows, out = self.run_main("A:", *PASSES, "--expect", "chunk=24576,pinned_gib=95..100")
        self.assertEqual(code, 0)
        kinds = [r["row"] for r in rows]
        self.assertEqual(kinds, ["ready", "req", "req", "facts"] + ["req"] * 6 + ["facts"] + ["engine"] * 8)
        check = rows[3]
        self.assertEqual((check["at"], check["ok"], check["chunk"], check["pinned_gib"]), ("check", True, 24576, 98.17))
        self.assertEqual(rows[10]["at"], "end")
        self.assertEqual(len({r["start"] for r in rows}), 1)
        self.assertEqual([r["req"] for r in rows if r["row"] == "req"], ["chat0", "p8k", "chat0", "chat1", "chat2",
                                                                          "p2k", "p8k", "p20k"])
        self.assertEqual({r["kind"] for r in rows if r["row"] == "req"}, {"warm", "m"})
        for key in ("pass", "prompt", "decode", "drive_mb", "cached_mb", "avail_mb", "prompt_n", "text"):
            self.assertIn(key, rows[1])

    def test_engine_rows_carry_the_engine_s_drive_reads_by_request(self):
        _, rows, _ = self.run_main("A:", *PASSES)
        eng = {(r["kind"], r["req"]): r for r in rows if r["row"] == "engine"}
        self.assertEqual((eng[("m", "p20k")]["file_mb"], eng[("m", "p20k")]["ptoks"]), (78.8, 2528.4))
        self.assertEqual(eng[("warm", "chat0")]["file_mb"], 9.1)

    def test_engine_rows_that_do_not_line_up_are_left_unlabeled(self):
        self.read["p2k"] = 2000
        _, rows, out = self.run_main("A:", *PASSES)
        self.assertEqual({r["req"] for r in rows if r["row"] == "engine"}, {None})
        self.assertIn("do not line up", out)

    def test_a_wrong_expect_stops_before_the_measured_passes_with_exit_4(self):
        code, rows, out = self.run_main("A:", "B:", *PASSES, "--expect", "chunk=8192")
        self.assertEqual(code, 4)
        self.assertEqual(self.posts, ["chat0", "p8k"])                      # the warm pass only
        check = next(r for r in rows if r["row"] == "facts" and r["at"] == "check")
        self.assertEqual((check["ok"], check["mismatch"]), (False, ["chunk is 24576, not 8192"]))
        self.assertEqual(len(self.started), 1)                              # B never starts
        self.assertTrue(self.started[0][2].terminated)
        self.assertIn("EXPECT MISMATCH", out)

    def test_a_fact_missing_from_the_log_is_a_mismatch(self):
        code, _, _ = self.run_main("A:", *PASSES, "--expect", "cached_share=0.9..1")
        self.assertEqual(code, 4)

    def test_without_a_measured_pass_the_facts_are_checked_at_the_end(self):
        code, rows, _ = self.run_main("A:", "--pass", "warm:chat0,p8k", "--expect", "chunk=24576")
        self.assertEqual(code, 0)
        self.assertEqual([r["row"] for r in rows][:4], ["ready", "req", "req", "facts"])

    def test_each_start_has_its_own_id_and_a_clock_row_with_sample(self):
        code, rows, _ = self.run_main("A:", "B:", *PASSES, "--sample")
        self.assertEqual(code, 0)
        clocks = [r for r in rows if r["row"] == "clock"]
        self.assertEqual([c["arm"] for c in clocks], ["A", "B"])
        self.assertNotEqual(clocks[0]["start"], clocks[1]["start"])
        self.assertEqual((clocks[0]["clk_max"], clocks[0]["clock_over"]), (1500, False))
        self.assertGreater(clocks[0]["samples"], 0)

    def test_each_arm_takes_its_config_exe_xargs_and_env(self):
        other = self.dir / "other.json"
        other.write_text(json.dumps({"exe": "/other", "args": ["--other"], "port": 2}))
        self.run_main("A:exe=/a,xargs=--x;--y", f"B:config={other},STRATA_FLAG=1", "--pass", "m:chat0")
        (cfg_a, env_a, _), (cfg_b, env_b, _) = self.started
        self.assertEqual((cfg_a["exe"], cfg_a["args"]), ("/a", ["--base", "--x", "--y"]))
        self.assertEqual((cfg_b["exe"], cfg_b["args"]), ("/other", ["--other"]))
        self.assertEqual(env_b["STRATA_FLAG"], "1")
        self.assertNotIn("STRATA_FLAG", env_a)

    def test_keep_logs_copies_the_engine_log(self):
        self.run_main("A:", "--pass", "m:chat0", "--keep-logs", str(self.dir / "t17-x-"))
        self.assertEqual((self.dir / "t17-x-A-engine.log").read_text(), self.engine_log.read_text())

    def test_a_bad_request_or_expect_is_refused_before_any_start(self):
        for flags in (["--pass", "m:p64k"], ["--pass", "m:chat9"], ["--pass", "m:chat0", "--expect", "chunks=1"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                self.run_main("A:", *flags)
            self.assertEqual(cm.exception.code, 2)
        self.assertEqual(self.started, [])


if __name__ == "__main__":
    unittest.main()
