"""Tests for tools/ab_engine.py (no GPU, no server, no sudo): arm parsing, the identity check, the cache drop, and the
flow of main() with a fake server.

    python tools/test_ab_engine.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_engine  # noqa: E402


class ParseArm(unittest.TestCase):
    def test_keys_in_any_order(self):
        self.assertEqual(ab_engine.parse_arm("B:config=/c.json,exe=/e,STRATA_A=1"),
                         ("B", {"config": "/c.json", "exe": "/e", "STRATA_A": "1"}))

    def test_a_bare_name(self):
        self.assertEqual(ab_engine.parse_arm("A:"), ("A", {}))
        self.assertEqual(ab_engine.parse_arm("A"), ("A", {}))

    def test_a_value_keeps_its_equals_signs_and_semicolons(self):
        self.assertEqual(ab_engine.parse_arm("A:xargs=--x=1;--y"), ("A", {"xargs": "--x=1;--y"}))


class FirstDifference(unittest.TestCase):
    def test_same_texts(self):
        self.assertIsNone(ab_engine.first_difference({"A": {"chat0": "x"}, "B": {"chat0": "x"}}))

    def test_different_texts(self):
        self.assertEqual(ab_engine.first_difference({"A": {"chat0": "x"}, "B": {"chat0": "y"}}), ("chat0", "B"))

    def test_every_arm_is_compared_with_the_first(self):
        t = {"A": {"chat0": "x"}, "B": {"chat0": "x"}, "C": {"chat0": "z"}}
        self.assertEqual(ab_engine.first_difference(t), ("chat0", "C"))

    def test_the_first_request_wins_over_the_first_arm(self):
        t = {"A": {"chat0": "x", "chat1": "x"}, "B": {"chat0": "x", "chat1": "y"}, "C": {"chat0": "z", "chat1": "x"}}
        self.assertEqual(ab_engine.first_difference(t), ("chat0", "C"))

    def test_a_missing_request_is_a_difference(self):
        self.assertEqual(ab_engine.first_difference({"A": {"chat0": "x", "long": "y"}, "B": {"chat0": "x"}}),
                         ("long", "B"))
        self.assertEqual(ab_engine.first_difference({"A": {"chat0": "x"}, "B": {"chat0": "x", "long": "y"}}),
                         ("long", "B"))

    def test_one_arm_or_none(self):
        self.assertIsNone(ab_engine.first_difference({"A": {"chat0": "x"}}))
        self.assertIsNone(ab_engine.first_difference({}))


class IdentityVerdict(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(ab_engine.identity_verdict({"A": {"chat0": "x"}, "B": {"chat0": "x"}}),
                         (0, "identity: identical"))

    def test_a_difference(self):
        self.assertEqual(ab_engine.identity_verdict({"A": {"chat0": "x"}, "B": {"chat0": "y"}}),
                         (1, "identity: first difference at chat0 in arm B (against A)"))

    def test_an_empty_table_is_an_error(self):
        code, line = ab_engine.identity_verdict({})
        self.assertEqual(code, 1)
        self.assertIn("identity: error", line)

    def test_an_arm_without_answers_is_an_error(self):
        self.assertEqual(ab_engine.identity_verdict({"A": {"chat0": "x"}, "B": {}}),
                         (1, "identity: error, arm B has no measured answers"))

    def test_an_empty_answer_is_an_error_even_when_both_are_empty(self):
        self.assertEqual(ab_engine.identity_verdict({"A": {"chat0": ""}, "B": {"chat0": ""}}),
                         (1, "identity: error, arm A gave an empty answer at chat0"))


class DropCaches(unittest.TestCase):
    def test_runs_sudo_without_a_prompt(self):
        with mock.patch.object(ab_engine.os, "sync") as sync, mock.patch.object(ab_engine.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stderr="")
            ab_engine.drop_caches()
        sync.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])

    def test_a_refused_sudo_stops_the_run_with_its_message(self):
        with mock.patch.object(ab_engine.os, "sync"), mock.patch.object(ab_engine.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=1, stderr="sudo: a password is required\n")
            with self.assertRaises(SystemExit) as cm:
                ab_engine.drop_caches()
        self.assertIn("sudo: a password is required", str(cm.exception.code))
        self.assertIn("--drop-caches", str(cm.exception.code))

    def test_a_missing_sudo_stops_the_run_too(self):
        with mock.patch.object(ab_engine.os, "sync"), \
                mock.patch.object(ab_engine.subprocess, "run", side_effect=FileNotFoundError("sudo")):
            with self.assertRaises(SystemExit) as cm:
                ab_engine.drop_caches()
        self.assertIn("sudo", str(cm.exception.code))


class FakeProc:
    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class Flow(unittest.TestCase):
    """main() with a fake server: Popen, wait_ready and post are replaced, the log goes to a temp directory."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.base = self.write_cfg("base.json", {"exe": "/base", "args": ["--base"], "port": 1})
        self.events: list = []      # ("drop",), ("start", exe) and ("post", exe, prompt) in the order they happen
        self.started: list = []     # (cfg, env) of every start
        self.cfg: dict = {}
        self.text = lambda arm, prompt, n: f"{arm}:{prompt[:12]}"
        self.calls: dict = {}
        self.nth = 0                # posts since the last start
        self.decode = lambda nth: 10.0

    def write_cfg(self, name, cfg):
        p = self.dir / name
        p.write_text(json.dumps(cfg))
        return str(p)

    def popen(self, argv, cwd=None, env=None, **kw):
        self.cfg = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
        self.started.append((self.cfg, env))
        self.nth = 0
        self.events.append(("start", self.cfg.get("exe")))
        return FakeProc()

    def post(self, port, prompt, max_tokens):
        arm = self.cfg["exe"]
        n = self.calls[(arm, prompt)] = self.calls.get((arm, prompt), 0) + 1
        self.events.append(("post", arm, prompt))
        self.nth += 1
        return {"timings": {"predicted_per_second": self.decode(self.nth), "prompt_per_second": 100.0, "prompt_n": 5},
                "usage": {"completion_tokens": 3}, "choices": [{"message": {"content": self.text(arm, prompt, n)}}]}

    def run_main(self, *argv, drop=None):
        log = self.dir / "log.jsonl"
        args = [self.base, *argv, "--log", str(log)]
        out = io.StringIO()
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(ab_engine.subprocess, "Popen", self.popen))
            st.enter_context(mock.patch.object(ab_engine, "wait_ready", lambda *a, **k: None))
            st.enter_context(mock.patch.object(ab_engine, "post", self.post))
            st.enter_context(mock.patch.object(ab_engine.time, "sleep", lambda s: None))
            st.enter_context(mock.patch.object(ab_engine.tempfile, "gettempdir", lambda: str(self.dir)))
            st.enter_context(mock.patch.object(ab_engine, "drop_caches",
                                               drop or (lambda: self.events.append(("drop",)))))
            st.enter_context(contextlib.redirect_stdout(out))
            code = ab_engine.main(args)
        rows = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
        return code, rows, out.getvalue()

    def test_no_new_flag_keeps_today_s_run(self):
        code, rows, out = self.run_main("A:exe=/a", "B:exe=/b", "--rounds", "1")
        self.assertEqual(code, 0)
        self.assertEqual([(r["arm"], r["req"]) for r in rows],
                         [(a, t) for a in "AB" for t in ("chat0", "chat1", "chat2", "long")])
        for key in ("round", "arm", "req", "decode", "prompt", "prompt_n", "n", "text"):
            self.assertIn(key, rows[0])
        self.assertEqual({r["pass"] for r in rows}, {0})
        self.assertFalse(any(r["cold"] for r in rows))
        self.assertFalse(any(e[0] == "drop" for e in self.events))
        self.assertNotIn("identity", out)
        self.assertNotIn("cold", out)
        self.assertEqual(self.events[-1][2], ab_engine.long_prompt(6000))

    def test_warm_and_measure_passes(self):
        code, rows, _ = self.run_main("A:exe=/a", "--rounds", "1", "--warm", "2", "--measure", "3")
        self.assertEqual(len([e for e in self.events if e[0] == "post"]), (2 + 3) * 4)
        self.assertEqual(len(rows), 3 * 4)
        self.assertEqual(sorted({r["pass"] for r in rows}), [0, 1, 2])
        self.assertFalse(any(r["cold"] for r in rows))
        first_post = next(i for i, e in enumerate(self.events) if e[0] == "post")
        self.assertEqual(first_post, 1)    # the start is event 0 and the warm passes follow

    def test_each_arm_takes_its_own_config(self):
        other = self.write_cfg("other.json", {"exe": "/other", "args": ["--other"], "port": 2})
        self.run_main(f"A:exe=/a,xargs=--x;--y", f"B:config={other},STRATA_FLAG=1", "--rounds", "1")
        (cfg_a, env_a), (cfg_b, env_b) = self.started
        self.assertEqual((cfg_a["exe"], cfg_a["args"]), ("/a", ["--base", "--x", "--y"]))
        self.assertEqual((cfg_b["exe"], cfg_b["args"]), ("/other", ["--other"]))
        self.assertEqual(cfg_a["port"], cfg_b["port"])
        self.assertNotIn("config", env_a)
        self.assertNotIn("config", env_b)
        self.assertEqual(env_b["STRATA_FLAG"], "1")
        self.assertNotIn("STRATA_FLAG", env_a)

    def test_a_missing_arm_config_fails_before_any_start(self):
        with self.assertRaises(FileNotFoundError):
            self.run_main("A:exe=/a", f"B:config={self.dir}/nope.json")
        self.assertEqual(self.started, [])

    def test_drop_caches_before_each_start_and_the_first_request_is_cold(self):
        code, rows, out = self.run_main("A:exe=/a", "B:exe=/b", "--rounds", "2", "--measure", "2", "--drop-caches")
        starts = [i for i, e in enumerate(self.events) if e[0] == "start"]
        self.assertEqual(len(starts), 4)
        for i in starts:
            self.assertEqual(self.events[i - 1], ("drop",))
        cold = [(r["round"], r["arm"], r["pass"], r["req"]) for r in rows if r["cold"]]
        self.assertEqual(cold, [(0, "A", 0, "chat0"), (0, "B", 0, "chat0"),
                                (1, "B", 0, "chat0"), (1, "A", 0, "chat0")])
        self.assertEqual(len([x for x in out.splitlines() if x.endswith("  cold")]), 4)

    def test_the_cold_request_can_fall_in_a_warm_pass_and_is_logged(self):
        _, rows, out = self.run_main("A:exe=/a", "--rounds", "1", "--warm", "2", "--measure", "2", "--drop-caches")
        self.assertEqual([r["pass"] for r in rows], ["warm"] * 8 + [0] * 4 + [1] * 4)
        self.assertEqual([(r["pass"], r["req"]) for r in rows if r["cold"]], [("warm", "chat0")])
        self.assertEqual(rows[0]["req"], "chat0")
        self.assertEqual(len([x for x in out.splitlines() if x.endswith("  pass warm  cold")]), 1)

    def test_with_drop_caches_every_visit_has_one_cold_request(self):
        _, rows, _ = self.run_main("A:exe=/a", "B:exe=/b", "--rounds", "2", "--warm", "1", "--drop-caches")
        self.assertEqual([(r["round"], r["arm"], r["pass"]) for r in rows if r["cold"]],
                         [(0, "A", "warm"), (0, "B", "warm"), (1, "B", "warm"), (1, "A", "warm")])

    def test_medians_leave_out_the_cold_request_and_it_gets_its_own_line(self):
        self.decode = lambda nth: 1.0 if nth == 1 else 20.0
        _, rows, out = self.run_main("A:exe=/a", "--rounds", "1", "--warm", "1", "--measure", "2", "--drop-caches")
        medians, cold = out.split("medians over rounds")[1].split("cold requests")
        self.assertIn("A: decode   20.0", medians)
        self.assertNotIn("A: decode    1.0", medians)
        line = next(x for x in cold.splitlines() if x.startswith("  A:"))
        self.assertIn("chat0", line)
        self.assertIn("decode 1.0", line)
        self.assertEqual(len([x for x in cold.splitlines() if x.startswith("  A:")]), 1)

    def test_a_cold_only_tag_has_no_median(self):
        self.decode = lambda nth: 1.0 if nth == 1 else 20.0
        _, _, out = self.run_main("A:exe=/a", "--rounds", "1", "--drop-caches")
        medians = out.split("medians over rounds")[1].split("cold requests")[0]
        self.assertIn("A: decode    nan", medians.splitlines()[1])
        self.assertIn("A: decode   20.0", medians.splitlines()[2])

    def test_the_cold_line_shows_every_visit_of_the_arm(self):
        _, _, out = self.run_main("A:exe=/a", "--rounds", "3", "--drop-caches")
        line = next(x for x in out.split("cold requests")[1].splitlines() if x.startswith("  A:"))
        self.assertEqual(line.count("chat0"), 3)

    def test_the_warm_passes_do_not_feed_the_identity_check_with_drop_caches(self):
        self.text = lambda arm, prompt, n: "same" if n == 2 else f"{arm}{n}"
        _, _, out = self.run_main("A:exe=/a", "B:exe=/b", "--rounds", "1", "--warm", "1", "--drop-caches", "--identity")
        self.assertIn("identity: identical", out)

    def test_a_refused_cache_drop_ends_the_run_before_the_start(self):
        def refuse():
            raise SystemExit("--drop-caches failed: sudo: a password is required")
        with self.assertRaises(SystemExit) as cm:
            self.run_main("A:exe=/a", "--drop-caches", drop=refuse)
        self.assertIn("password", str(cm.exception.code))
        self.assertEqual(self.started, [])

    def test_a_given_long_words_tags_each_prompt_with_its_size(self):
        _, rows, _ = self.run_main("A:exe=/a", "--rounds", "1", "--long-words", "6000")
        self.assertEqual([r["req"] for r in rows], ["chat0", "chat1", "chat2", "long6000"])

    def test_long_words_repeat(self):
        _, rows, _ = self.run_main("A:exe=/a", "--rounds", "1", "--long-words", "30", "--long-words", "60")
        self.assertEqual([r["req"] for r in rows], ["chat0", "chat1", "chat2", "long30", "long60"])
        prompts = {e[2] for e in self.events if e[0] == "post"}
        self.assertEqual(len([p for p in prompts if p.startswith("Here is a list of words")]), 2)
        self.assertEqual(self.events[-1][2], ab_engine.long_prompt(60))

    def test_the_log_gets_80_characters_or_the_whole_answer(self):
        self.text = lambda arm, prompt, n: "w" * 200
        _, rows, _ = self.run_main("A:exe=/a", "--rounds", "1")
        self.assertEqual({len(r["text"]) for r in rows}, {80})
        (self.dir / "log.jsonl").unlink()
        _, rows, _ = self.run_main("A:exe=/a", "--rounds", "1", "--full-text")
        self.assertEqual({len(r["text"]) for r in rows}, {200})

    def test_identity_identical(self):
        self.text = lambda arm, prompt, n: "same " + prompt[:12]
        code, _, out = self.run_main("A:exe=/a", "B:exe=/b", "--rounds", "1", "--identity")
        self.assertEqual(code, 0)
        self.assertIn("identity: identical", out)

    def test_without_identity_a_difference_does_not_fail_the_run(self):
        self.text = lambda arm, prompt, n: arm
        code, _, out = self.run_main("A:exe=/a", "B:exe=/b", "--rounds", "1")
        self.assertEqual(code, 0)
        self.assertNotIn("identity", out)

    def test_identity_names_the_first_difference(self):
        self.text = lambda arm, prompt, n: "x" if arm == "/a" or not prompt.startswith("Explain") else "y"
        code, _, out = self.run_main("A:exe=/a", "B:exe=/b", "--rounds", "1", "--identity")
        self.assertEqual(code, 1)
        self.assertIn("identity: first difference at chat1 in arm B", out)

    def test_identity_compares_whole_texts_even_when_the_log_cuts_them(self):
        self.text = lambda arm, prompt, n: "w" * 100 + ("1" if arm == "/a" else "2")
        code, rows, out = self.run_main("A:exe=/a", "B:exe=/b", "--rounds", "1", "--identity")
        self.assertEqual(code, 1)
        self.assertEqual(rows[0]["text"], rows[-1]["text"])
        self.assertIn("identity: first difference at chat0 in arm B", out)

    def test_identity_uses_the_first_measured_pass(self):
        self.text = lambda arm, prompt, n: "same" if n == 1 else f"{arm}{n}"
        _, _, out = self.run_main("A:exe=/a", "B:exe=/b", "--rounds", "2", "--measure", "2", "--identity")
        self.assertIn("identity: identical", out)

    def test_identity_ignores_the_warm_passes(self):
        self.text = lambda arm, prompt, n: "same" if n == 2 else f"{arm}{n}"
        _, _, out = self.run_main("A:exe=/a", "B:exe=/b", "--rounds", "1", "--warm", "1", "--identity")
        self.assertIn("identity: identical", out)

    def test_identity_needs_two_arms(self):
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            self.run_main("A:exe=/a", "--identity")
        self.assertEqual(cm.exception.code, 2)

    def test_duplicate_arm_names_are_refused_before_any_start(self):
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()) as err:
            self.run_main("A:exe=/a", "A:exe=/b", "--identity")
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("A", err.getvalue())
        self.assertEqual(self.started, [])

    def test_duplicate_arm_names_are_refused_without_identity_too(self):
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            self.run_main("A:exe=/a", "B:exe=/b", "A:exe=/c")
        self.assertEqual(cm.exception.code, 2)
        self.assertEqual(self.started, [])

    def test_identity_fails_when_every_answer_is_empty(self):
        self.text = lambda arm, prompt, n: ""
        code, _, out = self.run_main("A:exe=/a", "B:exe=/b", "--rounds", "1", "--identity")
        self.assertEqual(code, 1)
        self.assertNotIn("identity: identical", out)
        self.assertIn("identity: error, arm A gave an empty answer at chat0", out)

    def test_identity_fails_when_one_arm_has_an_empty_answer(self):
        self.text = lambda arm, prompt, n: "" if arm == "/b" and prompt.startswith("Explain") else "x"
        code, _, out = self.run_main("A:exe=/a", "B:exe=/b", "--rounds", "1", "--identity")
        self.assertEqual(code, 1)
        self.assertIn("identity: error, arm B gave an empty answer at chat1", out)

    def test_three_rounds_by_default(self):
        self.run_main("A:exe=/a", "B:exe=/b")
        self.assertEqual([cfg["exe"] for cfg, _ in self.started], ["/a", "/b", "/b", "/a", "/a", "/b"])

    def test_the_decode_identity_kind(self):
        self.text = lambda arm, prompt, n: "same " + prompt[:12]
        code, rows, out = self.run_main("A:exe=/a", "B:exe=/b,xargs=--z", "--identity-kind", "decode")
        self.assertEqual(code, 0)
        (cfg_a, env_a), (cfg_b, env_b) = self.started                      # one round
        kind = ["--adapt-swaps", "0", "--pcie-frac", "0", "--prompt-cache", "0"]
        self.assertEqual(cfg_a["args"], ["--base"] + kind)
        self.assertEqual(cfg_b["args"], ["--base"] + kind + ["--z"])       # the arm's own arguments come after
        self.assertEqual((env_a["STRATA_PF_PEER_HOST"], env_b["STRATA_PF_PEER_HOST"]), ("0", "0"))
        self.assertEqual(len([e for e in self.events if e[0] == "post"]), 2 * 4)   # no warm pass, one measured
        self.assertIn("identity: identical", out)
        self.text = lambda arm, prompt, n: "w" * 100
        (self.dir / "log.jsonl").unlink()
        _, rows, _ = self.run_main("A:exe=/a", "B:exe=/b", "--identity-kind", "decode")
        self.assertEqual({len(r["text"]) for r in rows}, {100})                     # whole answers

    def test_the_adapt_identity_kind_keeps_the_swaps(self):
        self.run_main("A:exe=/a", "B:exe=/b", "--identity-kind", "adapt", "--rounds", "2")
        self.assertEqual(len(self.started), 4)
        self.assertEqual(self.started[0][0]["args"], ["--base", "--pcie-frac", "0", "--prompt-cache", "0"])
        self.assertEqual(self.started[0][1]["STRATA_PF_PEER_HOST"], "0")

    def test_an_arm_s_own_env_wins_over_the_kind_s(self):
        self.run_main("A:exe=/a,STRATA_PF_PEER_HOST=1", "B:exe=/b", "--identity-kind", "adapt")
        self.assertEqual([env["STRATA_PF_PEER_HOST"] for _, env in self.started], ["1", "0"])

    def test_bad_pass_counts(self):
        for flags in (["--measure", "0"], ["--warm", "-1"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                self.run_main("A:exe=/a", *flags)
            self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
