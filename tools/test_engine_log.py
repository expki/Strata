"""Tests for tools/engine_log.py on short excerpts of Task 16's engine logs (tools/fixtures/engine-log-*.txt, lines kept
verbatim): one start's facts, the requests as rows, the decode timing rows, and the command line.

    python -m pytest tools/test_engine_log.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_log  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"


def facts(name: str) -> dict:
    return engine_log.facts(engine_log.read(FIX / f"engine-log-{name}.txt"))


class Facts(unittest.TestCase):
    def test_k16_chunk_loan_w6800_and_no_ring(self):
        f = facts("k16")
        self.assertEqual((f["chunk"], f["loan_slots"], f["w6800_experts"], f["ring_slots"]), (16384, 1600, 7472, 0))
        self.assertIs(f["tier_in_prompts"], True)

    def test_c16_at_the_default_reserve_reads_its_prompts_without_the_tier(self):
        f = facts("c16")
        self.assertIs(f["tier_in_prompts"], False)
        self.assertEqual(f["chunk"], 16384)

    def test_g305_opens_a_ring_of_128(self):
        self.assertEqual(facts("g305")["ring_slots"], 128)

    def test_rb60_pins_60_gib_in_56_s(self):
        f = facts("rb60")
        self.assertEqual((f["pinned_gib"], f["pinned_s"]), (60.0, 56.0))

    def test_pauto_picks_4352(self):
        self.assertEqual(facts("pauto")["chunk"], 4352)

    def test_k24r_loan_w6800_and_ring(self):
        f = facts("k24r")
        self.assertEqual((f["loan_slots"], f["w6800_experts"], f["ring_slots"]), (2329, 7188, 128))

    def test_re_pins_90_66_gib_in_92_s(self):
        f = facts("re")
        self.assertEqual((f["pinned_gib"], f["pinned_s"]), (90.66, 92.0))

    def test_re24_pinned_and_the_3090_low_on_vram(self):
        f = facts("re24")
        self.assertEqual((f["pinned_gib"], f["pinned_s"]), (98.17, 109.0))
        self.assertEqual((f["vram_free_mib"], f["vram_low"]), (231, 93))
        self.assertEqual(f["chunk"], 24576)

    def test_no_copy_reads_as_0_gib_and_no_time(self):
        f = facts("k16")
        self.assertEqual((f["pinned_gib"], f["pinned_s"]), (0.0, None))

    def test_slots_and_the_page_cache_share(self):
        self.assertEqual(facts("k16")["cuda_slots"], 3103)
        self.assertEqual(facts("k16")["cached_share"], 1.0)
        self.assertIsNone(facts("re24")["cached_share"])    # the copy path never asks the page cache

    def test_every_fact_is_a_key(self):
        self.assertEqual(set(facts("re24")), set(engine_log.FACTS))

    def test_only_the_last_start_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "two.log"
            p.write_text((FIX / "engine-log-rb60.txt").read_text() + (FIX / "engine-log-k16.txt").read_text())
            f = engine_log.facts(engine_log.read(p))
        self.assertEqual((f["pinned_gib"], f["chunk"], f["w6800_experts"]), (0.0, 16384, 7472))

    def test_a_log_without_a_start_line_is_read_whole(self):
        lines = [x for x in engine_log.read(FIX / "engine-log-g305.txt") if "engine started:" not in x]
        f = engine_log.facts(lines)
        self.assertEqual((f["ring_slots"], f["chunk"]), (128, None))   # the chunk comes from the start line

    def test_nothing_in_the_log_gives_none_and_no_tier(self):
        f = engine_log.facts([])
        self.assertIsNone(f["chunk"])
        self.assertIs(f["tier_in_prompts"], False)                     # no AMD tier at all
        self.assertIsNone(f["vram_low"])


class Rows(unittest.TestCase):
    def test_re24_requests(self):
        rows = engine_log.rows(engine_log.read(FIX / "engine-log-re24.txt"))
        self.assertEqual([r["prompt"] for r in rows], [15, 34, 8013, 34, 27, 29, 2041, 8013, 20053])
        last = rows[-1]
        self.assertEqual((last["read"], last["ptoks"], last["gen"], last["dtoks"]), (20053, 2528.4, 128, 64.4))
        self.assertEqual((last["acc"], last["drafts"]), (76, 109))
        self.assertEqual((last["file_mb"], last["hit"], last["own_ms"], last["amd_ms"], last["streamed_rows"]),
                         (78.8, 91.1, 59.3, 19.2, 0))

    def test_a_low_vram_warning_goes_with_the_request_it_comes_before(self):
        rows = engine_log.rows(engine_log.read(FIX / "engine-log-re24.txt"))
        self.assertEqual([r.get("vram_low") for r in rows], [231, None, None, 93, None, None, None, None, None])

    def test_timing_lines_only_when_asked(self):
        lines = engine_log.read(FIX / "engine-log-k16.txt")
        rows = engine_log.rows(lines)
        self.assertNotIn("timing", rows[0])
        rows = engine_log.rows(lines, timing=True)
        r20 = next(r for r in rows if r["prompt"] == 20053)
        self.assertTrue(any(t.startswith("strata prefill timing: 20046 tokens") for t in r20["timing"]))
        self.assertEqual((r20["own_ms"], r20["paged_gbs"]), (108.3, 13.9))   # the planner line after its serve line


class Decode(unittest.TestCase):
    def test_re24_decode_rows(self):
        d = engine_log.decode(engine_log.read(FIX / "engine-log-re24.txt"))
        self.assertEqual(len(d), 9)
        first = d[0]
        self.assertEqual((first["req"], first["prompt"], first["windows"], first["ms_window"], first["cpu"],
                          first["cpu_experts"], first["join"], first["swaps"]), (0, 15, 4, 106.94, 64.02, 9.11, 12.65, 24.0))
        self.assertEqual((d[-1]["prompt"], d[-1]["tok_window"], d[-1]["gpu_wait"]), (20053, 2.42, 12.81))

    def test_no_decode_timing_no_rows(self):
        self.assertEqual(engine_log.decode(engine_log.read(FIX / "engine-log-k16.txt")), [])


class Cli(unittest.TestCase):
    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = engine_log.main(list(argv))
        return code, out.getvalue()

    def test_facts_prints_one_json_object(self):
        code, out = self.run_main("facts", str(FIX / "engine-log-re24.txt"))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["pinned_gib"], 98.17)

    def test_rows_and_decode_print_json_lines(self):
        _, out = self.run_main("rows", str(FIX / "engine-log-re24.txt"))
        self.assertEqual(json.loads(out.splitlines()[-1])["prompt"], 20053)
        _, out = self.run_main("decode", str(FIX / "engine-log-re24.txt"))
        self.assertEqual(len(out.splitlines()), 9)

    def test_an_unknown_command_is_refused(self):
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            self.run_main("nope", str(FIX / "engine-log-re24.txt"))
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
