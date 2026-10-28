"""DeepSeek's faster mode (5 of 6 experts per word) is removed: old flags, saved presets and saved --experts flags are not used."""

from __future__ import annotations

import json
import unittest

from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config, drop_experts
from tests.terminal.launch_support import run_cli

NOTE = "Faster mode was removed"


class OldFlags(unittest.TestCase):
    def plan(self, *args):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "ist-das-2.00bpw", "--family", "deepseek41", *args])
        self.assertEqual(code, 0, err + out)
        return json.loads(out)["plan"]

    def test_old_flags_are_accepted_and_not_used(self):
        for args in (("--speed", "faster"), ("--speed", "full"), ("--full-quality",)):
            p = self.plan(*args)
            self.assertIsNone(p["settings"]["speed"], args)
            self.assertTrue(any(NOTE in w for w in p["warnings"]), args)
            self.assertNotIn("--speed", p["flags"])
            self.assertNotIn("--full-quality", p["flags"])

    def test_no_speed_line_and_no_note_without_old_flags(self):
        p = self.plan()
        self.assertNotIn("speed", p)
        self.assertFalse(any(NOTE in w for w in p["warnings"]))

    def test_another_model_says_the_same(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "q2_0", "--speed", "faster"])
        p = json.loads(out)["plan"]
        self.assertIsNone(p["settings"]["speed"])
        self.assertTrue(any(NOTE in w for w in p["warnings"]))


class SavedState(unittest.TestCase):
    def test_a_saved_preset_with_a_speed_still_loads(self):
        s = S.LaunchSettings.from_dict({"family": "deepseek41", "model": "IST-DAS-2.00bpw", "speed": "faster"})
        self.assertEqual(s.speed, "faster")                 # loaded as it was saved, then the launch says it is not used
        self.assertNotIn("--speed", S.flags_text(s))

    def test_a_saved_experts_flag_is_dropped(self):
        cfg = {"args": ["--pack", "p", "--experts", "5", "--dspark", "auto", "--experts=6"]}
        drop_experts(cfg)
        self.assertEqual(cfg["args"], ["--pack", "p", "--dspark", "auto"])
        base = {"exe": "x", "args": ["--pack", "p", "--experts", "5"]}
        self.assertEqual(derive_config(base, S.LaunchSettings(), bool_flags=frozenset())["args"], ["--pack", "p"])


if __name__ == "__main__":
    unittest.main()
