"""The 0.2.7 speed features and their switches: the contract names, the defaults (both ON), where each applies, and how the Terminal carries them.

This fork removed two upstream rows: "Faster long prompts" (KVA) and "Faster answers on 12 GB NVIDIA cards" (the route prior); the tests say they stay gone.

The app (apps/host/server/native-settings.mjs SPEED_SWITCHES) and Control (strata_control/speed_switches.py) say the same thing; the last test reads the
app's table with node and fails when the two differ.  `strata launch` shows the environment entries as rows of its Advanced table and writes them into
the run config's "env" (`--set STRATA_PCIE_FRAC_DEFAULT=old` turns the Radeon one off); a default launch has none of them.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

from strata_cli.launch import params as P
from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_control import speed_switches as W

ROOT = Path(__file__).resolve().parents[2]

ENV = {      # what each switch writes outside its default
    "faster_radeon_9000": {"STRATA_PCIE_FRAC_DEFAULT": "old"},
    "deepseek_speedups": {"DS4_DEFAULTS": "026", "DS4_FAST_ATTN": "0", "DS4_ATTN_SLAB": "0", "DS4_COPY_PIN": "0", "DS4_COPY_HUGETLB": "0",
                          "DS4_STORE_SPARE": "0", "DS4_STORE_PUT": "0"},
    "qwen_defaults_026": {"STRATA_DEFAULTS": "026"},         # the Terminal's troubleshooting switch: not in the app
}
DEFAULT = {"faster_radeon_9000": "on", "deepseek_speedups": "on", "qwen_defaults_026": "off"}
SIX = ("DS4_FAST_ATTN", "DS4_ATTN_SLAB", "DS4_COPY_PIN", "DS4_COPY_HUGETLB", "DS4_STORE_SPARE", "DS4_STORE_PUT")
RTX3060 = {"name": "NVIDIA GeForce RTX 3060", "vendor": "nvidia", "arch": "86", "vram_gb": 12.0}
RTX4070S = {"name": "NVIDIA GeForce RTX 4070 SUPER", "vendor": "nvidia", "arch": "89", "vram_gb": 12.0}
RTX4060TI16 = {"name": "NVIDIA GeForce RTX 4060 Ti 16GB", "vendor": "nvidia", "arch": "89", "vram_gb": 16.0}
R9700 = {"name": "AMD Radeon AI PRO R9700", "vendor": "amd", "arch": "gfx1201"}
RX7900 = {"name": "AMD Radeon RX 7900 XTX", "vendor": "amd", "arch": "gfx1100"}
STRIX = {"name": "AMD Radeon 8060S", "vendor": "amd", "arch": "gfx1151", "uma": True}
RTX = {"name": "NVIDIA GeForce RTX 4090", "vendor": "nvidia", "arch": "89"}
# A Qwen3.8-Flash-Next run config (the original, Swift, Coder, Unsloth, Gyro, NVFP4): setup writes no "arch" for it.  Qwen3.6-35B-A3B has an engine path of its own: "arch".
QWEN = {"model_name": "qwen3.8-flash-next-iq3_s", "backend": "cuda", "args": ["--native", "m.gguf"], "env": {"PATH": "/bin"}}
QWEN36 = {"arch": "qwen35moe", "model_name": "qwen3.6-35b-a3b-ud-q4_k_xl", "backend": "cuda", "args": ["--native", "m.gguf"], "env": {"PATH": "/bin"}}
DEEPSEEK = {"arch": "deepseek41", "exe": "C:/s/engine-dsv41/0.1.52-win/strata-dsv41.exe", "args": ["--pack", "p"], "env": {"DS4_ASYNC_T": "4"}}


def ids(cfg, gpus, backend=None):
    """The switches the app would show (the Terminal-only one is tested apart)."""
    return [s.id for s in W.applicable(cfg, gpus, backend) if not s.terminal_only]


class Contract(unittest.TestCase):
    def test_three_switches_with_the_contract_names_and_their_defaults(self):
        self.assertEqual([s.id for s in W.SWITCHES], list(ENV))
        for s in W.SWITCHES:
            self.assertEqual(dict(s.env), ENV[s.id], s.id)
            self.assertEqual(s.default, DEFAULT[s.id], s.id)

    def test_an_effect_text_says_what_the_feature_does_in_plain_words(self):
        for s in W.SWITCHES:
            if not s.terminal_only:
                self.assertRegex(s.effect, r"Off by default" if s.default == "off" else r"[Tt]urn off", s.id)
            self.assertNotRegex(s.effect, r"STRATA_|DS4_|--|env", s.id)

    def test_the_removed_rows_stay_gone(self):
        for gone in ("faster_long_prompts", "faster_12gb_nvidia"):
            self.assertNotIn(gone, W.BY_ID)
        for name in ("STRATA_KVA", "STRATA_ROUTE_PRIOR"):
            self.assertNotIn(name, W.env_names())
        self.assertEqual(ids(QWEN, [RTX3060]), [], "no row for a 12 GB NVIDIA card")

    def test_where_each_switch_applies(self):
        self.assertEqual(ids(QWEN, [RTX]), [])
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [RX7900]), [])
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [STRIX]), [], "Strix Halo is gfx1151, not gfx12")
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [R9700]), ["faster_radeon_9000"])
        self.assertEqual(ids({"model_name": "qwen3.8-flash-next-iq3_s"}, [R9700], "hip"), ["faster_radeon_9000"], "the installed engine's backend when the config has none")
        self.assertEqual(ids({**QWEN, "backend": "hip"}, [{"name": "AMD Radeon RX 9070 XT", "vendor": "amd"}]), ["faster_radeon_9000"])
        self.assertEqual(ids({**QWEN, "backend": "cuda"}, [RTX, R9700]), [], "the Radeon switch needs the HIP engine")
        self.assertEqual(ids({**QWEN36, "backend": "hip"}, [R9700]), ["faster_radeon_9000"], "the Radeon switch is not tied to the model's shape")
        for backend in ("sycl", "metal", "cpu", None):
            self.assertEqual(ids({**QWEN, "backend": backend}, [R9700]), [], backend)
        self.assertEqual(ids(DEEPSEEK, [R9700]), ["deepseek_speedups"])
        self.assertEqual(ids({**DEEPSEEK, "backend": "cuda"}, [RTX]), ["deepseek_speedups"])
        self.assertEqual(ids({"exe": "C:/x/strata-dsv41.exe"}, [RTX]), ["deepseek_speedups"], "found by the engine's name too")

    def test_deepseek_off_writes_the_master_switch_and_the_six_and_on_removes_all_seven(self):
        env = {"DS4_ASYNC_T": "4"}
        off = W.set_switch(env, "deepseek_speedups", False)
        self.assertEqual(list(off)[:2], ["DS4_ASYNC_T", "DS4_DEFAULTS"], "the master switch first")
        self.assertEqual(off, {"DS4_ASYNC_T": "4", "DS4_DEFAULTS": "026", **{k: "0" for k in SIX}})
        self.assertFalse(W.is_on(off, "deepseek_speedups"))
        self.assertEqual(W.set_switch(off, "deepseek_speedups", True), env)
        # an earlier test build wrote only the six: it still reads as off, and turning it on removes them
        six = {"DS4_ASYNC_T": "4", **{k: "0" for k in SIX}}
        self.assertFalse(W.is_on(six, "deepseek_speedups"))
        self.assertEqual(W.set_switch(six, "deepseek_speedups", True), env)
        self.assertTrue(W.is_on({"DS4_DEFAULTS": "026"}, "deepseek_speedups"), "the master switch alone is not the whole switch: a hand-edited file is left alone")

    def test_the_terminal_only_switch_is_for_qwen_and_never_in_the_app(self):
        sw = W.BY_ID["qwen_defaults_026"]
        self.assertTrue(sw.terminal_only)
        self.assertEqual(dict(sw.env), {"STRATA_DEFAULTS": "026"})
        self.assertNotIn("qwen_defaults_026", [s.id for s in W.app_switches()])
        for cfg, gpus in ((QWEN, [RTX]), (QWEN36, [RTX]), ({**QWEN, "backend": "sycl"}, []), ({**QWEN, "backend": "cpu"}, [])):
            self.assertIn("qwen_defaults_026", [s.id for s in W.applicable(cfg, gpus)])
        self.assertNotIn("qwen_defaults_026", [s.id for s in W.applicable(DEEPSEEK, [RTX])], "DeepSeek has its own master switch (DS4_DEFAULTS)")
        self.assertEqual(W.set_switch({"PATH": "/bin"}, sw, True), {"PATH": "/bin", "STRATA_DEFAULTS": "026"})
        self.assertFalse(W.is_on({}, sw))
        self.assertEqual(W.set_switch({"STRATA_DEFAULTS": "026"}, sw, False), {})

    def test_outside_the_default_and_back_is_the_identical_environment(self):
        env = {"PATH": "/bin", "STRATA_PREFILL": "2", "DS4_ASYNC_T": "4"}
        for s in W.SWITCHES:
            other = W.set_switch(env, s, s.default != "on")           # each one out of its default
            self.assertEqual(other, {**env, **ENV[s.id]})
            self.assertTrue(W.is_set(other, s))
            self.assertEqual(W.is_on(other, s), s.default != "on")
            back = W.set_switch(other, s, s.default == "on")
            self.assertEqual(json.dumps(back), json.dumps(env), s.id)

    def test_deepseek_is_on_by_default(self):
        self.assertTrue(W.is_on({}, "deepseek_speedups"))
        self.assertTrue(W.is_on({"DS4_FAST_ATTN": "0"}, "deepseek_speedups"), "only some of the entries reads as the default (on)")


class Terminal(unittest.TestCase):
    """`strata launch`: the run config a launch writes."""

    def config(self, base, extra=None):
        return derive_config(base, S.LaunchSettings(extra=extra or {}), bool_flags=frozenset())

    def test_a_default_launch_has_none_of_the_new_entries(self):
        for base in (QWEN, DEEPSEEK):
            env = self.config(base).get("env") or {}
            for name in W.env_names():
                self.assertNotIn(name, env, name)
            self.assertEqual(env, base["env"], "the environment is exactly the installed one")
            self.assertEqual(json.dumps(self.config(base), sort_keys=True), json.dumps(self.config(base, {}), sort_keys=True))

    def test_each_switch_outside_its_default_by_set_gives_exactly_its_entries(self):
        for sid, entries in ENV.items():          # each one is moved out of its default by its entries
            base = DEEPSEEK if sid == "deepseek_speedups" else QWEN
            cfg = self.config(base, S.settings_from_pairs([f"{k}={v}" for k, v in entries.items()]))
            self.assertEqual(cfg["env"], {**base["env"], **entries}, sid)
            self.assertEqual(cfg["args"], base["args"], "no engine flag changes")

    def test_off_again_puts_the_default_back(self):
        cfg = self.config(QWEN, S.settings_from_pairs(["STRATA_DEFAULTS=off"]))
        self.assertEqual(cfg["env"], QWEN["env"])
        turned_on = {**QWEN, "env": {**QWEN["env"], "STRATA_DEFAULTS": "026"}}
        self.assertEqual(self.config(turned_on, S.settings_from_pairs(["STRATA_DEFAULTS=off"]))["env"], QWEN["env"])

    def test_the_advanced_table_lists_only_the_switches_that_apply(self):
        def names(cfg, gpus, extra=None):
            rows = P.build_rows(None, cfg["args"], None, cfg["env"], None, extra or {}, P.switch_params(W.applicable(cfg, gpus)))
            return [r.name for r in rows if r.name in W.env_names()]
        # (STRATA_DEFAULTS, the Qwen "0.2.6 behaviour" row, closes every Qwen table; the DeepSeek table has DS4_DEFAULTS instead)
        self.assertEqual(names(QWEN, [RTX]), ["STRATA_DEFAULTS"])
        self.assertEqual(names(QWEN, [RTX3060]), ["STRATA_DEFAULTS"])
        self.assertEqual(names({**QWEN, "backend": "hip"}, [R9700]), ["STRATA_PCIE_FRAC_DEFAULT", "STRATA_DEFAULTS"])
        self.assertEqual(names(DEEPSEEK, [R9700]), list(ENV["deepseek_speedups"]), "the master switch and the six, in the contract order")
        self.assertEqual(names({**QWEN, "backend": "sycl"}, [R9700]), ["STRATA_DEFAULTS"])
        self.assertEqual(names(QWEN36, [RTX]), ["STRATA_DEFAULTS"])

    def test_the_qwen_0_2_6_row_in_the_advanced_table(self):
        rows = P.build_rows(None, QWEN["args"], None, QWEN["env"], None, {}, P.switch_params(W.applicable(QWEN, [RTX])))
        row = next(r for r in rows if r.name == "STRATA_DEFAULTS")
        self.assertEqual((row.default, row.yours, row.known), ("off", "", True))
        self.assertIn("Qwen: 0.2.6 behaviour", row.description)
        self.assertIn("off by default; set 026 to turn on", row.description)
        cfg = derive_config(QWEN, S.LaunchSettings(extra=S.settings_from_pairs(["STRATA_DEFAULTS=026"])), bool_flags=frozenset())
        self.assertEqual(cfg["env"], {**QWEN["env"], "STRATA_DEFAULTS": "026"})
        self.assertEqual(derive_config(QWEN, S.LaunchSettings(), bool_flags=frozenset())["env"], QWEN["env"], "a default launch carries nothing")

    def test_a_row_shows_its_default_until_it_is_changed_and_explains_itself(self):
        hip = {**QWEN, "backend": "hip"}
        radeon = next(r for r in P.build_rows(None, hip["args"], None, hip["env"], None, {}, P.switch_params(W.applicable(hip, [R9700]))) if r.name == "STRATA_PCIE_FRAC_DEFAULT")
        self.assertEqual(radeon.default, "on")
        self.assertIn("on by default; set old to turn off", radeon.description)

    def test_the_six_deepseek_rows_say_they_go_together(self):
        text = W.env_words("DS4_COPY_PIN")
        self.assertIn("all 7 speed-up rows go together", text)
        self.assertEqual({W.env_names()[n].id for n in ENV["deepseek_speedups"]}, {"deepseek_speedups"})


class SameAsTheApp(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_apps_table_has_the_same_names_and_words(self):
        script = ("import {SPEED_SWITCHES} from './apps/host/server/native-settings.mjs';"
                  "console.log(JSON.stringify(SPEED_SWITCHES.map(s=>({id:s.id,label:s.label,hint:s.hint,default:s.default,env:s.env,legacy:s.legacy??{}}))))")
        out = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=60, check=True).stdout
        app = json.loads(out)
        self.assertEqual([a["id"] for a in app], [s.id for s in W.app_switches()], "the Terminal-only switch is not in the app's table")
        for a, s in zip(app, W.app_switches()):
            self.assertEqual(a["label"], s.label)
            self.assertEqual(a["hint"], s.effect, s.id)
            self.assertEqual(a["env"], dict(s.env), s.id)
            self.assertEqual(a["default"], s.default, s.id)
            self.assertEqual(a["legacy"], dict(s.legacy), s.id)


if __name__ == "__main__":
    unittest.main()
