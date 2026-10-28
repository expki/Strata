"""The speed features of 0.2.7 that have a switch in the app's Expert settings and in the Terminal's Advanced table.

A switch has a default (``on`` or ``off``).  In its other state it writes exactly the engine environment below into the model's run config ("env"); back in
its default it removes those entries, so a default launch carries none of them and is byte for byte what 0.2.6 started.  The names are the engine switch
contract (phaseA-tests/notes-027-plan.md).  The app does the same in apps/host/server/native-settings.mjs (SPEED_SWITCHES); tests/terminal/test_speed_switches.py
reads that file and fails when the two disagree, so the Terminal and the app cannot drift apart.

- "Faster answers on Radeon RX 9000 / AI PRO R9700" and "DeepSeek speed-ups" are ON by default; turning them off writes their entries.
- This fork has no "Faster long prompts" (KVA) and no "Faster answers on 12 GB NVIDIA cards" (the route prior): both change which work or which experts
  run, so the engine features and their rows were removed.

The Terminal shows the switches as rows of its Advanced table (strata_cli/launch/params.py), only where they apply: ``strata launch --set STRATA_PCIE_FRAC_DEFAULT=old``
turns "Faster answers on Radeon RX 9000" off, ``--set STRATA_PCIE_FRAC_DEFAULT=off`` puts the default back (the same as removing the row).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Switch:
    id: str
    label: str                    # the words the app shows
    effect: str                   # one plain sentence: what it does and what the other choice does
    env: tuple[tuple[str, str], ...]   # the environment entries written when the switch is NOT in its default state
    only: str                     # "qwen-radeon-9000" | "deepseek" | "qwen-any": where it applies
    default: str = "on"           # "on" | "off": what a launch without the entries does
    legacy: tuple[tuple[str, str], ...] = ()    # entries an earlier 0.2.7 test build wrote for the same state: they read as the switch being set, and are removed with it
    terminal_only: bool = False   # shown in the Terminal's Advanced table only, never in the app


SWITCHES: tuple[Switch, ...] = (
    Switch("faster_radeon_9000", "Faster answers on Radeon RX 9000 / AI PRO R9700",
           "Shares the work between the graphics card and the processor in a way that makes answers faster on these cards. "
           "Turn off to go back to the previous setting, which is slower.",
           (("STRATA_PCIE_FRAC_DEFAULT", "old"),), "qwen-radeon-9000"),
    Switch("deepseek_speedups", "DeepSeek speed-ups",
           "Reads long prompts faster and loads the parts of the model it needs faster. "
           "Turn off if DeepSeek stops working or its answers look wrong; it then runs as in version 0.2.6.",
           # DS4_DEFAULTS=026: the engine's master switch back to the 0.2.6 defaults; the six are what older engines (and the test builds) read
           (("DS4_DEFAULTS", "026"), ("DS4_FAST_ATTN", "0"), ("DS4_ATTN_SLAB", "0"), ("DS4_COPY_PIN", "0"), ("DS4_COPY_HUGETLB", "0"),
            ("DS4_STORE_SPARE", "0"), ("DS4_STORE_PUT", "0")), "deepseek",
           legacy=(("DS4_FAST_ATTN", "0"), ("DS4_ATTN_SLAB", "0"), ("DS4_COPY_PIN", "0"), ("DS4_COPY_HUGETLB", "0"),
                   ("DS4_STORE_SPARE", "0"), ("DS4_STORE_PUT", "0"))),
    # Troubleshooting only, in the Terminal's Advanced table and nowhere else: every Qwen 0.2.7 default back to its 0.2.6 behaviour.
    Switch("qwen_defaults_026", "Qwen: 0.2.6 behaviour",
           "Starts every Qwen speed setting new in 0.2.7 the way 0.2.6 did, for troubleshooting. Use it if something that worked before no longer does.",
           (("STRATA_DEFAULTS", "026"),), "qwen-any", default="off", terminal_only=True),
)

BY_ID = {s.id: s for s in SWITCHES}

_GFX12 = re.compile(r"^gfx120[01]$", re.IGNORECASE)
_RADEON_9000 = re.compile(r"\b(?:RX\s*9\d{3}|AI\s*PRO\s*R97\d{2}|R97\d{2})\b", re.IGNORECASE)


def is_deepseek(cfg: Mapping[str, Any] | None) -> bool:
    cfg = cfg or {}
    return cfg.get("arch") == "deepseek41" or bool(re.search(r"strata-dsv41", str(cfg.get("exe") or ""), re.IGNORECASE))


def _radeon_9000(gpus: Iterable[Mapping[str, Any]] | None) -> bool:
    for g in gpus or []:
        if g.get("vendor") not in (None, "", "amd"):
            continue
        arch = str(g.get("arch") or "")
        if _GFX12.match(arch) or (not arch and _RADEON_9000.search(str(g.get("name") or ""))):
            return True
    return False


def applies(switch: Switch | str, cfg: Mapping[str, Any] | None, gpus: Iterable[Mapping[str, Any]] | None, backend: str | None = None) -> bool:
    """Does the switch make sense for this model on this PC? ``backend`` is the engine's (cuda, hip, sycl, metal, cpu); the run config's own wins."""
    sw = BY_ID[switch] if isinstance(switch, str) else switch
    if sw.only == "deepseek":
        return is_deepseek(cfg)
    if is_deepseek(cfg):
        return False
    if sw.only == "qwen-any":
        return True
    backend = (cfg or {}).get("backend") or backend
    if sw.only == "qwen-radeon-9000":
        return backend == "hip" and _radeon_9000(gpus)
    return False


def applicable(cfg: Mapping[str, Any] | None, gpus: Iterable[Mapping[str, Any]] | None, backend: str | None = None) -> list[Switch]:
    gpus = list(gpus or [])
    return [s for s in SWITCHES if applies(s, cfg, gpus, backend)]


def app_switches() -> list[Switch]:
    """The switches the app shows (apps/host/server/native-settings.mjs SPEED_SWITCHES): all but the Terminal-only ones."""
    return [s for s in SWITCHES if not s.terminal_only]


def is_set(env: Mapping[str, Any] | None, switch: Switch | str) -> bool:
    """Every entry the switch writes is there with its value (the switch is out of its default).  A file with only some of them, or other values
    is in the default state and is left alone."""
    sw = BY_ID[switch] if isinstance(switch, str) else switch
    have = lambda entries: bool(entries) and all((env or {}).get(name) == value for name, value in entries)   # noqa: E731
    return have(sw.env) or have(sw.legacy)


def is_on(env: Mapping[str, Any] | None, switch: Switch | str) -> bool:
    """Is the feature on for a launch with this environment?"""
    sw = BY_ID[switch] if isinstance(switch, str) else switch
    return is_set(env, sw) != (sw.default == "on")


def set_switch(env: Mapping[str, Any] | None, switch: Switch | str, on: bool) -> dict[str, Any]:
    """The environment with the feature on or off: the default state removes the switch's entries, the other one writes them. Nothing else changes."""
    sw = BY_ID[switch] if isinstance(switch, str) else switch
    out = dict(env or {})
    write = on != (sw.default == "on")
    for name, value in sw.env:
        if write:
            out[name] = value
        else:
            out.pop(name, None)
    if not write:
        for name, _ in sw.legacy:
            out.pop(name, None)
    return out


def env_names() -> dict[str, Switch]:
    """Every environment name a switch writes -> its switch."""
    return {name: sw for sw in SWITCHES for name, _ in sw.env}


def env_words(name: str) -> str:
    """The plain sentence of one environment row of the Terminal's Advanced table."""
    sw = env_names()[name]
    value = dict(sw.env)[name]
    together = f"; all {len(sw.env)} speed-up rows go together" if len(sw.env) > 1 else ""
    if sw.default == "off":
        return f"{sw.label} (off by default; set {value} to turn on{together}). {sw.effect}"
    return f"{sw.label} (on by default; set {value} to turn off{together}). {sw.effect}"
