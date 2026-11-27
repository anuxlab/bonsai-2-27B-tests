"""Load + normalise test cases from tests/*.json (static) and generators (long-context)."""
import json, re
from pathlib import Path
from . import generators

TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"
SUITES = ["basic", "thinking", "coding", "agentic", "advanced", "vision", "consistency", "long_context", "performance", "stress", "safety"]
HEAVY_DEFAULT_OFF = {"long_context"}  # expensive: needs enable_long_context
# Throughput measured in the 2026-11-25 Actions run (4 vCPU / 15 GiB, CPU only). Used for *planning estimates only*.
PREFILL_TPS, DECODE_TPS = 4.0, 1.6
LEVEL_NAMES = {1: "L1-basic", 2: "L2-intermediate", 3: "L3-advanced", 4: "L4-hard"}


def _norm(c, suite):
    c = dict(c)
    c["suite"] = suite
    c.setdefault("kind", "chat")
    c.setdefault("tags", [])
    c.setdefault("level", 2)
    c.setdefault("max_tokens", 512)
    c.setdefault("est_out_tokens", 200)
    if "est_prompt_tokens" not in c:
        c["est_prompt_tokens"] = int((len(c.get("prompt", "")) + len(c.get("system", ""))) / generators.CHARS_PER_TOKEN) + 12
    return c


def load_suite(suite, long_context_levels=("1k", "2k", "4k")):
    raw = json.loads((TESTS_DIR / f"{suite}.json").read_text(encoding="utf-8"))
    out = []
    for c in raw:
        if c.get("gen") == "long_context":
            if c["label"] in long_context_levels:
                out += [_norm(x, suite) for x in generators.long_context_cases(c["tokens"], c["seed"], c["label"])]
        else:
            out.append(_norm(c, suite))
    return out


def est_seconds(c, seen_groups=None):
    """Planning estimate: prefill (once per prefix group, thanks to the prompt cache) + decode."""
    pre = c["est_prompt_tokens"]
    g = c.get("prefix_group")
    if g and seen_groups is not None:
        pre = 40 if g in seen_groups else pre
        seen_groups.add(g)
    return pre / PREFILL_TPS + c["est_out_tokens"] / DECODE_TPS
