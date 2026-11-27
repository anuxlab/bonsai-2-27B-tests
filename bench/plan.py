"""Plan = which tests run, split into shards (one GitHub Actions matrix job each)."""
import json, math, re
from . import cases as C

UNSPLITTABLE = {"performance", "stress", "vision"}  # measured on ONE clean machine / need one server session


def parse_suites(s):
    s = (s or "all").strip().lower()
    if s in ("all", "*", ""):
        return [x for x in C.SUITES]
    names = [x.strip() for x in re.split(r"[,\s]+", s) if x.strip()]
    bad = [n for n in names if n not in C.SUITES]
    if bad:
        raise ValueError(f"unknown suite(s) {bad}; valid: {C.SUITES}")
    return [n for n in C.SUITES if n in names]


def build_plan(suites="all", repeat=1, long_context_levels="1k,2k,4k", enable_vision=True, enable_long_context=True,
               test_filter="", shard_minutes=60, max_shards=20, ctx_override=0, stress_level="medium", **_):
    names = parse_suites(suites)
    lcl = tuple(x.strip() for x in str(long_context_levels).split(",") if x.strip())
    flt = re.compile(test_filter) if test_filter else None
    shards, skipped, notes = [], [], []
    suite_cases = {}
    for s in names:
        if s == "vision" and not enable_vision:
            skipped.append({"suite": s, "reason": "disabled by input enable_vision=false"}); continue
        if s == "long_context" and not enable_long_context:
            skipped.append({"suite": s, "reason": "disabled by input enable_long_context=false"}); continue
        cs = C.load_suite(s, lcl)
        if flt:
            cs = [c for c in cs if flt.search(c["id"])]
        if not cs:
            skipped.append({"suite": s, "reason": "no tests selected"}); continue
        suite_cases[s] = cs
    # cost per suite (repeat multiplies cost)
    def cost(cs):
        seen = set()
        return sum(C.est_seconds(c, seen) for c in cs) * repeat
    totals = {s: cost(cs) for s, cs in suite_cases.items()}
    target = shard_minutes * 60.0
    wanted = {s: (1 if s in UNSPLITTABLE else max(1, math.ceil(totals[s] / target))) for s in suite_cases}
    while sum(wanted.values()) > max_shards:  # respect the concurrency cap: enlarge the biggest shards' target
        s = max((k for k in wanted if wanted[k] > 1), key=lambda k: totals[k] / wanted[k], default=None)
        if s is None:
            notes.append(f"{sum(wanted.values())} unsplittable shards exceed max_shards={max_shards}; they will queue")
            break
        wanted[s] -= 1
    for s, cs in suite_cases.items():
        k = wanted[s]
        groups = {}
        for c in cs:  # keep a prefix group (shared long prompt) together so the prompt cache is useful
            groups.setdefault(c.get("prefix_group") or c["id"], []).append(c)
        gl = sorted(groups.values(), key=lambda g: -sum(C.est_seconds(c) for c in g))
        bins = [[] for _ in range(k)]
        load = [0.0] * k
        for g in gl:  # longest-processing-time-first greedy packing
            i = min(range(k), key=lambda j: load[j])
            bins[i] += g
            load[i] += sum(C.est_seconds(c) for c in g)
        for i, b in enumerate(bins):
            if not b:
                continue
            b.sort(key=lambda c: (c.get("order", 5), c["id"]))
            seen = set()
            est = sum(C.est_seconds(c, seen) for c in b) * repeat
            need = max(c["est_prompt_tokens"] + c["max_tokens"] for c in b)
            ctx = ctx_override or max(8192, int(math.ceil(need * 1.25 / 1024) * 1024))
            sid = s if k == 1 else f"{s}-{i + 1}"
            shards.append({"shard": sid, "suite": s, "case_ids": [c["id"] for c in b], "n_tests": len(b) * repeat, "ctx": ctx,
                           "parallel": 2 if s in ("performance", "stress") else 1, "vision": s == "vision",
                           "est_minutes": round(est / 60, 1)})
    return {"shards": shards, "skipped_suites": skipped, "notes": notes,
            "total_tests": sum(s["n_tests"] for s in shards), "total_est_minutes_serial": round(sum(s["est_minutes"] for s in shards), 1),
            "longest_shard_minutes": max((s["est_minutes"] for s in shards), default=0), "dataset_version": __import__("bench").DATASET_VERSION}


def matrix_json(plan):
    inc = [{k: v for k, v in s.items() if k != "case_ids"} for s in plan["shards"]]
    return json.dumps({"include": inc}, separators=(",", ":"))
