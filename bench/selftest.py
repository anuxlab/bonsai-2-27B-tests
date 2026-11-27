"""`python -m bench selftest` — validates the test suite itself: every reference answer must pass its grader and
every `wrong` answer must fail. This is what keeps the benchmark honest (a broken grader looks like a model failure)."""
import json
from . import cases as C
from .graders import grade
from .plan import build_plan

KNOWN = {"numeric", "exact", "choice", "number_list", "lines", "contains_all", "contains_any", "not_contains", "regex", "json", "constraints", "order",
         "refusal", "uncertainty", "code", "unit_tests", "trace", "none"}


def _types(g):
    return [x["type"] for x in (g if isinstance(g, list) else [g])] + ([t for x in (g if isinstance(g, list) else [g]) if x["type"] == "trace" and x.get("final")
                                                                          for t in _types(x["final"])])


def run():
    problems, ok, ids = [], 0, set()
    for s in C.SUITES:
        for c in C.load_suite(s, ("1k", "2k", "4k", "8k")):
            if c["id"] in ids:
                problems.append(f"duplicate id {c['id']}")
            ids.add(c["id"])
            for t in _types(c["grade"]):
                if t not in KNOWN:
                    problems.append(f"{c['id']}: unknown grader {t}")
            for f in ("level", "category", "max_tokens", "est_out_tokens"):
                if f not in c:
                    problems.append(f"{c['id']}: missing {f}")
            if c["kind"] != "chat":
                continue
            ref, wrong = c.get("reference"), c.get("wrong")
            if ref is None:
                continue
            r = grade(c["grade"], ref)
            if r["passed"] is not True:
                problems.append(f"{c['id']}: REFERENCE FAILS its own grader: {[x for x in r['checks'] if not x['passed']][:2]} {r.get('grader_error','')}")
            elif wrong is not None:
                w = grade(c["grade"], wrong)
                if w["passed"] is True:
                    problems.append(f"{c['id']}: WRONG answer is accepted by the grader")
                else:
                    ok += 1
            else:
                ok += 1
    plan = build_plan()
    covered = sum(len(s["case_ids"]) for s in plan["shards"])
    print(f"selftest: {ok} cases verified (reference passes, wrong fails); {len(ids)} cases total; plan covers {covered} tests in {len(plan['shards'])} shards")
    for p in problems:
        print("  PROBLEM:", p)
    return 1 if problems else 0
