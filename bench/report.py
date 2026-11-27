"""Summaries (shard + overall). Accuracy is computed ONLY over graded tests (passed/failed/truncated/timeout);
errors, grading errors and skips are reported separately and never counted as model failures."""
import csv, json, os
from collections import Counter, defaultdict
from pathlib import Path
from .util import mean, percentile, read_jsonl, sha256_file, utcnow, write_json

GRADED = ("passed", "failed", "truncated", "timeout")
ALL_STATUSES = ("passed", "failed", "truncated", "timeout", "error", "grading_error", "skipped", "measured")


def _acc(rows):
    g = [r for r in rows if r["status"] in GRADED]
    return {"n": len(rows), "graded": len(g), "passed": sum(r["status"] == "passed" for r in g),
            "accuracy": round(sum(r["status"] == "passed" for r in g) / len(g), 4) if g else None,
            "mean_score": round(mean([r.get("score") for r in g]) or 0, 4) if g else None}


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(sum(vals) / len(vals), 3), "median": round(percentile(vals, 50), 3), "p95": round(percentile(vals, 95), 3),
            "min": round(min(vals), 3), "max": round(max(vals), 3), "n": len(vals)}


def consistency(records):
    """Per family: agreement = share of variants giving the modal normalized answer (only meaningful when
    every variant has the same expected answer); accuracy = share correct. Stochastic wording differences
    never count as disagreement because we compare *extracted* answers, not raw text."""
    fams = defaultdict(list)
    for r in records:
        if r.get("family") and r["status"] in GRADED:
            fams[r["family"]].append(r)
    out = {}
    for f, rs in fams.items():
        answers = []
        for r in rs:
            ex = (r.get("evaluation") or {}).get("extracted")
            answers.append(json.dumps(ex, sort_keys=True, default=str).strip().lower() if isinstance(ex, (dict, list)) else str(ex).strip().lower())
        same_expected = len({r.get("expected_key") for r in rs}) == 1
        modal = Counter(answers).most_common(1)[0][1] if answers else 0
        out[f] = {"variants": len(rs), "accuracy": round(sum(r["status"] == "passed" for r in rs) / len(rs), 3),
                  "answer_agreement": round(modal / len(rs), 3) if same_expected else None,
                  "all_correct": all(r["status"] == "passed" for r in rs),
                  "by_variant": {f"{r['variant']}#{r['repeat']}": r["status"] for r in rs}}
    return out


def long_context(records):
    rows = [r for r in records if r["suite"] == "long_context" and r["status"] in GRADED]
    by_len, by_pos = defaultdict(list), defaultdict(list)
    for r in rows:
        by_len[r.get("context_tokens_target")].append(r)
        by_pos[r.get("needle_position")].append(r)
    lat = {k: _stats([r["metrics"]["latency_s"] for r in v if r.get("metrics")]) for k, v in by_len.items()}
    return {"by_context_tokens": {str(k): {**_acc(v), "prompt_tokens": _stats([r["metrics"]["prompt_tokens"] for r in v if r.get("metrics")]), "latency_s": lat[k]}
                                  for k, v in sorted(by_len.items(), key=lambda x: x[0] or 0)},
            "by_needle_position": {str(k): _acc(v) for k, v in by_pos.items()}} if rows else None


def summarize(records):
    counts = Counter(r["status"] for r in records)
    chat = [r for r in records if r.get("metrics") and r["status"] in ("passed", "failed", "truncated") and r["kind"] != "proc"]
    by_level, by_cat = defaultdict(list), defaultdict(list)
    for r in records:
        by_level[r["level"]].append(r)
        by_cat[r.get("category") or "-"].append(r)
    s = {"schema": "bonsai-bench/summary/v1", "total": len(records), "counts": {k: counts.get(k, 0) for k in ALL_STATUSES}, **_acc(records),
         "truncation_rate": round(counts["truncated"] / max(1, len([r for r in records if r["status"] in GRADED])), 4),
         "latency_s": _stats([r["metrics"]["latency_s"] for r in chat]),
         "tokens_per_s": _stats([r["metrics"]["tokens_per_s"] for r in chat if r["metrics"].get("tokens_per_s_source") == "server_timings"]),
         "prefill_tokens_per_s": _stats([r["metrics"].get("prefill_tokens_per_s") for r in chat]),
         "tokens": {"prompt": sum((r.get("metrics") or {}).get("prompt_tokens") or 0 for r in chat), "completion": sum((r.get("metrics") or {}).get("completion_tokens") or 0 for r in chat)},
         "by_level": {str(k): _acc(v) for k, v in sorted(by_level.items())}, "by_category": {k: _acc(v) for k, v in sorted(by_cat.items())},
         "failure_classes": dict(Counter(r["failure_class"] for r in records if r.get("failure_class"))),
         "skip_reasons": dict(Counter(r["skip_reason"] for r in records if r["status"] == "skipped")),
         "heuristic_graded": sum(1 for r in records if (r.get("evaluation") or {}).get("heuristic")),
         "consistency": consistency(records) or None, "long_context": long_context(records),
         "perf": {r["test_id"]: r["metrics"] for r in records if r["suite"] in ("performance", "stress") and r.get("metrics")} or None,
         "failures": [{"test": r["test_id"], "rep": r["repeat"], "status": r["status"],
                       "why": next((f"{c['name']} {c.get('detail', '')}".strip() for c in (r.get("evaluation") or {}).get("checks", []) if not c["passed"]),
                                   str(r.get("error") or r.get("skip_reason") or ""))[:160]}
                      for r in records if r["status"] in ("failed", "truncated", "timeout", "error", "grading_error")][:60]}
    return s


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def _f(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def shard_markdown(s):
    c = s["counts"]
    L = [f"### Shard `{s.get('shard', '?')}` — {s.get('suite', '')}",
         f"**{c['passed']} passed · {c['failed']} failed · {c['truncated']} truncated · {c['timeout']} timeout · {c['error']} errors · {c['skipped']} skipped** "
         f"→ accuracy **{_pct(s['accuracy'])}** over {s['graded']} graded tests (wall {s.get('wall_seconds', '–')} s, {s.get('server_restarts', 0)} server restarts)", ""]
    if s.get("fatal"):
        L.append(f"> ⚠️ **{s['fatal'][:300]}**\n")
    lat, tps = s.get("latency_s"), s.get("tokens_per_s")
    if lat:
        L.append(f"latency mean/median/p95: {_f(lat['mean'])} / {_f(lat['median'])} / {_f(lat['p95'])} s" + (f" · decode {_f(tps['mean'], 2)} tok/s" if tps else ""))
    pk = s.get("resource_peaks") or {}
    if pk:
        L.append(f"peak server RSS {_f(pk.get('server_rss_mb'), 0)} MB · min MemAvailable {_f(pk.get('min_mem_available_mb'), 0)} MB · peak swap {_f(pk.get('peak_swap_used_mb'), 0)} MB")
    if s["by_level"]:
        L += ["", "| level | tests | passed | accuracy |", "|---|---|---|---|"]
        L += [f"| L{k} | {v['graded']} | {v['passed']} | {_pct(v['accuracy'])} |" for k, v in s["by_level"].items() if v["graded"]]
    if s["failures"]:
        L += ["", "<details><summary>Failures / errors</summary>", ""]
        L += [f"- `{f['test']}` r{f['rep']} **{f['status']}** — {f['why']}" for f in s["failures"][:25]]
        L.append("\n</details>")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------ aggregation across shards
def _load_shards(root):
    shards = []
    for sj in sorted(Path(root).rglob("summary.json")):
        d = sj.parent
        if not (d / "results.jsonl").exists() and not (d / "metadata").exists():
            continue
        shards.append({"dir": d, "summary": json.loads(sj.read_text()), "records": read_jsonl(d / "results.jsonl"),
                       "config": _j(d / "metadata" / "run_config.json"), "model": _j(d / "metadata" / "model_info.json"),
                       "runtime": _j(d / "metadata" / "runtime_info.json"), "env": _j(d / "diagnostics" / "environment.json") or _j(d / "environment.json")})
    return shards


def _j(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def aggregate(in_dir, out_dir, run_id="local", expected_shards=None):
    out = Path(out_dir)
    (out / "suite_summaries").mkdir(parents=True, exist_ok=True)
    shards = _load_shards(in_dir)
    records = [r for s in shards for r in s["records"]]
    by_suite = defaultdict(list)
    for r in records:
        by_suite[r["suite"]].append(r)
    suites = {k: summarize(v) for k, v in by_suite.items()}
    overall = summarize(records)
    present = {s["summary"]["shard"] for s in shards}
    missing = sorted(set(expected_shards or []) - present)
    first = next((s for s in shards if s["config"]), None)
    hw = sorted({(s["env"] or {}).get("cpu_model") or "?" for s in shards})
    meta = {"schema": "bonsai-bench/overall/v1", "run_id": run_id, "generated_at": utcnow(), "dataset_version": (first or {}).get("config", {}).get("dataset_version") if first else None,
            "git_sha": os.environ.get("GITHUB_SHA"), "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
            "inputs": (first["config"] or {}).get("inputs") if first else None, "model": (first or {}).get("model"),
            "runtime": ((first or {}).get("runtime") or {}).get("version_output"), "hardware": hw,
            "shards_total": len(expected_shards or present), "shards_reported": len(present), "shards_missing": missing,
            "shards_fatal": {s["summary"]["shard"]: s["summary"]["fatal"] for s in shards if s["summary"].get("fatal")},
            "server_restarts": sum(s["summary"].get("server_restarts", 0) for s in shards),
            "peak_server_rss_mb": max([(s["summary"].get("resource_peaks") or {}).get("server_rss_mb") or 0 for s in shards] or [0]),
            "min_mem_available_mb": min([v for v in [(s["summary"].get("resource_peaks") or {}).get("min_mem_available_mb") for s in shards] if v is not None] or [None])
            if shards else None, "overall": overall, "suites": suites}
    write_json(out / "overall.json", meta)
    for k, v in suites.items():
        write_json(out / "suite_summaries" / f"{k}.json", v)
        (out / "suite_summaries" / f"{k}.md").write_text(shard_markdown({**v, "shard": k, "suite": k}))
    cols = ["suite", "test_id", "repeat", "level", "category", "family", "variant", "status", "failure_class", "score", "duration_s",
            "prompt_tokens", "completion_tokens", "tokens_per_s", "latency_s", "ttft_s", "finish_reason", "shard"]
    with open(out / "overall.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            m, rs = r.get("metrics") or {}, r.get("response") or {}
            w.writerow([r["suite"], r["test_id"], r["repeat"], r["level"], r.get("category"), r.get("family"), r.get("variant"), r["status"], r.get("failure_class"),
                        r.get("score"), r.get("duration_s"), m.get("prompt_tokens"), m.get("completion_tokens"), m.get("tokens_per_s"), m.get("latency_s"),
                        m.get("ttft_s"), rs.get("finish_reason"), r.get("shard")])
    import gzip
    with gzip.open(out / "results_compact.jsonl.gz", "wt", encoding="utf-8") as gz:  # small enough to commit for longitudinal tracking
        for r in records:
            gz.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    (out / "overall.md").write_text(overall_markdown(meta))
    try:
        _charts(meta, out)
    except Exception as e:  # charts are optional
        (out / "charts_skipped.txt").write_text(repr(e))
    return meta


def write_manifest(root, run_id="local"):
    root = Path(root)
    files = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    write_json(root / "manifest.json", {"run_id": run_id, "generated_at": utcnow(), "n_files": len(files), "total_bytes": sum(f["bytes"] for f in files), "files": files})


def overall_markdown(m):
    o, suites = m["overall"], m["suites"]
    c = o["counts"]
    L = [f"# Bonsai-2-27B benchmark — run `{m['run_id']}`", "",
         f"- dataset `{m.get('dataset_version')}` · commit `{(m.get('git_sha') or 'n/a')[:10]}` · hardware: {', '.join(m['hardware'])}",
         f"- shards reported **{m['shards_reported']}/{m['shards_total']}**" + (f" — ⚠️ missing: {', '.join(m['shards_missing'])}" if m["shards_missing"] else ""),
         f"- **{o['total']} test executions**: {c['passed']} passed · {c['failed']} failed · {c['truncated']} truncated (hit max_tokens) · {c['timeout']} timeout · "
         f"{c['error']} errors · {c['grading_error']} grader errors · {c['skipped']} skipped · {c['measured']} measured-only",
         f"- **overall accuracy {_pct(o['accuracy'])}** over {o['graded']} graded tests (skips/errors excluded) · mean score {_f(o['mean_score'], 3)}",
         f"- server restarts: {m['server_restarts']} · peak server RSS {_f(m['peak_server_rss_mb'], 0)} MB · lowest MemAvailable {_f(m['min_mem_available_mb'], 0)} MB", ""]
    if m["shards_fatal"]:
        L += ["## ⚠️ Infrastructure problems"] + [f"- `{k}`: {v[:200]}" for k, v in m["shards_fatal"].items()] + [""]
    L += ["## Accuracy by suite", "", "| suite | graded | passed | accuracy | trunc | err | skip | median latency s | decode tok/s |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in suites.items():
        L.append(f"| {k} | {v['graded']} | {v['passed']} | {_pct(v['accuracy'])} | {v['counts']['truncated']} | {v['counts']['error'] + v['counts']['grading_error']} | {v['counts']['skipped']} | "
                 f"{_f((v['latency_s'] or {}).get('median'))} | {_f((v['tokens_per_s'] or {}).get('mean'), 2)} |")
    L += ["", "## Accuracy by difficulty level (all suites)", "", "| level | graded | accuracy |", "|---|---|---|"]
    L += [f"| L{k} | {v['graded']} | {_pct(v['accuracy'])} |" for k, v in o["by_level"].items()]
    L += ["", "<details><summary><b>Accuracy by category</b> (weakest first)</summary>", "", "| category | graded | accuracy |", "|---|---|---|"]
    L += [f"| {k} | {v['graded']} | {_pct(v['accuracy'])} |" for k, v in sorted(o["by_category"].items(), key=lambda x: (x[1]["accuracy"] is None, x[1]["accuracy"] or 0)) if v["graded"]]
    L += ["", "</details>"]
    cons = suites.get("consistency", {}).get("consistency")
    if cons:
        L += ["", "## Consistency", "", "| family | variants | accuracy | answer agreement |", "|---|---|---|---|"]
        L += [f"| {k} | {v['variants']} | {_pct(v['accuracy'])} | {_pct(v['answer_agreement'])} |" for k, v in cons.items()]
    lc = suites.get("long_context", {}).get("long_context")
    if lc:
        L += ["", "## Long context", "", "| target ctx | prompt tokens (actual) | accuracy | median latency s |", "|---|---|---|---|"]
        L += [f"| {k} | {_f((v['prompt_tokens'] or {}).get('mean'), 0)} | {_pct(v['accuracy'])} | {_f((v['latency_s'] or {}).get('median'), 0)} |" for k, v in lc["by_context_tokens"].items()]
        L += ["", "| needle position | accuracy |", "|---|---|"] + [f"| {k} | {_pct(v['accuracy'])} |" for k, v in lc["by_needle_position"].items()]
    perf = suites.get("performance", {}).get("perf")
    if perf:
        L += ["", "## Performance (single GitHub-hosted runner; treat small differences as noise)", "", "| experiment | latency s | TTFT s | decode tok/s | prefill tok/s |", "|---|---|---|---|---|"]
        for k, v in perf.items():
            g = lambda n: (v.get(n) or {}).get("mean") if isinstance(v.get(n), dict) else None
            if any(g(n) is not None for n in ("latency_s", "decode_tps")):
                L.append(f"| {k} | {_f(g('latency_s'), 2)} | {_f(g('ttft_s'), 2)} | {_f(g('decode_tps'), 2)} | {_f(g('prefill_tps'), 2)} |")
        ca = perf.get("perf-09-cache-reuse")
        if ca:
            L.append(f"\nPrompt cache: second identical request cached {ca['second'].get('cached_tokens')} tokens, latency speed-up ×{_f(ca.get('latency_speedup_x'), 2)}")
        cc = perf.get("perf-10-concurrency-2")
        if cc:
            L.append(f"\nConcurrency ×2: aggregate {_f(cc.get('aggregate_tps'), 2)} tok/s vs single {_f(cc.get('single_tps'), 2)} tok/s (gain ×{_f(cc.get('throughput_gain_x'), 2)})")
    st = suites.get("stress", {})
    if st:
        L += ["", "## Stress / reliability", "", "| test | status | detail |", "|---|---|---|"]
        for r in st.get("failures", []):
            L.append(f"| {r['test']} | {r['status']} | {r['why']} |")
        L.append(f"\n{st['counts']['passed']} passed, {st['counts']['failed']} failed, {st['counts']['skipped']} skipped.")
    if o["skip_reasons"]:
        L += ["", "## Environment limitations (skipped, NOT scored as model failures)"] + [f"- {n}× {k}" for k, n in o["skip_reasons"].items()]
    if o["failure_classes"]:
        L += ["", "## Failure classes", ""] + [f"- {k}: {n}" for k, n in o["failure_classes"].items()]
    if o["heuristic_graded"]:
        L += ["", f"_Note: {o['heuristic_graded']} results (safety/refusal/uncertainty) use heuristic marker matching and should be spot-checked in `transcript.md`._"]
    L += ["", "## All failures (first 60)", ""] + [f"- `{f['test']}` r{f['rep']} **{f['status']}** — {f['why']}" for f in o["failures"]]
    return "\n".join(L) + "\n"


def _charts(m, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    suites = [(k, v) for k, v in m["suites"].items() if v["accuracy"] is not None]
    if suites:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([k for k, _ in suites], [100 * v["accuracy"] for _, v in suites], color="#3366cc")
        ax.set_ylabel("accuracy % (graded tests)"); ax.set_ylim(0, 100); plt.xticks(rotation=35, ha="right"); plt.tight_layout()
        fig.savefig(out / "chart_suite_accuracy.png", dpi=110); plt.close(fig)
    lv = m["overall"]["by_level"]
    if lv:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar([f"L{k}" for k in lv], [100 * (v["accuracy"] or 0) for v in lv.values()], color="#109618")
        ax.set_ylabel("accuracy %"); ax.set_ylim(0, 100); plt.tight_layout()
        fig.savefig(out / "chart_level_accuracy.png", dpi=110); plt.close(fig)
    lc = (m["suites"].get("long_context") or {}).get("long_context")
    if lc:
        ks = list(lc["by_context_tokens"])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(ks, [100 * (lc["by_context_tokens"][k]["accuracy"] or 0) for k in ks], marker="o")
        ax.set_xlabel("target context tokens"); ax.set_ylabel("accuracy %"); ax.set_ylim(0, 105); plt.tight_layout()
        fig.savefig(out / "chart_long_context.png", dpi=110); plt.close(fig)
