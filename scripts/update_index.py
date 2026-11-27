"""Append this run to benchmark-index.json and write a regression note vs the previous *comparable* run."""
import json, sys, datetime
from pathlib import Path

report, results_root, run_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cur = json.loads((report / "overall.json").read_text())
idx_path = results_root / "benchmark-index.json"
idx = json.loads(idx_path.read_text()) if idx_path.exists() else {"runs": []}
inp = cur.get("inputs") or {}
key = {"dataset": cur.get("dataset_version"), "thinking": inp.get("thinking"), "temperature": inp.get("temperature"), "ctx": inp.get("ctx"),
       "repeat": inp.get("repeat"), "model_sha": ((cur.get("model") or {}).get("model") or {}).get("sha256"), "runtime": (cur.get("runtime") or "")[:80]}
entry = {"run_id": run_id, "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "git_sha": cur.get("git_sha"), "workflow_run_id": cur.get("workflow_run_id"),
         "comparability_key": key, "overall_accuracy": cur["overall"]["accuracy"], "counts": cur["overall"]["counts"],
         "suite_accuracy": {k: v["accuracy"] for k, v in cur["suites"].items()}, "hardware": cur.get("hardware"), "shards_missing": cur.get("shards_missing")}
prev = next((r for r in reversed(idx["runs"]) if r["comparability_key"] == key), None)
lines = [f"# Regression check — run {run_id}", ""]
if not prev:
    lines.append("No earlier run with the same dataset/model/runtime/sampling configuration — nothing to compare.")
else:
    lines += [f"Compared with `{prev['run_id']}` (same dataset, model hash, runtime, sampling). Hardware may differ; accuracy differences below ~1 test per suite are noise.", "",
              "| suite | previous | current | Δ pts |", "|---|---|---|---|"]
    for s, a in entry["suite_accuracy"].items():
        p = prev["suite_accuracy"].get(s)
        if a is None or p is None:
            continue
        d = 100 * (a - p)
        lines.append(f"| {s} | {100*p:.1f}% | {100*a:.1f}% | {d:+.1f}{' ⚠️' if d <= -10 else ''} |")
idx["runs"].append(entry)
idx_path.parent.mkdir(parents=True, exist_ok=True)
idx_path.write_text(json.dumps(idx, indent=1))
(results_root / "latest").mkdir(parents=True, exist_ok=True)
(results_root / "latest" / "regression.md").write_text("\n".join(lines) + "\n")
