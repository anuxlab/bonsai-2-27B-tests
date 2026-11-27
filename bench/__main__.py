import argparse, json, os, sys, time
from pathlib import Path


def tobool(x):
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def add_plan_args(p):
    p.add_argument("--suites", default="all", help="comma list or 'all'")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--long-context-levels", default="1k,2k,4k")
    p.add_argument("--enable-vision", type=tobool, default=True)
    p.add_argument("--enable-long-context", type=tobool, default=True)
    p.add_argument("--test-filter", default="", help="regex on test ids (re-run only some tests)")
    p.add_argument("--shard-minutes", type=float, default=60)
    p.add_argument("--max-shards", type=int, default=20)
    p.add_argument("--ctx", type=int, default=0, help="override llama-server context size")
    p.add_argument("--stress-level", default="medium", choices=["low", "medium", "high"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan"); add_plan_args(p)
    p.add_argument("--github-output", default=None); p.add_argument("--summary", action="store_true")

    p = sub.add_parser("preflight")
    p.add_argument("--out", required=True); p.add_argument("--policy", default="strict", choices=["strict", "permissive"])
    p.add_argument("--require-disk-gb", type=float, default=25); p.add_argument("--min-ram-gb", type=float, default=12)

    p = sub.add_parser("run"); add_plan_args(p)
    p.add_argument("--shard", required=True); p.add_argument("--out-dir", required=True); p.add_argument("--run-id", default="local")
    p.add_argument("--llama-server", default="./runtime/llama-server"); p.add_argument("--model", default="./models/model.gguf")
    p.add_argument("--mmproj", default=None); p.add_argument("--model-info", default=None)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4); p.add_argument("--port", type=int, default=8080)
    p.add_argument("--base-url", default=None, help="use an already running OpenAI-compatible server (e.g. http://127.0.0.1:8099/v1)")
    p.add_argument("--assume-vision", type=tobool, default=False)
    p.add_argument("--ready-timeout", type=int, default=900)
    p.add_argument("--max-tokens", type=int, default=0, help="global cap on max_tokens (0 = per-test values)")
    p.add_argument("--temperature", type=float, default=None); p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=1234); p.add_argument("--thinking", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--retries", type=int, default=2); p.add_argument("--budget-minutes", type=float, default=150)
    p.add_argument("--max-restarts", type=int, default=2); p.add_argument("--min-mem-available-mb", type=int, default=700)
    p.add_argument("--monitor-interval", type=float, default=5.0)

    p = sub.add_parser("aggregate")
    p.add_argument("--in-dir", required=True); p.add_argument("--out-dir", required=True); p.add_argument("--run-id", default="local")
    p.add_argument("--expected-shards", default="", help="comma list of shard ids that should exist")

    p = sub.add_parser("manifest"); p.add_argument("--root", required=True); p.add_argument("--run-id", default="local")
    p = sub.add_parser("selftest"); p.add_argument("--e2e", action="store_true", help="also run every suite against the fake server")
    p = sub.add_parser("fake-server"); p.add_argument("--port", type=int, default=8099)

    a = ap.parse_args(argv)
    if a.cmd == "plan":
        from .plan import build_plan, matrix_json
        kw = {k: v for k, v in vars(a).items() if k not in ("cmd", "github_output", "summary")}
        kw["ctx_override"] = kw.pop("ctx")
        plan = build_plan(**kw)
        if not plan["shards"]:
            print("::error::the selection produced zero tests", file=sys.stderr); return 2
        print(json.dumps({k: v for k, v in plan.items() if k != "shards"}, indent=1))
        for s in plan["shards"]:
            print(f"  {s['shard']:16} {s['n_tests']:4} tests  ~{s['est_minutes']:6.1f} min  ctx={s['ctx']} parallel={s['parallel']}")
        if a.github_output:
            with open(a.github_output, "a") as f:
                f.write(f"matrix={matrix_json(plan)}\n")
                f.write(f"expected_shards={','.join(s['shard'] for s in plan['shards'])}\n")
        if a.summary and os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write("## Plan\n\n| shard | tests | est. min | ctx |\n|---|---|---|---|\n" + "\n".join(f"| {s['shard']} | {s['n_tests']} | {s['est_minutes']} | {s['ctx']} |" for s in plan["shards"]) + "\n")
                for sk in plan["skipped_suites"]:
                    f.write(f"\n- skipped suite **{sk['suite']}**: {sk['reason']}")
        return 0
    if a.cmd == "preflight":
        from .resources import preflight
        ok, rep = preflight(a.out, a.policy, a.require_disk_gb, a.min_ram_gb)
        for f in rep["findings"]:
            print(f"[{f['level']}] {f['msg']}")
        return 0 if ok else 2
    if a.cmd == "run":
        from .runner import run_cli
        a.ctx = a.ctx
        return run_cli(a)
    if a.cmd == "aggregate":
        from .report import aggregate
        m = aggregate(a.in_dir, a.out_dir, a.run_id, [x for x in a.expected_shards.split(",") if x])
        print(Path(a.out_dir, "overall.md").read_text())
        return 0
    if a.cmd == "manifest":
        from .report import write_manifest
        write_manifest(a.root, a.run_id); return 0
    if a.cmd == "selftest":
        from .selftest import run
        rc = run()
        if a.e2e and rc == 0:
            from .e2e import run_e2e
            rc = run_e2e()
        return rc
    if a.cmd == "fake-server":
        from .fake_server import serve
        serve(a.port); print(f"fake server on :{a.port}"); time.sleep(10 ** 9)


if __name__ == "__main__":
    sys.exit(main())
