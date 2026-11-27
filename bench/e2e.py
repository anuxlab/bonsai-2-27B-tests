"""End-to-end smoke test of the whole pipeline against the fake server."""
import shutil, tempfile
from pathlib import Path
from .__main__ import main
from .fake_server import serve
from .plan import build_plan


def run_e2e(port=8099, keep=None):
    tmp = Path(keep or tempfile.mkdtemp(prefix="bonsai_e2e_"))
    srv = serve(port)
    plan = build_plan(suites="all", enable_long_context=True, long_context_levels="1k,2k,4k")
    rc_all = 0
    for sh in plan["shards"]:
        rc = main(["run", "--shard", sh["shard"], "--out-dir", str(tmp / "shards" / f"shard-{sh['shard']}"), "--base-url", f"http://127.0.0.1:{port}/v1",
                   "--assume-vision", "true", "--run-id", "e2e", "--monitor-interval", "1", "--stress-level", "high", "--retries", "0"])
        rc_all |= rc
    rc_all |= main(["aggregate", "--in-dir", str(tmp / "shards"), "--out-dir", str(tmp / "report"), "--run-id", "e2e",
                    "--expected-shards", ",".join(s["shard"] for s in plan["shards"])])
    srv.shutdown()
    print("e2e output in", tmp)
    return rc_all
