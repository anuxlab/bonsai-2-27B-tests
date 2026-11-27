"""Execute one shard: supervise llama-server, run every test, write a structured result for each (even failures)."""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
from . import DATASET_VERSION, __version__, cases as C, generators, resources
from .client import Client, LLMError
from .graders import grade
from .plan import build_plan
from .procs import Ctx, run_proc
from .server import ServerManager, ServerStartError
from .tools import ToolEnv, run_agent
from .util import append_jsonl, sha256_file, slug, trunc, utcnow, write_json


class AbortShard(Exception):
    pass


class Runner:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out_dir)
        self.t_start = time.monotonic()
        self.records, self.server, self.monitor, self.client = [], None, None, None
        for d in ("metadata", "diagnostics", "tests", "images"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.events = self.out / "diagnostics" / "events.log"

    # ---------------------------------------------------------------- logging
    def log(self, msg):
        line = f"[{utcnow()}] {msg}"
        print(line, flush=True)
        with self.events.open("a") as f:
            f.write(line + "\n")

    # ---------------------------------------------------------------- setup
    def plan(self):
        a = self.a
        plan = build_plan(suites=a.suites, repeat=a.repeat, long_context_levels=a.long_context_levels, enable_vision=a.enable_vision,
                          enable_long_context=a.enable_long_context, test_filter=a.test_filter, shard_minutes=a.shard_minutes,
                          max_shards=a.max_shards, ctx_override=a.ctx, stress_level=a.stress_level)
        sh = next((s for s in plan["shards"] if s["shard"] == a.shard), None)
        if sh is None:
            raise SystemExit(f"shard {a.shard!r} not in plan: {[s['shard'] for s in plan['shards']]}")
        lcl = tuple(x.strip() for x in a.long_context_levels.split(","))
        cmap = {c["id"]: c for c in C.load_suite(sh["suite"], lcl)}
        return sh, [cmap[i] for i in sh["case_ids"]]

    def write_metadata(self, sh):
        a = self.a
        cfg = {"schema": "bonsai-bench/run-config/v1", "bench_version": __version__, "dataset_version": DATASET_VERSION,
               "run_id": a.run_id, "shard": sh, "created_at": utcnow(),
               "inputs": {k: v for k, v in vars(a).items() if k not in ("func",)},
               "git_sha": os.environ.get("GITHUB_SHA"), "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
               "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "workflow": os.environ.get("GITHUB_WORKFLOW")}
        write_json(self.out / "metadata" / "run_config.json", cfg)
        for name in ("model_info.json",):
            src = Path(a.model_info) if a.model_info else None
            if src and src.exists():
                (self.out / "metadata" / name).write_text(src.read_text())

    def start_server(self, sh):
        a = self.a
        if a.base_url:  # external server (CI of the harness itself, or a manually started llama-server)
            self.client = Client(a.base_url, recover=self.ensure_server)
            self.vision_supported = a.assume_vision
            self.log(f"using external server {a.base_url}")
            return
        mm = a.mmproj if (sh["vision"] and a.mmproj and Path(a.mmproj).exists()) else None
        self.server = ServerManager(a.llama_server, a.model, mm, ctx=sh["ctx"], threads=a.threads, parallel=sh["parallel"],
                                    port=a.port, log_path=self.out / "diagnostics" / "server.log")
        self.log("starting: " + " ".join(self.server.command()))
        secs = self.server.start(ready_timeout=a.ready_timeout)
        self.log(f"server ready in {secs}s")
        self.client = Client(self.server.base + "/v1", recover=self.ensure_server)
        self.vision_supported = self.server.supports_vision()
        try:
            v = subprocess.run([a.llama_server, "--version"], capture_output=True, text=True, timeout=20)
            ver = (v.stdout + v.stderr).strip()[-600:]
        except Exception as e:
            ver = f"unavailable: {e}"
        write_json(self.out / "metadata" / "runtime_info.json", {"llama_server": a.llama_server, "version_output": ver, "command": self.server.command(),
                                                              "ctx": sh["ctx"], "parallel": sh["parallel"], "threads": a.threads,
                                                              "startup": self.server.startups, "mmproj_loaded": bool(mm), "vision_supported": self.vision_supported})

    def ensure_server(self):
        """Called before retries and after infra errors. Restarts a dead server (bounded)."""
        if not self.server:
            return
        if self.server.alive() and self.server.healthy(10):
            return
        info = self.server.exit_info()
        self.log(f"!! server not healthy: alive={self.server.alive()} exit={info}")
        snap = resources.snapshot()
        self.log(f"!! resources: mem_available={snap['mem_available_mb']}MB swap_used={snap['swap_used_mb']}MB psi={snap['psi']}")
        if self.server.restart_count >= self.a.max_restarts:
            raise AbortShard(f"server restarted {self.server.restart_count} times; giving up (last exit {info})")
        if self.server.alive():  # alive but unresponsive: kill it, a hung server is useless
            self.server.stop(grace=5)
        self.log("restarting server …")
        self.server.restart(self.a.ready_timeout)

    # ---------------------------------------------------------------- one test
    def params_for(self, case, rep):
        a = self.a
        mt = min(case["max_tokens"], a.max_tokens) if a.max_tokens else case["max_tokens"]
        think = {"on": True, "off": False}.get(a.thinking, case.get("thinking"))
        timeout = case.get("timeout") or int(120 + case["est_prompt_tokens"] / (C.PREFILL_TPS * 0.6) + mt / (C.DECODE_TPS * 0.5))
        return {"max_tokens": mt, "temperature": a.temperature if a.temperature is not None else case.get("temperature", 0.0),
                "top_p": a.top_p, "seed": a.seed + rep, "thinking": think, "timeout": timeout, "retries": a.retries}

    def build_messages(self, case):
        msgs = []
        if case.get("system"):
            msgs.append({"role": "system", "content": case["system"]})
        meta = {}
        if case.get("image"):
            png, sha, uri = generators.make_image(case["image"])
            (self.out / "images" / f"{slug(case['id'])}.png").write_bytes(png)
            meta = {"image_sha256": sha, "image_spec": case["image"], "image_bytes": len(png)}
            msgs.append({"role": "user", "content": [{"type": "text", "text": case["prompt"]}, {"type": "image_url", "image_url": {"url": uri}}]})
        else:
            msgs.append({"role": "user", "content": case["prompt"]})
        return msgs, meta

    def execute(self, case, rep):
        rec = {"schema": "bonsai-bench/test-result/v1", "run_id": self.a.run_id, "shard": self.a.shard, "suite": case["suite"], "test_id": case["id"],
               "repeat": rep, "dataset_version": DATASET_VERSION, "level": case["level"], "category": case.get("category"), "tags": case["tags"],
               "family": case.get("family"), "variant": case.get("variant"), "kind": case["kind"], "needle_position": case.get("needle_position"),
               "context_tokens_target": case.get("context_tokens_target"), "expected_key": json.dumps(case.get("reference"))[:80], "started_at": utcnow(), "status": None, "failure_class": None,
               "score": None, "passed": None, "error": None, "attempts": [], "resources_before": resources.snapshot(self.server.pid if self.server else None)}
        t0 = time.perf_counter()
        try:
            if case["kind"] == "proc":
                self._exec_proc(case, rec)
            elif case["kind"] == "agent":
                self._exec_agent(case, rec)
            else:
                self._exec_chat(case, rec)
        except LLMError as e:
            self._llm_error(rec, e)
        except AbortShard:
            raise
        except ImportError as e:
            rec.update(status="error", failure_class="configuration", error={"type": "ImportError", "message": str(e)})
        except Exception as e:
            rec.update(status="error", failure_class="evaluation" if rec.get("response") else "infrastructure",
                       error={"type": type(e).__name__, "message": str(e)[:500], "trace": traceback.format_exc()[-1500:]})
        rec["duration_s"] = round(time.perf_counter() - t0, 3)
        rec["finished_at"] = utcnow()
        rec["resources_after"] = resources.snapshot(self.server.pid if self.server else None)
        return rec

    def _llm_error(self, rec, e):
        rec["attempts"] = e.attempts
        rec["error"] = {"type": e.kind, "message": str(e)[:500], "http_status": e.status, "body": (e.body or "")[:500]}
        if e.kind == "timeout":
            rec.update(status="timeout", failure_class="timeout")
        elif e.kind == "http_4xx":
            rec.update(status="error", failure_class="configuration")
        else:
            rec.update(status="error", failure_class="infrastructure")
            if self.server and not self.server.alive():
                info = self.server.exit_info()
                rec["error"]["server_exit"] = info
                if info["probable_oom"]:
                    rec["failure_class"] = "resource"
        self.ensure_server()

    def _finish_grade(self, case, rec, text, finish_reason, ctx=None):
        g = grade(case["grade"], text, ctx)
        rec["evaluation"] = g
        if g.get("grader_error"):
            rec.update(status="grading_error", failure_class="evaluation", error={"type": "grader", "message": g["grader_error"]})
            return
        rec["score"], rec["passed"] = g["score"], g["passed"]
        if g["passed"] is None:
            rec["status"] = "measured"
        elif g["passed"]:
            rec["status"] = "passed"
        else:
            rec["status"] = "truncated" if finish_reason == "length" else "failed"
            rec["failure_class"] = "test_failure"

    def _metrics(self, res_list, rec):
        pt = sum(r["usage"].get("prompt_tokens", 0) for r in res_list)
        ct = sum(r["usage"].get("completion_tokens", 0) for r in res_list)
        el = sum(r["elapsed"] for r in res_list)
        tm = res_list[-1]["timings"] if res_list else {}
        srv_tps = tm.get("predicted_per_second") if len(res_list) == 1 else None
        rec["metrics"] = {"latency_s": round(el, 3), "ttft_s": res_list[0]["ttft"] if res_list and res_list[0]["ttft"] else None,
                          "prompt_tokens": pt, "completion_tokens": ct,
                          "tokens_per_s": round(srv_tps, 3) if srv_tps else (round(ct / el, 3) if el and ct else None),
                          "tokens_per_s_source": "server_timings" if srv_tps else "completion_tokens/latency (includes prefill)",
                          "prefill_tokens_per_s": tm.get("prompt_per_second") if len(res_list) == 1 else None,
                          "cached_tokens": tm.get("cache_n"), "reasoning_chars": sum(len(r["reasoning"] or "") for r in res_list)}

    def _exec_chat(self, case, rec):
        msgs, meta = self.build_messages(case)
        params = self.params_for(case, rec["repeat"])
        rec["config"] = {k: v for k, v in params.items()}
        rec["prompt"] = {"system": case.get("system"), "user": case["prompt"] if case["suite"] != "long_context" else trunc(case["prompt"], 1200) +
                         f"\n…[long document omitted; {len(case['prompt'])} chars; see run_config dataset_version]…\n" + case["prompt"][-400:], **meta}
        res = self.client.chat(msgs, **params)
        rec["request_id"], rec["attempts"] = res["request_id"], res["attempts"]
        rec["response"] = {"text": res["text"], "reasoning": res["reasoning"], "finish_reason": res["finish_reason"], "usage": res["usage"], "timings": res["timings"]}
        rec["raw_response"] = res["raw"]
        self._metrics([res], rec)
        self._finish_grade(case, rec, res["text"], res["finish_reason"])

    def _exec_agent(self, case, rec):
        params = self.params_for(case, rec["repeat"])
        params.pop("retries", None)
        params["retries"] = self.a.retries
        rec["config"] = dict(params, tools=case["tools"], max_steps=case.get("max_steps", 8))
        rec["prompt"] = {"system": case.get("system"), "user": case["prompt"], "fixtures": case.get("fixtures")}
        env = ToolEnv(case["tools"], case.get("fixtures"))
        out = run_agent(self.client, case, params, env, max_steps=case.get("max_steps", 8))
        rec["trace"], rec["steps"] = out["trace"], out["steps"]
        rec["response"] = {"text": out["final_text"], "reasoning": "\n---\n".join(s["reasoning"] for s in out["steps"] if s["reasoning"]),
                           "finish_reason": out["steps"][-1]["finish_reason"] if out["steps"] else None,
                           "usage": {"prompt_tokens": out["totals"]["prompt_tokens"], "completion_tokens": out["totals"]["completion_tokens"]}, "timings": {}}
        rec["attempts"] = [a for s in out["steps"] for a in s["attempts"]]
        res_like = [{"usage": s["usage"], "elapsed": s["elapsed_s"], "timings": s["timings"], "ttft": None, "reasoning": s["reasoning"]} for s in out["steps"]]
        self._metrics(res_like, rec)
        rec["metrics"]["agent_steps"] = len(out["steps"])
        rec["metrics"]["tool_calls"] = len(out["trace"])
        self._finish_grade(case, rec, out["final_text"], rec["response"]["finish_reason"],
                           {"trace": out["trace"], "state": out["state"], "incomplete": out["incomplete"], "format_violations": out["format_violations"]})

    def _exec_proc(self, case, rec):
        ctx = Ctx(self.client, self.server, self.monitor, self.a.stress_level, self.ctx_size, self.a.min_mem_available_mb,
                  ref_mode=bool(self.a.base_url))
        rec["config"] = {"params": case["params"], "stress_level": self.a.stress_level}
        rec["prompt"] = {"procedure": case["proc"]}
        r = run_proc(case, ctx)
        rec["evaluation"] = {"passed": r.get("passed"), "score": r.get("score"), "checks": r.get("checks", []), "method": "procedure", "heuristic": False}
        rec["metrics"] = r.get("metrics", {})
        rec["notes"] = r.get("notes")
        rec["score"], rec["passed"], rec["status"] = r.get("score"), r.get("passed"), r["status"]
        if r["status"] == "skipped":
            rec["skip_reason"] = r["skip_reason"]
            rec["failure_class"] = r.get("failure_class", "configuration")
        elif r["status"] == "failed":
            rec["failure_class"] = "test_failure"
        self.ensure_server()

    # ---------------------------------------------------------------- bookkeeping
    def skip_record(self, case, rep, reason, fclass="configuration"):
        return {"schema": "bonsai-bench/test-result/v1", "run_id": self.a.run_id, "shard": self.a.shard, "suite": case["suite"], "test_id": case["id"],
                "repeat": rep, "dataset_version": DATASET_VERSION, "level": case["level"], "category": case.get("category"), "tags": case["tags"],
                "family": case.get("family"), "variant": case.get("variant"), "kind": case["kind"], "status": "skipped", "skip_reason": reason,
                "failure_class": fclass, "score": None, "passed": None, "started_at": utcnow(), "finished_at": utcnow(), "duration_s": 0}

    def persist(self, rec):
        write_json(self.out / "tests" / f"{slug(rec['test_id'])}__r{rec['repeat']}.json", rec)
        compact = {k: v for k, v in rec.items() if k not in ("raw_response", "steps", "resources_before", "resources_after")}
        if isinstance(compact.get("response"), dict):
            compact["response"] = {**compact["response"], "text": trunc(compact["response"].get("text"), 4000), "reasoning": trunc(compact["response"].get("reasoning"), 1500)}
        if isinstance(compact.get("prompt"), dict):
            compact["prompt"] = {k: (trunc(v, 2500) if isinstance(v, str) else v) for k, v in compact["prompt"].items()}
        append_jsonl(self.out / "results.jsonl", compact)
        with (self.out / "transcript.md").open("a", encoding="utf-8") as f:
            f.write(f"\n## {rec['test_id']} (rep {rec['repeat']}) — **{rec['status'].upper()}** score={rec.get('score')} {rec.get('duration_s')}s\n")
            if rec.get("prompt"):
                f.write("\n**Prompt**\n```\n" + trunc(json.dumps(rec["prompt"], ensure_ascii=False) if not isinstance(rec["prompt"].get("user"), str) else rec["prompt"]["user"], 1800) + "\n```\n")
            r = rec.get("response") or {}
            if r.get("reasoning"):
                f.write("\n**Reasoning (exposed by API)**\n```\n" + trunc(r["reasoning"], 2500) + "\n```\n")
            if r.get("text") is not None:
                f.write("\n**Response**\n```\n" + trunc(r.get("text"), 3000) + "\n```\n")
            for c in (rec.get("evaluation") or {}).get("checks", []):
                f.write(f"- {'✅' if c['passed'] else '❌'} {c['name']} {c.get('detail') or ''}\n")
            if rec.get("error"):
                f.write(f"\n**Error**: {rec['error']}\n")
            if rec.get("skip_reason"):
                f.write(f"\n**Skipped**: {rec['skip_reason']}\n")

    def console(self, i, n, rec):
        icon = {"passed": "PASS", "failed": "FAIL", "truncated": "TRNC", "timeout": "TIME", "error": "ERR ", "skipped": "SKIP", "measured": "MEAS", "grading_error": "GERR"}[rec["status"]]
        m = rec.get("metrics") or {}
        extra = f"{m.get('completion_tokens', '-')} tok, {m.get('tokens_per_s', '-')} t/s" if m else ""
        print(f"[{i:>3}/{n}] {icon} {rec['test_id']} r{rec['repeat']}  L{rec['level']}  {rec.get('duration_s')}s  score={rec.get('score')}  {extra} {rec.get('skip_reason') or ''}", flush=True)
        print(f"::group::{rec['test_id']} r{rec['repeat']} details")
        r = rec.get("response") or {}
        if isinstance(rec.get("prompt"), dict):
            print("PROMPT:", trunc(rec["prompt"].get("user") or rec["prompt"].get("procedure"), 800))
        if r.get("reasoning"):
            print("REASONING:", trunc(r["reasoning"], 800))
        if r.get("text") is not None:
            print("RESPONSE:", trunc(r["text"], 800))
        for c in (rec.get("evaluation") or {}).get("checks", [])[:12]:
            print(("  ok  " if c["passed"] else "  FAIL"), c["name"], c.get("detail") or "")
        if rec.get("error"):
            print("ERROR:", rec["error"])
        print("::endgroup::", flush=True)

    # ---------------------------------------------------------------- main
    def run(self):
        from . import report
        a = self.a
        sh, cases = self.plan()
        self.ctx_size = sh["ctx"]
        self.write_metadata(sh)
        self.log(f"shard={sh['shard']} suite={sh['suite']} tests={sh['n_tests']} est={sh['est_minutes']}min ctx={sh['ctx']} budget={a.budget_minutes}min")
        self.monitor = resources.Monitor(self.out / "diagnostics" / "memory.jsonl", interval=a.monitor_interval,
                                         pid_fn=lambda: self.server.pid if self.server else None, path=str(self.out))
        self.monitor.start()
        units = [(c, r) for c in cases for r in range(a.repeat)]
        fatal = None
        try:
            try:
                self.start_server(sh)
            except ServerStartError as e:
                fatal = f"server_start_failed: {e}"
            except Exception as e:
                fatal = f"server_start_error: {type(e).__name__}: {e}"
            if fatal:
                self.log("FATAL " + fatal.splitlines()[0])
                (self.out / "diagnostics" / "fatal.txt").write_text(fatal)
                for i, (c, r) in enumerate(units, 1):
                    rec = self.skip_record(c, r, "server failed to start (see diagnostics/fatal.txt, server.log)", "infrastructure")
                    rec["status"] = "error"
                    rec["error"] = {"type": "server_start", "message": fatal[:500]}
                    self.persist(rec); self.records.append(rec)
            else:
                if sh["suite"] != "performance":  # perf measures the cold start itself
                    t = time.perf_counter()
                    try:
                        self.client.chat([{"role": "user", "content": "Hi"}], max_tokens=4, thinking=False, timeout=600)
                        self.log(f"warm-up request: {time.perf_counter() - t:.1f}s")
                        write_json(self.out / "metadata" / "warmup.json", {"seconds": round(time.perf_counter() - t, 2)})
                    except LLMError as e:
                        self.log(f"warm-up failed: {e}")
                for i, (c, r) in enumerate(units, 1):
                    rec = self.gate(sh, c, r) or self.run_one(c, r)
                    if rec is None:
                        break
                    self.persist(rec); self.records.append(rec); self.console(i, len(units), rec)
                    if rec.get("_abort"):
                        break
        except AbortShard as e:
            fatal = f"aborted: {e}"
            self.log("ABORT " + str(e))
            done = {(r["test_id"], r["repeat"]) for r in self.records}
            for c, r in units:
                if (c["id"], r) not in done:
                    rec = self.skip_record(c, r, f"shard aborted: {e}", "infrastructure")
                    self.persist(rec); self.records.append(rec)
        finally:
            peak = dict(self.monitor.peak, rss_growth_mb=self.monitor.memory_growth_mb())
            self.monitor.stop()
            restarts = self.server.restart_count if self.server else 0
            if self.server:
                self.server.stop()
        summary = report.summarize(self.records)
        summary.update(shard=sh["shard"], suite=sh["suite"], run_id=a.run_id, fatal=fatal, server_restarts=restarts, resource_peaks=peak,
                       wall_seconds=round(time.monotonic() - self.t_start, 1), est_minutes=sh["est_minutes"], finished_at=utcnow(), ctx=sh["ctx"],
                       dataset_version=DATASET_VERSION)
        write_json(self.out / "summary.json", summary)
        md = report.shard_markdown(summary)
        (self.out / "summary.md").write_text(md)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write(md + "\n")
        self.log(f"done: {summary['counts']} accuracy={summary['accuracy']}")
        return 3 if fatal and fatal.startswith("server_start") else 0

    def gate(self, sh, c, r):
        """Return a skip record if this test must not run, else None."""
        if (time.monotonic() - self.t_start) / 60 > self.a.budget_minutes:
            return self.skip_record(c, r, f"shard time budget of {self.a.budget_minutes} min exhausted", "resource")
        if c["suite"] == "vision" and not self.vision_supported:
            return self.skip_record(c, r, "vision unsupported: multimodal projector not loaded / server reports no vision modality")
        if c["suite"] == "long_context" and c["est_prompt_tokens"] + c["max_tokens"] > sh["ctx"]:
            return self.skip_record(c, r, f"needs ~{c['est_prompt_tokens'] + c['max_tokens']} tokens but ctx={sh['ctx']}")
        if "heavy" in c["tags"] or c["suite"] == "long_context":
            ma = resources.snapshot()["mem_available_mb"]
            if ma is not None and ma < self.a.min_mem_available_mb:
                return self.skip_record(c, r, f"MemAvailable {ma} MB < floor {self.a.min_mem_available_mb} MB", "resource")
        return None

    def run_one(self, c, r):
        try:
            return self.execute(c, r)
        except AbortShard:
            raise


def run_cli(a):
    r = Runner(a)
    return r.run()
