"""Procedural tests (performance + stress). Each proc returns a dict:
   {status: measured|passed|failed|skipped, passed, score, checks, metrics, notes, skip_reason}
Nothing here invents numbers: a metric the server did not report stays None."""
import json, random, threading, time, urllib.error, urllib.request
from . import generators, resources
from .client import Client, LLMError
from .graders import grade
from .util import mean

LEVELS = {"low": 0, "medium": 1, "high": 2}
COUNT_PROMPT = "Count upward starting from 1, writing the numbers separated by single spaces. Never stop counting."


class Ctx:
    def __init__(self, client, server, monitor, level="medium", ctx_size=8192, mem_floor_mb=700, ref_mode=False):
        self.client, self.server, self.monitor, self.level = client, server, monitor, level
        self.ctx_size, self.mem_floor_mb, self.ref_mode = ctx_size, mem_floor_mb, ref_mode


def filler(tokens, seed=5):
    rnd = random.Random(seed)
    s, n = [], 0
    while n < tokens * generators.CHARS_PER_TOKEN:
        x = generators._filler_sentence(rnd)
        s.append(x)
        n += len(x) + 1
    return " ".join(s)


def _meas(res):
    t, u = res["timings"], res["usage"]
    return {"latency_s": round(res["elapsed"], 3), "ttft_s": None if res["ttft"] is None else round(res["ttft"], 3),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "decode_tps": t.get("predicted_per_second"), "prefill_tps": t.get("prompt_per_second"),
            "prompt_ms": t.get("prompt_ms"), "predicted_ms": t.get("predicted_ms"), "cached_tokens": t.get("cache_n"),
            "finish_reason": res["finish_reason"]}


def _agg(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return {"mean": round(sum(v) / len(v), 3), "min": round(min(v), 3), "max": round(max(v), 3), "n": len(v)} if v else None


def _chat(ctx, prompt, max_tokens, **kw):
    kw.setdefault("thinking", False)
    kw.setdefault("timeout", 120 + max_tokens / 0.6 + len(prompt) / 12)
    return ctx.client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.0, seed=1, **kw)


def _skip(reason):
    return {"status": "skipped", "passed": None, "score": None, "checks": [], "metrics": {}, "skip_reason": reason}


def _mem_ok(ctx):
    ma = resources.snapshot()["mem_available_mb"]
    return ma is None or ma >= ctx.mem_floor_mb, ma


def _gate(case, ctx):
    p = case["params"]
    if p.get("min_level") and LEVELS[ctx.level] < LEVELS[p["min_level"]]:
        return _skip(f"requires stress_level>={p['min_level']} (current {ctx.level})")
    if p.get("needs_slots") and (ctx.server is None or ctx.server.parallel < p["needs_slots"]) and not ctx.ref_mode:
        return _skip(f"requires llama-server --parallel>={p['needs_slots']}")
    ok, ma = _mem_ok(ctx)
    if "stress" in case["tags"] and not ok:
        return {**_skip(f"MemAvailable {ma} MB below floor {ctx.mem_floor_mb} MB"), "failure_class": "resource"}
    return None


def _result(checks, metrics, status=None, notes=None):
    passed = all(c["passed"] for c in checks) if checks else None
    return {"status": status or ("passed" if passed else "failed" if passed is False else "measured"), "passed": passed,
            "score": (sum(c["passed"] for c in checks) / len(checks)) if checks else None, "checks": checks, "metrics": metrics, "notes": notes}


# ------------------------------------------------------------------------------------------ PERFORMANCE
def perf_request(case, ctx):
    p = case["params"]
    rows = []
    for i in range(p.get("repeats", 1)):
        if p.get("prompt_tokens"):
            prompt, extra = filler(p["prompt_tokens"], seed=100 + i) + "\n\nReply with the single word OK.", {"cache_prompt": False}
        elif p.get("force_length"):
            prompt, extra = COUNT_PROMPT, {"cache_prompt": False}
        else:
            prompt, extra = p["prompt"], None
        res = _chat(ctx, prompt, p["max_tokens"], stream=p.get("stream", False), extra=extra)
        m = _meas(res)
        m["repeat"] = i
        rows.append(m)
    metrics = {k: _agg(rows, k) for k in ("latency_s", "ttft_s", "decode_tps", "prefill_tps", "prompt_tokens", "completion_tokens")}
    metrics["runs"] = rows
    checks = [{"name": "all_requests_completed", "passed": True, "detail": f"{len(rows)} runs"}]
    if p.get("force_length"):
        full = all((r["completion_tokens"] or 0) >= 0.9 * p["max_tokens"] for r in rows)
        checks.append({"name": "generated_requested_length(>=90%)", "passed": full, "detail": f"completion_tokens={[r['completion_tokens'] for r in rows]}"})
    res_ = _result(checks, metrics, status="measured")
    res_["passed"] = None
    return res_


def perf_cache(case, ctx):
    p = case["params"]
    prompt = filler(p["prompt_tokens"], seed=77) + "\n\nReply with the single word OK."
    a = _meas(_chat(ctx, prompt, p["max_tokens"], stream=True))
    b = _meas(_chat(ctx, prompt, p["max_tokens"], stream=True))
    speed = round(a["latency_s"] / b["latency_s"], 2) if b["latency_s"] else None
    checks = [{"name": "second_request_used_prompt_cache", "passed": bool(b["cached_tokens"]), "detail": f"cached_tokens={b['cached_tokens']}"}]
    r = _result(checks, {"first": a, "second": b, "latency_speedup_x": speed}, status="measured")
    r["passed"] = None
    return r


def _parallel(ctx, n, prompts, max_tokens):
    out, errs = [None] * n, []

    def work(i):
        try:
            out[i] = _meas(_chat(ctx, prompts[i], max_tokens))
        except Exception as e:
            errs.append(repr(e))
    th = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    [t.start() for t in th]
    [t.join() for t in th]
    return out, errs, time.perf_counter() - t0


def perf_concurrency(case, ctx):
    p = case["params"]
    base = _meas(_chat(ctx, COUNT_PROMPT, p["max_tokens"], extra={"cache_prompt": False}))
    n = p["concurrency"]
    rows, errs, wall = _parallel(ctx, n, [COUNT_PROMPT] * n, p["max_tokens"])
    done = [r for r in rows if r]
    toks = sum(r["completion_tokens"] or 0 for r in done)
    agg_tps = round(toks / wall, 3) if wall else None
    single = base["completion_tokens"] / base["latency_s"] if base["latency_s"] and base["completion_tokens"] else None
    metrics = {"single_stream": base, "concurrent": done, "wall_s": round(wall, 2), "aggregate_tps": agg_tps,
               "single_tps": round(single, 3) if single else None,
               "throughput_gain_x": round(agg_tps / single, 2) if agg_tps and single else None}
    r = _result([{"name": "all_concurrent_requests_ok", "passed": not errs and len(done) == n, "detail": "; ".join(errs)[:200]}], metrics)
    r["status"] = "measured" if r["passed"] else "failed"
    return r


# ------------------------------------------------------------------------------------------ STRESS
def _rss(ctx):
    return (resources.snapshot(ctx.server.pid if ctx.server else None).get("server") or {}).get("rss_mb")


def stress_burst(case, ctx):
    p = case["params"]
    n = p["n_by_level"][ctx.level]
    rss0, lat, correct, errs = _rss(ctx), [], 0, []
    for i in range(n):
        try:
            res = _chat(ctx, f"What is {i + 3} + {i + 4}? Reply with only the number.", p["max_tokens"])
            lat.append(res["elapsed"])
            correct += grade({"type": "numeric", "answer": 2 * i + 7}, res["text"])["passed"] is True
        except LLMError as e:
            errs.append(f"{i}: {e.kind}")
    k = max(1, len(lat) // 3)
    drift = round(mean(lat[-k:]) / mean(lat[:k]), 2) if len(lat) >= 3 and mean(lat[:k]) else None
    rss1 = _rss(ctx)
    metrics = {"requests": n, "errors": errs, "latency_s": [round(x, 2) for x in lat], "latency_drift_last_vs_first_third": drift,
               "server_rss_mb_before": rss0, "server_rss_mb_after": rss1, "rss_growth_mb": None if rss0 is None or rss1 is None else round(rss1 - rss0, 1)}
    return _result([{"name": "no_errors", "passed": not errs, "detail": "; ".join(errs)[:200]},
                    {"name": "answers_correct>=90%", "passed": correct >= 0.9 * n, "detail": f"{correct}/{n}"},
                    {"name": "latency_drift<1.5x", "passed": drift is None or drift < 1.5, "detail": f"drift={drift}"}], metrics)


def stress_long_generation(case, ctx):
    target = case["params"]["tokens_by_level"][ctx.level]
    rss0 = _rss(ctx)
    try:
        m = _meas(_chat(ctx, COUNT_PROMPT, target, stream=True, extra={"cache_prompt": False}))
    except LLMError as e:
        return _result([{"name": "completed", "passed": False, "detail": f"{e.kind}: {e}"[:200]}], {"target_tokens": target})
    m.update(target_tokens=target, rss_growth_mb=None if rss0 is None or _rss(ctx) is None else round(_rss(ctx) - rss0, 1))
    return _result([{"name": "completed_without_error", "passed": True, "detail": ""},
                    {"name": "generated>=80%_of_target", "passed": (m["completion_tokens"] or 0) >= 0.8 * target, "detail": f"{m['completion_tokens']}/{target}"}], m)


def stress_mixed(case, ctx):
    p = case["params"]
    rows, errs = [], []
    for i in range(p["rounds_by_level"][ctx.level]):
        for kind, prompt in (("small", "What is 2+2? Reply with only the number."), ("large", filler(p["large_tokens"], seed=30 + i) + "\n\nReply with the single word OK.")):
            try:
                m = _meas(_chat(ctx, prompt, 8, extra={"cache_prompt": False}))
                m["kind"] = kind
                rows.append(m)
            except LLMError as e:
                errs.append(f"{kind}#{i}: {e.kind}")
    return _result([{"name": "no_errors", "passed": not errs, "detail": "; ".join(errs)[:200]}],
                   {"small_latency_s": _agg([r for r in rows if r["kind"] == "small"], "latency_s"),
                    "large_latency_s": _agg([r for r in rows if r["kind"] == "large"], "latency_s"), "runs": rows})


def stress_concurrent(case, ctx):
    p = case["params"]
    n = p["n_by_level"][ctx.level]
    rows, errs, wall = _parallel(ctx, n, [f"What is {i + 10} + 5? Reply with only the number." for i in range(n)], p["max_tokens"])
    done = [r for r in rows if r]
    lat = [r["latency_s"] for r in done]
    return _result([{"name": "all_requests_ok", "passed": not errs and len(done) == n, "detail": "; ".join(errs)[:200]}],
                   {"concurrency": n, "wall_s": round(wall, 2), "latency_spread_s": [min(lat), max(lat)] if lat else None, "runs": done})


def stress_abort(case, ctx):
    p = case["params"]
    t0 = time.perf_counter()
    aborted, finished_early = False, False
    try:
        _chat(ctx, COUNT_PROMPT, 400, timeout=p["abort_after_s"], retries=0)
        finished_early = True  # server so fast the request completed before we could abort it
    except LLMError as e:
        aborted = e.kind == "timeout"
    healthy = ctx.client.wait_idle_healthy(90)
    ans, ok = None, False
    try:
        res = _chat(ctx, "What is 2+2? Reply with only the number.", 16)
        ans = res["text"]
        ok = grade({"type": "numeric", "answer": 4}, ans)["passed"] is True
    except LLMError as e:
        ans = f"{e.kind}"
    return _result([{"name": "client_abort_triggered", "passed": aborted or finished_early, "detail": "request finished before abort (fast server)" if finished_early else ""}, {"name": "server_healthy_after_abort", "passed": healthy, "detail": ""},
                    {"name": "next_request_correct", "passed": ok, "detail": f"answer={ans!r}"}], {"recovery_seconds": round(time.perf_counter() - t0 - p["abort_after_s"], 2)})


def _raw_post(url, body, headers=None):
    req = urllib.request.Request(url, data=body, headers=headers or {"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:
        return f"conn:{type(e).__name__}"


def stress_malformed(case, ctx):
    url = ctx.client.base + "/chat/completions"
    probes = [("invalid_json_body", b"{not json"), ("missing_messages", b'{"max_tokens": 5}'), ("empty_messages", b'{"messages": []}'),
              ("messages_wrong_type", b'{"messages": "hello"}')]
    checks = []
    for name, body in probes:
        st = _raw_post(url, body)
        checks.append({"name": f"{name}->4xx", "passed": isinstance(st, int) and 400 <= st < 500, "detail": f"status={st}"})
    alive = ctx.client.health()
    checks.append({"name": "server_alive_after", "passed": alive, "detail": ""})
    return _result(checks, {})


def stress_overflow(case, ctx):
    big = "word " * int(ctx.ctx_size * 3)
    st = _raw_post(ctx.client.base + "/chat/completions",
                   json.dumps({"messages": [{"role": "user", "content": big}], "max_tokens": 4, "cache_prompt": False}).encode())
    alive = ctx.client.health(timeout=30)
    return _result([{"name": "oversized_prompt_rejected_gracefully(4xx)", "passed": isinstance(st, int) and 400 <= st < 500, "detail": f"status={st}"},
                    {"name": "server_alive_after", "passed": alive, "detail": ""}], {"prompt_chars": len(big), "ctx": ctx.ctx_size})


def stress_kill(case, ctx):
    if ctx.server is None:
        return _skip("server not managed by harness (external --base-url)")
    t0 = time.perf_counter()
    ctx.server.kill9()
    dead = not ctx.server.alive()
    try:
        ready = ctx.server.restart()
    except Exception as e:
        return _result([{"name": "server_killed", "passed": dead, "detail": ""}, {"name": "server_restarted", "passed": False, "detail": str(e)[:200]}], {})
    res = _chat(ctx, "What is 2+2? Reply with only the number.", 16)
    ok = grade({"type": "numeric", "answer": 4}, res["text"])["passed"] is True
    return _result([{"name": "server_killed", "passed": dead, "detail": ""}, {"name": "server_restarted", "passed": True, "detail": f"ready in {ready}s"},
                    {"name": "answers_correctly_after_restart", "passed": ok, "detail": ""}],
                   {"restart_ready_seconds": ready, "total_recovery_seconds": round(time.perf_counter() - t0, 2)})


PROCS = {"perf_request": perf_request, "perf_cache": perf_cache, "perf_concurrency": perf_concurrency, "stress_burst": stress_burst,
         "stress_long_generation": stress_long_generation, "stress_mixed": stress_mixed, "stress_concurrent": stress_concurrent,
         "stress_abort": stress_abort, "stress_malformed": stress_malformed, "stress_overflow": stress_overflow, "stress_kill": stress_kill}


def run_proc(case, ctx):
    gate = _gate(case, ctx)
    if gate:
        return gate
    return PROCS[case["proc"]](case, ctx)
