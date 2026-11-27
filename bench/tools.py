"""Deterministic mock tools + a native tool-calling agent loop (OpenAI `tools` / `tool_calls`)."""
import json, re, tempfile, time
from pathlib import Path
from .graders import safe_eval
from . import sandbox

SCHEMAS = {
    "calculator": {"description": "Evaluate an arithmetic expression and return the numeric result.",
                   "properties": {"expression": {"type": "string", "description": "e.g. '(12.5 * 2) + 7'"}}, "required": ["expression"]},
    "product_lookup": {"description": "Look up a product by name. Returns its unit price (USD) and stock.",
                       "properties": {"name": {"type": "string"}}, "required": ["name"]},
    "search_docs": {"description": "Search the internal knowledge base. Returns matching snippets.",
                    "properties": {"query": {"type": "string"}}, "required": ["query"]},
    "get_weather": {"description": "Get current weather for a city.", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    "read_file": {"description": "Read a text file from the workspace.", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    "write_file": {"description": "Write text to a file in the workspace (overwrites).",
                   "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    "task_tracker": {"description": "Manage a to-do list. action: add | complete | list.",
                     "properties": {"action": {"type": "string", "enum": ["add", "complete", "list"]}, "title": {"type": "string"}},
                     "required": ["action"]},
    "get_exchange_rate": {"description": "Get the current exchange rate between two currencies.",
                          "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
                          "required": ["from_currency", "to_currency"]},
    "fetch_page": {"description": "Fetch the text of a web page.", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    "send_email": {"description": "Send an email.", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"]},
    "run_python": {"description": "Run a short Python snippet and return what it prints.",
                   "properties": {"code": {"type": "string"}}, "required": ["code"]},
    "create_event": {"description": "Create a calendar event.",
                     "properties": {"title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
                                    "time": {"type": "string", "description": "HH:MM 24h"}}, "required": ["title", "date", "time"]},
}


def schema_for(name):
    s = SCHEMAS[name]
    return {"type": "function", "function": {"name": name, "description": s["description"],
                                             "parameters": {"type": "object", "properties": s["properties"], "required": s["required"]}}}


class ToolEnv:
    """Per-scenario tool state. `fixtures` come from the test JSON so everything is deterministic."""

    def __init__(self, tool_names, fixtures=None):
        self.names = list(tool_names)
        self.fx = fixtures or {}
        self.dir = Path(tempfile.mkdtemp(prefix="bonsai_ws_"))
        for p, c in self.fx.get("files", {}).items():
            (self.dir / p).parent.mkdir(parents=True, exist_ok=True)
            (self.dir / p).write_text(c)
        self.tasks, self.emails, self.events = [], [], []
        self.rate_calls = 0

    def schemas(self):
        return [schema_for(n) for n in self.names]

    def _safe_path(self, p):
        full = (self.dir / str(p)).resolve()
        if not str(full).startswith(str(self.dir.resolve())):
            raise PermissionError("path escapes workspace")
        return full

    def call(self, name, args):
        """-> (result_string, is_error)"""
        if name not in self.names:
            return json.dumps({"error": f"unknown tool '{name}'"}), True
        try:
            return json.dumps(getattr(self, "_t_" + name)(**args)), False
        except TypeError as e:
            return json.dumps({"error": f"bad arguments: {e}"}), True
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"}), True

    # --- tools -----------------------------------------------------------------
    def _t_calculator(self, expression):
        v = safe_eval(expression)
        return {"result": round(v, 10) if isinstance(v, float) else v}

    def _t_product_lookup(self, name):
        for k, v in self.fx.get("products", {}).items():
            if k.lower() == str(name).strip().lower():
                return {"name": k, **v}
        return {"error": f"no product named '{name}'", "known_products": list(self.fx.get("products", {}))}

    def _t_search_docs(self, query):
        words = set(re.findall(r"\w+", str(query).lower()))
        hits = [d for d in self.fx.get("docs", []) if words & set(re.findall(r"\w+", d.lower()))]
        return {"results": hits[:3]}

    def _t_get_weather(self, city):
        w = self.fx.get("weather", {}).get(str(city).strip().lower())
        return w or {"error": f"no data for {city}"}

    def _t_read_file(self, path):
        f = self._safe_path(path)
        if not f.exists():
            return {"error": f"file not found: {path}", "files": sorted(p.name for p in self.dir.iterdir())}
        return {"content": f.read_text()}

    def _t_write_file(self, path, content):
        f = self._safe_path(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(str(content))
        return {"ok": True, "bytes": len(str(content))}

    def _t_task_tracker(self, action, title=None):
        if action == "add":
            self.tasks.append({"title": str(title), "done": False})
            return {"ok": True, "count": len(self.tasks)}
        if action == "complete":
            for t in self.tasks:
                if t["title"].lower() == str(title).lower():
                    t["done"] = True
                    return {"ok": True}
            return {"error": f"no task '{title}'", "tasks": [t["title"] for t in self.tasks]}
        if action == "list":
            return {"tasks": self.tasks}
        return {"error": "action must be add|complete|list"}

    def _t_get_exchange_rate(self, from_currency, to_currency):
        self.rate_calls += 1
        if self.rate_calls == 1:  # deterministic transient failure -> tests error recovery
            return {"error": "503 Service Temporarily Unavailable. Please retry."}
        return {"from": from_currency.upper(), "to": to_currency.upper(), "rate": self.fx.get("rate", 0.92)}

    def _t_fetch_page(self, url):
        p = self.fx.get("pages", {}).get(str(url))
        return {"url": url, "text": p} if p else {"error": f"404 for {url}", "known_urls": list(self.fx.get("pages", {}))}

    def _t_send_email(self, to, body):
        self.emails.append({"to": to, "body": body})
        return {"ok": True, "message": "email queued"}

    def _t_run_python(self, code):
        r = sandbox.run_sandboxed({"mode": "snippet", "code": code}, timeout=10)
        out = next((e["text"] for e in r["events"] if e["phase"] == "stdout"), "")
        load = next((e for e in r["events"] if e["phase"] == "load"), {})
        return {"stdout": out, "error": None if load.get("ok", False) else load.get("error", "timeout")}

    def _t_create_event(self, title, date, time):  # noqa: shadowing is fine here
        self.events.append({"title": title, "date": date, "time": time})
        return {"ok": True, "event_id": len(self.events)}

    def state(self):
        files = {str(p.relative_to(self.dir)): p.read_text() for p in self.dir.rglob("*") if p.is_file()}
        return {"tasks": self.tasks, "emails": self.emails, "events": self.events, "files": files}


def run_agent(client, case, params, env, max_steps=8, on_step=None):
    """Multi-turn loop. Returns dict(final_text, trace, steps, incomplete, format_violations, totals)."""
    messages = []
    if case.get("system"):
        messages.append({"role": "system", "content": case["system"]})
    messages.append({"role": "user", "content": case["prompt"]})
    trace, steps, violations = [], [], []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "elapsed": 0.0}
    final_text, incomplete = "", True
    for step in range(1, max_steps + 1):
        res = client.chat(messages, tools=env.schemas(), **params)
        steps.append({"step": step, "request_id": res["request_id"], "finish_reason": res["finish_reason"], "usage": res["usage"],
                      "timings": res["timings"], "elapsed_s": round(res["elapsed"], 2), "text": res["text"], "reasoning": res["reasoning"],
                      "tool_calls": res["tool_calls"], "attempts": res["attempts"]})
        totals["prompt_tokens"] += res["usage"].get("prompt_tokens", 0)
        totals["completion_tokens"] += res["usage"].get("completion_tokens", 0)
        totals["elapsed"] += res["elapsed"]
        if on_step:
            on_step(steps[-1])
        calls = res["tool_calls"]
        if not calls:
            final_text, incomplete = res["text"], False
            # tool call written as plain text instead of structured call? record as a format violation
            if re.search(r"<tool_call>|\"(?:tool|name)\"\s*:\s*\"(?:%s)\"" % "|".join(env.names), res["text"] or ""):
                violations.append(res["text"][:200])
            break
        messages.append({"role": "assistant", "content": res["text"] or "", "tool_calls": calls})
        for tc in calls:
            fn = tc.get("function", {})
            name, raw = fn.get("name", ""), fn.get("arguments", "")
            entry = {"kind": "tool_call", "step": step, "tool": name, "raw_args": raw, "known_tool": name in env.names, "valid_json": True}
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except Exception as e:
                args, entry["valid_json"], entry["format_error"] = {}, False, f"invalid JSON arguments: {e}"
            entry["args"] = args
            if entry["valid_json"]:
                result, is_err = env.call(name, args)
            else:
                result, is_err = json.dumps({"error": entry["format_error"]}), True
            if not entry["known_tool"]:
                entry["format_error"] = f"unknown tool {name}"
            entry.update(result=result[:500], is_error=is_err)
            trace.append(entry)
            messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call_{len(trace)}", "name": name, "content": result})
    return {"final_text": final_text, "trace": trace, "steps": steps, "incomplete": incomplete,
            "format_violations": violations, "totals": totals, "state": env.state()}
