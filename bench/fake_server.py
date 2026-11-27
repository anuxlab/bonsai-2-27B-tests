"""Tiny OpenAI-compatible stand-in for llama-server used to test the HARNESS (not the model).
It replays each test's `reference` answer, so a green end-to-end run proves the plumbing, graders,
sandbox, tools, reporting and sharding all work before you spend 6 hours of runner time."""
import json, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from . import cases as C


def _answer_from_grade(g):
    g = g[0] if isinstance(g, list) else g
    if g.get("type") == "numeric":
        return str(g.get("answer", g.get("answers", [""])[0]))
    if g.get("type") in ("exact", "choice"):
        return str((g.get("answers") or [g.get("answer")])[0])
    return "OK"


def build_index():
    idx = {}
    for s in C.SUITES:
        for c in C.load_suite(s, ("1k", "2k", "4k", "8k")):
            if c["kind"] != "proc":
                idx[c["prompt"]] = c
    return idx


class Handler(BaseHTTPRequestHandler):
    index, ctx = {}, 8192
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send(200, {"status": "ok"})
        if self.path.startswith("/props"):
            return self._send(200, {"modalities": {"vision": True}})
        self._send(404, {"error": "nf"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            req = json.loads(raw)
            msgs = req["messages"]
            if not isinstance(msgs, list) or not msgs:
                raise ValueError("messages must be a non-empty list")
        except Exception as e:
            return self._send(400, {"error": {"message": f"bad request: {e}"}})
        user = next((m for m in reversed(msgs) if m["role"] == "user"), msgs[-1])
        text = user["content"] if isinstance(user["content"], str) else " ".join(p.get("text", "") for p in user["content"] if p.get("type") == "text")
        if len(text) / 3.8 > self.ctx:
            return self._send(400, {"error": {"message": "the request exceeds the available context size"}})
        time.sleep(0.01)
        mt = req.get("max_tokens", 64)
        finish, tool_calls = "stop", None
        case = self.index.get(text)
        if case and case["kind"] == "agent":
            calls = case["reference"]["calls"]
            done = sum(1 for m in msgs if m["role"] == "tool")
            if done < len(calls):
                c = calls[done]
                tool_calls = [{"id": f"call_{done}", "type": "function", "function": {"name": c["tool"], "arguments": json.dumps(c["args"])}}]
                out = ""
                finish = "tool_calls"
            else:
                out = case["reference"]["final"]
        elif case:
            out = case.get("reference") or _answer_from_grade(case["grade"])
        elif text.startswith("Count upward"):
            out, finish = " ".join(map(str, range(1, mt + 1))), "length"
        elif (m := re.match(r"What is (\d+) \+ (\d+)\?", text)):
            out = str(int(m.group(1)) + int(m.group(2)))
        elif text.startswith("What is 2+2"):
            out = "4"
        elif "Reply with the single word OK" in text:
            out = "OK"
        else:
            out = "Hello!"
        ntok = mt if finish == "length" else max(1, len(out) // 4)
        body = {"choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": out, "reasoning_content": "fake reasoning", "tool_calls": tool_calls}}],
                "usage": {"prompt_tokens": int(len(text) / 3.8) + 5, "completion_tokens": ntok},
                "timings": {"prompt_n": 5, "cache_n": 0 if text not in getattr(self.server, "seen", set()) else 4, "prompt_ms": 10, "prompt_per_second": 500.0,
                            "predicted_n": ntok, "predicted_ms": 10 * ntok, "predicted_per_second": 100.0}}
        self.server.seen = getattr(self.server, "seen", set()) | {text}
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            ch = lambda d, f=None, **x: ("data: " + json.dumps({"choices": [{"delta": d, "finish_reason": f}], **x}) + "\n\n").encode()
            self.wfile.write(ch({"content": out}))
            self.wfile.write(ch({}, finish, usage=body["usage"], timings=body["timings"]))
            self.wfile.write(b"data: [DONE]\n\n")
            self.close_connection = True
            return
        self._send(200, body)


def serve(port=8099, ctx=8192):
    Handler.index, Handler.ctx = build_index(), ctx
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv
