"""OpenAI-compatible client for llama-server: request IDs, retry+backoff, error classification,
optional streaming (TTFT), tool-call support. Stdlib only."""
import http.client, json, socket, time, urllib.error, urllib.request, uuid
from .util import utcnow


class LLMError(Exception):
    """kind: timeout | connect | http_4xx | http_5xx | bad_json | bad_shape"""

    def __init__(self, kind, message, status=None, body=None, attempts=None):
        super().__init__(f"{kind}: {message}")
        self.kind, self.status, self.body, self.attempts = kind, status, body, attempts or []


TRANSIENT_STATUS = {502, 503, 504}


class Client:
    def __init__(self, base_url, recover=None):
        self.base = base_url.rstrip("/")          # e.g. http://127.0.0.1:8080/v1
        self.root = self.base[:-3] if self.base.endswith("/v1") else self.base
        self.recover = recover                      # called before a retry after a connection-level failure

    def health(self, timeout=5):
        try:
            with urllib.request.urlopen(self.root + "/health", timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def wait_idle_healthy(self, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.health():
                return True
            time.sleep(2)
        return False

    # ------------------------------------------------------------------
    def build_payload(self, messages, max_tokens, temperature, top_p, seed, thinking, tools, stream, extra):
        p = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if top_p is not None:
            p["top_p"] = top_p
        if seed is not None:
            p["seed"] = seed
        if thinking is not None:
            p["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if tools:
            p["tools"] = tools
            p["tool_choice"] = "auto"
        if stream:
            p["stream"] = True
            p["stream_options"] = {"include_usage": True}
        if extra:
            p.update(extra)
        return p

    def chat(self, messages, *, max_tokens=512, temperature=0.0, top_p=None, seed=None, thinking=None,
             tools=None, stream=False, timeout=600, retries=3, extra=None, request_id=None):
        rid = request_id or uuid.uuid4().hex[:12]
        payload = self.build_payload(messages, max_tokens, temperature, top_p, seed, thinking, tools, stream, extra)
        attempts = []
        last = None
        for n in range(1, retries + 2):
            t_start = time.perf_counter()
            att = {"n": n, "started_at": utcnow(), "request_id": f"{rid}-{n}"}
            try:
                res = self._once(payload, timeout, stream, att["request_id"])
                att.update(ok=True, elapsed_s=round(time.perf_counter() - t_start, 3))
                attempts.append(att)
                res.update(request_id=rid, attempts=attempts, request=payload)
                return res
            except LLMError as e:
                att.update(ok=False, error_kind=e.kind, status=e.status, error=str(e)[:300],
                           elapsed_s=round(time.perf_counter() - t_start, 3))
                attempts.append(att)
                e.attempts = attempts
                last = e
                transient = e.kind == "connect" or (e.kind == "http_5xx" and e.status in TRANSIENT_STATUS)
                if not transient or n > retries:
                    raise
                if self.recover:
                    try:
                        self.recover()
                    except Exception as re_:
                        att["recover_error"] = repr(re_)
                time.sleep(min(2 ** n, 30))  # exponential backoff
        raise last

    # ------------------------------------------------------------------
    def _once(self, payload, timeout, stream, request_id):
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "X-Request-ID": request_id})
        t0 = time.perf_counter()
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:2000]
            raise LLMError("http_4xx" if e.code < 500 else "http_5xx", f"HTTP {e.code}: {body[:200]}", e.code, body)
        except (socket.timeout, TimeoutError):
            raise LLMError("timeout", f"no response within {timeout}s")
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise LLMError("timeout", f"no response within {timeout}s")
            raise LLMError("connect", repr(e.reason))
        except (ConnectionError, http.client.HTTPException, OSError) as e:
            raise LLMError("connect", repr(e))
        try:
            with resp:
                if stream:
                    return self._read_stream(resp, t0, timeout)
                raw = resp.read()
        except (socket.timeout, TimeoutError):
            raise LLMError("timeout", f"response body not complete within {timeout}s")
        except (ConnectionError, http.client.HTTPException, OSError) as e:
            raise LLMError("connect", f"connection lost mid-response: {e!r}")
        elapsed = time.perf_counter() - t0
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError("bad_json", str(e), body=raw[:500].decode("utf-8", "replace"))
        return self._normalize(body, elapsed, None)

    def _read_stream(self, resp, t0, timeout):
        text, reasoning, tool_calls, usage, timings, finish, ttft = [], [], {}, {}, {}, None, None
        chunks = 0
        try:
            for line in resp:
                if time.perf_counter() - t0 > timeout:
                    raise LLMError("timeout", f"stream exceeded {timeout}s")
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ch = json.loads(data)
                except json.JSONDecodeError:
                    continue
                chunks += 1
                if ch.get("usage"):
                    usage = ch["usage"]
                if ch.get("timings"):
                    timings = ch["timings"]
                for c in ch.get("choices", []):
                    d = c.get("delta", {})
                    if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                    if d.get("content"):
                        text.append(d["content"])
                    if d.get("reasoning_content"):
                        reasoning.append(d["reasoning_content"])
                    for tc in d.get("tool_calls") or []:
                        slot = tool_calls.setdefault(tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function", {})
                        slot["function"]["name"] += fn.get("name") or ""
                        slot["function"]["arguments"] += fn.get("arguments") or ""
                    if c.get("finish_reason"):
                        finish = c["finish_reason"]
        except (socket.timeout, TimeoutError):
            raise LLMError("timeout", f"stream stalled > {timeout}s")
        except (ConnectionError, http.client.HTTPException, OSError) as e:
            raise LLMError("connect", f"stream lost: {e!r}")
        elapsed = time.perf_counter() - t0
        body = {"choices": [{"message": {"content": "".join(text), "reasoning_content": "".join(reasoning),
                                         "tool_calls": [tool_calls[k] for k in sorted(tool_calls)] or None},
                             "finish_reason": finish}],
                "usage": usage, "timings": timings, "_stream_chunks": chunks}
        return self._normalize(body, elapsed, ttft)

    @staticmethod
    def _normalize(body, elapsed, ttft):
        try:
            choice = body["choices"][0]
            msg = choice.get("message", {})
        except (KeyError, IndexError, TypeError):
            raise LLMError("bad_shape", "response has no choices[0]", body=json.dumps(body)[:500])
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
        return {"text": content, "reasoning": msg.get("reasoning_content") or "", "tool_calls": msg.get("tool_calls") or [],
                "finish_reason": choice.get("finish_reason"), "usage": body.get("usage") or {},
                "timings": body.get("timings") or {}, "elapsed": elapsed, "ttft": ttft, "raw": body}
