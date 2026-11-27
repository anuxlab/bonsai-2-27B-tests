"""Run model-generated Python in an isolated child process.

Layers (best effort; this is NOT a hardened security boundary, so the eval jobs are also run with a
read-only token and no secrets):
  1. separate process, `python -I`, scrubbed environment (no GITHUB_TOKEN etc.), temp cwd
  2. rlimits: CPU, address space, file size, open files
  3. wall-clock timeout, whole process group killed
  4. network namespace via `unshare -rn` when the kernel allows it, otherwise in-process socket/subprocess/os guards
"""
import json, os, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path

HARNESS = r'''
import sys, os, json, signal, resource, builtins, io, contextlib, math, traceback
spec = json.load(open(sys.argv[1]))
MEM = spec.get("mem_mb", 1024) * 1024 * 1024
for lim, val in ((resource.RLIMIT_AS, MEM), (resource.RLIMIT_CPU, int(spec["timeout"]) + 2),
                 (resource.RLIMIT_FSIZE, 10 * 1024 * 1024), (resource.RLIMIT_NOFILE, 64)):
    try: resource.setrlimit(lim, (val, val))
    except Exception: pass
OUT = sys.__stdout__
def emit(o): OUT.write("\x1e" + json.dumps(o, default=repr) + "\n"); OUT.flush()
import socket, subprocess
def _deny(*a, **k): raise PermissionError("blocked by sandbox")
class _NoSock:
    def __init__(self, *a, **k): raise PermissionError("network blocked by sandbox")
socket.socket = _NoSock; socket.create_connection = _deny; subprocess.Popen = _deny
for n in ("system","popen","fork","forkpty","execv","execve","execvp","execl","execle","execlp","spawnl","spawnv","spawnlp","spawnvp","kill","killpg","setsid"):
    if hasattr(os, n): setattr(os, n, _deny)
_real_open = builtins.open
_root = os.path.realpath(os.getcwd())
def _open(file, *a, **k):
    if isinstance(file, (str, bytes, os.PathLike)):
        p = os.path.realpath(os.fspath(file))
        if isinstance(p, bytes): p = p.decode()
        if not (p == _root or p.startswith(_root + os.sep)):
            raise PermissionError("file access outside sandbox dir blocked")
    return _real_open(file, *a, **k)
builtins.open = _open
sys.stdin = io.StringIO("")
def canon(x):
    if isinstance(x, (tuple, list)): return [canon(i) for i in x]
    if isinstance(x, (set, frozenset)):
        try: return sorted(canon(i) for i in x)
        except Exception: return [canon(i) for i in x]
    if isinstance(x, dict): return {str(k): canon(v) for k, v in x.items()}
    return x
def eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool): return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)): return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(a, list) and isinstance(b, list): return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict): return a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
    return a == b
class _Alarm(Exception): pass
def _on_alarm(*a): raise _Alarm("per-test timeout")
signal.signal(signal.SIGALRM, _on_alarm)
def guarded(fn, secs):
    signal.setitimer(signal.ITIMER_REAL, secs)
    try: return fn()
    finally: signal.setitimer(signal.ITIMER_REAL, 0)
buf = io.StringIO()
mode = spec["mode"]
ns = {"__name__": "sandbox_module"}
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        guarded(lambda: exec(compile(spec["code"], "<model_code>", "exec"), ns), spec.get("load_timeout", 5))
except BaseException as e:
    emit({"phase": "load", "ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
    emit({"phase": "done"}); sys.exit(0)
emit({"phase": "load", "ok": True})
if mode == "tests":
    for i, (expr, expected) in enumerate(spec["tests"]):
        r = {"phase": "test", "i": i, "expr": expr[:200]}
        t0 = __import__("time").perf_counter()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                val = guarded(lambda: eval(expr, ns), spec.get("per_test_timeout", 5))
            if isinstance(expected, dict) and "raises" in expected:
                r.update(ok=False, error="expected %s but nothing was raised" % expected["raises"])
            else:
                got = canon(val)
                r.update(ok=eq(got, expected), got=repr(got)[:200])
        except BaseException as e:
            names = [c.__name__ for c in type(e).__mro__]
            if isinstance(expected, dict) and "raises" in expected and expected["raises"] in names:
                r.update(ok=True, got="raised " + type(e).__name__)
            else:
                r.update(ok=False, error=f"{type(e).__name__}: {e}"[:200])
        r["secs"] = round(__import__("time").perf_counter() - t0, 4)
        emit(r)
elif mode == "unit":
    tests = [(k, v) for k, v in ns.items() if k.startswith("test_") and callable(v)]
    emit({"phase": "discovered", "n": len(tests)})
    for k, fn in tests:
        r = {"phase": "test", "name": k}
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                guarded(fn, spec.get("per_test_timeout", 5))
            r["ok"] = True
        except BaseException as e:
            r.update(ok=False, error=f"{type(e).__name__}: {e}"[:200])
        emit(r)
elif mode == "snippet":
    pass  # code already ran at load time; its stdout is in buf
emit({"phase": "stdout", "text": buf.getvalue()[-4000:]})
emit({"phase": "done"})
'''

_ISOLATION = None


def isolation_mode():
    """'netns' if `unshare -rn` works here, else 'python-guard'. Probed once."""
    global _ISOLATION
    if _ISOLATION is None:
        _ISOLATION = "python-guard"
        if shutil.which("unshare"):
            try:
                r = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=5)
                if r.returncode == 0:
                    _ISOLATION = "netns"
            except Exception:
                pass
    return _ISOLATION


def run_sandboxed(spec, timeout=20, workdir=None):
    """spec: {mode: tests|unit|snippet, code, tests?, per_test_timeout?, mem_mb?}.
    Returns {events:[...], timed_out, returncode, duration_s, isolation, stderr}."""
    spec = dict(spec, timeout=timeout)
    tmp = Path(workdir or tempfile.mkdtemp(prefix="bonsai_sbx_"))
    own_tmp = workdir is None
    try:
        (tmp / "harness.py").write_text(HARNESS)
        (tmp / "spec.json").write_text(json.dumps(spec))
        cmd = [sys.executable, "-I", "harness.py", "spec.json"]
        iso = isolation_mode()
        if iso == "netns":
            cmd = ["unshare", "-rn", "--"] + cmd
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "HOME": str(tmp)}
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, cwd=tmp, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdin=subprocess.DEVNULL, start_new_session=True, text=True, errors="replace")
        timed_out = False
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
            out, err = p.communicate()
        events = []
        for line in (out or "").split("\n"):  # NOT splitlines(): it treats \x1e as a line break
            if line.startswith("\x1e"):
                try:
                    events.append(json.loads(line[1:]))
                except json.JSONDecodeError:
                    pass
        return {"events": events, "timed_out": timed_out, "returncode": p.returncode,
                "duration_s": round(time.perf_counter() - t0, 3), "isolation": iso, "stderr": (err or "")[-1500:]}
    finally:
        if own_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
