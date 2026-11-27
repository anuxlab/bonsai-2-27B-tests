"""llama-server supervisor: start, readiness, liveness, restart, clean stop, OOM evidence."""
import json, os, signal, subprocess, time, urllib.request
from pathlib import Path
from .util import utcnow


class ServerStartError(RuntimeError):
    pass


class ServerManager:
    def __init__(self, binary, model, mmproj=None, ctx=8192, threads=4, parallel=1, port=8080,
                 log_path="server.log", extra_args=None):
        self.binary, self.model, self.mmproj = str(binary), str(model), (str(mmproj) if mmproj else None)
        self.ctx, self.threads, self.parallel, self.port = ctx, threads, parallel, port
        self.log_path, self.extra_args = Path(log_path), list(extra_args or [])
        self.proc = None
        self.restart_count = 0
        self.startups = []  # [{started_at, ready_seconds}]
        self.base = f"http://127.0.0.1:{port}"

    def command(self):
        cmd = [self.binary, "-m", self.model, "-ngl", "0", "-fa", "on", "-c", str(self.ctx), "-t", str(self.threads),
               "-np", str(self.parallel), "--jinja", "--host", "127.0.0.1", "--port", str(self.port)]
        if self.mmproj:
            cmd += ["--mmproj", self.mmproj]
        return cmd + self.extra_args

    @property
    def pid(self):
        return self.proc.pid if self.proc and self.proc.poll() is None else None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def healthy(self, timeout=5):
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def _log_tail(self, n=60):
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except Exception:
            return ""

    def start(self, ready_timeout=900):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(self.log_path, "ab")
        logf.write(f"\n===== [{utcnow()}] START: {' '.join(self.command())}\n".encode())
        logf.flush()
        t0 = time.monotonic()
        started_at = utcnow()
        self.proc = subprocess.Popen(self.command(), stdout=logf, stderr=subprocess.STDOUT,
                                     start_new_session=True, env={**os.environ, "LLAMA_LOG_COLORS": "0"})
        while time.monotonic() - t0 < ready_timeout:
            if self.proc.poll() is not None:
                raise ServerStartError(f"llama-server exited with code {self.proc.returncode} during startup.\n{self._log_tail()}")
            if self.healthy(timeout=3):
                dt = round(time.monotonic() - t0, 2)
                self.startups.append({"started_at": started_at, "ready_seconds": dt})
                return dt
            time.sleep(2)
        self.stop()
        raise ServerStartError(f"llama-server not ready after {ready_timeout}s.\n{self._log_tail()}")

    def stop(self, grace=15):
        if not self.proc:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=grace)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass

    def kill9(self):
        """Used by the restart-recovery stress test."""
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=30)

    def restart(self, ready_timeout=900):
        self.restart_count += 1
        self.stop(grace=5)
        return self.start(ready_timeout)

    def exit_info(self):
        """Why did the server die? Negative return code = killed by signal (-9 is the OOM killer's signature)."""
        rc = self.proc.poll() if self.proc else None
        info = {"returncode": rc, "signal": (-rc if rc is not None and rc < 0 else None)}
        info["probable_oom"] = info["signal"] == 9
        return info

    def props(self):
        try:
            with urllib.request.urlopen(self.base + "/props", timeout=10) as r:
                return json.loads(r.read())
        except Exception:
            return {}

    def supports_vision(self):
        if not self.mmproj:
            return False
        p = self.props()
        mod = p.get("modalities") or {}
        if "vision" in mod:
            return bool(mod["vision"])
        try:  # older builds: fall back to the startup log
            return "loaded multimodal model" in self.log_path.read_text(errors="replace")
        except Exception:
            return False
