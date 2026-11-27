"""Resource diagnostics: preflight checks, point-in-time snapshots, background monitor."""
import os, platform, shutil, subprocess, threading, time
from pathlib import Path
from .util import utcnow, append_jsonl, write_json

_prev_cpu = [None]


def read_meminfo():
    d = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            d[k] = int(v.strip().split()[0])  # kB
    except Exception:
        pass
    return d


def _read_int(p):
    try:
        s = Path(p).read_text().strip()
        return None if s == "max" else int(s)
    except Exception:
        return None


def cgroup_memory():
    """cgroup v2 first, then v1. Values in MB (None if unlimited/unknown)."""
    cur = _read_int("/sys/fs/cgroup/memory.current")
    lim = _read_int("/sys/fs/cgroup/memory.max")
    if cur is None:
        cur = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        lim = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if lim and lim > 1 << 60:
            lim = None
    f = lambda x: None if x is None else round(x / 1048576, 1)
    return {"current_mb": f(cur), "limit_mb": f(lim)}


def psi_memory():
    """Memory-pressure stall info (avg10 = % of last 10 s that tasks were stalled)."""
    out = {}
    try:
        for line in Path("/proc/pressure/memory").read_text().splitlines():
            kind, rest = line.split(None, 1)
            kv = dict(x.split("=") for x in rest.split())
            out[kind + "_avg10"] = float(kv.get("avg10", 0))
    except Exception:
        pass
    return out


def proc_mem(pid):
    if not pid:
        return {}
    try:
        d = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS", "VmSwap", "VmHWM")):
                k, v = line.split(":", 1)
                d[k] = round(int(v.split()[0]) / 1024, 1)
        return {"rss_mb": d.get("VmRSS"), "swap_mb": d.get("VmSwap"), "hwm_mb": d.get("VmHWM")}
    except Exception:
        return {}


def cpu_pct():
    try:
        parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        vals = list(map(int, parts))
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        prev = _prev_cpu[0]
        _prev_cpu[0] = (idle, total)
        if prev is None or total == prev[1]:
            return None
        return round(100.0 * (1 - (idle - prev[0]) / (total - prev[1])), 1)
    except Exception:
        return None


def snapshot(pid=None, path="."):
    m = read_meminfo()
    mb = lambda k: round(m[k] / 1024, 1) if k in m else None
    du = shutil.disk_usage(path)
    return {
        "t": utcnow(),
        "mem_total_mb": mb("MemTotal"), "mem_available_mb": mb("MemAvailable"),
        "mem_used_mb": round((m.get("MemTotal", 0) - m.get("MemAvailable", 0)) / 1024, 1) if m else None,
        "swap_total_mb": mb("SwapTotal"),
        "swap_used_mb": round((m.get("SwapTotal", 0) - m.get("SwapFree", 0)) / 1024, 1) if m else None,
        "disk_free_gb": round(du.free / 1e9, 2),
        "load1": os.getloadavg()[0], "cpu_pct": cpu_pct(),
        "cgroup": cgroup_memory(), "psi": psi_memory(), "server": proc_mem(pid),
    }


class Monitor(threading.Thread):
    """Samples resources every `interval` s into a JSONL file and tracks peaks."""

    def __init__(self, out_file, interval=5.0, pid_fn=None, path="."):
        super().__init__(daemon=True)
        self.out_file, self.interval, self.pid_fn, self.path = Path(out_file), interval, pid_fn, path
        self._halt = threading.Event()
        self.peak = {"server_rss_mb": 0, "min_mem_available_mb": None, "peak_swap_used_mb": 0,
                     "peak_cpu_pct": 0, "peak_psi_some_avg10": 0, "samples": 0}
        self.history = []  # server RSS (MB) per sample

    def run(self):
        while not self._halt.is_set():
            try:
                s = snapshot(self.pid_fn() if self.pid_fn else None, self.path)
                append_jsonl(self.out_file, s)
                p = self.peak
                p["samples"] += 1
                rss = (s["server"] or {}).get("rss_mb") or 0
                p["server_rss_mb"] = max(p["server_rss_mb"], rss)
                ma = s["mem_available_mb"]
                if ma is not None:
                    p["min_mem_available_mb"] = ma if p["min_mem_available_mb"] is None else min(p["min_mem_available_mb"], ma)
                p["peak_swap_used_mb"] = max(p["peak_swap_used_mb"], s["swap_used_mb"] or 0)
                p["peak_cpu_pct"] = max(p["peak_cpu_pct"], s["cpu_pct"] or 0)
                p["peak_psi_some_avg10"] = max(p["peak_psi_some_avg10"], s["psi"].get("some_avg10", 0))
                if rss:
                    self.history.append(rss)
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()

    def memory_growth_mb(self):
        """Mean server RSS of last quartile minus first quartile (None if too few samples)."""
        h = self.history
        if len(h) < 8:
            return None
        q = max(1, len(h) // 4)
        return round(sum(h[-q:]) / q - sum(h[:q]) / q, 1)


def environment_info():
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    osr = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                osr[k] = v.strip('"')
    except Exception:
        pass
    # allow-list only: never dump the whole environment (it can contain secrets)
    keep = ["GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_WORKFLOW", "GITHUB_SHA", "GITHUB_REF",
            "GITHUB_REPOSITORY", "RUNNER_OS", "RUNNER_ARCH", "RUNNER_NAME", "ImageOS", "ImageVersion"]
    return {
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
        "os": osr.get("PRETTY_NAME"), "cpu_model": cpu_model, "cpu_count": os.cpu_count(),
        "env": {k: os.environ[k] for k in keep if k in os.environ},
        "memory_at_start": snapshot(),
    }


def _sh(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"<{cmd!r} failed: {e}>"


def preflight(out_dir, policy="strict", require_disk_gb=25.0, min_ram_gb=12.0):
    """Run BEFORE downloading anything. Returns (ok, report). Disk shortage always aborts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment_info()
    m = env["memory_at_start"]
    findings, abort = [], False

    def add(level, msg):
        findings.append({"level": level, "msg": msg})

    if m["disk_free_gb"] < require_disk_gb:
        add("ABORT", f"disk free {m['disk_free_gb']} GB < required {require_disk_gb} GB (aborting before model download)")
        abort = True
    else:
        add("OK", f"disk free {m['disk_free_gb']} GB >= {require_disk_gb} GB")

    ram_gb = (m["mem_total_mb"] or 0) / 1024
    if ram_gb < min_ram_gb:
        add("ABORT" if policy == "strict" else "WARN", f"RAM {ram_gb:.1f} GB < recommended {min_ram_gb} GB")
        abort = abort or policy == "strict"
    else:
        add("OK", f"RAM {ram_gb:.1f} GB (available {(m['mem_available_mb'] or 0) / 1024:.1f} GB)")

    if not m["swap_total_mb"]:
        # no swap is acceptable on a big-RAM runner; only strict + small RAM aborts
        add("WARN", "no swap configured")
    else:
        add("OK", f"swap {m['swap_total_mb']} MB (used {m['swap_used_mb']} MB)")

    cg = m["cgroup"]
    if cg.get("limit_mb") and cg["limit_mb"] < min_ram_gb * 1024:
        add("WARN", f"cgroup memory limit {cg['limit_mb']} MB is below {min_ram_gb} GB")
    if (m["psi"].get("some_avg10") or 0) > 10:
        add("WARN", f"memory pressure already high at start: {m['psi']}")

    extra = {"free -h": _sh("free -h"), "df -h": _sh("df -h"), "lscpu": _sh("lscpu | head -20"),
             "swapon --show": _sh("swapon --show"), "ulimit -a": _sh("bash -c 'ulimit -a'")}
    report = {"ok": not abort, "policy": policy, "findings": findings, "environment": env}
    write_json(out_dir / "environment.json", env)
    lines = [f"PREFLIGHT {utcnow()}  policy={policy}  result={'PASS' if not abort else 'ABORT'}", ""]
    lines += [f"[{f['level']}] {f['msg']}" for f in findings]
    for k, v in extra.items():
        lines += ["", f"$ {k}", v]
    (out_dir / "preflight.txt").write_text("\n".join(lines) + "\n")
    return (not abort), report
