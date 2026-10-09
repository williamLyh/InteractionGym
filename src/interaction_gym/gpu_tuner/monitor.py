"""Background sampling while episodes run: GPU utilisation + memory (``nvidia-smi``, local or over
ssh) and vLLM's Prometheus ``/metrics`` (requests running / waiting, KV-cache usage) per service.

    mon = Monitor(gpu=GpuSampler(ssh="gpu-host"), metrics={"llm": ["http://127.0.0.1:8000"]})
    async with mon.running():
        ... episodes ...
    mon.samples          # [Sample]
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from statistics import mean

QUERY = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]

# vLLM v1 names (older releases: gpu_cache_usage_perc); vLLM-Omni servers expose the same when they expose any
RUNNING = ("vllm:num_requests_running",)
WAITING = ("vllm:num_requests_waiting",)
KV = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")


@dataclass
class Sample:
    t: float
    gpu: dict[int, tuple] = field(default_factory=dict)  # id -> (util 0..1, memory used MB[, memory total MB])
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)  # service -> {running, waiting, kv}


class GpuSampler:
    def __init__(self, ssh: str | None = None):
        self.ssh = ssh

    def __call__(self) -> dict[int, tuple[float, float, float]]:
        cmd = (["ssh", "-o", "ConnectTimeout=5", self.ssh, " ".join(QUERY)] if self.ssh else QUERY)
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
        if not out.strip():
            raise RuntimeError("nvidia-smi returned nothing")
        res = {}
        for line in out.strip().splitlines():
            i, u, m, t = (x.strip() for x in line.split(","))
            res[int(i)] = (float(u) / 100, float(m), float(t))
        return res


def parse_prometheus(text: str) -> dict[str, float]:
    """Each metric summed over its label sets (several engines / replicas behind one server) — except
    pipeline stages (vLLM-Omni labels them ``stage``): a request sits in several stages at once, so
    the busiest stage counts, not the sum; and queue depth (``*waiting*``) is the entry stage's, since
    a streaming request shows as "waiting" in later stages while the first one feeds them."""
    per: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+0-9.eEnaNInf]+)", line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        st = re.search(r'stage="([^"]*)"', m.group(2) or "")
        d = per.setdefault(m.group(1), {})
        k = st.group(1) if st else ""
        d[k] = d.get(k, 0.0) + v
    out = {}
    for name, d in per.items():
        if "waiting" in name and len(d) > 1:  # a queue forms in front of the entry stage; later stages "wait" for chunks
            out[name] = d[min(d, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0))]
        else:
            out[name] = max(d.values())
    return out


def scrape(base: str, timeout: float = 3.0) -> dict[str, float] | None:
    url = re.sub(r"^ws", "http", base.split("/v1")[0].rstrip("/")) + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            m = parse_prometheus(r.read().decode())
    except Exception:  # noqa: BLE001 — a server without /metrics is just not sampled
        return None
    pick = lambda names: next((m[n] for n in names if n in m), None)  # noqa: E731
    vals = {"running": pick(RUNNING), "waiting": pick(WAITING), "kv": pick(KV)}
    return {k: v for k, v in vals.items() if v is not None} or None


class Monitor:
    """Samples every ``interval`` seconds. ``gpu`` and each metrics source are callables / URLs; a
    mock passes ``sample_fn`` instead (returns a ``Sample``)."""

    def __init__(self, gpu=None, metrics: dict[str, list[str]] | None = None, interval: float = 2.0, sample_fn=None, clock=time.perf_counter):
        self.gpu, self.metrics, self.interval, self.sample_fn, self.clock = gpu, metrics or {}, interval, sample_fn, clock
        self.samples: list[Sample] = []
        self.errors = 0

    def sample(self) -> Sample:
        if self.sample_fn is not None:
            return self.sample_fn()
        s = Sample(self.clock())
        if self.gpu is not None:
            try:
                s.gpu = self.gpu()
            except Exception:  # noqa: BLE001
                self.errors += 1
        for svc, urls in self.metrics.items():
            got = [x for x in (scrape(u) for u in urls) if x]
            if got:
                s.metrics[svc] = {k: sum(g.get(k, 0.0) for g in got) if k != "kv" else mean(g["kv"] for g in got if "kv" in g)
                                  for k in {k for g in got for k in g}}
        return s

    async def _loop(self, stop: asyncio.Event):
        while not stop.is_set():
            self.samples.append(await asyncio.to_thread(self.sample) if self.sample_fn is None else self.sample())
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), self.interval)

    @contextlib.asynccontextmanager
    async def running(self):
        stop = asyncio.Event()
        task = asyncio.create_task(self._loop(stop))
        try:
            yield self
        finally:
            stop.set()
            await task

    def window(self, t0: float, t1: float) -> list[Sample]:
        return [s for s in self.samples if t0 <= s.t <= t1]


# server-log lines that mean a stage was lost (vLLM-Omni evicts a stage replica that died, e.g. out of memory)
EVICTION = ("no live replica", "is dead", "CUDA out of memory", "OutOfMemoryError")


class LogWatch:
    """New lines matching ``patterns`` in server logs since ``mark()``. ``run(cmd) -> stdout`` runs a shell
    command where the logs are (locally or over ssh); ``paths`` are the log files."""

    def __init__(self, run, paths: list[str], patterns=EVICTION, max_lines: int = 5):
        self.run, self.paths, self.patterns, self.max_lines = run, list(paths), patterns, max_lines
        self.offsets: dict[str, int] = {}

    def mark(self) -> None:
        for p in self.paths:
            out = self.run(f"wc -c < {p} 2>/dev/null || echo 0").strip().splitlines()
            self.offsets[p] = int(out[-1]) if out and out[-1].isdigit() else 0

    def check(self) -> list[str]:
        pat = "|".join(self.patterns)  # plain words: no regex escaping (GNU grep warns on stray backslashes)
        hits = []
        for p in self.paths:
            off = self.offsets.get(p, 0)
            out = self.run(f"tail -c +{off + 1} {p} 2>/dev/null | grep -m {self.max_lines} -E '{pat}'")
            hits += [f"{p.rsplit('/', 1)[-1]}: {line.strip()[:200]}" for line in out.splitlines() if line.strip()]
        return hits
