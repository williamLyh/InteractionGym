"""Restarting a host into a layout (stage 2 of validation). This stops services: callers must gate it
behind an explicit flag and an explicit stop command — nothing here guesses what to stop."""

from __future__ import annotations

import asyncio
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from .launch import endpoints, launch_script
from .solve import Hardware, Plan, ServiceSpec


LOCAL = ("local", "localhost")


class SSHLayout:
    """``apply(plan)``: copy the plan's launch script to ``host``, run ``stop_cmd`` there, wait until the
    plan's GPUs are free, start the script, and wait until every endpoint answers ``GET /v1/models``
    (then ``warmup``, if given). ``host="local"``: the tuner runs on the GPU host itself (no ssh)."""

    def __init__(self, host: str, specs: dict[str, ServiceSpec], hw: Hardware, stop_cmd: str, *, ready_timeout_s: float = 900,
                 settle_s: float = 10, warmup=None, scratch: str | Path = ".", log=print):
        assert stop_cmd, "an explicit stop command is required (what to stop is the caller's decision)"
        self.host, self.specs, self.hw, self.stop_cmd = host, specs, hw, stop_cmd
        self.ready_timeout_s, self.settle_s, self.warmup, self.scratch, self.log = ready_timeout_s, settle_s, warmup, Path(scratch), log

    def _ssh(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        argv = ["bash", "-c", cmd] if self.host in LOCAL else ["ssh", self.host, cmd]
        return subprocess.run(argv, capture_output=True, text=True, check=check)

    def run(self, cmd: str) -> str:
        """stdout of a shell command on the host (for ``LogWatch``)."""
        return self._ssh(cmd, False).stdout

    def log_paths(self, plan: Plan) -> list[str]:
        """The server logs ``launch_script`` writes (``logs/<session>.log``)."""
        return [f"{self.hw.workdir}/logs/{self.hw.session_prefix}{e['name']}.log"
                for es in endpoints(plan, self.specs).values() for e in es if e.get("gpus")]

    async def alive(self, plan: Plan) -> bool:
        """Every server of the plan still answers ``GET /v1/models``."""
        ports = [e["port"] for es in endpoints(plan, self.specs).values() for e in es if e.get("port")]
        probe = " && ".join(f"curl -sf -m 5 http://127.0.0.1:{p}/v1/models >/dev/null" for p in ports)
        return not ports or (await asyncio.to_thread(self._ssh, probe, False)).returncode == 0

    async def _wait_gpus_free(self, gpus: list[int], *, max_mb: int = 1500, timeout_s: float = 180) -> None:
        """Killed servers free GPU memory asynchronously (vLLM workers can take tens of seconds); starting the
        next layout before that OOMs it at load time."""
        if not gpus:
            return
        q = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
        t0 = time.monotonic()
        while True:
            r = await asyncio.to_thread(self._ssh, q, False)
            used = {int(a): int(b) for a, b in (ln.split(",") for ln in r.stdout.strip().splitlines() if "," in ln)}
            busy = {g: used[g] for g in gpus if used.get(g, 0) > max_mb}
            if not busy:
                return
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"GPUs still in use {timeout_s:.0f} s after the stop command (MiB): {busy}")
            await asyncio.sleep(5)

    async def apply(self, plan: Plan) -> None:
        name = f"tune_{int(time.time())}.sh"
        local = self.scratch / name
        local.write_text(launch_script(plan, self.specs, self.hw))
        remote = f"{self.hw.workdir}/tuned/{name}"
        await asyncio.to_thread(self._ssh, f"mkdir -p {shlex.quote(self.hw.workdir)}/tuned")
        if self.host in LOCAL:
            Path(remote).parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copyfile, local, remote)
        else:
            await asyncio.to_thread(subprocess.run, ["scp", "-q", str(local), f"{self.host}:{remote}"], check=True)
        self.log(f"  stopping: {self.stop_cmd}")
        # the stop command may legitimately exit non-zero (nothing to stop): not checked
        await asyncio.to_thread(self._ssh, f"cd {shlex.quote(self.hw.workdir)} && {self.stop_cmd}", False)
        await asyncio.sleep(self.settle_s)
        await self._wait_gpus_free(sorted({g for reps in plan.placement.values() for r in reps for g in r}))
        self.log(f"  starting {remote}")
        await asyncio.to_thread(self._ssh, f"bash {shlex.quote(remote)}")
        eps = [e for es in endpoints(plan, self.specs).values() for e in es]
        ports = [e["port"] for e in eps if e.get("port")]
        sessions = [f"{self.hw.session_prefix}{e['name']}" for e in eps]
        t0 = time.monotonic()
        while True:
            probe = " && ".join(f"curl -sf -m 3 http://127.0.0.1:{p}/v1/models >/dev/null" for p in ports)
            if (await asyncio.to_thread(self._ssh, probe, False)).returncode == 0:
                break
            alive = (await asyncio.to_thread(self._ssh, "tmux ls -F '#S' 2>/dev/null", False)).stdout.split()
            dead = [x for x in sessions if x not in alive]
            if dead:  # a server exited while loading (e.g. out of memory): no point waiting for its port
                tail = (await asyncio.to_thread(self._ssh, f"cd {shlex.quote(self.hw.workdir)} && tail -n 20 logs/{dead[0]}.log", False)).stdout
                raise RuntimeError(f"server session(s) {dead} exited during start-up; logs/{dead[0]}.log ends with:\n{tail}")
            if time.monotonic() - t0 > self.ready_timeout_s:
                raise TimeoutError(f"layout not ready after {self.ready_timeout_s:.0f} s (ports {ports})")
            await asyncio.sleep(15)
        self.log(f"  layout ready in {time.monotonic() - t0:.0f} s")
        if self.warmup is not None:
            await self.warmup(plan)
