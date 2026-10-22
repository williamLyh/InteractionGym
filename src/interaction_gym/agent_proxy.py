"""One address in front of several full-duplex agent servers: a WebSocket proxy that pins each session to one server.

A vLLM-Omni duplex server serves one Thinker on one GPU (all current MiniCPM-o configs have a single Thinker replica;
the engine rejects ``--data-parallel-size``), so several GPUs mean several servers. A duplex session is stateful —
every unit continues the same KV cache — so it cannot be balanced per request the way the TTS proxy balances HTTP
calls; it can be balanced per *connection*. This proxy accepts a client's WebSocket, picks the server with the fewest
open sessions (ties rotate, so an idle pool fills evenly), opens the same path and query there, and relays messages
both ways unchanged until either side closes. Clients keep one ``IG_AGENT_URL``.

- ``max_sessions`` per server: a session that finds every server full waits for a free one (up to
  ``queue_timeout_s``, then the client is closed with 1013 "try again later") instead of being refused by a server.
- A server that does not accept the connection is skipped for that session (the next least-loaded one is tried).
- A plain HTTP request (no WebSocket upgrade) gets the open sessions per server as JSON, e.g. for health checks.

    python -m interaction_gym.agent_proxy --backends ws://127.0.0.1:8010,ws://127.0.0.1:8011 --port 8009 --max-sessions 16

The relay adds one local hop per message (measured on a laptop: tens of microseconds per round trip, see
tests/test_agent_proxy.py), which matters neither for real-time nor for lockstep sessions.

Future: if vLLM-Omni gains data-parallel Thinker replicas with per-session affinity (its stage pools already route
by affinity for Stage-1 replicas), one server could span the GPUs itself and this proxy would retire
(docs/agent_server.md, "Future: Thinker data parallelism").
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from http import HTTPStatus

from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

FORWARD_HEADERS = ("authorization",)


def servers_behind(url: str, timeout: float = 5.0) -> int:
    """How many agent servers answer at ``url`` (ws://...): the proxy's server count, or 1 for a server itself (or
    anything that does not answer like the proxy). For per-server work such as a warm-up session on each."""
    import urllib.parse
    import urllib.request

    u = urllib.parse.urlparse(url)
    http = urllib.parse.urlunparse(("https" if u.scheme == "wss" else "http", u.netloc, "/", "", "", ""))
    try:
        with urllib.request.urlopen(http, timeout=timeout) as r:
            return max(1, len(json.loads(r.read())["backends"]))
    except Exception:  # noqa: BLE001 — not the proxy
        return 1


class AgentProxy:
    def __init__(self, backends: list[str] | str, max_sessions: int | None = None, queue_timeout_s: float = 600.0,
                 log=print):
        backends = backends.split(",") if isinstance(backends, str) else list(backends)
        self.backends = [b.strip().rstrip("/") for b in backends if b.strip()]
        assert self.backends, "AgentProxy needs at least one backend"
        assert max_sessions is None or max_sessions > 0
        self.max_sessions, self.queue_timeout_s, self.log = max_sessions, queue_timeout_s, log
        self.open = {b: 0 for b in self.backends}
        self.served = {b: 0 for b in self.backends}
        self.waiting = 0
        self._next = 0
        self._free: asyncio.Condition | None = None

    # ---------------------------------------------------------------- choosing a server

    def _pick(self, exclude: set[str]) -> str | None:
        n = len(self.backends)
        order = [self.backends[(self._next + k) % n] for k in range(n)]
        ok = [b for b in order if b not in exclude and (self.max_sessions is None or self.open[b] < self.max_sessions)]
        if not ok:
            return None
        b = min(ok, key=lambda x: self.open[x])  # min keeps the first of equals: the rotation order
        self._next = (self.backends.index(b) + 1) % n
        return b

    async def acquire(self, exclude: set[str] = frozenset()) -> str | None:
        """A server with a free slot (counted as open), waiting for one if all are full; None if every server is
        excluded or the wait timed out."""
        if self._free is None:
            self._free = asyncio.Condition()
        if len(exclude) >= len(self.backends):
            return None
        deadline = time.monotonic() + self.queue_timeout_s
        async with self._free:
            self.waiting += 1
            try:
                while (b := self._pick(set(exclude))) is None:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return None
                    try:
                        await asyncio.wait_for(self._free.wait(), left)
                    except asyncio.TimeoutError:
                        return None
            finally:
                self.waiting -= 1
            self.open[b] += 1
            return b

    async def release(self, b: str) -> None:
        self.open[b] -= 1
        if self._free is not None:
            async with self._free:
                self._free.notify()

    def status(self) -> dict:
        return {"backends": [{"url": b, "open": self.open[b], "served": self.served[b]} for b in self.backends],
                "max_sessions": self.max_sessions, "waiting": self.waiting}

    # ---------------------------------------------------------------- relaying

    async def handle(self, client: ServerConnection) -> None:
        path = client.request.path
        headers = {k: v for k, v in client.request.headers.raw_items() if k.lower() in FORWARD_HEADERS}
        tried: set[str] = set()
        while True:
            b = await self.acquire(tried)
            if b is None:
                reason = "no agent server accepted the session" if tried else "every agent server is full"
                self.log(f"agent_proxy: {reason}; closing a client")
                await client.close(1013 if not tried else 1011, reason)
                return
            try:
                up = await connect(b + path, max_size=None, compression=None, ping_interval=None, open_timeout=30,
                                   additional_headers=headers)
            except Exception as e:  # noqa: BLE001 — refused, down, or a bad handshake: try another server
                self.log(f"agent_proxy: {b} did not accept a session ({type(e).__name__}: {e})")
                tried.add(b)
                await self.release(b)
                continue
            break
        self.served[b] += 1
        try:
            await self._relay(client, up)
        finally:
            await self.release(b)

    @staticmethod
    async def _relay(client, up) -> None:
        async def pump(src, dst):
            try:
                async for msg in src:
                    await dst.send(msg)
            except ConnectionClosed:
                pass

        tasks = [asyncio.ensure_future(pump(client, up)), asyncio.ensure_future(pump(up, client))]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            # whoever closed first, the other side gets the same close code
            src = up if up.close_code is not None else client
            dst = client if src is up else up
            code = src.close_code if src.close_code not in (None, 1005, 1006) else 1000
            await dst.close(code, src.close_reason or "")
            await src.close()

    def _http(self, connection: ServerConnection, request):
        if request.headers.get("Upgrade", "").lower() != "websocket":
            resp = connection.respond(HTTPStatus.OK, json.dumps(self.status()) + "\n")
            resp.headers["Content-Type"] = "application/json"
            return resp
        return None

    async def serve(self, host: str = "127.0.0.1", port: int = 8009):
        """Start listening; returns the websockets server (use as an async context manager or ``.serve_forever()``)."""
        return await serve(self.handle, host, port, max_size=None, compression=None, ping_interval=None,
                           process_request=self._http)


async def _main(args) -> None:
    proxy = AgentProxy(args.backends, args.max_sessions, args.queue_timeout_s)
    server = await proxy.serve(args.host, args.port)
    print(f"agent_proxy: ws://{args.host}:{args.port} -> {', '.join(proxy.backends)}"
          + (f" (max {args.max_sessions} sessions each)" if args.max_sessions else ""), flush=True)
    await server.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backends", default=os.environ.get("AGENT_BACKENDS", ""),
                    help="comma-separated ws://host:port of the agent servers (default: $AGENT_BACKENDS)")
    ap.add_argument("--host", default=os.environ.get("IG_BIND_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8009")))
    ap.add_argument("--max-sessions", type=int, default=int(os.environ["MAX_SESSIONS"]) if os.environ.get("MAX_SESSIONS") else None,
                    help="sessions per server; more wait for a free one (default: $MAX_SESSIONS, else no limit)")
    ap.add_argument("--queue-timeout-s", type=float, default=600.0)
    args = ap.parse_args()
    if not args.backends:
        ap.error("no backends: pass --backends or set AGENT_BACKENDS")
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
