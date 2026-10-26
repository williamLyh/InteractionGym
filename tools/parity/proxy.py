"""A record / replay proxy for the model servers behind an InteractionGym run — the instrument of a parity test.

Two runs that issue the same requests in front of the same recorded answers must produce the same trajectories; a
request the recording has not seen is a behavioural difference. So: run the reference implementation through the
proxy in ``record`` mode (every HTTP request keyed by method, path and canonical body; every WebSocket session as
the sequence of client messages with the server messages that followed each), then run the implementation under
test in ``replay`` mode against the same store — answers come from the store, a request without one is a **miss**:
logged to ``<store>/misses.jsonl`` with a diff against the nearest recorded request and answered with HTTP 460 (a
client error, so the run fails at once instead of retrying) / a WebSocket ``error`` event. ``summary.json`` counts
hits and misses per upstream.

    python tools/parity/proxy.py --mode record --store runs/parity/store \\
        --map 18000=http://127.0.0.1:8000 --map 18001=http://127.0.0.1:8001 --map 18010=ws://127.0.0.1:8010

Point the run at the proxy's ports (``IG_LLM_URL=http://127.0.0.1:18000/v1`` …). Identical requests repeated get
their recorded answers in order (a sampled model may answer the same prompt differently twice); more repetitions
than recorded reuse the last answer and are counted as ``overflow``. Only stdlib and ``websockets`` are needed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import difflib
import hashlib
import json
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MISS_STATUS = 460
LONG = 96  # characters kept at each end of a long string in a diff


# ----------------------------------------------------------------------------- canonical forms and diffs

def canonical(body: bytes, content_type: str) -> tuple[str, object]:
    """``(key material, a JSON-able picture)`` of a request body: JSON is re-serialised with sorted keys; multipart
    bodies keep their text fields and hash their file parts; anything else is hashed."""
    ct = (content_type or "").lower()
    if "json" in ct or body[:1] in (b"{", b"["):
        try:
            obj = json.loads(body)
            return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")), obj
        except ValueError:
            pass
    if "multipart/form-data" in ct:
        m = re.search(r"boundary=([^;]+)", ct)
        if m:
            parts = {}
            for part in body.split(b"--" + m.group(1).strip().encode()):
                if b"\r\n\r\n" not in part:
                    continue
                head, _, data = part.partition(b"\r\n\r\n")
                name = re.search(rb'name="([^"]*)"', head)
                data = data[:-2] if data.endswith(b"\r\n") else data
                if name:
                    k = name.group(1).decode()
                    parts[k] = ({"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} if b"filename=" in head
                                else data.decode("utf-8", "replace"))
            return json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":")), parts
    h = hashlib.sha256(body).hexdigest()
    return h, {"sha256": h, "bytes": len(body)}


def abbreviate(x):
    """Long strings (base64 audio, long prompts) shortened to their ends and a hash, so a diff stays readable."""
    if isinstance(x, str) and len(x) > 2 * LONG + 20:
        return f"{x[:LONG]}…<{len(x)} chars sha256:{hashlib.sha256(x.encode()).hexdigest()[:12]}>…{x[-LONG:]}"
    if isinstance(x, dict):
        return {k: abbreviate(v) for k, v in x.items()}
    if isinstance(x, list):
        return [abbreviate(v) for v in x]
    return x


def pretty(obj) -> str:
    return json.dumps(abbreviate(obj), indent=1, sort_keys=True, ensure_ascii=False)


def diff(recorded, got, n: int = 60) -> str:
    a, b = pretty(recorded).splitlines(), pretty(got).splitlines()
    lines = list(difflib.unified_diff(a, b, "recorded", "got", lineterm="", n=2))
    return "\n".join(lines[:n]) + ("\n…" if len(lines) > n else "")


def nearest(candidates: list, got) -> tuple[object, str]:
    """The recorded request most like ``got`` (by sequence similarity of their pretty forms) and the diff."""
    if not candidates:
        return None, "(no recorded request on this path)"
    g = pretty(got)
    best = max(candidates, key=lambda c: difflib.SequenceMatcher(None, pretty(c)[:20000], g[:20000]).quick_ratio())
    return best, diff(best, got)


# ----------------------------------------------------------------------------- the store

class Store:
    """``<store>/http/<key>/{req.json,0.resp,1.resp,…}`` (``.resp``: a JSON header line, then the body) and
    ``<store>/ws/<n>.jsonl`` (one session: ``{"c": msg}`` / ``{"s": msg}`` lines in order). Thread-safe."""

    def __init__(self, root: Path, mode: str):
        self.root, self.mode = Path(root), mode
        self.lock = threading.Lock()
        (self.root / "http").mkdir(parents=True, exist_ok=True)
        (self.root / "ws").mkdir(parents=True, exist_ok=True)
        self.served: dict[str, int] = {}  # replay: how many answers of a key were handed out
        self.stats = {"http": {"hit": 0, "miss": 0, "overflow": 0, "recorded": 0}, "ws": {"sessions": 0, "miss": 0, "recorded": 0}}
        self.misses = open(self.root / "misses.jsonl", "a")
        self.by_path: dict[str, list] = {}  # replay: path -> recorded request pictures (for the nearest-diff)
        self.ws_sessions: list[list[dict]] = []
        self.ws_taken: set[int] = set()
        if mode == "replay":
            for d in (self.root / "http").iterdir():
                req = json.loads((d / "req.json").read_text())
                self.by_path.setdefault(req["path"], []).append(req["body"])
            for p in sorted((self.root / "ws").glob("*.jsonl"), key=lambda p: int(p.stem)):
                self.ws_sessions.append([json.loads(l) for l in p.read_text().splitlines() if l.strip()])

    # http
    def key(self, method: str, path: str, canon: str) -> str:
        return hashlib.sha256(f"{method} {path}\n{canon}".encode()).hexdigest()

    def put(self, key: str, method: str, path: str, picture, status: int, headers: dict, body: bytes) -> None:
        d = self.root / "http" / key
        with self.lock:
            d.mkdir(exist_ok=True)
            if not (d / "req.json").exists():
                (d / "req.json").write_text(json.dumps({"method": method, "path": path, "body": picture}, ensure_ascii=False))
            n = len(list(d.glob("*.resp")))
            (d / f"{n}.resp").write_bytes(json.dumps({"status": status, "headers": headers}).encode() + b"\n" + body)
            self.stats["http"]["recorded"] += 1

    def get(self, key: str) -> tuple[int, dict, bytes] | None:
        d = self.root / "http" / key
        with self.lock:
            files = sorted(d.glob("*.resp"), key=lambda p: int(p.stem)) if d.exists() else []
            if not files:
                self.stats["http"]["miss"] += 1
                return None
            n = self.served.get(key, 0)
            self.served[key] = n + 1
            if n >= len(files):
                self.stats["http"]["overflow"] += 1
                req = json.loads((d / "req.json").read_text())
                self.misses.write(json.dumps({"t": time.time(), "kind": "overflow", "path": req["path"], "got": abbreviate(req["body"]),
                                              "note": f"request repeated {n + 1} times, recorded {len(files)}"}, ensure_ascii=False) + "\n")
                self.misses.flush()
            else:
                self.stats["http"]["hit"] += 1
            raw = files[min(n, len(files) - 1)].read_bytes()
        head, _, body = raw.partition(b"\n")
        h = json.loads(head)
        return h["status"], h["headers"], body

    def miss(self, kind: str, path: str, got, note: str = "") -> str:
        with self.lock:
            best, d = nearest(self.by_path.get(path, []), got) if kind == "http" else (None, note)
            rec = {"t": time.time(), "kind": kind, "path": path, "got": abbreviate(got), "nearest": abbreviate(best), "diff": d, "note": note}
            self.misses.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.misses.flush()
            if kind == "ws":
                self.stats["ws"]["miss"] += 1
        return d

    # ws
    def new_session(self) -> "SessionWriter":
        with self.lock:
            n = len(list((self.root / "ws").glob("*.jsonl")))
            self.stats["ws"]["recorded"] += 1
            return SessionWriter(self.root / "ws" / f"{n}.jsonl")

    def candidates(self, hashes: list[str], exclude: int | None = None) -> list[int]:
        """Recorded sessions not yet taken whose client messages start with ``hashes`` (lazy binding: sessions that
        start alike — the same ``session.update`` — are told apart by their later messages)."""
        with self.lock:
            out = []
            for i, s in enumerate(self.ws_sessions):
                if i in self.ws_taken and i != exclude:
                    continue
                cl = [e["h"] for e in s if "c" in e]
                if len(cl) >= len(hashes) and cl[:len(hashes)] == hashes:
                    out.append(i)
            return out

    def take_session(self, i: int) -> None:
        with self.lock:
            if i not in self.ws_taken:
                self.ws_taken.add(i)
                self.stats["ws"]["sessions"] += 1

    def summary(self) -> dict:
        return {"mode": self.mode, **self.stats, "ws_unused_sessions": len(self.ws_sessions) - len(self.ws_taken) if self.mode == "replay" else None}

    def flush(self) -> None:
        (self.root / "summary.json").write_text(json.dumps(self.summary(), indent=1))


class SessionWriter:
    def __init__(self, path: Path):
        self.f = open(path, "a")
        self.lock = threading.Lock()

    def write(self, side: str, msg, h: str | None = None) -> None:
        rec = {side: msg} if h is None else {"h": h, side: msg}
        with self.lock:
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.f.flush()

    def close(self) -> None:
        self.f.close()


def ws_hash(msg) -> str:
    m = msg if isinstance(msg, str) else base64.b64encode(msg).decode()
    try:  # JSON messages: key by content, not formatting
        m = json.dumps(json.loads(m), sort_keys=True, separators=(",", ":"))
    except (ValueError, TypeError):
        pass
    return hashlib.sha256(m.encode()).hexdigest()


def ws_picture(msg):
    try:
        return json.loads(msg) if isinstance(msg, str) else {"bytes": len(msg)}
    except ValueError:
        return msg


# ----------------------------------------------------------------------------- HTTP

FORWARD_HEADERS = ("content-type", "authorization", "accept")


def http_server(port: int, upstream: str, store: Store, log) -> ThreadingHTTPServer:
    base = upstream.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _reply(self, status: int, headers: dict, body: bytes) -> None:
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            path = self.path
            canon, picture = canonical(body, self.headers.get("Content-Type", ""))
            key = store.key(self.command, path, canon)
            if store.mode == "replay":
                hit = store.get(key)
                if hit is None:
                    d = store.miss("http", path, picture)
                    log(f"MISS {self.command} {path}\n{d}")
                    self._reply(MISS_STATUS, {"Content-Type": "application/json"},
                                json.dumps({"error": "parity proxy: no recorded answer for this request", "path": path}).encode())
                    return
                self._reply(*hit)
                return
            headers = {k: v for k, v in self.headers.items() if k.lower() in FORWARD_HEADERS}
            req = urllib.request.Request(base + path, body if n else None, headers, method=self.command)
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    status, rh, rb = r.status, {"Content-Type": r.headers.get("Content-Type", "application/octet-stream")}, r.read()
            except urllib.error.HTTPError as e:
                status, rh, rb = e.code, {"Content-Type": e.headers.get("Content-Type", "application/json")}, e.read()
            except Exception as e:  # upstream down: not recorded (the client retries)
                self._reply(502, {"Content-Type": "application/json"}, json.dumps({"error": f"parity proxy: upstream {e}"}).encode())
                return
            if status < 500:  # transient server errors are not part of the behaviour under test
                store.put(key, self.command, path, picture, status, rh, rb)
            self._reply(status, rh, rb)

        do_POST = do_GET = do_PUT = do_DELETE = _handle

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    return srv


# ----------------------------------------------------------------------------- WebSocket

async def ws_serve(port: int, upstream: str, store: Store, log):
    import websockets

    async def record(client):
        path = getattr(getattr(client, "request", None), "path", "") or getattr(client, "path", "")
        url = upstream.rstrip("/") + path
        async with websockets.connect(url, max_size=None) as server:
            w = store.new_session()

            async def c2s():
                try:
                    async for m in client:
                        w.write("c", ws_picture(m), ws_hash(m))
                        await server.send(m)
                finally:
                    await server.close()

            async def s2c():
                try:
                    async for m in server:
                        w.write("s", m if isinstance(m, str) else {"__bytes__": base64.b64encode(m).decode()})
                        await client.send(m)
                finally:
                    await client.close()

            await asyncio.gather(c2s(), s2c(), return_exceptions=True)
            w.close()

    async def replay(client):
        """Each client message narrows the recorded sessions whose client messages match so far; the server messages
        that followed the matching message in the chosen session are sent back. A session is taken for good at
        its end (so two sessions with the same prefix can still be told apart while both run)."""
        hashes, chosen = [], None
        try:
            async for m in client:
                h = ws_hash(m)
                hashes.append(h)
                cands = store.candidates(hashes, exclude=chosen)
                if not cands:
                    exp = None
                    prev = store.candidates(hashes[:-1], exclude=chosen) if len(hashes) > 1 else []
                    if prev:  # the nearest: the previous candidate's next client message
                        s0 = store.ws_sessions[chosen if chosen in prev else prev[0]]
                        cl = [e["c"] for e in s0 if "c" in e]
                        exp = cl[len(hashes) - 1] if len(cl) >= len(hashes) else None
                    d = diff(exp, ws_picture(m))
                    store.miss("ws", "message", ws_picture(m), d if exp is not None else "no recorded session continues this way")
                    log(f"WS MISS at client message {len(hashes) - 1} (candidates before it: {len(prev)})\n{d}")
                    await client.send(json.dumps({"type": "error", "error": {"code": "parity_mismatch",
                                                   "message": f"client message {len(hashes) - 1} differs from every recorded session"}}))
                    await client.close()
                    return
                chosen = chosen if chosen in cands else cands[0]
                session = store.ws_sessions[chosen]
                # the server messages after the (len(hashes)-1)-th client message of the chosen session
                k, i = -1, 0
                while i < len(session):
                    if "c" in session[i]:
                        k += 1
                        if k == len(hashes) - 1:
                            break
                    i += 1
                i += 1
                while i < len(session) and "s" in session[i]:
                    sm = session[i]["s"]
                    await client.send(base64.b64decode(sm["__bytes__"]) if isinstance(sm, dict) and "__bytes__" in sm else sm)
                    i += 1
        finally:
            if chosen is not None:
                store.take_session(chosen)
        await client.close()

    handler = record if store.mode == "record" else replay
    return await websockets.serve(handler, "127.0.0.1", port, max_size=None)


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["record", "replay"], required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--map", action="append", default=[], help="LISTEN_PORT=UPSTREAM_URL (http(s):// or ws(s)://)")
    ap.add_argument("--log", default=None, help="log file (default stderr)")
    a = ap.parse_args(argv)
    store = Store(Path(a.store), a.mode)
    out = open(a.log, "a") if a.log else sys.stderr

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=out, flush=True)

    servers, ws = [], []
    for m in a.map:
        port, up = m.split("=", 1)
        (ws if up.startswith("ws") else servers).append((int(port), up))
    for port, up in servers:
        srv = http_server(port, up, store, log)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        log(f"{a.mode}: http :{port} -> {up}")

    async def run():
        for port, up in ws:
            await ws_serve(port, up, store, log)
            log(f"{a.mode}: ws :{port} -> {up}")
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        while not stop.is_set():
            await asyncio.wait([asyncio.ensure_future(stop.wait())], timeout=10)
            store.flush()
        store.flush()
        log(f"stopped: {json.dumps(store.summary())}")

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
