"""tools/parity: the record / replay proxy answers a replayed run from the recording and reports what differs; the
comparer joins episodes across naming schemes and tells behavioural from float-only differences."""

import asyncio
import importlib.util
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "tools" / "parity"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


proxy = _load("proxy")
compare = _load("compare")


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Upstream(BaseHTTPRequestHandler):
    """Answers chat requests with a counter (so identical prompts get different answers, like a sampled model)."""
    n = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Upstream.n += 1
        out = json.dumps({"echo": body["messages"][-1]["content"], "n": Upstream.n}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def _post(port, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_http_record_then_replay_serves_identical_answers_and_reports_a_miss(tmp_path):
    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    store = tmp_path / "store"
    rec = proxy.Store(store, "record")
    p_rec = _port()
    srv = proxy.http_server(p_rec, f"http://127.0.0.1:{up.server_port}", rec, lambda m: None)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    a = _post(p_rec, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    b = _post(p_rec, {"messages": [{"role": "user", "content": "hi"}], "model": "m"})  # same request, other key order
    c = _post(p_rec, {"model": "m", "messages": [{"role": "user", "content": "bye"}]})
    assert (a["n"], b["n"], c["n"]) == (1, 2, 3)
    srv.shutdown()
    rec.flush()
    assert rec.summary()["http"]["recorded"] == 3

    rep = proxy.Store(store, "replay")
    p_rep = _port()
    srv2 = proxy.http_server(p_rep, "http://127.0.0.1:1", rep, lambda m: None)  # no upstream needed
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    assert _post(p_rep, {"model": "m", "messages": [{"role": "user", "content": "bye"}]}) == c
    assert _post(p_rep, {"model": "m", "messages": [{"role": "user", "content": "hi"}]}) == a  # answers in recorded order
    assert _post(p_rep, {"model": "m", "messages": [{"role": "user", "content": "hi"}]}) == b
    assert _post(p_rep, {"model": "m", "messages": [{"role": "user", "content": "hi"}]}) == b  # overflow: the last again
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(p_rep, {"model": "m", "messages": [{"role": "user", "content": "hi there"}]})
    assert e.value.code == proxy.MISS_STATUS
    srv2.shutdown()
    rep.flush()
    s = rep.summary()
    assert s["http"] == {"hit": 3, "miss": 1, "overflow": 1, "recorded": 0}
    miss = json.loads((store / "misses.jsonl").read_text().splitlines()[-1])
    assert miss["path"] == "/v1/chat/completions" and "hi there" in miss["diff"] and miss["nearest"]["messages"][0]["content"] == "hi"


def test_ws_record_then_replay(tmp_path):
    websockets = pytest.importorskip("websockets")

    async def go():
        async def upstream(ws):
            k = 0
            async for m in ws:
                ev = json.loads(m)
                k += 1
                if ev["type"] == "session.update":
                    await ws.send(json.dumps({"type": "session.created"}))
                else:
                    await ws.send(json.dumps({"type": "response.output_text.delta", "delta": f"u{k}"}))
                    await ws.send(json.dumps({"type": "input_audio_buffer.processed", "event_id": ev["event_id"]}))

        up = await websockets.serve(upstream, "127.0.0.1", 0)
        up_port = up.sockets[0].getsockname()[1]
        store = tmp_path / "store"
        rec = proxy.Store(store, "record")
        p_rec = _port()
        srv = await proxy.ws_serve(p_rec, f"ws://127.0.0.1:{up_port}", rec, lambda m: None)

        async def client(port, msgs):
            got = []
            async with websockets.connect(f"ws://127.0.0.1:{port}/v1/realtime?duplex=1") as ws:
                for m in msgs:
                    await ws.send(json.dumps(m))
                    got.append(json.loads(await ws.recv()))
                    if m["type"] != "session.update":
                        got.append(json.loads(await ws.recv()))
            return got

        msgs = [{"type": "session.update", "session": {"model": "x"}}, {"type": "input_audio_buffer.append", "audio": "AAAA", "event_id": "in1.0"},
                {"type": "input_audio_buffer.append", "audio": "BBBB", "event_id": "in2.0"}]
        got_rec = await client(p_rec, msgs)
        srv.close()
        await srv.wait_closed()
        rec.flush()

        rep = proxy.Store(store, "replay")
        p_rep = _port()
        srv2 = await proxy.ws_serve(p_rep, "ws://127.0.0.1:1", rep, lambda m: None)
        assert await client(p_rep, msgs) == got_rec
        # a diverging client message is a miss: an error event, then the connection closes
        async with websockets.connect(f"ws://127.0.0.1:{p_rep}/v1/realtime?duplex=1") as ws:
            await ws.send(json.dumps(msgs[0]))  # a second session with the same start: no recorded session left
            err = json.loads(await ws.recv())
            assert err["type"] == "error" and err["error"]["code"] == "parity_mismatch"
        srv2.close()
        await srv2.wait_closed()
        rep.flush()
        assert rep.summary()["ws"] == {"sessions": 1, "miss": 1, "recorded": 0}
        up.close()
        await up.wait_closed()

    asyncio.run(go())


def test_compare_joins_by_task_and_seed_and_separates_float_drift(tmp_path):
    def ep(eid, task, seed, score, text="hello", extra=None):
        return {"meta": {"episode_id": eid, "task": {"id": task}, "seed": seed, "run_id": eid, "wall_s": 1.0, "config": {"x": eid}},
                "turns": [{"role": "user", "text": text, "start_time": 0, "end_time": 1000}], "eval": {"scores": {"total": score}, **(extra or {})}}
    ref = tmp_path / "ref.jsonl"
    new = tmp_path / "new.jsonl"
    ref.write_text("\n".join(json.dumps(e) for e in [ep("a-s0", "a", 0, 0.5), ep("b-s1", "b", 1, 0.25), ep("c-s0", "c", 0, 1.0)]))
    new.write_text("\n".join(json.dumps(e) for e in [ep("a/s0", "a", 0, 0.5), ep("b/s1", "b", 1, 0.25 + 1e-12), ep("c/s0", "c", 0, 1.0, text="bye")]))
    rep = compare.compare(compare.load(str(ref)), compare.load(str(new)))
    assert rep["episodes"] == {"ref": 3, "new": 3, "common": 3, "identical": 1, "differ": 1, "float_only": 1}
    assert rep["verdict"] == "DIFFER" and rep["max_float_err"] == pytest.approx(1e-12)
    assert rep["diffs"]["c|s0|"]["fields"][0]["path"] == "turns[0].text"
    assert compare.main([str(ref), str(ref)]) == 0
