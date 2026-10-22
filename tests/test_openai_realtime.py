"""The OpenAI Realtime adapter against a local stand-in that behaves as the probe found the real API to (2026-10-22):
turn detection on the audio received (not wall time), audio streamed slower than real time, a response cancelled when
the user speaks, truncation / deletion of what was not heard, function calls. Episodes run in wall time (the adapter
paces the microphone), so they are kept to a few seconds."""

import asyncio
import base64
import json
import time
from array import array

import pytest

websockets = pytest.importorskip("websockets")

from interaction_gym import AgentSpec, Env, Task  # noqa: E402
from interaction_gym.agents.openai_realtime import OpenAIRealtimeAgent  # noqa: E402
from interaction_gym.audio import Audio  # noqa: E402
from interaction_gym.tools import FunctionBackend, ToolWorld  # noqa: E402
from interaction_gym.traj import episode  # noqa: E402
from interaction_gym.user import ReplayUser, Voice  # noqa: E402
from tests.test_traj import check  # noqa: E402

SR = 24000


class ToneSpeech:
    """A tone as long as the words take to say (the stand-in server's VAD hears it as speech)."""

    clones = True

    async def synth(self, text, voice="default", instructions=None, ref_audio=None, ref_text=None, language=None, seed=None):
        return Audio(array("h", [3000, -3000] * (len(text.split()) * SR // 3 // 2)), SR)  # 3 words/s


class Fake:
    """Server settings and what it saw."""

    def __init__(self, latency_s=0.4, deltas=5, delta_ms=200, gap_s=0.3, tool=False, wire=None):
        self.latency_s, self.deltas, self.delta_ms, self.gap_s, self.tool = latency_s, deltas, delta_ms, gap_s, tool
        self.wire = wire  # a slow link: audio leaves at this many x real time, every event queued behind it
        self.got: list[dict] = []
        self.created_wall: list[float] = []


async def fake_realtime(ws, f: Fake):
    assert ws.request.headers["Authorization"] == "Bearer sk-test" and "model=gpt-realtime-mini" in ws.request.path
    if f.wire:  # every message through one queue, audio drained at f.wire x real time (as over a slow WebSocket)
        q: asyncio.Queue = asyncio.Queue()
        raw_send = ws.send

        async def drain():
            while True:
                m = await q.get()
                if '"response.output_audio.delta"' in m:
                    await asyncio.sleep(f.delta_ms / 1000 / f.wire)
                await raw_send(m)

        asyncio.ensure_future(drain())

        async def queued(m):
            q.put_nowait(m)

        ws.send = queued
    await ws.send(json.dumps({"type": "session.created", "session": {}}))
    pos = speech_run = silence_run = 0
    speaking, active, called, n = False, None, False, 0

    async def respond(rid):
        nonlocal active, called
        await ws.send(json.dumps({"type": "response.created", "response": {"id": rid}}))
        f.created_wall.append(time.monotonic())
        await asyncio.sleep(f.latency_s)
        if f.tool and not called:
            called = True
            await ws.send(json.dumps({"type": "response.function_call_arguments.done", "response_id": rid, "call_id": "call_1",
                                      "name": "lookup", "arguments": json.dumps({"q": "hours"})}))
            await ws.send(json.dumps({"type": "response.done", "response": {"id": rid, "status": "completed"}}))
            active = None
            return
        for k in range(f.deltas):
            if active != rid:
                return
            pcm = array("h", [2000] * (SR * f.delta_ms // 1000)).tobytes()
            await ws.send(json.dumps({"type": "response.output_audio.delta", "response_id": rid, "item_id": f"item_{rid}",
                                      "delta": base64.b64encode(pcm).decode()}))
            await ws.send(json.dumps({"type": "response.output_audio_transcript.delta", "response_id": rid,
                                      "item_id": f"item_{rid}", "delta": f"w{k} "}))
            await asyncio.sleep(f.gap_s)
        if active == rid:
            active = None
            await ws.send(json.dumps({"type": "response.done", "response": {
                "id": rid, "status": "completed", "usage": {"total_tokens": 10, "output_token_details": {"audio_tokens": 7}}}}))

    async for raw in ws:
        ev = json.loads(raw)
        f.got.append(ev)
        kind = ev["type"]
        if kind == "session.update":
            await ws.send(json.dumps({"type": "session.updated", "session": ev["session"]}))
        elif kind == "input_audio_buffer.append":
            a = array("h")
            a.frombytes(base64.b64decode(ev["audio"]))
            for i in range(0, len(a) - 479, 480):  # 20 ms frames: VAD on the samples received
                pos += 20
                loud = (sum(v * v for v in a[i:i + 480]) / 480) ** 0.5 > 500
                if loud:
                    speech_run, silence_run = speech_run + 20, 0
                    if not speaking and speech_run >= 100:
                        speaking = True
                        await ws.send(json.dumps({"type": "input_audio_buffer.speech_started", "audio_start_ms": pos - 100}))
                        if active is not None:  # interrupt_response
                            rid, active = active, None
                            await ws.send(json.dumps({"type": "response.done", "response": {"id": rid, "status": "cancelled"}}))
                else:
                    speech_run = 0
                    if speaking:
                        silence_run += 20
                        if silence_run >= 500:
                            speaking, silence_run = False, 0
                            await ws.send(json.dumps({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": pos - 500}))
                            n += 1
                            active = f"r{n}"
                            asyncio.ensure_future(respond(active))
        elif kind == "conversation.item.retrieve":  # the adapter's barrier: answered in order, with an error
            await ws.send(json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                      "code": "item_retrieve_invalid_item_id", "event_id": ev.get("event_id")}}))
        elif kind == "response.create":
            n += 1
            active = f"r{n}"
            asyncio.ensure_future(respond(active))


def lookup(q: str) -> dict:
    """Look something up."""
    return {"open": "9am"}


def run(f: Fake, turns, stall=None, max_ms=8000, tools=False, **kw):
    """One episode; ``stall(t)`` returns seconds to block the env loop before step ``t`` (the simulated user computing)."""
    async def main():
        async with websockets.serve(lambda ws: fake_realtime(ws, f), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = AgentSpec(chunk_ms=100, audio="user.audio", sr=SR, obs=("user.speech", "tool.result") if tools else ("user.speech",))
            nodes = {"user": ReplayUser(turns, voice=Voice(ToneSpeech(), leveling=None))}
            if tools:
                nodes["tools"] = ToolWorld(FunctionBackend({"lookup": lookup}, initial_state={}))
            env = Env(nodes, spec, max_ms=max_ms, end_idle_ms=800)
            agent = OpenAIRealtimeAgent(spec, url=f"ws://127.0.0.1:{port}/v1/realtime", **kw)
            obs = await env.reset(Task(id="t", scenario={}))
            done = False
            while not (env.truncated or (done and not agent.busy)):
                if stall and stall(env.t):
                    time.sleep(stall(env.t))  # blocks the loop, as a slow user step would
                obs, _, done = await env.step(await agent.act(env.t, obs))
            await agent.close()
            return env, agent

    return asyncio.run(main())


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def agent_turns(env):
    return [t for t in check(episode(env, "e"))["turns"] if t["role"] == "agent"]


def test_streams_paced_audio_and_plays_a_slow_stream_without_gaps():
    f = Fake(latency_s=0.4, deltas=5, delta_ms=200, gap_s=0.3)  # 1 s of speech arriving over 1.5 s
    t0 = time.monotonic()
    env, agent = run(f, [{"t": 0, "text": "what time do you open"}])  # 5 words: ~1.67 s of tone
    wall = time.monotonic() - t0
    upd = f.got[0]
    assert upd["type"] == "session.update" and upd["session"]["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    appends = [e for e in f.got if e["type"] == "input_audio_buffer.append"]
    assert all(len(base64.b64decode(e["audio"])) == 2 * SR // 10 for e in appends)  # 100 ms at 24 kHz each
    (a,) = agent_turns(env)
    assert a["text"].strip() == "w0 w1 w2 w3 w4" and a["end_time"] - a["start_time"] == 1000  # waited out the slow stream: no gap
    # user speech ends ~1667 ms; the VAD needs 500 ms of silence; then ~400 ms latency (one step of slack each way)
    assert 1667 + 500 + 400 - 200 <= a["start_time"] <= 1667 + 500 + 400 + 300
    assert wall >= env.t / 1000 * 0.9  # the microphone ran at real-time pace
    assert agent.usage == {"total_tokens": 10, "output_token.audio_tokens": 7}
    d = agent.describe()
    assert d["provider"] == "openai" and d["clock"] == "wall, audio-paced" and not d["reproducible"]
    (r,) = agent.trace("e")["responses"]
    assert r["status"] == "completed" and r["played_ms"] == 1000 and 300 <= r["first_audio_wall_ms"] <= 700


def test_a_stall_does_not_shorten_the_model_latency():
    f = Fake(latency_s=0.8, deltas=2)
    # the env blocks for 1.5 s once, right after the user's turn ends (as a slow user step would), while the server is
    # already generating: the reply must still start ~0.8 s of simulated time after the server began it
    stalled = []

    def stall(t):
        if f.created_wall and not stalled:
            stalled.append(t)
            return 1.5
        return 0

    env, agent = run(f, [{"t": 0, "text": "what time do you open"}], stall=stall)
    (r,) = agent.trace("e")["responses"]
    assert r["start_t"] - r["created_t"] >= 700  # not ~0 (the stall swallowed in wall time)
    (a,) = agent_turns(env)
    assert a["start_time"] == r["start_t"] or abs(a["start_time"] - r["start_t"]) <= 100


def test_user_speech_cuts_the_reply_and_the_server_keeps_what_was_heard():
    f = Fake(latency_s=0.2, deltas=15, delta_ms=200, gap_s=0.1)  # a 3 s reply, streamed fast
    # the reply starts ~2.4 s; the user barges in at 3.5 s
    env, agent = run(f, [{"t": 0, "text": "what time do you open"}, {"t": 3500, "text": "sorry one more thing"}], max_ms=9000)
    first = agent_turns(env)[0]
    (r1, *_) = agent.trace("e")["responses"]
    assert r1["status"] == "cancelled" and r1["played_ms"] < 3000
    assert first["end_time"] - first["start_time"] == r1["played_ms"]
    assert first["end_time"] <= 3500 + 100 + 200  # stopped within the VAD's onset (100 ms) and a step or two
    trunc = [e for e in f.got if e["type"] == "conversation.item.truncate"]
    assert trunc == [{"type": "conversation.item.truncate", "item_id": "item_r1", "content_index": 0, "audio_end_ms": r1["played_ms"]}]


def test_function_calls_go_to_the_tools_and_their_results_come_back():
    f = Fake(latency_s=0.2, deltas=3, tool=True)
    env, agent = run(f, [{"t": 0, "text": "when do you open"}], tools=True)
    tools = f.got[0]["session"]["tools"]
    assert tools[0]["name"] == "lookup" and tools[0]["type"] == "function"
    out = [e for e in f.got if e["type"] == "conversation.item.create"]
    assert out and out[0]["item"]["type"] == "function_call_output" and out[0]["item"]["call_id"] == "call_1"
    assert "9am" in out[0]["item"]["output"]
    assert any(e["type"] == "response.create" for e in f.got)
    (a,) = agent_turns(env)
    assert a["text"].strip() == "w0 w1 w2"
    calls = [e for e in check(episode(env, "e")).get("events", []) if "tool" in json.dumps(e)]
    assert agent.trace("e")["responses"][0]["tool_calls"] == ["lookup"] or calls


def test_a_hosted_realtime_agent_needs_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIRealtimeAgent(AgentSpec(chunk_ms=100, audio="user.audio", sr=SR))


def test_a_barge_in_is_cut_where_the_server_detected_it_even_behind_a_slow_link():
    # the whole 3 s reply is produced at once and leaves at 0.7x real time: the server's speech_started queues behind it
    turns = [{"t": 0, "text": "what time do you open"}, {"t": 3500, "text": "sorry one more thing"}]
    env, agent = run(Fake(latency_s=0.2, deltas=15, gap_s=0, wire=0.7), turns, max_ms=9000)
    first = agent_turns(env)[0]
    assert first["end_time"] <= 3500 + 100 + 300 and agent.trace("e")["barriers"] > 0  # cut at once, as detected
    lag = [x for x in agent.trace("e")["input_lag"] if x["event"] == "speech_started"]
    assert lag[-1]["handled_t"] - lag[-1]["audio_t"] <= 400
    env, agent = run(Fake(latency_s=0.2, deltas=15, gap_s=0, wire=0.7), turns, max_ms=9000, sync=False)
    assert agent_turns(env)[0]["end_time"] > 3500 + 1000  # without barriers the cut comes late

