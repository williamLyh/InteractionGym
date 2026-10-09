"""The vLLM-Omni duplex adapter against a local stand-in server speaking the same protocol."""

import asyncio
import base64
from array import array
import json
from pathlib import Path

import pytest

websockets = pytest.importorskip("websockets")

from interaction_gym import AgentSpec, Env, Frame, Task  # noqa: E402
from interaction_gym.agents.vllm_omni import VllmOmniDuplexAgent  # noqa: E402
from interaction_gym.audio import Audio  # noqa: E402
from interaction_gym.clients import FakeSpeech  # noqa: E402
from interaction_gym.traj import episode  # noqa: E402
from interaction_gym.user import ReplayUser, Voice  # noqa: E402
from tests.test_traj import check  # noqa: E402

SR = 24000
TRACE_VALIDATOR = __import__("jsonschema").Draft202012Validator(json.loads((Path(__file__).parents[1] / "docs/agent_trace.schema.json").read_text()))


def pcm(ms: int) -> str:
    return base64.b64encode(bytes(2 * SR * ms // 1000)).decode()


async def fake_server(ws, reply_after_ms=1500, status="completed", got=None, late_ms=0, unit_ms=1000):
    """Answers once ``reply_after_ms`` of input has arrived: two 500 ms deltas with their text
    (the second ``late_ms`` of input later)."""
    first = json.loads(await ws.recv())  # like the real server: the session exists once it is configured
    assert first["type"] == "session.update"
    if got is not None:
        got.append(first)
    await ws.send(json.dumps({"type": "session.created", "session": {"capabilities": {"chunk_period_ms": unit_ms}}}))
    heard, replied = 0, False
    if late_ms:
        async for raw in ws:
            ev = json.loads(raw)
            if ev["type"] != "input_audio_buffer.append":
                continue
            heard += len(base64.b64decode(ev["audio"])) // 2 * 1000 // ev["sample_rate_hz"]
            for i, (at, text) in enumerate(((reply_after_ms, " Hello"), (reply_after_ms + late_ms, " there."))):
                if heard == at:
                    await ws.send(json.dumps({"type": "response.output_audio.delta", "response_id": "r1", "delta": pcm(500)}))
                    await ws.send(json.dumps({"type": "response.output_audio_transcript.delta", "response_id": "r1", "delta": text}))
                    if i == 1:
                        await ws.send(json.dumps({"type": "response.done", "response": {"id": "r1", "status": "completed"}}))
        return
    async for raw in ws:
        ev = json.loads(raw)
        if got is not None:
            got.append(ev)
        if ev["type"] == "input_audio_buffer.append":
            heard += len(base64.b64decode(ev["audio"])) // 2 * 1000 // ev["sample_rate_hz"]
            if heard >= reply_after_ms and not replied:
                replied = True
                for text in (" Hello", " there."):
                    await ws.send(json.dumps({"type": "response.output_audio.delta", "response_id": "r1", "delta": pcm(500)}))
                    await ws.send(json.dumps({"type": "response.output_audio_transcript.delta", "response_id": "r1", "delta": text}))
                if status == "completed":
                    await ws.send(json.dumps({"type": "response.done", "response": {"id": "r1", "status": status}}))
            if heard >= reply_after_ms + 300 and status != "completed" and replied != "done":
                replied = "done"  # stopped while speaking, e.g. cancelled
                await ws.send(json.dumps({"type": "response.done", "response": {"id": "r1", "status": status}}))


def run(status="completed", late_ms=0, end_idle_ms=500, prebuffer_ms=1000):
    async def main():
        got = []
        async with websockets.serve(lambda ws: fake_server(ws, status=status, got=got, late_ms=late_ms), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = AgentSpec(chunk_ms=100, audio="user.audio", sr=SR)
            user = ReplayUser([{"t": 0, "text": "hi how are you"}], voice=Voice(FakeSpeech(sr=SR)))
            env = Env({"user": user}, spec, max_ms=10_000, end_idle_ms=end_idle_ms)
            agent = VllmOmniDuplexAgent(spec, f"ws://127.0.0.1:{port}/v1/realtime?duplex=1", ref_audio=__file__,
                                        session={"temperature": 0.0}, prebuffer_ms=prebuffer_ms)
            obs = await env.reset(Task(id="t", scenario={}))
            done = False
            while not done:
                obs, _, done = await env.step(await agent.act(env.t, obs))
            await agent.close()
            return env, got

    return asyncio.run(main())


def test_streams_the_microphone_and_plays_the_reply():
    env, got = run()
    first = got[0]
    assert first["type"] == "session.update" and first["session"]["turn_detection"] is None and first["session"]["temperature"] == 0.0
    appends = [e for e in got if e["type"] == "input_audio_buffer.append"]
    assert {e["sample_rate_hz"] for e in appends} == {16000}
    assert all(len(base64.b64decode(e["audio"])) == 2 * 16000 // 10 for e in appends)  # 100 ms each, resampled
    ep = check(episode(env, "e"))
    (a,) = [t for t in ep["turns"] if t["role"] == "agent"]
    assert a["text"] == "Hello there." and a["end_time"] - a["start_time"] == 1000
    assert a["start_time"] >= 1500  # not before the server had heard enough to answer
    assert not env.truncated


def test_a_response_that_ends_early_drops_what_has_not_played():
    env, _ = run(status="cancelled")
    (a,) = [t for t in check(episode(env, "e"))["turns"] if t["role"] == "agent"]
    # 300 ms of the 500 ms " Hello" delta played before the cancel: round(6 * 300 / 500) = 4 chars were said
    assert a["end_time"] - a["start_time"] == 300 and a["text"] == "Hel"


def test_a_reply_that_arrives_in_pieces_is_one_utterance_with_a_pause():
    env, _ = run(late_ms=1500, end_idle_ms=2500, prebuffer_ms=200)  # starts before the rest has arrived
    (a,) = [t for t in check(episode(env, "e"))["turns"] if t["role"] == "agent"]
    assert a["text"] == "Hello there." and a["start_time"] == 1500 and a["end_time"] == 3500  # 1 s of speech + the wait
    env, _ = run(late_ms=1500, end_idle_ms=2500)  # the default jitter buffer waits for it: no pause
    (a,) = [t for t in check(episode(env, "e"))["turns"] if t["role"] == "agent"]
    assert a["end_time"] - a["start_time"] == 1000 and not env.truncated


async def fake_lockstep_server(ws, reply_after_ms=1500, unit_ms=1000):
    """The lockstep extension: input is cut into units; each finished unit may take a while to compute
    (random delay); every append is acknowledged with ``input_audio_buffer.processed`` after all the
    output it caused."""
    import random

    rng = random.Random()  # deliberately unseeded: compute time must not change the outcome
    first = json.loads(await ws.recv())
    assert first["type"] == "session.update" and first["session"]["extra_body"]["clock"] == "input"
    trace = first["session"]["extra_body"].get("trace_tokens")
    await ws.send(json.dumps({"type": "session.created", "session": {"capabilities": {"chunk_period_ms": unit_ms}}}))
    heard = units = 0
    replied = False
    async for raw in ws:
        ev = json.loads(raw)
        if ev["type"] != "input_audio_buffer.append":
            continue
        heard += len(base64.b64decode(ev["audio"])) // 2 * 1000 // ev["sample_rate_hz"]
        while (units + 1) * unit_ms <= heard:  # a unit completes: the model decides
            units += 1
            await asyncio.sleep(rng.uniform(0, 0.05))
            speak = units * unit_ms >= reply_after_ms and not replied
            if trace:  # what the model consumed / produced in this unit
                out = [[9, "<|speak|>"], [42, "Hello"]] if speak else [[8, "<|listen|>"]]
                await ws.send(json.dumps({"type": "debug.unit_tokens", "unit_index": units - 1, "end_ms": units * unit_ms,
                                          "decision": "speak" if speak else "listen", "special_ids": [7, 8, 9],
                                          "stages": [{"stage": "thinker", "input": [[7, "<unit>"]] + [[0, "<unk>"]] * 25, "output": out}]}))
            if speak:
                replied = True
                for text in (" Hello", " there."):
                    await ws.send(json.dumps({"type": "response.output_audio.delta", "response_id": "r1", "delta": pcm(500)}))
                    await ws.send(json.dumps({"type": "response.output_audio_transcript.delta", "response_id": "r1", "delta": text}))
                await ws.send(json.dumps({"type": "response.done", "response": {"id": "r1", "status": "completed"}}))
        await ws.send(json.dumps({"type": "input_audio_buffer.processed", "audio_end_ms": heard, "unit_end_ms": units * unit_ms}))


def run_lockstep(trace=False):
    import time

    async def main():
        async with websockets.serve(fake_lockstep_server, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = AgentSpec(chunk_ms=100, audio="user.audio", sr=SR)
            user = ReplayUser([{"t": 0, "text": "hi how are you"}], voice=Voice(FakeSpeech(sr=SR)))
            env = Env({"user": user}, spec, max_ms=20_000, end_idle_ms=3000)
            agent = VllmOmniDuplexAgent(spec, f"ws://127.0.0.1:{port}/v1/realtime?duplex=1", ref_audio=__file__, clock="input", trace_tokens=trace)
            obs = await env.reset(Task(id="t", scenario={}))
            wall = time.monotonic()
            done = False
            while not done:
                obs, _, done = await env.step(await agent.act(env.t, obs))
            wall = time.monotonic() - wall
            await agent.close()
            return env, wall, agent

    return asyncio.run(main())


def test_lockstep_runs_on_input_time_and_is_reproducible():
    runs = [run_lockstep() for _ in range(2)]
    turns = [[(t["role"], t["start_time"], t["end_time"], t["text"]) for t in check(episode(env, "e"))["turns"]] for env, _, _ in runs]
    assert turns[0] == turns[1]  # random compute delays change nothing
    agent_turns = [t for t in turns[0] if t[0] == "agent"]
    assert agent_turns == [("agent", 2000, 3000, "Hello there.")]  # spoken right when unit 2 (ending at 2 s) was decided
    for env, wall, agent in runs:
        assert agent.acks == agent.appends and wall < env.t / 1000 / 2  # far faster than real time


def test_the_step_must_divide_the_model_unit():
    async def main(step, unit):
        async with websockets.serve(lambda ws: fake_server(ws, unit_ms=unit), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = AgentSpec(chunk_ms=step, audio="user.audio", sr=SR)
            agent = VllmOmniDuplexAgent(spec, f"ws://127.0.0.1:{port}/v1/realtime?duplex=1", ref_audio=__file__)
            try:
                await agent.act(0, [])
                return agent.unit_ms
            finally:
                await agent.close()

    assert asyncio.run(main(200, 1000)) == 1000 and asyncio.run(main(80, 1040)) == 1040
    for step, unit in ((300, 1000), (2000, 1000)):  # boundaries would drift / an append would carry 2 units
        with pytest.raises(ValueError, match="does not divide"):
            asyncio.run(main(step, unit))


def test_token_trace_is_recorded_into_the_episode_and_rendered():
    from interaction_gym.viewer import export_agent_trace_html, render_html

    env, _, agent = run_lockstep(trace=True)
    ep = check(episode(env, "e"))
    assert "agent_trace" not in ep  # chunk-level internals are not part of the trajectory
    tr = agent.trace("e")
    assert not list(TRACE_VALIDATOR.iter_errors(tr)) and tr["episode_id"] == "e"
    assert tr["unit_ms"] == 1000 and tr["clock"] == "input"
    assert [u["end_ms"] for u in tr["units"]] == [1000 * (i + 1) for i in range(len(tr["units"]))]
    assert [u["decision"] for u in tr["units"][:3]] == ["listen", "speak", "listen"]
    assert "Tokens per unit" not in render_html([ep])  # not in the env viewer
    import tempfile
    page = export_agent_trace_html(ep, tr, Path(tempfile.mkdtemp()) / "e.agent.html").read_text()
    assert "Tokens per unit" in page and "<|speak|>" in page


async def fake_text_server(ws, reply_after_ms=1500):
    """Text-only output: the server sends the reply's text and no audio."""
    first = json.loads(await ws.recv())
    assert first["session"]["modalities"] == ["text"] and "ref_audio" not in first["session"]
    await ws.send(json.dumps({"type": "session.created", "session": {"capabilities": {"chunk_period_ms": 1000}}}))
    heard, replied = 0, False
    async for raw in ws:
        ev = json.loads(raw)
        if ev["type"] != "input_audio_buffer.append":
            continue
        heard += len(base64.b64decode(ev["audio"])) // 2 * 1000 // ev["sample_rate_hz"]
        if heard >= reply_after_ms and not replied:
            replied = True
            for text in (" Hello", " there, nice to meet you."):
                await ws.send(json.dumps({"type": "response.output_text.delta", "response_id": "r1", "delta": text}))
            await ws.send(json.dumps({"type": "response.done", "response": {"id": "r1", "status": "completed"}}))
        await ws.send(json.dumps({"type": "input_audio_buffer.processed", "audio_end_ms": heard}))


def test_a_realtime_session_is_not_traced():
    # the server's token trace needs the input clock (it refuses trace_tokens without clock="input")
    spec = AgentSpec(chunk_ms=100, audio="user.audio", sr=SR)
    with pytest.warns(UserWarning, match="trace_tokens needs clock='input'"):
        agent = VllmOmniDuplexAgent(spec, "ws://127.0.0.1:1/v1/realtime?duplex=1", clock="realtime", trace_tokens=True)
    assert not agent.trace_tokens and agent.trace("e") is None
    assert VllmOmniDuplexAgent(spec, "ws://127.0.0.1:1/v1/realtime?duplex=1", clock="input", trace_tokens=True).trace_tokens


def test_text_only_output_is_timed_by_the_speaking_rate():
    async def main():
        async with websockets.serve(fake_text_server, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = AgentSpec(chunk_ms=100, audio="user.audio", sr=SR)
            user = ReplayUser([{"t": 0, "text": "hi how are you"}], voice=Voice(FakeSpeech(sr=SR)))
            env = Env({"user": user}, spec, max_ms=10_000, end_idle_ms=1000)
            agent = VllmOmniDuplexAgent(spec, f"ws://127.0.0.1:{port}/v1/realtime?duplex=1", clock="input",
                                        audio_out=False, speech_cps=10.0)
            obs = await env.reset(Task(id="t", scenario={}))
            done = False
            while not done:
                obs, _, done = await env.step(await agent.act(env.t, obs))
            await agent.close()
            return env

    ep = check(episode(asyncio.run(main()), "e"))
    (a,) = [t for t in ep["turns"] if t["role"] == "agent"]
    text = "Hello there, nice to meet you."
    assert a["text"] == text and "media" not in a
    assert abs((a["end_time"] - a["start_time"]) - len(" Hello" + " there, nice to meet you.") / 10.0 * 1000) <= 100  # one step of rounding


def test_speech_rate_calibrates_from_complete_turns_with_audio():
    from interaction_gym.agents import speech_rate

    ep = {"turns": [{"role": "agent", "text": "x" * 20, "start_time": 0, "end_time": 2000, "media": {}},
                    {"role": "agent", "text": "y" * 99, "start_time": 0, "end_time": 100, "media": {}, "unsaid": "z"},  # cut: skipped
                    {"role": "user", "text": "u" * 50, "start_time": 0, "end_time": 1000, "media": {}}]}
    assert speech_rate([ep]) == 10.0
    with pytest.raises(ValueError):
        speech_rate([{"turns": []}])


def pcm_tone(ms: int, level: int = 3000) -> str:
    return base64.b64encode(array("h", [level, -level] * (SR * ms // 2000)).tobytes()).decode()


async def scripted_server(ws, script, *, unit_ms=1000, acks=True, got=None):
    """Replays ``script``: {number of appends received: [events]}; acks every append (lockstep)."""
    first = json.loads(await ws.recv())
    if got is not None:
        got.append(first)
    await ws.send(json.dumps({"type": "session.created", "session": {"capabilities": {"chunk_period_ms": unit_ms}}}))
    n = 0
    async for raw in ws:
        ev = json.loads(raw)
        if got is not None:
            got.append(ev)
        if ev["type"] == "input_audio_buffer.commit" and acks:  # the newer server acknowledges commits too
            await ws.send(json.dumps({"type": "input_audio_buffer.processed", "trigger": "input_audio_buffer.commit"}))
        if ev["type"] != "input_audio_buffer.append":
            continue
        n += 1
        for out in script.get(n, []):
            await ws.send(json.dumps(out))
        if acks:
            await ws.send(json.dumps({"type": "input_audio_buffer.processed", "trigger": "input_audio_buffer.append"}))


def scripted_episode(script, user_turns=None, **agent_kw):
    got = []

    async def main():
        async with websockets.serve(lambda ws: scripted_server(ws, script, got=got), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = AgentSpec(chunk_ms=100, audio="user.audio", sr=SR)
            user = ReplayUser(user_turns or [{"t": 0, "text": "hi there"}], voice=Voice(FakeSpeech(sr=SR)))
            env = Env({"user": user}, spec, max_ms=8000, end_idle_ms=1500)
            agent = VllmOmniDuplexAgent(spec, f"ws://127.0.0.1:{port}/v1/realtime?duplex=1", clock="input", **agent_kw)
            obs = await env.reset(Task(id="t", scenario={}))
            done = False
            while not done:
                obs, _, done = await env.step(await agent.act(env.t, obs))
            await agent.close()
            return env, agent

    env, agent = asyncio.run(main())
    return check(episode(env, "e")), agent, got


def test_text_that_arrives_before_its_audio_is_kept():
    r = "r1"
    ep, _, _ = scripted_episode({10: [{"type": "response.output_audio_transcript.delta", "response_id": r, "delta": " Sure thing."},
                                      {"type": "response.output_audio.delta", "response_id": r, "delta": pcm(500)},
                                      {"type": "response.done", "response": {"id": r, "status": "completed"}}]})
    (a,) = [t for t in ep["turns"] if t["role"] == "agent"]
    assert a["text"] == "Sure thing." and a["start_time"] == 1000  # reply to the 10th append (900–1000 ms), played from t=1000


def test_client_side_turn_detection_commits_after_the_user_goes_quiet():
    loud = Audio(array("h", [4000, -4000] * (SR // 2)), SR)  # 1 s of "speech"
    ep, agent, got = scripted_episode({}, user_turns=[{"t": 0, "text": "hello there", "audio": loud}],
                                      turn_detection={"type": "server_vad"}, commit_after_silence_ms=600)
    first = got[0]["session"]
    assert first["turn_detection"] == {"type": "server_vad"} and "auto_response" not in first["extra_body"]
    commits = [i for i, e in enumerate(got) if e["type"] == "input_audio_buffer.commit"]
    appends_before = sum(1 for e in got[: commits[0]] if e["type"] == "input_audio_buffer.append")
    assert len(commits) == 1 and appends_before == 10 + 6  # 1 s of speech, then 600 ms of quiet (100 ms steps)
    assert agent.commits == 1 and agent.acks == agent.appends  # commit acks are not counted as append acks


def test_near_silent_output_within_a_response_splits_utterances():
    r = "r1"
    quiet = base64.b64encode(array("h", [20] * (SR // 10)).tobytes()).decode()  # 100 ms of near-silence
    script = {10: [{"type": "response.output_audio.delta", "response_id": r, "delta": pcm_tone(500)},
                   {"type": "response.output_audio_transcript.delta", "response_id": r, "delta": " One."}]}
    for k in range(11, 31):  # 2 s of near-silence, still the same response
        script[k] = [{"type": "response.output_audio.delta", "response_id": r, "delta": quiet}]
    script[31] = [{"type": "response.output_audio.delta", "response_id": r, "delta": pcm_tone(500)},
                  {"type": "response.output_audio_transcript.delta", "response_id": r, "delta": " Two."},
                  {"type": "response.done", "response": {"id": r, "status": "completed"}}]
    ep, _, _ = scripted_episode(script, split_silence_ms=800)
    agent_turns = [t for t in ep["turns"] if t["role"] == "agent"]
    assert [t["text"] for t in agent_turns] == ["One.", "Two."]  # one response, two utterances
    assert agent_turns[0]["end_time"] - agent_turns[0]["start_time"] < 1500  # not 2.5 s of mostly silence


# ---- lockstep: refused inputs are resent ------------------------------------------------------------

async def refusing_server(ws, refuse, got):
    """Lockstep server that refuses chosen attempts. ``refuse(n, ev)`` sees the n-th input received (resends
    included) and returns ``None`` (accept), ``("rejected", reason)`` (error, then an acknowledgement with
    ``decision: "rejected"``), ``("transport", code)`` (only an error: no acknowledgement) or ``("handled", code)``
    (error, then a plain acknowledgement)."""
    first = json.loads(await ws.recv())
    await ws.send(json.dumps({"type": "session.created", "session": {"capabilities": {"chunk_period_ms": 1000}}}))
    n = index = 0
    async for raw in ws:
        ev = json.loads(raw)
        if ev["type"] not in ("input_audio_buffer.append", "input_audio_buffer.commit"):
            continue
        n += 1
        got.append(ev)
        verdict = refuse(n, ev)
        if verdict is not None:
            await ws.send(json.dumps({"type": "error", "error": {"type": "invalid_request_error", "code": verdict[1],
                                                                 "message": "refused", "event_id": ev["event_id"]}}))
            if verdict[0] == "transport":
                continue
        index += 1
        ack = {"type": "input_audio_buffer.processed", "input_index": index, "trigger": ev["type"], "units": []}
        if verdict is not None and verdict[0] == "rejected":
            ack |= {"decision": "rejected", "reason": verdict[1]}
        await ws.send(json.dumps(ack))
    assert first["session"]["extra_body"]["clock"] == "input"


def refusing_episode(refuse, steps=10, **agent_kw):
    got: list[dict] = []

    async def main():
        async with websockets.serve(lambda ws: refusing_server(ws, refuse, got), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = AgentSpec(chunk_ms=100, audio="user.audio", sr=SR)
            agent = VllmOmniDuplexAgent(spec, f"ws://127.0.0.1:{port}/v1/realtime?duplex=1", clock="input",
                                        retry_backoff_s=0.001, retry_backoff_max_s=0.004, **agent_kw)
            try:
                for i in range(steps):
                    tone = Audio(array("h", [i + 1] * (SR // 10)), SR)  # each step's audio is distinguishable
                    await agent.act(i * 100, [Frame("user.audio", i * 100, tone, dur=100)])
            finally:
                await agent.close()
            return agent

    return asyncio.run(main()), got


def _levels(got):
    return [array("h", base64.b64decode(e["audio"]))[0] for e in got if e["type"] == "input_audio_buffer.append"]


def test_a_rejected_append_is_resent_before_anything_later():
    agent, got = refusing_episode(lambda n, ev: ("rejected", "input_backpressure") if n in (3, 4) else None)
    assert _levels(got)[:6] == [1, 2, 3, 3, 3, 4]  # the 3rd step's audio, twice refused, resent until heard
    assert len({e["event_id"] for e in got}) == len(got), "every attempt has its own event_id"
    assert agent.acks == agent.appends == 10 and agent.input_stats["input_retries"] == 2
    assert agent.input_stats["dropped_inputs"] == [] and agent.failed is None


def test_a_transport_error_without_acknowledgement_is_resent():
    agent, got = refusing_episode(lambda n, ev: ("transport", "engine_backpressure") if n == 2 else None)
    assert _levels(got)[:4] == [1, 2, 2, 3]
    assert agent.acks == agent.appends == 10 and agent.input_stats["input_retries"] == 1


def test_an_input_still_refused_after_the_retry_limit_fails_the_step():
    from interaction_gym.agents.vllm_omni import InputDroppedError

    with pytest.raises(InputDroppedError, match="input_backpressure"):
        refusing_episode(lambda n, ev: ("rejected", "input_backpressure") if n >= 3 else None, max_input_retries=2)


def test_the_retry_limit_can_mark_the_episode_failed_and_go_on():
    agent, got = refusing_episode(lambda n, ev: ("rejected", "input_backpressure") if n in (3, 4, 5) else None,
                                  max_input_retries=2, on_input_dropped="mark")
    assert _levels(got)[:6] == [1, 2, 3, 3, 3, 4]  # 1 send + 2 resends, then the next step goes on
    assert agent.input_stats["dropped_inputs"] == [{"t": 200, "type": "input_audio_buffer.append",
                                                    "reason": "input_backpressure", "attempts": 3}]
    assert agent.input_stats["input_retries"] == 2 and "input_backpressure" in agent.failed
    assert agent.acks == 9 and agent.appends == 10


def test_a_refusal_a_resend_cannot_fix_is_not_retried():
    for layer in ("transport", "rejected"):  # the serving layer (no acknowledgement) or the session (rejected)
        agent, got = refusing_episode(lambda n, ev: (layer, "bad_audio") if n == 2 else None, on_input_dropped="mark",
                                      ack_grace_s=0.05)
        assert _levels(got)[:3] == [1, 2, 3]
        assert agent.input_stats["dropped_inputs"] == [{"t": 100, "type": "input_audio_buffer.append",
                                                        "reason": "bad_audio", "attempts": 1}]
        assert agent.input_stats["input_retries"] == 0 and agent.acks == 9, "every later ack matched its own input"


def test_an_input_rejected_while_handled_is_acknowledged_and_not_resent():
    agent, got = refusing_episode(lambda n, ev: ("handled", "input_audio_buffer_empty") if n == 2 else None)
    assert _levels(got)[:3] == [1, 2, 3] and agent.input_stats["input_retries"] == 0
    assert agent.input_stats["input_errors"] == [{"type": "input_audio_buffer.append", "code": "input_audio_buffer_empty"}]


def test_resends_back_off_exponentially_up_to_a_cap():
    spec = AgentSpec(chunk_ms=100, audio="user.audio", sr=SR)
    agent = VllmOmniDuplexAgent(spec, clock="input")
    assert agent.max_input_retries == 5
    assert [agent.backoff_s(a) for a in range(1, 7)] == [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]


def test_a_session_the_server_refuses_fails_at_once():
    """E.g. ``duplex_session_capacity_exhausted``: the error ends the connect, not a 30 s wait."""

    async def full_server(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "error", "error": {"type": "server_error", "message": "full",
                                                              "code": "duplex_session_capacity_exhausted"}}))
        await ws.close()

    async def main():
        async with websockets.serve(full_server, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            agent = VllmOmniDuplexAgent(AgentSpec(chunk_ms=100, audio="user.audio", sr=SR),
                                        f"ws://127.0.0.1:{port}/v1/realtime?duplex=1", clock="input")
            loop = asyncio.get_running_loop()
            started = loop.time()
            with pytest.raises(RuntimeError, match="duplex_session_capacity_exhausted"):
                await agent.act(0, [Frame("user.audio", 0, Audio(array("h", [0] * (SR // 10)), SR), dur=100)])
            assert loop.time() - started < 5
            await agent.close()

    asyncio.run(main())
