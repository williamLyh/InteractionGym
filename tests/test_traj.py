"""The standard output format (schema v1): episodes from every example validate, and the
fields carry what docs/FORMAT.md says they do."""

import asyncio
import json
from pathlib import Path

import jsonschema
import pytest

import examples.minimal as minimal
import examples.tools_demo as tools_demo
import examples.user_modes as um
from interaction_gym import AgentSpec, Background, Env, Task
from interaction_gym.audio import Audio
from interaction_gym.eval import duplex
from interaction_gym.media import MediaStore
from interaction_gym.traj import _turns, agent_view, episode, load, save, to_frames
from interaction_gym.user import ReplayUser
from interaction_gym.viewer import render_html

SCHEMA = json.loads((Path(__file__).parents[1] / "docs/trajectory.schema.json").read_text())
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


def run(coro):
    return asyncio.run(coro)


def check(ep):
    """Schema + the invariants JSON Schema cannot express."""
    errors = [f"{list(e.path)}: {e.message}" for e in VALIDATOR.iter_errors(ep)]
    assert not errors, errors
    ids = [t["id"] for t in ep["turns"]]
    assert len(ids) == len(set(ids))
    assert [t["start_time"] for t in ep["turns"]] == sorted(t["start_time"] for t in ep["turns"])
    for t in ep["turns"]:
        assert 0 <= t["start_time"] <= t["end_time"] <= ep["meta"]["duration_ms"]
        if "media" in t:
            assert t["media"]["end_ms"] - t["media"]["start_ms"] == t["end_time"] - t["start_time"]
    for key, d in ep["eval"]["duplex"].items():
        assert key in ids and d.get("target", ids[0]) in ids
    return ep


def episodes_from_examples(tmp):
    media = MediaStore(tmp)
    out = []
    for name, kw in minimal.VARIANTS.items():
        out.append(episode(run(minimal.run(**kw)), f"minimal/{name}", media=media))
    for mode, streaming in (("offline", False), ("script", False), ("online", False), ("online", True)):
        out.append(episode(run(um.run(mode, streaming=streaming)), f"modes/{mode}/{streaming}", media=media))
    for fn in (tools_demo.slow_tool_with_filler, tools_demo.user_talks_while_waiting, tools_demo.tool_error_and_retry):
        env, meta = run(fn())
        out.append(episode(env, f"tools/{fn.__name__}", media=media, meta=meta))
    return out


def test_every_example_episode_matches_the_schema(tmp_path):
    for ep in episodes_from_examples(tmp_path):
        check(ep)


def test_kind_comes_from_the_producer(tmp_path):
    ep = check(episode(run(um.run("offline")), "e", media=MediaStore(tmp_path)))
    kinds = {t["text"] or "(noise)": t.get("kind") for t in ep["turns"] if t["role"] == "user"}
    assert kinds["mm-hmm"] == "backchannel" and kinds["hold on, I'm on the phone"] == "aside" and kinds["(noise)"] == "noise"
    assert kinds["Two people, around seven please."] is None  # a normal turn carries no kind


def test_cut_turns_keep_what_was_said_and_unsaid():
    ep = check(episode(run(minimal.run()), "e"))
    a0 = next(t for t in ep["turns"] if t["id"] == "a0")
    assert a0["text"] + a0["unsaid"] == minimal.REPLIES[0]
    assert ep["eval"]["duplex"]["u1"] == {"behavior": "barge_in", "target": "a0", "reaction": "yielded", "latency_ms": a0["end_time"] - 3100}


def test_streamed_agent_turns_have_no_unsaid_but_still_yield():
    ep = check(episode(run(minimal.run(streaming=True)), "e"))
    assert not any("unsaid" in t for t in ep["turns"] if t["role"] == "agent")
    (d,) = ep["eval"]["duplex"].values()
    assert d["behavior"] == "barge_in" and d["reaction"] == "yielded"


def T(id, role, a, b, **kw):
    return {"id": id, "role": role, "start_time": a, "end_time": b, "text": "x", **kw}


def test_duplex_rules():
    turns = [
        T("a0", "agent", 0, 5000),
        T("u0", "user", 1000, 1300, kind="backchannel"),  # agent keeps talking -> continued
        T("u1", "user", 2000, 3000),  # normal turn over the agent that never stops -> kept_talking
        T("u2", "user", 6000, 7000, kind="aside"),  # nobody speaking, agent silent afterwards -> stayed_silent
        T("u3", "user", 9000, 9500, kind="noise"),
        T("a1", "agent", 10000, 11000),  # starts 500 ms after the noise -> responded
        T("u4", "user", 12000, 15000),
        T("a2", "agent", 13000, 14000),  # starts during a normal user turn -> agent_interrupt
        T("a3", "agent", 16000, 18000, unsaid="..."),
        T("u5", "user", 17000, 17300, kind="backchannel"),  # agent cut right after -> stopped
    ]
    turns[-2]["end_time"] = 17500
    d = duplex(turns)
    assert d["u0"] == {"behavior": "backchannel", "target": "a0", "reaction": "continued"}
    assert d["u1"] == {"behavior": "barge_in", "target": "a0", "reaction": "kept_talking"}
    assert d["u2"] == {"behavior": "aside", "reaction": "stayed_silent"}
    assert d["u3"] == {"behavior": "noise", "reaction": "responded"}
    assert d["a2"] == {"behavior": "agent_interrupt", "target": "u4"}
    assert d["u5"] == {"behavior": "backchannel", "target": "a3", "reaction": "stopped"}
    assert list(d) == ["u0", "u1", "u2", "u3", "a2", "u5"]  # ordered by time


def test_media_is_shared_deduplicated_and_sliced(tmp_path):
    store = MediaStore(tmp_path)
    audio = Audio.silence(1000)
    r1, r2 = store.ref(audio), store.ref(audio, 0, 400)
    assert r1["uri"] == r2["uri"] and len(list(tmp_path.rglob("*.wav"))) == 1
    assert store.load(r2).dur_ms == 400
    # a cut user turn references the start of its full audio
    ep = check(episode(run(um.run("offline")), "e", media=store))
    for t in ep["turns"]:
        if "media" in t:
            assert t["media"]["start_ms"] == 0 and (tmp_path / t["media"]["uri"]).exists()


def test_background_track_and_continuous_microphone(tmp_path):
    hum = Audio.silence(5000)
    hum.samples[:] = type(hum.samples)("h", [1000] * len(hum.samples))
    spec = AgentSpec(chunk_ms=200, audio="user.audio")
    env = Env({"user": ReplayUser([{"t": 1000, "text": "hello there"}])}, spec, background=[Background(hum, gain_db=-6, offset_ms=500)])

    async def main():
        mics = []
        obs = await env.reset(Task())
        while not env.done:
            obs, _, _ = await env.step([])
            mics += [f for f in obs if f.stream == "user.audio"]
        return mics

    mics = run(main())
    assert all(f.dur == 200 and f.data.dur_ms == 200 for f in mics)  # one mixed frame per step, even in silence
    assert all(max(f.data.samples) > 0 for f in mics)  # the ambience is always there
    ep = check(episode(env, "e", media=MediaStore(tmp_path)))
    (bg,) = ep["background"]
    assert bg["gain_db"] == -6 and bg["media"]["start_ms"] == 500 


def test_round_trip_to_runtime_frames():
    for env in (run(minimal.run()), run(tools_demo.tool_error_and_retry())[0]):
        ep = episode(env, "e")
        frames = to_frames(ep)
        rebuilt = _turns(frames, None, ep["meta"]["duration_ms"])
        speech = [{k: v for k, v in t.items() if k != "tool_calls"} for t in ep["turns"] if t["end_time"] > t["start_time"] or t["text"]]
        assert sorted(rebuilt, key=lambda t: t["start_time"]) == speech
        calls = [f.data.id for f in frames if f.stream.endswith("tool_call")]
        assert calls == [c["id"] for t in ep["turns"] for c in t.get("tool_calls", [])]


def test_agent_view_replays_what_was_heard():
    ep = episode(run(minimal.run(chunk_ms=200)), "e")
    steps = agent_view(ep, 200)
    assert len(steps) == -(-ep["meta"]["duration_ms"] // 200)
    for t in (t for t in ep["turns"] if t["role"] == "user"):
        assert "".join(h["text"] for s in steps for h in s["heard"] if h["id"] == t["id"]) == t["text"]


def test_save_load_and_viewer(tmp_path):
    eps = [episode(run(minimal.run()), "a"), episode(run(minimal.run(chunk_ms=80)), "b")]
    path = save(eps, tmp_path / "run" / "episodes.jsonl")
    assert len(path.read_text().splitlines()) == 2 and load(path) == json.loads(json.dumps(eps))
    html = render_html(eps, "media/")
    assert "__DATA__" not in html and "__MEDIA_BASE__" not in html and '"episode_id": "a"' in html


def test_tau_episode_carries_official_reward(tmp_path):
    pytest.importorskip("tau2")
    from interaction_gym.integrations import tau

    env = run(tools_demo.run(*_tau_case()))
    task = tau.load_task("mock", "create_task_1")
    res = tau.evaluate(env.log, task)
    ep = check(episode(env, "tau", media=MediaStore(tmp_path), reward={"total": res["reward"], "parts": res["breakdown"]}))
    assert ep["eval"]["reward"] == {"total": 1.0, "parts": res["breakdown"]}
    assert ep["meta"]["task"]["scenario"]["tau_id"] == "create_task_1"


def _tau_case():
    from interaction_gym.integrations import tau
    from interaction_gym.tools import ToolWorld

    task = tau.load_task("mock", "create_task_1")
    user = tools_demo.script_user("Hi, I need some help.", "Great, thanks!")
    return {"user": user, "tools": ToolWorld(tau.TauBackend("mock"))}, task, tau.OracleAgent(task, tools_demo.SPEC)


def test_eval_tells_a_naive_agent_from_one_that_listens():
    naive = episode(run(um.run("offline")), "naive")["eval"]["duplex"]
    listening = episode(run(um.run("offline", min_words=2)), "listening")["eval"]["duplex"]
    by_kind = lambda d: {v["behavior"]: v["reaction"] for v in d.values() if v["behavior"] != "barge_in"}  # noqa: E731
    assert by_kind(naive)["noise"] == "stopped" and by_kind(naive)["backchannel"] == "stopped"
    assert by_kind(listening)["noise"] == "continued" and by_kind(listening)["backchannel"] == "continued"
    assert by_kind(listening)["aside"] == "stopped"  # word counts cannot tell talking-to-someone-else apart
    assert all(v["reaction"] == "yielded" for v in listening.values() if v["behavior"] == "barge_in")
