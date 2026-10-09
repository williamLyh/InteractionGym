"""Audio MultiChallenge port: loader, sources, grading history (no models, no data)."""

import asyncio
import json

from interaction_gym.audio import Audio
from interaction_gym.benchmarks import audiomc
from interaction_gym.clients import FakeChat
from interaction_gym.user import Utterance


def _root(tmp_path):
    recs = [{"id": f"c{i}", "axis": ax, "rubric": json.dumps(["States X. ", "Mentions Y."]),
             "turns": [{"user": "Hi, I need help.", "assistant": "Sure, with what?", "audio": f"c{i}/u1.wav", "dur_ms": 1000},
                       {"user": "Thanks for asking! Add K4.", "assistant": "", "audio": f"c{i}/u2.wav", "dur_ms": 1000}]}
            for i, ax in enumerate(["VOICE_EDITING", "VOICE_EDITING", "VOICE_EDITING", "SELF_COHERENCE"])]
    (tmp_path / "index.json").write_text(json.dumps(recs))
    for r in recs:
        (tmp_path / r["id"]).mkdir(exist_ok=True)
        for t in r["turns"]:
            Audio.silence(t["dur_ms"]).write_wav(tmp_path / t["audio"])
    return tmp_path


def test_load(tmp_path):
    tasks = audiomc.load(_root(tmp_path))
    assert [t.id for t in tasks] == ["c0", "c1", "c2", "c3"]  # grouped by axis order
    t = tasks[0]
    assert t.criteria["rubric"] == ["States X.", "Mentions Y."] and t.criteria["axis"] == "VOICE_EDITING"
    assert t.scenario["first_turn"]["audio"].endswith("c0/u1.wav") and len(t.scenario["turns"]) == 2
    assert [t.id for t in audiomc.load(_root(tmp_path), per_axis=2)] == ["c0", "c1", "c3"]


def test_sources(tmp_path):
    t = audiomc.load(_root(tmp_path), only=["c0"])[0]
    a = audiomc.ScriptSource()
    assert asyncio.run(a.next(1, t, [])).text == "Thanks for asking! Add K4." and asyncio.run(a.next(2, t, [])) is None
    llm = FakeChat(["Got it. Add K4."])
    b = audiomc.ReplanSource(llm)
    assert asyncio.run(b.next(0, t, [])).text == "Hi, I need help."
    convo = [Utterance("user", "Hi, I need help.", 0, False), Utterance("agent", "Hello! How can I help?", 1, False)]
    turn = asyncio.run(b.next(1, t, convo))
    assert turn.text == "Got it. Add K4." and turn.final
    prompt = llm.calls[0][1]["content"]
    assert "ASSISTANT: Hello! How can I help?" in prompt and '"Thanks for asking! Add K4."' in prompt
    assert asyncio.run(b.next(2, t, convo)) is None


def test_history_and_final_reply():
    ep = {"turns": [{"role": "user", "text": "Hi", "start_time": 0, "end_time": 1000},
                    {"role": "agent", "text": "Hello.", "start_time": 1500, "end_time": 2000},
                    {"role": "agent", "text": "How can I help?", "start_time": 2100, "end_time": 3000},
                    {"role": "user", "text": "Add K4.", "start_time": 4000, "end_time": 5000},
                    {"role": "agent", "text": "Added K4.", "start_time": 5500, "end_time": 6000}]}
    assert audiomc.final_reply(ep) == "Added K4."
    assert audiomc.history(ep) == "User: Hi\n\nAssistant: Hello. How can I help?\n\nUser: Add K4.\n\nAssistant: Added K4."
    p = audiomc.judge_prompt(audiomc.history(ep), "States K4.")
    assert "# Rubric item\nStates K4." in p and "User: Add K4." in p
    ep["turns"] = ep["turns"][:4]
    assert audiomc.history(ep).endswith("User: Add K4.\n\nAssistant: ")
