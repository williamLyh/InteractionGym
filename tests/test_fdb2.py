"""Full-Duplex-Bench v2 port: loader, examiner source, static script, judge prompt / parser (no models, no data)."""

import asyncio
import json

from interaction_gym.benchmarks import fdb2
from interaction_gym.clients import FakeChat
from interaction_gym.user import Utterance


def _data(tmp_path):
    task = {"id": "Correction.x.001", "split": "Correction", "class_id": "x", "scenario_title": "t",
            "examiner_system_prompt": "Act like a customer {not a format field}. ", "examiner_task_prompt": "Order a pizza.",
            "staged_reveal": {"T1": "ask", "T2": "size", "T3": "change size", "T4": "confirm"}, "skills_tested": ["correction"]}
    data = {"settings": {"turn_limit": 5}, "splits": {s: {"classes": [], "tasks": [dict(task, id=f"{s}.x.001", split=s)]} for s in fdb2.SPLITS}}
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps(data))
    return p


def test_load(tmp_path):
    tasks = fdb2.load(_data(tmp_path))
    assert [t.id for t in tasks] == [f"{s}.x.001" for s in fdb2.SPLITS]
    t = tasks[1]
    assert t.criteria["split"] == "Correction" and t.criteria["staged_reveal"]["T3"] == "change size"
    assert t.scenario["instructions"].startswith("Act like a customer {not a format field}. Do not talk over the other speaker.")
    assert fdb2.load(_data(tmp_path), splits=["Safety"])[0].id == "Safety.x.001"


def test_examiner_source_end_phrase_and_first_turn(tmp_path):
    t = fdb2.load(_data(tmp_path))[0]
    src = fdb2.ExaminerSource(FakeChat(["Hi, I'd like a pizza.", "Great, a large one. The conversation is over."]))
    first = asyncio.run(src.next(0, t, []))
    assert first.text == "Hi, I'd like a pizza." and not first.final
    sys_msg = src.inner.llm.calls[0][0]["content"]
    assert "{not a format field}" in sys_msg and "spoken aloud" in sys_msg
    last = asyncio.run(src.next(1, t, [Utterance("user", first.text, 0, False), Utterance("agent", "Sure, what size?", 1, False)]))
    assert last.final and fdb2._end(last.text)
    t.scenario["first_turn"] = {"text": "fixed"}
    assert asyncio.run(src.next(0, t, [])).text == "fixed"


def test_make_script_and_schedule(tmp_path):
    t = fdb2.load(_data(tmp_path))[0]
    ex = FakeChat(["I want a pizza.", "Large please.", "Thanks. The conversation is over."])
    asst = FakeChat(["Sure, what size would you like?", "Large it is."])
    lines = asyncio.run(fdb2.make_script(t, ex, asst))
    assert [x["role"] for x in lines] == ["examiner", "assistant", "examiner", "assistant", "examiner"]
    assert asst.calls[0][0]["content"] == fdb2.REFERENCE_ASSISTANT and asst.calls[0][1] == {"role": "user", "content": "I want a pizza."}
    starts = fdb2.schedule(lines, [1000, 800, 1200], wps=3.0, reply_gap_ms=1000, user_gap_ms=400)
    assert starts == [0, 1000 + 1000 + 2000 + 400, 4400 + 800 + 1000 + 1000 + 400]


def test_judge_prompt_and_parse(tmp_path):
    t = fdb2.load(_data(tmp_path))[1]
    ep = {"turns": [{"role": "user", "text": "I want a pizza.", "start_time": 0, "end_time": 1500},
                    {"role": "agent", "text": "Sure, what size?", "start_time": 2500, "end_time": 3500},
                    {"role": "user", "text": "Late", "start_time": 130_000, "end_time": 131_000}]}
    p = fdb2.judge_prompt(t, ep)
    assert "Correction Handling" in p and "T3: change size" in p and "[2.50, 3.50]: Sure, what size?" in p and "Late" not in p
    assert "Entity Tracking\n\nWHAT" not in p  # only the split's task-specific rubric
    raw = '{\n "Turn-taking event and score": [\n  [2.5, 3.5]: 4, 5,\n  [[6.0, 8.25], 3, 2]\n ],\n "Task-specific score": 4\n}'
    j = fdb2.parse_judgement(raw)
    assert j == {"events": [[2.5, 3.5, 4.0, 5.0], [6.0, 8.25, 3.0, 2.0]], "task": 4.0}
    assert fdb2.episode_scores(j) == {"n_events": 2, "tt": 3.5, "if": 3.5, "task": 4.0}
    assert fdb2.parse_judgement('["6.00", "12.16"]: 5, 4,\n "[7.5, 9]: 3, 2"')["events"] == [[6.0, 12.16, 5.0, 4.0], [7.5, 9.0, 3.0, 2.0]]
    assert fdb2.episode_scores(j, "Daily")["task"] is None
    assert fdb2.parse_judgement('{"Turn-taking event and score": [], "Task-specific score": null}')["task"] is None
    assert not fdb2.reached_end(ep)


def test_stage_closing_forces_end_when_goals_covered(tmp_path):
    t = fdb2.load(_data(tmp_path))[0]
    tracker = FakeChat(['{"T1": true, "T2": false, "T3": false, "T4": false}', '{"T1": true, "T2": true, "T3": true, "T4": true}'])
    src = fdb2.ExaminerSource(FakeChat(["Large please.", "Great, thanks!"]), closing=fdb2.StageClosing(tracker))
    convo = [Utterance("user", "I want a pizza.", 0, False), Utterance("agent", "Sure, what size?", 1, False)]
    a = asyncio.run(src.next(1, t, convo))
    assert not a.final and "Note for you" not in src.inner.llm.calls[0][-1]["content"]
    b = asyncio.run(src.next(2, t, convo + [Utterance("user", a.text, 2, False), Utterance("agent", "Large it is, confirmed.", 3, False)]))
    assert b.final and b.text == "Great, thanks! The conversation is over." and "The conversation is over" in src.inner.llm.calls[1][-1]["content"]
    assert src.log[-1]["close"] == "all_covered" and src.log[-1]["forced_end"] and src.log[0]["covered"] == ["T1"]


def test_stage_closing_moves_on_and_stalls(tmp_path):
    t = fdb2.load(_data(tmp_path))[0]
    none = '{"T1": true, "T2": false, "T3": false, "T4": false}'
    c = fdb2.StageClosing(FakeChat([none] * 10), max_per_stage=2, max_stall=3)
    convo = [Utterance("user", "Hi", 0, False), Utterance("agent", "Hello?", 1, False)]
    out = [asyncio.run(c.step(i, t, convo, {})) for i in range(1, 5)]
    assert out[0][0] is None and out[1][0] is None                        # T2 current twice
    assert "move on" in out[2][0] and out[2][2]["moved_on"] == "T2" and not out[2][1]
    assert out[3][1] and out[3][2]["close"] == "stalled"                   # 3 lines without progress


def test_stage_analysis_and_reached_judge(tmp_path):
    t = fdb2.load(_data(tmp_path))[1]
    ep = {"meta": {"duration_ms": 120_000}, "turns": [
        {"role": "user", "text": "I want a pizza.", "start_time": 0, "end_time": 2000},
        {"role": "agent", "text": "Sure, what size? " * 10, "start_time": 3000, "end_time": 100_000},
        {"role": "user", "text": "Large.", "start_time": 101_000, "end_time": 102_000}]}
    assert "[3.0-100.0] ASSISTANT: Sure" in fdb2.stage_prompt(t, ep) and "T3: change size" in fdb2.stage_prompt(t, ep)
    raw = ('{"T1": {"reached": true, "completed": true, "agent": "yes", "retries": 0}, "T2": {"reached": true, "completed": false, '
           '"agent": "partial", "retries": 1}, "T3": {"reached": false, "completed": true, "agent": "n/a", "retries": 0}, '
           '"T4": {"reached": false}, "pleasantry_loop": false, "examiner_drift": false}')
    st = fdb2.parse_stages(raw)
    assert st["reached"] == ["T1", "T2"] and st["completed"] == ["T1"] and st["max_reached"] == 2 and st["stage_score"] == 0.75
    pace = fdb2.pacing(ep)
    assert pace["examiner_lines"] == 2 and round(pace["agent_share"], 2) == 0.81 and pace["gap_s"] == 1.0
    assert fdb2.cap_cause(st, pace) == "slow_pacing"
    st["stages"]["T2"]["retries"] = 2
    assert fdb2.cap_cause(st, pace) == "agent_failed_stage"
    p = fdb2.judge_prompt(t, ep, stages=["T1", "T2"])
    assert "T2: size" in p and "T3: change size" not in p and "never reached" in p
