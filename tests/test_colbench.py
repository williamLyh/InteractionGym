"""ColBench adapter without its data: the scoring subprocess, the submit tool, the spoken task mapping and the user's
round limit (the simulator's prompt is passed in, so no sweet_rl checkout is needed)."""

import asyncio

from interaction_gym import AgentSpec, Env, Frame
from interaction_gym.integrations import colbench as cb
from interaction_gym.tools import CALL, RESULT, ToolCall

REF = "def area(w, h, unit='m'):\n    if w < 0 or h < 0:\n        return None\n    return f'{w * h} {unit}2'\n"
TESTS = {"t1": "area(2, 3)", "t2": "area(1, 1, unit='cm')", "t3": "area(-1, 2)"}
REC = {"problem_description": "I need a function for the area of a rectangle with its unit.The signature of the function is: def area(w, h, unit='m')",
       "hidden_information": REF, "ground_truth": REF, "test_cases": TESTS}


def test_task_mapping_speaks_the_description_and_shows_the_signature():
    t = cb.to_task(REC, "cb-0")
    assert t.scenario["first_turn"] == "I need a function for the area of a rectangle with its unit."
    assert t.scenario["agent_context"] == {"function_signature": "def area(w, h, unit='m')"}
    assert t.criteria["test_cases"] == TESTS


def test_run_tests_matches_the_reference_output():
    assert cb.run_tests(REF, REF, TESTS) == [True, True, False]  # a None reference output never counts
    wrong = "def area(w, h, unit='m'):\n    return f'{w * h} {unit}^2'\n"
    assert cb.run_tests(wrong, REF, TESTS) == [False, False, False]
    assert cb.run_tests("```python\n" + REF + "```", REF, TESTS)[:2] == [True, True]  # fenced code is unwrapped


def test_run_tests_rejects_and_times_out():
    assert cb.run_tests("import os\n" + REF, REF, TESTS) == [False] * 3
    assert cb.run_tests("def area(*a, **k):\n    while True: pass\n", REF, {"t1": "area(2, 3)"}, timeout_s=0.3) == [False]
    assert cb.run_tests(None, REF, TESTS) == [False] * 3


def test_submit_once_then_evaluate():
    task = cb.to_task(REC, "cb-0")
    env = Env({"tools": cb.tool_world(latency_ms=100)}, AgentSpec(chunk_ms=100, obs=(RESULT["agent"],)))

    async def go():
        await env.reset(task)
        good = REF.replace("return None", "return 'invalid'")  # passes t1, t2; t3's reference is None
        obs = (await env.step([Frame(CALL["agent"], 0, ToolCall("c1", "submit_code", {"code": good}))]))[0]
        obs += (await env.step([Frame(CALL["agent"], env.t, ToolCall("c2", "submit_code", {"code": "x = 1"}))]))[0]
        for _ in range(3):
            obs += (await env.step([]))[0]
        return obs

    obs = asyncio.run(go())
    results = [f.data for f in obs if f.stream == RESULT["agent"]]
    assert [r.error for r in results] == [False, True]  # the second submission is refused
    res = cb.evaluate(env, task)
    assert res["submitted"] and res["tests"] == [True, True, False] and res["reward"] == 0.0
    assert res["submit_time"] == 0


class _Echo:
    model = "fake"

    def __init__(self):
        self.seen = []
        self.reply = "It should be in square metres."

    async def chat(self, messages, **kw):
        self.seen.append(messages)
        return self.reply


def test_user_uses_the_simulator_prompt_and_stops_after_the_rounds():
    task = cb.to_task(REC, "cb-0")
    llm = _Echo()
    user = cb.ColBenchUser(llm, max_rounds=2, prompt="Help with: {problem_description}\nHidden:\n{hidden_information}\n{dialogue_history}")
    first = asyncio.run(user.next(0, task, []))
    assert first.text == task.scenario["first_turn"]
    from interaction_gym.user import Utterance
    convo = [Utterance("user", first.text, 0, False), Utterance("agent", "Which unit?", 3000, False)]
    turn = asyncio.run(user.next(1, task, convo))
    assert turn.text == "It should be in square metres."
    system = llm.seen[0][0]["content"]
    assert "Hidden:\n" + REF in system and "{" not in system.split("Hidden:")[0] and "phone call" in system
    assert asyncio.run(user.next(3, task, convo)) is None  # description + 2 answers: rounds used up
    llm.reply = "Square metres. ###STOP###"
    assert not asyncio.run(user.next(1, task, convo)).final  # no hanging up before the code is submitted
    done = convo + [Utterance("agent", "I've written it: the function returns the area with its unit.", 9000, False)]
    assert asyncio.run(user.next(2, task, done)).final
