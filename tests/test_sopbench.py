"""SOPBench adapter: task mapping, the SOP formats in the agent's prompt, calls through the env and SOPBench's own
verdict on them. Needs a checkout at third_party/sopbench (or SOPBENCH_PATH); skipped otherwise."""

import asyncio

import pytest

from interaction_gym import AgentSpec, Env, Frame
from interaction_gym.integrations import sopbench as sb
from interaction_gym.tools import CALL, RESULT, ToolCall

try:
    sb.path()
except FileNotFoundError:
    pytest.skip("needs a SOPBench checkout (third_party/sopbench or SOPBENCH_PATH)", allow_module_level=True)


def _case(domain, goal, should_succeed):
    return next(t for t in sb.tasks(domain, goals=[goal]) if t.criteria["action_should_succeed"] == should_succeed)


def test_tasks_hold_the_users_information_not_the_agents():
    t = _case("bank", "apply_credit_card", True)
    rec = t.initial_state["sopbench_case"]
    assert t.id.startswith("sopbench-bank-apply_credit_card-") and rec["user_goal"] == "apply_credit_card"
    assert rec["user_instruction"] in t.scenario["instructions"]
    assert all(str(v) in t.scenario["instructions"] for v in rec["user_known"].values() if not isinstance(v, dict))
    assert sb.tasks("hotel")[0].initial_state["sopbench_case"]["user_goal"]  # filled in where the data leaves it out


def test_sop_formats_in_the_agent_prompt():
    t = _case("bank", "apply_credit_card", True)
    prompts = {f: sb.SOPBenchBackend(sb.to_task("bank", "apply_credit_card", 0, t.initial_state["sopbench_case"], f)).prompt
               for f in sb.FORMATS}
    assert "ALL of these conditions must be met" in prompts["structured"] and "phone call" in prompts["structured"]
    assert "ALL of these conditions must be met" not in prompts["list"]
    assert prompts["structured+order"].startswith(prompts["structured"].split("\n\n\n### This is a phone call")[0])
    assert len(prompts["structured+order"]) > len(prompts["structured"])
    names = {s.name for s in sb.SOPBenchBackend(t).tools()}
    assert "apply_credit_card" in names and "login_user" in names and "exit_conversation" not in names


def test_doing_nothing_passes_exactly_the_cases_to_decline():
    for should in (True, False):
        assert sb.replay(_case("bank", "apply_credit_card", should), [])["success"] is (not should)


def test_verify_then_act_passes_and_skipping_the_verification_fails():
    t = _case("bank", "apply_credit_card", True)
    known = t.initial_state["sopbench_case"]["user_known"]
    db = t.initial_state["sopbench_case"]["initial_database"]["accounts"][known["username"]]
    act = ("apply_credit_card", {k: known[k] for k in ("username", "monthly_income", "total_assets")})
    login = ("login_user", {"username": known["username"], "identification": db["identification"]})
    ok = sb.replay(t, [("internal_check_username_exist", {"username": known["username"]}), login,
                       ("internal_get_credit_score", {"username": known["username"]}), act])
    assert ok["success"] and ok["action_successfully_called"]
    assert not sb.replay(t, [act])["success"]  # acting without the SOP's verifications


def test_calls_through_the_env_and_evaluate():
    t = _case("bank", "apply_credit_card", False)
    known = t.initial_state["sopbench_case"]["user_known"]
    env = Env({"tools": sb.tool_world(t, latency_ms=100)}, AgentSpec(chunk_ms=100, obs=(RESULT["agent"],)))

    async def go():
        await env.reset(t)
        obs = (await env.step([Frame(CALL["agent"], 0, ToolCall("c1", "internal_check_username_exist", {"username": known["username"]}))]))[0]
        obs += (await env.step([Frame(CALL["agent"], env.t, ToolCall("c2", "no_such_tool", {}))]))[0]
        for _ in range(3):
            obs += (await env.step([]))[0]
        return obs

    results = [f.data for f in asyncio.run(go()) if f.stream == RESULT["agent"]]
    assert [r.error for r in results] == [False, True] and results[0].content.startswith("(")  # the (success, value) tuple
    res = sb.evaluate(env, t)
    assert res["reward"] == float(res["success"]) and res["calls"] == [{"name": "internal_check_username_exist",
                                                                        "arguments": {"username": known["username"]}}]
    assert res["target_call_time"] is None
