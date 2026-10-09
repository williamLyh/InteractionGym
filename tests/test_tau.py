"""τ-bench integration (layer 1): tasks, tool world, official evaluation — no real models involved."""

import asyncio

import pytest

pytest.importorskip("tau2")

from interaction_gym import AgentSpec, Chunk, Env, Frame, Segment  # noqa: E402
from interaction_gym.clients import FakeChat  # noqa: E402
from interaction_gym.integrations import tau  # noqa: E402
from interaction_gym.tools import CALL, RESULT, ToolCall, ToolWorld  # noqa: E402
from interaction_gym.user import STOP, LLMSource, UserSim  # noqa: E402

SPEC = AgentSpec(obs=("user.*", RESULT["agent"]))


def make_env(latency_ms=0):
    user = UserSim(LLMSource(FakeChat(["Hi, I need some help with my tasks.", f"Great, thanks! {STOP}"])))
    return Env({"user": user, "tools": ToolWorld(tau.TauBackend("mock"), latency_ms)}, SPEC, max_ms=120_000)


async def episode(task, agent, latency_ms=0):
    env = make_env(latency_ms)
    obs = await env.reset(task)
    done = False
    while not done:
        obs, _, done = await env.step(agent.act(env.t, obs) if agent else [])
    return env


class Lazy:
    """Talks but never uses a tool."""

    def __init__(self):
        self.replied = False

    def act(self, t, obs):
        if not self.replied and any(isinstance(f.data, Chunk) and f.data.last for f in obs):
            self.replied = True
            return [Frame("policy.speech", t, Segment("a0", t, 1000, "Sure, anything else?"))]
        return []


def test_tasks_load():
    ts = tau.tasks("mock")
    assert len(ts) == 10
    t = tau.load_task("mock", "create_task_1")
    assert t.id == "mock/create_task_1" and "Important Meeting" in t.scenario["instructions"]
    assert tau.tau_task(t).id == "create_task_1"
    backend = tau.TauBackend("mock")
    assert "create_task" in {spec.name for spec in backend.tools("agent")} and backend.instructions("agent")


# Tasks the oracle cannot solve by replaying gold agent actions:
NEEDS_INFERENCE = "update_task_with_history_and_env_assertions"  # no gold actions; only the end state is checked
NEEDS_USER_TOOL = "update_task_with_user_tools"  # the *user* must call one of their own tools
ORACLE_TASKS = [t for t in tau.tasks("mock") if t.scenario["tau_id"] not in (NEEDS_INFERENCE, NEEDS_USER_TOOL)]


@pytest.mark.parametrize("task", ORACLE_TASKS, ids=lambda t: t.id)
def test_oracle_gets_full_reward(task):
    env = asyncio.run(episode(task, tau.OracleAgent(task)))
    res = tau.evaluate(env.log, task)
    assert res["termination"] == "user_stop"
    if res["nl_skipped"]:
        assert res["reward"] is None
    else:
        assert res["reward"] == 1.0, res


class Extra:
    """Oracle plus frames injected once at a given step (what the oracle cannot know to do)."""

    def __init__(self, task, frames):
        self.oracle, self.frames = tau.OracleAgent(task), frames

    def act(self, t, obs):
        out = self.oracle.act(t, obs)
        if t >= 1000 and self.frames:
            out, self.frames = out + [Frame(f.stream, t, f.data, src=f.src) for f in self.frames], []
        return out


def test_env_assertion_task_needs_the_right_end_state():
    task = tau.load_task("mock", NEEDS_INFERENCE)
    assert tau.evaluate(asyncio.run(episode(task, tau.OracleAgent(task))).log, task)["reward"] == 0.0
    fix = Frame(CALL["agent"], 0, ToolCall("fix", "update_task_status", {"task_id": "task_2", "status": "completed"}))
    assert tau.evaluate(asyncio.run(episode(task, Extra(task, [fix]))).log, task)["reward"] == 1.0


def test_user_side_tool_calls_are_routed_and_evaluated():
    task = tau.load_task("mock", NEEDS_USER_TOOL)
    before = tau.evaluate(asyncio.run(episode(task, tau.OracleAgent(task))).log, task)
    assert before["breakdown"]["ENV_ASSERTION"] == 0.0  # the notification was never dismissed

    dismiss = Frame(CALL["user"], 0, ToolCall("u1", "dismiss_notification", {"notification_id": "notif_1"}), src="user")
    env = asyncio.run(episode(task, Extra(task, [dismiss])))
    res = next(f for f in env.log if f.stream == RESULT["user"])
    assert not res.data.error
    after = tau.evaluate(env.log, task)
    assert after["breakdown"]["ENV_ASSERTION"] == 1.0 and after["breakdown"]["ACTION"] == 1.0
    # This mock task's DB check and env assertion contradict each other (its gold actions omit the
    # user's dismissal), so the overall reward stays 0 under τ-bench's own evaluator as well.
    assert after["breakdown"]["DB"] == 0.0


def test_doing_nothing_fails():
    task = tau.load_task("mock", "create_task_1")
    env = asyncio.run(episode(task, Lazy()))
    res = tau.evaluate(env.log, task)
    assert res["reward"] == 0.0 and res["parts"]["env"] == 0.0


def test_tool_latency_and_fork_isolation():
    task = tau.load_task("mock", "create_task_1")

    async def main():
        env = make_env(latency_ms=700)
        await env.reset(task)
        call = ToolCall("c1", "create_task", {"user_id": "user_1", "title": "x"})
        a, b = env.fork(2)
        await a.step([Frame(CALL["agent"], a.t, call)])
        for _ in range(5):
            await a.step([])
            await b.step([])
        return env, a, b

    env, a, b = asyncio.run(main())
    call_t = next(f.t for f in a.log if f.stream == CALL["agent"])
    res = next(f for f in a.log if f.stream == RESULT["agent"])
    assert res.t - call_t == 700 and not res.data.error
    hashes = [e.sim.states["tools"]["services"]["mock"].get_db_hash() for e in (env, a, b)]
    assert hashes[0] == hashes[2] != hashes[1]  # only the fork that called the tool changed its DB


def test_no_done_means_max_steps():
    task = tau.load_task("mock", "create_task_1")
    env = Env({"tools": ToolWorld(tau.TauBackend("mock"))}, SPEC, max_ms=1_000)

    async def main():
        await env.reset(task)
        while not (await env.step([]))[2]:
            pass

    asyncio.run(main())
    assert tau.evaluate(env.log, task, truncated=env.truncated)["termination"] == "max_steps"
