import asyncio

from interaction_gym import REWARD, AgentSpec, Env, Frame, Node, Segment, Sim, Task


def run(coro):
    return asyncio.run(coro)


class Emitter(Node):
    """Emits the given frames once, at t=0."""

    def __init__(self, frames):
        self.frames = frames

    async def step(self, state, t, inbox):
        return (self.frames if t == 0 else []), None


class Recorder(Node):
    def __init__(self, reads):
        self.reads = reads

    def init_state(self, task, rng):
        return []

    async def step(self, state, t, inbox):
        state.append((t, [f.stream for f in inbox]))
        return [], None


def test_event_driven_delivery_skips_idle_time():
    sim = Sim({"a": Emitter([Frame("x", 500), Frame("x", 90_000)]), "b": Recorder(("x",))}, Task())
    run(sim.advance_to(100_000))
    # b runs at t=0 (initial call) and exactly when frames arrive, never in between
    assert sim.states["b"] == [(0, []), (500, ["x"]), (90_000, ["x"])]
    assert sim.t == 100_000


def test_routing_by_pattern_and_no_self_delivery():
    class Echo(Node):
        reads = ("*",)

        def init_state(self, task, rng):
            return []

        async def step(self, state, t, inbox):
            state.extend(f.stream for f in inbox)
            return ([Frame("echo.out", t)] if t == 0 else []), None

    sim = Sim({"e": Echo(), "u": Recorder(("echo.*",)), "v": Recorder(("other.*",))}, Task())
    run(sim.advance_to(10))
    assert sim.states["e"] == []  # never receives its own output
    assert sim.states["u"][-1] == (0, ["echo.out"])
    assert sim.states["v"] == [(0, [])]


def test_wake_schedule():
    class Ticker(Node):
        def init_state(self, task, rng):
            return []

        async def step(self, state, t, inbox):
            state.append(t)
            return [], (t + 100 if t < 300 else None)

    sim = Sim({"k": Ticker()}, Task())
    run(sim.advance_to(1_000))
    assert sim.states["k"] == [0, 100, 200, 300]


def test_same_instant_frames_are_delivered_in_order():
    class Relay(Node):
        reads = ("a",)

        async def step(self, state, t, inbox):
            return [Frame("b", t) for _ in inbox], None

    sim = Sim({"src": Emitter([Frame("a", 50)]), "r": Relay(), "rec": Recorder(("b",))}, Task())
    run(sim.advance_to(50))
    assert sim.states["rec"][-1] == (50, ["b"])
    assert [f.stream for f in sim.log] == ["a", "b"]


def test_segment_play_heard_cut():
    s = Segment("s", 1000, 1000, "abcdefghij")
    assert s.heard(999) == "" and s.heard(1500) == "abcde" and s.heard(5000) == "abcdefghij"
    assert s.play(1200, 1400) == "cd"
    c = s.cut(1300)
    assert (c.id, c.t0, c.end, c.data) == ("s", 1000, 1300, "abc")
    assert c.heard(1150) == s.heard(1150)
    assert Segment("a", 0, 100, [1, 2, 3, 4]).play(0, 50) == [1, 2]  # works for action sequences too


def test_reward_stream():
    sim = Sim({"j": Emitter([Frame(REWARD, 10, 0.5), Frame(REWARD, 20, 0.25)])}, Task())
    run(sim.advance_to(1_000))
    assert sim.reward == 0.75


def test_episode_ends_by_itself_without_cutting_anyone():
    async def main():
        env = Env({"j": Emitter([Frame("user.speech", 0, Segment("u0", 0, 1000, "hi"))])}, AgentSpec(chunk_ms=100), end_idle_ms=500)
        await env.reset(Task())
        done = False
        while not done:
            act = [Frame("policy.speech", env.t, Segment("a0", env.t, 2000, "a long answer"))] if env.t == 1200 else []
            _, _, done = await env.step(act)
        return env

    env = run(main())
    assert not env.truncated and env.t == 1200 + 2000 + 500  # the reply plays out fully, then 500 ms of quiet
    assert [f.data.end for f in env.log if isinstance(f.data, Segment) and f.data.id == "a0"] == [3200]  # never cut
    env2 = Env({"j": Emitter([Frame("x", 0)])}, AgentSpec(chunk_ms=100), max_ms=300)
    run(env2.reset(Task()))
    while not run(env2.step([]))[2]:
        pass
    assert env2.truncated


def test_log_filter():
    sim = Sim({"a": Emitter([Frame("keep.x", 1), Frame("drop.y", 2)])}, Task(), log=("keep.*",))
    run(sim.advance_to(10))
    assert [f.stream for f in sim.log] == ["keep.x"]


class Counter(Node):
    """Counts policy pings; replies with its running count."""

    reads = ("ping",)

    def init_state(self, task, rng):
        return {"n": 0}

    async def step(self, state, t, inbox):
        state["n"] += len(inbox)
        return [Frame("count", t, state["n"]) for _ in inbox], None


def test_env_step_latency_and_fork_isolation():
    async def main():
        env = Env({"c": Counter()}, AgentSpec(chunk_ms=100, obs=("count",)))
        await env.reset(Task())
        obs, _, _ = await env.step([Frame("ping", env.t + 250)])  # action lands after the step window
        assert obs == []
        obs, _, _ = await env.step([])
        obs, _, _ = await env.step([])
        assert [(f.t, f.data) for f in obs] == [(250, 1)]

        a, b = env.fork(2)
        await a.step([Frame("ping", a.t), Frame("ping", a.t)])
        await b.step([])
        assert a.sim.states["c"]["n"] == 3
        assert b.sim.states["c"]["n"] == 1
        assert env.sim.states["c"]["n"] == 1  # original untouched
        assert len(a.log) > len(b.log) == len(env.log)

    run(main())
