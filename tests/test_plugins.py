"""The plugin layer: registry, layered configuration, composition, and a run through ``ig``."""

import asyncio
import json
import warnings
from dataclasses import dataclass

import pytest

from interaction_gym import AgentSpec, Env, Task
from interaction_gym.agents import CannedAgent
from interaction_gym.plugins import REGISTRY, Composition, PluginError, Registry, compose, plugin
from interaction_gym.plugins import registry as R
from interaction_gym.plugins.cli import main
from interaction_gym.traj import episode
from interaction_gym.user import ReplayUser, Voice

MINIMAL = Composition.bundle("minimal")


def fresh(name="x"):
    """A composition with the minimal bundle's rows (not the shared object)."""
    return Composition.bundle("minimal")


# ---------------------------------------------------------------- registry


def test_unknown_plugin_names_the_near_misses_and_the_kind():
    with pytest.raises(PluginError, match="did you mean agent/canned"):
        REGISTRY.get("agent/caned")
    with pytest.raises(PluginError, match="role plugins: role/controlled"):
        REGISTRY.get("role/nothing-like-it")


def test_registering_twice_or_a_bad_kind_fails():
    reg = Registry()

    @plugin("sink", "mine", registry=reg)
    def mine(ctx, cfg):
        return None

    with pytest.raises(PluginError, match="already registered"):
        reg.add(R.Plugin("sink", "mine", lambda c, k: 1))
    with pytest.raises(PluginError, match="unknown kind"):
        reg.add(R.Plugin("widget", "x", lambda c, k: 1))
    with pytest.raises(PluginError, match="Config must be a dataclass"):
        reg.add(R.Plugin("sink", "y", lambda c, k: 1, config=dict))


def test_entry_points_are_loaded_once_and_a_broken_one_only_warns(monkeypatch):
    reg = Registry()
    loads = []

    class EP:
        def __init__(self, name, fn):
            self.name, self.value, self.fn = name, f"pkg:{name}", fn

        def load(self):
            loads.append(self.name)
            return self.fn()

    def good():
        plugin("scorer", "from_ep", registry=reg)(lambda ctx, cfg: (lambda ep, env, agent: {}))

    def bad():
        raise ImportError("no such module")

    monkeypatch.setattr(R.metadata, "entry_points", lambda group: [EP("good", good), EP("bad", bad)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reg.load()
        reg.load()
    assert loads == ["good", "bad"] and reg.get("scorer/from_ep")
    assert any("bad" in str(x.message) and "no such module" in str(x.message) for x in w)


# ---------------------------------------------------------------- configuration


def test_bundle_layers_and_patch_replace_a_row_s_whole_config():
    comp = fresh()
    assert comp.layers == ["core", "minimal"]
    assert comp.rows["env"].config == {"max_ms": 60000, "end_idle_ms": 3000}  # minimal's patch replaced core's
    comp.apply("- id: env\n  config: {max_ms: 1000}\n", "p1")
    assert comp.rows["env"].config == {"max_ms": 1000}  # no merging: end_idle_ms is gone (its default applies)
    comp.set("env.end_idle_ms=500")
    assert comp.rows["env"].config == {"max_ms": 1000, "end_idle_ms": 500}  # --set merges one key
    comp.set("agent.yield_after=null")
    comp.set("user.voice.words_per_sec=2.5")
    assert comp.rows["agent"].config["yield_after"] is None and comp.rows["user"].config["voice"] == {"words_per_sec": 2.5}
    comp.set("viewer.disabled=true")
    assert comp.rows["viewer"].disabled and "--set viewer.disabled=true" in comp.layers


def test_configuration_mistakes_fail_at_resolve_with_the_row_named():
    with pytest.raises(PluginError, match="row 'user' already exists .*patch it"):
        fresh().apply("- insert: [{id: user, plugin: role/replay}]", "p")
    with pytest.raises(PluginError, match="cannot patch unknown row 'nope'"):
        fresh().apply("- id: nope\n  config: {}", "p")
    with pytest.raises(PluginError, match="unknown fields \\['plugn'\\]"):
        fresh().apply("- insert: [{id: z, plugin: role/replay, plugn: 1}]", "p")
    c = fresh()
    c.apply("- id: user\n  plugin: role/user\n  config: {source: llmm}", "p")
    with pytest.raises(PluginError, match="row 'user': source names 'llmm', which is not the id of an enabled row$"):
        c.resolve()
    c = fresh()
    c.apply("- id: agent\n  config: {replies: [a], yield_afte: 1}", "p")
    with pytest.raises(PluginError, match="row 'agent' .*does not take yield_afte; its fields are replies"):
        c.resolve()
    c = fresh()
    c.apply("- id: agent\n  config: {streaming: true}", "p")
    with pytest.raises(PluginError, match="agent/canned needs replies"):
        c.resolve()
    c = fresh()
    c.set("env.max_ms=soon")
    with pytest.raises(PluginError, match="env.max_ms must be int or null, got 'soon'"):
        c.resolve()
    c = fresh()
    c.apply("- id: agent\n  plugin: agent/cannd", "p")
    with pytest.raises(PluginError, match="row 'agent' .*unknown plugin 'agent/cannd'; did you mean agent/canned"):
        c.resolve()
    c = fresh()
    c.apply("- remove: controlled", "p")
    with pytest.raises(PluginError, match="row 'agent' requires 'controlled': no enabled row"):
        c.resolve()
    c = fresh()
    c.apply("- id: env\n  requires: [agent]\n- id: controlled\n  requires: [env]", "p")
    with pytest.raises(PluginError, match="cycle"):
        c.resolve()
    c = fresh()
    c.apply("- insert: [{id: second, plugin: role/controlled}]", "p")
    with pytest.raises(PluginError, match="both provide 'controlled'"):
        c.resolve()
    with pytest.raises(PluginError, match="no bundle 'nope'; built-in bundles: .*core, .*minimal"):
        Composition.bundle("nope")


def test_requirements_order_rows_and_the_dump_reproduces_the_composition():
    res = fresh().resolve()
    ids = [r.id for r in res.rows]
    assert ids.index("controlled") < ids.index("agent")  # agent requires controlled
    again = Composition()
    again.apply(res.dump(), "dump")
    assert again.resolve().digest() == res.digest()
    assert res.meta()["layers"] == ["core", "minimal"] and len(res.digest()) == 12
    c = fresh()
    c.set("env.max_ms=59999")
    assert c.resolve().digest() != res.digest()


# ---------------------------------------------------------------- composition


def test_compose_sorts_objects_by_kind_and_checks_the_shape():
    run = compose(fresh().resolve())
    assert set(run.nodes) == {"user"} and run.controlled.name == "agent" and run.spec.chunk_ms == 200
    assert [t.id for t in run.tasks] == ["weather-correction"] and set(run.sinks) == {"episodes", "viewer", "stdout"}
    assert set(run.scorers) == {"interaction"} and run.driver is not None and run.env_config.max_ms == 60000
    c = fresh()
    c.apply("- insert: [{id: agent2, plugin: agent/canned, config: {replies: [x]}}]", "p")
    with pytest.raises(PluginError, match="two agent rows"):
        compose(c.resolve())
    c = fresh()
    c.apply("- remove: controlled\n- id: agent\n  requires: []\n- insert: [{id: k, plugin: role/controlled}]\n- remove: k", "p")
    with pytest.raises(PluginError, match="requires 'controlled'"):
        c.resolve()


def test_tools_and_fake_backends_compose_into_the_session():
    c = fresh()
    c.apply("""
- insert:
    - {id: llm, plugin: backend/fake_chat, config: {replies: ["Hi there.", "Bye."]}}
    - {id: tts, plugin: backend/tts_fake}
    - {id: lookup, plugin: tool/functions, config: {functions: {check: "tests.test_plugins:check_availability"}, name: tables}}
    - {id: tools, plugin: node/tool_world, config: {backends: [lookup], latency_ms: 250}}
    - {id: lines, plugin: source/llm, config: {llm: llm}}
- id: user
  plugin: role/user
  config: {source: lines, voice: {tts: tts, clone: false}, timing: {respond_after_ms: 500, response_delay: {median_ms: 200}}, behaviors: {noise_per_min: 0}}
""", "p")
    run = compose(c.resolve())
    ids = [r.id for r in run.resolved.rows]
    assert ids.index("llm") < ids.index("lines") < ids.index("user") and ids.index("lookup") < ids.index("tools")  # refs order the build
    assert set(run.nodes) == {"user", "tools"}
    env = run.make_env()
    asyncio.run(env.reset(Task("t", {"persona": "a caller", "instructions": "ask for a table"}), 0))
    assert [t.name for t in env.initial_session(Task("t")).tools] == ["check"]
    user = env.nodes["user"]  # built per episode (a factory): the same configuration, a fresh object each time
    assert user.timing.respond_after_ms == 500 and user.timing.response_delay.median_ms == 200 and run.make_env().nodes["user"] is not user


def check_availability(party: int) -> dict:
    """Is a table free?"""
    return {"available": party < 5}


# ---------------------------------------------------------------- a run


def test_ig_run_writes_episodes_with_the_config_and_resumes(tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["run", "minimal", "--out", str(out), "--set", "viewer.disabled=true"]) == 0
    eps = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert len(eps) == 1 and eps[0]["meta"]["episode_id"] == "weather-correction/s0"
    m = eps[0]["meta"]
    assert m["controlled"] == "agent" and m["config"]["layers"] == ["core", "minimal", "--set viewer.disabled=true"]
    assert {r["id"] for r in m["config"]["rows"]} >= {"env", "user", "agent", "tasks"} and "viewer" not in {r["id"] for r in m["config"]["rows"]}
    assert "interaction" in eps[0]["eval"] and eps[0]["eval"]["scores"]["total"] is not None
    assert not (out / "index.html").exists()
    roles = [(t["role"], "unsaid" in t) for t in eps[0]["turns"]]
    assert roles == [("user", False), ("agent", True), ("user", False), ("agent", False)]  # the first reply is cut
    # the same command again: nothing to do (resumed from episodes.jsonl)
    assert main(["run", "minimal", "--out", str(out), "--set", "viewer.disabled=true"]) == 0
    assert "0 pairs to run (1 episodes done)" in [x for x in capsys.readouterr().out.splitlines() if x.startswith("ig:")][-1]
    assert len((out / "episodes.jsonl").read_text().splitlines()) == 1


def test_ig_run_dump_config_and_errors(tmp_path, capsys):
    assert main(["run", "minimal", "--dump-config", "--set", "env.max_ms=1234"]) == 0
    text = capsys.readouterr().out
    assert "max_ms: 1234" in text and text.rstrip().endswith("layers: core > minimal > --set env.max_ms=1234")
    assert main(["run", "minimal", "--set", "agent.nope=1", "--out", str(tmp_path)]) == 2
    assert "does not take nope" in capsys.readouterr().err
    assert main(["plugins", "agent"]) == 0
    assert "agent/canned" in capsys.readouterr().out


def test_the_composed_run_matches_the_same_episode_built_by_hand():
    run = compose(fresh().resolve())
    task = run.tasks[0]
    from interaction_gym.plugins import EpisodeContext
    from interaction_gym.plugins.builtin.drivers import run_episode

    ep, env, agent = asyncio.run(run_episode(run, EpisodeContext(task, 0, episode_id="x")))

    turns = run.resolved.configs["user"].turns
    spec = AgentSpec(chunk_ms=200)
    env2 = Env({"user": ReplayUser(turns, voice=Voice())}, spec, max_ms=60000)
    pol = CannedAgent(run.resolved.configs["agent"].replies, spec, yield_after=160)

    async def go():
        obs = await env2.reset(task, 0)
        done = False
        while not done:
            obs, _, done = await env2.step(pol.act(env2.t, obs))
        return episode(env2, "x")

    ep2 = asyncio.run(go())
    assert ep["turns"] == ep2["turns"] and env.log == env2.log
