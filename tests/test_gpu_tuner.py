"""GPU tuner: allocation / rounding, demand accounting, measured rounds on a mock cluster."""

import asyncio
import json
from dataclasses import replace

import pytest

from interaction_gym import AgentSpec, Env, Task
from interaction_gym.agents import CannedAgent
from interaction_gym.clients import FakeChat, FakeSpeech, OpenAIChat
from interaction_gym.gpu_tuner import (Curve, DemandProfile, Layout, Meter, MeteredAgent, MeteredChat, MeteredSpeech,
                                         MockCluster, Monitor, SupplyProfile, allocate, analyze, launch_script, load_host,
                                         loop, measure, place, propose, run_batch, solve)
from interaction_gym.gpu_tuner.empirical import mock_truth
from interaction_gym.gpu_tuner.launch import endpoints, fmt, layout_plan, launch_config
from interaction_gym.gpu_tuner.monitor import parse_prometheus
from interaction_gym.gpu_tuner.rebalance import Round, ServiceStats
from interaction_gym.user import LLMInterrupt, LLMSource, UserSim, Voice


def run(c):
    return asyncio.run(c)


def specs8():
    hw, specs = load_host()  # the shipped example preset
    return hw, specs


# ---------------------------------------------------------------- layouts and placement


def test_layout_parse_roundtrip_and_gpu_share():
    lay = Layout.parse("agent=4+5,llm=0+1/2+3,tts=6/7,clone=6")
    assert lay.alloc == {"agent": 1, "llm": 2, "tts": 2, "clone": 1}
    assert str(lay) == "agent=4+5,llm=0+1/2+3,tts=6/7,clone=6"
    share = lay.gpu_share()
    assert share["llm"] == 4 and share["agent"] == 2 and share["tts"] == 1.5 and share["clone"] == 0.5


def test_place_whole_replicas_take_aligned_blocks_and_fractional_spread_then_pack():
    hw, specs = specs8()
    p = place({"agent": 2, "llm": 1, "tts": 2, "clone": 2}, specs, hw)
    assert p["agent"] == [[0, 1], [2, 3]] and p["llm"] == [[4, 5]]
    assert sorted(g for r in p["tts"] for g in r) == [6, 7]  # one TTS per free GPU first
    assert sorted(g for r in p["clone"] for g in r) == [6, 7]  # then packed next to them (12+12 GB <= 32)
    assert place({"agent": 3, "llm": 1, "tts": 1}, specs, hw) is None  # no GPU (or spare memory) left for TTS
    assert place({"tts": 6}, specs, replace(hw, gpus=[0, 1])) is None  # 3 x 12 GB > 32 GB per GPU


def test_place_respects_pins_and_gpu_subset():
    hw, specs = specs8()
    specs["agent"] = replace(specs["agent"], gpu_ids=[4, 5])
    p = place({"agent": 1, "llm": 1}, specs, replace(hw, gpus=[2, 3, 4, 5]))
    assert p == {"agent": [[4, 5]], "llm": [[2, 3]]}


# ---------------------------------------------------------------- allocation / rounding


def test_allocate_follows_targets_in_whole_replicas():
    hw, specs = specs8()
    lay = allocate({"agent": 5.1, "llm": 1.3, "tts": 0.6, "clone": 1.0}, specs, hw)
    assert lay.alloc == {"agent": 2, "llm": 1, "tts": 1, "clone": 1}
    assert len({g for r in lay.placement.values() for x in r for g in x}) == 8  # every GPU used


def test_allocate_gives_spare_gpus_to_the_most_underprovisioned_and_keeps_minimums():
    hw, specs = specs8()
    lay = allocate({"agent": 0.1, "llm": 7.0, "tts": 0.2, "clone": 0.2}, specs, hw)
    assert lay.alloc["agent"] == 1 and lay.alloc["tts"] == 1 and lay.alloc["clone"] == 1  # every service keeps one replica
    assert lay.alloc["llm"] == 2  # 2 + 2x2 + fractional on the rest
    lay = allocate({"agent": 1.0, "llm": 1.0, "tts": 3.0, "clone": 0.5}, specs, replace(hw, gpus=[0, 1, 2, 3, 4, 5]))
    assert lay.alloc["tts"] >= 2 and lay.alloc["clone"] == 1


def test_allocate_reserved_service_is_kept_out_of_the_split():
    hw, specs = specs8()
    specs["trainer"] = replace(specs["trainer"], gpus=2, min_replicas=1)
    lay = allocate({"agent": 4.0, "llm": 1.0, "tts": 0.5, "clone": 0.5}, specs, hw, keep={"trainer": 1})
    assert lay.alloc["trainer"] == 1 and lay.alloc["agent"] == 1  # 2 trainer + 2 agent + 2 llm + 2 shared TTS GPUs


def test_allocate_infeasible_minimum_raises():
    hw, specs = specs8()
    with pytest.raises(ValueError):
        allocate({"agent": 1, "llm": 1}, specs, replace(hw, gpus=[0, 1, 2]))


def _round(stats: dict, *, eph=400.0, n=4, cap=4, levels=None) -> Round:
    from interaction_gym.gpu_tuner.empirical import Level

    lay = Layout.parse("agent=4+5,llm=0+1/2+3,tts=6/7,clone=6")
    full = {}
    for s, (busy, waiting, share) in stats.items():
        held = lay.gpu_share()[s]
        full[s] = ServiceStats(s, lay.alloc[s], lay.gpus_of(s), held, busy, busy / held, None, None, waiting, None, share * 30, share, 1)
    lv = levels or [Level(2, 10, 60, eph * 0.55, 55), Level(n, 20, 60, eph, 55)]
    return analyze(Round(lay, lv, full, session_cap=cap))


def test_analyze_bottleneck_priority_queue_then_util_then_sessions_then_wait():
    r = _round({"agent": (1.4, 0, 0.5), "llm": (0.2, 3.0, 0.2), "tts": (0.1, 0, 0.05), "clone": (0.2, 0, 0.1)})
    assert r.bottleneck == "llm" and "queues" in r.why
    r = _round({"agent": (1.95, 0, 0.5), "llm": (0.2, 0, 0.2), "tts": (0.1, 0, 0.05), "clone": (0.2, 0, 0.1)})
    assert r.bottleneck == "agent" and "util" in r.why
    r = _round({"agent": (1.4, 0, 0.5), "llm": (0.2, 0, 0.2), "tts": (0.1, 0, 0.05), "clone": (0.2, 0, 0.1)})
    assert r.bottleneck == "agent" and "sessions" in r.why and set(r.idle) == {"llm", "tts"}


def test_propose_moves_gpus_from_idle_llm_to_the_agent():
    hw, specs = specs8()
    r = _round({"agent": (1.45, 0, 0.53), "llm": (0.11, 0, 0.21), "tts": (0.08, 0, 0.03), "clone": (0.26, 0, 0.14)})
    p = propose(r, specs, hw)
    assert p.layout.alloc == {"agent": 2, "llm": 1, "tts": 1, "clone": 1}
    assert p.concurrency == 8  # the new session capacity
    assert p.projected_eph == pytest.approx(800)  # session-limited: 400 ep/h x 8/4 sessions
    assert abs(sum(p.targets.values()) - 8) < 1e-6


def test_propose_without_gpu_samples_uses_wait_shares():
    hw, specs = specs8()
    r = _round({"agent": (1.45, 0, 0.53), "llm": (0.11, 0, 0.21), "tts": (0.08, 0, 0.03), "clone": (0.26, 0, 0.14)})
    for x in r.stats.values():
        x.busy = x.util = None
    p = propose(r, specs, hw)
    assert "wait share" in p.basis and p.layout.alloc["agent"] >= p.layout.alloc["llm"]


# ---------------------------------------------------------------- demand accounting


def test_meter_counts_user_side_demand_of_a_real_episode():
    meter = Meter()
    chat = FakeChat(["Sure, around seven.", "Thanks, bye!"])
    tts, clone = FakeSpeech(), FakeSpeech()
    llm = MeteredChat(chat, meter, "llm")
    user = UserSim(LLMSource(llm, first="Hi, I'd like to book a table."),
                   Voice(MeteredSpeech(tts, meter, "tts"), clone=MeteredSpeech(clone, meter, "clone")), interrupt=LLMInterrupt(llm))
    spec = AgentSpec(chunk_ms=200)
    env = Env({"user": user}, spec, max_ms=30_000)
    agent = MeteredAgent(CannedAgent(["Hello, what time?", "Booked, goodbye."], spec), meter, "agent")

    async def go():
        tok = meter.start_episode("e0")
        obs = await env.reset(Task(id="t", scenario={"persona": "caller"}))
        done = False
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
        meter.end_episode(env.t, tok)

    run(go())
    d = meter.demand()
    assert d.services["llm"].calls == len(chat.calls)
    assert d.services["tts"].calls + d.services["clone"].calls == tts.calls + clone.calls
    assert d.services["agent"].units == pytest.approx(env.t / 1000)  # simulated seconds stepped
    assert d.sim_s == env.t / 1000 and d.services["llm"].source == "estimated"  # FakeChat reports no usage
    assert d.services["tts"].units > 0


def test_metered_openai_chat_reads_usage(monkeypatch):
    from interaction_gym.gpu_tuner import meter as m

    def fake_post(url, payload, key):
        return json.dumps({"choices": [{"message": {"content": "<think>x</think> Hello"}}],
                           "usage": {"prompt_tokens": 123, "completion_tokens": 7}}).encode()

    monkeypatch.setattr(m, "_post", fake_post)
    meter = Meter()
    out = run(MeteredChat(OpenAIChat("http://x/v1", "m"), meter).chat([{"role": "user", "content": "hi"}]))
    assert out == "Hello"
    e = meter.events[0]
    assert (e.prompt_tokens, e.completion_tokens, e.estimated_tokens) == (123, 7, False)


def test_meter_concurrency():
    meter = Meter()
    from interaction_gym.gpu_tuner.meter import Event

    meter.events = [Event("llm", None, 0, 2, 1), Event("llm", None, 1, 3, 1)]
    c = meter.concurrency("llm")
    assert c["peak"] == 2 and c["mean"] == pytest.approx(4 / 3)


def test_demand_from_saved_episodes():
    ep = {"meta": {"episode_id": "e", "duration_ms": 40_000,
                   "task": {"scenario": {"first_turn": "Hi"}},
                   "user": {"mode": "online", "voice": {"clone": {}}, "turn_taking": {"check_ms": 1000}, "barge_in": {"min_words": 5}}},
          "turns": [{"role": "user", "start_time": 0, "end_time": 2000, "text": "Hi"},
                    {"role": "agent", "start_time": 3000, "end_time": 9000, "text": "Hello there how can I help you today"},
                    {"role": "user", "start_time": 10000, "end_time": 13000, "text": "A table for two"}]}
    tr = {"episode_id": "e", "unit_ms": 1000, "units": [{"unit_index": i} for i in range(-1, 40)]}
    d = DemandProfile.from_episodes([ep], [tr])
    assert d.sim_s == 40 and d.services["agent"].units == 40
    assert d.services["tts"].units == 2 and d.services["clone"].units == 3 and d.services["clone"].calls == 1
    assert d.services["llm"].calls > 2  # one LLM-written turn + the closing reply + barge-in checks


def test_demand_json_roundtrip(tmp_path):
    d = DemandProfile.synthetic()
    d.save(tmp_path / "d.json")
    assert DemandProfile.load(tmp_path / "d.json").to_json() == d.to_json()


# ---------------------------------------------------------------- monitor


def test_parse_prometheus_sums_label_sets():
    text = ('# HELP x\nvllm:num_requests_running{engine="0"} 3\nvllm:num_requests_running{engine="1"} 2\n'
            'vllm:num_requests_waiting 1.0\nvllm:kv_cache_usage_perc 0.25\n')
    m = parse_prometheus(text)
    assert m["vllm:num_requests_running"] == 5 and m["vllm:num_requests_waiting"] == 1 and m["vllm:kv_cache_usage_perc"] == 0.25
    staged = ('vllm:num_requests_running{replica="0",stage="0"} 2\nvllm:num_requests_running{replica="1",stage="0"} 1\n'
              'vllm:num_requests_running{replica="0",stage="1"} 1\n')
    assert parse_prometheus(staged)["vllm:num_requests_running"] == 3  # busiest stage (summed over replicas), not all stages
    hand_off = 'vllm:num_requests_waiting{stage="0"} 0\nvllm:num_requests_waiting{stage="1"} 1\n'
    assert parse_prometheus(hand_off)["vllm:num_requests_waiting"] == 0  # stage 1 waiting on stage 0's chunks is no queue


# ---------------------------------------------------------------- the measured loop on a mock cluster


def test_run_batch_measures_episodes_per_hour():
    async def ep(i):
        await asyncio.sleep(0.02)
        return 30_000

    lv = run(run_batch(ep, 4, episodes=12))
    assert lv.episodes == 12 and lv.eph == pytest.approx(4 / 0.02 * 3600, rel=0.25)


TS = 0.003  # mock time compression: wide enough that timer jitter does not matter


def _mock(hw, specs, **kw):
    return MockCluster(DemandProfile.synthetic(), mock_truth(SupplyProfile.synthetic(), {"llm": 1.1}), specs,
                       replace(hw, colocation_penalty=1.0), time_scale=TS, **kw)


def test_measure_ramps_to_session_capacity_and_finds_the_agent_bottleneck():
    hw, specs = specs8()
    mock = _mock(hw, specs)
    lay = Layout.parse("agent=4+5,llm=0+1/2+3,tts=6/7,clone=6")
    mock.start(lay.placement)

    async def go():
        mon = Monitor(sample_fn=mock.sample, interval=0.006)
        async with mon.running():
            return await measure(lay, mock.runner, specs, monitor=mon, levels=(2, 4, 8, 16), seconds=0.4, scale=1 / TS, log=lambda *a: None)

    r = run(go())
    assert [lv.concurrency for lv in r.levels] == [2, 4]  # capped at 1 replica x 4 sessions
    assert r.bottleneck == "agent" and "llm" in r.idle
    assert r.stats["agent"].util > r.stats["llm"].util
    assert 0.3 < r.stats["agent"].wait_share < 0.8


def test_loop_applies_the_proposal_and_doubles_measured_throughput():
    hw, specs = specs8()
    mock = _mock(hw, specs)
    lay = Layout.parse("agent=4+5,llm=0+1/2+3,tts=6/7,clone=6")
    mock.start(lay.placement)
    res = run(loop(lay, lambda lay_, meter: mock.runner(meter), specs, hw, apply=mock.apply_layout, caps=False,
                   monitor_for=lambda lay_: Monitor(sample_fn=mock.sample, interval=0.006), levels=(2, 4, 8), seconds=0.4, scale=1 / TS,
                   log=lambda *a: None))
    assert res.measured and len(res.rounds) in (2, 3)  # a third round may try moving a TTS/clone replica, then stops
    a = res.rounds[1].layout.alloc  # GPUs move from the idle LLM to the agent (TTS/clone split varies with measurement noise)
    assert a["agent"] == 2 and a["llm"] == 1 and a["tts"] >= 1 and a["clone"] >= 1
    assert res.rounds[1].best.eph > 1.6 * res.rounds[0].best.eph
    assert res.recommendation.alloc["agent"] == 2 and res.recommendation.alloc["llm"] == 1 and res.concurrency == 8
    assert mock.applied[-1] == res.recommendation.alloc  # the best measured layout is the one left running


def test_loop_without_restarts_only_proposes():
    hw, specs = specs8()
    mock = _mock(hw, specs)
    lay = Layout.parse("agent=4+5,llm=0+1/2+3,tts=6/7,clone=6")
    mock.start(lay.placement)
    res = run(loop(lay, lambda lay_, meter: mock.runner(meter), specs, hw, apply=None, caps=False,
                   monitor_for=lambda lay_: Monitor(sample_fn=mock.sample, interval=0.006), levels=(2, 4), seconds=0.3, scale=1 / TS,
                   log=lambda *a: None))
    assert not res.measured and len(res.rounds) == 1 and mock.applied == [{"agent": 1, "llm": 2, "tts": 2, "clone": 1}]
    assert res.recommendation.alloc["agent"] == 2 and "projected" in res.reason


def test_loop_stops_and_restores_when_a_change_does_not_help():
    hw, specs = specs8()
    mock = _mock(hw, specs)
    lay = Layout.parse("agent=0+1/2+3,llm=4+5,tts=6,clone=7")
    mock.start(lay.placement)
    res = run(loop(lay, lambda lay_, meter: mock.runner(meter), specs, hw, apply=mock.apply_layout, caps=False,
                   monitor_for=lambda lay_: Monitor(sample_fn=mock.sample, interval=0.006), levels=(2, 4, 8), seconds=0.3, scale=1 / TS,
                   log=lambda *a: None))
    assert res.recommendation.alloc == lay.alloc and "already matches" in res.reason and len(res.rounds) == 1


def test_loop_does_not_restart_for_a_proposal_that_projects_no_gain(monkeypatch):
    """Seen on an 8-GPU host: the agent was the bottleneck with no GPU left for another agent replica, so the proposal only
    added a clone replica, projected at the measured rate. Restarting for that buys nothing."""
    from interaction_gym.gpu_tuner import rebalance

    hw, specs = specs8()
    mock = _mock(hw, specs)
    lay = Layout.parse("agent=0+1/2+3,llm=4+5,tts=6,clone=7")
    mock.start(lay.placement)
    real = rebalance.propose

    def same_rate(r, *a, **k):
        p = real(r, *a, **k)
        return replace(p, layout=Layout.parse("agent=0+1/2+3,llm=4+5,tts=6,clone=7/6"), projected_eph=r.best.eph)

    monkeypatch.setattr(rebalance, "propose", same_rate)
    res = run(loop(lay, lambda lay_, meter: mock.runner(meter), specs, hw, apply=mock.apply_layout,
                   monitor_for=lambda lay_: Monitor(sample_fn=mock.sample, interval=0.006), levels=(2, 4), seconds=0.3, scale=1 / TS,
                   log=lambda *a: None))
    assert len(res.rounds) == 1 and "projects no gain" in res.reason and res.recommendation.alloc == lay.alloc
    assert mock.applied == [lay.alloc]  # never restarted


# ---------------------------------------------------------------- launch config


def test_thinker_preset_one_gpu_agents():
    hw, specs = load_host("example-8gpu-thinker")
    _, audio = load_host()
    assert specs["agent"].gpus == 1 and specs["agent"].max_sessions == 16 and audio["agent"].gpus == 2
    assert {k: v for k, v in specs.items() if k != "agent"} == {k: v for k, v in audio.items() if k != "agent"}
    eps = endpoints(layout_plan(Layout.parse("agent=2/3/4/5,llm=0+1,tts=6/7"), 64), specs)
    assert [e["port"] for e in eps["agent"]] == [8010, 8011, 8012, 8013]
    assert eps["agent"][3]["cmd"] == "AGENT_LAYOUT=thinker GPUS=5 PORT=8013 MAX_SESSIONS=16 exec scripts/run_minicpmo.sh"


def test_launch_config_endpoints_and_script():
    hw, specs = specs8()
    lay = Layout.parse("agent=0+1/2+3,llm=4+5,tts=6/7,clone=6")
    plan = layout_plan(lay, 8, 900)
    eps = endpoints(plan, specs)
    assert [e["port"] for e in eps["agent"]] == [8010, 8011]
    assert [e["port"] for e in eps["tts"]] == [8002, 8003, 8001] and eps["tts"][-1]["backends"].endswith(":8003")
    assert eps["clone"][0]["port"] == 8005 and eps["llm"][0]["port"] == 8000
    assert eps["llm"][0]["cmd"] == "GPUS=4,5 DP=1 PORT=8000 MAX_NUM_SEQS=128 exec scripts/run_llm.sh"
    assert eps["agent"][1]["cmd"] == "AGENT_LAYOUT=audio GPUS=2,3 PORT=8011 MAX_SESSIONS=4 exec scripts/run_minicpmo.sh"
    assert fmt('x {"enable_thinking": false} {port}', port=1) == 'x {"enable_thinking": false} 1'  # JSON braces survive
    cfg = launch_config(plan, specs, hw)
    assert cfg["env"]["concurrent_episodes"] == 8 and len(cfg["env"]["agent_urls"]) == 2
    sh = launch_script(plan, specs, hw)
    assert "GPUS=2,3 PORT=8011" in sh and "tmux new-session -d -s dig_agent1" in sh and "tmux kill" not in sh and "stop.sh" not in sh


def test_fmt_only_fills_known_fields():
    assert fmt("a {gpu} {x} {port}", gpu=3, port=9) == "a 3 {x} 9"


# ---------------------------------------------------------------- predictive fallback


def test_curve_interpolates_and_saturates():
    c = Curve("call", [(1, 1.0), (4, 2.0)])
    assert c.unit_time(0.5) == 1.0 and c.unit_time(2.5) == pytest.approx(1.5)
    assert c.unit_time(8) == pytest.approx(4.0) and c.throughput(8) == pytest.approx(c.throughput(4))


def test_predictive_solver_agrees_with_the_loop_on_synthetic_profiles():
    hw, specs = specs8()
    plans = solve(specs, DemandProfile.synthetic(), SupplyProfile.synthetic(), hw)
    assert plans[0].alloc == {"agent": 2, "llm": 1, "tts": 1, "clone": 1} and plans[0].concurrency == 8


def test_propose_adds_agent_sessions_when_session_bound_even_if_gpu_load_is_low():
    hw, specs = specs8()
    r = _round({"agent": (0.2, 0, 0.1), "llm": (0.5, 0, 0.6), "tts": (0.1, 0, 0.05), "clone": (0.2, 0, 0.2)})
    assert r.session_bound
    p = propose(r, specs, hw)
    assert p.layout.alloc["agent"] == 2 and p.concurrency > 4


def test_ssh_layout_local_waits_for_gpus_and_fails_fast(tmp_path, monkeypatch):
    """--ssh local runs on the host itself; the next layout starts only once the plan's GPUs are free, and a
    server that dies while loading raises with its log instead of waiting for its port."""
    import subprocess as sp

    from interaction_gym.gpu_tuner import remote

    hw, specs = specs8()
    hw = type(hw)(**{**hw.__dict__, "workdir": str(tmp_path)})
    calls, smi = [], iter(["4, 9000\n5, 100\n", "4, 200\n5, 100\n"])

    def fake_run(argv, **kw):
        cmd = argv[-1]
        calls.append(argv)
        if cmd.startswith("nvidia-smi"):
            return sp.CompletedProcess(argv, 0, next(smi, "4, 0\n5, 0\n"), "")
        if cmd.startswith("curl"):
            return sp.CompletedProcess(argv, 1, "", "")
        if cmd.startswith("tmux ls"):
            return sp.CompletedProcess(argv, 0, "dig_llm\n", "")
        return sp.CompletedProcess(argv, 0, "out of memory\n" if "tail" in cmd else "", "")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    s = remote.SSHLayout("local", specs, hw, "true", settle_s=0, scratch=tmp_path, log=lambda *a: None)
    plan = layout_plan(Layout.parse("agent=4+5,llm=0+1"), 4)
    with pytest.raises(RuntimeError, match=r"(?s)dig_agent.*out of memory"):
        asyncio.run(s.apply(plan))
    assert all(a[0] == "bash" for a in calls)  # no ssh
    assert sum(a[-1].startswith("nvidia-smi") for a in calls) == 2  # polled until GPU 4 was free


# ---------------------------------------------------------------- concurrency caps


def test_layout_caps_parse_roundtrip_and_session_capacity():
    from interaction_gym.gpu_tuner.rebalance import next_cap, session_capacity

    hw, specs = specs8()
    lay = Layout.parse("agent=0+1/2+3@6,llm=4+5,tts=6,clone=7")
    assert lay.caps == {"agent": 6} and str(lay) == "agent=0+1/2+3@6,llm=4+5,tts=6,clone=7"
    assert lay.cap("agent", specs) == 6 and lay.cap("llm", specs) == 128 and lay.cap("tts", specs) is None
    assert session_capacity(lay, specs) == 12
    base = Layout.parse("agent=0+1/2+3,llm=4+5,tts=6,clone=7")
    assert session_capacity(base, specs) == 8 and not base.same(lay, specs)
    assert base.same(Layout.parse("agent=0+1/2+3@4,llm=4+5,tts=6,clone=7"), specs)  # @4 is the spec's default
    assert base.with_cap("agent", 8).caps == {"agent": 8} and base.caps == {}
    assert next_cap(specs["agent"], 4) == 6 and next_cap(specs["agent"], 8) == 12 and next_cap(specs["agent"], 12) is None


def test_launch_carries_caps_into_commands_and_yaml():
    hw, specs = specs8()
    plan = layout_plan(Layout.parse("agent=0+1/2+3@8,llm=4+5@256,tts=6,clone=7"), 16)
    eps = endpoints(plan, specs)
    assert eps["agent"][0]["cmd"] == "AGENT_LAYOUT=audio GPUS=0,1 PORT=8010 MAX_SESSIONS=8 exec scripts/run_minicpmo.sh"
    assert "MAX_NUM_SEQS=256" in eps["llm"][0]["cmd"] and eps["clone"][0]["cmd"].startswith("MAX_NUM_SEQS=64 ")
    assert "{cap}" not in launch_script(plan, specs, hw)
    cfg = launch_config(plan, specs, hw)
    assert cfg["services"]["agent"]["cap"] == 8 and "cap" not in cfg["services"]["tts"]


def _capbound_round(agent_util: float, *, cap=None, waiting=2.0):
    """The example tuned layout as measured on an 8x RTX 5090 host: both agent servers full, stage requests queued."""
    from interaction_gym.gpu_tuner.empirical import Level
    from interaction_gym.gpu_tuner.rebalance import session_capacity, session_service

    hw, specs = specs8()
    lay = Layout.parse("agent=0+1/2+3,llm=4+5,tts=6,clone=7")
    if cap:
        lay = lay.with_cap("agent", cap)
    held = lay.gpu_share()
    row = lambda s, busy, w, share: ServiceStats(s, lay.alloc[s], lay.gpus_of(s), held[s], busy, busy / held[s], None,  # noqa: E731
                                                 None, w, None, share * 40, share, 1, cap=lay.cap(s, specs))
    stats = {"agent": row("agent", agent_util * 4, waiting, 0.8), "llm": row("llm", 0.5, 0.0, 0.05),
             "tts": row("tts", 0.04, 0.0, 0.01), "clone": row("clone", 0.3, 0.0, 0.05)}
    n = session_capacity(lay, specs)
    r = Round(lay, [Level(n // 2, 10, 60, 450, 55), Level(n, 20, 60, 690, 55)], stats, session_cap=n,
              session_service=session_service(lay, specs), caps={s: lay.cap(s, specs) for s in lay.alloc if lay.cap(s, specs)})
    return analyze(r), specs, hw


def test_propose_raises_the_cap_when_concurrency_bound_with_low_util():
    r, specs, hw = _capbound_round(0.33)
    assert r.bottleneck == "agent" and r.cap_bound == ["agent"]
    p = propose(r, specs, hw)
    assert p.action == "cap" and p.service == "agent" and p.layout.caps == {"agent": 6}
    assert p.layout.alloc == r.layout.alloc and p.layout.placement == r.layout.placement  # no GPU moves
    assert p.concurrency == 12 and p.projected_eph == pytest.approx(690 * 12 / 8)  # session-limited, util allows more
    assert "33% util" in p.why and "before adding replicas" in p.why


def test_propose_adds_replicas_instead_when_util_is_high_or_the_cap_is_exhausted():
    r, specs, hw = _capbound_round(0.75)
    p = propose(r, specs, hw)
    assert p.action == "replicas" and "busy" in p.why  # high util: the cheaper knob would not help
    r, specs, hw = _capbound_round(0.33, cap=12)
    p = propose(r, specs, hw)
    assert p.action == "replicas" and "last step" in p.why and p.layout.caps == {"agent": 12}  # the tuned cap is kept
    r, specs, hw = _capbound_round(0.33, cap=8)
    p = propose(r, specs, hw, frozen={"agent": "GPU memory 95%"})
    assert p.action == "replicas" and "raising it failed" in p.why
    r, specs, hw = _capbound_round(0.33, waiting=0.0)
    r.levels[-1].eph = r.levels[-2].eph  # sessions full but nothing queues and episodes/hour no longer rose: not cap-bound
    r = analyze(r)
    assert "agent" not in r.cap_bound and propose(r, specs, hw).action == "replicas"


def test_guard_rejects_memory_failures_eviction_and_small_cap_gains():
    from interaction_gym.gpu_tuner.rebalance import guard

    r0, specs, hw = _capbound_round(0.33)
    p = propose(r0, specs, hw)
    r1, _, _ = _capbound_round(0.4, cap=6)
    r1.levels[-1].eph = 690 * 1.2
    assert guard(r1, r0, p) == ""
    r1.levels[-1].eph = 690 * 1.03
    assert "gain under 5%" in guard(r1, r0, p)
    r1.levels[-1].eph = 690 * 1.2
    r1.stats["agent"].mem_peak = 0.95
    assert "GPU memory 95%" in guard(r1, r0, p)
    r0.stats["agent"].mem_peak = 0.95
    assert guard(r1, r0, p) == ""  # as full as before: vLLM's static preallocation, not per-session growth
    r1.stats["agent"].mem_peak = 0.8
    r1.levels[-1].errors, r1.levels[-1].error_kinds = 2, ["RuntimeError: vLLM-Omni sent no input_audio_buffer.processed for 60 s"]
    assert "2 failed episodes" in guard(r1, r0, p) and "processed" in guard(r1, r0, p)
    r1.levels[-1].errors = 0
    r1.problems = ["dig_agent0.log: Stage 2 replica 0 is dead"]
    assert "stage eviction" in guard(r1, r0, p)


def test_loop_sweeps_the_session_cap_and_rolls_back_when_memory_runs_out():
    """Session-bound agent at low util whose throughput rises sublinearly with the cap and whose GPUs fill up per
    session: the loop raises the cap 4 -> 6 -> 8 -> 12, rejects 12 (memory > 92 %) and leaves 8 running."""
    from interaction_gym.gpu_tuner.empirical import session_bound_mock

    hw, specs = specs8()
    mock = session_bound_mock(specs, hw, time_scale=TS)
    lay = Layout.parse("agent=0+1/2+3,llm=4+5,tts=6,clone=7")
    mock.start(lay.placement)
    res = run(loop(lay, lambda lay_, meter: mock.runner(meter), specs, hw, apply=mock.apply_layout,
                   monitor_for=lambda lay_: Monitor(sample_fn=mock.sample, interval=0.006), levels=(2, 4, 8, 16), seconds=0.4,
                   scale=1 / TS, rounds=6, log=lambda *a: None))
    assert res.actions[:4] == ["start", "cap", "cap", "cap"]
    caps = [r.layout.cap("agent", specs) for r in res.rounds]
    assert caps[:4] == [4, 6, 8, 12]
    assert [lv.concurrency for lv in res.rounds[1].levels] == [8, 12]  # a cap round starts at the previous best level
    assert "GPU memory" in res.rounds[3].guard and "GPU memory" in res.frozen["agent"]
    assert res.recommendation.cap("agent", specs) == 8 and res.concurrency == 16
    assert mock.applied_caps[-1] == {"agent": 8}  # the best measured setting is the one left running
    e = [r.best.eph for r in res.rounds]
    assert e[1] > 1.1 * e[0] and e[2] > 1.05 * e[1]  # more sessions help ...
    assert e[2] / 16 < e[0] / 8  # ... sublinearly
    assert all(r.layout.alloc == lay.alloc for r in res.rounds)  # no GPU moved


def test_loop_without_cap_tuning_only_moves_gpus():
    from interaction_gym.gpu_tuner.empirical import session_bound_mock

    hw, specs = specs8()
    mock = session_bound_mock(specs, hw, time_scale=TS)
    lay = Layout.parse("agent=0+1/2+3,llm=4+5,tts=6,clone=7")
    mock.start(lay.placement)
    res = run(loop(lay, lambda lay_, meter: mock.runner(meter), specs, hw, apply=mock.apply_layout, caps=False,
                   monitor_for=lambda lay_: Monitor(sample_fn=mock.sample, interval=0.006), levels=(4, 8), seconds=0.3,
                   scale=1 / TS, log=lambda *a: None))
    assert all(p.action == "replicas" for p in res.proposals) and all(not r.layout.caps for r in res.rounds)


def test_loop_restarts_and_remeasures_when_the_servers_died():
    """An external kill (some shared hosts kill long-running services): if the servers are gone after a round, restart the layout and measure it again."""
    hw, specs = specs8()
    mock = _mock(hw, specs)
    lay = Layout.parse("agent=0+1/2+3,llm=4+5,tts=6,clone=7")
    mock.start(lay.placement)
    state = {"checks": 0}

    def alive(lay_):
        state["checks"] += 1
        return state["checks"] > 1  # dead after the first round, fine after the restart

    res = run(loop(lay, lambda lay_, meter: mock.runner(meter), specs, hw, apply=mock.apply_layout, alive=alive,
                   monitor_for=lambda lay_: Monitor(sample_fn=mock.sample, interval=0.006), levels=(4, 8), seconds=0.3,
                   scale=1 / TS, rounds=1, log=lambda *a: None))
    assert len(res.rounds) == 1 and len(mock.applied) == 2 and state["checks"] == 2


def test_run_batch_records_failures_and_stops_a_broken_batch():
    async def bad(i):
        await asyncio.sleep(0.001)
        raise RuntimeError("vLLM-Omni sent no input_audio_buffer.processed for 60 s (append 3 of 9)\nmore")

    lv = run(run_batch(bad, 2, seconds=5))
    assert lv.aborted and lv.errors in (4, 5) and lv.episodes == 0 and lv.eph == 0
    assert lv.error_kinds[0] == "RuntimeError: vLLM-Omni sent no input_audio_buffer.processed for 60 s (append 3 of 9)"


def test_log_watch_reports_new_eviction_lines_only(tmp_path):
    import subprocess as sp

    from interaction_gym.gpu_tuner.monitor import LogWatch

    log = tmp_path / "dig_agent0.log"
    log.write_text("INFO old: Stage 2 replica 0 is dead\n")
    w = LogWatch(lambda cmd: sp.run(["bash", "-c", cmd], capture_output=True, text=True).stdout, [str(log)])
    w.mark()
    assert w.check() == []
    with open(log, "a") as f:
        f.write("INFO fine\nERROR [StagePool] no live replica for stage 2\n")
    assert w.check() == ["dig_agent0.log: ERROR [StagePool] no live replica for stage 2"]
