"""Benchmark adapters (Full-Duplex-Bench) on tiny synthetic fixtures — no real benchmark data in the repo."""

import asyncio
import json
import math
import struct
from array import array

import pytest

from interaction_gym import AgentSpec
from interaction_gym.audio import Audio
from interaction_gym.benchmarks import clip_bank
from interaction_gym.benchmarks import full_duplex_bench as fdb
from interaction_gym.benchmarks.wav import read_wav, resample, speech_segments
from interaction_gym.traj import episode

SR = 16000


def tone(spans, total_s, sr=SR, amp=8000):
    """Silence with a 220 Hz tone over each (start_s, end_s)."""
    x = [0] * round(total_s * sr)
    for a, b in spans:
        for i in range(round(a * sr), round(b * sr)):
            x[i] = int(amp * math.sin(2 * math.pi * 220 * i / sr))
    return x


def write_pcm16(path, x, sr=SR):
    Audio(array("h", x), sr).write_wav(path)


def write_float(path, x, sr=SR):
    data = struct.pack(f"<{len(x)}f", *(v / 32768 for v in x))
    fmt = struct.pack("<HHIIHH", 3, 1, sr, sr * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def words(spec):
    return [{"text": w, "timestamp": [a, b]} for w, a, b in spec]


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "fdb"
    # pause handling (float WAV like the released synthetic set): "I need | a book", pause 1.0–1.6
    d = r / "v1.0/synthetic_pause_handling/1"
    d.mkdir(parents=True)
    write_float(d / "input.wav", tone([(0.0, 1.0), (1.6, 2.4)], 3.0))
    (d / "pause.json").write_text(json.dumps([{"text": "[PAUSE]", "timestamp": [1.0, 1.6]}]))
    (d / "transcription.json").write_text(json.dumps(words([("I", 0.0, 0.3), ("need", 0.3, 1.0), ("a", 1.6, 1.8), ("book", 1.8, 2.4)])))
    # turn taking: speech 0.2–1.5, turn end annotated at 1.5
    d = r / "v1.0/candor_turn_taking/1"
    d.mkdir(parents=True)
    write_pcm16(d / "input.wav", tone([(0.2, 1.5)], 3.0))
    (d / "turn_taking.json").write_text(json.dumps([{"text": "[TURN-TAKING]", "timestamp": [1.5, 1.7]}]))
    # ICC: two word groups separated by a 1.5 s gap
    d = r / "v1.0/icc_backchannel/0"
    d.mkdir(parents=True)
    write_pcm16(d / "input.wav", tone([(0.0, 1.0), (2.5, 3.5)], 4.0))
    (d / "transcription.json").write_text(json.dumps(words([("so", 0.0, 0.5), ("yes", 0.5, 1.0), ("and", 2.5, 3.0), ("then", 3.0, 3.5)])))
    (r / "v1.0/icc_backchannel/aggregated_all_data.json").write_text(json.dumps({"0": [[1.2, 1.5], [2.0, 2.3]]}))
    # v1.0 interruption: context 0–1.0 (context.wav 1.2 s), interruption 3.0–4.0, input 6 s
    d = r / "v1.0/synthetic_user_interruption/1"
    d.mkdir(parents=True)
    write_pcm16(d / "input.wav", tone([(0.0, 1.0), (3.0, 4.0)], 6.0))
    write_float(d / "context.wav", tone([(0.0, 1.0)], 1.2))
    write_float(d / "interrupt.wav", tone([(0.0, 1.0)], 1.0))
    (d / "interrupt.json").write_text(json.dumps([{"context": "Tell me about cats.", "interrupt": "What about dogs?", "timestamp": [3.0, 4.0]}]))
    # v1.5 backchannel: context 0–1.0, backchannel 3.0–3.5
    for i in ("1", "2"):
        d = r / "v1.5/user_backchannel" / i
        d.mkdir(parents=True)
        write_pcm16(d / "input.wav", tone([(0.0, 1.0), (3.0, 3.5)], 5.0))
        write_pcm16(d / "clean_input.wav", tone([(0.0, 1.0)], 5.0))
        write_pcm16(d / "context.wav", tone([(0.0, 1.0)], 1.0))
        write_pcm16(d / "backchannel.wav", tone([(0.0, 0.5)], 0.5))
        (d / "metadata.json").write_text(json.dumps({"context_text": "Explain tides.", "backchannel_text": "uh huh", "timestamps": [3.0, 3.5]}))
    return r


def mix(sample):
    """What the agent's microphone carries: turns placed at their times + the background track."""
    n = round(sample.duration_ms * SR / 1000)
    acc = list(sample.background[0].audio.samples) if sample.background else [0] * n
    for t in sample.turns:
        off = round(t["t"] * SR / 1000)
        for i, v in enumerate(t["audio"].samples):
            acc[off + i] += v
    return acc


# ---------------------------------------------------------------- audio helpers


def test_read_wav_float_and_pcm(tmp_path):
    x = tone([(0.0, 0.1)], 0.2)
    write_float(tmp_path / "f.wav", x)
    write_pcm16(tmp_path / "p.wav", x)
    f, p = read_wav(tmp_path / "f.wav"), read_wav(tmp_path / "p.wav")
    assert f.sr == p.sr == SR and max(abs(a - b) for a, b in zip(f.samples, p.samples)) <= 1


def test_resample_and_vad():
    a = Audio(array("h", tone([(0.5, 1.5)], 2.0, sr=48000)), 48000)
    b = resample(a, SR)
    assert b.sr == SR and abs(b.dur_ms - 2000) <= 1
    (s, e), = speech_segments(b)
    assert abs(s - 0.47) < 0.05 and abs(e - 1.53) < 0.05


# ---------------------------------------------------------------- loading → turns


def test_pause_handling_turns_wait_then_normal(root):
    s = fdb.load_sample(root, "v1.0/synthetic_pause_handling", "1")
    assert [(t["t"], t.get("expects"), t["text"]) for t in s.turns] == [(0, "wait", "I need"), (1600, None, "a book")]
    assert abs(s.turns[0]["audio"].dur_ms - 1000) <= 20 and s.duration_ms == 3000
    assert s.task.scenario["benchmark"]["annotation"]["pause.json"] == [[1.0, 1.6]]
    assert mix(s) == list(fdb.read_wav(root / "v1.0/synthetic_pause_handling/1/input.wav").samples)  # exact input


def test_turn_taking_turn_ends_at_annotation(root):
    s = fdb.load_sample(root, "v1.0/candor_turn_taking", "1")
    (t,) = s.turns
    assert abs(t["t"] - 200) <= 20 and t["t"] + t["audio"].dur_ms == 1500 and "expects" not in t


def test_icc_split_into_wait_turns(root):
    s = fdb.load_sample(root, "v1.0/icc_backchannel", "0")
    assert [(t["t"], t.get("expects"), t["text"]) for t in s.turns] == [(0, "wait", "so yes"), (2500, None, "and then")]


def test_interruptions_and_v15_kinds(root):
    s = fdb.load_sample(root, "v1.0/synthetic_user_interruption", "1")
    assert [(t["text"], t.get("expects")) for t in s.turns] == [("Tell me about cats.", None), ("What about dogs?", "yield")]
    assert abs(s.turns[1]["t"] - 3000) <= 20
    s = fdb.load_sample(root, "v1.5/user_backchannel", "1")
    assert [(t["text"], t.get("kind")) for t in s.turns] == [("Explain tides.", None), ("uh huh", "backchannel")]
    assert mix(s) == list(fdb.read_wav(root / "v1.5/user_backchannel/1/input.wav").samples)
    c = fdb.load_sample(root, "v1.5/user_backchannel", "1", clean=True)
    assert len(c.turns) == 1 and c.id.endswith("-clean")


def test_replay_episode_runs_for_the_input_length(root):
    s = fdb.load_sample(root, "v1.5/user_backchannel", "1")
    spec = AgentSpec(chunk_ms=200, audio="user.audio", sr=SR)
    env = fdb.make_env(s, spec)

    async def go():
        obs, done = await env.reset(s.task), False
        while not done:
            obs, _, done = await env.step([])
    asyncio.run(go())
    ep = episode(env, s.id)
    assert ep["meta"]["duration_ms"] == 5000
    assert [t.get("kind") for t in ep["turns"]] == [None, "backchannel"]
    assert ep["eval"]["scores"]["by_expectation"]["ignore"] == 1.0  # a silent agent ignores the backchannel
    assert fdb.bench_meta(ep)["task"] == "user_backchannel"


def test_split_of_is_stable():
    ids = [str(i) for i in range(400)]
    m = [i for i in ids if fdb.split_of("v1.5/user_backchannel", i, 0.5) == "material"]
    assert 150 < len(m) < 250 and m == [i for i in ids if fdb.split_of("v1.5/user_backchannel", i, 0.5) == "material"]
    assert all(fdb.split_of("x", i) == "eval" for i in ids)


# ---------------------------------------------------------------- official metric ports


def ep_of(subset, ann, agent, user=(), dur_ms=10000, sid="0", clean=False):
    version, task, _ = fdb.SUBSETS[subset]
    turns = [{"id": f"u{i}", "role": "user", "start_time": a, "end_time": b, "text": txt} for i, (a, b, txt) in enumerate(user)]
    turns += [{"id": f"a{i}", "role": "agent", "start_time": a, "end_time": b, "text": txt} for i, (a, b, txt) in enumerate(agent)]
    bench = {"version": version, "subset": subset, "task": task, "sample_id": sid, "duration_ms": dur_ms, "annotation": ann, "clean": clean}
    return {"meta": {"episode_id": f"{subset}-{sid}", "task": {"scenario": {"benchmark": bench}}}, "turns": sorted(turns, key=lambda t: t["start_time"])}


def test_take_turn_rule():
    assert fdb._tor([]) == 0
    ch = lambda n, d: [{"text": "w", "timestamp": [i * d / n, (i + 1) * d / n]} for i in range(n)]  # noqa: E731
    assert fdb._tor(ch(3, 0.9)) == 0 and fdb._tor(ch(4, 0.9)) == 1 and fdb._tor(ch(2, 1.2)) == 1


def test_pause_and_turn_taking_metrics():
    ph = "v1.0/synthetic_pause_handling"
    assert fdb.sample_metrics(ep_of(ph, {}, [(1200, 1500, "mm hmm")]))["TOR"] == 0  # a short backchannel is not a turn
    assert fdb.sample_metrics(ep_of(ph, {}, [(1200, 3000, "sure I can help")]))["TOR"] == 1
    assert fdb.sample_metrics(ep_of(ph, {}, [(9800, 12000, "late long reply here")], dur_ms=10000))["TOR"] == 0  # past the input: unseen
    tt = "v1.0/candor_turn_taking"
    ann = {"turn_taking.json": [{"timestamp": [1.5, 1.7]}]}
    m = fdb.sample_metrics(ep_of(tt, ann, [(2000, 4000, "yes it is my first time")]))
    assert m["TOR"] == 1 and m["latency"] == pytest.approx(0.5)
    assert fdb.sample_metrics(ep_of(tt, ann, [(1000, 4000, "yes it is my first time")]))["latency"] == 0.0  # negative → 0
    agg = fdb.official_metrics([ep_of(tt, ann, [(2000, 4000, "a b c d")], sid="1"), ep_of(tt, ann, [], sid="2")])
    assert agg["TOR"] == 0.5 and agg["latency"] == pytest.approx(0.5)


def test_user_interruption_crops_after_interrupt_end():
    ui = "v1.0/synthetic_user_interruption"
    ann = {"interrupt.json": [{"context": "c", "interrupt": "i", "timestamp": [3.0, 4.0]}]}
    # talking before and during the interruption (cut at 3.4), replying from 5.0
    m = fdb.sample_metrics(ep_of(ui, ann, [(1500, 3400, "one two three four"), (5000, 7000, "sure dogs are great pets")]))
    assert m["TOR"] == 1 and m["latency"] == pytest.approx(1.0) and m["response_text"] == "sure dogs are great pets"


def test_backchannel_port_and_jsd():
    bc = "v1.0/icc_backchannel"
    m = fdb.sample_metrics(ep_of(bc, {}, [(1000, 1400, "yeah"), (5000, 5300, "mhm")], dur_ms=10000))
    assert m["TOR"] == 0 and m["freq"] == pytest.approx(0.2) and m["backchannels"] == [[1.0, 1.4], [5.0, 5.3]]
    m = fdb.sample_metrics(ep_of(bc, {}, [(1000, 1400, "yeah"), (5000, 9000, "let me tell you something")], dur_ms=10000))
    assert m["TOR"] == 1 and len(m["backchannels"]) == 1  # a > 3 s segment breaks the loop
    # official quirk: a later short backchannel resets TOR to 0
    m = fdb.sample_metrics(ep_of(bc, {}, [(1000, 2500, "well I think that"), (5000, 5300, "mhm")], dur_ms=10000))
    assert m["TOR"] == 0
    assert fdb.jensenshannon([1, 0], [0, 1]) == pytest.approx(math.sqrt(math.log(2)))
    assert fdb.backchannel_jsd([], 10, [0.5, 0.5]) == 1.0
    assert fdb.backchannel_jsd([[0.0, 0.1]], 0.3, [0.5, 0.5]) == pytest.approx(fdb.jensenshannon([1, 0], [0.5, 0.5]), abs=1e-6)  # bins [1, 0]
    assert fdb.backchannel_jsd([[0.0, 0.3]], 0.3, [0.5, 0.5]) == pytest.approx(0.0, abs=1e-6)  # bins [1, 1]
    agg = fdb.official_metrics([ep_of(bc, {}, [(1000, 1400, "yeah")], sid="0")], gt_distribution={"0": [1.0] * 51})
    assert 0 < agg["JSD"] < 1


def test_v15_timing_port():
    user = [(0.0, 1.0), (3.0, 4.0)]
    model = [(1.5, 3.4), (5.0, 6.0)]
    assert fdb.overlaps(user, model) == [[3.0, 3.4]]
    assert fdb.response_gaps(user, model) == [[1.0, 1.5], [4.0, 5.0]]
    sub = "v1.5/user_interruption"
    ann = {"metadata.json": {"timestamps": [3.0, 4.0]}}
    ep = ep_of(sub, ann, [(1500, 3400, "a b c"), (5000, 6000, "d e")], user=[(0, 1000, "q"), (3000, 4000, "i")])
    m = fdb.sample_metrics(ep)
    assert m["stop"] == pytest.approx(0.4) and m["resp"] == pytest.approx(1.0)
    agg = fdb.official_metrics([ep], behaviours={ep["meta"]["episode_id"]: "C_RESPOND"})
    assert agg["stop"] == pytest.approx(0.4) and agg["behaviour"] == {"C_RESPOND": 1.0}


def test_judges_parse_official_formats():
    class LLM:
        def __init__(self, reply):
            self.reply, self.seen = reply, None

        async def chat(self, messages, **kw):
            self.seen = messages
            return self.reply

    ui = "v1.0/synthetic_user_interruption"
    ann = {"interrupt.json": [{"context": "c", "interrupt": "dogs?", "timestamp": [3.0, 4.0]}]}
    ep = ep_of(ui, ann, [(5000, 7000, "sure dogs are great pets")])
    llm = LLM("Analysis: fine.\nI would rate the AI's response as 4")
    assert asyncio.run(fdb.judge_interruption(ep, llm)) == 4 and "dogs?" in llm.seen[1]["content"]
    sub = "v1.5/talking_to_other"
    ann = {"metadata.json": {"timestamps": [3.0, 4.0]}}
    noisy = ep_of(sub, ann, [(5000, 6000, "ok")], user=[(0, 1000, "q"), (3000, 4000, "coach hi")])
    clean = ep_of(sub, ann, [(1500, 3000, "answer")], user=[(0, 1000, "q")], clean=True)
    assert asyncio.run(fdb.judge_behaviour(noisy, clean, LLM('ok { "behaviour": ["C_RESUME"] }'), "instr")) == "C_RESUME"


# ---------------------------------------------------------------- clip bank


def test_clip_bank_extract_and_splits(root, tmp_path):
    out = tmp_path / "bank"
    path = clip_bank.extract(root, out, material_frac=1.0, subsets=["v1.5/user_backchannel"])
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    bcs = [r for r in recs if r["kind"] == "backchannel"]
    assert len(bcs) == 2 and all(r["split"] == "material" and r["source"]["license"] == "MIT" for r in recs)
    assert len([r for r in recs if r["use"] == "opening"]) == 1  # the same opening text twice: deduplicated
    assert abs(bcs[0]["dur_ms"] - 500) < 80 and (out / bcs[0]["audio"]).exists()
    assert len(clip_bank.load_manifest(path, kind="backchannel")) == 2
    path = clip_bank.extract(root, tmp_path / "bank0", material_frac=0.0, subsets=["v1.5/user_backchannel"])
    assert path.read_text() == ""  # everything is held out for evaluation


def test_human_stats(root):
    st = clip_bank.human_stats(root)
    p = st["pauses"]["v1.0/synthetic_pause_handling"]
    assert p["pause_s"]["n"] == 1 and p["pause_s"]["mean"] == pytest.approx(0.6) and p["words_before_pause"]["mean"] == 2
    b = st["backchannels"]
    assert b["duration_s"]["n"] == 2 and b["fraction_in_speaker_pause"] == 1.0
    assert st["interruptions"]["v1.0/synthetic_user_interruption"]["onset_after_context_end_s"]["mean"] == pytest.approx(1.8)


def test_shared_content_shares_a_split():
    for i in map(str, range(100)):
        assert fdb.split_of("v1.0/synthetic_user_interruption", i, 0.5) == fdb.split_of("v1.5/user_interruption", i, 0.5)


def test_clip_bank_drops_material_repeating_eval_text(root, tmp_path, monkeypatch):
    # sample 1 material, sample 2 eval, both say "Explain tides.": the material opening must go
    monkeypatch.setattr(clip_bank, "split_of", lambda subset, sid, frac: "material" if sid == "1" else "eval")
    monkeypatch.setattr(fdb, "split_of", lambda subset, sid, frac=0.0: "material" if sid == "1" else "eval")
    recs = [json.loads(x) for x in clip_bank.extract(root, tmp_path / "b", subsets=["v1.5/user_backchannel"]).read_text().splitlines()]
    assert [r["use"] for r in recs] == ["clip"]


def test_clip_bank_seeds_online_behaviors(root, tmp_path):
    path = clip_bank.extract(root, tmp_path / "bank", material_frac=1.0, subsets=["v1.5/user_backchannel"])
    bh = clip_bank.behaviors(path, {"pauses": {"v1.0/candor_pause_handling": {"pause_s": {"n": 9, "p10": 0.68, "p90": 1.28}}}},
                             aside_per_min=3.0)
    texts = {json.loads(x)["text"].strip().lower().rstrip(".") for x in path.read_text().splitlines() if json.loads(x)["kind"] == "backchannel"}
    assert set(bh.backchannels) == texts and bh.pause_ms == (680, 1280) and bh.aside_per_min == 3.0
    assert bh.asides == clip_bank.behaviors(path).asides  # no aside clips: the defaults stay


# ---------------------------------------------------------------- Easy Turn


@pytest.fixture
def et_root(tmp_path):
    r = tmp_path / "et"
    spec = {"complete": [("complete_real_001", "你好吗？<COMPLETE>", "G1", 1.0), ("complete_renzao_1", "几点了<COMPLETE>", "ZH_B1", 0.8)],
            "incomplete": [("incomplete_real_001", "因为小时候<INCOMPLETE>", "G2", 0.6)],
            "backchannel": [("backchannel_real_001", "嗯，也是<BACKCHANNEL>", "G1", 0.4)],
            "wait": [("wait_renzao_1", "别说了<WAIT>", "<NONE>", 0.5)]}
    for state, recs in spec.items():
        d = r / "testset" / state
        lines = []
        for key, txt, spk, dur in recs:
            sub = "real" if "real" in key else "synthetic"
            (d / sub).mkdir(parents=True, exist_ok=True)
            write_pcm16(d / sub / f"{key}.wav", tone([(0.2, 0.2 + dur)], dur + 0.6))  # 0.2 s silence before, 0.4 s after
            lines.append(json.dumps({"key": key, "wav": f"./{state}/{sub}/{key}.wav", "txt": txt, "speaker": spk, "duration": dur + 0.6}))
        (d / f"{state}_test.list").write_text("\n".join(lines) + "\n")
    return r


def test_easy_turn_episodes(et_root):
    from interaction_gym.benchmarks import easy_turn as et

    s = et.load(et_root, "easy_turn/complete")[0]
    assert len(s.turns) == 1 and s.turns[0]["text"] == "你好吗？" and "expects" not in s.turns[0]
    assert abs(s.turns[0]["audio"].dur_ms - 1060) < 40 and s.duration_ms == 500 + s.turns[0]["audio"].dur_ms + 5000
    s = et.load(et_root, "easy_turn/incomplete")[0]
    assert s.turns[0]["expects"] == "wait" and s.duration_ms == 500 + s.turns[0]["audio"].dur_ms + 3000
    s = et.load(et_root, "easy_turn/backchannel")[0]
    op, bc = s.turns
    assert op["text"] == "你好吗？" and bc["kind"] == "backchannel"  # same speaker G1 -> that opening
    assert bc["t"] == op["t"] + op["audio"].dur_ms + 4000
    s = et.load(et_root, "easy_turn/wait")[0]
    assert s.turns[0]["text"] == "几点了" and s.turns[1]["expects"] == "wait"  # no speaker: synthetic opening
    meta = s.task.scenario["benchmark"]
    assert meta["license"] == "Apache-2.0" and meta["annotation"]["opening"]["key"] == "complete_renzao_1"
    env = et.make_env(s, AgentSpec(chunk_ms=200, audio="user.audio", sr=SR))

    async def go():
        await env.reset(s.task, seed=0)
        done = False
        while not done:
            _, _, done = await env.step([])
        return episode(env, s.id)

    ep = asyncio.run(go())
    assert ep["meta"]["duration_ms"] == s.duration_ms and [t.get("expects") for t in ep["turns"] if t["role"] == "user"] == [None, "wait"]
    by = ep["eval"]["scores"]["by_expectation"]
    assert by["wait"] == 1.0 and "yield" not in by  # a silent agent: it waited, and there was nothing to yield


# ---------------------------------------------------------------- HumDial-FDBench


@pytest.fixture
def hd_root(tmp_path):
    r = tmp_path / "hd"

    def sample(cat, sid, segs, total, clean_segs=None, words=()):
        d = r / "test/en_test_nondev" / cat
        d.mkdir(parents=True, exist_ok=True)
        write_pcm16(d / f"{sid}.wav", tone([(a, b) for a, b, _ in segs], total))
        (d / f"{sid}.json").write_text(json.dumps({"final_duration": total, "speech_segments": [{"xmin": a, "xmax": b, "text": t} for a, b, t in segs]}))
        (d / f"{sid}_timestamp.json").write_text(json.dumps({"chunks": [{"text": w, "timestamp": [a, b]} for w, a, b in words]}))
        if clean_segs is not None:
            write_pcm16(d / f"clean_{sid}.wav", tone([(a, b) for a, b, _ in clean_segs], total))
            (d / f"clean_{sid}.json").write_text(json.dumps({"speech_segments": [{"xmin": a, "xmax": b, "text": t} for a, b, t in clean_segs]}))

    sample("ask", "0001_0001", [(0.0, 1.0, "Tell me about cats."), (6.0, 7.0, "And dogs?")], 12.0, clean_segs=[(0.0, 1.0, "Tell me about cats.")])
    sample("pause", "0001_0002", [(0.0, 2.5, "Could you add [break] some bananas?")], 8.0,
           words=[("Could", 0.0, 0.3), ("add", 0.3, 0.8), ("some", 1.6, 2.0), ("bananas", 2.0, 2.5)])
    sample("talk_to_others", "0001_0003_add", [(0.0, 1.0, "q"), (6.0, 7.0, "say again?"), (12.0, 13.0, "nice weather")], 20.0)
    sample("others_talk_to_user_before", "0001_0004", [(0.0, 1.0, "did you ice your ankle?"), (6.0, 7.0, "q")], 12.0)
    return r


def test_humdial_episodes(hd_root):
    from interaction_gym.benchmarks import humdial as hd

    s = hd.load(hd_root, "humdial/en/ask")[0]
    assert [(t.get("kind"), t.get("expects")) for t in s.turns] == [(None, None), (None, "yield")]
    assert s.turns[1]["t"] == 6000 and s.duration_ms == 12000 and s.task.scenario["benchmark"]["license"] == "Apache-2.0"
    c = hd.load_sample(hd_root, "humdial/en/ask", "0001_0001", clean=True)
    assert len(c.turns) == 1 and c.id.endswith("-clean")
    p = hd.load(hd_root, "humdial/en/pause")[0]
    assert [t["text"] for t in p.turns] == ["Could you add", "some bananas?"] and p.turns[0]["expects"] == "wait"
    assert abs(p.turns[0]["audio"].dur_ms - 800) < 40 and abs(p.turns[1]["t"] - 1600) < 40
    t = hd.load(hd_root, "humdial/en/talk_to_others")[0]
    assert [x.get("kind") for x in t.turns] == [None, None, "aside"]
    b = hd.load(hd_root, "humdial/en/others_talk_to_user_before")[0]
    assert [x.get("kind") for x in b.turns] == ["noise", None]
    with pytest.raises(ValueError):
        hd.load_sample(hd_root, "humdial/en/pause", "0001_0002", clean=True)


def test_clip_bank_from_humdial(hd_root, tmp_path):
    path = clip_bank.extract_humdial(hd_root, tmp_path / "bank", material_frac=1.0, langs=("en",), append=False)
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    by = {(r["use"], r["kind"]) for r in recs}
    assert by == {("opening", None), ("interruption", None), ("clip", "aside"), ("clip", "noise")}
    intr = next(r for r in recs if r["use"] == "interruption")
    assert intr["intent"] == "ask" and intr["text"] == "And dogs?" and abs(intr["dur_ms"] - 1060) < 80
    assert all(r["lang"] == "en" and r["source"]["license"] == "Apache-2.0" and not r["source"]["synthetic"] for r in recs)
    assert [r["kind"] for r in clip_bank.load_manifest(path, kind="noise", lang="en")] == ["noise"]
    assert clip_bank.load_manifest(path, lang="zh") == []
    assert clip_bank.humdial_split("en", "0001_0001", 0.5) == clip_bank.humdial_split("en", "0001_0099", 0.5)  # same speaker
    assert clip_bank.extract_humdial(hd_root, tmp_path / "b0", material_frac=0.0, append=False).read_text() == ""


def test_humdial_eval_split_matches_clip_bank(hd_root):
    from interaction_gym.benchmarks import humdial as hd

    all_ids = hd.ids(hd_root, "humdial/en/ask")
    assert hd.ids(hd_root, "humdial/en/ask", split="eval", material_frac=0.0) == all_ids
    assert hd.ids(hd_root, "humdial/en/ask", split="eval", material_frac=1.0) == []
    assert clip_bank.humdial_split("en", "0001_0001", 0.5) == hd.split_of("en", "0001_0001", 0.5)
