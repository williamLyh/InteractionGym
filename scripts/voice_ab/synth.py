"""Voice A/B, synthesis stage: re-synthesize the user turns of a fixed set of episodes under each voice variant.

Text, speaker and order are held fixed (taken from out/run_places.jsonl, a text-only run); turns are rendered one after
another per episode through interaction_gym.user.Voice exactly as an episode would (the clone reference is
the first normal turn with >= 3 words). Resumable: an episode/variant with meta.json is skipped.

Outputs: <out>/<variant>/<ep>/NN.wav, concat.wav (turns joined with 500 ms gaps), meta.json (per-turn text, kind,
duration, latency, cloned?).
"""
import argparse, asyncio, json, random, time
from pathlib import Path

from interaction_gym import Task
from interaction_gym.audio import Audio
from interaction_gym.clients import OpenAISpeech
from interaction_gym.user import QWEN3_TTS_VOICES, Leveling, UserTurn, Voice

SR = 24000
GAP_MS = 500


def variants(tts, clone):
    lv = Leveling()
    return {
        "V0": dict(style="persona"),                                        # current: CustomVoice per turn + persona style
        "V1": dict(style="persona", leveling=lv),                           # + trim / loudness
        "V2": dict(style="neutral", leveling=lv),                           # + neutral instruction
        "C0": dict(style="persona", clone=clone),                           # clone, no post-processing
        "V3": dict(style="persona", leveling=lv, clone=clone),              # V1 + clone
        "V4": dict(style="neutral", leveling=lv, clone=clone),              # V1 + neutral reference + clone
        "V5": dict(style="neutral", leveling=lv, clone=clone, seed=True),   # V4 + one sampling seed per episode
    }


def pick_episodes(path, n):
    """n episodes with >= 4 normal user turns, round-robin over surroundings, distinct persona names first."""
    eps = [json.loads(l) for l in open(path)]
    ok = [e for e in eps if sum(1 for t in e["turns"] if t["role"] == "user" and t.get("kind") is None and t["text"].strip()) >= 4]
    rng = random.Random(0)
    rng.shuffle(ok)
    name = lambda e: e["meta"]["task"]["scenario"]["profile"]["name"]
    place = lambda e: e["meta"]["task"]["scenario"]["profile"]["surroundings"]
    out, names = [], set()
    for unique in (True, False):
        for p in sorted({place(e) for e in ok}) * n:
            if len(out) >= n:
                break
            e = next((e for e in ok if place(e) == p and e not in out and (not unique or name(e) not in names)), None)
            if e is not None:
                out.append(e); names.add(name(e))
    return sorted(out, key=lambda e: e["meta"]["episode_id"])


def user_turns(ep):
    ts = [t for t in ep["turns"] if t["role"] == "user" and t.get("kind") in (None, "backchannel", "aside") and t["text"].strip()]
    return sorted(ts, key=lambda t: t["start_time"])


async def run_one(ep, name, cfg, out, sem):
    d = out / name / ep["meta"]["episode_id"]
    if (d / "meta.json").exists():
        return
    async with sem:
        d.mkdir(parents=True, exist_ok=True)
        task = Task(id=ep["meta"]["task"]["id"], scenario=ep["meta"]["task"]["scenario"])
        voice = Voice(cfg["tts"], voices=QWEN3_TTS_VOICES, **{k: v for k, v in cfg.items() if k != "tts"})
        ref, rows, parts = None, [], []
        for i, t in enumerate(user_turns(ep)):
            cloned = ref is not None
            t0 = time.perf_counter()
            for attempt in range(4):
                try:
                    seg = await voice.render(f"u{i}", 0, UserTurn(t["text"], kind=t.get("kind")), task, ref)
                    break
                except Exception as e:  # a server restart: wait and retry
                    if attempt == 3:
                        raise
                    await asyncio.sleep(20)
            lat = time.perf_counter() - t0
            if t.get("kind") is None:  # as UserSim: only the user's own lines set the reference
                ref = voice.reference(seg, ref)
            seg.data.write_wav(d / f"{i:02d}.wav")
            parts += [seg.data, Audio.silence(GAP_MS, SR)]
            rows.append({"i": i, "id": t["id"], "kind": t.get("kind"), "text": t["text"], "dur_ms": seg.data.dur_ms,
                         "latency_s": round(lat, 3), "cloned": cloned})
        cat = parts[0]
        for p in parts[1:]:
            cat = cat + p
        cat.write_wav(d / "concat.wav")
        meta = {"episode_id": ep["meta"]["episode_id"], "variant": name, "speaker": voice.speaker(task),
                "profile": task.scenario.get("profile"), "voice": voice.profile(task), "turns": rows}
        (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        print("done", name, ep["meta"]["episode_id"], flush=True)


async def main(a):
    out = Path(a.out)
    eps = pick_episodes(a.episodes, a.n)
    (out / "episodes.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "episodes.json").write_text(json.dumps([e["meta"]["episode_id"] for e in eps], indent=1))
    tts = OpenAISpeech(a.tts, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", sr=SR)
    clone = OpenAISpeech(a.clone, "Qwen/Qwen3-TTS-12Hz-1.7B-Base", sr=SR)
    vs = variants(tts, clone)
    sem = asyncio.Semaphore(a.concurrency)
    jobs = [run_one(e, n, {"tts": tts, **c}, out, sem) for n, c in vs.items() if n in a.variants.split(",") for e in eps]
    res = await asyncio.gather(*jobs, return_exceptions=True)
    bad = [r for r in res if isinstance(r, Exception)]
    for r in bad[:5]:
        print("ERROR", repr(r), flush=True)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="voice_ab_work/run_places.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="voice_ab_work/synth")
    ap.add_argument("--tts", default="http://127.0.0.1:8200/v1")
    ap.add_argument("--clone", default="http://127.0.0.1:8210/v1")
    ap.add_argument("--variants", default="V0,V1,V2,C0,V3,V4,V5")
    ap.add_argument("--concurrency", type=int, default=48)
    asyncio.run(main(ap.parse_args()))
