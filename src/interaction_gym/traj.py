"""Episode trajectories in the standard output format (schema v1, see docs/FORMAT.md).

- ``episode(env, ...)``: build the episode object from a finished (or running) Env.
- ``save`` / ``load``: a run is a JSONL file with one episode per line.
- ``to_frames``: back to a runtime event log (e.g. to replay as exogenous streams).
- ``agent_view``: what the agent heard at each step, derived from the turns.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from .audio import Audio
from .core import REWARD, SESSION, Env, Frame, Segment, Session
from .eval import duplex, scores
from .media import MediaStore
from .tools import CALL, CALLER, RESULT, ToolCall, ToolResult

SCHEMA = 1


def _jsonable(x: Any) -> Any:
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Audio):
        return {"audio_ms": x.dur_ms, "sr": x.sr}  # task data should reference media, not embed it
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return repr(x)


def role_of(stream: str) -> str:
    head = stream.split(".")[0]
    return "agent" if head == "policy" else head


def speech_stream(role: str) -> str:
    return "policy.speech" if role == "agent" else f"{role}.speech"


def _turns(log: list[Frame], media: MediaStore | None, duration: int) -> list[dict]:
    first: dict[tuple, Segment] = {}
    last: dict[tuple, Segment] = {}
    for f in log:
        if isinstance(f.data, Segment):
            key = (f.stream, f.data.id)
            first.setdefault(key, f.data)
            last[key] = f.data
    turns, ids = [], set()
    for key, seg in last.items():
        full = first[key] if first[key].end > seg.end else seg  # the longest version known (the plan, if it was cut)
        if seg.end > duration:  # still playing when the episode ended: it was cut there
            seg = seg.cut(duration)
        tid = seg.id if seg.id not in ids else f"{role_of(key[0])}:{seg.id}"
        ids.add(tid)
        t = {"id": tid, "role": role_of(key[0]), "start_time": seg.t0, "end_time": seg.end, "text": seg.transcript}
        if seg.kind:
            t["kind"] = seg.kind
        if seg.expects:
            t["expects"] = seg.expects
        if getattr(seg, "intent", None):
            t["intent"] = seg.intent
        if getattr(seg, "label", None):
            t["label"] = seg.label
        if full.end > seg.end and full.transcript[len(t["text"]) :]:
            t["unsaid"] = full.transcript[len(t["text"]) :]
        if media is not None and isinstance(full.data, Audio):
            t["media"] = media.ref(full.data, 0, seg.end - seg.t0)
        turns.append(t)
    return turns


def _attach_tool_calls(log: list[Frame], turns: list[dict]) -> None:
    caller_of = {stream: caller for caller, stream in RESULT.items()}
    results = {(caller_of[f.stream], f.data.id): f for f in log if f.stream in caller_of and isinstance(f.data, ToolResult)}  # ids are per caller
    for f in log:
        if f.stream not in CALLER or not isinstance(f.data, ToolCall):
            continue
        role, call, t = CALLER[f.stream], f.data, f.t
        mine = sorted((x for x in turns if x["role"] == role), key=lambda x: x["start_time"])
        host = next((x for x in mine if x["start_time"] <= t < x["end_time"]), None) or next(
            (x for x in mine if x["start_time"] >= t), None
        )
        if host is None:  # never spoke again: a silent turn carries the call
            host = {"id": f"{role}:{call.id}", "role": role, "start_time": t, "end_time": t, "text": ""}
            turns.append(host)
        rec = {"id": call.id, "name": call.name, "arguments": call.arguments, "call_time": t}
        res = results.get((role, call.id))
        if res is not None:
            rec.update(result_time=res.t, content=res.data.content)
            if res.data.error:
                rec["error"] = True
        host.setdefault("tool_calls", []).append(rec)


def _tool_dict(spec) -> dict:
    return {k: v for k, v in _jsonable(spec).items() if v not in ("", None)}


def _env_context(env: Env, log: list[Frame]) -> dict:
    """The environment's context: its components, and what it told the agent — the tool schemas
    available at t=0 (or the catalog to discover them), instructions, granted context, and the
    tools added later (with the time they became available)."""
    out: dict = {"components": {name: type(node).__name__ for name, node in env.nodes.items() if name != Env.POLICY}}
    if env.background_specs:  # the always-on background (context, no turn): what it is, even when nothing is mixed
        out["background"] = [dict(s) for s in env.background_specs]
    frames = [f for f in log if f.stream == SESSION]
    if not frames:
        return out
    first = frames[0].data
    out.update(
        tool_schema_mode=first.mode,
        instructions=first.instructions,
        tools=[_tool_dict(t) for t in first.tools],
        tool_services=list(first.services),
        agent_context=dict(first.context),
    )
    known, cur, updates = {t.name for t in first.tools}, first, []
    for f in frames[1:]:
        prev = cur
        if isinstance(f.data, Session):
            cur = f.data
        else:  # partial update from a node, merged the same way the agent receives it
            upd = {k: tuple(v) if isinstance(v, list) else v for k, v in f.data.items()}
            if "context" in upd:
                upd["context"] = {**cur.context, **upd["context"]}
            cur = dataclasses.replace(cur, **upd)
        added = [t for t in cur.tools if t.name not in known]
        upd = {"time": f.t}
        if added:
            upd["tools"] = [_tool_dict(t) for t in added]
        if cur.instructions != prev.instructions:
            upd["instructions"] = cur.instructions
        new_ctx = {k: v for k, v in cur.context.items() if prev.context.get(k) != v}
        if new_ctx:
            upd["context"] = new_ctx
        known |= {t.name for t in added}
        if len(upd) > 1:
            updates.append(upd)
    out["tool_updates"] = updates
    return {k: v for k, v in out.items() if v not in ("", [], {}, None)}


def _user(env: Env) -> dict | None:
    """The simulated user: the component that reports a ``profile`` (persona, goal, voice, models,
    barge-in and turn-taking policy). Only the producer knows these, so each user component reports its own."""
    for name, node in env.nodes.items():
        if hasattr(node, "profile"):
            out = {"component": name, **node.profile(env.sim.task)}
            if callable(getattr(node, "report", None)):  # what this episode's user did (decision counts, random events)
                out.update(node.report(env.sim.states[name]))
            return out
    return None


def _background(env: Env, media: MediaStore | None, duration: int) -> list[dict]:
    out = []
    for bg in env.background if media is not None else []:
        end = bg.end_time if bg.end_time is not None else duration
        span = end - bg.start_time
        stop = bg.offset_ms + (span if bg.loop else min(span, bg.audio.dur_ms - bg.offset_ms))
        rec = {"media": media.ref(bg.audio, bg.offset_ms, stop)}
        if bg.gain_db:
            rec["gain_db"] = bg.gain_db
        if bg.loop:
            rec["loop"] = True
        if bg.start_time:
            rec["start_time"] = bg.start_time
        if bg.end_time is not None:
            rec["end_time"] = bg.end_time
        if bg.spec:
            rec["spec"] = dict(bg.spec)
        out.append(rec)
    return out


def describe_agent(agent) -> dict:
    """``meta.agent``: ``agent.describe()`` (an agent describing itself), a dict as given, else the class name."""
    if isinstance(agent, dict):
        return dict(agent)
    if hasattr(agent, "describe"):
        return agent.describe()
    return {"name": type(agent).__name__}


def episode(
    env: Env,
    episode_id: str,
    *,
    run_id: str | None = None,
    media: MediaStore | None = None,
    reward: dict | None = None,
    meta: dict | None = None,
    agent=None,
) -> dict:
    """The episode in schema v1. Audio is written to ``media`` (if given) and referenced.
    ``reward`` defaults to the sum of the episode's ``reward`` stream; pass a task-specific
    ``{"total", "parts"}`` (e.g. τ-bench's breakdown) to override. ``meta`` adds fields to ``meta``.
    ``agent`` (an agent with ``describe()``, or a dict) is recorded as ``meta.agent``: which agent the episode ran.
    Chunk-/unit-level internals of the agent are not part of the trajectory; an adapter's trace is
    saved separately (docs/AGENT_TRACE.md)."""
    log = env.log
    turns = _turns(log, media, env.t)
    _attach_tool_calls(log, turns)
    turns.sort(key=lambda t: t["start_time"])
    m: dict = {"episode_id": episode_id}
    if run_id:
        m["run_id"] = run_id
    if env.seed is not None:
        m["seed"] = env.seed
    if media is not None:
        m["media_root"] = str(media.root)
    m.update(
        task=_jsonable(env.sim.task),
        env=_env_context(env, log),
        duration_ms=env.t,
        end_reason="max_duration" if env.truncated else "idle",
    )
    user = _user(env)
    if user is not None:
        m["user"] = user
    if agent is not None:
        m["agent"] = describe_agent(agent)
    m.update(meta or {})
    ep: dict = {"schema": SCHEMA, "meta": m}
    bg = _background(env, media, env.t)
    if bg:
        ep["background"] = bg
    ep["turns"] = turns
    if reward is None:  # from the episode's reward stream; null when nothing defines a reward
        rewarded = any(f.stream == REWARD for f in log)
        reward = {"total": env.sim.reward if rewarded else None, "parts": {}}
    ep["eval"] = {"reward": reward, "duplex": duplex(turns, end_ms=env.t), "scores": scores(turns, end_ms=env.t)}
    return ep


# ---------------------------------------------------------------- io


def save(episodes: list[dict], path: str | Path, append: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep, ensure_ascii=False) + "\n")
    return path


def load(path: str | Path) -> list[dict]:
    """Episodes from a JSONL run file, or a single (possibly pretty-printed) episode JSON."""
    text = Path(path).read_text()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------- conversions


def to_frames(ep: dict, media: MediaStore | None = None) -> list[Frame]:
    """Rebuild a runtime event log. A cut turn becomes its plan (its full audio from the media store, or
    for text a duration estimated from the speaking rate of what was said) plus a cut at ``end_time``."""
    frames = []
    for t in ep["turns"]:
        role = t["role"]
        stream, src = speech_stream(role), "policy" if role == "agent" else role
        dur = t["end_time"] - t["start_time"]
        if dur > 0 or t["text"]:
            data = media.load(t["media"]) if media is not None and "media" in t else t["text"]
            tags = {"kind": t.get("kind"), "expects": t.get("expects"), "intent": t.get("intent"), "label": t.get("label")}
            actual = Segment(t["id"], t["start_time"], dur, data, text=t["text"], **tags)
            if "unsaid" in t and t["text"]:
                full = t["text"] + t["unsaid"]
                if media is not None and "media" in t:  # the stored file holds the whole planned audio
                    plan = media.load({**t["media"], "end_ms": 10**12})
                    plan_dur, plan_data = plan.dur_ms, plan
                else:  # text only: estimate from the speaking rate of what was said
                    plan_dur, plan_data = round(dur * len(full) / len(t["text"])), full
                planned = Segment(t["id"], t["start_time"], plan_dur, plan_data, text=full, **tags)
                frames += [Frame(stream, t["start_time"], planned, src=src), Frame(stream, t["end_time"], actual, src=src)]
            else:
                frames.append(Frame(stream, t["start_time"], actual, src=src))
        for c in t.get("tool_calls", []):
            frames.append(Frame(CALL[role], c["call_time"], ToolCall(c["id"], c["name"], c["arguments"]), src=src))
            if "result_time" in c:
                res = ToolResult(c["id"], c["name"], c.get("content", ""), c.get("error", False))
                frames.append(Frame(RESULT[role], c["result_time"], res, src="tools"))
    return sorted(frames, key=lambda f: f.t)


def agent_view(ep: dict, chunk_ms: int, roles: tuple[str, ...] = ("user",)) -> list[dict]:
    """What an agent stepping every ``chunk_ms`` heard at each step: for each window
    [time, time + chunk_ms), the text of every turn by ``roles`` played in it (linear in time)."""
    chunk = chunk_ms
    turns = [t for t in ep["turns"] if t["role"] in roles and t["end_time"] > t["start_time"]]
    steps = []
    for i, a in enumerate(range(0, ep["meta"]["duration_ms"], chunk)):
        b, heard = a + chunk, []
        for t in turns:
            s = Segment(t["id"], t["start_time"], t["end_time"] - t["start_time"], t["text"])
            if s.t0 < b and s.end > a:
                lo, hi = max(a, s.t0), min(b, s.end)
                heard.append({"id": t["id"], "start_time": lo, "end_time": hi, "text": s.heard_text(hi)[len(s.heard_text(lo)) :]})
        steps.append({"step": i, "time": a, "heard": heard})
    return steps


def freeze_user(ep: dict, media: MediaStore | None = None, role: str = "user") -> list[dict]:
    """The ``role``'s side of an episode as fixed, timed turns for ``ReplayUser`` — an open-loop user.

    Each turn keeps what was actually said (a turn that was cut keeps only its spoken part, as a
    recording would), its start time, ``kind`` and, with ``media``, its audio. Replayed against any
    agent, the user then says exactly the same things at exactly the same times, whatever the agent
    does: the open-loop counterpart of the closed-loop episode it came from.
    """
    turns = []
    for t in ep["turns"]:
        if t["role"] != role or (t["end_time"] <= t["start_time"] and not t["text"]):
            continue
        turn = {"t": t["start_time"], "text": t["text"], "dur": t["end_time"] - t["start_time"]}
        if t.get("kind"):
            turn["kind"] = t["kind"]
        for k in ("expects", "intent", "label"):
            if t.get(k):
                turn[k] = t[k]
        if media is not None and "media" in t:
            turn["audio"] = media.load(t["media"])  # the spoken part only
        turns.append(turn)
    return turns
