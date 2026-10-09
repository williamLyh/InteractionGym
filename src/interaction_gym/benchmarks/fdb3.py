"""Full-Duplex-Bench v3 (tool use under real disfluency) as InteractionGym tasks.

Benchmark: Lin et al., "Full-Duplex-Bench-v3: Benchmarking Tool Use for Full-Duplex Voice Agents Under
Real-World Disfluency" (arXiv 2604.04847); code github.com/DanielLin94144/Full-Duplex-Bench/tree/main/v3.
**License: CC BY-NC 4.0** (data and code) — non-commercial research only. The mock APIs, tool descriptions,
agent instructions, judge prompts and pass logic ported from that repository live in the optional component
``interaction-gym-fdbench`` (``interaction_gym_fdbench.v3``, CC BY-NC 4.0; ``extras/fdbench`` in the
repository), loaded on first use and re-exported here; this module (loading, the closed-loop user, timing, episode
helpers) is Apache-2.0 and imports without it.

Data: 100 recordings ``{scenario_id}_{speaker}/`` (79 scenarios, 12 speakers), each a single spoken request
(``input.wav``, 48 kHz in the release; any WAV or an Opus file decoded with ffmpeg is accepted here) plus
``metadata.json``: ``dialogue[0].user`` (the script the speaker read, with its fillers, pauses, false starts and
self-corrections), ``dialogue[0].ai`` (reference reply), ``expected_tool_calls`` (``args`` may reference earlier
results, ``"$RESULT_0.flights[0].flight_id"``), ``difficulty`` (easy / medium / hard = 1 / 2 / 3 calls),
``disfluency_features``, ``state_rollback_test`` (+ ``state_rollback_details``: the original and the corrected
parameter), ``acting_notes``, ``latency_profile``.

What is here:

- ``load`` / ``load_recording``: one ``Task`` per recording. The user's turn is the recording from its start to
  the end of the request (``user_end_ms``: the first ≥ 2 s gap between voiced stretches, else the last voiced
  frame — the official rule on ASR word timestamps, here on an energy VAD) plus ``TAIL_MS``. The release's
  recordings run on for 20–40 s of room tone (sometimes faint background talk) after the request; that tail is
  dropped, in both open- and closed-loop runs, so that a simulated user can take the next turn.
- ``MockAPIBackend``: the 12 mock APIs of ``mock_apis.py``, returning exactly the same values, behind the
  official cascaded agent's tool schemas (``cascaded_agent.py``); arguments are coerced to the declared types as
  LiveKit does. ``latency_ms``: the scenario's ``latency_profile`` midpoint (deterministic).
- ``pass_at_1``: ``evaluate_pass_rate.py`` (strict: exactly the expected tools as a multiset, then every
  argument correct by an LLM judge with the official prompt, exact match as fallback). The official judge is
  GPT-4o; any ``TextGen`` can be passed (numbers are only comparable with the official model).
- ``judge_response``: ``evaluate_tool_calls.py``'s response-quality judge (official prompt).
- ``timing``: turn-take, interruption (Δt = agent start − user end < 0), first-response and first-tool-call
  latency (``analyze_tool_latency.py``).
- ``user_card``: a closed-loop scenario card (goal, facts, persona) for a simulated user who continues the
  conversation after the recording (``USER_SYSTEM``).
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..audio import Audio
from ..core import Task
from ..tools import ToolBackend, ToolCall, ToolResult, ToolSpec
from .wav import read_wav, resample
from ._fdbench import Lazy, reexport

# Full-Duplex-Bench v3 mock APIs, tool schemas, instructions, judges and pass logic (CC BY-NC 4.0): the optional
# component interaction-gym-fdbench, loaded on first use and re-exported here
_v3 = Lazy("v3")
__getattr__ = reexport(__name__, "v3", {
    "AGENT_INSTRUCTIONS", "ARG_PROMPT", "FUNCTIONS", "LATENCY_MS", "REQUIRED", "RESPONSE_PROMPT", "_TOOLS",
    "exact_match_args", "judge_args", "judge_response", "pass_at_1", "strip_json_fences", "tool_selection",
}, {"_strip_json_fences": "strip_json_fences"})  # (old name)

NAME = "Full-Duplex-Bench v3"
REPO = "https://github.com/DanielLin94144/Full-Duplex-Bench/tree/main/v3"
LICENSE = "CC BY-NC 4.0"
TAIL_MS = 300  # kept after the end of the request
GAP_S = 2.0  # official end-of-request rule: the first gap of more than 2 s between words


def _p(type_: str, desc: str) -> dict:
    return {"type": type_, "description": desc}


def tool_specs() -> list[ToolSpec]:
    """The official tool schemas (``_TOOLS``) as ``ToolSpec``s."""
    out = []
    for name, (desc, params) in _v3._TOOLS.items():
        props = {p: _p(t, d) for p, (t, d, _) in params.items()}
        req = [p for p, (_, _, dflt) in params.items() if dflt is _v3.REQUIRED]
        out.append(ToolSpec(name, desc, {"type": "object", "properties": props, "required": req}))
    return out


def official_args(name: str, args: dict) -> dict:
    """The arguments as the official agent logs them: the declared parameters (defaults filled in, as the
    LiveKit function wrapper passes them to ``registry.call``), coerced to their declared types."""
    params = _v3._TOOLS.get(name, ("", {}))[1]
    out = dict(args)
    for p, (t, _, dflt) in params.items():
        if p not in out and dflt is not _v3.REQUIRED:
            out[p] = dflt
        elif p in out and out[p] is not None:
            out[p] = _coerce(out[p], t)
    return out


def _coerce(v, t: str):
    try:
        if t == "integer" and not isinstance(v, bool):
            return int(float(v))
        if t == "number" and not isinstance(v, bool):
            return float(str(v).replace(",", "").replace("$", "")) if isinstance(v, str) else float(v)
        if t == "string" and not isinstance(v, str):
            return str(v)
    except (TypeError, ValueError):
        pass
    return v


class MockAPIBackend(ToolBackend):
    """The 12 mock APIs. State: the calls made, in order (``{"function", "args", "result"}``, like ``CallLogger``)."""

    name = "fdb3"
    description = "Full-Duplex-Bench v3 mock APIs (travel, finance, housing, e-commerce)"

    def tools(self, caller="agent"):
        return tool_specs() if caller == "agent" else []

    def reset(self, task, rng):
        return {"calls": []}

    async def call(self, state, caller, call: ToolCall) -> ToolResult:
        fn = _v3.FUNCTIONS.get(call.name)
        if fn is None:  # MockAPIRegistry.call answers unknown functions with an error object
            result = {"status": "error", "message": f"Unknown function: {call.name}"}
            return ToolResult(call.id, call.name, json.dumps(result), error=True)
        args = official_args(call.name, call.arguments)
        try:
            result = fn(**args)
        except Exception as e:  # e.g. a missing required argument: the tool layer reports it to the model
            return ToolResult(call.id, call.name, f"Error: {type(e).__name__}: {e}", error=True)
        state["calls"].append({"function": call.name, "args": args, "result": result})
        return ToolResult(call.id, call.name, json.dumps(result))


def latency_ms(task: Task):
    """ToolWorld latency for this task's ``latency_profile`` (the profile's midpoint)."""
    ms = _v3.LATENCY_MS.get(task.scenario.get("benchmark", {}).get("latency_profile", "instant"), 0)
    return lambda call: ms


# ---------------------------------------------------------------- loading


def read_audio(path: str | Path, sr: int = 16000) -> Audio:
    """A WAV (any PCM / float) or anything ffmpeg decodes (e.g. Opus), as mono 16-bit at ``sr``."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return resample(read_wav(path), sr)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
                         check=True, capture_output=True).stdout
    return Audio.from_pcm16(raw, sr)


def voiced_spans(audio: Audio, rms_threshold: float = 300.0, frame_ms: int = 20, min_speech_ms: int = 100,
                 min_silence_ms: int = 200) -> list[tuple[int, int]]:
    """Voiced stretches ``[(start_ms, end_ms)]`` by frame energy (the same detector the cascaded agent uses)."""
    n = round(audio.sr * frame_ms / 1000)
    s = audio.samples
    spans, start, quiet = [], None, 0
    nframes = len(s) // n
    for i in range(nframes):
        seg = s[i * n:(i + 1) * n]
        rms = (sum(v * v for v in seg) / n) ** 0.5
        if rms >= rms_threshold:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet * frame_ms >= min_silence_ms:
                spans.append((start, i - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        spans.append((start, nframes - quiet))
    return [(a * frame_ms, b * frame_ms) for a, b in spans if (b - a) * frame_ms >= min_speech_ms]


def request_end_ms(audio: Audio, gap_s: float = GAP_S) -> int:
    """End of the spoken request: the end of the voiced stretch before the first gap longer than ``gap_s``
    (official rule, ``run_tool_benchmark.py``), else of the last voiced stretch."""
    spans = voiced_spans(audio)
    if not spans:
        return audio.dur_ms
    for (a, b), (c, _) in zip(spans, spans[1:]):
        if c - b > gap_s * 1000:
            return b
    return spans[-1][1]


@dataclass
class Recording:
    id: str  # directory name: {scenario_id}_{speaker}
    meta: dict
    audio: Audio  # trimmed to the request (+ TAIL_MS)
    user_end_ms: int  # end of the request in the recording
    full_ms: int  # length of the released recording
    path: str = ""

    @property
    def scenario_id(self) -> str:
        return self.meta["id"]

    @property
    def speaker(self) -> str:
        return self.id[len(self.meta["id"]) + 1:]

    @property
    def script(self) -> str:
        return self.meta["dialogue"][0]["user"]


def ids(root: str | Path) -> list[str]:
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir() and (p / "metadata.json").exists())


def load_recording(root: str | Path, rid: str, sr: int = 16000) -> Recording:
    d = Path(root) / rid
    meta = json.loads((d / "metadata.json").read_text())
    src = next((d / f for f in ("input.wav", "input.opus", "input.flac") if (d / f).exists()), None)
    if src is None:
        raise FileNotFoundError(f"{d}: no input audio")
    audio = read_audio(src, sr)
    end = request_end_ms(audio)
    cut = min(audio.dur_ms, end + TAIL_MS)
    return Recording(rid, meta, audio[: round(cut * sr / 1000)], end, audio.dur_ms, str(src))


def task_of(rec: Recording) -> Task:
    """The task for one recording. ``scenario["turns"]`` (for ``ReplayUser``) and ``scenario["first_turn"]``
    (for an ``LLMSource`` user) are the recording; ``instructions`` / ``persona`` / ``facts`` are the
    closed-loop scenario card (``user_card``); ``benchmark`` keeps the official annotation."""
    m = rec.meta
    turn = {"t": 0, "text": rec.script, "audio": rec.audio}
    card = user_card(m)
    bench = {k: m.get(k) for k in ("id", "domain", "title", "difficulty", "disfluency_features", "state_rollback_test",
                                   "state_rollback_details", "acting_notes", "latency_profile")}
    bench.update(recording=rec.id, speaker=rec.speaker, script=rec.script, reference_reply=m["dialogue"][0].get("ai", ""),
                 user_end_ms=rec.user_end_ms, recording_ms=rec.full_ms, license=LICENSE)
    return Task(id=rec.id, scenario={"turns": [turn], "first_turn": {"text": rec.script, "audio": rec.audio},
                                     "instructions": card["goal"], "persona": card["persona"], "facts": card["facts"],
                                     "benchmark": bench},
                criteria={"expected_tool_calls": m["expected_tool_calls"]})


def load(root: str | Path, only: list[str] | None = None, sr: int = 16000) -> list[tuple[Recording, Task]]:
    out = []
    for rid in ids(root):
        if only is None or rid in only:
            rec = load_recording(root, rid, sr)
            out.append((rec, task_of(rec)))
    return out


# ---------------------------------------------------------------- closed-loop user card

_REF = re.compile(r"^\$RESULT_(\d+)\.(.+)$")


def _value(v) -> str:
    if isinstance(v, str):
        m = _REF.match(v)
        if m:
            return f"(whatever the result of step {int(m.group(1)) + 1} gives: {m.group(2).replace('_', ' ')})"
        return f'"{v}"'
    return json.dumps(v)


def user_card(meta: dict) -> dict:
    """What the simulated user wants and knows, from the annotation: ``goal`` (the request, in steps), ``facts``
    (every value the correct calls need — for a self-correction only the corrected value, never the original)
    and ``persona`` (from the acting notes)."""
    calls = meta["expected_tool_calls"]
    steps, facts = [], {}
    for i, c in enumerate(calls):
        args = ", ".join(f"{k.replace('_', ' ')} = {_value(v)}" for k, v in c.get("args", {}).items())
        steps.append(f"{i + 1}. {c['function'].replace('_', ' ')}" + (f" ({args})" if args else ""))
        for k, v in c.get("args", {}).items():
            if not (isinstance(v, str) and v.startswith("$")):
                facts[f"{c['function']}.{k}"] = v
    rollback = meta.get("state_rollback_details") or {}
    corrected = rollback.get("corrected_param") or {}
    goal = ("You are calling a voice assistant. What you asked for, in your own (first) message, was:\n"
            f"\"{meta['dialogue'][0]['user']}\"\n\n"
            "What you want done, in this order:\n" + "\n".join(steps))
    if corrected:
        goal += ("\n\nWhile asking, you corrected yourself; only the corrected value counts: "
                 + ", ".join(f"{k.replace('_', ' ')} = {_value(v)}" for k, v in corrected.items()) + ".")
    persona = "A customer on the phone. " + (f"How you spoke your first message: {meta['acting_notes']}" if meta.get("acting_notes") else "")
    return {"goal": goal, "facts": facts, "persona": persona.strip()}


USER_SYSTEM = """You are playing a customer who is talking to a voice assistant on the phone, in a live spoken conversation.

{persona}

{instructions}

Rules:
- Your first message (above) has already been said. From now on, say only what you say out loud: one or two short, natural spoken sentences.
- If the assistant asks you something, answer truthfully with the details above. Never invent other values, and never go back to a value you corrected.
- If the assistant got something wrong (a wrong detail, the wrong action), or skipped part of what you want done, say so briefly and ask for it.
- If the assistant asks you to confirm something that is right, confirm it.
- Want nothing beyond your list: if the assistant offers anything else (adding an item, booking, more help), say no thanks.
- Do not ask about the results themselves (prices, options, details): you only care that the steps in your list get done.
- Messages from the assistant marked [CURRENTLY SPEAKING, INCOMPLETE] are still being said.
- Never say that you are an AI or a simulation.
- As soon as every step in your list has been done (the assistant has reported it), or clearly cannot be done, say a brief thanks / goodbye and end with {stop}."""


class UserSource:
    """The closed-loop user's content: an ``LLMSource`` with ``USER_SYSTEM`` whose first turn is the recording, that
    stops after ``max_turns`` turns, and that is told when the assistant has stayed silent since the user's last
    turn (a nudge after ``TurnTaking.nudge_after_ms``) instead of being handed its own message to continue."""

    def __init__(self, llm, max_turns: int = 8, system: str | None = None):
        from ..user import LLMSource

        self.inner = LLMSource(llm, system=system or USER_SYSTEM)
        self.max_turns = max_turns

    def profile(self) -> dict:
        return {**self.inner.profile(), "max_turns": self.max_turns}

    async def next(self, i, task, convo):
        from ..user import STOP, UserTurn, _pause, _spoken, as_turn

        if i == 0:
            return as_turn(task.scenario["first_turn"])
        if i >= self.max_turns:
            return None
        msgs = self.inner.messages(task, convo)
        if msgs[-1]["role"] == "assistant":  # the user spoke last: the assistant has said nothing since
            msgs = [m for m in msgs if not m["content"].startswith("(The call has just connected")]
            msgs.append({"role": "user", "content": "(The assistant has not said anything since your last message.)"})
        raw = await self.inner.llm.chat(msgs)
        if not _spoken(raw) and STOP not in raw:
            raw = await self.inner.llm.chat(msgs)
        text = _spoken(raw)
        if STOP in text:
            text = text.replace(STOP, "").strip()
            return UserTurn(text, final=True) if text else None
        return UserTurn(text, pause=_pause(raw))


# ---------------------------------------------------------------- from an episode


def agent_calls(ep: dict, before_ms: int | None = None) -> list[dict]:
    """The agent's tool calls ``{"function", "args", "time"}`` in call order (official arguments), optionally
    only those made before ``before_ms``."""
    calls = [c for t in ep["turns"] if t["role"] == "agent" for c in t.get("tool_calls", [])]
    calls.sort(key=lambda c: c["call_time"])
    return [{"function": c["name"], "args": official_args(c["name"], c["arguments"]), "time": c["call_time"]}
            for c in calls if before_ms is None or c["call_time"] < before_ms]


def effective_calls(calls: list[dict], expected_calls: list[dict]) -> list[dict]:
    """The calls that stand at the end of a conversation: for a function the task expects ``k`` times, only its
    last ``k`` calls (a later call to the same function is a correction / retry that replaces an earlier one);
    calls to functions the task does not expect all stay (they are real extra actions)."""
    want: dict[str, int] = {}
    for c in expected_calls:
        want[c["function"]] = want.get(c["function"], 0) + 1
    keep = []
    for i, c in enumerate(calls):
        k = want.get(c["function"])
        later = sum(1 for d in calls[i + 1:] if d["function"] == c["function"])
        if k is None or later < k:
            keep.append(c)
    return keep


def user_turns(ep: dict) -> list[dict]:
    return sorted((t for t in ep["turns"] if t["role"] == "user" and t["end_time"] > t["start_time"]), key=lambda t: t["start_time"])


def agent_speech(ep: dict, before_ms: int | None = None) -> list[dict]:
    return sorted((t for t in ep["turns"] if t["role"] == "agent" and t["end_time"] > t["start_time"]
                   and (before_ms is None or t["start_time"] < before_ms)), key=lambda t: t["start_time"])


def timing(ep: dict, user_end_ms: int, before_ms: int | None = None) -> dict:
    """Official timing for the first user turn (``analyze_tool_latency.py`` / ``evaluate_tool_calls.py``):
    ``turn_taken`` (the agent said anything), ``delta_ms`` = first agent speech − end of the request (< 0 is an
    interruption), ``first_response_ms`` (only when not an interruption), ``first_tool_call_ms`` (first call −
    end of the request; negative when the agent called before the user had finished)."""
    speech = agent_speech(ep, before_ms)
    calls = agent_calls(ep, before_ms)
    out: dict = {"turn_taken": bool(speech), "user_end_ms": user_end_ms}
    if speech:
        d = speech[0]["start_time"] - user_end_ms
        out.update(delta_ms=d, interrupted=d < 0, first_response_ms=d if d >= 0 else None)
    else:
        out.update(delta_ms=None, interrupted=False, first_response_ms=None)
    out["first_tool_call_ms"] = calls[0]["time"] - user_end_ms if calls else None
    return out
