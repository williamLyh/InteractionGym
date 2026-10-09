"""Audio MultiChallenge (arXiv 2512.14865, Scale AI) in InteractionGym: open- vs closed-loop multi-turn episodes.

Data: Hugging Face ``ScaleAI/audiomc`` (**MIT**), one parquet (5.3 GB): 452 conversations by 47 speakers, 3–8 user
turns each (real human speech, median 18 s per turn), four axes — INFERENCE_MEMORY (132), INSTRUCTION_RETENTION
(120), VOICE_EDITING (117, mid-utterance repairs), SELF_COHERENCE (83) — and 1,712 rubric items about the reply to
the LAST user turn. Every user turn after the first was recorded in reaction to a *fixed* assistant reply (the
``assistant_turn_k_transcript`` columns). Official protocol: the model gets the conversation with those fixed
assistant turns as history and writes only the final reply; o4-mini judges it against each rubric item separately
(``JUDGE_PROMPT``, from the dataset card, MIT: ``benchmarks/third_party/audiomc``; ``history``).

A native duplex model cannot be given someone else's replies as its own history, so here it talks through the whole
conversation (``load`` expects the parquet unpacked by ``extract``: ``<root>/<id>/u<k>.wav`` + ``index.json``):

- **A, open loop** (``ScriptSource``): the recorded user turns, in order, each once the agent has stopped (reply gap
  from ``ResponseDelay``; the user never yields mid-turn). The content is fixed: later turns still react to the
  dataset's assistant, whatever the agent said.
- **B, closed loop** (``ReplanSource``): the same first recording; each later turn is the scripted turn rewritten by
  an LLM user to fit what the agent actually said — keeping every fact, request, instruction and self-repair of the
  script (so the final challenge and its rubric stay valid) — spoken in the speaker's cloned voice.
- Both: same number of user turns; the agent's reply to the last user turn is judged with the official prompt, on
  the conversation as it actually happened (``history(ep)``). A = B up to the user's second turn.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core import Task
from ..user import UserTurn, _spoken, as_turn

NAME = "Audio MultiChallenge"
HF = "ScaleAI/audiomc"
LICENSE = "MIT"
AXES = ("INFERENCE_MEMORY", "INSTRUCTION_RETENTION", "VOICE_EDITING", "SELF_COHERENCE")

from .third_party.audiomc import JUDGE_PROMPT  # noqa: E402,F401  (MIT, Scale AI; re-exported)


def extract(parquet: str | Path, out: str | Path) -> None:
    """The parquet → ``<out>/<id>/u<k>.wav`` (16 kHz mono) + ``index.json`` (needs pyarrow, numpy, soundfile)."""
    import io

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f, index = pq.ParquetFile(str(parquet)), []
    for rg in range(f.num_row_groups):
        for row in f.read_row_group(rg).to_pylist():
            d = out / row["id"]
            d.mkdir(exist_ok=True)
            rec = {"id": row["id"], "axis": row["axis"], "rubric": row["rubric"], "turns": []}
            for k in range(1, 9):
                a, tr = row.get(f"user_turn_{k}_audio"), (row.get(f"user_turn_{k}_transcript") or "").strip()
                if not tr and not (a and a.get("bytes")):
                    break
                x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
                x = x.mean(1)
                if sr != 16000:
                    n = int(round(len(x) * 16000 / sr))
                    x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")
                sf.write(d / f"u{k}.wav", x, 16000, subtype="PCM_16")
                rec["turns"].append({"user": tr, "assistant": (row.get(f"assistant_turn_{k}_transcript") or "").strip(),
                                     "audio": f"{row['id']}/u{k}.wav", "dur_ms": int(len(x) / 16)})
            index.append(rec)
    (out / "index.json").write_text(json.dumps(index, indent=1))


def load(root: str | Path, axes=None, only=None, per_axis: int | None = None) -> list[Task]:
    """``index.json`` → one ``Task`` per conversation. ``scenario["script"]`` = the recorded user turns (text + audio
    path) and the dataset's assistant turns; ``criteria["rubric"]``. ``per_axis`` keeps every k-th conversation of
    each axis (in index order) to get that many."""
    root = Path(root)
    index = json.loads((root / "index.json").read_text())
    tasks = []
    for axis in axes or AXES:
        recs = [r for r in index if r["axis"] == axis and (not only or r["id"] in only)]
        if per_axis and len(recs) > per_axis:
            step = len(recs) / per_axis
            recs = [recs[int(i * step)] for i in range(per_axis)]
        for r in recs:
            rubric = json.loads(r["rubric"]) if isinstance(r["rubric"], str) else r["rubric"]
            script = [{"text": t["user"], "audio": str(root / t["audio"]), "assistant": t["assistant"]} for t in r["turns"]]
            tasks.append(Task(
                id=r["id"],
                scenario={"script": script, "first_turn": {"text": script[0]["text"], "audio": script[0]["audio"]},
                          "turns": [{"text": s["text"], "audio": s["audio"]} for s in script], "persona": "",
                          "profile": {"language": "en"},
                          "benchmark": {"name": NAME, "license": LICENSE, "source": HF, "id": r["id"], "axis": axis,
                                        "n_turns": len(script)}},
                criteria={"rubric": [x.strip() for x in rubric], "axis": axis}))
    return tasks


REPLAN_SYSTEM = """You are the USER in a spoken conversation with a voice assistant. You are following a script of what you want to say, turn by turn, but the assistant you are talking to is not the one the script was written for, so you adapt each line to what this assistant actually said.

For your next turn you get the scripted line. Rewrite it so it makes sense after what THIS assistant just said:
- Keep EVERY fact, number, name, request, question, instruction, constraint and change of mind in the scripted line, with the same values. Do not add new facts or requests, and do not drop any.
- Keep the way it is said: casual spoken language, fillers, and any self-corrections inside the line ("K5 — no, make it K4") stay as spoken repairs.
- Only change the parts that react to the previous reply (thanks, agreement, references to things it said) so they fit what this assistant actually said; if the scripted line refers to something this assistant never said, rephrase it so it does not.
- Say what the scripted line says, not something from earlier in the conversation: never repeat a request you already made.
- If this assistant got something you said earlier wrong, you may point it out in one short phrase, at the start.
- Output only the words you say out loud, nothing else."""

REPLAN_PROMPT = """Conversation so far (as it actually happened):
{dialogue}

The scripted line for your next turn (rewrite THIS line):
"{line}"

Your next turn:"""


def _dialogue(convo) -> str:
    return "\n".join(("USER: " if u.role == "user" else "ASSISTANT: ") + u.text for u in convo if u.text) or "(nothing yet)"


class ScriptSource:
    """A: the recorded turns, in order (content fixed); the user never stops talking mid-turn."""

    def profile(self) -> dict:
        return {"mode": "semi_online", "content": "recorded script"}

    async def next(self, i, task, convo):
        turns = task.scenario["turns"]
        return as_turn(turns[i]) if i < len(turns) else None


class ReplanSource:
    """B: first turn = the recording; turn k > 1 = scripted turn k rewritten (LLM) to fit the conversation so far."""

    def __init__(self, llm):
        self.llm = llm

    def profile(self) -> dict:
        from ..clients import describe

        return {"mode": "online", "content": "script replanned to the agent's replies", "llm": describe(self.llm),
                "system_prompt": REPLAN_SYSTEM}

    async def next(self, i, task, convo):
        script = task.scenario["script"]
        if i == 0:
            return as_turn(task.scenario["first_turn"])
        if i >= len(script):
            return None
        msgs = [{"role": "system", "content": REPLAN_SYSTEM},
                {"role": "user", "content": REPLAN_PROMPT.format(dialogue=_dialogue(convo), line=script[i]["text"])}]
        text = _spoken(await self.llm.chat(msgs))
        if not text:
            text = script[i]["text"]
        return UserTurn(text, final=i == len(script) - 1)


def final_reply(ep: dict) -> str:
    """Everything the agent said after the last user turn began (its answer to the challenge)."""
    users = [t for t in ep["turns"] if t["role"] == "user" and t["text"]]
    if not users:
        return ""
    t0 = max(t["start_time"] for t in users)
    return " ".join(t["text"] for t in sorted(ep["turns"], key=lambda t: t["start_time"])
                    if t["role"] != "user" and t["text"] and t["start_time"] >= t0).strip()


def history(ep: dict) -> str:
    """The README's ``build_grading_conversation_history`` on the conversation as it happened: user turns and the
    agent's speech between them (merged per gap), the last assistant entry = ``final_reply``."""
    parts, cur = [], None
    for t in sorted((t for t in ep["turns"] if t["text"] and t["end_time"] > t["start_time"]), key=lambda t: t["start_time"]):
        role = "User" if t["role"] == "user" else "Assistant"
        if cur and cur[0] == role:
            cur[1].append(t["text"])
        else:
            cur = (role, [t["text"]])
            parts.append(cur)
    if parts and parts[-1][0] == "User":
        parts.append(("Assistant", [""]))
    return "\n\n".join(f"{r}: {' '.join(x).strip()}" for r, x in parts)


def judge_prompt(conversation: str, item: str) -> str:
    return JUDGE_PROMPT.replace("«conversation_history»", conversation).replace("«rubric_item»", item)
