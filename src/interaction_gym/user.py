"""Simulated users.

- ``ReplayUser`` (offline): turns play at fixed timestamps whatever the agent does.
- ``UserSim`` (reactive): a ``TurnTaking`` policy decides *when* to speak; the
  content comes from a ``ScriptSource`` (fixed texts, i.e. semi-online) or an
  ``LLMSource`` (generated, i.e. online); ``Voice`` decides *how it sounds*
  (given audio, TTS, or text with an estimated duration).

The user only ever sees what the agent has already said (``Segment.heard_text``).
LLM / TTS calls run inside ``step`` while simulated time stands still; realistic
reaction time comes from scheduling the utterance in the future instead.
"""

from __future__ import annotations

import hashlib
import random
import math
import re
import warnings
from dataclasses import asdict, dataclass, replace

from .audio import Audio, normalize_level, trim_silence
from .clients import Speech, TextGen, describe
from .core import Frame, Node, Segment, Task
from .soundscape import EVENT_RATES

STOP = "###STOP###"
WORDS_PER_SEC = 3.4  # FDGym's speaking-rate estimate for text-only timelines
CJK_CHARS_PER_WORD = 1.5  # Chinese / Japanese text has no spaces: ~1.5 characters per English-word-equivalent of speech

_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"  # kana + CJK ideographs (Hangul is space-separated)
_CJK_PUNCT = "\u3000-\u303f\uff01-\uff0f\uff1a-\uff1f"  # full-width punctuation separates tokens like a space
_UNIT = re.compile(rf"[{_CJK}]|[^\s{_CJK}{_CJK_PUNCT}]+")


def has_cjk(text: str) -> bool:
    return re.search(rf"[{_CJK}]", text or "") is not None


def n_words(text: str) -> float:
    """Spoken length of ``text`` in English-word equivalents: whitespace tokens, with CJK characters (no spaces between
    words) counted ``1 / CJK_CHARS_PER_WORD`` each. Equals ``len(text.split())`` for text without CJK."""
    text = text or ""
    if not has_cjk(text):
        return len(text.split())
    cjk = len(re.findall(rf"[{_CJK}]", text))
    other = len([u for u in re.findall(rf"[^\s{_CJK}{_CJK_PUNCT}]+", text) if re.search(r"\w", u)])  # punctuation is no word
    return cjk / CJK_CHARS_PER_WORD + other


@dataclass(frozen=True)
class UserTurn:
    text: str
    audio: Audio | None = None
    t: int | None = None  # absolute start time; only used by ReplayUser
    final: bool = False  # the conversation ends after this turn
    kind: str | None = None  # None (a normal turn) | "backchannel" | "aside" | "noise" | "away" — known only to the producer
    dur: int | None = None  # duration when there is neither audio nor text to time it (e.g. a noise, an absence)
    pause: str | None = None  # None | "short" | "long": the user needs a moment before saying this (thinking, checking)
    expects: str | None = None  # the reaction the user expects, when not the kind's default: "yield" | "wait" | "interrupt" | ...
    intent: str | None = None  # a barge-in's purpose: "correction" | "question" | "stop" | "other"
    label: str | None = None  # what a sound is (a noise's "door_slam", "called_away" ...)


def as_turn(x) -> UserTurn:
    if isinstance(x, UserTurn):
        return x
    if isinstance(x, str):
        return UserTurn(x)
    audio = Audio.read_wav(x["audio"]) if isinstance(x.get("audio"), str) else x.get("audio")
    return UserTurn(x.get("text", ""), audio, x.get("t"), x.get("final", False), x.get("kind"), x.get("dur"), x.get("pause"),
                    x.get("expects"), x.get("intent"), x.get("label"))


@dataclass(frozen=True)
class Utterance:
    role: str  # "user" | "agent"
    text: str  # what has been heard so far
    t0: int
    incomplete: bool  # still being spoken at the time of the snapshot
    kind: str | None = None  # the segment's kind: None (a normal turn) | "backchannel" | "aside" | "noise" | "away"


def conversation(user: list[Segment], agent: list[Segment], t: int) -> list[Utterance]:
    """What has been said by both sides up to time t, ordered by start time."""
    out = []
    for role, segs in (("user", user), ("agent", agent)):
        for s in segs:
            heard = s.heard_text(t)
            if s.t0 <= t and heard:
                out.append(Utterance(role, heard, s.t0, s.active(t), s.kind))
    return sorted(out, key=lambda u: u.t0)


# ---------------------------------------------------------------- who the user is, and how they sound

# The speakers of Qwen3-TTS CustomVoice (from its model card): gender, age group, native language.
QWEN3_TTS_VOICES = [
    {"speaker": "vivian", "gender": "female", "age": "young", "language": "zh", "description": "bright, slightly edgy young female voice"},
    {"speaker": "serena", "gender": "female", "age": "young", "language": "zh", "description": "warm, gentle young female voice"},
    {"speaker": "uncle_fu", "gender": "male", "age": "senior", "language": "zh", "description": "seasoned male voice with a low, mellow timbre"},
    {"speaker": "dylan", "gender": "male", "age": "young", "language": "zh", "accent": "beijing", "description": "youthful Beijing male voice"},
    {"speaker": "eric", "gender": "male", "age": "young", "language": "zh", "accent": "sichuan", "description": "lively Chengdu male voice, slightly husky"},
    {"speaker": "ryan", "gender": "male", "age": "adult", "language": "en", "description": "dynamic male voice with strong rhythmic drive"},
    {"speaker": "aiden", "gender": "male", "age": "young", "language": "en", "accent": "american", "description": "sunny American male voice"},
    {"speaker": "ono_anna", "gender": "female", "age": "young", "language": "ja", "description": "playful Japanese female voice"},
    {"speaker": "sohee", "gender": "female", "age": "adult", "language": "ko", "description": "warm Korean female voice"},
]


LANGUAGES = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "fr": "French", "de": "German",
             "es": "Spanish", "it": "Italian", "pt": "Portuguese", "ru": "Russian"}


def language_name(code: str | None) -> str | None:
    """``"en"`` → ``"English"`` (names pass through)."""
    return None if code is None else LANGUAGES.get(code.lower(), code)


def age_group(age) -> str | None:
    """``"child" | "young" | "adult" | "senior"`` from an age in years (or such a label)."""
    if age is None or isinstance(age, str):
        return age
    return "child" if age < 13 else "young" if age < 36 else "adult" if age < 60 else "senior"


def pick_voice(profile: dict, voices: list[dict]) -> dict:
    """The catalog voice that best fits a user profile: same gender (required when both are known),
    then age group (+3 same, +1 adjacent), native language (``native_language``, else ``language``; +1) and accent (+1) — who someone sounds like is
    mostly gender and age; these voices can speak any language, their native one only shows as accent.
    Ties are broken by a hash of the profile's name, so one person always gets the same voice while a
    population of users gets a spread."""
    gender, accent, age = profile.get("gender"), profile.get("accent"), age_group(profile.get("age"))
    lang = profile.get("native_language") or profile.get("language")  # a voice's native language shows as its accent
    pool = [v for v in voices if not gender or v.get("gender") == gender] or voices
    order = ["child", "young", "adult", "senior"]

    def score(v: dict) -> int:
        near = 0
        if age in order and v.get("age") in order:
            near = {0: 3, 1: 1}.get(abs(order.index(age) - order.index(v["age"])), 0)
        return near + (lang is not None and v.get("language") == lang) + (accent is not None and v.get("accent") == accent)

    best = max(map(score, pool))
    ties = [v for v in pool if score(v) == best]
    h = int(hashlib.sha256(str(profile.get("name", "")).encode()).hexdigest(), 16)
    return ties[h % len(ties)]


def describe_profile(profile: dict) -> str:
    """One line for the user LLM: "Alex Chen, female, 34 (young), speaks en with an american accent; ..."."""
    if not profile:
        return ""
    who = ", ".join(str(x) for x in (profile.get("name"), profile.get("gender")) if x)
    if profile.get("age") is not None:
        who += f", {profile['age']}" + (f" ({age_group(profile['age'])})" if not isinstance(profile["age"], str) else "")
    bits = [who] if who else []
    if profile.get("language"):
        native = profile.get("native_language")
        bits.append(f"speaks {language_name(profile['language'])}" + (f" with a {profile['accent']} accent" if profile.get("accent") else "")
                    + (f" (native language: {language_name(native)})" if native and native != profile["language"] else ""))
    for key in ("occupation", "speaking_style", "traits"):
        if profile.get(key):
            bits.append(f"{key.replace('_', ' ')}: {profile[key]}")
    return "; ".join(bits)


# How the user says things when the persona's speaking style is not passed to the TTS: an acted style instruction
# ("energetic, talks fast", "slow, gentle"), re-read on every turn, makes some turns much faster, slower or more
# emotional than the rest. The persona still picks the voice and the words.
NEUTRAL_STYLE = "Natural, relaxed everyday phone-call voice at a normal pace; not acted, not dramatic."
_QUIET = re.compile(r"\b(soft|quiet|shy|gentle|tired|careful)", re.I)
_LOUD = re.compile(r"\b(booming|loud|energetic|confident)", re.I)


NO_CLONE = ("Voice: a TTS was given without a voice-cloning TTS. By default every user turn after the first is cloned from "
            "it (one voice, pace and manner for the whole call): pass clone=<a cloning TTS, e.g. OpenAISpeech(url, "
            "'Qwen/Qwen3-TTS-12Hz-1.7B-Base')>, or clone=False to synthesize every turn on its own (not recommended: turns "
            "then differ in pace, loudness and emotion).")


def clones(speech) -> bool:
    """Whether a TTS client can clone a voice itself (``ref_audio``): it says so (``clones = True``, e.g. ``FakeSpeech``)
    or serves a Qwen3-TTS "Base" model. Wrappers exposing ``inner`` (``Cached``) are looked through."""
    while speech is not None:
        if getattr(speech, "clones", False) or str(getattr(speech, "model", "")).lower().endswith("-base"):
            return True
        speech = getattr(speech, "inner", None)
    return False


@dataclass(frozen=True)
class Leveling:
    """Post-processing of every synthesized turn: trim the leading / trailing silence TTS clips carry (keeping
    ``margin_ms``), then bring the active-speech RMS to ``level_dbfs`` plus a small fixed per-persona offset
    (``offset_db`` for a quiet / loud speaking style, else up to ``±offset_db / 2`` from a hash of the name), peak-limited
    to ``peak_dbfs``. ``level_dbfs=None`` only trims."""
    margin_ms: int = 120
    level_dbfs: float | None = -23.0
    offset_db: float = 2.0
    peak_dbfs: float = -1.0

    def offset(self, task: Task, speaker: str) -> float:
        prof = task.scenario.get("profile", {})
        style = task.scenario.get("voice_style") or prof.get("speaking_style") or ""
        if _QUIET.search(style):
            return -self.offset_db
        if _LOUD.search(style):
            return self.offset_db
        h = int(hashlib.sha256(str(prof.get("name") or speaker).encode()).hexdigest(), 16)
        return round(((h % 1001) / 1000 - 0.5) * self.offset_db, 2)

    def apply(self, audio: Audio, task: Task, speaker: str) -> Audio:
        audio = trim_silence(audio, self.margin_ms)
        if self.level_dbfs is not None:
            audio = normalize_level(audio, self.level_dbfs + self.offset(task, speaker), self.peak_dbfs)
        return audio

    def describe(self) -> dict:
        return {"trim_margin_ms": self.margin_ms, "level_dbfs": self.level_dbfs, "persona_offset_db": self.offset_db,
                "peak_dbfs": self.peak_dbfs}


class Voice:
    """Renders a turn: use its audio if given, else TTS, else text with an estimated duration.

    Defaults (since 2026-10-08, chosen by a judged A/B of user-voice variants, see CHANGELOG): the first turn is
    synthesized with a neutral everyday-speech instruction, every later turn is cloned from it, each synthesized turn is
    trimmed and loudness-normalised, and one sampling seed is used per episode. The previous behaviour (every turn
    synthesized on its own with the persona's speaking style, untouched) is
    ``Voice(tts, clone=False, style="persona", leveling=None, seed=False)``.

    The speaker is, in order: ``task.scenario["voice"]``; the ``voices`` catalog entry that best
    fits ``task.scenario["profile"]`` (gender, age, language, accent — see ``pick_voice``); ``voice``.

    ``style`` is what the TTS is told about how to say it (``instructions``, for TTS models that take them):
    ``"persona"`` — the profile's ``speaking_style`` (or ``scenario["voice_style"]``); ``"neutral"`` —
    ``NEUTRAL_STYLE``; any other string — that text; None — nothing.

    ``clone`` (a voice-cloning TTS, e.g. Qwen3-TTS "Base") is required with a TTS unless the TTS clones by itself
    (``clones``); ``clone=False`` opts out explicitly. With it the user keeps one voice for the whole episode: the first turn with words in it (synthesized as above, or given as recorded audio) becomes
    the reference, and every later turn is cloned from it (``ref_audio`` + ``ref_text``), which also carries the
    reference's pace and manner over. The reference lives in the user node's episode state (passed in as ``ref``).

    ``leveling`` (a ``Leveling``) post-processes every synthesized turn: silence trimmed, loudness normalised.
    ``seed=True`` sends one sampling seed (from the task id and speaker) with every turn of an episode, for TTS
    servers that take it.
    """

    def __init__(self, speech: Speech | None = None, voice: str = "default", words_per_sec: float = WORDS_PER_SEC,
                 voices: list[dict] | None = None, clone: Speech | bool | None = None, ref_min_words: int = 3,
                 style: str | None = "neutral", leveling: Leveling | None = Leveling(), seed: bool = True):
        self.speech = speech
        self.voice = voice
        self.wps = words_per_sec
        self.voices = voices
        if clone is None and speech is not None:  # cloning is the default: a TTS needs a cloning TTS beside it
            clone = speech if clones(speech) else None
            if clone is None:
                raise ValueError(NO_CLONE)
        if clone is True:  # the TTS clones by itself
            clone = speech
        self.clone = None if clone is False else clone
        self.ref_min_words = ref_min_words
        self.style = style
        self.leveling = leveling
        self.seed = seed

    def choice(self, task: Task) -> tuple[str, str]:
        """(speaker, why): ``"task"``, ``"profile"`` or ``"default"``."""
        sc = task.scenario
        if sc.get("voice"):
            return sc["voice"], "task"
        if self.voices and sc.get("profile"):
            return pick_voice(sc["profile"], self.voices)["speaker"], "profile"
        return self.voice, "default"

    def speaker(self, task: Task) -> str:
        return self.choice(task)[0]

    def language(self, task: Task) -> str | None:
        """The language the user speaks (``profile.language``), as a name, for TTS models that take it."""
        return language_name(task.scenario.get("profile", {}).get("language"))

    def instructions(self, task: Task) -> str | None:
        if self.style == "persona":
            sc = task.scenario
            return sc.get("voice_style") or sc.get("profile", {}).get("speaking_style")
        if self.style == "neutral":
            return NEUTRAL_STYLE
        return self.style or None

    def episode_seed(self, task: Task) -> int | None:
        if not self.seed:
            return None
        return int(hashlib.sha256(f"{task.id}/{self.speaker(task)}".encode()).hexdigest(), 16) % 2 ** 31

    def profile(self, task: Task) -> dict:
        if self.speech is None:  # a text-only timeline
            return {"tts": None, "words_per_sec": self.wps}
        speaker, why = self.choice(task)
        out = {"speaker": speaker, "chosen_by": why, "tts": describe(self.speech), "style": self.style}
        if self.instructions(task):
            out["instructions"] = self.instructions(task)
        if self.clone is not None:
            out["clone"] = {"tts": describe(self.clone), "reference": "first turn"}
        if self.leveling is not None:
            out["leveling"] = self.leveling.describe() | {"offset_db": self.leveling.offset(task, speaker)}
        if self.seed:
            out["seed"] = self.episode_seed(task)
        return out

    def reference(self, seg: Segment, ref: tuple | None) -> tuple | None:
        """The episode's voice reference after ``seg``: the first segment with real words and audio."""
        if ref is not None or self.clone is None or not isinstance(seg.data, Audio):
            return ref
        return (seg.data, seg.text) if n_words(seg.text or "") >= self.ref_min_words else None

    def max_ms(self, text: str) -> int:
        """The longest a turn with these words can plausibly take (TTS models occasionally run away,
        e.g. turning "mm-hmm" into half a minute of audio)."""
        return 3000 + round(n_words(text) / self.wps * 1000 * 2.5)

    async def _synth(self, text: str, task: Task, ref: tuple | None) -> Audio:
        seed = self.episode_seed(task)
        tts = self.clone if ref is not None and self.clone is not None else self.speech
        kw = {"seed": seed} if seed is not None and _accepts(tts.synth, "seed") else {}  # TTS clients without seeds: none sent
        if ref is not None and self.clone is not None:  # same voice as the reference turn
            audio = await self.clone.synth(text, self.speaker(task), self.instructions(task), ref_audio=ref[0], ref_text=ref[1],
                                           language=self.language(task), **kw)
        else:
            audio = await self.speech.synth(text, self.speaker(task), self.instructions(task), language=self.language(task), **kw)
        return audio if self.leveling is None else self.leveling.apply(audio, task, self.speaker(task))

    async def render(self, sid: str, t0: int, turn: UserTurn, task: Task, ref: tuple | None = None) -> Segment:
        audio = turn.audio
        if audio is None and self.speech is not None:
            if turn.text:
                audio = await self._synth(turn.text, task, ref)
                if audio.dur_ms > self.max_ms(turn.text):  # a runaway: try once more, then cut it to size
                    audio = await self._synth(turn.text, task, ref)
                    if audio.dur_ms > self.max_ms(turn.text):
                        audio = audio[: round(self.max_ms(turn.text) * audio.sr / 1000)]
            elif turn.dur:  # no words (an absence, a noise placeholder): nothing to synthesize
                audio = Audio.silence(turn.dur, getattr(self.speech, "sr", 16000))
        tags = {"kind": turn.kind, "expects": turn.expects, "intent": turn.intent, "label": turn.label}
        if audio is not None:
            return Segment(sid, t0, audio.dur_ms, audio, text=turn.text, **tags)
        dur = turn.dur if turn.dur is not None else round(n_words(turn.text) / self.wps * 1000)
        return Segment(sid, t0, dur, turn.text, text=turn.text, **tags)


# ---------------------------------------------------------------- offline


class ReplayUser(Node):
    """Plays pre-timed turns (``UserTurn.t``) regardless of the agent.

    Turns come from the constructor or from ``task.scenario["turns"]``.
    """

    def __init__(self, turns: list | None = None, voice: Voice | None = None, stream: str = "user.speech"):
        self.turns = turns
        self.voice = voice or Voice()
        self.stream = stream

    def init_state(self, task, rng):
        return {"task": task}

    async def step(self, st, t, inbox):
        if t != 0:
            return [], None
        task = st["task"]
        segs, ref = [], None
        for i, turn in enumerate(self._turns(task)):
            segs.append(await self.voice.render(f"u{i}", turn.t, turn, task, ref))
            ref = self.voice.reference(segs[-1], ref)
        return [Frame(self.stream, s.t0, s) for s in segs], None

    def _turns(self, task: Task) -> list[UserTurn]:
        return [as_turn(x) for x in (self.turns if self.turns is not None else task.scenario["turns"])]

    def profile(self, task: Task) -> dict:
        recorded = sum(t.audio is not None for t in self._turns(task))
        voice = self.voice.profile(task)
        if recorded:
            voice["recorded_turns"] = recorded
        return {"mode": "offline", **_scenario(task), "voice": voice}


# ---------------------------------------------------------------- content sources


class ScriptSource:
    """Fixed utterances in order (from the constructor or ``task.scenario["turns"]``); timing stays reactive."""

    def __init__(self, turns: list | None = None):
        self.turns = turns

    async def next(self, i: int, task: Task, convo: list[Utterance], barge_in: bool = False) -> UserTurn | None:
        turns = self.turns if self.turns is not None else task.scenario["turns"]
        return as_turn(turns[i]) if i < len(turns) else None

    def profile(self) -> dict:
        return {"mode": "semi_online"}


def _scenario(task: Task) -> dict:
    """Who the user is meant to be, as the task describes them."""
    sc = task.scenario
    return {k: v for k, v in (("persona", sc.get("persona")), ("profile", sc.get("profile")), ("goal", sc.get("instructions"))) if v}


def _persona(task: Task) -> str:
    """The persona text plus the structured profile, for the user LLM."""
    sc = task.scenario
    lang = language_name(sc.get("profile", {}).get("language"))
    return "\n".join(x for x in (sc.get("persona", ""), describe_profile(sc.get("profile", {})),
                                 f"Always speak {lang}." if lang else "") if x)




# "(pause)" / "(long pause)", also with full-width brackets or in Chinese ("（停顿）", "（长时间停顿）"), as LLMs writing
# Chinese turns produce them
_PAUSE_MARK = r"[(（]\s*(long pause|pause|长时间停顿|长停顿|停顿|停一下)\s*[)）]"
_PAUSE = re.compile(rf"^\s*{_PAUSE_MARK}\s*", re.I)
_INLINE = re.compile(rf"\s*{_PAUSE_MARK}\s*", re.I)


def _is_long(word: str) -> bool:
    return word.lower().startswith("long") or word.startswith("长")


def _spoken(raw: str, keep_pauses: bool = False) -> str:
    """What the user LLM wrote, cleaned to what a person would say: no bracketed markers or stage directions
    (copied prompt markers like "[CURRENTLY SPEAKING, INCOMPLETE]", "[Interrupting]", "[User cuts in]", or one cut
    off before its "]"), no *action* asides, no leading pause marker, no wrapping quotes. Inline pause markers are
    dropped, or with ``keep_pauses`` kept, normalised to "(pause)" / "(long pause)", where the user stops mid-thought
    (``pieces`` splits the turn there)."""
    text = re.sub(r"\[[^\]]*(\]|$)", "", raw)
    text = re.sub(r"\*[^*]*\*", "", text)
    text = _PAUSE.sub("", text.strip(), count=1)
    if keep_pauses:
        text = _INLINE.sub(lambda m: " (long pause) " if _is_long(m.group(1)) else " (pause) ", text)
    else:
        text = _INLINE.sub(" ", text)
        text = re.sub(rf"(?<=[{_CJK}，。！？、]) (?=[{_CJK}，。！？、])", "", text)  # no space left where a marker sat in Chinese
    return re.sub(r"\s+", " ", text).strip().strip('"“”「」').strip()


def _pause(raw: str) -> str | None:
    """The pause the user LLM asked for before this turn: "(pause)" → "short", "(long pause)" → "long"."""
    m = _PAUSE.match(raw.strip().strip('"“”'))
    return None if m is None else ("long" if _is_long(m.group(1)) else "short")


def pieces(text: str, max_pauses: int | None = None) -> list[tuple[str, str | None]]:
    """A turn split where the user stops mid-thought (its inline pause markers): ``[(words, pause after)]`` with the
    pause ``"short"`` / ``"long"`` / None (the last piece). A marker at the start or end, or two in a row, is no
    mid-thought pause; beyond ``max_pauses`` the rest stays one piece. Every piece but the last trails off ("...")."""
    parts = _INLINE.split(text or "")
    out: list[list] = []
    for i in range(0, len(parts), 2):
        words = parts[i].strip()
        after = None if i + 1 >= len(parts) else "long" if _is_long(parts[i + 1]) else "short"
        if not words:
            if out and after == "long":
                out[-1][1] = "long"
            continue
        out.append([words, after])
    if not out:
        return [("", None)]
    out[-1][1] = None
    if max_pauses is not None and len(out) > max_pauses + 1:
        out = out[:max_pauses] + [[" ".join(w for w, _ in out[max_pauses:]), None]]
    res = []
    for k, (words, after) in enumerate(out):
        if after is not None and not re.search(r"[.!?。！？…]$", words):
            words = words.rstrip(" ,;:，、；：") + "..."
        res.append((words, after))
    return res


# How the persona's hesitancy shows in what the user LLM writes (it places the mid-thought pauses itself)
HESITANCY = {
    "low": "You speak fluently: you almost never stop in the middle of a sentence.",
    "normal": "Like most people, now and then you stop mid-sentence for a moment when you have to recall or decide something.",
    "high": "You are hesitant: you often stop mid-sentence to think or to find the right word.",
}

USER_SYSTEM = """You are playing a user who is talking to a voice assistant in a live spoken conversation.

{persona}

Your goal:
{instructions}

Rules:
- Say only what the user says out loud: one or two short, natural spoken sentences.
- Reveal information gradually, only when it is asked for or needed.
- Messages from the assistant marked [CURRENTLY SPEAKING, INCOMPLETE] are still being said; you may cut in.
- Never say that you are an AI or a simulation.
- Speak only to the assistant: never write lines addressed to other people around you.
- If you would need a moment before answering (thinking, deciding, remembering), start with (pause); if it would take a while (looking something up, checking with someone), start with (long pause). Otherwise just answer.
- {hesitancy} Where you stop mid-sentence (to think, recall a detail, or find a word), write (pause) at that point, e.g. "It's for, (pause) let me think, six people." Never more than twice in one turn.
- When your goal is achieved, or clearly cannot be achieved, say a brief goodbye and end with {stop}."""


# Told to the user LLM when the listener decided to cut in: it picks why while writing the line (one call)
BARGE_IN_NOTE = ("(You are cutting in now, while the assistant is still speaking. First write one line \"INTENT: <correction | "
                 "question | stop | other>\" saying why: correction only if the assistant got something wrong (misheard or "
                 "misunderstood you, or said something false) — not when you simply answer, add details or change your mind; "
                 "question if you want to ask something; stop if you have heard enough or want it to stop or move on; other for "
                 "anything else (answering it early, adding or changing details, agreeing). Then, on the next line, write what "
                 "you say.)")
_INTENT = re.compile(r"^[\W_]*intent[\W_]*([a-z]+(?:/[a-z]+)?)", re.I)
_INTENT_WORDS = {"correction": "correction", "correct": "correction", "question": "question", "ask": "question", "stop": "stop",
                 "enough": "stop", "stop/enough": "stop", "other": "other"}


def parse_intent(raw: str) -> tuple[str, str]:
    """(intent, the rest of the text) of a barge-in line written as "INTENT: <intent>" then the words (the tag may also
    sit inline before the words). No tag, or an unknown intent: "other"."""
    m = _INTENT.match(raw or "")
    if m is None:
        return "other", raw or ""
    word = m.group(1).lower()
    intent = _INTENT_WORDS.get(word, _INTENT_WORDS.get(word.split("/")[0], "other"))
    return intent, re.sub(r"^[\s*_:)\].>-]+", "", raw[m.end():])


class LLMSource:
    """Generates each turn with a text LLM from the persona, goal and conversation heard so far.

    When the user cuts in (``barge_in``), the same generation call is told so (``BARGE_IN_NOTE``) and also returns
    why — ``INTENT: correction | question | stop | other`` on its first line (``parse_intent``); the turn carries it
    as ``intent`` ("other" when the tag is missing).

    The first turn can be given (text, or audio with its transcript) via ``first`` or
    ``task.scenario["first_turn"]``; otherwise it is generated too. The LLM places the user's mid-thought pauses
    itself ("(pause)" inside a turn, guided by the persona's ``hesitancy``); they stay in the returned text, and
    ``UserSim`` says the turn in pieces there.
    """

    def __init__(self, llm: TextGen, first=None, system: str = USER_SYSTEM):
        self.llm = llm
        self.first = first
        self.system = system

    def profile(self) -> dict:
        out = {"mode": "online", "llm": describe(self.llm)}
        if self.system != USER_SYSTEM:
            out["system_prompt"] = self.system
        return out

    def messages(self, task: Task, convo: list[Utterance], barge_in: bool = False) -> list[dict]:
        sc = task.scenario
        level = persona_levels(sc.get("profile") or {})[0]["hesitancy"]
        msgs = [{"role": "system", "content": self.system.format(persona=_persona(task), instructions=sc.get("instructions", ""), stop=STOP,
                                                                 hesitancy=HESITANCY[level])}]
        for u in convo:  # the simulator plays "assistant"; the agent's speech is its input
            if u.role == "user" and u.kind in SIDE_KINDS:
                # the user's own non-directed sounds are not this LLM's lines: as its own lines ("yeah", "wait, what was
                # that") they taught it to answer in one-word fragments ("I", "What", "Sorry")
                continue
            text = u.text + (" [CURRENTLY SPEAKING, INCOMPLETE]" if u.incomplete else "")
            role = "assistant" if u.role == "user" else "user"
            if len(msgs) > 1 and msgs[-1]["role"] == role:  # one turn said in pieces (a mid-thought pause, a stream
                # split by the other side's sounds) stays one message
                msgs[-1] = {"role": role, "content": msgs[-1]["content"].replace(" [CURRENTLY SPEAKING, INCOMPLETE]", "") + " " + text}
            else:
                msgs.append({"role": role, "content": text})
        if not any(m["role"] == "user" for m in msgs):  # nothing heard from the agent yet (e.g. it stayed silent)
            msgs.insert(1, {"role": "user", "content": "(The call has just connected. Start the conversation.)"})
        if msgs[-1]["role"] == "assistant":  # the agent said nothing since the user's last words (a nudge): without a
            # closing user message the chat model continues its own last line word by word ("How many" / "people" / "?")
            msgs.append({"role": "user", "content": "(Silence: the assistant has not said anything since your last words.)"})
        if barge_in:
            msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + "\n\n" + BARGE_IN_NOTE}
        return msgs

    async def next(self, i: int, task: Task, convo: list[Utterance], barge_in: bool = False) -> UserTurn | None:
        first = self.first if self.first is not None else task.scenario.get("first_turn")
        if i == 0 and first is not None:
            return as_turn(first)
        msgs = self.messages(task, convo, barge_in)
        intent = None
        for _ in range(2):  # nothing sayable (e.g. only a marker or empty): ask once more
            raw = await self.llm.chat(msgs)
            if barge_in:
                intent, raw = parse_intent(raw)
            if _spoken(raw) or STOP in raw:
                break
        text, pause = _spoken(raw, keep_pauses=True), _pause(raw)
        if STOP in text:
            text = text.replace(STOP, "").strip()
            return UserTurn(text, final=True, pause=pause, intent=intent) if _spoken(text) else None
        return UserTurn(text, pause=pause, intent=intent)


# ---------------------------------------------------------------- what the user does while the agent talks


class KeywordInterrupt:
    """A rule (for tests and scripted users): cut in once the agent's ongoing utterance contains any of the words."""

    def __init__(self, *words: str):
        self.words = [w.lower() for w in words]

    def profile(self) -> dict:
        return {"type": "keyword", "words": self.words}

    async def __call__(self, task: Task, convo: list[Utterance]) -> bool:
        cur = next((u for u in reversed(convo) if u.role == "agent" and u.incomplete), None)
        return cur is not None and any(w in cur.text.lower() for w in self.words)


LISTEN, BACKCHANNEL, INTERRUPT = "LISTEN", "BACKCHANNEL", "INTERRUPT"
INTENTS = ("correction", "question", "stop", "other")

# Adapted from τ-Voice (tau2-bench, MIT License, Sierra Research): its barge-in prompt, extended from a YES/NO
# interrupt decision to LISTEN / BACKCHANNEL / INTERRUPT, with the user's persona, goal and listening style.
LISTENER_PROMPT = """You are analyzing a live phone conversation to decide what the user does at this moment, while the agent is speaking.

The user:
{persona}

The user's goal:
{goal}

How the user listens: {style}

Conversation history (most recent at bottom):

<conversation_history>
{history}
</conversation_history>

The agent is CURRENTLY speaking (you can see their ongoing speech in the conversation above). {moment}

What does the user do NOW? The options:
{options}
{consider}
Respond with ONLY {answer}"""

_MOMENT = {"boundary": "It has just reached the end of a phrase.",
           "fallback": "It has been talking for a few seconds without a pause."}
_OPTION = {
    LISTEN: "- LISTEN: keep listening in silence.",
    BACKCHANNEL: ("- BACKCHANNEL: make a short listening sound to show they are following, without taking the turn (such as {sounds}). "
                  "Only where this user would naturally make one: typically when the agent has finished a point or a step; not in the "
                  "middle of a phrase, and not at every pause."),
    INTERRUPT: "- INTERRUPT: cut in and take the turn (to correct a mistake, ask something, stop the agent, or anything else).",
}
# τ-Voice's considerations for barging in
_CONSIDER = """
Consider, before interrupting:
- Has the user heard enough to understand what the agent is asking or saying?
- Has the user heard enough to have a response, question, or correction ready?
- Did the agent just complete the sentence which has all the pertinent information the user was looking for?
- Do NOT repeatedly interrupt the agent if it has spoken only a few words (say less than 5 words).
"""


@dataclass(frozen=True)
class Decision:
    """One decision while the agent talks: ``choice`` (LISTEN | BACKCHANNEL | INTERRUPT) and the backchannel's
    ``token`` if the model gave a usable one. A barge-in's ``intent`` is not decided here: the user LLM picks it while
    writing what it says (``LLMSource``); ``intent`` stays None."""

    choice: str
    token: str | None = None
    intent: str | None = None
    raw: str = ""


_CHOICE = re.compile(r"\b(LISTEN|BACKCHANNEL|INTERRUPT|YES|NO)\b\s*[:：\-—]?\s*(.*)", re.I | re.S)


def parse_decision(answer: str, options) -> Decision:
    """The model's answer as a ``Decision``: the first option word in it (τ-Voice's YES / NO count as INTERRUPT /
    LISTEN); anything unusable, or an option that was not offered, is LISTEN."""
    m = _CHOICE.search(answer or "")
    if m is None:
        return Decision(LISTEN, raw=answer or "")
    word = {"YES": INTERRUPT, "NO": LISTEN}.get(m.group(1).upper(), m.group(1).upper())
    if word not in options:
        return Decision(LISTEN, raw=answer)
    rest = (m.group(2).strip().splitlines() or [""])[0]
    if word == BACKCHANNEL:
        tok = _spoken(rest).strip(" .!,;:\"'“”()")
        ok = bool(tok) and n_words(tok) <= 3 and len(tok) <= 24 and not _CHOICE.search(tok)
        return Decision(word, tok if ok else None, raw=answer)
    return Decision(word, raw=answer)


def _history_line(u: Utterance) -> str | None:
    if u.role == "user" and u.kind in ("noise", "away"):
        return None
    who = {"backchannel": "USER (listening sound)", "aside": "USER (to someone else)"}.get(u.kind, u.role.upper()) if u.role == "user" else "AGENT"
    return f"{who}: {u.text}" + (" [CURRENTLY SPEAKING, INCOMPLETE]" if u.incomplete else "")


class LLMListener:
    """The user's listening behaviour, decided by a model: at each decision point while the agent talks (the end of
    one of its phrases, or ``TurnTaking.max_decision_gap_ms`` without one), one call returns LISTEN, BACKCHANNEL
    (with the sound) or INTERRUPT (why is left to the user LLM writing the line), given the heard conversation, the
    user's persona and goal, and their listening style (the profile's ``backchanneling`` level as a description; the model decides, the persona
    only informs). No call before ``min_words`` of the agent's turn were heard (τ-Voice). Sampling is greedy
    (``params``: temperature 0) and answers are memoised by prompt, so a seed replays the same decisions.

    Used as ``UserSim(interrupt=LLMListener(llm))`` it decides all three; as ``UserSim(listener=...)`` (with
    ``interrupt=None`` or a rule) it decides LISTEN / BACKCHANNEL only. Called as ``await listener(task, convo)`` it
    answers τ-Voice's question alone (barge in now? → bool)."""

    def __init__(self, llm: TextGen, min_words: int = 5, params: dict | None = None):
        self.llm = llm
        self.min_words = min_words
        self.params = {"temperature": 0.0, "max_tokens": 16} if params is None else dict(params)
        self._memo: dict[str, str] = {}
        self.calls = 0  # model calls actually made (memo misses)

    def profile(self) -> dict:
        return {"type": "llm", "llm": describe(self.llm), "min_words": self.min_words, "params": self.params}

    def prompt(self, task: Task, convo: list[Utterance], options, style: str = "", sounds=(), moment: str = "boundary") -> str:
        lines = [x for x in map(_history_line, convo) if x]
        opts = "\n".join(_OPTION[o].format(sounds=", ".join(f'"{s}"' for s in sounds[:4]) or '"mm-hmm"') for o in options)
        answer = "one of: " + ", ".join({LISTEN: "LISTEN", BACKCHANNEL: "BACKCHANNEL: <the sound>", INTERRUPT: "INTERRUPT"}[o]
                                        for o in options) + "."
        return LISTENER_PROMPT.format(persona=_persona(task) or "(no description)", goal=task.scenario.get("instructions", "") or "(not given)",
                                      style=style or "(not described)", history="\n".join(lines), moment=_MOMENT[moment],
                                      options=opts, consider=_CONSIDER if INTERRUPT in options else "", answer=answer)

    async def decide(self, task: Task, convo: list[Utterance], options=(LISTEN, BACKCHANNEL, INTERRUPT), style: str = "",
                     sounds=(), moment: str = "boundary") -> Decision:
        p = self.prompt(task, convo, options, style, sounds, moment)
        if p not in self._memo:
            self.calls += 1
            self._memo[p] = await self.llm.chat([{"role": "user", "content": p}], **self.params)
        return parse_decision(self._memo[p], options)

    async def __call__(self, task: Task, convo: list[Utterance]) -> bool:
        cur = next((u for u in reversed(convo) if u.role == "agent" and u.incomplete), None)
        if cur is None or n_words(cur.text) < self.min_words:
            return False
        return (await self.decide(task, convo, (LISTEN, INTERRUPT))).choice == INTERRUPT


LLMInterrupt = LLMListener  # the old name: interrupt=LLMInterrupt(llm) decides LISTEN / BACKCHANNEL / INTERRUPT


# ---------------------------------------------------------------- reactive user


@dataclass(frozen=True)
class Behaviors:
    """How an online user sounds besides its turns, and the safety nets around it.

    What the user decides (subjective) is decided by a model, never rolled: backchannels and barge-ins by the
    listener (``LLMListener``) at the agent's phrase boundaries; mid-thought pauses by the user LLM inside its turn
    ("(pause)"). The persona only informs them (``backchanneling`` / ``hesitancy`` levels as descriptions).

    What just happens to the user (not subjective) is random, on the call clock and seeded:

    - noise events (a cough, a door slam, a dog barking, a horn ...): a compound Poisson process
      (``soundscape.EventProcess``) with per-label onset rates by surroundings (``soundscape.EVENT_RATES``; ``noise_per_min``
      scales the table to that many onsets per minute, ``noise_rates`` gives {label: rate} outright), each onset a burst
      of one or more occurrences (``soundscape.EVENT_TYPES``), at a jittered per-label level. They happen whenever they
      happen: the user's own involuntary sounds (cough, sneeze) may overlap its speech, sounds of the surroundings
      overlap whoever is talking. No safety cap applies to them;
    - ``aside_per_min`` times being addressed by someone nearby (Poisson), taken up only when it is socially plausible
      (the user is not saying something, about to, or in the middle of a sequence; else it waits). The user then says
      a remark to that person; while the agent talks it is, with probability ``away_p``, being called away: "Sorry,
      hold on a second" (to the agent: ``expects: "wait"``), the remark, an absence of ``away_ms`` (``kind: "away"``,
      expects wait), and "Sorry about that, go on". The lines are written by the user LLM (``AsideWriter``) in the
      persona's language and voice, for the surroundings and the call so far; ``asides`` / ``HOLD_ON`` / ``BACK`` are
      the fixed fallback without a model.

    ``None`` for a rate = from the surroundings (``SURROUNDINGS``; asides x2 with low ``attentiveness``), as in ``PERSONA``.

    Safety nets only (they should rarely bind): user sounds at least ``min_gap_ms`` apart, at most
    ``max_backchannels_per_turn`` backchannels and ``max_asides_per_turn`` random events per agent turn, at most
    ``max_pauses_per_turn`` mid-thought pauses in one turn. Pause lengths: ``pause_ms`` ("(pause)"),
    ``long_pause_ms`` ("(long pause)"). ``backchannels``: the sounds the listener is offered (() = the language's).

    Deprecated: ``backchannel_per_min`` (now the nearest ``backchanneling`` level), ``pause_p`` (a random
    mid-thought split of a reply of ``pause_min_words``+ words; off by default) and the per-check ``*_p``."""

    backchannel_per_min: float | None = None  # deprecated
    aside_per_min: float | None = 0.0
    noise_per_min: float | None = 0.0
    pause_p: float | None = 0.0  # deprecated
    min_gap_ms: int = 3000
    max_backchannels_per_turn: int | None = 3
    max_asides_per_turn: int | None = 1  # asides (being addressed, called away) while one agent turn plays
    max_pauses_per_turn: int = 2
    away_p: float = 0.3
    away_ms: tuple[int, int] = (3000, 8000)
    pause_ms: tuple[int, int] = (1200, 2500)
    long_pause_ms: tuple[int, int] = (3000, 6000)
    pause_min_words: int = 6
    noise_ms: tuple[int, int] = (300, 700)  # unused (noise lengths come from the clips)
    noise_rates: dict | None = None  # {label: onsets per minute}: replaces the surroundings' table (soundscape.EVENT_RATES)
    backchannels: tuple[str, ...] = ()
    asides: tuple[str, ...] = ()
    backchannel_p: float = 0.0  # deprecated: per-check probabilities
    aside_p: float = 0.0
    noise_p: float = 0.0

    def __post_init__(self):
        if self.backchannel_p or self.aside_p or self.noise_p:
            warnings.warn("Behaviors backchannel_p / aside_p / noise_p (per check) are deprecated: use aside_per_min / noise_per_min "
                          "(per minute of the call); backchannels are decided by the listener model", DeprecationWarning, stacklevel=3)
        if self.backchannel_per_min is not None:
            warnings.warn("Behaviors.backchannel_per_min is deprecated: backchannels are decided by the listener model (LLMListener); "
                          "the rate is mapped to the nearest profile backchanneling level", DeprecationWarning, stacklevel=3)
        if self.pause_p:
            warnings.warn("Behaviors.pause_p is deprecated: mid-thought pauses are placed by the user LLM ('(pause)' in its turn, "
                          "guided by the profile's hesitancy)", DeprecationWarning, stacklevel=3)

    def rates(self) -> dict[str, float]:
        """{kind: random events per minute of the call}."""
        return {"aside": self.aside_per_min or 0.0, "noise": self.noise_per_min or 0.0}

    def resolve(self, check_ms: int = 1000, profile: dict | None = None, background: bool = False) -> tuple[Behaviors, dict]:
        """(the effective behaviours, the persona levels in effect): deprecated per-check probabilities turned into
        rates (``p * 60000 / check_ms``), unset event rates filled in from the profile's ``surroundings``; the levels
        dict has ``backchanneling`` / ``surroundings`` / ``hesitancy`` / ``attentiveness`` and ``inferred`` (the
        levels not given in the profile). A deprecated ``backchannel_per_min`` sets ``backchanneling``."""
        b = self
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for kind in ("backchannel", "aside", "noise"):
                p = getattr(b, f"{kind}_p")
                if p:
                    b = replace(b, **{f"{kind}_p": 0.0, f"{kind}_per_min": p * 60000 / check_ms})
            levels, inferred = persona_levels(profile or {}, background)
            if b.backchannel_per_min is not None:
                r = b.backchannel_per_min
                levels["backchanneling"] = "none" if r <= 0 else "rare" if r <= 1.5 else "normal" if r <= 3.5 else "frequent"
                inferred.discard("backchanneling")
                b = replace(b, backchannel_per_min=None)
            aside, noise = SURROUNDINGS[levels["surroundings"]]
            if levels["attentiveness"] == "low":  # distracted: busier with whatever is around them
                aside *= 2
            b = replace(b, aside_per_min=aside if b.aside_per_min is None else b.aside_per_min,
                        noise_per_min=noise if b.noise_per_min is None else b.noise_per_min, pause_p=b.pause_p or 0.0)
        out = {k: v for k, v in levels.items() if v is not None}
        if inferred:
            out["inferred"] = sorted(inferred)
        return b, out


# The persona's listening style, as the listener model is told it (it decides; the level only informs)
BACKCHANNELING = {
    "none": "The user listens in silence and never makes listening sounds.",
    "rare": "The user rarely makes listening sounds: mostly they listen in silence, at most an occasional \"mm-hmm\" after a long explanation.",
    "normal": ("Like most people on the phone, the user now and then makes a short listening sound (\"mm-hmm\", \"okay\") when the agent "
               "finishes a point or a step, but is silent most of the time."),
    "frequent": "The user often makes short listening sounds (\"mm-hmm\", \"yes\", \"right\") to show they are following, typically when the agent finishes a point.",
}
# Random events per minute of the call, by where the user is: (asides, noises). An aside needs someone nearby (home,
# office, cafe, a passenger); noisy places (street, cafe, car) make noise events likelier. With ~1-2 min calls this
# gives P(>= 1 event) of ~10-20% (quiet) to ~60-80% (street, cafe).
# The noise rate is the total of the per-label onset rates (soundscape.EVENT_RATES).
_ASIDE_RATES = {"quiet": 0.0, "home": 0.3, "office": 0.2, "car": 0.1, "cafe": 0.2, "street": 0.1}
SURROUNDINGS = {k: (a, round(sum(EVENT_RATES[k].values()), 4)) for k, a in _ASIDE_RATES.items()}

_FREQUENT = re.compile(r"backchannel|mm-hmm|uh-huh|while listening|chatty|talkative")
_RARE = re.compile(r"\b(shy|quiet|gruff|terse|curt|clipped|short answers|reserved)\b")
_HESITANT = re.compile(r"paus|hesitat|thinks out loud|find(?:s|ing)? (?:the )?words|\bslow")
_FLUENT = re.compile(r"\b(fast|quick|precise|clipped|brisk|businesslike|confident)\b")
_PLACES = [("street", r"\b(street|road|outside|traffic|walking)\b"), ("car", r"\b(car|driving)\b"),
           ("cafe", r"\b(cafe|café|coffee shop)\b"), ("office", r"\b(office|at work|desk|colleagues?)\b"),
           ("home", r"\b(home|kids|children|family|baby|dog)\b")]


def persona_levels(profile: dict, background: bool = False) -> tuple[dict, set]:
    """The profile's behaviour levels, and which of them were inferred.

    ``backchanneling`` (none | rare | normal | frequent), ``surroundings`` (quiet | home | office | car | cafe |
    street), ``hesitancy`` (low | normal | high); ``attentiveness`` (low | normal | high) passes through. Unset
    levels are inferred from the free text of ``speaking_style`` and ``traits`` (e.g. "says 'mm-hmm' while
    listening" → frequent, "shy, quiet" → rare, "hesitates, pauses mid-sentence" → high, "fast, businesslike" →
    low; a place named there → that surroundings), low attentiveness → rare backchannels, a scenario with a
    background noise bed (``background``) → cafe; otherwise normal / quiet / normal."""
    text = " ".join(str(profile.get(k) or "") for k in ("speaking_style", "traits")).lower()
    att = profile.get("attentiveness")
    levels, inferred = {"attentiveness": att}, set()

    def pick(key, guess):
        if profile.get(key):
            levels[key] = profile[key]
        else:
            levels[key] = guess
            inferred.add(key)

    pick("backchanneling", "frequent" if _FREQUENT.search(text) else "rare" if _RARE.search(text) or att == "low" else "normal")
    place = next((name for name, rx in _PLACES if re.search(rx, text)), None)
    pick("surroundings", place or ("cafe" if background else "quiet"))
    pick("hesitancy", "high" if _HESITANT.search(text) else "low" if _FLUENT.search(text) else "normal")
    for key, table in (("backchanneling", BACKCHANNELING), ("surroundings", SURROUNDINGS), ("hesitancy", HESITANCY)):
        if levels[key] not in table:
            raise ValueError(f"profile {key}={levels[key]!r}: one of {', '.join(table)}")
    return levels, inferred


PERSONA = Behaviors(aside_per_min=None, noise_per_min=None)  # random event rates from the persona's surroundings

# The sounds and lines of the defaults, by language (English for any other); explicit Behaviors values win
BACKCHANNELS = {"en": ("mm-hmm", "yeah", "okay", "uh-huh", "right"), "zh": ("嗯", "嗯嗯", "对", "好的", "哦")}
ASIDES = {  # said to someone else nearby, by surroundings (a Chinese "等一下" would be heard as a request to the agent)
    "en": {"quiet": ("Sorry, I'm on the phone.",),
           "home": ("Not now, sweetie, I'm on the phone.", "Can you turn that down, please?", "Just leave it on the table, thanks.",
                    "Mum, I'm on the phone!"),
           "office": ("Yeah, I'll be there in five minutes.", "Can it wait? I'm on a call.", "Thanks, just put it on my desk."),
           "car": ("Can you check the map for me?", "Turn left up here, I think.", "Not now, I'm on the phone."),
           "cafe": ("Oh, thank you, that's mine.", "A flat white, please.", "Sorry, is anyone sitting here?"),
           "street": ("Sorry, excuse me.", "No thanks, I'm fine.", "Oops, sorry!")},
    "zh": {"quiet": ("我在打电话呢。",),
           "home": ("你先别吵，我打电话呢。", "放桌上就行，谢谢。", "妈，我在打电话！", "嘘，小点声。"),
           "office": ("好，我五分钟后过去。", "等会儿再说，我在打电话。", "谢谢，放我桌上吧。"),
           "car": ("帮我看一下导航。", "前面右转。", "先别说话，我打电话呢。"),
           "cafe": ("谢谢，这是我的。", "一杯拿铁，谢谢。", "不好意思，这儿有人吗？"),
           "street": ("不好意思，借过一下。", "不用了，谢谢。", "哎呀，对不起！")},
}
HOLD_ON = {"en": ("Sorry, can you hold on a second?", "Hang on, one moment.", "Sorry, just a second."),
           "zh": ("不好意思，稍等一下。", "等我一下，马上回来。")}  # to the agent, before being called away
BACK = {"en": ("Sorry about that. Go on.", "Okay, I'm back, sorry.", "Sorry, where were we?"),
        "zh": ("不好意思，我回来了。", "好了，你接着说吧。")}

# Who or what needs the user, by surroundings (one is drawn per event from the episode seed; the user LLM writes the lines)
SITUATIONS = {
    "quiet": ("someone comes into the room and says something to them", "a family member asks them a quick question"),
    "home": ("their child asks them for something", "their partner asks them a question from the next room",
             "someone rings the doorbell", "a family member wants to show them something", "the dog wants to go out"),
    "office": ("a colleague stops by their desk with a question", "their manager calls them over",
               "someone asks whether they are coming to the meeting", "a courier brings a parcel to their desk"),
    "car": ("a passenger asks them something", "a passenger points out the turn coming up", "a child in the back seat asks for something"),
    "cafe": ("the barista calls out their order", "a stranger asks whether the seat next to them is free",
             "a waiter asks whether they want anything else", "a friend arrives and greets them"),
    "street": ("a passer-by asks them for directions", "someone bumps into them", "a street vendor offers them something",
               "a friend recognises them and says hello"),
}

ASIDE_PROMPT = """You write what a person on a phone call says when someone near them needs their attention.

The person:
{persona}

They speak {language}. They are {place}. What happens: {situation}.

The phone call so far (most recent at bottom):
{history}

Write {what}
Each line: short (at most 12 words), natural spoken {language}, in this person's voice, fitting the place, the situation and the call. No quotes, no stage directions, no names of the other person unless natural.

Answer in exactly this format:
{fmt}"""
_PLACE = {"quiet": "somewhere quiet", "home": "at home", "office": "at the office", "car": "in a car",
          "cafe": "in a cafe", "street": "outside on the street"}
_LINE = re.compile(r"^\W*(HOLD|ASIDE|BACK)\W*[:：]\s*(.+)$", re.I | re.M)


class AsideWriter:
    """The lines of an aside or of being called away, written by the user LLM (one call per event): the remark to the
    person nearby (``ASIDE``) and, when called away, the "hold on" to the agent before (``HOLD``) and the "I'm back"
    after (``BACK``) — in the persona's language and voice, for the surroundings, the drawn situation and the call so
    far. Greedy (``params``) and memoised by prompt, so a seed replays the same lines. A line that is missing or too
    long (over ``max_words``) falls back to the fixed lists (``ASIDES`` / ``HOLD_ON`` / ``BACK``)."""

    def __init__(self, llm: TextGen, params: dict | None = None, max_words: int = 16):
        self.llm = llm
        self.params = {"temperature": 0.0, "max_tokens": 96} if params is None else dict(params)
        self.max_words = max_words
        self._memo: dict[str, str] = {}
        self.calls = 0

    def profile(self) -> dict:
        return {"type": "llm", "llm": describe(self.llm), "params": self.params}

    def prompt(self, task: Task, convo: list[Utterance], place: str, situation: str, away: bool) -> str:
        lang = language_name(_lang(task)) or "English"
        lines = [x for x in map(_history_line, convo) if x][-8:]
        if away:
            what = ("three lines. HOLD: what they say to the caller (the assistant) to ask it to wait a moment. ASIDE: what they then "
                    "say to the person near them (only to that person, nothing to the caller). BACK: what they say to the caller when "
                    "they come back a little later.")
            fmt = "HOLD: <line>\nASIDE: <line>\nBACK: <line>"
        else:
            what = ("one line. ASIDE: only the words they say to the person near them — nothing addressed to the caller (they "
                    "go back to the call afterwards on their own).")
            fmt = "ASIDE: <line>"
        return ASIDE_PROMPT.format(persona=_persona(task) or "(no description)", language=lang, place=_PLACE.get(place, place),
                                   situation=situation, history="\n".join(lines) or "(it has just started)", what=what, fmt=fmt)

    async def write(self, task: Task, convo: list[Utterance], place: str, situation: str, away: bool) -> dict[str, str]:
        """{"ASIDE": ..., "HOLD": ..., "BACK": ...}: the usable lines the model wrote (possibly none)."""
        p = self.prompt(task, convo, place, situation, away)
        if p not in self._memo:
            self.calls += 1
            self._memo[p] = await self.llm.chat([{"role": "user", "content": p}], **self.params)
        out = {}
        for key, line in _LINE.findall(self._memo[p] or ""):
            text = _spoken(line).strip().strip("\"'“”")
            if text and n_words(text) <= self.max_words and key.upper() not in out:
                out[key.upper()] = text
        return out


@dataclass(frozen=True)
class ResponseDelay:
    """How long a user takes to start answering once the agent stops: a random, content-aware gap.

    A log-normal around ``median_ms`` (people answer fast, with a long tail); with probability
    ``distracted_p`` the user was not paying attention and the gap is drawn around
    ``distracted_median_ms`` instead; a turn the user LLM marked "(pause)" / "(long pause)" (it needs to
    think or look something up) adds ``pause_short_ms`` / ``pause_long_ms``. The profile's
    ``attentiveness`` ("low" / "normal" / "high") scales the distraction probability. Draws come from
    the episode's seeded RNG, so a seed reproduces the same delays."""

    median_ms: int = 300
    sigma: float = 0.6
    distracted_p: float = 0.1
    distracted_median_ms: int = 3000
    distracted_sigma: float = 0.4
    pause_short_ms: int = 800
    pause_long_ms: int = 2500
    max_ms: int = 10_000

    def sample(self, rng, pause: str | None = None, attentiveness: str | None = None) -> int:
        p = self.distracted_p * {"low": 2.5, "high": 0.3}.get(attentiveness or "", 1.0)
        median, sigma = (self.distracted_median_ms, self.distracted_sigma) if rng.random() < p else (self.median_ms, self.sigma)
        gap = median * math.exp(rng.gauss(0.0, sigma))
        gap += {"short": self.pause_short_ms, "long": self.pause_long_ms}.get(pause or "", 0)
        return int(min(max(gap, 0), self.max_ms))


@dataclass(frozen=True)
class TurnTaking:
    speaks_first: bool = True
    respond_after_ms: int = 1000  # silence after the agent before the user replies (τ-Voice wait-to-respond); fixed
    response_delay: ResponseDelay | None = None  # a random, content-aware gap instead (replaces respond_after_ms)
    nudge_after_ms: int = 5000  # silence after the user's own turn before speaking again
    max_decision_gap_ms: int = 3000  # while the agent talks without a phrase boundary, decide at least this often
    check_ms: int = 1000  # after yielding to the agent, how soon the user looks again
    reaction_ms: int = 300  # barge-in decision → start of speech
    yield_after_ms: int | None = 1000  # how long the user keeps talking when the agent talks over it; None = never yields


# Where a listener may react: after a phrase-final mark followed by a space (so not "3.5" or "e.g.x"), or after
# full-width CJK punctuation
_BOUNDARY = re.compile(r"[,.;:!?…—](?=\s)|[，。；：！？、]")
SIDE_KINDS = ("backchannel", "aside", "noise", "away")  # the user's turns that are not its own lines to the agent


def phrase_boundaries(seg: Segment) -> list[int]:
    """The times inside ``seg`` at which its listener has just heard a phrase end (a punctuation mark of its
    transcript, mapped to time the way ``heard_text`` maps it: linearly over the segment). Times only: what comes
    after a boundary is never part of a decision made there."""
    text = seg.transcript
    n = len(text)
    if not n or seg.dur <= 0:
        return []
    return [seg.t0 + math.ceil(seg.dur * m.end() / n) for m in _BOUNDARY.finditer(text) if m.end() < n]


def _lang(task: Task) -> str:
    return (task.scenario.get("profile", {}).get("language") or "en").lower()


class UserSim(Node):
    """Reactive simulated user: decides when to speak, gets the content from ``source``, renders it with ``voice``.

    While the agent talks, the user considers what to do at each *decision point*: the end of one of the agent's
    phrases (``phrase_boundaries``), or ``timing.max_decision_gap_ms`` after the last point if the agent never
    pauses. There ``interrupt`` (a rule such as ``KeywordInterrupt``) may barge in; then the listener model
    (``interrupt`` if it is an ``LLMListener``, else ``listener``) chooses LISTEN / BACKCHANNEL / INTERRUPT (only the
    options this user has: no INTERRUPT unless ``interrupt`` is the model; BACKCHANNEL only at a phrase boundary, never
    for a ``backchanneling: none`` persona, nor while a safety cap holds). ``interrupt=None`` and no ``listener``: the user only listens.

    Random events (noises, being addressed or called away; ``Behaviors``) come from seeded processes on the call clock
    that do not depend on what the agent does. ``soundscape`` adds the persona's background track to the episode
    (``soundscape.background_spec``) and supplies noise recordings; None: no background, synthetic noises.
    ``aside_writer`` writes the lines of asides and of being called away: an ``AsideWriter``, ``"auto"`` (one on the
    source's LLM when it has one) or None (the fixed lines).

    Every sound the user makes is its own turn, labelled by its producer (``kind`` / ``expects`` / ``intent`` /
    ``label``; docs/FORMAT.md §4.4)."""

    def __init__(
        self,
        source: ScriptSource | LLMSource,
        voice: Voice | None = None,
        timing: TurnTaking = TurnTaking(),
        interrupt=None,  # an LLMListener (model decides, barge-ins included), a rule async (task, convo) -> bool, or None
        stream: str = "user.speech",
        agent_stream: str = "policy.speech",
        behaviors: Behaviors = Behaviors(),
        listener: LLMListener | None = None,  # the model deciding backchannels when interrupt is not one
        soundscape=None,
        aside_writer="auto",
    ):
        from .soundscape import Soundscape

        self.behaviors = behaviors
        self.source = source
        self.voice = voice or Voice()
        self.timing = timing
        self.interrupt = interrupt
        self.listener = interrupt if isinstance(interrupt, LLMListener) else listener
        self.soundscape = Soundscape() if soundscape is None else soundscape
        llm = getattr(source, "llm", None)
        self.aside_writer = (AsideWriter(llm) if llm is not None else None) if aside_writer == "auto" else aside_writer
        self.stream = stream
        self.reads = (agent_stream,)

    # ------------------------------------------------------------ configuration

    def init_state(self, task, rng):
        from .soundscape import EventProcess

        b, levels = self._resolve(task)
        # the episode seed and the task: two tasks run with the same seed get different events
        base = f"{rng.getrandbits(64)}:{task.id}"
        st = {"task": task, "timing": self._timing(task), "behaviors": b, "levels": levels, "rng": rng, "agent": {}, "mine": {},
              "n": 0, "uid": 0, "done": False, "planned": None, "pending": [], "sounds": 0, "last_sound": None, "per_turn": {}, "dp": {},
              "erng": {k: random.Random(f"{base}:{k}") for k in ("noise", "aside")}, "next_event": {},
              "noise": EventProcess(self._noise_rates(b, levels), base),
              "log": {"points": 0, "boundary": 0, "fallback": 0, "too_few_words": 0, "capped": 0, "llm_calls": 0,
                      LISTEN: 0, BACKCHANNEL: 0, INTERRUPT: 0, "rule_interrupts": 0},
              "events": {"noise": 0, "noise_occurrences": 0, "noise_over_user": 0, "aside": 0, "called_away": 0, "capped": 0,
                         "lines_llm": 0, "lines_fixed": 0}, "noise_log": []}
        st["next_event"]["aside"] = self._draw(st, "aside", 0)
        return st

    @staticmethod
    def _noise_rates(b: Behaviors, levels: dict) -> dict[str, float]:
        from .soundscape import event_rates

        return event_rates(levels["surroundings"], b.noise_per_min, b.noise_rates)

    def _resolve(self, task: Task) -> tuple[Behaviors, dict]:
        """This user's habits and the persona levels in effect: the constructor's ``behaviors`` with any
        ``task.scenario["behaviors"]`` overrides, the language's sounds where none were given, then what is left
        unset derived from ``task.scenario["profile"]`` (``Behaviors.resolve``)."""
        sc = task.scenario
        over = {k: tuple(v) if isinstance(v, list) else v for k, v in sc.get("behaviors", {}).items()}
        for kind in ("aside", "noise"):  # whichever form the scenario gives replaces both forms of the base
            if f"{kind}_per_min" in over:
                over.setdefault(f"{kind}_p", 0.0)
            elif over.get(f"{kind}_p"):
                over[f"{kind}_per_min"] = 0.0
        lang = _lang(task)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # warned once already if the constructor used them
            b = replace(self.behaviors, **over)
            b, levels = b.resolve(self._timing(task).check_ms, sc.get("profile"), _has_background(sc))
            if not b.backchannels:
                b = replace(b, backchannels=BACKCHANNELS.get(lang, BACKCHANNELS["en"]))
            if not b.asides:
                b = replace(b, asides=ASIDES.get(lang, ASIDES["en"])[levels["surroundings"]])
        return b, levels

    def _behaviors(self, task: Task) -> Behaviors:
        return self._resolve(task)[0]

    def _timing(self, task: Task) -> TurnTaking:
        """This user's turn-taking: the defaults, with any ``task.scenario["turn_taking"]`` overrides
        (e.g. ``{"yield_after_ms": None}`` for someone who never stops when talked over, or
        ``{"response_delay": {"median_ms": 500}}``)."""
        over = dict(task.scenario.get("turn_taking", {}))
        if isinstance(over.get("response_delay"), dict):
            over["response_delay"] = replace(self.timing.response_delay or ResponseDelay(), **over["response_delay"])
        return replace(self.timing, **over)

    def background(self, task: Task, seed: int, sr: int) -> tuple[list[dict], list]:
        """The episode's background from the persona's surroundings or ``task.scenario["background"]`` (called by
        ``Env.reset``): (specs, tracks)."""
        from .soundscape import background_spec, render_background

        if not self.soundscape.background:
            return [], []
        sc = task.scenario
        surroundings = persona_levels(sc.get("profile") or {}, _has_background(sc))[0]["surroundings"]
        spec = background_spec(sc.get("background"), surroundings, self.soundscape.bank)
        track = render_background(spec, sr, seed, self.soundscape.bank)
        return [spec], [track] if track is not None else []

    def profile(self, task: Task) -> dict:
        if self.interrupt is None:
            barge_in = {"type": "never"}
        else:
            barge_in = self.interrupt.profile() if hasattr(self.interrupt, "profile") else {"type": type(self.interrupt).__name__}
        src = self.source.profile()
        b, levels = self._resolve(task)
        habits = {k: round(v, 4) for k, v in (("aside_per_min", b.aside_per_min), ("noise_per_min", b.noise_per_min),
                                              ("pause_p", b.pause_p)) if v}
        if b.aside_per_min:
            habits["away_p"] = b.away_p
            habits["aside_lines"] = self._lines_profile(task)
        if b.noise_per_min or b.noise_rates:
            from .soundscape import EventProcess

            habits["noise_process"] = EventProcess(self._noise_rates(b, levels), 0).params()
        habits |= {"min_gap_ms": b.min_gap_ms, "max_backchannels_per_turn": b.max_backchannels_per_turn,
                   "max_asides_per_turn": b.max_asides_per_turn, "max_pauses_per_turn": b.max_pauses_per_turn, "persona": levels}
        listening = {"decision_points": "phrase boundaries", "max_decision_gap_ms": self._timing(task).max_decision_gap_ms}
        if self.listener is not None:
            listening["model"] = self.listener.profile()
            listening["options"] = [LISTEN] + ([BACKCHANNEL] if levels["backchanneling"] != "none" else []) + (
                [INTERRUPT] if self.listener is self.interrupt else [])
        else:
            listening["model"] = None
        return {"mode": src.pop("mode"), **_scenario(task), "voice": self.voice.profile(task), **src,
                "barge_in": barge_in, "listening": listening, "turn_taking": asdict(self._timing(task)), "behaviors": habits}

    def _lines_profile(self, task: Task) -> dict:
        if self.aside_writer is None or self._fixed_asides(task):
            return {"type": "fixed"}
        return self.aside_writer.profile()

    def _fixed_asides(self, task: Task) -> bool:
        """Aside lines were given (constructor or scenario): use them, not the model."""
        return bool(self.behaviors.asides or task.scenario.get("behaviors", {}).get("asides"))

    def report(self, st) -> dict:
        """What this episode's user did beyond its turns: how its decisions were made (decision points, model calls
        and choices, safety caps that held) and how many random events happened (for ``meta.user``)."""
        out = {"decisions": dict(st["log"]), "random_events": dict(st["events"])}
        if st["noise_log"]:
            out["noise_events"] = [dict(e) for e in st["noise_log"]]
        return out

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _latest(segs: dict, t: int) -> Segment | None:
        started = [s for s in segs.values() if s.t0 <= t]
        return max(started, key=lambda s: s.t0, default=None)

    @staticmethod
    def _mine(st) -> Segment | None:
        """The user's latest own line (not a sound, an aside or an absence)."""
        return max((s for s in st["mine"].values() if s.kind is None), key=lambda s: s.t0, default=None)

    @staticmethod
    def _last_end(st) -> Segment | None:
        """The user's latest-ending sound of its own (noise events aside: they never hold the user up)."""
        return max((s for s in st["mine"].values() if s.kind != "noise"), key=lambda s: s.end, default=None)

    def _draw(self, st, kind: str, t: int) -> int | None:
        rate = st["behaviors"].rates()[kind]
        return None if rate <= 0 else t + max(1, round(st["erng"][kind].expovariate(rate / 60000)))

    def _busy(self, st, t: int) -> bool:
        """Saying something, about to, or in the middle of a sequence: no random event now."""
        last = self._last_end(st)
        return bool(st["pending"]) or st["planned"] is not None or (last is not None and (last.end > t or last.t0 > t))

    async def _emit(self, st, t0: int, turn: UserTurn, sid: str | None = None) -> Frame:
        if sid is None:
            st["sounds"] += 1
            sid = f"s{st['sounds']}"
        seg = await self.voice.render(sid, t0, turn, st["task"], st.get("ref"))
        st["mine"][seg.id] = seg
        return Frame(self.stream, t0, seg)

    def _count(self, st, agent: Segment | None, key: str) -> None:
        if agent is not None:
            n = st["per_turn"].setdefault(agent.id, {"backchannel": 0, "other": 0})
            n[key] += 1

    # ------------------------------------------------------------ speaking

    async def _speak(self, st, t0: int, t: int, turn: UserTurn | None = None, barge_in: bool = False,
                     resumed: bool = False, intent: str | None = None) -> tuple[list[Frame], int | None]:
        if turn is None:
            convo = conversation(list(st["mine"].values()), list(st["agent"].values()), t)
            kw = {"barge_in": True} if barge_in and _takes(self.source.next, "barge_in") else {}
            turn = await self.source.next(st["n"], st["task"], convo, **kw)
        if barge_in and turn is not None and turn.expects is None and turn.kind is None:  # cutting in: the agent should yield
            turn = replace(turn, expects="yield", intent=intent or turn.intent)
        b = st["behaviors"]
        if turn is not None and turn.audio is None and turn.kind is None and not resumed:
            ps = pieces(turn.text, b.max_pauses_per_turn)
            if len(ps) > 1:  # mid-thought pauses the LLM placed: said in pieces, each but the last expects the agent to wait
                rest, prev = [], ps[0][1]
                for k, (words, after) in enumerate(ps[1:], 1):
                    last = k == len(ps) - 1
                    gap = st["rng"].randint(*(b.long_pause_ms if prev == "long" else b.pause_ms))
                    rest.append((replace(turn, text=words, pause=None, expects=turn.expects if last and turn.expects != "yield" else
                                         None if last else "wait", intent=None, final=turn.final and last), gap))
                    prev = after
                st["pending"] = rest + st["pending"]
                turn = replace(turn, text=ps[0][0], expects=turn.expects or "wait", final=False)
            elif turn.text != ps[0][0] and _INLINE.search(turn.text or ""):  # a stray marker (at the start or end)
                turn = replace(turn, text=ps[0][0])
            elif b.pause_p and n_words(turn.text) >= b.pause_min_words and st["rng"].random() < b.pause_p:  # deprecated
                split = _split_mid_thought(turn.text, st["rng"])
                if split is not None:
                    st["pending"] = [(replace(turn, text=split[1], pause=None, expects=None if turn.expects == "yield" else turn.expects),
                                      st["rng"].randint(*b.pause_ms))] + st["pending"]
                    turn = replace(turn, text=split[0], expects=turn.expects or "wait", final=False)
        if turn is None:  # nothing more to say: the user simply stops talking
            st["done"] = True
            return [], None
        if not turn.text and turn.audio is None and not turn.dur:  # nothing sayable came back: stay quiet, try again later
            return [], t + st["timing"].nudge_after_ms
        frame = await self._emit(st, t0, turn, sid=f"u{st['uid']}")
        seg = frame.data
        st["ref"] = self.voice.reference(seg, st.get("ref"))
        st["uid"] += 1
        if not resumed:  # a new turn from the source (not the rest of one, nor a scripted step of being called away)
            st["n"] += 1
        if turn.final:  # the last thing the user says; the conversation then winds down by itself
            st["done"] = True
            return [frame], None
        return [frame], (seg.t0 if t0 > t else seg.end)

    def _delay(self, st, turn: UserTurn) -> int:
        attentiveness = st["task"].scenario.get("profile", {}).get("attentiveness")
        return st["timing"].response_delay.sample(st["rng"], turn.pause, attentiveness)

    # ------------------------------------------------------------ deciding while the agent talks

    def _next_point(self, st, agent: Segment, after: int) -> int:
        """The next decision point after ``after``: the agent's next phrase boundary, or ``max_decision_gap_ms`` on."""
        nxt = next((b for b in phrase_boundaries(agent) if b > after), None)
        fallback = after + st["timing"].max_decision_gap_ms
        return fallback if nxt is None else min(nxt, fallback)

    def _bc_allowed(self, st, t: int, agent_id: str) -> bool:
        """The safety nets on backchannels (they only stop runaway behaviour; the model decides)."""
        b = st["behaviors"]
        if st["last_sound"] is not None and t - st["last_sound"] < b.min_gap_ms:
            return False
        n = st["per_turn"].get(agent_id, {}).get("backchannel", 0)
        return b.max_backchannels_per_turn is None or n < b.max_backchannels_per_turn

    async def _decide(self, st, agent: Segment, t: int, at: str) -> tuple[list[Frame], int | None] | None:
        tt, log, task = st["timing"], st["log"], st["task"]
        log["points"] += 1
        log[at] += 1
        convo = conversation(list(st["mine"].values()), list(st["agent"].values()), t)
        if self.interrupt is not None and self.interrupt is not self.listener:  # a rule
            if await self.interrupt(task, convo):
                log["rule_interrupts"] += 1
                return await self._speak(st, t + tt.reaction_ms, t, barge_in=True)
        if self.listener is None:
            return None
        options = [LISTEN]
        if st["levels"]["backchanneling"] != "none" and at == "boundary":  # a listening sound only where a phrase ends
            if self._bc_allowed(st, t, agent.id):
                options.append(BACKCHANNEL)
            else:
                log["capped"] += 1
        if self.listener is self.interrupt:
            options.append(INTERRUPT)
        if len(options) == 1:
            return None
        if n_words(agent.heard_text(t)) < self.listener.min_words:
            log["too_few_words"] += 1
            return None
        d = await self.listener.decide(task, convo, tuple(options), BACKCHANNELING[st["levels"]["backchanneling"]],
                                       st["behaviors"].backchannels, at)
        log["llm_calls"] += 1
        log[d.choice] += 1
        if d.choice == INTERRUPT:
            return await self._speak(st, t + tt.reaction_ms, t, barge_in=True)
        if d.choice == BACKCHANNEL:
            text, last = d.token, st.get("last_backchannel")
            if not text or text == last:  # the model's sound, but never the same one twice in a row (greedy decoding
                # would say "mm-hmm" every time): else one from the language's sounds, seeded
                text = st["rng"].choice([v for v in st["behaviors"].backchannels if v != last] or st["behaviors"].backchannels)
            st["last_backchannel"] = text
            frame = await self._emit(st, t, UserTurn(text, kind="backchannel", expects="ignore"))
            self._count(st, agent, "backchannel")
            st["last_sound"] = t
            return [frame], frame.data.end
        return None

    # ------------------------------------------------------------ random events

    async def _noises(self, st, t: int) -> list[Frame]:
        """The noise events due by ``t`` (``soundscape.EventProcess``), whatever the user or the agent is doing."""
        from .soundscape import event_audio

        out, proc = [], st["noise"]
        while (ev := proc.peek()) is not None and ev.t <= t:
            proc.pop()
            sr = getattr(self.voice.speech, "sr", 16000)
            audio, source, n = event_audio(ev, sr, st["erng"]["noise"], self.soundscape.bank)
            if self.voice.speech is None:  # a text-only timeline: just its length
                turn = UserTurn("", kind="noise", expects="ignore", label=ev.label, dur=audio.dur_ms)
            else:
                turn = UserTurn("", audio=audio, kind="noise", expects="ignore", label=ev.label)
            mine = self._mine(st)
            over = mine is not None and mine.t0 <= t < mine.end
            ev_log = st["events"]
            ev_log["noise"] += 1
            ev_log["noise_occurrences"] += n
            ev_log["noise_over_user"] += over
            st["noise_log"].append({"t": t, "label": ev.label, "n": n, "level_db": ev.level_db, "dur_ms": audio.dur_ms,
                                     "source": ev.source, "clip": source, "over_user_speech": over})
            out.append(await self._emit(st, t, turn))
        return out

    async def _random_event(self, st, t: int) -> tuple[list[Frame], int | None] | None:
        """Being addressed by someone nearby, when it is due and socially plausible (else it waits)."""
        x = st["next_event"].get("aside")
        if x is None or x > t or self._busy(st, t):  # deferred until the user is free
            return None
        b, rng = st["behaviors"], st["erng"]["aside"]
        if st["last_sound"] is not None and t - st["last_sound"] < b.min_gap_ms:
            st["next_event"]["aside"] = st["last_sound"] + b.min_gap_ms
            return None
        st["next_event"]["aside"] = self._draw(st, "aside", t)
        agent = self._latest(st["agent"], t)
        agent = agent if agent is not None and agent.active(t) else None
        if agent is not None and b.max_asides_per_turn is not None and st["per_turn"].get(agent.id, {}).get("other", 0) >= b.max_asides_per_turn:
            st["events"]["capped"] += 1
            return None
        self._count(st, agent, "other")
        st["last_sound"] = t
        task, lang = st["task"], _lang(st["task"])
        away = agent is not None and rng.random() < b.away_p  # called away: tell the agent, go, come back
        place = st["levels"]["surroundings"]
        situation = rng.choice(SITUATIONS.get(place, SITUATIONS["quiet"]))
        lines = {"ASIDE": rng.choice(b.asides), "HOLD": rng.choice(HOLD_ON.get(lang, HOLD_ON["en"])),
                 "BACK": rng.choice(BACK.get(lang, BACK["en"]))}  # the fixed fallback (always drawn: the same seed stream)
        if self.aside_writer is not None and not self._fixed_asides(task):
            convo = conversation(list(st["mine"].values()), list(st["agent"].values()), t)
            try:
                got = await self.aside_writer.write(task, convo, place, situation, away)
            except Exception as e:  # noqa: BLE001  (a failed call must not end the episode: the fixed lines stand in)
                warnings.warn(f"AsideWriter failed, using fixed lines: {e!r}")
                got = {}
            need = ("HOLD", "ASIDE", "BACK") if away else ("ASIDE",)
            st["events"]["lines_llm"] += sum(k in got for k in need)
            st["events"]["lines_fixed"] += sum(k not in got for k in need)
            lines |= got
        else:
            st["events"]["lines_fixed"] += 3 if away else 1
        aside = UserTurn(lines["ASIDE"], kind="aside", expects="ignore")
        if away:
            st["events"]["called_away"] += 1
            hold = UserTurn(lines["HOLD"], expects="wait", label="called_away")
            gone = UserTurn("", kind="away", expects="wait", dur=rng.randint(*b.away_ms), label="called_away")
            back = UserTurn(lines["BACK"], label="called_away")
            st["pending"] = [(replace(aside, label="called_away"), rng.randint(300, 800)), (gone, 0), (back, 0)] + st["pending"]
            return await self._speak(st, t + st["timing"].reaction_ms, t, hold, barge_in=True, resumed=True)
        st["events"]["aside"] += 1
        frame = await self._emit(st, t, aside)
        return [frame], frame.data.end

    def _next_event(self, st, t: int) -> int | None:
        nxt = [x for x in st["next_event"].values() if x is not None and x > t]
        ev = st["noise"].peek()
        if ev is not None:
            nxt.append(max(ev.t, t + 1))
        return min(nxt, default=None)

    # ------------------------------------------------------------ the step

    async def step(self, st, t, inbox):
        for f in inbox:
            st["agent"][f.data.id] = f.data
        mine = self._mine(st)
        if st["done"] and not st["pending"] and (mine is None or mine.end <= t):  # said everything: never starts a new
            return [], None  # turn, but still yields if talked over during its last one (below)
        noises = [] if st["done"] else await self._noises(st, t)
        out = None if st["done"] else await self._random_event(st, t)
        frames, wake = out if out is not None else await self._react(st, t)
        nxt = None if st["done"] else self._next_event(st, t)
        if nxt is not None:
            wake = nxt if wake is None else min(wake, nxt)
        return noises + frames, wake

    async def _react(self, st, t):
        tt = st["timing"]
        mine = self._mine(st)
        agent = self._latest(st["agent"], t)
        agent_speaking = agent is not None and agent.active(t)

        if mine is None and tt.speaks_first and not st["mine"]:
            return await self._speak(st, t, t)
        if mine is not None and mine.t0 > t:  # scheduled but not started yet
            return [], mine.t0
        if mine is not None and mine.active(t):
            if agent_speaking and agent.t0 > mine.t0:  # the agent is talking over me
                if tt.yield_after_ms is not None and t - agent.t0 >= tt.yield_after_ms:
                    cut = mine.cut(t)
                    st["mine"][cut.id] = cut
                    return [Frame(self.stream, t, cut)], t + tt.check_ms
                return [], mine.end if tt.yield_after_ms is None else min(agent.t0 + tt.yield_after_ms, mine.end)
            return [], mine.end
        if st["pending"]:  # the rest of a turn after a mid-thought pause, or the next step of being called away
            last = self._last_end(st)
            if last is not None and last.end > t:
                return [], last.end
            turn, gap = st["pending"][0]
            due = (last.end if last is not None else t) + gap
            if t < due:
                return [], due
            st["pending"].pop(0)
            return await self._speak(st, t, t, turn, barge_in=agent_speaking and turn.kind is None, resumed=True)
        if st["planned"] is not None:  # an answer is ready, waiting for its moment
            turn, t0 = st["planned"]
            if agent_speaking:  # the agent started again meanwhile: hold it, re-time it after the agent stops
                st["planned"] = (turn, None)
                return [], agent.end
            if t0 is None:
                t0 = t + self._delay(st, turn)
                st["planned"] = (turn, t0)
            if t < t0:
                return [], t0
            st["planned"] = None
            return await self._speak(st, t, t, turn)
        sound = self._last_end(st)
        if sound is not None and sound.end > t:  # still making a sound
            return [], sound.end

        if agent_speaking:
            if self.listener is None and self.interrupt is None:  # nobody decides anything: the user just listens
                return [], agent.end
            last = st["dp"].setdefault(agent.id, agent.t0)
            nxt = self._next_point(st, agent, last)
            if t >= nxt:
                at = "boundary" if any(last < b <= t for b in phrase_boundaries(agent)) else "fallback"
                st["dp"][agent.id] = t
                out = await self._decide(st, agent, t, at)
                if out is not None:
                    return out
                nxt = self._next_point(st, agent, t)
            return [], min(nxt, agent.end)

        agent_end = max((s.end for s in st["agent"].values() if s.t0 <= t), default=None)
        my_end = mine.end if mine is not None else None
        if agent_end is not None and (my_end is None or agent_end > my_end):
            if agent_end == t:  # a streamed agent turn ends exactly here only so far: its next chunk for this instant may
                # still come (the env steps the agent after running the instant). Look again 1 ms later instead of
                # deciding on the words heard so far (the reply would answer a fragment such as "O" of "Okay, ...").
                return [], t + 1
            if tt.response_delay is not None:  # decide what to say now, then take a content-aware moment to say it
                convo = conversation(list(st["mine"].values()), list(st["agent"].values()), t)
                turn = await self.source.next(st["n"], st["task"], convo)
                if turn is None:
                    st["done"] = True
                    return [], None
                t0 = agent_end + self._delay(st, turn)
                if t0 <= t:
                    return await self._speak(st, t, t, turn)
                st["planned"] = (turn, t0)
                return [], t0
            if t - agent_end >= tt.respond_after_ms:
                return await self._speak(st, t, t)
            return [], agent_end + tt.respond_after_ms
        if my_end is not None:
            if t - my_end >= tt.nudge_after_ms:
                return await self._speak(st, t, t)
            return [], my_end + tt.nudge_after_ms
        return [], None


def _accepts(fn, name: str) -> bool:
    """``fn`` takes keyword ``name`` (by name, or through ``**kwargs``)."""
    import inspect

    try:
        ps = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in ps or any(q.kind is inspect.Parameter.VAR_KEYWORD for q in ps.values())


def _takes(fn, name: str) -> bool:
    import inspect

    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _has_background(sc: dict) -> bool:
    """The scenario lays a noise bed under the call (the old numeric form, or a non-silent type)."""
    bg = sc.get("background")
    return bg is not None and bg is not False and bg != "silence" and not (isinstance(bg, dict) and bg.get("type") == "silence")


def _split_mid_thought(text: str, rng) -> tuple[str, str] | None:
    """Deprecated (``Behaviors.pause_p``): a random place where the user stops to think: (first half + "...", the rest).
    Between words for spaced text; CJK text only after an inner comma-like mark; None if there is none."""
    if not has_cjk(text):
        words = text.split()
        k = rng.randint(2, len(words) - 2)
        return " ".join(words[:k]) + "...", " ".join(words[k:])
    cuts = [m.end() for m in re.finditer(r"[，,、；;：:]\s*", text)
            if n_words(text[:m.start()]) >= 1 and n_words(text[m.end():]) >= 1.5]
    if not cuts:
        return None
    cut = rng.choice(cuts)
    return text[:cut].rstrip().rstrip("，,、；;：: ") + "...", text[cut:].strip()
