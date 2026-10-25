"""Roles: the participants of an episode.

- ``role/controlled``: the role an external policy plays (its ``AgentSpec``); the ``agent`` row attaches the policy.
- ``role/replay``: pre-timed turns (``ReplayUser``), from the config or ``task.scenario["turns"]``.
- ``role/user``: the reactive simulated user (``UserSim``): a ``source`` row decides what it says, turn-taking
  timing and behaviours decide when. Backends are named by row id (``voice.tts``, ``voice.clone``, ``listener``).

Every role names the stream it speaks on and the stream it listens to; ``user.speech`` / ``policy.speech`` are only
the core bundle's defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core import AgentSpec
from ...user import Behaviors, LLMListener, ReplayUser, ResponseDelay, TurnTaking, UserSim, Voice
from ..registry import Controlled, PluginError, plugin


@dataclass
class ControlledConfig:
    name: str = "agent"  # what the trajectory calls this role (turns[].role, meta.controlled)
    chunk_ms: int = 200
    obs: tuple[str, ...] = ("user.speech",)
    out: str | None = None  # the stream it speaks on; default: policy.speech for "agent", <name>.speech otherwise
    audio: str | None = None  # e.g. "user.audio": a microphone frame every step
    sr: int = 16000
    preset: str | None = None  # a duplex model's name (agents.vllm_omni.DUPLEX_MODELS): its chunk_ms, audio on
    # how this role's reactions to each other role's turns are graded (eval.Standard fields: mid_ms, width_ms,
    # early_floor, respond_within_ms), by speaker role with "*" the default; None: the assistant standard for everyone
    standards: dict | None = None


@plugin("role", "controlled", provides="controlled", config=ControlledConfig)
def controlled(ctx, cfg: ControlledConfig) -> Controlled:
    """The role played by the external policy: what it observes, at which granularity, and where it speaks. The
    trajectory names roles after their streams (``policy.*`` is ``agent``), so ``out`` must agree with ``name``."""
    from ...traj import role_of

    out = cfg.out or ("policy.speech" if cfg.name == "agent" else f"{cfg.name}.speech")
    if role_of(out) != cfg.name:
        raise PluginError(f"role/controlled: name {cfg.name!r} but out {out!r} would be recorded as role {role_of(out)!r}; "
                          f"use out: {cfg.name}.speech (or policy.speech for the role named agent)")
    if cfg.preset:
        from ...agents.vllm_omni import duplex_spec

        spec = duplex_spec(cfg.preset, cfg.sr, cfg.audio or "user.audio")
        spec = AgentSpec(spec.chunk_ms, tuple(cfg.obs), out, spec.audio, cfg.sr)
    else:
        spec = AgentSpec(cfg.chunk_ms, tuple(cfg.obs), out, cfg.audio, cfg.sr)
    extra = {}
    if cfg.standards:
        from ...eval import Standard

        try:
            extra["standards"] = {k: Standard(**(v or {})) for k, v in cfg.standards.items()}
        except TypeError as e:
            raise PluginError(f"role/controlled: standards: {e}; the fields are mid_ms, width_ms, early_floor, respond_within_ms") from None
    return Controlled(spec, cfg.name, extra)


@dataclass
class VoiceConfig:
    tts: str | None = None  # row id of a backend/tts; None: text timed at words_per_sec
    clone: str | bool | None = None  # row id of a cloning TTS, False: no cloning, None: the tts clones by itself or not at all
    voice: str | None = None  # preset speaker when not cloning
    voices: str | None = None  # a speaker catalogue the persona picks from: "qwen3" (QWEN3_TTS_VOICES) | "openai" (OPENAI_TTS_VOICES)
    words_per_sec: float = 3.4
    style: str | None = "neutral"
    seed: bool = True


def _voice(ctx, v: VoiceConfig | dict | None) -> Voice:
    v = VoiceConfig(**v) if isinstance(v, dict) else (v or VoiceConfig())
    tts = ctx[v.tts] if v.tts else None
    kw = {}
    if v.clone is not None:
        kw["clone"] = ctx[v.clone] if isinstance(v.clone, str) else v.clone
    if v.voice:
        kw["voice"] = v.voice
    if v.voices:
        from ...user import OPENAI_TTS_VOICES, QWEN3_TTS_VOICES

        catalogues = {"qwen3": QWEN3_TTS_VOICES, "openai": OPENAI_TTS_VOICES}
        if v.voices not in catalogues:
            raise PluginError(f"voice.voices is one of {', '.join(catalogues)}, got {v.voices!r}")
        kw["voices"] = catalogues[v.voices]
    if v.style != "neutral":
        kw["style"] = v.style
    if not v.seed:
        kw["seed"] = False
    if tts is None:
        return Voice(words_per_sec=v.words_per_sec) if v.words_per_sec != 3.4 else Voice()
    return Voice(tts, **kw)


VOICE_REFS = {"voice.tts": (), "voice.clone": ()}


@dataclass
class ReplayConfig:
    stream: str = "user.speech"
    turns: list | None = None  # [{"t": ms, "text": ...}, ...]; None: task.scenario["turns"]
    voice: dict = field(default_factory=dict)  # VoiceConfig


@plugin("role", "replay", config=ReplayConfig, refs=VOICE_REFS)
def replay(ctx, cfg: ReplayConfig) -> ReplayUser:
    """Speaks pre-timed turns whatever the other side does (open loop)."""
    return ReplayUser(cfg.turns, voice=_voice(ctx, cfg.voice), stream=cfg.stream)


@dataclass
class UserConfig:
    source: str  # row id of a source/* row
    stream: str = "user.speech"
    listens_to: str = "policy.speech"
    listener: str | None = None  # row id of a backend/chat deciding LISTEN / BACKCHANNEL / INTERRUPT; None: only listens
    voice: dict = field(default_factory=dict)  # VoiceConfig
    timing: dict = field(default_factory=dict)  # TurnTaking fields; "response_delay": {ResponseDelay fields} | true
    behaviors: dict = field(default_factory=dict)  # Behaviors fields
    soundscape: dict | None = None  # Soundscape(**dict); background may be "scenario": a track only when the task defines one
    benchmark: bool = False  # the evaluation setting: no background track, no sound bank, no random noises or asides
    aside_writer: str = "auto"
    behaviours: list | None = None  # row ids of behaviour/* rows, in order; None: the default library (noise, asides)


@plugin("role", "user", config=UserConfig, refs={"source": (), "listener": (), "behaviours[]": (), **VOICE_REFS})
def user(ctx, cfg: UserConfig):
    """The reactive simulated user: content from its ``source``, timing from ``timing``, sounds from ``behaviors``.
    Built per episode (a factory), so a source that keeps per-episode state (an examiner's log) starts fresh."""
    from ...soundscape import Soundscape
    from ..compose import is_factory

    timing = dict(cfg.timing)
    rd = timing.pop("response_delay", None)
    if rd is not None and rd is not False:  # {} or true: the default ResponseDelay; a dict: its fields
        timing["response_delay"] = ResponseDelay(**(rd if isinstance(rd, dict) else {}))
    per_task_bg = cfg.soundscape is not None and cfg.soundscape.get("background") == "scenario"
    try:
        tt = TurnTaking(**timing)
        beh = Behaviors(**cfg.behaviors) if cfg.behaviors or not cfg.benchmark else None
        soundscape = Soundscape(**{**cfg.soundscape, **({"background": True} if per_task_bg else {})}) if cfg.soundscape is not None else None
    except TypeError as e:
        raise PluginError(f"role/user: {e}") from None
    if cfg.benchmark:
        from ...benchmarks import benchmark_behaviors, benchmark_soundscape

        soundscape = soundscape or benchmark_soundscape()
        beh = beh or benchmark_behaviors()
    listener = LLMListener(ctx[cfg.listener]) if cfg.listener else None
    voice, source = _voice(ctx, cfg.voice), ctx[cfg.source]
    behaviours = None if cfg.behaviours is None else [ctx[rid] for rid in cfg.behaviours]

    def make(ep) -> UserSim:
        scape = soundscape
        if per_task_bg:  # a background track only for tasks that define one
            scape = Soundscape(**{**cfg.soundscape, "background": ep is not None and "background" in ep.task.scenario})
        return UserSim(source(ep) if is_factory(source) else source, voice=voice, timing=tt, interrupt=listener, stream=cfg.stream,
                       agent_stream=cfg.listens_to, behaviors=beh if beh is not None else Behaviors(), soundscape=scape,
                       aside_writer=cfg.aside_writer, behaviours=behaviours)
    return make
