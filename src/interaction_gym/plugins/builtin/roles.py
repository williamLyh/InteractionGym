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
    name: str = "agent"  # what the trajectory calls this role
    chunk_ms: int = 200
    obs: tuple[str, ...] = ("user.speech",)
    out: str = "policy.speech"
    audio: str | None = None  # e.g. "user.audio": a microphone frame every step
    sr: int = 16000
    preset: str | None = None  # a duplex model's name (agents.vllm_omni.DUPLEX_MODELS): its chunk_ms, audio on


@plugin("role", "controlled", provides="controlled", config=ControlledConfig)
def controlled(ctx, cfg: ControlledConfig) -> Controlled:
    """The role played by the external policy: what it observes, at which granularity, and where it speaks."""
    if cfg.preset:
        from ...agents.vllm_omni import duplex_spec

        spec = duplex_spec(cfg.preset, cfg.sr, cfg.audio or "user.audio")
        spec = AgentSpec(spec.chunk_ms, tuple(cfg.obs), cfg.out, spec.audio, cfg.sr)
    else:
        spec = AgentSpec(cfg.chunk_ms, tuple(cfg.obs), cfg.out, cfg.audio, cfg.sr)
    return Controlled(spec, cfg.name)


@dataclass
class VoiceConfig:
    tts: str | None = None  # row id of a backend/tts; None: text timed at words_per_sec
    clone: str | bool | None = None  # row id of a cloning TTS, False: no cloning, None: the tts clones by itself or not at all
    voice: str | None = None  # preset speaker when not cloning
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
    soundscape: dict | None = None  # Soundscape(**dict); None: the default (a background from the persona's surroundings)
    benchmark: bool = False  # the evaluation setting: no background track, no sound bank, no random noises or asides
    aside_writer: str = "auto"


@plugin("role", "user", config=UserConfig, refs={"source": (), "listener": (), **VOICE_REFS})
def user(ctx, cfg: UserConfig) -> UserSim:
    """The reactive simulated user: content from its ``source``, timing from ``timing``, sounds from ``behaviors``."""
    from ...soundscape import Soundscape

    timing = dict(cfg.timing)
    rd = timing.pop("response_delay", None)
    if rd is not None and rd is not False:  # {} or true: the default ResponseDelay; a dict: its fields
        timing["response_delay"] = ResponseDelay(**(rd if isinstance(rd, dict) else {}))
    try:
        tt = TurnTaking(**timing)
        beh = Behaviors(**cfg.behaviors) if cfg.behaviors or not cfg.benchmark else None
        soundscape = Soundscape(**cfg.soundscape) if cfg.soundscape is not None else None
    except TypeError as e:
        raise PluginError(f"role/user: {e}") from None
    if cfg.benchmark:
        from ...benchmarks import benchmark_behaviors, benchmark_soundscape

        soundscape = soundscape or benchmark_soundscape()
        beh = beh or benchmark_behaviors()
    listener = LLMListener(ctx[cfg.listener]) if cfg.listener else None
    return UserSim(ctx[cfg.source], voice=_voice(ctx, cfg.voice), timing=tt, interrupt=listener, stream=cfg.stream,
                   agent_stream=cfg.listens_to, behaviors=beh if beh is not None else Behaviors(), soundscape=soundscape,
                   aside_writer=cfg.aside_writer)
