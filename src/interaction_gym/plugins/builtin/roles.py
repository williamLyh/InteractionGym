"""Roles: the participants of an episode.

- ``role/controlled``: the role an external policy plays (its ``AgentSpec``); the ``agent`` row attaches the policy.
- ``role/replay``: pre-timed turns (``ReplayUser``), from the config or ``task.scenario["turns"]``.
- ``role/user``: the reactive simulated user (``UserSim``): a script or an LLM decides what it says, turn-taking
  timing and behaviours decide when. Backends are named by row id (``llm``, ``tts``, ``clone``).

Every role names the stream it speaks on and the stream it listens to; ``user.speech`` / ``policy.speech`` are only
the core bundle's defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core import AgentSpec
from ...user import Behaviors, LLMListener, LLMSource, ReplayUser, ResponseDelay, ScriptSource, TurnTaking, UserSim, Voice
from ..registry import Controlled, PluginError, plugin


@dataclass
class ControlledConfig:
    name: str = "agent"  # what the trajectory calls this role
    chunk_ms: int = 200
    obs: tuple[str, ...] = ("user.speech",)
    out: str = "policy.speech"
    audio: str | None = None  # e.g. "user.audio": a microphone frame every step
    sr: int = 16000


@plugin("role", "controlled", provides="controlled", config=ControlledConfig)
def controlled(ctx, cfg: ControlledConfig) -> Controlled:
    """The role played by the external policy: what it observes, at which granularity, and where it speaks."""
    return Controlled(AgentSpec(cfg.chunk_ms, tuple(cfg.obs), cfg.out, cfg.audio, cfg.sr), cfg.name)


@dataclass
class VoiceConfig:
    tts: str | None = None  # row id of a backend/tts; None: text timed at words_per_sec
    clone: str | bool | None = None  # row id of a cloning TTS, False: no cloning, None: the tts clones by itself or not at all
    voice: str | None = None  # preset speaker when not cloning
    words_per_sec: float = 3.4


def _voice(ctx, v: VoiceConfig | dict | None) -> Voice:
    v = VoiceConfig(**v) if isinstance(v, dict) else (v or VoiceConfig())
    tts = ctx[v.tts] if v.tts else None
    kw = {}
    if v.clone is not None:
        kw["clone"] = ctx[v.clone] if isinstance(v.clone, str) else v.clone
    if v.voice:
        kw["voice"] = v.voice
    if tts is None:
        return Voice(words_per_sec=v.words_per_sec) if v.words_per_sec != 3.4 else Voice()
    return Voice(tts, **kw)


@dataclass
class ReplayConfig:
    stream: str = "user.speech"
    turns: list | None = None  # [{"t": ms, "text": ...}, ...]; None: task.scenario["turns"]
    voice: dict = field(default_factory=dict)  # VoiceConfig


VOICE_REFS = {"voice.tts": (), "voice.clone": ()}


@plugin("role", "replay", config=ReplayConfig, refs=VOICE_REFS)
def replay(ctx, cfg: ReplayConfig) -> ReplayUser:
    """Speaks pre-timed turns whatever the other side does (open loop)."""
    return ReplayUser(cfg.turns, voice=_voice(ctx, cfg.voice), stream=cfg.stream)


@dataclass
class UserConfig:
    stream: str = "user.speech"
    listens_to: str = "policy.speech"
    source: str = "script"  # "script" (turns below or task.scenario["turns"]) or the row id of a backend/chat
    turns: list | None = None
    system: str | None = None  # the user LLM's system prompt (default: user.USER_SYSTEM)
    listener: str | None = None  # row id of a backend/chat deciding LISTEN / BACKCHANNEL / INTERRUPT; None: only listens
    voice: dict = field(default_factory=dict)  # VoiceConfig
    timing: dict = field(default_factory=dict)  # TurnTaking fields; "response_delay": {ResponseDelay fields} | true
    behaviors: dict = field(default_factory=dict)  # Behaviors fields
    soundscape: str | None = None  # row id of a node/soundscape; None: the default bank
    aside_writer: str = "auto"


@plugin("role", "user", config=UserConfig, refs={"source": ("script",), "listener": (), "soundscape": (), **VOICE_REFS})
def user(ctx, cfg: UserConfig) -> UserSim:
    """The reactive simulated user: content from a script or an LLM, timing from ``timing``, sounds from ``behaviors``."""
    if cfg.source == "script":
        source = ScriptSource(cfg.turns)
    else:
        source = LLMSource(ctx[cfg.source], **({"system": cfg.system} if cfg.system else {}))
    timing = dict(cfg.timing)
    rd = timing.pop("response_delay", None)
    if rd:
        timing["response_delay"] = ResponseDelay(**(rd if isinstance(rd, dict) else {}))
    try:
        tt = TurnTaking(**timing)
        beh = Behaviors(**cfg.behaviors)
    except TypeError as e:
        raise PluginError(f"role/user: {e}") from None
    listener = LLMListener(ctx[cfg.listener]) if cfg.listener else None
    return UserSim(source, voice=_voice(ctx, cfg.voice), timing=tt, interrupt=listener, stream=cfg.stream,
                   agent_stream=cfg.listens_to, behaviors=beh, soundscape=ctx.get(cfg.soundscape), aside_writer=cfg.aside_writer)
