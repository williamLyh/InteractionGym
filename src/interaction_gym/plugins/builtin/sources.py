"""Sources: what a simulated role says. A ``role/user`` row names one by id; the timing of *when* stays the role's.

- ``source/script``: fixed lines in order (from the config or ``task.scenario["turns"]``).
- ``source/llm``: a text LLM writes each turn from the persona, goal and conversation so far (``LLMSource``).
- ``source/audiomc_script`` / ``source/audiomc_replan``: Audio MultiChallenge's open-loop recordings and its
  closed-loop variant (the scripted turn rewritten to fit what the agent actually said).
"""

from __future__ import annotations

from dataclasses import dataclass

from ...user import LLMSource, ScriptSource
from ..registry import plugin


@dataclass
class ScriptConfig:
    turns: list | None = None  # None: task.scenario["turns"]


@plugin("source", "script", config=ScriptConfig)
def script(ctx, cfg: ScriptConfig) -> ScriptSource:
    """Fixed utterances in order; timing stays reactive."""
    return ScriptSource(cfg.turns)


@dataclass
class LLMSourceConfig:
    llm: str  # row id of a backend/chat
    system: str | None = None  # the user's system prompt (default: user.USER_SYSTEM)
    first: str | None = None  # a given first line (else task.scenario["first_turn"], else generated)


@plugin("source", "llm", config=LLMSourceConfig, refs=("llm",))
def llm(ctx, cfg: LLMSourceConfig) -> LLMSource:
    """Each turn written by a text LLM from the persona, goal and the conversation heard so far."""
    kw = {k: v for k, v in (("system", cfg.system), ("first", cfg.first)) if v is not None}
    return LLMSource(ctx[cfg.llm], **kw)


@plugin("source", "audiomc_script")
def audiomc_script(ctx, kw: dict):
    """Audio MultiChallenge, open loop: the recorded user turns in order, content fixed."""
    from ...benchmarks import audiomc

    return audiomc.ScriptSource()


@dataclass
class ReplanConfig:
    llm: str  # row id of a backend/chat (the dataset's prompt rewrites each scripted turn to the agent's replies)


@plugin("source", "audiomc_replan", config=ReplanConfig, refs=("llm",))
def audiomc_replan(ctx, cfg: ReplanConfig):
    """Audio MultiChallenge, closed loop: the first turn is the recording, every later turn is the script replanned."""
    from ...benchmarks import audiomc

    return audiomc.ReplanSource(ctx[cfg.llm])
