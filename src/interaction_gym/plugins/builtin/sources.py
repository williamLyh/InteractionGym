"""Sources: what a simulated role says. A ``role/user`` row names one by id; the timing of *when* stays the role's.

- ``source/script``: fixed lines in order (from the config or ``task.scenario["turns"]``).
- ``source/llm``: a text LLM writes each turn from the persona, goal and conversation so far (``LLMSource``).
- ``source/audiomc_script`` / ``source/audiomc_replan``: Audio MultiChallenge's open-loop recordings and its
  closed-loop variant (the scripted turn rewritten to fit what the agent actually said).
"""

from __future__ import annotations

from dataclasses import dataclass

from ...user import LLMSource, ScriptSource
from ..registry import PluginError, plugin


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
class ExaminerConfig:
    llm: str  # row id of a backend/chat (the official examiner prompts on it)
    max_turns: int = 12
    closing: bool = False  # stage-progress-aware closing rules (fdb2.StageClosing) on closing_llm
    closing_llm: str | None = None
    max_per_stage: int = 3
    max_stall: int = 5


@plugin("source", "fdb2_examiner", config=ExaminerConfig, refs=("llm", "closing_llm"))
def fdb2_examiner(ctx, cfg: ExaminerConfig):
    """Full-Duplex-Bench v2's live examiner (closed loop): the official prompts on a text LLM; the first line can be
    the open-loop script's (``task.scenario["first_turn"]``). One examiner per episode: its line log goes to
    ``meta.examiner_log``."""
    from ...benchmarks import fdb2

    if cfg.closing and not cfg.closing_llm:
        raise PluginError("source/fdb2_examiner: closing needs closing_llm (a backend/chat row)")

    class Examiner(fdb2.ExaminerSource):
        def episode_meta(self) -> dict:
            return {**({"examiner_log": self.log} if self.log else {}), "closing": self.closing is not None}

    llm, closing_llm = ctx[cfg.llm], ctx.get(cfg.closing_llm)

    def make(ep):
        closing = fdb2.StageClosing(closing_llm, cfg.max_per_stage, cfg.max_stall) if cfg.closing else None
        return Examiner(llm, cfg.max_turns, closing=closing)
    return make


@dataclass
class Fdb3UserConfig:
    llm: str  # row id of a backend/chat
    max_turns: int = 8
    system: str | None = None  # default: fdb3.USER_SYSTEM


@plugin("source", "fdb3_user", config=Fdb3UserConfig, refs=("llm",))
def fdb3_user(ctx, cfg: Fdb3UserConfig):
    """Full-Duplex-Bench v3's closed-loop caller: the recording first, then an LLM on the scenario card."""
    from ...benchmarks import fdb3

    return fdb3.UserSource(ctx[cfg.llm], cfg.max_turns, system=cfg.system)


@dataclass
class ReplanConfig:
    llm: str  # row id of a backend/chat (the dataset's prompt rewrites each scripted turn to the agent's replies)


@plugin("source", "audiomc_replan", config=ReplanConfig, refs=("llm",))
def audiomc_replan(ctx, cfg: ReplanConfig):
    """Audio MultiChallenge, closed loop: the first turn is the recording, every later turn is the script replanned."""
    from ...benchmarks import audiomc

    return audiomc.ReplanSource(ctx[cfg.llm])
