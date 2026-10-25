"""Behaviours: what a simulated role does besides its turns, as separate objects the role runs in a fixed order
(docs/ARCHITECTURE.md §3). Each owns its random process on the role's seeded streams and labels what it produces
(``kind`` / ``expects`` / ``label``), so the scorer judges the other side against it without knowing the behaviour.

A behaviour has three entry points, all optional:

- ``init(user, st)``: per-episode state (draw the first due time, build an event process) on ``st``;
- ``sounds(user, st, t)``: sounds that happen whatever the role is doing (a cough, a door): frames to emit now;
- ``act(user, st, t)``: something the role itself does when free (an aside, being called away): ``(frames, wake)``
  to take over this step, or None;

plus ``due(st, t)`` (its next wake, or None) and ``report(st)`` (counts for ``meta.user``). The role core keeps the
decision points while the other side talks (barge-in / backchannel) and the turn itself; these move here next.

The default library (``default_behaviours()``) is the simulated user's historical habits: ``NoiseEvents`` and
``Asides`` (being addressed by someone nearby, being called away), both driven by the resolved ``Behaviors`` rates
and the persona's surroundings. A role with ``behaviours=[]`` makes no sound but its turns.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import replace

from .core import Frame


class Behaviour:
    name = "behaviour"

    def init(self, user, st: dict) -> None:
        pass

    def due(self, st: dict, t: int) -> int | None:
        return None

    async def sounds(self, user, st: dict, t: int) -> list[Frame]:
        return []

    async def act(self, user, st: dict, t: int) -> tuple[list[Frame], int | None] | None:
        return None

    def report(self, st: dict) -> dict:
        return {}


class NoiseEvents(Behaviour):
    """Noise events on the call clock — a cough, a door, a phone, third-party speech — a compound Poisson process
    (``soundscape.EventProcess``) at the persona's surroundings' rates (``Behaviors.noise_per_min`` /
    ``noise_rates``, ``third_party_per_min``); each is a turn with ``kind="noise"``, ``expects="ignore"`` and the
    event's ``label``. They overlap anyone's speech, the user's own included."""

    name = "noise"

    def init(self, user, st: dict) -> None:
        from .soundscape import EventProcess

        b = st["behaviors"]
        st["noise"] = EventProcess(self.rates(b, st["levels"]), st["erng_base"])
        st["noise_log"] = []
        st["events"] |= {"noise": 0, "noise_occurrences": 0, "noise_over_user": 0}
        if b.third_party_per_min:
            from .soundscape import _bank

            bank = _bank(user.soundscape.bank, fetch=False)
            if bank is None or not bank.files("events", "speech"):
                raise ValueError("Behaviors.third_party_per_min needs a sound bank with events/speech/*.wav recordings")

    @staticmethod
    def rates(b, levels: dict) -> dict[str, float]:
        from .soundscape import event_rates

        rates = event_rates(levels["surroundings"], b.noise_per_min, b.noise_rates)
        if b.third_party_per_min:
            rates["speech"] = rates.get("speech", 0.0) + b.third_party_per_min
        return rates

    def due(self, st: dict, t: int) -> int | None:
        ev = st["noise"].peek()
        return None if ev is None else max(ev.t, t + 1)

    async def sounds(self, user, st: dict, t: int) -> list[Frame]:
        """The noise events due by ``t``, whatever the user or the agent is doing."""
        from .soundscape import RECORDED_ONLY, event_audio
        from .user import UserTurn, _lang

        out, proc = [], st["noise"]
        while (ev := proc.peek()) is not None and ev.t <= t:
            proc.pop()
            sr = getattr(user.voice.speech, "sr", 16000)
            # third-party speech in the caller's language where the bank has it (events/speech_<lang>/)
            variant = _lang(st["task"]) if ev.label in RECORDED_ONLY else None
            audio, source, n = event_audio(ev, sr, st["erng"]["noise"], user.soundscape.bank, variant)
            if user.voice.speech is None:  # a text-only timeline: just its length
                turn = UserTurn("", kind="noise", expects="ignore", label=ev.label, dur=audio.dur_ms)
            else:
                turn = UserTurn("", audio=audio, kind="noise", expects="ignore", label=ev.label)
            mine = user._mine(st)
            over = mine is not None and mine.t0 <= t < mine.end
            ev_log = st["events"]
            ev_log["noise"] += 1
            ev_log["noise_occurrences"] += n
            ev_log["noise_over_user"] += over
            st["noise_log"].append({"t": t, "label": ev.label, "n": n, "level_db": ev.level_db, "dur_ms": audio.dur_ms,
                                     "source": ev.source, "clip": source, "over_user_speech": over})
            out.append(await user._emit(st, t, turn))
        return out

    def report(self, st: dict) -> dict:
        return {"noise_events": [dict(e) for e in st["noise_log"]]} if st.get("noise_log") else {}


class Asides(Behaviour):
    """Being addressed by someone nearby: a remark to them (``kind="aside"``, ``expects="ignore"``) or, with
    probability ``Behaviors.away_p`` while the agent talks, being called away — "hold on" (``expects="wait"``), the
    remark, an absence (``kind="away"``) and "I'm back" (all labelled ``called_away``). A Poisson process at
    ``Behaviors.aside_per_min`` on the call clock, deferred while the user is busy and capped per agent turn; the lines
    are written by the role's ``aside_writer`` or taken from the fixed ones."""

    name = "aside"

    def init(self, user, st: dict) -> None:
        st["events"] |= {"aside": 0, "called_away": 0, "capped": 0, "lines_llm": 0, "lines_fixed": 0}
        st["next_event"]["aside"] = self._draw(st, 0)

    @staticmethod
    def _draw(st: dict, t: int) -> int | None:
        rate = st["behaviors"].rates()["aside"]
        return None if rate <= 0 else t + max(1, round(st["erng"]["aside"].expovariate(rate / 60000)))

    def due(self, st: dict, t: int) -> int | None:
        x = st["next_event"].get("aside")
        return x if x is not None and x > t else None

    async def act(self, user, st: dict, t: int) -> tuple[list[Frame], int | None] | None:
        """Being addressed by someone nearby, when it is due and socially plausible (else it waits)."""
        from .user import BACK, HOLD_ON, SITUATIONS, UserTurn, _lang, conversation

        x = st["next_event"].get("aside")
        if x is None or x > t or user._busy(st, t):  # deferred until the user is free
            return None
        b, rng = st["behaviors"], st["erng"]["aside"]
        if st["last_sound"] is not None and t - st["last_sound"] < b.min_gap_ms:
            st["next_event"]["aside"] = st["last_sound"] + b.min_gap_ms
            return None
        st["next_event"]["aside"] = self._draw(st, t)
        agent = user._latest(st["agent"], t)
        agent = agent if agent is not None and agent.active(t) else None
        if agent is not None and b.max_asides_per_turn is not None and st["per_turn"].get(agent.id, {}).get("other", 0) >= b.max_asides_per_turn:
            st["events"]["capped"] += 1
            return None
        user._count(st, agent, "other")
        st["last_sound"] = t
        task, lang = st["task"], _lang(st["task"])
        away = agent is not None and rng.random() < b.away_p  # called away: tell the agent, go, come back
        place = st["levels"]["surroundings"]
        situation = rng.choice(SITUATIONS.get(place, SITUATIONS["quiet"]))
        lines = {"ASIDE": rng.choice(b.asides), "HOLD": rng.choice(HOLD_ON.get(lang, HOLD_ON["en"])),
                 "BACK": rng.choice(BACK.get(lang, BACK["en"]))}  # the fixed fallback (always drawn: the same seed stream)
        if user.aside_writer is not None and not user._fixed_asides(task):
            convo = conversation(list(st["mine"].values()), list(st["agent"].values()), t)
            try:
                got = await user.aside_writer.write(task, convo, place, situation, away)
            except Exception as e:  # noqa: BLE001  (a failed call must not end the episode: the fixed lines stand in)
                warnings.warn(f"AsideWriter failed, using fixed lines: {e!r}")
                got = {}
            need = ("HOLD", "ASIDE", "BACK") if away else ("ASIDE",)
            st["events"]["lines_llm"] += sum(k in got for k in need)
            st["events"]["lines_fixed"] += sum(k not in got for k in need)
            lines |= got
        else:
            st["events"]["lines_fixed"] += 3 if away else 1
        aside = user._pack_clip(st, "aside") or UserTurn(lines["ASIDE"], kind="aside", expects="ignore")
        if away:
            st["events"]["called_away"] += 1
            hold = UserTurn(lines["HOLD"], expects="wait", label="called_away")
            gone = UserTurn("", kind="away", expects="wait", dur=rng.randint(*b.away_ms), label="called_away")
            back = UserTurn(lines["BACK"], label="called_away")
            st["pending"] = [(replace(aside, label="called_away"), rng.randint(300, 800)), (gone, 0), (back, 0)] + st["pending"]
            return await user._speak(st, t + st["timing"].reaction_ms, t, hold, barge_in=True, resumed=True)
        st["events"]["aside"] += 1
        frame = await user._emit(st, t, aside)
        return [frame], frame.data.end


def default_behaviours() -> list[Behaviour]:
    """The simulated user's historical habits, in the order the role runs them."""
    return [NoiseEvents(), Asides()]


def seeded_streams(base: str, names=("noise", "aside", "barge")) -> dict[str, random.Random]:
    """One seeded stream per behaviour, independent of the others (``f"{base}:{name}"``)."""
    return {k: random.Random(f"{base}:{k}") for k in names}
