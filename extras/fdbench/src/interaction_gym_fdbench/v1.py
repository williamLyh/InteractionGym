# SPDX-License-Identifier: CC-BY-NC-4.0
# Ported from Full-Duplex-Bench v1 / v1.5 (github.com/DanielLin94144/Full-Duplex-Bench, v1_v1.5/evaluation/), (c) the
# Full-Duplex-Bench authors, licensed under CC BY-NC 4.0 (see LICENSE at the root of this package, extras/fdbench/LICENSE): NON-COMMERCIAL USE ONLY.
# Changes: the thresholds and per-sample rules of eval_pause_handling.py / eval_smooth_turn_taking.py /
# eval_user_interruption.py / eval_backchannel.py and get_timing.py re-implemented as pure functions over word
# chunks and speech spans (no numpy / scipy), and the user-interruption judge prompt + reply parsing and the
# behaviour judge's input layout of eval_user_interruption.py / eval_behavior.py.
"""Full-Duplex-Bench v1 / v1.5 material (CC BY-NC 4.0): official metric rules and judge prompts.

Used by ``interaction_gym.benchmarks.full_duplex_bench`` (Apache-2.0), which loads this optional package lazily and re-exports these names.
"""

from __future__ import annotations

import json
import re

# official thresholds (v1_v1.5/evaluation/eval_*.py, get_timing.py)
TURN_DURATION_THRESHOLD = 1  # s
TURN_NUM_WORDS_THRESHOLD = 3
BC_WINDOW = 0.2  # eval_backchannel.py window_size
BC_EPS = 1e-10
BC_TIME_THRESHOLD = 3
USER_MERGE_GAP = 0.6  # get_timing.py
MODEL_MERGE_GAP = 0.5


def merge(seg: list[tuple[float, float]], gap_thr: float) -> list[tuple[float, float]]:
    """get_timing.py ``_merge``."""
    if not seg:
        return []
    seg = sorted(seg)
    merged = [seg[0]]
    for s, e in seg[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap_thr:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def take_turn(chunks: list[dict]) -> int:
    """The take-turn rule shared by eval_pause_handling.py / eval_smooth_turn_taking.py / eval_user_interruption.py:
    no words → 0; words spanning < 1 s and ≤ 3 words → 0 (a backchannel); otherwise 1."""
    if not chunks:
        return 0
    duration = chunks[-1]["timestamp"][-1] - chunks[0]["timestamp"][0]
    if duration < TURN_DURATION_THRESHOLD and len(chunks) <= TURN_NUM_WORDS_THRESHOLD:
        return 0
    return 1


def backchannel_tor(segments: list[tuple[float, float]], chunks: list[dict]) -> tuple[int, list[list[float]]]:
    """eval_backchannel.py, step for step (including its quirk that a later short segment can reset TOR to 0 unless
    a > 3 s segment broke the loop): (TOR, predicted backchannel intervals)."""
    pred, tor = [], 0
    for start_time, end_time in segments:
        duration = end_time - start_time
        if duration > BC_TIME_THRESHOLD:
            tor = 1
            break
        curr = []
        for c in chunks:
            t_start, t_end = c["timestamp"]
            if (t_start >= start_time and t_end <= end_time) or (t_start <= end_time and t_end > end_time) or (
                t_start <= start_time and t_end > start_time):
                curr.append(c["text"])
        if len(curr) > 3:
            tor = 1
        elif duration < 1:
            tor = 0 if len(curr) <= 2 else 1
        else:
            tor = 1
        pred.append([start_time, end_time])
    return tor, pred


def backchannel_histogram(pred: list[list[float]], duration_s: float) -> list[float]:
    """eval_backchannel.py: the predicted backchannels as a normalized histogram over 0.2 s windows."""
    bins = [0.0] * (int(duration_s / BC_WINDOW) + 1)
    for s, e in pred:
        for i in range(int(s / BC_WINDOW), int(e / BC_WINDOW) + 1):
            if i < len(bins):
                bins[i] += 1
    bins = [x + BC_EPS for x in bins]
    tot = sum(bins)
    return [x / tot for x in bins]


def overlaps(user, model):
    """get_timing.py ``overlaps`` (verbatim logic)."""
    raw = []
    i = j = 0
    while i < len(user) and j < len(model):
        u_s, u_e = user[i]
        m_s, m_e = model[j]
        s, e = max(u_s, m_s), min(u_e, m_e)
        if e > s:
            raw.append((s, e))
        if u_e < m_e:
            i += 1
        else:
            j += 1
    best = {}
    for s, e in raw:
        key = int(round(e * 1000))
        if key not in best or (e - s) < (best[key][1] - best[key][0]):
            best[key] = (s, e)
    return [[round(s, 3), round(e, 3)] for s, e in sorted(best.values(), key=lambda x: x[1])]


def response_gaps(user, model):
    """get_timing.py ``response_gaps`` (verbatim logic)."""
    model_starts = [s for s, _ in model]
    tmp = {}
    for u_s, u_e in user:
        nxt = next((s for s in model_starts if s > u_e), None)
        if nxt is None:
            continue
        key = int(round(nxt * 1000))
        cand = [round(u_e, 3), round(nxt, 3)]
        if key not in tmp or cand[0] > tmp[key][0]:
            tmp[key] = cand
    return [iv for _, iv in sorted(tmp.items(), key=lambda kv: kv[1][1])]


INTERRUPTION_JUDGE = """
   The scenario is that the user and AI are talking in the spoken conversation.
   The user first speaks, then the AI responds. But when AI is speaking, the user interrupts the AI's turn.
   Your task is to rate the quality of AI's response after the user interrupt the turn.


   Below is the rating guideline (from 0 to 5, 0 is the worst and 5 is the best):
   - 0: The AI's response is totally unrelated to the user's interrupting turn.
   - 1: The AI's response is not related to the user's interrupting turn.
   - 2: The AI's response is slightly related to the user's interrupting turn.
   - 3: The AI's response is related to the user's interrupting turn.
   - 4: The AI's response is highly related to the user's interrupting turn.
   - 5: The AI's response is perfectly related to the user's interrupting turn.


   Firstly, briefly analyze the user's interrupting turn and the AI's response
   Then, you must return the overall output as the following format:
   Analysis: [Your analysis].
   I would rate the AI's response as [Rating].
   """  # eval_user_interruption.py system_msg (official judge: gpt-4-turbo, seed 0)


def interruption_user_message(context: str, interrupt: str, response: str) -> str:
    """eval_user_interruption.py: the judge's user message."""
    return f"""
                - Contextual user turn: {context}
                - User interrupting turn: {interrupt}
                - AI's response: {response}
                """


def parse_interruption_rating(reply: str) -> int | None:
    """eval_user_interruption.py: the last "I would rate the AI's response as N"."""
    found = re.findall(r"Analysis:\s*(.*?)\nI would rate the AI's response as (\d+)", reply + "\n", re.DOTALL)
    return int(found[-1][1]) if found else None


def _compact(chunks: list[dict]) -> str:
    return json.dumps(chunks, separators=(",", ":"), ensure_ascii=False)


def behaviour_input(input_clean: list[dict], input_noisy: list[dict], output_clean: list[dict], output_noisy: list[dict]) -> str:
    """eval_behavior.py: the judge's user message (word chunks of both inputs and outputs). The judge's
    instruction itself is read from a checkout of the official repo (``behavior.txt``), not shipped here."""
    return f"""
        {{
            "input_clean": {_compact(input_clean)},
            "input_noisy": {_compact(input_noisy)},
            "output_clean": {_compact(output_clean)},
            "output_noisy": {_compact(output_noisy)}
        }}
        """


def parse_behaviour(reply: str) -> str | None:
    """eval_behavior.py: the first ``behaviour`` tag in the judge's reply."""
    dec, pos = json.JSONDecoder(), reply.find("{")
    while pos != -1:
        try:
            obj, end = dec.raw_decode(reply, pos)
            if "behaviour" in obj and obj["behaviour"]:
                return obj["behaviour"][0]
            pos = reply.find("{", end)
        except json.JSONDecodeError:
            pos = reply.find("{", pos + 1)
    return None
