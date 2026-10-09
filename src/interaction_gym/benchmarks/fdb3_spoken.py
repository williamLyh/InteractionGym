"""Full-Duplex-Bench v3 for agents that cannot call tools (native duplex models such as MiniCPM-o 4.5): a
spoken-fulfilment outcome in place of Pass@1.

Did the agent's speech capture the request's parameters with their FINAL values? The parameters are the literal
arguments of the task's ``expected_tool_calls`` (``$RESULT_k`` references are excluded: they come from tool results
the agent never had); for a self-correction (``state_rollback_details``) the target is the corrected value and the
original one is *superseded*. One slot per distinct parameter value (the same value needed by two calls is one
slot). The rules and the LLM fallback follow the authors' slot-based outcome reward for multi-step scenarios (from
their RL experiments), adapted to this benchmark's parameters:

1. **Rule check** on the agent's turns in order (``mentions``: normalised text, number words → digits, spelled-out
   codes collapsed, per-type rules — see ``SLOT_TYPES``): the agent's last turn that states the target or a superseded
   value decides — target → ``correct``, superseded only → ``wrong`` (both in one turn, "150, not 100" → correct).
2. **LLM fallback** (any ``TextGen``, T = 0) only for slots no rule settled (paraphrases, "a tablet" for quantity 1,
   misheard values): ``confirmed`` → correct, ``wrong`` → wrong, ``not_confirmed`` → missing.

``outcome`` = share of slots correct; ``fulfilled`` = all slots correct. Which agent turns count is the caller's
choice (``before_ms``): the first response (open loop, or a closed-loop run up to the user's second turn) or the
whole conversation (closed-loop final). This module is original code (Apache-2.0); the benchmark data it scores is
CC BY-NC 4.0.
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------- text normalisation

UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
TEENS = {w: 10 + i for i, w in enumerate("ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
       "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
       "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30}


def _num_words(tokens: list[str]) -> list[str]:
    """Number words → digits: compounds become one number ("twenty five" → 25, "fifteen hundred" → 1500), a digit
    word after a digit word starts a new number ("one two three" → 1 2 3); ordinals become their number."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "double" and i + 1 < len(tokens) and (tokens[i + 1] in UNITS or tokens[i + 1].isdigit()):
            d = str(UNITS.get(tokens[i + 1], tokens[i + 1]))
            if len(d) == 1:
                out += [d, d]
                i += 2
                continue
        if t in ORD:
            out.append(str(ORD[t]))
            i += 1
            continue
        if not (t in UNITS or t in TEENS or t in TENS):
            out.append(t)
            i += 1
            continue
        total, cur, j, last = 0, 0, i, None
        while j < len(tokens):
            w = tokens[j]
            if w in TENS and last in (None, "mult"):
                cur += TENS[w]
                last = "tens"
            elif w in UNITS and last in (None, "tens", "mult") and not (w == "zero" and last is not None):
                cur += UNITS[w]
                last = "unit"
            elif w in ORD and last == "tens" and ORD[w] < 10:
                cur += ORD[w]
                j += 1
                break
            elif w in TEENS and last in (None, "mult"):
                cur += TEENS[w]
                last = "teen"
            elif w == "hundred" and last in ("unit", "teen", "tens"):
                cur *= 100
                last = "mult"
            elif w == "thousand" and last in ("unit", "teen", "tens", "mult"):
                total += cur * 1000
                cur = 0
                last = "mult"
            elif w == "and" and last == "mult" and j + 1 < len(tokens) and (tokens[j + 1] in UNITS or tokens[j + 1] in TEENS or tokens[j + 1] in TENS):
                pass
            else:
                break
            j += 1
        out.append(str(total + cur))
        i = j
    return out


def normalize(text: str) -> str:
    """Lowercase words (keeping ':' / '.' inside numbers), number words → digits, "14th" → "14", "1,800" → "1800",
    "$1.5k" untouched."""
    s = text.lower().replace("’", "'")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", s)
    toks = re.findall(r"\d+(?:[:.]\d+)?|[a-z]+(?:'[a-z]+)?", s)
    return " ".join(_num_words(toks))


def _collapse_spelled(norm: str) -> str:
    """Spelled-out codes: runs of ≥ 2 single letters / digits joined ('b o b 1 2' → 'bob12')."""
    return re.sub(r"(?<![\w'])(?:[a-z0-9] )+[a-z0-9]\b", lambda m: m.group(0).replace(" ", ""), norm)


def _ints(norm: str) -> list[tuple[int, int]]:
    return [(m.start(), int(m.group(0))) for m in re.finditer(r"(?<![\d.:])\d+(?![\d:]|\.\d)", norm)]


def _word(norm: str, w: str) -> bool:
    return bool(w) and re.search(rf"(?<![\w']){re.escape(w)}(?![\w'])", norm) is not None


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s)


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


# ---------------------------------------------------------------- the benchmark's parameters as slots

STOP = {"the", "a", "an", "my", "on", "to", "of", "at", "in", "for", "and"}
CURRENCY = {"USD": ["usd", "us dollars", "u s dollars", "american dollars", "us dollar", "american dollar"],
            "EUR": ["eur", "euro", "euros"], "GBP": ["gbp", "pound", "pounds", "sterling", "british pounds"],
            "JPY": ["jpy", "yen", "japanese yen"], "CAD": ["cad", "canadian", "canadian dollars", "canadian dollar"]}
MODE = {"driving": ["drive", "driving", "car", "drives"], "transit": ["transit", "public transport", "public transportation", "bus", "train", "subway", "metro"],
        "walking": ["walk", "walking", "on foot", "walks"], "biking": ["bike", "biking", "bicycle", "cycling", "cycle", "bikes"]}
PETS = ["pet", "pets", "pet friendly", "pets allowed", "dog", "dogs", "cat", "cats"]
DOC = {"driver_license": ["driver license", "driver's license", "drivers license", "driving license", "driving licence", "driver licence", "driver's licence"],
       "passport": ["passport"], "visa": ["visa"]}
FILTER = {"max_price": ["max price", "maximum price", "price", "budget", "rent", "under", "maximum", "max"],
          "min_bedrooms": ["bedroom", "bedrooms", "minimum bedrooms"], "neighborhood": ["neighborhood", "neighbourhood", "area"],
          "pets_allowed": PETS}
CITY = {"Las Vegas": ["las vegas", "vegas"], "San Francisco": ["san francisco", "sf"]}
BILL = {"credit_card": ["credit card"], "mortgage": ["mortgage"]}
ACCOUNT = {"checking": ["checking", "chequing"], "savings": ["savings", "saving"]}
COUNT_UNITS = ["of", "units", "unit", "items", "item", "pairs", "pair", "quantity", "times", "copies", "x"]

# (function, argument) -> slot type; arguments not listed (and $RESULT references) are not slots
SLOT_TYPES = {
    ("search_flights", "destination"): "city", ("search_flights", "date"): "monthday", ("book_flight", "passenger_name"): "name",
    ("update_identity_doc", "doc_type"): "enum", ("update_identity_doc", "doc_number"): "code", ("get_card_benefits", "card_type"): "enum",
    ("get_exchange_rate", "amount"): "amount", ("get_exchange_rate", "from_currency"): "enum", ("get_exchange_rate", "to_currency"): "enum",
    ("modify_autopay", "bill_type"): "enum", ("modify_autopay", "source_account"): "enum", ("search_apartments", "city"): "city",
    ("search_apartments", "bedrooms"): "count", ("search_apartments", "max_price"): "amount", ("search_apartments", "pets_allowed"): "enum",
    ("calculate_commute", "origin_address"): "phrase", ("calculate_commute", "destination_address"): "phrase",
    ("calculate_commute", "mode"): "enum", ("update_search_filter", "filter_name"): "enum", ("update_search_filter", "value"): "value",
    ("track_order", "order_id"): "code", ("search_products", "query"): "phrase", ("search_products", "max_price"): "amount",
    ("search_products", "category"): "enum", ("add_to_cart", "product_id"): "code", ("add_to_cart", "quantity"): "count",
}
_SYN = {"from_currency": CURRENCY, "to_currency": CURRENCY, "mode": MODE, "doc_type": DOC, "filter_name": FILTER, "bill_type": BILL,
        "source_account": ACCOUNT}


def _is_ref(v) -> bool:
    return isinstance(v, str) and v.startswith("$")


def _label(fn: str, arg: str) -> str:
    return f"{fn.replace('_', ' ')}: {arg.replace('_', ' ')}"


# Stated by the caller although the rules cannot see it in the script ("just put 1 in there", "add 2... to the cart").
STATED = {("ecommerce_11", "add_to_cart.quantity"), ("ecommerce_19", "add_to_cart.quantity"), ("ecommerce_20", "add_to_cart.quantity")}
# Checked by hand, all 100 scripts: the only other annotated values the rules do not find in the script are not said
# by the caller — ecommerce_12 query "gift", ecommerce_21 quantity 1 ("add it"), housing_11 city "Austin",
# housing_20 mode "driving" (the API default) — or contradict it: travel_02 doc_number "P9-9-9-90011" (spoken
# "P-8-8-9-9-0-0-1-1"). Those are not slots: an agent cannot be asked to confirm what it was never told.


def task_slots(expected_calls: list[dict], rollback: dict | None = None, script: str | None = None,
               scenario_id: str | None = None) -> list[dict]:
    """The slots of one task: ``{"key", "label", "type", "value", "superseded", "synonyms"?, "units"?}``. ``value`` is
    the final (corrected) value, ``superseded`` the original values of a self-correction. With ``script`` (the
    caller's words), only the values the caller states (rule match, or ``STATED``)."""
    slots = _all_slots(expected_calls, rollback)
    if script is None:
        return slots
    return [s for s in slots if mentions(s, s["value"], script) or (scenario_id, s["key"]) in STATED]


def _all_slots(expected_calls: list[dict], rollback: dict | None) -> list[dict]:
    orig = (rollback or {}).get("original_param") or {}
    corrected = (rollback or {}).get("corrected_param") or {}
    queries = [str(c["args"]["query"]) for c in expected_calls if "query" in c.get("args", {}) and not _is_ref(c["args"]["query"])]
    out, seen = [], set()
    for c in expected_calls:
        fn = c["function"]
        for arg, v in c.get("args", {}).items():
            typ = SLOT_TYPES.get((fn, arg))
            if typ is None or _is_ref(v):
                continue
            s = {"key": f"{fn}.{arg}", "label": _label(fn, arg), "type": typ, "value": v, "superseded": []}
            if arg in corrected and arg in orig and corrected[arg] == v:
                s["superseded"] = [orig[arg]]
            if typ == "value":  # a filter value: by what it looks like
                if isinstance(v, bool) or str(v).lower() == "true":
                    s.update(type="enum", value="true", synonyms={"true": PETS})
                else:
                    try:
                        s.update(type="amount", value=float(v))
                    except (TypeError, ValueError):
                        s.update(type="name")
            if s["type"] == "amount":
                s["value"], s["superseded"] = float(s["value"]), [float(x) for x in s["superseded"]]
            if typ == "enum":
                syn = _SYN.get(arg, {})
                if arg == "pets_allowed":
                    syn = {str(v): PETS}
                s["synonyms"] = {k: list(vals) for k, vals in syn.items()} if syn else {}
                s["value"] = str(v)
                s["superseded"] = [str(x) for x in s["superseded"]]
            if typ == "city":
                s["synonyms"] = {str(v): CITY.get(str(v), [str(v)]), **{str(x): CITY.get(str(x), [str(x)]) for x in s["superseded"]}}
            if typ == "count":
                units = ["bedroom", "bedrooms", "bed", "beds", "br", "room", "rooms"] if arg == "bedrooms" else list(COUNT_UNITS)
                if arg == "quantity":
                    for q in queries + [str(x) for x in (orig.get("query"),) if x]:
                        units += [w for w in normalize(q).split() if w not in STOP]
                s["units"] = units
            sig = (s["type"], json.dumps(s["value"]))
            if sig in seen:
                continue
            seen.add(sig)
            out.append(s)
    return out


def _join_digits(norm: str) -> str:
    """Streaming transcripts split numbers into tokens ("1 5 0 euros", "1 50", "October 7 th"): joined."""
    norm = re.sub(r"\b(\d+) (st|nd|rd|th)\b", r"\1", norm)
    return re.sub(r"(?<![\d.:])\d+(?: \d+)+(?![\d.:])", lambda m: m.group(0).replace(" ", ""), norm)


def mentions(slot: dict, value, text: str) -> bool:
    """Whether ``text`` states ``value`` for ``slot`` (rules only), as written or with split numbers joined."""
    norm = normalize(text)
    joined = _join_digits(norm)
    return _mentions(slot, value, norm) or (joined != norm and _mentions(slot, value, joined))


def _mentions(slot: dict, value, norm: str) -> bool:
    typ = slot["type"]
    if typ == "count":
        n = int(float(value))
        for pos, v in _ints(norm):
            if v != n:
                continue
            window = norm[max(0, pos - 28): pos + len(str(v)) + 28]
            if any(_word(window, u) or _word(window, _stem(u)) or _word(window, u + "s") for u in slot.get("units", [])):
                return True
        return False
    if typ == "monthday":
        m = re.match(r"([A-Za-z]+)\s+(\d+)", str(value))
        if not m:
            return False
        month, day = m.group(1).lower(), int(m.group(2))
        return (re.search(rf"\b{month} (?:the )?{day}\b", norm) is not None
                or re.search(rf"\b(?:the )?{day} (?:of )?{month}\b", norm) is not None)
    if typ == "name":
        words = [w for w in normalize(str(value)).split()]
        spelled = _collapse_spelled(norm)
        return bool(words) and all(_word(norm, w) or _word(spelled, w) for w in words)
    if typ == "code":  # the characters in order, starting and ending at word boundaries ("B O B one two" = BOB12)
        want, toks = _alnum(str(value).lower()), norm.split()
        for i in range(len(toks)):
            cat = ""
            for tok in toks[i:]:
                cat += _alnum(tok)
                if cat == want:
                    return True
                if len(cat) >= len(want) or not want.startswith(cat):
                    break
        return False
    if typ == "amount":
        n = float(value)
        return any(abs(v - n) < 1e-6 for _, v in _ints(norm)) or bool(re.search(rf"(?<![\d.]){re.escape(f'{n:g}')}(?![\d])", norm))
    if typ == "phrase":
        want = [_stem(w) for w in normalize(str(value)).split() if w not in STOP]
        have = {_stem(w) for w in norm.split()}
        return bool(want) and all(w in have for w in want)
    if typ in ("city", "enum"):
        syn = slot.get("synonyms", {}).get(str(value)) or [str(value).replace("_", " ")]
        return any(_word(norm, normalize(s)) for s in syn)
    raise ValueError(f"unknown slot type {typ}")


# ---------------------------------------------------------------- outcome of an episode


def _agent_turns(ep: dict, before_ms: int | None = None) -> list[dict]:
    return sorted((t for t in ep["turns"] if t["role"] == "agent" and t.get("text") and t["end_time"] > t["start_time"]
                   and (before_ms is None or t["start_time"] < before_ms)), key=lambda t: t["start_time"])


def rule_check(turns: list[dict], slots: list[dict]) -> list[dict]:
    """Per slot ``status``: 'correct' / 'wrong' / None (no rule mention), decided by the last agent turn that states the
    target or a superseded value."""
    out = []
    for s in slots:
        last = None
        for a in turns:
            hit_t = mentions(s, s["value"], a["text"])
            hit_w = any(mentions(s, v, a["text"]) for v in s["superseded"])
            if hit_t:
                last = "correct"
            elif hit_w:
                last = "wrong"
        out.append({"key": s["key"], "status": last})
    return out


FALLBACK_PROMPT = """Below is a phone call between a caller and a voice assistant, and some details of the caller's request that the assistant needed to take down.
For each detail, decide from the ASSISTANT's lines whether, by the end of the transcript, the assistant had confirmed (repeated back, acknowledged or used) that detail with exactly the value given, allowing for paraphrases, number formats and small spelling differences in names:
- "confirmed": the assistant confirmed this value (and did not end up with a different one);
- "wrong": the assistant's final version of this detail is a different value;
- "not_confirmed": the assistant never stated this detail.

Transcript (times in seconds):
{transcript}

Details:
{details}

Answer with JSON only, one entry per detail key: {{{keys}}}"""


def transcript(ep: dict, before_ms: int | None = None) -> str:
    lines = []
    for t in sorted((t for t in ep["turns"] if t.get("text") and t["end_time"] > t["start_time"]
                     and (before_ms is None or t["start_time"] < before_ms)), key=lambda t: t["start_time"]):
        who = "CALLER" if t["role"] == "user" else "ASSISTANT"
        lines.append(f"[{t['start_time'] / 1000:.1f}] {who}: {t['text']}")
    return "\n".join(lines)


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v).replace("_", " ")


def fallback_prompt(ep: dict, unresolved: list[dict], before_ms: int | None = None) -> str:
    lines = []
    for s in unresolved:
        extra = f" (the caller first said {_fmt(s['superseded'][0])} and then corrected it)" if s["superseded"] else ""
        lines.append(f"- {s['key']}: {s['label']} = {_fmt(s['value'])}{extra}")
    keys = ", ".join(f'"{s["key"]}": "confirmed|wrong|not_confirmed"' for s in unresolved)
    return FALLBACK_PROMPT.format(transcript=transcript(ep, before_ms), details="\n".join(lines), keys=keys)


def slots_of(task: dict) -> list[dict]:
    """The slots of a task as stored in an episode (``meta.task``)."""
    b = task["scenario"]["benchmark"]
    return task_slots(task["criteria"]["expected_tool_calls"], b.get("state_rollback_details"), b.get("script"), b.get("id"))


async def outcome(ep: dict, llm=None, before_ms: int | None = None, slots: list[dict] | None = None) -> dict:
    """``{"outcome", "fulfilled", "n_slots", "slots": [{"key", "status", "source", "superseded"}], "rule_resolved"}``
    over the agent's turns that start before ``before_ms`` (all when None). ``llm``: a ``TextGen`` for the
    fallback (None: unresolved slots count as missing). A task without slots scores ``outcome`` None."""
    task = ep["meta"]["task"]
    if slots is None:
        slots = slots_of(task)
    if not slots:
        return {"outcome": None, "fulfilled": None, "n_slots": 0, "slots": [], "rule_resolved": 0}
    turns = _agent_turns(ep, before_ms)
    checked = rule_check(turns, slots)
    unresolved = [s for s, c in zip(slots, checked) if c["status"] is None]
    verdict: dict[str, str] = {}
    if unresolved and llm is not None and turns:
        msg = fallback_prompt(ep, unresolved, before_ms)
        for _ in range(3):
            try:
                raw = await llm.chat([{"role": "user", "content": msg}], temperature=0, max_tokens=200)
                js = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
                verdict = {s["key"]: str(js.get(s["key"], "not_confirmed")).strip().lower() for s in unresolved}
                break
            except Exception:  # noqa: BLE001
                verdict = {}
    per = []
    for s, c in zip(slots, checked):
        if c["status"] is not None:
            per.append({"key": s["key"], "status": c["status"], "source": "rule", "superseded": bool(s["superseded"])})
        else:
            v = verdict.get(s["key"], "not_confirmed")
            per.append({"key": s["key"], "status": "correct" if v == "confirmed" else "wrong" if v == "wrong" else "missing",
                        "source": "llm" if verdict else "none", "superseded": bool(s["superseded"])})
    ok = sum(p["status"] == "correct" for p in per)
    return {"outcome": round(ok / len(per), 4), "fulfilled": ok == len(per), "n_slots": len(per), "slots": per,
            "rule_resolved": sum(c["status"] is not None for c in checked)}
