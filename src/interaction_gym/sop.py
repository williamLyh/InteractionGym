"""Standard operating procedures (SOPs): one structured source format, many ways to show it to the agent.

An SOP is plain JSON-able data, so it lives in a task (``task.scenario["sop"]``) and in the trajectory
(``meta.policy["source"]``)::

    {"name": "Trattoria Roma table bookings",
     "rules": ["Keep replies short.", ...],                      # general conduct, always in force
     "actions": {
       "book_table": {
         "description": "Book a table.",
         "requires": {"chain": [                                  # a condition tree, see below
             {"check": "name_phone", "text": "The caller has given a name and a phone number.", "by": "ask"},
             {"all": [{"check": "open", "text": "...", "by": "look_up"},
                      {"any": [{"check": "small", ...}, {"check": "manager_ok", ...}]}]}]},
         "then": ["Read the booking back and confirm it."],      # what to do once it is allowed
         "otherwise": "Explain which condition is not met and decline."}}}

A condition tree is made of

- ``{"check": id, "text": ..., "by": "ask" | "look_up", "on_fail": ...}``: one condition. ``by`` says how the agent
  establishes it: ``ask`` the caller, or ``look_up`` information it has (records, a tool); ``on_fail`` is what to do
  when it does not hold (e.g. "offer the events line instead");
- ``{"all": [...]}``: every part must hold; ``{"any": [...]}``: at least one must hold;
- ``{"chain": [...]}``: every part must hold, established in this order.

**Formats** turn an SOP into the text the agent reads. Built in: ``structured`` (nested ALL / ANY / IN ORDER
bullets, as SOPBench's structured format), ``list`` (numbered steps; alternatives as lettered sub-items, so the
logic survives flattening), ``code`` (Python-style pseudo-code) and ``mermaid`` (a flowchart). A format is a
function ``(sop: dict) -> str``; add your own with ``@register_format("name")`` or, from another package, an entry
point in the group ``interaction_gym.sop_formats`` (``name = "pkg.module:function"``); ``render(sop, "name")`` uses
it.

**Checking** works on the tree and a case's ground truth (``{check id: holds?}``): ``holds`` says whether the action
is allowed; ``witnessed`` whether a sequence of established checks justifies acting (every required check, a holding
alternative of each ``any``, ``chain`` parts in order); ``refuted`` whether it justifies declining (a failing check
that makes the tree false).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from importlib import metadata

Format = Callable[[dict], str]
_FORMATS: dict[str, Format] = {}
ENTRY_POINT_GROUP = "interaction_gym.sop_formats"
KINDS = ("all", "any", "chain")


# ---------------------------------------------------------------- the tree

def kind(node: dict) -> str:
    if "check" in node:
        return "check"
    ks = [k for k in KINDS if k in node]
    if len(ks) != 1:
        raise ValueError(f"an SOP node has one of check / {' / '.join(KINDS)}: {node}")
    return ks[0]


def checks(node: dict | None) -> list[dict]:
    """The leaf conditions of a tree, in order."""
    if not node:
        return []
    k = kind(node)
    return [node] if k == "check" else [c for part in node[k] for c in checks(part)]


def all_checks(sop: dict) -> dict[str, dict]:
    """Every condition of every action, by id."""
    return {c["check"]: c for a in sop.get("actions", {}).values() for c in checks(a.get("requires"))}


def validate(sop: dict) -> dict:
    """Raises ``ValueError`` on a malformed SOP (unknown node kinds, a check id used twice with different texts)."""
    seen: dict[str, str] = {}
    for name, a in sop.get("actions", {}).items():
        for c in checks(a.get("requires")):
            if c.get("by", "look_up") not in ("ask", "look_up"):
                raise ValueError(f"{name}: check {c['check']}: by must be ask or look_up")
            if seen.setdefault(c["check"], c.get("text", "")) != c.get("text", ""):
                raise ValueError(f"check id {c['check']} has two texts")
    return sop


def holds(node: dict | None, truth: dict[str, bool]) -> bool:
    """Is the condition met, given which checks hold? (An absent tree always holds.)"""
    if not node:
        return True
    k = kind(node)
    if k == "check":
        return bool(truth[node["check"]])
    parts = [holds(p, truth) for p in node[k]]
    return any(parts) if k == "any" else all(parts)


def witnessed(node: dict | None, established: list[str], truth: dict[str, bool], known: set[str] = frozenset()) -> bool:
    """Do the checks the agent established (ids, in the order it established them) justify acting? Every check of
    an ``all`` / ``chain``, a holding alternative of an ``any``; the parts of a ``chain`` in order. ``known``: checks
    the agent knows without establishing them in the conversation (e.g. what is on its screen); they count wherever
    they are needed and put no constraint on the order."""
    if not node:
        return True
    pos = {c: i for i, c in reversed(list(enumerate(established)))}  # first time each was established
    NO = -1  # not established; None: established, at no particular time

    def first(n):  # when n was fully established (max over what it needed)
        k = kind(n)
        if k == "check":
            if not truth.get(n["check"]):
                return NO
            return None if n["check"] in known else pos.get(n["check"], NO)
        ts = [first(p) for p in n[k]]
        if k == "any":
            ok = [t for t in ts if t != NO]
            return NO if not ok else None if None in ok else min(ok)
        if NO in ts:
            return NO
        timed = [t for t in ts if t is not None]
        if k == "chain" and any(b < a for a, b in zip(timed, timed[1:])):
            return NO
        return max(timed) if timed else None

    return first(node) != NO


def refuted(node: dict | None, established: list[str], truth: dict[str, bool], known: set[str] = frozenset()) -> bool:
    """Do the established (or ``known``) checks justify declining: a failing check (or failing alternatives) that
    makes the tree false?"""
    if not node:
        return False
    k = kind(node)
    if k == "check":
        return (node["check"] in established or node["check"] in known) and not truth.get(node["check"], True)
    parts = [refuted(p, established, truth, known) for p in node[k]]
    return all(parts) if k == "any" else any(parts)


def failing(node: dict | None, truth: dict[str, bool]) -> list[str]:
    """The checks that do not hold, in order."""
    return [c["check"] for c in checks(node) if not truth.get(c["check"], True)]


# ---------------------------------------------------------------- formats (plugins)

def register_format(name: str):
    """Decorator: ``@register_format("mine") def mine(sop) -> str``."""
    def deco(fn: Format) -> Format:
        _FORMATS[name] = fn
        return fn
    return deco


def formats() -> dict[str, Format]:
    """The available formats: built-in, registered, and entry points of ``interaction_gym.sop_formats``."""
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        if ep.name not in _FORMATS:
            try:
                _FORMATS[ep.name] = ep.load()
            except Exception as e:  # noqa: BLE001  (a broken plugin should not take the others down)
                warnings.warn(f"SOP format plugin {ep.name} ({ep.value}) failed to load: {e}", stacklevel=2)
    return dict(_FORMATS)


def render(sop: dict, fmt: str = "structured") -> str:
    fs = formats()
    if fmt not in fs:
        raise KeyError(f"unknown SOP format {fmt!r}; available: {', '.join(sorted(fs))}")
    return fs[fmt](sop).strip()


def _cond(c: dict) -> str:
    how = {"ask": "ask the caller", "look_up": "check your information"}.get(c.get("by", "look_up"))
    out = f"{c['text']} ({how})"
    return out + (f" If not: {c['on_fail']}" if c.get("on_fail") else "")


def _rules(sop: dict) -> str:
    rs = sop.get("rules") or []
    return ("General rules:\n" + "\n".join(f"- {r}" for r in rs) + "\n\n") if rs else ""


@register_format("structured")
def structured(sop: dict) -> str:
    head = {"all": "ALL of these must hold:", "any": "ANY ONE of these must hold:", "chain": "These steps, IN THIS ORDER:"}

    def tree(n, ind=0) -> list[str]:
        k, pad = kind(n), "  " * ind
        if k == "check":
            return [pad + _cond(n)]
        out = [pad + head[k]]
        for i, p in enumerate(n[k], 1):
            sub = tree(p, ind + 1)
            mark = f"{i}. " if k == "chain" else "• "
            out.append(pad + "  " + mark + sub[0].strip())
            out += sub[1:]
        return out

    parts = []
    for name, a in sop.get("actions", {}).items():
        lines = [f"* {name}: {a.get('description', '')}".rstrip(": ")]
        lines += ["  Before doing it:"] + ["  " + x for x in tree(a["requires"], 1)] if a.get("requires") else ["  No conditions."]
        lines += [f"  Then: {t}" for t in a.get("then", [])]
        if a.get("otherwise"):
            lines.append(f"  Otherwise: {a['otherwise']}")
        parts.append("\n".join(lines))
    return _rules(sop) + "Actions and their conditions:\n\n" + "\n\n".join(parts)


@register_format("list")
def numbered(sop: dict) -> str:
    """Numbered steps per action. ``any`` becomes "one of: (a) ... (b) ..."; ``chain`` and ``all`` are steps (a chain's
    order is stated); the logic of the tree is kept, unlike a flat list of every condition."""
    def items(n) -> list[str]:
        k = kind(n)
        if k == "check":
            return [_cond(n)]
        if k == "any":
            alts = []
            for j, p in enumerate(n[k]):
                sub = items(p)
                alts.append(f"({chr(97 + j)}) " + ("; and ".join(sub) if len(sub) > 1 else sub[0]))
            return ["At least one of: " + " OR ".join(alts)]
        return [x for p in n[k] for x in items(p)]

    parts = []
    for name, a in sop.get("actions", {}).items():
        req = a.get("requires")
        steps = items(req) if req else []
        order = "in this order" if req and kind(req) == "chain" else "all of them"
        lines = [f"{name}: {a.get('description', '')}".rstrip(": ")]
        if steps:
            lines.append(f"Before doing it, check ({order}):")
            lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
        lines += [f"Then: {t}" for t in a.get("then", [])]
        if a.get("otherwise"):
            lines.append(f"Otherwise: {a['otherwise']}")
        parts.append("\n".join(lines))
    return _rules(sop) + "\n\n".join(parts)


@register_format("code")
def code(sop: dict) -> str:
    def expr(n) -> str:
        k = kind(n)
        if k == "check":
            return f"{n['check']}()"
        joined = (" or " if k == "any" else " and ").join(expr(p) for p in n[k])
        return f"({joined})" if len(n[k]) > 1 else joined

    out = []
    for name, a in sop.get("actions", {}).items():
        for c in checks(a.get("requires")):
            how = "ask the caller" if c.get("by") == "ask" else "check your information"
            line = f'def {c["check"]}() -> bool:\n    """{c["text"]} ({how})"""\n    ...'
            if line not in out:
                out.append(line)
    out.append("")
    for name, a in sop.get("actions", {}).items():
        req = a.get("requires")
        body = [f"def {name}():"]
        if a.get("description"):
            body.append(f'    """{a["description"]}"""')
        steps = [] if not req else req["chain"] if kind(req) == "chain" else [req]
        steps = [q for p in steps for q in (p["all"] if kind(p) == "all" else [p])]  # an ALL step: one test per part
        for p in steps:  # in order
            body.append(f"    if not {expr(p)}:")
            fails = [c.get("on_fail") for c in checks(p) if c.get("on_fail")]
            body.append(f"        return decline({(fails[-1] if fails else a.get('otherwise', 'decline'))!r})")
        for t in a.get("then", []):
            body.append(f"    # then: {t}")
        body.append(f"    return do_{name}()")
        out.append("\n".join(body))
    return _rules(sop) + "```python\n" + "\n\n".join(out) + "\n```"


@register_format("mermaid")
def mermaid(sop: dict) -> str:
    lines, n = ["flowchart TD"], [0]

    def nid() -> str:
        n[0] += 1
        return f"n{n[0]}"

    def q(s: str) -> str:
        return s.replace('"', "'")

    def flow(node, yes: str, no: str) -> str:  # entry node id of a sub-flow that goes to yes / no
        k = kind(node)
        if k == "check":
            i = nid()
            lines.append(f'  {i}{{"{q(node["text"])}"}}')
            lines.append(f"  {i} -->|yes| {yes}")
            if node.get("on_fail"):
                f = nid()
                lines.append(f'  {f}["{q(node["on_fail"])}"]')
                lines.append(f"  {i} -->|no| {f}")
            else:
                lines.append(f"  {i} -->|no| {no}")
            return i
        parts = node[k]
        if k == "any":  # try each alternative in turn
            nxt = no
            for p in reversed(parts):
                nxt = flow(p, yes, nxt)
            return nxt
        nxt = yes  # all / chain: each part leads to the next
        for p in reversed(parts):
            nxt = flow(p, nxt, no)
        return nxt

    for name, a in sop.get("actions", {}).items():
        start, do, dec = nid(), nid(), nid()
        lines.append(f'  {start}(["{q(name)}: {q(a.get("description", ""))}"])')
        lines.append(f'  {do}["do {q(name)}' + ("; then " + q("; ".join(a["then"])) if a.get("then") else "") + '"]')
        lines.append(f'  {dec}["{q(a.get("otherwise", "decline"))}"]')
        entry = flow(a["requires"], do, dec) if a.get("requires") else do
        lines.append(f"  {start} --> {entry}")
    return _rules(sop) + "```mermaid\n" + "\n".join(lines) + "\n```"
