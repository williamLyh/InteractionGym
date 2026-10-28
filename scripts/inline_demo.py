"""Put freshly recorded takes into the demo page (``demo/demo/index.html``).

    python scripts/inline_demo.py --export runs/demo_export --raw a=runs/demo/a/a-...-s3.json b=... c=... \
        [--note a="what happens in this take ..."] [--out demo/demo/index.html]

``--export`` is the folder ``export_demo.py`` wrote (``<key>.json``, ``<key>.user.m4a``, ``<key>.agent.m4a`` per key
a / b / c); ``--raw`` names each take's raw recording (``record_demo.py`` output) for the context card (the agent's
prompt, the caller's persona, goal, profile and first turn). The page keeps its own prose: the agent / environment
notes of the context card stay as they were, and a take's scenario narration is shown only when ``--note`` gives one
for it (otherwise the page falls back to the export's title). Standard library only."""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "demo" / "demo" / "index.html"
PREFIX = "<script>window.__EPS = "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--raw", nargs="+", default=[], help="key=path of the raw recording, per key")
    ap.add_argument("--note", nargs="*", default=[], help='key="scenario narration" for takes with a written note')
    ap.add_argument("--context", nargs="*", default=[], help="key=path of a hand-written context card (JSON: agent / user / env) for a take, "
                    "e.g. a new key the page has no card for yet; merged over the page's own before the raw refresh")
    ap.add_argument("--page", type=Path, default=PAGE)
    ap.add_argument("--out", type=Path, default=PAGE)
    a = ap.parse_args()
    raw = {k: Path(v) for k, v in (x.split("=", 1) for x in a.raw)}
    notes = {k: v for k, v in (x.split("=", 1) for x in a.note)}
    cards = {k: json.loads(Path(v).read_text()) for k, v in (x.split("=", 1) for x in a.context)}

    html = a.page.read_text()
    lines = html.split("\n")
    i = next(n for n, l in enumerate(lines) if l.startswith(PREFIX))
    old = json.loads(lines[i][len(PREFIX):].rstrip().removesuffix("</script>").rstrip(";"))

    eps = {}
    keys = [k for k in old if (a.export / f"{k}.json").exists()] + sorted(k for k in cards if k not in old)
    for key in keys:
        page = json.loads((a.export / f"{key}.json").read_text())
        audio = {}
        for who in ("user", "agent"):
            f = a.export / page["audio"][who]
            audio[who] = "data:audio/mp4;base64," + base64.b64encode(f.read_bytes()).decode()
        page["audio"] = audio
        ctx = json.loads(json.dumps(old.get(key, {}).get("context", {})))  # the hand-written parts stay (agent.note, env)
        for part, v in cards.get(key, {}).items():
            ctx[part] = {**ctx.get(part, {}), **v} if isinstance(v, dict) and isinstance(ctx.get(part), dict) else v
        if key in raw:
            r = json.loads(raw[key].read_text())
            m, sc = r["meta"], r.get("task", {}).get("scenario", {}) or r.get("scenario", {})
            ctx.setdefault("agent", {})["prompt"] = m.get("agent_prompt") or ctx.get("agent", {}).get("prompt")
            user = ctx.setdefault("user", {})
            for k_from, k_to in (("persona", "persona"), ("profile", "profile"), ("first_turn", "first_turn")):
                if sc.get(k_from) is not None:
                    user[k_to] = sc[k_from]
            if m.get("scenario_goal"):
                user["goal"] = m["scenario_goal"]
            if isinstance(user.get("profile"), dict):  # the card shows rows
                user["profile"] = [[k.replace("_", " "), str(v)] for k, v in user["profile"].items()]
            if isinstance(user.get("first_turn"), str):
                user["first_turn"] = {"text": user["first_turn"]}
        page["context"] = ctx
        eps[key] = page
    lines[i] = PREFIX + json.dumps(eps, ensure_ascii=False, separators=(",", ":")) + ";</script>"
    html = "\n".join(lines)

    # NOTES: the takes whose narration is given here, plus the page's own notes for takes that did not change (a note is
    # keyed by id@sim_ms, so a replaced take drops out and shows the export's title until it gets a note)
    old_notes = json.loads(re.search(r"const NOTES = (\{.*?\});", html).group(1))
    now = {k: f"{eps[k]['meta']['id']}@{eps[k]['meta']['sim_ms']}" for k in eps}
    kept = {v: 1 for k, v in now.items() if v in old_notes or k in notes}
    html = re.sub(r"const NOTES = \{.*?\};", "const NOTES = " + json.dumps(kept) + ";", html, count=1)
    for k, text in notes.items():
        pat = re.compile(r"(\n  %s: \{ file: DATA \+ \"%s\.json\".*?scenarioRow\(e, )\"(?:[^\"\\]|\\.)*\"(\))" % (k, k), re.S)
        html, n = pat.subn(lambda mm: mm.group(1) + json.dumps(text, ensure_ascii=False) + mm.group(2), html, count=1)
        if n != 1:
            raise SystemExit(f"could not place the note of {k}")
    a.out.write_text(html)
    print(f"-> {a.out} ({a.out.stat().st_size / 1e6:.1f} MB; takes: " + ", ".join(f"{k}={eps[k]['meta']['id']}@{eps[k]['meta']['sim_ms']}" for k in eps) + ")")


if __name__ == "__main__":
    main()
