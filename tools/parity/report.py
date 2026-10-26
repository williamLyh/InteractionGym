"""One table over the parity stages: ``python tools/parity/report.py $PARITY_OUT [--out report.md]``.

For every stage directory with a ``verdict.txt``: the proxy's replay statistics (hits / misses / overflow, WebSocket
sessions), each compared file's verdict (identical episodes, float-only, differing, only-in-one-side) and the first
differing fields, so the state of the parity test reads at a glance."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def stage_rows(d: Path) -> tuple[dict, list[dict]]:
    v = (d / "verdict.txt").read_text().splitlines()
    head = {"rc": v[0] if v else "", "proxy": {}, "misses": ""}
    for line in v:
        if line.startswith("proxy: "):
            try:
                head["proxy"] = json.loads(line[7:])
            except ValueError:
                head["proxy"] = line[7:]
        elif line.startswith("misses: "):
            head["misses"] = line[8:].strip()
    rows = []
    for p in sorted(d.glob("compare_*.json")):
        rep = json.loads(p.read_text())
        name = p.stem[len("compare_"):]
        if "episodes" in rep:
            e = rep["episodes"]
            first = next(iter(rep["diffs"].values()), {})
            fields = first.get("fields", [])[:3] or first.get("floats", [])[:3]
            rows.append({"file": name, "verdict": rep["verdict"], "n": f"{e['identical']}/{e['common']}",
                         "float_only": e["float_only"], "differ": e["differ"], "only_ref": len(rep["only_ref"]), "only_new": len(rep["only_new"]),
                         "max_float_err": rep.get("max_float_err", 0.0),
                         "first": "; ".join(f"{f['path']}: {str(f['ref'])[:40]!r}→{str(f['new'])[:40]!r}" for f in fields)})
        else:
            fields = rep.get("fields", [])[:3]
            rows.append({"file": name, "verdict": rep["verdict"], "n": "", "float_only": rep.get("n_floats", 0), "differ": rep.get("n", 0),
                         "only_ref": "", "only_new": "", "max_float_err": "", "first": "; ".join(f"{f['path']}: {str(f['ref'])[:40]!r}→{str(f['new'])[:40]!r}" for f in fields)})
    missing = [l for l in v if ": MISSING" in l]
    for m in missing:
        rows.append({"file": m.split(":")[0], "verdict": "MISSING", "n": "", "float_only": "", "differ": "", "only_ref": "", "only_new": "", "max_float_err": "", "first": m})
    return head, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    root = Path(a.root)
    L = ["# Parity: main scripts vs plugin bundles (record / replay)", ""]
    overall = "IDENTICAL"
    for d in sorted(p for p in root.iterdir() if (p / "verdict.txt").exists()):
        head, rows = stage_rows(d)
        px = head["proxy"] if isinstance(head["proxy"], dict) else {}
        http, ws = px.get("http", {}), px.get("ws", {})
        L += [f"## {d.name}", "",
              f"{head['rc']} · proxy replay: http hit {http.get('hit', '?')} / miss {http.get('miss', '?')} / overflow {http.get('overflow', '?')}, "
              f"ws sessions {ws.get('sessions', '?')} / miss {ws.get('miss', '?')} (unused recorded {px.get('ws_unused_sessions', '?')}) · misses logged {head['misses']}", "",
              "| file | verdict | identical | float-only | differ | only old | only new | max float err | first differences |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['file']} | **{r['verdict']}** | {r['n']} | {r['float_only']} | {r['differ']} | {r['only_ref']} | {r['only_new']} | {r['max_float_err']} | {r['first']} |")
            if r["verdict"] != "IDENTICAL":
                overall = "DIFFER" if r["verdict"] in ("DIFFER", "MISSING") or overall == "DIFFER" else "FLOAT_DRIFT"
        L.append("")
    L.insert(2, f"**Overall: {overall}**\n")
    text = "\n".join(L)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
