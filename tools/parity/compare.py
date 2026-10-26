"""Compare two sets of episodes (reference vs under test) field by field — the verdict of a parity test.

    python tools/parity/compare.py REF.jsonl NEW.jsonl [--media-ref DIR --media-new DIR] [--out report.json]

Episodes are joined by task id + seed (+ arm / condition when recorded), so the two runs may name their episodes
differently (``fdb-s0`` vs ``fdb/s0``). Fields that cannot agree between two runs are ignored (``meta.run_id``,
``wall_s``, ``meta.config`` and its digest, timestamps, file paths, ``meta.env`` class names); everything else —
every turn, timing, score, judge verdict, duplex metric — must be identical. Floats are compared exactly, and a
difference that is only in floats is reported apart with its largest absolute error (a precision drift, not a
behavioural one). With media roots, the referenced WAV files are compared by content hash too. Exit status 0 when
the sets are identical, 1 otherwise; the JSON report lists every differing path with both values (first 40 per
episode)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

IGNORE = {  # dotted paths (prefix match) that two runs cannot share
    "meta.run_id", "meta.wall_s", "meta.config", "meta.config_digest", "meta.created", "meta.started", "meta.finished",
    "meta.host", "meta.out", "meta.media_root", "meta.episode_id", "meta.arm_layers", "meta.pid", "meta.git",
    "meta.timestamp", "meta.date", "meta.version",
}
IGNORE_KEYS = {"wall_s", "wall_ms", "elapsed_s", "started_at", "finished_at", "created_at", "run_id", "episode_id"}  # anywhere (ids: the join key)
WALL_SUFFIXES = ("_wall_ms", "_wall_s")  # measured latencies of model calls (asr_wall_ms …): wall-clock, never equal
PATH_KEYS = {"wav", "path", "file", "audio_path", "media", "audio"}  # values that are file paths: compared by basename, media by hash
# fields the plugin runs record and the scripts never did (added metadata, not a behavioural difference): present in
# NEW and absent in REF is reported apart as "added"; any other disagreement on them is a difference
DESCRIPTIVE = {"meta.agent", "meta.task.scenario._agent"}  # how a run describes its agent (each script had its own dict): reported apart, not a verdict
ADDED = {"meta.config", "meta.roles", "meta.controlled", "meta.standards", "meta.agent_events", "meta.agent_name", "meta.task_id",
         "meta.user.behaviours", "meta.task.scenario.agent_instructions", "meta.nodes", "eval.interaction", "meta.condition",
         "meta.agent_base", "meta.group", "meta.clock"}


def norm_id(eid: str) -> str:
    """``name``, ``name-s3``, ``name/s3``, ``tag/name/s3/A`` → ``name|s<seed>|<arm>`` (the schemes of the scripts and of
    ``ig run`` unified; no seed means seed 0)."""
    parts = eid.split("/")
    arm = ""
    if len(parts) >= 3 and re.fullmatch(r"s\d+", parts[-2]):
        arm, parts = parts[-1], parts[:-1]
    eid = "/".join(parts)
    m = re.search(r"[-/]s(\d+)$", eid)
    seed = m.group(1) if m else "0"
    name = eid[:m.start()] if m else eid
    return f"{name.split('/')[-1]}|s{seed}|{arm}"  # a run-folder prefix (the scripts' --agent, the bundle's agent name) is not identity


def join_key(ep: dict) -> str:
    if "meta" not in ep:  # an agent trace or another per-episode record: by its episode id, naming schemes unified
        return norm_id(str(ep.get("episode_id") or ep.get("id") or ""))
    m = ep.get("meta", {})
    if m.get("episode_id"):
        return norm_id(str(m["episode_id"]))
    task = (m.get("task") or {}).get("id") if isinstance(m.get("task"), dict) else m.get("task")
    arm = m.get("arm") or m.get("condition") or (m.get("ab") or {}).get("arm") or ""
    seed = m.get("seed")
    eid = m.get("episode_id", "")
    if task is None:
        task = re.sub(r"[-/]s\d+$", "", eid)
    if seed is None:
        s = re.search(r"[-/]s(\d+)$", eid)
        seed = int(s.group(1)) if s else 0
    return f"{task}|s{seed}|{arm}"


def norm(x, path: str = ""):
    if isinstance(x, dict):
        return {k: norm(v, f"{path}.{k}" if path else k) for k, v in x.items() if k not in IGNORE_KEYS and not k.endswith(WALL_SUFFIXES)}
    if isinstance(x, list):
        return [norm(v, f"{path}[]") for v in x]
    if isinstance(x, str) and path.split(".")[-1].split("[")[0] in PATH_KEYS and "/" in x:
        return Path(x).name
    return x


def walk(a, b, path, out, floats, added=None):
    if any(path == p or path.startswith(p + ".") for p in IGNORE):
        return
    if added is not None and any(path == p or path.startswith(p + ".") for p in DESCRIPTIVE):
        if a != b:
            added.append((path, a, b))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if added is not None and any(p == d or p.startswith(d + ".") for d in DESCRIPTIVE):
                if a.get(k) != b.get(k):
                    added.append((p, a.get(k, "<absent>"), b.get(k, "<absent>")))
            elif k not in a and k in b and p in ADDED:
                (added if added is not None else out).append((p, "<absent>", b[k]))
            elif k not in a or k not in b:
                out.append((p, a.get(k, "<absent>"), b.get(k, "<absent>")))
            else:
                walk(a[k], b[k], p, out, floats, added)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}.len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out, floats, added)
        return
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not (isinstance(a, bool) or isinstance(b, bool)):
            if a != b and not (math.isnan(a) and math.isnan(b)):
                floats.append((path, a, b, abs(a - b)))
            return
    if a != b:
        out.append((path, a, b))


def load(path: str) -> dict[str, dict]:
    eps = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            ep = json.loads(line)
            k = join_key(ep)
            if k in eps:
                k = f"{k}#{sum(1 for e in eps if e.startswith(k))}"
            eps[k] = ep
    return eps


def media_hashes(ep: dict, root: Path | None) -> dict[str, str]:
    """``{relative wav: sha256}`` of the files an episode references under ``root``."""
    if root is None:
        return {}
    out = {}

    def rec(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in PATH_KEYS and isinstance(v, str) and v.endswith((".wav", ".flac", ".opus")):
                    p = root / v if not Path(v).is_absolute() else Path(v)
                    if not p.exists():
                        p = next(root.rglob(Path(v).name), None)
                    out[Path(v).name] = hashlib.sha256(p.read_bytes()).hexdigest() if p and p.exists() else "<missing>"
                else:
                    rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)
    rec(ep)
    return out


def compare(ref: dict[str, dict], new: dict[str, dict], media_ref=None, media_new=None, expect_only_ref: str | None = None) -> dict:
    only_ref = sorted(set(ref) - set(new))
    expected = [k for k in only_ref if expect_only_ref and re.search(expect_only_ref, k)]
    report = {"episodes": {"ref": len(ref), "new": len(new), "common": 0, "identical": 0, "differ": 0, "float_only": 0},
              "only_ref": [k for k in only_ref if k not in expected], "only_ref_expected": expected,
              "only_new": sorted(set(new) - set(ref)), "diffs": {}, "added": {}, "max_float_err": 0.0}
    for k in sorted(set(ref) & set(new)):
        report["episodes"]["common"] += 1
        a, b = norm(ref[k]), norm(new[k])
        out, floats, added = [], [], []
        walk(a, b, "", out, floats, added)
        if added:
            report["added"][k] = sorted({p for p, _, _ in added})
        if media_ref or media_new:
            ha, hb = media_hashes(ref[k], media_ref), media_hashes(new[k], media_new)
            for f in sorted(set(ha) | set(hb)):
                if ha.get(f) != hb.get(f):
                    out.append((f"media.{f}", ha.get(f, "<absent>")[:16], hb.get(f, "<absent>")[:16]))
        if floats:
            report["max_float_err"] = max(report["max_float_err"], max(f[3] for f in floats))
        if out:
            report["episodes"]["differ"] += 1
            report["diffs"][k] = {"fields": [{"path": p, "ref": r, "new": n} for p, r, n in out[:40]], "n": len(out),
                                  "floats": [{"path": p, "ref": r, "new": n, "err": e} for p, r, n, e in floats[:20]]}
        elif floats:
            report["episodes"]["float_only"] += 1
            report["diffs"][k] = {"fields": [], "n": 0, "floats": [{"path": p, "ref": r, "new": n, "err": e} for p, r, n, e in floats[:20]]}
        else:
            report["episodes"]["identical"] += 1
    report["verdict"] = ("IDENTICAL" if report["episodes"]["identical"] == report["episodes"]["common"] and not report["only_ref"] and not report["only_new"]
                         else "FLOAT_DRIFT" if report["episodes"]["differ"] == 0 and not report["only_ref"] and not report["only_new"] else "DIFFER")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--json", action="store_true", help="REF and NEW are single JSON documents (a report summary), not episode files")
    ap.add_argument("--media-ref")
    ap.add_argument("--media-new")
    ap.add_argument("--out")
    ap.add_argument("--expect-only-ref", help="regex of join keys the reference may have alone (runs the bundle does not make)")
    a = ap.parse_args(argv)
    if a.json:
        def keyed(x):  # dict keys that are episode ids: the two naming schemes unified
            if isinstance(x, dict):
                ids = any(isinstance(k, str) and re.search(r"/s\d+(/[A-Za-z]+)?$", k) for k in x)  # a dict keyed by episode ids on at least one side
                return {(norm_id(k) if ids and isinstance(k, str) else k): keyed(v) for k, v in x.items()}
            if isinstance(x, list):
                xs = [keyed(v) for v in x]
                if xs and all(isinstance(v, dict) and isinstance(v.get("id"), str) for v in xs):  # rows per episode: by id, not by completion order
                    xs = sorted(xs, key=lambda v: norm_id(v["id"]))
                return xs
            return x
        ra, rb = keyed(json.loads(Path(a.ref).read_text())), keyed(json.loads(Path(a.new).read_text()))
        out, floats = [], []
        walk(norm(ra), norm(rb), "", out, floats)
        verdict = "IDENTICAL" if not out and not floats else "FLOAT_DRIFT" if not out else "DIFFER"
        rep = {"verdict": verdict, "fields": [{"path": p, "ref": r, "new": n} for p, r, n in out[:80]],
               "floats": [{"path": p, "ref": r, "new": n, "err": e} for p, r, n, e in floats[:40]], "n": len(out), "n_floats": len(floats)}
        if a.out:
            Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
        print(f"{verdict}: {len(out)} fields differ, {len(floats)} float-only" + (f" (max err {max(f[3] for f in floats):.3g})" if floats else ""))
        for f in rep["fields"][:12]:
            print(f"    {f['path']}: {str(f['ref'])[:80]!r} -> {str(f['new'])[:80]!r}")
        return 0 if verdict == "IDENTICAL" else 1
    rep = compare(load(a.ref), load(a.new), Path(a.media_ref) if a.media_ref else None, Path(a.media_new) if a.media_new else None, a.expect_only_ref)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    e = rep["episodes"]
    print(f"{rep['verdict']}: {e['identical']}/{e['common']} identical, {e['float_only']} float-only (max err {rep['max_float_err']:.3g}), "
          f"{e['differ']} differ; only in ref {len(rep['only_ref'])} (+{len(rep['only_ref_expected'])} expected), only in new {len(rep['only_new'])}; "
          f"added-by-new fields in {len(rep['added'])} episodes")
    for k, d in list(rep["diffs"].items())[:5]:
        print(f"  {k}: {d['n']} fields differ")
        for f in d["fields"][:6]:
            print(f"    {f['path']}: {str(f['ref'])[:80]!r} -> {str(f['new'])[:80]!r}")
        for f in d["floats"][:3]:
            print(f"    {f['path']}: {f['ref']} -> {f['new']} (float err {f['err']:.3g})")
    return 0 if rep["verdict"] == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())
