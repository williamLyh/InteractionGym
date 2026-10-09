#!/usr/bin/env python3
"""Apply (or revert) InteractionGym's vLLM-Omni patch to an installed vLLM-Omni or a source checkout.

The patch (patches/vllm-omni/, see its README.md) adds what InteractionGym's ``VllmOmniDuplexAgent`` needs from
the server and is not upstream yet: input-clocked (lockstep) duplex sessions, ``silence_continuation``, the
per-unit token trace, MiniCPM-o 4.5's per-session audio feature extractor and its Thinker-only text sessions.

    # the vLLM-Omni of the Python running this script (a pip install: patches site-packages)
    <omni-env>/bin/python scripts/apply_vllm_omni_patch.py --dry-run
    <omni-env>/bin/python scripts/apply_vllm_omni_patch.py
    # another environment, or a source checkout
    python scripts/apply_vllm_omni_patch.py --python <omni-env>/bin/python
    python scripts/apply_vllm_omni_patch.py --target ~/src/vllm-omni
    # state / undo
    python scripts/apply_vllm_omni_patch.py --python <omni-env>/bin/python --status
    python scripts/apply_vllm_omni_patch.py --python <omni-env>/bin/python --revert

It finds the ``vllm_omni`` package, checks that its version (pip) or commit (git checkout) is a supported base,
dry-runs the patch (GNU ``patch``, no fuzz), backs up every file the patch touches into
``<root>/.interaction_gym_vllm_omni_patch/`` (``<root>`` = site-packages or the checkout), and applies it.
``--revert`` restores those files byte for byte (checked against the recorded SHA-256) and deletes the files the
patch added. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PATCH_DIR = Path(__file__).resolve().parents[1] / "patches" / "vllm-omni"
BACKUP_DIR = ".interaction_gym_vllm_omni_patch"

#: Supported bases: a pip version or a git commit of vllm-project/vllm-omni -> (patch file, vllm version it needs).
RC1 = ("vllm_omni-0.31.0rc1-interactiongym.patch", "0.31.0")
MAIN = ("vllm_omni-main-61cae20d-interactiongym.patch", "0.31.0")
BY_VERSION = {"0.31.0rc1": RC1}
BY_COMMIT = {
    "6dd0d1f9310f7598b773c796b434c8000c2816ec": RC1,  # tag v0.31.0rc1
    "61cae20d0f5c2b438564a892e250e08e72e5afa2": MAIN,  # main, 2026-10-09
}

PROBE = r"""
import json, os, sys
try:
    import importlib.metadata as md
except ImportError:
    md = None
import importlib.util
spec = importlib.util.find_spec("vllm_omni")
out = {"file": spec.origin if spec else None, "python": sys.executable}
for dist in ("vllm-omni", "vllm"):
    try:
        out[dist] = md.version(dist) if md else None
    except Exception:
        out[dist] = None
print(json.dumps(out))
"""


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(root: Path) -> str | None:
    """The commit of ``root`` when it is the top of a git work tree."""
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if top.returncode or Path(top.stdout.strip()).resolve() != root.resolve():
            return None
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def dist_version(root: Path) -> str | None:
    """vllm-omni's version from a dist-info next to the package (``--target`` on a site-packages directory)."""
    for d in sorted(root.glob("vllm_omni-*.dist-info")):
        meta = d / "METADATA"
        if meta.is_file():
            m = re.search(r"^Version: (.+)$", meta.read_text(errors="replace"), re.M)
            if m:
                return m.group(1).strip()
    return None


def locate(args: argparse.Namespace) -> dict:
    """Where vllm_omni is and what it is: {root, kind (pip|source), version, commit, vllm}."""
    if args.target:
        root = Path(args.target).expanduser().resolve()
        if not (root / "vllm_omni" / "__init__.py").is_file():
            die(f"{root} has no vllm_omni/ package (give a site-packages directory or a vllm-omni checkout)")
        info = {"vllm-omni": dist_version(root), "vllm": None}
    else:
        py = args.python or sys.executable
        res = subprocess.run([py, "-c", PROBE], capture_output=True, text=True)
        if res.returncode:
            die(f"could not run {py}: {res.stderr.strip()}")
        info = json.loads(res.stdout.strip().splitlines()[-1])
        if not info.get("file"):
            die(f"vllm_omni is not importable by {py} (pass --python <env>/bin/python or --target <dir>)")
        root = Path(info["file"]).resolve().parent.parent
    commit = git_head(root)
    return {"root": root, "kind": "source" if commit else "pip", "commit": commit,
            "version": info.get("vllm-omni"), "vllm": info.get("vllm")}


def choose_patch(loc: dict, args: argparse.Namespace) -> Path:
    if args.patch:
        patch = Path(args.patch).expanduser().resolve()
        if not patch.is_file():
            die(f"no such patch: {patch}")
        return patch
    base = BY_COMMIT.get(loc["commit"]) if loc["kind"] == "source" else BY_VERSION.get(loc["version"] or "")
    supported = ", ".join([f"vllm-omni=={v}" for v in BY_VERSION] + [f"commit {c[:8]}" for c in BY_COMMIT])
    if base is None:
        what = f"commit {loc['commit']}" if loc["kind"] == "source" else f"version {loc['version']}"
        msg = f"unsupported vLLM-Omni base ({loc['kind']}, {what}); supported: {supported}"
        if not args.force:
            die(msg + ". --force tries the patch for vllm-omni 0.31.0rc1 anyway (the dry run still has to pass), "
                "or pass --patch <file>")
        print(f"warning: {msg}; --force: trying {RC1[0]}", file=sys.stderr)
        base = RC1
    name, vllm_needed = base
    if loc["vllm"] and loc["vllm"].split("+")[0] != vllm_needed:
        print(f"warning: vllm {loc['vllm']} is installed; this base was tested with vllm=={vllm_needed}",
              file=sys.stderr)
    return PATCH_DIR / name


def touched_files(patch: Path) -> list[tuple[str, bool]]:
    """(path, created) of every file in a unified diff (-p1 paths)."""
    out, old = [], None
    for line in patch.read_text().splitlines():
        if line.startswith("--- "):
            old = line[4:].split("\t")[0]
        elif line.startswith("+++ ") and old is not None:
            new = line[4:].split("\t")[0]
            created = old == "/dev/null"
            path = (new if new != "/dev/null" else old).split("/", 1)[1]
            out.append((path, created))
            old = None
    seen, uniq = set(), []
    for p, c in out:
        if p not in seen:
            seen.add(p)
            uniq.append((p, c))
    return uniq


def gnu_patch() -> str:
    """GNU patch (``gpatch`` on macOS with Homebrew): BSD patch reports a failed reverse dry run as success."""
    for name in ("gpatch", "patch"):
        exe = shutil.which(name)
        if exe and "GNU patch" in subprocess.run([exe, "--version"], capture_output=True, text=True).stdout:
            return exe
    die("GNU patch is needed (Linux: apt-get install patch; macOS: brew install gpatch)")
    return ""


def run_patch(root: Path, patch: Path, *, reverse: bool = False, dry: bool = False) -> subprocess.CompletedProcess:
    exe = gnu_patch()
    # --forward in both directions: without it, --batch takes a patch that looks reversed as reversed and applies it
    # the other way, so a reverse dry run would "succeed" on an unpatched tree (and a forward one on a patched tree).
    cmd = [exe, "-p1", "--batch", "--forward", "--fuzz=0", "--no-backup-if-mismatch", "-d", str(root), "-i", str(patch)]
    if reverse:
        cmd.append("--reverse")
    if dry:
        cmd.append("--dry-run")
    return subprocess.run(cmd, capture_output=True, text=True)


def state(root: Path, patch: Path) -> str:
    """applied | not-applied | conflict."""
    if run_patch(root, patch, reverse=True, dry=True).returncode == 0:
        return "applied"
    if run_patch(root, patch, dry=True).returncode == 0:
        return "not-applied"
    return "conflict"


def apply(root: Path, patch: Path, dry_only: bool) -> None:
    backup = root / BACKUP_DIR
    st = state(root, patch)
    if st == "applied":
        print(f"already applied: {patch.name} in {root}")
        return
    if st == "conflict":
        res = run_patch(root, patch, dry=True)
        die(f"{patch.name} does not apply cleanly to {root}:\n{res.stdout}{res.stderr}")
    res = run_patch(root, patch, dry=True)
    print(f"dry run OK: {patch.name} -> {root}")
    if dry_only:
        print(res.stdout.rstrip())
        return
    if backup.exists():
        die(f"{backup} exists from an earlier apply; run --revert first (or remove it if those files were reinstalled)")
    files = touched_files(patch)
    manifest = {"patch": patch.name, "patch_sha256": sha256(patch), "root": str(root), "files": {}}
    for rel, created in files:
        src = root / rel
        entry = {"created": not src.exists()}
        if src.exists():
            dst = backup / "files" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            entry["sha256_before"] = sha256(src)
        manifest["files"][rel] = entry
    res = run_patch(root, patch)
    if res.returncode:
        # GNU patch checks every hunk before writing in --dry-run, so this is unexpected: put the originals back.
        restore(root, manifest, strict=False)
        die(f"patch failed after a clean dry run; files restored:\n{res.stdout}{res.stderr}")
    for rel, entry in manifest["files"].items():
        entry["sha256_after"] = sha256(root / rel) if (root / rel).exists() else None
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"applied {patch.name}: {len(files)} files; backup in {backup}")
    print("check: python scripts/apply_vllm_omni_patch.py --status  (with the same --python / --target)")


def restore(root: Path, manifest: dict, *, strict: bool) -> list[str]:
    backup = root / BACKUP_DIR
    problems = []
    for rel, entry in manifest["files"].items():
        path = root / rel
        after = entry.get("sha256_after")
        if strict and after and path.exists() and sha256(path) != after:
            problems.append(f"{rel} changed since the patch was applied")
        if entry["created"]:
            if path.exists():
                path.unlink()
            pyc = path.parent / "__pycache__"
            for f in pyc.glob(path.stem + ".*.pyc") if pyc.is_dir() else ():
                f.unlink()
        else:
            shutil.copy2(backup / "files" / rel, path)
            if sha256(path) != entry["sha256_before"]:
                problems.append(f"{rel} restored but its SHA-256 differs from the original")
    return problems


def revert(root: Path, force: bool) -> None:
    backup = root / BACKUP_DIR
    mf = backup / "manifest.json"
    if not mf.is_file():
        die(f"no backup in {backup}: nothing to revert (was the patch applied with this script?)")
    manifest = json.loads(mf.read_text())
    changed = [rel for rel, e in manifest["files"].items()
               if e.get("sha256_after") and (root / rel).exists() and sha256(root / rel) != e["sha256_after"]]
    if changed and not force:
        die("files changed since the patch was applied (a reinstall?): " + ", ".join(changed[:5])
            + (" ..." if len(changed) > 5 else "") + "; --force restores the backed-up originals anyway")
    problems = [p for p in restore(root, manifest, strict=False) if "differs" in p]
    if problems:
        die("; ".join(problems))
    shutil.rmtree(backup)
    print(f"reverted {manifest['patch']}: {len(manifest['files'])} files restored byte for byte in {root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--python", help="Python of the vLLM-Omni environment (default: the one running this script)")
    where.add_argument("--target", help="a site-packages directory or a vllm-omni source checkout")
    ap.add_argument("--patch", help="patch file to use instead of the one chosen for the detected base")
    act = ap.add_mutually_exclusive_group()
    act.add_argument("--dry-run", action="store_true", help="only check that the patch applies")
    act.add_argument("--revert", action="store_true", help="restore the backed-up files and remove added ones")
    act.add_argument("--status", action="store_true", help="report whether the patch is applied")
    ap.add_argument("--force", action="store_true", help="unsupported base / changed files: go on anyway")
    args = ap.parse_args()

    loc = locate(args)
    print(f"vllm_omni: {loc['root'] / 'vllm_omni'} ({loc['kind']}; "
          + (f"commit {loc['commit'][:12]}" if loc["commit"] else f"version {loc['version']}")
          + (f"; vllm {loc['vllm']}" if loc["vllm"] else "") + ")")
    if args.revert:
        revert(loc["root"], args.force)
        return
    mf = loc["root"] / BACKUP_DIR / "manifest.json"
    if args.status and mf.is_file() and not args.patch:
        patch = PATCH_DIR / json.loads(mf.read_text())["patch"]
    else:
        patch = choose_patch(loc, args)
    if args.status:
        st = state(loc["root"], patch)
        print(f"{patch.name}: {st}" + ("" if mf.is_file() or st != "applied" else " (not by this script: no backup)"))
        sys.exit(0 if st == "applied" else 1)
    apply(loc["root"], patch, args.dry_run)


if __name__ == "__main__":
    main()
