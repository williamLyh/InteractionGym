#!/usr/bin/env python3
"""Build the GitHub Pages site in ``demo/``: the demo page plus a browsable run of example episodes.

    python scripts/build_site.py --examples runs/suite_v2          # copy a run into demo/examples/
    python scripts/build_site.py                                   # only refresh the tab bar on existing pages

The site has two tabs, shown as one bar at the top of every page:

- ``demo/index.html``: the interactive demo (three recorded episodes, written by hand).
- ``demo/examples/``: a run written by ``viewer.export_run`` (``index.html``, ``env.html``, ``*.agent.html``,
  ``media/``), e.g. from ``examples/minicpmo_suite.py``.

With ``--examples`` the run's pages are copied, its WAV media are transcoded to MP3 (mono, ``--bitrate``) so
the repository stays small, and absolute sound-bank paths are shortened to ``soundbank/...``. Needs ffmpeg.
Standard library only. ``.github/workflows/pages.yml`` publishes ``demo/`` on push.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "demo"
REPO_URL = "https://github.com/williamLyh/InteractionGym"
TABS = [("demo", "Demo", "index.html"), ("examples", "Example episodes", "examples/index.html")]
BAR_H = 38  # px

START, END = "<!-- site-nav -->", "<!-- /site-nav -->"


def nav(active: str, root: str) -> str:
    links = "".join(
        f'<a href="{root}{href}"{" aria-current=\"page\"" if key == active else ""}>{label}</a>' for key, label, href in TABS
    )
    return f"""{START}
<style>
.site-nav {{ --sn-bg: #ffffff; --sn-ink: #1d1d1f; --sn-muted: #6b6b70; --sn-line: #e3e3e0; --sn-on: #2f6fd6;
  display: flex; align-items: center; gap: 4px; height: {BAR_H}px; box-sizing: border-box; padding: 0 16px;
  background: var(--sn-bg); border-bottom: 1px solid var(--sn-line); overflow-x: auto; white-space: nowrap;
  font: 500 13px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) .site-nav {{
  --sn-bg: #1e1e21; --sn-ink: #ececef; --sn-muted: #9a9aa2; --sn-line: #333338; --sn-on: #6ea0ff; }} }}
:root[data-theme="dark"] .site-nav {{ --sn-bg: #1e1e21; --sn-ink: #ececef; --sn-muted: #9a9aa2; --sn-line: #333338; --sn-on: #6ea0ff; }}
.site-nav b {{ color: var(--sn-ink); margin-right: 12px; font-weight: 600; }}
.site-nav a {{ color: var(--sn-muted); text-decoration: none; padding: 11px 10px; border-bottom: 2px solid transparent; }}
.site-nav a:hover {{ color: var(--sn-ink); }}
.site-nav a[aria-current] {{ color: var(--sn-ink); border-bottom-color: var(--sn-on); }}
.site-nav .gh {{ margin-left: auto; }}
@media (min-width: 761px) {{ .app {{ height: calc(100vh - {BAR_H}px); }} }}  /* the trajectory viewer fills the window below the bar */
</style>
<nav class="site-nav"><b>InteractionGym</b>{links}<a class="gh" href="{REPO_URL}">GitHub</a></nav>
{END}
"""


def put_nav(page: Path, active: str, root: str) -> None:
    s = page.read_text()
    s = re.sub(r"\n?" + re.escape(START) + r".*?" + re.escape(END) + r"\n?", "", s, flags=re.S)
    bar = nav(active, root)
    m = re.search(r"<body[^>]*>\n?", s)
    if m:
        s = s[: m.end()] + bar + s[m.end():]
    else:  # the demo page has no <body> tag: the bar goes before its first element after <style>
        i = s.index("</style>") + len("</style>")
        s = s[:i] + "\n" + bar + s[i:]
    page.write_text(s)


def copy_examples(run: Path, bitrate: str) -> None:
    out = SITE / "examples"
    if out.exists():
        shutil.rmtree(out)
    (out / "media").mkdir(parents=True)
    for page in run.glob("*.html"):
        s = page.read_text()
        s = re.sub(r"(media/[0-9a-f]{2}/[0-9a-f]{64})\.wav", r"\1.mp3", s)
        s = re.sub(r"/[^\"'\s<>]*/((?:ambience|events)/[^\"'\s<>]+)", r"soundbank/\1", s)
        (out / page.name).write_text(s)
    wavs = sorted((run / "media").rglob("*.wav"))
    for wav in wavs:
        mp3 = out / "media" / wav.relative_to(run / "media").with_suffix(".mp3")
        mp3.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-ac", "1", "-b:a", bitrate, str(mp3)], check=True)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"examples: {len(list(out.glob('*.html')))} pages, {len(wavs)} audio files, {size / 1e6:.1f} MB -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=Path, help="a run directory written by viewer.export_run")
    ap.add_argument("--bitrate", default="48k", help="MP3 bitrate for the examples' audio")
    args = ap.parse_args()
    if args.examples:
        copy_examples(args.examples, args.bitrate)
    put_nav(SITE / "index.html", "demo", "")
    for page in sorted((SITE / "examples").glob("*.html")):
        put_nav(page, "examples", "../")


if __name__ == "__main__":
    main()
