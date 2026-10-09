"""Self-contained HTML viewer for schema-v1 episodes (see docs/FORMAT.md)."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

from .commentary import headlines, units, view_data  # noqa: F401  (re-exported)

TEMPLATE = Path(__file__).with_name("template.html")
AGENT_TEMPLATE = Path(__file__).with_name("agent_trace.html")


def _back(href: str | None, label: str) -> str:
    return f'<a class="back" href="{html.escape(href)}">← {html.escape(label)}</a>' if href else ""


def _json(x) -> str:
    return json.dumps(x, ensure_ascii=False).replace("</", "<\\/")


def render_html(episodes: list[dict], media_base: str = "", back: str | None = None, traces: list[dict] | None = None,
                agent_pages: dict[str, str] | None = None) -> str:
    """``back``: a link (e.g. to an index of runs) shown at the top of the sidebar. ``traces``: agent
    trace records (docs/AGENT_TRACE.md) whose per-unit decisions are drawn under the agent lane;
    ``agent_pages``: ``episode_id`` -> link to that episode's agent trace page."""
    view = _json(view_data(episodes, traces, agent_pages))
    return (TEMPLATE.read_text().replace("__BACK_LINK__", _back(back, "all runs"))
            .replace("__VIEW__", view).replace("__DATA__", _json(episodes)).replace("__MEDIA_BASE__", media_base))


def export_html(episodes: list[dict], path: str | Path, back: str | None = None, traces: list[dict] | None = None,
                agent_pages: dict[str, str] | None = None) -> Path:
    """Write the viewer page. Audio plays from the episodes' ``meta.media_root``, referenced
    relative to the page so the folder can be moved as a whole."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    roots = {ep["meta"].get("media_root") for ep in episodes} - {None}
    base = ""
    if roots:
        base = os.path.relpath(Path(roots.pop()).resolve(), path.parent.resolve()).replace(os.sep, "/") + "/"
    path.write_text(render_html(episodes, base, back, traces, agent_pages))
    return path


def export_agent_trace_html(episode: dict, trace: dict, path: str | Path, back: str | None = None, env_page: str | None = None) -> Path:
    """A separate page for an agent server's trace (one record of ``agent_traces.jsonl``, docs/AGENT_TRACE.md):
    the token sequences its model consumed and produced per unit, on the episode's timeline. The agent
    server is not part of the environment, so this is kept out of the environment viewer and the trajectory;
    it is for monitoring and debugging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"episode": episode, "trace": trace}, ensure_ascii=False).replace("</", "<\\/")
    links = " ".join(x for x in (_back(back, "all runs"), _back(env_page, "environment view")) if x)
    path.write_text(AGENT_TEMPLATE.read_text().replace("__BACK_LINK__", links).replace("__DATA__", data))
    return path


def export_run(episodes: list[dict], out_dir: str | Path, notes: dict[str, str] | None = None, title: str = "Run",
               traces: list[dict] | None = None) -> Path:
    """A browsable run: ``index.html`` (one row per episode), ``env.html`` (all episodes, with a link back to
    the index) and, for episodes with an agent trace (``traces``, records of ``agent_traces.jsonl``),
    ``<episode>.agent.html``. Episodes are listed in order; ``notes`` adds a description per ``episode_id``.
    Returns the index path."""
    by_id = {t["episode_id"]: t for t in traces or [] if t}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, pages = [], {}
    for ep in episodes:
        eid = ep["meta"]["episode_id"]
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in eid)
        page = None
        if eid in by_id:
            page = export_agent_trace_html(ep, by_id[eid], out / f"{safe}.agent.html", back="index.html", env_page=f"env.html#{eid}").name
            pages[eid] = page
        a, u = ep["meta"].get("agent", {}), ep["meta"].get("user", {})
        who = ", ".join(str(x) for x in (u.get("profile", {}).get("name"), u.get("profile", {}).get("gender"), u.get("profile", {}).get("age")) if x)
        cells = [f'<a href="env.html#{html.escape(eid)}">{html.escape(eid)}</a>', f'{ep["meta"]["duration_ms"] / 1000:.1f}s',
                 str(len(ep["turns"])), str(len(ep["eval"]["duplex"])), html.escape(str(a.get("clock", "—"))),
                 html.escape(str(a.get("output", "audio"))), html.escape(who or "—"),
                 f'<a href="{page}">tokens</a>' if page else "—", html.escape((notes or {}).get(eid, ""))]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    export_html(episodes, out / "env.html", back="index.html", traces=list(by_id.values()), agent_pages=pages)
    head = "".join(f"<th>{h}</th>" for h in ("episode", "duration", "turns", "duplex events", "clock", "output", "user", "agent trace", "notes"))
    (out / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e3e0;--link:#2f6fd6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151517;--panel:#1e1e21;--ink:#ececef;--muted:#9a9aa2;--line:#333338;--link:#6ea0ff}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1300px;margin:0 auto;padding:20px 16px}} h1{{font-size:18px;margin:0 0 6px}} a{{color:var(--link)}} p{{color:var(--muted);margin:0 0 12px}}
.wrap{{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px}}
table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
td:last-child{{white-space:normal;color:var(--muted);min-width:240px}} th{{color:var(--muted);font-weight:500}}
</style></head><body><main><h1>{html.escape(title)}</h1><p>{len(episodes)} episodes · click a name for the environment view (all episodes are in its sidebar), "tokens" for the agent server trace.</p>
<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></main></body></html>""")
    return out / "index.html"
