"""``ig``: run a composition, inspect the plugins.

    ig run minimal                                   # a built-in bundle
    ig run fdb3 --patch my.yml --set env.max_ms=60000 --out runs/x --seeds 0,1
    ig run my_bundle.yml --dump-config               # print the complete configuration and stop
    ig prepare fdb2 --out runs/fdb2_ab --seeds 0     # a benchmark's preparation stage (reference scripts)
    ig report fdb2 --out runs/fdb2_ab                # the run's post-hoc report (judge, tables) on those files
    ig plugins [kind]                                # what can go in a row, with each plugin's config fields
    ig bundles                                       # the built-in bundles
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .compose import compose
from .config import BUNDLES, Composition
from .registry import REGISTRY, KINDS, PluginError, config_fields


def _composition(args) -> Composition:
    comp = Composition.bundle(args.bundle)
    for p in args.patch or ():
        comp.apply(open(p).read(), p)
    for s in args.set or ():
        comp.set(s)
    return comp


def cmd_run(args) -> int:
    resolved = _composition(args).resolve()
    if args.dump_config:
        print(resolved.dump(), end="")
        print(f"# digest {resolved.digest()}; layers: {' > '.join(resolved.layers)}")
        return 0
    run = compose(resolved)
    if run.driver is None:
        raise PluginError("the composition has no driver row (kind driver), e.g. {id: driver, plugin: driver/eval}")
    seeds = [int(s) for s in args.seeds.split(",")]
    asyncio.run(run.driver.run(run, out=args.out, seeds=seeds, limit=args.limit, ids=args.ids.split(",") if args.ids else (), tag=args.tag))
    return 0


def cmd_prepare(args) -> int:
    """A task source's preparation stage (reference scripts, caches) for the tasks and seeds a run would use."""
    from .builtin.drivers import select_tasks

    run = compose(_composition(args).resolve())
    sources = [s for s in run.task_sources if callable(getattr(s, "build", None))]
    if not sources:
        raise PluginError("nothing to prepare: no task row of this composition has a preparation stage")
    seeds = [int(s) for s in args.seeds.split(",")]
    stride = bool(getattr(getattr(run.driver, "cfg", None), "limit_stride", False))
    tasks = select_tasks(run.tasks, args.ids.split(",") if args.ids else (), args.limit, stride)
    for src in sources:
        if hasattr(src, "out"):
            src.out = Path(args.out)
        print(f"ig: preparing {len(tasks)} tasks × {len(seeds)} seeds -> {args.out}", flush=True)
        n = asyncio.run(src.build(tasks, seeds, concurrency=args.concurrency))
        print(f"ig: {n} prepared", flush=True)
    return 0


def cmd_report(args) -> int:
    """The run's report rows (post-hoc judges and tables) on the files under --out."""
    run = compose(_composition(args).resolve())
    if not run.reports:
        raise PluginError("nothing to report: no report row in this composition (e.g. {id: report, plugin: report/fdb2, config: {judge: judge_llm}})")
    tag = args.tag or ((run.agent_label or "") if getattr(getattr(run.driver, "cfg", None), "arms", None) else "")
    for rid, rep in run.reports.items():
        print(f"ig: report {rid} on {args.out}" + (f" (tag {tag})" if tag else ""), flush=True)
        asyncio.run(rep.run(Path(args.out), tag))
    return 0


def cmd_plugins(args) -> int:
    for p in REGISTRY.all(args.kind):
        print(f"{p.key}" + (f"  (provides {p.provides})" if p.provides else "") + (f"  requires {', '.join(p.requires)}" if p.requires else ""))
        if p.doc:
            print(f"    {p.doc.splitlines()[0]}")
        for f in config_fields(p):
            print(f"      {f['name']}: {f['type']} = {f['default']!r}")
        if p.config is None:
            print("      (keyword arguments of the wrapped class)")
    return 0


def cmd_bundles(args) -> int:
    for p in sorted(BUNDLES.glob("*.yml")):
        first = next((line.lstrip("# ").rstrip() for line in p.read_text().splitlines() if line.startswith("#")), "")
        print(f"{p.stem:12s} {first}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ig", description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a bundle")
    r.add_argument("bundle", help="a built-in bundle name or a YAML file")
    r.add_argument("--patch", action="append", help="a patch layer (YAML); repeatable, applied in order")
    r.add_argument("--set", action="append", help="row.key=value (YAML value); repeatable")
    r.add_argument("--out", default="runs/ig")
    r.add_argument("--seeds", default="0")
    r.add_argument("--limit", type=int, default=0, help="only the first N tasks")
    r.add_argument("--ids", default="", help="only these task ids (comma-separated)")
    r.add_argument("--tag", default="", help="prefix of the episode ids")
    r.add_argument("--dump-config", action="store_true", help="print the resolved configuration and exit")
    r.set_defaults(fn=cmd_run)
    pr = sub.add_parser("prepare", help="a benchmark's preparation stage (e.g. FD-Bench v2 reference scripts) for the run's tasks and seeds")
    for a in ("bundle", "--patch", "--set", "--out", "--seeds", "--limit", "--ids"):
        act = next(x for x in r._actions if a in x.option_strings or x.dest == a)
        pr._add_action(act)
    pr.add_argument("--concurrency", type=int, default=4)
    pr.set_defaults(fn=cmd_prepare)
    rp = sub.add_parser("report", help="the run's post-hoc report (judges, tables, pages) on the files under --out")
    for a in ("bundle", "--patch", "--set", "--out", "--tag"):
        act = next(x for x in r._actions if a in x.option_strings or x.dest == a)
        rp._add_action(act)
    rp.set_defaults(fn=cmd_report)
    p = sub.add_parser("plugins", help="list the plugins")
    p.add_argument("kind", nargs="?", choices=KINDS)
    p.set_defaults(fn=cmd_plugins)
    b = sub.add_parser("bundles", help="list the built-in bundles")
    b.set_defaults(fn=cmd_bundles)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except PluginError as e:
        print(f"ig: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
