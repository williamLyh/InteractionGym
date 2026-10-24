"""``ig``: run a composition, inspect the plugins.

    ig run minimal                                   # a built-in bundle
    ig run fdb3 --patch my.yml --set env.max_ms=60000 --out runs/x --seeds 0,1
    ig run my_bundle.yml --dump-config               # print the complete configuration and stop
    ig plugins [kind]                                # what can go in a row, with each plugin's config fields
    ig bundles                                       # the built-in bundles
"""

from __future__ import annotations

import argparse
import asyncio
import sys

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
