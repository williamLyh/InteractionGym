"""Composition: the rows that make up a run, layered from bundles, patches and ``--set``, then resolved.

A composition is an ordered set of **rows** ``{id, plugin, config, disabled, requires}``. Layers apply in order,
each a YAML list of operations:

    - extends: core                       # bundles only: apply that bundle first
    - insert:                             # add rows (an existing id is an error: patch it instead)
        - {id: user, plugin: role/replay, config: {stream: user.speech}}
    - id: env                             # patch a row: `config` replaces the row's WHOLE config (no merging),
      config: {max_ms: 60000}             # `disabled` / `requires` / `plugin` replace that field
    - id: viewer
      disabled: true
    - remove: judge                       # drop a row

``--set row.key.sub=value`` is the one merging operation: it sets a single key inside a row's config (the value is
parsed as YAML, so ``--set env.max_ms=60000`` is an int and ``--set agent.replies='[a, b]'`` a list).

``resolve()`` checks every enabled row against the registry (plugin exists, config valid, requirements met),
orders rows so that requirements come first, and returns a ``Resolved`` composition whose ``dump()`` is the complete
configuration the run was built from — it goes into the trajectory's ``meta.config`` with its ``digest()``.
Whether a row's requirement is satisfied is decided by row ids and service keys (what a row's plugin ``provides``),
never by row order.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any

import yaml

from .registry import REGISTRY, Plugin, PluginError, Registry, config_refs, parse_config

BUNDLES = Path(__file__).resolve().parent.parent / "bundles"


@dataclass
class Row:
    id: str
    plugin: str
    config: dict = field(default_factory=dict)
    disabled: bool = False
    requires: tuple[str, ...] = ()
    source: str = ""  # the layer that inserted it (for messages)

    def as_dict(self) -> dict:
        d: dict[str, Any] = {"id": self.id, "plugin": self.plugin}
        if self.config:
            d["config"] = self.config
        if self.disabled:
            d["disabled"] = True
        if self.requires:
            d["requires"] = list(self.requires)
        return d


def _ops(text_or_ops, source: str) -> list:
    ops = yaml.safe_load(text_or_ops) if isinstance(text_or_ops, str) else text_or_ops
    if ops is None:
        return []
    if not isinstance(ops, list):
        raise PluginError(f"{source}: a layer is a YAML list of operations, got {type(ops).__name__}")
    return ops


class Composition:
    def __init__(self):
        self.rows: dict[str, Row] = {}
        self.layers: list[str] = []

    # ---------------------------------------------------------------- building

    @classmethod
    def bundle(cls, name: str, search: tuple[Path, ...] = (BUNDLES,)) -> "Composition":
        """A composition from a bundle: a built-in name (``core``, ``minimal``, ...) or a path to a YAML file."""
        c = cls()
        c.extend(name, search)
        return c

    def extend(self, name: str, search: tuple[Path, ...] = (BUNDLES,), _seen: tuple[str, ...] = ()) -> None:
        path = Path(name)
        if not path.suffix or not path.exists():
            found = [d / f"{name}.yml" for d in search if (d / f"{name}.yml").exists()]
            if not found:
                known = sorted(p.stem for d in search for p in d.glob("*.yml"))
                raise PluginError(f"no bundle {name!r}; built-in bundles: {', '.join(known)}")
            path = found[0]
        key = str(path.resolve())
        if key in _seen:
            raise PluginError(f"bundle {name} extends itself: {' -> '.join(_seen + (key,))}")
        ops = _ops(path.read_text(), str(path))
        if ops and isinstance(ops[0], dict) and "extends" in ops[0]:
            base = ops.pop(0)["extends"]
            self.extend(base, search + (path.parent,), _seen + (key,))
        self.apply(ops, path.stem if path.parent in search else str(path))

    def apply(self, ops, source: str) -> None:
        """Apply one layer of operations (a YAML string, or the parsed list)."""
        self.layers.append(source)
        for op in _ops(ops, source):
            if not isinstance(op, dict):
                raise PluginError(f"{source}: each operation is a mapping, got {op!r}")
            if "extends" in op:
                raise PluginError(f"{source}: 'extends' is allowed only as the first operation of a bundle")
            if "insert" in op:
                for r in op["insert"] or []:
                    self._insert(r, source)
            elif "remove" in op:
                if op["remove"] not in self.rows:
                    raise PluginError(f"{source}: cannot remove unknown row {op['remove']!r}")
                del self.rows[op["remove"]]
            elif "id" in op:
                self._patch(op, source)
            else:
                raise PluginError(f"{source}: an operation is insert / remove / a row patch (with id), got {sorted(op)}")

    def _insert(self, r: dict, source: str) -> None:
        if not isinstance(r, dict) or "id" not in r or "plugin" not in r:
            raise PluginError(f"{source}: an inserted row needs id and plugin, got {r!r}")
        if r["id"] in self.rows:
            raise PluginError(f"{source}: row {r['id']!r} already exists (from {self.rows[r['id']].source}); patch it by id instead of inserting")
        extra = set(r) - {"id", "plugin", "config", "disabled", "requires"}
        if extra:
            raise PluginError(f"{source}: row {r['id']!r} has unknown fields {sorted(extra)}")
        self.rows[r["id"]] = Row(r["id"], r["plugin"], dict(r.get("config") or {}), bool(r.get("disabled", False)),
                                 tuple(r.get("requires") or ()), source)

    def _patch(self, op: dict, source: str) -> None:
        row = self.rows.get(op["id"])
        if row is None:
            raise PluginError(f"{source}: cannot patch unknown row {op['id']!r}; rows: {', '.join(self.rows) or '(none)'}")
        extra = set(op) - {"id", "plugin", "config", "disabled", "requires"}
        if extra:
            raise PluginError(f"{source}: patch of {op['id']!r} has unknown fields {sorted(extra)}")
        if "plugin" in op:
            row.plugin = op["plugin"]
        if "config" in op:
            row.config = dict(op["config"] or {})
        if "disabled" in op:
            row.disabled = bool(op["disabled"])
        if "requires" in op:
            row.requires = tuple(op["requires"] or ())

    def set(self, assignment: str) -> None:
        """``row.key[.sub]=value``: set one key inside a row's config (value parsed as YAML)."""
        if "=" not in assignment:
            raise PluginError(f"--set takes row.key=value, got {assignment!r}")
        path, text = assignment.split("=", 1)
        parts = path.split(".")
        if len(parts) < 2 or not all(parts):
            raise PluginError(f"--set takes row.key=value, got {assignment!r}")
        row = self.rows.get(parts[0])
        if row is None:
            raise PluginError(f"--set: unknown row {parts[0]!r}; rows: {', '.join(self.rows) or '(none)'}")
        if parts[1] == "disabled" and len(parts) == 2:
            row.disabled = bool(yaml.safe_load(text))
        else:
            d = row.config
            for k in parts[1:-1]:
                d = d.setdefault(k, {})
                if not isinstance(d, dict):
                    raise PluginError(f"--set: {'.'.join(parts[:-1])} is not a mapping")
            d[parts[-1]] = yaml.safe_load(text)
        self.layers.append(f"--set {assignment}")

    # ---------------------------------------------------------------- resolving

    def resolve(self, registry: Registry = REGISTRY) -> "Resolved":
        rows = [copy.deepcopy(r) for r in self.rows.values() if not r.disabled]
        plugins: dict[str, Plugin] = {}
        configs: dict[str, Any] = {}
        provides: dict[str, str] = {}  # service key -> row id
        for r in rows:
            try:
                plugins[r.id] = p = registry.get(r.plugin)
            except PluginError as e:
                raise PluginError(f"row {r.id!r} ({r.source}): {e}") from None
            configs[r.id] = parse_config(p, r.config, f"row {r.id!r} ({r.source})")
            provides[r.id] = r.id
            provides[p.key] = r.id
            if p.provides:
                if p.provides in provides and provides[p.provides] != r.id:
                    raise PluginError(f"rows {provides[p.provides]!r} and {r.id!r} both provide {p.provides!r}; disable one")
                provides[p.provides] = r.id
        deps: dict[str, set[str]] = {}
        for r in rows:
            p = plugins[r.id]
            needs = [(need, None) for need in tuple(p.requires) + tuple(r.requires)]
            for path, value in config_refs(p, r.config):
                if value in p.refs[path]:
                    continue
                if value not in provides:
                    allowed = f" or one of {', '.join(p.refs[path])}" if p.refs[path] else ""
                    raise PluginError(f"row {r.id!r}: {path} names {value!r}, which is not the id of an enabled row{allowed}")
                needs.append((value, path))
            deps[r.id] = set()
            for need, path in needs:
                if need not in provides:
                    raise PluginError(f"row {r.id!r} requires {need!r}: no enabled row has that id or provides that service")
                if provides[need] != r.id:
                    deps[r.id].add(provides[need])
        try:
            order = list(TopologicalSorter(deps).static_order())
        except CycleError as e:
            raise PluginError(f"rows require each other in a cycle: {e.args[1]}") from None
        by_id = {r.id: r for r in rows}
        # the dependency order decides only what must come first; among independent rows the layer order is kept
        rank = {rid: i for i, rid in enumerate(by_id)}
        ordered = sorted(by_id.values(), key=lambda r: (_depth(r.id, deps), rank[r.id]))
        assert [r.id for r in ordered] and set(order) == set(by_id)
        return Resolved(ordered, plugins, configs, provides, tuple(self.layers))


def _depth(rid: str, deps: dict[str, set[str]], _seen=()) -> int:
    return 0 if not deps[rid] else 1 + max(_depth(d, deps, _seen + (rid,)) for d in deps[rid])


@dataclass
class Resolved:
    rows: list[Row]  # enabled rows, requirements first
    plugins: dict[str, Plugin]
    configs: dict[str, Any]
    provides: dict[str, str]  # service key / row id / plugin key -> row id
    layers: tuple[str, ...] = ()

    def dump(self) -> str:
        """The complete configuration as YAML (one insert of the resolved rows): what ``--dump-config`` prints and
        what a run records."""
        return yaml.safe_dump([{"insert": [r.as_dict() for r in self.rows]}], sort_keys=False, allow_unicode=True, width=120)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps([r.as_dict() for r in self.rows], sort_keys=True, default=str).encode()).hexdigest()[:12]

    def meta(self) -> dict:
        """What the trajectory records: the rows, their digest and the layers they came from."""
        return {"digest": self.digest(), "layers": list(self.layers), "rows": [r.as_dict() for r in self.rows]}
