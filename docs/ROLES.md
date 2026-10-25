# Roles: any participant can be the one under evaluation

Status: step 3 of [ARCHITECTURE.md](ARCHITECTURE.md) §8, 2026-10-25. The role table and the controlled role are
recorded and used; per-pair standards, several controlled roles and role-named tool callers are deferred (§6).

## 1. What a role is

A role is a participant of an episode: it **speaks on one stream** (`caller.speech`), **perceives** other streams
(`assistant.speech`, `world.screen`), has an identity (persona, goal, SOP, hidden preferences), a voice and a
*policy* — simulated (a source and turn-taking rules), replayed (recorded turns at fixed times), or **controlled**
(an external policy stepped through `Env`). The kernel knows none of this: it sees nodes and streams. The role is a
composition-level notion (`role/*` plugins) and a trajectory-level one (`meta.roles`).

`user` and `agent` are the two roles of the core bundle, not kernel concepts. Nothing in the kernel, the trajectory
or the scores depends on those names.

## 2. Names

The trajectory names a role after the stream it speaks on: `turns[].role` is the stream's first segment
(`caller.speech` → `caller`), with one legacy mapping, `policy.*` → `agent` (the controlled role's default stream,
`AgentSpec.out = "policy.speech"`). `role/controlled` checks that its `name` and `out` agree (`name: assistant` with
`out: assistant.speech`; `out` defaults to `<name>.speech`, `policy.speech` for `agent`).

## 3. What the trajectory records

| Field | Meaning |
|---|---|
| `meta.roles` | every role by name: `{stream, perceives: [streams], controlled: true}` for the external policy's role, `{stream, perceives, component}` for a simulated or replayed one (the node that plays it). Nodes that do not speak (tools, soundscapes) are components (`meta.env.components`), not roles |
| `meta.controlled` | the controlled role's name: the role the rule scores judge |
| `turns[].role` | the role name (§2) |

Files written before this (no `meta.controlled`) read as `controlled = "agent"`; every reader uses that default
(`interaction.controlled(ep)`), so nothing changes for them.

## 4. Evaluation is relative to the controlled role

The rule scores (`eval.scores`, FORMAT.md §6.3) are **one expectation per turn of every other role, judged on the
controlled role's reaction**: `scores(turns, agent=meta.controlled)`. The same for `eval.duplex`, the timing metrics,
`interaction.metrics`, `turn_validity`, the SOP judge's transcript and the viewer's headlines and lanes: "agent" means
the controlled role, "user" means every other role. So:

- `controlled: agent` (today's runs): unchanged.
- `controlled: user` with a simulated `agent` role (a replay or an LLM assistant): the scores judge how the user model
  reacts to the assistant's turns — the user-side evaluation of DESIGN.md §5.1, with the same code.
- three roles (`caller`, `assistant` controlled, `business` simulated): the assistant is judged on its reactions to
  both other roles' turns; `turns[].to` can say whom a line addresses (FORMAT.md §4.3).

What the expectation *is* still comes from the producer's labels (`kind`, `expects`) and the fixed table of §6.3; the
**standards** (latency thresholds, overlap tolerance) are the assistant standard for every controlled role. A user
held to the assistant standard will be penalised for overlapping — see §6.

## 5. Composition

```yaml
- insert:
    - {id: controlled, plugin: role/controlled, config: {name: assistant, obs: [caller.speech], chunk_ms: 200}}
    - {id: caller, plugin: role/user, config: {source: caller_lines, stream: caller.speech, listens_to: assistant.speech}}
```

Swapping sides is a configuration change: `role/controlled` with `name: user`, `out: user.speech`, `obs:
[policy.speech]` and a `role/replay` (or `role/user`) speaking on `policy.speech` makes the external policy the
caller and the simulated node the assistant; the trajectory then says `controlled: user`.

## 6. Deferred

- **Per-pair standards.** One expectation table keyed by (speaker role, listener role): a caller may overlap the
  assistant, the assistant may not overlap the caller (`eval.reply_score`'s early floor). Comes with the behaviour
  plugins (ARCHITECTURE.md step 4), which bring each behaviour's own `expects`.
- **Several controlled roles** (self-play): waits for real-human data to anchor them (DESIGN.md §5.1).
- **Tool callers by role.** Tool calls are keyed by caller class (`agent` → `policy.tool_call`, `user` →
  `user.tool_call`; `tools.CALL`), not by role name: a third speaking role has no tool stream yet.
- **The session for non-controlled roles.** `Env` sends the `session` (instructions, tools) to the controlled role
  only; simulated roles take their identity from the task (`persona`, `instructions`).
- **The viewer** draws two main lanes (the controlled role and the first other role) and puts further roles in
  `other` lanes; a per-role lane layout and colours come with the three-party tasks.
