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

What the expectation *is* comes from the producer's labels (`kind`, `expects`) and the fixed table of §6.3. **How
strictly** the controlled role's reaction is graded is a **standard per speaker** (`eval.Standard`: the latency
logistic `mid_ms` / `width_ms`, the early-reply floor `early_floor`, the reply window `respond_within_ms`), set on the
controlled role by speaker role with `"*"` as the default and recorded as `meta.standards`:

```yaml
- id: controlled
  config: {name: assistant, standards: {"*": {}, business: {mid_ms: 3000}}}   # the caller: the assistant standard; the business may wait
- id: controlled
  config: {name: user, out: user.speech, obs: [policy.speech], standards: {"*": {early_floor: 1.0}}}  # a caller may answer at once
```

Without `standards`, every speaker is graded by the assistant standard (`eval.ASSISTANT_STANDARD`; `CALLER_STANDARD`
drops the early-reply penalty). The outcome rules themselves (talked over = 0, kept talking = 0, ...) are the same for
every pair; a tolerance for overlap would be a new outcome rule, not a standard — see §6.

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

- **Outcome rules per pair.** The standards (§4) scale the grading; the outcomes are fixed: an overlap is
  `talked_over` (0) whoever does it. A role allowed to overlap another (a caller cutting a long-winded assistant short
  without penalty) needs an outcome rule keyed by the pair, which would change what the expectations mean; it waits
  for the real-corpus timing statistics of DESIGN.md §5.2.
- **Several controlled roles** (self-play): waits for real-human data to anchor them (DESIGN.md §5.1).
- **Tool callers by role.** Tool calls are keyed by caller class (`agent` → `policy.tool_call`, `user` →
  `user.tool_call`; `tools.CALL`), not by role name: a third speaking role has no tool stream yet.
- **The session for non-controlled roles.** `Env` sends the `session` (instructions, tools) to the controlled role
  only; simulated roles take their identity from the task (`persona`, `instructions`).
- **The viewer** draws two main lanes (the controlled role and the first other role) and puts further roles in
  `other` lanes; a per-role lane layout and colours come with the three-party tasks.
