# SOPs: one source format, pluggable presentation

Work tasks (booking a table, refunding an order) sit between tool-call tasks and open conversation: the agent may or
may not have tools, but it must follow a **standard operating procedure** before it acts, and decline when the
procedure does not allow it. InteractionGym keeps the SOP as **structured data** and lets the task, the experiment or
the user decide **how it is written for the model**.

- `interaction_gym.sop`: the source format, the condition logic, and the formats (plugins).
- `interaction_gym.sop_task`: spoken SOP tasks for agents that only talk, and their judge.
- `examples/sop_calls.py` (or the bundle `ig run sop_calls`, the same scenarios from `interaction_gym/data/sop_calls.json`): service calls with MiniCPM-o 4.5: in English a restaurant (×2), a clinic and an online shop; in Chinese a courier (changing a parcel's address, ×2) and a hotel (late checkout, ×2).

## 1. Source format

An SOP is JSON-able data (in a task: `task.scenario["sop"]`; in a trajectory: `meta.policy.source`):

```json
{"name": "Trattoria Roma: table bookings",
 "rules": ["Keep replies short."],
 "actions": {"book_table": {
   "description": "Book a table for the caller.",
   "requires": {"chain": [
     {"check": "request_details", "text": "The caller has told you the day, the time and the number of people.", "by": "ask"},
     {"all": [
       {"check": "opening_hours", "text": "The time is within opening hours ...", "by": "look_up", "on_fail": "Suggest the nearest time within opening hours."},
       {"any": [{"check": "party_of_8_or_fewer", "text": "The party is 8 people or fewer.", "by": "look_up"},
                {"check": "deposit_agreed", "text": "For 9 or more: the caller agrees to a deposit ...", "by": "ask", "on_fail": "Give them the events line ..."}]},
       {"check": "table_free", "text": "A table ... is free at that time.", "by": "look_up", "on_fail": "Offer the nearest free time."}]},
     {"check": "name_and_phone", "text": "The caller has given a name and a phone number.", "by": "ask"}]},
   "then": ["Read the booking back and confirm it."],
   "otherwise": "Explain which condition is not met, decline, and offer the alternative if there is one."}}}
```

| Node | Meaning |
|---|---|
| `{"check": id, "text", "by", "on_fail"}` | One condition. `by`: `ask` (the caller says it) or `look_up` (records, a tool, the agent's screen). `on_fail`: what to do when it does not hold |
| `{"all": [...]}` | Every part must hold, in any order |
| `{"any": [...]}` | At least one part must hold |
| `{"chain": [...]}` | Every part must hold, established in this order |

`lang` (optional, default `en`; built in: `en`, `zh`) is the language the SOP's texts are written in: the built-in
formats write their own words in it ("ALL of these must hold" / "以下条件全部满足"), and `sop_task` frames the agent's
instructions in it.

This is the same shape as SOPBench's constraint graphs (`and` / `or` / `chain` over single conditions) and as a τ²
policy's preconditions, written once so that it can be rendered, checked and generated.

**Logic** (`sop.holds`, `sop.witnessed`, `sop.refuted`): given a case's ground truth (`{check id: holds?}`), whether
the action is allowed; whether the conditions the agent established, in the order it established them, justify acting
(every required check, a holding alternative of each `any`, `chain` parts in order); whether they justify declining
(a failing check that makes the tree false). `known` checks (e.g. what is on the agent's screen) count without being
established and do not constrain the order.

## 2. Formats are plugins

A format is a function `(sop: dict) -> str`. Built in:

| Format | What the model reads |
|---|---|
| `structured` | Nested ALL / ANY / IN THIS ORDER bullets (as SOPBench's structured format) |
| `list` | Numbered steps; an `any` becomes "At least one of: (a) ... OR (b) ...". Unlike SOPBench's `list` (its `old` format), which flattens every condition into one numbered set, it keeps the logic and a fixed order |
| `code` | Python-style pseudo-code: one function per condition, one per action with an early `return decline(...)` per step |
| `mermaid` | A flowchart: one decision per condition, yes / no edges, the `on_fail` alternatives as end nodes |

Add your own in code:

```python
from interaction_gym import sop

@sop.register_format("checklist")
def checklist(s: dict) -> str:
    return "\n".join(f"[ ] {c['text']}" for c in sop.checks(s["actions"]["book_table"]["requires"]))

sop.render(my_sop, "checklist")
```

or from another package, with an entry point (loaded by `sop.formats()`, so every runner's `--format` sees it):

```toml
[project.entry-points."interaction_gym.sop_formats"]
checklist = "my_package.formats:checklist"
```

Which format works best is an empirical question per model (FlowBench, arXiv 2406.14884: flowcharts lead its turn-level
scores, but in full simulated sessions the best format differs by model and by one to three points), so runners take the format as a parameter
(`examples/sop_calls.py --format`) and record it in `meta.policy.format`.

## 3. Spoken SOP tasks (`sop_task`)

For agents that only talk (a full-duplex speech model without tool calls): the information the agent would look up is
in its instructions as "your screen" (`sop_case.facts`), the rest it must ask the caller. A task carries the SOP, the
case (`action`, `facts`, `truth`, `expect`: the details of a correct outcome when it should be done), the agent's role
and the caller (persona, goal with the caller's details). Cases come in pairs: the same SOP with a caller for whom the
action should be done, and one for whom it should be declined.

**Judge** (`sop_task.judge`, recorded as `eval.verifier`): an LLM reads the transcript and *reports* what happened,
without scoring: the decision (done / declined / unfinished) and its turn, the conditions the agent established and
where, the reason given for declining, the details it committed to, whether it offered the SOP's alternative. Rules on
the SOP tree then score three parts, and the reward is 1 only if all hold:

| Part | Done | Declined |
|---|---|---|
| `decision_correct` | the SOP allows it for this caller | the SOP does not allow it |
| `procedure_followed` | every `ask` condition it needed was established before it committed (`chain` in order); `look_up` ones are on its screen | a failing condition that rules it out was established and given as the reason |
| `details_correct` | what it confirmed matches `expect` (an LLM compares meanings) | it offered the SOP's alternative, where there is one |

The viewer shows the SOP (this case's action excerpted first, then the full SOP the agent got, folded, with the excerpt highlighted), the agent's screen and the ground truth in the "SOP policy"
card, and the judge's report with the three parts under Evaluation → Content.

## 4. Not yet

- SOPBench's cases as source SOPs (its constraint graphs map one to one), so its domains can be rendered with any format.
- A generator that writes SOP, facts, ground truth and caller together (the data factory's task level).
- Tool-using agents on these tasks: the same tree, checked against tool calls and the final state instead of the transcript.
