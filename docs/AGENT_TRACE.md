# Agent trace (optional companion format)

Per-chunk / per-unit information internal to the agent server is not part of the trajectory ([FORMAT.md](FORMAT.md) records only environment-level facts). When you need to monitor or debug an agent (for example, to check that a model's chunk layout is what you expect), the agent adapter can save a separate **agent trace**:

- File: `agent_traces.jsonl`, next to the trajectory's `episodes.jsonl`, one record per episode per line.
- Correspondence: a record's `episode_id` equals the trajectory's `meta.episode_id`. Episodes without a trace have no line.
- JSON Schema: [agent_trace.schema.json](agent_trace.schema.json).
- Source: `VllmOmniDuplexAgent(trace_tokens=True)`, which requires a server that supports the token trace (see [agent_server.md](agent_server.md#per-unit-token-trace-debug)). Retrieve a record with `agent.trace(episode_id)`.
- Viewing: `viewer.export_agent_trace_html(episode, trace, path)`, or `viewer.export_run(episodes, out_dir, traces=[...])`, which generates an `<episode>.agent.html` page for every episode that has a trace. `export_run` also draws a condensed per-unit decision (listen / speak, generated text, special tokens such as `<|turn_eos|>`; not the full token list) on a units strip below the agent track of the environment page, lit up during playback (`viewer.units(trace)`; also available via `export_html(..., traces=[...])`).

## Record fields

| Field | Meaning |
|---|---|
| `episode_id` | the corresponding trajectory |
| `server`, `model` | source |
| `unit_ms` | the model's own chunk length (the `chunk_period_ms` reported by the server) |
| `clock` | `input` (lockstep) or `realtime` |
| `units[]` | one entry per model unit, in submission order; fields below |

Fields of each `units[]` entry:

| Field | Meaning |
|---|---|
| `unit_index` | 0-based; `-1` denotes the initial prompt at session start |
| `end_ms` | input time at the end of the unit, equal to episode time |
| `decision` | `listen` / `speak` / `prompt` |
| `stages[]` | one entry per model stage (e.g. `thinker`, `talker`): `input` is the tokens this unit appended to the model context, `output` is the tokens this stage generated for this unit. Each token is `[id, raw token piece]`; special tokens are shown verbatim |
| `special_ids` | ids of the tokenizer special tokens that appear above |

## Example (MiniCPM-o 4.5)

```json
{"episode_id": "booking-alex", "server": "vllm-omni duplex", "model": "openbmb/MiniCPM-o-4_5", "unit_ms": 1000, "clock": "input",
 "units": [
  {"unit_index": 0, "end_ms": 1000, "decision": "listen", "stages": [{"stage": "thinker", "input": [[151683, "<unit>"], [151697, "<|audio|>"]], "output": [[151705, "<|listen|>"]]}]},
  {"unit_index": 3, "end_ms": 4000, "decision": "speak", "stages": [
     {"stage": "thinker", "input": [[151705, "<|listen|>"], [151684, "</unit>"], [151683, "<unit>"]], "output": [[151706, "<|speak|>"], [39814, "Sure"]]},
     {"stage": "talker", "input": [[39814, "Sure"], [151687, "<audio_bos>"]], "output": [[4192, "s3:4192"]]}]}
 ]}
```

The token lists in the example are abbreviated; a real record lists every token, e.g. all 10 `<|audio|>` placeholders of each unit.
